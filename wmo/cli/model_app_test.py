"""CLI tests for `wmo optimize distill`, driven via CliRunner.

`run_distillation` is monkeypatched to a recorder (no real tinker or harbor):
these tests pin the `run` command's CLI lifecycle around it (input loading and
pinning, the cost-confirmation rule, resume conflicts, progress rendering, the
completion output, and the `--promote` settings write), plus what `report`
reads back out of a finished run dir.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner, Result

from wmo.agents.default import default_agent
from wmo.cli import app
from wmo.cli.model_app import PROBE_EXIT_INSUFFICIENT, PROBE_EXIT_NO_GAP
from wmo.config.settings import load_settings
from wmo.distill.config import DistillConfig
from wmo.distill.gate import DistillGateRecord
from wmo.distill.loop import (
    STUDENT_AFTER_EVAL,
    STUDENT_BEFORE_EVAL,
    TEACHER_BASELINE_EVAL,
    DistillBudgetError,
    DistillEvalReport,
    DistillProgress,
    DistillResult,
    ProgressCallback,
    SpendSummary,
    StepMetrics,
    resume_command,
)
from wmo.distill.rollouts import E2B_SANDBOXES_PER_TRIAL
from wmo.distill.store import DEFAULT_TINKER_OPENAI_ENDPOINT, AdapterStore, DistillRunStore
from wmo.harness.doc import HarnessDoc
from wmo.harness.e2b_reap import CapacityCheck, ReapOutcome
from wmo.harness.store import HarnessStore
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import ProviderKind, TokenUsage
from wmo.providers.pool import PoolEntry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wmo.distill.loop import DistillServiceClient, LiveTrialPreflight

model_app_module = importlib.import_module("wmo.cli.model_app")

runner = CliRunner()

_SAMPLER_PATH = "tinker://weights/pi-final"

_PRICING_TOML = """\
[pricing]
student_prefill = 0.5
student_sample = 1.4
student_train = 1.3
teacher_prefill = 2.5
teacher_sample = 6.25
"""

_BUDGET_TOML = """\
[budget]
max_usd = 50.0
"""


def _gate(accepted: bool) -> DistillGateRecord:
    return DistillGateRecord(
        accepted=accepted,
        reason="accepted: after 0.500" if accepted else "rejected: after 0.100",
        teacher_solve_rate=0.6,
        student_before_solve_rate=0.2,
        student_after_solve_rate=0.5 if accepted else 0.1,
        min_teacher_fraction=0.7,
    )


def _result(run_dir: str, *, accepted: bool = True) -> DistillResult:
    return DistillResult(
        name="pi",
        run_dir=run_dir,
        steps_completed=2,
        final_sampler_path=_SAMPLER_PATH,
        final_state_path="tinker://state/pi-final",
        gate=_gate(accepted),
        adapter_version=1 if accepted else None,
        spend=SpendSummary(lines=[], session_usd=12.5, prior_usd=0.0, total_usd=12.5),
    )


class _RunRecorder:
    """Stands in for `run_distillation`: records the call, emits canned progress."""

    def __init__(
        self,
        *,
        accepted: bool = True,
        error: Exception | None = None,
        progress: tuple[DistillProgress, ...] = (),
        promote_adapter: bool = True,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self._accepted = accepted
        self._error = error
        self._progress = progress
        self._promote_adapter = promote_adapter

    def __call__(
        self,
        name: str,
        cfg: DistillConfig,
        harness: HarnessDoc,
        train_task_ids: Sequence[str],
        holdout_task_ids: Sequence[str],
        run_dir: Path,
        *,
        resume: bool = False,
        on_progress: ProgressCallback | None = None,
        service_client: DistillServiceClient | None = None,
        adapter_store: AdapterStore | None = None,
        live_trial_preflight: LiveTrialPreflight | None = None,
        cli_agent: str | None = None,
    ) -> DistillResult:
        self.calls.append(
            {
                "name": name,
                "cfg": cfg,
                "harness": harness,
                "train": tuple(train_task_ids),
                "holdout": tuple(holdout_task_ids),
                "run_dir": run_dir,
                "resume": resume,
                "cli_agent": cli_agent,
                "adapter_store": adapter_store,
            }
        )
        # Mirror the real loop's first durable action: the config snapshot is
        # what marks a run dir as started (the CLI's fresh/resume guards key
        # on it), so the stand-in must uphold that contract too.
        DistillRunStore(run_dir).snapshot_config(cfg)
        if on_progress is not None:
            for event in self._progress:
                on_progress(event)
        if self._error is not None:
            raise self._error
        result = _result(str(run_dir), accepted=self._accepted)
        if not self._promote_adapter:
            result = result.model_copy(update={"adapter_version": None})
        return result


def _write_inputs(
    tmp_path: Path,
    *,
    extra_toml: str = _BUDGET_TOML,
    train_ids: tuple[str, ...] = ("t1", "t2"),
    holdout_ids: tuple[str, ...] = ("h1",),
) -> None:
    (tmp_path / "job.yaml").write_text("job_name: template\n", encoding="utf-8")
    (tmp_path / "distill.toml").write_text(
        "[student]\n"
        'base_model = "test/student"\n'
        "[teacher]\n"
        'model = "test/teacher"\n'
        "[harbor]\n"
        f'job_template = "{tmp_path / "job.yaml"}"\n'
        "[train]\n"
        "steps = 2\n"
        "tasks_per_batch = 1\n"
        "group_size = 1\n" + extra_toml,
        encoding="utf-8",
    )
    (tmp_path / "train.json").write_text(json.dumps(list(train_ids)), encoding="utf-8")
    (tmp_path / "holdout.json").write_text(json.dumps(list(holdout_ids)), encoding="utf-8")


def _patch_run(monkeypatch: pytest.MonkeyPatch, recorder: _RunRecorder) -> None:
    monkeypatch.setattr(model_app_module, "run_distillation", recorder)


def _invoke(
    tmp_path: Path, *extra: str, harness: str | None = None, input: str | None = None
) -> Result:
    """Start a fresh run through the real CLI; `harness` types `--harness` explicitly."""
    return runner.invoke(
        app,
        [
            "optimize",
            "distill",
            "run",
            "--config",
            str(tmp_path / "distill.toml"),
            "--task-ids",
            str(tmp_path / "train.json"),
            "--holdout-task-ids",
            str(tmp_path / "holdout.json"),
            "--run-dir",
            str(tmp_path / "run"),
            "--root",
            str(tmp_path / ".wmo"),
            *(() if harness is None else ("--harness", harness)),
            *extra,
        ],
        input=input,
    )


def _flat(result: Result) -> str:
    """Collapse rich wrapping (and typer's error-box borders) for substring asserts."""
    return " ".join(result.output.replace("│", " ").split())


# -- routing and the happy path ---------------------------------------------------------------


def test_distill_routes_to_run_distillation_with_pinned_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    recorder = _RunRecorder()
    _patch_run(monkeypatch, recorder)

    result = _invoke(tmp_path, "--yes")

    assert result.exit_code == 0, result.output
    [call] = recorder.calls
    assert call["name"] == "pi"
    assert call["train"] == ("t1", "t2")
    assert call["holdout"] == ("h1",)
    assert call["resume"] is False
    assert call["run_dir"] == tmp_path / "run"
    harness = call["harness"]
    assert isinstance(harness, HarnessDoc)
    assert harness.runtime_kind() == "pi-node"  # the built-in pi seed
    adapter_store = call["adapter_store"]
    assert isinstance(adapter_store, AdapterStore)
    assert adapter_store.root == tmp_path / ".wmo"  # --root, not the cwd default
    # The CLI-level pins land in distill-run.json for --resume.
    record = json.loads((tmp_path / "run" / "distill-run.json").read_text(encoding="utf-8"))
    assert record["agent"] == "pi"
    assert record["train_task_ids"] == ["t1", "t2"]
    assert record["holdout_task_ids"] == ["h1"]
    assert record["seed_version"] is None
    flat = _flat(result)
    assert "student test/student" in flat and "teacher test/teacher" in flat
    assert "gate" in flat and "accepted: after 0.500" in flat
    assert _SAMPLER_PATH in flat
    assert "[models.agent]" in flat  # the handoff snippet is printed


def test_distill_completion_prints_adapter_path_and_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    _patch_run(monkeypatch, _RunRecorder())

    result = _invoke(tmp_path, "--yes")

    assert result.exit_code == 0, result.output
    flat = _flat(result)
    assert "adapter pi v1 (champion)" in flat
    assert str(AdapterStore(tmp_path / ".wmo").dir_for("pi") / "v1") in flat
    assert "$12.50 total" in flat


def test_distill_rejected_gate_prints_no_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    _patch_run(monkeypatch, _RunRecorder(accepted=False))

    result = _invoke(tmp_path, "--yes")

    assert result.exit_code == 0, result.output
    flat = _flat(result)
    assert "rejected: after 0.100" in flat
    assert "adapter not promoted" in flat
    assert "(champion)" not in flat


def test_distill_progress_events_render_phase_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    events = (
        DistillProgress(
            phase="preflight",
            message="running preflight checks before any spend",
            total_steps=2,
            spent_usd=0.0,
        ),
        DistillProgress(
            phase="training",
            message="step 1/2: solve rate 0.50",
            total_steps=2,
            step=0,
            spent_usd=3.25,
        ),
    )
    _patch_run(monkeypatch, _RunRecorder(progress=events))

    result = _invoke(tmp_path, "--yes")

    assert result.exit_code == 0, result.output
    flat = _flat(result)
    assert "[preflight] running preflight checks before any spend" in flat
    assert "[training] step 1/2: solve rate 0.50 ($3.25 spent)" in flat


def test_distill_budget_abort_prints_the_resume_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    # The real helper, not a literal: what is printed has to be the command that
    # actually resumes, so a rename must break here rather than in a user's shell.
    error = DistillBudgetError(
        "budget exhausted: $51.00 spent against the $50.00 cap",
        resume_command=resume_command("pi", tmp_path / "run"),
        spent_usd=51.0,
        max_usd=50.0,
    )
    _patch_run(monkeypatch, _RunRecorder(error=error))

    result = _invoke(tmp_path, "--yes")

    assert result.exit_code == 1
    flat = _flat(result)
    assert "budget exhausted" in flat
    assert "resume with:" in flat
    assert f"wmo optimize distill run --run-dir {tmp_path / 'run'} --resume" in flat


def test_distill_runtime_error_exits_nonzero_with_the_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    _patch_run(monkeypatch, _RunRecorder(error=RuntimeError("teacher preflight ping failed")))

    result = _invoke(tmp_path, "--yes")

    assert result.exit_code == 1
    assert "teacher preflight ping failed" in _flat(result)


# -- cost confirmation -------------------------------------------------------------------------


def test_distill_unpriced_meters_without_budget_reject_yes_non_interactively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unpriced meters force interactive confirmation ONLY when max_usd is also unset."""
    _write_inputs(tmp_path, extra_toml="")  # no pricing, no budget cap
    recorder = _RunRecorder()
    _patch_run(monkeypatch, recorder)

    result = _invoke(tmp_path, "--yes")

    assert result.exit_code == 2
    flat = _flat(result)
    assert "unbounded" in flat
    assert "budget.max_usd" in flat
    assert recorder.calls == []


def test_distill_unpriced_meters_with_a_budget_cap_honor_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)  # budget cap set, meters unpriced
    recorder = _RunRecorder()
    _patch_run(monkeypatch, recorder)

    result = _invoke(tmp_path, "--yes")

    assert result.exit_code == 0, result.output
    assert len(recorder.calls) == 1
    flat = _flat(result)
    assert "unknown" in flat  # unpriced meters still print "unknown" in the table
    assert "hard cap budget.max_usd=$50.00" in flat


def test_distill_fully_priced_estimate_proceeds_without_budget_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The four full prices plus teacher_sample fully price the run: the two
    # cached-prefill meters derive their 20% defaults.
    _write_inputs(tmp_path, extra_toml=_PRICING_TOML)
    recorder = _RunRecorder()
    _patch_run(monkeypatch, recorder)

    result = _invoke(tmp_path, "--yes")

    assert result.exit_code == 0, result.output
    assert len(recorder.calls) == 1
    flat = _flat(result)
    assert "unknown" not in flat
    assert "priced total $" in flat


def test_distill_cost_table_lists_cached_and_teacher_sample_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cost table renders every meter, with derived cached prices shown."""
    _write_inputs(tmp_path, extra_toml=_PRICING_TOML + _BUDGET_TOML)
    recorder = _RunRecorder()
    _patch_run(monkeypatch, recorder)

    result = _invoke(tmp_path, "--yes")

    assert result.exit_code == 0, result.output
    flat = _flat(result)
    for meter in (
        "student_prefill",
        "student_cached_prefill",
        "student_sample",
        "student_train",
        "teacher_prefill",
        "teacher_cached_prefill",
        "teacher_sample",
    ):
        assert meter in flat, meter
    # Derived cached rates: 20% of 0.5 and 2.5; teacher_sample as configured.
    assert "0.100" in flat
    assert "0.500" in flat
    assert "6.250" in flat


# -- input validation --------------------------------------------------------------------------


def test_distill_requires_its_input_flags_to_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    _patch_run(monkeypatch, _RunRecorder())

    result = runner.invoke(
        app,
        [
            "optimize",
            "distill",
            "run",
            "--run-dir",
            str(tmp_path / "run"),
            "--root",
            str(tmp_path / ".wmo"),
        ],
    )

    assert result.exit_code == 2
    flat = _flat(result)
    assert "--config" in flat
    assert "--task-ids" in flat
    assert "--holdout-task-ids" in flat


def test_distill_requires_a_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--run-dir holds every durable artifact, so it is mandatory on both paths."""
    _write_inputs(tmp_path)
    _patch_run(monkeypatch, _RunRecorder())

    result = runner.invoke(
        app,
        [
            "optimize",
            "distill",
            "run",
            "--config",
            str(tmp_path / "distill.toml"),
            "--task-ids",
            str(tmp_path / "train.json"),
            "--holdout-task-ids",
            str(tmp_path / "holdout.json"),
            "--root",
            str(tmp_path / ".wmo"),
        ],
    )

    assert result.exit_code == 2
    assert "Missing option '--run-dir'" in _flat(result)


def test_distill_reports_config_errors_as_usage_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    (tmp_path / "distill.toml").write_text("[student\nnot toml", encoding="utf-8")
    _patch_run(monkeypatch, _RunRecorder())

    result = _invoke(tmp_path, "--yes")

    assert result.exit_code == 2
    assert "invalid distill config" in _flat(result)


def test_distill_names_the_failing_config_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    (tmp_path / "distill.toml").write_text(
        '[student]\nbase_model = "s"\n[teacher]\nmodel = "t"\n', encoding="utf-8"
    )
    _patch_run(monkeypatch, _RunRecorder())

    result = _invoke(tmp_path, "--yes")

    assert result.exit_code == 2
    assert "harbor" in _flat(result)  # the missing required section is named


def test_distill_rejects_overlapping_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path, holdout_ids=("t1",))
    _patch_run(monkeypatch, _RunRecorder())

    result = _invoke(tmp_path, "--yes")

    assert result.exit_code == 2
    flat = _flat(result)
    assert "t1" in flat and "BOTH" in flat


def test_distill_rejects_a_missing_job_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    (tmp_path / "job.yaml").unlink()
    _patch_run(monkeypatch, _RunRecorder())

    result = _invoke(tmp_path, "--yes")

    assert result.exit_code == 2
    assert "job_template" in _flat(result)


def test_distill_requires_a_pi_node_seed_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    _patch_run(monkeypatch, _RunRecorder())
    HarnessStore(str(tmp_path / ".wmo")).save_version(HarnessDoc.baseline("soft"))

    result = _invoke(tmp_path, "--yes", harness="soft")

    assert result.exit_code == 2
    flat = _flat(result)
    assert "runtime kind" in flat and "kit-python" in flat


# -- resume ------------------------------------------------------------------------------------


def test_distill_fresh_run_refuses_a_used_run_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    _patch_run(monkeypatch, _RunRecorder())
    first = _invoke(tmp_path, "--yes")
    assert first.exit_code == 0, first.output

    second = _invoke(tmp_path, "--yes")

    assert second.exit_code == 2
    assert "already holds a distillation run" in _flat(second)


def test_distill_record_without_config_snapshot_starts_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run record with no config.toml means the previous start failed before
    the loop's first durable action (e.g. a missing SDK): the dir must stay
    usable, not brick both the fresh and resume paths."""
    _write_inputs(tmp_path)
    recorder = _RunRecorder()
    _patch_run(monkeypatch, recorder)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "distill-run.json").write_text("{}", encoding="utf-8")

    result = _invoke(tmp_path, "--yes")

    assert result.exit_code == 0, result.output
    assert "never began" in _flat(result)
    assert len(recorder.calls) == 1


def test_distill_resume_reuses_the_pinned_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    recorder = _RunRecorder()
    _patch_run(monkeypatch, recorder)
    first = _invoke(tmp_path, "--yes")
    assert first.exit_code == 0, first.output

    resumed = runner.invoke(
        app,
        [
            "optimize",
            "distill",
            "run",
            "--resume",
            "--config",
            str(tmp_path / "distill.toml"),
            "--run-dir",
            str(tmp_path / "run"),
            "--root",
            str(tmp_path / ".wmo"),
            "--yes",
        ],
    )

    assert resumed.exit_code == 0, resumed.output
    assert len(recorder.calls) == 2
    call = recorder.calls[1]
    assert call["resume"] is True
    assert call["train"] == ("t1", "t2")
    assert call["holdout"] == ("h1",)


def test_distill_resume_rejects_conflicting_task_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    recorder = _RunRecorder()
    _patch_run(monkeypatch, recorder)
    first = _invoke(tmp_path, "--yes")
    assert first.exit_code == 0, first.output
    (tmp_path / "train.json").write_text('["other"]', encoding="utf-8")

    conflict = _invoke(tmp_path, "--yes", "--resume")

    assert conflict.exit_code == 2
    assert "differs from the recorded train split" in _flat(conflict)
    assert len(recorder.calls) == 1


def test_distill_resume_without_a_record_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    _patch_run(monkeypatch, _RunRecorder())

    result = _invoke(tmp_path, "--yes", "--resume")

    assert result.exit_code == 2
    assert "start the run once without --resume" in _flat(result)


# -- promote -----------------------------------------------------------------------------------


def test_promote_writes_models_agent_after_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    _patch_run(monkeypatch, _RunRecorder())

    result = _invoke(tmp_path, "--yes", "--promote", input="y\n")

    assert result.exit_code == 0, result.output
    assert "wrote" in _flat(result)
    settings = load_settings(tmp_path / ".wmo")
    assert settings.models.agent is not None
    assert settings.models.agent.provider == "openai"
    assert settings.models.agent.model == _SAMPLER_PATH
    assert settings.models.agent.model_type == "test/student"
    assert settings.models.agent.endpoint == DEFAULT_TINKER_OPENAI_ENDPOINT


def test_promote_declined_leaves_settings_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    _patch_run(monkeypatch, _RunRecorder())

    result = _invoke(tmp_path, "--yes", "--promote", input="n\n")

    assert result.exit_code == 0, result.output
    assert "skipped writing" in _flat(result)
    assert load_settings(tmp_path / ".wmo").models.agent is None


def test_promote_confirmation_is_required_even_with_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--yes covers the cost prompt only; the settings write always asks (EOF declines)."""
    _write_inputs(tmp_path)
    _patch_run(monkeypatch, _RunRecorder())

    result = _invoke(tmp_path, "--yes", "--promote")  # no input: the confirm sees EOF

    assert result.exit_code == 0, result.output
    assert "skipped writing" in _flat(result)
    assert load_settings(tmp_path / ".wmo").models.agent is None


def test_promote_skips_on_a_rejected_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_inputs(tmp_path)
    _patch_run(monkeypatch, _RunRecorder(accepted=False))

    result = _invoke(tmp_path, "--yes", "--promote", input="y\n")

    assert result.exit_code == 0, result.output
    assert "--promote skipped" in _flat(result)
    assert load_settings(tmp_path / ".wmo").models.agent is None


# -- e2b capacity preflight --------------------------------------------------------------------


def _capacity(
    *, cap: int = 100, alive_before: int, alive: int, required: int, freed: int = 0
) -> CapacityCheck:
    outcome = (
        ReapOutcome(
            killed=tuple(f"orphan-{index}" for index in range(freed)),
            already_gone=(),
            failed=(),
            pruned_ledgers=(),
        )
        if freed
        else None
    )
    return CapacityCheck(
        cap=cap, alive_before=alive_before, alive=alive, required=required, outcome=outcome
    )


def _patch_capacity(monkeypatch: pytest.MonkeyPatch, check: CapacityCheck | Exception) -> list[int]:
    """Stand in for the live account count; returns the `required` values it was asked for."""
    asked: list[int] = []

    def fake_check_capacity(*, required: int) -> CapacityCheck:
        asked.append(required)
        if isinstance(check, Exception):
            raise check
        return check

    monkeypatch.setattr(model_app_module, "check_capacity", fake_check_capacity)
    return asked


def test_e2b_preflight_passes_when_enough_slots_are_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path, extra_toml=f"trial_concurrency = 12\n{_BUDGET_TOML}")
    recorder = _RunRecorder()
    _patch_run(monkeypatch, recorder)
    asked = _patch_capacity(monkeypatch, _capacity(alive_before=40, alive=40, required=12))

    result = _invoke(tmp_path, "--yes", "--backend", "e2b")

    assert result.exit_code == 0, result.output
    # One sandbox per trial: harbor's task environment. Terminus-2 runs in this process.
    assert asked == [12 * E2B_SANDBOXES_PER_TRIAL]
    flat = _flat(result)
    assert "e2b capacity ok: 40/100 sandbox(es) in use, 60 free, 12 needed" in flat
    assert "(1 per trial x train.trial_concurrency=12)" in flat
    assert len(recorder.calls) == 1


def test_e2b_preflight_reaps_dead_owner_orphans_and_then_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path, extra_toml=f"trial_concurrency = 8\n{_BUDGET_TOML}")
    recorder = _RunRecorder()
    _patch_run(monkeypatch, recorder)
    _patch_capacity(monkeypatch, _capacity(alive_before=97, alive=84, required=8, freed=13))

    result = _invoke(tmp_path, "--yes", "--backend", "e2b")

    assert result.exit_code == 0, result.output
    flat = _flat(result)
    assert (
        "reaped 13 orphaned E2B sandbox(es) from dead local runs (97 -> 84 of 100 in use)" in flat
    )
    assert "e2b capacity ok: 84/100" in flat
    assert len(recorder.calls) == 1


def test_e2b_preflight_fails_fast_when_slots_stay_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live failure mode: 73 orphans held the account and every trial 429'd instead."""
    _write_inputs(tmp_path, extra_toml=f"trial_concurrency = 12\n{_BUDGET_TOML}")
    recorder = _RunRecorder()
    _patch_run(monkeypatch, recorder)
    _patch_capacity(monkeypatch, _capacity(alive_before=98, alive=95, required=12, freed=3))

    result = _invoke(tmp_path, "--yes", "--backend", "e2b")

    assert result.exit_code != 0
    flat = _flat(result)
    assert "not enough free E2B sandbox slots: 95 of 100 concurrent sandboxes are in use" in flat
    assert "leaving 5 free, but this run needs 12 (1 per trial x train.trial_concurrency=12" in flat
    assert "harbor's task environment)" in flat
    assert "freed 3 slot(s) and was not enough" in flat
    assert "wmo e2b reap --stale-minutes 60 --yes" in flat
    assert "lower train.trial_concurrency to at most 5" in flat  # 5 free // 1 per trial
    assert "WMO_E2B_SANDBOX_CAP" in flat
    assert recorder.calls == []  # nothing was spent


def test_e2b_preflight_message_names_the_missing_orphan_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path, extra_toml=f"trial_concurrency = 4\n{_BUDGET_TOML}")
    _patch_run(monkeypatch, _RunRecorder())
    _patch_capacity(monkeypatch, _capacity(alive_before=99, alive=99, required=8))

    result = _invoke(tmp_path, "--yes", "--backend", "e2b")

    assert result.exit_code != 0
    assert "No orphan of a dead local run was left to reclaim." in _flat(result)


def test_a_missing_e2b_extra_fails_the_preflight_with_the_sync_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    recorder = _RunRecorder()
    _patch_run(monkeypatch, recorder)
    _patch_capacity(monkeypatch, ImportError("the e2b SDK is not installed; run `uv sync`"))

    result = _invoke(tmp_path, "--yes", "--backend", "e2b")

    assert result.exit_code != 0
    assert "uv sync" in _flat(result)
    assert recorder.calls == []


def test_an_unreachable_account_warns_but_still_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A monitoring call must not brick a resume: an unreachable API warns and continues."""
    _write_inputs(tmp_path)
    recorder = _RunRecorder()
    _patch_run(monkeypatch, recorder)
    _patch_capacity(monkeypatch, RuntimeError("connection reset"))

    result = _invoke(tmp_path, "--yes", "--backend", "e2b")

    assert result.exit_code == 0, result.output
    assert "could not check E2B sandbox capacity" in _flat(result)
    assert len(recorder.calls) == 1


def test_a_missing_credential_fails_fast_instead_of_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unset key is a configuration error: every trial would 401, so do not start."""

    class AuthenticationException(Exception):
        """Name-matched stand-in for the e2b SDK's own credential error."""

    _write_inputs(tmp_path)
    recorder = _RunRecorder()
    _patch_run(monkeypatch, recorder)
    _patch_capacity(monkeypatch, AuthenticationException("API key is required"))

    result = _invoke(tmp_path, "--yes", "--backend", "e2b")

    assert result.exit_code != 0
    flat = _flat(result)
    assert "E2B rejected the sandbox capacity check" in flat
    assert "$E2B_API_KEY" in flat
    assert "backend = 'local'" in flat
    assert recorder.calls == []


def test_a_local_backend_never_touches_the_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    recorder = _RunRecorder()
    _patch_run(monkeypatch, recorder)

    def fake_check_capacity(*, required: int) -> CapacityCheck:
        raise AssertionError(f"the local backend must not check E2B capacity ({required})")

    monkeypatch.setattr(model_app_module, "check_capacity", fake_check_capacity)

    result = _invoke(tmp_path, "--yes")

    assert result.exit_code == 0, result.output
    assert len(recorder.calls) == 1


# -- the --harness flag ------------------------------------------------------------------------


def test_harness_defaults_to_the_builtin_pi_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The harness is a dependency of the run, not its subject, so it defaults."""
    _write_inputs(tmp_path)
    recorder = _RunRecorder()
    _patch_run(monkeypatch, recorder)

    result = _invoke(tmp_path, "--yes")

    assert result.exit_code == 0, result.output
    [call] = recorder.calls
    assert call["name"] == "pi"
    assert call["cli_agent"] == "pi"
    record = json.loads((tmp_path / "run" / "distill-run.json").read_text(encoding="utf-8"))
    assert record["agent"] == "pi"
    assert record["seed_version"] is None  # the built-in seed, never a stored 'pi'


def test_an_explicit_harness_ref_is_pinned_into_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    recorder = _RunRecorder()
    _patch_run(monkeypatch, recorder)
    saved = HarnessStore(str(tmp_path / ".wmo")).save_version(default_agent("pi"))

    result = _invoke(tmp_path, "--yes", harness="pi@1")

    assert result.exit_code == 0, result.output
    [call] = recorder.calls
    assert call["cli_agent"] == "pi@1"  # as typed, so the printed resume command works
    record = json.loads((tmp_path / "run" / "distill-run.json").read_text(encoding="utf-8"))
    assert record["agent"] == "pi@1"
    assert record["seed_version"] == saved.version
    assert record["seed_doc_hash"] == saved.doc_hash


def test_resume_without_harness_adopts_the_recorded_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume carries only --run-dir, so the option's default must not read as
    a conflicting explicit value against a run started from a stored version."""
    _write_inputs(tmp_path)
    recorder = _RunRecorder()
    _patch_run(monkeypatch, recorder)
    HarnessStore(str(tmp_path / ".wmo")).save_version(default_agent("pi"))
    first = _invoke(tmp_path, "--yes", harness="pi@1")
    assert first.exit_code == 0, first.output

    resumed = runner.invoke(
        app,
        [
            "optimize",
            "distill",
            "run",
            "--run-dir",
            str(tmp_path / "run"),
            "--root",
            str(tmp_path / ".wmo"),
            "--resume",
            "--yes",
        ],
    )

    assert resumed.exit_code == 0, resumed.output
    assert recorder.calls[1]["cli_agent"] == "pi@1"


def test_resume_rejects_a_conflicting_explicit_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    recorder = _RunRecorder()
    _patch_run(monkeypatch, recorder)
    first = _invoke(tmp_path, "--yes")
    assert first.exit_code == 0, first.output
    HarnessStore(str(tmp_path / ".wmo")).save_version(default_agent("pi"))

    conflict = _invoke(tmp_path, "--yes", "--resume", harness="pi@1")

    assert conflict.exit_code == 2
    flat = _flat(conflict)
    assert "--harness 'pi@1'" in flat and "recorded 'pi'" in flat
    assert len(recorder.calls) == 1


def test_an_explicit_backend_overrides_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_inputs(tmp_path)
    recorder = _RunRecorder()
    _patch_run(monkeypatch, recorder)
    monkeypatch.setattr(
        model_app_module, "_preflight_e2b_capacity", lambda console, *, trial_concurrency: None
    )

    result = _invoke(tmp_path, "--yes", "--backend", "e2b")

    assert result.exit_code == 0, result.output
    [call] = recorder.calls
    cfg = call["cfg"]
    assert isinstance(cfg, DistillConfig)
    assert cfg.harbor is not None
    assert cfg.harbor.backend == "e2b"  # the config says "local"
    record = json.loads((tmp_path / "run" / "distill-run.json").read_text(encoding="utf-8"))
    assert record["backend"] == "e2b"


# -- report ------------------------------------------------------------------------------------


def _eval_report(
    name: str, model: str, *, solve: float, graded: float, graded_trials: int = 6
) -> DistillEvalReport:
    return DistillEvalReport(
        name=name,
        provider_model=model,
        base_model="test/student",
        task_ids=["h1", "h2"],
        attempts=3,
        trials=6,
        solve_rate=solve,
        graded_solve_rate=graded,
        graded_trials=graded_trials,
        empty_span_trials=0,
        executed_trials=6,
        infra_failed_trials=0,
        scaffold_loss_rate=0.25,
    )


def _finished_run(tmp_path: Path, *, accepted: bool = True) -> Path:
    """A run dir holding exactly what the loop persists: gate, evals, metrics."""
    run_dir = tmp_path / "run"
    store = DistillRunStore(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    store.write_gate(_gate(accepted))
    store.write_eval(
        TEACHER_BASELINE_EVAL, _eval_report("teacher", "test/teacher", solve=0.6, graded=0.7)
    )
    store.write_eval(
        STUDENT_BEFORE_EVAL, _eval_report("before", "test/student", solve=0.2, graded=0.3)
    )
    store.write_eval(
        STUDENT_AFTER_EVAL,
        _eval_report("after", _SAMPLER_PATH, solve=0.5 if accepted else 0.1, graded=0.62),
    )
    return run_dir


def _step_metrics() -> StepMetrics:
    """A training row in the real shape, so a field rename breaks this test."""
    return StepMetrics(
        loss="importance_sampling",
        tasks=2,
        trials=4,
        solve_rate=0.5,
        graded_solve_rate=0.6,
        graded_trials=4,
        raw_solve_rate=0.5,
        executed_trials=4,
        infra_failed_trials=0,
        scaffold_loss_rate=0.25,
        stop_reason_counts={"submitted": 3, "max_turns": 1},
        empty_span_trials=0,
        truncated_spans=0,
        datums=4,
        fragments=0,
        fragmentation_rate=0.0,
        overflow_drops=0,
        overlong_drops=0,
        mismatch_drops=0,
        clipped_tokens=0,
        optimizer_updates=1,
        loss_tokens=400,
        context_tokens=2000,
        reverse_kl_per_token=0.4123,
        entropy_per_token=0.16,
        mean_generation_tokens=2140.0,
        entropy_baseline=0.2,
        entropy_ratio=0.8,
        generation_tokens_baseline=2400.0,
        generation_tokens_ratio=0.89,
        reward_mean=0.5,
        advantage_mean=0.05,
        advantage_std=1.2,
        clip_fraction=0.0,
        pg_loss=0.42,
        grad_norm=1.5,
        sampler_path=_SAMPLER_PATH,
        student_prefill_tokens=120,
        student_cached_prefill_tokens=30,
        student_sample_tokens=40,
        student_train_tokens=200,
        teacher_prefill_tokens=160,
        teacher_cached_prefill_tokens=0,
        teacher_sample_tokens=0,
        usd=0.75,
        cumulative_usd=12.5,
    )


def _report(run_dir: Path) -> Result:
    return runner.invoke(app, ["optimize", "distill", "report", "--run-dir", str(run_dir)])


def test_report_prints_the_verdict_and_the_before_after_table(tmp_path: Path) -> None:
    run_dir = _finished_run(tmp_path)
    DistillRunStore(run_dir).append_metrics(0, _step_metrics())

    result = _report(run_dir)

    assert result.exit_code == 0, result.output
    flat = _flat(result)
    assert "accepted: after 0.500" in flat
    # every measurement the gate compared, with the model that produced it
    assert "teacher test/teacher 0.600" in flat
    assert "student before test/student 0.200" in flat
    assert "student after" in flat
    # The sampler path is wider than its table cell at 80 columns, so it folds there.
    # It is also the one value a reader copies out of this report (into a pool entry, or
    # a follow-on run's init_from_state), so it gets an unbroken line of its own.
    assert f"trained artifact: {_SAMPLER_PATH}" in flat
    assert "0.700" in flat and "0.620" in flat  # the graded column
    assert "6/6" in flat and "25%" in flat  # executed trials, scaffold loss
    assert "paired delta (after - before): +0.300 solve rate" in flat
    assert "graded (same trials at test resolution): +0.320" in flat
    assert "after / teacher: 0.833 against gate minimum 0.70; gate passed" in flat
    assert "1 training step(s) recorded" in flat
    assert "reverse KL/token 0.4123" in flat
    assert "tokens/episode 2140" in flat
    assert "$12.50 spent" in flat


def test_report_names_a_rejected_gate(tmp_path: Path) -> None:
    result = _report(_finished_run(tmp_path, accepted=False))

    assert result.exit_code == 0, result.output
    flat = _flat(result)
    assert "rejected: after 0.100" in flat
    assert "paired delta (after - before): -0.100 solve rate" in flat
    assert "gate FAILED" in flat


def test_report_works_without_eval_reports_or_metrics(tmp_path: Path) -> None:
    """gate.json alone still answers the question the command exists for."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    DistillRunStore(run_dir).write_gate(_gate(True))

    result = _report(run_dir)

    assert result.exit_code == 0, result.output
    flat = _flat(result)
    assert "teacher unknown 0.600" in flat
    assert "no training step recorded in metrics.jsonl" in flat


def test_report_on_a_run_that_never_gated_says_how_to_finish_it(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    result = _report(run_dir)

    assert result.exit_code == 2
    flat = _flat(result)
    assert "has not reached its gate yet" in flat
    assert "wmo optimize distill run --run-dir <dir> --resume" in flat


# -- `probe`: the teacher-search gate over a measured matrix ------------------------------------


def _probe(matrix_file: Path, *extra: str) -> Result:
    return runner.invoke(app, ["optimize", "distill", "probe", str(matrix_file), *extra])


def _matrix_file(tmp_path: Path, *, gain: float, scenarios: int = 12) -> Path:
    """A two-candidate matrix where the dearer model is `gain` reward points better."""
    pool = [
        PoolEntry(
            name=name,
            kind=ProviderKind.OPENAI,
            model=f"{name}-runtime",
            tier="open",
            input_per_mtok=rate,
            output_per_mtok=rate,
        )
        for name, rate in (("small", 0.10), ("big", 1.15))
    ]
    outcomes = [
        ScenarioOutcome(
            scenario_id=f"s{i:02d}",
            task=f"task {i}",
            model=name,
            reward=0.30 + (gain + (0.02 if i % 2 else -0.02) if name == "big" else 0.0),
            success=True,
            usage=TokenUsage(input_tokens=100, output_tokens=50),
            # Distinct per-episode spend, so the measured ladder has a cheapest model to pick
            # the student from rather than a tie.
            cost_usd=0.002 if name == "small" else 0.020,
        )
        for name in ("small", "big")
        for i in range(scenarios)
    ]
    path = tmp_path / "matrix.json"
    OutcomeMatrix(pool=pool, outcomes=outcomes).save(path)
    return path


def test_probe_exits_zero_and_names_the_teacher_when_the_gap_is_real(tmp_path: Path) -> None:
    result = _probe(_matrix_file(tmp_path, gain=0.30))

    assert result.exit_code == 0, result.output
    flat = _flat(result)
    assert "distill" in flat
    assert "big *" in flat  # the gain table stars the chosen teacher
    assert "+30.0" in flat


def test_probe_exits_three_when_the_matrix_shows_no_gap(tmp_path: Path) -> None:
    """The documented no-gap code: a script branches on it without parsing the sentence."""
    result = _probe(_matrix_file(tmp_path, gain=0.02))

    assert result.exit_code == PROBE_EXIT_NO_GAP
    assert "DO NOT DISTILL" in _flat(result)


def test_probe_exits_four_when_the_matrix_is_too_thin_to_decide(tmp_path: Path) -> None:
    result = _probe(_matrix_file(tmp_path, gain=0.30, scenarios=3))

    assert result.exit_code == PROBE_EXIT_INSUFFICIENT
    assert "INSUFFICIENT EVIDENCE" in _flat(result)


def test_probe_takes_the_bar_and_the_student_from_the_caller(tmp_path: Path) -> None:
    matrix_file = _matrix_file(tmp_path, gain=0.30)

    lowered = _probe(matrix_file, "--min-gap", "0.5")
    assert lowered.exit_code == PROBE_EXIT_NO_GAP  # +30 points no longer clears a 50-point bar

    inverted = _probe(matrix_file, "--student", "big")
    assert inverted.exit_code == PROBE_EXIT_NO_GAP  # the small model is 30 points WORSE
    assert "against 'big'" in _flat(inverted)


def test_probe_on_a_missing_matrix_says_which_command_writes_one(tmp_path: Path) -> None:
    result = _probe(tmp_path / "nope.json")

    assert result.exit_code == 2
    flat = _flat(result)
    assert "no outcome matrix at" in flat
    assert "wmo optimize route sweep" in flat


def test_probe_on_a_file_that_is_not_a_matrix_says_so(tmp_path: Path) -> None:
    path = tmp_path / "matrix.json"
    path.write_text('{"pool": []}', encoding="utf-8")

    result = _probe(path)

    assert result.exit_code == 2
    assert "is not a readable outcome matrix" in _flat(result)
