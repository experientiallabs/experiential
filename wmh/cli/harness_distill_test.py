"""CLI tests for `wmh optimize <agent> harbor --mode distill`, driven via CliRunner.

`run_distillation` is monkeypatched to a recorder (no real tinker or harbor):
these tests pin the CLI lifecycle around it: input loading and pinning, the
cost-confirmation rule, resume conflicts, progress rendering, the completion
output, and the `--promote` settings write.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner, Result

from wmh.cli import app
from wmh.config.settings import load_settings
from wmh.distill.gate import DistillGateRecord
from wmh.distill.loop import (
    DistillBudgetError,
    DistillProgress,
    DistillResult,
    ProgressCallback,
    SpendSummary,
)
from wmh.distill.store import DEFAULT_TINKER_OPENAI_ENDPOINT, AdapterStore, DistillRunStore
from wmh.harness.doc import HarnessDoc
from wmh.harness.store import HarnessStore

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wmh.distill.config import DistillConfig
    from wmh.distill.loop import DistillServiceClient, LiveTrialPreflight

harness_distill_module = importlib.import_module("wmh.cli.harness_distill")

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
    monkeypatch.setattr(harness_distill_module, "run_distillation", recorder)


def _invoke(tmp_path: Path, *extra: str, agent: str = "pi", input: str | None = None) -> Result:
    return runner.invoke(
        app,
        [
            "optimize",
            agent,
            "harbor",
            "--mode",
            "distill",
            "--distill-config",
            str(tmp_path / "distill.toml"),
            "--task-ids",
            str(tmp_path / "train.json"),
            "--holdout-task-ids",
            str(tmp_path / "holdout.json"),
            "--run-dir",
            str(tmp_path / "run"),
            "--root",
            str(tmp_path / ".wmh"),
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
    assert adapter_store.root == tmp_path / ".wmh"  # --root, not the cwd default
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
    assert str(AdapterStore(tmp_path / ".wmh").dir_for("pi") / "v1") in flat
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
    error = DistillBudgetError(
        "budget exhausted: $51.00 spent against the $50.00 cap",
        resume_command=f"wmh optimize pi harbor --mode distill --run-dir {tmp_path / 'run'} "
        "--resume",
        spent_usd=51.0,
        max_usd=50.0,
    )
    _patch_run(monkeypatch, _RunRecorder(error=error))

    result = _invoke(tmp_path, "--yes")

    assert result.exit_code == 1
    flat = _flat(result)
    assert "budget exhausted" in flat
    assert "resume with:" in flat and "--resume" in flat


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
            "pi",
            "harbor",
            "--mode",
            "distill",
            "--run-dir",
            str(tmp_path / "run"),
            "--root",
            str(tmp_path / ".wmh"),
        ],
    )

    assert result.exit_code == 2
    flat = _flat(result)
    assert "--distill-config" in flat
    assert "--task-ids" in flat
    assert "--holdout-task-ids" in flat


def test_distill_requires_a_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_inputs(tmp_path)
    _patch_run(monkeypatch, _RunRecorder())

    result = runner.invoke(
        app,
        [
            "optimize",
            "pi",
            "harbor",
            "--mode",
            "distill",
            "--distill-config",
            str(tmp_path / "distill.toml"),
            "--task-ids",
            str(tmp_path / "train.json"),
            "--holdout-task-ids",
            str(tmp_path / "holdout.json"),
            "--root",
            str(tmp_path / ".wmh"),
        ],
    )

    assert result.exit_code == 2
    assert "--run-dir is required" in _flat(result)


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
    HarnessStore(str(tmp_path / ".wmh")).save_version(HarnessDoc.baseline("soft"))

    result = _invoke(tmp_path, "--yes", agent="soft")

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
            "pi",
            "harbor",
            "--mode",
            "distill",
            "--resume",
            "--distill-config",
            str(tmp_path / "distill.toml"),
            "--run-dir",
            str(tmp_path / "run"),
            "--root",
            str(tmp_path / ".wmh"),
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
    settings = load_settings(tmp_path / ".wmh")
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
    assert load_settings(tmp_path / ".wmh").models.agent is None


def test_promote_confirmation_is_required_even_with_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--yes covers the cost prompt only; the settings write always asks (EOF declines)."""
    _write_inputs(tmp_path)
    _patch_run(monkeypatch, _RunRecorder())

    result = _invoke(tmp_path, "--yes", "--promote")  # no input: the confirm sees EOF

    assert result.exit_code == 0, result.output
    assert "skipped writing" in _flat(result)
    assert load_settings(tmp_path / ".wmh").models.agent is None


def test_promote_skips_on_a_rejected_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_inputs(tmp_path)
    _patch_run(monkeypatch, _RunRecorder(accepted=False))

    result = _invoke(tmp_path, "--yes", "--promote", input="y\n")

    assert result.exit_code == 0, result.output
    assert "--promote skipped" in _flat(result)
    assert load_settings(tmp_path / ".wmh").models.agent is None
