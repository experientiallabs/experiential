"""Harness selection and report tests for the model-distillation CLI."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from typer.testing import Result

from wmo.cli import app
from wmo.cli.model_app_test import (
    _SAMPLER_PATH,
    _flat,
    _gate,
    _invoke,
    _patch_run,
    _RunRecorder,
    _write_inputs,
    runner,
)
from wmo.optimize.model.config import DistillConfig
from wmo.optimize.model.loop import (
    STUDENT_AFTER_EVAL,
    STUDENT_BEFORE_EVAL,
    TEACHER_BASELINE_EVAL,
    DistillEvalReport,
    StepMetrics,
)
from wmo.optimize.model.store import DistillRunStore
from wmo.runtime.agents.default import default_agent
from wmo.runtime.harness.store import HarnessStore

model_app_module = importlib.import_module("wmo.cli.model_app")

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


def test_report_survives_the_half_written_last_metrics_line_of_an_aborted_run(
    tmp_path: Path,
) -> None:
    """`report` advertises itself as safe on an aborted run dir, so it must read one.

    A run that died mid-append leaves a torn final line. That used to end the command in a
    traceback, after the gate verdict and the solve-rate table had already printed.
    """
    run_dir = _finished_run(tmp_path)
    store = DistillRunStore(run_dir)
    store.append_metrics(0, _step_metrics())
    with store.metrics_path.open("a", encoding="utf-8") as handle:
        handle.write('{"step": 1, "reverse_')

    result = _report(run_dir)

    assert result.exit_code == 0, result.output
    flat = _flat(result)
    assert "ignoring a half-written last line" in flat
    assert "reporting the 1 complete row(s)" in flat
    assert "1 training step(s) recorded" in flat  # the complete row is still reported
    assert "Traceback" not in result.output


def test_report_on_metrics_damaged_above_the_last_line_is_a_usage_error(tmp_path: Path) -> None:
    """Only the final line is excusable: a broken row above it means content was lost."""
    run_dir = _finished_run(tmp_path)
    store = DistillRunStore(run_dir)
    store.metrics_path.write_text('{"step": 0}\n[1, 2, 3]\n{"step": 2}\n', encoding="utf-8")

    result = _report(run_dir)

    assert result.exit_code == 2
    flat = _flat(result)
    assert "corrupt metrics row on line 2" in flat
    assert "Traceback" not in result.output


def test_report_refuses_a_last_metrics_line_that_parses_into_a_non_object(tmp_path: Path) -> None:
    """The tail excuse is for a line that fails to parse, which is all a torn append leaves.

    A whole `[]`/`null` at the end was written intact, so skipping it would report the previous
    step as the run's latest state while hiding that the file had been damaged.
    """
    run_dir = _finished_run(tmp_path)
    store = DistillRunStore(run_dir)
    store.append_metrics(0, _step_metrics())
    with store.metrics_path.open("a", encoding="utf-8") as handle:
        handle.write("[]\n")

    result = _report(run_dir)

    assert result.exit_code == 2
    flat = _flat(result)
    assert "corrupt metrics row on line 2" in flat
    assert "expected a JSON object" in flat
    assert "ignoring a half-written last line" not in flat
    assert "Traceback" not in result.output
