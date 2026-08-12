"""Read-only reporting helpers for completed model-distillation runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from wmo.common.core.types import JsonObject

if TYPE_CHECKING:
    from wmo.optimize.model.gate import DistillGateRecord
    from wmo.optimize.model.loop import DistillEvalReport
    from wmo.optimize.model.store import DistillRunStore

_REPORT_ROWS: tuple[tuple[str, str], ...] = (
    ("teacher", "baseline-teacher"),
    ("student before", "baseline-student-before"),
    ("student after", "student-after"),
)


def _load_gate(store: DistillRunStore) -> DistillGateRecord:
    """Read the run's gate record or raise a usage error."""
    from wmo.optimize.model.gate import DistillGateRecord

    try:
        text = store.gate_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            f"no {store.gate_path}: this run has not reached its gate yet (or "
            f"{store.run_dir} is not a distillation run dir). Finish or resume it with "
            "`wmo optimize distill run --run-dir <dir> --resume`"
        ) from exc
    try:
        return DistillGateRecord.model_validate_json(text)
    except ValidationError as exc:
        raise typer.BadParameter(f"cannot load {store.gate_path}: {exc}") from exc


def _load_eval_report(store: DistillRunStore, key: str) -> DistillEvalReport | None:
    """Read one evaluation report, returning ``None`` when it was not written."""
    from wmo.optimize.model.loop import DistillEvalReport

    try:
        text = (store.evals_dir / f"{key}.json").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        return DistillEvalReport.model_validate_json(text)
    except ValidationError as exc:
        raise typer.BadParameter(
            f"cannot load {store.evals_dir / f'{key}.json'}: {exc}; delete the file to "
            "report from gate.json alone"
        ) from exc


def _solve_rate_table(store: DistillRunStore, gate: DistillGateRecord) -> Table:
    """Build the held-out teacher and student solve-rate comparison."""
    rates = (
        gate.teacher_solve_rate,
        gate.student_before_solve_rate,
        gate.student_after_solve_rate,
    )
    table = Table(title=f"Held-out solve rates ({store.run_dir})")
    table.add_column("Measurement", no_wrap=True)
    table.add_column("Model", overflow="fold")
    table.add_column("Solve", justify="right")
    table.add_column("Graded", justify="right")
    table.add_column("Executed", justify="right")
    table.add_column("Scaffold", justify="right")
    for (label, key), rate in zip(_REPORT_ROWS, rates, strict=True):
        eval_report = _load_eval_report(store, key)
        if eval_report is None:
            table.add_row(label, "unknown", f"{rate:.3f}", "-", "-", "-")
            continue
        graded = (
            f"{eval_report.graded_solve_rate:.3f}" if eval_report.graded_trials else "unmeasured"
        )
        table.add_row(
            label,
            eval_report.provider_model,
            f"{rate:.3f}",
            graded,
            f"{eval_report.executed_trials}/{eval_report.trials}",
            f"{eval_report.scaffold_loss_rate:.0%}",
        )
    return table


def _print_trained_artifact(console: Console, store: DistillRunStore) -> None:
    """Print the exact sampler path behind the student-after measurement."""
    after = _load_eval_report(store, "student-after")
    if after is None or not after.provider_model:
        return
    console.print(f"trained artifact: {escape(after.provider_model)}")


def _print_paired_delta(console: Console, store: DistillRunStore, gate: DistillGateRecord) -> None:
    """Print what training moved on the gate's held-out split."""
    binary = gate.student_after_solve_rate - gate.student_before_solve_rate
    console.print(f"paired delta (after - before): {binary:+.3f} solve rate")
    before = _load_eval_report(store, "baseline-student-before")
    after = _load_eval_report(store, "student-after")
    if before is not None and after is not None and before.graded_trials and after.graded_trials:
        graded = after.graded_solve_rate - before.graded_solve_rate
        console.print(f"  graded (same trials at test resolution): {graded:+.3f}")
    fraction = (
        gate.student_after_solve_rate / gate.teacher_solve_rate
        if gate.teacher_solve_rate > 0
        else None
    )
    reached = "unmeasurable (teacher solved nothing)" if fraction is None else f"{fraction:.3f}"
    verdict = "passed" if gate.accepted else "FAILED"
    console.print(
        f"  after / teacher: {reached} against gate minimum "
        f"{gate.min_teacher_fraction:.2f}; gate {verdict}"
    )


def _read_metrics(console: Console, store: DistillRunStore) -> list[JsonObject]:
    """Read complete metrics rows while tolerating one half-written tail row."""
    try:
        return store.read_metrics()
    except ValueError:
        pass
    try:
        rows = store.read_metrics(tolerate_partial_tail=True)
    except ValueError as fatal:
        raise typer.BadParameter(str(fatal)) from fatal
    console.print(
        f"[yellow]note[/yellow] ignoring a half-written last line in "
        f"{escape(str(store.metrics_path))} (a run killed mid-append leaves one); "
        f"reporting the {len(rows)} complete row(s)"
    )
    return rows


def _print_training_summary(console: Console, store: DistillRunStore) -> None:
    """Print the last complete training row's health metrics."""
    rows = [row for row in _read_metrics(console, store) if row.get("phase") is None]
    if not rows:
        console.print("no training step recorded in metrics.jsonl")
        return
    last = rows[-1]
    step = _row_int(last, "step")
    parts = [f"{len(rows)} training step(s) recorded"]
    for label, key, spec in (
        ("reverse KL/token", "reverse_kl_per_token", ".4f"),
        ("entropy ratio", "entropy_ratio", ".2f"),
        ("tokens/episode", "mean_generation_tokens", ".0f"),
        ("tokens/episode ratio", "generation_tokens_ratio", ".2f"),
    ):
        value = _row_float(last, key)
        if value is not None:
            parts.append(f"{label} {value:{spec}}")
    spent = _row_float(last, "cumulative_usd")
    if spent is not None:
        parts.append(f"${spent:.2f} spent")
    head = "training" if step is None else f"training (last row step {step})"
    console.print(f"{head}: {', '.join(parts)}")


def _row_float(row: JsonObject, key: str) -> float | None:
    """Return one numeric metrics-row field as a float."""
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _row_int(row: JsonObject, key: str) -> int | None:
    """Return one integer metrics-row field."""
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
