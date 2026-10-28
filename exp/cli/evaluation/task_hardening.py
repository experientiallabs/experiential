"""Continuation actions for task hardening and task-validity checks in the task factory.

Both actions validate saved evidence first, stop there on ``--dry-run``, show the cost inventory
and ask for spend consent before any paid draw, and replay saved reports without asking again.
The terminal shows a short summary; the full report is saved beside the project's runtime state.
"""

from rich.text import Text

from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.cli.evaluation.learning_summary import save_details
from exp.optimize.evaluation.factory.hardening import (
    TaskHardeningReport,
    TaskHardeningSpec,
    preflight_task_hardening,
    run_task_hardening,
    saved_task_hardening,
)
from exp.optimize.evaluation.factory.validity import (
    TaskValidityReport,
    TaskValiditySpec,
    preflight_task_validity,
    run_task_validity,
    saved_task_validity,
)


def _validity_lines(report: TaskValidityReport) -> list[str]:
    """One line per check, with its first reason."""
    lines = [f"Task validity: {report.status}"]
    for check in report.checks:
        reason = f": {check.reasons[0]}" if check.reasons else ""
        lines.append(f"  {check.name}: {check.status}{reason}")
    return lines


def _hardening_lines(report: TaskHardeningReport) -> list[str]:
    """One line per examined candidate, with its pass rate, interval and band."""
    lines = [f"Task hardening: {report.status} (one family, lineage {report.lineage})"]
    for level in report.levels:
        if level.failure is not None:
            lines.append(f"  level {level.level}: author failed ({level.failure})")
        for row in level.candidates:
            line = f"  level {level.level} world {row.proposal}: {row.disposition}"
            difficulty = row.difficulty
            if difficulty is not None and difficulty.pass_rate is not None:
                interval = difficulty.interval or (0.0, 1.0)
                line += (
                    f", pass rate {difficulty.pass_rate:.2f} "
                    f"[{interval[0]:.2f}, {interval[1]:.2f}] over {len(difficulty.actors)} "
                    f"actors, {difficulty.unknown} unknown, band {difficulty.band}"
                )
            elif row.rejections or row.refusals:
                line += f": {(row.rejections or row.refusals)[0]}"
            lines.append(line)
    return lines


def handle_check_task_validity(context: ContinuationContext, spec: TaskValiditySpec) -> None:
    """Run or replay leak, closed-book and control checks for one qualified variant.

    Args:
        context: Validated ``--continuations`` invocation.
        spec: Saved scenario-search cell and the registered validity models and thresholds.

    Returns:
        None. A summary and the saved report path are printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
    saved = saved_task_validity(store, spec, code_revision=revision)
    if saved is None:
        preflight_task_validity(store, spec)
    if context.dry_run:
        console.print("Validity inputs and models checked; no calls or reports created.")
        return
    if saved is None and not context.consent(spec):
        return
    report = saved or run_task_validity(store, spec, code_revision=revision)
    for line in _validity_lines(report):
        console.print(Text(line))
    path = save_details(
        store, "task-validity", spec.identity(revision), report.model_dump_json(indent=2)
    )
    console.print(Text(f"Report: {path}"))


def handle_harden_tasks(context: ContinuationContext, spec: TaskHardeningSpec) -> None:
    """Run or replay a bounded hardening experiment over one qualified seed world.

    Args:
        context: Validated ``--continuations`` invocation.
        spec: Seed world, panel, saturation rule, band, validity settings and level allowance.

    Returns:
        None. A summary and the saved report path are printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
    saved = saved_task_hardening(store, spec, code_revision=revision)
    if saved is None:
        preflight_task_hardening(store, spec)
    console.print(
        Text(
            f"Task hardening: up to {spec.maximum_levels} harder levels, "
            f"{len(spec.actors)} panel actors, {spec.repeats} attempts each"
        )
    )
    if context.dry_run:
        console.print("Seed, panel, validity and author inputs checked; no calls made.")
        return
    if saved is None and not context.consent(spec):
        return
    report = saved or run_task_hardening(store, spec, code_revision=revision)
    for line in _hardening_lines(report):
        console.print(Text(line))
    path = save_details(
        store, "task-hardening", spec.identity(revision), report.model_dump_json(indent=2)
    )
    console.print(Text(f"Report: {path}"))
