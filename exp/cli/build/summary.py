"""The few lines ``exp build`` prints when it finishes; the full detail is in its report file."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.text import Text

from exp.cli.shared.progress import duration_text
from exp.common.core.files import write_text_atomic
from exp.common.project import ProjectStore
from exp.common.project.paths import validate_local_id
from exp.simulation.build import ProjectBuild

if TYPE_CHECKING:  # learning loads the fidelity lifecycle; keep `exp --help` free of it
    from exp.cli.build.learning import BuildLearning


def render_completed_build(
    console: Console,
    completed: ProjectBuild,
    *,
    project: str,
    learning: BuildLearning,
    report: Path | None,
    duration_seconds: float,
) -> None:
    """Present a few result lines: traces, learning, retest, the report and the next command.

    Args:
        console: The build command's terminal.
        completed: Persisted project build and mining results.
        project: Local project identifier used in the next recommended command.
        learning: What the learning stage did.
        report: Saved full learning and retest output, when learning ran.
        duration_seconds: Wall-clock time the whole command took.
    """
    dataset = completed.artifacts.trace_dataset.dataset
    tasks = completed.artifacts.mining.tasks
    duplicate_count = len(dataset.trace_ids) - len(completed.artifacts.mining.analysis.candidates)
    console.print(
        f"[green]✓[/green] ingested {len(dataset.trace_ids)} traces "
        f"[dim]({dataset.invalid_trace_count} invalid, {duplicate_count} duplicate)[/dim] · "
        f"split {sum(task.partition == 'fit' for task in tasks)} fit / "
        f"{sum(task.partition == 'held_out' for task in tasks)} held out"
    )
    if len(dataset.trace_ids) < 100 or len(dataset.trace_ids) > 1_000:
        console.print("[yellow]guidance[/yellow] 100 to 1,000 traces is the usual starting range")
    for line in learning_lines(learning):
        console.print(line)
    if report is not None:
        console.print(Text(f"Report: {report}"))
    console.print(f"[green]Complete[/green] [dim]in {duration_text(duration_seconds)}[/dim]")
    if learning.outcome is not None:
        console.print(f"next: exp eval {project} --audit N  [dim](check on held-out traces)[/dim]")
    else:
        console.print(f"next: exp optimize router {project}")


def learning_lines(learning: BuildLearning, noun: str = "traces") -> tuple[str, ...]:
    """The learning and retest result lines, warnings included, in terminal markup."""
    outcome = learning.outcome
    if outcome is None:
        return ()
    calls = (
        "no new model calls (saved results)"
        if outcome.attempts_used == 0
        else f"{outcome.attempts_used:,} model attempts"
    )
    learned = (
        f"[green]✓[/green] learned {outcome.learned}/{outcome.traces} {noun} · "
        f"memory v{outcome.head_version} · {calls}"
    )
    curve = learning.curve
    if curve is not None and curve.points:
        exact = sum(point.exact for point in curve.points)
        steps = sum(point.steps for point in curve.points)
        learned += f"\n  replies exact while learning {_percent(exact, steps)} ({exact}/{steps})"
        if curve.quarters:
            last = curve.quarters[-1]
            learned += f", last {last.last - last.first + 1} traces {last.share}"
    lines = [learned]
    if outcome.incomplete:
        lines.append(
            f"[yellow]warning[/yellow] {outcome.incomplete} trace(s) stopped on every allowed "
            "attempt and were not learned"
        )
    if outcome.stopped is not None:
        lines.append(f"[yellow]warning[/yellow] {outcome.stopped}")
    retest = learning.retest
    if retest is not None:
        total, changes = retest.total, retest.changes
        lines.append(
            f"[green]✓[/green] retested {retest.planned} "
            f"{'trace' if retest.planned == 1 else 'traces'} that had misses · "
            f"{_percent(total.exact, total.steps)} replies exact ({total.exact}/{total.steps}) · "
            f"fixed {changes.fixed} of {changes.missed} missed turns · "
            f"{changes.regressed} regressed"
        )
    return tuple(lines)


def _percent(part: int, whole: int) -> str:
    """A whole-number percentage, or a dash without a denominator."""
    return f"{part / whole:.0%}" if whole else "-"


def save_details(store: ProjectStore, kind: str, identity: str, text: str) -> Path:
    """Write a command's full output beside its saved run state, like an evaluation report.

    Args:
        store: Project owning the run.
        kind: Runtime family, such as ``learning`` or ``audits``.
        identity: Exact identity of the saved state the output describes.
        text: The rendered full output.

    Returns:
        ``<project>/runtime/<kind>/<identity>/report.txt``, written atomically.
    """
    validate_local_id(identity, label=f"{kind} identity")
    path = store.paths.runtime_directory / kind / identity / "report.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, text)
    return path
