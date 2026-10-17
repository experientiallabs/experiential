"""The few lines ``exp build`` prints when it finishes; the full detail is in its report file."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.text import Text

from exp.cli.evaluation.learning_summary import LearningSummary, learning_lines
from exp.cli.shared.progress import duration_text
from exp.simulation.build import ProjectBuild


def render_completed_build(
    console: Console,
    completed: ProjectBuild,
    *,
    project: str,
    learning: LearningSummary,
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
