"""Terminal presentation shared by replay panels: the held-out audit and the miss retest."""

from collections.abc import Callable
from contextlib import nullcontext

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from exp.cli.evaluation.fidelity import render_fidelity
from exp.cli.evaluation.fidelity_turns import render_turns, turn_marks
from exp.cli.shared.progress import progress_display
from exp.common.progress import ProgressHook
from exp.common.project import ProjectStore
from exp.optimize.evaluation.fidelity.panel import execute_panel, require_resumable
from exp.optimize.evaluation.fidelity.replay import ReplayAggregate, ReplayResult
from exp.optimize.evaluation.fidelity.runner import PreparedFidelity
from exp.runtime.models.providers.transport import ProviderTransportError


def run_panel(
    console: Console,
    store: ProjectStore,
    registrations: tuple[PreparedFidelity, ...],
    progress: ProgressHook | None = None,
    *,
    guard: Callable[[], None] | None = None,
    resume: bool = False,
) -> None:
    """Run a registered panel, explaining a provider failure.

    ``progress`` is the calling command's single progress line; without it, the panel owns a
    display with one row per trace.

    Raises:
        ValueError: A trace was interrupted and ``resume`` was not given; nothing is sent.
        typer.Exit: A provider request failed; completed traces stay saved.
    """
    stopped = require_resumable(store, registrations, resume=resume)
    if stopped:
        console.print(
            Text(
                f"{stopped} interrupted trace(s) resume in place; saved steps replay "
                "without provider calls",
                style="dim",
            )
        )
    try:
        with nullcontext(progress) if progress is not None else progress_display(console) as hook:
            execute_panel(store, registrations, progress=hook, guard=guard, resume=resume)
    except ProviderTransportError as exc:
        console.print(f"[red]error[/red] a provider request failed: {exc}")
        console.print(
            "Completed traces are saved. Run the same command again to continue; add "
            "--retry-incomplete to resume a trace that stopped before saving its result."
        )
        raise typer.Exit(code=1) from exc


def aggregate_line(total: ReplayAggregate, planned: int, title: str = "Held-out fidelity") -> str:
    """One line: replies exact, reply-field agreement, and the judge mean.

    Args:
        total: Sums over completed traces.
        planned: Traces in the panel, including any without a saved report.
        title: Leading label naming what was measured.

    Returns:
        Plain summary text with explicit denominators.
    """
    exact = f"{total.exact}/{total.steps} replies exact ({share(total.exact, total.steps)})"
    fields = (
        f"{share(total.fields_matched, total.fields_total)} reply fields match "
        f"({total.fields_matched}/{total.fields_total})"
    )
    judge = "judge -" if total.judge_mean is None else f"judge {total.judge_mean:.1f}/5"
    line = f"{title}  {exact} · {fields} · {judge} · {total.traces}/{planned} traces"
    if total.judge_turns:
        line += (
            f"\n  judge agrees with exact replies on {total.judge_agreements}/{total.judge_turns} "
            f"turns ({share(total.judge_agreements, total.judge_turns)})"
        )
    return line


def render_traces(
    console: Console,
    store: ProjectStore,
    registrations: tuple[PreparedFidelity, ...],
    results: tuple[ReplayResult, ...],
    label: str,
) -> None:
    """Per-trace rows, then every turn and criterion of each trace."""
    table = Table(box=None, padding=(0, 2))
    for column in (label, "Replies exact", "Field match", "Judge", "Turns"):
        table.add_column(column)
    for result in results:
        table.add_row(
            Text(result.trace_id),
            f"{result.exact}/{result.steps}",
            share(result.fields_matched, result.fields_total),
            _judge(result),
            turn_marks(result.turns),
        )
    console.print()
    console.print(table)
    prepared = {p.spec.run_id: p for p in registrations}
    for result in results:
        registration = prepared[result.run_id]
        console.print()
        console.print(Text(f"{result.trace_id} · {result.run_id}", style="bold"))
        for disclosure in result.recovery_lines():
            console.print(Text(f"  {disclosure}", style="yellow"))
        render_turns(console, store, registration, result.report, registration.spec.judge)
        render_fidelity(console, result.report, registration.spec.judge)


def _judge(result: ReplayResult) -> str:
    """Mean assessed score of one trace, or its judge status."""
    if not result.judge_scores:
        return "-" if result.judge_status == "completed" else result.judge_status
    return f"{sum(result.judge_scores) / len(result.judge_scores):.1f}"


def share(part: int, whole: int) -> str:
    """A whole-number percentage, or a dash without a denominator."""
    return f"{part / whole:.0%}" if whole else "-"
