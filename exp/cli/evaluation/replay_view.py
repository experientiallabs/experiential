"""Terminal presentation shared by replay panels: the held-out audit and the miss retest."""

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from statistics import median

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from exp.cli.evaluation.fidelity import render_fidelity
from exp.cli.evaluation.fidelity_turns import render_turns, turn_marks
from exp.cli.shared.progress import duration_text, progress_display
from exp.common.models.client import RequestAdmissionRefused
from exp.common.models.unknown_outcome import DispatchOutcomeUnknown
from exp.common.progress import ProgressHook
from exp.common.project import ProjectStore
from exp.optimize.evaluation.fidelity.panel import execute_panel, require_resumable
from exp.optimize.evaluation.fidelity.replay import (
    TRIVIAL_REPLY_CHARACTERS,
    ReplayAggregate,
    ReplayResult,
)
from exp.optimize.evaluation.fidelity.runner import PreparedFidelity
from exp.runtime.models.structured_completion import CompletionNotDispatched
from exp.runtime.sessions.dispatch_ledger import DispatchCapExhaustedError
from exp.simulation.evaluation.workflow import INFRASTRUCTURE_STOPS


def run_panel(
    console: Console,
    store: ProjectStore,
    registrations: tuple[PreparedFidelity, ...],
    progress: ProgressHook | None = None,
    *,
    guard: Callable[[PreparedFidelity], None] | None = None,
    resume: bool = False,
) -> None:
    """Run a registered panel, explaining an infrastructure stop.

    ``progress`` is the calling command's single progress line; without it, the panel owns a
    display with one row per trace.

    Raises:
        ValueError: A trace was interrupted and ``resume`` was not given; nothing is sent.
        typer.Exit: The execution machinery stopped a trace (see ``INFRASTRUCTURE_STOPS``);
            completed traces stay saved and the stopped one is never scored.
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
    except INFRASTRUCTURE_STOPS as exc:
        console.print(f"[red]error[/red] {stop_reason(exc)}")
        console.print(
            "Completed traces are saved. The stopped trace is never bought again silently: "
            "run the same command with --retry-incomplete to resume it."
        )
        raise typer.Exit(code=1) from exc


def stop_reason(error: BaseException) -> str:
    """What stopped a replay or learning run, for its error line.

    Args:
        error: One of ``INFRASTRUCTURE_STOPS``.

    Returns:
        A plain reason naming the failed part of the execution machinery.
    """
    if isinstance(error, DispatchCapExhaustedError):
        return f"the run spent its physical-attempt ceiling: {error}"
    if isinstance(error, DispatchOutcomeUnknown):
        return f"a provider request has an unknown outcome: {error}"
    if isinstance(error, (CompletionNotDispatched, RequestAdmissionRefused)):
        return f"a provider request was refused before dispatch: {error}"
    return f"a provider request failed: {error}"


def aggregate_line(total: ReplayAggregate, planned: int, title: str = "Held-out fidelity") -> str:
    """The pooled headline, then the per-lineage rate, the reply-kind split, and replay time.

    The headline pools every turn of every trace; the per-lineage mean weighs each independent
    lineage equally and carries its 95 percent interval at the lineage count. The judge mean
    states how many traces it covers and how many judge calls failed.

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
    line = f"{title}  {exact} · {fields} · {judge_summary(total)} · {total.traces}/{planned} traces"
    if total.lineage_exact is not None and total.lineage_interval is not None:
        low, high = total.lineage_interval
        line += (
            f"\n  per-lineage mean exact {total.lineage_exact:.0%} (95% interval {low:.0%} to "
            f"{high:.0%}, {total.lineages} {'lineage' if total.lineages == 1 else 'lineages'})"
        )
    kinds = [row for row in total.kinds if row.steps]
    if kinds:
        line += (
            "\n  exact by recorded reply: "
            + " · ".join(
                f"{row.kind} {row.exact}/{row.steps} ({share(row.exact, row.steps)})"
                for row in kinds
            )
            + f" (trivial: empty or under {TRIVIAL_REPLY_CHARACTERS} characters)"
        )
    if total.replay_seconds:
        line += (
            f"\n  replay time per trace {duration_text(median(total.replay_seconds))} median, "
            f"{duration_text(max(total.replay_seconds))} longest (execution start to saved report)"
        )
    if total.judge_turns:
        line += (
            f"\n  judge agrees with exact replies on {total.judge_agreements}/{total.judge_turns} "
            f"turns ({share(total.judge_agreements, total.judge_turns)})"
        )
    return line


def facts_line(registrations: Sequence[PreparedFidelity]) -> str:
    """How many replayed traces had their predictions conditioned on supplied instance facts.

    Exactness on a trace given its starting-world facts measures fidelity given those facts,
    so a panel states how many of its traces were measured that way.

    Args:
        registrations: Every registered trace of the panel.

    Returns:
        ``Instance facts: ...`` with the supplied count over the planned traces.
    """
    supplied = [row.spec.facts for row in registrations if row.spec.facts is not None]
    if not supplied:
        return "  instance facts: none supplied; predictions read only the task and its history"
    facts = sum(len(row.facts.facts) for row in supplied)
    return (
        f"  instance facts: supplied to {len(supplied)}/{len(registrations)} traces "
        f"({facts} validated facts); exactness there is measured given those facts"
    )


def judge_summary(total: ReplayAggregate) -> str:
    """The judge mean with the traces it covers and the judge failures, for the headline.

    Args:
        total: Sums over completed traces.

    Returns:
        ``judge M/5 (J/T traces judged, F judge failures)``, with ``-`` for no assessed score.
    """
    mean = "-" if total.judge_mean is None else f"{total.judge_mean:.1f}/5"
    failures = (
        f"{total.judge_failures} judge {'failure' if total.judge_failures == 1 else 'failures'}"
    )
    return f"judge {mean} ({total.judged_traces}/{total.traces} traces judged, {failures})"


def render_traces(
    console: Console,
    store: ProjectStore,
    registrations: tuple[PreparedFidelity, ...],
    results: tuple[ReplayResult, ...],
    label: str,
) -> None:
    """Per-trace rows, then every turn and criterion of each trace."""
    table = Table(box=None, padding=(0, 2))
    for column in (label, "Replies exact", "Field match", "Judge", "Replay time", "Turns"):
        table.add_column(column)
    for result in results:
        table.add_row(
            Text(result.trace_id),
            f"{result.exact}/{result.steps}",
            share(result.fields_matched, result.fields_total),
            _judge(result),
            "-" if result.replay_seconds is None else duration_text(result.replay_seconds),
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
