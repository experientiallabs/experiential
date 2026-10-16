"""Held-out simulation fidelity in the existing evaluation terminal conventions."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from exp.cli.evaluation.fidelity import render_fidelity
from exp.cli.evaluation.fidelity_turns import render_turns, turn_marks
from exp.cli.evaluation.view import heading
from exp.cli.shared.consent import can_prompt, require_spend_consent
from exp.cli.shared.options import usage_error
from exp.cli.shared.progress import progress_display, qualified
from exp.cli.shared.theme import EXP_THEME
from exp.common.project import ProjectStore
from exp.common.release_revision import installed_release_revision
from exp.optimize.evaluation.fidelity import (
    PreparedFidelity,
    execute_fidelity_evaluation,
    saved_fidelity_report,
)
from exp.optimize.evaluation.held_out_exposure import LIMITS, LineageExposure
from exp.optimize.evaluation.held_out_fidelity import (
    AuditCheckpoint,
    HeldOutPanel,
    register_held_out_fidelity,
    saved_held_out_results,
)
from exp.optimize.evaluation.held_out_receipts import (
    AuditReceipt,
    audit_receipts,
    record_audit_receipt,
)
from exp.optimize.evaluation.trace_learning import fit_cases, learning_curve
from exp.optimize.evaluation.trace_replay import ReplayAggregate, ReplayResult, aggregate
from exp.runtime.models.providers.transport import ProviderTransportError

PANEL_CONCURRENCY = 4  # traces of a frozen panel replayed at once


def run_held_out_fidelity(
    project: str,
    count: int,
    *,
    root: Path,
    yes: bool,
    non_interactive: bool,
    dry_run: bool,
    verbose: bool = False,
    retry_incomplete: bool = False,
    include_inspected: bool = False,
) -> None:
    """Measure the learned simulator on held-out traces it never learned, or replay the result.

    Uses the hybrid profile: the memory learned with ``--hold-out`` at its final checkpoint,
    with every held-out lineage reserved, plus recorded facts from the fit-only index. Every
    run saves an audit receipt before reporting; ``--verbose`` shows per-trace detail, which
    makes the audited lineages development data that later audits skip.

    Args:
        project: Existing project with a completed build.
        count: Maximum held-out traces, selected deterministically.
        root: Existing project and artifact root.
        yes: Explicit authorization after the undefined-cost warning.
        non_interactive: Disable terminal questions.
        dry_run: Register the runs and stop before credentials or provider calls.
        verbose: Also print every trace, every turn, and every criterion.
        retry_incomplete: Replace interrupted runs, buying their calls again.
        include_inspected: Also re-run lineages an earlier audit showed in detail, reported as
            a development re-audit.
    """
    console = Console(theme=EXP_THEME)
    with usage_error(OSError, ValueError):
        store = ProjectStore(root, project)
        revision = installed_release_revision()
        panel = register_held_out_fidelity(
            store,
            count,
            profile="memory",
            code_revision=revision,
            clock=lambda: datetime.now(UTC),
            retry_incomplete=retry_incomplete,
            include_inspected=include_inspected,
        )

        pending = [p for p in panel.prepared if saved_fidelity_report(store, p) is None]

        def record(*, results_shown: bool) -> AuditReceipt:
            """Save this audit's receipt, and any exposure, before anything it covers is shown."""
            return record_audit_receipt(
                store,
                panel,
                fresh=bool(pending),
                results_shown=results_shown,
                detail_shown=results_shown and verbose,
                code_revision=revision,
                clock=lambda: datetime.now(UTC),
            )

        heading(console, project, "Held-out simulation fidelity")
        judge = panel.prepared[0].spec.judge_alias
        steps = sum(len(case.steps) for case in panel.cases)
        version = panel.checkpoint.version if panel.checkpoint else None
        console.print(
            Text(
                f"{len(panel.cases)} held-out traces · {steps} turns · hybrid simulator "
                f"{panel.world_model.artifact_id} · memory v{version} · judge {judge}",
                style="dim",
            )
        )
        console.print(
            Text(
                f"panel {panel.identity} · exposure ledger {panel.exposure.ledger} "
                f"({panel.exposure.scheme})",
                style="dim",
            )
        )
        if panel.excluded:
            console.print(Text(skipped_line(panel.excluded), style="dim"))
        if panel.retired:
            console.print(
                Text(
                    f"{len(panel.retired)} earlier attempts could not complete; "
                    "registered fresh attempts",
                    style="dim",
                )
            )
        fit = len(fit_cases(store))
        if version is not None and version < fit:
            console.print(
                f"[yellow]warning[/yellow] memory v{version} learned "
                f"{version} of {fit} fit traces; finish `exp build --learn` "
                "to measure the fully learned simulator"
            )
        if dry_run:
            console.print(Text(receipt_line(record(results_shown=False), store), style="dim"))
            for prepared in panel.prepared:
                console.print(Text(f"  {prepared.spec.run_id}"))
            console.print(
                "Prepared without provider calls. Run the same command without --dry-run."
            )
            return
        if not pending:
            console.print(Text("Saved results. No provider calls.", style="dim"))
            # The lifecycle owner replays each report and completes any interrupted publication.
            for prepared in panel.prepared:
                execute_fidelity_evaluation(store, prepared)
        else:
            interactive = not non_interactive and can_prompt(console)
            if not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=None,
                cost_is_upper_bound=False,
                command=f"exp eval {project} --held-out-fidelity {count}",
                non_interactive=not interactive,
            ):
                record(results_shown=False)
                return
            try:
                execute_runs(console, store, panel.prepared)
            except typer.Exit:
                record(results_shown=False)
                raise
        results = saved_held_out_results(store, panel)
        receipt = record(results_shown=True)
        if verbose:
            render_traces(console, store, panel.prepared, results, "Held-out trace")
        console.print()
        console.print(
            Text(aggregate_line(aggregate(results), len(panel.cases), title(panel, receipt)))
        )
        curve = learning_curve(store, "fit")
        if curve is not None:
            console.print(Text(curve.line()))
        render_exposure(console, store, panel, receipt, verbose=verbose)


REAUDIT_TITLE = "Re-audit on inspected cases (development)"


def title(panel: HeldOutPanel, receipt: AuditReceipt) -> str:
    """The aggregate's label: never independent once a case is exposed or results repeat."""
    if not panel.independent:
        return REAUDIT_TITLE
    if receipt.status == "development-re-audit":
        return "Held-out re-audit (development)"
    if receipt.status == "replay":
        return "Held-out fidelity (replay)"
    return "Held-out fidelity"


def skipped_line(excluded: int) -> str:
    """Why the default panel left out exposed held-out lineages."""
    noun, pronoun = ("lineage", "it is") if excluded == 1 else ("lineages", "they are")
    return (
        f"Skipped {excluded} held-out {noun} exposed in the exposure ledger (shown in detail by "
        f"an earlier audit, or grouped with content that was); {pronoun} development data now "
        "(--include-inspected re-runs exposed cases)"
    )


def receipt_line(receipt: AuditReceipt, store: ProjectStore) -> str:
    """The receipt's sequence, ID and status, naming the original for a replay."""
    status = receipt.status
    if receipt.replays is not None:
        original = next(r for r in audit_receipts(store) if r.receipt_id == receipt.replays)
        status = f"replay of receipt #{original.sequence} ({original.status})"
    return (
        f"Audit receipt #{receipt.sequence} {receipt.receipt_id}: {status} · "
        f"code {receipt.code_revision[:12]}"
    )


def exposure_line(item: LineageExposure, panel: HeldOutPanel) -> str:
    """One exposed lineage: whether it was audited, its component size, and the path."""
    where = "in this panel" if item.lineage in panel.inspected else "skipped"
    hops = [
        (
            f"same task {edge.task_lineage}"
            if edge.reason == "declared-task-identity"
            else f"{edge.duplicate_kind} near-duplicate {edge.similarity:.2f}"
        )
        + f" ({edge.project}/{edge.build})"
        for edge in item.path.path
    ]
    route = " → ".join(hops) if hops else "its own content was shown"
    return (
        f"  {item.lineage} ({where}): component of {item.path.component_size} · "
        f"{route} · exposed {item.path.exposed_node[:12]}"
    )


def checkpoint_lines(checkpoint: AuditCheckpoint) -> tuple[str, str]:
    """The audited checkpoint, and that an all-traces build is a different one."""
    return (
        f"Audited checkpoint: memory {checkpoint.lineage} v{checkpoint.version}, "
        f"scope {checkpoint.scope} (held-out lineages reserved before learning)",
        "An all-traces build (exp build --learn without --hold-out) is a different checkpoint "
        "and was not independently evaluated.",
    )


def render_exposure(
    console: Console,
    store: ProjectStore,
    panel: HeldOutPanel,
    receipt: AuditReceipt,
    *,
    verbose: bool,
) -> None:
    """What was audited, its status, the exposure decision and its limits, and what was shown."""
    if panel.checkpoint is not None:
        for line in checkpoint_lines(panel.checkpoint):
            console.print(Text(line, style="dim"))
    console.print(Text(receipt_line(receipt, store), style="dim"))
    exposed = panel.exposure.exposed
    console.print(
        Text(
            f"Exposure ledger {panel.exposure.ledger} ({panel.exposure.scheme}): "
            f"{len(exposed)} held-out {'lineage' if len(exposed) == 1 else 'lineages'} exposed; "
            "independence holds only relative to this ledger.",
            style="dim",
        )
    )
    for item in exposed:
        console.print(Text(exposure_line(item, panel), style="dim"))
    for limit in LIMITS:
        console.print(Text(f"Limit: {limit}", style="dim"))
    if panel.independent:
        console.print(
            Text(
                "Held-out lineages were reserved before learning; recorded facts come only "
                "from fit traces.",
                style="dim",
            )
        )
    else:
        console.print(
            f"[yellow]warning[/yellow] {len(panel.inspected)} of {len(panel.cases)} traces "
            "belong to exposed lineages; this is a development re-audit, not an independent "
            "measurement"
        )
    if verbose:
        audited = (
            "this lineage is" if len(panel.cases) == 1 else f"these {len(panel.cases)} lineages are"
        )
        console.print(
            Text(
                f"Per-trace detail shown: {audited} now development data, and later audits "
                "skip them.",
                style="dim",
            )
        )


def execute_runs(
    console: Console, store: ProjectStore, registrations: tuple[PreparedFidelity, ...]
) -> None:
    """Run the registered traces concurrently through the fidelity owner.

    Panel traces read frozen memory and never publish, so they are independent; each keeps its
    own ledger, session and progress row.
    """
    total = len(registrations)
    try:
        with progress_display(console) as progress, ThreadPoolExecutor(PANEL_CONCURRENCY) as pool:
            futures = [
                pool.submit(
                    execute_fidelity_evaluation,
                    store,
                    prepared,
                    progress=qualified(progress, f"trace {index}/{total}"),
                )
                for index, prepared in enumerate(registrations, start=1)
            ]
            for future in futures:
                future.result()
    except ProviderTransportError as exc:
        console.print(f"[red]error[/red] a provider request failed: {exc}")
        console.print(
            "Completed traces are saved. Run the same command again to continue; "
            "the stopped trace starts a fresh attempt."
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
