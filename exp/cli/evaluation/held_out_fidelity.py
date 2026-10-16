"""Held-out simulation fidelity in the existing evaluation terminal conventions."""

import io
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from time import monotonic

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from exp.cli.evaluation.fidelity import render_fidelity
from exp.cli.evaluation.fidelity_turns import render_turns, turn_marks
from exp.cli.evaluation.view import heading
from exp.cli.shared.consent import can_prompt, one_cost_warning, require_spend_consent
from exp.cli.shared.options import usage_error
from exp.cli.shared.progress import (
    Phase,
    duration_text,
    progress_display,
    progress_phases,
    qualified,
)
from exp.cli.shared.theme import EXP_THEME
from exp.common.progress import ProgressHook, report
from exp.common.project import ProjectStore
from exp.common.release_revision import installed_release_revision
from exp.optimize.evaluation.fidelity import (
    PreparedFidelity,
    execute_fidelity_evaluation,
    saved_fidelity_report,
)
from exp.optimize.evaluation.fidelity_recovery import stage_line
from exp.optimize.evaluation.held_out_audit import (
    HeldOutAudit,
    open_held_out_audit,
    outcome_unknown,
)
from exp.optimize.evaluation.held_out_exposure import LIMITS, LineageExposure
from exp.optimize.evaluation.held_out_fidelity import AuditCheckpoint, saved_held_out_results
from exp.optimize.evaluation.held_out_receipts import (
    QUALIFIED,
    AuditReceipt,
    AuditReport,
    ReceiptView,
    ReportKind,
    record_report,
)
from exp.optimize.evaluation.trace_learning import (
    fit_cases,
    learned_runs,
    learning_curve,
    learning_plan,
    recorded_plan,
)
from exp.optimize.evaluation.trace_replay import (
    DEADLINE,
    ReplayAggregate,
    ReplayResult,
    aggregate,
    interrupted,
)
from exp.runtime.models.providers.transport import ProviderTransportError
from exp.runtime.models.structured_completion import CompletionNotDispatched
from exp.simulation.mining.exposure_ledger import PanelInUseError

PANEL_CONCURRENCY = 4  # traces of a frozen panel replayed at once
PANEL_STAGE = "Replaying traces"  # counts finished panel traces
# The audit's single progress line: fit-only learning, then the held-out panel.
AUDIT_PHASES = (
    Phase("learning", "Learning fit traces", 0.6, driver="Learning from traces", unit="traces"),
    Phase("panel", "Replaying held-out traces", 0.4, driver=PANEL_STAGE, unit="traces"),
)
REAUDIT_TITLE = "Re-audit on inspected cases (development)"


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
    details: Console | None = None,
    terminal: Console | None = None,
    progress: ProgressHook | None = None,
    pause: Callable[[], AbstractContextManager[None]] | None = None,
) -> None:
    """Measure the learned simulator on held-out traces it never learned, or replay the result.

    Uses the hybrid profile: the audit's fit-only memory at its final checkpoint,
    with every held-out lineage reserved, plus recorded facts from the fit-only index. The
    panel's content is reserved for its audit receipt before any run is registered, and every
    invocation appends a report to that receipt before showing anything; ``--verbose`` shows
    per-trace detail, which exposes the audited lineages so later audits skip them.

    Args:
        project: Existing project with a completed build.
        count: Maximum held-out traces, selected deterministically.
        root: Existing project and artifact root.
        yes: Explicit authorization after the undefined-cost warning.
        non_interactive: Disable terminal questions.
        dry_run: Reserve and register the runs and stop before credentials or provider calls.
        verbose: Also print every trace, every turn, and every criterion.
        retry_incomplete: Resume interrupted runs in place: saved work replays, and a request
            lost in flight is sent again and reported as an unknown-outcome retry.
        include_inspected: Also re-run exposed or reserved lineages, as a development re-audit.
        details: Where the panel's description and notes are written; defaults to the terminal.
        terminal: The user's terminal, for results, warnings, the spend question and errors.
        progress: The calling command's single progress line; without it, the panel owns one.
        pause: Releases that progress line while the spend question is asked.
    """
    screen = terminal or Console(theme=EXP_THEME)
    console = details or screen
    with usage_error(OSError, ValueError, PanelInUseError):
        store = ProjectStore(root, project)
        clock = lambda: datetime.now(UTC)  # noqa: E731 - one shared aware clock
        with open_held_out_audit(
            store,
            count,
            profile="memory",
            code_revision=installed_release_revision(),
            clock=clock,
            include_inspected=include_inspected,
        ) as audit:
            _run_audit(
                screen,
                console,
                store,
                audit,
                command=f"exp eval {project} --audit {count}",
                root=root,
                yes=yes,
                non_interactive=non_interactive,
                dry_run=dry_run,
                verbose=verbose,
                resume=retry_incomplete,
                clock=clock,
                progress=progress,
                pause=pause,
            )


def _run_audit(
    screen: Console,
    console: Console,
    store: ProjectStore,
    audit: HeldOutAudit,
    *,
    command: str,
    root: Path,
    yes: bool,
    non_interactive: bool,
    dry_run: bool,
    verbose: bool,
    resume: bool,
    clock: Callable[[], datetime],
    progress: ProgressHook | None,
    pause: Callable[[], AbstractContextManager[None]] | None,
) -> None:
    """Execute or replay an open audit, recording its report before showing anything.

    An interrupted trace is refused before spend consent unless ``resume`` continues it in
    place under this same receipt. ``screen`` receives the result, warnings, the spend question
    and errors; ``console`` receives the panel's description and notes (the same terminal when
    the panel runs on its own).
    """
    panel, receipt = audit.panel, audit.receipt
    prior = audit.view

    def record(kind: ReportKind, *, detail: bool = False) -> AuditReport:
        """Append this invocation's report, and any exposure, before anything it covers shows."""
        return record_report(store, receipt, panel, kind=kind, detail_shown=detail, clock=clock)

    if console is screen:
        heading(screen, store.paths.project_id, "Held-out simulation fidelity")
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
            f"panel {receipt.panel} · receipt #{receipt.sequence} {receipt.receipt_id} · "
            f"exposure ledger {receipt.ledger} ({schemes_label(receipt)})",
            style="dim",
        )
    )
    if audit.excluded:
        console.print(Text(skipped_line(audit.excluded), style="dim"))
    for line in attempt_lines(prior):
        console.print(Text(line, style="dim"))
    if outcome_unknown(store, audit):
        screen.print(
            "[yellow]warning[/yellow] dispatched, outcome unknown: a run claimed provider "
            "slots and stopped; only this receipt can complete it, with --retry-incomplete"
        )
    fit = len(fit_cases(store))
    # Count learned traces, not versions: an identical repeat trace publishes no version.
    learned = (
        len(learned_runs(store, learning_plan(store, scope="fit")))
        if recorded_plan(store, "fit")
        else 0
    )
    if version is not None and learned < fit:
        screen.print(
            f"[yellow]warning[/yellow] memory v{version} learned {learned} of {fit} fit "
            "traces; run the audit again to finish learning and measure the fully learned simulator"
        )
    if dry_run:
        record("prepared")
        for prepared in panel.prepared:
            console.print(Text(f"  {prepared.spec.run_id}"))
        console.print("Prepared without provider calls. Run the same command without --dry-run.")
        return
    pending = [p for p in panel.prepared if saved_fidelity_report(store, p) is None]
    if not pending:
        console.print(Text("Saved results. No provider calls.", style="dim"))
        # The lifecycle owner replays each report and completes any interrupted publication.
        for prepared in panel.prepared:
            execute_fidelity_evaluation(store, prepared)
    else:
        try:
            require_resumable(store, panel.prepared, resume=resume)
        except ValueError:
            record("stopped")
            raise
        interactive = not non_interactive and can_prompt(screen)
        with pause() if pause is not None else nullcontext():
            consented = require_spend_consent(
                screen,
                root=root,
                yes=yes,
                estimated_cost_usd=None,
                cost_is_upper_bound=False,
                command=command,
                non_interactive=not interactive,
            )
        if not consented:
            record("stopped")
            return
        try:
            execute_runs(screen, store, panel.prepared, progress, guard=audit.guard, resume=resume)
        except BaseException:
            record("stopped")
            raise
    results = saved_held_out_results(store, panel)
    audit.reassess(clock)
    kind: ReportKind = "replay" if not pending and prior.reported else "evaluation"
    record(kind, detail=verbose)
    view = audit.view
    if verbose:
        render_traces(screen, store, panel.prepared, results, "Held-out trace")
    lines = [aggregate_line(aggregate(results), len(panel.cases), title(audit, view))]
    if any(view.redispatched.values()):
        lines.append(stage_line(view.redispatched, len(view.affected_traces), len(panel.cases)))
    lines.extend(sensitivity_lines(view, results))
    for target in (screen, console) if console is not screen else (screen,):
        target.print()
        for line in lines:
            target.print(Text(line))
    curve = learning_curve(store, "fit")
    if curve is not None:
        console.print(Text(curve.line()))
    render_exposure(screen, console, audit, view, verbose=verbose)


def title(audit: HeldOutAudit, view: ReceiptView) -> str:
    """The aggregate's label: never plain independent once exposed, re-audited or re-attempted."""
    if audit.receipt.reaudited:
        return REAUDIT_TITLE
    if view.receipt.basis == "development-re-audit":
        return "Held-out re-audit (development)"
    if view.headline == QUALIFIED:
        return "Held-out fidelity (independent, qualified: trace re-attempts)"
    return "Held-out fidelity"


def sensitivity_lines(view: ReceiptView, results: tuple[ReplayResult, ...]) -> tuple[str, ...]:
    """With re-attempted traces, the score over unaffected traces and over affected ones."""
    affected = view.affected_traces
    if not affected:
        return ()
    planned = len(view.receipt.cases)
    unaffected = tuple(result for result in results if result.trace_id not in affected)
    touched = tuple(result for result in results if result.trace_id in affected)
    return (
        aggregate_line(aggregate(unaffected), planned - len(affected), "  Unaffected traces only"),
        aggregate_line(aggregate(touched), len(affected), "  Re-attempted traces only"),
    )


def attempt_lines(view: ReceiptView) -> tuple[str, ...]:
    """Every replaced or resumed registration of the receipt, its reason and claimed slots."""
    return tuple(
        f"{attempt.reason} of {attempt.trace_id}: {attempt.retired_run} resumed in place "
        f"({attempt.claimed_slots} claimed slots); "
        + "; ".join(row.line() for row in attempt.redispatched)
        if attempt.in_place
        else f"{attempt.reason} of {attempt.trace_id}: {attempt.retired_run} -> "
        f"{attempt.replacement_run} ({attempt.claimed_slots} claimed slots)"
        for attempt in view.attempts
    )


def skipped_line(excluded: int) -> str:
    """Why the default panel left out exposed, reserved or unresolved held-out lineages."""
    noun, pronoun = ("lineage", "it is") if excluded == 1 else ("lineages", "they are")
    return (
        f"Skipped {excluded} held-out {noun} exposed, reserved or unresolved in the exposure "
        "ledger (shown in detail by an earlier audit, grouped with content that was, held by "
        f"another audit, or not fingerprintable); {pronoun} not independent "
        "(--include-inspected re-runs them)"
    )


def receipt_line(view: ReceiptView) -> str:
    """The receipt's sequence, ID and headline, and what this invocation was."""
    receipt = view.receipt
    last = view.reports[-1].kind if view.reports else "opened"
    status = f"replay ({view.headline})" if last == "replay" else view.headline
    follows = f" · follows {receipt.follows} ({receipt.follow_reason})" if receipt.follows else ""
    return (
        f"Audit receipt #{receipt.sequence} {receipt.receipt_id}: {status} · "
        f"this run: {last} · code {receipt.code_revision[:12]}{follows}"
    )


def schemes_label(receipt: AuditReceipt) -> str:
    """Every fingerprint scheme the match used, with its definition and vector digests."""
    return ", ".join(
        f"{scheme.name} v{scheme.version} definition {scheme.definition_sha256[:12]} "
        f"vectors {scheme.vectors_sha256[:12]}"
        for scheme in receipt.schemes
    )


def exposure_line(item: LineageExposure, audit: HeldOutAudit) -> str:
    """One excluded lineage: whether audited, its component size and path, or why unresolved."""
    where = "in this panel" if item.lineage in audit.receipt.lineages else "skipped"
    if item.path is None:
        return (
            f"  {item.lineage} ({item.kind}, {where}): members "
            f"{', '.join(item.unresolved)} cannot be fingerprinted, so its exposure history is "
            "unresolved"
        )
    hops = [
        (
            f"same task {edge.task_lineage}"
            if edge.reason == "declared-task-identity"
            else f"{edge.duplicate_kind} near-duplicate {edge.similarity:.2f}"
        )
        + f" ({edge.project}/{edge.build})"
        for edge in item.path.path
    ]
    route = " → ".join(hops) if hops else "its own content"
    return (
        f"  {item.lineage} ({item.kind}, {where}): component of {item.path.component_size} · "
        f"{route} · marked {item.path.exposed_node[:12]}"
    )


def checkpoint_lines(checkpoint: AuditCheckpoint, simulator: str) -> tuple[str, str]:
    """The audited checkpoint, and that an all-traces build is a different one."""
    return (
        f"Audited checkpoint: memory {checkpoint.lineage} v{checkpoint.version}, "
        f"scope {checkpoint.scope}, {checkpoint.fit_traces_learned} fit traces learned, "
        f"simulator {simulator} (held-out lineages reserved before learning)",
        "The build's all-traces memory is a different checkpoint "
        "and was not independently evaluated.",
    )


def render_exposure(
    screen: Console, console: Console, audit: HeldOutAudit, view: ReceiptView, *, verbose: bool
) -> None:
    """What was audited, the receipt, the exposure decision and its limits, and what was shown.

    Warnings, and the note that per-trace detail exposed the audited lineages, go to ``screen``;
    the rest goes to ``console``.
    """
    receipt = view.receipt
    if receipt.checkpoint is not None:
        for line in checkpoint_lines(receipt.checkpoint, receipt.world_model.artifact_id):
            console.print(Text(line, style="dim"))
    console.print(Text(receipt_line(view), style="dim"))
    exposed = receipt.exposed
    console.print(
        Text(
            f"Exposure ledger {receipt.ledger} ({schemes_label(receipt)}): {len(exposed)} "
            f"held-out {'lineage' if len(exposed) == 1 else 'lineages'} exposed, reserved or "
            "unresolved; "
            "independence holds only relative to this ledger.",
            style="dim",
        )
    )
    for item in exposed:
        console.print(Text(exposure_line(item, audit), style="dim"))
    for notice in view.notices:
        screen.print(
            f"[yellow]warning[/yellow] invalidation notice: links recorded after this receipt "
            f"reserved its panel reach exposure that predates it on {len(notice.exposed)} "
            f"audited {'lineage' if len(notice.exposed) == 1 else 'lineages'} "
            f"(ledger {notice.ledger}); the receipt and its snapshot are unchanged"
        )
    for limit in LIMITS:
        console.print(Text(f"Limit: {limit}", style="dim"))
    if receipt.reaudited:
        screen.print(
            f"[yellow]warning[/yellow] {len(receipt.reaudited)} of {len(receipt.cases)} traces "
            "belong to exposed or reserved lineages; this is a development re-audit, not an "
            "independent measurement"
        )
    else:
        console.print(
            Text(
                "Held-out lineages were reserved before learning; recorded facts come only "
                "from fit traces.",
                style="dim",
            )
        )
    if verbose:
        audited = (
            "this lineage is"
            if len(receipt.cases) == 1
            else f"these {len(receipt.cases)} lineages are"
        )
        screen.print(
            Text(
                f"Per-trace detail shown: {audited} now development data, and later audits "
                "skip them.",
                style="dim",
            )
        )


def execute_runs(
    console: Console,
    store: ProjectStore,
    registrations: tuple[PreparedFidelity, ...],
    progress: ProgressHook | None = None,
    *,
    guard: Callable[[], None] | None = None,
    resume: bool = False,
) -> None:
    """Run the registered traces concurrently through the fidelity owner.

    Panel traces read frozen memory and never publish, so they are independent; each keeps its
    own ledger and session. ``progress`` receives every trace's events qualified by ``trace
    i/n``, plus ``Replaying traces k/n`` as traces finish; without it, the panel owns a display.
    Finished traces replay untouched; with ``resume``, each interrupted trace continues in place
    from its first uncommitted step.

    Raises:
        ValueError: A trace was interrupted and ``resume`` was not given; nothing is sent.
    """
    total = len(registrations)
    # Once any trace fails or the user interrupts, no further trace starts; running ones finish.
    stopped = Event()
    resumed = require_resumable(store, registrations, resume=resume)
    if resumed:
        console.print(
            Text(
                f"{resumed} interrupted trace(s) resume in place; saved steps replay "
                "without provider calls",
                style="dim",
            )
        )
    window = datetime.now(UTC) + DEADLINE if resume else None
    try:
        with (
            nullcontext(progress) if progress is not None else progress_display(console) as hook,
            ThreadPoolExecutor(PANEL_CONCURRENCY) as pool,
        ):

            def replay(index: int, prepared: PreparedFidelity) -> None:
                """Replay one trace unless the panel has already stopped."""
                if stopped.is_set():
                    return
                try:
                    execute_fidelity_evaluation(
                        store,
                        prepared,
                        progress=qualified(hook, f"trace {index}/{total}"),
                        dispatch_guard=guard,
                        resume_until=window,
                    )
                except BaseException:
                    stopped.set()
                    raise

            report(hook, PANEL_STAGE, completed=0, total=total)
            futures = [
                pool.submit(replay, index, prepared)
                for index, prepared in enumerate(registrations, start=1)
            ]
            try:
                for finished, future in enumerate(as_completed(futures), start=1):
                    future.result()
                    report(hook, PANEL_STAGE, completed=finished, total=total)
            except BaseException:
                stopped.set()
                for future in futures:
                    future.cancel()
                raise
    except (ProviderTransportError, CompletionNotDispatched) as exc:
        console.print(f"[red]error[/red] a provider request failed: {exc}")
        console.print(
            "Completed traces are saved. The stopped trace is never bought again silently: "
            "run the same command with --retry-incomplete to resume it."
        )
        raise typer.Exit(code=1) from exc


def require_resumable(
    store: ProjectStore, registrations: tuple[PreparedFidelity, ...], *, resume: bool
) -> int:
    """Count interrupted traces, refusing them before any consent or call without ``resume``.

    Raises:
        ValueError: A trace was interrupted and ``resume`` was not given.
    """
    stopped = sum(interrupted(store, prepared) for prepared in registrations)
    if stopped and not resume:
        raise ValueError(
            f"{stopped} trace(s) were interrupted before saving a result; run the same "
            "command with --retry-incomplete to resume them in place"
        )
    return stopped


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


@one_cost_warning()
def run_audit(
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
    """Independently audit the simulator on traces it never learned.

    The default build learns every trace, so the audit learns its own memory from fit traces
    only, with held-out lineages reserved first; saved results replay without calls. It then
    replays up to ``count`` held-out traces on that memory's final checkpoint.

    Args:
        project: Existing project with a completed build.
        count: Maximum held-out traces, selected deterministically.
        root: Existing project and artifact root.
        yes: Explicit authorization after the undefined-cost warning.
        non_interactive: Disable terminal questions.
        dry_run: Stop before credentials or provider calls.
        verbose: Also print every trace, every turn, and every criterion.
        retry_incomplete: Resume interrupted runs in place.
        include_inspected: Also re-run exposed or reserved lineages, as a development re-audit.
    """
    # Deferred: learning loads the fidelity lifecycle.
    from exp.cli.build.learning import BuildLearning, resolve_memory_recall, run_learning
    from exp.cli.build.summary import learning_lines, save_details
    from exp.common.models import load_model_catalog

    started = monotonic()
    console = Console(theme=EXP_THEME)
    with usage_error(OSError, ValueError):
        store = ProjectStore(root, project)
        if dry_run and not recorded_plan(store, "fit"):
            console.print(
                "The audit first learns a separate memory from fit traces only, then replays "
                "held-out traces on it. Run without --dry-run to see its preflight."
            )
            return
    if dry_run:
        run_held_out_fidelity(
            project,
            count,
            root=root,
            yes=yes,
            non_interactive=non_interactive,
            dry_run=True,
            verbose=verbose,
            retry_incomplete=retry_incomplete,
            include_inspected=include_inspected,
        )
        return
    with usage_error(OSError, ValueError):
        recall = resolve_memory_recall(load_model_catalog(store.model_catalog_path))
        if recall is None:
            raise ValueError(
                "the audit learns a fit-only memory and needs a native decision model to "
                "recall it (for example cloudflare/clef-flash); add one to the model catalog"
            )
    heading(console, project, "Held-out simulation fidelity")
    buffer = io.StringIO()
    details = Console(file=buffer, width=110, theme=EXP_THEME, no_color=True)
    with progress_phases(console, AUDIT_PHASES) as pipeline:
        with usage_error(OSError, ValueError):
            outcome = run_learning(
                details,
                store,
                project=project,
                root=root,
                recall_alias=recall,
                maximum_attempts=None,
                yes=yes,
                non_interactive=non_interactive,
                code_revision=installed_release_revision(),
                retry_incomplete=retry_incomplete,
                hold_out=True,
                terminal=console,
                progress=pipeline.phase("learning"),
                pause=pipeline.paused,
            )
        if outcome is None or not recorded_plan(store, "fit"):
            return
        pipeline.complete("learning")
        learned = BuildLearning(recall, outcome, learning_curve(store, "fit"))
        for line in learning_lines(learned, "fit traces"):
            console.print(line)
        run_held_out_fidelity(
            project,
            count,
            root=root,
            yes=yes,
            non_interactive=non_interactive,
            dry_run=False,
            verbose=verbose,
            retry_incomplete=retry_incomplete,
            include_inspected=include_inspected,
            details=details,
            terminal=console,
            progress=pipeline.phase("panel"),
            pause=pipeline.paused,
        )
    console.print(
        Text(f"Report: {save_details(store, 'audits', outcome.plan.lineage, buffer.getvalue())}")
    )
    console.print(f"[green]Complete[/green] [dim]in {duration_text(monotonic() - started)}[/dim]")
