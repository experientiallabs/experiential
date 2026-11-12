"""The ``exp variance run`` and ``exp variance watch`` flows (imported only when invoked).

They follow ``exp build``: a preflight block, one consent, then the one shared progress line
(:class:`~exp.cli.shared.progress.ProgressDisplay`) while the front door runs, and a few result
lines at the end. The full detail (every phase with its time, each start's pipeline stages,
calls per role and tier, the attempt cap, the outcome) is saved to a report file beside the
run, like ``exp eval``'s reports; ``exp variance watch --once`` prints the same detail.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.text import Text

from exp.cli.evaluation.front_door_action import front_door_lines
from exp.cli.shared.consent import can_prompt, require_spend_consent
from exp.cli.shared.progress import duration_text, progress_display
from exp.cli.variance import launch
from exp.cli.variance.runs import (
    RunRecord,
    finish_record,
    locate,
    now_iso,
    process_alive,
    save_record,
    spawn_detached,
)
from exp.cli.variance.sources import (
    Traffic,
    admission_path,
    journal_segments,
    outcome,
    parse_time,
    pipeline_stages,
    read_admission,
    segment_after,
)
from exp.cli.variance.view import FrontDoorView, progress_view
from exp.common.progress import report
from exp.common.project import ProjectStore
from exp.common.project.database import content_database_path
from exp.optimize.evaluation.variance.frontdoor import (
    FrontDoorPlan,
    FrontDoorReport,
    FrontDoorSpec,
    plan_front_door,
    run_front_door,
)
from exp.optimize.evaluation.variance.frontdoor_states import StageState

FINAL = frozenset({"finished", "stopped", "failed"})
STORE_EVERY = 15.0
"""Seconds between project-store probes of the pipelines' saved stages."""


@dataclass(frozen=True)
class RunOptions:
    """Everything ``exp variance run`` was asked to do (see its option help)."""

    project: str
    spec: Path
    root: Path
    dry_run: bool = False
    yes: bool = False
    consent: str | None = None
    non_interactive: bool = False
    detach: bool = False
    log: Path | None = None
    code_revision: str | None = None

    def rerun(self, digest: str) -> list[str]:
        """The ``exp`` arguments that run exactly this plan without a question."""
        arguments = [
            "variance", "run", self.project, str(self.spec.resolve()),
            "--root", str(self.root.resolve()), "--consent", digest, "--non-interactive",
        ]  # fmt: skip
        if self.code_revision is not None:
            arguments += ["--code-revision", self.code_revision]
        return arguments


def run(console: Console, options: RunOptions) -> None:
    """Preflight, plan, consent once by digest, then run (in this process or detached)."""
    store = ProjectStore(options.root, options.project)
    spec = launch.load_spec(options.spec)
    aliases = launch.check_spec(spec)
    revision = options.code_revision or launch.source_revision()
    view = FrontDoorView(title="front door", project=options.project)
    dependencies = launch.front_door_dependencies(
        store, spec, revision, root=options.root, announce=view.announce, progress=view.observe
    )
    render_preflight(console, options, launch.slot_names(spec), revision, aliases)
    with progress_display(console) as hook:
        report(hook, "Planning (reads the store, no model call)")
        plan = plan_front_door(store, spec, code_revision=revision, dependencies=dependencies)
    if options.dry_run:
        for line in plan_view(plan):
            console.print(Text(line))
        console.print("[green]dry run complete[/green] No provider calls. To run this plan:")
        console.print(Text(f"  exp {' '.join(options.rerun(plan.digest))}"))
        return
    render_plan(console, plan)
    rerun = "exp " + " ".join(options.rerun(plan.digest))
    if not plan.complete and not consented(console, options, plan, rerun):
        return
    view.set_plan([row.stage for row in plan.stages])
    view.title = f"front door {plan.digest[:12]}"
    if options.detach:
        detach(console, options, store, spec, plan, revision)
        return
    run_here(console, options, store, spec, plan, revision, dependencies, view)


def render_preflight(
    console: Console,
    options: RunOptions,
    slots: list[str],
    revision: str,
    aliases: dict[str, str | None],
) -> None:
    """The ``exp build``-style preflight block."""
    console.print("[bold]Front-door preflight[/bold]")
    console.print(f"  [dim]spec[/dim]         {options.spec}")
    console.print(f"  [dim]code[/dim]         {revision}")
    console.print(f"  [dim]decisions[/dim]    clef ({len(aliases)} roles); no astra_alias")
    console.print(f"  [dim]slots[/dim]        {', '.join(slots)}")
    console.print(
        f"  [dim]spend[/dim]        at most ${launch.DEFAULT_CEILING_USD:.2f} and "
        f"{launch.DEFAULT_REAL_ATTEMPTS:,} real attempts"
    )


def render_plan(console: Console, plan: FrontDoorPlan) -> None:
    """The plan in three lines: what replays, what will run and what waits on it."""

    def named(state: StageState) -> list[str]:
        """Stages in one state."""
        return [row.stage for row in plan.stages if row.status == state]

    pending = [row for row in plan.stages if row.status == StageState.PLANNED]
    requests = sum(line.attempts for row in pending for line in row.costs)
    console.print(f"[bold]Front-door plan[/bold] [dim]{plan.digest}[/dim]")
    console.print(f"  [dim]complete[/dim]     {', '.join(named(StageState.DONE)) or 'none'}")
    console.print(
        f"  [dim]to run[/dim]       {', '.join(named(StageState.PLANNED)) or 'none'} "
        f"[dim](at most {requests:,} physical requests)[/dim]"
    )
    console.print(f"  [dim]waiting[/dim]      {', '.join(named(StageState.BLOCKED)) or 'none'}")


def plan_view(plan: FrontDoorPlan) -> list[str]:
    """The full plan (``--dry-run``): every stage, completed, remaining and unpriced work."""
    lines = list(front_door_lines(plan))
    if "State machines (Mermaid):" in lines:
        lines = lines[: lines.index("State machines (Mermaid):")]
    lines.append(f"Code revision {plan.code_revision}")
    lines.append(f"Plan digest {plan.digest}")
    return lines


def consented(console: Console, options: RunOptions, plan: FrontDoorPlan, rerun: str) -> bool:
    """One consent keyed by the plan digest: ``--consent``, ``--yes`` or one question.

    Raises:
        ValueError: ``--consent`` names another plan.
        typer.Exit: Non-interactive with neither ``--consent`` nor ``--yes``.
    """
    if options.consent is not None:
        if not launch.matches(options.consent, plan.digest):
            raise ValueError(
                f"--consent {options.consent} does not name this plan ({plan.digest}); the "
                "store or code changed since it was planned. Review the plan and consent again"
            )
        return True
    return require_spend_consent(
        console,
        root=options.root,
        yes=options.yes,
        estimated_cost_usd=min(launch.priced_bound(plan), launch.DEFAULT_CEILING_USD),
        command=rerun,
        non_interactive=options.non_interactive,
        cost_is_upper_bound=False,
    )


def detach(
    console: Console,
    options: RunOptions,
    store: ProjectStore,
    spec: FrontDoorSpec,
    plan: FrontDoorPlan,
    revision: str,
) -> None:
    """Start the consented plan in its own session and record it; in a terminal, watch it."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log = store.paths.runtime_directory / "front-door-cli" / f"{plan.digest[:16]}-{stamp}.log"
    pid = spawn_detached([*options.rerun(plan.digest), "--log", str(log)], log)
    record = RunRecord(
        digest=plan.digest,
        journal=launch.journal_name(store, spec),
        operation=launch.operation_name(plan.digest),
        spec=str(options.spec.resolve()),
        revision=revision,
        pid=pid,
        log=str(log),
        started=now_iso(),
    )
    save_record(store, record)
    console.print(
        f"[green]✓[/green] detached as pid {pid} "
        "[dim](closing this terminal keeps it running)[/dim]"
    )
    console.print(Text(f"Log: {log}"))
    console.print(f"next: exp variance watch {options.project} {plan.digest[:12]}")
    if not options.non_interactive and can_prompt(console):
        watch(console, project=options.project, root=options.root, run=plan.digest, once=False)


def run_here(
    console: Console,
    options: RunOptions,
    store: ProjectStore,
    spec: FrontDoorSpec,
    plan: FrontDoorPlan,
    revision: str,
    dependencies: object,
    view: FrontDoorView,
) -> None:
    """Install the ceiling and run the front door in this process under the progress line."""
    from exp.cli.evaluation.front_door_action import front_door_summary
    from exp.cli.evaluation.learning_summary import save_details
    from exp.runtime.models.spend_ceiling import operation_spend_scope

    record = RunRecord(
        digest=plan.digest,
        journal=launch.journal_name(store, spec),
        operation=launch.operation_name(plan.digest),
        spec=str(options.spec.resolve()),
        revision=revision,
        pid=os.getpid(),
        log=None if options.log is None else str(options.log),
        started=now_iso(),
    )
    started = time.monotonic()
    with operation_spend_scope():
        approval = None
        if not plan.complete:
            launch.install_ceiling(
                store,
                plan,
                ceiling_usd=launch.DEFAULT_CEILING_USD,
                real_attempts=launch.DEFAULT_REAL_ATTEMPTS,
                command="exp " + " ".join(options.rerun(plan.digest)),
            )
            approval = launch.approval(plan, launch.DEFAULT_ADMISSION_ATTEMPTS)
        save_record(store, record)
        refresh = refresher(options.root, options.project, view, record)
        try:
            with progress_view(console, view, refresh):
                result = run_front_door(
                    store,
                    spec,
                    code_revision=revision,
                    approval=approval,
                    dependencies=dependencies,  # ty: ignore[invalid-argument-type]
                )
                records = {
                    row.task_id: {stage: ref.artifact_id for stage, ref in row.stages.items()}
                    for row in result.pipelines
                }
                view.update(stopped=result.stopped, outcome=outcome(store, records))
        except BaseException as error:
            finish_record(store, record, "failed", f"{type(error).__name__}: {error}"[:2000])
            raise
    finish_record(
        store, record, "finished" if result.stopped is None else "stopped", result.stopped
    )
    finish(console, options, store, spec, view, result, front_door_summary, save_details, started)


def finish(  # noqa: PLR0913 - the closing lines need each of these
    console: Console,
    options: RunOptions,
    store: ProjectStore,
    spec: FrontDoorSpec,
    view: FrontDoorView,
    result: FrontDoorReport,
    summary: Callable[[FrontDoorReport], object],
    save: Callable[[ProjectStore, str, str, str], Path],
    started: float,
) -> None:
    """The result lines, the saved report and how long it took, as ``exp build`` ends."""
    import json

    for line in view.result_lines():
        console.print(line)
    text = "\n".join([*view.lines(), "", json.dumps(summary(result), indent=2), ""])
    details = save(store, "front-door", launch.journal_name(store, spec), text)
    console.print(Text(f"Report: {details}"))
    took = duration_text(time.monotonic() - started)
    if result.stopped is None:
        console.print(f"[green]Complete[/green] [dim]in {took}[/dim]")
        return
    console.print(Text(f"Stopped: {result.stopped}", style="yellow"))
    console.print(
        f"next: exp variance run {options.project} {options.spec}  "
        "[dim](resumes; saved work replays)[/dim]"
    )
    raise typer.Exit(code=2)


def refresher(
    root: Path, project_id: str, view: FrontDoorView, record: RunRecord | None, *,
    journal: Path | None = None, since: datetime | None = None, ledger: Path | None = None,
) -> Callable[[], None]:  # fmt: skip
    """Re-read every durable source into the view; the store is probed every 15 seconds.

    The view thread opens its own project store, never sharing the run's connections.
    """
    runtime = root / "projects" / project_id / "runtime"
    if record is not None:
        journal = journal or runtime / "front-door" / record.journal / "transitions.jsonl"
        since = since or parse_time(record.started)
        ledger = ledger or runtime / "spend" / record.operation / "ledger.sqlite"
    traffic = None if ledger is None else Traffic(ledger, content_database_path(root), project_id)
    admission = admission_path()
    state: dict[str, object] = {"store": None, "probed": 0.0}

    def refresh() -> None:
        """One pass over the sources."""
        segment = None if journal is None else segment_after(journal_segments(journal), since)
        if traffic is not None:
            with view.guard():
                traffic.refresh()
        values: dict[str, object] = {
            "segment": segment,
            "traffic": traffic,
            "admission": read_admission(admission),
        }
        if segment is not None:
            values["note"] = ""
        elif not view.note:
            values["note"] = "waiting for the run's journal (a run re-plans first; no model call)"
        tasks = view.tasks()
        if tasks and time.monotonic() - float(state["probed"]) >= STORE_EVERY:  # ty: ignore
            if state["store"] is None:
                state["store"] = ProjectStore(root, project_id)
            values["starts"] = pipeline_stages(state["store"], tasks, since)  # ty: ignore
            state["probed"] = time.monotonic()
        view.update(**values)

    return refresh


def watch(
    console: Console,
    *,
    project: str,
    root: Path,
    run: str | None,
    once: bool,
    interval: float = 2.0,
) -> None:
    """Reopen the progress line on one run until it ends; Ctrl-C leaves the run running.

    With ``once``, print the run's full detail (the report's content) instead.
    """
    store = ProjectStore(root, project)
    located = locate(store, run)
    view = FrontDoorView(title=f"front door {located.name}", project=project)
    if located.record is not None and located.record.started:
        moment = parse_time(located.record.started)
        if moment is not None:
            view.started = moment.timestamp()
    refresh = refresher(
        root, project, view, located.record,
        journal=located.journal, since=located.since, ledger=located.ledger,
    )  # fmt: skip
    refresh = _with_outcome(refresh, view, store, located.record)
    if once:
        _settle(refresh)
        for line in view.lines():
            console.print(Text(line), soft_wrap=True)
        return
    try:
        with progress_view(console, view, refresh, interval=interval):
            while not _ended(view, located.record):
                time.sleep(interval)
    except KeyboardInterrupt:
        console.print("[dim]stopped watching; the run continues[/dim]")
        return
    for line in view.result_lines():
        console.print(line)
    if view.note:
        console.print(Text(view.note, style="yellow"))
    console.print(
        f"next: exp variance watch {project} {located.name} --once  [dim](full detail)[/dim]"
    )


def _settle(refresh: Callable[[], None]) -> None:
    """Refresh until the store probe has run once (it needs the journal's starts first)."""
    refresh()
    refresh()


def _with_outcome(
    refresh: Callable[[], None], view: FrontDoorView, store: ProjectStore, record: RunRecord | None
) -> Callable[[], None]:
    """Also summarize the saved pipelines once the run has ended, and say why it ended."""

    def wrapped() -> None:
        """Refresh, then add the outcome of an ended run."""
        refresh()
        ready = bool(view.starts) or not view.tasks()
        if _ended(view, record) and view.outcome is None and ready:
            records = {task: dict(start.saved) for task, start in view.starts.items()}
            view.update(outcome=outcome(store, records))
            if record is not None and record.stopped:
                view.update(stopped=record.stopped)

    return wrapped


def _last_activity(view: FrontDoorView) -> float:
    """The last journal move or call reservation of a run."""
    moments = [0.0]
    if view.segment is not None:
        for row in view.segment.phases.values():
            for moment in (row.started, row.ended):
                if moment is not None:
                    moments.append(moment.timestamp())
    if view.traffic is not None:
        moments.extend(call.reserved for call in view.traffic.calls.values())
    return max(moments) or time.time()


def _ended(view: FrontDoorView, record: RunRecord | None) -> bool:
    """Whether the run reached a final state or its process is gone."""
    state = view.run_state()
    if state in FINAL:
        return True
    pids = set()
    if record is not None:
        pids.add(record.pid)
    if view.traffic is not None:
        pids |= view.traffic.pids()
    if pids and not any(process_alive(pid) for pid in pids):
        if view.segment is not None and view.segment.state not in FINAL:
            view.update(
                note=f"no live process ({', '.join(map(str, sorted(pids)))} gone); the run was "
                "interrupted",
                interrupted=_last_activity(view),
            )
        return True
    return False
