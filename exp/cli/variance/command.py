"""The ``exp variance run`` and ``exp variance watch`` flows (imported only when invoked)."""

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
from exp.cli.shared.consent import require_spend_consent
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
from exp.cli.variance.view import FrontDoorView, live_view
from exp.common.config.dotenv import load_env_file
from exp.common.progress import report
from exp.common.project import ProjectStore
from exp.common.project.database import content_database_path
from exp.optimize.evaluation.variance.frontdoor import (
    FrontDoorPlan,
    FrontDoorReport,
    FrontDoorSpec,
    plan_front_door,
    run_directory,
    run_front_door,
)

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
    watch: bool = True
    log: Path | None = None
    code_revision: str | None = None
    ceiling_usd: float = launch.DEFAULT_CEILING_USD
    real_attempts: int = launch.DEFAULT_REAL_ATTEMPTS
    admission_attempts: int = launch.DEFAULT_ADMISSION_ATTEMPTS
    statecharts: bool = False

    def rerun(self, digest: str) -> list[str]:
        """The ``exp`` arguments that run exactly this plan without a question."""
        arguments = [
            "variance", "run", self.project, str(self.spec.resolve()),
            "--root", str(self.root.resolve()), "--consent", digest, "--non-interactive",
            "--ceiling-usd", repr(self.ceiling_usd), "--real-attempts", str(self.real_attempts),
            "--admission-attempts", str(self.admission_attempts),
        ]  # fmt: skip
        if self.code_revision is not None:
            arguments += ["--code-revision", self.code_revision]
        return arguments


def load_env(path: Path) -> None:
    """Load one KEY=VALUE file without overriding the environment; values are never shown."""
    if not path.is_file():
        raise ValueError(f"env file {path} does not exist")
    load_env_file(path)


def run(console: Console, options: RunOptions) -> None:
    """Preflight, plan, consent once by digest, then run (in process or detached)."""
    store = ProjectStore(options.root, options.project)
    spec = FrontDoorSpec.model_validate_json(options.spec.read_bytes())
    aliases = launch.check_spec(spec)
    revision = options.code_revision or launch.source_revision()
    view = FrontDoorView(title="front door (planning)", project=options.project)
    dependencies = launch.front_door_dependencies(
        store, spec, revision, root=options.root, announce=view.announce, progress=view.observe
    )
    render_preflight(console, options, spec, revision, aliases)
    started = time.monotonic()
    with progress_display(console) as hook:
        report(hook, "planning: reading the store, no model call")
        plan = plan_front_door(store, spec, code_revision=revision, dependencies=dependencies)
    console.print(f"[dim]planned in {duration_text(time.monotonic() - started)}[/dim]")
    for line in plan_view(plan, statecharts=options.statecharts):
        console.print(Text(line))
    rerun = "exp " + " ".join(options.rerun(plan.digest))
    if options.dry_run:
        console.print(
            "[green]plan only[/green] No consent, no model call. To run exactly this plan:"
        )
        console.print(Text(f"  {rerun}"))
        return
    if not plan.complete and not consented(console, options, plan, rerun):
        console.print("Not consented; nothing ran.")
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
    spec: FrontDoorSpec,
    revision: str,
    aliases: dict[str, str | None],
) -> None:
    """The ``exp build``-style preflight block."""
    from exp.optimize.evaluation.variance.frontdoor import configured_slots

    console.print("[bold]Front-door preflight[/bold]")
    console.print(Text.assemble(("  spec        ", "dim"), str(options.spec)))
    console.print(Text.assemble(("  project     ", "dim"), f"{options.project} at {options.root}"))
    console.print(Text.assemble(("  code        ", "dim"), revision))
    console.print(
        Text.assemble(
            ("  decisions   ", "dim"),
            f"clef ({', '.join(where.split('.')[-2] for where in aliases)})",
        )
    )
    console.print(Text.assemble(("  roles       ", "dim"), "open-weight only (no astra_alias)"))
    console.print(Text.assemble(("  slots       ", "dim"), ", ".join(configured_slots(spec))))
    console.print(
        Text.assemble(
            ("  spend       ", "dim"),
            f"at most ${options.ceiling_usd:,.2f} real and {options.real_attempts:,} real "
            f"attempts; phases admitted up to {options.admission_attempts:,} worst-case attempts",
        )
    )


def plan_view(plan: FrontDoorPlan, *, statecharts: bool = False) -> list[str]:
    """The plan: every stage, completed work, remaining and unpriced work, bound and digest."""
    lines = list(front_door_lines(plan))
    if not statecharts and "State machines (Mermaid):" in lines:
        lines = lines[: lines.index("State machines (Mermaid):")]
    lines.append(f"Code revision {plan.code_revision}")
    lines.append(f"Priced bound ${launch.priced_bound(plan):.6f}")
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
                "store or code changed since it was planned. Review the plan above and consent "
                "to its digest"
            )
        console.print(Text(f"Consented to plan {plan.digest} by --consent."))
        return True
    console.print(
        Text(
            f"Consent covers exactly plan {plan.digest[:12]}: real spend at most "
            f"${min(launch.priced_bound(plan), options.ceiling_usd):.2f}, at most "
            f"{options.real_attempts:,} real attempts."
        )
    )
    return require_spend_consent(
        console,
        root=options.root,
        yes=options.yes,
        estimated_cost_usd=min(launch.priced_bound(plan), options.ceiling_usd),
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
    """Start the consented plan in its own session, record it, then (optionally) watch it."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log = options.log or store.paths.runtime_directory / "front-door-cli" / (
        f"{plan.digest[:16]}-{stamp}.log"
    )
    arguments = [*options.rerun(plan.digest), "--log", str(log)]
    if options.code_revision is None:
        arguments += ["--code-revision", revision]
    pid = spawn_detached(arguments, log)
    record = RunRecord(
        digest=plan.digest,
        journal=run_directory(store, spec).name,
        operation=launch.operation_name(plan.digest),
        spec=str(options.spec.resolve()),
        revision=revision,
        pid=pid,
        log=str(log),
        started=now_iso(),
    )
    save_record(store, record)
    console.print(f"[green]detached[/green] pid {pid}; it re-plans (no model call) and checks "
                  "the digest before its first phase.")  # fmt: skip
    console.print(Text(f"  log    {log}"))
    console.print(Text(f"  watch  exp variance watch {options.project} {plan.digest[:12]} "
                       f"--root {options.root}"))  # fmt: skip
    if options.watch:
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
    """Install the ceiling and run the front door in this process under the live view."""
    from exp.cli.evaluation.front_door_action import front_door_summary
    from exp.cli.evaluation.learning_summary import save_details
    from exp.runtime.models.spend_ceiling import operation_spend_scope

    record = RunRecord(
        digest=plan.digest,
        journal=run_directory(store, spec).name,
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
            operation, receipt = launch.install_ceiling(
                store,
                plan,
                ceiling_usd=options.ceiling_usd,
                real_attempts=options.real_attempts,
                command="exp " + " ".join(options.rerun(plan.digest)),
            )
            console.print(Text(f"Spend receipt: {receipt}", style="dim"))
            approval = launch.approval(plan, options.admission_attempts)
        save_record(store, record)
        view.update(note="re-planning inside the run to check the consent (no model call)")
        refresh = refresher(options.root, options.project, view, record)
        result: FrontDoorReport | None = None
        try:
            with live_view(console, view, refresh):
                result = run_front_door(
                    store,
                    spec,
                    code_revision=revision,
                    approval=approval,
                    dependencies=dependencies,  # ty: ignore[invalid-argument-type]
                )
                view.update(stopped=result.stopped, note="")
                records = {
                    row.task_id: {stage: ref.artifact_id for stage, ref in row.stages.items()}
                    for row in result.pipelines
                }
                view.update(outcome=outcome(store, records))
        except BaseException as error:
            finish_record(store, record, "failed", f"{type(error).__name__}: {error}"[:2000])
            raise
    state = "finished" if result.stopped is None else "stopped"
    finish_record(store, record, state, result.stopped)
    if not console.is_interactive:
        for line in view.summary_lines():
            console.print(Text(f"  {line}"))
    details = save_details(
        store, "front-door", run_directory(store, spec).name, result.model_dump_json(indent=2)
    )
    console.print_json(data=front_door_summary(result))
    console.print(Text(f"Report: {details}"))
    took = duration_text(time.monotonic() - started)
    if result.stopped is None:
        console.print(f"[green]Complete[/green] [dim]in {took}[/dim]")
        return
    console.print(f"[yellow]Stopped[/yellow] [dim]after {took}[/dim]")
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
    """Reopen the live view on one run until it ends (Ctrl-C leaves the run running)."""
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
    if located.journal is None:
        view.update(note="no transition journal found yet (the run may still be planning)")
    if once:
        _settle(refresh)
        for line in view.lines():
            console.print(Text(line), soft_wrap=True)
        return
    try:
        with live_view(console, view, refresh, interval=interval):
            while not _ended(view, located.record):
                time.sleep(interval)
    except KeyboardInterrupt:
        console.print("[dim]stopped watching; the run (if any) continues[/dim]")
        return
    if not console.is_interactive:
        for line in view.summary_lines():
            console.print(Text(f"  {line}"))


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
