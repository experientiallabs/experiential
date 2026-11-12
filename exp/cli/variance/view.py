"""The front-door live view, in the ``exp build`` style.

One model (:class:`FrontDoorView`) folds every source into rows and panels; two renderers show
it. An interactive terminal gets a rich ``Live`` region redrawn in place: the stage tree with
each phase's status and elapsed time (each start's pipeline with its six stages and what they
cover), then the running stage's call counts, the calls panel (per role: calls, done, in flight,
errors, 429s, timeouts, calls in the last minute, median latency), the real-attempt cap and the
host's admission, and the last phase preview. A non-interactive stream gets stable lines with no
cursor control, like :mod:`exp.cli.shared.progress`: the tree once, then one line each time a
row changes state. Both end with the same summary of what the pipelines produced.

Sources (:mod:`exp.cli.variance.sources`): the run's transition journal (phase states and
times), progress events (:class:`~exp.common.progress.ProgressEvent`) from the front door and
its pipeline stages when the view runs in the same process, the spend ledger joined to gateway
attempt records (calls), the host admission database (calls in flight, breakers) and the
project store (each start's saved pipeline stages and, at the end, their outcomes).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from exp.cli.shared.progress import duration_text
from exp.cli.variance.sources import (
    PIPELINE_STAGES,
    Admission,
    Outcome,
    RunSegment,
    StartStages,
    Traffic,
)
from exp.common.progress import ProgressEvent

DEFAULT_PHASES = (
    "project-build",
    "variance-build",
    "slot tag-banks",
    "slot task-types",
    "slot state-rag",
    "bootstrap-preparation",
    "slot rule-admission",
    "pipelines",
    "slot blast-radius",
    "slot curriculum",
    "slot dense-hybrid",
)
"""The phases a full front door names, in order, until a plan or journal says otherwise."""

STAGE_COVERS = {
    "variance-build": "the start's variance build (replays the front-door build)",
    "rules": "rule book · rule memory · general rubric",
    "traceml": "actor groups ×5 tiers · rubric judging · behavior differences · TraceML · "
    "point bar",
    "error-mining": "error mining · judge verdicts · selectors/critics · blast radius",
    "families": "adversarial linters · environment families · continuations",
    "records": "records · curriculum · dense reward · Harbor export",
}
"""What each pipeline stage covers, shown beside it in the tree."""

_FINAL = {"finished", "stopped", "failed"}

LIFECYCLE = {"replayed", "running", "finished", "stopped", "refused", "raised"}
"""Progress ``detail`` values the front door emits for a phase's lifecycle."""

_EVENT_STATE = {
    "replayed": "done",
    "running": "running",
    "finished": "done",
    "stopped": "stopped_at_cap",
    "refused": "stopped_at_cap",
    "raised": "failed",
}
_GLYPH = {
    "done": ("✓", "green"),
    "running": ("●", "cyan"),
    "failed": ("✗", "red"),
    "stopped_at_cap": ("!", "yellow"),
    "interrupted": ("!", "yellow"),
    "approved": ("○", "cyan"),
    "planned": ("·", "dim"),
    "blocked": ("·", "dim"),
    "pending": ("·", "dim"),
}
_WORDS = {
    "stopped_at_cap": "stopped",
    "blocked": "waiting",
}


@dataclass(frozen=True)
class Row:
    """One line of the stage tree.

    Attributes:
        key: Identity of the row across refreshes.
        depth: Indentation level (0 phases, 1 starts, 2 pipeline stages).
        label: What it is.
        state: Its state (``done``, ``running``, ``failed``, ``planned``...).
        elapsed: Seconds it ran, or has been running, when known.
        note: Short qualifier (``saved``, counts, what a pipeline stage covers).
    """

    key: str
    depth: int
    label: str
    state: str
    elapsed: float | None = None
    note: str = ""

    def text(self) -> str:
        """The row as one stable plain line."""
        glyph = _GLYPH.get(self.state, ("·", "dim"))[0]
        word = _WORDS.get(self.state, self.state)
        parts = [f"{'  ' * (self.depth + 1)}{glyph} {self.label}", word]
        if self.elapsed is not None:
            parts.append(duration_text(self.elapsed))
        if self.note:
            parts.append(self.note)
        return "  ".join(parts)


@dataclass
class _SubStage:
    """A pipeline stage the in-process progress events last reported."""

    stage: str
    finished: bool
    at: float


@dataclass
class FrontDoorView:
    """Everything the live view shows, folded from its sources. Safe across threads.

    Attributes:
        title: The run's name (``front door <digest12>``).
        project: The project.
        phases: Phase names in plan order.
        segment: The run's journal segment, once it exists.
        traffic: The operation's calls, once its ledger exists.
        admission: Host-wide admission.
        starts: Each start's newest pipeline registration and saved stages.
        outcome: What the pipelines produced, once the run ended.
        started: When the view (or the run) started, for elapsed time.
        stopped: Why the run stopped, when it did.
        note: A status line under the title (re-planning, waiting on the ledger...).
        interrupted: When an unfinished run's process was found gone: the last activity, which
            ends every running row's clock.
    """

    title: str
    project: str
    phases: list[str] = field(default_factory=lambda: list(DEFAULT_PHASES))
    segment: RunSegment | None = None
    traffic: Traffic | None = None
    admission: Admission = field(default_factory=Admission)
    starts: dict[str, StartStages] = field(default_factory=dict)
    outcome: Outcome | None = None
    started: float = field(default_factory=time.time)
    stopped: str | None = None
    note: str = ""
    interrupted: float | None = None
    _events: dict[str, tuple[str, float]] = field(default_factory=dict)
    _sub: dict[str, _SubStage] = field(default_factory=dict)
    _pipelines: tuple[int, int] | None = None
    _preview: tuple[str, ...] = ()
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _spinner: Spinner = field(default_factory=lambda: Spinner("dots", style="cyan"))

    # ------------------------------------------------------------ inputs

    def guard(self) -> threading.RLock:
        """The view's lock, for a source that mutates in place while the view draws it."""
        return self._lock

    def observe(self, event: ProgressEvent) -> None:
        """Fold one progress event (the hook handed to the front door)."""
        now = time.time()
        with self._lock:
            if event.stage == "pipelines" and event.total is not None:
                self._pipelines = (event.completed or 0, event.total)
            elif event.stage.startswith("pipeline ") and event.total is not None and event.detail:
                finished = (event.completed or 0) > PIPELINE_STAGES.index(event.detail)
                self._sub[event.stage] = _SubStage(event.detail, finished, now)
            elif event.detail in LIFECYCLE:
                self._events[event.stage] = (_EVENT_STATE[event.detail], now)
                self._add_phase(event.stage)

    def announce(self, lines: tuple[str, ...]) -> None:
        """Keep the front door's preview of the phase it is about to admit."""
        with self._lock:
            self._preview = tuple(lines)

    def set_plan(self, stages: Sequence[str]) -> None:
        """Take the phase order from the consented plan."""
        with self._lock:
            self.phases = list(stages)

    def update(self, **values: object) -> None:
        """Replace source snapshots (segment, traffic, admission, starts, outcome, note...)."""
        with self._lock:
            for name, value in values.items():
                setattr(self, name, value)
            if self.segment is not None:
                for name in self.segment.phases:
                    self._add_phase(name)

    def _add_phase(self, name: str) -> None:
        """Place a phase the plan did not name: a start's pipeline goes after ``pipelines``."""
        if name in self.phases:
            return
        if name.startswith("pipeline "):
            anchor = max(
                (i for i, row in enumerate(self.phases) if row.startswith("pipeline")), default=-1
            )
            if anchor >= 0:
                self.phases.insert(anchor + 1, name)
                return
            later = [i for i, row in enumerate(self.phases) if row.startswith("slot blast")]
            self.phases.insert(later[0] if later else len(self.phases), name)
            return
        self.phases.append(name)

    # ------------------------------------------------------------ model

    def run_state(self) -> str:
        """The run's state: from the journal, else ``planning`` or ``running``."""
        with self._lock:
            if self.segment is not None:
                if self.interrupted is not None and self.segment.state not in _FINAL:
                    return "interrupted"
                return self.segment.state
            return "running" if self._events else "planning"

    def tasks(self) -> list[str]:
        """Every start whose pipeline phase is known."""
        with self._lock:
            return [
                name.removeprefix("pipeline ")
                for name in self.phases
                if name.startswith("pipeline ")
            ]

    def rows(self, now: float | None = None) -> list[Row]:
        """The stage tree, in order."""
        now = self._clock() if now is None else now
        with self._lock:
            rows: list[Row] = []
            for name in self.phases:
                rows.append(self._phase_row(name, now))
                if name.startswith("pipeline "):
                    rows.extend(self._stage_rows(name, rows[-1].state, now))
            if self.interrupted is not None:
                rows = [
                    Row(row.key, row.depth, row.label, "interrupted", row.elapsed, row.note)
                    if row.state == "running"
                    else row
                    for row in rows
                ]
            return rows

    def _phase_row(self, name: str, now: float) -> Row:
        """One phase's row from the journal, else from progress events."""
        state, elapsed, note = "planned", None, ""
        journal = self.segment.phases.get(name) if self.segment is not None else None
        if journal is not None:
            state = journal.state
            if journal.started is not None:
                end = journal.ended.timestamp() if journal.ended is not None else now
                elapsed = max(0.0, end - journal.started.timestamp())
            elif journal.replayed:
                note = "saved"
        elif name in self._events:
            state = self._events[name][0]
        starts = [row for row in self.phases if row.startswith("pipeline ")]
        if name == "pipelines" and self._pipelines is not None:
            done, total = self._pipelines
            note = f"{done}/{total} starts"
        elif name == "pipelines" and starts:
            finished = sum(
                1
                for row in starts
                if self.segment is not None
                and row in self.segment.phases
                and self.segment.phases[row].state == "done"
            )
            note = f"{len(starts)} starts, {finished} complete"
        if state == "running" and self.traffic is not None and journal and journal.started:
            calls = self.traffic.roles(now, since=journal.started.timestamp())
            total = sum(row.total for row in calls)
            flight = sum(row.in_flight for row in calls)
            note = f"{total} calls, {flight} in flight"
        return Row(name, 0, name, state, elapsed, note)

    def _stage_rows(self, phase: str, phase_state: str, now: float) -> list[Row]:
        """A start's six pipeline stages: saved, running or pending."""
        task = phase.removeprefix("pipeline ")
        start = self.starts.get(task)
        saved = dict(start.saved_at) if start is not None else {}
        sub = self._sub.get(phase)
        rows = []
        previous: datetime | None = None
        first_open = True
        for stage in PIPELINE_STAGES:
            note = STAGE_COVERS[stage]
            if stage in saved:
                state = "done"
                moment = saved[stage]
                note = f"saved {moment.astimezone(UTC):%H:%M}  {note}"
                previous = moment
            elif sub is not None and sub.stage == stage:
                state = "done" if sub.finished else "running"
                if sub.finished:
                    previous = datetime.fromtimestamp(sub.at, UTC)
            elif first_open and phase_state in ("running", "failed", "stopped_at_cap"):
                state = "running" if phase_state == "running" else phase_state
            else:
                state = "pending"
            if stage not in saved:
                first_open = first_open and state not in ("running", "failed", "stopped_at_cap")
            elapsed = None
            if state == "running" and previous is not None:
                elapsed = max(0.0, now - previous.timestamp())
            rows.append(Row(f"{phase}/{stage}", 2, stage, state, elapsed, note))
        return rows

    def _clock(self) -> float:
        """Now, or the last activity of an interrupted run."""
        return self.interrupted if self.interrupted is not None else time.time()

    # ------------------------------------------------------------ panels

    def running(self) -> list[str]:
        """The running phase: its elapsed time and its calls by role since it started."""
        with self._lock:
            if self.segment is None or self.traffic is None:
                return []
            now = self._clock()
            lines = []
            for row in self.segment.phases.values():
                if row.state != "running" or row.started is None:
                    continue
                calls = self.traffic.roles(now, since=row.started.timestamp())
                parts = [
                    f"{_role(call.role)} {call.done} done"
                    + (f" +{call.in_flight} in flight" if call.in_flight else "")
                    for call in calls
                ]
                elapsed = duration_text(max(0.0, now - row.started.timestamp()))
                lines.append(f"{row.name} · {elapsed} · " + (", ".join(parts) or "no calls yet"))
            return lines

    def calls_table(self) -> Table | None:
        """Per-role calls across the run."""
        with self._lock:
            if self.traffic is None or not self.traffic.calls:
                return None
            table = Table(box=None, pad_edge=False, header_style="dim", show_edge=False)
            for column in ("role", "calls", "done", "in flight", "errors", "429", "timeouts",
                           "last min", "p50"):  # fmt: skip
                table.add_column(column, justify="left" if column == "role" else "right")
            for row in self.traffic.roles(time.time()):
                table.add_row(
                    _role(row.role),
                    str(row.total),
                    str(row.done),
                    str(row.in_flight),
                    _warn(row.errors),
                    _warn(row.rate_limited),
                    _warn(row.timeouts),
                    str(row.last_minute),
                    "-" if row.p50 is None else f"{row.p50:.1f}s",
                )
            return table

    def calls_lines(self) -> list[str]:
        """Per-role calls as plain lines."""
        with self._lock:
            if self.traffic is None or not self.traffic.calls:
                return []
            return [
                f"{_role(row.role)}: {row.total} calls, {row.done} done, {row.in_flight} in "
                f"flight, {row.errors} errors, {row.rate_limited} 429s, {row.timeouts} timeouts"
                + ("" if row.p50 is None else f", p50 {row.p50:.1f}s")
                for row in self.traffic.roles(time.time())
            ]

    def cap_line(self) -> str:
        """Use of the real-attempt cap, host calls in flight and open breakers."""
        with self._lock:
            parts = []
            if self.traffic is not None:
                used = self.traffic.used()
                cap = self.traffic.allowance
                if cap:
                    parts.append(f"{used:,} / {cap:,} real attempts ({used / cap:.1%})")
                else:
                    parts.append(f"{used:,} real attempts")
                if self.traffic.error:
                    parts.append(self.traffic.error)
            flight = {k: v for k, v in self.admission.in_flight.items() if v}
            if flight:
                parts.append(
                    "host-wide in flight "
                    + ", ".join(f"{k} {v}" for k, v in sorted(flight.items()))
                )
            if self.admission.open_breakers:
                parts.append("breakers open: " + ", ".join(self.admission.open_breakers))
            return " · ".join(parts)

    def summary_lines(self) -> list[str]:
        """What the pipelines produced, once the run ended."""
        with self._lock:
            result = self.outcome
            if result is None:
                return []
            points = result.qualified_points
            qualified = sum(
                n for state, n in points.items() if state not in ("rejected", "uncertain")
            )
            lines = [
                f"qualified points  {qualified}"
                + (
                    f"  ({', '.join(f'{k} {v}' for k, v in points.most_common())})"
                    if points
                    else ""
                ),
                f"error points      {sum(result.error_points.values())}"
                + (
                    "  ("
                    + ", ".join(f"{_role(k)} {v}" for k, v in result.error_points.most_common())
                    + ")"
                    if result.error_points
                    else ""
                ),
                "judged            "
                + (
                    "; ".join(
                        f"{task}: {len(tiers)} tiers"
                        for task, tiers in result.judged.items()
                        if tiers
                    )
                    or "none recorded"
                ),
                f"families          {result.families.get('admitted', 0)} admitted, "
                f"{result.families.get('refused', 0)} refused",
                f"admitted tasks    {result.admitted}",
            ]
            lines.append(
                f"harbor packages   {len(result.harbor)} qualified, "
                f"{len(result.diagnostic)} diagnostic"
            )
            lines.extend(f"  {path}" for path in result.harbor)
            lines.extend(f"  {path} (diagnostic)" for path in result.diagnostic)
            return lines

    # ------------------------------------------------------------ renderers

    def header(self) -> str:
        """The title line."""
        with self._lock:
            state = self.run_state()
            start, end = self.started, self._clock()
            if self.segment is not None and self.segment.planned is not None:
                start = min(start, self.segment.planned.timestamp())
                if self.segment.ended is not None and state in ("finished", "stopped", "failed"):
                    end = self.segment.ended.timestamp()
            elapsed = duration_text(max(0.0, end - start))
            return f"{self.title} · {self.project} · {state} · {elapsed}"

    def lines(self) -> list[str]:
        """The whole view as stable plain lines (``watch --once`` and logs)."""
        out = [self.header()]
        if self.note:
            out.append(f"  {self.note}")
        out.extend(row.text() for row in self.rows())
        running = self.running()
        if running:
            out.append("Running")
            out.extend(f"  {line}" for line in running)
        calls = self.calls_lines()
        if calls:
            out.append("Calls")
            out.extend(f"  {line}" for line in calls)
        cap = self.cap_line()
        if cap:
            out.append(f"Cap  {cap}")
        if self.stopped:
            out.append(f"Stopped  {self.stopped}")
        summary = self.summary_lines()
        if summary:
            out.append("Summary")
            out.extend(f"  {line}" for line in summary)
        return out

    def render(self) -> RenderableType:
        """The interactive region: tree, running stage, calls, cap and the last preview."""
        with self._lock:
            parts: list[RenderableType] = [Text(self.header(), style="bold")]
            if self.note:
                parts.append(Text(f"  {self.note}", style="dim"))
            for row in self.rows():
                parts.append(self._render_row(row))
            running = self.running()
            if running:
                parts.append(Text("Running", style="bold"))
                parts.extend(Text(f"  {line}", style="cyan") for line in running)
            table = self.calls_table()
            if table is not None:
                parts.append(Text("Calls", style="bold"))
                parts.append(table)
            cap = self.cap_line()
            if cap:
                parts.append(Text.assemble(("Cap  ", "bold"), (cap, "dim")))
            if self._preview:
                parts.append(Text("Last phase preview", style="bold"))
                parts.extend(Text(line, style="dim") for line in self._preview[:5])
            if self.stopped:
                parts.append(Text.assemble(("Stopped  ", "bold yellow"), self.stopped))
            summary = self.summary_lines()
            if summary:
                parts.append(Text("Summary", style="bold"))
                parts.extend(Text(f"  {line}") for line in summary)
            return Group(*parts)

    def _render_row(self, row: Row) -> RenderableType:
        """One tree row with its glyph (a spinner while running)."""
        glyph, style = _GLYPH.get(row.state, ("·", "dim"))
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append("  " * (row.depth + 1))
        if row.state == "running":
            frames = self._spinner.frames
            frame = frames[int(time.time() * 1000 / self._spinner.interval) % len(frames)]
            line.append(frame or glyph, style=style)
        else:
            line.append(glyph, style=style)
        label_style = "dim" if row.state in ("planned", "blocked", "pending") else ""
        line.append(f" {row.label:<{max(1, 38 - 2 * row.depth)}}", style=label_style)
        line.append(f" {_WORDS.get(row.state, row.state):<11}", style=style)
        if row.elapsed is not None:
            line.append(f" {duration_text(row.elapsed):>7}", style="dim")
        if row.note:
            line.append(f"  {row.note}", style="dim")
        return line


def _role(role: str) -> str:
    """A role's display name: actor roles are the five tiers."""
    return f"tier {role.removeprefix('actor-')}" if role.startswith("actor-") else role


def _warn(count: int) -> str:
    """A count styled as a warning when it is not zero."""
    return f"[yellow]{count}[/yellow]" if count else "0"


class LinePrinter:
    """Stable lines for a non-interactive stream: the tree once, then each row that changes."""

    def __init__(self, console: Console) -> None:
        """Print nothing yet."""
        self._console = console
        self._seen: dict[str, str] = {}

    def __call__(self, view: FrontDoorView) -> None:
        """Print the rows whose state changed since the last call."""
        first = not self._seen
        if first:
            self._console.print(view.header(), markup=False, highlight=False)
        for row in view.rows():
            if self._seen.get(row.key) == row.state:
                continue
            self._seen[row.key] = row.state
            self._console.print(row.text(), markup=False, highlight=False)


@contextmanager
def live_view(
    console: Console,
    view: FrontDoorView,
    refresh: Callable[[], None],
    *,
    interval: float = 2.0,
) -> Iterator[FrontDoorView]:
    """Own the terminal while a run (or a watch) is in progress, refreshing sources in a thread.

    Args:
        console: The command's console.
        view: The model to draw.
        refresh: Re-reads every source into the view; errors are kept as the view's note.
        interval: Seconds between source refreshes.

    Yields:
        The view, for the caller's progress hook and announcements.
    """
    stop = threading.Event()
    printer = None if console.is_interactive else LinePrinter(console)

    def tick() -> None:
        """Refresh once; a failing source is shown, never raised."""
        try:
            refresh()
        except Exception as error:  # noqa: BLE001 - a view must not kill the run
            view.update(note=f"view refresh failed: {type(error).__name__}: {error}")
        if printer is not None:
            printer(view)

    def loop() -> None:
        """Refresh until stopped."""
        while not stop.wait(interval):
            tick()

    tick()
    worker = threading.Thread(target=loop, name="front-door-view", daemon=True)
    live = (
        Live(console=console, get_renderable=view.render, refresh_per_second=4, transient=True)
        if printer is None
        else None
    )
    if live is not None:
        live.start(refresh=True)
    worker.start()
    try:
        yield view
    finally:
        stop.set()
        worker.join(timeout=interval + 5)
        tick()
        if live is not None:
            live.stop()
            console.print(view.render())
