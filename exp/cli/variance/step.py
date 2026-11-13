"""``exp variance run --step``: pause before and after each stage to show its evidence.

Before a phase (and before each pipeline stage that runs now) a short card says what it does,
what it reads and its bound, then waits for Enter. After it, a card says whether it ran now (and
for how long, with how many model attempts) or replayed from the store (and when it was saved),
its key outputs, and the path of its evidence page in the demo wiki, then waits again. A replayed
start's pipeline lists its six saved stages, so a pure replay walks every stage with no model
call. Cards use the ``exp build`` preflight style; there is no progress line in step mode.

The hooks are the front door's progress events (each phase's ``running``, ``replayed`` and end
events arrive synchronously, before and after its work) and the reporting pipeline executors'
results, so pausing never changes what runs.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.text import Text

from exp.cli.shared.progress import duration_text
from exp.cli.variance.sources import PIPELINE_STAGES, Traffic, journal_segments, pipeline_stages
from exp.cli.variance.view import STAGE_COVERS, FrontDoorView
from exp.common.progress import ProgressEvent
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.frontdoor import FrontDoorPlan

WIKI = Path("/workspace/t/dv-demo/wiki")
"""The demo's evidence wiki: one page per stage, ``<stage-key>.html``."""

READS = {
    "project-build": "the trace file: import, task mining, both RAG indexes, the world model",
    "variance-build": "the project build: decision tags, tag clusters, the families review",
    "slot tag-banks": "each environment family's traces and decision tags",
    "slot task-types": "each family's tasks and its tag bank",
    "slot state-rag": "each family's tag bank and its traces' states",
    "bootstrap-preparation": "the families, their state-tag indexes and the selected task types",
    "slot rule-admission": "each family's frozen rules and the registered starts",
    "slot blast-radius": "every start's mined error points and continuations",
    "slot curriculum": "every start's admitted tasks and their strengths",
    "slot dense-hybrid": "every start's records and point options",
}
"""What each phase reads, for the before card."""

STAGE_READS = {
    "variance-build": "the front door's saved variance build (replayed)",
    "rules": "the family's frozen rules and the start's job",
    "traceml": "the start, five actor tiers and the general rubric",
    "error-mining": "the judged groups, TraceML error points and point bars",
    "families": "qualified points, judged groups and the behavior library",
    "records": "admitted families, selectors and the curriculum",
}


def stage_key(name: str) -> str:
    """The wiki page key of a phase (``slot tag-banks`` -> ``slot-tag-banks``)."""
    return "-".join(name.lower().split())


def evidence(name: str) -> str:
    """The evidence page path of one phase or pipeline stage, and whether it exists yet."""
    path = WIKI / f"{stage_key(name)}.html"
    return str(path) if path.exists() else f"{path} [dim](not built yet)[/dim]"


def one_line(summary: Mapping[str, object], limit: int = 110) -> str:
    """A stage summary in one line: scalars as values, collections as counts."""
    parts = []
    for key, value in summary.items():
        if isinstance(value, dict | list):
            parts.append(f"{key} {len(value)}")
        elif value is not None:
            parts.append(f"{key} {value}")
    text = " · ".join(parts)
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class Stepper:
    """Shows the cards and waits; receives the front door's events and stage results.

    Attributes:
        console: The command console.
        store: The project store (read only here).
        view: The live model, for calls and the journal.
        plan: The consented (or replayed) plan.
        journal: The registration's transition journal, for when replayed phases were saved.
        pause: Wait for Enter after each card (only in a terminal).
    """

    console: Console
    store: ProjectStore
    view: FrontDoorView
    plan: FrontDoorPlan
    journal: Path | None
    traffic: Traffic | None = None
    pause: bool = True
    start_at: str | None = None
    _started: dict[str, float] = field(default_factory=dict)
    _target: str | None = None
    _position: int = 0

    def __post_init__(self) -> None:
        """Every card in order: each phase, and each stage of a start's pipeline."""
        self.steps: list[str] = []
        for row in self.plan.stages:
            self.steps.append(row.stage)
            if row.stage.startswith("pipeline "):
                task = row.stage.removeprefix("pipeline ")
                self.steps.extend(f"{task} {stage}" for stage in PIPELINE_STAGES)
        if self.start_at is not None:
            self._target = self._resolve(self.start_at, 0)
            if self._target is None:
                raise ValueError(
                    f"--from {self.start_at!r} names no stage; stages: " + ", ".join(self.steps)
                )

    def _resolve(self, wanted: str, after: int) -> str | None:
        """A step by its number (1-based) or by name (exact, else first containing it)."""
        wanted = wanted.strip()
        if wanted.isdigit():
            index = int(wanted) - 1
            return self.steps[index] if 0 <= index < len(self.steps) else None
        lower = wanted.lower()
        for step in self.steps[after:]:
            if step.lower() == lower:
                return step
        for step in self.steps[after:]:
            if lower in step.lower():
                return step
        return None

    def _show(self, name: str) -> bool:
        """Whether this card is shown; cards before a jump target pass silently."""
        if name in self.steps:
            self._position = self.steps.index(name)
        if self._target is None:
            return True
        if name == self._target:
            self._target = None
            return True
        number = self._position + 1
        self.console.print(f"  [dim]· {number}. {name} (skipped, replayed)[/dim]")
        return False

    def listing(self) -> None:
        """Every card with its number."""
        for number, step in enumerate(self.steps, 1):
            here = "  ◂ here" if number - 1 == self._position else ""
            self.console.print(f"  [dim]{number:>2}.[/dim] {step}{here}")

    def wait(self, verb: str) -> None:
        """Wait for Enter; a number or a stage name jumps forward to that card, l lists them."""
        while self.pause:
            answer = self.console.input(
                f"  [dim]Enter to {verb} · number or name to jump · l to list[/dim] "
            ).strip()
            if not answer:
                return
            if answer.lower() in ("l", "list", "?"):
                self.listing()
                continue
            target = self._resolve(answer, self._position + 1)
            if target is None:
                self.console.print(
                    "  [yellow]no later stage matches; the replay only moves forward "
                    "(restart with --from to begin elsewhere)[/yellow]"
                )
                continue
            self._target = target
            return

    # ------------------------------------------------------------ front-door phases

    def observe(self, event: ProgressEvent) -> None:
        """The progress hook: fold into the view, then show a card at each lifecycle step."""
        self.view.observe(event)
        if event.total is not None or event.detail is None:
            return
        name = event.stage
        if event.detail == "running":
            shown = self._show(name)
            if shown:
                self.before(name)
                self.wait("run")
            self._started[name] = time.time()
        elif event.detail == "replayed":
            if self._show(name):
                self.before(name)
                self.replayed(name)
            elif name.startswith("pipeline "):
                self._pipeline_steps(name.removeprefix("pipeline "))
        elif event.detail in ("finished", "stopped", "raised", "refused"):
            if self._target is None:
                self.after(name, event.detail)

    def before(self, name: str) -> None:
        """What the phase does, what it reads and its bound."""
        row = next((row for row in self.plan.stages if row.stage == name), None)
        self.console.print(f"\n[bold]▸ {name}[/bold]")
        does = row.lines[0] if row is not None and row.lines else None
        if name.startswith("pipeline "):
            does = "six stages for this start: " + ", ".join(PIPELINE_STAGES)
        if does:
            self.console.print(Text.assemble(("  does       ", "dim"), does))
        reads = READS.get(name, "the start registered by the preparation")
        self.console.print(Text.assemble(("  reads      ", "dim"), reads))
        if row is not None and row.costs:
            requests = sum(line.attempts for line in row.costs)
            self.console.print(f"  [dim]bound[/dim]      at most {requests:,} physical requests")

    def replayed(self, name: str) -> None:
        """A saved phase: when it was saved and, for a start, its six saved stages."""
        saved = self._saved_at(name)
        when = f"saved {saved:%Y-%m-%d %H:%M} UTC" if saved else "saved earlier"
        self.console.print(f"[green]✓[/green] {name}  replayed ({when}) · no model call")
        self.console.print(f"  [dim]evidence[/dim]   {evidence(name)}")
        self.wait("continue")
        if name.startswith("pipeline "):
            self._pipeline_steps(name.removeprefix("pipeline "))

    def after(self, name: str, outcome: str) -> None:
        """A phase that ran now: how long, how many model attempts, its evidence."""
        started = self._started.pop(name, None)
        took = "" if started is None else f" in {duration_text(time.time() - started)}"
        attempts = self._attempts_since(started)
        mark = "[green]✓[/green]" if outcome == "finished" else "[yellow]![/yellow]"
        self.console.print(
            f"{mark} {name}  ran now{took} · {attempts:,} model attempts · {outcome}"
        )
        if name.startswith("pipeline "):
            self._pipeline_cards(name.removeprefix("pipeline "), only_saved=True)
        self.console.print(f"  [dim]evidence[/dim]   {evidence(name)}")
        self.wait("continue")

    # ------------------------------------------------------------ pipeline stages

    def before_stage(self, task: str, stage: str) -> None:
        """A pipeline stage about to run now."""
        if not self._show(f"{task} {stage}"):
            self._started[f"{task}/{stage}"] = time.time()
            return
        self.console.print(f"\n  [bold]▸ {stage}[/bold] [dim]{task}[/dim]")
        self.console.print(Text.assemble(("    does     ", "dim"), STAGE_COVERS[stage]))
        self.console.print(Text.assemble(("    reads    ", "dim"), STAGE_READS[stage]))
        self._started[f"{task}/{stage}"] = time.time()
        self.wait("run")

    def after_stage(self, task: str, stage: str, summary: Mapping[str, object]) -> None:
        """A pipeline stage that just ran: its summary and evidence."""
        if self._target is not None:
            self._started.pop(f"{task}/{stage}", None)
            return
        started = self._started.pop(f"{task}/{stage}", None)
        took = "" if started is None else f" in {duration_text(time.time() - started)}"
        attempts = self._attempts_since(started)
        self.console.print(
            f"  [green]✓[/green] {stage}  ran now{took} · {attempts:,} model attempts"
        )
        self.console.print(Text.assemble(("    outputs  ", "dim"), one_line(summary)))
        self.console.print(f"    [dim]evidence[/dim] {evidence(f'pipeline {task} {stage}')}")
        self.wait("continue")

    def _pipeline_cards(self, task: str, *, only_saved: bool = False) -> None:
        """Each saved stage of a start: when it was saved and its summary in one line."""
        start = pipeline_stages(self.store, [task]).get(task)
        if start is None or start.run is None:
            self.console.print("  [dim]no pipeline registration saved yet[/dim]")
            return
        for stage in PIPELINE_STAGES:
            if stage not in start.saved:
                if not only_saved:
                    self.console.print(f"  [dim]·[/dim] {stage}  not saved")
                continue
            record = json.loads(self.store.artifacts.read_bytes(start.saved[stage], "record.json"))
            moment = start.saved_at[stage].astimezone(UTC)
            self.console.print(
                f"  [green]✓[/green] {stage:<13} [dim]saved {moment:%H:%M}[/dim]  "
                f"{one_line(record.get('summary') or {})}"
            )
            self.console.print(f"    [dim]evidence[/dim] {evidence(f'pipeline {task} {stage}')}")

    def _pipeline_steps(self, task: str) -> None:
        """A replayed start's saved stages, each its own card and pause."""
        start = pipeline_stages(self.store, [task]).get(task)
        if start is None or start.run is None:
            self.console.print("  [dim]no pipeline registration saved yet[/dim]")
            return
        for stage in PIPELINE_STAGES:
            card = f"{task} {stage}"
            if not self._show(card):
                continue
            self.console.print(f"\n  [bold]▸ {stage}[/bold] [dim]{task}[/dim]")
            self.console.print(Text.assemble(("    does     ", "dim"), STAGE_COVERS[stage]))
            self.console.print(Text.assemble(("    reads    ", "dim"), STAGE_READS[stage]))
            if stage not in start.saved:
                self.console.print("    [dim]·[/dim] not saved")
            else:
                record = json.loads(
                    self.store.artifacts.read_bytes(start.saved[stage], "record.json")
                )
                moment = start.saved_at[stage].astimezone(UTC)
                self.console.print(
                    f"  [green]✓[/green] {stage}  replayed (saved {moment:%Y-%m-%d %H:%M} UTC)"
                    " · no model call"
                )
                self.console.print(
                    Text.assemble(("    outputs  ", "dim"), one_line(record.get("summary") or {}))
                )
            self.console.print(f"    [dim]evidence[/dim] {evidence(f'pipeline {task} {stage}')}")
            self.wait("continue")

    # ------------------------------------------------------------ sources

    def _attempts_since(self, started: float | None) -> int:
        """Model attempts the operation reserved since a moment."""
        traffic = self.traffic
        if traffic is None or started is None:
            return 0
        with self.view.guard():
            traffic.refresh()
            return sum(1 for call in traffic.calls.values() if call.reserved >= started)

    def _saved_at(self, name: str) -> datetime | None:
        """When an earlier run of this registration finished the phase."""
        if self.journal is None:
            return None
        ends = [
            segment.phases[name].ended
            for segment in journal_segments(self.journal)
            if name in segment.phases and segment.phases[name].state == "done"
        ]
        return max((end for end in ends if end is not None), default=None)


def stage_hooks(
    stepper: Stepper | None,
) -> tuple[
    Callable[[str, str], None] | None, Callable[[str, str, Mapping[str, object]], None] | None
]:
    """The before and after callbacks the reporting pipeline executors call in step mode."""
    if stepper is None:
        return None, None
    return stepper.before_stage, stepper.after_stage
