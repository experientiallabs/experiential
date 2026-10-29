"""The one in-place progress line every long-running CLI command uses.

An interactive terminal sees a single line that never grows: a spinner, an overall bar, the
percent done, a status (the phase with its exact counts and the stage running inside it), a
rate-based remaining-time estimate once one is observable, and elapsed time. A command with
several phases (``exp build``: indexes, learning, retest) weights them into one bar; a command
with one phase reads its bar from the counts of whichever stage is running. A run that fails
leaves its interrupted status marked ``[ ]``.

A non-interactive stream (logs, CI, agents) gets stable lines with no cursor control: a phase
prints once when it starts and once when its counting stage completes; a phase without one
prints each stage once when it starts and once when its counts complete.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from time import monotonic

from rich.console import Console, RenderableType
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from exp.common.progress import ProgressEvent, ProgressHook

_ROW_INDENT = "  "
_BAR_WIDTH = 20
_ONLY = "only"  # the implicit phase of a single-phase display


@dataclass(frozen=True)
class Phase:
    """One phase of a command, as the single progress line shows it.

    Attributes:
        key: Identity the command uses to report into this phase.
        label: Short status prefix, such as ``Learning``; empty for a single-phase display.
        weight: Share of the overall bar this phase fills.
        driver: Stage whose exact counts measure the phase's completion; other stages only
            update the status. None lets the counts of whichever stage is running measure it.
        unit: Plural noun for the driver counts, such as ``traces``.
    """

    key: str
    label: str
    weight: float
    driver: str | None = None
    unit: str = ""


class ProgressDisplay:
    """Renders one command's progress events as a single line. Events may come from threads."""

    def __init__(self, console: Console, phases: Sequence[Phase]) -> None:
        """Prepare the display without drawing anything yet."""
        self._console = console
        self._phases = {phase.key: phase for phase in phases}
        self._fraction = {phase.key: 0.0 for phase in phases}
        self._counts: dict[str, tuple[int, int]] = {}
        self._active: str | None = None
        self._status = ""
        self._printed: ProgressEvent | None = None
        self._started = 0.0
        self._measured: float | None = None
        self._lock = threading.Lock()
        self._spinner = Spinner("dots")
        self._live = (
            Live(
                console=console,
                auto_refresh=True,
                refresh_per_second=4,
                transient=True,
                get_renderable=self._render,
            )
            if console.is_interactive
            else None
        )

    def start(self) -> None:
        """Begin owning the in-place line."""
        self._started = monotonic()
        if self._live is not None:
            self._live.start(refresh=True)

    def stop(self) -> None:
        """Release the in-place line; nothing of it stays on screen."""
        if self._live is not None:
            self._live.stop()

    def abort(self) -> None:
        """Release the line, leaving the interrupted status marked unfinished."""
        if self._live is not None:
            status = self._status_text()
            self._live.stop()
            if status:
                self._console.print(Text(f"{_ROW_INDENT}[ ] {status}", style="red"))

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Release the line while the command asks a question, then take it back."""
        if self._live is not None:
            self._live.stop()
        try:
            yield
        finally:
            if self._live is not None:
                self._live.start(refresh=True)

    def phase(self, key: str) -> ProgressHook:
        """The hook a command hands to the services running one phase."""
        phase = self._phases[key]

        def observe(event: ProgressEvent) -> None:
            """Fold one service event into the single line."""
            with self._lock:
                self._enter(key)
                drives = event.total is not None and (
                    event.stage == phase.driver or phase.driver is None
                )
                if drives and event.total:
                    self._fraction[key] = (event.completed or 0) / event.total
                    if self._measured is None and event.completed:
                        self._measured = monotonic()
                if event.stage == phase.driver and event.total:
                    self._counts[key] = (event.completed or 0, event.total)
                else:
                    self._status = _label(event)
                self._log(phase, event)

        return observe

    def complete(self, key: str) -> None:
        """Mark one phase finished, whether or not it reported any work."""
        with self._lock:
            self._fraction[key] = 1.0

    def _enter(self, key: str) -> None:
        """Make ``key`` the active phase, announcing a labelled phase once in a log."""
        if self._active == key:
            return
        if self._active is not None:
            self._fraction[self._active] = 1.0
        self._active, self._status = key, ""
        label = self._phases[key].label
        if self._live is None and label:
            self._console.print(f"{_ROW_INDENT}{label}...", markup=False, highlight=False)

    def _log(self, phase: Phase, event: ProgressEvent) -> None:
        """In a log, print a stage when it starts and when its counts complete; nothing else."""
        if self._live is not None:
            return
        if phase.driver is not None and event.stage != phase.driver:
            return  # inner stages of a counted phase stay off the log
        starts = self._printed is None or _key(self._printed) != _key(event)
        finishes = (
            event.total is not None
            and event.completed == event.total
            and self._printed != event
            and not (phase.driver is not None and starts and event.completed == 0)
        )
        if starts and phase.driver is None or finishes:
            self._console.print(f"{_ROW_INDENT}. {_label(event)}", markup=False, highlight=False)
            self._printed = event

    def _share(self) -> float:
        """Overall completion across every phase, by weight."""
        total = sum(phase.weight for phase in self._phases.values()) or 1.0
        done = sum(self._phases[key].weight * value for key, value in self._fraction.items())
        return min(1.0, done / total)

    def _status_text(self) -> str:
        """The phase label, its exact counts and the stage running inside it."""
        if self._active is None:
            return ""
        phase = self._phases[self._active]
        parts = [phase.label] if phase.label else []
        if self._active in self._counts:
            completed, of = self._counts[self._active]
            parts.append(f"{completed}/{of} {phase.unit}".rstrip())
        status = " ".join(parts)
        if self._status:
            status = f"{status} · {self._status}" if status else self._status
        return status

    def _render(self) -> RenderableType:
        """The single line: spinner, bar, percent, status, remaining estimate and elapsed."""
        with self._lock:
            share, status, measured = self._share(), self._status_text(), self._measured
        now = monotonic()
        filled = round(_BAR_WIDTH * share)
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append("\u2501" * filled, style="cyan")
        line.append("\u2501" * (_BAR_WIDTH - filled), style="dim")
        line.append(f" {share:4.0%}", style="cyan")
        if status:
            line.append(f" · {status}", style="cyan")
        if measured is not None and 0.0 < share < 1.0 and now > self._started:
            remaining = (now - self._started) * (1.0 - share) / share
            line.append(f" · eta {_duration(remaining)}", style="dim")
        elapsed = max(0, int(now - self._started))
        line.append(f" · {_duration(elapsed) if elapsed else '0s'} elapsed", style="dim")
        self._spinner.update(text=line)
        return self._spinner


@contextmanager
def progress_phases(console: Console, phases: Sequence[Phase]) -> Iterator[ProgressDisplay]:
    """Own the single progress line for a whole multi-phase command."""
    display = ProgressDisplay(console, phases)
    display.start()
    try:
        yield display
    except BaseException:
        display.abort()
        raise
    display.stop()


@contextmanager
def progress_display(console: Console) -> Iterator[ProgressHook]:
    """Own the single progress line for one long-running section with one phase.

    Args:
        console: Terminal, or non-terminal stream, receiving the updates.

    Yields:
        The hook the command hands to its long-running services.
    """
    with progress_phases(console, (Phase(_ONLY, "", 1.0),)) as display:
        yield display.phase(_ONLY)


def qualified(hook: ProgressHook | None, detail: str) -> ProgressHook | None:
    """Re-emit a service's events under one distinguishing qualifier.

    Args:
        hook: Downstream observer, or ``None`` when nobody is watching.
        detail: Short qualifier separating repeated stages, such as an index name.

    Returns:
        A forwarding hook carrying the qualifier, or ``None`` when the observer is absent.
    """
    if hook is None:
        return None

    def forward(event: ProgressEvent) -> None:
        """Forward one event with the owning command's qualifier attached."""
        hook(
            ProgressEvent(
                stage=event.stage,
                completed=event.completed,
                total=event.total,
                detail=event.detail or detail,
            )
        )

    return forward


def _key(event: ProgressEvent) -> tuple[str, str | None]:
    """Return the identity that separates one stage row from the next."""
    return (event.stage, event.detail)


def _name(event: ProgressEvent) -> str:
    """Format one stage name with its optional qualifier.

    Args:
        event: Progress update being rendered.

    Returns:
        The stage name with its optional parenthesized detail.
    """
    if event.detail:
        return f"{event.stage} ({event.detail})"
    return event.stage


def _label(event: ProgressEvent) -> str:
    """Format one stage with its qualifier and exact counts.

    Args:
        event: Progress update being rendered.

    Returns:
        The stage name, optional parenthesized detail, and ``completed/total`` counts.
    """
    parts = [_name(event)]
    if event.completed is not None and event.total is not None:
        parts.append(f"{event.completed}/{event.total}")
    return " ".join(parts)


def _duration(seconds: float) -> str:
    """Format a positive duration compactly for the in-place line.

    Args:
        seconds: Positive estimated remaining seconds.

    Returns:
        Whole seconds below one minute, then ``NmSSs``, then ``NhMMm``.
    """
    rounded = max(round(seconds), 1)
    if rounded < 60:
        return f"{rounded}s"
    minutes, secs = divmod(rounded, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def duration_text(seconds: float) -> str:
    """How long a whole command took, for its final line: ``45s``, ``12m08s`` or ``1h03m``."""
    return _duration(seconds) if seconds >= 1 else "under 1s"
