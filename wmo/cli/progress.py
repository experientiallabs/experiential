"""One in-place progress line shared by every long-running CLI command.

An interactive terminal shows a single rich ``Live`` line naming the active stage with its exact
completed and total counts; each finished stage is printed once above the line as a permanent
``[x]`` row, matching the picker screens. A non-interactive stream receives stable
newline-delimited stage updates with no cursor control, so piped and scripted sessions stay
readable line by line.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.live import Live
from rich.text import Text

from wmo.common.progress import ProgressEvent, ProgressHook

_ROW_INDENT = "  "


class ProgressDisplay:
    """Renders progress events for one command run on one console."""

    def __init__(self, console: Console) -> None:
        """Prepare the display without drawing anything yet.

        Args:
            console: Interactive terminal, or a stream that can only be appended to.
        """
        self._console = console
        self._live = (
            Live(console=console, auto_refresh=False, transient=True)
            if console.is_interactive
            else None
        )
        self._current: ProgressEvent | None = None

    def start(self) -> None:
        """Begin owning the in-place line on an interactive terminal."""
        if self._live is not None:
            self._live.start(refresh=False)

    def stop(self) -> None:
        """Print the final stage as finished and release the in-place line."""
        if self._current is not None:
            self._print_finished(self._current)
            self._current = None
        if self._live is not None:
            self._live.stop()

    def abort(self) -> None:
        """Print the interrupted stage as unfinished and release the in-place line."""
        if self._current is not None:
            if self._live is not None:
                self._live.console.print(
                    Text(f"{_ROW_INDENT}[ ] {_label(self._current)}", style="red")
                )
            self._current = None
        if self._live is not None:
            self._live.stop()

    def observe(self, event: ProgressEvent) -> None:
        """Render one truthful stage update.

        Args:
            event: Current stage name with optional exact completed and total counts.
        """
        if self._live is None:
            self._console.print(f"{_ROW_INDENT}. {_label(event)}", markup=False, highlight=False)
            return
        if self._current is not None and _key(self._current) != _key(event):
            self._print_finished(self._current)
        self._current = event
        self._live.update(Text(f"{_ROW_INDENT}> {_label(event)}", style="cyan"), refresh=True)

    def _print_finished(self, event: ProgressEvent) -> None:
        """Print one permanent finished-stage row above the in-place line.

        Args:
            event: Last update observed for the stage that just finished.
        """
        if self._live is None:
            return
        self._live.console.print(Text(f"{_ROW_INDENT}[x] {_label(event)}", style="dim"))


@contextmanager
def progress_display(console: Console) -> Iterator[ProgressHook]:
    """Own one progress region for the duration of a long-running command section.

    Args:
        console: Terminal, or non-terminal stream, receiving the updates.

    Yields:
        The hook the command hands to its long-running services.
    """
    display = ProgressDisplay(console)
    display.start()
    try:
        yield display.observe
    except BaseException:
        display.abort()
        raise
    display.stop()


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


def _label(event: ProgressEvent) -> str:
    """Format one stage with its qualifier and exact counts.

    Args:
        event: Progress update being rendered.

    Returns:
        The stage name, optional parenthesized detail, and ``completed/total`` counts.
    """
    parts = [event.stage]
    if event.detail:
        parts.append(f"({event.detail})")
    if event.completed is not None and event.total is not None:
        parts.append(f"{event.completed}/{event.total}")
    return " ".join(parts)
