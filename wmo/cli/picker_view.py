"""One reactive terminal region shared by every keyboard-driven picker screen.

A keyboard picker owns no output of its own: it hands the current rows, focus, status, and search
state to this view, which redraws a single region in place through one rich ``Live``. An open
search shows its query under the heading and swaps the keyboard hint for the search bindings.
Long lists scroll around the focused row inside a small window so only a few options show at
once, with dim ``... more`` indicators for the rest. A console that cannot be redrawn in place,
such as a pipe or a dumb terminal, prints each frame instead, which keeps scripted and piped
sessions readable.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.text import Text

_MINIMUM_VISIBLE_ROWS = 3
_MAXIMUM_VISIBLE_ROWS = 8
_RESERVED_LINES = 6
_ROW_INDENT = " "

_MULTI_HINT = "up/down + space, Enter on Complete; / search, b back, q cancel"
_SINGLE_HINT = "up/down + Enter; / search, b back, q cancel"
_MULTI_SEARCH_HINT = "type to search; Enter keeps matches, Esc clears"
_SINGLE_SEARCH_HINT = "type to search; Enter confirms, Esc clears"


class PickerMode(StrEnum):
    """Whether a screen collects several values or exactly one."""

    MULTIPLE = "multiple"
    SINGLE = "single"


@dataclass(frozen=True)
class PickerRow:
    """One rendered row of a picker screen.

    Attributes:
        label: Row text, already carrying any provider and model identity.
        detail: Optional short annotation shown dimly after the label.
        marked: Whether a multi-select screen currently holds this row as selected.
        action: Whether the row submits the screen instead of naming a value.
    """

    label: str
    detail: str = ""
    marked: bool = False
    action: bool = False


class PickerView:
    """A single terminal region that a picker screen redraws in place.

    The view owns every visual decision: the heading, selection marks, focus pointer, scroll
    indicators for a long list, the transient status line, and the keyboard hint. Screens supply
    only state, so provider, model, role, and candidate screens stay visually identical.
    """

    def __init__(self, console: Console, *, title: str, mode: PickerMode) -> None:
        """Prepare the region for one screen without drawing anything yet.

        Args:
            console: Interactive terminal, or a stream that can only be appended to.
            title: Heading describing what is being chosen.
            mode: Whether the screen collects several values or exactly one.
        """
        self._console = console
        self._title = title
        self._mode = mode
        self._live = (
            Live(console=console, auto_refresh=False, transient=False)
            if console.is_interactive
            else None
        )

    def start(self) -> None:
        """Begin owning the region on an interactive terminal, drawing nothing yet."""
        if self._live is not None:
            self._live.start(refresh=False)

    def stop(self) -> None:
        """Release the region and leave the last frame plus a restored cursor behind."""
        if self._live is not None:
            self._live.stop()

    def show(
        self,
        rows: Sequence[PickerRow],
        *,
        focus: int,
        status: str = "",
        query: str = "",
        searching: bool = False,
    ) -> None:
        """Replace the region with the current state of the screen.

        Args:
            rows: Every row of the screen, in presentation order.
            focus: Index of the focused row.
            status: Optional message explaining why the last key changed nothing.
            query: Search text currently narrowing the rows.
            searching: Whether the search line is open for typing.
        """
        frame = self._frame(rows, focus=focus, status=status, query=query, searching=searching)
        if self._live is None:
            self._console.print(frame)
            return
        self._live.update(frame, refresh=True)

    def _frame(
        self,
        rows: Sequence[PickerRow],
        *,
        focus: int,
        status: str,
        query: str,
        searching: bool,
    ) -> RenderableType:
        """Build the complete renderable for one moment of the screen.

        Args:
            rows: Every row of the screen, in presentation order.
            focus: Index of the focused row.
            status: Optional message shown under the rows.
            query: Search text currently narrowing the rows.
            searching: Whether the search line is open for typing.

        Returns:
            The heading, the visible window of rows, and the keyboard hint as one renderable.
        """
        title = Text(self._title, style="bold")
        title.append(f"  ({self._hint(searching=False)})", style="dim")
        lines: list[Text] = [title]
        if searching:
            lines.append(Text(f"Search: {query}_", style="cyan"))
        elif query:
            lines.append(Text(f"Filter: {query}", style="dim"))
        first, last = _window(
            count=len(rows),
            focus=focus,
            capacity=self._capacity(rows, searching=searching, query=query),
        )
        if first > 0:
            lines.append(Text(f"{_ROW_INDENT}... {first} more above", style="dim"))
        for index in range(first, last):
            lines.extend(self._row_lines(rows[index], focused=index == focus))
        hidden_below = len(rows) - last
        if hidden_below > 0:
            lines.append(Text(f"{_ROW_INDENT}... {hidden_below} more below", style="dim"))
        if status:
            lines.append(Text(status, style="yellow"))
        if searching:
            lines.append(Text(self._hint(searching=True), style="dim"))
        return Group(*lines)

    def _row_lines(self, row: PickerRow, *, focused: bool) -> tuple[Text, ...]:
        """Render one row as a single line with its dim annotation inline.

        Args:
            row: Row being rendered.
            focused: Whether the row currently holds focus.

        Returns:
            One label line carrying the focus pointer, mark, and annotation.
        """
        mark = "" if self._mode is PickerMode.SINGLE or row.action else _mark(row.marked) + " "
        line = Text(_ROW_INDENT)
        if focused:
            line.append("\u276f", style="bold cyan")
        else:
            line.append(" ")
        line.append(f" {mark}{row.label}")
        if row.detail:
            line.append(f"  ({row.detail})", style="dim")
        return (line,)

    def _capacity(self, rows: Sequence[PickerRow], *, searching: bool, query: str) -> int:
        """Return how many rows fit in the region on this terminal.

        Args:
            rows: Every row of the screen, used to detect metadata lines.
            searching: Whether the search line is open for typing.
            query: Search text currently narrowing the rows.

        Returns:
            The largest row count that keeps the region inside the terminal height and
            never exceeds the compact visible-row ceiling.
        """
        del rows
        height = self._console.size.height
        reserved = _RESERVED_LINES
        if searching or query:
            reserved += 1
        available = max(_MINIMUM_VISIBLE_ROWS, height - reserved)
        return min(available, _MAXIMUM_VISIBLE_ROWS)

    def _hint(self, *, searching: bool) -> str:
        """Return the compact keyboard hint for this screen and search state.

        Args:
            searching: Whether the search line is open for typing.

        Returns:
            One short hint string.
        """
        if searching:
            return _SINGLE_SEARCH_HINT if self._mode is PickerMode.SINGLE else _MULTI_SEARCH_HINT
        return _SINGLE_HINT if self._mode is PickerMode.SINGLE else _MULTI_HINT


@contextmanager
def picker_view(console: Console, *, title: str, mode: PickerMode) -> Iterator[PickerView]:
    """Own one redraw region for the duration of a screen.

    Args:
        console: Terminal, or non-terminal stream, receiving the screen.
        title: Heading describing what is being chosen.
        mode: Whether the screen collects several values or exactly one.

    Yields:
        The view the screen redraws after every key press.
    """
    view = PickerView(console, title=title, mode=mode)
    view.start()
    try:
        yield view
    finally:
        view.stop()


def _mark(marked: bool) -> str:
    """Return the selection mark for one multi-select row."""
    return "[x]" if marked else "[ ]"


def _window(*, count: int, focus: int, capacity: int) -> tuple[int, int]:
    """Return the half-open row range that keeps the focused row visible.

    Args:
        count: Total number of rows on the screen.
        focus: Index of the focused row.
        capacity: Largest number of rows the region may show at once.

    Returns:
        First and last row indexes of the visible window, the last exclusive.
    """
    if count <= capacity:
        return 0, count
    first = min(max(0, focus - capacity // 2), count - capacity)
    return first, first + capacity
