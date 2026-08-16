"""Selection prompts shared by interactive CLI setup screens.

Provider setup uses a keyboard multi-select on a real terminal: Up and Down move focus, Enter
selects or deselects the focused row, and a final Complete row submits. Other screens still read
whole lines so a long list can filter, collapse, and accept back or cancel words. The same
line-based path remains available for scripted non-terminal sessions.
"""

from __future__ import annotations

import select
import sys
import termios
import tty
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from rich.console import Console
from rich.markup import escape

_BACK_WORDS = frozenset({"b", "back"})
_CANCEL_WORDS = frozenset({"q", "quit", "cancel"})
_ALL_WORDS = frozenset({"a", "all"})
_NONE_WORDS = frozenset({"none", "clear"})
_MORE_WORDS = frozenset({"more", "m"})
_COLLAPSED_LIMIT = 12
_COMPLETE_LABEL = "Complete"
_NARROW_WIDTH = 72


class PickerAction(StrEnum):
    """Navigation the user requested instead of completing a selection."""

    BACK = "back"
    CANCEL = "cancel"


class PickerKey(StrEnum):
    """One decoded key from a keyboard multi-select list."""

    UP = "up"
    DOWN = "down"
    ENTER = "enter"
    BACK = "back"
    CANCEL = "cancel"
    IGNORE = "ignore"


@dataclass(frozen=True)
class PickerOption:
    """One selectable row with its stable value and readable presentation."""

    value: str
    label: str
    detail: str = ""


@dataclass(frozen=True)
class PickerResult:
    """Either the values the user chose, or the navigation they requested."""

    values: tuple[str, ...] = ()
    action: PickerAction | None = None


def select_many(
    console: Console,
    *,
    title: str,
    options: Sequence[PickerOption],
    preselected: Sequence[str] = (),
    minimum: int = 1,
) -> PickerResult:
    """Choose several rows from one searchable list.

    Args:
        console: Terminal used for the list and its prompt.
        title: Screen heading describing what is being chosen.
        options: Every selectable row, in presentation order.
        preselected: Values already chosen, kept when the screen is shown again.
        minimum: Smallest accepted number of selected values.

    Returns:
        The chosen values, or the requested back or cancel navigation.

    Raises:
        ValueError: The screen has no rows to choose from.
    """
    if not options:
        raise ValueError(f"{title} has no available choices")
    values = {option.value for option in options}
    selected = [value for value in preselected if value in values]
    query = ""
    expanded = False
    console.print(f"[bold]{title}[/bold]")
    while True:
        visible = _filtered(options, query)
        _render(console, visible, selected=selected, expanded=expanded, query=query)
        console.print(
            "[dim]Numbers or ranges toggle rows, 'all' selects, 'none' clears, text filters, "
            "empty line accepts, 'b' goes back, 'q' cancels.[/dim]"
        )
        answer = console.input("> ").strip()
        word = answer.casefold()
        if word in _BACK_WORDS:
            return PickerResult(action=PickerAction.BACK)
        if word in _CANCEL_WORDS:
            return PickerResult(action=PickerAction.CANCEL)
        if not answer:
            if len(selected) >= minimum:
                return PickerResult(values=tuple(selected))
            console.print(f"[yellow]Select at least {minimum}.[/yellow]")
            continue
        if word in _MORE_WORDS:
            expanded = True
            continue
        if word in _NONE_WORDS:
            selected = []
            continue
        if word in _ALL_WORDS:
            selected = _merged(selected, tuple(option.value for option in visible))
            continue
        positions = _positions(answer, count=len(visible))
        if positions is None:
            query = answer
            expanded = False
            if not _filtered(options, query):
                console.print(f"[yellow]No row matches {answer!r}.[/yellow]")
                query = ""
            continue
        for position in positions:
            value = visible[position - 1].value
            if value in selected:
                selected.remove(value)
            else:
                selected.append(value)


def select_one(
    console: Console,
    *,
    title: str,
    options: Sequence[PickerOption],
    default: str | None = None,
) -> PickerResult:
    """Choose one row from a searchable list.

    Args:
        console: Terminal used for the list and its prompt.
        title: Screen heading describing what is being chosen.
        options: Every selectable row, in presentation order.
        default: Value accepted when the user submits an empty line.

    Returns:
        The chosen value, or the requested back or cancel navigation.

    Raises:
        ValueError: The screen has no rows to choose from.
    """
    if not options:
        raise ValueError(f"{title} has no available choices")
    values = {option.value for option in options}
    fallback = default if default in values else None
    query = ""
    expanded = False
    console.print(f"[bold]{title}[/bold]")
    while True:
        visible = _filtered(options, query)
        _render(console, visible, selected=[], expanded=expanded, query=query)
        suffix = f" [dim](empty line keeps {fallback})[/dim]" if fallback is not None else ""
        console.print(
            f"[dim]Enter one number, text filters, 'b' goes back, 'q' cancels.[/dim]{suffix}"
        )
        answer = console.input("> ").strip()
        word = answer.casefold()
        if word in _BACK_WORDS:
            return PickerResult(action=PickerAction.BACK)
        if word in _CANCEL_WORDS:
            return PickerResult(action=PickerAction.CANCEL)
        if not answer and fallback is not None:
            return PickerResult(values=(fallback,))
        if word in _MORE_WORDS:
            expanded = True
            continue
        positions = _positions(answer, count=len(visible))
        if positions is not None and len(positions) == 1:
            return PickerResult(values=(visible[positions[0] - 1].value,))
        if answer in values:
            return PickerResult(values=(answer,))
        if positions is not None:
            console.print("[yellow]Enter exactly one number.[/yellow]")
            continue
        query = answer
        expanded = False
        if not _filtered(options, query):
            console.print(f"[yellow]No row matches {answer!r}.[/yellow]")
            query = ""


def _filtered(options: Sequence[PickerOption], query: str) -> tuple[PickerOption, ...]:
    """Keep the rows whose value, label, or detail contains the query text."""
    if not query:
        return tuple(options)
    needle = query.casefold()
    return tuple(
        option
        for option in options
        if needle in option.value.casefold()
        or needle in option.label.casefold()
        or needle in option.detail.casefold()
    )


def _render(
    console: Console,
    visible: Sequence[PickerOption],
    *,
    selected: Sequence[str],
    expanded: bool,
    query: str,
) -> None:
    """Print the visible rows, collapsing a long list until it is expanded."""
    if query:
        console.print(f"[dim]filter: {query}[/dim]")
    limit = len(visible) if expanded else min(len(visible), _COLLAPSED_LIMIT)
    chosen = frozenset(selected)
    for position, option in enumerate(visible[:limit], start=1):
        mark = "[x]" if option.value in chosen else "[ ]"
        detail = f" [dim]{escape(option.detail)}[/dim]" if option.detail else ""
        console.print(f"  {escape(mark)} {position}. {escape(option.label)}{detail}")
    hidden = len(visible) - limit
    if hidden > 0:
        console.print(f"  ... {hidden} more (type 'more' to show all, or type text to filter)")


def _positions(answer: str, *, count: int) -> tuple[int, ...] | None:
    """Parse one comma-separated list of row numbers and inclusive ranges."""
    positions: list[int] = []
    for part in answer.replace(" ", ",").split(","):
        if not part:
            continue
        bounds = part.split("-")
        if len(bounds) > 2 or not all(bound.isdigit() for bound in bounds):
            return None
        first = int(bounds[0])
        last = int(bounds[-1])
        if first < 1 or last < first or last > count:
            return None
        positions.extend(range(first, last + 1))
    return tuple(dict.fromkeys(positions)) or None


def _merged(selected: Sequence[str], additions: Sequence[str]) -> list[str]:
    """Append every addition that is not already selected, preserving order."""
    merged = list(selected)
    merged.extend(value for value in additions if value not in merged)
    return merged


def uses_keyboard_list(console: Console) -> bool:
    """Return whether this console can drive the keyboard multi-select list.

    Args:
        console: Terminal used for the screen.

    Returns:
        True only when both stdout and stdin belong to an interactive terminal.
    """
    return console.is_terminal and _stdin_is_tty()


def interpret_key_bytes(data: bytes) -> PickerKey:
    """Map one raw terminal key sequence to a picker action.

    Args:
        data: Bytes read from the terminal for a single key press.

    Returns:
        The decoded action, or ``IGNORE`` for an unrecognized sequence.
    """
    if data in {b"\r", b"\n"}:
        return PickerKey.ENTER
    if data in {b"b", b"B"}:
        return PickerKey.BACK
    if data in {b"q", b"Q", b"\x03"}:
        return PickerKey.CANCEL
    if data in {b"\x1b", b"\x1b\x1b"}:
        return PickerKey.CANCEL
    if data in {b"\x1b[A", b"\x1bOA"}:
        return PickerKey.UP
    if data in {b"\x1b[B", b"\x1bOB"}:
        return PickerKey.DOWN
    return PickerKey.IGNORE


def read_terminal_key() -> PickerKey:
    """Read one key from the controlling terminal in raw mode.

    Returns:
        The decoded picker action for the next key press.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        first = sys.stdin.read(1)
        if first != "\x1b":
            return interpret_key_bytes(first.encode())
        readable, _, _ = select.select([sys.stdin], [], [], 0.05)
        if not readable:
            return PickerKey.CANCEL
        rest = sys.stdin.read(1)
        if rest in {"[", "O"}:
            extra_ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if extra_ready:
                rest += sys.stdin.read(1)
        return interpret_key_bytes(("\x1b" + rest).encode())
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def select_many_list(
    console: Console,
    *,
    title: str,
    options: Sequence[PickerOption],
    preselected: Sequence[str] = (),
    minimum: int = 1,
    read_key: Callable[[], PickerKey] | None = None,
) -> PickerResult:
    """Choose several rows from a keyboard-driven list with a Complete action.

    Up and Down move focus. Enter selects or deselects the focused option. Enter on the
    Complete row submits the current selection and does nothing on any other row.

    Args:
        console: Terminal used for the list.
        title: Screen heading describing what is being chosen.
        options: Every selectable row, in presentation order.
        preselected: Values already chosen, kept when the screen is shown again.
        minimum: Smallest accepted number of selected values.
        read_key: Optional key source used by tests instead of the controlling terminal.

    Returns:
        The chosen values, or the requested back or cancel navigation.

    Raises:
        ValueError: The screen has no rows to choose from.
    """
    if not options:
        raise ValueError(f"{title} has no available choices")
    values = {option.value for option in options}
    selected = [value for value in preselected if value in values]
    focus = 0
    complete_index = len(options)
    reader = read_key if read_key is not None else read_terminal_key
    console.print(f"[bold]{title}[/bold]")
    while True:
        _render_list(
            console,
            options,
            selected=selected,
            focus=focus,
        )
        key = reader()
        if key is PickerKey.BACK:
            return PickerResult(action=PickerAction.BACK)
        if key is PickerKey.CANCEL:
            return PickerResult(action=PickerAction.CANCEL)
        if key is PickerKey.UP:
            focus = (focus - 1) % (complete_index + 1)
            continue
        if key is PickerKey.DOWN:
            focus = (focus + 1) % (complete_index + 1)
            continue
        if key is not PickerKey.ENTER:
            continue
        if focus == complete_index:
            if len(selected) >= minimum:
                return PickerResult(values=tuple(selected))
            console.print(f"[yellow]Select at least {minimum}.[/yellow]")
            continue
        value = options[focus].value
        if value in selected:
            selected.remove(value)
        else:
            selected.append(value)


def _render_list(
    console: Console,
    options: Sequence[PickerOption],
    *,
    selected: Sequence[str],
    focus: int,
) -> None:
    """Print the option rows, selection marks, focus marker, and Complete action."""
    chosen = frozenset(selected)
    width = console.width or 80
    narrow = width < _NARROW_WIDTH
    for index, option in enumerate(options):
        pointer = ">" if index == focus else " "
        mark = "[x]" if option.value in chosen else "[ ]"
        console.print(f"  {pointer} {escape(mark)} {escape(option.label)}")
        if option.detail:
            console.print(f"      [dim]{escape(option.detail)}[/dim]")
    pointer = ">" if focus == len(options) else " "
    console.print(f"  {pointer} {_COMPLETE_LABEL}")
    if narrow:
        console.print("[dim]Up/Down moves focus. Enter selects or deselects.[/dim]")
        console.print("[dim]Activate Complete to submit. b goes back, q cancels.[/dim]")
        return
    console.print(
        "[dim]Up/Down moves focus, Enter selects or deselects, Complete submits, "
        "b goes back, q cancels.[/dim]"
    )


def _stdin_is_tty() -> bool:
    """Return whether the input stream is a terminal."""
    stdin = sys.stdin
    try:
        return stdin is not None and stdin.isatty()
    except ValueError:
        return False
