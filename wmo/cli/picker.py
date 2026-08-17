"""Selection prompts shared by interactive CLI setup screens.

Every provider, model, role, and candidate screen is keyboard driven on a real terminal: Up and
Down move focus inside one region redrawn in place by ``wmo.cli.picker_view``, Space or Enter
toggles a row on a multi-select screen whose Complete row submits, and Enter confirms the focused
row on a single-select screen. A console without a terminal cannot be driven by raw keys, so the
same screens fall back to the line-based path, which reads whole lines and accepts numbers, ranges,
filter text, and back or cancel words.
"""

from __future__ import annotations

import os
import select
import sys
import termios
import tty
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from functools import partial

from rich.console import Console
from rich.markup import escape

from wmo.cli.picker_view import PickerMode, PickerRow, picker_view

_BACK_WORDS = frozenset({"b", "back"})
_CANCEL_WORDS = frozenset({"q", "quit", "cancel"})
_ALL_WORDS = frozenset({"a", "all"})
_NONE_WORDS = frozenset({"none", "clear"})
_MORE_WORDS = frozenset({"more", "m"})
_COLLAPSED_LIMIT = 12
_COMPLETE_LABEL = "Complete"


class PickerAction(StrEnum):
    """Navigation the user requested instead of completing a selection."""

    BACK = "back"
    CANCEL = "cancel"


class PickerKey(StrEnum):
    """One decoded key from a keyboard-driven picker screen."""

    UP = "up"
    DOWN = "down"
    ENTER = "enter"
    SPACE = "space"
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
    if data == b" ":
        return PickerKey.SPACE
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


def _read_terminal_key_from_fd(fd: int) -> PickerKey:
    """Read one key sequence without passing through a buffered text stream.

    Args:
        fd: Raw terminal file descriptor already configured for immediate input.

    Returns:
        The decoded picker action for the next key press.
    """
    first = os.read(fd, 1)
    if first != b"\x1b":
        return interpret_key_bytes(first)
    readable, _, _ = select.select([fd], [], [], 0.05)
    if not readable:
        return PickerKey.CANCEL
    rest = os.read(fd, 1)
    if rest in {b"[", b"O"}:
        extra_ready, _, _ = select.select([fd], [], [], 0.05)
        if extra_ready:
            rest += os.read(fd, 1)
    return interpret_key_bytes(first + rest)


@contextmanager
def _terminal_key_reader(
    read_key: Callable[[], PickerKey] | None,
) -> Iterator[Callable[[], PickerKey]]:
    """Yield a scripted reader or hold the controlling terminal in raw input mode.

    Args:
        read_key: Optional injected reader that requires no terminal configuration.

    Yields:
        A zero-argument reader for one decoded keyboard event.
    """
    if read_key is not None:
        yield read_key
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        raw = termios.tcgetattr(fd)
        raw[1] = old[1]
        termios.tcsetattr(fd, termios.TCSANOW, raw)
        yield partial(_read_terminal_key_from_fd, fd)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_terminal_key() -> PickerKey:
    """Read one key from the controlling terminal in raw mode.

    Returns:
        The decoded picker action for the next key press.
    """
    with _terminal_key_reader(None) as reader:
        return reader()


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

    Up and Down move focus. Space or Enter selects or deselects the focused option. Either key on
    the Complete row submits the current selection.

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
    status = ""
    with (
        _terminal_key_reader(read_key) as reader,
        picker_view(console, title=title, mode=PickerMode.MULTIPLE) as view,
    ):
        while True:
            view.show(
                _multi_rows(options, selected=selected),
                focus=focus,
                status=status,
            )
            key = reader()
            status = ""
            if key is PickerKey.BACK:
                return PickerResult(action=PickerAction.BACK)
            if key is PickerKey.CANCEL:
                return PickerResult(action=PickerAction.CANCEL)
            if key in (PickerKey.UP, PickerKey.DOWN):
                focus = _moved(focus, key=key, count=complete_index + 1)
                continue
            if key not in (PickerKey.ENTER, PickerKey.SPACE):
                continue
            if focus == complete_index:
                if len(selected) >= minimum:
                    return PickerResult(values=tuple(selected))
                status = f"Select at least {minimum}."
                continue
            value = options[focus].value
            if value in selected:
                selected.remove(value)
            else:
                selected.append(value)


def select_one_list(
    console: Console,
    *,
    title: str,
    options: Sequence[PickerOption],
    default: str | None = None,
    read_key: Callable[[], PickerKey] | None = None,
) -> PickerResult:
    """Choose exactly one row from a keyboard-driven list.

    Up and Down move focus, and Enter confirms the focused row immediately. A prior answer starts
    focused so the same key confirms it again.

    Args:
        console: Terminal used for the list.
        title: Screen heading describing what is being chosen.
        options: Every selectable row, in presentation order.
        default: Value focused first when it is still available.
        read_key: Optional key source used by tests instead of the controlling terminal.

    Returns:
        The chosen value, or the requested back or cancel navigation.

    Raises:
        ValueError: The screen has no rows to choose from.
    """
    if not options:
        raise ValueError(f"{title} has no available choices")
    focus = next(
        (index for index, option in enumerate(options) if option.value == default),
        0,
    )
    rows = tuple(PickerRow(label=option.label, detail=option.detail) for option in options)
    with (
        _terminal_key_reader(read_key) as reader,
        picker_view(console, title=title, mode=PickerMode.SINGLE) as view,
    ):
        while True:
            view.show(rows, focus=focus)
            key = reader()
            if key is PickerKey.BACK:
                return PickerResult(action=PickerAction.BACK)
            if key is PickerKey.CANCEL:
                return PickerResult(action=PickerAction.CANCEL)
            if key in (PickerKey.UP, PickerKey.DOWN):
                focus = _moved(focus, key=key, count=len(options))
                continue
            if key in (PickerKey.ENTER, PickerKey.SPACE):
                return PickerResult(values=(options[focus].value,))


def choose_many(
    console: Console,
    *,
    title: str,
    options: Sequence[PickerOption],
    preselected: Sequence[str] = (),
    minimum: int = 1,
    read_key: Callable[[], PickerKey] | None = None,
) -> PickerResult:
    """Collect several values from the keyboard list, or from typed lines without a terminal.

    Args:
        console: Terminal used for the screen.
        title: Screen heading describing what is being chosen.
        options: Every selectable row, in presentation order.
        preselected: Values already chosen, kept when the screen is shown again.
        minimum: Smallest accepted number of selected values.
        read_key: Optional key source used by tests instead of the controlling terminal.

    Returns:
        The chosen values, or the requested back or cancel navigation.
    """
    if read_key is not None or uses_keyboard_list(console):
        return select_many_list(
            console,
            title=title,
            options=options,
            preselected=preselected,
            minimum=minimum,
            read_key=read_key,
        )
    return select_many(
        console,
        title=title,
        options=options,
        preselected=preselected,
        minimum=minimum,
    )


def choose_one(
    console: Console,
    *,
    title: str,
    options: Sequence[PickerOption],
    default: str | None = None,
    read_key: Callable[[], PickerKey] | None = None,
) -> PickerResult:
    """Collect one value from the keyboard list, or from typed lines without a terminal.

    Args:
        console: Terminal used for the screen.
        title: Screen heading describing what is being chosen.
        options: Every selectable row, in presentation order.
        default: Value focused first, or accepted by an empty line on the line-based path.
        read_key: Optional key source used by tests instead of the controlling terminal.

    Returns:
        The chosen value, or the requested back or cancel navigation.
    """
    if read_key is not None or uses_keyboard_list(console):
        return select_one_list(
            console,
            title=title,
            options=options,
            default=default,
            read_key=read_key,
        )
    return select_one(console, title=title, options=options, default=default)


def _multi_rows(
    options: Sequence[PickerOption],
    *,
    selected: Sequence[str],
) -> tuple[PickerRow, ...]:
    """Build the rendered rows of a multi-select screen, ending with the Complete action.

    Args:
        options: Every selectable row, in presentation order.
        selected: Values currently selected.

    Returns:
        One row per option, plus the Complete row that submits the screen.
    """
    chosen = frozenset(selected)
    rows = [
        PickerRow(label=option.label, detail=option.detail, marked=option.value in chosen)
        for option in options
    ]
    rows.append(PickerRow(label=_COMPLETE_LABEL, action=True))
    return tuple(rows)


def _moved(focus: int, *, key: PickerKey, count: int) -> int:
    """Return the focus index after one Up or Down key, wrapping at both ends.

    Args:
        focus: Current focus index.
        key: Decoded Up or Down key.
        count: Number of focusable rows.

    Returns:
        The next focus index.
    """
    step = -1 if key is PickerKey.UP else 1
    return (focus + step) % count


def _stdin_is_tty() -> bool:
    """Return whether the input stream is a terminal."""
    stdin = sys.stdin
    try:
        return stdin is not None and stdin.isatty()
    except ValueError:
        return False
