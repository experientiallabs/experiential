"""Tests for the shared reactive picker region."""

from __future__ import annotations

import io
import re
from collections.abc import Callable

import pytest
from rich.console import Console

from wmo.cli.shared.picker_view import PickerMode, PickerRow, picker_view


def _terminal(*, width: int = 100, height: int = 24) -> Console:
    """Return a console that emits terminal control sequences into a buffer.

    Args:
        width: Reported terminal width.
        height: Reported terminal height.

    Returns:
        A console whose ``file`` collects everything the region draws.
    """
    return Console(
        file=io.StringIO(),
        force_terminal=True,
        force_interactive=True,
        no_color=True,
        width=width,
        height=height,
        legacy_windows=False,
    )


def _output(console: Console) -> str:
    """Return everything written to a buffered console.

    Args:
        console: Console created by ``_terminal``.

    Returns:
        The decoded transcript, including cursor control sequences.
    """
    assert isinstance(console.file, io.StringIO)
    return console.file.getvalue()


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _plain(console: Console) -> str:
    """Return the console transcript with ANSI sequences removed and wraps rejoined.

    Args:
        console: Console created by ``_terminal``.

    Returns:
        The visible text as one whitespace-collapsed string.
    """
    return " ".join(_ANSI.sub("", _output(console)).split())


def _rows(count: int, *, detail: bool = False) -> tuple[PickerRow, ...]:
    """Build numbered rows for a screen.

    Args:
        count: Number of rows to build.
        detail: Whether each row carries a metadata line.

    Returns:
        Rows in presentation order.
    """
    return tuple(
        PickerRow(
            label=f"model-{index + 1}",
            detail="roles: judge; pricing: api" if detail else "",
        )
        for index in range(count)
    )


def test_repeated_frames_redraw_one_region_on_a_terminal(
    rendered_screen: Callable[[str], tuple[str, ...]],
) -> None:
    """Moving focus many times leaves a single copy of the heading and every row.

    Args:
        rendered_screen: Replay of terminal output into the visible screen.
    """
    console = _terminal()
    rows = _rows(3)

    with picker_view(console, title="Select the providers", mode=PickerMode.MULTIPLE) as view:
        for focus in (0, 1, 2, 1, 0, 2):
            view.show(rows, focus=focus)

    screen = rendered_screen(_output(console))
    text = "\n".join(screen)
    assert text.count("Select the providers") == 1
    assert text.count("model-1") == 1
    assert text.count("model-3") == 1
    assert " \u276f [ ] model-3" in screen
    assert _output(console).endswith("\x1b[?25h")


def test_a_console_without_a_terminal_prints_each_frame() -> None:
    """A piped console cannot be redrawn, so every frame is printed in order."""
    console = Console(file=io.StringIO(), width=100, no_color=True)

    with picker_view(console, title="Select the providers", mode=PickerMode.MULTIPLE) as view:
        view.show(_rows(2), focus=0)
        view.show(_rows(2), focus=1)

    assert _output(console).count("Select the providers") == 2


def test_multi_select_marks_selection_and_names_the_submit_row() -> None:
    """A multi-select frame shows selection marks, and the action row carries no mark."""
    console = _terminal()
    rows = (
        PickerRow(label="openai", marked=True),
        PickerRow(label="anthropic"),
        PickerRow(label="Complete", action=True),
    )

    with picker_view(console, title="Providers", mode=PickerMode.MULTIPLE) as view:
        view.show(rows, focus=2)

    text = _plain(console)
    assert "[x] openai" in text
    assert "[ ] anthropic" in text
    assert "\u276f Complete" in text
    assert "Enter on Complete" in text


def test_single_select_shows_metadata_without_selection_marks() -> None:
    """A single-select frame keeps its dim annotation inline but drops selection marks."""
    console = _terminal()
    rows = (PickerRow(label="sonnet", detail="anthropic"),)

    with picker_view(console, title="World model", mode=PickerMode.SINGLE) as view:
        view.show(rows, focus=0)

    text = _plain(console)
    assert "\u276f sonnet (anthropic)" in text
    assert "[ ]" not in text
    assert "up/down + Enter" in text


def test_a_long_list_scrolls_around_the_focused_row(
    rendered_screen: Callable[[str], tuple[str, ...]],
) -> None:
    """A list taller than the terminal shows a window plus both hidden-row counts.

    Args:
        rendered_screen: Replay of terminal output into the visible screen.
    """
    console = _terminal(height=14)
    rows = _rows(40)

    with picker_view(console, title="Models", mode=PickerMode.SINGLE) as view:
        view.show(rows, focus=0)
        first = rendered_screen(_output(console))
        view.show(rows, focus=20)

    last = rendered_screen(_output(console))

    def _below(line: str) -> bool:
        """Match the dim indicator for rows hidden below the visible window."""
        return line.strip().startswith("\u2026") and line.strip().endswith("more")

    assert "more above" not in "\n".join(first)
    assert any(_below(line) for line in first)
    assert any("more above" in line for line in last)
    assert any(_below(line) for line in last)
    assert " \u276f model-21" in last
    assert not any(line.strip() == "model-1" for line in last)
    assert len(last) <= 14


def test_a_narrow_terminal_wraps_the_keyboard_hint() -> None:
    """A narrow terminal wraps the inline hint and annotation without losing either."""
    console = _terminal(width=40)

    with picker_view(console, title="Providers", mode=PickerMode.MULTIPLE) as view:
        view.show(_rows(1, detail=True), focus=0)

    text = _plain(console)
    assert "Enter on Complete" in text
    assert "pricing: api" in text


def test_an_open_search_shows_the_query_and_swaps_the_hint() -> None:
    """An open search draws its query with a caret and explains the search bindings."""
    console = _terminal()

    with picker_view(console, title="Models", mode=PickerMode.SINGLE) as view:
        view.show(_rows(2), focus=0, query="gpt", searching=True)

    output = _output(console)
    assert "Search: gpt_" in output
    assert "Enter confirms, Esc clears" in output


def test_a_retained_filter_is_shown_with_the_normal_hint() -> None:
    """A closed search keeps its filter visible while the normal keyboard hint returns."""
    console = _terminal()

    with picker_view(console, title="Models", mode=PickerMode.MULTIPLE) as view:
        view.show(_rows(2), focus=0, query="gpt")

    output = _output(console)
    assert "Filter: gpt" in output
    assert "Search: " not in output
    assert "up/down" in output


def test_a_searching_narrow_terminal_keeps_the_search_hint() -> None:
    """A narrow terminal still shows the search bindings while a search is open."""
    console = _terminal(width=40)

    with picker_view(console, title="Models", mode=PickerMode.MULTIPLE) as view:
        view.show(_rows(1), focus=0, query="gpt", searching=True)

    text = _plain(console)
    assert "Search: gpt_" in text
    assert "Esc clears" in text


def test_a_search_frame_still_fits_a_short_terminal(
    rendered_screen: Callable[[str], tuple[str, ...]],
) -> None:
    """The search line costs one row, and a long match list still fits the terminal.

    Args:
        rendered_screen: Replay of terminal output into the visible screen.
    """
    console = _terminal(height=14)

    with picker_view(console, title="Models", mode=PickerMode.SINGLE) as view:
        view.show(_rows(40), focus=0, query="model", searching=True)

    screen = rendered_screen(_output(console))
    assert "Search: model_" in "\n".join(screen)
    assert len(screen) <= 14, screen


def test_a_status_message_appears_with_the_frame_that_explains_it() -> None:
    """A refused submission explains itself inside the same region."""
    console = _terminal()

    with picker_view(console, title="Providers", mode=PickerMode.MULTIPLE) as view:
        view.show(_rows(1), focus=0, status="Select at least 1.")

    assert "Select at least 1." in _output(console)


def test_an_exception_inside_a_screen_still_releases_the_region() -> None:
    """The region stops and restores the cursor even when a screen fails."""
    console = _terminal()

    with pytest.raises(RuntimeError, match="picker failed"):
        with picker_view(console, title="Providers", mode=PickerMode.MULTIPLE) as view:
            view.show(_rows(2), focus=0)
            raise RuntimeError("picker failed")

    assert _output(console).endswith("\x1b[?25h")
