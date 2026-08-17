"""Tests for the shared reactive picker region."""

from __future__ import annotations

import io
from collections.abc import Callable

import pytest
from rich.console import Console

from wmo.cli.picker_view import PickerMode, PickerRow, picker_view


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
    assert "  > [ ] model-3" in screen
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

    output = _output(console)
    assert "[x] openai" in output
    assert "[ ] anthropic" in output
    assert "> Complete" in output
    assert "Space or Enter selects or deselects" in output


def test_single_select_shows_metadata_without_selection_marks() -> None:
    """A single-select frame keeps provider, role, and pricing metadata but drops marks."""
    console = _terminal()
    rows = (
        PickerRow(
            label="sonnet (anthropic/claude-sonnet-4)",
            detail="roles: judge, world_model; pricing: api",
        ),
    )

    with picker_view(console, title="World model", mode=PickerMode.SINGLE) as view:
        view.show(rows, focus=0)

    output = _output(console)
    assert "> sonnet (anthropic/claude-sonnet-4)" in output
    assert "roles: judge, world_model; pricing: api" in output
    assert "[ ]" not in output
    assert "Enter confirms the focused row" in output


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
    assert "  ... 1 more above" not in "\n".join(first)
    assert any(line.startswith("  ... ") and "more below" in line for line in first)
    assert any("more above" in line for line in last)
    assert any("more below" in line for line in last)
    assert "  > model-21" in last
    assert not any(line.strip() == "model-1" for line in last)
    assert len(last) <= 14


def test_a_narrow_terminal_splits_the_keyboard_hint() -> None:
    """A narrow terminal keeps the hint readable by using two shorter lines."""
    console = _terminal(width=40)

    with picker_view(console, title="Providers", mode=PickerMode.MULTIPLE) as view:
        view.show(_rows(1, detail=True), focus=0)

    output = _output(console)
    assert "Activate Complete to submit." in output
    assert "roles: judge; pricing: api" in output


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
