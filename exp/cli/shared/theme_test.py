"""Tests for the shared Experiential Labs terminal palette."""

from __future__ import annotations

import io

from rich.console import Console

from exp.cli.shared.theme import ACCENT, ERROR, EXP_THEME, MUTED, WARNING


def _console() -> Console:
    """Return one truecolor capturing console using the shared theme."""
    return Console(
        file=io.StringIO(),
        force_terminal=True,
        color_system="truecolor",
        width=100,
        theme=EXP_THEME,
    )


def _color_name(console: Console, style: str) -> str:
    """Return the resolved foreground color name for one style.

    Args:
        console: Themed console under test.
        style: Style name to resolve.

    Returns:
        Lowercase resolved color name.
    """
    color = console.get_style(style).color
    assert color is not None
    return color.name


def test_standard_markup_names_resolve_to_the_brand_palette() -> None:
    """Existing CLI markup adopts the palette without per-call style changes."""
    console = _console()

    assert _color_name(console, "green") == ACCENT.casefold()
    assert _color_name(console, "dim") == MUTED.casefold()
    assert _color_name(console, "yellow") == WARNING.casefold()
    assert _color_name(console, "red") == ERROR.casefold()


def test_prompt_styles_use_accent_and_muted_tones() -> None:
    """Interactive choices render in accent green with muted defaults."""
    console = _console()

    assert _color_name(console, "prompt.choices") == ACCENT.casefold()
    assert _color_name(console, "prompt.default") == MUTED.casefold()
