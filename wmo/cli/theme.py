"""Experiential Labs terminal palette shared by every WMO command console.

The palette mirrors the Experiential Labs website: a pale mint accent over dark surfaces, a
muted gray-green for secondary detail, warm amber for warnings, and a soft red for failures.
Standard markup names used across the CLI (``green``, ``dim``, ``yellow``, ``red``) map onto
the palette so every command renders one consistent brand without per-call style changes.
"""

from __future__ import annotations

from rich.theme import Theme

ACCENT = "#b6f7c8"
MUTED = "#626b65"
WARNING = "#e3b341"
ERROR = "#c25b4e"

WMO_THEME = Theme(
    {
        "green": ACCENT,
        "dim": MUTED,
        "yellow": WARNING,
        "red": ERROR,
        "rule.line": MUTED,
        "prompt.choices": ACCENT,
        "prompt.default": MUTED,
        "prompt.invalid": ERROR,
    }
)
