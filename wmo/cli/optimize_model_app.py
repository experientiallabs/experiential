"""Composition and shared console for staged model optimization."""

from __future__ import annotations

from rich.console import Console

_console = Console()

from wmo.cli.optimize_model_cmd import optimize_model  # noqa: E402

__all__ = ("optimize_model",)
