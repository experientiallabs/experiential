"""Tests for role-separated judge transcript rendering helpers."""

from __future__ import annotations

from click import unstyle
from rich.console import Console

from wmo.cli.judge_render import model_name, render_field
from wmo.common.models import ModelSnapshot


def test_model_name_omits_internal_hashes() -> None:
    """Operator output shows provider and model identity only."""
    model = ModelSnapshot(
        provider="openai",
        model_id="judge-model",
        revision="rev-1",
        capabilities_sha256="0" * 64,
        connection_sha256="1" * 64,
    )

    assert model_name(model) == "openai/judge-model (revision rev-1)"


def test_render_field_marks_truncated_transcripts() -> None:
    """Truncation keeps the admitted prefix and names the omitted remainder."""
    console = Console(record=True, width=80)

    render_field(console, "Assistant output", "abcdefghij", character_limit=4)
    printed = unstyle(console.export_text())

    assert "Assistant output:" in printed
    assert "abcd" in printed
    assert "truncated 6 characters" in printed
    assert "use --page for the full transcript" in printed
