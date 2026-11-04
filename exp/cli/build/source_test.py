"""Tests for selecting the local trace path a build reads."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from exp.cli.build.source import resolve_trace_path


def test_a_session_directory_is_accepted_only_for_a_directory_source(tmp_path: Path) -> None:
    """agent-sessions reads a directory of session files; file-only sources refuse one."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    export = tmp_path / "export.json"
    export.write_text("{}")

    assert resolve_trace_path(sessions, source="agent-sessions") == sessions
    assert resolve_trace_path(export, source="chat-json") == export
    with pytest.raises(typer.BadParameter, match="not a directory"):
        resolve_trace_path(sessions, source="chat-json")
    with pytest.raises(typer.BadParameter, match="not found"):
        resolve_trace_path(tmp_path / "missing", source="agent-sessions")
