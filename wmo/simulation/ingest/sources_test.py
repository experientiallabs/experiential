"""Tests for declared trace-source resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmo.simulation.ingest.sources import (
    CANONICAL_TRACE_SOURCES,
    TraceSourceError,
    load_trace_source,
)


def _chat_json_export(path: Path) -> Path:
    """Write one minimal chat conversation export.

    Args:
        path: Directory receiving the export.

    Returns:
        Path of the written export.
    """
    export = path / "chat.json"
    export.write_text(
        json.dumps(
            {
                "conversation_id": "conversation-1",
                "messages": [
                    {"role": "user", "content": "plan the trip"},
                    {"role": "assistant", "content": "here is the plan"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return export


def test_declared_sources_are_the_supported_set() -> None:
    """The declared source names are exactly the supported normalizers, sorted."""
    assert CANONICAL_TRACE_SOURCES == (
        "braintrust",
        "chat-json",
        "langfuse",
        "langsmith",
        "mastra",
        "otel-genai",
        "otlp",
        "phoenix",
        "posthog",
    )


def test_load_trace_source_dispatches_to_the_declared_loader(tmp_path: Path) -> None:
    """A declared source normalizes its own export through one canonical loader."""
    result = load_trace_source("chat-json", _chat_json_export(tmp_path))

    assert len(result.traces) == 1
    assert result.traces[0].source.identity.kind == "file"
    assert result.issues == ()


def test_load_trace_source_normalizes_the_declared_name(tmp_path: Path) -> None:
    """Surrounding whitespace and letter case do not change the resolved loader."""
    result = load_trace_source("  Chat-JSON ", _chat_json_export(tmp_path))

    assert len(result.traces) == 1


def test_load_trace_source_rejects_an_undeclared_source(tmp_path: Path) -> None:
    """An unsupported name fails closed and lists the supported names."""
    with pytest.raises(TraceSourceError, match="unsupported trace source 'weave'"):
        load_trace_source("weave", _chat_json_export(tmp_path))


def test_load_trace_source_reports_the_source_that_failed(tmp_path: Path) -> None:
    """A source-specific failure is raised as one seam error naming the declared source."""
    with pytest.raises(TraceSourceError, match="chat-json normalization failed"):
        load_trace_source("chat-json", tmp_path / "absent.json")
