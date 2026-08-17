"""Tests for Mastra span export normalization."""

from __future__ import annotations

import json
from pathlib import Path

from wmo.simulation.ingest.mastra import MASTRA_SOURCE


def _spans(provider: str | None = "openai") -> list[dict[str, object]]:
    """Return one Mastra trace as a model span, a tool span, and an agent run span.

    Args:
        provider: Declared model provider, or ``None`` to omit the provider attribute.

    Returns:
        Mastra span records in source order.
    """
    model_attributes: dict[str, object] = {
        "model": "gpt-4.1",
        "threadId": "thread-2",
        "usage": {"promptTokens": 20, "completionTokens": 6},
    }
    if provider is not None:
        model_attributes["provider"] = provider
    return [
        {
            "traceId": "trace-1",
            "id": "span-1",
            "type": "model_generation",
            "name": "generate",
            "startTime": "2026-04-01T00:00:01Z",
            "endTime": "2026-04-01T00:00:02Z",
            "attributes": model_attributes,
            "input": {"messages": [{"role": "user", "content": "Summarize the incident"}]},
            "output": {
                "toolCalls": [
                    {
                        "toolCallId": "call-3",
                        "toolName": "fetch_incident",
                        "input": {"id": "INC-1"},
                    }
                ]
            },
        },
        {
            "traceId": "trace-1",
            "id": "span-2",
            "parentSpanId": "span-1",
            "type": "tool_call",
            "name": "fetch_incident",
            "startTime": "2026-04-01T00:00:03Z",
            "endTime": "2026-04-01T00:00:04Z",
            "attributes": {"toolName": "fetch_incident", "toolCallId": "call-3"},
            "input": {"input": {"id": "INC-1"}},
            "output": {"text": "disk pressure"},
        },
        {
            "traceId": "trace-1",
            "id": "span-3",
            "type": "agent_run",
            "name": "support-agent",
            "startTime": "2026-04-01T00:00:00Z",
            "input": {"messages": [{"role": "user", "content": "Summarize the incident"}]},
        },
    ]


def test_load_mastra_file_accepts_epoch_millisecond_timestamps(tmp_path: Path) -> None:
    """Epoch millisecond timestamps are read without changing observation order."""
    spans = _spans()
    spans[0]["startTime"] = 1_775_001_000_000
    spans[1]["startTime"] = 1_775_001_002_000
    path = tmp_path / "mastra.jsonl"
    path.write_text("\n".join(json.dumps(span) for span in spans), encoding="utf-8")

    result = MASTRA_SOURCE.load(path)

    assert [span.name for span in result.traces[0].spans] == [
        "agent.model_call",
        "agent.tool_call",
    ]
