"""Tests for Mastra span export normalization."""

from __future__ import annotations

import json
from pathlib import Path

from wmo.simulation.ingest.mastra import load_mastra_file


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


def test_load_mastra_file_converts_model_and_tool_spans(tmp_path: Path) -> None:
    """Mastra model and tool spans become paired canonical spans with retained identity."""
    path = tmp_path / "mastra.json"
    path.write_text(json.dumps({"spans": _spans()}), encoding="utf-8")

    result = load_mastra_file(path)

    assert result.issues == ()
    trace = result.traces[0]
    assert trace.task == "Summarize the incident"
    assert trace.conversation_id == "thread-2"
    assert [span.name for span in trace.spans] == ["agent.model_call", "agent.tool_call"]
    call, tool_result = trace.spans
    assert call.attributes["gen_ai.tool.name"] == "fetch_incident"
    assert call.attributes["gen_ai.tool.call.arguments"] == '{"id":"INC-1"}'
    assert call.attributes["gen_ai.tool.call.id"] == "call-3"
    assert call.model is not None
    assert (call.model.provider, call.model.model_id) == ("openai", "gpt-4.1")
    assert call.usage is not None
    assert (call.usage.input_tokens, call.usage.output_tokens) == (20, 6)
    assert tool_result.attributes["gen_ai.tool.call.id"] == "call-3"
    assert tool_result.attributes["gen_ai.tool.message"] == "disk pressure"
    assert tool_result.parent_span_id == call.span_id


def test_load_mastra_file_accepts_epoch_millisecond_timestamps(tmp_path: Path) -> None:
    """Epoch millisecond timestamps are read without changing observation order."""
    spans = _spans()
    spans[0]["startTime"] = 1_775_001_000_000
    spans[1]["startTime"] = 1_775_001_002_000
    path = tmp_path / "mastra.jsonl"
    path.write_text("\n".join(json.dumps(span) for span in spans), encoding="utf-8")

    result = load_mastra_file(path)

    assert [span.name for span in result.traces[0].spans] == [
        "agent.model_call",
        "agent.tool_call",
    ]


def test_load_mastra_file_retains_error_info(tmp_path: Path) -> None:
    """Mastra errorInfo becomes a structured span failure and a failure outcome."""
    spans = _spans()
    spans[1]["errorInfo"] = {"message": "tool unavailable"}
    path = tmp_path / "mastra.json"
    path.write_text(json.dumps(spans), encoding="utf-8")

    result = load_mastra_file(path)

    trace = result.traces[0]
    assert trace.spans[1].failure is not None
    assert trace.outcome is not None
    assert trace.outcome.status == "failure"


def test_load_mastra_file_excludes_span_without_trace_id(tmp_path: Path) -> None:
    """A span with a blank traceId is excluded with an explicit issue."""
    spans = _spans()
    spans[0]["traceId"] = "  "
    path = tmp_path / "mastra.json"
    path.write_text(json.dumps(spans), encoding="utf-8")

    result = load_mastra_file(path)

    assert [issue.source_record for issue in result.issues] == ["record-1", "trace-trace-1"]
    assert result.traces == ()


def test_load_mastra_file_keeps_model_name_without_provider(tmp_path: Path) -> None:
    """A model span naming only a model keeps it as evidence without resolved identity."""
    spans = _spans(provider=None)
    path = tmp_path / "mastra.json"
    path.write_text(json.dumps(spans), encoding="utf-8")

    result = load_mastra_file(path)

    span = result.traces[0].spans[0]
    assert span.model is None
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1"
