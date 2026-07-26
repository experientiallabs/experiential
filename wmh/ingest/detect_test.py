"""Tests for trace-format auto-detection (`wmh.ingest.detect`)."""

from __future__ import annotations

import pytest

from wmh.ingest.detect import detect_format

# One minimal, representative payload per source shape: exactly the markers detection keys on.
_OTLP_ENVELOPE = {"resourceSpans": [{"scopeSpans": [{"spans": [{"traceId": "t1"}]}]}]}
_OTLP_BARE_SPAN = {
    "traceId": "a" * 32,
    "spanId": "b" * 16,
    "startTimeUnixNano": 1,
    "attributes": [{"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}}],
}
_CHAT = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
_LANGFUSE = {"id": "lf-1", "observations": [{"id": "o1", "type": "GENERATION"}]}
_LANGSMITH = {"id": "r1", "trace_id": "t1", "run_type": "llm", "outputs": {}}
_BRAINTRUST = {"span_id": "s1", "root_span_id": "r1", "span_attributes": {"type": "llm"}}
_POSTHOG = {"event": "$ai_generation", "properties": {"$ai_trace_id": "t1"}}
_MASTRA = {"id": "s1", "traceId": "t1", "type": "model_generation", "input": [], "output": {}}
_PHOENIX = {
    "name": "agent_step",
    "context": {"trace_id": "t1", "span_id": "s1"},
    "attributes": {"openinference.span.kind": "LLM"},
}


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (_OTLP_ENVELOPE, "otel-genai"),
        (_OTLP_BARE_SPAN, "otel-genai"),
        (_CHAT, "chat-json"),
        (_LANGFUSE, "langfuse"),
        (_LANGSMITH, "langsmith"),
        (_BRAINTRUST, "braintrust"),
        (_POSTHOG, "posthog"),
        (_MASTRA, "mastra"),
        (_PHOENIX, "phoenix"),
    ],
)
def test_detects_each_source_shape(payload: dict, expected: str) -> None:
    assert detect_format([payload]) == expected


def test_detects_inside_wrappers_and_lists() -> None:
    # API page wrappers and JSON arrays defer to their elements' shape.
    assert detect_format([{"data": [_LANGFUSE]}]) == "langfuse"
    assert detect_format([{"events": [_BRAINTRUST]}]) == "braintrust"
    assert detect_format([{"runs": [_LANGSMITH]}]) == "langsmith"
    assert detect_format([{"results": [_POSTHOG]}]) == "posthog"
    assert detect_format([{"spans": [_MASTRA]}]) == "mastra"
    assert detect_format([[_PHOENIX, _PHOENIX]]) == "phoenix"
    # A bare message list is one chat-json conversation.
    assert detect_format([[{"role": "user", "content": "hi"}]]) == "chat-json"
    # JSONL: one payload per line, all agreeing.
    assert detect_format([_OTLP_BARE_SPAN, _OTLP_BARE_SPAN]) == "otel-genai"


def test_unknown_shape_raises_with_guidance() -> None:
    with pytest.raises(ValueError, match="--source"):
        detect_format([{"nothing": "recognizable"}])
    with pytest.raises(ValueError, match="--source"):
        detect_format([])


def test_conflicting_payloads_raise_naming_both() -> None:
    with pytest.raises(ValueError, match="langfuse.*langsmith|langsmith.*langfuse"):
        detect_format([_LANGFUSE, _LANGSMITH])
