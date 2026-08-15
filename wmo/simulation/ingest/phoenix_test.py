"""Tests for Phoenix and OpenInference span export normalization."""

from __future__ import annotations

import json
from pathlib import Path

from wmo.simulation.ingest.phoenix import load_phoenix_file


def _native_spans(provider: str | None = "openai") -> list[dict[str, object]]:
    """Return one Phoenix trace as an LLM span, a tool span, and a chain span.

    Args:
        provider: Declared model provider, or ``None`` to omit the provider attribute.

    Returns:
        Native Phoenix span records in source order.
    """
    llm_attributes: dict[str, object] = {
        "model_name": "gpt-4o-mini",
        "token_count": {"prompt": 40, "completion": 5},
        "input_messages": [{"message": {"role": "user", "content": "Refund my order"}}],
        "output_messages": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "tool_call": {
                                "id": "call-5",
                                "function": {
                                    "name": "refund_order",
                                    "arguments": '{"order": "A1"}',
                                },
                            }
                        }
                    ],
                }
            }
        ],
    }
    if provider is not None:
        llm_attributes["provider"] = provider
    return [
        {
            "context": {"trace_id": "trace-1", "span_id": "span-1"},
            "name": "ChatCompletion",
            "start_time": "2026-05-01T00:00:01Z",
            "end_time": "2026-05-01T00:00:02Z",
            "attributes": {
                "openinference": {"span": {"kind": "LLM"}},
                "llm": llm_attributes,
            },
        },
        {
            "context": {"trace_id": "trace-1", "span_id": "span-2"},
            "parent_id": "span-1",
            "name": "refund_order",
            "start_time": "2026-05-01T00:00:03Z",
            "end_time": "2026-05-01T00:00:04Z",
            "attributes": {
                "openinference": {"span": {"kind": "TOOL"}},
                "tool": {"name": "refund_order", "parameters": {"order": "A1"}},
                "output": {"value": "refunded"},
            },
        },
        {
            "context": {"trace_id": "trace-1", "span_id": "span-3"},
            "name": "AgentExecutor",
            "start_time": "2026-05-01T00:00:00Z",
            "end_time": "2026-05-01T00:00:05Z",
            "attributes": {"openinference": {"span": {"kind": "CHAIN"}}},
        },
    ]


def test_load_phoenix_file_converts_native_llm_and_tool_spans(tmp_path: Path) -> None:
    """Native Phoenix LLM and TOOL spans become paired canonical spans."""
    path = tmp_path / "phoenix.json"
    path.write_text(json.dumps(_native_spans()), encoding="utf-8")

    result = load_phoenix_file(path)

    assert result.issues == ()
    trace = result.traces[0]
    assert trace.task == "Refund my order"
    assert [span.name for span in trace.spans] == ["agent.model_call", "agent.tool_call"]
    call, tool_result = trace.spans
    assert call.attributes["gen_ai.tool.name"] == "refund_order"
    assert call.attributes["gen_ai.tool.call.id"] == "call-5"
    assert call.model is not None
    assert (call.model.provider, call.model.model_id) == ("openai", "gpt-4o-mini")
    assert call.usage is not None
    assert (call.usage.input_tokens, call.usage.output_tokens) == (40, 5)
    assert tool_result.attributes["gen_ai.tool.call.id"] == "call-5"
    assert tool_result.attributes["gen_ai.tool.message"] == "refunded"
    assert tool_result.parent_span_id == call.span_id


def test_load_phoenix_file_converts_flat_dotted_rows(tmp_path: Path) -> None:
    """Flat span rows with dotted column names normalize like native spans."""
    rows = [
        {
            "context.trace_id": "trace-9",
            "context.span_id": "span-9",
            "name": "ChatCompletion",
            "start_time": "2026-05-02T00:00:01Z",
            "end_time": "2026-05-02T00:00:02Z",
            "attributes.openinference.span.kind": "LLM",
            "attributes.llm.provider": "anthropic",
            "attributes.llm.model_name": "claude-sonnet-4",
            "attributes.input.value": "Where is my package",
            "attributes.output.value": "It ships tomorrow",
        }
    ]
    path = tmp_path / "phoenix_rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    result = load_phoenix_file(path)

    trace = result.traces[0]
    assert trace.task == "Where is my package"
    span = trace.spans[0]
    assert span.model is not None
    assert (span.model.provider, span.model.model_id) == ("anthropic", "claude-sonnet-4")
    assert span.attributes["gen_ai.completion"] == "It ships tomorrow"


def test_load_phoenix_file_converts_otlp_envelope(tmp_path: Path) -> None:
    """An OTLP envelope carrying OpenInference attributes normalizes to one trace."""
    envelope = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "6b8b45671f2c4a7dbb1e9c2a3d4e5f60",
                                "spanId": "1a2b3c4d5e6f7a8b",
                                "name": "ChatCompletion",
                                "startTimeUnixNano": "1778000000000000000",
                                "endTimeUnixNano": 1778000001000000000,
                                "status": {"code": 2, "message": "rate limited"},
                                "attributes": [
                                    {
                                        "key": "openinference.span.kind",
                                        "value": {"stringValue": "LLM"},
                                    },
                                    {"key": "llm.provider", "value": {"stringValue": "openai"}},
                                    {"key": "llm.model_name", "value": {"stringValue": "gpt-4o"}},
                                    {
                                        "key": "input.value",
                                        "value": {"stringValue": "Cancel my plan"},
                                    },
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }
    path = tmp_path / "phoenix_otlp.json"
    path.write_text(json.dumps(envelope), encoding="utf-8")

    result = load_phoenix_file(path)

    trace = result.traces[0]
    assert trace.task == "Cancel my plan"
    assert trace.spans[0].attributes["wmo.source.span.id"] == "1a2b3c4d5e6f7a8b"
    assert trace.spans[0].failure is not None
    assert trace.outcome is not None
    assert trace.outcome.status == "failure"


def test_load_phoenix_file_excludes_span_without_end_time(tmp_path: Path) -> None:
    """A span with no end time is excluded with an explicit issue."""
    spans = _native_spans()
    del spans[0]["end_time"]
    path = tmp_path / "phoenix.json"
    path.write_text(json.dumps(spans), encoding="utf-8")

    result = load_phoenix_file(path)

    assert [issue.source_record for issue in result.issues] == ["record-1", "trace-trace-1"]
    assert result.traces == ()


def test_load_phoenix_file_keeps_model_name_without_provider(tmp_path: Path) -> None:
    """An LLM span naming only a model keeps it as evidence without resolved identity."""
    spans = _native_spans(provider=None)
    path = tmp_path / "phoenix.json"
    path.write_text(json.dumps(spans), encoding="utf-8")

    result = load_phoenix_file(path)

    span = result.traces[0].spans[0]
    assert span.model is None
    assert span.attributes["gen_ai.request.model"] == "gpt-4o-mini"
