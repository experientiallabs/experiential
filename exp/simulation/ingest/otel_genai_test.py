"""Tests for exported OpenTelemetry GenAI span normalization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.simulation.ingest.otel_genai import load_otel_genai_file
from exp.simulation.ingest.vendor_records import VendorTraceFormatError


def _spans() -> list[dict[str, object]]:
    """Return one exported GenAI trace as a model span and its tool span."""
    return [
        {
            "trace_id": "9" * 32,
            "span_id": "a" * 16,
            "name": "agent.model_call",
            "start_time": "2026-06-01T00:00:01Z",
            "end_time": "2026-06-01T00:00:02Z",
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": "openai",
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.input.messages": json.dumps(
                    [{"role": "user", "content": "Reset my password"}]
                ),
                "gen_ai.tool.name": "reset_password",
                "gen_ai.tool.call.id": "call-1",
                "gen_ai.usage.input_tokens": 15,
                "gen_ai.usage.output_tokens": 2,
            },
        },
        {
            "trace_id": "9" * 32,
            "span_id": "b" * 16,
            "parent_span_id": "a" * 16,
            "name": "agent.tool_call",
            "start_time": "2026-06-01T00:00:03Z",
            "end_time": "2026-06-01T00:00:04Z",
            "attributes": {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "reset_password",
                "gen_ai.tool.call.id": "call-1",
                "gen_ai.tool.message": "password reset",
            },
        },
    ]


def test_load_otel_genai_file_preserves_opaque_source_identity(tmp_path: Path) -> None:
    """Opaque identifiers are mapped deterministically and retained as source evidence."""
    spans = _spans()
    spans[0]["trace_id"] = "session-77"
    spans[1]["trace_id"] = "session-77"
    spans[0]["span_id"] = "call-a"
    spans[1]["span_id"] = "tool-b"
    spans[1]["parent_span_id"] = "call-a"
    path = tmp_path / "otel.jsonl"
    path.write_text("\n".join(json.dumps(span) for span in spans), encoding="utf-8")

    result = load_otel_genai_file(path)

    trace = result.traces[0]
    call, tool_result = trace.spans
    assert call.attributes["exp.source.trace.id"] == "session-77"
    assert call.attributes["exp.source.span.id"] == "call-a"
    assert tool_result.parent_span_id == call.span_id
    assert len(trace.trace_id) == 32


def test_load_otel_genai_file_accepts_otlp_envelope(tmp_path: Path) -> None:
    """An OTLP envelope is passed through to the canonical normalizer unchanged."""
    envelope = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "9" * 32,
                                "spanId": "a" * 16,
                                "name": "agent.model_call",
                                "startTimeUnixNano": "1780000000000000000",
                                "endTimeUnixNano": "1780000001000000000",
                                "attributes": [
                                    {
                                        "key": "gen_ai.operation.name",
                                        "value": {"stringValue": "chat"},
                                    },
                                    {
                                        "key": "gen_ai.provider.name",
                                        "value": {"stringValue": "openai"},
                                    },
                                    {
                                        "key": "gen_ai.request.model",
                                        "value": {"stringValue": "gpt-4o"},
                                    },
                                    {
                                        "key": "gen_ai.input.messages",
                                        "value": {
                                            "stringValue": json.dumps(
                                                [{"role": "user", "content": "Ping"}]
                                            )
                                        },
                                    },
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }
    path = tmp_path / "otel_envelope.json"
    path.write_text(json.dumps(envelope), encoding="utf-8")

    result = load_otel_genai_file(path)

    assert result.traces[0].task == "Ping"


def test_load_otel_genai_file_retains_malformed_jsonl_issue(tmp_path: Path) -> None:
    """A malformed JSONL line is retained as an explicit issue without repair."""
    spans = _spans()
    lines = [json.dumps(spans[0]), "{not json}", json.dumps(spans[1])]
    path = tmp_path / "otel.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")

    result = load_otel_genai_file(path)

    assert [issue.source_record for issue in result.issues] == ["line-2"]
    assert len(result.traces[0].spans) == 2


def test_load_otel_genai_file_rejects_unsupported_document(tmp_path: Path) -> None:
    """A document that declares no span records fails loudly."""
    path = tmp_path / "otel.json"
    path.write_text(json.dumps({"unexpected": True}), encoding="utf-8")

    result = load_otel_genai_file(path)

    assert result.traces == ()
    assert result.issues[0].source_record == "record-1"


def test_load_otel_genai_file_requires_an_existing_file(tmp_path: Path) -> None:
    """A missing export path fails loudly."""
    with pytest.raises(VendorTraceFormatError):
        load_otel_genai_file(tmp_path / "missing.json")
