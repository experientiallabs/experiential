"""Tests for the Arize Phoenix adapter against realistic Phoenix span exports.

Phoenix exports flat OpenInference span dicts whose ids live under `context` and whose timestamps
are ISO strings (NOT OTLP `traceId`/`startTimeUnixNano`), so these fixtures exercise the adapter's
own field mapping plus the shared OpenInference classifier. No network; file fixtures only.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from wmh.core.types import ActionKind
from wmh.ingest.adapter import VendorPull, get_adapter
from wmh.ingest.normalize import spans_to_traces
from wmh.ingest.phoenix import PhoenixAdapter

# A small, realistic Phoenix span export: an AGENT/LLM span issuing a tool call (OpenInference
# `tool.name` + `input.value`), followed by the TOOL span carrying the tool's `output.value`.
_PHOENIX_SPANS = [
    {
        "name": "agent_step",
        "context": {
            "trace_id": "f1e2d3c4b5a6978800112233445566aa",
            "span_id": "aaaa000000000001",
        },
        "parent_id": None,
        "start_time": "2024-01-01T00:00:00.000000+00:00",
        "end_time": "2024-01-01T00:00:00.500000+00:00",
        "status_code": "OK",
        "attributes": {
            "openinference.span.kind": "LLM",
            "llm.model_name": "gpt-4o",
            # An LLM span's prompt lives under llm.input_messages (a real OpenInference key); the
            # structured tool-call args ride on the following TOOL span's input.value.
            "llm.input_messages": "look up user u1",
            "tool.name": "get_user",
        },
    },
    {
        "name": "get_user",
        "context": {
            "trace_id": "f1e2d3c4b5a6978800112233445566aa",
            "span_id": "aaaa000000000002",
        },
        "parent_id": "aaaa000000000001",
        "start_time": "2024-01-01T00:00:00.600000+00:00",
        "end_time": "2024-01-01T00:00:00.800000+00:00",
        "status_code": "OK",
        "attributes": {
            "openinference.span.kind": "TOOL",
            "tool.name": "get_user",
            "input.value": '{"id": "u1"}',
            "output.value": "user u1: Ada Lovelace",
        },
    },
]


def test_from_file_maps_phoenix_native_spans(tmp_path: Path) -> None:
    path = tmp_path / "phoenix_spans.json"
    path.write_text(json.dumps(_PHOENIX_SPANS), encoding="utf-8")

    traces = PhoenixAdapter().from_file(str(path))

    assert len(traces) == 1
    assert traces[0].trace_id == "f1e2d3c4b5a6978800112233445566aa"
    assert traces[0].source.startswith("phoenix:")
    assert len(traces[0].steps) == 1
    step = traces[0].steps[0]
    assert step.action.kind == ActionKind.TOOL_CALL
    assert step.action.name == "get_user"
    # The LLM span had no args; the normalizer backfills from the TOOL span's input.value.
    assert step.action.arguments == {"id": "u1"}
    assert step.observation.content == "user u1: Ada Lovelace"
    assert step.observation.is_error is False


def test_from_file_handles_jsonl_and_error_status(tmp_path: Path) -> None:
    # One span per line (JSONL), and a TOOL span flagged with an ERROR status_code.
    llm_span = {
        "name": "agent_step",
        "context": {"trace_id": "e" * 32, "span_id": "bbbb000000000001"},
        "start_time": "2024-02-02T00:00:00.000000+00:00",
        "status_code": "OK",
        "attributes": {"openinference.span.kind": "LLM", "tool.name": "get_user"},
    }
    tool_span = {
        "name": "get_user",
        "context": {"trace_id": "e" * 32, "span_id": "bbbb000000000002"},
        "start_time": "2024-02-02T00:00:01.000000+00:00",
        "status_code": "ERROR",
        "attributes": {
            "openinference.span.kind": "TOOL",
            "input.value": '{"id": "u1"}',
            "output.value": "not found",
        },
    }
    path = tmp_path / "phoenix_spans.jsonl"
    path.write_text(f"{json.dumps(llm_span)}\n{json.dumps(tool_span)}\n", encoding="utf-8")

    traces = PhoenixAdapter().from_file(str(path))

    assert len(traces) == 1
    step = traces[0].steps[0]
    assert step.observation.is_error is True
    assert step.observation.content == "not found"


def test_from_file_accepts_otlp_envelope(tmp_path: Path) -> None:
    # Phoenix can also export standard OTLP spans; those delegate to the shared collect_spans.
    otlp = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "0a0b0c0d0e0f00112233445566778899",
                                "spanId": "1111000000000001",
                                "name": "chat",
                                "startTimeUnixNano": 1,
                                "attributes": {
                                    "openinference.span.kind": "LLM",
                                    "tool.name": "search",
                                    "input.value": '{"q": "hi"}',
                                },
                            },
                            {
                                "traceId": "0a0b0c0d0e0f00112233445566778899",
                                "spanId": "1111000000000002",
                                "name": "search",
                                "startTimeUnixNano": 2,
                                "attributes": {
                                    "openinference.span.kind": "TOOL",
                                    "output.value": "1 result",
                                },
                            },
                        ]
                    }
                ]
            }
        ]
    }
    path = tmp_path / "phoenix_otlp.json"
    path.write_text(json.dumps(otlp), encoding="utf-8")

    traces = PhoenixAdapter().from_file(str(path))

    assert len(traces) == 1
    step = traces[0].steps[0]
    assert step.action.name == "search"
    assert step.action.arguments == {"q": "hi"}
    assert step.observation.content == "1 result"


def test_dataframe_records_flat_columns_indexed_tool_calls() -> None:
    """The REAL `get_spans_dataframe().reset_index().to_dict("records")` shape: ids and attributes
    are FLAT dotted columns (`context.trace_id`, `attributes.<key>`), the tool call rides on the LLM
    span as INDEXED OpenInference keys (`...output_messages.0.message.tool_calls.0...`), and
    `start_time` is a `datetime` object, not an ISO string. Passed in-memory (a dataframe can't
    round-trip a datetime through JSON), which is how the SDK export is actually consumed."""
    tc = "attributes.llm.output_messages.0.message.tool_calls.0.tool_call.function"
    records = [
        {
            "name": "llm",
            "context.trace_id": "d" * 32,
            "context.span_id": "cccc000000000001",
            "parent_id": None,
            "start_time": datetime(2024, 3, 3, tzinfo=UTC),
            "status_code": "OK",
            "attributes.openinference.span.kind": "LLM",
            "attributes.llm.model_name": "gpt-4o",
            f"{tc}.name": "get_user",
            f"{tc}.arguments": '{"id": "u1"}',
        },
        {
            "name": "get_user",
            "context.trace_id": "d" * 32,
            "context.span_id": "cccc000000000002",
            "parent_id": "cccc000000000001",
            "start_time": datetime(2024, 3, 3, 0, 0, 1, tzinfo=UTC),
            "status_code": "OK",
            "attributes.openinference.span.kind": "TOOL",
            "attributes.output.value": "user u1: Ada Lovelace",
        },
    ]

    adapter = PhoenixAdapter()
    traces = spans_to_traces(adapter._collect_all([records]), source="phoenix:df")

    assert len(traces) == 1
    assert traces[0].trace_id == "d" * 32
    step = traces[0].steps[0]
    assert step.action.kind == ActionKind.TOOL_CALL
    assert step.action.name == "get_user"
    assert step.action.arguments == {"id": "u1"}
    assert step.observation.content == "user u1: Ada Lovelace"


def test_registered_under_phoenix() -> None:
    assert get_adapter("phoenix").name == "phoenix"


def test_vendor_pull_left_as_friendly_default() -> None:
    try:
        PhoenixAdapter().from_vendor(VendorPull())
    except ValueError as exc:
        assert "does not support live vendor pulls" in str(exc)
        assert "phoenix" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
