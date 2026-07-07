"""Tests for BaseTraceAdapter + the shared normalizer's OpenInference handling.

Proves a provider adapter that only sets `name` + (optionally) `spans_from_payload` gets correct
file/JSONL loading and span->Trace normalization for free — including OpenInference-vocabulary spans
(`openinference.span.kind`, `tool.name`, `output.value`), which providers like Phoenix/Langfuse use.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import JsonValue

from wmh.core.types import ActionKind, JsonObject
from wmh.ingest.adapter import VendorPull
from wmh.ingest.base import BaseTraceAdapter


def _oi_span(
    span_id: str, kind: str, attrs: JsonObject, *, start: int, name: str = ""
) -> JsonObject:
    """An OpenInference-style OTLP span with a FLAT attribute map (provider export shape)."""
    return {
        "traceId": "oitrace0000000000000000000000000",
        "spanId": span_id,
        "name": name,
        "startTimeUnixNano": start,
        "attributes": {"openinference.span.kind": kind, **attrs},
    }


def _otlp(spans: list[JsonObject]) -> JsonObject:
    return {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}


class _DefaultAdapter(BaseTraceAdapter):
    name = "test-default"


def test_base_from_file_normalizes_openinference_spans(tmp_path: Path) -> None:
    # LLM span issues a tool call (OpenInference: tool.name + input.value); TOOL span has output.
    spans = [
        _oi_span(
            "a1",
            "LLM",
            {"tool.name": "get_user", "input.value": '{"id": "u1"}', "input": "look up u1"},
            start=1,
        ),
        _oi_span("t1", "TOOL", {"tool.name": "get_user", "output.value": "found u1"}, start=2),
    ]
    path = tmp_path / "oi.json"
    path.write_text(json.dumps(_otlp(spans)), encoding="utf-8")

    traces = _DefaultAdapter().from_file(str(path))

    assert len(traces) == 1
    assert traces[0].source.startswith("test-default:")
    step = traces[0].steps[0]
    assert step.action.kind == ActionKind.TOOL_CALL
    assert step.action.name == "get_user"
    assert step.action.arguments == {"id": "u1"}
    assert step.observation.content == "found u1"


def test_base_from_file_handles_jsonl_and_skips_corrupt_lines(tmp_path: Path) -> None:
    good = json.dumps(_otlp([_oi_span("a1", "LLM", {"llm.model_name": "gpt"}, start=1)]))
    path = tmp_path / "spans.jsonl"
    path.write_text(f"{good}\n{{truncated\n{good}\n", encoding="utf-8")

    traces = _DefaultAdapter().from_file(str(path))
    # Two valid lines -> two payloads -> same trace id grouped into one trace; corrupt line skipped.
    assert len(traces) == 1


def test_base_vendor_pull_unsupported_is_friendly() -> None:
    try:
        _DefaultAdapter().from_vendor(VendorPull())
    except ValueError as exc:
        assert "does not support live vendor pulls" in str(exc)
        assert "test-default" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_subclass_can_override_spans_from_payload(tmp_path: Path) -> None:
    """A provider whose export is NOT OTLP maps its own shape via spans_from_payload."""
    from wmh.ingest.normalize import SpanRecord

    class CustomAdapter(BaseTraceAdapter):
        name = "custom"

        def spans_from_payload(self, payload: JsonValue) -> list[SpanRecord]:
            # payload is {"events": [{"call": "...", "result": "..."}]}
            spans: list[SpanRecord] = []
            events = payload.get("events", []) if isinstance(payload, dict) else []
            if not isinstance(events, list):
                return spans
            for i, ev in enumerate(events):
                if not isinstance(ev, dict):
                    continue
                spans.append(
                    SpanRecord(
                        trace_id="c" * 32,
                        span_id=f"a{i}",
                        start_nano=i * 2,
                        attributes={
                            "gen_ai.operation.name": "chat",
                            "gen_ai.tool.name": ev["call"],
                            "gen_ai.tool.call.arguments": "{}",
                        },
                    )
                )
                spans.append(
                    SpanRecord(
                        trace_id="c" * 32,
                        span_id=f"t{i}",
                        start_nano=i * 2 + 1,
                        attributes={
                            "gen_ai.operation.name": "execute_tool",
                            "gen_ai.tool.message": ev["result"],
                        },
                    )
                )
            return spans

    path = tmp_path / "custom.json"
    path.write_text(json.dumps({"events": [{"call": "ping", "result": "pong"}]}), encoding="utf-8")

    traces = CustomAdapter().from_file(str(path))
    assert len(traces) == 1
    assert traces[0].steps[0].action.name == "ping"
    assert traces[0].steps[0].observation.content == "pong"


def test_multi_step_trace_split_across_jsonl_lines_keeps_order(tmp_path: Path) -> None:
    """A trace whose spans arrive one-per-JSONL-line must not collide on span_id / start_nano.

    Row-shaped adapters assign per-payload ordinals (0,1,...). With one row per JSONL line, every
    payload restarts at 0, so without globally-unique span ids the action/observation spans would
    share `start_nano=0` and the same span_id, scrambling the pairing (regression guard).
    """
    from wmh.ingest.normalize import SpanRecord

    class RowAdapter(BaseTraceAdapter):
        name = "row"

        def spans_from_payload(self, payload: JsonValue) -> list[SpanRecord]:
            # Each payload (one JSONL line) is a single {kind, tool, content} row -> one span,
            # always at the per-payload ordinal 0 (the collision-prone case).
            if not isinstance(payload, dict):
                return []
            tool = payload.get("tool")
            is_tool = payload.get("kind") == "tool"
            attrs: JsonObject = {
                "gen_ai.operation.name": "execute_tool" if is_tool else "chat",
                "gen_ai.tool.name": tool if isinstance(tool, str) else "",
            }
            if is_tool:
                attrs["gen_ai.tool.message"] = payload.get("content", "")
            else:
                attrs["gen_ai.tool.call.arguments"] = "{}"
            return [SpanRecord(trace_id="t" * 32, span_id="x0", start_nano=0, attributes=attrs)]

    # Two steps (callA->resultA, callB->resultB), four JSONL lines, all one trace.
    lines = [
        {"kind": "llm", "tool": "callA"},
        {"kind": "tool", "tool": "callA", "content": "resultA"},
        {"kind": "llm", "tool": "callB"},
        {"kind": "tool", "tool": "callB", "content": "resultB"},
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    traces = RowAdapter().from_file(str(path))
    assert len(traces) == 1
    steps = traces[0].steps
    assert [(s.action.name, s.observation.content) for s in steps] == [
        ("callA", "resultA"),
        ("callB", "resultB"),
    ]
