"""Tests for Opik trace export normalization."""

from __future__ import annotations

import json
from pathlib import Path

from exp.simulation.ingest.opik import OPIK_SOURCE
from exp.simulation.ingest.sources import CANONICAL_TRACE_SOURCES, load_trace_source


def _llm_span(
    *,
    trace_id: str = "trace-1",
    span_id: str = "span-1",
    parent_id: str | None = None,
    name: str = "answer",
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    inputs: object | None = None,
    outputs: object | None = None,
) -> dict[str, object]:
    """Return one Opik LLM span with explicit timing and model identity.

    Args:
        trace_id: Declared trace identifier.
        span_id: Declared span identifier.
        parent_id: Declared parent span identifier, when any.
        name: Declared span name.
        model: Declared model name.
        provider: Declared provider name.
        inputs: Declared span input, defaulting to a user message list.
        outputs: Declared span output, defaulting to an assistant completion.

    Returns:
        One Opik span object.
    """
    span: dict[str, object] = {
        "trace_id": trace_id,
        "id": span_id,
        "name": name,
        "type": "llm",
        "start_time": "2026-09-04T12:00:00+00:00",
        "end_time": "2026-09-04T12:00:01+00:00",
        "input": inputs
        if inputs is not None
        else {"messages": [{"role": "user", "content": "Where is my order?"}]},
        "output": outputs
        if outputs is not None
        else {"role": "assistant", "content": "It ships tomorrow."},
        "model": model,
        "provider": provider,
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }
    if parent_id is not None:
        span["parent_span_id"] = parent_id
    return span


def _tool_span(
    *,
    trace_id: str = "trace-1",
    span_id: str = "span-2",
    parent_id: str = "span-1",
    name: str = "lookup_order",
) -> dict[str, object]:
    """Return one Opik TOOL span that reports its arguments and result."""
    return {
        "trace_id": trace_id,
        "id": span_id,
        "parent_span_id": parent_id,
        "name": name,
        "type": "tool",
        "start_time": "2026-09-04T12:00:01+00:00",
        "end_time": "2026-09-04T12:00:02+00:00",
        "input": {"order": "A1"},
        "output": "ships tomorrow",
    }


def test_load_opik_file_keeps_completion_and_model_identity(tmp_path: Path) -> None:
    """The model span keeps its completion text and declared provider model."""
    path = tmp_path / "opik.json"
    path.write_text(
        json.dumps([_llm_span(), _tool_span(), _llm_span(span_id="span-3", outputs="Done.")]),
        encoding="utf-8",
    )

    result = OPIK_SOURCE.load(path)

    assert result.issues == ()
    assert len(result.traces) == 1
    assert result.traces[0].task == "Where is my order?"
    completions = [span.attributes.get("gen_ai.completion") for span in result.traces[0].spans]
    assert "It ships tomorrow." in completions
    assert "Done." in completions
    assert result.traces[0].spans[0].attributes["gen_ai.request.model"] == "gpt-4o-mini"


def test_load_opik_file_preserves_usage_when_declared(tmp_path: Path) -> None:
    """Token accounting declared on LLM spans is retained on the model span."""
    path = tmp_path / "opik.json"
    path.write_text(json.dumps([_llm_span()]), encoding="utf-8")

    result = OPIK_SOURCE.load(path)

    usage = result.traces[0].spans[0].usage
    assert usage is not None
    assert usage.input_tokens == 12
    assert usage.output_tokens == 8


def test_load_opik_file_tool_result_is_paired_with_requesting_call(tmp_path: Path) -> None:
    """A tool span is normalized as a tool_result paired with the earlier model call."""
    tool_call_outputs = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "function": {"name": "lookup_order", "arguments": '{"order": "A1"}'},
            }
        ],
    }
    path = tmp_path / "opik.json"
    path.write_text(
        json.dumps(
            [
                _llm_span(span_id="span-1", outputs=tool_call_outputs),
                _tool_span(span_id="span-2", parent_id="span-1"),
            ]
        ),
        encoding="utf-8",
    )

    result = OPIK_SOURCE.load(path)

    tool_names = [span.attributes.get("gen_ai.tool.name") for span in result.traces[0].spans]
    assert "lookup_order" in tool_names
    assert len(result.traces[0].spans) == 2


def test_load_opik_file_ignores_orchestration_spans(tmp_path: Path) -> None:
    """General and guardrail spans carry no direct evidence and are ignored."""
    general_span = {
        "trace_id": "trace-1",
        "id": "span-99",
        "name": "chat",
        "type": "general",
        "start_time": "2026-09-04T12:00:02+00:00",
        "end_time": "2026-09-04T12:00:03+00:00",
        "input": {"question": "Where is my order?"},
        "output": "It ships tomorrow.",
    }
    guardrail_span = dict(general_span, id="span-98", type="guardrail")
    path = tmp_path / "opik.json"
    path.write_text(
        json.dumps([_llm_span(span_id="span-1"), general_span, guardrail_span]),
        encoding="utf-8",
    )

    result = OPIK_SOURCE.load(path)

    assert len(result.traces[0].spans) == 1


def test_load_opik_file_accepts_spans_envelope_wrapper(tmp_path: Path) -> None:
    """A ``spans`` envelope is flattened into one trace."""
    path = tmp_path / "opik.json"
    path.write_text(json.dumps({"spans": [_llm_span(span_id="span-7")]}), encoding="utf-8")

    result = OPIK_SOURCE.load(path)

    assert len(result.traces) == 1
    assert result.traces[0].spans[0].attributes["gen_ai.request.model"] == "gpt-4o-mini"


def test_load_opik_file_accepts_jsonl(tmp_path: Path) -> None:
    """JSONL with one span per line is supported."""
    path = tmp_path / "opik.jsonl"
    spans = [
        _llm_span(trace_id="trace-2", span_id="span-a"),
        _tool_span(trace_id="trace-2", span_id="span-b", parent_id="span-a"),
    ]
    path.write_text("\n".join(json.dumps(span) for span in spans), encoding="utf-8")

    result = OPIK_SOURCE.load(path)

    assert len(result.traces) == 1
    assert len(result.traces[0].spans) == 2


def test_load_opik_file_marks_failed_span(tmp_path: Path) -> None:
    """A span with error_info retains the declared failure message."""
    span = _tool_span(span_id="span-err")
    assert isinstance(span, dict)
    span["error_info"] = {"exception_type": "ValueError", "message": "not refundable"}
    path = tmp_path / "opik.json"
    path.write_text(json.dumps([_llm_span(span_id="span-1"), span]), encoding="utf-8")

    result = OPIK_SOURCE.load(path)

    assert result.issues == ()
    failed = [span for span in result.traces[0].spans if span.failure is not None]
    assert len(failed) == 1
    assert failed[0].failure is not None
    assert failed[0].failure.message == "not refundable"


def test_load_opik_file_accepts_sdk_serialized_shapes(tmp_path: Path) -> None:
    """Payloads serialized by the Opik SDK keep identity, usage, and failures."""
    llm_span = {
        "trace_id": "0196a3b4-9c8f-7e2d-a1b0-c5d4e3f2a190",
        "id": "span-llm",
        "parent_span_id": "span-root",
        "name": "answer",
        "type": "llm",
        "start_time": "2026-09-04 12:00:00+00:00",
        "end_time": "2026-09-04 12:00:01+00:00",
        "input": {"messages": [{"role": "user", "content": "Where is my order?"}]},
        "output": {"role": "assistant", "content": "It ships tomorrow."},
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        "model": "gpt-4o-mini",
        "provider": "openai",
        "project_name": "experiential-check",
    }
    path = tmp_path / "opik.json"
    path.write_text(json.dumps([llm_span]), encoding="utf-8")

    result = OPIK_SOURCE.load(path)

    assert result.issues == ()
    assert result.traces[0].task == "Where is my order?"
    usage = result.traces[0].spans[0].usage
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens) == (12, 8)


def test_load_opik_file_reports_typeless_record_as_exclusion(tmp_path: Path) -> None:
    """A record with identity but no declared type is reported, not silently dropped."""
    path = tmp_path / "opik.json"
    path.write_text(
        json.dumps(
            [
                {
                    "trace_id": "trace-1",
                    "id": "span-typeless",
                    "name": "chat",
                    "start_time": "2026-09-04T12:00:00+00:00",
                    "end_time": "2026-09-04T12:00:01+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = OPIK_SOURCE.load(path)

    assert result.traces == ()
    assert len(result.issues) == 1


def test_opik_is_registered_as_canonical_source() -> None:
    """The Opik source is discoverable through the canonical source table."""
    assert "opik" in CANONICAL_TRACE_SOURCES
    assert "opik" in sorted(CANONICAL_TRACE_SOURCES)


def test_load_trace_source_dispatches_to_opik(tmp_path: Path) -> None:
    """The generic loader routes ``opik`` to the Opik adapter."""
    path = tmp_path / "opik.json"
    path.write_text(json.dumps([_llm_span()]), encoding="utf-8")

    result = load_trace_source("opik", path)

    assert len(result.traces) == 1


def test_load_opik_file_rejects_unsupported_shape(tmp_path: Path) -> None:
    """A payload without any Opik record keys is reported as an exclusion."""
    path = tmp_path / "opik.json"
    path.write_text(json.dumps({"unknown": 1}), encoding="utf-8")

    result = OPIK_SOURCE.load(path)

    assert result.traces == ()
    assert len(result.issues) == 1
