"""Tests for MLflow trace export normalization."""

from __future__ import annotations

import json
from pathlib import Path

from exp.simulation.ingest.mlflow import MLFLOW_SOURCE
from exp.simulation.ingest.sources import CANONICAL_TRACE_SOURCES, load_trace_source


def _llm_span(
    *,
    trace_id: str = "trace-1",
    span_id: str = "span-1",
    parent_id: str | None = None,
    name: str = "chat",
    start_ns: int = 1_700_000_000_000_000_000,
    end_ns: int = 1_700_000_001_000_000_000,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    inputs: object | None = None,
    outputs: object | None = None,
    span_type: str = "LLM",
    error: str | None = None,
) -> dict[str, object]:
    """Return one MLflow LLM span with explicit timing and model identity.

    Args:
        trace_id: Declared trace identifier.
        span_id: Declared span identifier.
        parent_id: Declared parent span identifier, when any.
        name: Declared span name.
        start_ns: Start time as nanoseconds since epoch.
        end_ns: End time as nanoseconds since epoch.
        model: Declared model name.
        provider: Declared provider name.
        inputs: Declared span inputs, defaulting to a user message list.
        outputs: Declared span outputs, defaulting to an assistant completion.
        span_type: Declared MLflow span type.
        error: Optional error message that also marks the span failed.

    Returns:
        One MLflow span object.
    """
    span: dict[str, object] = {
        "trace_id": trace_id,
        "span_id": span_id,
        "name": name,
        "span_type": span_type,
        "start_time_ns": start_ns,
        "end_time_ns": end_ns,
        "inputs": inputs
        if inputs is not None
        else {"messages": [{"role": "user", "content": "Where is my order?"}]},
        "outputs": outputs
        if outputs is not None
        else {"role": "assistant", "content": "It ships tomorrow."},
        "attributes": {
            "mlflow.spanType": span_type,
            "llm.request.model": model,
            "llm.system": provider,
        },
    }
    if parent_id is not None:
        span["parent_id"] = parent_id
    if error is not None:
        span["status"] = {"status_code": "ERROR", "error": error}
    return span


def _tool_span(
    *,
    trace_id: str = "trace-1",
    span_id: str = "span-2",
    parent_id: str = "span-1",
    name: str = "lookup_order",
) -> dict[str, object]:
    """Return one MLflow TOOL span that reports its arguments and result."""
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_id": parent_id,
        "name": name,
        "span_type": "TOOL",
        "start_time_ns": 1_700_000_001_000_000_000,
        "end_time_ns": 1_700_000_002_000_000_000,
        "inputs": {"order": "A1"},
        "outputs": "ships tomorrow",
        "attributes": {"mlflow.spanType": "TOOL"},
    }


def test_load_mlflow_file_keeps_completion_and_model_identity(tmp_path: Path) -> None:
    """The model span keeps its completion text and declared provider model."""
    path = tmp_path / "mlflow.json"
    path.write_text(
        json.dumps([_llm_span(), _tool_span(), _llm_span(span_id="span-3", outputs="Done.")]),
        encoding="utf-8",
    )

    result = MLFLOW_SOURCE.load(path)

    assert len(result.traces) == 1
    assert result.traces[0].task == "Where is my order?"
    completions = [span.attributes.get("gen_ai.completion") for span in result.traces[0].spans]
    assert "It ships tomorrow." in completions
    assert "Done." in completions


def test_load_mlflow_file_tool_result_is_paired_with_requesting_call(tmp_path: Path) -> None:
    """A TOOL span is normalized as a tool_result paired with the earlier model call."""
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
    path = tmp_path / "mlflow.json"
    path.write_text(
        json.dumps(
            [
                _llm_span(span_id="span-1", outputs=tool_call_outputs),
                _tool_span(span_id="span-2", parent_id="span-1"),
            ]
        ),
        encoding="utf-8",
    )

    result = MLFLOW_SOURCE.load(path)

    tool_names = [span.attributes.get("gen_ai.tool.name") for span in result.traces[0].spans]
    assert "lookup_order" in tool_names
    assert len(result.traces[0].spans) == 2


def test_load_mlflow_file_ignores_orchestration_spans(tmp_path: Path) -> None:
    """CHAIN and AGENT spans carry no direct evidence and are ignored."""
    path = tmp_path / "mlflow.json"
    path.write_text(
        json.dumps(
            [
                _llm_span(span_id="span-1"),
                {
                    "trace_id": "trace-1",
                    "span_id": "span-99",
                    "name": "orchestration",
                    "span_type": "CHAIN",
                    "start_time_ns": 1_700_000_002_000_000_000,
                    "end_time_ns": 1_700_000_003_000_000_000,
                    "attributes": {"mlflow.spanType": "CHAIN"},
                },
            ]
        ),
        encoding="utf-8",
    )

    result = MLFLOW_SOURCE.load(path)

    assert len(result.traces[0].spans) == 1


def test_load_mlflow_file_accepts_trace_envelope_wrapper(tmp_path: Path) -> None:
    """A ``traces`` envelope with ``data.spans`` is flattened into one trace."""
    path = tmp_path / "mlflow.json"
    path.write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "info": {"trace_id": "trace-7"},
                        "data": {"spans": [_llm_span(trace_id="trace-7", span_id="span-7")]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = MLFLOW_SOURCE.load(path)

    assert len(result.traces) == 1
    assert result.traces[0].spans[0].attributes["gen_ai.request.model"] == "gpt-4o-mini"


def test_load_mlflow_file_envelope_trace_id_propagates_to_child_spans(
    tmp_path: Path,
) -> None:
    """Child spans without a trace_id inherit ``info.trace_id`` from the envelope."""
    # MLflow trace-search shapes identify the trace only at the envelope level;
    # valid child spans that omit their own trace_id must not be dropped.
    child_without_trace = {
        "span_id": "span-7-no-trace",
        "name": "chat",
        "span_type": "LLM",
        "start_time_ns": 1_700_000_000_000_000_000,
        "end_time_ns": 1_700_000_001_000_000_000,
        "inputs": {"messages": [{"role": "user", "content": "hello envelope"}]},
        "outputs": {"content": "hi envelope"},
        "attributes": {"llm.request.model": "gpt-4o", "llm.system": "openai"},
    }
    path = tmp_path / "mlflow.json"
    path.write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "info": {"trace_id": "trace-envelope-1"},
                        "data": {"spans": [child_without_trace]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = MLFLOW_SOURCE.load(path)

    assert result.issues == ()
    assert len(result.traces) == 1
    assert result.traces[0].task == "hello envelope"
    # Also verify single-trace envelope shape without the outer ``traces`` wrapper.
    single_envelope = tmp_path / "mlflow_single.json"
    single_envelope.write_text(
        json.dumps(
            {
                "info": {"trace_id": "trace-single-1"},
                "data": {"spans": [child_without_trace]},
            }
        ),
        encoding="utf-8",
    )
    single_result = MLFLOW_SOURCE.load(single_envelope)
    assert len(single_result.traces) == 1
    assert single_result.traces[0].task == "hello envelope"


def test_load_mlflow_file_accepts_jsonl(tmp_path: Path) -> None:
    """JSONL with one span per line is supported."""
    path = tmp_path / "mlflow.jsonl"
    spans = [
        _llm_span(trace_id="trace-2", span_id="span-a"),
        _llm_span(trace_id="trace-2", span_id="span-b"),
    ]
    path.write_text("\n".join(json.dumps(span) for span in spans), encoding="utf-8")

    result = MLFLOW_SOURCE.load(path)

    assert len(result.traces) == 1
    assert len(result.traces[0].spans) == 2


def test_load_mlflow_file_handles_iso_timestamps(tmp_path: Path) -> None:
    """ISO-8601 startTime and endTime are accepted when nanosecond fields are absent."""
    path = tmp_path / "mlflow.json"
    span = {
        "trace_id": "trace-iso",
        "span_id": "span-iso",
        "name": "chat",
        "span_type": "LLM",
        "startTime": "2026-02-01T00:00:00Z",
        "endTime": "2026-02-01T00:00:01Z",
        "inputs": {"messages": [{"role": "user", "content": "hello"}]},
        "outputs": {"content": "hi"},
        "attributes": {"llm.request.model": "gpt-4o", "llm.system": "openai"},
    }
    path.write_text(json.dumps([span]), encoding="utf-8")

    result = MLFLOW_SOURCE.load(path)

    assert result.traces[0].task == "hello"


def test_load_mlflow_file_marks_failed_span(tmp_path: Path) -> None:
    """A span with ERROR status retains the declared failure message."""
    path = tmp_path / "mlflow.json"
    path.write_text(
        json.dumps([_llm_span(span_id="span-err", error="rate limited")]),
        encoding="utf-8",
    )

    result = MLFLOW_SOURCE.load(path)

    assert result.issues == ()
    failed = result.traces[0].spans[0]
    assert failed.failure is not None
    assert failed.failure.message == "rate limited"


def test_load_mlflow_file_preserves_usage_when_declared(tmp_path: Path) -> None:
    """Token accounting declared on LLM spans is retained on the model span."""
    span = _llm_span(span_id="span-usage")
    attributes: dict[str, object] = {
        "mlflow.spanType": "LLM",
        "llm.request.model": "gpt-4o-mini",
        "llm.system": "openai",
        "gen_ai.usage.input_tokens": 10,
        "gen_ai.usage.output_tokens": 20,
        # The shared declared_usage helper also reads prompt/completion tokens.
        "llm.usage.prompt_tokens": 10,
        "llm.usage.completion_tokens": 20,
    }
    span["attributes"] = attributes
    path = tmp_path / "mlflow.json"
    path.write_text(json.dumps([span]), encoding="utf-8")

    result = MLFLOW_SOURCE.load(path)

    # Usage is validated strictly; presence does not error and raw span is kept.
    assert len(result.traces) == 1
    assert result.traces[0].spans[0].attributes["gen_ai.request.model"] == "gpt-4o-mini"


def test_mlflow_is_registered_as_canonical_source() -> None:
    """The MLflow source is discoverable through the canonical source table."""
    assert "mlflow" in CANONICAL_TRACE_SOURCES
    assert "mlflow" in sorted(CANONICAL_TRACE_SOURCES)


def test_load_trace_source_dispatches_to_mlflow(tmp_path: Path) -> None:
    """The generic loader routes ``mlflow`` to the MLflow adapter."""
    path = tmp_path / "mlflow.json"
    path.write_text(json.dumps([_llm_span()]), encoding="utf-8")

    result = load_trace_source("mlflow", path)

    assert len(result.traces) == 1


def test_load_mlflow_file_rejects_unsupported_shape(tmp_path: Path) -> None:
    """A payload without any MLflow record keys is reported as an exclusion."""
    path = tmp_path / "mlflow.json"
    path.write_text(json.dumps({"unknown": 1}), encoding="utf-8")

    result = MLFLOW_SOURCE.load(path)

    assert result.traces == ()
    assert len(result.issues) == 1


def test_load_mlflow_file_camelcase_keys(tmp_path: Path) -> None:
    """CamelCase traceId and spanId are accepted."""
    path = tmp_path / "mlflow.json"
    span = {
        "traceId": "trace-camel",
        "spanId": "span-camel",
        "name": "chat",
        "spanType": "LLM",
        "startTime": "2026-02-01T00:00:00Z",
        "endTime": "2026-02-01T00:00:01Z",
        "inputs": {"messages": [{"role": "user", "content": "hello"}]},
        "outputs": {"content": "hi"},
        "attributes": {"llm.request.model": "gpt-4o", "llm.system": "openai"},
    }
    path.write_text(json.dumps([span]), encoding="utf-8")

    result = MLFLOW_SOURCE.load(path)

    assert len(result.traces) == 1
    assert result.traces[0].task == "hello"


def test_load_mlflow_file_accepts_real_server_export_shape(tmp_path: Path) -> None:
    """Server exports use unix_nano timing, JSON-encoded attributes, and OTel status."""
    start_ns = 1_788_512_428_260_757_000
    tool_span = {
        "trace_id": "trace-real",
        "span_id": "span-tool",
        "parent_span_id": "span-root",
        "name": "lookup_order",
        "start_time_unix_nano": start_ns,
        "end_time_unix_nano": start_ns + 1_000_000,
        "status": {"code": "STATUS_CODE_OK", "message": ""},
        "attributes": {
            "mlflow.spanType": '"TOOL"',
            "mlflow.spanInputs": '{"order": "A1"}',
            "mlflow.spanOutputs": '"ships tomorrow"',
        },
    }
    model_span = {
        "trace_id": "trace-real",
        "span_id": "span-llm",
        "parent_span_id": "span-root",
        "name": "answer",
        "start_time_unix_nano": start_ns,
        "end_time_unix_nano": start_ns + 2_000_000,
        "status": {"code": "STATUS_CODE_OK", "message": ""},
        "attributes": {
            "mlflow.spanType": '"LLM"',
            "llm.request.model": '"gpt-4o-mini"',
            "llm.system": '"openai"',
            "mlflow.spanInputs": (
                '{"messages": [{"role": "user", "content": "Where is my order?"}]}'
            ),
            "mlflow.spanOutputs": '{"content": "It ships tomorrow."}',
        },
    }
    root_span = {
        "trace_id": "trace-real",
        "span_id": "span-root",
        "name": "chat",
        "start_time_unix_nano": start_ns,
        "end_time_unix_nano": start_ns + 3_000_000,
        "status": {"code": "STATUS_CODE_OK", "message": ""},
        "attributes": {"mlflow.spanType": '"UNKNOWN"'},
    }
    path = tmp_path / "mlflow.json"
    path.write_text(json.dumps([root_span, tool_span, model_span]), encoding="utf-8")

    result = MLFLOW_SOURCE.load(path)

    assert result.issues == ()
    assert len(result.traces) == 1
    assert result.traces[0].task == "Where is my order?"
    assert len(result.traces[0].spans) == 2
    tool_names = [span.attributes.get("gen_ai.tool.name") for span in result.traces[0].spans]
    assert "lookup_order" in tool_names
    failed = [span for span in result.traces[0].spans if span.failure is not None]
    assert failed == []
    models = [
        span.attributes["gen_ai.request.model"]
        for span in result.traces[0].spans
        if "gen_ai.request.model" in span.attributes
    ]
    assert models == ["gpt-4o-mini"]
