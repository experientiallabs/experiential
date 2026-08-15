"""Normalize Mastra AI tracing exports into canonical trace evidence.

Mastra exports one record per span with ``traceId``, ``id``, ``parentSpanId``, ``name``, ``type``,
``input``, ``output``, ``attributes``, ``startTime``, ``endTime``, and any ``errorInfo``. Exports
arrive as a span array, a ``spans`` or ``traces`` envelope, or JSONL with one span per line.

Span types map to canonical evidence by what they observe:

- ``model_generation`` and ``llm_generation`` become model calls, including the tool calls their
  output requests in either Vercel AI SDK or OpenAI shape,
- ``tool_call`` and ``mcp_tool_call`` become tool results paired with the earlier requesting call,
- ``agent_run``, ``workflow_run``, ``workflow_step``, ``model_step``, ``model_chunk``, and
  ``generic`` observe orchestration rather than a visible agent step and are not converted.

Model identity comes from the span attributes and is retained only when Mastra declares both a
provider and a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject, SourceIdentity
from wmo.simulation.ingest.otlp import (
    GENAI_SEMANTIC_CONVENTION_VERSION,
    TraceNormalizationIssue,
    TraceNormalizationResult,
)
from wmo.simulation.ingest.vendor_observations import (
    VendorObservation,
    VendorTokenUsage,
    declared_completion_text,
    declared_error_message,
    declared_model_identity,
    declared_tool_calls,
    declared_usage,
)
from wmo.simulation.ingest.vendor_records import (
    VendorTraceFormatError,
    first_text,
    first_user_text,
    flatten_records,
    json_text,
    json_value,
    read_vendor_export,
    required_text,
    source_timestamp,
)
from wmo.simulation.ingest.vendor_trace import approved_extensions, build_vendor_traces

VENDOR = "mastra"

_MODEL_TYPES = frozenset({"model_generation", "llm_generation"})
_TOOL_TYPES = frozenset({"tool_call", "mcp_tool_call"})
_MODEL_KEYS = ("model", "modelId", "model_id", "modelName")
_PROVIDER_KEYS = ("provider", "providerId", "provider_id", "modelProvider")
_ERROR_KEYS = ("errorInfo", "error")


def load_mastra_file(
    path: Path,
    *,
    semantic_convention_version: str = GENAI_SEMANTIC_CONVENTION_VERSION,
    source_id: str | None = None,
) -> TraceNormalizationResult:
    """Read a Mastra span export into canonical trace evidence.

    Args:
        path: Mastra span array, envelope, or JSONL export.
        semantic_convention_version: Pinned GenAI semantic-convention version for the traces.
        source_id: Optional durable source label. The local path is used when omitted.

    Returns:
        Canonical traces and every retained parse or validation exclusion.

    Raises:
        VendorTraceFormatError: The export cannot be read or decoded.
    """
    export = read_vendor_export(path, vendor=VENDOR, source_id=source_id)
    return normalize_mastra_payloads(
        export.payloads,
        source=export.source,
        semantic_convention_version=semantic_convention_version,
        initial_issues=export.issues,
    )


def normalize_mastra_payloads(
    payloads: Sequence[JsonValue],
    *,
    source: SourceIdentity,
    semantic_convention_version: str = GENAI_SEMANTIC_CONVENTION_VERSION,
    initial_issues: Sequence[TraceNormalizationIssue] = (),
) -> TraceNormalizationResult:
    """Normalize decoded Mastra spans into canonical traces.

    Args:
        payloads: Decoded Mastra documents in source order.
        source: Immutable identity of the source bytes or transport result.
        semantic_convention_version: Pinned GenAI semantic-convention version for the traces.
        initial_issues: Parse exclusions collected before span mapping.

    Returns:
        Canonical traces and every retained validation exclusion.
    """
    issues = list(initial_issues)
    observations: list[VendorObservation] = []
    ordinal = 0
    for index, payload in enumerate(payloads, start=1):
        try:
            spans = flatten_records(
                payload,
                vendor=VENDOR,
                wrapper_keys=("spans", "traces", "data", "results", "items"),
                record_keys=("traceId", "trace_id"),
            )
        except VendorTraceFormatError as exc:
            issues.append(TraceNormalizationIssue(f"record-{index}", str(exc)))
            continue
        for span in spans:
            try:
                converted = _span_observation(span, ordinal)
            except VendorTraceFormatError as exc:
                issues.append(TraceNormalizationIssue(f"record-{index}", str(exc)))
                continue
            if converted is None:
                continue
            observations.append(converted)
            ordinal += 1
    return build_vendor_traces(
        observations,
        vendor=VENDOR,
        source=source,
        semantic_convention_version=semantic_convention_version,
        initial_issues=issues,
    )


def _span_observation(span: JsonObject, ordinal: int) -> VendorObservation | None:
    """Convert one Mastra span to a declared model or tool-result observation.

    Args:
        span: Mastra span record.
        ordinal: Source order position for the emitted observation.

    Returns:
        Declared observation, or ``None`` for orchestration-only span types.

    Raises:
        VendorTraceFormatError: The span lacks identity, timing, or tool evidence.
    """
    span_type = (first_text(span, ("type", "spanType")) or "").casefold()
    if span_type not in _MODEL_TYPES | _TOOL_TYPES:
        return None
    source_trace_id = required_text(span.get("traceId", span.get("trace_id")), "Mastra traceId")
    source_span_id = required_text(span.get("id", span.get("spanId")), "Mastra span id")
    started_at, ended_at = _interval(span)
    attributes = span.get("attributes")
    attribute_object: JsonObject = attributes if isinstance(attributes, dict) else {}
    inputs = json_value(span.get("input"))
    outputs = json_value(span.get("output"))
    extensions = _extensions(span, attribute_object)
    failure = declared_error_message(span, keys=_ERROR_KEYS, label="Mastra span")
    parent = first_text(span, ("parentSpanId", "parent_span_id"))
    if span_type in _TOOL_TYPES:
        return VendorObservation(
            source_trace_id=source_trace_id,
            source_span_id=source_span_id,
            ordinal=ordinal,
            started_at=started_at,
            ended_at=ended_at,
            kind="tool_result",
            source_parent_span_id=parent,
            request_text=first_user_text(inputs),
            tool_name=_tool_name(span, attribute_object),
            tool_arguments=json_text(_tool_arguments(inputs)),
            tool_message=declared_completion_text(outputs),
            tool_call_id=first_text(attribute_object, ("toolCallId", "tool_call_id")),
            failure_message=failure,
            extensions=extensions,
        )
    model, declared_model = declared_model_identity(
        attribute_object,
        model_keys=_MODEL_KEYS,
        provider_keys=_PROVIDER_KEYS,
    )
    return VendorObservation(
        source_trace_id=source_trace_id,
        source_span_id=source_span_id,
        ordinal=ordinal,
        started_at=started_at,
        ended_at=ended_at,
        kind="model",
        source_parent_span_id=parent,
        request_text=first_user_text(inputs),
        input_messages=_input_messages(inputs),
        completion_text=declared_completion_text(outputs) or None,
        tool_calls=declared_tool_calls(outputs),
        model=model,
        usage=_usage(attribute_object, span),
        failure_message=failure,
        declared_attributes=(
            {} if declared_model is None else {"gen_ai.request.model": declared_model}
        ),
        extensions=extensions,
    )


def _interval(span: JsonObject) -> tuple[datetime, datetime]:
    """Read the declared source interval of one Mastra span.

    Args:
        span: Mastra span record.

    Returns:
        Source start and end instants, equal when the span declares no end.

    Raises:
        VendorTraceFormatError: The span declares no readable start time.
    """
    started_at = source_timestamp(
        span.get("startTime", span.get("start_time")), "Mastra span startTime"
    )
    end_value = span.get("endTime", span.get("end_time"))
    ended_at = (
        source_timestamp(end_value, "Mastra span endTime") if end_value is not None else started_at
    )
    return started_at, ended_at


def _tool_name(span: JsonObject, attributes: JsonObject) -> str:
    """Read the executed tool name from Mastra span attributes or the span name.

    Args:
        span: Mastra span record.
        attributes: Declared span attributes.

    Returns:
        Declared tool name.

    Raises:
        VendorTraceFormatError: The span declares no tool name.
    """
    name = first_text(attributes, ("toolName", "tool_name", "toolId", "name"))
    if name is not None:
        return name
    return required_text(span.get("name"), "Mastra tool span name")


def _tool_arguments(inputs: JsonValue | None) -> JsonValue | None:
    """Read declared tool arguments from a Mastra tool span input.

    Args:
        inputs: Decoded span input.

    Returns:
        Declared arguments, unwrapping the AI SDK ``input`` and v4 ``args`` fields.
    """
    if isinstance(inputs, dict):
        for key in ("input", "args", "arguments", "parameters"):
            if key in inputs:
                return inputs[key]
    return inputs


def _input_messages(inputs: JsonValue | None) -> JsonValue | None:
    """Return the declared model input messages for one Mastra model span.

    Args:
        inputs: Decoded span input.

    Returns:
        The declared message list, or the raw input when the span declares no list.
    """
    if isinstance(inputs, dict):
        for key in ("messages", "prompt", "input"):
            value = inputs.get(key)
            if isinstance(value, list):
                return value
    return inputs


def _usage(attributes: JsonObject, span: JsonObject) -> VendorTokenUsage | None:
    """Read declared token accounting from Mastra span attributes or the span itself.

    Args:
        attributes: Declared span attributes.
        span: Mastra span record.

    Returns:
        Declared usage, or ``None`` when the span declares no complete accounting.

    Raises:
        VendorTraceFormatError: A declared token count is not a non-negative integer.
    """
    for candidate in (attributes.get("usage"), span.get("usage"), attributes):
        usage = declared_usage(candidate)
        if usage is not None:
            return usage
    return None


def _extensions(span: JsonObject, attributes: JsonObject) -> JsonObject:
    """Read approved WMO extensions, thread identity, and metadata from one Mastra span.

    Args:
        span: Mastra span record.
        attributes: Declared span attributes.

    Returns:
        Approved extension attributes for the spans of this record.
    """
    extensions = approved_extensions(span)
    extensions.update(approved_extensions(attributes))
    metadata = span.get("metadata")
    if isinstance(metadata, dict):
        extensions.update(approved_extensions(metadata))
        if "wmo.trace.metadata" not in extensions:
            extensions["wmo.trace.metadata"] = metadata
    thread_id = first_text(attributes, ("threadId", "thread_id")) or first_text(
        span, ("threadId", "thread_id")
    )
    if thread_id is not None and "wmo.conversation.id" not in extensions:
        extensions["wmo.conversation.id"] = thread_id
    return extensions
