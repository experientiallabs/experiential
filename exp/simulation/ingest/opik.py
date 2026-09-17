"""Normalize Opik tracing exports into canonical trace evidence.

Opik exports one record per span with ``trace_id``, ``id``, ``parent_span_id``,
``name``, ``type``, ``input``, ``output``, ``metadata``, ``start_time``,
``end_time``, and any ``error_info``. Exports arrive as a span array, a ``spans``
or ``traces`` envelope, or JSONL with one span per line.

Span types map to canonical evidence by what they observe:

- ``llm`` becomes model calls, including the tool calls their output requests,
- ``tool`` becomes tool results paired with the earlier requesting call,
- ``general``, ``guardrail``, and other orchestration types are not converted.

Model identity comes from the span ``model`` and ``provider`` fields and is
retained only when Opik declares both; usage comes from the flat ``usage``
mapping with ``prompt_tokens`` and ``completion_tokens``.
"""

from __future__ import annotations

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.simulation.ingest.vendor_observations import (
    VendorObservation,
    VendorTokenUsage,
    declared_completion_text,
    declared_error_message,
    declared_model_identity,
    declared_tool_calls,
    declared_usage,
)
from exp.simulation.ingest.vendor_records import (
    VendorTraceFormatError,
    first_text,
    first_user_text,
    json_text,
    json_value,
    required_text,
    source_interval,
)
from exp.simulation.ingest.vendor_source import VendorSource, record_flattener
from exp.simulation.ingest.vendor_trace import approved_extensions

VENDOR = "opik"

_MODEL_TYPES = frozenset({"llm"})
_TOOL_TYPES = frozenset({"tool"})
_MODEL_KEYS = ("model", "model_name", "modelName")
_PROVIDER_KEYS = ("provider", "model_provider", "modelProvider", "llm.system")
_ERROR_KEYS = ("error", "errorInfo", "error_info")


def _span_observation(span: JsonObject, ordinal: int) -> tuple[VendorObservation, ...]:
    """Convert one Opik span to a declared model or tool-result observation.

    Args:
        span: Opik span record.
        ordinal: Source order position for the emitted observation.

    Returns:
        Declared observation, or nothing for orchestration-only span types.

    Raises:
        VendorTraceFormatError: The span lacks a type, identity, timing, or tool
            evidence.
    """
    raw_type = first_text(span, ("type", "spanType"))
    if raw_type is None:
        raise VendorTraceFormatError("Opik span declares no type")
    span_type = raw_type.casefold()
    if span_type not in _MODEL_TYPES | _TOOL_TYPES:
        return ()
    source_trace_id = required_text(span.get("trace_id", span.get("traceId")), "Opik trace_id")
    source_span_id = required_text(
        span.get("id", span.get("span_id", span.get("spanId"))), "Opik span id"
    )
    started_at, ended_at = source_interval(
        span.get("start_time", span.get("startTime")),
        span.get("end_time", span.get("endTime")),
        start_label="Opik span start_time",
        end_label="Opik span end_time",
    )
    attributes = _attributes(span)
    inputs = json_value(span.get("input", span.get("inputs")))
    outputs = json_value(span.get("output", span.get("outputs")))
    extensions = _extensions(span, attributes)
    failure = _failure_message(span)
    parent = first_text(span, ("parent_span_id", "parentSpanId", "parent_id"))
    if span_type in _TOOL_TYPES:
        return (
            VendorObservation(
                source_trace_id=source_trace_id,
                source_span_id=source_span_id,
                ordinal=ordinal,
                started_at=started_at,
                ended_at=ended_at,
                kind="tool_result",
                source_parent_span_id=parent,
                request_text=first_user_text(inputs),
                tool_name=_tool_name(span, attributes),
                tool_arguments=json_text(inputs),
                tool_message=declared_completion_text(outputs),
                tool_call_id=first_text(attributes, ("toolCallId", "tool_call_id")),
                failure_message=failure,
                extensions=extensions,
            ),
        )
    model, declared_model = declared_model_identity(
        {**attributes, **_identity_fields(span)},
        model_keys=_MODEL_KEYS,
        provider_keys=_PROVIDER_KEYS,
    )
    return (
        VendorObservation(
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
            usage=_usage(attributes, span),
            failure_message=failure,
            declared_attributes=(
                {} if declared_model is None else {"gen_ai.request.model": declared_model}
            ),
            extensions=extensions,
        ),
    )


def _attributes(span: JsonObject) -> JsonObject:
    """Return the Opik span metadata object.

    Args:
        span: Opik span record.

    Returns:
        Metadata mapping, empty when the span declares none.
    """
    value = span.get("metadata", span.get("attributes"))
    if isinstance(value, dict):
        return value
    return {}


def _identity_fields(span: JsonObject) -> JsonObject:
    """Return the top-level Opik model and provider declarations.

    Args:
        span: Opik span record.

    Returns:
        Mapping with the span-level model and provider values, when declared.
    """
    fields: JsonObject = {}
    for key in _MODEL_KEYS:
        value = span.get(key)
        if isinstance(value, str) and value.strip():
            fields[key] = value.strip()
    for key in _PROVIDER_KEYS:
        value = span.get(key)
        if isinstance(value, str) and value.strip():
            fields[key] = value.strip()
    return fields


def _input_messages(inputs: JsonValue | None) -> JsonValue | None:
    """Return the declared model input messages for one Opik model span.

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


def _tool_name(span: JsonObject, attributes: JsonObject) -> str:
    """Read the executed tool name from an Opik tool span.

    Args:
        span: Opik span record.
        attributes: Declared span metadata.

    Returns:
        Declared tool name.

    Raises:
        VendorTraceFormatError: The span declares no tool name.
    """
    name = first_text(attributes, ("toolName", "tool_name", "name"))
    if name is not None:
        return name
    return required_text(span.get("name"), "Opik tool span name")


def _failure_message(span: JsonObject) -> str | None:
    """Read an Opik span failure message from its error info, when present.

    Args:
        span: Opik span record.

    Returns:
        Declared error text, or ``None`` when the span declares no error.
    """
    error = span.get("error_info", span.get("errorInfo"))
    if isinstance(error, dict) and error:
        message = first_text(error, ("message", "error", "exception_type"))
        if message is not None:
            return message
        return json_text(error)
    return declared_error_message(span, keys=_ERROR_KEYS, label="Opik span")


def _usage(attributes: JsonObject, span: JsonObject) -> VendorTokenUsage | None:
    """Read declared token accounting from an Opik span or its metadata.

    Args:
        attributes: Declared span metadata.
        span: Opik span record.

    Returns:
        Declared usage, or ``None`` when the span declares no complete accounting.

    Raises:
        VendorTraceFormatError: A declared token count is not a non-negative integer.
    """
    for candidate in (span.get("usage"), attributes.get("usage"), attributes):
        usage = declared_usage(candidate)
        if usage is not None:
            return usage
    return None


def _extensions(span: JsonObject, attributes: JsonObject) -> JsonObject:
    """Read approved EXP extensions and the Opik project from one span.

    Args:
        span: Opik span record.
        attributes: Declared span metadata.

    Returns:
        Approved extension attributes for this record.
    """
    extensions = approved_extensions(span)
    extensions.update(approved_extensions(attributes))
    metadata = span.get("metadata")
    if isinstance(metadata, dict):
        extensions.update(approved_extensions(metadata))
        if "exp.trace.metadata" not in extensions:
            extensions["exp.trace.metadata"] = metadata
    project = first_text(span, ("project_name", "projectName"))
    if project is not None and "exp.opik.project" not in extensions:
        extensions["exp.opik.project"] = project
    return extensions


OPIK_SOURCE: VendorSource[JsonObject] = VendorSource(
    vendor=VENDOR,
    records=record_flattener(
        vendor=VENDOR,
        wrapper_keys=("spans", "traces", "data", "results", "items"),
        record_keys=("trace_id", "traceId"),
    ),
    convert=_span_observation,
)
