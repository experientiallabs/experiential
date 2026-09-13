"""Normalize MLflow tracing exports into canonical trace evidence.

MLflow exports agent runs as spans grouped by a trace identifier. A span declares
``trace_id`` (or ``traceId``), ``span_id`` (or ``spanId``), ``parent_id``, ``name``,
``span_type`` (or ``type``), ``start_time_unix_nano``/``start_time_ns``/``start_time``
and ``end_time``, plus ``inputs``, ``outputs``, ``attributes``, and ``status``.
Exports arrive as a span array, a ``traces`` or ``spans`` envelope, or JSONL with
one span per line. A trace object that carries ``info.trace_id`` and ``data.spans``
is also supported because ``GET /mlflow/traces/search`` returns that shape.

Span types map to canonical evidence by what they observe:

- ``LLM`` and ``CHAT_MODEL`` become model calls, including the tool calls their
  output requests,
- ``TOOL`` and ``FUNCTION`` become tool results paired with the earlier
  requesting call,
- ``CHAIN``, ``AGENT``, ``RETRIEVER``, ``EMBEDDING``, ``UNKNOWN``, and other
  orchestration types are not converted.

MLflow names a model in span attributes under ``mlflow.model``, ``llm.request.model``,
or ``gen_ai.request.model``. Provider identity comes from ``llm.system`` or
``gen_ai.system``. The declared model name is retained as ``gen_ai.request.model``
evidence, and resolved model identity is retained only when the export also declares
a provider.
"""

from __future__ import annotations

from datetime import datetime

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
    source_timestamp,
)
from exp.simulation.ingest.vendor_source import VendorSource, record_flattener
from exp.simulation.ingest.vendor_trace import approved_extensions

VENDOR = "mlflow"

_MODEL_TYPES = frozenset({"llm", "chat_model", "chat", "completion"})
_TOOL_TYPES = frozenset({"tool", "function"})
_MODEL_KEYS = (
    "mlflow.model",
    "llm.request.model",
    "gen_ai.request.model",
    "model",
    "model_name",
    "modelName",
)
_PROVIDER_KEYS = (
    "llm.system",
    "gen_ai.system",
    "provider",
    "model_provider",
    "modelProvider",
)
_ERROR_KEYS = ("error", "exception", "statusMessage", "error_message")


def _span_observation(span: JsonObject, ordinal: int) -> tuple[VendorObservation, ...]:
    """Convert one MLflow span to a declared model or tool-result observation.

    Args:
        span: MLflow span record, which may be a flat span or a trace envelope
            carrying ``info`` and ``data.spans``.

    Returns:
        Declared observation, or nothing for orchestration-only span types or
        for envelope records that are expanded elsewhere.

    Raises:
        VendorTraceFormatError: The span lacks identity, timing, or tool evidence.
    """
    # Trace envelope: ``{"info": {"trace_id": ...}, "data": {"spans": [...]}}``.
    # These are not vendor records themselves when handled via flattening, but a
    # bare envelope that was identified as a record needs explicit expansion.
    if "data" in span and isinstance(span["data"], dict):
        data = span["data"]
        spans = data.get("spans")
        if isinstance(spans, list):
            # This record is actually a trace envelope; it should have been
            # flattened via wrapper_keys. If it arrives here, treat it as no-op
            # rather than misclassifying the envelope as a span.
            return ()
    if "info" in span and "data" not in span:
        # Standalone trace info without data is not a span.
        return ()

    span_type = _span_type(span)
    if span_type not in _MODEL_TYPES | _TOOL_TYPES:
        return ()

    source_trace_id = _trace_id(span)
    source_span_id = _span_id(span)
    started_at, ended_at = _interval(span)
    attributes = _attributes(span)
    inputs = _inputs(span, attributes)
    outputs = _outputs(span, attributes)
    extensions = _extensions(span, attributes)
    failure = _failure_message(span, attributes)
    parent = _parent_id(span)

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
        attributes,
        model_keys=_MODEL_KEYS,
        provider_keys=_PROVIDER_KEYS,
    )
    # Fallback: some MLflow spans carry model directly on the span rather than attributes.
    if model is None and declared_model is None:
        fallback_attrs: JsonObject = {}
        for key in _MODEL_KEYS:
            if key in span:
                fallback_attrs[key] = span[key]
        for key in _PROVIDER_KEYS:
            if key in span:
                fallback_attrs[key] = span[key]
        if fallback_attrs:
            model, declared_model = declared_model_identity(
                fallback_attrs,
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


def _span_type(span: JsonObject) -> str:
    """Return the lowercase declared MLflow span type.

    Args:
        span: MLflow span record.

    Returns:
        Lowercase span type, empty when the span declares none.
    """
    attributes = span.get("attributes")
    attr_obj: JsonObject = attributes if isinstance(attributes, dict) else {}
    for key in ("mlflow.spanType", "span_type", "spanType", "type"):
        value = json_value(span.get(key))
        if isinstance(value, str) and value.strip():
            return value.strip().casefold()
        attr_value = json_value(attr_obj.get(key))
        if isinstance(attr_value, str) and attr_value.strip():
            return attr_value.strip().casefold()
    # Also check nested attributes like attributes.spanType
    for key in ("mlflow.spanType", "span_type"):
        attr_value = json_value(attr_obj.get(key))
        if isinstance(attr_value, str) and attr_value.strip():
            return attr_value.strip().casefold()
    return ""


def _trace_id(span: JsonObject) -> str:
    """Read the MLflow trace identifier.

    Args:
        span: MLflow span record.

    Returns:
        Declared trace identifier.

    Raises:
        VendorTraceFormatError: The span declares no trace identifier.
    """
    for key in ("trace_id", "traceId", "traceID"):
        value = span.get(key)
        if isinstance(value, str) and value.strip():
            return required_text(value, "MLflow trace_id")
    # Check info wrapper
    info = span.get("info")
    if isinstance(info, dict):
        for key in ("trace_id", "traceId", "request_id"):
            value = info.get(key)
            if isinstance(value, str) and value.strip():
                return required_text(value, "MLflow trace_id")
    attributes = span.get("attributes")
    if isinstance(attributes, dict):
        for key in ("mlflow.traceId", "trace_id"):
            value = attributes.get(key)
            if isinstance(value, str) and value.strip():
                return required_text(value, "MLflow trace_id")
    return required_text(None, "MLflow trace_id")


def _span_id(span: JsonObject) -> str:
    """Read the MLflow span identifier.

    Args:
        span: MLflow span record.

    Returns:
        Declared span identifier.
    """
    for key in ("span_id", "spanId", "spanID", "id", "request_id"):
        value = span.get(key)
        if isinstance(value, str) and value.strip():
            return required_text(value, "MLflow span_id")
    info = span.get("info")
    if isinstance(info, dict):
        for key in ("span_id", "request_id"):
            value = info.get(key)
            if isinstance(value, str) and value.strip():
                return required_text(value, "MLflow span_id")
    return required_text(None, "MLflow span_id")


def _parent_id(span: JsonObject) -> str | None:
    """Return the declared MLflow parent span identifier, if any.

    Args:
        span: MLflow span record.

    Returns:
        Declared parent span identifier, or ``None`` when absent.
    """
    for key in ("parent_id", "parentId", "parentSpanId", "parent_span_id"):
        value = span.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    attributes = span.get("attributes")
    if isinstance(attributes, dict):
        for key in ("mlflow.parentSpanId", "parent_id"):
            value = attributes.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _attributes(span: JsonObject) -> JsonObject:
    """Return the MLflow span attributes object.

    The server JSON-encodes attribute values one extra layer (so the model reads
    ``'"gpt-4o-mini"'`` rather than ``'gpt-4o-mini"'``); each value is decoded so
    downstream readers see the declared text. Values that are not JSON-encoded
    text are left unchanged.

    Args:
        span: MLflow span record.

    Returns:
        Attribute mapping, empty when the span declares none.
    """
    value = span.get("attributes")
    if not isinstance(value, dict):
        # Some exports carry attributes flat on the span.
        return {}
    decoded: JsonObject = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        text = json_value(item)
        if text is not None:
            decoded[key] = text
    return decoded


def _inputs(span: JsonObject, attributes: JsonObject) -> JsonValue | None:
    """Read the declared MLflow span inputs.

    Args:
        span: MLflow span record.
        attributes: Declared span attributes.

    Returns:
        Decoded inputs, or ``None`` when the span declares none.
    """
    for key in ("inputs", "input", "spanInputs"):
        value = span.get(key)
        if value is not None:
            return json_value(value)
    for key in ("mlflow.spanInputs", "mlflow.inputs", "spanInputs"):
        value = attributes.get(key)
        if value is not None:
            return json_value(value)
    return None


def _outputs(span: JsonObject, attributes: JsonObject) -> JsonValue | None:
    """Read the declared MLflow span outputs.

    Args:
        span: MLflow span record.
        attributes: Declared span attributes.

    Returns:
        Decoded outputs, or ``None`` when the span declares none.
    """
    for key in ("outputs", "output", "spanOutputs"):
        value = span.get(key)
        if value is not None:
            return json_value(value)
    for key in ("mlflow.spanOutputs", "mlflow.outputs", "spanOutputs"):
        value = attributes.get(key)
        if value is not None:
            return json_value(value)
    return None


def _input_messages(inputs: JsonValue | None) -> JsonValue | None:
    """Return the declared model input messages for one MLflow model span.

    Args:
        inputs: Decoded span inputs.

    Returns:
        The declared message list, or the raw inputs when no list is declared.
    """
    if isinstance(inputs, dict):
        for key in ("messages", "prompt", "input"):
            value = inputs.get(key)
            if isinstance(value, list):
                return value
        # MLflow chat inputs may wrap messages under ``inputs.messages``
        nested = inputs.get("inputs")
        if isinstance(nested, dict):
            for key in ("messages", "prompt"):
                value = nested.get(key)
                if isinstance(value, list):
                    return value
    return inputs


def _tool_name(span: JsonObject, attributes: JsonObject) -> str:
    """Read the executed tool name from an MLflow tool span.

    Args:
        span: MLflow span record.
        attributes: Declared span attributes.

    Returns:
        Declared tool name.

    Raises:
        VendorTraceFormatError: The span declares no tool name.
    """
    for key in ("toolName", "tool_name", "functionName", "mlflow.spanFunctionName", "name"):
        value = attributes.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    name = span.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return required_text(None, "MLflow tool span name")


def _interval(span: JsonObject) -> tuple[datetime, datetime]:
    """Read the source interval from MLflow nanosecond, millisecond, or ISO fields.

    Args:
        span: MLflow span record.

    Returns:
        Source start and end instants, equal when the span declares no end.

    Raises:
        VendorTraceFormatError: The span declares no readable start time.
    """
    # MLflow commonly uses nanosecond epoch fields (``start_time_unix_nano`` in
    # server exports, ``start_time_ns`` elsewhere) or ``start_time`` / ``end_time``
    # (milliseconds or ISO). Try nanoseconds first by converting to seconds for
    # the shared timestamp helper.
    ns_start = span.get("start_time_ns", span.get("startTimeNs", span.get("start_time_unix_nano")))
    ns_end = span.get("end_time_ns", span.get("endTimeNs", span.get("end_time_unix_nano")))
    if isinstance(ns_start, int | float):
        # Convert nanoseconds to seconds for source_timestamp epoch handling.
        start_seconds = float(ns_start) / 1_000_000_000
        end_seconds = float(ns_end) / 1_000_000_000 if isinstance(ns_end, int | float) else None
        return source_interval(
            start_seconds,
            end_seconds,
            start_label="MLflow span start_time_ns",
            end_label="MLflow span end_time_ns",
        )
    # Millisecond epoch variants
    ms_start = span.get("start_time", span.get("startTime"))
    ms_end = span.get("end_time", span.get("endTime"))
    # If value is a large int (>1e12) it's likely already milliseconds; source_timestamp
    # handles both seconds and milliseconds, so delegate directly.
    if ms_start is not None:
        return source_interval(
            ms_start,
            ms_end,
            start_label="MLflow span startTime",
            end_label="MLflow span endTime",
        )
    # ISO fallback from attributes
    attributes = span.get("attributes")
    attr_obj: JsonObject = attributes if isinstance(attributes, dict) else {}
    iso_start = attr_obj.get("startTime") or span.get("timestamp")
    iso_end = attr_obj.get("endTime")
    if iso_start is not None:
        return source_interval(
            iso_start,
            iso_end,
            start_label="MLflow span startTime",
            end_label="MLflow span endTime",
        )
    # Final fallback: trace info timestamps
    info = span.get("info")
    if isinstance(info, dict):
        for key in ("request_time", "timestamp", "createdAt"):
            if key in info:
                start = info[key]
                end = info.get("execution_duration")
                if isinstance(end, int | float) and isinstance(start, int | float):
                    # execution_duration is often milliseconds
                    try:
                        started = source_timestamp(start, "MLflow trace request_time")
                        ended = source_timestamp(
                            float(start) / 1000 + float(end) / 1000
                            if float(start) > 1e11
                            else float(start) + float(end) / 1000,
                            "MLflow trace end",
                        )
                        return started, ended
                    except VendorTraceFormatError:
                        pass
                return source_interval(
                    start,
                    None,
                    start_label="MLflow trace request_time",
                    end_label="MLflow trace end",
                )
    raise VendorTraceFormatError("MLflow span declares no readable start time")


def _usage(attributes: JsonObject, span: JsonObject) -> VendorTokenUsage | None:
    """Read declared token accounting from MLflow span attributes or the span itself.

    Args:
        attributes: Declared span attributes.
        span: MLflow span record.

    Returns:
        Declared usage, or ``None`` when the span declares no complete accounting.

    Raises:
        VendorTraceFormatError: A declared token count is not a non-negative integer.
    """
    for candidate in (
        attributes.get("llm.usage"),
        attributes.get("gen_ai.usage"),
        attributes.get("usage"),
        span.get("usage"),
        attributes,
    ):
        usage = declared_usage(candidate)
        if usage is not None:
            return usage
    return None


def _failure_message(span: JsonObject, attributes: JsonObject) -> str | None:
    """Read an MLflow span failure message when the status marks the span failed.

    Args:
        span: MLflow span record.
        attributes: Declared span attributes.

    Returns:
        Declared error text, or ``None`` when the span is not failed.
    """
    status = span.get("status")
    if isinstance(status, dict):
        code = str(status.get("status_code") or status.get("code") or "").upper()
        # Server exports use OpenTelemetry codes (``STATUS_CODE_OK``); strip the
        # prefix before comparing so healthy spans are not marked failed.
        code = code.removeprefix("STATUS_CODE_")
        if code and code not in {"OK", "SUCCESS", "UNSET"}:
            return (
                declared_error_message(
                    {**span, **attributes, **status}, keys=_ERROR_KEYS, label="MLflow span"
                )
                or f"MLflow span failed with status {code}"
            )
        # Some MLflow spans put error directly on status object.
        for key in _ERROR_KEYS:
            if key in status and isinstance(status[key], str) and status[key].strip():
                return status[key].strip()
    elif isinstance(status, str) and status.strip():
        if status.strip().upper().removeprefix("STATUS_CODE_") not in {"OK", "SUCCESS", "UNSET"}:
            return declared_error_message(span, keys=_ERROR_KEYS, label="MLflow span") or (
                declared_error_message(attributes, keys=_ERROR_KEYS, label="MLflow span")
                or f"MLflow span failed with status {status}"
            )
    # Fallback: error keys even without explicit failed status code
    failure = declared_error_message(span, keys=_ERROR_KEYS, label="MLflow span")
    if failure is not None:
        return failure
    return declared_error_message(attributes, keys=_ERROR_KEYS, label="MLflow span")


def _extensions(span: JsonObject, attributes: JsonObject) -> JsonObject:
    """Read approved EXP extensions and MLflow metadata from one span.

    Args:
        span: MLflow span record.
        attributes: Declared span attributes.

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
    # Preserve MLflow experiment and run identifiers when present
    for key in ("mlflow.experimentId", "mlflow.runId", "experiment_id", "run_id"):
        value = attributes.get(key) or span.get(key)
        if isinstance(value, str) and value.strip() and "exp.mlflow.experiment" not in extensions:
            extensions["exp.mlflow.experiment"] = value.strip()
            break
    return extensions


def _envelope_trace_id(envelope: JsonObject) -> str | None:
    """Read the trace identifier declared only at the envelope level.

    Args:
        envelope: MLflow trace envelope with ``info`` and ``data``.

    Returns:
        Declared envelope trace identifier, or ``None`` when absent.
    """
    info = envelope.get("info")
    if isinstance(info, dict):
        for key in ("trace_id", "traceId", "traceID", "request_id"):
            value = info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("trace_id", "traceId", "traceID", "request_id"):
        value = envelope.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _with_envelope_trace_id(span: JsonObject, envelope_trace_id: str | None) -> JsonObject:
    """Return a span record that carries the envelope trace identifier when missing.

    Args:
        span: MLflow span record.
        envelope_trace_id: Trace identifier declared only at the envelope level.

    Returns:
        Original span when it already carries a trace identifier or no envelope
        identifier exists, otherwise a shallow copy with the envelope identifier.
    """
    if envelope_trace_id is None:
        return span
    if any(
        key in span for key in ("trace_id", "traceId", "traceID", "request_id", "mlflow.traceId")
    ):
        return span
    # Preserve original mapping; inject the envelope identifier as trace_id.
    return {**span, "trace_id": envelope_trace_id}


def _mlflow_records(payload: JsonValue) -> tuple[JsonObject, ...]:
    """Flatten MLflow export shapes, including trace envelopes with ``data.spans``.

    Args:
        payload: One decoded export document, array, or trace envelope.

    Returns:
        MLflow span record objects in source order.

    Raises:
        VendorTraceFormatError: The payload is not a supported MLflow shape.
    """
    # Handle explicit trace envelope(s) where a trace has ``info`` + ``data.spans``.
    # MLflow trace-search responses identify the trace only through ``info.trace_id``;
    # child spans may not repeat that identifier. Preserve the envelope identity so
    # every valid child span is retained.
    if isinstance(payload, dict) and "traces" in payload and isinstance(payload["traces"], list):
        records: list[JsonObject] = []
        for trace in payload["traces"]:
            if not isinstance(trace, dict):
                raise VendorTraceFormatError("MLflow trace envelope must be an object")
            data = trace.get("data")
            if isinstance(data, dict) and isinstance(data.get("spans"), list):
                envelope_trace_id = _envelope_trace_id(trace)
                for span in data["spans"]:
                    if not isinstance(span, dict):
                        raise VendorTraceFormatError("MLflow spans must be objects")
                    records.append(_with_envelope_trace_id(span, envelope_trace_id))
                continue
            # Fallback: trace itself may already be a span-shaped record
            if any(
                key in trace for key in ("trace_id", "traceId", "span_id", "spanId", "request_id")
            ):
                records.append(trace)
                continue
            # Also handle data being a flat span list under data directly
            if isinstance(trace.get("data"), list):
                envelope_trace_id = _envelope_trace_id(trace)
                for span in trace["data"]:
                    if isinstance(span, dict):
                        records.append(_with_envelope_trace_id(span, envelope_trace_id))
                continue
            raise VendorTraceFormatError("MLflow trace envelope needs data.spans or a span record")
        return tuple(records)
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], dict):
        data = payload["data"]
        spans = data.get("spans")
        if isinstance(spans, list):
            envelope_trace_id = _envelope_trace_id(payload)
            return tuple(
                _with_envelope_trace_id(span, envelope_trace_id)
                for span in spans
                if isinstance(span, dict)
            )
    # Single trace envelope without the outer ``traces`` wrapper.
    if isinstance(payload, dict) and "info" in payload and "data" in payload:
        info = payload.get("info")
        data = payload.get("data")
        if (
            isinstance(info, dict)
            and isinstance(data, dict)
            and isinstance(data.get("spans"), list)
        ):
            envelope_trace_id = _envelope_trace_id(payload)
            return tuple(
                _with_envelope_trace_id(span, envelope_trace_id)
                for span in data["spans"]
                if isinstance(span, dict)
            )
    # Fallback to strict wrapper flattening for flat span arrays, spans envelopes, JSONL.
    return record_flattener(
        vendor=VENDOR,
        wrapper_keys=("traces", "data", "spans", "results", "items", "events"),
        record_keys=(
            "trace_id",
            "traceId",
            "span_id",
            "spanId",
            "request_id",
            "mlflow.traceId",
        ),
    )(payload)


MLFLOW_SOURCE: VendorSource[JsonObject] = VendorSource(
    vendor=VENDOR,
    records=_mlflow_records,
    convert=_span_observation,
)
