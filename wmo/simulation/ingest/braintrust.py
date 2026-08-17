"""Normalize Braintrust log exports into canonical trace evidence.

Braintrust exports one row per span. A row declares ``span_id``, ``root_span_id``, ``span_parents``,
``input``, ``output``, ``metrics``, ``metadata``, ``span_attributes``, and any ``error``. Exports
arrive as a row array, an envelope with a row array, or JSONL with one row per line, and the rows of
one trace group by ``root_span_id``.

Span types map to canonical evidence by what they observe:

- ``llm`` becomes a model call, including the tool calls its output requests,
- ``tool`` becomes the tool result paired with the earlier requesting call,
- ``task``, ``function``, ``score``, and ``eval`` observe orchestration or grading rather than a
  visible agent step and are not converted.

Braintrust records timing in ``metrics.start`` and ``metrics.end`` as epoch seconds, and token
counts in ``metrics.prompt_tokens`` and ``metrics.completion_tokens``. Model identity is retained
only when the row's metadata declares both a provider and a model.
"""

from __future__ import annotations

from datetime import datetime

from wmo.common.core.artifacts import JsonObject
from wmo.simulation.ingest.vendor_observations import (
    VendorModelIdentity,
    VendorObservation,
    VendorTokenUsage,
    declared_completion_text,
    declared_error_message,
    declared_model_identity,
    declared_tool_calls,
    declared_usage,
)
from wmo.simulation.ingest.vendor_records import (
    first_text,
    first_user_text,
    json_text,
    json_value,
    required_text,
    source_interval,
)
from wmo.simulation.ingest.vendor_source import VendorSource, record_flattener
from wmo.simulation.ingest.vendor_trace import approved_extensions

VENDOR = "braintrust"

_MODEL_TYPES = frozenset({"llm"})
_TOOL_TYPES = frozenset({"tool"})
_MODEL_KEYS = ("model", "model_name")
_PROVIDER_KEYS = ("provider", "model_provider")
_ERROR_KEYS = ("error",)


def _row_observation(row: JsonObject, ordinal: int) -> tuple[VendorObservation, ...]:
    """Convert one Braintrust row to a declared model or tool-result observation.

    Args:
        row: Braintrust span row.
        ordinal: Source order position for the emitted observation.

    Returns:
        Declared observation, or nothing for orchestration-only span types.

    Raises:
        VendorTraceFormatError: The row lacks identity, timing, or tool evidence.
    """
    span_type = _span_type(row)
    if span_type not in _MODEL_TYPES | _TOOL_TYPES:
        return ()
    source_span_id = required_text(row.get("span_id", row.get("id")), "Braintrust span_id")
    source_trace_id = first_text(row, ("root_span_id",)) or source_span_id
    started_at, ended_at = _interval(row)
    inputs = json_value(row.get("input"))
    outputs = json_value(row.get("output"))
    extensions = _extensions(row)
    failure = declared_error_message(row, keys=_ERROR_KEYS, label="Braintrust row")
    if span_type in _TOOL_TYPES:
        return (
            VendorObservation(
                source_trace_id=source_trace_id,
                source_span_id=source_span_id,
                ordinal=ordinal,
                started_at=started_at,
                ended_at=ended_at,
                kind="tool_result",
                source_parent_span_id=_parent(row),
                request_text=first_user_text(inputs),
                tool_name=_tool_name(row),
                tool_arguments=json_text(inputs),
                tool_message=declared_completion_text(outputs),
                failure_message=failure,
                extensions=extensions,
            ),
        )
    model, declared_model = _model_identity(row)
    return (
        VendorObservation(
            source_trace_id=source_trace_id,
            source_span_id=source_span_id,
            ordinal=ordinal,
            started_at=started_at,
            ended_at=ended_at,
            kind="model",
            source_parent_span_id=_parent(row),
            request_text=first_user_text(inputs),
            input_messages=inputs if isinstance(inputs, list) else None,
            completion_text=declared_completion_text(outputs) or None,
            tool_calls=declared_tool_calls(outputs),
            model=model,
            usage=_usage(row),
            failure_message=failure,
            declared_attributes=(
                {} if declared_model is None else {"gen_ai.request.model": declared_model}
            ),
            extensions=extensions,
        ),
    )


def _span_type(row: JsonObject) -> str:
    """Read the declared Braintrust span type.

    Args:
        row: Braintrust span row.

    Returns:
        Lowercase declared span type, empty when the row declares none.
    """
    attributes = row.get("span_attributes")
    if isinstance(attributes, dict):
        declared = first_text(attributes, ("type",))
        if declared is not None:
            return declared.casefold()
    return (first_text(row, ("type",)) or "").casefold()


def _tool_name(row: JsonObject) -> str:
    """Read the executed tool name declared by one Braintrust tool row.

    Args:
        row: Braintrust span row of type tool.

    Returns:
        Declared tool name.

    Raises:
        VendorTraceFormatError: The row declares no tool name.
    """
    attributes = row.get("span_attributes")
    if isinstance(attributes, dict):
        name = first_text(attributes, ("name",))
        if name is not None:
            return name
    return required_text(row.get("name"), "Braintrust tool row name")


def _parent(row: JsonObject) -> str | None:
    """Return the single declared parent span key, if the row declares exactly one.

    Args:
        row: Braintrust span row.

    Returns:
        Declared parent span key, or ``None`` when absent or ambiguous.
    """
    parents = row.get("span_parents")
    if isinstance(parents, list) and len(parents) == 1 and isinstance(parents[0], str):
        return parents[0]
    return first_text(row, ("parent_span_id",))


def _interval(row: JsonObject) -> tuple[datetime, datetime]:
    """Read the source interval from Braintrust metrics or the row creation time.

    Args:
        row: Braintrust span row.

    Returns:
        Source start and end instants, equal when the row declares no end.

    Raises:
        VendorTraceFormatError: The row declares no readable start time.
    """
    metrics = row.get("metrics")
    metrics_object: JsonObject = metrics if isinstance(metrics, dict) else {}
    return source_interval(
        metrics_object.get("start", row.get("created")),
        metrics_object.get("end"),
        start_label="Braintrust metrics start",
        end_label="Braintrust metrics end",
    )


def _usage(row: JsonObject) -> VendorTokenUsage | None:
    """Read declared token accounting from Braintrust metrics.

    Args:
        row: Braintrust span row.

    Returns:
        Declared usage, or ``None`` when metrics declare no complete accounting.

    Raises:
        VendorTraceFormatError: A declared token count is not a non-negative integer.
    """
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return declared_usage(metrics)


def _model_identity(row: JsonObject) -> tuple[VendorModelIdentity | None, str | None]:
    """Read declared model identity from Braintrust row metadata.

    Args:
        row: Braintrust span row.

    Returns:
        Retained model identity when a provider is declared, and the declared model name.
    """
    metadata = row.get("metadata")
    merged: JsonObject = dict(metadata) if isinstance(metadata, dict) else {}
    for key in (*_MODEL_KEYS, *_PROVIDER_KEYS):
        if key in row:
            merged[key] = row[key]
    return declared_model_identity(
        merged,
        model_keys=_MODEL_KEYS,
        provider_keys=_PROVIDER_KEYS,
    )


def _extensions(row: JsonObject) -> JsonObject:
    """Read approved WMO extensions and Braintrust metadata from one row.

    Args:
        row: Braintrust span row.

    Returns:
        Approved extension attributes for the spans of this row.
    """
    extensions = approved_extensions(row)
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        extensions.update(approved_extensions(metadata))
        if "wmo.trace.metadata" not in extensions:
            extensions["wmo.trace.metadata"] = metadata
    return extensions


BRAINTRUST_SOURCE: VendorSource[JsonObject] = VendorSource(
    vendor=VENDOR,
    records=record_flattener(
        vendor=VENDOR,
        wrapper_keys=("events", "rows", "data", "results", "items"),
        record_keys=("span_id", "root_span_id", "span_attributes"),
    ),
    convert=_row_observation,
)
