"""Normalize Langfuse trace exports into canonical trace evidence.

Langfuse exports an agent run as a trace object with an ``observations`` array, and its public API
returns the same shape. Exports of bare observations that each carry ``traceId`` are also supported,
because Langfuse's observation endpoints page over observations rather than traces.

Observation types map to canonical evidence by what they observe:

- ``GENERATION`` becomes a model call, including any tool calls its output requests,
- ``TOOL`` and ``RETRIEVER`` become tool results paired with the earlier requesting call,
- ``SPAN``, ``EVENT``, ``CHAIN``, ``AGENT``, ``EVALUATOR``, and ``GUARDRAIL`` observe orchestration
  rather than a visible agent step and are not converted,
- a trace with no convertible observation is retained as one agent-level record when it declares
  visible output.

Langfuse names a model without naming a provider. The declared model name is retained as
``gen_ai.request.model`` evidence, and resolved model identity is retained only when the export also
declares a provider in the trace or observation metadata.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject, SourceIdentity
from wmo.simulation.ingest.otlp import (
    GENAI_SEMANTIC_CONVENTION_VERSION,
    TraceNormalizationIssue,
    TraceNormalizationResult,
)
from wmo.simulation.ingest.vendor_observations import (
    VendorModelIdentity,
    VendorObservation,
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

VENDOR = "langfuse"

_MODEL_TYPES = frozenset({"GENERATION"})
_TOOL_TYPES = frozenset({"TOOL", "RETRIEVER"})
_MODEL_KEYS = ("model", "modelName")
_PROVIDER_KEYS = ("provider", "modelProvider", "model_provider")
_ERROR_KEYS = ("statusMessage", "error")


def load_langfuse_file(
    path: Path,
    *,
    semantic_convention_version: str = GENAI_SEMANTIC_CONVENTION_VERSION,
    source_id: str | None = None,
) -> TraceNormalizationResult:
    """Read a Langfuse JSON or JSONL export into canonical trace evidence.

    Args:
        path: Langfuse trace export, observation export, or JSONL export.
        semantic_convention_version: Pinned GenAI semantic-convention version for the traces.
        source_id: Optional durable source label. The local path is used when omitted.

    Returns:
        Canonical traces and every retained parse or validation exclusion.

    Raises:
        VendorTraceFormatError: The export cannot be read or decoded.
    """
    export = read_vendor_export(path, vendor=VENDOR, source_id=source_id)
    return normalize_langfuse_payloads(
        export.payloads,
        source=export.source,
        semantic_convention_version=semantic_convention_version,
        initial_issues=export.issues,
    )


def normalize_langfuse_payloads(
    payloads: Sequence[JsonValue],
    *,
    source: SourceIdentity,
    semantic_convention_version: str = GENAI_SEMANTIC_CONVENTION_VERSION,
    initial_issues: Sequence[TraceNormalizationIssue] = (),
) -> TraceNormalizationResult:
    """Normalize decoded Langfuse payloads into canonical traces.

    Args:
        payloads: Decoded Langfuse documents in source order.
        source: Immutable identity of the source bytes or transport result.
        semantic_convention_version: Pinned GenAI semantic-convention version for the traces.
        initial_issues: Parse exclusions collected before record mapping.

    Returns:
        Canonical traces and every retained validation exclusion.
    """
    issues = list(initial_issues)
    observations: list[VendorObservation] = []
    ordinal = 0
    for index, payload in enumerate(payloads, start=1):
        try:
            records = flatten_records(
                payload,
                vendor=VENDOR,
                wrapper_keys=("data", "traces", "results", "items"),
                record_keys=("observations", "traceId", "trace_id"),
            )
        except VendorTraceFormatError as exc:
            issues.append(TraceNormalizationIssue(f"record-{index}", str(exc)))
            continue
        for record in records:
            try:
                emitted = _record_observations(record, ordinal)
            except VendorTraceFormatError as exc:
                issues.append(TraceNormalizationIssue(f"record-{index}", str(exc)))
                continue
            observations.extend(emitted)
            ordinal += len(emitted)
    return build_vendor_traces(
        observations,
        vendor=VENDOR,
        source=source,
        semantic_convention_version=semantic_convention_version,
        initial_issues=issues,
    )


def _record_observations(record: JsonObject, ordinal: int) -> tuple[VendorObservation, ...]:
    """Convert one Langfuse trace or bare observation record.

    Args:
        record: Langfuse trace object or single observation object.
        ordinal: Source order offset for the first emitted observation.

    Returns:
        Declared observations for this record.

    Raises:
        VendorTraceFormatError: The record is not a supported Langfuse shape.
    """
    if "observations" in record:
        return _trace_observations(record, ordinal)
    source_trace_id = required_text(
        record.get("traceId", record.get("trace_id")), "Langfuse observation traceId"
    )
    converted = _observation(
        record,
        source_trace_id=source_trace_id,
        ordinal=ordinal,
        request_text=first_user_text(json_value(record.get("input"))),
        extensions={},
        trace_provider=None,
    )
    return () if converted is None else (converted,)


def _trace_observations(trace: JsonObject, ordinal: int) -> tuple[VendorObservation, ...]:
    """Convert one Langfuse trace object and its observations.

    Args:
        trace: Langfuse trace object with an observations array.
        ordinal: Source order offset for the first emitted observation.

    Returns:
        Declared observations for this trace.

    Raises:
        VendorTraceFormatError: The trace has no identity, observations array, or request text.
    """
    raw_observations = trace.get("observations")
    if not isinstance(raw_observations, list):
        raise VendorTraceFormatError("Langfuse traces need an observations array")
    source_trace_id = required_text(trace.get("id"), "Langfuse trace id")
    request_text = first_user_text(json_value(trace.get("input")))
    extensions = _trace_extensions(trace)
    trace_provider = _provider(trace) or _provider_from_metadata(trace.get("metadata"))
    emitted: list[VendorObservation] = []
    for raw in raw_observations:
        if not isinstance(raw, dict):
            raise VendorTraceFormatError("Langfuse observations must be objects")
        converted = _observation(
            raw,
            source_trace_id=source_trace_id,
            ordinal=ordinal + len(emitted),
            request_text=request_text if not emitted else None,
            extensions=extensions,
            trace_provider=trace_provider,
        )
        if converted is not None:
            emitted.append(converted)
    if emitted:
        return tuple(emitted)
    return _trace_only_observation(
        trace,
        source_trace_id=source_trace_id,
        ordinal=ordinal,
        request_text=request_text,
        extensions=extensions,
    )


def _trace_only_observation(
    trace: JsonObject,
    *,
    source_trace_id: str,
    ordinal: int,
    request_text: str | None,
    extensions: JsonObject,
) -> tuple[VendorObservation, ...]:
    """Retain a trace with no convertible observation as agent-level evidence.

    Args:
        trace: Langfuse trace object.
        source_trace_id: Langfuse trace identity.
        ordinal: Source order position for the emitted observation.
        request_text: Request text declared by the trace input.
        extensions: Approved WMO extension attributes for the trace.

    Returns:
        One agent-level observation, or nothing when the trace declares no visible output.

    Raises:
        VendorTraceFormatError: The trace declares output but no timestamp or request text.
    """
    completion = declared_completion_text(json_value(trace.get("output")))
    if not completion:
        return ()
    started_at = source_timestamp(
        trace.get("timestamp", trace.get("createdAt")), "Langfuse trace timestamp"
    )
    return (
        VendorObservation(
            source_trace_id=source_trace_id,
            source_span_id=source_trace_id,
            ordinal=ordinal,
            started_at=started_at,
            ended_at=started_at,
            kind="agent",
            request_text=request_text,
            completion_text=completion,
            failure_message=declared_error_message(trace, keys=_ERROR_KEYS, label="Langfuse trace"),
            extensions=extensions,
        ),
    )


def _observation(
    raw: JsonObject,
    *,
    source_trace_id: str,
    ordinal: int,
    request_text: str | None,
    extensions: JsonObject,
    trace_provider: str | None,
) -> VendorObservation | None:
    """Convert one Langfuse observation to a declared model or tool-result record.

    Args:
        raw: Langfuse observation object.
        source_trace_id: Langfuse trace identity.
        ordinal: Source order position for the emitted observation.
        request_text: Trace request text, supplied only for the first emitted observation.
        extensions: Approved WMO extension attributes for the trace.
        trace_provider: Provider declared at trace level, when any.

    Returns:
        Declared observation, or ``None`` for orchestration-only observation types.

    Raises:
        VendorTraceFormatError: The observation lacks identity, timing, or tool evidence.
    """
    observation_type = (first_text(raw, ("type",)) or "").upper()
    if observation_type not in _MODEL_TYPES | _TOOL_TYPES:
        return None
    source_span_id = required_text(raw.get("id"), "Langfuse observation id")
    started_at = source_timestamp(raw.get("startTime"), "Langfuse observation startTime")
    ended_at = (
        source_timestamp(raw.get("endTime"), "Langfuse observation endTime")
        if raw.get("endTime") is not None
        else started_at
    )
    parent = first_text(raw, ("parentObservationId", "parent_observation_id"))
    failure = _failure_message(raw)
    if observation_type in _TOOL_TYPES:
        return VendorObservation(
            source_trace_id=source_trace_id,
            source_span_id=source_span_id,
            ordinal=ordinal,
            started_at=started_at,
            ended_at=ended_at,
            kind="tool_result",
            source_parent_span_id=parent,
            request_text=request_text,
            tool_name=required_text(raw.get("name"), "Langfuse tool observation name"),
            tool_arguments=json_text(json_value(raw.get("input"))),
            tool_message=declared_completion_text(json_value(raw.get("output"))),
            failure_message=failure,
            extensions=extensions,
        )
    output = json_value(raw.get("output"))
    tool_calls = declared_tool_calls(output)
    model, declared_model = _model_identity(raw, trace_provider)
    return VendorObservation(
        source_trace_id=source_trace_id,
        source_span_id=source_span_id,
        ordinal=ordinal,
        started_at=started_at,
        ended_at=ended_at,
        kind="model",
        source_parent_span_id=parent,
        request_text=request_text,
        input_messages=json_value(raw.get("input")),
        completion_text=declared_completion_text(output) or None,
        tool_calls=tool_calls,
        model=model,
        usage=declared_usage(raw.get("usage", raw.get("usageDetails"))),
        failure_message=failure,
        declared_attributes=(
            {} if declared_model is None else {"gen_ai.request.model": declared_model}
        ),
        extensions=extensions,
    )


def _failure_message(raw: JsonObject) -> str | None:
    """Read an observation error only when Langfuse marks the record at error level.

    Args:
        raw: Langfuse observation object.

    Returns:
        Declared error text, or ``None`` when the observation is not an error.
    """
    level = (first_text(raw, ("level",)) or "").upper()
    if level != "ERROR":
        return None
    return (
        declared_error_message(raw, keys=_ERROR_KEYS, label="Langfuse observation")
        or "Langfuse marked this observation at ERROR level without a message."
    )


def _model_identity(
    raw: JsonObject, trace_provider: str | None
) -> tuple[VendorModelIdentity | None, str | None]:
    """Read declared model identity from the observation, its metadata, and the trace.

    Args:
        raw: Langfuse observation object.
        trace_provider: Provider declared at trace level, when any.

    Returns:
        Retained model identity when a provider is declared, and the declared model name.
    """
    metadata = raw.get("metadata")
    merged: JsonObject = dict(metadata) if isinstance(metadata, dict) else {}
    merged.update({key: raw[key] for key in _MODEL_KEYS if key in raw})
    provider = _provider(raw) or _provider_from_metadata(metadata) or trace_provider
    if provider is not None:
        merged["provider"] = provider
    return declared_model_identity(
        merged,
        model_keys=_MODEL_KEYS,
        provider_keys=_PROVIDER_KEYS,
    )


def _provider(record: JsonObject) -> str | None:
    """Read an explicitly declared provider name from one Langfuse record.

    Args:
        record: Langfuse trace or observation object.

    Returns:
        Declared provider name, or ``None`` when the record declares none.
    """
    return first_text(record, _PROVIDER_KEYS)


def _provider_from_metadata(metadata: JsonValue | None) -> str | None:
    """Read a declared provider name from Langfuse metadata.

    Args:
        metadata: Langfuse metadata object, when present.

    Returns:
        Declared provider name, or ``None`` when metadata declares none.
    """
    if not isinstance(metadata, dict):
        return None
    return first_text(metadata, _PROVIDER_KEYS)


def _trace_extensions(trace: JsonObject) -> JsonObject:
    """Read approved WMO extensions and Langfuse metadata from one trace object.

    Args:
        trace: Langfuse trace object.

    Returns:
        Approved extension attributes for every span of this trace.
    """
    extensions = approved_extensions(trace)
    metadata = trace.get("metadata")
    if isinstance(metadata, dict):
        extensions.update(approved_extensions(metadata))
        if "wmo.trace.metadata" not in extensions:
            extensions["wmo.trace.metadata"] = metadata
    session_id = first_text(trace, ("sessionId", "session_id"))
    if session_id is not None and "wmo.conversation.id" not in extensions:
        extensions["wmo.conversation.id"] = session_id
    user_id = first_text(trace, ("userId", "user_id"))
    if user_id is not None and "wmo.customer.id" not in extensions:
        extensions["wmo.customer.id"] = user_id
    return extensions
