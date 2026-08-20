"""Normalize validated OpenTelemetry GenAI JSON into canonical trace evidence.

This module accepts only OpenTelemetry JSON and JSONL records that carry W3C trace identities
and GenAI semantic-convention attributes. It intentionally does not detect other vendor formats.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import JsonValue

from exp.common.core.artifacts import FailureCode, JsonObject, SourceIdentity, StructuredFailure
from exp.common.core.text import normalize_durable_text
from exp.common.models import BillingSource, ModelSnapshot, Usage
from exp.common.traces import Trace, TraceSource, TraceSpan
from exp.simulation.ingest.environment_capture import canonicalize_environment_capture_payloads
from exp.simulation.ingest.json_strict import DuplicateJsonKeyError, reject_duplicate_json_keys
from exp.simulation.ingest.model_identity import (
    TraceModelIdentityEvidence,
    normalized_capabilities_sha256,
    normalized_connection_sha256,
    normalized_model_identity_evidence,
)
from exp.simulation.ingest.trace_extensions import (
    CONVERSATION_ID_KEYS,
    REQUEST_CONTEXT_KEY,
    REQUEST_TOOLS_KEY,
    collect_outcome,
    collect_tools,
    consistent_json_object,
    consistent_text,
    json_value,
    required_text,
)

GENAI_SEMANTIC_CONVENTION_VERSION = "1.37.0"

_TRACE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_SPAN_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")
_MODEL_OPERATIONS = frozenset(
    {"chat", "text_completion", "generate_content", "invoke_agent", "embeddings"}
)
_TOOL_OPERATION = "execute_tool"


class OtlpTraceFormatError(ValueError):
    """Raised when a trace upload cannot be decoded as OpenTelemetry JSON."""


@dataclass(frozen=True)
class TraceNormalizationIssue:
    """One source record excluded from canonical trace evidence.

    Args:
        source_record: Stable source location, such as a JSONL line number or trace identity.
        message: Human-readable validation reason.
    """

    source_record: str
    message: str


@dataclass(frozen=True)
class TraceNormalizationResult:
    """The valid canonical traces and explicit exclusions from one source conversion.

    Args:
        traces: Valid normalized production traces in deterministic order.
        issues: Corrupt or incomplete source records that were excluded without repair.
        identity_evidence: Exact model-span provenance from a telemetry-aware normalizer. ``None``
            marks direct or programmatic records whose digest origin is unspecified.
    """

    traces: tuple[Trace, ...]
    issues: tuple[TraceNormalizationIssue, ...]
    identity_evidence: tuple[TraceModelIdentityEvidence, ...] | None = None

    @property
    def invalid_trace_count(self) -> int:
        """Return the number of source records excluded by validation."""
        return len(self.issues)


@dataclass(frozen=True)
class _RawSpan:
    """One decoded OTLP span with inherited resource attributes."""

    raw: JsonObject
    resource_attributes: JsonObject
    ordinal: int


def load_otlp_file(
    path: Path,
    *,
    semantic_convention_version: str = GENAI_SEMANTIC_CONVENTION_VERSION,
    source_id: str | None = None,
) -> TraceNormalizationResult:
    """Read an OTLP JSON or JSONL file into canonical trace evidence.

    Args:
        path: JSON document or JSONL file containing OTLP spans.
        semantic_convention_version: Pinned GenAI semantic-convention version accepted by this run.
        source_id: Optional durable source label. The local path is used when omitted.

    Returns:
        Valid canonical traces and all excluded-record validation issues.

    Raises:
        OtlpTraceFormatError: The file cannot be decoded as UTF-8 JSON or JSONL.
    """
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise OtlpTraceFormatError(f"cannot read OTLP trace file {path}: {exc}") from exc
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OtlpTraceFormatError(f"OTLP trace file is not UTF-8: {path}") from exc

    source = SourceIdentity(
        kind="otlp",
        source_id=source_id or str(path),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    try:
        document: JsonValue = json.loads(text)
    except json.JSONDecodeError:
        return _normalize_jsonl(
            text,
            source=source,
            semantic_convention_version=semantic_convention_version,
        )
    return normalize_otlp_payload(
        document,
        source=source,
        semantic_convention_version=semantic_convention_version,
    )


def normalize_otlp_payload(
    payload: JsonValue,
    *,
    source: SourceIdentity,
    semantic_convention_version: str = GENAI_SEMANTIC_CONVENTION_VERSION,
) -> TraceNormalizationResult:
    """Normalize one decoded OTLP JSON payload into canonical traces.

    Args:
        payload: One decoded OTLP document, array of spans, or direct OTLP span.
        source: Immutable identity of the source bytes or transport result.
        semantic_convention_version: Pinned GenAI semantic-convention version accepted by this run.

    Returns:
        Valid canonical traces and explicit validation exclusions.

    Raises:
        OtlpTraceFormatError: The semantic convention version is blank.
    """
    return normalize_otlp_payloads(
        (payload,),
        source=source,
        semantic_convention_version=semantic_convention_version,
    )


def normalize_otlp_payloads(
    payloads: Sequence[JsonValue],
    *,
    source: SourceIdentity,
    semantic_convention_version: str = GENAI_SEMANTIC_CONVENTION_VERSION,
    initial_issues: Sequence[TraceNormalizationIssue] = (),
) -> TraceNormalizationResult:
    """Normalize OTLP payload fragments that may contribute spans to the same trace.

    Args:
        payloads: Decoded JSON documents or JSONL records in source order.
        source: Immutable identity shared by all source records.
        semantic_convention_version: Pinned GenAI semantic-convention version accepted by this run.
        initial_issues: Parse failures collected before structured OTLP validation.

    Returns:
        Valid canonical traces and explicit validation exclusions.

    Raises:
        OtlpTraceFormatError: The semantic convention version is blank.
    """
    if not semantic_convention_version.strip():
        raise OtlpTraceFormatError("semantic convention version must not be blank")
    raw_spans: list[_RawSpan] = []
    issues = list(initial_issues)
    ordinal = 0
    for payload_index, payload in enumerate(payloads, start=1):
        try:
            extracted = _extract_raw_spans(payload, ordinal)
        except OtlpTraceFormatError as exc:
            issues.append(TraceNormalizationIssue(f"record-{payload_index}", str(exc)))
            continue
        raw_spans.extend(extracted)
        ordinal += len(extracted)

    by_trace: dict[str, list[_RawSpan]] = defaultdict(list)
    for raw_span in raw_spans:
        trace_id = raw_span.raw.get("traceId")
        if not isinstance(trace_id, str):
            issues.append(
                TraceNormalizationIssue(
                    f"span-{raw_span.ordinal}", "OTLP span is missing string traceId"
                )
            )
            continue
        by_trace[trace_id].append(raw_span)

    traces: list[Trace] = []
    for trace_id in sorted(by_trace):
        group = by_trace[trace_id]
        try:
            trace = _normalize_trace_group(
                group,
                source=source,
                semantic_convention_version=semantic_convention_version,
            )
        except OtlpTraceFormatError as exc:
            issues.append(TraceNormalizationIssue(f"trace-{trace_id}", str(exc)))
        else:
            traces.append(trace)
    traces.sort(key=lambda trace: (trace.spans[0].started_at, trace.trace_id))
    normalized_traces = tuple(traces)
    return TraceNormalizationResult(
        traces=normalized_traces,
        issues=tuple(issues),
        identity_evidence=normalized_model_identity_evidence(normalized_traces),
    )


def _normalize_jsonl(
    text: str,
    *,
    source: SourceIdentity,
    semantic_convention_version: str,
) -> TraceNormalizationResult:
    """Decode JSONL and admit only an unambiguous complete capture profile.

    Args:
        text: UTF-8 JSONL source text.
        source: Immutable identity of the original source bytes.
        semantic_convention_version: Pinned GenAI convention used for normalization.

    Returns:
        Canonical traces plus every retained parse or validation exclusion.

    Raises:
        OtlpTraceFormatError: The source contains neither records nor parse issues.
    """
    payloads: list[JsonValue] = []
    profile_payloads: list[JsonValue] = []
    profile_eligible = True
    issues: list[TraceNormalizationIssue] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            profile_eligible = False
            issues.append(
                TraceNormalizationIssue(f"line-{line_number}", f"invalid JSONL record: {exc.msg}")
            )
            continue
        payloads.append(payload)
        try:
            profile_payloads.append(json.loads(line, object_pairs_hook=reject_duplicate_json_keys))
        except DuplicateJsonKeyError:
            profile_eligible = False
    if not payloads and not issues:
        raise OtlpTraceFormatError("OTLP JSONL file contains no records")
    if profile_eligible and not issues:
        canonical_payloads = canonicalize_environment_capture_payloads(profile_payloads)
        if canonical_payloads is not None:
            payloads = list(canonical_payloads)
    return normalize_otlp_payloads(
        payloads,
        source=source,
        semantic_convention_version=semantic_convention_version,
        initial_issues=issues,
    )


def _extract_raw_spans(payload: JsonValue, ordinal: int) -> list[_RawSpan]:
    """Extract direct and resource-scoped OTLP spans without format guessing."""
    if isinstance(payload, list):
        spans: list[_RawSpan] = []
        next_ordinal = ordinal
        for item in payload:
            extracted = _extract_raw_spans(item, next_ordinal)
            spans.extend(extracted)
            next_ordinal += len(extracted)
        return spans
    if not isinstance(payload, dict):
        raise OtlpTraceFormatError("OTLP payload must be an object, array, or JSONL object")
    resource_spans = payload.get("resourceSpans")
    if isinstance(resource_spans, list):
        spans = []
        next_ordinal = ordinal
        for resource_span in resource_spans:
            if not isinstance(resource_span, dict):
                raise OtlpTraceFormatError("resourceSpans entries must be objects")
            resource = resource_span.get("resource")
            resource_attributes = _decode_attributes(
                resource.get("attributes") if isinstance(resource, dict) else None,
                label="resource attributes",
            )
            scope_spans = resource_span.get("scopeSpans")
            if not isinstance(scope_spans, list):
                raise OtlpTraceFormatError("resourceSpans entries must contain scopeSpans arrays")
            for scope_span in scope_spans:
                if not isinstance(scope_span, dict):
                    raise OtlpTraceFormatError("scopeSpans entries must be objects")
                raw_spans = scope_span.get("spans")
                if not isinstance(raw_spans, list):
                    raise OtlpTraceFormatError("scopeSpans entries must contain spans arrays")
                for raw in raw_spans:
                    if not isinstance(raw, dict):
                        raise OtlpTraceFormatError("OTLP span entries must be objects")
                    spans.append(
                        _RawSpan(
                            raw=raw, resource_attributes=resource_attributes, ordinal=next_ordinal
                        )
                    )
                    next_ordinal += 1
        return spans
    if "traceId" in payload or "spanId" in payload:
        return [_RawSpan(raw=payload, resource_attributes={}, ordinal=ordinal)]
    raise OtlpTraceFormatError("OTLP payload has no resourceSpans or direct span identity")


def _normalize_trace_group(
    raw_spans: Sequence[_RawSpan],
    *,
    source: SourceIdentity,
    semantic_convention_version: str,
) -> Trace:
    """Validate one W3C trace group and build its canonical trace record."""
    raw_trace_id = _required_text(raw_spans[0].raw.get("traceId"), "traceId")
    trace_id = _validate_w3c_id(raw_trace_id, kind="trace")
    normalized_evidence: list[tuple[TraceSpan, JsonObject, int]] = []
    for raw_span in raw_spans:
        if _required_text(raw_span.raw.get("traceId"), "traceId") != trace_id:
            raise OtlpTraceFormatError("one trace group contains conflicting traceId values")
        attributes = dict(raw_span.resource_attributes)
        attributes.update(
            _decode_attributes(raw_span.raw.get("attributes"), label="span attributes")
        )
        if not _is_genai_span(attributes):
            continue
        normalized_evidence.append(
            (_normalize_span(raw_span.raw, attributes), attributes, raw_span.ordinal)
        )
    if not normalized_evidence:
        raise OtlpTraceFormatError("trace has no GenAI semantic-convention spans")
    normalized_evidence.sort(
        key=lambda evidence: (evidence[0].started_at, evidence[0].span_id, evidence[2])
    )
    normalized_spans = tuple(evidence[0] for evidence in normalized_evidence)
    all_attributes = tuple(evidence[1] for evidence in normalized_evidence)
    _validate_tool_pairs(normalized_spans)
    initial_request = _initial_request(all_attributes)
    if initial_request is None:
        raise OtlpTraceFormatError("trace has no initial user prompt in GenAI attributes")
    task, initial_attributes = initial_request
    initial_context = consistent_json_object(
        (initial_attributes,), REQUEST_CONTEXT_KEY, error_type=OtlpTraceFormatError
    )
    conversation_id = consistent_text(
        # Lenient: foreign exporters populate identity attributes with non-text values, so an
        # invalid conversation id is ignored instead of excluding the trace.
        all_attributes,
        CONVERSATION_ID_KEYS,
        error_type=OtlpTraceFormatError,
        lenient=True,
    )
    tools = collect_tools(
        (initial_attributes,),
        keys=("gen_ai.tool.definitions", REQUEST_TOOLS_KEY),
        error_type=OtlpTraceFormatError,
    )
    outcome = collect_outcome(all_attributes, failures=(), error_type=OtlpTraceFormatError)
    return Trace(
        trace_id=trace_id,
        conversation_id=conversation_id,
        task=task,
        initial_context=initial_context,
        tools=tools,
        spans=normalized_spans,
        outcome=outcome,
        source=TraceSource(
            identity=source,
            semantic_convention_version=semantic_convention_version,
        ),
    )


def _normalize_span(raw: JsonObject, attributes: JsonObject) -> TraceSpan:
    """Validate one GenAI span and convert it to the canonical span contract."""
    span_id = _validate_w3c_id(_required_text(raw.get("spanId"), "spanId"), kind="span")
    parent_raw = raw.get("parentSpanId")
    parent_span_id = None
    if parent_raw is not None:
        parent_span_id = _validate_w3c_id(_required_text(parent_raw, "parentSpanId"), kind="span")
    name = _required_text(raw.get("name"), "span name")
    started_at = _timestamp(raw.get("startTimeUnixNano"), "startTimeUnixNano")
    ended_at = _timestamp(raw.get("endTimeUnixNano"), "endTimeUnixNano")
    if ended_at < started_at:
        raise OtlpTraceFormatError("span endTimeUnixNano is before startTimeUnixNano")
    operation = attributes.get("gen_ai.operation.name")
    if not isinstance(operation, str) or not operation.strip():
        raise OtlpTraceFormatError("GenAI span is missing gen_ai.operation.name")
    model = _model_snapshot(attributes, operation)
    usage = _usage(attributes)
    failure = _span_failure(raw, attributes)
    return TraceSpan(
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=normalize_durable_text(name),
        started_at=started_at,
        ended_at=ended_at,
        attributes=attributes,
        model=model,
        usage=usage,
        failure=failure,
    )


def decode_otlp_attributes(raw: JsonValue, *, label: str = "OTLP attributes") -> JsonObject:
    """Decode one OTLP key/value attribute array into a canonical attribute mapping.

    Args:
        raw: OTLP ``attributes`` array, or ``None`` for a span that declares none.
        label: Source label used in validation messages.

    Returns:
        Decoded attribute mapping keyed by the exact source attribute names.

    Raises:
        OtlpTraceFormatError: The array, an entry, or an ``AnyValue`` is malformed.
    """
    return _decode_attributes(raw, label=label)


def _decode_attributes(raw: JsonValue, *, label: str) -> JsonObject:
    """Decode OpenTelemetry key/value attributes into JSON values."""
    if raw is None:
        return {}
    if not isinstance(raw, list):
        raise OtlpTraceFormatError(f"{label} must be an array")
    attributes: JsonObject = {}
    for item in raw:
        if not isinstance(item, dict):
            raise OtlpTraceFormatError(f"{label} entries must be objects")
        key = item.get("key")
        if not isinstance(key, str) or not key:
            raise OtlpTraceFormatError(f"{label} entries need non-empty keys")
        value = _decode_attribute_value(item.get("value"))
        if key in attributes and attributes[key] != value:
            raise OtlpTraceFormatError(f"{label} repeat {key!r} with conflicting values")
        attributes[key] = value
    return attributes


def _decode_attribute_value(value: JsonValue) -> JsonValue:
    """Decode one OTLP AnyValue object without coercing unsupported types."""
    if not isinstance(value, dict):
        raise OtlpTraceFormatError("OTLP attribute value must be an AnyValue object")
    scalar_keys = ("stringValue", "boolValue", "intValue", "doubleValue", "bytesValue")
    found = [key for key in scalar_keys if key in value]
    if "arrayValue" in value:
        found.append("arrayValue")
    if "kvlistValue" in value:
        found.append("kvlistValue")
    if len(found) != 1:
        raise OtlpTraceFormatError("OTLP AnyValue must contain exactly one supported value")
    key = found[0]
    if key == "arrayValue":
        array = value["arrayValue"]
        if not isinstance(array, dict) or not isinstance(array.get("values"), list):
            raise OtlpTraceFormatError("OTLP arrayValue must contain a values array")
        return [_decode_attribute_value(item) for item in array["values"]]
    if key == "kvlistValue":
        key_values = value["kvlistValue"]
        if not isinstance(key_values, dict) or not isinstance(key_values.get("values"), list):
            raise OtlpTraceFormatError("OTLP kvlistValue must contain a values array")
        return _decode_attributes(key_values["values"], label="OTLP kvlistValue")
    raw_value = value[key]
    if key == "intValue":
        return _integer(raw_value, "OTLP intValue")
    if key == "doubleValue":
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise OtlpTraceFormatError("OTLP doubleValue must be numeric")
        return float(raw_value)
    if key == "boolValue":
        if not isinstance(raw_value, bool):
            raise OtlpTraceFormatError("OTLP boolValue must be boolean")
        return raw_value
    if not isinstance(raw_value, str):
        raise OtlpTraceFormatError(f"OTLP {key} must be text")
    return normalize_durable_text(raw_value)


def _is_genai_span(attributes: JsonObject) -> bool:
    """Return whether one span declares GenAI semantics rather than an unrelated root span."""
    return any(key.startswith("gen_ai.") for key in attributes)


def _model_snapshot(attributes: JsonObject, operation: str) -> ModelSnapshot | None:
    """Build resolved model evidence for GenAI model operations."""
    if operation not in _MODEL_OPERATIONS:
        return None
    model_id = _first_text(attributes, ("gen_ai.response.model", "gen_ai.request.model"))
    provider = _first_text(attributes, ("gen_ai.provider.name", "gen_ai.system"))
    if model_id is None or provider is None:
        if operation == "invoke_agent" and model_id is None and provider is None:
            return None
        raise OtlpTraceFormatError(
            "GenAI model span needs gen_ai.provider.name (or gen_ai.system) and a request or "
            "response model"
        )
    revision = _first_text(
        attributes, ("gen_ai.response.model.version", "gen_ai.request.model.version")
    )
    capabilities_sha256 = normalized_capabilities_sha256(
        attributes,
        provider,
        model_id,
        revision,
        error_type=OtlpTraceFormatError,
    )
    connection_sha256 = normalized_connection_sha256(
        attributes,
        provider,
        error_type=OtlpTraceFormatError,
    )
    return ModelSnapshot(
        provider=provider,
        model_id=model_id,
        revision=revision,
        billing_source=BillingSource.CUSTOMER_MANAGED,
        capabilities_sha256=capabilities_sha256,
        connection_sha256=connection_sha256,
    )


def _usage(attributes: JsonObject) -> Usage | None:
    """Read complete GenAI token usage or reject a partial accounting record."""
    input_tokens = attributes.get("gen_ai.usage.input_tokens")
    output_tokens = attributes.get("gen_ai.usage.output_tokens")
    if input_tokens is None and output_tokens is None:
        return None
    if input_tokens is None or output_tokens is None:
        raise OtlpTraceFormatError("GenAI usage needs both input and output token counts")
    cached = attributes.get("gen_ai.usage.cached_input_tokens")
    return Usage(
        input_tokens=_integer(input_tokens, "gen_ai.usage.input_tokens"),
        output_tokens=_integer(output_tokens, "gen_ai.usage.output_tokens"),
        cached_input_tokens=None
        if cached is None
        else _integer(cached, "gen_ai.usage.cached_input_tokens"),
    )


def _span_failure(raw: JsonObject, attributes: JsonObject) -> StructuredFailure | None:
    """Map an OTLP error status to structured, non-secret span failure evidence."""
    status = raw.get("status")
    status_code = status.get("code") if isinstance(status, dict) else None
    is_error = status_code == 2 or status_code == "STATUS_CODE_ERROR" or status_code == "ERROR"
    if not is_error:
        return None
    message: str = "OpenTelemetry span reported an error"
    if isinstance(status, dict):
        status_message = status.get("message")
        if isinstance(status_message, str) and status_message:
            message = status_message
    if message == "OpenTelemetry span reported an error":
        attribute_message = attributes.get("error.message")
        if isinstance(attribute_message, str) and attribute_message:
            message = attribute_message
    return StructuredFailure(code=FailureCode.INTERNAL, message=normalize_durable_text(message))


def _validate_tool_pairs(spans: Sequence[TraceSpan]) -> None:
    """Reject incomplete, ambiguous, or causally inconsistent explicit tool-call pairs."""
    calls: dict[str, TraceSpan] = {}
    results: dict[str, TraceSpan] = {}
    for span in spans:
        call_id = span.attributes.get("gen_ai.tool.call.id")
        if call_id is None:
            continue
        if not isinstance(call_id, str) or not call_id:
            raise OtlpTraceFormatError("gen_ai.tool.call.id must be non-empty text")
        operation = span.attributes.get("gen_ai.operation.name")
        target = results if operation == _TOOL_OPERATION else calls
        if call_id in target:
            raise OtlpTraceFormatError(f"tool call ID {call_id!r} appears more than once")
        target[call_id] = span
    if set(calls) != set(results):
        missing_results = sorted(set(calls).difference(results))
        missing_calls = sorted(set(results).difference(calls))
        pieces = []
        if missing_results:
            pieces.append(f"missing results for {missing_results[:3]}")
        if missing_calls:
            pieces.append(f"missing calls for {missing_calls[:3]}")
        raise OtlpTraceFormatError("unpaired explicit tool call IDs: " + ", ".join(pieces))
    for call_id in sorted(calls):
        _validate_tool_pair(call_id, calls[call_id], results[call_id])


def _validate_tool_pair(call_id: str, call: TraceSpan, result: TraceSpan) -> None:
    """Validate one explicit tool call and result as one causal, non-contradictory pair."""
    call_name = _required_text(
        call.attributes.get("gen_ai.tool.name"),
        f"tool call {call_id!r} name",
    )
    result_name = _required_text(
        result.attributes.get("gen_ai.tool.name"),
        f"tool result {call_id!r} name",
    )
    if call_name != result_name:
        raise OtlpTraceFormatError(
            f"tool result {call_id!r} names {result_name!r}, not paired call name {call_name!r}"
        )
    if result.started_at < call.ended_at:
        raise OtlpTraceFormatError(f"tool result {call_id!r} starts before paired call completes")
    if call.parent_span_id == result.span_id:
        raise OtlpTraceFormatError(f"tool call {call_id!r} cannot name its paired result as parent")
    if result.parent_span_id is not None and result.parent_span_id != call.span_id:
        raise OtlpTraceFormatError(
            f"tool result {call_id!r} parent contradicts paired call span {call.span_id!r}"
        )


def _initial_request(
    attributes_by_span: Iterable[JsonObject],
) -> tuple[str, JsonObject] | None:
    """Return the chronologically first request text and only its visible evidence."""
    for attributes in attributes_by_span:
        task = _task_from_attributes(attributes)
        if task is not None:
            return task, attributes
    return None


def _task_from_attributes(attributes: JsonObject) -> str | None:
    """Extract one user-visible request from standard GenAI message attributes."""
    messages = json_value(attributes.get("gen_ai.input.messages"))
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role not in {"user", "human"}:
                continue
            text = _message_content(message.get("content"))
            if text:
                return text
    prompt = attributes.get("gen_ai.prompt")
    if isinstance(prompt, str) and prompt.strip():
        return normalize_durable_text(prompt.strip())
    return None


def _message_content(value: JsonValue) -> str | None:
    """Read plain text from a GenAI message content value."""
    if isinstance(value, str) and value.strip():
        return normalize_durable_text(value.strip())
    if isinstance(value, list):
        texts = [
            item["text"].strip()
            for item in value
            if isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
            and item["text"].strip()
        ]
        if texts:
            return normalize_durable_text("\n".join(texts))
    return None


def _first_text(attributes: JsonObject, keys: tuple[str, ...]) -> str | None:
    """Return the first non-empty text attribute from an ordered key list."""
    for key in keys:
        value = attributes.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_durable_text(value.strip())
    return None


def _required_text(value: JsonValue | None, label: str) -> str:
    """Require a durable non-empty string field."""
    return required_text(value, label, error_type=OtlpTraceFormatError)


def _validate_w3c_id(value: str, *, kind: str) -> str:
    """Validate one lowercase, non-zero W3C trace or span identifier."""
    pattern = _TRACE_ID_PATTERN if kind == "trace" else _SPAN_ID_PATTERN
    if not pattern.fullmatch(value) or set(value) == {"0"}:
        expected = "32" if kind == "trace" else "16"
        raise OtlpTraceFormatError(f"{kind} ID must be a non-zero lowercase {expected}-hex W3C ID")
    return value


def _timestamp(value: JsonValue | None, label: str) -> datetime:
    """Convert integer epoch nanoseconds to an aware UTC timestamp."""
    nanoseconds = _integer(value, label)
    if nanoseconds <= 0:
        raise OtlpTraceFormatError(f"{label} must be a positive epoch-nanosecond value")
    seconds, remainder = divmod(nanoseconds, 1_000_000_000)
    try:
        return datetime.fromtimestamp(seconds, UTC) + timedelta(microseconds=remainder // 1_000)
    except (OverflowError, OSError, ValueError) as exc:
        raise OtlpTraceFormatError(f"{label} is outside the supported timestamp range") from exc


def _integer(value: JsonValue | None, label: str) -> int:
    """Read a non-boolean integral JSON value without silent numeric conversion."""
    if isinstance(value, bool):
        raise OtlpTraceFormatError(f"{label} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    raise OtlpTraceFormatError(f"{label} must be an integer")
