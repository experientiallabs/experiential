"""Normalize exported OpenTelemetry GenAI spans that are not wrapped in an OTLP envelope.

Collectors, file exporters, and SDK dumps publish the same GenAI semantic-convention spans in a
flatter shape than the OTLP JSON envelope: one span object per record, with identity in
``trace_id`` and ``span_id`` (or a nested ``context``), timing in ``start_time`` and ``end_time``,
and attributes as a plain JSON mapping instead of an ``AnyValue`` array.

This module reads those records, re-encodes each one as an OTLP span, and hands the result to the
canonical OTLP normalizer, so GenAI semantic-convention interpretation keeps exactly one owner.
Records that already arrive as an OTLP envelope are passed through untouched.

Identity is preserved when a record declares W3C-shaped hexadecimal identifiers. An opaque
identifier is mapped deterministically and the exact source identity is retained on the span as
``wmo.source.trace.id`` and ``wmo.source.span.id``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject, SourceIdentity
from wmo.simulation.ingest.otlp import (
    GENAI_SEMANTIC_CONVENTION_VERSION,
    OtlpTraceFormatError,
    TraceNormalizationIssue,
    TraceNormalizationResult,
    normalize_otlp_payloads,
)
from wmo.simulation.ingest.vendor_records import (
    VendorTraceFormatError,
    dotted_lookup,
    first_text,
    flatten_records,
    read_vendor_export,
    required_text,
    source_timestamp,
    vendor_w3c_id,
)

VENDOR = "otel-genai"

_SOURCE_TRACE_KEY = "wmo.source.trace.id"
_SOURCE_SPAN_KEY = "wmo.source.span.id"
_WRAPPER_KEYS = ("spans", "data", "results", "items", "resource_spans")
_RECORD_KEYS = ("trace_id", "traceId", "context", "span_id", "spanId")
_TRACE_ID_KEYS = ("trace_id", "traceId", "context.trace_id")
_SPAN_ID_KEYS = ("span_id", "spanId", "context.span_id")
_PARENT_ID_KEYS = ("parent_id", "parentId", "parent_span_id", "parentSpanId")
_START_KEYS = ("start_time", "startTime", "startTimeUnixNano", "start_time_unix_nano")
_END_KEYS = ("end_time", "endTime", "endTimeUnixNano", "end_time_unix_nano")


def load_otel_genai_file(
    path: Path,
    *,
    semantic_convention_version: str = GENAI_SEMANTIC_CONVENTION_VERSION,
    source_id: str | None = None,
) -> TraceNormalizationResult:
    """Read exported OpenTelemetry GenAI spans into canonical trace evidence.

    Args:
        path: Span array, span envelope, OTLP envelope, or JSONL export.
        semantic_convention_version: Pinned GenAI semantic-convention version for the traces.
        source_id: Optional durable source label. The local path is used when omitted.

    Returns:
        Canonical traces and every retained parse or validation exclusion.

    Raises:
        VendorTraceFormatError: The export cannot be read or decoded.
    """
    export = read_vendor_export(path, vendor=VENDOR, source_id=source_id)
    return normalize_otel_genai_payloads(
        export.payloads,
        source=export.source,
        semantic_convention_version=semantic_convention_version,
        initial_issues=export.issues,
    )


def normalize_otel_genai_payloads(
    payloads: Sequence[JsonValue],
    *,
    source: SourceIdentity,
    semantic_convention_version: str = GENAI_SEMANTIC_CONVENTION_VERSION,
    initial_issues: Sequence[TraceNormalizationIssue] = (),
) -> TraceNormalizationResult:
    """Normalize decoded GenAI span documents through the canonical OTLP normalizer.

    Args:
        payloads: Decoded span documents in source order.
        source: Immutable identity of the source bytes or transport result.
        semantic_convention_version: Pinned GenAI semantic-convention version for the traces.
        initial_issues: Parse exclusions collected before span mapping.

    Returns:
        Canonical traces and every retained validation exclusion.
    """
    issues = list(initial_issues)
    envelopes: list[JsonValue] = []
    spans: list[JsonValue] = []
    for index, payload in enumerate(payloads, start=1):
        if isinstance(payload, dict) and (
            "resourceSpans" in payload or "resource_spans" in payload
        ):
            envelopes.append(payload)
            continue
        try:
            records = flatten_records(
                payload,
                vendor=VENDOR,
                wrapper_keys=_WRAPPER_KEYS,
                record_keys=_RECORD_KEYS,
            )
        except VendorTraceFormatError as exc:
            issues.append(TraceNormalizationIssue(f"record-{index}", str(exc)))
            continue
        for record in records:
            try:
                spans.append(_otlp_span(record))
            except VendorTraceFormatError as exc:
                issues.append(TraceNormalizationIssue(f"record-{index}", str(exc)))
    if spans:
        envelopes.append({"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]})
    try:
        result = normalize_otlp_payloads(
            envelopes,
            source=source,
            semantic_convention_version=semantic_convention_version,
        )
    except OtlpTraceFormatError as exc:
        raise VendorTraceFormatError(str(exc)) from None
    return TraceNormalizationResult(
        traces=result.traces,
        issues=(*issues, *result.issues),
        identity_evidence=result.identity_evidence,
    )


def _otlp_span(record: JsonObject) -> JsonObject:
    """Re-encode one exported GenAI span record as an OTLP span object.

    Args:
        record: Exported span record with mapping attributes.

    Returns:
        OTLP span object with an ``AnyValue`` attribute array.

    Raises:
        VendorTraceFormatError: Identity, naming, or timing evidence is absent or malformed.
    """
    source_trace_id = required_text(dotted_lookup(record, _TRACE_ID_KEYS), "OTel GenAI trace_id")
    source_span_id = required_text(dotted_lookup(record, _SPAN_ID_KEYS), "OTel GenAI span_id")
    trace_id = vendor_w3c_id(source_trace_id, vendor=VENDOR, kind="trace", namespace="trace")
    span_id = vendor_w3c_id(source_span_id, vendor=VENDOR, kind="span", namespace="span")
    attributes = _attributes(record)
    attributes.setdefault(_SOURCE_TRACE_KEY, source_trace_id)
    attributes.setdefault(_SOURCE_SPAN_KEY, source_span_id)
    span: JsonObject = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": required_text(record.get("name"), "OTel GenAI span name"),
        "startTimeUnixNano": _unix_nano(
            dotted_lookup(record, _START_KEYS), "OTel GenAI start_time"
        ),
        "endTimeUnixNano": _unix_nano(dotted_lookup(record, _END_KEYS), "OTel GenAI end_time"),
        "attributes": _any_value_array(attributes),
    }
    parent = dotted_lookup(record, _PARENT_ID_KEYS)
    if isinstance(parent, str) and parent.strip():
        span["parentSpanId"] = vendor_w3c_id(
            parent.strip(), vendor=VENDOR, kind="span", namespace="span"
        )
    status = _status(record)
    if status is not None:
        span["status"] = status
    return span


def _attributes(record: JsonObject) -> JsonObject:
    """Read the declared attribute mapping of one exported GenAI span record.

    Args:
        record: Exported span record.

    Returns:
        Mutable copy of the declared attributes.

    Raises:
        VendorTraceFormatError: The record declares attributes that are not a JSON object.
    """
    declared = record.get("attributes")
    if declared is None:
        return {}
    if not isinstance(declared, dict):
        raise VendorTraceFormatError("OTel GenAI span attributes must be a JSON object")
    return dict(declared)


def _status(record: JsonObject) -> JsonObject | None:
    """Read the declared span status of one exported GenAI span record.

    Args:
        record: Exported span record.

    Returns:
        OTLP status object, or ``None`` when the record declares no status.
    """
    declared = record.get("status")
    if isinstance(declared, dict):
        return dict(declared)
    code = first_text(record, ("status_code", "statusCode"))
    if code is None:
        return None
    status: JsonObject = {"code": code}
    message = first_text(record, ("status_message", "statusMessage"))
    if message is not None:
        status["message"] = message
    return status


def _unix_nano(value: JsonValue | None, label: str) -> str:
    """Encode one declared instant as OTLP epoch nanoseconds.

    Args:
        value: ISO-8601 text, epoch seconds, epoch milliseconds, or epoch nanoseconds.
        label: Field label used in the validation message.

    Returns:
        Epoch nanoseconds as decimal text.

    Raises:
        VendorTraceFormatError: The value is absent or is not a supported instant.
    """
    if isinstance(value, int) and not isinstance(value, bool) and value > 10**17:
        return str(value)
    if isinstance(value, str) and value.strip().isdigit() and len(value.strip()) >= 18:
        return value.strip()
    parsed: datetime = source_timestamp(value, label)
    return str(int(parsed.timestamp() * 1_000_000_000))


def _any_value_array(attributes: JsonObject) -> list[JsonValue]:
    """Encode an attribute mapping as an OTLP key/value array.

    Args:
        attributes: Attribute mapping keyed by semantic-convention names.

    Returns:
        OTLP attribute entries in sorted key order.
    """
    entries: list[JsonValue] = []
    for key in sorted(attributes):
        value = attributes[key]
        if value is None:
            continue
        entries.append({"key": key, "value": _any_value(value)})
    return entries


def _any_value(value: JsonValue) -> JsonObject:
    """Encode one JSON value as an OTLP ``AnyValue`` object.

    Args:
        value: Attribute value.

    Returns:
        OTLP ``AnyValue`` object preserving the declared JSON type.

    Raises:
        VendorTraceFormatError: The value is null, which OTLP cannot represent.
    """
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, list):
        return {"arrayValue": {"values": [_any_value(item) for item in value if item is not None]}}
    if isinstance(value, dict):
        return {
            "kvlistValue": {
                "values": [
                    {"key": key, "value": _any_value(entry)}
                    for key, entry in sorted(value.items())
                    if entry is not None
                ]
            }
        }
    raise VendorTraceFormatError("exported GenAI attribute values cannot be null")
