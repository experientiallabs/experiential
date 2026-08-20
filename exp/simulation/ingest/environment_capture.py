"""Canonicalize the owned environment-capture action and observation profile."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.simulation.ingest.json_strict import DuplicateJsonKeyError, reject_duplicate_json_keys

_TRACE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_ACTION_KEY_ORDER = (
    "gen_ai.operation.name",
    "gen_ai.request.model",
    "gen_ai.tool.name",
    "gen_ai.tool.call.arguments",
)
_FIRST_ACTION_KEY_ORDER = _ACTION_KEY_ORDER + ("gen_ai.prompt", "wmh.trace.metadata")
_RESULT_KEY_ORDER = (
    "gen_ai.operation.name",
    "gen_ai.tool.name",
    "gen_ai.tool.message",
)
_SPAN_KEYS = frozenset(
    {
        "traceId",
        "spanId",
        "parentSpanId",
        "name",
        "startTimeUnixNano",
        "endTimeUnixNano",
        "status",
        "attributes",
    }
)
_PROFILE_TIMESTAMP_OFFSET_NANOS = 1_000_000_000
_PROFILE_TIMESTAMP_SCALE = 1_000
_ACTION_NAME = "chat terminal"
_RESULT_NAME = "execute_tool terminal"
_BENCHMARK = "terminal-tasks"
_MODEL = "terminal-agent"
_TOOL = "bash"


def canonicalize_environment_capture_payloads(
    payloads: Sequence[JsonValue],
) -> tuple[JsonValue, ...] | None:
    """Return canonical spans when every payload matches the owned capture profile.

    Args:
        payloads: Decoded JSONL records in their original source order.

    Returns:
        Canonical direct-span payloads, or ``None`` when any record falls outside the profile.
    """
    spans = _direct_spans(payloads)
    if spans is None:
        return None
    groups = _contiguous_trace_groups(spans)
    if groups is None:
        return None
    canonical: list[JsonValue] = []
    for group in groups:
        converted = _canonical_trace(group)
        if converted is None:
            return None
        canonical.extend(converted)
    return tuple(canonical)


def _direct_spans(payloads: Sequence[JsonValue]) -> tuple[JsonObject, ...] | None:
    """Return exact direct-span objects without accepting nested OTLP envelopes.

    Args:
        payloads: Decoded JSON records offered to the profile.

    Returns:
        Direct span objects, or ``None`` when the input uses another OTLP shape.
    """
    if not payloads:
        return None
    spans: list[JsonObject] = []
    for payload in payloads:
        if not isinstance(payload, dict) or set(payload) != _SPAN_KEYS:
            return None
        spans.append(payload)
    return tuple(spans)


def _contiguous_trace_groups(
    spans: Sequence[JsonObject],
) -> tuple[tuple[JsonObject, ...], ...] | None:
    """Group source-contiguous spans while rejecting repeated trace blocks.

    Args:
        spans: Exact direct-span records in source order.

    Returns:
        Contiguous trace groups, or ``None`` for invalid or repeated identities.
    """
    groups: list[list[JsonObject]] = []
    seen: set[str] = set()
    current_trace_id: str | None = None
    for span in spans:
        trace_id = span.get("traceId")
        if not isinstance(trace_id, str) or not _valid_trace_id(trace_id):
            return None
        if trace_id != current_trace_id:
            if trace_id in seen:
                return None
            seen.add(trace_id)
            groups.append([])
            current_trace_id = trace_id
        groups[-1].append(span)
    return tuple(tuple(group) for group in groups)


def _canonical_trace(spans: Sequence[JsonObject]) -> tuple[JsonObject, ...] | None:
    """Validate and canonicalize one complete alternating action-result trace.

    Args:
        spans: Source-contiguous spans for one trace identity.

    Returns:
        Canonical spans, or ``None`` when the trace is incomplete or malformed.
    """
    if not spans or len(spans) % 2:
        return None
    trace_id = spans[0].get("traceId")
    if not isinstance(trace_id, str):
        return None
    canonical: list[JsonObject] = []
    for ordinal in range(len(spans) // 2):
        action = spans[ordinal * 2]
        result = spans[ordinal * 2 + 1]
        converted = _canonical_pair(action, result, trace_id=trace_id, ordinal=ordinal)
        if converted is None:
            return None
        canonical.extend(converted)
    return tuple(canonical)


def _canonical_pair(
    action: JsonObject,
    result: JsonObject,
    *,
    trace_id: str,
    ordinal: int,
) -> tuple[JsonObject, JsonObject] | None:
    """Validate one exact pair and return its canonical direct OTLP spans.

    Args:
        action: Captured terminal action span.
        result: Captured terminal observation span.
        trace_id: Shared validated trace identity.
        ordinal: Zero-based action and result pair position.

    Returns:
        Canonical action and result spans, or ``None`` for any profile mismatch.
    """
    if action.get("traceId") != trace_id or result.get("traceId") != trace_id:
        return None
    prefix = f"{trace_id[:12]}{ordinal:04x}"
    if action.get("spanId") != f"{prefix}a" or result.get("spanId") != f"{prefix}b":
        return None
    if action.get("parentSpanId") != "" or result.get("parentSpanId") != "":
        return None
    if action.get("name") != _ACTION_NAME or result.get("name") != _RESULT_NAME:
        return None
    if not _exact_timestamps(action, ordinal * 10, ordinal * 10 + 1):
        return None
    if not _exact_timestamps(result, ordinal * 10 + 2, ordinal * 10 + 3):
        return None
    if action.get("status") != {"code": "STATUS_CODE_OK"}:
        return None
    if result.get("status") not in (
        {"code": "STATUS_CODE_OK"},
        {"code": "STATUS_CODE_ERROR"},
    ):
        return None
    action_attributes = _string_attributes(action.get("attributes"))
    result_attributes = _string_attributes(result.get("attributes"))
    if action_attributes is None or result_attributes is None:
        return None
    expected_action_keys = _FIRST_ACTION_KEY_ORDER if ordinal == 0 else _ACTION_KEY_ORDER
    if tuple(action_attributes) != expected_action_keys:
        return None
    if tuple(result_attributes) != _RESULT_KEY_ORDER:
        return None
    if action_attributes["gen_ai.operation.name"] != "chat":
        return None
    if result_attributes["gen_ai.operation.name"] != "execute_tool":
        return None
    if action_attributes["gen_ai.request.model"] != _MODEL:
        return None
    tool_name = action_attributes["gen_ai.tool.name"]
    if tool_name != _TOOL or result_attributes["gen_ai.tool.name"] != tool_name:
        return None
    if not _command_arguments(action_attributes["gen_ai.tool.call.arguments"]):
        return None
    if ordinal == 0 and not _first_action_evidence(action_attributes):
        return None
    action_span_id = str(action["spanId"])[-16:]
    result_span_id = str(result["spanId"])[-16:]
    call_id = f"environment-capture-{trace_id}-{ordinal}"
    return (
        _canonical_span(
            action,
            span_id=action_span_id,
            parent_span_id=None,
            start=ordinal * 10,
            end=ordinal * 10 + 1,
            status_code=1,
            attributes=_canonical_action_attributes(
                action_attributes,
                trace_id=trace_id,
                call_id=call_id,
                first=ordinal == 0,
            ),
        ),
        _canonical_span(
            result,
            span_id=result_span_id,
            parent_span_id=action_span_id,
            start=ordinal * 10 + 2,
            end=ordinal * 10 + 3,
            status_code=2 if result["status"] == {"code": "STATUS_CODE_ERROR"} else 1,
            attributes=(
                _attribute("gen_ai.operation.name", "execute_tool"),
                _attribute("gen_ai.tool.name", result_attributes["gen_ai.tool.name"]),
                _attribute("gen_ai.tool.call.id", call_id),
                _attribute("gen_ai.tool.message", result_attributes["gen_ai.tool.message"]),
            ),
        ),
    )


def _canonical_span(
    raw: JsonObject,
    *,
    span_id: str,
    parent_span_id: str | None,
    start: int,
    end: int,
    status_code: int,
    attributes: Sequence[JsonObject],
) -> JsonObject:
    """Build one direct span with canonical identity, time, status, and attributes.

    Args:
        raw: Validated source span supplying trace identity and name.
        span_id: Canonical W3C span identity.
        parent_span_id: Canonical parent identity, when present.
        start: Relative source start ordinal.
        end: Relative source end ordinal.
        status_code: Canonical OTLP numeric status code.
        attributes: Canonical semantic attributes.

    Returns:
        Strict direct OTLP span accepted by the shared normalizer.
    """
    span: JsonObject = {
        "traceId": raw["traceId"],
        "spanId": span_id,
        "name": raw["name"],
        "startTimeUnixNano": _canonical_timestamp(start),
        "endTimeUnixNano": _canonical_timestamp(end),
        "status": {"code": status_code},
        "attributes": list(attributes),
    }
    if parent_span_id is not None:
        span["parentSpanId"] = parent_span_id
    return span


def _canonical_action_attributes(
    attributes: dict[str, str], *, trace_id: str, call_id: str, first: bool
) -> tuple[JsonObject, ...]:
    """Build provider-free invoke-agent attributes with stable episode lineage.

    Args:
        attributes: Validated source action attributes.
        trace_id: Source trace identity used as episode lineage.
        call_id: Deterministic action and result pairing identity.
        first: Whether this action owns initial prompt and metadata evidence.

    Returns:
        Canonical invoke-agent attributes in deterministic order.
    """
    result = [
        _attribute("gen_ai.operation.name", "invoke_agent"),
        _attribute("gen_ai.tool.name", attributes["gen_ai.tool.name"]),
        _attribute("gen_ai.tool.call.id", call_id),
        _attribute("gen_ai.tool.call.arguments", attributes["gen_ai.tool.call.arguments"]),
    ]
    if first:
        result.extend(
            (
                _attribute("gen_ai.prompt", attributes["gen_ai.prompt"]),
                _attribute("wmh.trace.metadata", attributes["wmh.trace.metadata"]),
                _attribute("exp.conversation.id", trace_id),
            )
        )
    return tuple(result)


def _string_attributes(value: JsonValue | None) -> dict[str, str] | None:
    """Decode an exact list of unique OTLP string attributes.

    Args:
        value: Candidate OTLP attribute-list value.

    Returns:
        Ordered decoded attributes, or ``None`` for any structural mismatch.
    """
    if not isinstance(value, list):
        return None
    decoded: dict[str, str] = {}
    for item in value:
        if not isinstance(item, dict) or set(item) != {"key", "value"}:
            return None
        key = item.get("key")
        wrapped = item.get("value")
        if (
            not isinstance(key, str)
            or key in decoded
            or not isinstance(wrapped, dict)
            or set(wrapped) != {"stringValue"}
            or not isinstance(wrapped.get("stringValue"), str)
        ):
            return None
        decoded[key] = wrapped["stringValue"]
    return decoded


def _first_action_evidence(attributes: dict[str, str]) -> bool:
    """Return whether the first action carries exact terminal evidence.

    Args:
        attributes: Validated first-action string attributes.

    Returns:
        Whether prompt and metadata satisfy the terminal profile.
    """
    if not attributes["gen_ai.prompt"].strip():
        return False
    try:
        metadata = json.loads(
            attributes["wmh.trace.metadata"], object_pairs_hook=reject_duplicate_json_keys
        )
    except (json.JSONDecodeError, DuplicateJsonKeyError):
        return False
    return (
        isinstance(metadata, dict)
        and set(metadata) == {"benchmark", "returncode", "task_category"}
        and metadata.get("benchmark") == _BENCHMARK
        and isinstance(metadata.get("task_category"), str)
        and bool(metadata["task_category"].strip())
        and type(metadata.get("returncode")) is int
    )


def _command_arguments(value: str) -> bool:
    """Return whether text contains the exact nonempty terminal command object.

    Args:
        value: JSON text from the captured tool-call arguments.

    Returns:
        Whether the decoded object contains only one nonempty command string.
    """
    try:
        decoded = json.loads(value, object_pairs_hook=reject_duplicate_json_keys)
    except (json.JSONDecodeError, DuplicateJsonKeyError):
        return False
    return (
        isinstance(decoded, dict)
        and set(decoded) == {"command"}
        and isinstance(decoded.get("command"), str)
        and bool(decoded["command"].strip())
    )


def _exact_timestamps(span: JsonObject, start: int, end: int) -> bool:
    """Return whether one span retains exact non-boolean ordinal timestamps.

    Args:
        span: Captured direct span.
        start: Required relative start ordinal.
        end: Required relative end ordinal.

    Returns:
        Whether both timestamp fields are exact integer matches.
    """
    raw_start = span.get("startTimeUnixNano")
    raw_end = span.get("endTimeUnixNano")
    return type(raw_start) is int and type(raw_end) is int and raw_start == start and raw_end == end


def _canonical_timestamp(value: int) -> int:
    """Map one relative ordinal to a positive, microsecond-distinct epoch value."""
    return _PROFILE_TIMESTAMP_OFFSET_NANOS + value * _PROFILE_TIMESTAMP_SCALE


def _valid_trace_id(value: str) -> bool:
    """Return whether one trace ID is a non-zero lowercase W3C identity."""
    return bool(_TRACE_ID_PATTERN.fullmatch(value)) and set(value) != {"0"}


def _attribute(key: str, value: str) -> JsonObject:
    """Encode one OTLP string attribute."""
    return {"key": key, "value": {"stringValue": value}}
