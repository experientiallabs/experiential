"""Shared strict readers and value helpers for vendor trace exports.

Vendor observability products export agent runs as JSON documents, JSON arrays, or JSONL files
whose records are neither OpenTelemetry spans nor PostHog events. This module owns the transport
and value-reading rules those source families share: immutable source identity for the exact
bytes, explicit per-record parse exclusions instead of silent line skipping, deterministic
W3C-shaped identifiers for opaque vendor keys, and conservative text and timestamp readers.

Format knowledge stays in the per-vendor modules, and canonical span and trace construction stays
in :mod:`wmo.simulation.ingest.vendor_trace`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject, SourceIdentity, canonical_json_bytes
from wmo.common.core.text import normalize_durable_text
from wmo.simulation.ingest.otlp import TraceNormalizationIssue
from wmo.simulation.ingest.trace_extensions import json_value as json_value
from wmo.simulation.ingest.trace_extensions import required_text as _required_extension_text

_TRACE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_SPAN_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")
_USER_ROLES = frozenset({"user", "human"})


class VendorTraceFormatError(ValueError):
    """Raised when a vendor trace export cannot be decoded or validated."""


@dataclass(frozen=True)
class VendorExport:
    """One decoded vendor export with immutable source identity.

    Args:
        source: Immutable identity of the exact source bytes.
        payloads: Decoded JSON documents in source order.
        issues: Retained parse exclusions for records that could not be decoded.
    """

    source: SourceIdentity
    payloads: tuple[JsonValue, ...]
    issues: tuple[TraceNormalizationIssue, ...]


def read_vendor_export(
    path: Path,
    *,
    vendor: str,
    source_id: str | None = None,
    error_type: type[ValueError] = VendorTraceFormatError,
) -> VendorExport:
    """Read one vendor JSON or JSONL export without repairing malformed records.

    Args:
        path: Local vendor export file.
        vendor: Vendor label used in source identity and error messages.
        source_id: Optional durable source label. The local path is used when omitted.
        error_type: Error class raised for transport failures, for source families whose
            callers catch their own error type.

    Returns:
        Decoded payloads, retained parse exclusions, and immutable source identity.

    Raises:
        ValueError: The declared ``error_type`` when the file cannot be read, is not UTF-8,
            or contains no records.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise error_type(f"cannot read {vendor} export {path}: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise error_type(f"{vendor} export is not UTF-8: {path}") from exc
    source = SourceIdentity(
        kind="file",
        source_id=source_id or f"{vendor}:{path}",
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    try:
        document: JsonValue = json.loads(text)
    except json.JSONDecodeError:
        payloads, issues = _decode_jsonl(text, vendor=vendor, error_type=error_type)
        return VendorExport(source=source, payloads=payloads, issues=issues)
    return VendorExport(source=source, payloads=(document,), issues=())


def _decode_jsonl(
    text: str, *, vendor: str, error_type: type[ValueError]
) -> tuple[tuple[JsonValue, ...], tuple[TraceNormalizationIssue, ...]]:
    """Decode JSONL records and retain every malformed line as an explicit exclusion.

    Args:
        text: UTF-8 JSONL source text.
        vendor: Vendor label used in the no-record error message.
        error_type: Error class raised when the source declares no records.

    Returns:
        Decoded records in source order and retained parse exclusions.

    Raises:
        ValueError: The declared ``error_type`` when the source contains neither records
            nor parse issues.
    """
    payloads: list[JsonValue] = []
    issues: list[TraceNormalizationIssue] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payloads.append(json.loads(line))
        except json.JSONDecodeError as exc:
            issues.append(
                TraceNormalizationIssue(f"line-{line_number}", f"invalid JSONL record: {exc.msg}")
            )
    if not payloads and not issues:
        raise error_type(f"{vendor} export contains no records")
    return tuple(payloads), tuple(issues)


def flatten_records(
    payload: JsonValue,
    *,
    vendor: str,
    wrapper_keys: Sequence[str],
    record_keys: Sequence[str],
) -> tuple[JsonObject, ...]:
    """Flatten explicitly supported export wrappers into vendor record objects.

    Args:
        payload: One decoded export document, array, or record.
        vendor: Vendor label used in error messages.
        wrapper_keys: Envelope keys whose array values contain records or nested wrappers.
        record_keys: Keys whose presence identifies a bare vendor record.

    Returns:
        Vendor record objects in source order.

    Raises:
        VendorTraceFormatError: The payload is not a supported wrapper or record shape.
    """
    records: list[JsonObject] = []
    _append_records(
        payload,
        records,
        vendor=vendor,
        wrapper_keys=tuple(wrapper_keys),
        record_keys=tuple(record_keys),
    )
    return tuple(records)


def _append_records(
    payload: JsonValue,
    records: list[JsonObject],
    *,
    vendor: str,
    wrapper_keys: tuple[str, ...],
    record_keys: tuple[str, ...],
) -> None:
    """Append records from one payload node, recursing only through declared wrappers.

    Args:
        payload: Decoded export node.
        records: Accumulator for discovered record objects.
        vendor: Vendor label used in error messages.
        wrapper_keys: Envelope keys whose array values contain records or nested wrappers.
        record_keys: Keys whose presence identifies a bare vendor record.

    Raises:
        VendorTraceFormatError: The node is neither a declared wrapper nor a vendor record.
    """
    if isinstance(payload, list):
        for item in payload:
            _append_records(
                item,
                records,
                vendor=vendor,
                wrapper_keys=wrapper_keys,
                record_keys=record_keys,
            )
        return
    if not isinstance(payload, dict):
        raise VendorTraceFormatError(f"{vendor} exports must contain record objects")
    if any(key in payload for key in record_keys):
        records.append(payload)
        return
    for wrapper_key in wrapper_keys:
        wrapped = payload.get(wrapper_key)
        if isinstance(wrapped, list):
            _append_records(
                wrapped,
                records,
                vendor=vendor,
                wrapper_keys=wrapper_keys,
                record_keys=record_keys,
            )
            return
    expected = ", ".join((*record_keys, *wrapper_keys))
    raise VendorTraceFormatError(f"{vendor} record has none of the expected keys: {expected}")


def vendor_w3c_id(value: str, *, vendor: str, kind: str, namespace: str) -> str:
    """Preserve a valid W3C identifier or deterministically map an opaque vendor key.

    Args:
        value: Source trace or span identity from the vendor export.
        vendor: Vendor label included in the digest input.
        kind: ``trace`` for a 32-character identity, otherwise ``span``.
        namespace: Stable source-role namespace preventing cross-role collisions.

    Returns:
        Lowercase nonzero W3C-shaped trace or span identity.
    """
    normalized = value.casefold()
    pattern = _TRACE_ID_PATTERN if kind == "trace" else _SPAN_ID_PATTERN
    if pattern.fullmatch(normalized) and set(normalized) != {"0"}:
        return normalized
    width = 32 if kind == "trace" else 16
    digest = hashlib.sha256(f"{vendor}\0{namespace}\0{value}".encode()).hexdigest()
    return digest[:width]


def dotted_lookup(record: JsonObject, paths: Sequence[str]) -> JsonValue | None:
    """Read the first declared value among dotted record paths.

    Args:
        record: Source record whose keys may be flat dotted names or nested objects.
        paths: Dotted candidate paths in preference order.

    Returns:
        First declared value, or ``None`` when the record declares none.
    """
    for path in paths:
        if path in record:
            return record[path]
        node: JsonValue | None = record
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node is not None:
            return node
    return None


def required_text(value: JsonValue | None, label: str) -> str:
    """Require one non-empty durable source text value.

    Args:
        value: Raw source value.
        label: Field label used in the validation message.

    Returns:
        Normalized durable text.

    Raises:
        VendorTraceFormatError: The value is absent, not text, or blank.
    """
    return _required_extension_text(value, label, error_type=VendorTraceFormatError)


def first_text(record: JsonObject, keys: Sequence[str]) -> str | None:
    """Return the first non-empty text value from an ordered key list.

    Args:
        record: Source record or attribute mapping.
        keys: Ordered candidate keys.

    Returns:
        Normalized durable text, or ``None`` when no key holds text.
    """
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_durable_text(value.strip())
    return None


def json_text(value: JsonValue | None) -> str:
    """Render a source value as durable text or stable compact JSON.

    Args:
        value: Raw source value.

    Returns:
        Normalized text for string input, empty text for ``None``, canonical JSON otherwise.
    """
    if isinstance(value, str):
        return normalize_durable_text(value)
    if value is None:
        return ""
    return canonical_json_bytes(value).decode("utf-8")


def message_text(value: JsonValue | None) -> str:
    """Read plain text from a chat message content value.

    Args:
        value: Message content as text or as normalized multi-part content.

    Returns:
        Normalized text, empty when the content holds no readable text parts.
    """
    if isinstance(value, str):
        return normalize_durable_text(value)
    if not isinstance(value, list):
        return ""
    texts = [
        item["text"].strip()
        for item in value
        if isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
        and item["text"].strip()
    ]
    return normalize_durable_text("\n".join(texts))


def message_role(message: JsonObject) -> str:
    """Return a lowercase chat-message role from the supported role fields.

    Args:
        message: One chat or framework message object.

    Returns:
        Lowercase role name, empty when the message declares none.
    """
    for key in ("role", "type"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().casefold()
    return ""


def first_user_text(value: JsonValue | None) -> str | None:
    """Read the first user or human message text from a message list or plain input.

    Args:
        value: Message list, input object, or plain request text.

    Returns:
        Normalized request text, or ``None`` when the input holds none.
    """
    if isinstance(value, str) and value.strip():
        return normalize_durable_text(value.strip())
    if isinstance(value, list):
        for message in value:
            if not isinstance(message, dict) or message_role(message) not in _USER_ROLES:
                continue
            text = message_text(message.get("content"))
            if text:
                return text
        return None
    if isinstance(value, dict):
        for key in ("messages", "input", "prompt", "query", "question", "message"):
            nested = value.get(key)
            if nested is None:
                continue
            text = first_user_text(nested)
            if text is not None:
                return text
    return None


def source_timestamp(value: JsonValue | None, label: str) -> datetime:
    """Parse a timezone-aware source timestamp from ISO-8601 text or epoch numbers.

    Args:
        value: ISO-8601 text, epoch seconds, or epoch milliseconds.
        label: Field label used in the validation message.

    Returns:
        Timezone-aware UTC timestamp.

    Raises:
        VendorTraceFormatError: The value is absent or is not a supported timestamp.
    """
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise VendorTraceFormatError(f"{label} is not an ISO-8601 timestamp: {text!r}") from exc
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise VendorTraceFormatError(f"{label} must be an ISO-8601 or epoch timestamp")
    seconds = float(value)
    if seconds > 1e11:  # Vendors export either epoch seconds or epoch milliseconds.
        seconds /= 1000.0
    try:
        return datetime.fromtimestamp(seconds, UTC)
    except (OSError, OverflowError, ValueError) as exc:
        raise VendorTraceFormatError(f"{label} is outside the supported range") from exc


def source_interval(
    start: JsonValue | None,
    end: JsonValue | None,
    *,
    start_label: str,
    end_label: str,
) -> tuple[datetime, datetime]:
    """Read a declared source interval whose end instant is optional.

    Args:
        start: Declared start instant.
        end: Declared end instant, when any.
        start_label: Start field label used in the validation message.
        end_label: End field label used in the validation message.

    Returns:
        Source start and end instants, equal when no end is declared.

    Raises:
        VendorTraceFormatError: The start is absent or either value is not a timestamp.
    """
    started_at = source_timestamp(start, start_label)
    ended_at = source_timestamp(end, end_label) if end is not None else started_at
    return started_at, ended_at
