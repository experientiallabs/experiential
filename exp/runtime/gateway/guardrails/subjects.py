"""Fingerprint and budget the entire in-memory classifier subject, including private fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Literal

from pydantic import BaseModel, JsonValue

from exp.common.core.artifacts import canonical_json_bytes
from exp.runtime.gateway.contracts import GatewayRequest


@dataclass
class ObservedSubjects:
    """Keep content-free request-local identities and terminal coverage admission state.

    Attributes:
        fingerprints: Complete encoded subject hashes considered while admission is open.
        closed_reason: Content-free terminal admission reason, or None while open.
        _lock: Serialize terminal admission metadata when request callbacks overlap.
    """

    fingerprints: set[bytes] = field(default_factory=set)
    closed_reason: Literal["subject_oversized", "preparation_unavailable"] | None = None
    _lock: Lock = field(default_factory=Lock, repr=False)

    @property
    def closed(self) -> bool:
        """Report whether optional coverage ended without retaining the rejected input."""
        return self.closed_reason is not None

    def close(self, reason: Literal["subject_oversized", "preparation_unavailable"]) -> bool:
        """Claim the session's one terminal admission record without storing its subject."""
        with self._lock:
            if self.closed_reason is not None:
                return False
            self.closed_reason = reason
            return True


def observation_subject_exceeds(request: GatewayRequest, limit: int) -> bool:
    """Reject a definitely oversized subject using character counts without encoding text.

    Character counts are a lower bound on UTF-8 JSON bytes. This bounded walk
    stops as soon as the limit is exceeded; fitting subjects still receive exact
    canonical byte accounting before admission.
    """
    remaining = limit

    def visit(value: object) -> bool:
        """Consume a conservative size lower bound without copying string payloads."""
        nonlocal remaining
        if isinstance(value, str):
            remaining -= len(value) + 2
        elif isinstance(value, BaseModel):
            remaining -= 2
            for name in type(value).model_fields:
                if visit(name) or visit(getattr(value, name)):
                    return True
        elif isinstance(value, dict):
            remaining -= 2
            for key, item in value.items():
                if visit(key) or visit(item):
                    return True
        elif isinstance(value, (tuple, list)):
            remaining -= 2
            for item in value:
                if visit(item):
                    return True
        elif value is None:
            remaining -= 4
        elif isinstance(value, (bool, int, float)):
            remaining -= 1
        return remaining < 0

    return visit(request)


def observation_subject_bytes(request: GatewayRequest) -> bytes:
    """Encode every classifier-visible field for ephemeral deduplication and byte accounting.

    Provider serialization and replay hashes intentionally exclude some request fields.
    An observer receives the complete object, so neither narrower format can identify or
    budget its subject. These bytes must never be persisted or included in telemetry.
    """
    return canonical_json_bytes(_subject_value(request))


def _subject_value(value: object) -> JsonValue:
    """Read actual model fields recursively instead of applying wire serialization exclusions."""
    if isinstance(value, BaseModel):
        return {name: _subject_value(getattr(value, name)) for name in type(value).model_fields}
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("observation subject requires string object keys")
            result[key] = _subject_value(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_subject_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("observation subject contains a non-JSON value")
