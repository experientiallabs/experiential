"""Fingerprint and budget the entire in-memory classifier subject, including private fields."""

from __future__ import annotations

from pydantic import BaseModel, JsonValue

from exp.common.core.artifacts import canonical_json_bytes
from exp.runtime.gateway.contracts import GatewayRequest


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
