"""Strict JSON object decoding that rejects duplicate keys."""

from __future__ import annotations

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject


class DuplicateJsonKeyError(ValueError):
    """Raised when a JSON object repeats a key."""


def reject_duplicate_json_keys(pairs: list[tuple[str, JsonValue]]) -> JsonObject:
    """Decode one JSON object while rejecting repeated keys.

    Args:
        pairs: Object members in their original source order.

    Returns:
        Decoded JSON object with unique keys.

    Raises:
        DuplicateJsonKeyError: A key occurs more than once in the object.
    """
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
