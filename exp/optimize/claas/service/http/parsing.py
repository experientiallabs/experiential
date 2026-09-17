"""Strict JSON object decoding for learner requests and nested function arguments."""

import json
import math

from pydantic import TypeAdapter

from exp.common.core.artifacts import JsonObject, JsonValue

_OBJECT = TypeAdapter(JsonObject)


def _object(pairs: list[tuple[str, JsonValue]]) -> JsonObject:
    """Reject duplicate keys before one input can silently replace another."""
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON objects must not contain duplicate keys")
        result[key] = value
    return result


def _constant(value: str) -> None:
    """Reject nonfinite JavaScript constants that are not legal JSON numbers."""
    raise ValueError("JSON numbers must be finite")


def _float(value: str) -> float:
    """Reject finite-looking exponent literals that overflow the numeric representation."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON numbers must be finite")
    return number


def parse_object(payload: bytes | bytearray | str) -> JsonObject:
    """Decode exactly one JSON object with finite values and no duplicate properties."""
    try:
        return _OBJECT.validate_python(
            json.loads(
                payload, object_pairs_hook=_object, parse_constant=_constant, parse_float=_float
            )
        )
    except RecursionError:
        raise ValueError("JSON nesting exceeds the supported request depth") from None
