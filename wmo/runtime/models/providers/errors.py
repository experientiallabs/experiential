"""Provider-neutral errors and shared wire validation for completed provider responses."""

from __future__ import annotations

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject


class ProviderResponseError(ValueError):
    """A provider returned a completed response that violates WMO's typed contract."""


def require_array(value: JsonValue | None, label: str) -> list[JsonValue]:
    """Return a response JSON array or raise a focused conversion error.

    Args:
        value: Decoded response value to validate.
        label: Provider-prefixed wire location used in the error message.

    Returns:
        The value unchanged when it is a JSON array.

    Raises:
        ProviderResponseError: The value is not a JSON array.
    """
    if not isinstance(value, list):
        raise ProviderResponseError(f"{label} must be an array")
    return value


def require_object(value: JsonValue | None, label: str) -> JsonObject:
    """Return a response JSON object or raise a focused conversion error.

    Args:
        value: Decoded response value to validate.
        label: Provider-prefixed wire location used in the error message.

    Returns:
        The value unchanged when it is a JSON object.

    Raises:
        ProviderResponseError: The value is not a JSON object.
    """
    if not isinstance(value, dict):
        raise ProviderResponseError(f"{label} must be an object")
    return value


def require_string(value: JsonValue | None, label: str) -> str:
    """Return a required non-empty response string or raise a focused conversion error.

    Args:
        value: Decoded response value to validate.
        label: Provider-prefixed wire location used in the error message.

    Returns:
        The value unchanged when it is a non-empty string.

    Raises:
        ProviderResponseError: The value is not a non-empty string.
    """
    if not isinstance(value, str) or not value:
        raise ProviderResponseError(f"{label} must be a non-empty string")
    return value


def require_integer(value: JsonValue | None, label: str, *, default: int = 0) -> int:
    """Read an optional non-negative response integer or the documented omitted default.

    Args:
        value: Decoded response value to validate.
        label: Provider-prefixed wire location used in the error message.
        default: Value substituted when the field is absent.

    Returns:
        The value when it is a non-negative integer, or the default when absent.

    Raises:
        ProviderResponseError: The value is present but not a non-negative integer.
    """
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProviderResponseError(f"{label} must be a non-negative integer")
    return value
