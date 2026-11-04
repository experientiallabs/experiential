"""Shallow validation of the Messages fields the gateway forwards byte-for-byte.

``context_management``, ``diagnostics`` and ``safeguards`` are evolving
Anthropic objects that Anthropic rungs receive verbatim, so each is checked
only for its outer JSON shape. The ``anthropic-beta`` header is partitioned
against the exact forwarding allowlist.
"""

from __future__ import annotations

from typing import cast

from exp.common.core.artifacts import JsonObject
from exp.runtime.anthropic_protocol.manifest import (
    MESSAGES_BETA_TOKENS_FIELD_BOUND,
    MESSAGES_BETA_TOKENS_FORWARDED,
)
from exp.runtime.openai_protocol.errors import invalid_field


def context_management(payload: JsonObject) -> JsonObject | None:
    """Validate the caller's context-editing config as an object, verbatim.

    The nested shape is an evolving Anthropic beta the gateway forwards
    byte-for-byte (with the required beta header), so validation is
    deliberately shallow: a closed model here would recreate the
    reject-what-real-clients-send incident class.

    Raises:
        OpenAIProtocolError: The field is present but not a JSON object.
    """
    value = payload.get("context_management")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise invalid_field("context_management", "context_management must be a JSON object.")
    return cast(JsonObject, value)


def diagnostics(payload: JsonObject) -> JsonObject | None:
    """Validate the caller's diagnostics correlation as an object, verbatim.

    Like ``context_management``, the nested shape is an evolving Anthropic
    beta the gateway forwards byte-for-byte (with the required beta
    header), so validation is deliberately shallow.

    Raises:
        OpenAIProtocolError: The field is present but not a JSON object.
    """
    value = payload.get("diagnostics")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise invalid_field("diagnostics", "diagnostics must be a JSON object.")
    return cast(JsonObject, value)


def safeguards(payload: JsonObject) -> tuple[JsonObject, ...] | None:
    """Validate Claude Code's auto-mode safeguards as an array of objects.

    Each entry is an evolving provider object forwarded byte-for-byte on
    Anthropic rungs, so validation stays shallow.

    Raises:
        OpenAIProtocolError: The field is present but not an array of objects.
    """
    value = payload.get("safeguards")
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(entry, dict) for entry in value):
        raise invalid_field("safeguards", "safeguards must be an array of JSON objects.")
    return tuple(cast(JsonObject, entry) for entry in value)


def beta_tokens(header: str | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition a caller ``anthropic-beta`` header into forward and drop sets.

    Forwarding is an exact allowlist (:data:`MESSAGES_BETA_TOKENS_FORWARDED`):
    a caller header is operator-trust surface and is never blind-forwarded.
    Dropped tokens are disclosed per token, never rejected, because the
    provider itself tolerates unknown beta tokens. Field-bound tokens
    (:data:`MESSAGES_BETA_TOKENS_FIELD_BOUND`) land in neither set: the
    gateway injects them itself beside their field.

    Args:
        header: Raw comma-separated header value, or ``None``.

    Returns:
        ``(forwarded_tokens, dropped_disclosures)`` in caller order, deduped.

    Raises:
        OpenAIProtocolError: The header carries a non-display-safe value.
    """
    if header is None:
        return (), ()
    if len(header) > 4_096 or any(ord(char) < 32 for char in header):
        raise invalid_field("anthropic-beta", "anthropic-beta must be a display-safe header value.")
    forwarded: list[str] = []
    dropped: list[str] = []
    for raw_token in header.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if token in MESSAGES_BETA_TOKENS_FIELD_BOUND:
            continue
        if token in MESSAGES_BETA_TOKENS_FORWARDED:
            if token not in forwarded:
                forwarded.append(token)
        else:
            disclosure = f"anthropic-beta.{token}"
            if disclosure not in dropped:
                dropped.append(disclosure)
    return tuple(forwarded), tuple(dropped)
