"""OpenCode-style ``cache_control`` normalization for the Chat surface.

The @ai-sdk stack attaches Anthropic-style ephemeral cache hints to recent
messages for Claude-family model ids. Placements are classified in
``CHAT_CACHE_CONTROL_PLACEMENTS``: message-level and text-part hints are
validated and dropped here before official validation, while a hint inside a
``tool_calls`` entry is carried by the request decoder onto the canonical
tool call for the one wire that honors it.
"""

from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError, field_validator

from exp.common.core.artifacts import JsonObject
from exp.runtime.openai_protocol.errors import invalid_field

_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})


class _HintWireModel(BaseModel):
    """Strict private wire model rejecting unknown nested fields."""

    model_config = ConfigDict(extra="forbid")


class EphemeralCacheControl(_HintWireModel):
    """OpenCode/Anthropic cache breakpoint accepted only so it can be dropped.

    The object form is ``{"type": "ephemeral"}`` with an optional ``ttl`` of
    ``5m`` or ``1h``. An explicit ``ttl: null`` is not in that allowlist.
    """

    type: Literal["ephemeral"]
    ttl: Literal["5m", "1h"] | None = None

    @field_validator("ttl", mode="before")
    @classmethod
    def _reject_null_ttl(cls, value: object) -> object:
        """Reject an explicit null TTL while still allowing the key to be omitted."""
        if value is None:
            raise ValueError("ttl must be 5m or 1h when present")
        return value


def drop_opencode_cache_control(payload: JsonObject) -> JsonObject:
    """Remove supported OpenCode ``cache_control`` annotations from Chat messages.

    Args:
        payload: Parsed Chat Completions body.

    Returns:
        The original payload, or a shallow copy whose messages no longer carry
        a supported ``cache_control`` annotation.

    Raises:
        OpenAIProtocolError: A ``cache_control`` value is malformed or unsupported.
    """
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        return payload
    cleaned_messages: list[JsonValue] = []
    changed = False
    for index, raw_message in enumerate(raw_messages):
        message, message_changed = _without_message_hint(raw_message, index)
        cleaned_messages.append(message)
        changed = changed or message_changed
    if not changed:
        return payload
    cleaned_payload = dict(payload)
    cleaned_payload["messages"] = cleaned_messages
    return cleaned_payload


def _without_message_hint(raw_message: JsonValue, index: int) -> tuple[JsonValue, bool]:
    """Drop a supported ``cache_control`` annotation from one Chat message.

    Args:
        raw_message: One ``messages`` entry.
        index: Zero-based message index used in public error paths.

    Returns:
        The message (copied when an annotation is removed) and whether it changed.

    Raises:
        OpenAIProtocolError: The annotation is present but not a supported form.
    """
    if not isinstance(raw_message, dict):
        return raw_message, False
    message = cast(JsonObject, raw_message)
    changed = False
    if "cache_control" in message:
        require_supported_cache_control(message["cache_control"], f"messages.{index}.cache_control")
        message = {key: value for key, value in message.items() if key != "cache_control"}
        changed = True
    content = message.get("content")
    if isinstance(content, list):
        cleaned_content, content_changed = _without_text_part_hint(
            cast(list[JsonValue], content), index
        )
        if content_changed:
            if not changed:
                message = dict(message)
            message["content"] = cleaned_content
            changed = True
    return message, changed


def _without_text_part_hint(
    parts: list[JsonValue], message_index: int
) -> tuple[list[JsonValue], bool]:
    """Drop supported ``cache_control`` from OpenCode text content parts.

    Args:
        parts: Message ``content`` array.
        message_index: Zero-based parent message index used in public error paths.

    Returns:
        The content array (copied when an annotation is removed) and whether it changed.

    Raises:
        OpenAIProtocolError: A text-part annotation is present but not a supported form.
    """
    cleaned: list[JsonValue] = []
    changed = False
    for part_index, raw_part in enumerate(parts):
        if not isinstance(raw_part, dict) or "cache_control" not in raw_part:
            cleaned.append(raw_part)
            continue
        part = cast(JsonObject, raw_part)
        if part.get("type") not in _TEXT_PART_TYPES:
            cleaned.append(raw_part)
            continue
        require_supported_cache_control(
            part["cache_control"],
            f"messages.{message_index}.content.{part_index}.cache_control",
        )
        cleaned.append({key: value for key, value in part.items() if key != "cache_control"})
        changed = True
    return (cleaned, True) if changed else (parts, False)


def require_supported_cache_control(value: JsonValue, param: str) -> None:
    """Accept null or a supported ephemeral ``cache_control`` object.

    Args:
        value: Raw ``cache_control`` annotation.
        param: Public dotted field path used in the error.

    Raises:
        OpenAIProtocolError: The annotation is malformed or unsupported.
    """
    if value is None:
        return
    try:
        EphemeralCacheControl.model_validate(value)
    except ValidationError as exc:
        raise invalid_field(param) from exc
