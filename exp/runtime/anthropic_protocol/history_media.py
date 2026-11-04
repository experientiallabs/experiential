"""Disclosed degrade of unservable inline media baked into a Messages history.

An Anthropic Messages client resends its whole conversation on every turn,
so an inline image or PDF the gateway cannot forward (larger than the
gateway ceiling, not valid base64, or of a media type no wire carries) fails
every later turn identically once it sits in history: a typed 400 there
wedges the session for good, because the caller cannot edit what its client
replays. Such a block is therefore replaced, before strict wire validation,
by a text block stating what was omitted and why, and the substitution is
disclosed in ``ignored_parameters``.

The rule follows the tool-result image precedent in ``capability_policy``:
media inside a ``tool_result`` (a tool screenshot, a PDF a tool read) is
history the moment the tool returns it, so it degrades in every turn; a
top-level image or document degrades in every user turn before the latest
one. A top-level block in the LATEST user turn keeps the fail-closed
contract, since the caller can still resend it differently: an oversized one
gets a typed 400 that states the limit, and an undecodable one keeps the
decoder's own typed rejection.
"""

from __future__ import annotations

from typing import cast

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import DOCUMENT_MEDIA_TYPES, IMAGE_MEDIA_TYPES
from exp.runtime.openai_protocol.history_media import (
    MediaKind,
    UnservableReason,
    inline_reason,
    oversized_error,
    placeholder_part,
)

_MEDIA_TYPES: dict[MediaKind, frozenset[str]] = {
    "image": frozenset(IMAGE_MEDIA_TYPES),
    "document": frozenset(DOCUMENT_MEDIA_TYPES),
}
"""Media types each kind may declare on a base64 source, matched exactly as
the Messages decoder matches them."""


def unservable_reason(kind: MediaKind, block: JsonObject) -> UnservableReason | None:
    """Return why one inline media block cannot be forwarded, if it cannot.

    Only a ``base64`` source is judged: URL, file, and plain-text sources
    carry no inline bytes, and a malformed shape is left to the strict wire
    models so it keeps its field-specific error. Decodability is checked with
    the canonical content part's own rule, so every block judged servable
    here is one the decoder accepts.

    Args:
        kind: The block's content type.
        block: Raw caller content block.

    Returns:
        The reason the bytes cannot ride, or ``None`` when they can (or when
        the block is not an inline base64 source).
    """
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    source_object = cast(JsonObject, source)
    data = source_object.get("data")
    if source_object.get("type") != "base64" or not isinstance(data, str):
        return None
    if source_object.get("media_type") not in _MEDIA_TYPES[kind]:
        return "unsupported_media_type"
    return inline_reason(kind, None, data)


def _media_kind(block: object) -> MediaKind | None:
    """Return the media kind of one raw content block, if it is one."""
    if not isinstance(block, dict):
        return None
    block_type = cast(JsonObject, block).get("type")
    if block_type == "image":
        return "image"
    if block_type == "document":
        return "document"
    return None


def degrade_history_media(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Replace unservable inline media in conversation history with placeholders.

    The payload is never mutated: a changed message is rebuilt with a copy
    of its content list. Disclosures name the block position class and the
    reason once each, so a long history cannot grow the disclosure list.

    Args:
        payload: Parsed JSON request body, before strict wire validation.

    Returns:
        The payload to validate (the same object when nothing changed) and
        the ordered disclosures for ``ignored_parameters``.

    Raises:
        OpenAIProtocolError: A top-level image or document in the latest
            user turn exceeds its encoded size ceiling.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload, ()
    latest_user = max(
        (
            index
            for index, message in enumerate(messages)
            if isinstance(message, dict) and cast(JsonObject, message).get("role") == "user"
        ),
        default=-1,
    )
    disclosures: dict[str, None] = {}
    rebuilt: list[object] = list(messages)
    changed = False
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        message_object = cast(JsonObject, message)
        content = message_object.get("content")
        if message_object.get("role") != "user" or not isinstance(content, list):
            continue
        blocks: list[object] = list(cast(list[object], content))
        message_changed = False
        for block_index, block in enumerate(blocks):
            param = f"messages.{message_index}.content.{block_index}"
            kind = _media_kind(block)
            if kind is not None:
                block_object = cast(JsonObject, block)
                reason = unservable_reason(kind, block_object)
                if reason is None:
                    continue
                if message_index == latest_user:
                    if reason == "oversized":
                        data = cast(str, cast(JsonObject, block_object["source"])["data"])
                        raise oversized_error(kind, f"{param}.source.data", len(data))
                    continue
                blocks[block_index] = placeholder_part("text", kind, reason, block_object)
                disclosures[f"messages.content.{kind}->placeholder({reason})"] = None
                message_changed = True
                continue
            replaced = _degrade_tool_result(block, disclosures)
            if replaced is not None:
                blocks[block_index] = replaced
                message_changed = True
        if message_changed:
            rebuilt[message_index] = {**message_object, "content": blocks}
            changed = True
    if not changed:
        return payload, ()
    return {**payload, "messages": rebuilt}, tuple(disclosures)


def _degrade_tool_result(block: object, disclosures: dict[str, None]) -> JsonObject | None:
    """Degrade unservable media inside one ``tool_result`` block.

    Args:
        block: Raw caller content block.
        disclosures: Ordered disclosure set, extended in place.

    Returns:
        The rebuilt tool result, or ``None`` when the block is not a tool
        result or none of its media needed replacing.
    """
    if not isinstance(block, dict):
        return None
    block_object = cast(JsonObject, block)
    inner = block_object.get("content")
    if block_object.get("type") != "tool_result" or not isinstance(inner, list):
        return None
    parts: list[object] = list(cast(list[object], inner))
    changed = False
    for index, part in enumerate(parts):
        kind = _media_kind(part)
        if kind is None:
            continue
        reason = unservable_reason(kind, cast(JsonObject, part))
        if reason is None:
            continue
        parts[index] = placeholder_part("text", kind, reason, cast(JsonObject, part))
        disclosures[f"messages.content.tool_result.{kind}->placeholder({reason})"] = None
        changed = True
    if not changed:
        return None
    return {**block_object, "content": parts}
