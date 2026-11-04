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
one. A top-level block in the LATEST turn (the user messages since the last
assistant message) keeps the fail-closed contract, since the caller can still
resend it differently: an oversized one gets a typed 400 that states the
limit, and an undecodable one keeps the decoder's own typed rejection.
"""

from __future__ import annotations

from typing import cast

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import DOCUMENT_MEDIA_TYPES, IMAGE_MEDIA_TYPES
from exp.runtime.openai_protocol.history_media import (
    ANTHROPIC_CACHE_KEYS,
    MediaKind,
    UnservableReason,
    inline_reason,
    latest_turn,
    oversized_error,
    placeholder_part,
)

_MEDIA_TYPES: dict[MediaKind, frozenset[str]] = {
    "image": frozenset(IMAGE_MEDIA_TYPES),
    "document": frozenset(DOCUMENT_MEDIA_TYPES),
}
"""Media types each kind may declare on a base64 source, matched exactly as
the Messages decoder matches them."""


def unservable_reason(
    kind: MediaKind, block: JsonObject, *, check_decodable: bool = True
) -> UnservableReason | None:
    """Return why one inline media block cannot be forwarded, if it cannot.

    Only a ``base64`` source is judged: URL, file, and plain-text sources
    carry no inline bytes, and a malformed shape is left to the strict wire
    models so it keeps its field-specific error. Decodability is checked with
    the canonical content part's own rule, so every block judged servable
    here is one the decoder accepts.

    Args:
        kind: The block's content type.
        block: Raw caller content block.
        check_decodable: Whether to validate the base64 itself; a latest-turn
            block skips it, because the strict decoder owns that rejection.

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
    return inline_reason(kind, None, data, check_decodable=check_decodable)


def _media_kind(block: JsonValue) -> MediaKind | None:
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
    latest = latest_turn(messages[:-1] if _ends_in_empty_prefill(messages) else messages)
    disclosures: dict[str, None] = {}
    rebuilt: list[JsonValue] | None = None
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        blocks: list[JsonValue] | None = None
        for block_index, block in enumerate(content):
            replaced = _degrade_block(
                block,
                f"messages.{message_index}.content.{block_index}",
                latest=message_index in latest,
                disclosures=disclosures,
            )
            if replaced is not None:
                if blocks is None:
                    blocks = list(content)
                blocks[block_index] = replaced
        if blocks is not None:
            if rebuilt is None:
                rebuilt = list(messages)
            rebuilt[message_index] = {**message, "content": blocks}
    if rebuilt is None:
        return payload, ()
    return {**payload, "messages": rebuilt}, tuple(disclosures)


def _ends_in_empty_prefill(messages: list[JsonValue]) -> bool:
    """Whether the last turn is an empty assistant prefill the decoder drops.

    Such a turn is not the model speaking, so the user turn before it is
    still the latest turn and stays strict. The rule mirrors the decoder's
    own: an empty string, or only empty text blocks with no cache marker and
    no citations, after at least one non-system turn. A turn the decoder
    keeps (and refuses) still counts as the model's.
    """
    last = messages[-1] if messages else None
    if not isinstance(last, dict) or last.get("role") != "assistant":
        return False
    if not any(isinstance(turn, dict) and turn.get("role") != "system" for turn in messages[:-1]):
        return False
    content = last.get("content")
    if isinstance(content, str):
        return not content
    return (
        isinstance(content, list)
        and bool(content)
        and all(
            isinstance(block, dict)
            and block.get("type") == "text"
            and not block.get("text")
            and block.get("cache_control") is None
            and not block.get("citations")
            for block in content
        )
    )


def _degrade_block(
    block: JsonValue, param: str, *, latest: bool, disclosures: dict[str, None]
) -> JsonObject | None:
    """Return one top-level user block's replacement, or ``None`` to keep it.

    Args:
        block: Raw caller content block.
        param: Public path of the block.
        latest: Whether the block's message belongs to the latest turn.
        disclosures: Ordered disclosure set, extended in place.

    Returns:
        A placeholder for unservable history media, a rebuilt tool result,
        or ``None`` when the block rides unchanged.

    Raises:
        OpenAIProtocolError: An oversized top-level block sits in the latest turn.
    """
    kind = _media_kind(block)
    if kind is None:
        return _degrade_tool_result(block, disclosures)
    block_object = cast(JsonObject, block)
    reason = unservable_reason(kind, block_object, check_decodable=not latest)
    if reason is None:
        return None
    if latest:
        if reason == "oversized":
            data = cast(str, cast(JsonObject, block_object["source"])["data"])
            raise oversized_error(kind, f"{param}.source.data", len(data))
        return None
    disclosures[f"messages.content.{kind}->placeholder({reason})"] = None
    return placeholder_part("text", kind, reason, block_object, cache_keys=ANTHROPIC_CACHE_KEYS)


def _degrade_tool_result(block: JsonValue, disclosures: dict[str, None]) -> JsonObject | None:
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
    inner = block.get("content")
    if block.get("type") != "tool_result" or not isinstance(inner, list):
        return None
    parts: list[JsonValue] | None = None
    for index, part in enumerate(inner):
        kind = _media_kind(part)
        if kind is None:
            continue
        part_object = cast(JsonObject, part)
        reason = unservable_reason(kind, part_object)
        if reason is None:
            continue
        if parts is None:
            parts = list(inner)
        parts[index] = placeholder_part(
            "text", kind, reason, part_object, cache_keys=ANTHROPIC_CACHE_KEYS
        )
        disclosures[f"messages.content.tool_result.{kind}->placeholder({reason})"] = None
    if parts is None:
        return None
    return {**block, "content": parts}
