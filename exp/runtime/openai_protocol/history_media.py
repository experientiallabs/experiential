"""Disclosed degrade of unservable inline media baked into a conversation history.

Chat Completions, Responses, and Anthropic Messages clients all resend their
whole conversation on every turn, so an inline image or file the gateway
cannot forward (larger than the gateway ceiling, not valid base64, or of a
media type no wire carries) fails every later turn identically once it sits
in history: a typed 400 there wedges the session for good, because the
caller cannot edit what its client replays. Such a part is therefore
replaced, before strict wire validation, by a text part stating what was
omitted and why, and the substitution is disclosed in ``ignored_parameters``.

The rule follows the tool-result image precedent in ``capability_policy``:
media in a tool output (a tool screenshot, a PDF a tool read) is history the
moment the tool returns it, so it degrades in every turn; media in any other
message degrades everywhere except the LATEST turn (the user messages since
the model last spoke). Those keep the fail-closed contract, since the caller can still resend it
differently: an oversized part gets a typed 400 that states the limit, and an
undecodable or unsupported one keeps the decoder's own typed rejection.

This module owns the shared rule (classification, placeholder text, the
limit-stating error) and the Chat and Responses walkers; the Anthropic
walker lives beside the Messages decoder.
"""

from __future__ import annotations

import binascii
import re
from typing import Final, Literal, cast

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import (
    DOCUMENT_MEDIA_TYPES,
    IMAGE_MEDIA_TYPES,
    MAXIMUM_DOCUMENT_BASE64_BYTES,
    MAXIMUM_DOCUMENT_BYTES,
    MAXIMUM_IMAGE_BASE64_BYTES,
    MAXIMUM_IMAGE_BYTES,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, invalid_field
from exp.runtime.openai_protocol.wire_models import HOSTED_TOOL_ITEM_TYPES_ASSISTANT

MediaKind = Literal["image", "document"]
"""Media kinds whose inline bytes this pass inspects."""

UnservableReason = Literal["oversized", "undecodable", "unsupported_media_type"]
"""Why an inline media part cannot be forwarded."""

_CEILINGS: dict[MediaKind, tuple[int, int]] = {
    "image": (MAXIMUM_IMAGE_BASE64_BYTES, MAXIMUM_IMAGE_BYTES),
    "document": (MAXIMUM_DOCUMENT_BASE64_BYTES, MAXIMUM_DOCUMENT_BYTES),
}
"""Per-kind ``(encoded character ceiling, decoded byte ceiling)``."""

_MEDIA_TYPES: dict[MediaKind, frozenset[str]] = {
    "image": frozenset(IMAGE_MEDIA_TYPES),
    "document": frozenset(DOCUMENT_MEDIA_TYPES),
}
"""Media types each kind may declare."""

_FORMATS: dict[MediaKind, str] = {
    "image": "a PNG, JPEG, GIF, or WebP image",
    "document": "a PDF",
}
"""Human description of the accepted formats of each kind."""

_DATA_URL = re.compile(
    r"^data:(?P<media_type>[\w.+-]+/[\w.+-]+)(?P<parameters>;[^,]*)?,(?P<data>.*)$",
    re.DOTALL,
)
"""The ``data:`` URL shape the canonical content constructors parse."""

OPENAI_CACHE_KEYS: Final = ("cache_control", "prompt_cache_breakpoint")
"""Cache-marker keys an OpenAI-surface placeholder inherits from the part it replaces."""

ANTHROPIC_CACHE_KEYS: Final = ("cache_control",)
"""Cache-marker keys a Messages placeholder inherits: its text block allows no other."""

_VALIDATION_CHUNK: Final = 1 << 20
"""Characters validated per step; a multiple of 4 so every inner chunk is whole quanta."""


def base64_decodable(data: str, *, chunk: int = _VALIDATION_CHUNK) -> bool:
    """Return ``base64.b64decode(data, validate=True)``'s verdict without decoding it whole.

    The canonical content parts decode every inline payload once already, so
    this pre-validation pass must not hold a second full decoded copy: it
    checks strictly, one bounded chunk at a time, in linear time. Padding is
    legal only at the very end, so an inner chunk ending in ``=`` (which
    validates on its own) is refused here exactly as the whole string is.

    Args:
        data: The base64 payload.
        chunk: Characters per step, a positive multiple of 4.

    Returns:
        Whether strict base64 decoding of ``data`` succeeds.
    """
    if not data.isascii():
        return False
    end = len(data)
    for start in range(0, end, chunk):
        piece = data[start : start + chunk]
        if start + chunk < end and piece.endswith("="):
            return False
        try:
            binascii.a2b_base64(piece.encode("ascii"), strict_mode=True)
        except binascii.Error:
            return False
    return True


def _mebibytes(size: int) -> int:
    """Return a byte ceiling in whole MiB."""
    return size // (1024 * 1024)


def inline_reason(
    kind: MediaKind, media_type: str | None, data: str, *, check_decodable: bool = True
) -> UnservableReason | None:
    """Return why inline base64 bytes cannot be forwarded, if they cannot.

    Decodability uses the canonical content parts' own rule, so bytes judged
    servable here are bytes the decoder accepts.

    Args:
        kind: The media kind the bytes claim to be.
        media_type: Declared media type, ``None`` when the carrier states none
            (a bare base64 file, which the canonical document treats as a PDF).
        data: The base64 payload, without any ``data:`` preamble.
        check_decodable: Whether to validate the base64 itself. A latest-turn
            part skips it: only its size is judged here, and the strict decoder
            owns its undecodable rejection.

    Returns:
        The reason the bytes cannot ride, or ``None`` when they can.
    """
    if media_type is not None and media_type.lower() not in _MEDIA_TYPES[kind]:
        return "unsupported_media_type"
    if len(data) > _CEILINGS[kind][0]:
        return "oversized"
    if check_decodable and not base64_decodable(data):
        return "undecodable"
    return None


def data_url_reason(
    kind: MediaKind, value: str, *, bare_base64: bool, check_decodable: bool = True
) -> UnservableReason | None:
    """Return why one caller URL or file value cannot be forwarded, if it cannot.

    Args:
        kind: The media kind of the carrying part.
        value: Caller ``image_url`` or ``file_data`` value.
        bare_base64: Whether a value that is not a ``data:`` URL is base64
            bytes (``file_data``) rather than a remote URL (``image_url``).
        check_decodable: Whether to validate the base64 itself
            (:func:`inline_reason`).

    Returns:
        The reason the inline bytes cannot ride, or ``None`` when they can
        or when the value is a remote URL the provider fetches itself. A
        ``data:`` value that does not parse is undecodable, never a remote URL.
    """
    match = _DATA_URL.match(value)
    if match is None:
        if bare_base64:
            return inline_reason(kind, None, value.strip(), check_decodable=check_decodable)
        if value[:5].lower() == "data:":
            return "undecodable" if check_decodable else None
        return None
    if match["media_type"].lower() not in _MEDIA_TYPES[kind]:
        return "unsupported_media_type"
    if ";base64" not in (match["parameters"] or ""):
        return "undecodable"
    return inline_reason(kind, None, match["data"].strip(), check_decodable=check_decodable)


def inline_characters(value: str) -> int:
    """Return the base64 length of one caller value, without a ``data:`` preamble."""
    match = _DATA_URL.match(value)
    return len((match["data"] if match is not None else value).strip())


def placeholder_text(kind: MediaKind, reason: UnservableReason) -> str:
    """Return the text left in place of an omitted history part.

    Args:
        kind: The omitted part's media kind.
        reason: Why its bytes could not ride.

    Returns:
        One bracketed line the model reads where the part was.
    """
    match reason:
        case "oversized":
            detail = f"it is larger than the gateway's {_mebibytes(_CEILINGS[kind][1])} MiB limit"
        case "undecodable":
            detail = "its data is not valid base64"
        case "unsupported_media_type":
            detail = f"it is not {_FORMATS[kind]}"
    return f"[{kind} omitted from conversation history: {detail}]"


def oversized_error(kind: MediaKind, param: str, characters: int) -> OpenAIProtocolError:
    """Build the typed 400 for an oversized part in the latest turn's user messages.

    Args:
        kind: The part's media kind.
        param: Public path of the field carrying the bytes.
        characters: The caller's base64 length.

    Returns:
        The invalid-parameter error stating the limit.
    """
    encoded, decoded = _CEILINGS[kind]
    return invalid_field(
        param,
        f"The {kind} at '{param}' is {characters:,} base64 characters; an inline {kind} may "
        f"carry at most {encoded:,} base64 characters ({_mebibytes(decoded)} MiB decoded). "
        f"Resize or compress the {kind} and resend it.",
    )


def placeholder_part(
    text_type: str,
    kind: MediaKind,
    reason: UnservableReason,
    part: JsonObject,
    *,
    cache_keys: tuple[str, ...] = OPENAI_CACHE_KEYS,
) -> JsonObject:
    """Build the text part that replaces one omitted media part.

    The caller's cache marker moves onto the placeholder, so a prompt-cache
    breakpoint placed on the media part keeps its position.

    Args:
        text_type: The text part tag valid at this position.
        kind: The omitted part's media kind.
        reason: Why its bytes could not ride.
        part: The caller's media part.
        cache_keys: The cache-marker keys the surface's text part accepts.

    Returns:
        The replacement text part.
    """
    replacement: JsonObject = {"type": text_type, "text": placeholder_text(kind, reason)}
    for key in cache_keys:
        if key in part:
            replacement[key] = part[key]
    return replacement


def _openai_media(part: JsonValue, param: str) -> tuple[MediaKind, str, str, bool] | None:
    """Return one OpenAI media part's kind, value, value path, and value kind.

    Args:
        part: Raw caller content part.
        param: Public path of the part.

    Returns:
        ``(kind, value, value_param, bare_base64)`` for an image or file part
        that carries a string value, else ``None``.
    """
    if not isinstance(part, dict):
        return None
    part_object = cast(JsonObject, part)
    part_type = part_object.get("type")
    nested_image = part_object.get("image_url")
    nested_file = part_object.get("file")
    if part_type == "image_url" and isinstance(nested_image, dict):
        value = cast(JsonObject, nested_image).get("url")
        return ("image", value, f"{param}.image_url.url", False) if isinstance(value, str) else None
    if part_type == "input_image" and isinstance(nested_image, str):
        return "image", nested_image, f"{param}.image_url", False
    if part_type == "file" and isinstance(nested_file, dict):
        value = cast(JsonObject, nested_file).get("file_data")
        return (
            ("document", value, f"{param}.file.file_data", True) if isinstance(value, str) else None
        )
    value = part_object.get("file_data")
    if part_type == "input_file" and isinstance(value, str):
        return "document", value, f"{param}.file_data", True
    return None


def _degrade_parts(
    parts: list[JsonValue],
    param: str,
    *,
    text_type: str,
    latest: bool,
    position: str,
    disclosures: dict[str, None],
) -> list[JsonValue] | None:
    """Degrade the unservable media in one ordered part list.

    Args:
        parts: Raw caller parts of one message or tool output.
        param: Public path of the list.
        text_type: The text part tag valid in this list.
        latest: Whether the list is a user message of the latest turn.
        position: Disclosure position class (``messages.content`` or a tool output).
        disclosures: Ordered disclosure set, extended in place.

    Returns:
        The rebuilt list, or ``None`` when nothing was replaced.

    Raises:
        OpenAIProtocolError: An oversized part sits in a latest-turn user message.
    """
    rebuilt: list[JsonValue] | None = None
    for index, part in enumerate(parts):
        media = _openai_media(part, f"{param}.{index}")
        if media is None:
            continue
        kind, value, value_param, bare_base64 = media
        reason = data_url_reason(kind, value, bare_base64=bare_base64, check_decodable=not latest)
        if reason is None:
            continue
        if latest:
            if reason == "oversized":
                raise oversized_error(kind, value_param, inline_characters(value))
            continue
        if rebuilt is None:
            rebuilt = list(parts)
        rebuilt[index] = placeholder_part(text_type, kind, reason, cast(JsonObject, part))
        disclosures[f"{position}.{kind}->placeholder({reason})"] = None
    return rebuilt


_MODEL_ITEM_TYPES: Final = frozenset(
    {"reasoning", "function_call", "custom_tool_call", *HOSTED_TOOL_ITEM_TYPES_ASSISTANT}
)
"""Responses item types the model authored, beside assistant-role messages."""


def _model_turn(item: JsonObject) -> bool:
    """Whether one history entry is model output: an assistant message or a model item."""
    return item.get("role") == "assistant" or item.get("type") in _MODEL_ITEM_TYPES


def latest_turn(items: list[JsonValue]) -> frozenset[int]:
    """Return the indexes of the user entries after the last model-output entry.

    Model output is an assistant message, a reasoning or function/custom tool
    call item, or any provider-authored hosted item (compaction and program
    items included).

    The latest turn is everything the caller added since the model last
    spoke, so consecutive user messages (an image, then its question) all
    stay strict, exactly as the document-count pass reads the turn.

    Args:
        items: Raw conversation entries, in order.

    Returns:
        The latest turn's user-entry indexes.
    """
    last_model = max(
        (index for index, item in enumerate(items) if isinstance(item, dict) and _model_turn(item)),
        default=-1,
    )
    return frozenset(
        index
        for index, item in enumerate(items)
        if index > last_model and isinstance(item, dict) and item.get("role") == "user"
    )


def degrade_chat_media(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Replace unservable inline media in a Chat Completions history.

    Tool messages degrade in every position; every other message degrades
    except the latest turn's user messages. The payload is never mutated.

    Args:
        payload: Parsed Chat Completions body, before strict wire validation.

    Returns:
        The payload to validate (the same object when nothing changed) and
        the ordered disclosures for ``ignored_parameters``.

    Raises:
        OpenAIProtocolError: An oversized image or file sits in the latest
            user message.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload, ()
    items = messages
    latest = latest_turn(items)
    disclosures: dict[str, None] = {}
    rebuilt: list[JsonValue] | None = None
    for index, message in enumerate(items):
        if not isinstance(message, dict):
            continue
        message_object = cast(JsonObject, message)
        content = message_object.get("content")
        if not isinstance(content, list):
            continue
        tool = message_object.get("role") == "tool"
        parts = _degrade_parts(
            content,
            f"messages.{index}.content",
            text_type="text",
            latest=index in latest and not tool,
            position="messages.tool.content" if tool else "messages.content",
            disclosures=disclosures,
        )
        if parts is not None:
            if rebuilt is None:
                rebuilt = list(items)
            rebuilt[index] = {**message_object, "content": parts}
    if rebuilt is None:
        return payload, ()
    return {**payload, "messages": rebuilt}, tuple(disclosures)


def degrade_responses_media(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Replace unservable inline media in a Responses input history.

    ``function_call_output`` parts degrade in every position; message items
    degrade except the latest turn's user messages. The payload is never mutated.

    Args:
        payload: Parsed Responses body, before strict wire validation.

    Returns:
        The payload to validate (the same object when nothing changed) and
        the ordered disclosures for ``ignored_parameters``.

    Raises:
        OpenAIProtocolError: An oversized image or file sits in the latest
            user message.
    """
    raw_input = payload.get("input")
    if not isinstance(raw_input, list):
        return payload, ()
    items = raw_input
    latest = latest_turn(items)
    disclosures: dict[str, None] = {}
    rebuilt: list[JsonValue] | None = None
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        item_object = cast(JsonObject, item)
        if item_object.get("type") == "function_call_output":
            field, text_type, position, is_latest = (
                "output",
                "input_text",
                "input.function_call_output.output",
                False,
            )
        elif "role" in item_object:
            assistant = item_object.get("role") == "assistant"
            field, text_type, position, is_latest = (
                "content",
                "output_text" if assistant else "input_text",
                "input.content",
                index in latest,
            )
        else:
            continue
        parts_value = item_object.get(field)
        if not isinstance(parts_value, list):
            continue
        parts = _degrade_parts(
            parts_value,
            f"input.{index}.{field}",
            text_type=text_type,
            latest=is_latest,
            position=position,
            disclosures=disclosures,
        )
        if parts is not None:
            if rebuilt is None:
                rebuilt = list(items)
            rebuilt[index] = {**item_object, field: parts}
    if rebuilt is None:
        return payload, ()
    return {**payload, "input": rebuilt}, tuple(disclosures)
