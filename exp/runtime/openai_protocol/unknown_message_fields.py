"""Disclosed removal of unknown keys on conversation objects in a request body.

Chat apps and agent frameworks stamp their own bookkeeping on the messages
they keep and send the whole history back each turn: WorkBuddy tags every
message with ``conversationRequestId``, other clients add
``isInternalToolCall``, ``model_id``, ``message_id`` or ``max_tokens``. No
provider defines these keys, and a named 400 for one fails every later turn
of the conversation, because the client replays the same history on retry.

So a key the gateway's wire model does not define is removed from a message,
from an assistant tool call, from a content part, and from a part's media
object (``image_url``, ``video_url``, ``input_audio``, ``file``; the Vercel AI
SDK puts ``providerOptions`` inside ``image_url``) before the strict wire
models read the body. The removal is disclosed in ``ignored_parameters`` and
never reaches a provider: dispatch is built from the validated wire models,
which by then hold only defined fields. Every DEFINED key keeps its strict
value validation, and top-level request parameters keep their named
rejection. Keys the cache-marker passes own (``cache_control`` and
``prompt_cache_breakpoint``) are never removed here; those passes validate
them.

Disclosures are bounded: each removed key appears once per request however
many messages carried it, its name is reduced to an identifier-shaped token,
and at most :data:`MAXIMUM_NAMED_DISCLOSURES` keys are named, the rest being
counted in one summary entry, so a hostile history cannot grow the
disclosure list.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import NamedTuple, get_args

from pydantic import BaseModel

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.openai_protocol.custom_tool_wire import ChatCustomToolCall
from exp.runtime.openai_protocol.prompt_cache_controls import BREAKPOINT
from exp.runtime.openai_protocol.wire_models import (
    _AssistantToolCall,
    _ContentPart,
    _Message,
    _ResponseMessage,
)

UNKNOWN_MESSAGE_FIELD = "unknown_message_field"
"""Reason recorded for a removed key no client research has named."""

CLIENT_ANNOTATION = "client_annotation"
"""Reason recorded for a removed key traced to a specific client's bookkeeping."""

MAXIMUM_NAMED_DISCLOSURES = 16
"""Most distinct removed keys named in one request's disclosures."""

MAXIMUM_KEY_CHARACTERS = 64
"""Longest key token written into a disclosure."""

_RESERVED_KEYS = frozenset({"cache_control", BREAKPOINT})
"""Keys the cache-marker passes validate; never treated as unknown here."""

CHAT_MESSAGE_CLIENT_ANNOTATIONS = frozenset(
    {"cot_id", "cot_summary", "timestamp", "startsNewUserRequest", "agent"}
)
"""Chat message keys traced to one client, disclosed with :data:`CLIENT_ANNOTATION`.

GitHub Copilot Chat stamps ``cot_id`` and ``cot_summary``, other agents
``timestamp`` and ``startsNewUserRequest``, and WorkBuddy CLI ``agent``."""

_CHAT_REASONS: Mapping[str, str] = dict.fromkeys(CHAT_MESSAGE_CLIENT_ANNOTATIONS, CLIENT_ANNOTATION)

_MEDIA_REASONS: Mapping[str, str] = {"dimensions": CLIENT_ANNOTATION}
"""Media-object keys traced to one client: a caller size hint on an image."""

_NON_IDENTIFIER = re.compile(r"[^A-Za-z0-9_]")


class _PartKeys(NamedTuple):
    """Keys one content-part kind accepts.

    Attributes:
        part: Keys on the part object itself.
        media: Accepted keys of each media object, by the part field holding it.
    """

    part: frozenset[str]
    media: Mapping[str, frozenset[str]]


def _model_keys(model: type[BaseModel]) -> frozenset[str]:
    """Return every key ``model`` accepts on the wire, aliases included."""
    keys = set(_RESERVED_KEYS)
    for name, field in model.model_fields.items():
        keys.add(field.alias or name)
    return frozenset(keys)


def _nested_model(annotation: object) -> type[BaseModel] | None:
    """Return the wire model a field holds, unwrapping ``X | None``, else None."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    models = [arg for arg in get_args(annotation) if _nested_model(arg) is not None]
    return _nested_model(models[0]) if len(models) == 1 else None


def _part_keys_by_type() -> Mapping[str, _PartKeys]:
    """Map each content-part ``type`` value to its accepted keys and media objects.

    A part's media object (``image_url``, ``video_url``, ``input_audio``,
    ``file``) is the field whose type is itself a wire model; its accepted keys
    are recorded under the field name.
    """
    keys: dict[str, _PartKeys] = {}
    for model in get_args(get_args(_ContentPart)[0]):
        nested = {
            (field.alias or name): _model_keys(inner)
            for name, field in model.model_fields.items()
            if (inner := _nested_model(field.annotation)) is not None
        }
        for value in get_args(model.model_fields["type"].annotation):
            keys[value] = _PartKeys(_model_keys(model), nested)
    return keys


CHAT_MESSAGE_KEYS = _model_keys(_Message)
"""Keys a Chat message may carry."""

RESPONSES_MESSAGE_KEYS = _model_keys(_ResponseMessage)
"""Keys a Responses ``message`` input item may carry."""

_FUNCTION_CALL_KEYS = _model_keys(_AssistantToolCall)
_CUSTOM_CALL_KEYS = _model_keys(ChatCustomToolCall)
_PART_KEYS = _part_keys_by_type()


def key_token(key: str) -> str:
    """Reduce a caller key to a bounded identifier-shaped disclosure token.

    Args:
        key: The removed key exactly as the caller sent it.

    Returns:
        The key with every character outside ``[A-Za-z0-9_]`` replaced by
        ``_``, cut to :data:`MAXIMUM_KEY_CHARACTERS`; never empty.
    """
    return _NON_IDENTIFIER.sub("_", key[:MAXIMUM_KEY_CHARACTERS]) or "_"


class DroppedKeys:
    """Ordered, de-duplicated record of the keys removed from one request body."""

    def __init__(self, container: str) -> None:
        """Start an empty record for one top-level array.

        Args:
            container: The array the removals come from (``messages`` or ``input``).
        """
        self._container = container
        self._entries: dict[str, str] = {}

    def strip(
        self,
        value: JsonObject,
        known: frozenset[str],
        level: str = "",
        *,
        reasons: Mapping[str, str] | None = None,
    ) -> JsonObject:
        """Return ``value`` without the keys outside ``known``, recording each.

        Args:
            value: One message, tool call, or content part.
            known: Keys its wire model accepts.
            level: Path segment below the container (``tool_calls``,
                ``content``) or empty for the message itself.
            reasons: Researched keys disclosed with their own reason instead
                of :data:`UNKNOWN_MESSAGE_FIELD`.

        Returns:
            ``value`` itself when every key is known, else a copy without the
            unknown keys.
        """
        unknown = [key for key in value if key not in known]
        if not unknown:
            return value
        prefix = f"{self._container}.{level}." if level else f"{self._container}."
        for key in unknown:
            reason = (reasons or {}).get(key, UNKNOWN_MESSAGE_FIELD)
            self._entries.setdefault(f"{prefix}{key_token(key)}", reason)
        return {key: item for key, item in value.items() if key in known}

    def disclosures(self) -> tuple[str, ...]:
        """Return the bounded ``ignored_parameters`` entries for every removal.

        Returns:
            One ``<path>->dropped(<reason>)`` entry per distinct removed key in
            first-seen order, at most :data:`MAXIMUM_NAMED_DISCLOSURES` of
            them, followed by one ``<container>.*`` entry counting the rest.
        """
        entries = [f"{path}->dropped({reason})" for path, reason in self._entries.items()]
        if len(entries) <= MAXIMUM_NAMED_DISCLOSURES:
            return tuple(entries)
        hidden = len(entries) - MAXIMUM_NAMED_DISCLOSURES
        return (
            *entries[:MAXIMUM_NAMED_DISCLOSURES],
            f"{self._container}.*->dropped({hidden}_more_{UNKNOWN_MESSAGE_FIELD}s)",
        )


def _strip_parts(content: JsonValue, dropped: DroppedKeys) -> JsonValue:
    """Strip unknown keys from known-kind content parts and their media objects."""
    if not isinstance(content, list):
        return content
    parts: list[JsonValue] = []
    for part in content:
        known = _PART_KEYS.get(part.get("type")) if isinstance(part, dict) else None
        if isinstance(part, dict) and known is not None:
            part = dropped.strip(part, known.part, "content")
            for field, media_keys in known.media.items():
                media = part.get(field)
                if isinstance(media, dict):
                    stripped = dropped.strip(
                        media, media_keys, f"content.{field}", reasons=_MEDIA_REASONS
                    )
                    if stripped is not media:
                        part = {**part, field: stripped}
        parts.append(part)
    return parts


def _strip_tool_calls(calls: JsonValue, dropped: DroppedKeys) -> JsonValue:
    """Strip unknown keys from every assistant tool call object."""
    if not isinstance(calls, list):
        return calls
    return [
        dropped.strip(
            call,
            _CUSTOM_CALL_KEYS if call.get("type") == "custom" else _FUNCTION_CALL_KEYS,
            "tool_calls",
        )
        if isinstance(call, dict)
        else call
        for call in calls
    ]


def _strip_message(
    message: JsonObject,
    known: frozenset[str],
    dropped: DroppedKeys,
    reasons: Mapping[str, str] | None = None,
) -> JsonObject:
    """Strip one message and its content parts and tool calls."""
    stripped = dropped.strip(message, known, reasons=reasons)
    content = _strip_parts(stripped.get("content"), dropped)
    calls = _strip_tool_calls(stripped.get("tool_calls"), dropped)
    if content is stripped.get("content") and calls is stripped.get("tool_calls"):
        return stripped
    nested = dict(stripped)
    if "content" in stripped:
        nested["content"] = content
    if "tool_calls" in stripped:
        nested["tool_calls"] = calls
    return nested


def drop_unknown_chat_message_fields(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Remove unknown keys from Chat messages, their tool calls, and content parts.

    Args:
        payload: Parsed Chat Completions body.

    Returns:
        The body without those keys (the original object when there are
        none) and the bounded disclosures naming them.
    """
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        return payload, ()
    dropped = DroppedKeys("messages")
    messages = [
        _strip_message(message, CHAT_MESSAGE_KEYS, dropped, _CHAT_REASONS)
        if isinstance(message, dict)
        else message
        for message in raw_messages
    ]
    disclosures = dropped.disclosures()
    if not disclosures:
        return payload, ()
    return {**payload, "messages": messages}, disclosures


def drop_unknown_responses_message_fields(
    payload: JsonObject,
) -> tuple[JsonObject, tuple[str, ...]]:
    """Remove unknown keys from Responses ``message`` input items and their parts.

    Only message items (``type: "message"``, or no ``type`` with a ``role``)
    are touched; every other input item keeps its own strict validation.

    Args:
        payload: Parsed Responses body.

    Returns:
        The body without those keys (the original object when there are
        none) and the bounded disclosures naming them.
    """
    raw_input = payload.get("input")
    if not isinstance(raw_input, list):
        return payload, ()
    dropped = DroppedKeys("input")
    items: list[JsonValue] = []
    for item in raw_input:
        is_message = isinstance(item, dict) and (
            item.get("type") == "message" or ("type" not in item and "role" in item)
        )
        items.append(
            _strip_message(item, RESPONSES_MESSAGE_KEYS, dropped)
            if is_message and isinstance(item, dict)
            else item
        )
    disclosures = dropped.disclosures()
    if not disclosures:
        return payload, ()
    return {**payload, "input": items}, disclosures


def drop_unknown_message_keys(
    payload: JsonObject, known: frozenset[str]
) -> tuple[JsonObject, tuple[str, ...]]:
    """Remove unknown keys from the objects of a ``messages`` array, one level only.

    The Anthropic Messages decoder uses this for its turn objects; their
    content blocks keep the decoder's own strict validation.

    Args:
        payload: Parsed request body.
        known: Keys a message object may carry.

    Returns:
        The body without those keys (the original object when there are
        none) and the bounded disclosures naming them.
    """
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        return payload, ()
    dropped = DroppedKeys("messages")
    known = known | _RESERVED_KEYS
    messages = [
        dropped.strip(message, known) if isinstance(message, dict) else message
        for message in raw_messages
    ]
    disclosures = dropped.disclosures()
    if not disclosures:
        return payload, ()
    return {**payload, "messages": messages}, disclosures
