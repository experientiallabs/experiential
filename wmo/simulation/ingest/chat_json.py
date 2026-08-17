"""Normalize OpenAI-style chat JSON conversations into canonical trace evidence.

A chat JSON export is the smallest useful trace source: an OpenAI Chat Completions style
conversation, which most agent frameworks can dump without any observability vendor. The supported
shapes are one conversation object with a ``messages`` array, an array of such conversations, a bare
message array, or JSONL with one conversation or message array per line.

Message mapping follows the roles the conversation already declares:

- the first ``user`` or ``human`` message supplies the trace request text,
- an ``assistant`` message with ``tool_calls`` becomes one model call per requested tool,
- an ordinary ``assistant`` message becomes a model call carrying its completion text,
- a ``tool`` message becomes the tool result paired with the earlier call,
- ``system`` and ``developer`` messages carry no agent step and are not converted.

Chat JSON carries no source timing. When a message declares no timestamp, WMO assigns ordinal
timestamps that preserve message order only, and every resulting span records
``wmo.source.time.synthetic`` so downstream consumers can never mistake assigned order for measured
latency. Provider and model identity is retained only when the conversation declares both.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject, canonical_json_bytes
from wmo.simulation.ingest.vendor_observations import (
    VendorModelIdentity,
    VendorObservation,
    declared_tool_calls,
)
from wmo.simulation.ingest.vendor_records import (
    VendorTraceFormatError,
    first_text,
    flatten_records,
    message_role,
    message_text,
    required_text,
    source_timestamp,
)
from wmo.simulation.ingest.vendor_source import VendorSource
from wmo.simulation.ingest.vendor_trace import approved_extensions

VENDOR = "chat-json"

_TRACE_ID_KEYS = ("trace_id", "conversation_id", "session_id", "thread_id", "id")
_MODEL_KEYS = ("model", "model_id")
_PROVIDER_KEYS = ("provider", "system")
_TIMESTAMP_KEYS = ("timestamp", "created_at", "created", "time")
_USER_ROLES = frozenset({"user", "human"})
_SYNTHETIC_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _conversations(payload: JsonValue) -> tuple[JsonObject, ...]:
    """Flatten supported chat export shapes into conversation objects.

    Args:
        payload: Decoded conversation, conversation array, or bare message array.

    Returns:
        Conversation objects in source order.

    Raises:
        VendorTraceFormatError: The payload is not a supported conversation shape.
    """
    if isinstance(payload, list) and payload and _is_message_array(payload):
        return ({"messages": payload},)
    return flatten_records(
        payload,
        vendor=VENDOR,
        wrapper_keys=("conversations", "data", "results"),
        record_keys=("messages", "conversation"),
    )


def _is_message_array(payload: Sequence[JsonValue]) -> bool:
    """Return whether a bare array holds chat messages rather than conversations.

    Args:
        payload: Decoded JSON array.

    Returns:
        Whether every array item is an object declaring a role.
    """
    return all(isinstance(item, dict) and "role" in item for item in payload)


def _conversation_observations(
    conversation: JsonObject, ordinal: int
) -> tuple[VendorObservation, ...]:
    """Convert one conversation's messages into declared vendor observations.

    Args:
        conversation: One conversation object with a message array.
        ordinal: Source order offset for the first emitted observation.

    Returns:
        Declared observations for every convertible message.

    Raises:
        VendorTraceFormatError: The conversation has no message array or no user message.
    """
    raw_messages = conversation.get("messages", conversation.get("conversation"))
    if not isinstance(raw_messages, list):
        raise VendorTraceFormatError("chat JSON conversations need a messages array")
    messages = [message for message in raw_messages if isinstance(message, dict)]
    if len(messages) != len(raw_messages):
        raise VendorTraceFormatError("chat JSON messages must be objects")
    source_trace_id = _source_trace_id(conversation)
    extensions = _extensions(conversation)
    model = _model_identity(conversation)
    request_text = _first_user_message(messages)
    if request_text is None:
        raise VendorTraceFormatError("chat JSON conversation has no user message")
    observations: list[VendorObservation] = []
    input_messages: list[JsonValue] = []
    for index, message in enumerate(messages):
        role = message_role(message)
        timestamp, synthetic = _message_timestamp(message, index)
        emitted = _message_observations(
            message,
            role=role,
            index=index,
            ordinal=ordinal + len(observations),
            timestamp=timestamp,
            synthetic=synthetic,
            source_trace_id=source_trace_id,
            request_text=request_text if not observations else None,
            input_messages=tuple(input_messages),
            model=model,
            extensions=extensions,
        )
        observations.extend(emitted)
        input_messages.append(message)
    if not observations:
        raise VendorTraceFormatError("chat JSON conversation has no assistant or tool message")
    return tuple(observations)


def _message_observations(
    message: JsonObject,
    *,
    role: str,
    index: int,
    ordinal: int,
    timestamp: datetime,
    synthetic: bool,
    source_trace_id: str,
    request_text: str | None,
    input_messages: tuple[JsonValue, ...],
    model: VendorModelIdentity | None,
    extensions: JsonObject,
) -> tuple[VendorObservation, ...]:
    """Convert one chat message into zero or more declared observations.

    Args:
        message: One chat message object.
        role: Lowercase message role.
        index: Message position in the conversation.
        ordinal: Source order position for the emitted observation.
        timestamp: Source or assigned message timestamp.
        synthetic: Whether the timestamp is WMO-assigned.
        source_trace_id: Vendor conversation key.
        request_text: Trace request text, supplied only on the first emitted observation.
        input_messages: Messages preceding this one, retained as canonical input evidence.
        model: Declared conversation model identity.
        extensions: Approved WMO extension attributes for the conversation.

    Returns:
        Declared observations for this message.

    Raises:
        VendorTraceFormatError: A tool message declares no resolvable tool name.
    """
    if role == "tool":
        return (
            VendorObservation(
                source_trace_id=source_trace_id,
                source_span_id=f"message-{index}",
                ordinal=ordinal,
                started_at=timestamp,
                ended_at=timestamp,
                kind="tool_result",
                request_text=request_text,
                tool_name=required_text(
                    message.get("name", message.get("tool_name")),
                    "chat JSON tool message name",
                ),
                tool_message=message_text(message.get("content")),
                tool_call_id=first_text(message, ("tool_call_id", "call_id", "id")),
                extensions=extensions,
                synthetic_time=synthetic,
            ),
        )
    if role != "assistant":
        return ()
    completion = message_text(message.get("content"))
    return (
        VendorObservation(
            source_trace_id=source_trace_id,
            source_span_id=f"message-{index}",
            ordinal=ordinal,
            started_at=timestamp,
            ended_at=timestamp,
            kind="model",
            request_text=request_text,
            input_messages=list(input_messages),
            completion_text=completion or None,
            tool_calls=declared_tool_calls(message),
            model=model,
            extensions=extensions,
            synthetic_time=synthetic,
        ),
    )


def _first_user_message(messages: Sequence[JsonObject]) -> str | None:
    """Return the first user-visible request text in one conversation.

    Args:
        messages: Conversation messages in source order.

    Returns:
        Normalized request text, or ``None`` when the conversation declares none.
    """
    for message in messages:
        if message_role(message) not in _USER_ROLES:
            continue
        text = message_text(message.get("content"))
        if text:
            return text
    return None


def _message_timestamp(message: JsonObject, index: int) -> tuple[datetime, bool]:
    """Read a declared message timestamp or assign an order-preserving one.

    Args:
        message: One chat message object.
        index: Message position in the conversation.

    Returns:
        The timestamp and whether WMO assigned it.
    """
    for key in _TIMESTAMP_KEYS:
        if key in message:
            return source_timestamp(message[key], f"chat JSON message {key}"), False
    return _SYNTHETIC_EPOCH + timedelta(seconds=index), True


def _source_trace_id(conversation: JsonObject) -> str:
    """Resolve one stable conversation key for grouping and identity.

    Args:
        conversation: One conversation object.

    Returns:
        Declared conversation key, or a content digest when the export declares none.
    """
    declared = first_text(conversation, _TRACE_ID_KEYS)
    if declared is not None:
        return declared
    digest = hashlib.sha256(canonical_json_bytes(conversation)).hexdigest()
    return f"chat-json:{digest[:32]}"


def _extensions(conversation: JsonObject) -> JsonObject:
    """Read approved WMO extensions and declared metadata from one conversation.

    Args:
        conversation: One conversation object.

    Returns:
        Approved extension attributes for every span of this conversation.
    """
    extensions = approved_extensions(conversation)
    metadata = conversation.get("metadata")
    if isinstance(metadata, dict) and "wmo.trace.metadata" not in extensions:
        extensions["wmo.trace.metadata"] = metadata
    return extensions


def _model_identity(conversation: JsonObject) -> VendorModelIdentity | None:
    """Retain conversation model identity only when provider and model are declared.

    Args:
        conversation: One conversation object.

    Returns:
        Declared model identity, or ``None`` when the export declares neither field.

    Raises:
        VendorTraceFormatError: Only one of provider and model is declared.
    """
    model_id = first_text(conversation, _MODEL_KEYS)
    provider = first_text(conversation, _PROVIDER_KEYS)
    if model_id is None and provider is None:
        return None
    if model_id is None or provider is None:
        raise VendorTraceFormatError(
            "chat JSON model identity needs both a provider and a model to be retained"
        )
    return VendorModelIdentity(
        provider=provider,
        model_id=model_id,
        revision=first_text(conversation, ("model_revision", "revision")),
    )


CHAT_JSON_SOURCE: VendorSource[JsonObject] = VendorSource(
    vendor=VENDOR,
    records=_conversations,
    convert=_conversation_observations,
)
