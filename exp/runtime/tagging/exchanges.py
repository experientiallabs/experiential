"""Build tagging exchanges from captured gateway traffic.

Two capture forms feed tagging. A local ``CapturedExchange`` comes from the gateway capture
store. A capture record's parts are what a store keeps from one native collector record: the
effective request messages, the request envelope (the effective request without its messages,
plus provider-internal settings) and the response document (``{"kind": "json", ...}`` or
``{"kind": "sse", ...}``). Both decode protocol output through the gateway ingest conversion, so
Chat Completions, Responses and Messages traffic all become the same ``TagExchange``.

Only complete, successful responses are tagged, matching what the local store retains. The
adapters read the stored, queryable projection rather than its exact ``source_json`` sidecar:
tagging needs readable text, and the projection has already replaced the code points (NUL,
lone surrogates) that cannot be stored or sent as JSON text.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from typing import Final

from exp_gateway_native import CaptureResponse
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.tags.exchange import (
    MessageRole,
    TagContentPart,
    TagExchange,
    TagFunction,
    TagImagePart,
    TagMessage,
    TagTextPart,
    TagTool,
    TagToolCall,
)
from exp.runtime.gateway.contracts import TOOL_ERROR_TEXT_PREFIX
from exp.runtime.gateway.ingest.conversion import (
    CaptureProtocol,
    output_messages,
    restore_provider_context,
)
from exp.runtime.gateway.local_capture_contracts import CapturedExchange
from exp.runtime.gateway.native_capture import CaptureRecord

_ROLES: Final[dict[str, MessageRole]] = {
    "system": "system",
    "developer": "developer",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
}
_MESSAGES_EVENTS: Final = frozenset(
    {
        "ping",
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    }
)
_NATIVE_PROTOCOL: Final[dict[CaptureProtocol, str]] = {
    "chat_completions": "chat",
    "responses": "responses",
    "messages": "messages",
}


class TagExchangeError(ValueError):
    """Raised when captured traffic cannot become a tagging exchange; says why."""


def exchange_from_captured(experience: CapturedExchange) -> TagExchange:
    """Build a tagging exchange from one local gateway capture.

    The caller session label, when the capture has one, becomes the conversation key.

    Args:
        experience: A completed exchange from the local capture store.

    Returns:
        The exchange.

    Raises:
        TagExchangeError: The capture lacks its effective request or a complete response.
    """
    try:
        context = experience.request.get("exp_context")
        if not isinstance(context, dict) or context.get("schema_version") != 1:
            raise ValueError("it has no effective request context")
        effective = context.get("request")
        if not isinstance(effective, dict):
            raise ValueError("it has no effective request")
        output = experience.request.get("exp_capture_output")
        return _from_parts(
            exchange_id=experience.experience_id,
            protocol=experience.protocol,
            messages=_list(effective.get("messages")),
            envelope=effective,
            capture_context=context,
            capture_output=output if isinstance(output, dict) else None,
            body=experience.response,
            conversation_key=experience.episode_id,
        )
    except ValueError as exc:
        raise TagExchangeError(
            f"capture {experience.experience_id} cannot be tagged: {exc}. Collect fresh traffic "
            "with capture enabled."
        ) from exc


def exchange_from_capture_parts(
    *,
    exchange_id: str,
    protocol: CaptureProtocol,
    messages: Sequence[JsonValue],
    request: JsonObject | None,
    response: JsonObject,
    capture_context: JsonObject | None = None,
    capture_output: JsonObject | None = None,
    conversation_key: str | None = None,
) -> TagExchange:
    """Build a tagging exchange from the stored parts of one capture record.

    Args:
        exchange_id: The record's request id.
        protocol: The record's public surface.
        messages: The effective request messages, ``context["request"]["messages"]``.
        request: The request envelope, the effective request without its messages, or None
            when it was not retained.
        response: The record's response document, as ``CaptureJsonResponse`` or
            ``CaptureSseResponse`` serialize it; extra keys are ignored.
        capture_context: The record's capture context without its request
            (``provider_context``, ``provider_internal``, ``session_id``), or None.
        capture_output: The record's output evidence (``provider_tool_calls_json``), or None.
        conversation_key: The store's conversation key, or None to use the caller's
            conversation label (``metadata.conversation_id``, else the session id) as the local
            capture store does, or failing that the opening key.

    Returns:
        The exchange.

    Raises:
        TagExchangeError: The response failed or is incomplete, or no message can be read.
    """
    try:
        return _from_parts(
            exchange_id=exchange_id,
            protocol=protocol,
            messages=messages,
            envelope=request or {},
            capture_context=capture_context or {},
            capture_output=capture_output,
            body=completed_response(protocol, response),
            conversation_key=conversation_key,
        )
    except ValueError as exc:
        raise TagExchangeError(f"capture {exchange_id} cannot be tagged: {exc}") from exc


def exchange_from_capture_record(
    record: CaptureRecord, *, conversation_key: str | None = None
) -> TagExchange:
    """Build a tagging exchange from one complete native collector record.

    The record is split into the parts ``exchange_from_capture_parts`` reads.

    Args:
        record: A record whose response has been captured.
        conversation_key: The store's conversation key, or None to use the caller's label.

    Returns:
        The exchange.

    Raises:
        TagExchangeError: The record has no response or cannot be read.
    """
    request_id = record.request.request_id
    if record.response is None:
        raise TagExchangeError(f"capture {request_id} has no response yet; tag it once complete")
    context = record.request.context
    effective = context.get("request")
    if not isinstance(effective, dict):
        raise TagExchangeError(f"capture {request_id} has no effective request")
    return exchange_from_capture_parts(
        exchange_id=request_id,
        protocol=record.request.protocol,
        messages=_list(effective.get("messages")),
        request={key: value for key, value in effective.items() if key != "messages"},
        response=record.response.model_dump(mode="json"),
        capture_context={key: value for key, value in context.items() if key != "request"},
        capture_output={"provider_tool_calls_json": record.provider_tool_calls_json},
        conversation_key=conversation_key,
    )


def _list(value: JsonValue) -> list[JsonValue]:
    """Return a JSON list, or an empty list for any other value."""
    return value if isinstance(value, list) else []


def _from_parts(
    *,
    exchange_id: str,
    protocol: CaptureProtocol,
    messages: Sequence[JsonValue],
    envelope: JsonObject,
    capture_context: JsonObject,
    capture_output: JsonObject | None,
    body: JsonObject,
    conversation_key: str | None,
) -> TagExchange:
    """Assemble an exchange from request parts and a complete response body.

    Args:
        exchange_id: Exchange identifier.
        protocol: Public surface.
        messages: Canonical request messages; they are copied, never modified.
        envelope: Effective request fields; ``tools``, ``structured_text`` and ``metadata`` are
            read, and a ``messages`` entry is ignored.
        capture_context: Capture context; ``provider_context``, ``provider_internal`` and
            ``session_id`` are read.
        capture_output: Output evidence restoring exact tool arguments, or None.
        body: Complete response body.
        conversation_key: Explicit conversation key, or None to use the caller's label.

    Returns:
        The exchange.

    Raises:
        ValueError: The provider context does not match the messages, the response has no
            output, or no request message can be read.
    """
    canonical = restore_provider_context(
        [copy.deepcopy(message) for message in messages if isinstance(message, dict)],
        capture_context.get("provider_context"),
    )
    request_turns = tuple(turn for turn in map(_message, canonical) if turn is not None)
    if not request_turns:
        raise ValueError("the request has no readable messages")
    output = output_messages(
        protocol, _refusals_as_text(body) if protocol == "responses" else body, capture_output
    )
    response_turns = tuple(turn for turn in map(_message, output) if turn is not None)
    internal = capture_context.get("provider_internal")
    structured = envelope.get("structured_text") is not None or (
        isinstance(internal, dict) and internal.get("json_object_output") is True
    )
    return TagExchange(
        exchange_id=exchange_id,
        conversation_key=conversation_key or _caller_conversation(envelope, capture_context),
        tools=_tools(envelope.get("tools")),
        messages=request_turns,
        response=response_turns or (TagMessage(role="assistant"),),
        truncated=_finished_by_length(protocol, body),
        structured_output=structured,
    )


def _caller_conversation(envelope: JsonObject, capture_context: JsonObject) -> str | None:
    """Return the caller's conversation label the way the local capture store reads it.

    Args:
        envelope: Effective request fields.
        capture_context: Capture context.

    Returns:
        A non-blank ``metadata.conversation_id`` of at most 512 UTF-8 bytes, else the session
        id, else None.
    """
    metadata = envelope.get("metadata")
    label = metadata.get("conversation_id") if isinstance(metadata, dict) else None
    if isinstance(label, str) and label.strip() and len(label.encode("utf-8")) <= 512:
        return label
    session = capture_context.get("session_id")
    return session if isinstance(session, str) and session else None


def completed_response(protocol: CaptureProtocol, document: JsonObject) -> JsonObject:
    """Return the complete response body of a stored response document.

    Streams are reassembled by the native collector's own frame assembly. A response counts
    as complete when it has a 2xx status, reached a terminal state with no error, and (for a
    stream) lost no frames and was not disconnected.

    Args:
        protocol: The record's public surface.
        document: The record's response document.

    Returns:
        The response body.

    Raises:
        ValueError: The response failed, is incomplete, or is not a JSON object.
    """
    status = document.get("status")
    if not isinstance(status, int) or not 200 <= status < 300:
        raise ValueError(f"the response status is {status}; only successful responses are tagged")
    kind = document.get("kind")
    if kind == "json":
        body = document.get("body")
        complete = True
    elif kind == "sse":
        if document.get("truncated") is not False or document.get("client_disconnected"):
            raise ValueError("the stream was cut short before it completed")
        frames = document.get("frames")
        if not isinstance(frames, list) or any(_has_error(frame) for frame in frames):
            raise ValueError("the stream has no frames or reports an error")
        captured = CaptureResponse(_NATIVE_PROTOCOL[protocol], _sse_bytes(frames), True)
        body = json.loads(captured.body_json)
        content = body.get("content") if isinstance(body, dict) else None
        for block in content if isinstance(content, list) else ():
            # The assembler keeps a tool input with integers beyond 64 bits as exact source
            # text beside a rounded copy; restore the exact input, as Messages carries it.
            if isinstance(block, dict) and isinstance(
                source := block.pop("capture_input_source_json", None), str
            ):
                block["input"] = json.loads(source)
        # A Responses stream body is its terminal event's response, so the terminal check
        # below also accepts an incomplete (length-limited) response. Chat and Messages
        # streams must also have reached their end-of-stream event.
        complete = protocol == "responses" or (captured.completed and _lifecycle(protocol, frames))
    else:
        raise ValueError(f"unknown response kind {kind!r}")
    if not isinstance(body, dict) or not complete or not _terminal(protocol, body):
        raise ValueError("the response did not reach a terminal state")
    if _has_error(body):
        raise ValueError("the response reports an error")
    return body


def _lifecycle(protocol: CaptureProtocol, frames: Sequence[JsonValue]) -> bool:
    """Apply the collector's stream validity rules for Chat and Messages streams.

    Args:
        protocol: Chat Completions or Messages.
        frames: Stored ``data`` payloads in order.

    Returns:
        For Chat, every chunk carries choices, the stream ends with ``[DONE]`` and its first
        chunk names an id and a model. For Messages, every frame is a known event, one
        ``message_start`` opens the stream, ``message_stop`` ends it, and every content block
        that started (in index order) also stopped.
    """
    if protocol == "chat_completions":
        chunks = frames[:-1]
        first = chunks[0] if chunks else None
        return (
            frames[-1:] == ["[DONE]"]
            and isinstance(first, dict)
            and first.get("id") is not None
            and first.get("model") is not None
            and all(
                isinstance(chunk, dict) and isinstance(chunk.get("choices"), list)
                for chunk in chunks
            )
        )
    if not all(
        isinstance(frame, dict) and frame.get("type") in _MESSAGES_EVENTS for frame in frames
    ):
        return False
    events = [frame for frame in frames if isinstance(frame, dict) and frame.get("type") != "ping"]
    kinds = [event.get("type") for event in events]
    started = [event.get("index") for event in events if event.get("type") == "content_block_start"]
    stopped = {event.get("index") for event in events if event.get("type") == "content_block_stop"}
    return (
        kinds[:1] == ["message_start"]
        and kinds.count("message_start") == 1
        and kinds[-1:] == ["message_stop"]
        and kinds.count("message_stop") == 1
        and started == list(range(len(started)))
        and stopped == set(started)
    )


def _refusals_as_text(body: JsonObject) -> JsonObject:
    """Turn Responses refusal parts into output text the output decoder accepts.

    The output decoder reuses the request schema, which has no refusal part; the refusal text
    is what a tag question needs to see.

    Args:
        body: A complete Responses body.

    Returns:
        The body, copied with each refusal part rewritten when it has any.
    """
    output = body.get("output")
    if not isinstance(output, list):
        return body
    items: list[JsonValue] = []
    for item in output:
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(item, dict) and isinstance(content, list):
            parts: list[JsonValue] = [
                {"type": "output_text", "text": part.get("refusal"), "annotations": []}
                if isinstance(part, dict) and part.get("type") == "refusal"
                else part
                for part in content
            ]
            item = {**item, "content": parts}
        items.append(item)
    return {**body, "output": items}


def _sse_bytes(frames: Sequence[JsonValue]) -> bytes:
    """Encode stored frames back into the event-stream text they were parsed from.

    Args:
        frames: Stored ``data`` payloads; JSON values, or text for non-JSON payloads.

    Returns:
        UTF-8 event-stream bytes.
    """
    events: list[str] = []
    for frame in frames:
        payload = frame if isinstance(frame, str) else json.dumps(frame, ensure_ascii=False)
        events.append("".join(f"data: {line}\n" for line in payload.split("\n")) + "\n")
    return "".join(events).encode("utf-8")


def _terminal(protocol: CaptureProtocol, body: JsonObject) -> bool:
    """Report whether a response body reached a terminal state.

    Args:
        protocol: The public surface.
        body: The response body.

    Returns:
        True for a completed or incomplete Response, a Message with a stop reason, or Chat
        choices that all carry a finish reason.
    """
    if protocol == "responses":
        return body.get("status") in ("completed", "incomplete")
    if protocol == "messages":
        return (
            body.get("type") == "message"
            and isinstance(body.get("content"), list)
            and isinstance(body.get("stop_reason"), str)
        )
    choices = body.get("choices")
    return (
        isinstance(choices, list)
        and bool(choices)
        and all(isinstance(choice, dict) and choice.get("finish_reason") for choice in choices)
    )


def _has_error(value: JsonValue) -> bool:
    """Report whether a body or frame carries a provider error.

    Args:
        value: A response body or stream frame.

    Returns:
        True for a non-null ``error`` or an error or failed event.
    """
    return isinstance(value, dict) and (
        value.get("error") is not None or value.get("type") in ("error", "response.failed")
    )


def _finished_by_length(protocol: CaptureProtocol, body: JsonObject) -> bool:
    """Report whether the response stopped at its output length limit.

    Args:
        protocol: The public surface.
        body: The complete response body.

    Returns:
        True for a ``length`` finish, an incomplete Response over ``max_output_tokens`` or a
        Message that stopped at ``max_tokens``.
    """
    if protocol == "responses":
        details = body.get("incomplete_details")
        return isinstance(details, dict) and details.get("reason") == "max_output_tokens"
    if protocol == "messages":
        return body.get("stop_reason") == "max_tokens"
    choices = body.get("choices")
    return isinstance(choices, list) and any(
        isinstance(choice, dict) and choice.get("finish_reason") == "length" for choice in choices
    )


def _tools(value: JsonValue) -> tuple[TagTool, ...]:
    """Read the offered tool definitions.

    Args:
        value: The envelope's ``tools`` list, in canonical form.

    Returns:
        The tools with a name.
    """
    if not isinstance(value, list):
        return ()
    tools: list[TagTool] = []
    for tool in value:
        if not isinstance(tool, dict) or not isinstance(name := tool.get("name"), str) or not name:
            continue
        description = tool.get("description")
        parameters = tool.get("parameters")
        tools.append(
            TagTool(
                name=name,
                description=description if isinstance(description, str) else None,
                parameters=parameters if isinstance(parameters, dict) else {},
            )
        )
    return tuple(tools)


def _message(value: JsonObject) -> TagMessage | None:
    """Convert one canonical or chat-form message, dropping provider-native items.

    A Chat refusal becomes the assistant's text, and a failed tool result keeps the
    gateway's ``[tool error]`` prefix, so both stay visible to tag questions.

    Args:
        value: A message from the request or the decoded response.

    Returns:
        The message, or None for a native item or an unknown role.
    """
    role = _ROLES.get(str(value.get("role")))
    if role is None:
        return None
    native = value.get("provider_native_item", value.get("provider_anthropic_block"))
    if native is not None:
        return _native(role, native)
    content = _content(value)
    refusal = value.get("refusal")
    if role == "assistant" and not content and isinstance(refusal, str):
        content = refusal
    if role == "tool" and value.get("tool_is_error") is True:
        if isinstance(content, tuple):
            content = (TagTextPart(text=TOOL_ERROR_TEXT_PREFIX), *content)
        else:
            content = f"{TOOL_ERROR_TEXT_PREFIX}{content or ''}"
    call_id = value.get("tool_call_id")
    return TagMessage(
        role=role,
        content=content,
        tool_calls=_tool_calls(value.get("tool_calls")) if role == "assistant" else (),
        tool_call_id=call_id if role == "tool" and isinstance(call_id, str) else None,
    )


def _native(role: MessageRole, item: JsonValue) -> TagMessage | None:
    """Keep what a tag question can read from a provider-native output item.

    Cited text keeps its text, and server-side tool use (a Responses ``*_call`` item or an
    Anthropic ``server_tool_use`` block) becomes a tool call. Reasoning, encrypted results and
    other opaque items are dropped.

    Args:
        role: The carrying message's role.
        item: The native item or block.

    Returns:
        The readable message, or None when nothing in the item is readable.
    """
    if not isinstance(item, dict) or role != "assistant":
        return None
    kind = str(item.get("type"))
    if kind == "text" and isinstance(text := item.get("text"), str):
        return TagMessage(role=role, content=text)
    if kind == "server_tool_use":
        call: JsonObject = {
            "id": item.get("id"),
            "name": item.get("name"),
            "arguments": item.get("input"),
        }
    elif kind.endswith("_call") and kind != "function_call":
        details = {key: field for key, field in item.items() if key not in ("type", "id", "status")}
        call = {"id": item.get("id"), "name": kind.removesuffix("_call"), "arguments": details}
    else:
        return None
    calls = _tool_calls([call])
    return TagMessage(role=role, tool_calls=calls) if calls else None


def _content(value: JsonObject) -> str | tuple[TagContentPart, ...] | None:
    """Read message content, keeping image positions without their bytes.

    Args:
        value: A canonical message (``content`` plus ``content_parts``) or a chat message
            (``content`` as text or parts).

    Returns:
        Text, ordered parts when the message carries images, or None.
    """
    parts = value.get("content_parts")
    if not isinstance(parts, list) or not parts:
        parts = value.get("content")
        if not isinstance(parts, list):
            return parts if isinstance(parts, str) else None
    converted: list[TagContentPart] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        kind = part.get("kind", part.get("type"))
        if kind == "text" and isinstance(text := part.get("text"), str):
            converted.append(TagTextPart(text=text))
        elif kind in ("image", "image_url"):
            converted.append(_image(part))
        elif isinstance(kind, str):
            converted.append(TagTextPart(text=f"[{kind}]"))
    if not any(isinstance(part, TagImagePart) for part in converted):
        return "".join(part.text for part in converted if isinstance(part, TagTextPart))
    return tuple(converted)


def _image(part: JsonObject) -> TagImagePart:
    """Record an image part by media type and remote URL only.

    Args:
        part: A canonical image part or a chat ``image_url`` part.

    Returns:
        The image marker; inline data is never copied.
    """
    url = part.get("url")
    nested = part.get("image_url")
    if isinstance(nested, dict):
        url = nested.get("url")
    media_type = part.get("media_type")
    remote = isinstance(url, str) and url.startswith(("http://", "https://"))
    return TagImagePart(
        media_type=media_type if isinstance(media_type, str) else None,
        url=url if remote and isinstance(url, str) else None,
    )


def _tool_calls(value: JsonValue) -> tuple[TagToolCall, ...]:
    """Read assistant tool calls in canonical or chat form.

    Args:
        value: The message's ``tool_calls`` list.

    Returns:
        The calls with a tool name; arguments become JSON text.
    """
    if not isinstance(value, list):
        return ()
    calls: list[TagToolCall] = []
    for call in value:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        source = function if isinstance(function, dict) else call
        name = source.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = source.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {}, ensure_ascii=False, separators=(",", ":"))
        call_id = call.get("id", call.get("call_id"))
        calls.append(
            TagToolCall(
                id=call_id if isinstance(call_id, str) and call_id else f"{name}-{len(calls)}",
                function=TagFunction(name=name, arguments=arguments),
            )
        )
    return tuple(calls)


__all__ = [
    "TagExchangeError",
    "completed_response",
    "exchange_from_capture_parts",
    "exchange_from_capture_record",
    "exchange_from_captured",
]
