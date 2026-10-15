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
import re
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
# Native fields that identify an item, and fields no tag question can read.
_IDENTITY_FIELDS: Final = frozenset({"type", "id", "call_id", "name", "namespace", "status"})
_OPAQUE_FIELDS: Final = frozenset({"encrypted_content", "encrypted_index", "signature"})
_IMAGE_TYPES: Final = frozenset({"image", "input_image", "computer_screenshot"})
# Long runs of base64 characters with no spaces are binary payloads, not readable text.
_BINARY: Final = re.compile(r"[A-Za-z0-9+/_=-]{256,}")
_TEXT_PARTS: Final = frozenset({"text", "input_text", "output_text"})
_IMAGE_PARTS: Final = frozenset({"image", "image_url", "input_image"})
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
    readable, partial = _readable_responses(body) if protocol == "responses" else (body, {})
    output = output_messages(protocol, readable, capture_output)
    for message in output:
        for call in _list(message.get("tool_calls")):
            if isinstance(call, dict) and isinstance(
                raw := partial.get(str(call.get("call_id"))), str
            ):
                call["arguments"] = raw
    response_turns = tuple(turn for turn in map(_message, output) if turn is not None)
    internal = capture_context.get("provider_internal")
    internal = internal if isinstance(internal, dict) else {}
    output_config = internal.get("provider_output_config")
    output_format = output_config.get("format") if isinstance(output_config, dict) else None
    structured = (
        envelope.get("structured_text") is not None
        or internal.get("json_object_output") is True
        or (isinstance(output_format, dict) and output_format.get("type") == "json_schema")
    )
    return TagExchange(
        exchange_id=exchange_id,
        conversation_key=conversation_key or _caller_conversation(envelope, capture_context),
        tools=_tools(
            envelope.get("tools"),
            internal.get("provider_native_tools"),
            internal.get("provider_server_tools"),
        ),
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
        # A Responses stream body is its terminal event's response, so the terminal check
        # below also accepts an incomplete (length-limited) response. Chat and Messages
        # streams must also have reached their end-of-stream event with a valid lifecycle.
        complete = protocol == "responses" or (captured.completed and _lifecycle(protocol, frames))
        content = body.get("content") if isinstance(body, dict) and complete else None
        for block in content if isinstance(content, list) else ():
            if isinstance(block, dict) and "capture_partial_input" in block:
                raise ValueError("a streamed tool input is not complete JSON")
            # The assembler keeps a tool input with integers beyond 64 bits as exact source
            # text beside a rounded copy; restore the exact input, as Messages carries it.
            if isinstance(block, dict) and isinstance(
                source := block.pop("capture_input_source_json", None), str
            ):
                block["input"] = json.loads(source)
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
    return _messages_lifecycle(frames)


def _messages_lifecycle(frames: Sequence[JsonValue]) -> bool:
    """Replay a Messages stream's block lifecycle the way the collector's assembler does.

    Args:
        frames: Stored ``data`` payloads in order.

    Returns:
        True when one ``message_start`` comes first, blocks start in index order and receive
        deltas only while open, every block closes, and nothing follows ``message_stop``.
    """
    started = stopped = False
    opened: list[JsonValue] = []
    closed: set[str] = set()
    for frame in frames:
        if not isinstance(frame, dict) or stopped:
            return False
        kind = frame.get("type")
        index = frame.get("index")
        key = json.dumps(index)
        if kind == "message_start":
            if started:
                return False
            started = True
        elif kind == "content_block_start":
            if not started or index != len(opened):
                return False
            opened.append(index)
        elif kind in ("content_block_delta", "content_block_stop"):
            if index not in opened or key in closed:
                return False
            if kind == "content_block_stop":
                closed.add(key)
        elif kind == "message_delta":
            if not started or not isinstance(frame.get("delta"), dict):
                return False
        elif kind == "message_stop":
            stopped = True
        elif kind != "ping":
            return False
    return started and stopped and len(closed) == len(opened)


def _readable_responses(body: JsonObject) -> tuple[JsonObject, dict[str, str]]:
    """Prepare Responses output for the output decoder, which reuses the request schema.

    Refusal parts become output text, since the request schema has no refusal part. A
    function call whose arguments are not a JSON object (a call cut off by the length limit)
    decodes with empty arguments, and its exact text is returned for restoring afterwards.

    Args:
        body: A complete Responses body.

    Returns:
        A copied body the decoder accepts, and the cut-off argument text by call id.
    """
    output = body.get("output")
    if not isinstance(output, list):
        return body, {}
    partial: dict[str, str] = {}
    items: list[JsonValue] = []
    for item in output:
        if isinstance(item, dict) and isinstance(content := item.get("content"), list):
            parts: list[JsonValue] = [
                {"type": "output_text", "text": part.get("refusal"), "annotations": []}
                if isinstance(part, dict) and part.get("type") == "refusal"
                else part
                for part in content
            ]
            item = {**item, "content": parts}
        if (
            isinstance(item, dict)
            and item.get("type") == "function_call"
            and isinstance(arguments := item.get("arguments"), str)
            and not _json_object(arguments)
        ):
            partial[str(item.get("call_id"))] = arguments
            item = {**item, "arguments": "{}"}
        items.append(item)
    return {**body, "output": items}, partial


def _json_object(text: str) -> bool:
    """Report whether text parses as a JSON object."""
    try:
        return isinstance(json.loads(text), dict)
    except (ValueError, RecursionError):
        return False


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
        Message that stopped at ``max_tokens`` or at the model's context window.
    """
    if protocol == "responses":
        details = body.get("incomplete_details")
        return isinstance(details, dict) and details.get("reason") == "max_output_tokens"
    if protocol == "messages":
        return body.get("stop_reason") in ("max_tokens", "model_context_window_exceeded")
    choices = body.get("choices")
    return isinstance(choices, list) and any(
        isinstance(choice, dict) and choice.get("finish_reason") == "length" for choice in choices
    )


def _tools(value: JsonValue, native: JsonValue, server: JsonValue) -> tuple[TagTool, ...]:
    """Read the offered tools in caller order, including provider-native declarations.

    Args:
        value: The envelope's ``tools`` list, in canonical form.
        native: The captured Responses ``provider_native_tools`` entries (caller position
            plus declaration), or None.
        server: The captured Messages ``provider_server_tools`` declarations, or None.

    Returns:
        Every tool with a name; a namespace contributes its functions as ``namespace.name``,
        and server tools follow the others.
    """
    ordinary = [_tool(tool) for tool in _list(value) if isinstance(tool, dict)]
    placed: dict[int, list[TagTool]] = {}
    for entry in _list(native):
        if isinstance(entry, dict) and isinstance(tool := entry.get("tool"), dict):
            index = entry.get("index")
            placed.setdefault(index if isinstance(index, int) else 1 << 30, []).extend(
                _native_tools(tool)
            )
    tools: list[TagTool | None] = []
    remaining = iter(ordinary)
    for position in range(len(ordinary) + len(placed)):
        if position in placed:
            tools.extend(placed.pop(position))
        else:
            tools.append(next(remaining, None))
    tools.extend(remaining)
    for rest in placed.values():
        tools.extend(rest)
    tools.extend(_tool(tool) for tool in _list(server) if isinstance(tool, dict))
    return tuple(tool for tool in tools if tool is not None)


def _tool(tool: JsonObject, prefix: str = "") -> TagTool | None:
    """Read one function-like tool declaration.

    Args:
        tool: A declaration with ``name``, ``description`` and ``parameters``.
        prefix: Namespace prefix for nested functions.

    Returns:
        The tool, or None when it has no name.
    """
    name = tool.get("name", tool.get("server_label", tool.get("type")))
    if not isinstance(name, str) or not name:
        return None
    description = tool.get("description")
    parameters = tool.get("parameters")
    return TagTool(
        name=f"{prefix}{name}",
        description=description if isinstance(description, str) else None,
        parameters=parameters if isinstance(parameters, dict) else {},
    )


def _native_tools(tool: JsonObject) -> list[TagTool]:
    """Read one provider-native declaration, expanding a namespace into its functions.

    Args:
        tool: The native declaration.

    Returns:
        The tools it offers.
    """
    if tool.get("type") == "namespace" and isinstance(tool.get("tools"), list):
        prefix = f"{tool.get('name')}."
        nested = [
            _tool(item, prefix) for item in _list(tool.get("tools")) if isinstance(item, dict)
        ]
        return [item for item in nested if item is not None]
    single = _tool(tool)
    return [single] if single is not None else []


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
    """Keep what a tag question can read from a provider-native item.

    Cited text keeps its text. A tool result item (a Responses ``*_output`` item or an
    Anthropic ``*_tool_result`` block) becomes a tool result with its readable output, and a
    tool call item (``*_call``, ``server_tool_use``) becomes a tool call under its own name.
    Encrypted fields, signatures and binary payloads are dropped and images become markers.

    Args:
        role: The carrying message's role.
        item: The native item or block.

    Returns:
        The readable message, or None for reasoning and other unreadable items.
    """
    if not isinstance(item, dict):
        return None
    kind = str(item.get("type"))
    if kind == "text" and isinstance(text := item.get("text"), str):
        return TagMessage(role="assistant", content=text) if role == "assistant" else None
    call_id = item.get("call_id", item.get("tool_use_id"))
    if role == "tool" or kind.endswith(("_output", "_result")):
        output = item.get("output", item.get("content"))
        readable, images = _readable(output)
        text = readable if isinstance(readable, str) else _compact_json(readable)
        return TagMessage(
            role="tool",
            content=(TagTextPart(text=text), *(TagImagePart() for _ in range(images)))
            if images
            else text,
            tool_call_id=call_id if isinstance(call_id, str) else None,
        )
    if role != "assistant" or not (kind.endswith("_call") or kind.endswith("tool_use")):
        return None
    arguments = item.get("input", item.get("arguments"))
    if arguments is None:
        arguments = {key: field for key, field in item.items() if key not in _IDENTITY_FIELDS}
    readable, _ = _readable(arguments)
    call: JsonObject = {
        "id": call_id if isinstance(call_id, str) else item.get("id"),
        "name": item.get("name", kind.removesuffix("_call")),
        "namespace": item.get("namespace"),
        "arguments": readable,
    }
    calls = _tool_calls([call])
    return TagMessage(role="assistant", tool_calls=calls) if calls else None


def _readable(value: JsonValue) -> tuple[JsonValue, int]:
    """Project a native value onto what a reader can use.

    Args:
        value: Any JSON value from a native item.

    Returns:
        The value without encrypted fields or signatures, with binary strings shown as
        ``[binary]`` and images as ``[image]``, and the number of images found.
    """
    if isinstance(value, str):
        binary = value.startswith("data:") or _BINARY.fullmatch(value) is not None
        return ("[binary]", 0) if binary else (value, 0)
    if isinstance(value, list):
        items = [_readable(item) for item in value]
        return [item for item, _ in items], sum(count for _, count in items)
    if not isinstance(value, dict):
        return value, 0
    if str(value.get("type")) in _IMAGE_TYPES:
        return "[image]", 1
    projected: JsonObject = {}
    images = 0
    for key, field in value.items():
        if key in _OPAQUE_FIELDS:
            continue
        projected[key], count = _readable(field)
        images += count
    return projected, images


def _compact_json(value: JsonValue) -> str:
    """Render a JSON value compactly for a window."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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
        if kind in _TEXT_PARTS and isinstance(text := part.get("text"), str):
            converted.append(TagTextPart(text=text))
        elif kind in _IMAGE_PARTS:
            converted.append(_image(part))
        elif isinstance(kind, str):
            converted.append(TagTextPart(text=f"[{kind}]"))
    if not any(isinstance(part, TagImagePart) for part in converted):
        return "".join(part.text for part in converted if isinstance(part, TagTextPart))
    return tuple(converted)


def _image(part: JsonObject) -> TagImagePart:
    """Record an image part by media type and remote URL only.

    Args:
        part: A canonical image part, a Chat ``image_url`` part or a Responses
            ``input_image`` part.

    Returns:
        The image marker; inline data is never copied.
    """
    url = part.get("url")
    nested = part.get("image_url")
    if isinstance(nested, dict):
        url = nested.get("url")
    elif isinstance(nested, str):
        url = nested
    media_type = part.get("media_type")
    remote = isinstance(url, str) and url.startswith(("http://", "https://"))
    return TagImagePart(
        media_type=media_type if isinstance(media_type, str) else None,
        url=url if remote and isinstance(url, str) else None,
    )


def _tool_calls(value: JsonValue) -> tuple[TagToolCall, ...]:
    """Read assistant tool calls in canonical or chat form, function or custom.

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
        nested = call.get("function", call.get("custom"))
        source = nested if isinstance(nested, dict) else call
        name = source.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = source.get("arguments", source.get("input"))
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {}, ensure_ascii=False, separators=(",", ":"))
        call_id = call.get("id", call.get("call_id"))
        provider = call.get("provider_context")
        namespace = call.get(
            "namespace", provider.get("provider_namespace") if isinstance(provider, dict) else None
        )
        if isinstance(namespace, str) and namespace:
            name = f"{namespace}.{name}"
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
