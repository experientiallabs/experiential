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
from exp.runtime.tagging.responses import (
    completed_response,
    finished_by_length,
    readable_responses,
)

_ROLES: Final[dict[str, MessageRole]] = {
    "system": "system",
    "developer": "developer",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
}
# Native fields that identify an item, and fields no tag question can read.
_IDENTITY_FIELDS: Final = frozenset({"type", "id", "call_id", "tool_use_id", "name", "namespace"})
_REASONING_TYPES: Final = frozenset({"reasoning", "thinking", "redacted_thinking", "compaction"})
_OPAQUE_FIELDS: Final = frozenset({"encrypted_content", "encrypted_index", "signature"})
_IMAGE_TYPES: Final = frozenset({"image", "input_image", "computer_screenshot"})
# Long runs of base64 characters with no spaces are binary payloads, not readable text.
_BINARY: Final = re.compile(r"[A-Za-z0-9+/_=-]{256,}")
_TEXT_PARTS: Final = frozenset({"text", "input_text", "output_text"})
_IMAGE_PARTS: Final = frozenset({"image", "image_url", "input_image"})


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
    readable, partial = readable_responses(body) if protocol == "responses" else (body, {})
    # A reply that hit its limit before producing anything is a valid, empty terminal turn,
    # which the output decoder would reject as missing output.
    field = {"responses": "output", "messages": "content"}.get(protocol)
    empty = field is not None and readable.get(field) == []
    output = [] if empty else output_messages(protocol, readable, capture_output)
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
        truncated=finished_by_length(protocol, body),
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

    A Chat refusal becomes the assistant's text, generated Chat ``images`` become image
    markers, and a failed tool result keeps the gateway's ``[tool error]`` prefix, so all stay
    visible to tag questions.

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
    images = value.get("images")
    if role == "assistant" and isinstance(images, list) and images:
        text = content if isinstance(content, str) else ""
        content = (TagTextPart(text=text), *(TagImagePart() for _ in images))
    refusal = value.get("refusal")
    if role == "assistant" and not content and isinstance(refusal, str):
        content = refusal
    if role == "tool" and value.get("tool_is_error") is True:
        if isinstance(content, tuple):
            content = (TagTextPart(text=TOOL_ERROR_TEXT_PREFIX), *content)
        else:
            content = f"{TOOL_ERROR_TEXT_PREFIX}{content or ''}"
    call_id = value.get("tool_call_id")
    name = value.get("name", value.get("provider_tool_name"))
    return TagMessage(
        role=role,
        content=content,
        tool_calls=_tool_calls(value.get("tool_calls")) if role == "assistant" else (),
        tool_call_id=call_id if role == "tool" and isinstance(call_id, str) else None,
        name=name if isinstance(name, str) and name else None,
    )


def _native(role: MessageRole, item: JsonValue) -> TagMessage | None:
    """Keep what a tag question can read from a provider-native item.

    Items are read by shape, not by name. Cited text keeps its text. A result item (role
    ``tool``, or a type ending in ``_output`` or ``_result``) becomes a tool result, and any
    other assistant item except reasoning becomes a tool call named after the item. Both carry
    every readable field (outputs, errors, failed statuses, executed code), while encrypted
    fields, signatures and binary payloads are dropped and images become markers.

    Args:
        role: The carrying message's role.
        item: The native item or block.

    Returns:
        The readable message, or None for reasoning and items with nothing readable.
    """
    if not isinstance(item, dict):
        return None
    kind = str(item.get("type"))
    if kind == "text" and isinstance(text := item.get("text"), str):
        return TagMessage(role="assistant", content=text) if role == "assistant" else None
    call_id = item.get("call_id", item.get("tool_use_id"))
    fields = {
        key: field
        for key, field in item.items()
        if key not in _IDENTITY_FIELDS and not (key == "status" and field == "completed")
    }
    if role == "tool" or kind.endswith(("_output", "_result")):
        readable, images = _readable(_sole(fields, ("output", "content")))
        text = readable if isinstance(readable, str) else _compact_json(readable)
        return TagMessage(
            role="tool",
            content=(TagTextPart(text=text), *(TagImagePart() for _ in range(images)))
            if images
            else text,
            tool_call_id=call_id if isinstance(call_id, str) else None,
        )
    if role != "assistant" or kind in _REASONING_TYPES or not fields:
        return None
    readable, images = _readable(_sole(fields, ("input", "arguments")))
    call: JsonObject = {
        "id": call_id if isinstance(call_id, str) else item.get("id"),
        "name": item.get("name", kind.removesuffix("_call")),
        "namespace": item.get("namespace"),
        "arguments": readable,
    }
    calls = _tool_calls([call])
    if not calls:
        return None
    markers = tuple(TagImagePart() for _ in range(images))
    return TagMessage(role="assistant", content=markers or None, tool_calls=calls)


def _sole(fields: JsonObject, names: tuple[str, ...]) -> JsonValue:
    """Return a lone payload field's value, or every field when there are others.

    Args:
        fields: An item's readable fields.
        names: Payload field names, such as ``output`` or ``input``.

    Returns:
        The payload value when it is the only field, otherwise all the fields.
    """
    if len(fields) == 1 and next(iter(fields)) in names:
        return next(iter(fields.values()))
    return fields


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
        named = [
            nested
            for nested in (call.get("function"), call.get("custom"))
            if isinstance(nested, dict) and nested.get("name")
        ]
        source = named[0] if named else call
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
    "exchange_from_capture_parts",
    "exchange_from_capture_record",
    "exchange_from_captured",
]
