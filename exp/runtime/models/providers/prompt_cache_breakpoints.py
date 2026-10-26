"""Re-emit a caller's cache boundaries as OpenAI ``prompt_cache_breakpoint`` markers.

The canonical request carries every caller cache boundary as an ephemeral
``cache_control`` on its block (``GatewayMessage.provider_text_blocks``, a content
part, or a tool message). When the caller wrote those boundaries as OpenAI
``prompt_cache_breakpoint`` markers (``GatewayRequest.prompt_cache_breakpoints``) and
the rung's origin carries the field (``GatewayWireProfile.forwards_prompt_cache_options``),
the Chat and Responses builders hand each built message here, which marks the same
block on the wire. A text-only message that would otherwise flatten to one string
becomes its text parts so the boundary keeps its exact position.
"""

from __future__ import annotations

from typing import Literal

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayMessage
from exp.runtime.models.providers.prompt_cache_options import BREAKPOINT_MARKER
from exp.runtime.models.providers.wire_messages import retained_cache_marked_blocks

_FIELD = "prompt_cache_breakpoint"


def carries_breakpoints(message: GatewayMessage) -> bool:
    """Whether a message has any boundary this module would mark."""
    return (
        any("cache_control" in block for block in message.provider_text_blocks)
        or message.cache_control is not None
        or any(_part_markers(message))
    )


def chat_breakpoints(message: GatewayMessage, wire: JsonObject) -> JsonObject:
    """Mark one built Chat Completions message.

    Args:
        message: The canonical message ``wire`` was built from.
        wire: The Chat message JSON, built without ``cache_control`` markers.

    Returns:
        The message with its boundaries marked (``wire`` itself when it has none).
    """
    return _marked(message, wire, key="content", text_type="text")


def responses_breakpoints(message: GatewayMessage, item: JsonObject) -> JsonObject:
    """Mark one built Responses input item (a user, instruction or tool-output item).

    Args:
        message: The canonical message ``item`` was built from.
        item: The Responses input item JSON.

    Returns:
        The item with its boundaries marked (``item`` itself when it has none).
    """
    key = "output" if message.role == "tool" else "content"
    return _marked(message, item, key=key, text_type="input_text")


def _marked(
    message: GatewayMessage,
    wire: JsonObject,
    *,
    key: Literal["content", "output"],
    text_type: Literal["text", "input_text"],
) -> JsonObject:
    """Attach the breakpoint to every marked part of ``wire[key]``."""
    content = wire.get(key)
    if message.role == "tool":
        if message.cache_control is None:
            return wire
        # A tool result is one block upstream; its boundary is the result's end.
        if isinstance(content, str):
            return {**wire, key: [{"type": text_type, "text": content, _FIELD: BREAKPOINT_MARKER}]}
        if isinstance(content, list) and content and isinstance(content[-1], dict):
            return {**wire, key: [*content[:-1], {**content[-1], _FIELD: BREAKPOINT_MARKER}]}
        return wire
    if message.content_parts and isinstance(content, list):
        markers = _part_markers(message)
        if not any(markers) or len(content) != len(markers):
            return wire
        return {
            **wire,
            key: [
                {**part, _FIELD: BREAKPOINT_MARKER} if marked and isinstance(part, dict) else part
                for part, marked in zip(content, markers, strict=True)
            ],
        }
    blocks = retained_cache_marked_blocks(message.provider_text_blocks)
    if isinstance(content, str) and any("cache_control" in block for block in blocks):
        return {
            **wire,
            key: [
                {
                    "type": text_type,
                    "text": block["text"],
                    **({_FIELD: BREAKPOINT_MARKER} if "cache_control" in block else {}),
                }
                for block in blocks
            ],
        }
    return wire


def _part_markers(message: GatewayMessage) -> list[bool]:
    """Whether each content part ends a marked boundary, aligned with ``content_parts``.

    Text parts take their marker from the aligned retained text block (the Chat
    decoder's carrier) or the part itself; images and documents carry their own.
    """
    marked_blocks = iter(retained_cache_marked_blocks(message.provider_text_blocks))
    markers: list[bool] = []
    for part in message.content_parts:
        if part.kind == "text":
            block = next(marked_blocks, {})
            markers.append("cache_control" in block or part.cache_control is not None)
        elif part.kind == "image" or part.kind == "document":
            markers.append(part.cache_control is not None)
        else:
            markers.append(False)
    return markers
