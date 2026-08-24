"""OpenAI response assembly from normalized serving events."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from exp.common.core.artifacts import JsonObject
from exp.common.models import AssistantAction

if TYPE_CHECKING:
    from exp.runtime.anthropic_protocol.encoding import MessagesSseEncoder
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayMessage,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.streaming import (
    ChatSseEncoder,
    ResponsesSseEncoder,
    stable_public_id,
)


def stream_encoder(
    *,
    request: GatewayRequest,
    authorization: AuthorizationSnapshot,
    created_at: float,
) -> ChatSseEncoder | ResponsesSseEncoder | MessagesSseEncoder:
    """Create the surface-specific stateful SSE encoder."""
    if request.surface == GatewayApiSurface.CHAT_COMPLETIONS:
        return ChatSseEncoder(
            request_id=authorization.request_id,
            model=authorization.alias,
            created_at=int(created_at),
            include_usage=request.include_usage,
        )
    if request.surface == GatewayApiSurface.MESSAGES:
        # Imported here to break the real package cycle: this module is
        # imported by the openai_protocol package initializer, and the
        # anthropic_protocol modules import openai_protocol errors and IDs.
        from exp.runtime.anthropic_protocol.encoding import MessagesSseEncoder

        return MessagesSseEncoder(
            request_id=authorization.request_id,
            model=authorization.alias,
        )
    return ResponsesSseEncoder(
        request_id=authorization.request_id,
        model=authorization.alias,
        created_at=created_at,
        request=request,
    )


def capture_frame(
    buffer: bytearray,
    data: bytes,
    replayable: bool,
    *,
    maximum_bytes: int,
) -> bool:
    """Append one frame while it remains within the replay capture ceiling."""
    if not replayable or len(buffer) + len(data) > maximum_bytes:
        return False
    buffer.extend(data)
    return True


def is_terminal(event: GatewayEvent) -> bool:
    """Return whether one normalized event ends provider execution."""
    return event.kind in {
        GatewayEventKind.COMPLETED,
        GatewayEventKind.INCOMPLETE,
        GatewayEventKind.FAILED,
    }


def completed_body(
    *,
    request: GatewayRequest,
    request_id: str,
    model: str,
    created_at: float,
    events: tuple[GatewayEvent, ...],
) -> JsonObject:
    """Build one non-streaming public result from bounded normalized events."""
    if request.surface == GatewayApiSurface.MESSAGES:
        # Imported here to break the real package cycle (see stream_encoder).
        from exp.runtime.anthropic_protocol.encoding import completed_messages_body

        return completed_messages_body(request_id=request_id, model=model, events=events)
    if request.surface == GatewayApiSurface.RESPONSES:
        encoder = ResponsesSseEncoder(
            request_id=request_id,
            model=model,
            created_at=created_at,
            request=request,
        )
        encoder.start()
        frames: tuple[str, ...] = ()
        for event in events:
            produced = encoder.feed(event)
            if is_terminal(event):
                frames = produced
        if not frames:
            raise OpenAIProtocolError(
                status_code=502,
                code="all_routes_failed",
                message="Responses encoding produced no terminal result.",
                error_type="api_error",
            )
        payload = json.loads(frames[-1].partition("data: ")[2])
        return cast(JsonObject, payload["response"])
    terminal = next(event for event in reversed(events) if is_terminal(event))
    text = "".join(
        event.text_delta or "" for event in events if event.kind == GatewayEventKind.TEXT_DELTA
    )
    refusal = "".join(
        event.text_delta or "" for event in events if event.kind == GatewayEventKind.REFUSAL_DELTA
    )
    tool_calls = tuple(
        event.tool_call
        for event in events
        if event.kind == GatewayEventKind.TOOL_CALL_COMPLETED and event.tool_call is not None
    )
    message: JsonObject = {
        "role": "assistant",
        "content": text or None,
        "refusal": refusal or None,
        "tool_calls": [
            {
                "id": tool.call_id,
                "type": "function",
                "function": {"name": tool.name, "arguments": tool.arguments_json()},
            }
            for tool in tool_calls
        ]
        or None,
    }
    usage = next(
        (
            event.usage
            for event in reversed(events)
            if event.usage is not None and event.usage.has_token_counts
        ),
        None,
    )
    return {
        "id": stable_public_id("chatcmpl", request_id),
        "object": "chat.completion",
        "created": int(created_at),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": (
                    "length"
                    if terminal.kind == GatewayEventKind.INCOMPLETE
                    else "tool_calls"
                    if tool_calls
                    else "stop"
                ),
                "logprobs": None,
            }
        ],
        "usage": chat_usage(usage),
    }


def chat_usage(usage: GatewayUsage | None) -> JsonObject | None:
    """Encode normalized usage in the Chat completion shape."""
    if usage is None or not usage.has_token_counts:
        return None
    assert usage.input_tokens is not None
    assert usage.output_tokens is not None
    details: JsonObject = {}
    if usage.cached_input_tokens is not None:
        details["cached_tokens"] = usage.cached_input_tokens
    output_details: JsonObject = {}
    if usage.reasoning_tokens is not None:
        output_details["reasoning_tokens"] = usage.reasoning_tokens
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
        "prompt_tokens_details": details or None,
        "completion_tokens_details": output_details or None,
    }


def assistant_message(events: tuple[GatewayEvent, ...]) -> GatewayMessage | None:
    """Build continuation history without converting typed refusals into assistant text."""
    if any(event.kind == GatewayEventKind.REFUSAL_DELTA for event in events):
        return None
    text = "".join(
        event.text_delta or "" for event in events if event.kind == GatewayEventKind.TEXT_DELTA
    )
    tool_calls = tuple(
        event.tool_call
        for event in events
        if event.kind == GatewayEventKind.TOOL_CALL_COMPLETED and event.tool_call is not None
    )
    if not text and not tool_calls:
        return None
    action = AssistantAction(content=text or None, tool_calls=tool_calls)
    return GatewayMessage(
        role="assistant",
        content=action.content,
        tool_calls=action.tool_calls,
    )
