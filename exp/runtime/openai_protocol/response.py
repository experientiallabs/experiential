"""OpenAI response assembly from normalized serving events."""

from __future__ import annotations

import json
from typing import cast

from exp.common.core.artifacts import JsonObject
from exp.common.models import OpaqueReasoningContentBlock
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayMessage,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.gateway.reasoning_carrier import parse_reasoning_content_carrier
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.streaming import (
    ResponsesSseEncoder,
    stable_public_id,
)


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
    reasoning_content_carrier: str | None = None,
) -> JsonObject:
    """Build one non-streaming public result from bounded normalized events.

    The Anthropic Messages surface is rendered by the native data plane, so
    this builder serves only the OpenAI Chat Completions and Responses shapes.
    """
    if request.surface == GatewayApiSurface.RESPONSES:
        encoder = ResponsesSseEncoder(
            request_id=request_id,
            model=model,
            created_at=created_at,
            request=request,
        )
        if reasoning_content_carrier is not None:
            encoder.set_reasoning_content_carrier(reasoning_content_carrier)
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
    reasoning_content = _reasoning_content_block(events)
    if reasoning_content is not None and tool_calls and terminal.kind == GatewayEventKind.COMPLETED:
        if reasoning_content_carrier is None:
            raise OpenAIProtocolError(
                status_code=502,
                code="invalid_provider_stream",
                message="Chat reasoning content was not sealed by the gateway authority.",
                error_type="api_error",
            )
        message["reasoning_content"] = reasoning_content_carrier
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
        **_ignored_parameters_extension(request),
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


def _ignored_parameters_extension(request: GatewayRequest) -> JsonObject:
    """Expose accepted-but-ignored controls so compatibility behavior is never silent."""
    if not request.ignored_parameters:
        return {}
    return {"x-experiential-ignored-parameters": list(request.ignored_parameters)}


def assistant_message(
    events: tuple[GatewayEvent, ...],
    *,
    reasoning_content_carrier: str | None = None,
) -> GatewayMessage | None:
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
    terminal = next((event for event in reversed(events) if is_terminal(event)), None)
    reasoning_content = _reasoning_content_block(events)
    provider_reasoning = ()
    if (
        reasoning_content is not None
        and tool_calls
        and terminal is not None
        and terminal.kind == GatewayEventKind.COMPLETED
    ):
        if reasoning_content_carrier is None:
            raise OpenAIProtocolError(
                status_code=502,
                code="invalid_provider_stream",
                message="Responses reasoning content was not sealed by gateway authority.",
                error_type="api_error",
            )
        provider_reasoning = (parse_reasoning_content_carrier(reasoning_content_carrier),)
    if not text and not tool_calls:
        return None
    return GatewayMessage(
        role="assistant",
        content=text or None,
        tool_calls=tool_calls,
        provider_reasoning=provider_reasoning,
    )


def _reasoning_content_block(
    events: tuple[GatewayEvent, ...],
) -> OpaqueReasoningContentBlock | None:
    """Aggregate one route-stable opaque Fireworks reasoning payload."""
    route_sha256: str | None = None
    parts: list[str] = []
    for event in events:
        if event.kind != GatewayEventKind.REASONING_CONTENT_DELTA:
            continue
        if event.reasoning_content_route_sha256 is None or event.text_delta is None:
            raise OpenAIProtocolError(
                status_code=502,
                code="invalid_provider_stream",
                message="Chat reasoning content omitted route identity or text.",
                error_type="api_error",
            )
        if route_sha256 is not None and route_sha256 != event.reasoning_content_route_sha256:
            raise OpenAIProtocolError(
                status_code=502,
                code="invalid_provider_stream",
                message="Chat reasoning content changed provider route.",
                error_type="api_error",
            )
        route_sha256 = event.reasoning_content_route_sha256
        parts.append(event.text_delta)
    if route_sha256 is None or not parts:
        return None
    return OpaqueReasoningContentBlock(route_sha256=route_sha256, content="".join(parts))
