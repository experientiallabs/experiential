"""OpenAI response assembly from normalized serving events."""

from __future__ import annotations

import json
from typing import cast

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayRequest,
    GatewayUsage,
)
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
    probability_value = _chat_logprobs(events)
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
                "logprobs": probability_value,
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


def _chat_logprobs(events: tuple[GatewayEvent, ...]) -> JsonObject | None:
    """Aggregate ordered choice probability observations into Chat output."""
    observations = tuple(
        event.choice_logprobs_delta
        for event in events
        if event.kind == GatewayEventKind.CHOICE_LOGPROBS_DELTA
    )
    if not observations:
        return None
    if any(observation is None for observation in observations):
        raise OpenAIProtocolError(
            status_code=502,
            code="invalid_provider_stream",
            message="Chat probability event omitted its payload.",
            error_type="api_error",
        )
    observations = tuple(observation for observation in observations if observation is not None)
    if any(observation.choice_index != 0 for observation in observations):
        raise OpenAIProtocolError(
            status_code=502,
            code="invalid_provider_stream",
            message="Chat probability events support choice index 0 only.",
            error_type="api_error",
        )
    values = tuple(
        observation.logprobs for observation in observations if observation.logprobs is not None
    )
    if not values:
        return None
    content = tuple(record for value in values for record in (value.content or ()))
    refusal = tuple(record for value in values for record in (value.refusal or ()))
    # A probability object is retained even when both channels are null. This
    # distinguishes an observed empty update from no provider metadata.
    return {
        "content": [record.model_dump(mode="json") for record in content]
        if any(value.content is not None for value in values)
        else None,
        "refusal": [record.model_dump(mode="json") for record in refusal]
        if any(value.refusal is not None for value in values)
        else None,
    }


def _ignored_parameters_extension(request: GatewayRequest) -> JsonObject:
    """Expose accepted-but-ignored controls so compatibility behavior is never silent."""
    if not request.ignored_parameters:
        return {}
    return {"x-experiential-ignored-parameters": list(request.ignored_parameters)}
