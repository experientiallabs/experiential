"""Tests for completed OpenAI response assembly and continuation history."""

import base64
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.reasoning_carrier import FIREWORKS_REASONING_CONTENT_PREFIX
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.response import assistant_message, completed_body

_FIREWORKS_MODELS = (
    "accounts/fireworks/models/deepseek-v4-flash-0731",
    "accounts/fireworks/models/glm-5p2",
    "accounts/fireworks/models/kimi-k2p7-code",
)
_ROUTE_SHA256 = "a" * 64
_CARRIER = (
    f"{FIREWORKS_REASONING_CONTENT_PREFIX}"
    f"{base64.urlsafe_b64encode(b'fireworks-rung').rstrip(b'=').decode()}:"
    f"{base64.urlsafe_b64encode(b'opaque-envelope').rstrip(b'=').decode()}"
)


def _events(terminal: GatewayEventKind = GatewayEventKind.COMPLETED) -> tuple[GatewayEvent, ...]:
    """Build one buffered Fireworks tool-call result."""
    return (
        GatewayEvent(
            kind=GatewayEventKind.REASONING_CONTENT_DELTA,
            sequence_number=0,
            text_delta="first private ",
            reasoning_content_route_sha256=_ROUTE_SHA256,
        ),
        GatewayEvent(
            kind=GatewayEventKind.REASONING_CONTENT_DELTA,
            sequence_number=1,
            text_delta="reasoning",
            reasoning_content_route_sha256=_ROUTE_SHA256,
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_COMPLETED,
            sequence_number=2,
            tool_call_index=0,
            tool_call=ToolCall(
                call_id="call-one",
                name="lookup",
                arguments={"q": 1},
                raw_arguments='{ "q": 1 }',
            ),
        ),
        GatewayEvent(kind=terminal, sequence_number=3),
    )


@pytest.mark.parametrize("model", _FIREWORKS_MODELS)
def test_non_streaming_chat_uses_only_an_injected_authenticated_carrier(
    model: str,
) -> None:
    """DeepSeek, GLM, and Kimi expose only the authority-sealed token."""
    events = _events()
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="Use a tool"),),
    )

    body = completed_body(
        request=request,
        request_id="request-one",
        model=model,
        created_at=123,
        events=events,
        reasoning_content_carrier="authenticated-carrier-v2",
    )
    choices = cast("list[JsonObject]", body["choices"])
    public_message = cast("JsonObject", choices[0]["message"])
    assert public_message["reasoning_content"] == "authenticated-carrier-v2"
    with pytest.raises(OpenAIProtocolError, match="not sealed"):
        assistant_message(events)
    retained = assistant_message(events, reasoning_content_carrier=_CARRIER)
    assert retained is not None
    assert retained.provider_reasoning[0].kind == "sealed_reasoning_content"


def test_non_streaming_responses_round_trips_the_authenticated_carrier() -> None:
    """Responses exposes the opaque carrier only through encrypted_content."""
    events = (
        *_events()[:2],
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_STARTED,
            sequence_number=2,
            tool_call_index=0,
            tool_call_id="call-one",
            tool_name="lookup",
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            sequence_number=3,
            tool_call_index=0,
            raw_arguments_delta='{ "q": 1 }',
        ),
        _events()[2].model_copy(update={"sequence_number": 4}),
        GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=5),
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="Use a tool"),),
        include_encrypted_reasoning=True,
    )

    body = completed_body(
        request=request,
        request_id="request-responses",
        model=_FIREWORKS_MODELS[0],
        created_at=123,
        events=events,
        reasoning_content_carrier=_CARRIER,
    )

    output = cast("list[JsonObject]", body["output"])
    reasoning = next(item for item in output if item["type"] == "reasoning")
    assert reasoning["encrypted_content"] == _CARRIER
    assert "first private" not in str(body)


def test_non_streaming_chat_discards_reasoning_on_incomplete_tool_output() -> None:
    """Only a completed tool action is safe to replay as agent history."""
    events = _events(GatewayEventKind.INCOMPLETE)
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="Use a tool"),),
    )

    body = completed_body(
        request=request,
        request_id="request-incomplete",
        model="coding",
        created_at=123,
        events=events,
    )
    choices = cast("list[JsonObject]", body["choices"])
    public_message = cast("JsonObject", choices[0]["message"])
    retained = assistant_message(events)

    assert "reasoning_content" not in public_message
    assert retained is not None
    assert retained.provider_reasoning == ()
