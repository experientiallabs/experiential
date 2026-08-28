"""Tests for completed OpenAI response assembly and continuation history."""

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
from exp.runtime.models.providers.fireworks import decode_reasoning_content
from exp.runtime.openai_protocol.response import assistant_message, completed_body

_FIREWORKS_MODELS = (
    "accounts/fireworks/models/deepseek-v4-flash-0731",
    "accounts/fireworks/models/glm-5p2",
    "accounts/fireworks/models/kimi-k2p7-code",
)
_ROUTE_SHA256 = "a" * 64


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
def test_non_streaming_chat_and_internal_continuation_share_the_same_carrier(
    model: str,
) -> None:
    """DeepSeek, GLM, and Kimi expose and retain one identical route-bound token."""
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
    )
    choices = cast("list[JsonObject]", body["choices"])
    public_message = cast("JsonObject", choices[0]["message"])
    public_block = decode_reasoning_content(str(public_message["reasoning_content"]))
    retained = assistant_message(events)

    assert public_block.route_sha256 == _ROUTE_SHA256
    assert public_block.content == "first private reasoning"
    assert retained is not None
    assert retained.provider_reasoning == (public_block,)


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
