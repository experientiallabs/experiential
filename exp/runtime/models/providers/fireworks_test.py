"""Fireworks opaque tool-continuation contract regressions."""

from __future__ import annotations

from typing import Literal

import pytest

from exp.common.models import ToolCall
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
    OpaqueReasoningContentBlock,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.fireworks import (
    prepare_gateway_reasoning_history,
    require_responses_continuation_channel,
)
from exp.runtime.models.providers.streaming_requests import dialect_stream_payload

_ROUTE_SHA256 = "a" * 64
_OTHER_ROUTE_SHA256 = "b" * 64


def _call(call_id: str = "call-one") -> ToolCall:
    """Build one exact provider tool call."""
    return ToolCall(
        call_id=call_id,
        name="lookup",
        arguments={"q": "x"},
        raw_arguments='{ "q" : "x" }',
    )


def _assistant(
    *,
    call_id: str = "call-one",
    route_sha256: str = _ROUTE_SHA256,
    content: str = "private Fireworks reasoning",
) -> GatewayMessage:
    """Build one carrier-bound assistant tool turn."""
    return GatewayMessage(
        role="assistant",
        tool_calls=(_call(call_id),),
        provider_reasoning=(
            OpaqueReasoningContentBlock(
                route_sha256=route_sha256,
                content=content,
            ),
        ),
    )


def _profile(route_sha256: str | None = _ROUTE_SHA256) -> GatewayWireProfile:
    """Build one Fireworks Chat wire profile."""
    return GatewayWireProfile(
        dialect="openai_compatible",
        url="https://api.fireworks.ai/inference/v1/chat/completions",
        headers={"Authorization": "Bearer test-secret"},
        model_id="accounts/fireworks/models/deepseek-v4-flash-0731",
        fireworks_reasoning_route_sha256=route_sha256,
    )


@pytest.mark.parametrize(
    ("store", "include_encrypted", "tool_choice"),
    (
        (True, False, "auto"),
        (None, False, "required"),
        (False, True, "auto"),
        (False, False, "none"),
    ),
)
def test_fireworks_responses_accepts_each_safe_continuation_shape(
    store: bool | None,
    include_encrypted: bool,
    tool_choice: Literal["auto", "none", "required"],
) -> None:
    """Storage, an encrypted carrier, or disabled tools each make replay safe."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="Use a tool"),),
        tools=(
            GatewayToolDefinition(
                name="lookup",
                parameters={"type": "object"},
            ),
        ),
        tool_choice=tool_choice,
        response_store=store,
        include_encrypted_reasoning=include_encrypted,
    )

    require_responses_continuation_channel(request)


def test_fireworks_responses_rejects_tool_capable_turn_without_replay_channel() -> None:
    """A stateless tool turn cannot expose a call without its authenticated carrier."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="Use a tool"),),
        tools=(
            GatewayToolDefinition(
                name="lookup",
                parameters={"type": "object"},
            ),
        ),
        tool_choice="auto",
        response_store=False,
    )

    with pytest.raises(ProviderParameterError) as raised:
        require_responses_continuation_channel(request)
    assert raised.value.param == "store"


@pytest.mark.parametrize(
    "surface",
    (GatewayApiSurface.CHAT_COMPLETIONS, GatewayApiSurface.RESPONSES),
)
def test_chat_and_responses_continuations_replay_exact_reasoning_on_issuing_route(
    surface: GatewayApiSurface,
) -> None:
    """Both public surfaces replay plaintext only on the authenticated Fireworks wire."""
    request = GatewayRequest(
        surface=surface,
        messages=(
            GatewayMessage(role="user", content="Use a tool"),
            _assistant(),
            GatewayMessage(role="tool", content="done", tool_call_id="call-one"),
        ),
    )

    payload = dialect_stream_payload(_profile(), request)

    assert payload["reasoning_history"] == "interleaved"
    messages = payload["messages"]
    assert isinstance(messages, list)
    assistant = messages[1]
    assert isinstance(assistant, dict)
    assert assistant["reasoning_content"] == "private Fireworks reasoning"
    tool_calls = assistant["tool_calls"]
    assert isinstance(tool_calls, list)
    tool_call = tool_calls[0]
    assert isinstance(tool_call, dict)
    function = tool_call["function"]
    assert isinstance(function, dict)
    assert function["arguments"] == '{ "q" : "x" }'


def test_active_continuation_rejects_cross_route_replay() -> None:
    """An authenticated carrier cannot replay on a different Fireworks route."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(role="user", content="Use a tool"),
            _assistant(route_sha256=_OTHER_ROUTE_SHA256),
            GatewayMessage(role="tool", content="done", tool_call_id="call-one"),
        ),
    )

    with pytest.raises(ProviderParameterError) as raised:
        dialect_stream_payload(_profile(), request)
    assert raised.value.param == "messages.reasoning_content"


def test_stale_reasoning_is_stripped_at_the_next_user_boundary() -> None:
    """A prior turn's carrier is stripped before a new user request."""
    messages = (
        GatewayMessage(role="user", content="First turn"),
        _assistant(content="stale private reasoning"),
        GatewayMessage(role="tool", content="done", tool_call_id="call-one"),
        GatewayMessage(role="user", content="Second turn"),
    )

    prepared, active = prepare_gateway_reasoning_history(
        messages,
        route_sha256=_OTHER_ROUTE_SHA256,
    )

    assert not active
    assert prepared[1].provider_reasoning == ()


def test_active_reasoning_rejects_duplicate_tool_results() -> None:
    """Every carrier-bound call must receive exactly one linked result."""
    messages = (
        GatewayMessage(role="user", content="Use a tool"),
        _assistant(),
        GatewayMessage(role="tool", content="first", tool_call_id="call-one"),
        GatewayMessage(role="tool", content="duplicate", tool_call_id="call-one"),
    )

    with pytest.raises(ProviderParameterError, match="exactly one result"):
        prepare_gateway_reasoning_history(messages, route_sha256=_ROUTE_SHA256)
