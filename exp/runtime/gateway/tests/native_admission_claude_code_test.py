"""Claude Code request shapes on an effort-ladder lane admit, disclosed, instead of refusing.

Pins the behavior youtop.ai's Claude Code and opencode sessions hit on
2026-10-17 (``xhigh`` effort and numeric thinking budgets refused by name on
deepseek-v4.1-flash-uncensored, a ``none/low/high/max`` ladder): an effort the
route cannot carry snaps to the nearest tier, a budget maps onto the ladder,
and Claude Code history the provider cannot carry is dropped or folded, each
disclosed through ``x-experiential-ignored-parameters``. Content the model
would need to see (an image or PDF in a user turn on a text-only lane) still
refuses by name.
"""

from __future__ import annotations

from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.catalog import GatewayDeploymentCapabilities, GatewayDeploymentMetadata
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayRequest
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_admission import admitted_route_requests
from exp.runtime.gateway.native_admission_test import (
    _AdmissionCoercionCounter,
    _deployment,
    _mixed_route,
)
from exp.runtime.gateway.native_dispatch import NativeWireClient
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderCapabilityError

_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
)
_TOOLS = [{"name": "Read", "description": "Read a file", "input_schema": {"type": "object"}}]


def _admit(body: JsonObject) -> tuple[GatewayRequest, GatewayRequest]:
    """Admit one Messages body on a text-only none/low/high/max ladder lane."""
    # Imported here: the Messages decoder imported first at module scope closes
    # an import cycle through the reasoning carrier.
    from exp.runtime.anthropic_protocol.requests import decode_messages

    request = decode_messages(body).request.model_copy(update={"include_usage": True})
    metadata = GatewayDeploymentMetadata(
        capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True, supports_streaming_tool_arguments=True
        )
    )
    route = _mixed_route(
        "maximize_availability",
        (_deployment("ds", provider="experiential_cloud", gateway=metadata),),
        GatewayApiSurface.MESSAGES,
    )
    wire = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://ec.test/v1/chat/completions",
        model_id="deepseek-v4.1-flash-uncensored",
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
        supported_reasoning_efforts=("none", "low", "high", "max"),
    )
    _route, _wires, public, provider, _placement = admitted_route_requests(
        route,
        ((wire, cast(NativeWireClient, object())),),
        request,
        accounting=cast(NativeAttemptAccounting, _AdmissionCoercionCounter()),
        authorization=route.snapshot.authorization,
    )
    return public, provider


def _body(**extra: object) -> JsonObject:
    """Build one Claude Code-shaped streaming Messages body."""
    body: JsonObject = {
        "model": "deepseek-v4.1-flash-uncensored",
        "max_tokens": 32000,
        "stream": True,
        "tools": _TOOLS,
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(cast(JsonObject, extra))
    return body


@pytest.mark.parametrize(
    ("extra", "effort", "disclosure"),
    [
        ({"output_config": {"effort": "xhigh"}}, "high", "reasoning_effort->high"),
        (
            {"output_config": {"effort": "xhigh"}, "thinking": {"type": "adaptive"}},
            "high",
            "reasoning_effort->high",
        ),
        (
            {"output_config": {"effort": "medium"}, "thinking": {"type": "adaptive"}},
            "low",
            "reasoning_effort->low",
        ),
        (
            {"thinking": {"type": "enabled", "budget_tokens": 31999}},
            "high",
            "thinking.budget_tokens->reasoning_effort:high(route_effort_ladder)",
        ),
        (
            {"thinking": {"type": "enabled", "budget_tokens": 4000}},
            "low",
            "thinking.budget_tokens->reasoning_effort:low(route_effort_ladder)",
        ),
    ],
)
def test_claude_code_depth_controls_snap_onto_the_ladder(
    extra: JsonObject, effort: str, disclosure: str
) -> None:
    """An effort or budget the ladder lacks is served at the nearest tier, disclosed."""
    public, provider = _admit(_body(**extra))
    assert provider.reasoning_effort == effort
    assert disclosure in public.ignored_parameters


_ASSISTANT_TOOL_USE = {
    "role": "assistant",
    "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
}


@pytest.mark.parametrize(
    ("messages", "disclosure"),
    [
        (
            [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "t", "signature": "sig"},
                        {"type": "text", "text": "a"},
                    ],
                },
                {"role": "user", "content": "more"},
            ],
            "messages.thinking->dropped(unsupported_by_provider)",
        ),
        (
            [
                {"role": "user", "content": "hi"},
                _ASSISTANT_TOOL_USE,
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": _PNG,
                                    },
                                }
                            ],
                        }
                    ],
                },
            ],
            "messages.content.tool_result.image->placeholder",
        ),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
                    ],
                }
            ],
            "messages.content.cache_control->not_forwarded(provider_decides_caching; "
            "cache reads reported in usage.cache_read_input_tokens)",
        ),
        (
            [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "server_tool_use",
                            "id": "s1",
                            "name": "web_search",
                            "input": {"query": "x"},
                        },
                        {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []},
                        {"type": "text", "text": "a"},
                    ],
                },
                {"role": "user", "content": "more"},
            ],
            "messages.server_tool_blocks->dropped(unsupported_by_provider)",
        ),
    ],
)
def test_claude_code_history_the_lane_cannot_carry_degrades_disclosed(
    messages: list[JsonObject], disclosure: str
) -> None:
    """Replayed Claude Code history is dropped or folded with disclosure, not refused."""
    public, _provider = _admit(_body(messages=messages))
    assert disclosure in public.ignored_parameters


@pytest.mark.parametrize(
    ("block", "capability"),
    [
        (
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": _PNG},
            },
            "image_input",
        ),
        (
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBERi0xLjQK",
                },
            },
            "pdf_input",
        ),
    ],
)
def test_user_content_a_text_only_lane_cannot_see_still_refuses(
    block: JsonObject, capability: str
) -> None:
    """An image or PDF the caller asks about is never silently dropped."""
    messages = [{"role": "user", "content": [block, {"type": "text", "text": "what is this"}]}]
    with pytest.raises(ProviderCapabilityError) as raised:
        _admit(_body(messages=messages))
    assert raised.value.capability == capability
