"""Tests for routing the Responses ``max_tool_calls`` budget."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayProviderNativeTool,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.gateway.tool_contracts import GatewayAllowedToolsChoice, GatewayNamedToolChoice
from exp.runtime.gateway.tool_search.contracts import GatewayToolSearch
from exp.runtime.gateway.web_search.contracts import GatewayWebSearch
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.openai_payloads import openai_responses_stream_payload
from exp.runtime.models.providers.tool_call_budget import (
    TOOL_CHOICE_CLEARED,
    remaining_tool_calls,
    shape_tool_call_budget,
)

_NATIVE = GatewayWireProfile(dialect="openai_responses", url="https://openai.test")
_PLAN = GatewayWireProfile(
    dialect="openai_responses",
    url="https://chatgpt.com/backend-api/codex/responses",
    omits_output_token_limit=True,
)
_CHAT = GatewayWireProfile(dialect="openai_compatible", url="https://chat.test")


def _request(budget: int | None, **updates: object) -> GatewayRequest:
    """Build a Responses request: a hosted tool, a function, then a custom tool."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="go"),),
        tools=(GatewayToolDefinition(name="f", parameters={"type": "object"}),),
        provider_native_tools=(
            GatewayProviderNativeTool(index=0, tool={"type": "web_search"}),
            GatewayProviderNativeTool(index=2, tool={"type": "custom", "name": "exec"}),
        ),
        max_tool_calls=budget,
        stream=True,
        include_usage=True,
    )
    return request.model_copy(update=updates)


def _shape(
    request: GatewayRequest, *profiles: GatewayWireProfile
) -> tuple[dict[str, object], list[str]]:
    """Run the shaping step and return its updates and disclosures."""
    updates: dict[str, object] = {}
    ignored: list[str] = []
    shape_tool_call_budget(profiles, request, provider_updates=updates, ignored=ignored)
    return updates, ignored


def test_remaining_budget_subtracts_the_gateway_calls() -> None:
    """The caller's cap stays put; the gateway's own calls are counted beside it."""
    assert remaining_tool_calls(_request(None)) is None
    assert remaining_tool_calls(_request(3, gateway_tool_calls=1)) == 2
    assert remaining_tool_calls(_request(1, gateway_tool_calls=1)) == 0


def test_a_native_route_receives_the_unspent_remainder_over_hosted_tools() -> None:
    """The provider runs its hosted tools, so it enforces what the gateway left."""
    assert _shape(_request(None), _NATIVE) == ({}, [])
    assert _shape(_request(3, gateway_tool_calls=1), _NATIVE) == ({"max_tool_calls": 2}, [])
    # With no hosted tool declared the provider can run none: nothing to send.
    no_hosted = _request(3, provider_native_tools=())
    assert _shape(no_hosted, _NATIVE) == ({"max_tool_calls": None}, [])


def test_a_foreign_route_never_sends_the_budget_and_discloses_nothing() -> None:
    """Off the native wire the gateway's own planners honor the budget."""
    assert _shape(_request(2), _NATIVE, _CHAT) == ({"max_tool_calls": None}, [])


def test_a_plan_backend_cannot_carry_a_positive_cap_over_hosted_tools() -> None:
    """The plan backend refuses the field, so its rung narrows out like an output ceiling."""
    with pytest.raises(ProviderParameterError) as captured:
        _shape(_request(2), _PLAN)
    assert captured.value.param == "max_tool_calls"
    # Without hosted tools, or with nothing left, it is honored without the field.
    assert _shape(_request(2, provider_native_tools=()), _PLAN) == ({"max_tool_calls": None}, [])
    updates, _ = _shape(_request(0), _PLAN)
    assert updates["max_tool_calls"] is None
    # A no-fallback preview strips the hosted search carrier but keeps the
    # declaration, which the plan rung would still serve: it narrows out there too.
    preview = _request(
        2,
        provider_native_tools=(),
        web_search=GatewayWebSearch(declared_as="responses_tool"),
    )
    with pytest.raises(ProviderParameterError):
        _shape(preview, _PLAN)
    assert _shape(preview, _NATIVE) == ({"max_tool_calls": None}, [])


def test_a_choice_that_excludes_every_hosted_tool_needs_no_provider_cap() -> None:
    """With hosted tools unreachable no built-in call can happen, even on a plan rung."""
    assert _shape(_request(2, tool_choice="none"), _PLAN) == ({"max_tool_calls": None}, [])
    custom_only = GatewayAllowedToolsChoice(
        mode="auto", provider_entries=({"type": "custom", "name": "exec"},)
    )
    assert _shape(_request(2, tool_choice=custom_only), _PLAN) == ({"max_tool_calls": None}, [])
    forced = GatewayNamedToolChoice(name="f")
    assert _shape(_request(2, tool_choice=forced), _PLAN) == ({"max_tool_calls": None}, [])
    search_allowed = GatewayAllowedToolsChoice(
        mode="auto", provider_entries=({"type": "web_search"},)
    )
    with pytest.raises(ProviderParameterError):
        _shape(_request(2, tool_choice=search_allowed), _PLAN)


def test_a_spent_budget_withholds_hosted_tools_and_keeps_the_rest_in_order() -> None:
    """Zero is never sent: the hosted tools it forbids are withheld, survivors close up."""
    request = _request(0)
    updates, ignored = _shape(request, _NATIVE)
    assert updates["max_tool_calls"] is None
    assert ignored == ["tools.web_search->dropped(max_tool_calls)"]
    provider = request.model_copy(update=updates)
    payload = openai_responses_stream_payload("gpt-6-luna", provider, supports_temperature=False)
    tools = payload["tools"]
    assert isinstance(tools, list)
    assert [tool["type"] for tool in tools if isinstance(tool, dict)] == ["function", "custom"]
    assert "max_tool_calls" not in payload


def test_a_spent_budget_narrows_or_clears_the_tool_choice() -> None:
    """An allowed-tools selector keeps only caller tools, and is cleared when none remain."""
    narrowed = GatewayAllowedToolsChoice(
        mode="required",
        provider_entries=({"type": "web_search"}, {"type": "custom", "name": "exec"}),
    )
    updates, _ = _shape(_request(0, tool_choice=narrowed), _NATIVE)
    choice = updates["tool_choice"]
    assert isinstance(choice, GatewayAllowedToolsChoice)
    assert choice.provider_entries == ({"type": "custom", "name": "exec"},)

    hosted_only = _request(
        0,
        tools=(),
        provider_native_tools=(GatewayProviderNativeTool(index=0, tool={"type": "web_search"}),),
        tool_choice="required",
    )
    updates, ignored = _shape(hosted_only, _NATIVE)
    assert updates["tool_choice"] is None
    assert TOOL_CHOICE_CLEARED in ignored
    updates, _ = _shape(_request(0, tool_choice="required"), _NATIVE)
    assert "tool_choice" not in updates


def test_a_spent_budget_withholding_native_tool_search_loads_deferred_tools() -> None:
    """Without the provider's search a deferred function would be unreachable."""
    request = _request(
        0,
        tools=(GatewayToolDefinition(name="f", parameters={"type": "object"}, defer_loading=True),),
        provider_native_tools=(GatewayProviderNativeTool(index=1, tool={"type": "tool_search"}),),
        tool_search=GatewayToolSearch(declared_as="responses_tool", tool_type="tool_search"),
    )
    updates, ignored = _shape(request, _NATIVE)
    assert updates["tool_search"] is None
    # The provider body carries every function loaded and no deferral marker.
    payload = openai_responses_stream_payload(
        "gpt-6-luna", request.model_copy(update=updates), supports_temperature=False
    )
    tools_on_wire = payload["tools"]
    assert isinstance(tools_on_wire, list)
    assert [tool.get("name") for tool in tools_on_wire if isinstance(tool, dict)] == ["f"]
    assert all(isinstance(tool, dict) and "defer_loading" not in tool for tool in tools_on_wire)
    assert ignored == ["tools.tool_search->dropped(max_tool_calls)"]
    tools = updates["tools"]
    assert isinstance(tools, tuple)
    assert [tool.defer_loading for tool in tools if isinstance(tool, GatewayToolDefinition)] == [
        None
    ]
