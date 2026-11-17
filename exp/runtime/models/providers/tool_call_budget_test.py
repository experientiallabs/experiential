"""Tests for routing the Responses ``max_tool_calls`` budget."""

from __future__ import annotations

from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayProviderNativeTool,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.gateway.tool_contracts import GatewayAllowedToolsChoice
from exp.runtime.models.providers.tool_call_budget import (
    TOOL_CHOICE_CLEARED,
    shape_tool_call_budget,
)


def _request(budget: int | None, **updates: object) -> GatewayRequest:
    """Build a Responses request with one hosted and one caller-executed native tool."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="go"),),
        provider_native_tools=(
            GatewayProviderNativeTool(index=0, tool={"type": "web_search"}),
            GatewayProviderNativeTool(index=1, tool={"type": "custom", "name": "exec"}),
        ),
        max_tool_calls=budget,
    )
    return request.model_copy(update=updates)


def _shape(request: GatewayRequest, *, native_only: bool) -> tuple[dict[str, object], list[str]]:
    """Run the shaping step and return its updates and disclosures."""
    updates: dict[str, object] = {}
    ignored: list[str] = []
    shape_tool_call_budget(
        request, native_only=native_only, provider_updates=updates, ignored=ignored
    )
    return updates, ignored


def test_absent_or_positive_budget_on_a_native_route_is_forwarded_untouched() -> None:
    """The provider enforces a positive budget itself."""
    assert _shape(_request(None), native_only=True) == ({}, [])
    assert _shape(_request(2), native_only=True) == ({}, [])


def test_a_foreign_route_never_sends_the_budget_and_discloses_nothing() -> None:
    """Off the native wire the gateway's own planners honor the budget."""
    assert _shape(_request(2), native_only=False) == ({"max_tool_calls": None}, [])


def test_a_spent_budget_withholds_hosted_tools_on_a_native_route() -> None:
    """Zero is never sent: the hosted tools it would forbid are withheld instead."""
    updates, ignored = _shape(_request(0), native_only=True)
    assert updates["max_tool_calls"] is None
    kept = updates["provider_native_tools"]
    assert isinstance(kept, tuple)
    assert [entry.tool["type"] for entry in kept] == ["custom"]
    assert ignored == ["tools.web_search->dropped(max_tool_calls)"]


def test_a_spent_budget_narrows_or_clears_the_tool_choice() -> None:
    """An allowed-tools selector keeps only caller tools, and is cleared when none remain."""
    narrowed = GatewayAllowedToolsChoice(
        mode="required",
        provider_entries=({"type": "web_search"}, {"type": "custom", "name": "exec"}),
    )
    updates, _ = _shape(_request(0, tool_choice=narrowed), native_only=True)
    choice = updates["tool_choice"]
    assert isinstance(choice, GatewayAllowedToolsChoice)
    assert choice.provider_entries == ({"type": "custom", "name": "exec"},)

    hosted_only = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="go"),),
        provider_native_tools=(GatewayProviderNativeTool(index=0, tool={"type": "web_search"}),),
        tool_choice="required",
        max_tool_calls=0,
    )
    updates, ignored = _shape(hosted_only, native_only=True)
    assert updates["tool_choice"] is None
    assert TOOL_CHOICE_CLEARED in ignored
    with_function = hosted_only.model_copy(
        update={"tools": (GatewayToolDefinition(name="f", parameters={"type": "object"}),)}
    )
    updates, _ = _shape(with_function, native_only=True)
    assert "tool_choice" not in updates
