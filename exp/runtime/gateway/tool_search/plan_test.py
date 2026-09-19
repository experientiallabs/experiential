"""Tests for the tool-search planner."""

from typing import cast

from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayProviderNativeTool,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.gateway.tool_search.contracts import GatewayToolSearch
from exp.runtime.gateway.tool_search.plan import (
    DROPPED_NO_DEFERRED,
    TOOL_CHOICE_CLEARED,
    natively_served,
    plan_tool_search,
    strip_search_carriers,
    surface_shape,
)


def _tool(name: str, *, deferred: bool = False) -> GatewayToolDefinition:
    return GatewayToolDefinition(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object"},
        defer_loading=deferred or None,
    )


_MESSAGES = GatewayToolSearch(
    declared_as="messages_server_tool",
    mode="bm25",
    tool_type="tool_search_tool_bm25",
    tool_name="ts",
)


def test_native_routes_keep_the_provider_search() -> None:
    assert natively_served(_MESSAGES, ["anthropic_messages"])
    assert not natively_served(_MESSAGES, ["anthropic_messages", "openai_compatible"])
    responses = GatewayToolSearch(declared_as="responses_tool", tool_type="tool_search")
    assert natively_served(responses, ["openai_responses"])
    assert not natively_served(responses, ["openai_compatible"])
    openrouter = GatewayToolSearch(
        declared_as="openrouter_tool", tool_type="openrouter:tool_search"
    )
    assert not natively_served(openrouter, ["openai_compatible"])
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hi"),),
        tools=(_tool("a", deferred=True),),
        provider_server_tools=({"type": "tool_search_tool_bm25", "name": "ts"},),
        tool_search=_MESSAGES,
    )
    plan = plan_tool_search(request, ["anthropic_messages"])
    assert plan.request is request and plan.state is None and plan.admission is None


def test_gateway_search_partitions_tools_and_offers_the_search_tool() -> None:
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hi"),),
        tools=(
            _tool("loaded"),
            _tool("deferred_a", deferred=True),
            _tool("deferred_b", deferred=True),
        ),
        provider_server_tools=({"type": "tool_search_tool_bm25", "name": "ts"},),
        tool_choice=GatewayNamedToolChoice(name="ts"),
        tool_search=_MESSAGES,
    )
    plan = plan_tool_search(request, ["openai_compatible"])
    assert plan.state is not None and plan.admission is not None
    assert [tool.name for tool in plan.request.tools] == ["loaded", "tool_search"]
    assert plan.request.tools[0].defer_loading is None
    assert plan.request.provider_server_tools == ()
    assert plan.request.tool_choice is None
    assert TOOL_CHOICE_CLEARED in plan.request.ignored_parameters
    assert plan.admission == {
        "tool_name": "tool_search",
        "max_rounds": 3,
        "deferred": 2,
        "surface_shape": "messages_bm25",
    }
    assert [tool.name for tool in plan.state.deferred] == ["deferred_a", "deferred_b"]
    properties = cast("dict[str, object]", plan.request.tools[1].parameters["properties"])
    assert "query" in properties
    assert "pattern" not in properties


def test_gateway_tool_name_avoids_a_caller_collision() -> None:
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="hi"),),
        tools=(_tool("tool_search"), _tool("other", deferred=True)),
        provider_native_tools=(GatewayProviderNativeTool(index=0, tool={"type": "tool_search"}),),
        tool_search=GatewayToolSearch(declared_as="responses_tool", tool_type="tool_search"),
    )
    plan = plan_tool_search(request, ["openai_compatible"])
    assert plan.state is not None
    assert plan.state.tool_name == "gateway_tool_search"
    assert plan.request.provider_native_tools == ()
    assert surface_shape(plan.state.search) == "responses"


def test_no_deferred_tools_is_disclosed_and_carriers_stripped() -> None:
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
        tools=(_tool("loaded"),),
        provider_native_tools=(
            GatewayProviderNativeTool(index=1, tool={"type": "openrouter:tool_search"}),
        ),
        tool_search=GatewayToolSearch(
            declared_as="openrouter_tool", tool_type="openrouter:tool_search"
        ),
    )
    plan = plan_tool_search(request, ["openai_compatible"])
    assert plan.state is None and plan.admission is None
    assert DROPPED_NO_DEFERRED in plan.request.ignored_parameters
    assert plan.request.provider_native_tools == ()
    assert [tool.name for tool in plan.request.tools] == ["loaded"]


def test_strip_carriers_leaves_unrelated_tools() -> None:
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="hi"),),
        provider_native_tools=(
            GatewayProviderNativeTool(index=0, tool={"type": "custom", "name": "apply_patch"}),
            GatewayProviderNativeTool(index=1, tool={"type": "tool_search"}),
        ),
    )
    stripped = strip_search_carriers(request)
    assert [entry.tool["type"] for entry in stripped.provider_native_tools] == ["custom"]
