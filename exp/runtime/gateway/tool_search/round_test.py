"""Tests for one gateway tool-search round."""

import json
from typing import cast

from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.gateway.tool_search.contracts import GatewayToolSearch, ToolSearchMode
from exp.runtime.gateway.tool_search.plan import ToolSearchPlan, plan_tool_search
from exp.runtime.gateway.tool_search.round import WithheldSearchCall, parse_calls, perform_round


def _tool(name: str, description: str, *, deferred: bool = True) -> GatewayToolDefinition:
    return GatewayToolDefinition(
        name=name,
        description=description,
        parameters={"type": "object"},
        defer_loading=deferred or None,
    )


def _planned(mode: ToolSearchMode = "any", max_rounds: int = 3) -> ToolSearchPlan:
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="What's the weather in Bern?"),),
        tools=(
            _tool("send_email", "Send an email", deferred=False),
            _tool("get_weather", "Current weather for a city"),
            _tool("get_forecast", "Weather forecast for the week"),
            _tool("book_flight", "Book a flight"),
        ),
        tool_search=GatewayToolSearch(
            declared_as="openrouter_tool",
            mode=mode,
            tool_type="openrouter:tool_search",
            max_rounds=max_rounds,
        ),
    )
    plan = plan_tool_search(request, ["openai_compatible"])
    assert plan.state is not None
    return plan


def test_parse_calls_skips_malformed_entries() -> None:
    calls = parse_calls(
        [
            {"call_id": "c1", "name": "tool_search", "arguments": '{"query":"weather"}'},
            {"call_id": "", "name": "tool_search"},
            "junk",
            {"call_id": "c2", "name": "tool_search"},
        ]
    )
    assert [call.call_id for call in calls] == ["c1", "c2"]
    assert calls[1].raw_arguments == "{}"
    assert parse_calls(None) == []


def test_round_loads_matches_and_extends_the_conversation() -> None:
    plan = _planned()
    state = plan.state
    assert state is not None
    outcome = perform_round(
        plan.request,
        state,
        [WithheldSearchCall("c1", "tool_search", '{"query":"weather forecast","limit":2}')],
    )
    assert outcome.rounds[0]["matched"] == ["get_forecast", "get_weather"] or outcome.rounds[0][
        "matched"
    ] == ["get_weather", "get_forecast"]
    assert outcome.rounds[0]["query"] == "weather forecast"
    assert outcome.rounds[0]["pattern"] is None
    matched_tools = cast("list[dict[str, object]]", outcome.rounds[0]["matched_tools"])
    assert {tool["name"] for tool in matched_tools} == {"get_weather", "get_forecast"}
    assert not outcome.exhausted
    roles = [message.role for message in outcome.request.messages]
    assert roles == ["user", "assistant", "tool"]
    assert outcome.request.messages[1].tool_calls[0].call_id == "c1"
    assert outcome.request.messages[1].tool_calls[0].name == "tool_search"
    result = json.loads(outcome.request.messages[2].content or "")
    assert {entry["name"] for entry in result["matched"]} == {"get_weather", "get_forecast"}
    names = [tool.name for tool in outcome.request.tools]
    assert names[:1] == ["send_email"]
    assert set(names) == {"send_email", "get_weather", "get_forecast", "tool_search"}
    assert all(tool.defer_loading is None for tool in outcome.request.tools)
    assert [tool.name for tool in state.deferred] == ["book_flight"]
    assert state.rounds_done == 1


def test_regex_and_error_results_are_reported() -> None:
    plan = _planned()
    state = plan.state
    assert state is not None
    outcome = perform_round(
        plan.request,
        state,
        [
            WithheldSearchCall("c1", "tool_search", '{"pattern":"^book"}'),
            WithheldSearchCall("c2", "tool_search", '{"pattern":"("}'),
            WithheldSearchCall("c3", "tool_search", "{}"),
        ],
    )
    assert outcome.rounds[0]["matched"] == ["book_flight"]
    assert outcome.rounds[0]["pattern"] == "^book"
    assert outcome.rounds[1]["matched"] == []
    assert "error" in json.loads(outcome.request.messages[3].content or "")
    assert "error" in json.loads(outcome.request.messages[4].content or "")
    assert len(outcome.request.messages[1].tool_calls) == 3


def test_last_round_removes_the_search_tool() -> None:
    plan = _planned(max_rounds=1)
    state = plan.state
    assert state is not None
    outcome = perform_round(
        plan.request, state, [WithheldSearchCall("c1", "tool_search", '{"query":"nothing here"}')]
    )
    assert outcome.exhausted
    assert "tool_search" not in [tool.name for tool in outcome.request.tools]
    assert outcome.rounds[0]["matched"] == []
