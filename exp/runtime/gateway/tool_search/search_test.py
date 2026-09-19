"""Tests for BM25 and regex tool search."""

import pytest

from exp.runtime.gateway.contracts import GatewayToolDefinition
from exp.runtime.gateway.tool_search.search import (
    ToolSearchPatternError,
    bm25_search,
    clamp_limit,
    regex_search,
    tokenize,
    tool_document,
)


def _tool(name: str, description: str, **properties: str) -> GatewayToolDefinition:
    return GatewayToolDefinition(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                key: {"type": "string", "description": value} for key, value in properties.items()
            },
        },
    )


_TOOLS = (
    _tool("get_weather", "Current weather for a city", city="City name"),
    _tool("getForecast", "Seven day weather forecast", city="City name", days="Days ahead"),
    _tool("send_email", "Send an email message", to="Recipient address"),
    _tool("create_calendar_event", "Add an event to the user's calendar", when="Start time"),
)


def test_tokenizer_splits_identifiers() -> None:
    assert tokenize("getForecast send_email create-calendar.Event") == [
        "get",
        "forecast",
        "send",
        "email",
        "create",
        "calendar",
        "event",
    ]
    assert "days" in tokenize(tool_document(_TOOLS[1]))


def test_bm25_ranks_the_weather_tools_first_and_bounds_results() -> None:
    ranked = bm25_search("what is the weather", _TOOLS, limit=5)
    assert [tool.name for tool in ranked][:2] == ["get_weather", "getForecast"]
    assert "send_email" not in {tool.name for tool in ranked}
    assert bm25_search("weather", _TOOLS, limit=1) == [_TOOLS[0]]
    assert bm25_search("", _TOOLS, limit=5) == []
    assert bm25_search("weather", (), limit=5) == []


def test_regex_matches_names_and_descriptions_case_insensitively() -> None:
    assert [tool.name for tool in regex_search(r"^get", _TOOLS, limit=5)] == [
        "get_weather",
        "getForecast",
    ]
    assert [tool.name for tool in regex_search(r"CALENDAR", _TOOLS, limit=5)] == [
        "create_calendar_event"
    ]
    assert regex_search(r"weather", _TOOLS, limit=1) == [_TOOLS[0]]
    with pytest.raises(ToolSearchPatternError):
        regex_search("(", _TOOLS, limit=5)
    with pytest.raises(ToolSearchPatternError):
        regex_search("a" * 201, _TOOLS, limit=5)
    # Backtracking-only constructs are refused by RE2 rather than run.
    with pytest.raises(ToolSearchPatternError):
        regex_search(r"(a+)+\1", _TOOLS, limit=5)
    # A classic catastrophic-backtracking shape completes in linear time.
    assert regex_search(r"(a+)+$", _TOOLS, limit=5) == []


def test_limit_is_clamped() -> None:
    assert clamp_limit(None) == 5
    assert clamp_limit(True) == 5
    assert clamp_limit(0) == 1
    assert clamp_limit(50) == 10
    assert clamp_limit(3) == 3
