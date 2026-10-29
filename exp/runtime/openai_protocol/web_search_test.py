"""Tests for the OpenAI-surface web-search spellings."""

import pytest

from exp.runtime.gateway.contracts import GatewayProviderNativeTool
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.web_search import (
    ChatPlugin,
    WebSearchOptions,
    chat_web_search,
    responses_web_search,
    split_online_suffix,
)


def test_online_suffix_is_split_only_when_it_is_a_suffix() -> None:
    assert split_online_suffix("gpt-5.6-luna:online") == ("gpt-5.6-luna", True)
    assert split_online_suffix("gpt-5.6-luna") == ("gpt-5.6-luna", False)
    assert split_online_suffix(":online") == (":online", False)


def test_plugin_wins_over_options_and_carries_its_detail() -> None:
    search = chat_web_search(
        options=WebSearchOptions(search_context_size="low"),
        plugins=[ChatPlugin(id="web", max_results=7, exclude_domains=("spam.example",))],
        online_suffix=True,
    )
    assert search is not None
    assert search.declared_as == "plugin"
    assert search.max_results == 7
    assert search.search_context_size == "low"
    assert search.blocked_domains == ("spam.example",)
    disabled = chat_web_search(
        options=None, plugins=[ChatPlugin(id="web", enabled=False)], online_suffix=False
    )
    assert disabled is None


def test_unknown_plugins_are_rejected_by_position() -> None:
    with pytest.raises(OpenAIProtocolError) as captured:
        chat_web_search(
            options=None, plugins=[ChatPlugin(id="response-healing")], online_suffix=False
        )
    assert captured.value.detail.param == "plugins.0.id"


def test_responses_tool_shapes_normalize() -> None:
    tools = (
        GatewayProviderNativeTool(index=0, tool={"type": "custom", "name": "apply_patch"}),
        GatewayProviderNativeTool(
            index=1,
            tool={
                "type": "web_search_preview",
                "search_context_size": "high",
                "filters": {"allowed_domains": ["docs.python.org"]},
                "user_location": {"type": "approximate", "country": "US"},
            },
        ),
    )
    search = responses_web_search(tools, online_suffix=False)
    assert search is not None
    assert search.declared_as == "responses_tool"
    assert search.max_results == 8
    assert search.allowed_domains == ("docs.python.org",)
    assert search.user_location == {"type": "approximate", "country": "US"}
    assert responses_web_search(tools[:1], online_suffix=False) is None
    suffix = responses_web_search((), online_suffix=True)
    assert suffix is not None and suffix.declared_as == "model_suffix"
    with pytest.raises(OpenAIProtocolError) as captured:
        responses_web_search(
            (
                GatewayProviderNativeTool(
                    index=2,
                    tool={
                        "type": "web_search",
                        "filters": {"allowed_domains": ["a.com"], "blocked_domains": ["b.com"]},
                    },
                ),
            ),
            online_suffix=False,
        )
    assert captured.value.detail.param == "tools.2"
