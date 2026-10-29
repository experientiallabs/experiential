# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Tests for naming Anthropic server tools in the non-Anthropic route rejection."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.models.providers.server_tools import (
    anthropic_server_tool_names,
    anthropic_server_tools_message,
    anthropic_server_tools_present,
)


def _messages_request(
    *,
    server_tools: tuple[JsonObject, ...] = (),
    echoed_blocks: tuple[JsonObject, ...] = (),
) -> GatewayRequest:
    messages: list[GatewayMessage] = [GatewayMessage(role="user", content="search for it")]
    for block in echoed_blocks:
        messages.append(
            GatewayMessage(role="assistant", content=None, provider_anthropic_block=block)
        )
    return GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=tuple(messages),
        provider_server_tools=server_tools,
    )


def test_names_come_from_declared_tools_and_echoed_history_without_duplicates() -> None:
    """Declared entries, server_tool_use blocks, and *_tool_result blocks all name the tool once."""
    request = _messages_request(
        server_tools=(
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 8},
            {"type": "web_fetch_20250910", "name": "web_fetch"},
        ),
        echoed_blocks=(
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}},
            {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []},
        ),
    )
    assert anthropic_server_tool_names(request) == ("web_search", "web_fetch")


def test_a_result_block_alone_still_names_its_tool() -> None:
    """A transcript carrying only the echoed result block still identifies web_search."""
    request = _messages_request(
        echoed_blocks=(
            {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []},
        ),
    )
    assert anthropic_server_tool_names(request) == ("web_search",)


def test_no_server_tools_means_no_names() -> None:
    """A plain Messages request names nothing and is not flagged as carrying server tools."""
    assert anthropic_server_tool_names(_messages_request()) == ()
    assert anthropic_server_tools_present(_messages_request()) is False


def test_a_citation_text_block_is_server_tool_output_but_never_a_tool_name() -> None:
    """Echoed citation text is detected (only Anthropic can replay it) yet not misnamed."""
    request = _messages_request(
        echoed_blocks=(
            {
                "type": "text",
                "text": "cited answer",
                "citations": [{"type": "web_search_result_location"}],
            },
        ),
    )
    assert anthropic_server_tools_present(request) is True
    assert anthropic_server_tool_names(request) == ()
    message = anthropic_server_tools_message(())
    assert "'text'" not in message
    assert "echoed Anthropic server-tool output" in message
    assert "Anthropic only allows them on Anthropic (Claude) models" in message


def test_message_names_the_tool_its_claude_code_label_and_the_anthropic_only_rule() -> None:
    """The caller reads which tool, what Claude Code calls it, and why it needs a Claude model."""
    message = anthropic_server_tools_message(("web_search",))
    assert "'web_search' (Claude Code's WebSearch)" in message
    assert "Anthropic only allows them on Anthropic (Claude) models" in message
    assert "Switch to a Claude model alias" in message
    assert "this tool" in message and "these tools" not in message

    plural = anthropic_server_tools_message(("web_search", "mystery_tool"))
    assert "server tools 'web_search' (Claude Code's WebSearch), 'mystery_tool'" in plural
    assert "these tools" in plural
