"""Tests for OpenAI ``tool_choice`` decoding and Responses reflection."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayNamedToolChoice
from exp.runtime.gateway.tool_contracts import GatewayAllowedToolsChoice, GatewayProviderNativeTool
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import decode_responses
from exp.runtime.openai_protocol.tool_choice import (
    chat_tool_choice,
    responses_tool_choice,
    responses_tool_choice_echo,
)


def test_chat_decodes_every_current_selector_shape() -> None:
    """Strings, a named function and the nested allowed-tools set all decode."""
    assert chat_tool_choice("required") == "required"
    assert chat_tool_choice(
        {"type": "function", "function": {"name": "lookup"}}
    ) == GatewayNamedToolChoice(name="lookup")
    assert chat_tool_choice(
        {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": "auto",
                "tools": [
                    {"type": "function", "function": {"name": "lookup"}},
                    {"type": "function", "function": {"name": "clock"}},
                ],
            },
        }
    ) == GatewayAllowedToolsChoice(mode="auto", names=("lookup", "clock"))


def test_responses_decodes_the_flat_allowed_tools_set() -> None:
    """The Responses spelling carries mode and tools beside the type."""
    assert responses_tool_choice(
        {"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "a"}]}
    ) == GatewayAllowedToolsChoice(mode="required", names=("a",))


@pytest.mark.parametrize(
    ("value", "param"),
    [
        ({"type": "allowed_tools"}, "tool_choice.allowed_tools"),
        ({"type": "allowed_tools", "allowed_tools": {"mode": "none", "tools": []}}, "mode"),
        ({"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": []}}, "tools"),
        (
            {
                "type": "allowed_tools",
                "allowed_tools": {"mode": "auto", "tools": [{"type": "function", "name": "a"}]},
            },
            "tools[0]",
        ),
        (
            {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "auto",
                    "tools": [{"type": "function", "function": {"name": "a"}}] * 2,
                },
            },
            "tools[1]",
        ),
    ],
)
def test_chat_names_the_malformed_allowed_tools_field(value: JsonValue, param: str) -> None:
    """A malformed set is refused with the exact field path, never a bare tool_choice."""
    with pytest.raises(OpenAIProtocolError) as raised:
        chat_tool_choice(value)

    assert raised.value.detail.param is not None
    assert raised.value.detail.param.endswith(param)


def test_responses_refuses_a_chat_spelled_allowed_tool() -> None:
    """Each surface accepts only its own function spelling, as OpenAI does."""
    with pytest.raises(OpenAIProtocolError) as raised:
        responses_tool_choice(
            {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "function", "function": {"name": "a"}}],
            }
        )

    assert raised.value.detail.param == "tool_choice.tools[0]"


def test_responses_reflects_the_callers_allowed_tools_selector() -> None:
    """The response object echoes the selector in official Responses form."""
    request = decode_responses(
        cast(
            JsonObject,
            {
                "model": "gpt-5-nano",
                "input": "hi",
                "tools": [
                    {"type": "function", "name": "a", "parameters": {"type": "object"}},
                    {"type": "function", "name": "b", "parameters": {"type": "object"}},
                ],
                "tool_choice": {
                    "type": "allowed_tools",
                    "mode": "auto",
                    "tools": [{"type": "function", "name": "b"}],
                },
            },
        )
    ).request

    assert responses_tool_choice_echo(request) == {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [{"type": "function", "name": "b"}],
    }


def test_an_allowed_name_outside_the_tools_list_is_refused() -> None:
    """Every allowed function must be declared, like a named choice."""
    with pytest.raises(OpenAIProtocolError):
        decode_responses(
            cast(
                JsonObject,
                {
                    "model": "gpt-5-nano",
                    "input": "hi",
                    "tools": [{"type": "function", "name": "a", "parameters": {"type": "object"}}],
                    "tool_choice": {
                        "type": "allowed_tools",
                        "mode": "auto",
                        "tools": [{"type": "function", "name": "missing"}],
                    },
                },
            )
        )


@pytest.mark.parametrize("mode", [[], {}, 1, None])
def test_a_non_string_mode_is_a_field_scoped_client_error(mode: JsonValue) -> None:
    """An unhashable or non-string mode is a 400 on the mode field, never a TypeError."""
    with pytest.raises(OpenAIProtocolError) as raised:
        responses_tool_choice(
            {"type": "allowed_tools", "mode": mode, "tools": [{"type": "function", "name": "a"}]}
        )

    assert raised.value.detail.param == "tool_choice.mode"


def test_responses_carries_non_function_entries_verbatim() -> None:
    """The official SDK's mcp and image_generation entries are allowed, not refused."""
    choice = responses_tool_choice(
        {
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [
                {"type": "function", "name": "a"},
                {"type": "mcp", "server_label": "deepwiki"},
                {"type": "image_generation"},
            ],
        },
        native_tools=(
            GatewayProviderNativeTool(
                index=1, tool={"type": "mcp", "server_label": "deepwiki", "server_url": "u"}
            ),
            GatewayProviderNativeTool(index=2, tool={"type": "image_generation"}),
        ),
    )

    assert choice == GatewayAllowedToolsChoice(
        mode="auto",
        names=("a",),
        provider_entries=(
            {"type": "mcp", "server_label": "deepwiki"},
            {"type": "image_generation"},
        ),
    )


def test_a_native_entry_naming_no_declaration_is_refused() -> None:
    """An allowed MCP server absent from tools is a field-scoped 400, like a function."""
    with pytest.raises(OpenAIProtocolError) as raised:
        responses_tool_choice(
            {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "mcp", "server_label": "missing"}],
            },
            native_tools=(
                GatewayProviderNativeTool(index=0, tool={"type": "mcp", "server_label": "other"}),
            ),
        )

    assert raised.value.detail.param == "tool_choice.tools"


def test_an_oversized_allowed_set_is_refused_before_decoding() -> None:
    """The set is bounded at OpenAI's tool ceiling, so decoding stays linear."""
    tools: list[JsonValue] = [{"type": "function", "name": f"f{index}"} for index in range(129)]
    with pytest.raises(OpenAIProtocolError) as raised:
        responses_tool_choice({"type": "allowed_tools", "mode": "auto", "tools": tools})

    assert raised.value.detail.param == "tool_choice.tools"
