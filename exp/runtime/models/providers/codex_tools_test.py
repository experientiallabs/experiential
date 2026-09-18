# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Tests for Codex native-tool translation on foreign provider wires."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayProviderNativeTool,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.models.providers.codex_tools import (
    NativeToolMapping,
    convert_native_history,
    invert_tool_call,
    translate_native_tools,
)

_FUNCTION = {
    "type": "function",
    "name": "exec_command",
    "description": "Execute shell commands",
    "strict": False,
    "parameters": {"type": "object", "properties": {}},
}
_CUSTOM: JsonObject = {
    "type": "custom",
    "name": "apply_patch",
    "description": "Use the `apply_patch` tool to edit files.",
    "format": {"type": "grammar", "syntax": "lark", "definition": "start: x"},
}
_NAMESPACE: JsonObject = {
    "type": "namespace",
    "name": "multi_agent_v1",
    "description": "Tools for spawning and managing sub-agents.",
    "tools": [
        {
            "type": "function",
            "name": "close_agent",
            "description": "Close an agent.",
            "strict": False,
            "parameters": {"type": "object", "properties": {}},
        }
    ],
}
_WEB_SEARCH: JsonObject = {"type": "web_search", "external_web_access": False}
_TOOL_SEARCH: JsonObject = {
    "type": "tool_search",
    "description": "Search for additional tools.",
    "parameters": {"type": "object", "properties": {}},
    "execution": {"type": "server"},
}


def _request(
    *,
    tools: tuple[GatewayToolDefinition, ...] = (),
    provider_native_tools: tuple[GatewayProviderNativeTool, ...] = (),
    messages: tuple[GatewayMessage, ...] | None = None,
) -> GatewayRequest:
    return GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=messages if messages is not None else (GatewayMessage(role="user", content="hi"),),
        tools=tools,
        provider_native_tools=provider_native_tools,
    )


def test_translate_hoists_functions_converts_custom_drops_hosted() -> None:
    request = _request(
        tools=(GatewayToolDefinition(name="exec_command", parameters={"type": "object"}),),
        provider_native_tools=(
            GatewayProviderNativeTool(index=1, tool=_CUSTOM),
            GatewayProviderNativeTool(index=2, tool=_NAMESPACE),
            GatewayProviderNativeTool(index=3, tool=_WEB_SEARCH),
            GatewayProviderNativeTool(index=4, tool=_TOOL_SEARCH),
        ),
    )
    result = translate_native_tools(request)
    names = [tool.name for tool in result.tools]
    assert names == ["exec_command", "apply_patch", "multi_agent_v1__close_agent"]
    # hosted tools dropped with disclosure
    assert "tools.web_search->dropped(unsupported_by_provider)" in result.disclosures
    assert "tools.tool_search->dropped(unsupported_by_provider)" in result.disclosures
    # custom tool presents a single required input string
    apply_patch = next(t for t in result.tools if t.name == "apply_patch")
    assert apply_patch.parameters["required"] == ["input"]
    # mapping inverts both translated tools
    assert result.mapping.resolve("apply_patch") == ("apply_patch", None, True)
    assert result.mapping.resolve("multi_agent_v1__close_agent") == (
        "close_agent",
        "multi_agent_v1",
        False,
    )


def test_mangled_name_collision_is_suffixed() -> None:
    request = _request(
        tools=(
            GatewayToolDefinition(
                name="multi_agent_v1__close_agent", parameters={"type": "object"}
            ),
        ),
        provider_native_tools=(GatewayProviderNativeTool(index=1, tool=_NAMESPACE),),
    )
    result = translate_native_tools(request)
    names = [t.name for t in result.tools]
    assert names == ["multi_agent_v1__close_agent", "multi_agent_v1__close_agent_2"]
    assert result.mapping.resolve("multi_agent_v1__close_agent_2") == (
        "close_agent",
        "multi_agent_v1",
        False,
    )


def test_invert_custom_unwraps_input() -> None:
    mapping = NativeToolMapping()
    mapping.record("apply_patch", "apply_patch", None, True)
    name, ns, custom, text = invert_tool_call(
        "apply_patch", '{"input": "*** Begin Patch"}', mapping
    )
    assert (name, ns, custom, text) == ("apply_patch", None, True, "*** Begin Patch")


def test_invert_custom_guards_malformed_arguments() -> None:
    mapping = NativeToolMapping()
    mapping.record("apply_patch", "apply_patch", None, True)
    # not a JSON object with a string input -> raw text passes through, no crash
    name, ns, custom, text = invert_tool_call("apply_patch", "raw patch text", mapping)
    assert (name, custom, text) == ("apply_patch", True, "raw patch text")


def test_invert_namespaced_function_restores_namespace() -> None:
    mapping = NativeToolMapping()
    mapping.record("multi_agent_v1__close_agent", "close_agent", "multi_agent_v1", False)
    name, ns, custom, text = invert_tool_call("multi_agent_v1__close_agent", "{}", mapping)
    assert (name, ns, custom, text) == ("close_agent", "multi_agent_v1", False, None)


def test_invert_unknown_name_is_plain_function() -> None:
    assert invert_tool_call("something", "{}", NativeToolMapping()) == (
        "something",
        None,
        False,
        None,
    )


def test_convert_history_custom_tool_call_roundtrips() -> None:
    request = _request(
        messages=(
            GatewayMessage(
                role="assistant",
                provider_native_item={
                    "type": "custom_tool_call",
                    "call_id": "call_1",
                    "name": "apply_patch",
                    "input": "*** Begin Patch",
                },
            ),
            GatewayMessage(
                role="assistant",
                provider_native_item={
                    "type": "custom_tool_call_output",
                    "call_id": "call_1",
                    "output": "done",
                },
            ),
        ),
    )
    mapping = NativeToolMapping()
    messages, disclosures = convert_native_history(request.messages, mapping)
    assert messages[0].role == "assistant"
    assert messages[0].tool_calls[0].name == "apply_patch"
    assert messages[0].tool_calls[0].arguments == {"input": "*** Begin Patch"}
    assert messages[1].role == "tool"
    assert messages[1].tool_call_id == "call_1"
    assert messages[1].content == "done"
    assert mapping.resolve("apply_patch") == ("apply_patch", None, True)


def test_convert_history_additional_tools_dropped() -> None:
    request = _request(
        messages=(
            GatewayMessage(
                role="assistant",
                provider_native_item={"type": "additional_tools", "tools": []},
            ),
        ),
    )
    messages, disclosures = convert_native_history(request.messages, NativeToolMapping())
    assert messages == ()
    assert "input.additional_tools->dropped(declared_inline)" in disclosures
