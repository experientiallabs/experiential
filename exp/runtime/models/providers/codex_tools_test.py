# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Tests for Codex native-tool translation on foreign provider wires."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayProviderNativeTool,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.gateway.replay_identity import canonical_request_sha256
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.codex_tools import (
    NativeToolMapping,
    convert_native_history,
    invert_tool_call,
    translate_native_tools,
)
from exp.runtime.models.providers.dialect_dispatch import dialect_stream_payload
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.streaming_requests import route_generation_parameter_requests
from exp.runtime.openai_protocol import decode_responses

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


def test_convert_history_replays_a_gateway_tool_search_round_as_a_function_pair() -> None:
    request = _request(
        messages=(
            GatewayMessage(
                role="assistant",
                provider_native_item={
                    "type": "tool_search_call",
                    "id": "tsc_1",
                    "call_id": "call_ts",
                    "status": "completed",
                    "execution": "server",
                    "arguments": {"goal": "weather"},
                },
            ),
            GatewayMessage(
                role="assistant",
                provider_native_item={
                    "type": "tool_search_output",
                    "id": "tso_1",
                    "call_id": "call_ts",
                    "status": "completed",
                    "execution": "server",
                    "tools": [{"type": "function", "name": "get_weather"}],
                },
            ),
        ),
    )
    messages, disclosures = convert_native_history(request.messages, NativeToolMapping())
    assert disclosures == []
    assert messages[0].role == "assistant"
    assert messages[0].tool_calls[0].name == "tool_search"
    assert messages[0].tool_calls[0].arguments == {"query": "weather"}
    assert messages[1].role == "tool"
    assert messages[1].tool_call_id == "call_ts"
    assert '"get_weather"' in (messages[1].content or "")


def test_custom_history_reuses_allocated_name_without_overwriting_plain_function() -> None:
    """Custom history follows its declaration even when a plain name is identical."""

    for tools in [
        [{"type": "function", "name": "apply_patch", "parameters": {"type": "object"}}, _CUSTOM],
        [_CUSTOM, {"type": "function", "name": "apply_patch", "parameters": {"type": "object"}}],
    ]:
        request = decode_responses(
            {
                "model": "coding",
                "tools": tools,
                "input": [
                    {
                        "type": "custom_tool_call",
                        "name": "apply_patch",
                        "call_id": "c",
                        "input": "patch",
                    },
                    {"type": "custom_tool_call_output", "call_id": "c", "output": "done"},
                    {
                        "type": "function_call",
                        "name": "apply_patch",
                        "call_id": "f",
                        "arguments": "{}",
                    },
                    {"type": "function_call_output", "call_id": "f", "output": "done"},
                ],
            }
        ).request
        translated = translate_native_tools(request)
        messages, _ = convert_native_history(request.messages, translated.mapping)
        calls = [call for message in messages for call in message.tool_calls]
        assert [(call.call_id, call.name) for call in calls] == [
            ("c", "apply_patch_2"),
            ("f", "apply_patch"),
        ]
        assert translated.mapping.as_dict() == {"apply_patch_2": ("apply_patch", None, True)}
        assert [message.tool_call_id for message in messages if message.role == "tool"] == [
            "c",
            "f",
        ]


def test_namespaced_history_uses_full_origin_and_reserves_plain_history_suffixes() -> None:
    """Flattened collisions retain exact origins and reserve genuine plain history names."""

    request = decode_responses(
        {
            "model": "coding",
            "tools": [
                {
                    "type": "namespace",
                    "name": "a__b",
                    "tools": [{"type": "function", "name": "c", "parameters": {"type": "object"}}],
                },
                {
                    "type": "namespace",
                    "name": "a",
                    "tools": [
                        {"type": "function", "name": "b__c", "parameters": {"type": "object"}}
                    ],
                },
            ],
            "input": [
                {
                    "type": "function_call",
                    "name": "c",
                    "namespace": "a__b",
                    "call_id": "one",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "name": "b__c",
                    "namespace": "a",
                    "call_id": "two",
                    "arguments": "{}",
                },
                {"type": "function_call", "name": "a__b__c", "call_id": "plain", "arguments": "{}"},
                {
                    "type": "function_call",
                    "name": "a__b__c_2",
                    "call_id": "suffix",
                    "arguments": "{}",
                },
            ],
        }
    ).request
    translated = translate_native_tools(request)
    messages, _ = convert_native_history(request.messages, translated.mapping)
    assert [tool.name for tool in translated.tools] == ["a__b__c_3", "a__b__c_4"]
    assert [(call.call_id, call.name) for message in messages for call in message.tool_calls] == [
        ("one", "a__b__c_3"),
        ("two", "a__b__c_4"),
        ("plain", "a__b__c"),
        ("suffix", "a__b__c_2"),
    ]
    assert translated.mapping.resolve("a__b__c_3") == ("c", "a__b", False)
    assert translated.mapping.resolve("a__b__c_4") == ("b__c", "a", False)


@pytest.mark.parametrize("declared", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("result_namespace", [False, True])
def test_namespaced_results_keep_their_calls_identity_on_each_wire(
    declared: bool, mixed: bool, result_namespace: bool
) -> None:
    """Mixed-wire translation renames namespaced results together with their calls.

    Args:
        declared: Whether the matching namespace is declared on this turn.
        mixed: Whether the route also contains a Chat wire requiring flattening.
        result_namespace: Whether the result includes its optional namespace.
    """
    body: JsonObject = {
        "model": "coding",
        "input": [
            {
                "type": "function_call",
                "name": "close",
                "namespace": "agents",
                "call_id": "one",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "name": "close",
                "namespace": "agents",
                "call_id": "one",
                "output": "done",
            },
        ],
    }
    if not result_namespace:
        inputs = body["input"]
        assert isinstance(inputs, list)
        del inputs[1]["namespace"]
    if declared:
        body["tools"] = [
            {
                "type": "namespace",
                "name": "agents",
                "tools": [{"type": "function", "name": "close", "parameters": {}}],
            }
        ]
    request = decode_responses(body).request
    native = GatewayWireProfile(dialect="openai_responses", url="http://127.0.0.1:9/v1")
    chat = GatewayWireProfile(dialect="openai_compatible", url="http://127.0.0.1:9/v1")
    profiles = (native, chat) if mixed else (native,)
    _, shaped = route_generation_parameter_requests(profiles, request)
    payload = dialect_stream_payload(native, shaped)
    items = payload["input"]
    assert isinstance(items, list)
    call, result = items
    expected_name = "agents__close" if mixed else "close"
    assert call["name"] == result["name"] == expected_name
    assert call.get("namespace") == (None if mixed else "agents")
    assert result.get("namespace") == ("agents" if result_namespace and not mixed else None)
    if mixed:
        messages = dialect_stream_payload(chat, shaped)["messages"]
        assert isinstance(messages, list)
        assert messages[0]["tool_calls"][0]["function"]["name"] == expected_name
        assert messages[1]["name"] == expected_name


@pytest.mark.parametrize("mixed", [False, True])
def test_namespace_only_results_drop_only_the_translated_namespace(mixed: bool) -> None:
    """A namespace-only output never gains an invented name during flattening.

    Args:
        mixed: Whether a Chat wire requires translation of the native history.
    """
    request = decode_responses(
        {
            "model": "coding",
            "input": [
                {
                    "type": "function_call",
                    "name": "close",
                    "namespace": "agents",
                    "call_id": "one",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "namespace": "agents",
                    "call_id": "one",
                    "output": "done",
                },
            ],
        }
    ).request
    native = GatewayWireProfile(dialect="openai_responses", url="http://127.0.0.1:9/v1")
    chat = GatewayWireProfile(dialect="openai_compatible", url="http://127.0.0.1:9/v1")
    _, shaped = route_generation_parameter_requests((native, chat) if mixed else (native,), request)
    items = dialect_stream_payload(native, shaped)["input"]
    assert isinstance(items, list)
    assert "name" not in items[1]
    assert items[1].get("namespace") == (None if mixed else "agents")
    assert items[1]["call_id"] == "one"
    assert items[1]["output"] == "done"


@pytest.mark.parametrize("duplicate_id", [False, True])
def test_result_attribution_is_not_invented_or_inferred_from_ambiguous_ids(
    duplicate_id: bool,
) -> None:
    """Unnamed outputs remain unnamed and duplicate IDs never select a namespace.

    Args:
        duplicate_id: Whether two different call namespaces share the same ID.
    """
    request = decode_responses(
        {
            "model": "coding",
            "input": [
                {
                    "type": "function_call",
                    "name": "close",
                    "namespace": "agents",
                    "call_id": "one",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "one", "output": "done"},
            ],
        }
    ).request
    messages = request.messages
    if duplicate_id:
        call = messages[0].tool_calls[0].model_copy(update={"provider_namespace": "other"})
        messages = (
            messages[0],
            messages[0].model_copy(update={"tool_calls": (call,)}),
            messages[1].model_copy(update={"provider_tool_name": "close"}),
        )
    converted, _ = convert_native_history(messages, NativeToolMapping())
    assert converted[-1].content == "done"
    assert converted[-1].tool_call_id == "one"
    assert converted[-1].provider_tool_name == ("close" if duplicate_id else None)
    assert converted[-1].provider_tool_namespace is None


def test_history_only_native_names_do_not_claim_ordinary_declarations() -> None:
    """Historical tools keep unique replay names without inventing current declarations."""

    request = decode_responses(
        {
            "model": "coding",
            "tools": [
                {"type": "function", "name": "apply_patch", "parameters": {"type": "object"}},
            ],
            "input": [
                {
                    "type": "custom_tool_call",
                    "name": "apply_patch",
                    "call_id": "old",
                    "input": "patch",
                },
                {"type": "custom_tool_call_output", "call_id": "old", "output": "done"},
            ],
        }
    ).request
    translated = translate_native_tools(request)
    messages, _ = convert_native_history(request.messages, translated.mapping)
    assert [tool.name for tool in translated.tools] == ["apply_patch"]
    assert messages[0].tool_calls[0].name == "apply_patch_2"
    assert translated.mapping.resolve("apply_patch") is None
    assert translated.mapping.resolve("apply_patch_2") == ("apply_patch", None, True)


def test_collision_shaping_preserves_public_identity_and_is_idempotent() -> None:
    """Repeated provider shaping preserves its map and never mutates public replay identity."""

    request = decode_responses(
        {
            "model": "coding",
            "tools": [
                {"type": "function", "name": "apply_patch", "parameters": {"type": "object"}},
                _CUSTOM,
            ],
            "input": [
                {
                    "type": "custom_tool_call",
                    "name": "apply_patch",
                    "call_id": "c",
                    "input": "patch",
                },
                {"type": "custom_tool_call_output", "call_id": "c", "output": "done"},
            ],
        }
    ).request
    digest = canonical_request_sha256(request)
    chat = GatewayWireProfile(
        dialect="openai_compatible", url="http://127.0.0.1:9/v1", model_id="model"
    )
    native = GatewayWireProfile(
        dialect="openai_responses", url="http://127.0.0.1:10/v1", model_id="model"
    )
    for profiles in [(chat,), (native, chat)]:
        public, shaped = route_generation_parameter_requests(profiles, request)
        _, reshaped = route_generation_parameter_requests(profiles, shaped)
        assert canonical_request_sha256(public) == digest
        assert canonical_request_sha256(request) == digest
        assert (
            shaped.native_tool_translation
            == reshaped.native_tool_translation
            == {"apply_patch_2": ("apply_patch", None, True)}
        )
        assert shaped.tools == reshaped.tools
        assert shaped.messages == reshaped.messages
        translated = translate_native_tools(shaped)
        assert translated.mapping.as_dict() == shaped.native_tool_translation
    _, unchanged = route_generation_parameter_requests((native,), request)
    assert unchanged.native_tool_translation is None
    assert unchanged.provider_native_tools == request.provider_native_tools
    assert unchanged.messages == request.messages


def test_duplicate_native_origin_fails_typed_before_dispatch() -> None:
    """Ambiguous duplicate declarations fail before any provider dispatch."""

    request = _request(
        provider_native_tools=(
            GatewayProviderNativeTool(index=0, tool=_CUSTOM),
            GatewayProviderNativeTool(
                index=1, tool={**_CUSTOM, "description": "Different declaration"}
            ),
        )
    )
    with pytest.raises(ProviderParameterError) as rejected:
        translate_native_tools(request)
    assert rejected.value.param == "tools" and rejected.value.code == "invalid_parameter"


def test_inverse_mapping_never_overwrites_a_different_origin() -> None:
    """An occupied wire identity cannot be rebound to another caller tool."""

    mapping = NativeToolMapping({"wire": ("a", "namespace", False)})
    with pytest.raises(ProviderParameterError):
        mapping.record("wire", "b", None, True)
    assert mapping.as_dict() == {"wire": ("a", "namespace", False)}


@pytest.mark.parametrize("forced_plain", [False, True])
def test_public_same_name_history_matches_actual_provider_payload(forced_plain: bool) -> None:
    """Declarations, history and inverse align without redirecting an ordinary named choice."""
    body: JsonObject = {
        "model": "coding",
        "tools": [
            {"type": "function", "name": "apply_patch", "parameters": {"type": "object"}},
            _CUSTOM,
        ],
        "input": [
            {
                "type": "custom_tool_call",
                "name": "apply_patch",
                "call_id": "custom",
                "input": "patch",
            },
            {"type": "custom_tool_call_output", "call_id": "custom", "output": "done"},
            {
                "type": "function_call",
                "name": "apply_patch",
                "call_id": "function",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "function", "output": "done"},
        ],
    }
    if forced_plain:
        body["tool_choice"] = {"type": "function", "name": "apply_patch"}
    request = decode_responses(body).request
    profile = GatewayWireProfile(
        dialect="openai_compatible", url="http://127.0.0.1:9/v1", model_id="same-model"
    )
    public, shaped = route_generation_parameter_requests((profile,), request)
    payload = dialect_stream_payload(profile, shaped)
    tools = payload["tools"]
    assert isinstance(tools, list)
    assert [tool["function"]["name"] for tool in tools] == ["apply_patch", "apply_patch_2"]
    messages = payload["messages"]
    assert isinstance(messages, list)
    calls = [call for message in messages for call in message.get("tool_calls", [])]
    assert [(call["id"], call["function"]["name"]) for call in calls] == [
        ("custom", "apply_patch_2"),
        ("function", "apply_patch"),
    ]
    assert shaped.native_tool_translation == {"apply_patch_2": ("apply_patch", None, True)}
    if forced_plain:
        assert payload["tool_choice"] == {"type": "function", "function": {"name": "apply_patch"}}
    assert canonical_request_sha256(public) == canonical_request_sha256(request)
