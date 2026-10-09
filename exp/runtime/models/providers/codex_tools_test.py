# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Tests for Codex native-tool translation on foreign provider wires."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayProviderNativeTool,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.gateway.replay_identity import canonical_request_sha256
from exp.runtime.gateway.tool_contracts import GatewayCustomToolChoice
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.capability_policy import coerce_capability
from exp.runtime.models.providers.codex_tools import (
    CUSTOM_GRAMMAR_DISCLOSURE,
    CUSTOM_TOOL_DISCLOSURE,
    NativeToolMapping,
    convert_native_history,
    translate_native_tools,
    translated_tool_choice,
)
from exp.runtime.models.providers.dialect_dispatch import dialect_stream_payload
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.streaming_requests import route_generation_parameter_requests
from exp.runtime.openai_protocol import decode_responses
from exp.runtime.openai_protocol.model_adapter import model_request
from exp.runtime.openai_protocol.requests import decode_chat

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


_OPENAI_RESPONSES = GatewayWireProfile(
    dialect="openai_responses", url="https://api.openai.test/v1", model_id="gpt-5-nano"
)
_AZURE_CHAT = GatewayWireProfile(
    dialect="openai_compatible", url="https://azure.test/openai/v1", model_id="gpt-5-nano"
)
_ANTHROPIC = GatewayWireProfile(
    dialect="anthropic_messages",
    url="https://anthropic.test",
    model_id="claude-sonnet-5",
    maximum_output_tokens=64_000,
)
_LARK = 'start: "print(" /[0-9+]+/ ")"'


def _custom_chat(tool_choice: JsonValue = None, *, grammar: bool = True) -> GatewayRequest:
    """A Chat request declaring one custom tool, the drift-audit repro shape."""
    custom: JsonObject = {"name": "code_exec", "description": "Executes python code"}
    if grammar:
        custom["format"] = {"type": "grammar", "grammar": {"syntax": "lark", "definition": _LARK}}
    payload: JsonObject = {
        "model": "gpt-5-nano",
        "messages": [{"role": "user", "content": "print 2+2"}],
        "tools": [{"type": "custom", "custom": custom}],
    }
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return decode_chat(payload).request


def _custom_responses(tool_choice: JsonValue = None) -> GatewayRequest:
    """The Responses spelling of the same custom tool."""
    payload: JsonObject = {
        "model": "gpt-5-nano",
        "input": "print 2+2",
        "tools": [
            {
                "type": "custom",
                "name": "code_exec",
                "description": "Executes python code",
                "format": {"type": "grammar", "syntax": "lark", "definition": _LARK},
            }
        ],
    }
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return decode_responses(payload).request


@pytest.mark.parametrize(
    ("request_factory", "choice"),
    [
        (_custom_chat, {"type": "custom", "custom": {"name": "code_exec"}}),
        (_custom_responses, {"type": "custom", "name": "code_exec"}),
    ],
)
def test_native_responses_route_forwards_custom_tools_and_choice_verbatim(
    request_factory: Callable[[JsonValue], GatewayRequest], choice: JsonObject
) -> None:
    """A native Responses rung receives the custom declaration, grammar and selector as is."""
    request = request_factory(choice)
    public, provider = route_generation_parameter_requests((_OPENAI_RESPONSES,), request)

    payload = dialect_stream_payload(_OPENAI_RESPONSES, provider)

    assert payload["tools"] == [
        {
            "type": "custom",
            "name": "code_exec",
            "description": "Executes python code",
            "format": {"type": "grammar", "syntax": "lark", "definition": _LARK},
        }
    ]
    assert payload["tool_choice"] == {"type": "custom", "name": "code_exec"}
    assert not any("custom" in path for path in public.ignored_parameters)


@pytest.mark.parametrize("profile", [_AZURE_CHAT, _ANTHROPIC])
@pytest.mark.parametrize("request_factory", [_custom_chat, _custom_responses])
def test_foreign_wires_translate_custom_tools_and_the_choice_with_disclosure(
    profile: GatewayWireProfile, request_factory: Callable[[JsonValue], GatewayRequest]
) -> None:
    """Off the native wire a custom tool is a one-``input`` function, grammar stated, disclosed."""
    request = request_factory(
        {"type": "custom", "custom": {"name": "code_exec"}}
        if request_factory is _custom_chat
        else {"type": "custom", "name": "code_exec"}
    )
    public, provider = route_generation_parameter_requests((profile,), request)

    assert CUSTOM_TOOL_DISCLOSURE in public.ignored_parameters
    assert CUSTOM_GRAMMAR_DISCLOSURE in public.ignored_parameters
    assert provider.tool_choice == GatewayNamedToolChoice(name="code_exec")
    assert provider.native_tool_translation == {"code_exec": ("code_exec", None, True)}
    (tool,) = provider.tools
    properties = cast(JsonObject, tool.parameters["properties"])
    assert cast(JsonObject, properties["input"])["description"] == (
        "The raw tool input (this tool was declared as a freeform tool). "
        f"The input must match this lark grammar:\n{_LARK}"
    )
    payload = dialect_stream_payload(profile, provider)
    choice = payload["tool_choice"]
    assert choice in (
        {"type": "function", "function": {"name": "code_exec"}},
        {"type": "tool", "name": "code_exec"},
    )
    # The public request keeps the caller's selector for reflection.
    assert isinstance(public.tool_choice, GatewayCustomToolChoice)


def test_a_text_custom_tool_translates_without_a_grammar_disclosure() -> None:
    """Only a declared grammar owes the grammar disclosure."""
    public, _provider = route_generation_parameter_requests(
        (_AZURE_CHAT,), _custom_chat(grammar=False)
    )

    assert CUSTOM_TOOL_DISCLOSURE in public.ignored_parameters
    assert CUSTOM_GRAMMAR_DISCLOSURE not in public.ignored_parameters


@pytest.mark.parametrize("profile", [_OPENAI_RESPONSES, _AZURE_CHAT])
def test_required_over_custom_tools_alone_is_kept_not_refused(profile: GatewayWireProfile) -> None:
    """Custom declarations are tool definitions, so ``required`` and parallel controls stay."""
    request = _custom_chat("required").model_copy(update={"parallel_tool_calls": False})

    public, provider = route_generation_parameter_requests((profile,), request)

    assert provider.tool_choice == "required"
    assert provider.parallel_tool_calls is False
    assert "tool_choice" not in public.ignored_parameters


def test_custom_choice_relaxes_to_auto_on_a_rung_that_cannot_force_tools() -> None:
    """The forced-choice policy treats a custom selector like any forced choice."""
    coercion = coerce_capability(
        "forced_tool_choice", _custom_chat({"type": "custom", "custom": {"name": "code_exec"}})
    )

    assert coercion is not None
    assert coercion.request.tool_choice == "auto"


def test_untranslated_custom_choice_fails_closed_in_foreign_encoders() -> None:
    """Every foreign encoder refuses a custom selector that skipped route shaping."""
    request = _custom_chat({"type": "custom", "custom": {"name": "code_exec"}})
    translated_tools = translate_native_tools(request).tools
    bypassed = request.model_copy(update={"tools": translated_tools, "provider_native_tools": ()})

    with pytest.raises(ValueError, match="custom-tool"):
        model_request(bypassed)
    with pytest.raises(ValueError, match="custom-tool"):
        dialect_stream_payload(_ANTHROPIC, bypassed)


def test_translated_tool_choice_leaves_other_choices_alone() -> None:
    """Only a custom selector is renamed; modes and named functions pass through."""
    mapping = NativeToolMapping({"code_exec_2": ("code_exec", None, True)})

    assert translated_tool_choice(
        GatewayCustomToolChoice(type="custom", name="code_exec"), mapping
    ) == GatewayNamedToolChoice(name="code_exec_2")
    assert translated_tool_choice("required", mapping) == "required"
    named = GatewayNamedToolChoice(name="lookup")
    assert translated_tool_choice(named, mapping) is named


def test_chat_custom_history_translates_on_a_foreign_wire() -> None:
    """Chat custom calls and results reach a Chat wire as the translated function pair."""
    request = decode_chat(
        {
            "model": "gpt-5-nano",
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_c",
                            "type": "custom",
                            "custom": {"name": "code_exec", "input": "print(1)"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_c", "content": "1"},
            ],
            "tools": [{"type": "custom", "custom": {"name": "code_exec"}}],
        }
    ).request

    _public, provider = route_generation_parameter_requests((_AZURE_CHAT,), request)
    payload = dialect_stream_payload(_AZURE_CHAT, provider)

    messages = cast(list[JsonObject], payload["messages"])
    assert messages[1]["tool_calls"] == [
        {
            "id": "call_c",
            "type": "function",
            "function": {"name": "code_exec", "arguments": '{"input": "print(1)"}'},
        }
    ]
    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == "call_c"


def test_chat_custom_history_replays_verbatim_on_native_responses() -> None:
    """On a native Responses rung the Chat history becomes the official custom items."""
    request = decode_chat(
        {
            "model": "gpt-5-nano",
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_c",
                            "type": "custom",
                            "custom": {"name": "code_exec", "input": "print(1)"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_c", "content": "1"},
            ],
            "tools": [{"type": "custom", "custom": {"name": "code_exec"}}],
        }
    ).request

    _public, provider = route_generation_parameter_requests((_OPENAI_RESPONSES,), request)
    payload = dialect_stream_payload(_OPENAI_RESPONSES, provider)

    items = payload["input"]
    assert isinstance(items, list)
    assert items[1:] == [
        {"type": "custom_tool_call", "call_id": "call_c", "name": "code_exec", "input": "print(1)"},
        {"type": "custom_tool_call_output", "call_id": "call_c", "output": "1"},
    ]


def test_one_chat_turn_of_parallel_custom_and_function_calls_stays_one_message() -> None:
    """A Chat wire needs every result to answer the turn right before it."""
    request = decode_chat(
        {
            "model": "gpt-5-nano",
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "both",
                    "tool_calls": [
                        {
                            "id": "f",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        },
                        {
                            "id": "a",
                            "type": "custom",
                            "custom": {"name": "code_exec", "input": "1"},
                        },
                        {
                            "id": "b",
                            "type": "custom",
                            "custom": {"name": "code_exec", "input": "2"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "f", "content": "x"},
                {"role": "tool", "tool_call_id": "a", "content": "y"},
                {"role": "tool", "tool_call_id": "b", "content": "z"},
            ],
            "tools": [
                {"type": "function", "function": {"name": "lookup", "parameters": {}}},
                {"type": "custom", "custom": {"name": "code_exec"}},
            ],
        }
    ).request

    _public, provider = route_generation_parameter_requests((_AZURE_CHAT,), request)
    messages = cast(list[JsonObject], dialect_stream_payload(_AZURE_CHAT, provider)["messages"])

    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "tool",
    ]
    calls = cast(list[JsonObject], messages[1]["tool_calls"])
    assert [call["id"] for call in calls] == ["f", "a", "b"]
    assert messages[1]["content"] == "both"


def test_custom_only_parallel_false_is_emulated_on_wires_without_the_control() -> None:
    """Translated custom tools still get the per-dialect parallel-tool shaping."""
    gemini = GatewayWireProfile(
        dialect="gemini_generate_content", url="https://gemini.test", model_id="gemini-3-pro"
    )
    request = _custom_chat().model_copy(update={"parallel_tool_calls": False})

    public, provider = route_generation_parameter_requests((gemini,), request)

    assert "parallel_tool_calls->emulated(serialized_by_gateway)" in public.ignored_parameters
    assert provider.serialize_tool_calls is True
