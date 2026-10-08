"""Tests for OpenAI ``allowed_tools`` decoding, route shaping and per-wire encoding."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.gateway.tool_contracts import GatewayAllowedToolsChoice
from exp.runtime.models.providers.allowed_tools import (
    ALLOWED_TOOLS_CLEARED,
    ALLOWED_TOOLS_DISCLOSURE,
    restrict_to_allowed_tools,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.capability_policy import coerce_capability
from exp.runtime.models.providers.dialect_dispatch import dialect_stream_payload
from exp.runtime.models.providers.streaming_requests import route_generation_parameter_requests
from exp.runtime.openai_protocol.model_adapter import model_request
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses

_PARAMETERS: JsonObject = {"type": "object", "properties": {"city": {"type": "string"}}}

_OPENAI = GatewayWireProfile(
    dialect="openai_responses",
    url="https://api.openai.test/v1",
    model_id="gpt-5.5",
    supports_reasoning=True,
    reasoning_wire_format="openai_responses",
)
_AZURE_CHAT = GatewayWireProfile(
    dialect="openai_compatible",
    url="https://azure.test/openai/v1",
    model_id="gpt-5-nano",
    supports_reasoning=True,
    reasoning_wire_format="reasoning_effort",
)
_ANTHROPIC = GatewayWireProfile(
    dialect="anthropic_messages",
    url="https://anthropic.test",
    model_id="claude-opus-5",
    maximum_output_tokens=128_000,
)
_GEMINI = GatewayWireProfile(
    dialect="gemini_generate_content",
    url="https://gemini.test",
    model_id="gemini-3-pro",
)


def _chat(mode: str) -> GatewayRequest:
    """The current OpenAI Chat ``allowed_tools`` shape over two declared functions."""
    return decode_chat(
        cast(
            JsonObject,
            {
                "model": "gpt-5-nano",
                "messages": [{"role": "user", "content": "weather in paris?"}],
                "tools": [
                    {"type": "function", "function": {"name": name, "parameters": _PARAMETERS}}
                    for name in ("get_weather", "get_time")
                ],
                "tool_choice": {
                    "type": "allowed_tools",
                    "allowed_tools": {
                        "mode": mode,
                        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                    },
                },
            },
        )
    ).request


def _responses(mode: str) -> GatewayRequest:
    """The current OpenAI Responses ``allowed_tools`` shape over two declared functions."""
    return decode_responses(
        cast(
            JsonObject,
            {
                "model": "gpt-5-nano",
                "input": "weather in paris?",
                "tools": [
                    {"type": "function", "name": name, "parameters": _PARAMETERS}
                    for name in ("get_weather", "get_time")
                ],
                "tool_choice": {
                    "type": "allowed_tools",
                    "mode": mode,
                    "tools": [{"type": "function", "name": "get_weather"}],
                },
            },
        )
    ).request


@pytest.mark.parametrize("decode", [_chat, _responses])
@pytest.mark.parametrize("mode", ["auto", "required"])
def test_both_surfaces_decode_allowed_tools(
    decode: Callable[[str], GatewayRequest], mode: str
) -> None:
    """Chat and Responses spellings decode to one canonical allowed-set choice."""
    request = decode(mode)

    assert request.tool_choice == GatewayAllowedToolsChoice(
        mode=cast(Literal["auto", "required"], mode), names=("get_weather",)
    )
    assert [tool.name for tool in request.tools] == ["get_weather", "get_time"]


@pytest.mark.parametrize("decode", [_chat, _responses])
def test_native_responses_route_passes_the_selector_through(
    decode: Callable[[str], GatewayRequest],
) -> None:
    """An OpenAI Responses rung keeps every tool and receives the selector verbatim."""
    public, provider = route_generation_parameter_requests((_OPENAI,), decode("required"))

    payload = dialect_stream_payload(_OPENAI, provider)

    assert [tool["name"] for tool in cast(list[JsonObject], payload["tools"])] == [
        "get_weather",
        "get_time",
    ]
    assert payload["tool_choice"] == {
        "type": "allowed_tools",
        "mode": "required",
        "tools": [{"type": "function", "name": "get_weather"}],
    }
    assert ALLOWED_TOOLS_DISCLOSURE not in public.ignored_parameters


@pytest.mark.parametrize(
    ("profile", "mode", "expected_choice"),
    [
        (_AZURE_CHAT, "auto", "auto"),
        (_AZURE_CHAT, "required", "required"),
        (_ANTHROPIC, "auto", {"type": "auto"}),
        (_ANTHROPIC, "required", {"type": "any"}),
    ],
)
def test_foreign_wires_restrict_the_tools_and_map_the_mode(
    profile: GatewayWireProfile, mode: str, expected_choice: object
) -> None:
    """A wire without the selector declares only the allowed tools under the mapped mode."""
    public, provider = route_generation_parameter_requests((profile,), _chat(mode))

    payload = dialect_stream_payload(profile, provider)

    tools = cast(list[JsonObject], payload["tools"])
    names = [
        cast(JsonObject, tool["function"])["name"] if "function" in tool else tool["name"]
        for tool in tools
    ]
    assert names == ["get_weather"]
    choice = payload["tool_choice"]
    if isinstance(choice, dict) and "disable_parallel_tool_use" in choice:
        choice = {key: value for key, value in choice.items() if key != "disable_parallel_tool_use"}
    assert choice == expected_choice
    assert ALLOWED_TOOLS_DISCLOSURE in public.ignored_parameters
    # The public copy keeps the caller's selector for reflection.
    assert isinstance(public.tool_choice, GatewayAllowedToolsChoice)


@pytest.mark.parametrize(("mode", "gemini_mode"), [("auto", "AUTO"), ("required", "ANY")])
def test_gemini_restricts_the_tools_and_maps_the_mode(mode: str, gemini_mode: str) -> None:
    """Gemini receives only the allowed declaration under AUTO or ANY."""
    _public, provider = route_generation_parameter_requests((_GEMINI,), _responses(mode))

    payload = dialect_stream_payload(_GEMINI, provider)

    declarations = cast(list[JsonObject], payload["tools"])[0]["functionDeclarations"]
    assert [entry["name"] for entry in cast(list[JsonObject], declarations)] == ["get_weather"]
    config = cast(JsonObject, cast(JsonObject, payload["toolConfig"])["functionCallingConfig"])
    assert config["mode"] == gemini_mode


def test_mixed_route_dispatches_the_restricted_form_on_every_rung() -> None:
    """One shaped request serves the whole route, so a mixed route restricts everywhere."""
    _public, provider = route_generation_parameter_requests((_OPENAI, _AZURE_CHAT), _chat("auto"))

    assert provider.tool_choice == "auto"
    assert [tool.name for tool in provider.tools] == ["get_weather"]
    assert dialect_stream_payload(_OPENAI, provider)["tool_choice"] == "auto"


def test_model_adapter_projects_the_restricted_form() -> None:
    """The model-client projection has no selector, so it carries the restricted form."""
    projected = model_request(_chat("required"))

    assert projected.tool_choice == "required"
    assert [tool.name for tool in projected.tools] == ["get_weather"]


def test_forced_tool_choice_coercion_relaxes_required_to_auto() -> None:
    """A rung that rejects forced tool use relaxes the selector's mode, keeping the set."""
    coercion = coerce_capability("forced_tool_choice", _chat("required"))

    assert coercion is not None
    assert coercion.request.tool_choice == GatewayAllowedToolsChoice(
        mode="auto", names=("get_weather",)
    )
    assert coerce_capability("forced_tool_choice", _chat("auto")) is None


def _native_request(mode: str) -> GatewayRequest:
    """A Responses request allowing one function and one hosted MCP server."""
    return decode_responses(
        cast(
            JsonObject,
            {
                "model": "gpt-5.5",
                "input": "hi",
                "tools": [
                    {"type": "function", "name": "get_weather", "parameters": _PARAMETERS},
                    {"type": "function", "name": "get_time", "parameters": _PARAMETERS},
                    {"type": "mcp", "server_label": "deepwiki", "server_url": "https://d.test"},
                    {"type": "mcp", "server_label": "other", "server_url": "https://o.test"},
                ],
                "tool_choice": {
                    "type": "allowed_tools",
                    "mode": mode,
                    "tools": [
                        {"type": "function", "name": "get_weather"},
                        {"type": "mcp", "server_label": "deepwiki"},
                    ],
                },
            },
        )
    ).request


def test_native_responses_forwards_non_function_entries() -> None:
    """A native route forwards the allowed MCP entry beside the function."""
    _public, provider = route_generation_parameter_requests((_OPENAI,), _native_request("auto"))

    assert dialect_stream_payload(_OPENAI, provider)["tool_choice"] == {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [
            {"type": "function", "name": "get_weather"},
            {"type": "mcp", "server_label": "deepwiki"},
        ],
    }


def test_restriction_keeps_only_the_named_native_declaration() -> None:
    """An allowed entry names a native declaration by every field it spells."""
    restricted = restrict_to_allowed_tools(_native_request("required"))

    assert [entry.tool["server_label"] for entry in restricted.provider_native_tools] == [
        "deepwiki"
    ]
    assert [tool.name for tool in restricted.tools] == ["get_weather"]
    assert restricted.tool_choice == "required"


def test_nothing_allowed_on_the_wire_clears_the_choice_with_disclosure() -> None:
    """A set naming only hosted tools a foreign wire drops cannot force a call."""
    request = _native_request("required").model_copy(
        update={
            "tool_choice": GatewayAllowedToolsChoice(
                mode="required", provider_entries=({"type": "mcp", "server_label": "deepwiki"},)
            )
        }
    )

    public, provider = route_generation_parameter_requests((_AZURE_CHAT,), request)

    assert provider.tool_choice is None
    assert provider.tools == ()
    assert public.ignored_parameters[-2:] == (ALLOWED_TOOLS_DISCLOSURE, ALLOWED_TOOLS_CLEARED)


def _translated(choice: GatewayAllowedToolsChoice) -> GatewayRequest:
    """A foreign-wire request whose tools were translated from Codex native declarations."""
    return GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="hi"),),
        tools=tuple(
            GatewayToolDefinition(name=name, parameters={"type": "object"})
            for name in ("foo", "foo__read", "ns__foo")
        ),
        native_tool_translation={
            "foo": ("foo", None, True),
            "foo__read": ("read", "foo", False),
            "ns__foo": ("foo", "ns", False),
        },
        tool_choice=choice,
    )


def test_custom_and_namespace_entries_match_by_kind_and_identity() -> None:
    """A custom ``foo`` never admits a ``foo`` namespace or a namespaced ``foo``."""
    custom = restrict_to_allowed_tools(
        _translated(
            GatewayAllowedToolsChoice(
                mode="required", provider_entries=({"type": "custom", "name": "foo"},)
            )
        )
    )
    namespace = restrict_to_allowed_tools(
        _translated(
            GatewayAllowedToolsChoice(
                mode="required", provider_entries=({"type": "namespace", "name": "foo"},)
            )
        )
    )

    assert [tool.name for tool in custom.tools] == ["foo"]
    assert [tool.name for tool in namespace.tools] == ["foo__read"]


def test_native_only_declarations_keep_the_allowed_set() -> None:
    """With no function tools, the selector still restricts the native declarations."""
    request = decode_responses(
        cast(
            JsonObject,
            {
                "model": "gpt-5.5",
                "input": "hi",
                "tools": [
                    {"type": "mcp", "server_label": "deepwiki", "server_url": "https://d.test"},
                    {"type": "mcp", "server_label": "other", "server_url": "https://o.test"},
                ],
                "tool_choice": {
                    "type": "allowed_tools",
                    "mode": "auto",
                    "tools": [{"type": "mcp", "server_label": "deepwiki"}],
                },
            },
        )
    ).request

    _public, provider = route_generation_parameter_requests((_OPENAI,), request)

    assert dialect_stream_payload(_OPENAI, provider)["tool_choice"] == {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [{"type": "mcp", "server_label": "deepwiki"}],
    }
