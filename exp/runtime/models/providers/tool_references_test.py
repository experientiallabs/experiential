"""Tests for client-side tool search on every wire dialect.

Each case decodes a Messages body shaped like Claude Code's ToolSearch turn
(deferred MCP tools, no server tool-search tool, a tool_result holding a
``tool_reference``), routes it, and builds the exact upstream payload. The
OpenAI-rung cases also cover a turn that declares a server tool search.
"""

from __future__ import annotations

from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import ImageContentPart, TextContentPart
from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.gateway.replay_identity import canonical_request_sha256
from exp.runtime.gateway.tool_search.plan import plan_tool_search
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.dialect_dispatch import dialect_stream_payload
from exp.runtime.models.providers.streaming_requests import route_generation_parameter_requests
from exp.runtime.models.providers.tool_references import (
    DEFER_LOADING_DROPPED_DISCLOSURE,
    TOOL_REFERENCE_TEXT_DISCLOSURE,
    fold_tool_references,
    resolve_client_tool_search,
    tool_reference_text,
    undeferred_tools,
)

_LOADED = "mcp__github__create_issue"
_UNLOADED = "mcp__slack__post_message"
_SCHEMA: JsonObject = {"type": "object", "properties": {"q": {"type": "string"}}}


def _claude_code_body(*, stream: bool) -> JsonObject:
    """Return one Claude Code ToolSearch turn as the Messages wire carries it."""
    return {
        "model": "coding",
        "max_tokens": 256,
        "stream": stream,
        "tools": [
            {"name": "Bash", "input_schema": {"type": "object"}},
            {"name": "ToolSearch", "input_schema": {"type": "object"}},
            {"name": _LOADED, "input_schema": _SCHEMA, "defer_loading": True},
            {"name": _UNLOADED, "input_schema": _SCHEMA, "defer_loading": True},
        ],
        "messages": [
            {"role": "user", "content": "open an issue"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "ToolSearch",
                        "input": {"query": f"select:{_LOADED}"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "tool_reference", "tool_name": _LOADED}],
                    }
                ],
            },
        ],
    }


def _route(
    dialect: str, *, stream: bool, body: JsonObject | None = None
) -> tuple[GatewayRequest, JsonObject]:
    """Decode a Messages turn, plan its tool search, and build one rung's upstream payload.

    Args:
        dialect: The rung's wire dialect.
        stream: Whether the caller streams.
        body: The Messages body; the Claude Code ToolSearch turn by default.

    Returns:
        The public request (carrying the route's disclosures) and the payload.
    """
    request = decode_messages(body or _claude_code_body(stream=stream)).request
    request = plan_tool_search(request, (dialect,)).request
    profile = GatewayWireProfile(
        dialect=dialect, url="https://upstream.test", maximum_output_tokens=64_000
    )
    public_request, provider_request = route_generation_parameter_requests((profile,), request)
    return public_request, dialect_stream_payload(profile, provider_request)


def _with_server_tool_search(*, stream: bool) -> JsonObject:
    """Return the Claude Code turn with Anthropic's server tool search declared too."""
    body = _claude_code_body(stream=stream)
    tools = cast("list[JsonObject]", body["tools"])
    body["tools"] = [
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
        *tools,
    ]
    return body


@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_passes_references_and_deferred_tools_through_verbatim(stream: bool) -> None:
    """Native Anthropic expands references itself: the reference block and every
    deferred tool (marker intact) forward unchanged, with nothing disclosed."""
    public_request, payload = _route("anthropic_messages", stream=stream)

    assert TOOL_REFERENCE_TEXT_DISCLOSURE not in public_request.ignored_parameters
    assert DEFER_LOADING_DROPPED_DISCLOSURE not in public_request.ignored_parameters
    tools = cast("list[JsonObject]", payload["tools"])
    assert [tool["name"] for tool in tools] == ["Bash", "ToolSearch", _LOADED, _UNLOADED]
    assert [tool.get("defer_loading") for tool in tools] == [None, None, True, True]
    messages = cast("list[JsonObject]", payload["messages"])
    result = cast("list[JsonObject]", messages[-1]["content"])[-1]
    assert result == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": [{"type": "tool_reference", "tool_name": _LOADED}],
    }


@pytest.mark.parametrize("stream", [False, True])
def test_chat_states_the_reference_and_loads_every_deferred_tool(stream: bool) -> None:
    """Chat has no reference block or deferral: the reference becomes text and
    every tool, deferred ones included, ships loaded with its full definition."""
    public_request, payload = _route("openai_compatible", stream=stream)

    assert TOOL_REFERENCE_TEXT_DISCLOSURE in public_request.ignored_parameters
    assert DEFER_LOADING_DROPPED_DISCLOSURE in public_request.ignored_parameters
    tools = cast("list[JsonObject]", payload["tools"])
    functions = [cast("JsonObject", tool["function"]) for tool in tools]
    assert [function["name"] for function in functions] == [
        "Bash",
        "ToolSearch",
        _LOADED,
        _UNLOADED,
    ]
    assert functions[2]["parameters"] == _SCHEMA
    assert all("defer_loading" not in tool for tool in tools)
    messages = cast("list[JsonObject]", payload["messages"])
    assert messages[-1] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": tool_reference_text(_LOADED),
    }


@pytest.mark.parametrize("stream", [False, True])
def test_an_openai_rung_never_receives_defer_loading_without_tool_search(stream: bool) -> None:
    """OpenAI refuses ``defer_loading`` without its hosted tool_search ("Deferred
    tools require tools.tool_search", production 2026-10-22 on gpt-6-luna). A
    Messages caller searching client-side declares none, so every tool ships
    loaded, the drop is disclosed, and the reference becomes text."""
    public_request, payload = _route("openai_responses", stream=stream)

    assert TOOL_REFERENCE_TEXT_DISCLOSURE in public_request.ignored_parameters
    assert DEFER_LOADING_DROPPED_DISCLOSURE in public_request.ignored_parameters
    tools = cast("list[JsonObject]", payload["tools"])
    assert [tool["name"] for tool in tools] == ["Bash", "ToolSearch", _LOADED, _UNLOADED]
    assert all("defer_loading" not in tool for tool in tools)
    assert all(tool.get("type") != "tool_search" for tool in tools)
    items = cast("list[JsonObject]", payload["input"])
    assert items[-1]["type"] == "function_call_output"
    assert items[-1]["output"] == tool_reference_text(_LOADED)


@pytest.mark.parametrize("dialect", ["openai_responses", "openai_compatible"])
@pytest.mark.parametrize("stream", [False, True])
def test_an_openai_rung_runs_the_gateway_search_for_a_declared_tool_search(
    dialect: str, stream: bool
) -> None:
    """A Claude Code turn that also declares Anthropic's server tool search maps
    to the documented gateway search on an OpenAI rung: the loaded tools plus
    one gateway ``tool_search`` function, deferred tools held in its corpus,
    and no provider-native marker or declaration on the wire."""
    _public_request, payload = _route(
        dialect, stream=stream, body=_with_server_tool_search(stream=stream)
    )

    tools = cast("list[JsonObject]", payload["tools"])
    names = [
        str(cast("JsonObject", tool["function"])["name"])
        if dialect == "openai_compatible"
        else str(tool["name"])
        for tool in tools
    ]
    assert names == ["Bash", "ToolSearch", "tool_search"]
    assert "defer_loading" not in str(payload)
    assert "tool_search_tool_regex" not in str(tools)


@pytest.mark.parametrize("dialect", ["bedrock_converse_stream", "gemini_generate_content"])
def test_other_text_only_wires_state_the_reference(dialect: str) -> None:
    """Converse and Gemini serve the turn too: no request fails on a reference."""
    public_request, payload = _route(dialect, stream=True)

    assert TOOL_REFERENCE_TEXT_DISCLOSURE in public_request.ignored_parameters
    assert tool_reference_text(_LOADED) in str(payload)
    assert _UNLOADED in str(payload)
    assert "defer_loading" not in str(payload)


def test_a_mixed_route_discloses_for_its_text_only_rungs() -> None:
    """A route with an Anthropic rung and a Chat rung still discloses the Chat fold."""
    request = decode_messages(_claude_code_body(stream=True)).request
    profiles = (
        GatewayWireProfile(dialect="anthropic_messages", url="https://a.test"),
        GatewayWireProfile(dialect="openai_compatible", url="https://c.test"),
    )
    public_request, provider_request = route_generation_parameter_requests(profiles, request)

    assert TOOL_REFERENCE_TEXT_DISCLOSURE in public_request.ignored_parameters
    anthropic_payload = dialect_stream_payload(profiles[0], provider_request)
    assert len(cast("list[JsonObject]", anthropic_payload["tools"])) == 4


def _deferred(name: str) -> GatewayToolDefinition:
    """Return one deferred function tool."""
    return GatewayToolDefinition(name=name, parameters=_SCHEMA, defer_loading=True)


@pytest.mark.parametrize(
    "surface",
    [GatewayApiSurface.MESSAGES, GatewayApiSurface.CHAT_COMPLETIONS, GatewayApiSurface.RESPONSES],
)
def test_deferred_tools_load_immediately_without_a_tool_search(
    surface: GatewayApiSurface,
) -> None:
    """With no tool search to map, every surface's deferred tools ship loaded and unmarked."""
    request = GatewayRequest(
        surface=surface,
        messages=(GatewayMessage(role="user", content="go"),),
        tools=(GatewayToolDefinition(name="plain", parameters=_SCHEMA), _deferred("d")),
    )
    resolved = resolve_client_tool_search(request)

    assert [tool.name for tool in resolved.tools] == ["plain", "d"]
    assert all(tool.defer_loading is None for tool in resolved.tools)


def test_a_request_without_deferred_tools_or_references_is_untouched() -> None:
    """Nothing to map returns the same request object."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="go"),),
        tools=(GatewayToolDefinition(name="plain", parameters=_SCHEMA),),
    )

    assert undeferred_tools(request) is None
    assert resolve_client_tool_search(request) is request


def test_a_multimodal_tool_result_gains_the_reference_as_a_trailing_text_part() -> None:
    """The folded parts still flatten to the folded content."""
    message = GatewayMessage(
        role="tool",
        tool_call_id="c1",
        content="shot:",
        content_parts=(
            TextContentPart(text="shot:"),
            ImageContentPart(media_type="image/png", data="aGk="),
        ),
        tool_references=("Lookup",),
    )
    (folded,) = fold_tool_references((message,))

    assert folded.tool_references == ()
    assert folded.content == "shot:\n" + tool_reference_text("Lookup")
    texts = [part.text for part in folded.content_parts if part.kind == "text"]
    assert "".join(texts) == folded.content


def test_references_join_replay_identity() -> None:
    """A loaded tool changes what the model can call, so it changes the digest."""
    plain = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="tool", tool_call_id="c1", content=""),),
    )
    referenced = plain.model_copy(
        update={
            "messages": (
                GatewayMessage(
                    role="tool", tool_call_id="c1", content="", tool_references=("Lookup",)
                ),
            )
        }
    )

    assert canonical_request_sha256(plain) != canonical_request_sha256(referenced)


def test_references_are_valid_only_on_tool_messages() -> None:
    """The carrier belongs to a tool result."""
    with pytest.raises(ValueError, match="tool_references are valid only for tool messages"):
        GatewayMessage(role="user", content="hi", tool_references=("Lookup",))
