"""OpenAI-shaped tools and selectors fold into their Messages spellings."""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.anthropic_protocol.openai_tool_shapes import (
    OPENAI_FUNCTION_TOOL_TRANSLATED,
    OPENAI_TOOL_CHOICE_TRANSLATED,
    fold_openai_tool_shapes,
)
from exp.runtime.anthropic_protocol.requests import decode_messages, decode_messages_count_tokens
from exp.runtime.openai_protocol.errors import OpenAIProtocolError

_SCHEMA: JsonObject = {"type": "object", "properties": {"q": {"type": "string"}}}

_CHAT_TOOL: JsonObject = {
    "type": "function",
    "function": {"name": "search", "description": "look up", "parameters": _SCHEMA},
}

_CUSTOM_TOOL: JsonObject = {"name": "search", "description": "look up", "input_schema": _SCHEMA}


def _count_body(**overrides: JsonValue) -> JsonObject:
    """One minimal ``count_tokens`` body with overrides applied."""
    payload: JsonObject = {"model": "coding", "messages": [{"role": "user", "content": "foo"}]}
    payload.update(overrides)
    return payload


def test_count_tokens_with_chat_function_tools_decodes() -> None:
    """A Chat-shaped tool list on ``count_tokens`` decodes to the custom tools it names."""
    decoded = decode_messages_count_tokens(_count_body(tools=[_CHAT_TOOL]))
    (tool,) = decoded.request.tools
    assert tool.name == "search"
    assert tool.description == "look up"
    assert tool.parameters == _SCHEMA
    assert decoded.request.ignored_parameters == (OPENAI_FUNCTION_TOOL_TRANSLATED,)


def test_generation_with_chat_tools_and_selector_decodes() -> None:
    """A generation body folds Chat tools and the Chat ``tool_choice`` beside them."""
    decoded = decode_messages(
        _count_body(
            max_tokens=128,
            tools=[_CHAT_TOOL],
            tool_choice={"type": "function", "function": {"name": "search"}},
        )
    )
    assert [tool.name for tool in decoded.request.tools] == ["search"]
    assert decoded.request.tool_choice is not None
    assert decoded.request.ignored_parameters == (
        OPENAI_FUNCTION_TOOL_TRANSLATED,
        OPENAI_TOOL_CHOICE_TRANSLATED,
    )


def test_chat_and_responses_shapes_fold_to_the_same_custom_tool() -> None:
    """The nested Chat and flat Responses declarations fold to one custom tool."""
    flat: JsonObject = {
        "type": "function",
        "name": "search",
        "description": "look up",
        "parameters": _SCHEMA,
    }
    for tool in (_CHAT_TOOL, flat):
        repaired, disclosures = fold_openai_tool_shapes({"tools": [tool]})
        assert repaired["tools"] == [_CUSTOM_TOOL]
        assert disclosures == (OPENAI_FUNCTION_TOOL_TRANSLATED,)


def test_annotations_carry_and_missing_parameters_is_the_decoders_empty_schema() -> None:
    """``strict`` and the tool annotations carry verbatim; no parameters is ``{}``."""
    marker: JsonObject = {"type": "ephemeral"}
    repaired, _ = fold_openai_tool_shapes(
        {
            "tools": [
                {"type": "function", "function": {"name": "ping", "strict": True}},
                {"type": "function", "name": "pong", "defer_loading": True},
                {
                    "type": "function",
                    "function": {"name": "pang"},
                    "defer_loading": True,
                    "cache_control": marker,
                },
            ]
        }
    )
    assert repaired["tools"] == [
        {"name": "ping", "input_schema": {}, "strict": True},
        {"name": "pong", "input_schema": {}, "defer_loading": True},
        {"name": "pang", "input_schema": {}, "defer_loading": True, "cache_control": marker},
    ]


def test_a_deferred_responses_tool_decodes_on_messages() -> None:
    """A deferred Responses function decodes as a deferred custom tool."""
    decoded = decode_messages_count_tokens(
        _count_body(
            tools=[
                {"type": "function", "name": "search", "parameters": _SCHEMA},
                {"type": "function", "name": "lookup", "defer_loading": True},
            ]
        )
    )
    assert [tool.name for tool in decoded.request.tools] == ["search", "lookup"]
    assert decoded.request.tools[1].defer_loading is True


def test_a_chat_wrapper_cache_marker_decodes_on_messages() -> None:
    """A Chat tool's wrapper ``cache_control`` decodes as the custom tool's marker."""
    decoded = decode_messages_count_tokens(
        _count_body(tools=[{**_CHAT_TOOL, "cache_control": {"type": "ephemeral"}}])
    )
    (tool,) = decoded.request.tools
    assert tool.name == "search"
    assert decoded.request.ignored_parameters == (OPENAI_FUNCTION_TOOL_TRANSLATED,)


def test_messages_tools_are_left_untouched() -> None:
    """A valid Messages declaration (custom or server tool) is never reinterpreted."""
    payload: JsonObject = {
        "tools": [_CUSTOM_TOOL, {"type": "web_search_20250305", "name": "web_search"}],
        "tool_choice": {"type": "auto"},
    }
    repaired, disclosures = fold_openai_tool_shapes(payload)
    assert repaired is payload
    assert disclosures == ()


def test_mixed_tool_lists_fold_only_the_openai_entries() -> None:
    """Only OpenAI entries fold; Messages entries in the same list are kept as sent."""
    repaired, disclosures = fold_openai_tool_shapes(
        {"tools": [_CUSTOM_TOOL | {"name": "other"}, _CHAT_TOOL]}
    )
    assert repaired["tools"] == [_CUSTOM_TOOL | {"name": "other"}, _CUSTOM_TOOL]
    assert disclosures == (OPENAI_FUNCTION_TOOL_TRANSLATED,)


@pytest.mark.parametrize(
    ("choice", "selector"),
    (
        ("auto", {"type": "auto"}),
        ("none", {"type": "none"}),
        ("required", {"type": "any"}),
        ({"type": "function", "function": {"name": "search"}}, {"type": "tool", "name": "search"}),
        ({"type": "function", "name": "search"}, {"type": "tool", "name": "search"}),
    ),
)
def test_openai_tool_choice_folds(choice: JsonValue, selector: JsonObject) -> None:
    """Every OpenAI ``tool_choice`` spelling folds to its Messages selector."""
    repaired, disclosures = fold_openai_tool_shapes({"tool_choice": choice})
    assert repaired["tool_choice"] == selector
    assert disclosures == (OPENAI_TOOL_CHOICE_TRANSLATED,)


@pytest.mark.parametrize(
    "tool",
    (
        {"type": "function", "function": {"name": "s", "parameters": {}, "extra": 1}},
        {"type": "function", "function": {"name": 3}},
        {"type": "function", "function": {"name": "s", "parameters": "{}"}},
        {"type": "function", "function": "search"},
        {"type": "function", "function": {"name": "s"}, "max_results": 3},
    ),
)
def test_a_malformed_openai_tool_keeps_the_wire_refusal(tool: JsonObject) -> None:
    """An entry that is not exactly an OpenAI shape is left for the named wire refusal."""
    repaired, disclosures = fold_openai_tool_shapes({"tools": [tool]})
    assert repaired["tools"] == [tool]
    assert disclosures == ()
    with pytest.raises(OpenAIProtocolError) as error:
        decode_messages_count_tokens(_count_body(tools=[tool]))
    assert error.value.detail.param is not None
    assert error.value.detail.param.startswith("tools.0")


@pytest.mark.parametrize(
    "choice", ("always", {"type": "function"}, {"type": "function", "name": 1})
)
def test_an_unknown_tool_choice_keeps_the_wire_refusal(choice: JsonValue) -> None:
    """A selector that is no OpenAI spelling is left for the wire model."""
    repaired, disclosures = fold_openai_tool_shapes({"tool_choice": choice})
    assert repaired["tool_choice"] == choice
    assert disclosures == ()


def test_an_untyped_chat_wrapper_folds_as_chat_defaults_it() -> None:
    """A Chat tool without ``type`` (Chat's default is ``function``) folds the same way."""
    untyped: JsonObject = {"function": _CHAT_TOOL["function"]}
    repaired, disclosures = fold_openai_tool_shapes({"tools": [untyped]})
    assert repaired["tools"] == [_CUSTOM_TOOL]
    assert disclosures == (OPENAI_FUNCTION_TOOL_TRANSLATED,)
    decoded = decode_messages_count_tokens(_count_body(tools=[untyped]))
    assert [tool.name for tool in decoded.request.tools] == ["search"]
