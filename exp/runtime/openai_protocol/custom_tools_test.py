"""Tests for decoding OpenAI Chat custom tools into the canonical Responses form."""

from __future__ import annotations

from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.tool_contracts import GatewayCustomToolChoice
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import decode_chat

_LARK = 'start: "print(" /[0-9+]+/ ")"'


def _chat(**fields: object) -> GatewayRequest:
    """Decode one Chat body with a user turn and the given extra fields."""
    payload: JsonObject = {
        "model": "gpt-5-nano",
        "messages": [{"role": "user", "content": "Use code_exec to print 2+2."}],
    }
    payload.update(cast(JsonObject, fields))
    return decode_chat(payload).request


def _custom_tool(format_: JsonObject | None = None) -> JsonObject:
    """One Chat custom tool declaration, optionally with a format."""
    body: JsonObject = {"name": "code_exec", "description": "Executes python code"}
    if format_ is not None:
        body["format"] = format_
    return {"type": "custom", "custom": body}


@pytest.mark.parametrize(
    ("format_", "canonical_format"),
    [
        (
            {"type": "grammar", "grammar": {"syntax": "lark", "definition": _LARK}},
            {"type": "grammar", "syntax": "lark", "definition": _LARK},
        ),
        (
            {"type": "grammar", "grammar": {"syntax": "regex", "definition": "^[0-9]+$"}},
            {"type": "grammar", "syntax": "regex", "definition": "^[0-9]+$"},
        ),
        ({"type": "text"}, {"type": "text"}),
        (None, None),
    ],
)
def test_chat_custom_tool_decodes_to_the_flat_responses_declaration(
    format_: JsonObject | None, canonical_format: JsonObject | None
) -> None:
    """The nested Chat spelling becomes the flat Responses one at the caller's position."""
    request = _chat(
        tools=[
            {"type": "function", "function": {"name": "lookup", "parameters": {}}},
            _custom_tool(format_),
        ]
    )

    expected: JsonObject = {
        "type": "custom",
        "name": "code_exec",
        "description": "Executes python code",
    }
    if canonical_format is not None:
        expected["format"] = canonical_format
    assert [tool.name for tool in request.tools] == ["lookup"]
    assert [(entry.index, entry.tool) for entry in request.provider_native_tools] == [(1, expected)]


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "custom"},
        {"type": "custom", "custom": {"name": "x"}, "function": {"name": "x"}},
        {"type": "custom", "custom": {"name": "x", "format": {"type": "grammar"}}},
        {
            "type": "custom",
            "custom": {
                "name": "x",
                "format": {"type": "grammar", "grammar": {"syntax": "ebnf", "definition": "a"}},
            },
        },
    ],
)
def test_malformed_chat_custom_tools_are_refused_on_the_tools_field(tool: JsonObject) -> None:
    """A custom tool without its body, or with a broken grammar, is a tools-scoped 400."""
    with pytest.raises(OpenAIProtocolError) as raised:
        _chat(tools=[tool])

    assert (raised.value.detail.param or "").startswith("tools")


def test_chat_custom_tool_choice_decodes_and_must_name_a_declared_custom_tool() -> None:
    """``{"type": "custom", "custom": {"name"}}`` selects a declared custom tool."""
    request = _chat(
        tools=[_custom_tool()],
        tool_choice={"type": "custom", "custom": {"name": "code_exec"}},
    )

    assert request.tool_choice == GatewayCustomToolChoice(type="custom", name="code_exec")
    with pytest.raises(OpenAIProtocolError) as raised:
        _chat(tools=[_custom_tool()], tool_choice={"type": "custom", "custom": {"name": "other"}})
    assert raised.value.detail.param == "tool_choice.custom.name"


def test_chat_history_custom_calls_become_native_items_in_order() -> None:
    """A mixed assistant turn keeps text and function calls first, then each custom call."""
    request = _chat(
        messages=[
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "Running both.",
                "tool_calls": [
                    {
                        "id": "call_f",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    },
                    {
                        "id": "call_c1",
                        "type": "custom",
                        "custom": {"name": "code_exec", "input": "print(1)"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_f", "content": "found"},
            {"role": "tool", "tool_call_id": "call_c1", "content": "1"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_c2",
                        "type": "custom",
                        "custom": {"name": "code_exec", "input": "print(2)"},
                    },
                    {
                        "id": "call_c3",
                        "type": "custom",
                        "custom": {"name": "code_exec", "input": "print(3)"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_c2", "content": [{"type": "text", "text": "2"}]},
            {"role": "tool", "tool_call_id": "call_c3", "content": "3"},
        ],
        tools=[
            {"type": "function", "function": {"name": "lookup", "parameters": {}}},
            _custom_tool(),
        ],
    )

    shapes = [
        (message.role, message.provider_native_item or message.content, message.tool_calls)
        for message in request.messages
    ]
    assert [role for role, _item, _calls in shapes] == [
        "user",
        "assistant",
        "assistant",
        "tool",
        "tool",
        "assistant",
        "assistant",
        "tool",
        "tool",
    ]
    assistant = request.messages[1]
    assert assistant.content == "Running both."
    assert [call.name for call in assistant.tool_calls] == ["lookup"]
    assert request.messages[2].provider_native_item == {
        "type": "custom_tool_call",
        "call_id": "call_c1",
        "name": "code_exec",
        "input": "print(1)",
    }
    assert request.messages[3].tool_call_id == "call_f"
    assert request.messages[4].provider_native_item == {
        "type": "custom_tool_call_output",
        "call_id": "call_c1",
        "output": "1",
    }
    assert [message.provider_native_item for message in request.messages[5:]] == [
        {
            "type": "custom_tool_call",
            "call_id": "call_c2",
            "name": "code_exec",
            "input": "print(2)",
        },
        {
            "type": "custom_tool_call",
            "call_id": "call_c3",
            "name": "code_exec",
            "input": "print(3)",
        },
        {"type": "custom_tool_call_output", "call_id": "call_c2", "output": "2"},
        {"type": "custom_tool_call_output", "call_id": "call_c3", "output": "3"},
    ]


def test_a_cache_marked_custom_call_turn_still_maps_one_to_one() -> None:
    """Cache restoration runs before expansion, so a marked custom turn decodes cleanly."""
    request = _chat(
        messages=[
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "cache_control": {"type": "ephemeral"},
                "tool_calls": [
                    {"id": "a", "type": "custom", "custom": {"name": "code_exec", "input": "1"}},
                    {"id": "b", "type": "custom", "custom": {"name": "code_exec", "input": "2"}},
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "x"},
            {"role": "tool", "tool_call_id": "b", "content": "y"},
            {"role": "user", "content": "again"},
        ],
        tools=[_custom_tool()],
    )

    items = [message.provider_native_item for message in request.messages]
    assert [item["call_id"] if item else None for item in items] == [None, "a", "b", "a", "b", None]


def test_a_custom_tool_result_with_an_image_is_a_field_scoped_400() -> None:
    """A custom tool output carries text only; media is refused by its field path."""
    with pytest.raises(OpenAIProtocolError) as raised:
        _chat(
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "a", "type": "custom", "custom": {"name": "code_exec", "input": "1"}}
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "a",
                    "content": [
                        {"type": "text", "text": "chart"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                        },
                    ],
                },
            ],
            tools=[_custom_tool()],
        )

    assert raised.value.detail.param == "messages.2.content"
