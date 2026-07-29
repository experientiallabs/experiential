"""Tests for the Anthropic Messages tool-calling translation."""

from __future__ import annotations

import json

from llm_waterfall import ChatRequest

from wmo.providers._anthropic_chat import (
    CACHE_CONTROL_EPHEMERAL,
    messages_request,
    messages_response,
)

_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "run a shell command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    },
}


def _request(**overrides: object) -> ChatRequest:
    payload: dict[str, object] = {
        "messages": [
            {"role": "system", "content": "you are an agent"},
            {"role": "user", "content": "fix the bug"},
        ],
        "tools": [_TOOL],
    }
    payload.update(overrides)
    return ChatRequest.model_validate(payload)


def test_system_lifts_out_of_messages() -> None:
    payload = messages_request(_request(), "claude-fable-5", default_max_tokens=1024)

    assert payload["system"] == "you are an agent"
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert [m["role"] for m in messages] == ["user"]


def test_max_tokens_falls_back_to_the_default() -> None:
    payload = messages_request(_request(), "claude-fable-5", default_max_tokens=1024)
    assert payload["max_tokens"] == 1024

    explicit = messages_request(_request(max_tokens=77), "claude-fable-5", default_max_tokens=1024)
    assert explicit["max_tokens"] == 77


def test_tools_and_tool_choice_translate() -> None:
    payload = messages_request(
        _request(tool_choice="required"), "claude-fable-5", default_max_tokens=1024
    )

    tools = payload["tools"]
    assert isinstance(tools, list)
    assert tools[0]["name"] == "bash"
    assert tools[0]["input_schema"] == _TOOL["function"]["parameters"]
    assert payload["tool_choice"] == {"type": "any"}


def test_tool_choice_none_drops_the_schemas() -> None:
    """`none` means the model may not call a tool, so the schemas are not sent at all."""
    payload = messages_request(
        _request(tool_choice="none"), "claude-fable-5", default_max_tokens=1024
    )

    assert "tools" not in payload
    assert "tool_choice" not in payload


def test_named_tool_choice_translates() -> None:
    payload = messages_request(
        _request(tool_choice={"type": "function", "function": {"name": "bash"}}),
        "claude-fable-5",
        default_max_tokens=1024,
    )
    assert payload["tool_choice"] == {"type": "tool", "name": "bash"}


def test_assistant_tool_call_and_result_round_trip() -> None:
    """An agent turn plus its observation must survive as tool_use / tool_result blocks."""
    request = _request(
        messages=[
            {"role": "user", "content": "fix the bug"},
            {
                "role": "assistant",
                "content": "looking",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "a.py b.py"},
        ]
    )

    payload = messages_request(request, "claude-fable-5", default_max_tokens=1024)
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]

    assistant_blocks = messages[1]["content"]
    assert isinstance(assistant_blocks, list)
    assert assistant_blocks[0] == {"type": "text", "text": "looking"}
    assert assistant_blocks[1]["type"] == "tool_use"
    assert assistant_blocks[1]["id"] == "call_1"
    assert assistant_blocks[1]["input"] == {"command": "ls"}

    result_blocks = messages[2]["content"]
    assert isinstance(result_blocks, list)
    assert result_blocks[0]["type"] == "tool_result"
    assert result_blocks[0]["tool_use_id"] == "call_1"


def test_consecutive_same_role_messages_merge() -> None:
    """The Messages API rejects two consecutive messages with one role; tool results merge."""
    request = _request(
        messages=[
            {"role": "user", "content": "fix the bug"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "one"},
            {"role": "tool", "tool_call_id": "c2", "content": "two"},
        ]
    )

    payload = messages_request(request, "claude-fable-5", default_max_tokens=1024)
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    merged = messages[2]["content"]
    assert isinstance(merged, list)
    assert [block["tool_use_id"] for block in merged] == ["c1", "c2"]


def test_malformed_tool_arguments_keep_the_call() -> None:
    """A malformed argument string is the call the model made; the block must not vanish."""
    request = _request(
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{not json"},
                    }
                ],
            }
        ]
    )

    payload = messages_request(request, "claude-fable-5", default_max_tokens=1024)
    messages = payload["messages"]
    assert isinstance(messages, list)
    blocks = messages[0]["content"]
    assert isinstance(blocks, list)
    assert blocks[0]["type"] == "tool_use"
    assert blocks[0]["input"] == {}


def test_one_breakpoint_closes_the_whole_prefix() -> None:
    """One breakpoint at the conversation end; the schemas must NOT carry a second one.

    A tools-block breakpoint sits under the minimum cacheable length and was measured to void
    the valid breakpoint on haiku-4-5, so its absence is the behavior under test.
    """
    payload = messages_request(_request(), "claude-fable-5", default_max_tokens=1024)

    tools = payload["tools"]
    assert isinstance(tools, list)
    assert "cache_control" not in tools[-1]
    messages = payload["messages"]
    assert isinstance(messages, list)
    blocks = messages[-1]["content"]
    assert isinstance(blocks, list)
    assert blocks[-1]["cache_control"] == CACHE_CONTROL_EPHEMERAL
    # The system prompt is inside the cached prefix, so it spends no breakpoint of its own.
    assert payload["system"] == "you are an agent"


def test_system_carries_the_breakpoint_when_there_are_no_messages() -> None:
    """With nothing in the conversation there is no last block, so the system prompt marks it."""
    payload = messages_request(
        _request(messages=[{"role": "system", "content": "you are an agent"}]),
        "claude-fable-5",
        default_max_tokens=1024,
    )

    system = payload["system"]
    assert isinstance(system, list)
    assert system[0]["cache_control"] == CACHE_CONTROL_EPHEMERAL


def test_caching_can_be_turned_off_to_measure_uncached_cost() -> None:
    payload = messages_request(
        _request(), "claude-fable-5", default_max_tokens=1024, cache_prompt=False
    )

    messages = payload["messages"]
    assert isinstance(messages, list)
    blocks = messages[-1]["content"]
    assert isinstance(blocks, list)
    assert "cache_control" not in blocks[-1]
    assert payload["system"] == "you are an agent"


def test_response_maps_text_tool_calls_and_finish_reason() -> None:
    raw = {
        "content": [
            {"type": "text", "text": "running it"},
            {"type": "tool_use", "id": "c1", "name": "bash", "input": {"command": "ls"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    response = messages_response(raw, "claude-fable-5")

    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == "running it"
    assert choice.message.tool_calls is not None
    call = choice.message.tool_calls[0]
    assert call.id == "c1"
    assert call.function.name == "bash"
    assert json.loads(call.function.arguments) == {"command": "ls"}


def test_response_usage_normalizes_cache_tokens_as_a_subset() -> None:
    """The API reports cache reads/writes BESIDE input_tokens; TokenUsage wants them inside."""
    raw = {
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 20,
        },
    }

    response = messages_response(raw, "claude-fable-5")

    assert response.usage is not None
    assert response.usage.prompt_tokens == 130
    assert response.usage.completion_tokens == 5
    extra = response.usage.model_dump()
    assert extra["cache_read_input_tokens"] == 100
    assert extra["cache_creation_input_tokens"] == 20


def test_response_maps_length_and_refusal_stops() -> None:
    for stop_reason, expected in (
        ("max_tokens", "length"),
        ("refusal", "content_filter"),
        ("end_turn", "stop"),
        ("pause_turn", "stop"),
    ):
        raw = {
            "content": [{"type": "text", "text": ""}],
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        assert messages_response(raw, "m").choices[0].finish_reason == expected
