# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Chat tool turns keep their results directly after the calling message."""

from __future__ import annotations

from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.models import ToolCall
from exp.runtime.gateway.contracts import (
    ExposedReasoningContentBlock,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.openai_payloads import openai_compatible_stream_payload
from exp.runtime.models.providers.tool_turn_order import (
    TOOL_TURN_TEXT_FOLD_DISCLOSURE,
    disclose_tool_turn_text_fold,
    fold_assistant_text_into_tool_turns,
)
from exp.runtime.openai_protocol import decode_responses

_PROGRESS = "I will read the synthetic markers now."
_STRING_OUTPUTS: tuple[JsonValue, JsonValue] = ("SYNTHETIC_TOOLS_OK", "SYNTHETIC_TOOLS_OK_2")
_MCP_OUTPUTS: tuple[JsonValue, JsonValue] = (
    [{"type": "input_text", "text": "SYNTHETIC_TOOLS_OK"}],
    [{"type": "input_text", "text": "SYNTHETIC_TOOLS_OK_2"}],
)


def _codex_history(outputs: tuple[JsonValue, JsonValue]) -> GatewayRequest:
    """The customer-reported Codex history: calls, progress message, then outputs."""
    return decode_responses(
        cast(
            JsonObject,
            {
                "model": "deepseek-v4.1-flash",
                "stream": True,
                "store": False,
                "provider": {"zdr": True},
                "reasoning": {"effort": "max"},
                "max_output_tokens": 2048,
                "tools": [
                    {
                        "type": "function",
                        "name": "read_marker",
                        "parameters": {
                            "type": "object",
                            "properties": {"index": {"type": "integer"}},
                            "required": ["index"],
                        },
                    }
                ],
                "input": [
                    {"role": "user", "content": "Read both synthetic markers."},
                    {
                        "type": "function_call",
                        "call_id": "call_fixture_1",
                        "name": "read_marker",
                        "arguments": '{"index":1}',
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_fixture_2",
                        "name": "read_marker",
                        "arguments": '{"index":2}',
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": _PROGRESS}],
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_fixture_1",
                        "output": outputs[0],
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_fixture_2",
                        "output": outputs[1],
                    },
                ],
            },
        )
    ).request


@pytest.mark.parametrize("outputs", [_STRING_OUTPUTS, _MCP_OUTPUTS], ids=["string", "input_text"])
def test_codex_progress_message_folds_into_the_tool_call_message(
    outputs: tuple[JsonValue, JsonValue],
) -> None:
    payload = openai_compatible_stream_payload("deepseek-v4.1-flash", _codex_history(outputs))
    messages = cast(list[JsonObject], payload["messages"])
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "tool"]
    assistant = messages[1]
    assert assistant["content"] == _PROGRESS
    assert [call["id"] for call in cast(list[JsonObject], assistant["tool_calls"])] == [
        "call_fixture_1",
        "call_fixture_2",
    ]
    assert [message["tool_call_id"] for message in messages[2:]] == [
        "call_fixture_1",
        "call_fixture_2",
    ]
    assert [message["content"] for message in messages[2:]] == [
        "SYNTHETIC_TOOLS_OK",
        "SYNTHETIC_TOOLS_OK_2",
    ]


def test_the_rewrite_is_disclosed_on_a_chat_route_only() -> None:
    messages = _codex_history(_STRING_OUTPUTS).messages
    chat = GatewayWireProfile(dialect="openai_compatible", url="https://chat.test")
    responses = GatewayWireProfile(dialect="openai_responses", url="https://responses.test")
    ignored: list[str] = []
    disclose_tool_turn_text_fold((responses,), messages, ignored)
    assert ignored == []
    disclose_tool_turn_text_fold((responses, chat), messages, ignored)
    disclose_tool_turn_text_fold((chat,), messages, ignored)
    assert ignored == [TOOL_TURN_TEXT_FOLD_DISCLOSURE]


def _call(call_id: str) -> ToolCall:
    return ToolCall(call_id=call_id, name="read_marker", arguments={"index": 1})


def _tool(call_id: str) -> GatewayMessage:
    return GatewayMessage(role="tool", content="ok", tool_call_id=call_id)


def test_a_valid_order_is_returned_untouched() -> None:
    messages = (
        GatewayMessage(role="user", content="go"),
        GatewayMessage(role="assistant", content=_PROGRESS),
        GatewayMessage(role="assistant", tool_calls=(_call("a"),)),
        _tool("a"),
        GatewayMessage(role="assistant", tool_calls=(_call("b"),)),
        GatewayMessage(role="assistant", content="Waiting on the result."),
    )
    assert fold_assistant_text_into_tool_turns(messages) is messages


def test_existing_content_and_a_single_reasoning_carrier_merge() -> None:
    reasoning = ExposedReasoningContentBlock(content="thinking")
    messages = (
        GatewayMessage(role="user", content="go"),
        GatewayMessage(role="assistant", content="First.", tool_calls=(_call("a"),)),
        GatewayMessage(role="assistant", content="Second.", provider_reasoning=(reasoning,)),
        _tool("a"),
    )
    folded = fold_assistant_text_into_tool_turns(messages)
    assert [message.role for message in folded] == ["user", "assistant", "tool"]
    assert folded[1].content == "First.\n\nSecond."
    assert folded[1].tool_calls == (_call("a"),)
    assert folded[1].provider_reasoning == (reasoning,)


def test_text_that_cannot_merge_moves_ahead_of_the_calls() -> None:
    first = ExposedReasoningContentBlock(content="one")
    second = ExposedReasoningContentBlock(content="two")
    calls = GatewayMessage(role="assistant", tool_calls=(_call("a"),), provider_reasoning=(first,))
    text = GatewayMessage(role="assistant", content=_PROGRESS, provider_reasoning=(second,))
    messages = (GatewayMessage(role="user", content="go"), calls, text, _tool("a"))
    assert fold_assistant_text_into_tool_turns(messages) == (
        messages[0],
        text,
        calls,
        messages[3],
    )
