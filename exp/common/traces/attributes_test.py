"""Tests for the canonical span-attribute readers."""

from __future__ import annotations

import json

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.traces.attributes import (
    assistant_completion,
    decoded_json_value,
    is_tool_result,
    jsonish_text,
    message_text,
    tool_arguments,
    tool_name,
    tool_result,
    user_input,
)


def test_decoded_json_value_returns_malformed_text_unchanged() -> None:
    """Attributes that only look like JSON stay readable instead of raising."""
    assert decoded_json_value('{"a": 1}') == {"a": 1}
    assert decoded_json_value("{not json") == "{not json"
    assert decoded_json_value(None) is None
    assert decoded_json_value({"already": "decoded"}) == {"already": "decoded"}


def test_message_text_joins_structured_content_parts() -> None:
    """Both plain strings and OpenAI-style content parts yield readable text."""
    assert message_text("  hello  ") == "hello"
    assert message_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    assert message_text([{"type": "image"}]) is None
    assert message_text("   ") is None


def test_assistant_completion_prefers_direct_attributes_then_messages() -> None:
    """Direct completion attributes win; output messages are the documented fallback."""
    assert assistant_completion({"gen_ai.completion": " done "}) == "done"
    assert assistant_completion({"gen_ai.response.text": "answered"}) == "answered"
    encoded = json.dumps([{"role": "assistant", "content": "from messages"}])
    assert assistant_completion({"gen_ai.output.messages": encoded}) == "from messages"
    assert assistant_completion({}) is None


def test_assistant_completion_reads_the_latest_assistant_turn() -> None:
    """A multi-turn span reports its last assistant message, not its first."""
    messages = [
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "follow up"},
        {"role": "model", "content": "second"},
    ]
    assert assistant_completion({"gen_ai.output.messages": messages}) == "second"


def test_user_input_reads_the_latest_user_turn() -> None:
    """Human and user roles are both accepted, newest first."""
    messages = [{"role": "user", "content": "older"}, {"role": "human", "content": "newer"}]
    assert user_input({"gen_ai.input.messages": messages}) == "newer"
    assert user_input({}) is None


def test_tool_name_rejects_blank_evidence() -> None:
    """A blank tool name is absent evidence, never an empty tool."""
    assert tool_name({"gen_ai.tool.name": " lookup_order "}) == "lookup_order"
    assert tool_name({"gen_ai.tool.name": "   "}) is None
    assert tool_name({"gen_ai.tool.name": 7}) is None
    assert tool_name({}) is None


def test_tool_arguments_decode_without_losing_absent_evidence() -> None:
    """Encoded arguments decode; a span that records none reports none."""
    assert tool_arguments({"gen_ai.tool.call.arguments": '{"id":"INC-1"}'}) == {"id": "INC-1"}
    assert tool_arguments({}) is None


@pytest.mark.parametrize("key", ["gen_ai.tool.message", "gen_ai.tool.output"])
def test_tool_result_accepts_either_normalizer_key(key: str) -> None:
    """The two normalizers disagree on the result key, so both are read."""
    assert tool_result({key: '{"status":"ok"}'}) == {"status": "ok"}


def test_tool_result_prefers_the_message_key_when_both_are_present() -> None:
    """A span carrying both keys resolves deterministically."""
    attributes: JsonObject = {
        "gen_ai.tool.message": "preferred",
        "gen_ai.tool.output": "ignored",
    }
    assert tool_result(attributes) == "preferred"
    assert tool_result({}) is None


def test_is_tool_result_distinguishes_calls_from_observations() -> None:
    """Only the canonical tool-result operation reports an observation."""
    assert is_tool_result({"gen_ai.operation.name": "execute_tool"}) is True
    assert is_tool_result({"gen_ai.operation.name": "chat"}) is False
    assert is_tool_result({}) is False


def test_jsonish_text_formats_json_and_passes_through_plain_text() -> None:
    """Readable evidence stays readable whether or not it was encoded JSON."""
    assert jsonish_text('{"b":1,"a":2}') == '{\n  "a": 2,\n  "b": 1\n}'
    assert jsonish_text("plain prose") == "plain prose"
