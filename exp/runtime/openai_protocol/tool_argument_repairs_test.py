"""Replayed tool-call arguments that encode no object decode to a faithful stand-in."""

from __future__ import annotations

import json

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.model import ToolArgumentsTreatment, ToolCall
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses
from exp.runtime.openai_protocol.tool_argument_repairs import (
    UNPARSED_ARGUMENTS_KEY,
    closed_truncated_object,
    replayed_tool_arguments,
)

_CASES: tuple[tuple[str, JsonObject, ToolArgumentsTreatment | None], ...] = (
    ('{"city": "Paris"}', {"city": "Paris"}, None),
    (json.dumps(json.dumps({"city": "Paris"})), {"city": "Paris"}, "double_encoded_json"),
    ('{"path": "src/ma', {"path": "src/ma"}, "truncated_json"),
    ('{"a": 1, "b', {"a": 1}, "truncated_json"),
    ("[1, 2]", {UNPARSED_ARGUMENTS_KEY: "[1, 2]"}, "non_object_json"),
    ('"ls -la"', {UNPARSED_ARGUMENTS_KEY: '"ls -la"'}, "non_object_json"),
    ("null", {UNPARSED_ARGUMENTS_KEY: "null"}, "non_object_json"),
    ('{"a": 1}{"b": 2}', {UNPARSED_ARGUMENTS_KEY: '{"a": 1}{"b": 2}'}, "invalid_json"),
    (
        "<tool_call>ls</tool_call>",
        {UNPARSED_ARGUMENTS_KEY: "<tool_call>ls</tool_call>"},
        "invalid_json",
    ),
)


@pytest.mark.parametrize(("raw", "arguments", "treatment"), _CASES)
def test_replayed_text_maps_to_one_object_and_names_its_treatment(
    raw: str, arguments: JsonObject, treatment: ToolArgumentsTreatment | None
) -> None:
    """An object decodes as itself; everything else names how its stand-in was made."""
    assert replayed_tool_arguments(raw) == (arguments, treatment)


@pytest.mark.parametrize(
    ("raw", "closed"),
    (
        ("{", {}),
        ('{"a"', {}),
        ('{"a":', {}),
        ('{"a": tru', {}),
        # A number at the cut may have been longer: dropped, never guessed.
        ('{"a": "x", "n": 12', {"a": "x"}),
        ('{"a": [1, 2', {"a": [1]}),
        ('{"a": ["x", "y', {"a": ["x", "y"]}),
        ('{"a": {"b": "c"}, ', {"a": {"b": "c"}}),
        ('{"a": "x\\', {"a": "x"}),
        ('{"a": "x\\u12', {"a": "x"}),
        # Half a surrogate pair is dropped, so the stand-in never carries a lone one.
        ('{"a": "x\\ud83d', {"a": "x"}),
        ('{"a": "x\\ud83d\\ude00', {"a": "x\U0001f600"}),
        ('  {"a": "b"', {"a": "b"}),
    ),
)
def test_a_cut_off_object_closes_on_what_it_completed(raw: str, closed: JsonObject) -> None:
    """Closing keeps complete members and string text, and drops what the cut shortened."""
    assert closed_truncated_object(raw) == closed


@pytest.mark.parametrize("raw", ('{"a": 1}}', '{"a" 1', "[1, 2", "nope", '{"a": 1}{"b"'))
def test_malformed_text_is_never_closed(raw: str) -> None:
    """Only a valid object prefix is repaired; malformed text is wrapped instead."""
    assert closed_truncated_object(raw) is None


def _chat_history(arguments: str) -> JsonObject:
    """A Chat body replaying one assistant call with ``arguments`` and its result."""
    return {
        "model": "coding",
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": arguments},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
            {"role": "user", "content": "next"},
        ],
    }


def _responses_history(arguments: str) -> JsonObject:
    """A Responses body replaying one function call with ``arguments`` and its output."""
    return {
        "model": "coding",
        "input": [
            {"role": "user", "content": "go"},
            {"type": "function_call", "call_id": "call_1", "name": "read", "arguments": arguments},
            {"type": "function_call_output", "call_id": "call_1", "output": "done"},
            {"role": "user", "content": "next"},
        ],
    }


def _only_call(messages: tuple[object, ...]) -> ToolCall:
    """Return the single replayed tool call of a decoded history."""
    calls = [call for message in messages for call in getattr(message, "tool_calls", ())]
    assert len(calls) == 1
    return calls[0]


@pytest.mark.parametrize(("raw", "arguments", "treatment"), _CASES)
def test_chat_history_serves_every_argument_text(
    raw: str, arguments: JsonObject, treatment: ToolArgumentsTreatment | None
) -> None:
    """A Chat history call decodes whatever text it carries, keeping the text byte-exact."""
    decoded = decode_chat(_chat_history(raw))
    call = _only_call(decoded.request.messages)
    assert (call.raw_arguments, call.arguments, call.arguments_treatment) == (
        raw,
        arguments,
        treatment,
    )
    # The decoder rewrites nothing on the wire, so it discloses nothing; a
    # rung that sends the stand-in discloses it (history_tool_arguments).
    assert decoded.request.ignored_parameters == ()


@pytest.mark.parametrize(("raw", "arguments", "treatment"), _CASES)
def test_responses_history_serves_every_argument_text(
    raw: str, arguments: JsonObject, treatment: ToolArgumentsTreatment | None
) -> None:
    """A Responses function_call decodes whatever text it carries.

    A double-encoded string is repaired at the payload, before the decoder, so
    the call retains the inner object text under the existing disclosure.
    """
    decoded = decode_responses(_responses_history(raw))
    call = _only_call(decoded.request.messages)
    if treatment == "double_encoded_json":
        assert call.raw_arguments == json.loads(raw)
        assert call.arguments_treatment is None
        assert decoded.request.ignored_parameters == (
            "input.arguments->decoded(double_encoded_json)",
        )
        return
    assert (call.raw_arguments, call.arguments, call.arguments_treatment) == (
        raw,
        arguments,
        treatment,
    )
    assert decoded.request.ignored_parameters == ()


@pytest.mark.parametrize("blank", ("", "   ", "\n"))
def test_blank_arguments_are_the_empty_object(blank: str) -> None:
    """A zero-argument call echoed as blank text is the canonical empty object, untreated."""
    call = _only_call(decode_chat(_chat_history(blank)).request.messages)
    assert (call.raw_arguments, call.arguments, call.arguments_treatment) == ("{}", {}, None)
