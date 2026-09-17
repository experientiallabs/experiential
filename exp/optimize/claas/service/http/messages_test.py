"""Learner tool history retains arguments and rejects unsupported input."""

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.optimize.claas.service.http.inputs import parse_generation
from exp.optimize.claas.service.http.messages import chat_messages, response_messages


def test_response_function_history() -> None:
    """Keep a function call and result addressable by their original call ID."""
    messages = response_messages(
        [
            {"role": "user", "content": "look up an item"},
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": '{"id":2}',
            },
            {"type": "function_call_output", "call_id": "call-1", "output": "found"},
        ],
        None,
    )
    assert messages[1].assistant_action is not None
    assert messages[1].assistant_action.tool_calls[0].arguments == {"id": 2}
    assert messages[2].tool_call_id == "call-1"


def test_media_is_not_silently_dropped() -> None:
    """Reject unsupported observations before student generation."""
    with pytest.raises(ValueError, match="text and function"):
        chat_messages(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
        )


@pytest.mark.parametrize(
    "history",
    [
        [{"type": "function_call_output", "call_id": "unknown", "output": "value"}],
        [{"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{}"}],
        [
            {"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call-1", "output": "first"},
            {"type": "function_call_output", "call_id": "call-1", "output": "duplicate"},
        ],
        [
            {"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{}"},
            {"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{}"},
        ],
        [
            {"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{}"},
            {"role": "user", "content": "continue before the tool finished"},
        ],
    ],
)
def test_incomplete_or_ambiguous_tool_history_is_rejected(history: list[JsonObject]) -> None:
    """Require complete, uniquely identified calls and outputs before sampling again."""
    with pytest.raises(ValueError):
        parse_generation({"model": "student", "input": history}, "responses", "response")


def test_parallel_response_calls_accept_independent_outputs() -> None:
    """Multiple official Responses function items form one complete tool turn."""
    value = parse_generation(
        {
            "model": "student",
            "input": [
                {"type": "function_call", "call_id": "a", "name": "lookup", "arguments": "{}"},
                {"type": "function_call", "call_id": "b", "name": "lookup", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "b", "output": "second"},
                {"type": "function_call_output", "call_id": "a", "output": "first"},
            ],
        },
        "responses",
        "response",
    )
    assert [message.tool_call_id for message in value.messages[-2:]] == ["b", "a"]


@pytest.mark.parametrize("strict", [True, "false", 0])
def test_unsupported_or_malformed_strict_tool_option_is_rejected(strict: JsonValue) -> None:
    """Do not let lazy iterable SDK validation silently coerce tool generation settings."""
    with pytest.raises(ValueError, match="strict"):
        parse_generation(
            {
                "model": "student",
                "input": "hello",
                "tools": [
                    {"type": "function", "name": "lookup", "strict": strict},
                ],
            },
            "responses",
            "response",
        )


def test_duplicate_argument_keys_and_partial_output_metadata_are_rejected() -> None:
    """Ambiguous function arguments and unfinished model items are never treated as complete."""
    with pytest.raises(ValueError, match="duplicate"):
        response_messages(
            [
                {
                    "type": "function_call",
                    "call_id": "a",
                    "name": "lookup",
                    "arguments": '{"key":1,"key":2}',
                }
            ],
            None,
        )
    with pytest.raises(ValueError, match="completed"):
        response_messages(
            [{"role": "assistant", "content": "partial", "status": "incomplete"}], None
        )
    with pytest.raises(ValueError, match="annotations"):
        response_messages(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "answer",
                            "annotations": [{"type": "citation"}],
                        }
                    ],
                }
            ],
            None,
        )
