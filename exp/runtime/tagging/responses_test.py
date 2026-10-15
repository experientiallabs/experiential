"""Tests for reading complete response bodies from stored response documents.

The frames follow the shapes the native collector tests use: Chat chunks ending in ``[DONE]``
and Messages block lifecycles. The capture adapter tests reuse them.
"""

from __future__ import annotations

import json

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.tagging.responses import completed_response, readable_responses


def _chunk(delta: JsonObject, finish: str | None = None) -> JsonObject:
    """Build one Chat stream chunk.

    Args:
        delta: Choice delta.
        finish: Finish reason, or None mid-stream.

    Returns:
        The chunk.
    """
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "worker",
        "created": 1,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


CHAT_FRAMES: list[JsonValue] = [
    _chunk({"role": "assistant", "content": "Record A "}),
    _chunk({"content": "is missing."}),
    _chunk(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"id":'},
                }
            ]
        }
    ),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"B"}'}}]}, "tool_calls"),
    "[DONE]",
]
CHAT_BODY: JsonObject = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Record A is missing.",
                "tool_calls": [
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"id":"B"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
}
MESSAGES_FRAMES: list[JsonValue] = [
    {
        "type": "message_start",
        "message": {
            "type": "message",
            "id": "msg",
            "role": "assistant",
            "content": [],
            "usage": {"input_tokens": 2},
            "stop_reason": None,
        },
    },
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "x"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "signature_delta", "signature": "s"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "content_block_start",
        "index": 1,
        "content_block": {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {}},
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": '{"cmd":'},
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": '"ls"}'},
    },
    {"type": "content_block_stop", "index": 1},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 3}},
    {"type": "message_stop"},
]


def _sse(frames: list[JsonValue]) -> JsonObject:
    """Wrap frames in a complete stored stream document.

    Args:
        frames: Stored ``data`` payloads.

    Returns:
        The response document.
    """
    return {
        "kind": "sse",
        "status": 200,
        "frames": frames,
        "truncated": False,
        "client_disconnected": False,
    }


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"kind": "json", "status": 500, "body": CHAT_BODY}, "status is 500"),
        ({**_sse(CHAT_FRAMES), "truncated": True}, "cut short"),
        ({**_sse(CHAT_FRAMES), "client_disconnected": True}, "cut short"),
        (_sse(CHAT_FRAMES[:-1]), "end marker"),
        (_sse([*CHAT_FRAMES[:2], {"error": {"message": "x"}}, "[DONE]"]), "reports an error"),
        (
            {"kind": "json", "status": 200, "body": {**CHAT_BODY, "error": {"message": "x"}}},
            "error",
        ),
        ({"kind": "wire", "status": 200}, "unknown response kind"),
    ],
)
def test_failed_or_unfinished_responses_are_not_tagged(document: JsonObject, message: str) -> None:
    """Failed, cut-short, unterminated or erroring responses are rejected with a reason.

    Args:
        document: Stored response document.
        message: Expected reason fragment.
    """
    with pytest.raises(ValueError, match=message):
        completed_response("chat_completions", document)


def test_streams_are_taken_as_the_collector_assembles_them() -> None:
    """Chat and Messages streams come back as the native assembler builds them."""
    chat = completed_response("chat_completions", _sse(CHAT_FRAMES))
    assert chat["choices"] == CHAT_BODY["choices"]
    message = completed_response("messages", _sse(MESSAGES_FRAMES))
    content = message["content"]
    assert isinstance(content, list) and isinstance(content[1], dict)
    assert content[1]["input"] == {"cmd": "ls"}
    assert message["stop_reason"] == "tool_use"


def test_messages_tool_inputs_keep_exact_wide_integers() -> None:
    """The collector's exact source for a wide integer tool input replaces its rounded copy."""
    frames = list(MESSAGES_FRAMES)
    frames[7] = {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": "18446744073709551617}"},
    }
    content = completed_response("messages", _sse(frames))["content"]
    assert isinstance(content, list) and isinstance(content[1], dict)
    assert content[1]["input"] == {"cmd": 18446744073709551617}
    assert "capture_input_source_json" not in content[1]


def test_exact_sources_are_preferred_over_projections() -> None:
    """Retained source JSON keeps wide integers and text the projection had to normalize."""
    exact = json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "a\u0000b \ud800",
                        "n": 18446744073709551617,
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    )
    projected: JsonObject = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "a\ufffdb \ufffd", "n": 1.8e19},
                "finish_reason": "stop",
            }
        ]
    }
    document: JsonObject = {"kind": "json", "status": 200, "body": projected, "source_json": exact}
    choices = completed_response("chat_completions", document)["choices"]
    assert isinstance(choices, list) and isinstance(choices[0], dict)
    message = choices[0]["message"]
    assert isinstance(message, dict)
    assert message["n"] == 18446744073709551617
    assert message["content"] == "a\u0000b \ufffd"


@pytest.mark.parametrize("status", ["completed", "incomplete"])
def test_responses_streams_take_their_finished_terminal_response(status: str) -> None:
    """A Responses stream body is its terminal response, including a length-limited one.

    Args:
        status: The terminal response status.
    """
    response: JsonObject = {
        "status": status,
        "output": [
            {
                "type": "message",
                "id": "m",
                "role": "assistant",
                "status": status,
                "content": [{"type": "output_text", "text": "Hi", "annotations": []}],
            }
        ],
    }
    frames: list[JsonValue] = [
        {"type": "response.created", "response": {"status": "in_progress"}},
        {"type": f"response.{status}", "response": response},
    ]
    assert completed_response("responses", _sse(frames)) == response


def test_unfinished_responses_are_not_tagged() -> None:
    """A Responses stream without a terminal event, or a body still in progress, is rejected."""
    created: JsonValue = {"type": "response.created", "response": {"status": "in_progress"}}
    with pytest.raises(ValueError, match="end marker"):
        completed_response("responses", _sse([created]))
    queued: JsonObject = {"kind": "json", "status": 200, "body": {"status": "queued", "output": []}}
    with pytest.raises(ValueError, match="not finished"):
        completed_response("responses", queued)


def test_responses_streams_keep_exact_wide_integers() -> None:
    """The collector's exact terminal event replaces a rounded Responses body."""
    response: JsonObject = {
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "id": "fc",
                "call_id": "c1",
                "name": "seek",
                "arguments": "{}",
                "status": "completed",
                "limit": 18446744073709551617,
            }
        ],
    }
    frames: list[JsonValue] = [{"type": "response.completed", "response": response}]
    body = completed_response("responses", _sse(frames))
    output = body["output"]
    assert isinstance(output, list) and isinstance(output[0], dict)
    assert output[0]["limit"] == 18446744073709551617


def test_readable_responses_drop_response_only_item_metadata() -> None:
    """Item metadata only responses carry is removed before request-schema decoding."""
    body: JsonObject = {
        "status": "completed",
        "output": [
            {"type": "function_call_output", "call_id": "c1", "output": "ok", "created_by": "x"},
            {"type": "custom_tool_call_output", "call_id": "c2", "output": "ok", "caller": {}},
        ],
    }
    readable, _ = readable_responses(body)
    output = readable["output"]
    assert isinstance(output, list)
    assert all(
        isinstance(item, dict) and not {"caller", "created_by"} & item.keys() for item in output
    )
