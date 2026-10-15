"""Tests for reading complete response bodies from stored response documents."""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.tagging.exchanges_test import CHAT_BODY, CHAT_FRAMES, MESSAGES_FRAMES, _chunk
from exp.runtime.tagging.responses import completed_response


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"kind": "json", "status": 500, "body": CHAT_BODY}, "status is 500"),
        (
            {
                "kind": "sse",
                "status": 200,
                "frames": CHAT_FRAMES,
                "truncated": True,
                "client_disconnected": False,
            },
            "cut short",
        ),
        (
            {
                "kind": "sse",
                "status": 200,
                "frames": CHAT_FRAMES,
                "truncated": False,
                "client_disconnected": True,
            },
            "cut short",
        ),
        (
            {
                "kind": "sse",
                "status": 200,
                "frames": CHAT_FRAMES[:-1],
                "truncated": False,
                "client_disconnected": False,
            },
            "terminal state",
        ),
        (
            {
                "kind": "sse",
                "status": 200,
                "frames": [*CHAT_FRAMES[:2], {"error": {"message": "x"}}],
                "truncated": False,
                "client_disconnected": False,
            },
            "reports an error",
        ),
        ({"kind": "json", "status": 200, "body": {"choices": [], "error": None}}, "terminal"),
        ({"kind": "wire", "status": 200}, "unknown response kind"),
    ],
)
def test_incomplete_or_failed_responses_are_not_tagged(document: JsonObject, message: str) -> None:
    """Failed, cut-short, unterminated or erroring responses are rejected with a reason.

    Args:
        document: Stored response document.
        message: Expected reason fragment.
    """
    with pytest.raises(ValueError, match=message):
        completed_response("chat_completions", document)


@pytest.mark.parametrize(
    "frames",
    [
        [frame for index, frame in enumerate(MESSAGES_FRAMES) if index != 8],
        [
            *MESSAGES_FRAMES[:9],
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": "x"},
            },
            *MESSAGES_FRAMES[9:],
        ],
        [*MESSAGES_FRAMES, {"type": "ping"}],
    ],
    ids=["unclosed-block", "delta-after-stop", "frame-after-stop"],
)
def test_messages_streams_with_broken_lifecycles_are_rejected(frames: list[JsonValue]) -> None:
    """Streams the collector's assembler would mark invalid are not complete captures.

    Args:
        frames: Stored frames with one lifecycle violation.
    """
    document: JsonObject = {
        "kind": "sse",
        "status": 200,
        "frames": frames,
        "truncated": False,
        "client_disconnected": False,
    }
    with pytest.raises(ValueError, match="terminal state"):
        completed_response("messages", document)


def test_chat_stream_with_an_unindexed_choice_is_rejected() -> None:
    """A chunk whose choice names no index fails the assembler and is not tagged."""
    frames: list[JsonValue] = [
        _chunk({"role": "assistant", "content": "done"}, "stop"),
        {"choices": [{}]},
        "[DONE]",
    ]
    document: JsonObject = {
        "kind": "sse",
        "status": 200,
        "frames": frames,
        "truncated": False,
        "client_disconnected": False,
    }
    with pytest.raises(ValueError, match="terminal state"):
        completed_response("chat_completions", document)


def test_chat_stream_with_an_unindexed_tool_delta_is_rejected() -> None:
    """A tool call delta without its index fails the assembler and is not tagged."""
    frames: list[JsonValue] = [
        _chunk({"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "f"}}]}),
        _chunk({"content": "done"}, "stop"),
        "[DONE]",
    ]
    document: JsonObject = {
        "kind": "sse",
        "status": 200,
        "frames": frames,
        "truncated": False,
        "client_disconnected": False,
    }
    with pytest.raises(ValueError, match="terminal state"):
        completed_response("chat_completions", document)


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


def test_chat_stream_whose_deltas_cannot_merge_is_rejected() -> None:
    """Text content followed by list content fails the assembler's merge and is not tagged."""
    frames: list[JsonValue] = [
        _chunk({"role": "assistant", "content": "Hello"}),
        _chunk({"content": [{"type": "text", "text": "world"}]}),
        _chunk({}, "stop"),
        "[DONE]",
    ]
    with pytest.raises(ValueError, match="terminal state"):
        completed_response("chat_completions", _sse(frames))
    assert completed_response("chat_completions", _sse(CHAT_FRAMES))["choices"]


def test_messages_stream_with_an_unknown_delta_is_rejected() -> None:
    """A block delta of unknown type fails the assembler and is not tagged."""
    frames = list(MESSAGES_FRAMES)
    frames[2] = {"type": "content_block_delta", "index": 0, "delta": {"type": "mystery_delta"}}
    with pytest.raises(ValueError, match="terminal state"):
        completed_response("messages", _sse(frames))


def test_messages_block_start_without_a_block_is_rejected() -> None:
    """A block start that carries no block fails the assembler and is not tagged."""
    frames = list(MESSAGES_FRAMES)
    frames[1] = {"type": "content_block_start", "index": 0}
    with pytest.raises(ValueError, match="terminal state"):
        completed_response("messages", _sse(frames))


def test_custom_call_repair_keeps_text_in_the_same_delta() -> None:
    """Text arriving in a delta after a custom call continuation is kept."""
    frames: list[JsonValue] = [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "c1",
                        "type": "custom",
                        "custom": {"name": "apply_patch", "input": "*** A"},
                    }
                ],
            }
        ),
        _chunk({"tool_calls": [{"index": 0, "custom": {"input": "B"}}], "content": "Patched."}),
        _chunk({}, "tool_calls"),
        "[DONE]",
    ]
    choices = completed_response("chat_completions", _sse(frames))["choices"]
    assert isinstance(choices, list) and isinstance(choices[0], dict)
    message = choices[0]["message"]
    assert isinstance(message, dict) and message["content"] == "Patched."
