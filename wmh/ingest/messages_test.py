"""Tests for the chat/tool-call -> Trace converter (chat-json adapter)."""

from __future__ import annotations

import json
from pathlib import Path

from wmh.core.types import ActionKind
from wmh.ingest import get_adapter
from wmh.ingest.messages import ChatMessagesAdapter

_CONVO = {
    "id": "conv-1",
    "metadata": {"benchmark": "demo"},
    "messages": [
        {"role": "user", "content": "what's the weather in Paris?"},
        {
            "role": "assistant",
            "content": "let me check",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "18C and sunny"},
        {"role": "assistant", "content": "It's 18C and sunny in Paris."},
    ],
}


def test_chat_json_adapter_is_registered() -> None:
    assert get_adapter("chat-json").name == "chat-json"


def test_converts_tool_call_and_final_message(tmp_path: Path) -> None:
    path = tmp_path / "convo.json"
    path.write_text(json.dumps(_CONVO), encoding="utf-8")

    traces = ChatMessagesAdapter().from_file(str(path))

    assert len(traces) == 1
    trace = traces[0]
    assert trace.metadata == {"benchmark": "demo"}
    # 2 steps: the get_weather tool call (paired with its result) + the final assistant message.
    assert len(trace.steps) == 2

    call = trace.steps[0]
    assert call.action.kind == ActionKind.TOOL_CALL
    assert call.action.name == "get_weather"
    assert call.action.arguments == {"city": "Paris"}
    assert call.observation.content == "18C and sunny"
    assert call.task == "what's the weather in Paris?"

    final = trace.steps[1]
    assert final.action.kind == ActionKind.MESSAGE
    assert final.action.content == "It's 18C and sunny in Paris."
    assert final.observation.content == ""


def test_tool_result_matched_by_order_when_no_id(tmp_path: Path) -> None:
    convo = {
        "messages": [
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "ls", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "a.txt\nb.txt"},  # no tool_call_id -> matched in order
        ]
    }
    path = tmp_path / "c.json"
    path.write_text(json.dumps(convo), encoding="utf-8")

    traces = ChatMessagesAdapter().from_file(str(path))
    assert len(traces) == 1
    step = traces[0].steps[0]
    assert step.action.name == "ls"
    assert step.observation.content == "a.txt\nb.txt"


def test_jsonl_multiple_conversations(tmp_path: Path) -> None:
    c2 = {
        "id": "conv-2",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }
    path = tmp_path / "convos.jsonl"
    path.write_text(json.dumps(_CONVO) + "\n" + json.dumps(c2) + "\n", encoding="utf-8")

    traces = ChatMessagesAdapter().from_file(str(path))
    assert len(traces) == 2


def test_bare_message_list_is_one_conversation(tmp_path: Path) -> None:
    path = tmp_path / "bare.json"
    path.write_text(json.dumps(_CONVO["messages"]), encoding="utf-8")

    traces = ChatMessagesAdapter().from_file(str(path))
    assert len(traces) == 1
    assert traces[0].steps[0].action.name == "get_weather"


def test_flattened_tool_call_shape(tmp_path: Path) -> None:
    # Some frameworks flatten the call (no nested "function").
    convo = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "tool_calls": [{"id": "x", "name": "f", "arguments": {"a": 1}}]},
            {"role": "tool", "tool_call_id": "x", "content": "ok"},
        ]
    }
    path = tmp_path / "flat.json"
    path.write_text(json.dumps(convo), encoding="utf-8")

    step = ChatMessagesAdapter().from_file(str(path))[0].steps[0]
    assert step.action.name == "f"
    assert step.action.arguments == {"a": 1}  # arguments object re-parsed by the normalizer
    assert step.observation.content == "ok"


def test_empty_tool_arguments_become_empty_dict(tmp_path: Path) -> None:
    """A tool call with no/blank arguments yields {} args, not a junk {"value": ""}."""
    convo = {
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "noop"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "done"},
        ]
    }
    path = tmp_path / "c.json"
    path.write_text(json.dumps(convo), encoding="utf-8")

    step = ChatMessagesAdapter().from_file(str(path))[0].steps[0]
    assert step.action.name == "noop"
    assert step.action.arguments == {}
