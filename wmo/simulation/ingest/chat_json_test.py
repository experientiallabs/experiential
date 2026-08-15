"""Tests for chat JSON conversation normalization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmo.simulation.ingest.chat_json import load_chat_json_file
from wmo.simulation.ingest.vendor_records import VendorTraceFormatError
from wmo.simulation.ingest.vendor_trace import SYNTHETIC_TIME_ATTRIBUTE


def _conversation() -> dict[str, object]:
    """Return one tool-using conversation with declared model identity."""
    return {
        "trace_id": "conversation-1",
        "provider": "openai",
        "model": "gpt-4o",
        "metadata": {"tenant": "acme"},
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "name": "get_weather", "tool_call_id": "call-1", "content": "18C"},
            {"role": "assistant", "content": "It is 18C in Paris."},
        ],
    }


def test_load_chat_json_file_pairs_tool_calls(tmp_path: Path) -> None:
    """One conversation becomes one trace with paired call, result, and completion spans."""
    path = tmp_path / "chat.json"
    path.write_text(json.dumps(_conversation()), encoding="utf-8")

    result = load_chat_json_file(path)

    assert result.issues == ()
    assert len(result.traces) == 1
    trace = result.traces[0]
    assert trace.task == "What is the weather in Paris?"
    assert trace.initial_context == {}
    assert [span.name for span in trace.spans] == [
        "agent.model_call",
        "agent.tool_call",
        "agent.model_call",
    ]
    call, tool_result, completion = trace.spans
    assert call.attributes["gen_ai.tool.name"] == "get_weather"
    assert call.attributes["gen_ai.tool.call.id"] == "call-1"
    assert tool_result.attributes["gen_ai.tool.call.id"] == "call-1"
    assert tool_result.attributes["gen_ai.tool.message"] == "18C"
    assert tool_result.parent_span_id == call.span_id
    assert completion.attributes["gen_ai.completion"] == "It is 18C in Paris."
    assert call.model is not None
    assert call.model.provider == "openai"
    assert call.model.model_id == "gpt-4o"
    assert call.attributes[SYNTHETIC_TIME_ATTRIBUTE] is True
    assert call.attributes["wmo.trace.metadata"] == {"tenant": "acme"}
    assert call.attributes["wmo.source.vendor"] == "chat-json"
    assert call.attributes["wmo.source.trace.id"] == "conversation-1"


def test_load_chat_json_file_accepts_bare_message_array(tmp_path: Path) -> None:
    """A bare message array is read as one conversation with a derived trace key."""
    path = tmp_path / "chat.json"
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    path.write_text(json.dumps(messages), encoding="utf-8")

    result = load_chat_json_file(path)

    assert len(result.traces) == 1
    assert result.traces[0].task == "hello"


def test_load_chat_json_file_uses_declared_timestamps(tmp_path: Path) -> None:
    """Declared message timestamps replace assigned ordinal timestamps."""
    path = tmp_path / "chat.jsonl"
    conversation = {
        "trace_id": "conversation-2",
        "messages": [
            {"role": "user", "content": "hello", "timestamp": "2026-01-01T00:00:00Z"},
            {"role": "assistant", "content": "hi", "timestamp": "2026-01-01T00:00:05Z"},
        ],
    }
    path.write_text(f"{json.dumps(conversation)}\n", encoding="utf-8")

    result = load_chat_json_file(path)

    span = result.traces[0].spans[0]
    assert SYNTHETIC_TIME_ATTRIBUTE not in span.attributes
    assert span.attributes["wmo.source.span.started_at"] == "2026-01-01T00:00:05+00:00"


def test_load_chat_json_file_retains_malformed_jsonl_lines(tmp_path: Path) -> None:
    """A malformed JSONL line is excluded with an explicit issue instead of being skipped."""
    path = tmp_path / "chat.jsonl"
    good = {"trace_id": "conversation-3", "messages": _conversation()["messages"]}
    path.write_text(f"{json.dumps(good)}\n{{not json\n", encoding="utf-8")

    result = load_chat_json_file(path)

    assert len(result.traces) == 1
    assert [issue.source_record for issue in result.issues] == ["line-2"]


def test_load_chat_json_file_excludes_conversation_without_user_message(tmp_path: Path) -> None:
    """A conversation with no user message is excluded rather than given invented intent."""
    path = tmp_path / "chat.json"
    path.write_text(
        json.dumps({"trace_id": "conversation-4", "messages": [{"role": "assistant", "c": 1}]}),
        encoding="utf-8",
    )

    result = load_chat_json_file(path)

    assert result.traces == ()
    assert [issue.source_record for issue in result.issues] == ["record-1"]


def test_load_chat_json_file_rejects_partial_model_identity(tmp_path: Path) -> None:
    """A conversation that names a model without a provider is excluded, not completed."""
    path = tmp_path / "chat.json"
    path.write_text(
        json.dumps(
            {
                "trace_id": "conversation-5",
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = load_chat_json_file(path)

    assert result.traces == ()
    assert "provider" in result.issues[0].message


def test_load_chat_json_file_rejects_unreadable_export(tmp_path: Path) -> None:
    """A missing export fails fast instead of returning an empty result."""
    with pytest.raises(VendorTraceFormatError):
        load_chat_json_file(tmp_path / "missing.json")
