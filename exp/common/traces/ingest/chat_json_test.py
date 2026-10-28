"""Tests for chat JSON conversation normalization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, SourceIdentity
from exp.common.traces.ingest.chat_json import CHAT_JSON_SOURCE, attach_request_context
from exp.common.traces.ingest.vendor_records import VendorTraceFormatError
from exp.common.traces.ingest.vendor_trace import SYNTHETIC_TIME_ATTRIBUTE


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


def test_load_chat_json_file_keeps_synthetic_timing_and_source_metadata(tmp_path: Path) -> None:
    """One conversation keeps synthetic timing, source metadata, and its completion."""
    path = tmp_path / "chat.json"
    path.write_text(json.dumps(_conversation()), encoding="utf-8")

    result = CHAT_JSON_SOURCE.load(path)

    trace = result.traces[0]
    assert trace.initial_context == {
        "instruction_messages": [{"role": "system", "content": "You are helpful."}]
    }
    call, completion = trace.spans[0], trace.spans[2]
    assert completion.attributes["gen_ai.completion"] == "It is 18C in Paris."
    assert call.attributes[SYNTHETIC_TIME_ATTRIBUTE] is True
    assert call.attributes["exp.trace.metadata"] == {"tenant": "acme"}
    assert call.attributes["exp.source.vendor"] == "chat-json"
    assert call.attributes["exp.source.trace.id"] == "conversation-1"


def test_load_chat_json_file_accepts_bare_message_array(tmp_path: Path) -> None:
    """A bare message array is read as one conversation with a derived trace key."""
    path = tmp_path / "chat.json"
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    path.write_text(json.dumps(messages), encoding="utf-8")

    result = CHAT_JSON_SOURCE.load(path)

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

    result = CHAT_JSON_SOURCE.load(path)

    span = result.traces[0].spans[0]
    assert SYNTHETIC_TIME_ATTRIBUTE not in span.attributes
    assert span.attributes["exp.source.span.started_at"] == "2026-01-01T00:00:05+00:00"


def test_load_chat_json_file_excludes_conversation_without_user_message(tmp_path: Path) -> None:
    """A conversation with no user message is excluded rather than given invented intent."""
    path = tmp_path / "chat.json"
    path.write_text(
        json.dumps({"trace_id": "conversation-4", "messages": [{"role": "assistant", "c": 1}]}),
        encoding="utf-8",
    )

    result = CHAT_JSON_SOURCE.load(path)

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

    result = CHAT_JSON_SOURCE.load(path)

    assert result.traces == ()
    assert "provider" in result.issues[0].message


def test_reused_call_id_keeps_each_unnamed_result_with_its_preceding_tool() -> None:
    """Later call-ID reuse cannot rewrite an earlier tool result or mutate the source export."""
    conversation: JsonObject = {
        "trace_id": "reused-call-id",
        "messages": [
            {"role": "user", "content": "Search and extract the company website."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "parallel_search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "Search result"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "parallel_extract", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "Extract result"},
            {"role": "assistant", "content": "Done."},
        ],
    }
    original = json.dumps(conversation, sort_keys=True)

    result = CHAT_JSON_SOURCE.normalize(
        (conversation,), source=SourceIdentity(kind="manual", source_id="reused-call-id")
    )

    assert not result.issues
    assert len(result.traces) == 1
    search_call, search_result, extract_call, extract_result, _ = result.traces[0].spans
    tool_spans = (search_call, search_result, extract_call, extract_result)
    assert [span.attributes["gen_ai.tool.name"] for span in tool_spans] == [
        "parallel_search",
        "parallel_search",
        "parallel_extract",
        "parallel_extract",
    ]
    assert search_result.parent_span_id == search_call.span_id
    assert extract_result.parent_span_id == extract_call.span_id
    assert all(span.attributes["gen_ai.tool.call.id"] == "call-1" for span in tool_spans)
    assert json.dumps(conversation, sort_keys=True) == original


def test_attach_request_context_reaches_trace_instance_facts_in_json_and_jsonl(
    tmp_path: Path,
) -> None:
    """Matched conversations gain the entry; import moves it out of the request context."""
    first = {**_conversation(), "exp.request.context": {"task_lineage": "a"}}
    second = {**_conversation(), "trace_id": "conversation-2"}
    facts = {"facts": [{"key": "input.xlsx/workbook", "value": 1}]}
    for name, payload in (
        ("chat.json", json.dumps({"conversations": [first, second]}).encode()),
        ("chat.jsonl", (json.dumps(first) + "\nnot json\n" + json.dumps(second)).encode()),
    ):
        annotated, matched = attach_request_context(
            payload, "instance_facts", {"conversation-1": facts}
        )
        assert matched == frozenset({"conversation-1"})
        path = tmp_path / name
        path.write_bytes(annotated)
        traces = {
            trace.spans[0].attributes["exp.source.trace.id"]: trace
            for trace in CHAT_JSON_SOURCE.load(path).traces
        }
        assert traces["conversation-1"].instance_facts == facts
        assert traces["conversation-1"].initial_context["task_lineage"] == "a"
        assert "instance_facts" not in traces["conversation-1"].initial_context
        assert traces["conversation-2"].instance_facts is None
    assert b"not json" in annotated


def test_attach_request_context_refuses_ambiguous_or_existing_entries() -> None:
    """A repeated conversation key or an existing entry stops instead of guessing."""
    conversation = _conversation()
    twice = json.dumps([conversation, conversation]).encode()
    with pytest.raises(VendorTraceFormatError, match="declared twice"):
        attach_request_context(twice, "instance_facts", {"conversation-1": {}})
    existing = json.dumps({**conversation, "exp.request.context": {"instance_facts": {}}}).encode()
    with pytest.raises(VendorTraceFormatError, match="already declares 'instance_facts'"):
        attach_request_context(existing, "instance_facts", {"conversation-1": {}})
    with pytest.raises(VendorTraceFormatError, match="not UTF-8"):
        attach_request_context(b"\xff", "instance_facts", {})
