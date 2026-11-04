"""Session wire formats read into format-neutral events without executing content."""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.traces.ingest.agent_session_formats import (
    FREE_FORM_ARGUMENT,
    content_text,
    detect_session_format,
    export_session,
    is_injected_text,
    object_arguments,
    read_sessions,
)
from exp.common.traces.ingest.vendor_records import VendorTraceFormatError


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        ([{"type": "session", "id": "s", "cwd": "/w"}], "pi"),
        ([{"type": "mode", "sessionId": "s"}], "claude-code"),
        ([{"id": "x", "projectPath": "/p", "messages": []}], "session-export"),
        ([{"trace_id": "x", "label": "good"}], None),
        ([{"messages": [], "type": "chat"}], None),
    ],
)
def test_detect_session_format(records: list[JsonObject], expected: str | None) -> None:
    """Leading records identify the wire format; unrelated JSONL is not a session."""
    assert detect_session_format(records) == expected


def test_claude_code_merges_split_turns_and_drops_duplicates() -> None:
    """Records repeating one ``message.id`` form one turn; repeated records are counted."""
    turn: JsonObject = {
        "type": "assistant",
        "sessionId": "s",
        "uuid": "a1",
        "message": {"id": "m1", "role": "assistant", "content": [{"type": "text", "text": "a"}]},
    }
    second: JsonObject = {
        "type": "assistant",
        "sessionId": "s",
        "uuid": "a2",
        "message": {
            "id": "m1",
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
        },
    }
    prompt: JsonObject = {
        "type": "user",
        "sessionId": "s",
        "uuid": "u1",
        "message": {"role": "user", "content": "go"},
    }

    (session,) = read_sessions("claude-code", [prompt, turn, second, dict(second)])

    kinds = [event.kind for event in session.events]
    assert kinds == ["user", "assistant"]
    assert session.events[1].text == "a"
    assert [call.name for call in session.events[1].tool_calls] == ["Bash"]
    skipped = session.metadata["skipped_records"]
    assert isinstance(skipped, dict) and skipped["duplicate-record"] == 1


def test_pi_without_complete_links_keeps_file_order() -> None:
    """Entries without parent links are read in file order instead of a guessed branch."""
    records: list[JsonObject] = [
        {"type": "session", "id": "s", "timestamp": "2026-01-01T00:00:00Z"},
        {"type": "message", "id": "m1", "message": {"role": "system", "content": "be brief"}},
        {"type": "message", "id": "m2", "message": {"role": "user", "content": "hi"}},
    ]

    (session,) = read_sessions("pi", records)

    assert [event.kind for event in session.events] == ["system", "user"]
    assert session.events[1].prompt is True


def test_pi_requires_a_session_header() -> None:
    """A pi file without its header is rejected with a format error."""
    with pytest.raises(VendorTraceFormatError, match="session header"):
        read_sessions("pi", [{"type": "message", "id": "m"}])


def test_export_tool_call_arguments_fall_back_to_text() -> None:
    """Non-JSON tool-call content is kept exactly, under the free-form key."""
    session = export_session(
        {
            "id": "e",
            "messages": [
                {"role": "assistant", "content": "[Tool call: run] not json", "toolName": "run"}
            ],
        },
        position=0,
    )

    (event,) = session.events
    assert event.tool_calls[0].arguments == {FREE_FORM_ARGUMENT: "not json"}
    assert session.synthetic_time is True


def test_injected_text_and_content_rendering() -> None:
    """Harness wrappers are not prompts, and image data is never copied into text."""
    assert is_injected_text("  <environment_context>cwd</environment_context>")
    assert not is_injected_text("Fix the bug")
    assert (
        content_text(
            [{"type": "text", "text": "see"}, {"type": "image", "source": {"data": "AAAA"}}]
        )
        == "see\n[image]"
    )


def test_export_drops_repeated_calls_and_results() -> None:
    """A repeated tool call or result for one call identity is dropped and counted."""
    call = {
        "role": "assistant",
        "content": "[Tool call: run] {}",
        "toolName": "run",
        "toolCallId": "c",
    }
    result = {"role": "tool-result", "content": "ok", "toolCallId": "c"}

    session = export_session({"id": "e", "messages": [call, call, result, result]}, position=0)

    assert [event.kind for event in session.events] == ["assistant", "tool_result"]
    assert session.events[1].tool_name == "run"
    assert session.metadata["skipped_records"] == {"duplicate-record": 2}


@pytest.mark.parametrize("recorded", ["ls -la", ["ls", "-la"], 3, None])
def test_non_object_arguments_are_held_under_the_free_form_key(recorded: JsonValue) -> None:
    """Replay needs object arguments, so any other recorded value is wrapped, never dropped."""
    assert object_arguments(recorded) == {FREE_FORM_ARGUMENT: recorded}
    assert object_arguments({"path": "a"}) == {"path": "a"}
