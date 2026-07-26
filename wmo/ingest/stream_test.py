"""Tests for the streaming ingest generator and its D-INGEST event vocabulary."""

from __future__ import annotations

import json
from pathlib import Path

from wmo.ingest.adapter import SourceCredentialError
from wmo.ingest.stream import (
    DetectedEvent,
    DoneEvent,
    ErrorCode,
    ErrorEvent,
    ProgressEvent,
    _classify,
    event_json,
    ingest_events,
)

_CONVERSATIONS = [
    {
        "trace_id": f"{i:032x}",
        "messages": [
            {"role": "user", "content": f"task {i}"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "function": {"name": "get", "arguments": '{"k": 1}'}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ],
    }
    for i in range(3)
]


def _write_corpus(tmp_path: Path) -> Path:
    src = tmp_path / "conversations.jsonl"
    src.write_text("\n".join(json.dumps(c) for c in _CONVERSATIONS), encoding="utf-8")
    return src


def test_file_ingest_emits_detected_progress_done_in_order(tmp_path: Path) -> None:
    src = _write_corpus(tmp_path)
    out = tmp_path / "out.otel.jsonl"
    events = list(ingest_events(file=str(src), out=out))

    detected = events[0]
    assert isinstance(detected, DetectedEvent)
    assert detected.format == "chat-json"
    assert detected.traces == 3

    progress = [e for e in events[1:-1] if isinstance(e, ProgressEvent)]
    assert progress, "expected at least one progress event"
    assert all(e.total == 3 for e in progress)
    assert [e.normalized for e in progress] == sorted(e.normalized for e in progress)
    assert progress[-1].normalized == 3

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.traces == 3
    assert done.steps == 6  # 2 steps per conversation (tool call + final message)
    assert done.otel_object == str(out)
    # The written corpus is a real otel-genai file: re-ingesting it yields the same traces.
    reingested = list(ingest_events(file=str(out), out=tmp_path / "again.otel.jsonl"))
    assert isinstance(reingested[0], DetectedEvent)
    assert reingested[0].format == "otel-genai"
    assert isinstance(reingested[-1], DoneEvent)
    assert reingested[-1].traces == 3


def test_explicit_source_skips_detection(tmp_path: Path) -> None:
    src = _write_corpus(tmp_path)
    events = list(ingest_events(file=str(src), source="chat-json", out=tmp_path / "o.jsonl"))
    assert isinstance(events[0], DetectedEvent)
    assert events[0].format == "chat-json"


def test_unknown_format_yields_bad_format_error(tmp_path: Path) -> None:
    src = tmp_path / "junk.json"
    src.write_text('{"nothing": "recognizable"}', encoding="utf-8")
    (event,) = list(ingest_events(file=str(src), out=tmp_path / "o.jsonl"))
    assert isinstance(event, ErrorEvent)
    assert event.code == ErrorCode.BAD_FORMAT
    assert "--source" in event.message


def test_missing_file_yields_bad_format_error(tmp_path: Path) -> None:
    (event,) = list(ingest_events(file=str(tmp_path / "nope.json"), out=tmp_path / "o.jsonl"))
    assert isinstance(event, ErrorEvent)
    assert event.code == ErrorCode.BAD_FORMAT


def test_empty_corpus_yields_empty_error(tmp_path: Path) -> None:
    src = tmp_path / "empty.jsonl"
    # Valid chat-json shape but zero usable conversations after the first (detectable) one is
    # message-free: normalization produces no steps and no traces.
    src.write_text('{"messages": []}', encoding="utf-8")
    events = list(ingest_events(file=str(src), source="chat-json", out=tmp_path / "o.jsonl"))
    assert isinstance(events[-1], ErrorEvent)
    assert events[-1].code == ErrorCode.EMPTY


def test_event_json_matches_pinned_contract_shapes() -> None:
    """Serialized events carry exactly the D-INGEST keys (`total` present even when null)."""
    assert event_json(DetectedEvent(format="chat-json", traces=2)) == {
        "type": "detected",
        "format": "chat-json",
        "traces": 2,
    }
    assert event_json(ProgressEvent(normalized=1, total=None)) == {
        "type": "progress",
        "normalized": 1,
        "total": None,
    }
    assert event_json(ProgressEvent(normalized=1, total=9, note="n")) == {
        "type": "progress",
        "normalized": 1,
        "total": 9,
        "note": "n",
    }
    assert event_json(DoneEvent(traces=1, steps=2, otel_object="p.jsonl")) == {
        "type": "done",
        "traces": 1,
        "steps": 2,
        "otel_object": "p.jsonl",
    }
    assert event_json(ErrorEvent(message="boom")) == {"type": "error", "message": "boom"}
    assert event_json(ErrorEvent(message="no", code=ErrorCode.BAD_CREDENTIALS)) == {
        "type": "error",
        "message": "no",
        "code": "bad_credentials",
    }


def test_driver_failures_classify_to_credentials_and_unreachable() -> None:
    """Adapters raise SourceCredentialError/ConnectionError; codes must map."""
    assert _classify(SourceCredentialError("postgres authentication failed")).code is (
        ErrorCode.BAD_CREDENTIALS
    )
    assert _classify(ConnectionError("could not connect")).code is ErrorCode.UNREACHABLE


def test_unreadable_file_is_bad_format_not_bad_credentials(tmp_path: Path) -> None:
    """An OS PermissionError (unreadable local file) must not render "check your credentials".

    Regression (Greptile P1): bare PermissionError was mapped to bad_credentials, so a
    chmod-000 upload told the user their API key was wrong.
    """
    assert _classify(PermissionError("denied")).code is ErrorCode.BAD_FORMAT
    src = tmp_path / "locked.json"
    src.write_text('{"messages": []}', encoding="utf-8")
    src.chmod(0)
    try:
        (event,) = list(ingest_events(file=str(src), out=tmp_path / "o.jsonl"))
    finally:
        src.chmod(0o644)
    assert isinstance(event, ErrorEvent)
    assert event.code is ErrorCode.BAD_FORMAT


def test_file_ingest_honors_limit(tmp_path: Path) -> None:
    """`limit` caps traces for FILE ingests too, not only vendor pulls (cost control)."""
    src = _write_corpus(tmp_path)
    events = list(ingest_events(file=str(src), out=tmp_path / "o.jsonl", limit=2))
    assert isinstance(events[0], DetectedEvent)
    assert events[0].traces == 2
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.traces == 2
