"""Tests for the Postgres trace source (fake row source; never a live database)."""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from wmh.ingest.adapter import VendorPull, get_adapter
from wmh.ingest.postgres import PostgresAdapter, _validate_table

_OTLP_SPAN_ROWS: list[tuple[str | None, JsonValue]] = [
    (
        "sess-1",
        {
            "traceId": "ignored",
            "spanId": "a1",
            "startTimeUnixNano": 1,
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                {"key": "gen_ai.tool.name", "value": {"stringValue": "get_user"}},
                {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": '{"id": "u1"}'}},
            ],
        },
    ),
    (
        "sess-1",
        {
            "traceId": "ignored",
            "spanId": "a2",
            "startTimeUnixNano": 2,
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
                {"key": "gen_ai.tool.message", "value": {"stringValue": "found u1"}},
            ],
        },
    ),
    (
        "sess-2",
        {
            "traceId": "ignored",
            "spanId": "b1",
            "startTimeUnixNano": 1,
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                {"key": "gen_ai.completion", "value": {"stringValue": "all done"}},
            ],
        },
    ),
]

_MESSAGE_ROWS: list[tuple[str | None, JsonValue]] = [
    ("chat-9", {"role": "user", "content": "what's the weather?"}),
    (
        "chat-9",
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "wx", "arguments": "{}"}}],
        },
    ),
    ("chat-9", {"role": "tool", "tool_call_id": "c1", "content": "sunny"}),
    ("chat-9", {"role": "assistant", "content": "It's sunny."}),
]

_CONVERSATION_BLOB_ROWS: list[tuple[str | None, JsonValue]] = [
    (
        None,
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        },
    ),
    (
        None,
        {
            "messages": [
                {"role": "user", "content": "bye"},
                {"role": "assistant", "content": "later"},
            ]
        },
    ),
]


class _FakeRows(PostgresAdapter):
    """A PostgresAdapter with the database swapped for canned rows."""

    def __init__(self, rows: list[tuple[str | None, JsonValue]]) -> None:
        self._rows = rows

    def _fetch_rows(self, pull: VendorPull) -> list[tuple[str | None, JsonValue]]:
        return self._rows


def test_postgres_adapter_is_registered() -> None:
    assert isinstance(get_adapter("postgres"), PostgresAdapter)


def test_row_per_step_groups_by_trace_id_column() -> None:
    """Span-shaped payload rows group by the table's trace-id column, not the payload's own id."""
    traces = _FakeRows(_OTLP_SPAN_ROWS).from_vendor(VendorPull(table="t"))
    assert [t.trace_id for t in traces] == ["sess-1", "sess-2"]
    step = traces[0].steps[0]
    assert step.action.name == "get_user"
    assert step.observation.content == "found u1"
    assert traces[1].steps[0].action.content == "all done"


def test_message_per_row_assembles_conversations() -> None:
    """Rows that are individual chat messages (a session/thread table) become one episode."""
    (trace,) = _FakeRows(_MESSAGE_ROWS).from_vendor(VendorPull(table="t"))
    assert trace.trace_id == "chat-9"
    assert len(trace.steps) == 2  # the tool-call step + the final assistant message
    assert trace.steps[0].action.name == "wx"
    assert trace.steps[0].observation.content == "sunny"
    assert trace.steps[0].task == "what's the weather?"


def test_row_per_trace_blobs_need_no_trace_id_column() -> None:
    """Self-contained conversation blobs (one session per row) each become their own trace."""
    traces = _FakeRows(_CONVERSATION_BLOB_ROWS).from_vendor(VendorPull(table="t"))
    assert len(traces) == 2
    assert all(len(t.steps) == 1 for t in traces)


def test_message_rows_without_trace_id_column_error() -> None:
    rows = [(None, payload) for _, payload in _MESSAGE_ROWS]
    with pytest.raises(ValueError, match="trace"):
        _FakeRows(rows).from_vendor(VendorPull(table="t"))


def test_unrecognizable_payloads_error_with_source_guidance() -> None:
    with pytest.raises(ValueError, match="auto-detect"):
        _FakeRows([("t1", {"nothing": "known"})]).from_vendor(VendorPull(table="t"))


def test_from_file_is_a_friendly_error() -> None:
    with pytest.raises(ValueError, match="database"):
        PostgresAdapter().from_file("whatever.json")


def test_pull_requires_table() -> None:
    with pytest.raises(ValueError, match="table"):
        PostgresAdapter().from_vendor(VendorPull())


def test_table_identifiers_are_validated() -> None:
    assert _validate_table("public.agent_traces") == "public.agent_traces"
    with pytest.raises(ValueError, match="table"):
        _validate_table("traces; drop table users")
