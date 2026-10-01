"""Source lineage boundaries distinguish observed timestamps from message order."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from exp.common.core.artifacts import JsonObject, SourceIdentity
from exp.common.traces.ingest.chat_json import CHAT_JSON_SOURCE
from exp.common.traces.ingest.vendor_trace import SYNTHETIC_TIME_ATTRIBUTE
from exp.simulation.mining.lineage import assign_source_lineages


def _record(trace_id: str, *, timestamp: str | None = None) -> JsonObject:
    """Return one chat record with optional observed assistant timing."""
    assistant: JsonObject = {"role": "assistant", "content": "Recorded answer."}
    if timestamp is not None:
        assistant["timestamp"] = timestamp
    return {
        "trace_id": trace_id,
        "messages": [{"role": "user", "content": f"Handle {trace_id}"}, assistant],
    }


def test_synthetic_chat_times_keep_source_trace_boundaries_without_invented_metadata() -> None:
    """Unrelated records in one export cannot acquire a shared epoch-day lineage."""
    normalized = CHAT_JSON_SOURCE.normalize(
        (_record("first"), _record("second")),
        source=SourceIdentity(kind="file", source_id="chat-export"),
    )

    assignments = assign_source_lineages(normalized.traces)
    reversed_assignments = assign_source_lineages(tuple(reversed(normalized.traces)))

    assert len({item.lineage_group_id for item in assignments}) == 2
    assert all(item.time_bucket is None and item.conversation_id is None for item in assignments)
    assert {item.trace_id: item.lineage_group_id for item in assignments} == {
        item.trace_id: item.lineage_group_id for item in reversed_assignments
    }
    assert all(trace.conversation_id is None for trace in normalized.traces)
    assert all(
        span.attributes[SYNTHETIC_TIME_ATTRIBUTE] is True
        for trace in normalized.traces
        for span in trace.spans
    )


@pytest.mark.parametrize(
    "day", [datetime(1970, 1, 1, tzinfo=UTC), datetime(2026, 8, 11, tzinfo=UTC)]
)
def test_measured_time_bucket_ignores_synthetic_spans_and_still_groups_observed_activity(
    day: datetime,
) -> None:
    """A mixed trace uses its observed timestamp, never its synthetic epoch span."""
    mixed = _record("mixed")
    messages = mixed["messages"]
    assert isinstance(messages, list)
    messages.append(
        {
            "role": "assistant",
            "content": "Follow-up.",
            "timestamp": (day + timedelta(hours=12)).isoformat(),
        }
    )
    normalized = CHAT_JSON_SOURCE.normalize(
        (
            mixed,
            _record("same-day", timestamp=(day + timedelta(hours=13)).isoformat()),
            _record("next-day", timestamp=(day + timedelta(days=1, hours=13)).isoformat()),
        ),
        source=SourceIdentity(kind="file", source_id="chat-export"),
    )
    by_source = {
        trace.spans[0].attributes["exp.source.trace.id"]: assignment
        for trace, assignment in zip(
            normalized.traces, assign_source_lineages(normalized.traces), strict=True
        )
    }

    assert by_source["mixed"].lineage_group_id == by_source["same-day"].lineage_group_id
    assert by_source["mixed"].lineage_group_id != by_source["next-day"].lineage_group_id
    assert by_source["mixed"].time_bucket == int(day.timestamp()) // 86400


def test_declared_conversation_still_groups_synthetic_traces() -> None:
    """An actual conversation boundary outranks missing measurement timestamps."""
    normalized = CHAT_JSON_SOURCE.normalize(
        (
            {**_record("first"), "exp.conversation.id": "declared-thread"},
            {**_record("second"), "exp.conversation.id": "declared-thread"},
        ),
        source=SourceIdentity(kind="file", source_id="chat-export"),
    )

    assignments = assign_source_lineages(normalized.traces)

    assert len({item.lineage_group_id for item in assignments}) == 1
    assert all(item.conversation_id == "declared-thread" for item in assignments)
    assert all(item.time_bucket is None for item in assignments)
