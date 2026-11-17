"""Tests for projecting canonical traces into reconstruction inputs."""

from __future__ import annotations

from exp.common.traces.ingest.chat_json_test import _two_request_trace
from exp.simulation.reconstruction.sources import from_canonical_traces


def test_shared_message_history_is_retained_once_per_episode() -> None:
    """Spans keep their prefix references and the episode header carries the history."""
    trace = _two_request_trace()

    (episode,) = from_canonical_traces((trace,)).episodes

    header, *spans = episode.records
    assert header["message_history"] == list(trace.message_history)
    assert len(spans) == len(trace.spans)
