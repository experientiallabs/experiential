"""Tests for readable calibration transcript rendering."""

from __future__ import annotations

from exp.cli.judge.transcript import span_evidence_text, user_input
from exp.common.traces.ingest.chat_json_test import _inline_history, _two_request_trace


def test_user_input_reads_the_latest_request_from_shared_history() -> None:
    """Shared and inline history render the same user message for every span."""
    trace = _two_request_trace()
    inline = _inline_history(trace)

    assert [user_input(trace, span) for span in trace.spans] == [
        "Book a flight.",
        "Paris, please.",
        None,
        "Paris, please.",
    ]
    assert [user_input(trace, span) for span in trace.spans] == [
        user_input(inline, span) for span in inline.spans
    ]
    assert [span_evidence_text(trace, span) for span in trace.spans] == [
        span_evidence_text(inline, span) for span in inline.spans
    ]
