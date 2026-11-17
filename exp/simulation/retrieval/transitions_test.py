"""Tests for real-trace RAG transition extraction."""

from __future__ import annotations

from exp.common.traces.ingest.chat_json_test import _inline_history, _two_request_trace
from exp.simulation.retrieval import RAGLineageBinding
from exp.simulation.retrieval.transitions import extract_fit_transitions


def test_shared_history_yields_the_same_transitions_as_inline_history() -> None:
    """A later user request read through the shared history becomes the reply evidence."""
    trace = _two_request_trace()
    bindings = (RAGLineageBinding(trace_id=trace.trace_id, lineage_id="travel", partition="fit"),)

    shared = extract_fit_transitions((trace,), bindings)

    assert shared == extract_fit_transitions((_inline_history(trace),), bindings)
    replies = [item for item in shared if item.observation.kind == "message"]
    assert [(item.action.content, item.observation.content) for item in replies] == [
        ("Where to?", "Paris, please.")
    ]
