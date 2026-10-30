"""Tests for runtime RAG dataset refresh verification."""

from __future__ import annotations

import pytest

from exp.common.traces.ingest.chat_json_test import _two_request_trace
from exp.simulation.retrieval.refresh_dataset import (
    RuntimeRAGDatasetError,
    _verify_stitched_runtime_traces,
)


def test_stitched_trace_cannot_alter_the_shared_message_history() -> None:
    """Spans point into the message history, so a changed history is changed source evidence."""
    trace = _two_request_trace()
    altered = trace.model_copy(
        update={"message_history": ({"role": "user", "content": "Rewritten."},)}
    )

    _verify_stitched_runtime_traces((trace,), {trace.trace_id: trace})
    with pytest.raises(RuntimeRAGDatasetError, match="changed canonical source evidence"):
        _verify_stitched_runtime_traces((trace,), {trace.trace_id: altered})
