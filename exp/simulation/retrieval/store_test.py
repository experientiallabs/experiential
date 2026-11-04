"""Loaded RAG indexes hold vectors as one exact read-only float64 matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from exp.simulation.retrieval import load_rag_index, persist_trace_rag
from exp.simulation.retrieval.tests.retrieval_test import (
    _CREATED_AT,
    _bindings,
    _persist_traces,
    _store,
)


def test_loaded_matrix_is_read_only_and_rebuilds_the_persisted_vectors(tmp_path: Path) -> None:
    """The matrix matches the envelope shape, refuses writes, and round-trips every record.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    store = _store(tmp_path, "matrix")
    source, traces = _persist_traces(store, count=3)
    built = persist_trace_rag(
        store, (source,), _bindings(traces), created_at=_CREATED_AT, code_revision="revision-a"
    )
    loaded = load_rag_index(store, built.index.rag_id)

    assert loaded.matrix.dtype == np.float64
    assert loaded.matrix.shape == (built.index.transition_count, built.index.embedding_dimension)
    assert loaded.vectors == built.vectors
    with pytest.raises(ValueError, match="read-only"):
        loaded.matrix[0, 0] = 0.0
