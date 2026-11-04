"""Equivalence of matrix-shortlisted ranking with the full pure-Python cosine scan."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from exp.common.models import Embedding, EmbeddingClient
from exp.simulation.retrieval import RAGAction, RAGQuery, TraceRAGRetriever, load_rag_index
from exp.simulation.retrieval.build import persist_trace_rag
from exp.simulation.retrieval.embedding import RAGEmbedderBinding
from exp.simulation.retrieval.embedding_checkpoint import RAGEmbeddingCheckpoint
from exp.simulation.retrieval.retriever import TagTerm, rank_rows
from exp.simulation.retrieval.tests.retrieval_test import (
    _CREATED_AT,
    _bindings,
    _constant_binding,
    _persist_traces,
    _store,
)

Ranked = tuple[tuple[int, float, float | None], ...]


def _reference(
    rows: tuple[tuple[float, ...], ...],
    query: tuple[float, ...],
    eligible: tuple[int, ...],
    transition_ids: tuple[str, ...],
    limit: int,
    tag: tuple[float, tuple[float, ...]] | None,
) -> Ranked:
    """Rank with the previous full pure-Python scan over every eligible row.

    Args:
        rows: Row vectors as Python floats.
        query: Query vector.
        eligible: Eligible row indexes.
        transition_ids: Tie-break key per row.
        limit: Number of rows to return.
        tag: Optional weight and per-row overlap.

    Returns:
        The previous implementation's ``(row, score, overlap)`` ranking.
    """
    candidates: list[tuple[int, float, float | None]] = []
    for row in eligible:
        score = sum(left * right for left, right in zip(query, rows[row], strict=True))
        if tag is None:
            candidates.append((row, score, None))
            continue
        weight, overlaps = tag
        candidates.append((row, score + weight * overlaps[row], overlaps[row]))
    candidates.sort(key=lambda item: (-item[1], transition_ids[item[0]]))
    return tuple(candidates[:limit])


def _unit_rows(
    generator: np.random.Generator, count: int, dimension: int
) -> npt.NDArray[np.float64]:
    """Return random float64 unit rows.

    Args:
        generator: Seeded random source.
        count: Number of rows.
        dimension: Row width.

    Returns:
        A ``(count, dimension)`` matrix of unit rows.
    """
    matrix = generator.standard_normal((count, dimension))
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


@pytest.mark.parametrize("seed", range(12))
def test_rank_rows_matches_pure_python_scan_with_ties(seed: int) -> None:
    """Shortlisted ranking equals the full scan bit for bit, ties and tag terms included.

    Args:
        seed: Random case selector.
    """
    generator = np.random.default_rng(seed)
    dimension = int(generator.choice((3, 16, 64)))
    base = _unit_rows(generator, 40, dimension)
    if seed % 3 == 0:
        # Few distinct coarse values produce many exact score ties.
        base = np.round(base * 2) / 2
        base[np.linalg.norm(base, axis=1) == 0, 0] = 1.0
        base = base / np.linalg.norm(base, axis=1, keepdims=True)
    duplicates = base[generator.integers(0, base.shape[0], size=60)]
    matrix = np.concatenate((base, duplicates))
    order = generator.permutation(matrix.shape[0])
    matrix = np.ascontiguousarray(matrix[order])
    matrix.flags.writeable = False
    count = matrix.shape[0]
    transition_ids = tuple(f"t-{value:04d}" for value in generator.permutation(count))
    query = tuple(float(value) for value in matrix[int(generator.integers(0, count))])
    rows = tuple(tuple(row.tolist()) for row in matrix)
    trace_codes = generator.integers(0, 7, size=count).astype(np.intp)
    overlaps = tuple(float(value) for value in generator.choice((0.0, 0.25, 0.5, 1.0), size=7))
    for eligible_fraction in (1.0, 0.6):
        mask = generator.random(count) < eligible_fraction
        eligible = np.flatnonzero(mask)
        for limit in (1, 3, 5, 17, count + 1):
            for weight in (None, 0.0, 0.3, 1.0):
                term = (
                    None
                    if weight is None
                    else TagTerm(weight=weight, overlaps=overlaps, trace_codes=trace_codes)
                )
                expected = _reference(
                    rows,
                    query,
                    tuple(eligible.tolist()),
                    transition_ids,
                    limit,
                    None
                    if weight is None
                    else (weight, tuple(overlaps[code] for code in trace_codes.tolist())),
                )
                actual = rank_rows(
                    matrix,
                    query,
                    eligible,
                    transition_ids=transition_ids,
                    limit=limit,
                    tag_term=term,
                )
                assert actual == expected


def test_rank_rows_keeps_every_row_tied_at_the_boundary() -> None:
    """Identical rows straddling the cutoff resolve by transition ID, as in the full scan."""
    matrix = np.tile(np.array([[1.0, 0.0]]), (10, 1))
    matrix.flags.writeable = False
    transition_ids = tuple(f"t-{9 - index}" for index in range(10))
    ranked = rank_rows(
        matrix,
        (1.0, 0.0),
        np.arange(10, dtype=np.intp),
        transition_ids=transition_ids,
        limit=3,
    )
    assert [transition_ids[row] for row, _, _ in ranked] == ["t-0", "t-1", "t-2"]


def test_rank_rows_refuses_non_finite_scores() -> None:
    """A non-finite query still fails closed instead of ranking silently."""
    matrix = np.eye(4)
    with pytest.raises(ValueError, match="finite"):
        rank_rows(
            matrix,
            (math.nan, 0.0, 0.0, 0.0),
            np.arange(4, dtype=np.intp),
            transition_ids=("a", "b", "c", "d"),
            limit=1,
        )


class _CountingClient:
    """Constant embedding client that counts provider calls."""

    def __init__(self, inner: EmbeddingClient) -> None:
        """Wrap one client.

        Args:
            inner: Client producing the vectors.
        """
        self.inner = inner
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Count and delegate one batch.

        Args:
            texts: Provider inputs.

        Returns:
            The wrapped client's embeddings.
        """
        self.calls += 1
        return self.inner.embed(texts)


def test_query_cache_reuses_durable_query_embeddings(tmp_path: Path) -> None:
    """A repeated identical query embeds once with a query cache and every time without one."""
    store = _store(tmp_path, "query-cache")
    source, traces = _persist_traces(store, count=3)
    binding = _constant_binding()
    built = persist_trace_rag(
        store,
        (source,),
        _bindings(traces),
        created_at=_CREATED_AT,
        code_revision="revision-a",
        embedder=binding,
    )
    loaded = load_rag_index(store, built.index.rag_id)
    query = RAGQuery(task="reset password", action=RAGAction(kind="message", content="email?"))
    counting = _CountingClient(binding.client)
    counted = RAGEmbedderBinding(client=counting, snapshot=binding.snapshot)
    cache = RAGEmbeddingCheckpoint(
        store.paths, binding.snapshot, maximum_chunk_bytes=loaded.index.embedding_chunk_bytes
    )
    cached = TraceRAGRetriever(loaded, embedder=counted, query_cache=cache)
    first = cached.retrieve(query)
    second = cached.retrieve(query)
    assert first == second
    assert counting.calls == 1
    uncached = TraceRAGRetriever(loaded, embedder=counted)
    assert uncached.retrieve(query) == first
    assert uncached.retrieve(query) == first
    assert counting.calls == 3


def test_query_cache_must_share_the_index_namespace(tmp_path: Path) -> None:
    """A query cache under another chunk size is refused before any query."""
    store = _store(tmp_path, "query-cache-namespace")
    source, traces = _persist_traces(store, count=1)
    binding = _constant_binding()
    built = persist_trace_rag(
        store,
        (source,),
        _bindings(traces),
        created_at=_CREATED_AT,
        code_revision="revision-a",
        embedder=binding,
    )
    loaded = load_rag_index(store, built.index.rag_id)
    cache = RAGEmbeddingCheckpoint(store.paths, binding.snapshot, maximum_chunk_bytes=64)
    with pytest.raises(ValueError, match="query cache"):
        TraceRAGRetriever(loaded, embedder=binding, query_cache=cache)
