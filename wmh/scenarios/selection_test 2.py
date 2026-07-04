"""Tests for representative selection (SemDeDup, hybrid allocation, failure pinning)."""

from __future__ import annotations

import numpy as np
import pytest

from wmh.scenarios.facets import Outcome, TraceFacet
from wmh.scenarios.selection import hybrid_select, semdedup_keep


def _facet(
    trace_id: str,
    *,
    outcome: Outcome = Outcome.SUCCESS,
    category: str | None = None,
) -> TraceFacet:
    return TraceFacet(
        trace_id=trace_id,
        task_summary=f"task {trace_id}",
        tool_signature="t",
        outcome=outcome,
        failure_category=category,
    )


def test_semdedup_drops_near_duplicates_within_cluster() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]])
    labels = np.asarray([0, 0, 1])
    kept = semdedup_keep(embeddings, labels, threshold=0.95)
    assert kept == [0, 2]


def test_semdedup_keeps_duplicates_across_clusters() -> None:
    embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    labels = np.asarray([0, 1])
    assert semdedup_keep(embeddings, labels) == [0, 1]


def _corpus(n_big: int = 8, n_small: int = 2) -> tuple[list[TraceFacet], np.ndarray, np.ndarray]:
    """Two clusters whose members are similar (shared center) but never SemDeDup-duplicates.

    Member j of a cluster is 0.8 * center + 0.6 * (its own basis dimension), unit norm, so
    within-cluster cosine is 0.64 — clustered together by construction, far below the 0.95
    dedup threshold.
    """
    n = n_big + n_small
    embeddings = np.zeros((n, n + 2))
    for j in range(n_big):
        embeddings[j, 0] = 0.8
        embeddings[j, 2 + j] = 0.6
    for j in range(n_small):
        embeddings[n_big + j, 1] = 0.8
        embeddings[n_big + j, 2 + n_big + j] = 0.6
    labels = np.asarray([0] * n_big + [1] * n_small)
    facets = [_facet(f"t{i}") for i in range(n)]
    return facets, embeddings, labels


def test_hybrid_select_respects_budget_and_weights_sum_to_one() -> None:
    facets, embeddings, labels = _corpus()
    selections = hybrid_select(facets, embeddings, labels, budget=4)
    assert len(selections) == 4
    assert sum(s.weight for s in selections) == pytest.approx(1.0)
    assert len({s.trace_id for s in selections}) == 4


def test_hybrid_select_covers_the_small_cluster() -> None:
    facets, embeddings, labels = _corpus(n_big=20, n_small=2)
    selections = hybrid_select(facets, embeddings, labels, budget=5)
    assert any(s.cluster_id == 1 for s in selections)


def test_hybrid_select_budget_above_corpus_returns_everything() -> None:
    facets, embeddings, labels = _corpus(n_big=3, n_small=2)
    selections = hybrid_select(facets, embeddings, labels, budget=50)
    assert len(selections) == 5
    assert sum(s.weight for s in selections) == pytest.approx(1.0)


def test_hybrid_select_pins_missing_failure_category() -> None:
    facets, embeddings, labels = _corpus(n_big=20, n_small=2)
    # One rare failure inside the big cluster; without pinning it would never win a slot.
    facets[7] = _facet("t7", outcome=Outcome.FAILURE, category="data_loss")
    selections = hybrid_select(facets, embeddings, labels, budget=3)
    pinned = [s for s in selections if s.pinned_failure == "data_loss"]
    assert len(pinned) == 1
    assert pinned[0].trace_id == "t7"
    assert sum(s.weight for s in selections) == pytest.approx(1.0)


def test_hybrid_select_rejects_bad_budget() -> None:
    facets, embeddings, labels = _corpus()
    with pytest.raises(ValueError, match="budget"):
        hybrid_select(facets, embeddings, labels, budget=0)
