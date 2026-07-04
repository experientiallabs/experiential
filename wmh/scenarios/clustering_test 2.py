"""Tests for facet clustering (k-means determinism/separation) and LLM cluster naming."""

from __future__ import annotations

import numpy as np

from wmh.scenarios.clustering import (
    TraceCluster,
    cluster_facets,
    default_k,
    kmeans_labels,
    name_clusters,
)
from wmh.scenarios.facets import TraceFacet
from wmh.scenarios.facets_test import FakeProvider


def _facet(trace_id: str, summary: str) -> TraceFacet:
    return TraceFacet(trace_id=trace_id, task_summary=summary, tool_signature="t")


def _two_blobs(per_blob: int = 5) -> np.ndarray:
    rng = np.random.default_rng(7)
    blob_a = rng.normal(loc=(1.0, 0.0), scale=0.01, size=(per_blob, 2))
    blob_b = rng.normal(loc=(0.0, 1.0), scale=0.01, size=(per_blob, 2))
    return np.vstack([blob_a, blob_b])


def test_kmeans_separates_two_blobs() -> None:
    labels = kmeans_labels(_two_blobs(), 2, seed=0)
    assert len(set(labels[:5].tolist())) == 1
    assert len(set(labels[5:].tolist())) == 1
    assert labels[0] != labels[5]


def test_kmeans_is_deterministic_under_seed() -> None:
    embeddings = _two_blobs()
    first = kmeans_labels(embeddings, 2, seed=3)
    second = kmeans_labels(embeddings, 2, seed=3)
    assert np.array_equal(first, second)


def test_kmeans_k_at_least_n_returns_identity() -> None:
    labels = kmeans_labels(np.eye(3), 5, seed=0)
    assert labels.tolist() == [0, 1, 2]


def test_default_k_is_sqrt_n_clamped() -> None:
    assert default_k(1) == 1
    assert default_k(4) == 2
    assert default_k(100) == 10


def test_cluster_facets_orders_clusters_by_size() -> None:
    facets = [_facet(f"t{i}", f"task {i}") for i in range(10)]
    embeddings = np.vstack([_two_blobs(3), _two_blobs(3)[:4]])  # 6 + 4 rows
    labels, clusters = cluster_facets(facets, embeddings, k=2, seed=0)
    assert len(labels) == 10
    assert len(clusters) == 2
    assert len(clusters[0].member_trace_ids) >= len(clusters[1].member_trace_ids)


def test_name_clusters_parses_and_falls_back() -> None:
    facets = [_facet("t1", "cancel a flight"), _facet("t2", "cancel a hotel")]
    good = TraceCluster(cluster_id=0, member_trace_ids=["t1", "t2"])
    name_clusters(
        FakeProvider('{"name": "Cancellations", "description": "Cancel bookings."}'),
        [good],
        facets,
    )
    assert good.name == "Cancellations"
    assert good.description == "Cancel bookings."

    bad = TraceCluster(cluster_id=3, member_trace_ids=["t1"])
    name_clusters(FakeProvider("garbage"), [bad], facets)
    assert bad.name == "cluster 3"
    assert bad.description == "cancel a flight"
