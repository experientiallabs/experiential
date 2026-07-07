"""Clustering of facet embeddings: numpy k-means (cosine) + LLM cluster naming.

k-means over L2-normalized facet embeddings (so squared-euclidean ranks like cosine), kmeans++
init, deterministic under a seed. Cluster naming is the Clio step: an LLM reads a sample of each
cluster's task summaries and writes a short name + description, which is what makes the resulting
scenario set auditable ("8 scenarios about baggage claims" instead of "cluster 3").
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ValidationError

from wmh.core.parsing import extract_json_object
from wmh.providers.base import Message, Provider
from wmh.scenarios.mining.facets import TraceFacet

_KMEANS_ITERS = 50
_NAME_SAMPLE = 10


class TraceCluster(BaseModel):
    """One discovered intent cluster over the facet corpus."""

    cluster_id: int
    name: str = ""
    description: str = ""
    member_trace_ids: list[str]


def default_k(n: int) -> int:
    """Heuristic base-layer cluster count: sqrt(n), clamped to [2, n]."""
    if n <= 2:
        return max(1, n)
    return min(n, max(2, round(float(np.sqrt(n)))))


def normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize rows (zero rows stay zero) so euclidean k-means ranks like cosine."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def kmeans_labels(embeddings: np.ndarray, k: int, *, seed: int = 0) -> np.ndarray:
    """Deterministic k-means (kmeans++ init, Lloyd iterations) over unit-normalized rows.

    Returns an int label per row. Empty clusters are re-seeded on the farthest point from its
    centroid so exactly `k` non-empty clusters come back whenever `k <= n_distinct_rows`.
    """
    matrix = normalize_rows(embeddings)
    n = matrix.shape[0]
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    centroids = _kmeans_pp_init(matrix, k, rng)
    labels = np.full(n, -1, dtype=np.int64)  # impossible sentinel: never false-converges on iter 1
    for _ in range(_KMEANS_ITERS):
        distances = _sq_distances(matrix, centroids)
        new_labels = distances.argmin(axis=1)
        for cluster in range(k):
            members = matrix[new_labels == cluster]
            if len(members) > 0:
                centroids[cluster] = members.mean(axis=0)
            else:
                # Re-seed an empty cluster on the point farthest from its current centroid.
                farthest = int(np.argmax(distances.min(axis=1)))
                centroids[cluster] = matrix[farthest]
                new_labels[farthest] = cluster
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return labels


def _kmeans_pp_init(matrix: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """kmeans++ seeding: spread initial centroids proportionally to squared distance."""
    n = matrix.shape[0]
    centroids = np.empty((k, matrix.shape[1]), dtype=np.float64)
    centroids[0] = matrix[rng.integers(n)]
    closest = _sq_distances(matrix, centroids[:1]).min(axis=1)
    for i in range(1, k):
        total = float(closest.sum())
        if total <= 0.0:  # all remaining points coincide with a centroid
            centroids[i:] = centroids[0]
            break
        probabilities = closest / total
        centroids[i] = matrix[rng.choice(n, p=probabilities)]
        closest = np.minimum(closest, _sq_distances(matrix, centroids[i : i + 1]).min(axis=1))
    return centroids


def _sq_distances(matrix: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Squared euclidean distance from every row to every centroid, shape (n, k)."""
    diff = matrix[:, None, :] - centroids[None, :, :]
    return np.einsum("nkd,nkd->nk", diff, diff)


def cluster_facets(
    facets: list[TraceFacet],
    embeddings: np.ndarray,
    *,
    k: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, list[TraceCluster]]:
    """Cluster the facet corpus; returns (labels per facet, clusters ordered by descending size)."""
    if len(facets) != len(embeddings):
        raise ValueError(f"{len(facets)} facets but {len(embeddings)} embeddings")
    if not facets:
        return np.empty(0, dtype=np.int64), []
    chosen_k = k if k is not None else default_k(len(facets))
    labels = kmeans_labels(embeddings, chosen_k, seed=seed)
    clusters: list[TraceCluster] = []
    for cluster_id in sorted(set(labels.tolist())):
        member_ids = [facets[i].trace_id for i in np.flatnonzero(labels == cluster_id)]
        clusters.append(TraceCluster(cluster_id=int(cluster_id), member_trace_ids=member_ids))
    clusters.sort(key=lambda c: len(c.member_trace_ids), reverse=True)
    return labels, clusters


NAMING_SYSTEM = """You name one cluster of related AI-agent tasks. You see a sample of short task
summaries that all landed in the same cluster.

Respond with ONLY a JSON object, no prose around it:
{"name": "<2-5 word noun phrase naming the shared task intent>",
 "description": "<one sentence describing what these tasks have in common>"}"""


class _RawName(BaseModel):
    name: str
    description: str = ""


def name_clusters(
    provider: Provider,
    clusters: list[TraceCluster],
    facets: list[TraceFacet],
    *,
    sample_size: int = _NAME_SAMPLE,
) -> None:
    """Fill in `name`/`description` on every cluster via one LLM call each (mutates in place)."""
    by_id = {facet.trace_id: facet for facet in facets}
    for cluster in clusters:
        summaries = [
            by_id[trace_id].task_summary
            for trace_id in cluster.member_trace_ids[:sample_size]
            if trace_id in by_id
        ]
        prompt = "TASK SUMMARIES:\n" + "\n".join(f"- {s}" for s in summaries)
        completion = provider.complete(
            NAMING_SYSTEM,
            [Message(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=256,
        )
        raw = extract_json_object(completion.text)
        parsed: _RawName | None = None
        if raw is not None:
            try:
                parsed = _RawName.model_validate_json(raw)
            except ValidationError:
                parsed = None
        if parsed is not None and parsed.name.strip():
            cluster.name = parsed.name.strip()
            cluster.description = parsed.description.strip()
        else:
            cluster.name = f"cluster {cluster.cluster_id}"
            cluster.description = summaries[0] if summaries else ""
