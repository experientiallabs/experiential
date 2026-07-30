"""Recorded-vector embedder: serve a fit's original embeddings from a published cache.

The cache is a plain `.npy` array row-aligned to the matrix's scenarios in FIRST-APPEARANCE
order, which is the order every matrix producer writes and the cheapest alignment to verify
(row count must equal distinct-scenario count, checked at load). Serving recorded vectors is
what makes a `matrix`-kind reproduction offline, credential-free, and bit-exact: the
reproduction cannot drift from the published fit through embedding-provider nondeterminism,
because there is no embedding provider in the loop.

The honesty rule: the policy artifact still records the spec the vectors BELONG to (their
geometry), never "cache" - a cache cannot embed text it has not seen, so it is not an
embedding function an endpoint could serve with. `embed()` on unseen text raises for the
same reason, loudly, naming the count.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wmo.optimize.outcomes import OutcomeMatrix


class CachedTaskEmbedder:
    """Embedder over a fixed task set, backed by recorded vectors."""

    def __init__(self, matrix: OutcomeMatrix, cache_path: Path) -> None:
        order: list[str] = []
        tasks: dict[str, str] = {}
        for outcome in matrix.outcomes:
            if outcome.scenario_id not in tasks:
                tasks[outcome.scenario_id] = outcome.task
                order.append(outcome.scenario_id)
        vectors = np.load(cache_path)
        if vectors.ndim != 2 or vectors.shape[0] != len(order):
            raise ValueError(
                f"embedding cache {cache_path} has shape {vectors.shape} but the matrix has "
                f"{len(order)} distinct scenarios; the cache was built for a different matrix"
            )
        self.dim = int(vectors.shape[1])
        self._by_text: dict[str, np.ndarray] = {
            tasks[sid]: vectors[index] for index, sid in enumerate(order)
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = sum(1 for text in texts if text not in self._by_text)
        if missing:
            raise ValueError(
                f"{missing} of {len(texts)} texts are not in the embedding cache; a recorded-"
                "vector reproduction can only embed the matrix's own tasks (serve-time traffic "
                "needs the real embedder the policy records)"
            )
        return [self._by_text[text].tolist() for text in texts]
