"""Tests for centroid distance tables."""

from __future__ import annotations

import numpy as np

from exp.simulation.mining.clustering.centroids import distances, squared_distances


def test_the_gram_trick_matches_a_naive_loop() -> None:
    """The fast form must be the same arithmetic, not an approximation of it."""
    rng = np.random.default_rng(7)
    points, centroids = rng.normal(size=(9, 4)), rng.normal(size=(3, 4))

    naive = np.array([[((p - c) ** 2).sum() for c in centroids] for p in points])

    assert np.allclose(squared_distances(points, centroids), naive)
    assert np.allclose(distances(points, centroids), np.sqrt(naive))


def test_a_point_on_its_centroid_never_goes_negative() -> None:
    """Cancellation produces a tiny negative square whose root would be NaN."""
    points = np.array([[1e8, 1e8, 1e8]])

    assert squared_distances(points, points).min() >= 0.0
    assert np.isfinite(distances(points, points)).all()
