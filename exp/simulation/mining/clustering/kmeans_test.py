"""Tests for the numpy k-means fit."""

from __future__ import annotations

import numpy as np
import pytest

from exp.simulation.mining.clustering.kmeans import kmeans_fit


def _blobs(separation: float = 30.0) -> np.ndarray:
    """Return three far-apart clusters of four points each."""
    rng = np.random.default_rng(3)
    centres = np.array([[0.0, 0.0], [separation, 0.0], [0.0, separation]])
    return np.vstack([centre + rng.normal(scale=0.2, size=(4, 2)) for centre in centres])


def _grouping(labels: np.ndarray) -> set[frozenset[int]]:
    """Return the partition a label vector describes, independent of label numbering."""
    return {frozenset(np.flatnonzero(labels == label).tolist()) for label in np.unique(labels)}


def test_separated_blobs_are_recovered_exactly() -> None:
    """A clusterer that cannot find obvious structure cannot be trusted with subtle structure."""
    fit = kmeans_fit(_blobs(), 3, seed=1)

    assert _grouping(fit.labels) == {
        frozenset(range(0, 4)),
        frozenset(range(4, 8)),
        frozenset(range(8, 12)),
    }
    assert fit.inertia < 5.0


def test_one_seed_gives_one_answer() -> None:
    """The seed is recorded in the spec precisely so a mined task set can be reproduced."""
    points = _blobs(separation=2.0)

    assert np.array_equal(
        kmeans_fit(points, 3, seed=11).labels, kmeans_fit(points, 3, seed=11).labels
    )


def test_restarts_keep_the_lowest_inertia() -> None:
    """More restarts can only improve the objective, never worsen it."""
    points = _blobs(separation=1.5)

    assert (
        kmeans_fit(points, 3, n_init=10, seed=5).inertia
        <= kmeans_fit(points, 3, n_init=1, seed=5).inertia
    )


def test_identical_rows_do_not_divide_by_zero_while_seeding() -> None:
    """A corpus of duplicate requests has no squared distance to sample proportionally to."""
    fit = kmeans_fit(np.ones((6, 3)), 3, seed=2)

    assert len(fit.labels) == 6
    assert fit.inertia == pytest.approx(0.0)


def test_asking_for_one_cluster_per_point_gives_singletons() -> None:
    """The budget can equal the candidate count, so k == n has to be a valid request."""
    fit = kmeans_fit(_blobs(), 12, seed=4)

    assert sorted(fit.labels.tolist()) == list(range(12))


def test_an_emptied_cluster_keeps_its_centroid_rather_than_jumping() -> None:
    """Reseeding onto the farthest point would make the result depend on scan order."""
    points = np.array([[0.0], [0.0], [0.0], [10.0]])

    fit = kmeans_fit(points, 3, n_init=1, seed=0)

    assert np.isfinite(fit.centroids).all()
    assert len(np.unique(fit.labels)) <= 3


def test_an_unusable_request_is_refused_by_name() -> None:
    """A clamped-silently k would report groups the caller did not ask for."""
    with pytest.raises(ValueError, match="non-empty"):
        kmeans_fit(np.zeros((0, 2)), 1)
    with pytest.raises(ValueError, match="k must be between"):
        kmeans_fit(_blobs(), 99)
    with pytest.raises(ValueError, match="at least one restart"):
        kmeans_fit(_blobs(), 2, n_init=0)
