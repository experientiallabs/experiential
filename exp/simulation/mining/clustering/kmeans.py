"""k-means++ seeding and Lloyd iteration, in numpy.

Written out rather than taken from a library because this project ships numpy and deliberately no
sklearn or scipy, and because the parts that matter for reproducible mining are exactly the parts a
library hides: where the randomness enters, what happens to a cluster that loses every member, and
how a tie between restarts is broken. All three are decided here and stated.

Randomness enters in one place, ``default_rng(seed + restart)``, so a run is reproducible from the
seed recorded in the spec. An emptied cluster keeps its previous centroid rather than being reseeded
onto the farthest point: reseeding is what sklearn does and it finds slightly better optima, but it
makes the outcome depend on which point happens to be farthest at that moment, and a selection
pipeline that must reproduce its artifacts byte for byte trades that quality for determinism.
Restart ties keep the earlier restart, so the result does not depend on floating-point noise in the
inertia comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from exp.simulation.mining.clustering.centroids import squared_distances


@dataclass(frozen=True)
class KMeansFit:
    """One completed k-means fit.

    Attributes:
        labels: ``(N,)`` cluster index per point, always in ``[0, k)``.
        centroids: ``(k, D)`` final centroid coordinates.
        inertia: Summed squared distance from every point to its own centroid.
        iterations: Lloyd iterations the winning restart used.
    """

    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    iterations: int


def kmeans_fit(
    features: np.ndarray,
    k: int,
    *,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-4,
    seed: int = 42,
) -> KMeansFit:
    """Fit k-means with k-means++ seeding, keeping the lowest-inertia restart.

    Args:
        features: ``(N, D)`` point coordinates.
        k: Number of clusters, at least one and at most ``N``.
        n_init: Independent seedings to try, at least one.
        max_iter: Lloyd iterations per restart.
        tol: Stop a restart once the summed squared centroid shift falls to this or below.
        seed: Base seed. Restart ``r`` draws from ``seed + r``.

    Returns:
        The winning restart's labels, centroids, inertia and iteration count.

    Raises:
        ValueError: The point set is empty, or ``k`` is outside ``[1, N]``, or a bound is not
            positive.
    """
    points = np.asarray(features, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("k-means needs a non-empty two-dimensional feature matrix")
    if not 1 <= k <= points.shape[0]:
        raise ValueError(f"k must be between 1 and the {points.shape[0]} available points, got {k}")
    if n_init < 1 or max_iter < 1:
        raise ValueError("k-means needs at least one restart and at least one iteration")
    best: KMeansFit | None = None
    for restart in range(n_init):
        candidate = _one_fit(
            points, k, rng=np.random.default_rng(seed + restart), max_iter=max_iter, tol=tol
        )
        if best is None or candidate.inertia < best.inertia:
            best = candidate
    assert best is not None  # n_init >= 1, so the loop always produced a fit.
    return best


def _one_fit(
    points: np.ndarray, k: int, *, rng: np.random.Generator, max_iter: int, tol: float
) -> KMeansFit:
    """Seed once with k-means++ and run Lloyd to convergence or the iteration bound."""
    centroids = _plus_plus(points, k, rng)
    labels = np.zeros(len(points), dtype=np.int64)
    inertia = 0.0
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        squared = squared_distances(points, centroids)
        labels = squared.argmin(axis=1)
        inertia = float(squared[np.arange(len(points)), labels].sum())
        moved = _recentre(points, labels, centroids)
        shift = float(((moved - centroids) ** 2).sum())
        centroids = moved
        if shift <= tol:
            break
    return KMeansFit(labels=labels, centroids=centroids, inertia=inertia, iterations=iterations)


def _plus_plus(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Choose k seeds by D-squared sampling, which spreads them without placing them by hand.

    A corpus of identical rows leaves nothing to sample proportionally to, so the draw falls back
    to uniform instead of dividing by a zero total.
    """
    chosen = [int(rng.integers(len(points)))]
    closest = squared_distances(points, points[chosen]).reshape(-1)
    while len(chosen) < k:
        total = float(closest.sum())
        weights = None if total <= 0.0 else closest / total
        chosen.append(int(rng.choice(len(points), p=weights)))
        closest = np.minimum(closest, squared_distances(points, points[chosen[-1:]]).reshape(-1))
    return points[chosen].copy()


def _recentre(points: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Move every centroid to the mean of its members, leaving an emptied cluster where it is."""
    sums = np.zeros_like(centroids)
    np.add.at(sums, labels, points)
    counts = np.bincount(labels, minlength=len(centroids))
    moved = centroids.copy()
    occupied = counts > 0
    moved[occupied] = sums[occupied] / counts[occupied, None]
    return moved
