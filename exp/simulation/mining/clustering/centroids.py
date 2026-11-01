"""Distances from points to centroids, computed the way that stays numerically honest.

The expansion ``||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2`` turns an N-by-K distance table into one
matrix product, which is the difference between a clustering pass that finishes and one that does
not. It also subtracts nearly equal numbers, so a point sitting on its centroid can come out at a
small negative value; clamping at zero before the square root is what keeps that from becoming a
NaN that then poisons every downstream comparison.

Two obvious companions are deliberately absent: a nearest-versus-second-nearest confidence margin
and an incremental running mean for updating centroids in place. Mining fits a clusterer once over
one frozen partition, so there is no online assignment that would report a margin and no
incremental pass that would move a centroid, and the persisted coverage contract has nowhere to put
a margin anyway. Either can be added when something actually reads it.
"""

from __future__ import annotations

import numpy as np


def squared_distances(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return the squared Euclidean distance from every point to every centroid.

    Args:
        features: ``(N, D)`` point coordinates.
        centroids: ``(K, D)`` centroid coordinates.

    Returns:
        A ``(N, K)`` table, clamped at zero so cancellation never produces a negative square.
    """
    squared = (
        (features**2).sum(axis=1, keepdims=True)
        - 2 * features @ centroids.T
        + (centroids**2).sum(axis=1)
    )
    return np.maximum(squared, 0.0)


def distances(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return the Euclidean distance from every point to every centroid.

    Args:
        features: ``(N, D)`` point coordinates.
        centroids: ``(K, D)`` centroid coordinates.

    Returns:
        A ``(N, K)`` table of nonnegative distances.
    """
    return np.sqrt(squared_distances(features, centroids))
