"""Tests for the provider-free descriptor clustering."""

from __future__ import annotations

import math
from collections.abc import Sequence

from exp.simulation.mining.clustering.descriptor import (
    MAXIMUM_ITERATIONS,
    default_cluster_count,
    descriptor_clusters,
)
from exp.simulation.mining.geometry import CoreSetGeometry, cosine


def _ring(count: int) -> CoreSetGeometry:
    """Place identities evenly on a circle, which gives every pair a distinct distance."""
    vectors = {
        f"trace-{index:02d}": (
            math.cos(2 * math.pi * index / count),
            math.sin(2 * math.pi * index / count),
        )
        for index in range(count)
    }
    return CoreSetGeometry.of(vectors, dict.fromkeys(vectors, 1))


def _main_clusters(
    ids: Sequence[str],
    vectors: dict[str, tuple[float, ...]],
    masses: dict[str, int],
    target: int,
) -> dict[str, int]:
    """Replicate the pre-geometry clustering, transcribed from the implementation it replaced.

    An independent transcription is what makes this a check of equivalence rather than a check of
    the port against itself.
    """

    def medoid(member_ids: Sequence[str]) -> str:
        """Weighted central member, ties by identity."""
        return min(
            member_ids,
            key=lambda one: (
                -sum(cosine(vectors[one], vectors[other]) * masses[other] for other in member_ids),
                one,
            ),
        )

    def farthest(selected: Sequence[str]) -> str:
        """Farthest-first point, ties by identity."""
        remaining = [one for one in ids if one not in set(selected)]
        if not selected:
            return medoid(list(ids))
        return min(
            remaining,
            key=lambda one: (max(cosine(vectors[one], vectors[sel]) for sel in selected), one),
        )

    cluster_count = min(len(ids), max(1, round(math.sqrt(len(ids)))), target)
    if cluster_count == 1:
        return dict.fromkeys(ids, 0)
    centers = [min(ids)]
    while len(centers) < cluster_count:
        centers.append(farthest(centers))
    assignments: dict[str, int] = {}
    for _ in range(20):
        updated = {}
        for one in ids:
            ranked = sorted(
                (-cosine(vectors[one], vectors[center]), center, index)
                for index, center in enumerate(centers)
            )
            updated[one] = ranked[0][2]
        if updated == assignments:
            break
        assignments = updated
        rebuilt: list[str] = []
        for cluster_id in range(cluster_count):
            members = [one for one in ids if assignments[one] == cluster_id]
            if members:
                rebuilt.append(medoid(members))
        while len(rebuilt) < cluster_count:
            rebuilt.append(farthest(rebuilt))
        centers = rebuilt
    return assignments


def test_the_port_matches_the_implementation_it_replaced() -> None:
    """This is the default path, so a drift here changes every existing project's artifacts."""
    for count in (1, 2, 5, 9, 16, 25):
        geometry = _ring(count)
        for target in (1, 3, count):
            expected = _main_clusters(
                geometry.ids, dict(geometry.vectors), dict(geometry.masses), target
            )
            assert (
                descriptor_clusters(geometry, default_cluster_count(count, target)) == expected
            ), (count, target)


def test_the_group_count_is_bounded_by_the_budget() -> None:
    """Promising one representative per group needs a slot per group to deliver it."""
    assert default_cluster_count(100, 70) == 10
    assert default_cluster_count(100, 4) == 4
    assert default_cluster_count(3, 70) == 2
    assert default_cluster_count(1, 70) == 1
    # An empty partition would give zero groups, which is why both the selection entry point and
    # the backend dispatch return before reaching this at all.
    assert default_cluster_count(0, 70) == 0


def test_every_candidate_is_assigned_exactly_once() -> None:
    """Selection looks a cluster up per candidate, so a gap would be a KeyError mid-build."""
    geometry = _ring(17)

    clusters = descriptor_clusters(geometry, 4)

    assert set(clusters) == set(geometry.ids)
    assert min(clusters.values()) >= 0


def test_clustering_does_not_depend_on_input_order() -> None:
    """Artifacts must reproduce however the corpus arrived."""
    geometry = _ring(12)
    reversed_vectors = dict(reversed(list(geometry.vectors.items())))
    shuffled = CoreSetGeometry.of(reversed_vectors, dict(geometry.masses))

    assert descriptor_clusters(geometry, 4) == descriptor_clusters(shuffled, 4)


def test_one_group_needs_no_iteration() -> None:
    """A partition too small to split is the common case on a small corpus."""
    geometry = _ring(5)

    assert set(descriptor_clusters(geometry, 1).values()) == {0}
    assert MAXIMUM_ITERATIONS == 20
