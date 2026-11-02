"""The provider-free clustering a build has always used.

Deterministic Lloyd iteration over the selection geometry, with two properties that distinguish it
from an ordinary k-means and are the reason it is the default. There is no randomness anywhere: the
first centre is the lexicographically smallest trace identity and the rest are placed
farthest-first, so a corpus clusters the same way on every machine without a seed to record. And a
centre is always a real trace rather than a computed mean, so every centre is a case someone can
read.

The group count is ``round(sqrt(n))``, bounded by the number of candidates and by the task budget.
Tying it to the budget is what matters: asking for more groups than there are slots would promise
one representative per group and then fail to deliver, and the cluster record would claim coverage
the task set does not have.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from exp.simulation.mining.geometry import CoreSetGeometry

MAXIMUM_ITERATIONS = 20
"""Lloyd passes before the assignment is accepted as it stands.

The loop normally reaches a fixed point well inside this, and a corpus that is still oscillating
after twenty passes is one whose geometry has no stable partition to find, so spending more passes
buys a different arbitrary answer rather than a better one.
"""


def default_cluster_count(candidate_count: int, target: int) -> int:
    """Return the group count the provider-free path uses.

    Args:
        candidate_count: Deduplicated candidates in this partition.
        target: Representative slots this partition may fill.

    Returns:
        ``round(sqrt(candidate_count))``, at least one, never above the candidate count or the
        target.
    """
    return min(candidate_count, max(1, round(math.sqrt(candidate_count))), target)


def default_k_grid(candidate_count: int) -> tuple[int, ...]:
    """Return the group counts a backend searches when nobody has said how many groups there are.

    A fixed grid cannot answer that question on two corpora of different sizes: a grid topping out
    at fifty is most of the plausible range for a thousand candidates and a narrow slice of it for
    a hundred thousand. The grid doubles from two up to ``2 * sqrt(n)``, which spans the orders of
    magnitude a selection criterion has to choose between while staying short enough that every
    candidate count is actually fitted.

    Args:
        candidate_count: Deduplicated candidates in this partition.

    Returns:
        Ascending distinct group counts, at least ``(1,)`` for a partition too small to split.
    """
    ceiling = min(candidate_count, max(2, round(2 * math.sqrt(candidate_count))))
    if ceiling < 2:
        return (max(1, candidate_count),)
    grid = {2}
    while max(grid) * 2 < ceiling:
        grid.add(max(grid) * 2)
    grid.add(ceiling)
    return tuple(sorted(grid))


def default_atom_count(candidate_count: int, groups: int) -> int:
    """Return the number of fine atoms a hierarchy is built over.

    Atoms exist so the tree above them can be re-cut to any group count without refitting, which
    only works when there are several atoms per group. ``4 * sqrt(n)`` keeps that ratio roughly
    constant as a corpus grows, and the result is never below the cut it has to support or above
    the candidates there are to fit.

    Args:
        candidate_count: Deduplicated candidates in this partition.
        groups: Groups the tree will be cut into.

    Returns:
        The atom count, at least ``groups`` and never above ``candidate_count``.
    """
    scaled = min(candidate_count, round(4 * math.sqrt(candidate_count)))
    return max(scaled, min(groups, candidate_count))


def descriptor_clusters(geometry: CoreSetGeometry, cluster_count: int) -> dict[str, int]:
    """Assign every candidate to a deterministic cluster around a real central trace.

    Args:
        geometry: Core-set positions and workload for one partition.
        cluster_count: Groups to form, at least one.

    Returns:
        One cluster identity per candidate identity.
    """
    if cluster_count <= 1:
        return dict.fromkeys(geometry.ids, 0)
    centers = _initial_centers(geometry, cluster_count)
    assignments: dict[str, int] = {}
    for _ in range(MAXIMUM_ITERATIONS):
        updated = {
            candidate_id: _nearest_center(geometry, candidate_id, centers)
            for candidate_id in geometry.ids
        }
        if updated == assignments:
            break
        assignments = updated
        centers = _reseed_centers(geometry, assignments, cluster_count)
    return assignments


def _initial_centers(geometry: CoreSetGeometry, count: int) -> list[str]:
    """Seed clusters with a lexical anchor followed by deterministic farthest-first points."""
    selected = [min(geometry.ids)]
    while len(selected) < count:
        selected.append(geometry.farthest_first(selected))
    return selected


def _nearest_center(geometry: CoreSetGeometry, candidate_id: str, centers: Sequence[str]) -> int:
    """Return the best center index, breaking vector ties by center identity."""
    ranked = sorted(
        (
            (-geometry.similarity(candidate_id, center_id), center_id, index)
            for index, center_id in enumerate(centers)
        )
    )
    return ranked[0][2]


def _reseed_centers(
    geometry: CoreSetGeometry, assignments: Mapping[str, int], cluster_count: int
) -> list[str]:
    """Choose a real weighted medoid for each non-empty cluster, retaining stable empty centers."""
    centers: list[str] = []
    for cluster_id in range(cluster_count):
        member_ids = [
            candidate_id for candidate_id in geometry.ids if assignments[candidate_id] == cluster_id
        ]
        if member_ids:
            centers.append(geometry.medoid(member_ids))
    while len(centers) < cluster_count:
        centers.append(geometry.farthest_first(centers))
    return centers
