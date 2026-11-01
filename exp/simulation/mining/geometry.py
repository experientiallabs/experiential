"""The one metric every core-set phase measures with.

Clustering, lineage anchors, cluster medoids, farthest-first coverage and nearest-representative
assignment all have to agree, bit for bit, about which case is central and which two cases are far
apart. When they disagree the phases quietly work against each other: the medoid one phase chose is
not the medoid the next phase reports, and the coverage record stops describing the selection it
claims to audit. Holding the metric in one place rather than one copy per caller makes that
agreement structural instead of a convention.

:class:`CoreSetGeometry` deliberately knows nothing about traces, candidates or descriptors. It
sees identities, one vector each and one workload mass each, which is the whole of what a core-set
decision depends on. That keeps selection geometry swappable by building a different mapping rather
than by editing an algorithm, and it keeps this module importable from anywhere in mining without a
dependency cycle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Return a bounded dot product for pre-normalized vectors.

    Args:
        left: One unit vector.
        right: Another unit vector of the same width.

    Returns:
        Their similarity, clamped into ``[-1, 1]`` against floating-point drift.

    Raises:
        ValueError: The vectors have different widths, which means the two positions were built in
            different geometries and no distance between them is meaningful.
    """
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=True))))


@dataclass(frozen=True)
class CoreSetGeometry:
    """One partition's positions and workload, with the core-set questions asked of them.

    Attributes:
        vectors: Unit selection vector per identity. Every vector shares one width, because a
            distance between two widths is not defined.
        masses: Uncollapsed workload mass per identity, used to weight centrality so the centre of
            a group is the case that best represents the traffic it carries rather than the case
            that sits in the middle of its coordinates.
        ids: Every identity in ascending order, so iteration never depends on input order.
    """

    vectors: Mapping[str, tuple[float, ...]]
    masses: Mapping[str, int]
    ids: tuple[str, ...]

    @classmethod
    def of(
        cls, vectors: Mapping[str, tuple[float, ...]], masses: Mapping[str, int]
    ) -> CoreSetGeometry:
        """Build a geometry over one partition's positions.

        Args:
            vectors: Unit selection vector per identity.
            masses: Workload mass per identity, covering exactly the same identities.

        Returns:
            The geometry, with identities sorted.

        Raises:
            ValueError: The two mappings name different identities.
        """
        if set(vectors) != set(masses):
            raise ValueError("core-set geometry needs one workload mass per positioned identity")
        return cls(vectors=vectors, masses=masses, ids=tuple(sorted(vectors)))

    def similarity(self, left_id: str, right_id: str) -> float:
        """Return the bounded similarity between two positioned identities."""
        return cosine(self.vectors[left_id], self.vectors[right_id])

    def medoid(self, member_ids: Sequence[str]) -> str:
        """Return a real weighted central member rather than a synthetic centroid.

        A centroid is an average request nobody ever sent, so it cannot become a task.

        Args:
            member_ids: Identities to centre, at least one.

        Returns:
            The most central member, with ties broken by ascending identity.
        """
        return min(
            member_ids,
            key=lambda candidate_id: (
                -sum(
                    self.similarity(candidate_id, other_id) * self.masses[other_id]
                    for other_id in member_ids
                ),
                candidate_id,
            ),
        )

    def farthest_first(
        self, selected_ids: Sequence[str], within: Sequence[str] | None = None
    ) -> str:
        """Choose the identity farthest from every already selected representative.

        Args:
            selected_ids: Identities already retained, possibly empty.
            within: Identities eligible to be chosen, or None for the whole partition. A sampling
                strategy that fills one cluster at a time passes that cluster's members, while the
                distance is still measured against every selected representative, so filling a
                cluster never places a point next to one already selected elsewhere.

        Returns:
            The eligible identity whose closest selected neighbour is furthest away, with ties
            broken by ascending identity. With nothing selected yet the weighted medoid of the
            eligible set is returned, so the first coverage point is a centre, not an outlier.

        Raises:
            ValueError: Every eligible identity is already selected.
        """
        taken = set(selected_ids)
        eligible = [
            candidate_id
            for candidate_id in (self.ids if within is None else within)
            if candidate_id not in taken
        ]
        if not eligible:
            raise ValueError("farthest-first selection has no remaining candidate")
        if not selected_ids:
            return self.medoid(eligible)
        return min(
            eligible,
            key=lambda candidate_id: (
                max(self.similarity(candidate_id, selected_id) for selected_id in selected_ids),
                candidate_id,
            ),
        )
