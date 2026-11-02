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

import numpy as np

GAIN_TIE_TOLERANCE = 1e-12
"""Width of the facility-location gain tie set, relative to the mass the gains are measured in.

:meth:`CoreSetGeometry.representatives` scores its candidates with a blocked matmul, which does not
return bitwise-identical rows for bitwise-identical inputs: a row scored inside the micro-kernel and
a row scored on the remainder path accumulate in different orders. Two candidates that are
mathematically tied, two duplicate vectors for instance, therefore come out a few last bits apart,
and an exactly-equal tie set holds one of them, leaving the documented identity tie-break nothing to
break. The tie set is built with a tolerance instead.

It has to be relative because a gain is a sum of ``mass * similarity`` terms, so it scales with the
workload mass of the group: an absolute epsilon that is generous on a group carrying ten traces is
no tolerance at all on one carrying a million. Scaled this way it is many orders of magnitude above
that rounding and far below any difference between two candidates a reviewer would call different.
"""


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

    def representatives(
        self,
        member_ids: Sequence[str],
        count: int,
        *,
        covered_ids: Sequence[str] = (),
    ) -> tuple[str, ...]:
        """Return the members most of a group looks like, most typical first.

        Each pick is the member that most increases the workload-weighted similarity of the group
        to its nearest chosen representative. The first pick maximizes :meth:`medoid`'s own
        objective shifted by a constant, so it is that medoid, and every later one is the centre of
        whatever mass the earlier picks stand in for badly. The objective is monotone and
        submodular, so this order is within ``1 - 1/e`` of the best set of its size, and nothing
        has to be configured to say how much "typical" to prefer over "spread": taking the common
        case first is what maximizing it does.

        The two measure that shared objective in different arithmetic, numpy here and pure Python
        there, which is visible only on a group whose centre two members hold to within the last
        bits of a double: this method hands those to the lowest identity while :meth:`medoid` hands
        them to whichever measured higher. Members positioned identically are a tie in both, and
        both resolve a tie to the lowest identity.

        This is the opposite emphasis to :meth:`farthest_first`, which spends budget on the cases
        least like anything chosen so far. Both are defensible and which one a run wants is what a
        sampling spec records, so neither is reachable without naming it.

        The gains are computed in numpy rather than pair by pair, because the objective is
        quadratic in the group size and a mining partition is routinely thousands of candidates:
        scoring one pick over two thousand members costs seconds in a Python double loop and
        milliseconds as a matmul. The candidates are scored in blocks so a large group never
        materializes a members-by-members matrix.

        Args:
            member_ids: Identities whose workload is to be represented, at least one. Repeats are
                ignored, since an identity cannot be represented twice.
            count: Representatives to return, clamped to the number of members still eligible.
            covered_ids: Identities already retained elsewhere. Mass they already represent well
                is not paid for twice, so a group whose centre a reserved slot already took
                contributes its next most typical case rather than repeating that centre.

        Returns:
            Representatives in greedy order, each appearing once and none of them in
            ``covered_ids``. Shorter than ``count`` when the group runs out of members. The order
            does not depend on the order ``member_ids`` arrived in: every pick comes from the gain
            tie set :data:`GAIN_TIE_TOLERANCE` defines, by lowest identity.

        Raises:
            ValueError: ``count`` is negative, which no caller means, or the members were
                positioned in different geometries, between which no distance is defined.
        """
        if count < 0:
            raise ValueError("a group cannot be asked for a negative number of representatives")
        members = tuple(dict.fromkeys(member_ids))
        taken = set(covered_ids)
        eligible = tuple(member_id for member_id in members if member_id not in taken)
        if not eligible or count == 0:
            return ()
        positions = self._positions(members)
        mass = np.asarray([float(self.masses[member_id]) for member_id in members])
        row_of = {member_id: row for row, member_id in enumerate(members)}
        live = [row_of[member_id] for member_id in eligible]
        # How well each member is represented already. A member no representative covers sits at
        # -1, below any similarity, so the first pick is scored against the whole group's mass and
        # lands on its weighted centre.
        best = np.full(len(members), -1.0)
        for covered_id in covered_ids:
            best = np.maximum(best, self._similarities(positions, self.vectors[covered_id]))
        chosen: list[str] = []
        width = GAIN_TIE_TOLERANCE * float(mass.sum())
        while len(chosen) < min(count, len(eligible)):
            rows = np.asarray(live, dtype=np.int64)
            gains = self._gains(positions, mass, best, rows)
            # The tie-break is the whole of this method's input-order independence: among the
            # candidates that would add the most, the lowest identity wins, exactly as every other
            # core-set question here breaks a tie. The tie set is taken within
            # GAIN_TIE_TOLERANCE of the peak rather than at it, because the blocked matmul that
            # scored these candidates leaves mathematically tied rows a few last bits apart and an
            # exactly-equal tie set would hold whichever of them rounded upwards.
            peak = float(gains.max())
            pick = min(members[int(row)] for row in rows[gains >= peak - width])
            chosen.append(pick)
            live.remove(row_of[pick])
            best = np.maximum(best, self._similarities(positions, self.vectors[pick]))
        return tuple(chosen)

    def _positions(self, member_ids: Sequence[str]) -> np.ndarray:
        """Stack one row per member, refusing positions built in different geometries."""
        widths = {len(self.vectors[member_id]) for member_id in member_ids}
        if len(widths) > 1:
            raise ValueError(
                "core-set geometry cannot measure between positions of different widths, so "
                f"these {len(member_ids)} members were not all positioned in one geometry"
            )
        return np.asarray([self.vectors[member_id] for member_id in member_ids], dtype=np.float64)

    @staticmethod
    def _similarities(positions: np.ndarray, vector: tuple[float, ...]) -> np.ndarray:
        """Return every stacked position's bounded similarity to one vector."""
        return np.clip(positions @ np.asarray(vector, dtype=np.float64), -1.0, 1.0)

    @staticmethod
    def _gains(
        positions: np.ndarray, mass: np.ndarray, best: np.ndarray, rows: np.ndarray
    ) -> np.ndarray:
        """Return the workload-weighted similarity each named candidate would add to the group.

        One candidate's gain is ``sum(mass * max(0, similarity - best))`` over the whole group,
        which is a clipped similarity block reduced against the mass vector. The block height is
        chosen so the intermediate stays a few tens of megabytes whatever the group size.
        """
        gains = np.empty(len(rows))
        height = max(1, 4_000_000 // max(1, positions.shape[0]))
        for start in range(0, len(rows), height):
            block = rows[start : start + height]
            similarity = np.clip(positions[block] @ positions.T, -1.0, 1.0)
            gains[start : start + len(block)] = np.maximum(0.0, similarity - best) @ mass
        return gains

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
