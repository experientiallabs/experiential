"""Tests for the shared core-set metric."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import pytest

from exp.simulation.mining.geometry import CoreSetGeometry, cosine


def _geometry(**rows: tuple[float, ...]) -> CoreSetGeometry:
    """Build a geometry with unit mass per identity."""
    return CoreSetGeometry.of(dict(rows), dict.fromkeys(rows, 1))


def test_cosine_clamps_drift_into_the_valid_range() -> None:
    """Coverage records a similarity bounded at one, so the metric cannot hand them 1.0000001."""
    almost = 1.0 / math.sqrt(2.0)
    assert cosine((almost, almost), (almost, almost)) <= 1.0
    assert cosine((1.0, 0.0), (-1.0, 0.0)) == -1.0


def test_cosine_refuses_two_widths() -> None:
    """Two geometries have no distance between them, so pairing them must fail rather than pad."""
    with pytest.raises(ValueError):
        cosine((1.0, 0.0), (1.0, 0.0, 0.0))


def test_the_medoid_follows_workload_rather_than_coordinates() -> None:
    """The centre of a group is the case representing its traffic, not its middle coordinate."""
    vectors = {"a": (1.0, 0.0), "b": (0.0, 1.0), "c": (0.8, 0.6)}
    balanced = CoreSetGeometry.of(vectors, {"a": 1, "b": 1, "c": 1})
    skewed = CoreSetGeometry.of(vectors, {"a": 50, "b": 1, "c": 1})

    assert balanced.medoid(("a", "b", "c")) == "c"
    assert skewed.medoid(("a", "b", "c")) == "a"


def test_every_tie_breaks_on_ascending_identity() -> None:
    """Identical positions must not let input order decide, or artifacts stop reproducing."""
    geometry = _geometry(zeta=(1.0, 0.0), alpha=(1.0, 0.0), mu=(1.0, 0.0))

    assert geometry.medoid(("zeta", "mu", "alpha")) == "alpha"
    assert geometry.medoid(("alpha", "mu", "zeta")) == "alpha"
    assert geometry.farthest_first(("alpha",)) == "mu"


def test_the_first_coverage_point_is_a_centre_not_an_outlier() -> None:
    """With nothing selected, farthest-first has no reference, so it starts from the medoid."""
    geometry = _geometry(a=(1.0, 0.0), b=(0.98, 0.199), c=(0.0, 1.0))

    # b sits between a and the outlier c, so it is the weighted centre of the three.
    assert geometry.farthest_first(()) == "b"
    assert geometry.farthest_first(()) == geometry.medoid(geometry.ids)


def test_farthest_first_measures_against_everything_already_selected() -> None:
    """Filling one cluster must not place a point beside a representative chosen elsewhere."""
    geometry = _geometry(near=(1.0, 0.0), far=(0.0, 1.0), middle=(0.7071, 0.7071))

    assert geometry.farthest_first(("near",), within=("far", "middle")) == "far"
    assert geometry.farthest_first(("near", "far"), within=("far", "middle")) == "middle"


def test_an_exhausted_eligible_set_is_a_failure_rather_than_a_repeat() -> None:
    """Returning an already-selected identity would silently shrink the task set."""
    geometry = _geometry(a=(1.0, 0.0))

    with pytest.raises(ValueError, match="no remaining candidate"):
        geometry.farthest_first(("a",))


def test_positions_without_workload_are_refused() -> None:
    """A candidate with no mass would vanish from conservation accounting."""
    with pytest.raises(ValueError, match="one workload mass per positioned identity"):
        CoreSetGeometry.of({"a": (1.0,), "b": (0.0,)}, {"a": 1})


def test_identities_are_sorted_so_iteration_never_depends_on_input_order() -> None:
    """Every backend builds its matrix from this order, so it has to be canonical."""
    assert _geometry(zeta=(1.0,), alpha=(1.0,), mu=(1.0,)).ids == ("alpha", "mu", "zeta")


def test_the_first_representative_is_exactly_the_medoid() -> None:
    """Facility location generalizes the medoid, so its first pick must not be a second opinion."""
    vectors = {"a": (1.0, 0.0), "b": (0.0, 1.0), "c": (0.8, 0.6), "d": (0.6, 0.8)}
    for masses in ({"a": 1, "b": 1, "c": 1, "d": 1}, {"a": 50, "b": 1, "c": 1, "d": 1}):
        candidates = CoreSetGeometry.of(vectors, masses)
        members = tuple(vectors)
        assert candidates.representatives(members, 1) == (candidates.medoid(members),)


def test_representatives_prefer_common_cases_to_distinctive_ones() -> None:
    """A crowd carrying the workload plus one rare outlier must report the crowd.

    The crowd has to actually be the workload for this to be the claim. Facility location maximizes
    workload-weighted coverage, so under equal masses a lone outlier is a hundredth of the corpus
    that nothing else covers at all, and spending one slot of three on it is the right answer
    rather than a bug. What distinguishes this picker from farthest-first is that it will not buy
    that coverage out of proportion to the traffic behind it, which is what the masses here say.
    """
    vectors = {f"crowd-{index}": (math.cos(index / 8), math.sin(index / 8)) for index in range(8)}
    vectors["outlier"] = (-1.0, 0.0)
    masses = dict.fromkeys(vectors, 100)
    masses["outlier"] = 1
    candidates = CoreSetGeometry.of(vectors, masses)

    picked = candidates.representatives(tuple(vectors), 3)

    assert "outlier" not in picked
    assert candidates.farthest_first(list(picked)) == "outlier"


def test_representatives_skip_mass_a_reserved_slot_already_covers() -> None:
    """A centre another phase already took is covered, so the next pick is the next centre."""
    vectors = {"a": (1.0, 0.0), "b": (0.99, 0.141), "c": (0.0, 1.0), "d": (0.141, 0.99)}
    candidates = CoreSetGeometry.of(vectors, dict.fromkeys(vectors, 1))
    members = tuple(vectors)
    first = candidates.representatives(members, 1)[0]

    after = candidates.representatives(members, 1, covered_ids=[first])

    assert after[0] != first
    assert candidates.similarity(after[0], first) < 0.9


def test_representatives_stop_at_the_members_there_are() -> None:
    """A cluster cannot be asked for more representatives than it has distinct members."""
    candidates = _geometry(a=(1.0, 0.0), b=(0.0, 1.0))

    assert len(candidates.representatives(("a", "b", "a"), 5)) == 2
    assert candidates.representatives(("a", "b"), 0) == ()
    assert candidates.representatives(("a", "b"), 2, covered_ids=["a", "b"]) == ()


def test_a_negative_representative_count_is_refused() -> None:
    """No caller means a negative count, so it is a mistake rather than an empty result."""
    with pytest.raises(ValueError):
        _geometry(a=(1.0, 0.0)).representatives(("a",), -1)


def test_representative_order_never_depends_on_member_order() -> None:
    """Selection order fixes artifact bytes, so it cannot move when the member list is reversed."""
    vectors = {f"t-{index:02d}": (math.cos(index), math.sin(index)) for index in range(9)}
    candidates = CoreSetGeometry.of(vectors, dict.fromkeys(vectors, 1))

    forward = candidates.representatives(tuple(vectors), 4)
    backward = candidates.representatives(tuple(reversed(tuple(vectors))), 4)

    assert forward == backward


def _scalar_representatives(
    geometry: CoreSetGeometry,
    member_ids: Sequence[str],
    count: int,
    covered_ids: Sequence[str] = (),
) -> tuple[str, ...]:
    """Pick representatives one pair at a time, which is the objective's plain definition.

    The shipped picker computes the same thing as blocked matmuls because the objective is
    quadratic in the group size. This is the reference it has to agree with: nothing here is
    vectorized, so a disagreement is the fast path's bug and not a question of which is right.
    """
    members = tuple(dict.fromkeys(member_ids))
    taken = set(covered_ids)
    eligible = [member for member in members if member not in taken]
    best = {
        member: max((geometry.similarity(member, c) for c in covered_ids), default=-1.0)
        for member in members
    }

    def gain(candidate: str) -> float:
        """Return the workload-weighted similarity one candidate would add."""
        return sum(
            geometry.masses[member]
            * max(0.0, geometry.similarity(candidate, member) - best[member])
            for member in members
        )

    chosen: list[str] = []
    while len(chosen) < min(count, len(eligible)):
        pick = min(
            (member for member in eligible if member not in chosen),
            key=lambda candidate: (-gain(candidate), candidate),
        )
        chosen.append(pick)
        for member in members:
            best[member] = max(best[member], geometry.similarity(member, pick))
    return tuple(chosen)


def test_the_blocked_picker_agrees_with_the_plain_definition() -> None:
    """Scoring picks as matmuls is an optimization, so it must choose what the definition does."""
    generator = random.Random(11)
    for _ in range(40):
        count = generator.randint(2, 25)
        width = generator.randint(1, 5)
        vectors: dict[str, tuple[float, ...]] = {}
        for index in range(count):
            raw = [generator.gauss(0, 1) for _ in range(width)]
            norm = math.sqrt(sum(value * value for value in raw)) or 1.0
            vectors[f"id-{generator.randint(0, 999):03d}-{index}"] = tuple(
                value / norm for value in raw
            )
        candidates = CoreSetGeometry.of(
            vectors, {identity: generator.randint(1, 40) for identity in vectors}
        )
        order = list(vectors)
        generator.shuffle(order)
        covered = tuple(generator.sample(order, generator.randint(0, min(3, count - 1))))
        wanted = generator.randint(0, count)

        assert candidates.representatives(
            order, wanted, covered_ids=covered
        ) == _scalar_representatives(candidates, order, wanted, covered)


def test_identical_members_are_separated_by_identity_alone() -> None:
    """Exact gain ties are the common case on tagged data, so the tie-break decides the result."""
    vectors = {f"t-{index:03d}": (1.0, 0.0) for index in range(12)}
    vectors["t-900"] = (0.0, 1.0)
    candidates = CoreSetGeometry.of(vectors, dict.fromkeys(vectors, 1))
    reversed_order = list(vectors)[::-1]

    picked = candidates.representatives(reversed_order, 4)

    assert picked == _scalar_representatives(candidates, reversed_order, 4)
    assert picked == ("t-000", "t-900", "t-001", "t-002")


def test_members_positioned_in_different_geometries_are_refused() -> None:
    """A distance between two widths is undefined, so it fails by name rather than inside numpy."""
    mixed = CoreSetGeometry.of({"a": (1.0, 0.0), "b": (1.0,)}, {"a": 1, "b": 1})

    with pytest.raises(ValueError, match="different widths"):
        mixed.representatives(("a", "b"), 1)
