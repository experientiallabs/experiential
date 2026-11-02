"""Tests for how leftover representative budget is spent."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pytest

from exp.simulation.mining.clustering import Hierarchy
from exp.simulation.mining.geometry import CoreSetGeometry
from exp.simulation.mining.sampling import (
    BalancedSamplingSpec,
    DiversitySamplingSpec,
    HierarchicalSamplingSpec,
    MedoidSamplingSpec,
    ProportionalSamplingSpec,
    RepresentativeSamplingSpec,
    SamplingSpec,
    StratifiedSamplingSpec,
    _largest_remainder,
    fill_budget,
    members_by_group,
)

_SPECS: tuple[SamplingSpec, ...] = (
    MedoidSamplingSpec(),
    ProportionalSamplingSpec(),
    DiversitySamplingSpec(),
    StratifiedSamplingSpec(),
    RepresentativeSamplingSpec(),
    BalancedSamplingSpec(),
    HierarchicalSamplingSpec(),
)


def _ring(count: int, *, masses: dict[str, int] | None = None) -> CoreSetGeometry:
    """Place `count` identities evenly on a circle so every pair has a distinct distance."""
    vectors = {
        f"trace-{index:02d}": (
            math.cos(2 * math.pi * index / count),
            math.sin(2 * math.pi * index / count),
        )
        for index in range(count)
    }
    return CoreSetGeometry.of(vectors, masses or dict.fromkeys(vectors, 1))


def _labels(geometry: CoreSetGeometry, groups: int) -> dict[str, int]:
    """Label identities round-robin, which is enough structure to exercise every strategy."""
    return {identity: index % groups for index, identity in enumerate(geometry.ids)}


def _cascade(groups: int) -> Hierarchy:
    """Build a tree over `groups` leaves, each joined in turn to the one root above it.

    One leaf per flat cluster, so a tree-reading strategy can be exercised on the same fixtures as
    every other strategy. The heights ascend with the merges, which is what a cut reads.
    """
    parent = np.full(max(2 * groups - 1, 1), -1)
    height = np.zeros(len(parent))
    current = 0
    for leaf in range(1, groups):
        node = groups + leaf - 1
        parent[current] = parent[leaf] = node
        height[node] = leaf / groups
        current = node
    return Hierarchy(parent, np.arange(groups), height)


def _run(
    spec: SamplingSpec, geometry: CoreSetGeometry, labels: dict[str, int], target: int
) -> tuple[list[str], dict[str, list[str]]]:
    """Fill one budget from empty and return the selection order with its reasons.

    The flat clusters double as the tree's assignment units, so the one strategy that reads a tree
    is held to every rule the others are held to on the same fixture.
    """
    selected: list[str] = []
    reasons: dict[str, list[str]] = defaultdict(list)
    fill_budget(
        spec,
        geometry=geometry,
        labels=labels,
        selected_ids=selected,
        reasons=reasons,
        target=target,
        hierarchy=_cascade(len(set(labels.values()))),
        units=labels,
    )
    return selected, reasons


def _main_medoid_fill(geometry: CoreSetGeometry, labels: dict[str, int], target: int) -> list[str]:
    """Replicate the pre-geometry fill loop, transcribed from the implementation it replaced.

    Keeping an independent transcription here means the default strategy is checked against the
    algorithm it claims to preserve, not merely against its own output.
    """
    selected: list[str] = []
    reasons: dict[str, list[str]] = defaultdict(list)
    for cluster_id in sorted(set(labels.values())):
        member_ids = [identity for identity in geometry.ids if labels[identity] == cluster_id]
        medoid_id = geometry.medoid(member_ids)
        if medoid_id in selected:
            reasons[medoid_id].append(f"cluster-medoid:{cluster_id}")
            continue
        if len(selected) >= target:
            break
        selected.append(medoid_id)
    while len(selected) < target:
        selected.append(geometry.farthest_first(selected))
    return selected


def test_medoid_fill_matches_the_implementation_it_replaced() -> None:
    """The default strategy is the provider-free path and must not drift from it."""
    geometry = _ring(24)
    labels = _labels(geometry, 5)

    for target in (1, 3, 5, 7, 12, 24):
        selected, _ = _run(MedoidSamplingSpec(), geometry, labels, target)
        assert selected == _main_medoid_fill(geometry, labels, target)


def test_medoid_fill_records_a_covered_centre_without_spending_budget() -> None:
    """A cluster whose centre another phase already took is still reported as medoid-covered."""
    geometry = _ring(6)
    labels = _labels(geometry, 3)
    selected = [geometry.medoid([i for i in geometry.ids if labels[i] == 0])]
    reasons: dict[str, list[str]] = defaultdict(list)

    fill_budget(
        MedoidSamplingSpec(),
        geometry=geometry,
        labels=labels,
        selected_ids=selected,
        reasons=reasons,
        target=1,
    )

    assert selected == selected[:1]
    assert reasons[selected[0]] == ["cluster-medoid:0"]


@pytest.mark.parametrize("spec", _SPECS)
def test_every_strategy_fills_exactly_the_budget(spec: SamplingSpec) -> None:
    """Underfilling loses coverage and overfilling breaks the task-count contract."""
    geometry = _ring(20)
    labels = _labels(geometry, 4)

    for target in (1, 2, 7, 19, 20):
        selected, _ = _run(spec, geometry, labels, target)
        assert len(selected) == target
        assert len(set(selected)) == target


@pytest.mark.parametrize("spec", _SPECS)
def test_every_strategy_is_independent_of_label_mapping_order(spec: SamplingSpec) -> None:
    """A strategy that read dict order would make artifacts depend on how the corpus arrived."""
    geometry = _ring(16)
    labels = _labels(geometry, 4)
    shuffled = dict(reversed(list(labels.items())))

    assert _run(spec, geometry, labels, 9)[0] == _run(spec, geometry, shuffled, 9)[0]


@pytest.mark.parametrize("spec", _SPECS)
def test_every_strategy_leaves_an_already_full_selection_alone(spec: SamplingSpec) -> None:
    """Reserved slots and lineage anchors can consume the whole budget on a small corpus."""
    geometry = _ring(5)
    labels = _labels(geometry, 2)
    selected = list(geometry.ids[:3])
    reasons: dict[str, list[str]] = defaultdict(list)

    fill_budget(
        spec,
        geometry=geometry,
        labels=labels,
        selected_ids=selected,
        reasons=reasons,
        target=3,
        hierarchy=_cascade(2),
        units=labels,
    )

    assert selected == list(geometry.ids[:3])


def test_proportional_gives_a_heavy_cluster_more_slots() -> None:
    """Matching the corpus's shape is the whole reason to choose this strategy."""
    geometry = _ring(
        12, masses={f"trace-{index:02d}": 50 if index < 4 else 1 for index in range(12)}
    )
    labels = {identity: 0 if index < 4 else 1 for index, identity in enumerate(geometry.ids)}

    selected, _ = _run(ProportionalSamplingSpec(), geometry, labels, 6)

    assert sum(labels[identity] == 0 for identity in selected) > 3


def test_diversity_ignores_clusters_entirely() -> None:
    """Relabelling must not move a selection that is defined to be label-blind."""
    geometry = _ring(14)

    one = _run(DiversitySamplingSpec(), geometry, _labels(geometry, 2), 6)[0]
    many = _run(DiversitySamplingSpec(), geometry, _labels(geometry, 7), 6)[0]

    assert one == many


def test_stratified_reaches_a_small_stratum_a_mass_split_would_starve() -> None:
    """Rare behaviour surviving selection is the point of equal shares."""
    geometry = _ring(
        12, masses={f"trace-{index:02d}": 1 if index < 2 else 40 for index in range(12)}
    )
    labels = {identity: 0 if index < 2 else 1 for index, identity in enumerate(geometry.ids)}

    stratified, _ = _run(StratifiedSamplingSpec(), geometry, labels, 4)
    proportional, _ = _run(ProportionalSamplingSpec(), geometry, labels, 4)

    assert sum(labels[identity] == 0 for identity in stratified) == 2
    assert sum(labels[identity] == 0 for identity in proportional) == 0


def test_stratified_prefers_explicit_strata_over_flat_clusters() -> None:
    """A hierarchy's coarse cut is the grouping a reviewer asked to spread budget over."""
    geometry = _ring(12)
    labels = _labels(geometry, 6)
    strata = {identity: index % 2 for index, identity in enumerate(geometry.ids)}
    selected: list[str] = []
    reasons: dict[str, list[str]] = defaultdict(list)

    fill_budget(
        StratifiedSamplingSpec(),
        geometry=geometry,
        labels=labels,
        selected_ids=selected,
        reasons=reasons,
        target=4,
        strata=strata,
    )

    assert sorted(reasons[identity][0] for identity in selected) == [
        "stratum:0",
        "stratum:0",
        "stratum:1",
        "stratum:1",
    ]


def test_largest_remainder_hands_out_every_slot() -> None:
    """Rounding each share down independently would silently drop slots from the budget."""
    assert sum(_largest_remainder({0: 1, 1: 1, 2: 1}, 7).values()) == 7
    assert _largest_remainder({0: 3, 1: 1}, 4) == {0: 3, 1: 1}
    assert _largest_remainder({0: 0, 1: 0}, 3) == {0: 0, 1: 0}
    assert _largest_remainder({}, 3) == {}


def test_members_by_group_omits_an_emptied_group_and_sorts_its_members() -> None:
    """Coverage validators reject an empty cluster, so grouping must never produce one."""
    assert members_by_group({"b": 2, "a": 2, "c": 7}) == {2: ["a", "b"], 7: ["c"]}


def test_proportional_records_a_covered_centre_like_the_medoid_strategy() -> None:
    """Audit reasons should not depend on which strategy spent the budget."""
    geometry = _ring(9)
    labels = _labels(geometry, 3)
    taken = geometry.medoid([i for i in geometry.ids if labels[i] == 0])
    selected = [taken]
    reasons: dict[str, list[str]] = defaultdict(list)

    fill_budget(
        ProportionalSamplingSpec(),
        geometry=geometry,
        labels=labels,
        selected_ids=selected,
        reasons=reasons,
        target=4,
    )

    assert "cluster-medoid:0" in reasons[taken]
    assert len(selected) == 4


def test_representative_fill_never_falls_through_to_farthest_first() -> None:
    """Spending the tail on outliers is the one emphasis these two strategies exist to avoid."""
    geometry = _ring(12)
    labels = _labels(geometry, 3)

    for spec in (RepresentativeSamplingSpec(), BalancedSamplingSpec()):
        selected, reasons = _run(spec, geometry, labels, 9)
        assert len(selected) == 9
        assert all(
            reason.startswith("cluster-representative:")
            for identity in selected
            for reason in reasons[identity]
        )


def test_representative_fill_reaches_the_budget_when_a_cluster_runs_dry() -> None:
    """Slots a tiny cluster cannot spend go to clusters that can, not to the farthest-first tail."""
    geometry = _ring(10)
    labels = {identity: (0 if index else 1) for index, identity in enumerate(geometry.ids)}

    selected, reasons = _run(BalancedSamplingSpec(), geometry, labels, 8)

    assert len(selected) == 8
    assert sum(1 for identity in selected if labels[identity] == 1) == 1
    assert all(
        reason.startswith("cluster-representative:")
        for identity in selected
        for reason in reasons[identity]
    )


def test_representative_weights_clusters_by_mass_and_balanced_does_not() -> None:
    """The two strategies differ in allocation alone, which is the choice between them."""
    masses = {f"trace-{index:02d}": (20 if index % 3 == 0 else 1) for index in range(12)}
    geometry = _ring(12, masses=masses)
    labels = {identity: index % 3 for index, identity in enumerate(geometry.ids)}

    def heavy_picks(spec: SamplingSpec) -> int:
        return sum(1 for i in _run(spec, geometry, labels, 6)[0] if labels[i] == 0)

    heavy = heavy_picks(RepresentativeSamplingSpec())
    even = heavy_picks(BalancedSamplingSpec())

    assert heavy > even
    assert even == 2


def test_representative_fill_starts_from_each_cluster_centre() -> None:
    """One slot per cluster has to land on the cluster's medoid, as every other strategy's does."""
    geometry = _ring(9)
    labels = _labels(geometry, 9)

    selected, _ = _run(RepresentativeSamplingSpec(), geometry, labels, 9)

    assert sorted(selected) == sorted(
        geometry.medoid(members) for members in members_by_group(labels).values()
    )


def _siblings() -> tuple[CoreSetGeometry, dict[str, int], Hierarchy]:
    """Build one heavy archetype beside twenty light siblings under a single parent.

    Ten cases of mass ten carry a hundred units of workload between them and share one leaf.
    Twenty cases of mass one each hold a leaf of their own, and all twenty of those leaves are
    siblings under one node, so the tree says they are one broad kind of work. Flat apportionment
    cannot see that: it compares each light leaf's mass of one against the heavy leaf's hundred,
    while the descent compares the family's pooled twenty against it.
    """
    masses = {f"trace-{index:02d}": (10 if index < 10 else 1) for index in range(30)}
    geometry = _ring(30, masses=masses)
    units = {
        identity: (0 if index < 10 else index - 9) for index, identity in enumerate(geometry.ids)
    }
    parent = np.full(23, -1)
    height = np.zeros(23)
    for leaf in range(1, 21):
        parent[leaf] = 21
    parent[0] = parent[21] = 22
    height[21], height[22] = 0.4, 0.9
    return geometry, units, Hierarchy(parent, np.arange(21), height)


def _fill(
    spec: SamplingSpec,
    geometry: CoreSetGeometry,
    units: dict[str, int],
    hierarchy: Hierarchy,
    target: int,
) -> tuple[list[str], dict[str, list[str]]]:
    """Fill one budget over a hand-built tree whose leaves are also the flat clusters."""
    selected: list[str] = []
    reasons: dict[str, list[str]] = defaultdict(list)
    fill_budget(
        spec,
        geometry=geometry,
        labels=units,
        selected_ids=selected,
        reasons=reasons,
        target=target,
        hierarchy=hierarchy,
        units=units,
    )
    return selected, reasons


def test_hierarchical_fill_refuses_a_partition_with_no_tree() -> None:
    """Filling the budget from flat clusters instead would report coverage it did not have."""
    geometry = _ring(8)
    labels = _labels(geometry, 2)

    for hierarchy, units in ((None, labels), (_cascade(2), None), (None, None)):
        with pytest.raises(ValueError, match="pattern_hierarchy or atoms_hierarchy"):
            fill_budget(
                HierarchicalSamplingSpec(),
                geometry=geometry,
                labels=labels,
                selected_ids=[],
                reasons=defaultdict(list),
                target=4,
                hierarchy=hierarchy,
                units=units,
            )


def test_hierarchical_fill_refuses_units_that_name_no_leaf_of_the_tree() -> None:
    """A tree and the units indexing it come from one assignment and cannot be mixed."""
    geometry = _ring(4)
    labels = _labels(geometry, 2)

    with pytest.raises(ValueError, match="not one of the 2 units"):
        fill_budget(
            HierarchicalSamplingSpec(),
            geometry=geometry,
            labels=labels,
            selected_ids=[],
            reasons=defaultdict(list),
            target=2,
            hierarchy=_cascade(2),
            units={identity: 7 for identity in geometry.ids},
        )


def test_hierarchical_fill_does_not_starve_a_group_of_small_siblings() -> None:
    """This is the claim the strategy makes, so it is measured against flat apportionment.

    Flat mass-proportional apportionment hands the heavy archetype nine of ten slots, because each
    light case's own share rounds to nothing and only one of them wins a remainder. Pooling the
    twenty light siblings under their parent earns the family two whole slots instead of one, and
    the two come from two different light cases rather than one case twice.
    """
    geometry, units, hierarchy = _siblings()

    descended, reasons = _fill(HierarchicalSamplingSpec(), geometry, units, hierarchy, 10)
    flat, _ = _fill(RepresentativeSamplingSpec(), geometry, units, hierarchy, 10)

    def light(selected: list[str]) -> int:
        """Count the picks that came from the group of small siblings."""
        return sum(1 for identity in selected if units[identity] > 0)

    assert len(descended) == len(flat) == 10
    assert light(flat) == 1
    assert light(descended) == 2
    assert len({units[identity] for identity in descended if units[identity] > 0}) == 2
    assert all(
        reason == "hierarchy-representative"
        for identity in descended
        for reason in reasons[identity]
    )


def test_hierarchical_fill_records_one_reason_a_reader_can_resolve() -> None:
    """The audit reason says the tree bought the slot and deliberately not which node did.

    A node index is a pattern or atom index below the leaf count, a merge index above it, and
    possibly a synthetic root. Nothing persists the tree beside the coverage report, so no reader
    of the artifact could ever map one back to anything, which is why the reason carries no
    identity at all rather than one that only this process understood.
    """
    geometry, units, hierarchy = _siblings()

    selected, reasons = _fill(HierarchicalSamplingSpec(), geometry, units, hierarchy, 4)

    assert selected
    assert all(reasons[identity] == ["hierarchy-representative"] for identity in selected)


def test_hierarchical_fill_spends_the_whole_budget_and_never_more() -> None:
    """Underfilling loses coverage and overfilling breaks the task-count contract."""
    geometry, units, hierarchy = _siblings()

    for target in (1, 2, 3, 9, 17, 29, 30):
        selected, _ = _fill(HierarchicalSamplingSpec(), geometry, units, hierarchy, target)
        assert len(selected) == target
        assert len(set(selected)) == target


def test_hierarchical_fill_reaches_every_candidate_when_the_budget_allows() -> None:
    """A subtree the descent could not reach would hold cases selection can never choose."""
    geometry, units, hierarchy = _siblings()

    selected, _ = _fill(HierarchicalSamplingSpec(), geometry, units, hierarchy, 30)

    assert sorted(selected) == sorted(geometry.ids)


def test_hierarchical_fill_returns_when_a_pass_can_apportion_nothing() -> None:
    """A pass that selects nothing has to end the descent rather than repeat itself.

    Mass is what the descent apportions, so a partition carrying none leaves every child with a
    share of zero and the descent with nothing to spend. The guarantee is that it returns instead
    of looping, and the budget then falls through to the farthest-first tail that every strategy
    ends in.
    """
    geometry, units, hierarchy = _siblings()
    massless = CoreSetGeometry.of(geometry.vectors, dict.fromkeys(geometry.ids, 0))

    selected, reasons = _fill(HierarchicalSamplingSpec(), massless, units, hierarchy, 6)

    assert len(selected) == 6
    assert all(reasons[identity] == ["farthest-first"] for identity in selected)


def test_hierarchical_fill_ignores_the_order_the_units_arrived_in() -> None:
    """Every tie in the descent is broken by node id or by identity, for exactly this reason."""
    geometry, units, hierarchy = _siblings()
    reversed_units = dict(reversed(list(units.items())))

    forward, _ = _fill(HierarchicalSamplingSpec(), geometry, units, hierarchy, 9)
    backward, _ = _fill(HierarchicalSamplingSpec(), geometry, reversed_units, hierarchy, 9)

    assert forward == backward


def test_hierarchical_fill_repeats_itself_exactly() -> None:
    """There is no seed on this spec because there is no random source, which is checked here."""
    geometry, units, hierarchy = _siblings()

    answers = {
        tuple(_fill(HierarchicalSamplingSpec(), geometry, units, hierarchy, 7)[0]) for _ in range(5)
    }

    assert len(answers) == 1
