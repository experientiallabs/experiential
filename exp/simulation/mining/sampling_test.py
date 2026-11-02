"""Tests for how leftover representative budget is spent."""

from __future__ import annotations

import math
from collections import defaultdict

import pytest

from exp.simulation.mining.geometry import CoreSetGeometry
from exp.simulation.mining.sampling import (
    BalancedSamplingSpec,
    DiversitySamplingSpec,
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


def _run(
    spec: SamplingSpec, geometry: CoreSetGeometry, labels: dict[str, int], target: int
) -> tuple[list[str], dict[str, list[str]]]:
    """Fill one budget from empty and return the selection order with its reasons."""
    selected: list[str] = []
    reasons: dict[str, list[str]] = defaultdict(list)
    fill_budget(
        spec,
        geometry=geometry,
        labels=labels,
        selected_ids=selected,
        reasons=reasons,
        target=target,
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
