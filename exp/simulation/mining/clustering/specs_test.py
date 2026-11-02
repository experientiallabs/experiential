"""Tests for clustering spec dispatch and validation."""

from __future__ import annotations

import math
from collections.abc import Mapping

import pytest
from pydantic import TypeAdapter, ValidationError

from exp.simulation.mining.clustering.descriptor import default_cluster_count, descriptor_clusters
from exp.simulation.mining.clustering.specs import (
    AtomsHierarchyClusterSpec,
    BernoulliMixtureClusterSpec,
    ClusteringError,
    ClusterSpec,
    DescriptorClusterSpec,
    KMeansClusterSpec,
    PatternHierarchyClusterSpec,
    cluster_candidates,
    publishes_tree,
)
from exp.simulation.mining.geometry import CoreSetGeometry
from exp.simulation.mining.sampling import members_by_group

_ADAPTER: TypeAdapter[ClusterSpec] = TypeAdapter(ClusterSpec)


def _ring(count: int) -> CoreSetGeometry:
    """Place identities evenly on a circle."""
    vectors = {
        f"trace-{index:02d}": (
            math.cos(2 * math.pi * index / count),
            math.sin(2 * math.pi * index / count),
        )
        for index in range(count)
    }
    return CoreSetGeometry.of(vectors, dict.fromkeys(vectors, 1))


def _tagged(*patterns: str) -> tuple[CoreSetGeometry, dict[str, tuple[float, ...]]]:
    """Build a tagged partition whose answers threshold back to the named bit patterns.

    The geometry stays continuous, because that is what a tagged selection actually carries; only
    the clusterer thresholds, and it thresholds these answers rather than these coordinates.
    """
    answers = {
        f"trace-{index:02d}": tuple(0.9 if bit == "1" else 0.1 for bit in pattern)
        for index, pattern in enumerate(patterns)
    }
    vectors: dict[str, tuple[float, ...]] = {}
    for identity, row in answers.items():
        norm = math.sqrt(sum(value * value for value in row))
        vectors[identity] = tuple(value / norm for value in row)
    return CoreSetGeometry.of(vectors, dict.fromkeys(vectors, 1)), answers


def _grouping(labels: Mapping[str, int]) -> set[frozenset[str]]:
    """Return the partition a labelling describes, free of the identities it numbered it with."""
    return {frozenset(members) for members in members_by_group(labels).values()}


def _probabilities(geometry: CoreSetGeometry) -> dict[str, tuple[float, ...]]:
    """Return plausible tag answers, half the corpus triggering the opposite tags."""
    return {
        identity: (0.9, 0.8, 0.1, 0.1) if index % 2 else (0.1, 0.1, 0.85, 0.9)
        for index, identity in enumerate(geometry.ids)
    }


def test_the_descriptor_backend_dispatches_to_the_default_path() -> None:
    """The default spec has to be the same code the provider-free path always ran."""
    geometry = _ring(16)

    assignment = cluster_candidates(DescriptorClusterSpec(), geometry=geometry, target=8)

    assert assignment.labels == descriptor_clusters(geometry, default_cluster_count(16, 8))
    assert assignment.strata is None
    assert assignment.hierarchy is None


@pytest.mark.parametrize(
    "spec",
    [
        DescriptorClusterSpec(),
        KMeansClusterSpec(seed=3),
        KMeansClusterSpec(k=3, seed=3),
        AtomsHierarchyClusterSpec(atom_count=8, groups=4, seed=3),
    ],
)
def test_every_backend_labels_every_candidate_nonnegatively(spec: ClusterSpec) -> None:
    """Coverage validators reject a negative identity and selection looks every candidate up."""
    geometry = _ring(14)

    assignment = cluster_candidates(spec, geometry=geometry, target=7)

    assert set(assignment.labels) == set(geometry.ids)
    assert min(assignment.labels.values()) >= 0


@pytest.mark.parametrize(
    "spec",
    [
        DescriptorClusterSpec(),
        KMeansClusterSpec(seed=3),
        AtomsHierarchyClusterSpec(atom_count=8, groups=4, seed=3),
    ],
)
def test_every_backend_ignores_input_order(spec: ClusterSpec) -> None:
    """Every numpy backend builds its matrix from sorted identities for exactly this reason."""
    geometry = _ring(12)
    shuffled = CoreSetGeometry.of(
        dict(reversed(list(geometry.vectors.items()))), dict(geometry.masses)
    )

    assert (
        cluster_candidates(spec, geometry=geometry, target=6).labels
        == cluster_candidates(spec, geometry=shuffled, target=6).labels
    )


def test_the_mixture_reads_untransformed_answers() -> None:
    """Reading probabilities as probabilities is the whole reason this backend exists."""
    geometry = _ring(12)

    assignment = cluster_candidates(
        BernoulliMixtureClusterSpec(k=2, seed=1),
        geometry=geometry,
        target=6,
        probabilities=_probabilities(geometry),
    )

    groups = {
        assignment.labels[identity] for index, identity in enumerate(geometry.ids) if index % 2
    }
    assert len(groups) == 1


def test_the_mixture_refuses_to_guess_answers_from_a_geometry() -> None:
    """A normalized selection vector is not a probability, and inferring one would be a fiction."""
    geometry = _ring(10)

    with pytest.raises(ClusteringError, match="decision_tag selection geometry"):
        cluster_candidates(BernoulliMixtureClusterSpec(), geometry=geometry, target=5)


def test_the_mixture_names_a_partly_tagged_corpus() -> None:
    """Tagging only part of a corpus is the usual way to get here, so say which part."""
    geometry = _ring(6)
    partial = dict(list(_probabilities(geometry).items())[:3])

    with pytest.raises(ClusteringError, match="no answers for 3 candidates"):
        cluster_candidates(
            BernoulliMixtureClusterSpec(k=2),
            geometry=geometry,
            target=3,
            probabilities=partial,
        )


def test_the_hierarchy_backend_publishes_coarse_strata() -> None:
    """Stratified sampling needs groups coarser than the atoms it spreads budget over."""
    geometry = _ring(16)

    assignment = cluster_candidates(
        AtomsHierarchyClusterSpec(atom_count=8, groups=3, seed=1), geometry=geometry, target=8
    )

    assert assignment.strata is not None
    assert assignment.hierarchy is not None
    assert len(set(assignment.strata.values())) <= len(set(assignment.labels.values()))


def test_a_group_count_above_the_candidate_count_is_clamped() -> None:
    """A spec written for a large corpus still has to run on a small partition."""
    geometry = _ring(3)

    for spec in (
        KMeansClusterSpec(k=50, seed=1),
        AtomsHierarchyClusterSpec(atom_count=50, groups=20, seed=1),
    ):
        assignment = cluster_candidates(spec, geometry=geometry, target=3)
        assert set(assignment.labels) == set(geometry.ids)


def test_an_empty_partition_clusters_into_nothing() -> None:
    """A partition can be empty when every lineage went to the other side of the split."""
    empty = CoreSetGeometry.of({}, {})

    assert cluster_candidates(KMeansClusterSpec(), geometry=empty, target=0).labels == {}


def test_a_cut_finer_than_its_atoms_is_refused_at_the_spec() -> None:
    """Catching this when the spec is written beats discovering it mid-build."""
    with pytest.raises(ValidationError, match="needs at least that many"):
        AtomsHierarchyClusterSpec(atom_count=4, groups=16)


def test_a_mixture_with_no_k_needs_a_grid_to_choose_from() -> None:
    """Leaving both unset would leave nothing to fit."""
    with pytest.raises(ValidationError, match="at least one candidate group count"):
        BernoulliMixtureClusterSpec(k=None, k_grid=())


def test_the_discriminator_round_trips_through_json() -> None:
    """A spec is persisted with the task set and replayed, so its kind must select it back."""
    for spec in (
        DescriptorClusterSpec(),
        KMeansClusterSpec(k=7, seed=9),
        BernoulliMixtureClusterSpec(k_grid=(3, 6), novelty_quantile=0.05),
        AtomsHierarchyClusterSpec(atom_count=32, groups=8),
        PatternHierarchyClusterSpec(cut=0.15),
    ):
        restored = _ADAPTER.validate_python(spec.model_dump(mode="json"))
        assert restored == spec
        assert type(restored) is type(spec)


def test_an_unknown_backend_fails_closed() -> None:
    """A spec from a newer revision must not quietly fall back to the default grouping."""
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python({"kind": "spectral", "k": 4})


@pytest.mark.parametrize("count", [3, 16, 400])
def test_an_unsized_spec_sizes_every_backend_to_the_partition(count: int) -> None:
    """A spec carrying no sizes has to run on a tiny partition and a large one alike."""
    geometry = _ring(count)
    target = max(1, count // 4)

    for spec in (AtomsHierarchyClusterSpec(seed=1), KMeansClusterSpec(seed=1)):
        assignment = cluster_candidates(spec, geometry=geometry, target=target)
        assert set(assignment.labels) == set(geometry.ids)
        assert all(label >= 0 for label in assignment.labels.values())
        assert len(set(assignment.labels.values())) <= count

    mixture = cluster_candidates(
        BernoulliMixtureClusterSpec(seed=1),
        geometry=geometry,
        target=target,
        probabilities=_probabilities(geometry),
    )
    assert set(mixture.labels) == set(geometry.ids)


def test_an_unsized_hierarchy_always_has_atoms_enough_for_its_cut() -> None:
    """Naming only one of the two sizes must stay valid, since the other is derived to fit."""
    assert AtomsHierarchyClusterSpec(groups=8).atom_count is None
    assert AtomsHierarchyClusterSpec(atom_count=8).groups is None

    assignment = cluster_candidates(
        AtomsHierarchyClusterSpec(groups=6, seed=1), geometry=_ring(40), target=10
    )

    assert assignment.strata is not None
    assert len(set(assignment.strata.values())) <= len(set(assignment.labels.values()))


def test_the_pattern_backend_groups_identical_fired_sets_at_the_finest_cut() -> None:
    """A cut of zero means "the same propositions fired", which is the finest real grouping."""
    geometry, answers = _tagged("1100", "1100", "1110", "0011")

    assignment = cluster_candidates(
        PatternHierarchyClusterSpec(cut=0.0), geometry=geometry, target=4, probabilities=answers
    )

    assert _grouping(assignment.labels) == {
        frozenset({"trace-00", "trace-01"}),
        frozenset({"trace-02"}),
        frozenset({"trace-03"}),
    }
    # The strata, the units and the tree come with every pattern-hierarchy assignment now that the
    # backend publishes the agglomeration it already built, so this cut publishes them too.
    assert assignment.strata is not None
    assert assignment.units is not None
    assert assignment.hierarchy is not None


def test_raising_the_pattern_cut_rolls_near_identical_cases_together() -> None:
    """Re-cutting at a coarser similarity is what the cut is for, so the gradient is checked."""
    geometry, answers = _tagged("1100", "1110", "0011")

    def _groups(cut: float) -> set[frozenset[str]]:
        """Return the partition one cut produces."""
        return _grouping(
            cluster_candidates(
                PatternHierarchyClusterSpec(cut=cut),
                geometry=geometry,
                target=3,
                probabilities=answers,
            ).labels
        )

    assert len(_groups(0.1)) == 3
    assert _groups(0.4) == {frozenset({"trace-00", "trace-01"}), frozenset({"trace-02"})}


def test_the_pattern_threshold_decides_which_patterns_exist() -> None:
    """Which answers count as fired is a clustering decision, so this backend owns the choice."""
    geometry = CoreSetGeometry.of(
        {"trace-00": (1.0, 0.0), "trace-01": (0.0, 1.0)}, {"trace-00": 1, "trace-01": 1}
    )
    answers = {"trace-00": (0.8, 0.2), "trace-01": (0.8, 0.6)}

    def _groups(threshold: float) -> set[frozenset[str]]:
        """Return the partition one threshold produces at the finest cut."""
        return _grouping(
            cluster_candidates(
                PatternHierarchyClusterSpec(cut=0.0, threshold=threshold),
                geometry=geometry,
                target=2,
                probabilities=answers,
            ).labels
        )

    # 0.6 fires at the default and not above it, so the two cases are one pattern or two.
    assert _groups(0.5) == {frozenset({"trace-00"}), frozenset({"trace-01"})}
    assert _groups(0.7) == {frozenset({"trace-00", "trace-01"})}


def test_the_cases_that_fired_nothing_share_exactly_one_cluster() -> None:
    """Jaccard between two empty sets is undefined, and both substitutes are wrong answers."""
    geometry, answers = _tagged("0000", "1100", "0000", "1110", "0000")

    assignment = cluster_candidates(
        PatternHierarchyClusterSpec(cut=0.4), geometry=geometry, target=5, probabilities=answers
    )

    assert frozenset({"trace-00", "trace-02", "trace-04"}) in _grouping(assignment.labels)
    assert min(assignment.labels.values()) >= 0


def test_the_pattern_backend_refuses_to_guess_answers_from_a_geometry() -> None:
    """Thresholding a normalized vector would invent the one input this backend has."""
    with pytest.raises(ClusteringError, match="decision_tag selection geometry"):
        cluster_candidates(PatternHierarchyClusterSpec(), geometry=_ring(8), target=4)


def test_the_pattern_refusal_names_both_ways_out() -> None:
    """A failure a user can hit must say what to do about it, not only what went wrong."""
    with pytest.raises(ClusteringError) as caught:
        cluster_candidates(PatternHierarchyClusterSpec(), geometry=_ring(8), target=4)

    assert "decision_tag" in str(caught.value)
    assert "atoms_hierarchy" in str(caught.value)


def test_the_pattern_backend_names_a_partly_tagged_corpus() -> None:
    """Tagging only part of a corpus is the usual way to get here, so say which part."""
    geometry, answers = _tagged("1100", "1110", "0011")
    partial = dict(list(answers.items())[:2])

    with pytest.raises(ClusteringError, match="no answers for 1 candidates"):
        cluster_candidates(
            PatternHierarchyClusterSpec(), geometry=geometry, target=3, probabilities=partial
        )


def test_a_finest_cut_cluster_keeps_members_the_sampler_can_tell_apart() -> None:
    """Thresholding in the clusterer is what buys this, and it is the reason for the split.

    Every member of a cluster cut at zero fired exactly the same propositions. Had the selection
    vector been that same thresholded row, the members would be at identical positions and a
    medoid or facility-location pick inside the cluster would be pure alphabetical tie-break.
    """
    geometry, answers = _tagged("1100", "1100", "1100")
    geometry = CoreSetGeometry.of(
        {
            "trace-00": (0.8, 0.6, 0.0, 0.0),
            "trace-01": (0.6, 0.8, 0.0, 0.0),
            "trace-02": (0.7071, 0.7071, 0.0, 0.0),
        },
        dict.fromkeys(answers, 1),
    )

    assignment = cluster_candidates(
        PatternHierarchyClusterSpec(cut=0.0), geometry=geometry, target=3, probabilities=answers
    )

    assert len(_grouping(assignment.labels)) == 1
    similarities = {
        round(geometry.similarity("trace-00", other), 6) for other in ("trace-01", "trace-02")
    }
    assert len(similarities) == 2
    assert geometry.medoid(tuple(geometry.ids)) == "trace-02"


def test_the_pattern_backend_ignores_input_order() -> None:
    """Patterns are ordered canonically, not by first appearance, for exactly this reason."""
    geometry, answers = _tagged("1100", "1110", "0011", "1100", "0000", "1010")
    shuffled = CoreSetGeometry.of(
        dict(reversed(list(geometry.vectors.items()))), dict(geometry.masses)
    )
    spec = PatternHierarchyClusterSpec(cut=0.45)

    assert (
        cluster_candidates(spec, geometry=geometry, target=6, probabilities=answers).labels
        == cluster_candidates(
            spec, geometry=shuffled, target=6, probabilities=dict(reversed(list(answers.items())))
        ).labels
    )


def test_the_pattern_backend_repeats_itself_exactly() -> None:
    """There is no seed on this spec because there is no random source, which is checked here."""
    geometry, answers = _tagged("1100", "1110", "0011", "1010", "0111", "1100", "0000")
    spec = PatternHierarchyClusterSpec(cut=0.5)

    answered = {
        tuple(
            sorted(
                cluster_candidates(
                    spec, geometry=geometry, target=4, probabilities=answers
                ).labels.items()
            )
        )
        for _ in range(5)
    }

    assert len(answered) == 1


def test_the_pattern_cut_and_threshold_stay_in_their_intervals() -> None:
    """The cut is a distance rather than a group count, so the interval is part of the contract."""
    assert PatternHierarchyClusterSpec().cut == 0.3
    assert PatternHierarchyClusterSpec().threshold == 0.5
    for cut in (-0.1, 1.1):
        with pytest.raises(ValidationError):
            PatternHierarchyClusterSpec(cut=cut)
    # A bound at either end would fire every proposition or none, so neither is a threshold.
    for threshold in (0.0, 1.0):
        with pytest.raises(ValidationError):
            PatternHierarchyClusterSpec(threshold=threshold)


def test_the_pattern_backend_publishes_strata_coarser_than_its_clusters() -> None:
    """Strata are what a sampler covers before spending inside one, so they have to be coarser."""
    geometry, answers = _tagged("1100", "1110", "0011", "0111", "0000")

    assignment = cluster_candidates(
        PatternHierarchyClusterSpec(cut=0.1, strata_cut=0.8),
        geometry=geometry,
        target=5,
        probabilities=answers,
    )

    assert assignment.strata is not None
    clusters = _grouping(assignment.labels)
    strata = _grouping(assignment.strata)
    assert len(strata) < len(clusters)
    # Each cluster sits inside exactly one stratum, which is what makes the pair a hierarchy of
    # two partitions rather than two unrelated groupings.
    assert all(any(cluster <= stratum for stratum in strata) for cluster in clusters)


def test_a_stratum_finer_than_its_cluster_is_refused() -> None:
    """One cluster spread across several strata is not a coarser grouping of anything."""
    with pytest.raises(ValidationError, match="at or above cut"):
        PatternHierarchyClusterSpec(cut=0.6, strata_cut=0.3)
    assert PatternHierarchyClusterSpec().strata_cut == 0.6
    assert PatternHierarchyClusterSpec(cut=0.6, strata_cut=0.6).strata_cut == 0.6


def test_the_pattern_backend_publishes_the_units_its_tree_is_indexed_by() -> None:
    """A tree is only descendable if something says which leaf a candidate sits under."""
    geometry, answers = _tagged("1100", "1100", "1110", "0011", "0000")

    assignment = cluster_candidates(
        PatternHierarchyClusterSpec(cut=0.3), geometry=geometry, target=5, probabilities=answers
    )

    assert assignment.units is not None
    assert assignment.hierarchy is not None
    # The unit is the fired-proposition set, so the two identical cases share one and the tree has
    # one leaf per distinct pattern.
    assert assignment.units["trace-00"] == assignment.units["trace-01"]
    assert len(set(assignment.units.values())) == 4
    leaves = {int(assignment.hierarchy.leaf_of_unit[unit]) for unit in assignment.units.values()}
    assert leaves <= set(assignment.hierarchy.leaves(assignment.hierarchy.root))


def test_the_atoms_backend_states_that_its_units_are_its_atoms() -> None:
    """Leaving a consumer to infer that the labels happen to be the units would be a trap."""
    geometry = _ring(16)

    assignment = cluster_candidates(
        AtomsHierarchyClusterSpec(atom_count=8, groups=3, seed=1), geometry=geometry, target=8
    )

    assert assignment.units is not None
    assert assignment.units == assignment.labels
    assert assignment.hierarchy is not None
    assert sorted(set(assignment.units.values())) == list(range(8))


@pytest.mark.parametrize(
    ("spec", "tagged"),
    [
        (DescriptorClusterSpec(), False),
        (KMeansClusterSpec(k=3, seed=3), False),
        (AtomsHierarchyClusterSpec(atom_count=6, groups=3, seed=3), False),
        (BernoulliMixtureClusterSpec(k=2, seed=3), True),
        (PatternHierarchyClusterSpec(), True),
    ],
)
def test_the_units_and_the_tree_are_published_together_or_not_at_all(
    spec: ClusterSpec, tagged: bool
) -> None:
    """One without the other names a mapping into nothing, or a tree nothing maps into."""
    geometry = _ring(12)
    answers = _probabilities(geometry) if tagged else None

    assignment = cluster_candidates(spec, geometry=geometry, target=6, probabilities=answers)

    assert (assignment.units is None) == (assignment.hierarchy is None)
    assert (assignment.units is None) == (assignment.strata is None)


@pytest.mark.parametrize(
    ("spec", "tagged"),
    [
        (DescriptorClusterSpec(), False),
        (KMeansClusterSpec(k=3, seed=3), False),
        (AtomsHierarchyClusterSpec(atom_count=6, groups=3, seed=3), False),
        (BernoulliMixtureClusterSpec(k=2, seed=3), True),
        (PatternHierarchyClusterSpec(), True),
    ],
)
def test_the_tree_predicate_answers_what_the_dispatch_actually_published(
    spec: ClusterSpec, tagged: bool
) -> None:
    """One owner for "does this backend publish a tree?", checked against the dispatch itself.

    Everything that has to know the answer before a fit runs, a spec validator or a CLI option
    table, reads the predicate, so the predicate disagreeing with what the dispatch fills would
    refuse a usable combination or accept an unusable one.
    """
    geometry = _ring(12)
    answers = _probabilities(geometry) if tagged else None

    assignment = cluster_candidates(spec, geometry=geometry, target=6, probabilities=answers)

    assert publishes_tree(spec) == (assignment.hierarchy is not None)
