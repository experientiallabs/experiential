"""Tests for clustering spec dispatch and validation."""

from __future__ import annotations

import math

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
    cluster_candidates,
)
from exp.simulation.mining.geometry import CoreSetGeometry

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
    ):
        restored = _ADAPTER.validate_python(spec.model_dump(mode="json"))
        assert restored == spec
        assert type(restored) is type(spec)


def test_an_unknown_backend_fails_closed() -> None:
    """A spec from a newer revision must not quietly fall back to the default grouping."""
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python({"kind": "spectral", "k": 4})
