"""Tests for weighted Ward linkage and the re-cuttable tree over atoms."""

from __future__ import annotations

import numpy as np
import pytest

from exp.simulation.mining.clustering.hierarchy import (
    Hierarchy,
    atoms_hierarchy_fit,
    weighted_ward,
)


def test_three_points_merge_in_the_hand_computed_order() -> None:
    """Ward is specified by its merge costs, so the first check is arithmetic, not behaviour."""
    centroids = np.array([[0.0], [1.0], [10.0]])
    sizes = np.array([1.0, 1.0, 1.0])

    parent, height = weighted_ward(centroids, sizes)

    # Nearest pair first: cost 1*1/(1+1) * 1^2 = 0.5.
    assert parent[0] == parent[1] == 3
    assert height[3] == pytest.approx(0.5)
    # Then the merged pair (weight 2, centre 0.5) against the far point, by Lance-Williams.
    assert parent[2] == parent[3] == 4
    assert height[4] == pytest.approx(2 * 1 / 3 * (10.0 - 0.5) ** 2, rel=0.2)
    assert parent[4] == -1


def test_size_weighting_reproduces_ward_over_the_traces_themselves() -> None:
    """The equivalence is the whole claim: a cut equals Ward with atoms kept whole."""
    centroids = np.array([[0.0], [2.0]])

    light = weighted_ward(centroids, np.array([1.0, 1.0]))[1][2]
    heavy = weighted_ward(centroids, np.array([10.0, 10.0]))[1][2]

    assert light == pytest.approx(1.0 / 2 * 4.0)
    assert heavy == pytest.approx(100.0 / 20 * 4.0)


def test_a_single_point_is_its_own_root() -> None:
    """A partition can collapse to one atom, and a tree over it still has to be well formed."""
    parent, height = weighted_ward(np.array([[1.0, 2.0]]), np.array([3.0]))

    assert parent.tolist() == [-1]
    assert height.tolist() == [0.0]
    assert Hierarchy(parent, np.array([0])).root == 0


def test_an_empty_or_mismatched_input_is_refused() -> None:
    """Silently inventing a weight would silently change every merge cost."""
    with pytest.raises(ValueError, match="at least one point"):
        weighted_ward(np.zeros((0, 2)), np.zeros(0))
    with pytest.raises(ValueError, match="one weight per point"):
        weighted_ward(np.zeros((3, 2)), np.ones(2))


def _linear_tree() -> Hierarchy:
    """Return the tree over four evenly spaced atoms."""
    centroids = np.array([[0.0], [1.0], [8.0], [9.0]])
    parent, height = weighted_ward(centroids, np.ones(4))
    return Hierarchy(parent, np.arange(4), height)


def test_one_fit_answers_several_group_counts() -> None:
    """Re-cutting without refitting is the reason to pay for a hierarchy at all."""
    tree = _linear_tree()

    assert len(set(tree.labels_at(np.arange(4), 1).tolist())) == 1
    assert tree.labels_at(np.arange(4), 2).tolist() == [0, 0, 1, 1]
    assert len(set(tree.labels_at(np.arange(4), 4).tolist())) == 4


def test_group_identities_are_contiguous_from_zero() -> None:
    """Cluster identities are persisted coverage evidence and must be nonnegative and dense."""
    labels = _linear_tree().labels_at(np.arange(4), 3)

    assert sorted(set(labels.tolist())) == list(range(len(set(labels.tolist()))))
    assert labels.min() >= 0


def test_a_cut_finer_than_the_tree_does_not_raise() -> None:
    """The reference implementation raises here, taking a minimum over an empty axis.

    Asking a two-atom tree for sixteen groups is an ordinary consequence of a spec written for a
    larger corpus meeting a small partition, so it has to degrade to the finest available
    partition rather than fail the whole build.
    """
    parent, height = weighted_ward(np.array([[0.0], [5.0]]), np.ones(2))
    tree = Hierarchy(parent, np.arange(2), height)

    labels = tree.labels_at(np.arange(2), 16)

    assert sorted(labels.tolist()) == [0, 1]


def test_a_path_runs_from_the_root_down_to_the_leaf() -> None:
    """Paths are how a reviewer reads which coarse group a case sits inside."""
    tree = _linear_tree()

    path = tree.path(0)

    assert path[0] == tree.root
    assert path[-1] == 0
    assert tree.leaves(tree.root) == [0, 1, 2, 3]


def _blobs() -> np.ndarray:
    """Return four well-separated pairs of points."""
    rng = np.random.default_rng(5)
    centres = np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 20.0], [20.0, 20.0]])
    return np.vstack([centre + rng.normal(scale=0.1, size=(3, 2)) for centre in centres])


def test_atoms_stay_finer_than_the_groups_above_them() -> None:
    """Atoms are the stable assignment unit; the groups are a view over them."""
    fit = atoms_hierarchy_fit(_blobs(), atom_count=8, groups=4, seed=1)

    assert len(set(fit.atoms.tolist())) >= len(set(fit.labels.tolist()))
    assert fit.labels.min() >= 0
    assert len(fit.group_of_atom) == len(fit.centroids)
    assert fit.sizes.sum() == len(_blobs())


def test_the_four_blobs_land_in_four_groups() -> None:
    """A hierarchy that cannot recover obvious structure is not worth cutting."""
    fit = atoms_hierarchy_fit(_blobs(), atom_count=8, groups=4, seed=1)

    grouping = {
        frozenset(np.flatnonzero(fit.labels == label).tolist())
        for label in set(fit.labels.tolist())
    }
    assert grouping == {frozenset(range(start, start + 3)) for start in (0, 3, 6, 9)}


def test_more_atoms_than_points_is_clamped() -> None:
    """A spec written for a big corpus must still fit a small partition."""
    fit = atoms_hierarchy_fit(_blobs(), atom_count=500, groups=3, seed=1)

    assert len(fit.centroids) == len(_blobs())


def test_an_unusable_request_is_refused() -> None:
    """A zero-atom hierarchy has nothing to cut."""
    with pytest.raises(ValueError, match="non-empty"):
        atoms_hierarchy_fit(np.zeros((0, 2)), atom_count=2, groups=1)
    with pytest.raises(ValueError, match="at least one atom"):
        atoms_hierarchy_fit(_blobs(), atom_count=0, groups=1)
