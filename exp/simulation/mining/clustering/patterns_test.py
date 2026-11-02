"""Tests for average-linkage Jaccard agglomeration over distinct fired-proposition sets."""

from __future__ import annotations

import numpy as np
import pytest

from exp.simulation.mining.clustering.patterns import (
    average_linkage,
    jaccard_distances,
    pattern_hierarchy_fit,
    pattern_labels_at,
    pattern_tree,
)


def _rows(*patterns: str) -> np.ndarray:
    """Build an indicator matrix from bit strings, one row per string."""
    return np.array([[float(bit) for bit in pattern] for pattern in patterns])


def _partition(labels: np.ndarray) -> set[frozenset[int]]:
    """Return the grouping a label vector describes, free of the identities it used."""
    groups: dict[int, set[int]] = {}
    for position, label in enumerate(labels.tolist()):
        groups.setdefault(int(label), set()).add(position)
    return {frozenset(members) for members in groups.values()}


def test_jaccard_distance_is_one_minus_the_overlap_share() -> None:
    """The distance is the definition, so it is checked against hand-computed overlaps."""
    distances = jaccard_distances(_rows("1100", "1110", "0011"))

    assert distances[0, 0] == pytest.approx(0.0)
    assert distances[0, 1] == pytest.approx(1.0 - 2.0 / 3.0)
    assert distances[0, 2] == pytest.approx(1.0)
    assert distances[1, 2] == pytest.approx(1.0 - 1.0 / 4.0)
    assert distances == pytest.approx(distances.T)


def test_jaccard_distance_refuses_an_empty_set() -> None:
    """Between two empty sets the ratio is 0/0, and no substitute value is defensible."""
    with pytest.raises(ValueError, match="undefined"):
        jaccard_distances(_rows("1100", "0000"))


def test_average_linkage_merges_the_closest_pair_and_records_its_height() -> None:
    """The merge order and the heights are what a cut reads, so both are pinned."""
    distances = np.array([[0.0, 0.2, 0.9], [0.2, 0.0, 0.8], [0.9, 0.8, 0.0]])

    merges, heights = average_linkage(distances, np.ones(3))

    assert merges.tolist() == [[0, 1], [0, 2]]
    assert heights == pytest.approx([0.2, (0.9 + 0.8) / 2])


def test_average_linkage_weights_the_update_by_multiplicity() -> None:
    """Weighting is what makes a run over patterns equal a run over every candidate."""
    distances = np.array([[0.0, 0.1, 0.7], [0.1, 0.0, 0.3], [0.7, 0.3, 0.0]])

    weighted = average_linkage(distances, np.array([3.0, 1.0, 1.0]))[1]
    unweighted = average_linkage(distances, np.ones(3))[1]

    assert weighted == pytest.approx([0.1, (3 * 0.7 + 1 * 0.3) / 4])
    assert unweighted == pytest.approx([0.1, (0.7 + 0.3) / 2])


def test_average_linkage_never_inverts() -> None:
    """A cut at a height is only a partition if the heights it reads are non-decreasing."""
    generator = np.random.default_rng(7)
    for _ in range(20):
        patterns = (generator.random((9, 12)) > 0.5).astype(float)
        patterns[:, 0] = 1.0
        _, heights = average_linkage(jaccard_distances(patterns), np.ones(9))
        assert np.all(np.diff(heights) >= 0.0)


def test_a_weighted_update_that_rounds_below_its_own_merge_is_clamped() -> None:
    """The published heights have to be non-decreasing in float64, not only in exact arithmetic.

    Average linkage cannot invert mathematically: an updated distance is the weighted mean of two
    distances that were both at or above the merge height, so it is too. In float64 that mean can
    round below them. Three points at exactly 1/3 with multiplicities 2 and 7 is such a case: the
    second merge measures ``(2/3 + 7/3) / 9``, which is one ulp under 1/3, and without the clamp a
    cut at 1/3 would stop in front of a merge whose exact height is the cut.
    """
    third = 1.0 / 3.0
    distances = np.array(
        [
            [0.0, third, third, 0.9],
            [third, 0.0, third, 0.9],
            [third, third, 0.0, 0.9],
            [0.9, 0.9, 0.9, 0.0],
        ]
    )

    merges, heights = average_linkage(distances, np.array([2.0, 7.0, 1.0, 1.0]))

    assert (2.0 * third + 7.0 * third) / 9.0 < third
    assert merges.tolist() == [[0, 1], [0, 2], [0, 3]]
    assert heights.tolist() == [third, third, 0.9]


def test_merge_heights_stay_non_decreasing_across_many_random_corpora() -> None:
    """The invariant a cut's prefix rule rests on is a property of every fit, not of one corpus.

    Multiplicity is what exposes it. A weighted mean of two equal distances can round below them,
    and a corpus with duplicate rows is the only way to get weights other than one, so the sweep
    draws candidate rows rather than distinct patterns.
    """
    generator = np.random.default_rng(34)
    for _ in range(200):
        rows = (
            generator.random((int(generator.integers(6, 30)), int(generator.integers(3, 9))))
            < generator.uniform(0.3, 0.8)
        ).astype(float)
        fit = pattern_hierarchy_fit(rows, cut=0.3)

        assert np.all(np.diff(fit.heights) >= 0.0)


def test_a_merge_at_exactly_the_cut_height_is_applied() -> None:
    """The cut's contract is every merge at or below it, so the boundary is what has to hold.

    Eight fired propositions and nine sharing seven of them is a union of ten, so the pair sits at
    exactly 3/10 and has to merge at the default cut of 0.3. Reading the distance as
    ``1 - 7 / 10`` lands an ulp above 0.3 and leaves the two apart, which is the common case rather
    than a corner: these are small integer counts and the cuts a run names are round numbers.
    """
    tenths = _rows("1111111100", "1111111011")
    thirds = _rows("1000", "1110")

    assert jaccard_distances(tenths)[0, 1] == 0.3
    assert jaccard_distances(thirds)[0, 1] == 2.0 / 3.0
    assert _partition(pattern_hierarchy_fit(tenths, cut=0.3).labels) == {frozenset({0, 1})}
    assert _partition(pattern_hierarchy_fit(thirds, cut=2.0 / 3.0).labels) == {frozenset({0, 1})}


def test_a_merge_above_the_cut_is_still_refused() -> None:
    """Reading a height at the cut as at the cut must not widen into reading any height as at it."""
    thirds = _rows("1000", "1110")

    assert _partition(pattern_hierarchy_fit(thirds, cut=0.66).labels) == {
        frozenset({0}),
        frozenset({1}),
    }
    assert _partition(pattern_hierarchy_fit(_rows("1100", "1110"), cut=0.0).labels) == {
        frozenset({0}),
        frozenset({1}),
    }


def test_average_linkage_breaks_a_tie_by_the_lowest_index_pair() -> None:
    """Indicator data is full of equal distances, and which merges first fixes the whole tree."""
    distances = np.full((4, 4), 0.5)
    np.fill_diagonal(distances, 0.0)

    merges, heights = average_linkage(distances, np.ones(4))

    assert merges.tolist() == [[0, 1], [0, 2], [0, 3]]
    assert heights == pytest.approx([0.5, 0.5, 0.5])


def _exhaustive_linkage(
    distances: np.ndarray, weights: np.ndarray
) -> tuple[list[list[int]], list[float]]:
    """Agglomerate while picking each merge by scanning the whole upper triangle.

    This is the tie-break's definition: the lowest index pair among every pair at the lowest live
    cost. The shipped search finds the same pair with two array reductions because listing the ties
    costs Python time proportional to how many there are, so a disagreement is the fast search's
    bug rather than a question of which rule is right.
    """
    cost = np.asarray(distances, dtype=np.float64).copy()
    mass = np.asarray(weights, dtype=np.float64).copy()
    np.fill_diagonal(cost, np.inf)
    merges: list[list[int]] = []
    heights: list[float] = []
    for _ in range(cost.shape[0] - 1):
        lowest = cost.min()
        left, right = min(
            (row, column)
            for row in range(cost.shape[0])
            for column in range(row + 1, cost.shape[0])
            if cost[row, column] == lowest
        )
        merges.append([left, right])
        heights.append(max(float(cost[left, right]), heights[-1] if heights else -np.inf))
        updated = (mass[left] * cost[left] + mass[right] * cost[right]) / (mass[left] + mass[right])
        cost[left, :] = cost[:, left] = updated
        cost[left, left] = np.inf
        cost[right, :] = cost[:, right] = np.inf
        mass[left] += mass[right]
    return merges, heights


def test_the_closest_pair_search_agrees_with_an_exhaustive_scan() -> None:
    """Ties are most of a binary corpus, so the cheap search has to break them identically."""
    generator = np.random.default_rng(99)
    for trial in range(120):
        count = int(generator.integers(2, 12))
        if trial % 3 == 0:
            distances = np.full((count, count), 0.5)
        else:
            coarse = np.round(generator.random((count, count)), 1)
            distances = np.minimum(coarse, coarse.T)
        np.fill_diagonal(distances, 0.0)
        weights = generator.integers(1, 6, count).astype(float)

        merges, heights = average_linkage(distances, weights)
        wanted_merges, wanted_heights = _exhaustive_linkage(distances, weights)

        assert merges.tolist() == wanted_merges
        assert heights.tolist() == wanted_heights


def test_average_linkage_refuses_evidence_it_cannot_read() -> None:
    """A mismatched weight vector is a caller error, not something to pad or truncate."""
    with pytest.raises(ValueError, match="square"):
        average_linkage(np.zeros((2, 3)), np.ones(2))
    with pytest.raises(ValueError, match="one weight per point"):
        average_linkage(np.zeros((3, 3)), np.ones(2))
    with pytest.raises(ValueError, match="positive weight"):
        average_linkage(np.zeros((2, 2)), np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="symmetric"):
        average_linkage(np.array([[0.0, 0.2], [0.3, 0.0]]), np.ones(2))


def test_a_cut_of_zero_groups_only_identical_fired_sets() -> None:
    """Distinct sets are never at distance zero, so the finest cut is one cluster per pattern."""
    fit = pattern_hierarchy_fit(_rows("1100", "1100", "1110", "0011"), cut=0.0)

    assert _partition(fit.labels) == {frozenset({0, 1}), frozenset({2}), frozenset({3})}
    assert fit.weights.tolist() == [1.0, 2.0, 1.0]


def test_raising_the_cut_rolls_near_identical_patterns_together() -> None:
    """A re-cuttable gradient is the reason for average linkage, so the gradient is checked."""
    rows = _rows("1100", "1110", "0011")

    assert _partition(pattern_hierarchy_fit(rows, cut=0.2).labels) == {
        frozenset({0}),
        frozenset({1}),
        frozenset({2}),
    }
    assert _partition(pattern_hierarchy_fit(rows, cut=0.4).labels) == {
        frozenset({0, 1}),
        frozenset({2}),
    }
    assert _partition(pattern_hierarchy_fit(rows, cut=1.0).labels) == {frozenset({0, 1, 2})}


def test_every_case_that_fired_nothing_shares_one_cluster() -> None:
    """These are the least distinctive cases, so isolating each one would be exactly backwards."""
    for cut in (0.0, 0.5, 1.0):
        fit = pattern_hierarchy_fit(_rows("0000", "1100", "0000", "1110", "0000"), cut=cut)
        empty = {int(fit.labels[position]) for position in (0, 2, 4)}

        assert len(empty) == 1
        assert fit.empty_pattern is not None
        assert empty.isdisjoint({int(fit.labels[1]), int(fit.labels[3])})
        assert not fit.merges.tolist() or fit.empty_pattern not in set(fit.merges.reshape(-1))


def test_a_corpus_that_fired_nothing_at_all_is_one_cluster() -> None:
    """With no set to overlap there is nothing to agglomerate, which is an answer not a failure."""
    fit = pattern_hierarchy_fit(_rows("000", "000"), cut=0.3)

    assert fit.labels.tolist() == [0, 0]
    assert fit.merges.shape == (0, 2)


def test_patterns_are_ordered_canonically_rather_than_by_first_appearance() -> None:
    """Input-order independence is a tested invariant of the whole pipeline, down to this order."""
    first = pattern_hierarchy_fit(_rows("1110", "0011", "1100"), cut=0.0)
    second = pattern_hierarchy_fit(_rows("1100", "1110", "0011"), cut=0.0)

    assert first.patterns.tolist() == second.patterns.tolist()


def test_the_grouping_does_not_depend_on_row_order() -> None:
    """Rows arrive in whatever order a corpus was read in, which cannot move the partition."""
    rows = _rows("1100", "1110", "0011", "1100", "0000", "1010")
    order = [4, 0, 5, 2, 1, 3]

    direct = pattern_hierarchy_fit(rows, cut=0.45).labels.tolist()
    shuffled = pattern_hierarchy_fit(rows[order], cut=0.45).labels.tolist()

    assert _partition(np.array(direct)) == {
        frozenset(order[position] for position in group) for group in _partition(np.array(shuffled))
    }


def test_cluster_identities_are_numbered_by_their_lowest_row() -> None:
    """A cluster identity is persisted coverage evidence, so the numbering cannot be arbitrary."""
    fit = pattern_hierarchy_fit(_rows("0011", "1100", "0011", "1110"), cut=0.4)

    assert fit.labels.tolist() == [0, 1, 0, 1]


def test_the_same_input_gives_the_same_answer_every_run() -> None:
    """There is no random source here, and a repeated run must prove there is none."""
    rows = _rows("1100", "1110", "0011", "1010", "0111", "1100", "0000")
    answers = {tuple(pattern_hierarchy_fit(rows, cut=0.5).labels.tolist()) for _ in range(5)}

    assert len(answers) == 1


def test_agglomerating_over_patterns_equals_agglomerating_over_every_row() -> None:
    """The whole reason to collapse first is that it changes nothing, so that is checked."""
    distinct = _rows("1100", "1110", "0011", "1010")
    expanded = np.repeat(distinct, (3, 1, 2, 1), axis=0)

    collapsed = average_linkage(jaccard_distances(distinct), np.array([3.0, 1.0, 2.0, 1.0]))
    raw = average_linkage(jaccard_distances(expanded), np.ones(len(expanded)))

    assert collapsed[1] == pytest.approx([height for height in raw[1].tolist() if height > 0.0])


def test_a_matrix_that_is_not_membership_is_refused() -> None:
    """Thresholding here would invent this backend's one input, so it refuses instead."""
    with pytest.raises(ValueError, match="exactly 0.0 or 1.0"):
        pattern_hierarchy_fit(np.array([[0.4, 0.9], [1.0, 0.0]]), cut=0.3)
    with pytest.raises(ValueError, match="non-empty two-dimensional"):
        pattern_hierarchy_fit(np.zeros((0, 3)), cut=0.3)
    with pytest.raises(ValueError, match=r"distance in \[0, 1\]"):
        pattern_hierarchy_fit(_rows("10"), cut=1.5)


def test_re_cutting_a_fit_matches_a_fresh_fit_at_that_height() -> None:
    """A fit answers every height, so re-cutting it cannot disagree with fitting it again."""
    rows = _rows("1100", "1110", "0011", "1010", "0111", "1100", "0000")
    fit = pattern_hierarchy_fit(rows, cut=0.0)

    for cut in (0.0, 0.2, 0.45, 0.6, 0.875, 1.0):
        assert (
            pattern_labels_at(fit, cut).tolist()
            == pattern_hierarchy_fit(rows, cut=cut).labels.tolist()
        )


def test_re_cutting_refuses_a_height_that_is_not_a_jaccard_distance() -> None:
    """The interval is the contract, and a cut outside it would silently become 0.0 or 1.0."""
    fit = pattern_hierarchy_fit(_rows("1100", "0011"), cut=0.3)

    for cut in (-0.1, 1.5):
        with pytest.raises(ValueError, match=r"distance in \[0, 1\]"):
            pattern_labels_at(fit, cut)


def test_the_published_tree_is_the_agglomeration_it_was_built_from() -> None:
    """The tree is checked against a hand-worked dendrogram rather than against itself.

    Four rows over three distinct non-empty patterns plus the all-zero one. Canonical ascending
    bit order makes the patterns ``0000``, ``0011``, ``1100`` and ``1110``, so the agglomeration
    runs over patterns 1, 2 and 3. ``1100`` and ``1110`` share two of the three propositions
    either mentions and merge first at ``1 - 2/3``; the pair then stands at ``(1.0 + 0.75) / 2``
    from ``0011``, which is the second merge.
    """
    fit = pattern_hierarchy_fit(_rows("1100", "1110", "0011", "0000"), cut=0.3)

    tree = pattern_tree(fit)

    assert fit.merges.tolist() == [[2, 3], [1, 2]]
    assert fit.heights == pytest.approx([1.0 - 2.0 / 3.0, 0.875])
    # Leaves 0 to 3 are the patterns, node 4 is the first merge, node 5 the second, and node 6 the
    # synthetic root joining the second merge to the held-out all-zero leaf 0.
    assert tree.parent.tolist() == [6, 5, 4, 4, 5, 6, -1]
    assert tree.height == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0 - 2.0 / 3.0, 0.875, 1.0])
    assert tree.root == 6
    assert tree.leaf_of_unit.tolist() == [0, 1, 2, 3]


def test_the_held_out_all_zero_leaf_is_joined_at_the_maximum_distance() -> None:
    """It shares nothing with anything, so a cut has to separate it before anything else."""
    fit = pattern_hierarchy_fit(_rows("1100", "1110", "0011", "0000"), cut=0.3)

    tree = pattern_tree(fit)
    empty_leaf = fit.empty_pattern

    assert empty_leaf is not None
    assert tree.height is not None
    assert tree.height[tree.root] == pytest.approx(1.0)
    assert tree.parent[empty_leaf] == tree.root
    assert tree.children(tree.root).tolist() == [empty_leaf, 5]
    assert tree.cut(2).tolist() == [0, 5, 5, 5]


def test_a_leaf_of_the_tree_is_the_pattern_it_was_built_for() -> None:
    """The finest assignment unit of this backend is the pattern, not the candidate row."""
    fit = pattern_hierarchy_fit(_rows("1100", "1110", "0011", "1100", "0000"), cut=0.3)

    tree = pattern_tree(fit)

    assert tree.leaf_of_unit.tolist() == list(range(fit.patterns.shape[0]))
    assert [tree.leaves(leaf) for leaf in tree.leaf_of_unit.tolist()] == [
        [leaf] for leaf in range(fit.patterns.shape[0])
    ]


def test_cutting_the_published_tree_expands_the_highest_merge_first() -> None:
    """A cut that expanded the largest subtree instead would describe a different dendrogram.

    Two well-separated pairs, one tight and one loose: ``1111000000`` with ``1111100000`` at 0.2
    and ``0000011100`` with ``0000001110`` at 0.5. Three groups therefore has to split the loose
    pair and leave the tight one whole, which is the opposite of what expanding the first or the
    largest expandable node would do, since the two subtrees are the same size and the tight pair
    comes first.
    """
    fit = pattern_hierarchy_fit(
        _rows("1111000000", "1111100000", "0000011100", "0000001110"), cut=0.3
    )

    tree = pattern_tree(fit)

    assert fit.heights == pytest.approx([0.2, 0.5, 1.0])
    assert tree.parent.tolist() == [5, 5, 4, 4, 6, 6, -1]
    assert tree.cut(2).tolist() == [5, 5, 4, 4]
    assert tree.cut(3).tolist() == [0, 1, 4, 4]
    assert tree.cut(4).tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("rows", "leaves"),
    [
        (("1100", "1110", "0011"), 3),
        (("0000", "0000"), 1),
        (("1100", "1100"), 1),
        (("1010",), 1),
        (("0000", "1100"), 2),
        (("0000", "1100", "1110"), 3),
    ],
)
def test_every_degenerate_fit_still_publishes_one_root(rows: tuple[str, ...], leaves: int) -> None:
    """A forest would make the tree unusable, and these are the shapes that produce one.

    No all-zero pattern, nothing but all-zero rows, one pattern overall, an all-zero pattern with
    a single other pattern and therefore no merges, and an agglomeration of two beside the
    all-zero leaf.
    """
    fit = pattern_hierarchy_fit(_rows(*rows), cut=0.3)

    tree = pattern_tree(fit)

    assert fit.patterns.shape[0] == leaves
    assert tree.leaf_of_unit.tolist() == list(range(leaves))
    assert int(tree.root) == len(tree.parent) - 1
    assert sorted(tree.leaves(tree.root)) == list(range(leaves))
    assert len(set(tree.cut(leaves).tolist())) == leaves


def test_the_tree_is_the_same_whatever_order_the_rows_arrived_in() -> None:
    """The patterns are canonically ordered, so the tree over them cannot move either."""
    rows = _rows("1100", "1110", "0011", "1100", "0000", "1010")
    order = [4, 0, 5, 2, 1, 3]

    direct = pattern_tree(pattern_hierarchy_fit(rows, cut=0.3))
    shuffled = pattern_tree(pattern_hierarchy_fit(rows[order], cut=0.3))

    assert direct.height is not None and shuffled.height is not None
    assert direct.parent.tolist() == shuffled.parent.tolist()
    assert direct.height.tolist() == shuffled.height.tolist()
    assert direct.leaf_of_unit.tolist() == shuffled.leaf_of_unit.tolist()
