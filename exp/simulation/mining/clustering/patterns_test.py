"""Tests for average-linkage Jaccard agglomeration over distinct fired-proposition sets."""

from __future__ import annotations

import numpy as np
import pytest

from exp.simulation.mining.clustering.patterns import (
    average_linkage,
    jaccard_distances,
    pattern_hierarchy_fit,
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
        assert np.all(np.diff(heights) >= -1e-12)


def test_average_linkage_breaks_a_tie_by_the_lowest_index_pair() -> None:
    """Indicator data is full of equal distances, and which merges first fixes the whole tree."""
    distances = np.full((4, 4), 0.5)
    np.fill_diagonal(distances, 0.0)

    merges, heights = average_linkage(distances, np.ones(4))

    assert merges.tolist() == [[0, 1], [0, 2], [0, 3]]
    assert heights == pytest.approx([0.5, 0.5, 0.5])


def test_average_linkage_refuses_evidence_it_cannot_read() -> None:
    """A mismatched weight vector is a caller error, not something to pad or truncate."""
    with pytest.raises(ValueError, match="square"):
        average_linkage(np.zeros((2, 3)), np.ones(2))
    with pytest.raises(ValueError, match="one weight per point"):
        average_linkage(np.zeros((3, 3)), np.ones(2))
    with pytest.raises(ValueError, match="positive weight"):
        average_linkage(np.zeros((2, 2)), np.array([1.0, 0.0]))


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
