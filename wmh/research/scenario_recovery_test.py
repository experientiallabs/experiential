"""Tests for task-identity recovery metrics (purity, ARI, label extraction)."""

from __future__ import annotations

import pytest

from wmh.core.types import Trace
from wmh.research.scenario_recovery import ground_truth_labels, recovery_report


def test_perfect_clustering_scores_one() -> None:
    report = recovery_report([0, 0, 1, 1], ["a", "a", "b", "b"])
    assert report.purity == 1.0
    assert report.adjusted_rand_index == pytest.approx(1.0)


def test_label_permutation_does_not_matter() -> None:
    report = recovery_report([5, 5, 2, 2], ["a", "a", "b", "b"])
    assert report.adjusted_rand_index == pytest.approx(1.0)


def test_single_cluster_over_mixed_labels_scores_low() -> None:
    report = recovery_report([0, 0, 0, 0], ["a", "a", "b", "b"])
    assert report.purity == 0.5
    assert report.adjusted_rand_index == pytest.approx(0.0)


def test_known_ari_values() -> None:
    # One intruder in an otherwise clean 2-cluster split: hand-computed ARI = 1.2 / 3.7.
    report = recovery_report([0, 0, 0, 1, 1, 1], ["a", "a", "a", "b", "b", "a"])
    assert report.adjusted_rand_index == pytest.approx(1.2 / 3.7)
    assert report.purity == pytest.approx(5 / 6)
    # Symmetric disagreement (each cluster 2:1 mixed): hand-computed ARI = -1/9. Negative ARI
    # (worse than chance) is valid and must come through un-clamped.
    worse = recovery_report([0, 0, 0, 1, 1, 1], ["a", "a", "b", "b", "b", "a"])
    assert worse.adjusted_rand_index == pytest.approx(-1 / 9)


def test_recovery_report_rejects_misaligned_or_empty() -> None:
    with pytest.raises(ValueError, match="labels"):
        recovery_report([0], ["a", "b"])
    with pytest.raises(ValueError, match="empty"):
        recovery_report([], [])


def test_ground_truth_labels_from_metadata() -> None:
    traces = [
        Trace(trace_id="t1", metadata={"domain": "airline", "task_id": "3"}),
        Trace(trace_id="t2", metadata={"domain": "retail", "task_id": 7}),
        Trace(trace_id="t3"),
    ]
    assert ground_truth_labels(traces) == ["airline/3", "retail/7", "unknown"]
