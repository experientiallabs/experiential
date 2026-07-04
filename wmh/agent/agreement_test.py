"""Tests for the pure sim-real agreement metric (confusion, outcome agreement, rank correlation)."""

from __future__ import annotations

import pytest

from wmh.agent.agreement import _ranks, _spearman, compute_agreement
from wmh.agent.closed_loop import ClosedLoopReport, TaskOutcome


def _report(name: str, per_task: dict[str, float]) -> ClosedLoopReport:
    outcomes = {
        tid: TaskOutcome(task_id=tid, success_rate=r, passes=3) for tid, r in per_task.items()
    }
    rates = list(per_task.values())
    return ClosedLoopReport(
        harness=name,
        success_rate=sum(rates) / len(rates) if rates else 0.0,
        per_task=outcomes,
    )


def test_perfect_agreement() -> None:
    sim = [_report("a", {"t1": 1.0, "t2": 0.0}), _report("b", {"t1": 1.0, "t2": 1.0})]
    real = [_report("a", {"t1": 1.0, "t2": 0.0}), _report("b", {"t1": 1.0, "t2": 1.0})]
    rep = compute_agreement(sim, real, k=3)
    assert rep.outcome_agreement == 1.0
    assert rep.confusion.total == 4
    assert rep.confusion.sim_pass_real_fail == 0
    assert rep.mean_abs_gap == 0.0
    # a<b in both worlds -> perfect positive rank correlation.
    assert rep.rank_correlation == pytest.approx(1.0)


def test_sim_over_credits_is_the_dangerous_cell() -> None:
    # Sim says variant passes t2; reality says it fails. That is the mirage evolution would chase.
    sim = [_report("a", {"t1": 1.0, "t2": 1.0})]
    real = [_report("a", {"t1": 1.0, "t2": 0.0})]
    rep = compute_agreement(sim, real, k=3)
    assert rep.confusion.sim_pass_real_fail == 1
    assert rep.confusion.sim_pass_real_pass == 1
    assert rep.outcome_agreement == 0.5


def test_rank_correlation_detects_inverted_ranking() -> None:
    # Sim ranks a>b>c; reality ranks a<b<c. Rank correlation should be strongly negative.
    sim = [_report("a", {"t": 0.9}), _report("b", {"t": 0.5}), _report("c", {"t": 0.1})]
    real = [_report("a", {"t": 0.1}), _report("b", {"t": 0.5}), _report("c", {"t": 0.9})]
    rep = compute_agreement(sim, real, k=3)
    assert rep.rank_correlation == pytest.approx(-1.0)


def test_unmatched_variants_and_tasks_are_skipped() -> None:
    sim = [_report("a", {"t1": 1.0, "extra": 1.0}), _report("only_sim", {"t1": 1.0})]
    real = [_report("a", {"t1": 1.0})]  # no 'extra' task, no 'only_sim' variant
    rep = compute_agreement(sim, real, k=3)
    # Only variant 'a', only task 't1' is shared -> one cell.
    assert [v.harness for v in rep.per_variant] == ["a"]
    assert rep.confusion.total == 1


def test_pass_threshold_binarizes() -> None:
    sim = [_report("a", {"t": 0.66})]  # 2/3 passes
    real = [_report("a", {"t": 0.33})]  # 1/3 passes
    # At threshold 0.5: sim passes, real fails.
    rep = compute_agreement(sim, real, k=3, pass_threshold=0.5)
    assert rep.confusion.sim_pass_real_fail == 1


def test_spearman_none_when_constant_or_singleton() -> None:
    assert _spearman([1.0], [1.0]) is None  # too few points
    assert _spearman([0.5, 0.5, 0.5], [0.1, 0.2, 0.3]) is None  # constant x -> undefined


def test_ranks_average_ties() -> None:
    # Values [10, 10, 20] -> the two 10s share rank (0+1)/2 = 0.5; 20 gets rank 2.
    assert _ranks([10.0, 10.0, 20.0]) == [0.5, 0.5, 2.0]
