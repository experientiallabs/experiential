"""Tests for the report-vs-report agreement metric (pure, offline)."""

from __future__ import annotations

from wmh.evals.agreement import AgreementReport, compute_agreement
from wmh.evals.closed_loop import ClosedLoopReport, TaskOutcome


def _report(label: str, per_task: dict[str, float], k: int = 3) -> ClosedLoopReport:
    outcomes = {
        tid: TaskOutcome(task_id=tid, success_rate=r, passes=k) for tid, r in per_task.items()
    }
    rates = list(per_task.values())
    return ClosedLoopReport(
        label=label,
        success_rate=sum(rates) / len(rates) if rates else 0.0,
        k=k,
        per_task=outcomes,
    )


def test_perfect_agreement() -> None:
    a = _report("sim", {"t1": 1.0, "t2": 0.0})
    b = _report("real", {"t1": 1.0, "t2": 0.0})
    result = compute_agreement(a, b)
    assert result.outcome_agreement == 1.0
    assert result.confusion.total == 2
    assert result.success_gap == 0.0


def test_over_optimism_lands_in_the_dangerous_cell() -> None:
    # Report A (e.g. the world model) says t2 passes; report B (reality) says it fails.
    a = _report("sim", {"t1": 1.0, "t2": 1.0})
    b = _report("real", {"t1": 1.0, "t2": 0.0})
    result = compute_agreement(a, b)
    assert result.confusion.a_pass_b_fail == 1
    assert result.outcome_agreement == 0.5
    assert result.success_gap == 0.5  # sim over-credits by half


def test_unshared_tasks_are_skipped_and_zero_cells_is_none() -> None:
    a = _report("sim", {"only_in_a": 1.0})
    b = _report("real", {"only_in_b": 1.0})
    result = compute_agreement(a, b)
    assert result.confusion.total == 0
    # None, not 0.0: "no data" must not read as "total disagreement".
    assert result.outcome_agreement is None
    assert "n/a" in result.summary()


def test_pass_threshold_binarizes() -> None:
    a = _report("sim", {"t": 2 / 3})  # 2 of 3 passes
    b = _report("real", {"t": 1 / 3})  # 1 of 3 passes
    result = compute_agreement(a, b, pass_threshold=0.5)
    assert result.confusion.a_pass_b_fail == 1


def test_report_json_roundtrips() -> None:
    # The `--out` artifact and agreement inputs must reload cleanly (incl. None agreement).
    result = compute_agreement(_report("sim", {}), _report("real", {}))
    assert AgreementReport.model_validate_json(result.model_dump_json()) == result
