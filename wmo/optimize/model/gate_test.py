"""Tests for the distillation promotion gate: threshold, regression, tie epsilon."""

from __future__ import annotations

import pytest

from wmo.optimize.model.config import GateConfig
from wmo.optimize.model.gate import gate_distillation

# Half the gate's tie epsilon: a delta this small must count as a tie.
_HALF_EPS = 5e-10


def _spec(
    *,
    min_teacher_fraction: float = 0.7,
    require_no_regression: bool = True,
    k: int = 3,
) -> GateConfig:
    return GateConfig(
        k=k,
        min_teacher_fraction=min_teacher_fraction,
        require_no_regression=require_no_regression,
    )


def test_accepts_when_threshold_met_and_no_regression() -> None:
    record = gate_distillation(0.6, 0.3, 0.5, _spec())
    assert record.accepted
    assert record.teacher_solve_rate == 0.6
    assert record.student_before_solve_rate == 0.3
    assert record.student_after_solve_rate == 0.5
    assert record.min_teacher_fraction == 0.7


def test_rejects_below_teacher_fraction() -> None:
    record = gate_distillation(0.6, 0.3, 0.35, _spec())
    assert not record.accepted
    assert record.reason.startswith("rejected: ")
    # The reason states the compared numbers: after, fraction, teacher, threshold.
    assert "after 0.350" in record.reason
    assert "0.70 x teacher 0.600 = 0.420" in record.reason


def test_rejects_regression_even_above_teacher_threshold() -> None:
    record = gate_distillation(0.4, 0.5, 0.45, _spec())
    assert not record.accepted
    assert "after 0.450 regressed below before 0.500" in record.reason


def test_regression_check_can_be_disabled() -> None:
    spec = _spec(require_no_regression=False)
    record = gate_distillation(0.4, 0.5, 0.45, spec)
    assert record.accepted
    assert "regression check disabled" in record.reason


def test_both_failures_are_reported_together() -> None:
    record = gate_distillation(0.8, 0.5, 0.2, _spec())
    assert not record.accepted
    assert "below 0.70 x teacher 0.800" in record.reason
    assert "regressed below before 0.500" in record.reason


def test_threshold_tie_within_epsilon_passes() -> None:
    # Exactly at the threshold and a hair (half epsilon) under it both pass.
    spec = _spec(min_teacher_fraction=0.5, require_no_regression=False)
    assert gate_distillation(0.8, 0.0, 0.4, spec).accepted
    assert gate_distillation(0.8, 0.0, 0.4 - _HALF_EPS, spec).accepted


def test_threshold_miss_beyond_epsilon_fails() -> None:
    spec = _spec(min_teacher_fraction=0.5, require_no_regression=False)
    assert not gate_distillation(0.8, 0.0, 0.4 - 1e-6, spec).accepted


def test_regression_tie_within_epsilon_passes() -> None:
    spec = _spec(min_teacher_fraction=0.1)
    assert gate_distillation(0.2, 0.5, 0.5, spec).accepted
    assert gate_distillation(0.2, 0.5, 0.5 - _HALF_EPS, spec).accepted


def test_regression_beyond_epsilon_fails() -> None:
    spec = _spec(min_teacher_fraction=0.1)
    record = gate_distillation(0.2, 0.5, 0.5 - 1e-6, spec)
    assert not record.accepted
    assert "regressed below before 0.500" in record.reason


def test_zero_teacher_rate_accepts_any_after() -> None:
    # A zero teacher solve rate makes the threshold zero; the gate degrades
    # gracefully instead of dividing or rejecting everything.
    record = gate_distillation(0.0, 0.0, 0.0, _spec())
    assert record.accepted


def test_accepted_reason_states_the_numbers_and_attempts() -> None:
    record = gate_distillation(0.6, 0.3, 0.5, _spec(k=3))
    assert record.reason.startswith("accepted: ")
    assert "after 0.500" in record.reason
    assert "0.70 x teacher 0.600 = 0.420" in record.reason
    assert "before 0.300" in record.reason
    assert "k=3 attempts" in record.reason


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_out_of_range_rates_are_rejected_with_guidance(bad: float) -> None:
    with pytest.raises(ValueError, match="solve rate"):
        gate_distillation(bad, 0.5, 0.5, _spec())
    with pytest.raises(ValueError, match="student-after"):
        gate_distillation(0.5, 0.5, bad, _spec())
