"""One cumulative phase ledger: never reset, refuses what does not fit, records usage."""

from __future__ import annotations

from pathlib import Path

from exp.optimize.evaluation.variance.frontdoor_ledger import Measure, PhaseLedger


def test_the_ledger_is_cumulative_across_runs_and_refuses_before_a_phase(tmp_path: Path) -> None:
    """A reopened ledger keeps every phase; a bound over what is left is refused."""
    path = tmp_path / "ledger.json"
    ledger = PhaseLedger(path, ceiling_usd=1.0, unpriced_attempts=100)
    assert ledger.admit("a", 0.4, 60) is None
    ledger.settle("a", Measure(attempts=10, usd=0.0), Measure(attempts=40, usd=0.25))
    reopened = PhaseLedger(path, ceiling_usd=1.0, unpriced_attempts=100)
    assert reopened.remaining_usd == 0.75 and reopened.remaining_attempts == 70
    reason = reopened.admit("b", 0.9, 0)
    assert reason is not None and "cumulative ceiling" in reason
    assert reopened.admit("b", 0.1, 80) is not None
    assert reopened.admit("b", 0.1, 70) is None
    assert [row.phase for row in PhaseLedger(path).state.phases] == ["a", "b"]


def test_a_running_phase_may_spend_its_reservation_plus_what_is_left(tmp_path: Path) -> None:
    """The live allowance a gated pipeline reads is its own reservation plus the free balance."""
    ledger = PhaseLedger(tmp_path / "ledger.json", ceiling_usd=1.0, unpriced_attempts=100)
    assert ledger.admit("pipeline a", 0.0, 30) is None
    assert ledger.remaining_attempts == 70
    assert ledger.allowance("pipeline a") == 100
    assert ledger.admit("pipeline b", 0.0, 50) is None
    assert ledger.allowance("pipeline b") == 70 and ledger.allowance("pipeline a") == 50
