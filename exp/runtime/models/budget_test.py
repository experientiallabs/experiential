"""Parallel spend admission, crash reservations and exact request replay."""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from exp.runtime.models.budget import RequestBudget, SpendLimitReached


def _call(budget: RequestBudget, *, key: str, calls: list[str], maximum: float = 1) -> str:
    """Run a deterministic charged request through the production ledger."""
    with budget.scope(key):

        def operation() -> str:
            """Record only new provider dispatches."""
            calls.append(key)
            return key

        return budget.call(
            role="assistant",
            fingerprint=key,
            maximum_cost_usd=maximum,
            operation=operation,
            encode=str,
            decode=str,
            charge=lambda result: maximum,
        )


def test_increase_limit_replays_saved_calls_without_new_spend(tmp_path: Path) -> None:
    """Budget pauses are before dispatch, and a later process can continue with more allowance."""
    calls: list[str] = []
    budget = RequestBudget(tmp_path, identity="fixture", maximum_cost_usd=1)
    assert _call(budget, key="first", calls=calls) == "first"
    with pytest.raises(SpendLimitReached) as paused:
        _call(budget, key="second", calls=calls)
    assert paused.value.required_usd == 2
    assert budget.accounted_usd == 1
    resumed = RequestBudget(tmp_path, identity="fixture", maximum_cost_usd=2)
    assert _call(resumed, key="first", calls=calls) == "first"
    assert _call(resumed, key="second", calls=calls) == "second"
    assert calls == ["first", "second"]
    assert resumed.accounted_usd == 2


def test_parallel_reservations_share_one_allowance(tmp_path: Path) -> None:
    """Concurrent cells cannot each independently spend the entire approved run budget."""
    budget = RequestBudget(tmp_path, identity="fixture", maximum_cost_usd=3)
    calls: list[str] = []
    start = threading.Barrier(8)

    def worker(index: int) -> bool:
        """Compete for three available dollars from eight independent cells."""
        start.wait(timeout=10)
        try:
            _call(budget, key=str(index), calls=calls)
            return True
        except SpendLimitReached:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(worker, range(8))) == 3
    assert len(calls) == 3
    assert budget.accounted_usd == 3


def test_unknown_dispatch_retains_its_reservation_and_never_replays(tmp_path: Path) -> None:
    """A provider failure is not evidence that a paid request was free."""
    budget = RequestBudget(tmp_path, identity="fixture", maximum_cost_usd=2)

    def fail() -> str:
        """Simulate loss of a provider response after dispatch."""
        raise TimeoutError

    with budget.scope("cell"), pytest.raises(TimeoutError):
        budget.call(
            role="assistant",
            fingerprint="request",
            maximum_cost_usd=1,
            operation=fail,
            encode=str,
            decode=str,
            charge=lambda result: 0,
        )
    assert budget.accounted_usd == 1
    with budget.scope("cell"), pytest.raises(ValueError, match="unresolved spend"):
        budget.call(
            role="assistant",
            fingerprint="request",
            maximum_cost_usd=1,
            operation=fail,
            encode=str,
            decode=str,
            charge=lambda result: 0,
        )


def test_changed_request_at_same_coordinate_fails_before_dispatch(tmp_path: Path) -> None:
    """Saved response reuse requires exact request identity, not merely matching text or model."""
    budget = RequestBudget(tmp_path, identity="fixture", maximum_cost_usd=2)
    _call(budget, key="first", calls=[])
    with budget.scope("first"), pytest.raises(ValueError, match="request changed"):
        budget.call(
            role="assistant",
            fingerprint="different",
            maximum_cost_usd=1,
            operation=lambda: "unexpected",
            encode=str,
            decode=str,
            charge=lambda r: 1,
        )


def test_selected_request_charges_include_unknowns_without_counting_other_roles(
    tmp_path: Path,
) -> None:
    """Explicit coordinates recover paid or unresolved calls without inspecting their responses."""
    budget = RequestBudget(tmp_path, identity="roles", maximum_cost_usd=10)
    _call(budget, key="worker", calls=[], maximum=3)
    _call(budget, key="probe", calls=[], maximum=2)
    assert budget.accounted_requests((("probe", "assistant", 0), ("probe", "assistant", 0))) == 2
    assert budget.accounted_requests((("probe", "judge", 0),)) == 0
    assert budget.accounted_requests(()) == 0

    def fail() -> str:
        """Leave a reserved request without a returned response."""
        raise TimeoutError

    with budget.scope("interrupted-probe"), pytest.raises(TimeoutError):
        budget.call(
            role="judge",
            fingerprint="x",
            maximum_cost_usd=1,
            operation=fail,
            encode=str,
            decode=str,
            charge=lambda response: 0,
        )
    assert budget.accounted_requests((("interrupted-probe", "judge", 0),)) == 1
    assert budget.accounted_usd == 6
