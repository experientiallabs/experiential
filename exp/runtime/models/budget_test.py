"""Parallel spend admission, crash reservations and exact request replay."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from exp.common.project import ProjectStore
from exp.runtime.models.budget import RequestBudget, SpendLimitReached
from exp.runtime.models.providers.async_transport import (
    ProviderDeadlineExceeded,
    RequestDeadline,
    run_with_retry_async,
)
from exp.runtime.models.providers.transport import (
    ProviderTransportError,
    RetryPolicy,
    is_known_unbilled_failure,
    known_unbilled_attempts,
)


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
    budget = RequestBudget(
        ProjectStore(tmp_path, "budget-test"), identity="fixture", maximum_cost_usd=1
    )
    assert _call(budget, key="first", calls=calls) == "first"
    with pytest.raises(SpendLimitReached) as paused:
        _call(budget, key="second", calls=calls)
    assert paused.value.required_usd == 2
    assert budget.accounted_usd == 1
    resumed = RequestBudget(
        ProjectStore(tmp_path, "budget-test"), identity="fixture", maximum_cost_usd=2
    )
    assert _call(resumed, key="first", calls=calls) == "first"
    assert _call(resumed, key="second", calls=calls) == "second"
    assert calls == ["first", "second"]
    assert resumed.accounted_usd == 2


def test_parallel_reservations_share_one_allowance(tmp_path: Path) -> None:
    """Concurrent cells cannot each independently spend the entire approved run budget."""
    budget = RequestBudget(
        ProjectStore(tmp_path, "budget-test"), identity="fixture", maximum_cost_usd=3
    )
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


def test_uncapped_parallel_requests_keep_durable_accounting_and_replay(tmp_path: Path) -> None:
    """No aggregate cap blocks concurrent finite reservations, and reopening preserves charges."""
    project = ProjectStore(tmp_path, "budget-test")
    budget = RequestBudget(project, identity="uncapped", maximum_cost_usd=None)
    admitted = threading.Barrier(8)
    calls: list[str] = []

    def worker(index: int) -> str:
        """Require all eight requests to hold their reservations before any settles."""
        key = str(index)

        def operation() -> str:
            """Prove that independent requests are admitted together without an aggregate cap."""
            calls.append(key)
            admitted.wait(timeout=10)
            return key

        with budget.scope(key):
            return budget.call(
                role="assistant",
                fingerprint=key,
                maximum_cost_usd=100,
                operation=operation,
                encode=str,
                decode=str,
                charge=lambda result: 100,
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(worker, range(8))) == [str(index) for index in range(8)]
    assert budget.accounted_usd == 800
    reopened = RequestBudget(ProjectStore(tmp_path, "budget-test"), identity="uncapped")
    for index in range(8):
        assert _call(reopened, key=str(index), calls=calls, maximum=100) == str(index)
    assert len(calls) == 8
    assert reopened.accounted_usd == 800


def test_aggregate_limit_can_be_removed_and_lowered_without_rewriting_paid_work(
    tmp_path: Path,
) -> None:
    """Explicit replacement affects new admission while exact paid replay remains free."""
    project = ProjectStore(tmp_path, "budget-test")
    calls: list[str] = []
    limited = RequestBudget(project, identity="replace", maximum_cost_usd=1)
    _call(limited, key="first", calls=calls)
    with pytest.raises(SpendLimitReached):
        _call(limited, key="second", calls=calls)
    unlimited = RequestBudget(project, identity="replace", maximum_cost_usd=None)
    _call(unlimited, key="second", calls=calls)
    reduced = RequestBudget(project, identity="replace", maximum_cost_usd=0.5)
    assert _call(reduced, key="first", calls=calls) == "first"
    with pytest.raises(SpendLimitReached) as paused:
        _call(reduced, key="third", calls=calls)
    assert paused.value.limit_usd == 0.5
    assert paused.value.accounted_usd == 2
    assert calls == ["first", "second"]


@pytest.mark.parametrize("maximum", [-1, float("inf"), float("nan")])
def test_uncapped_execution_still_requires_finite_request_reservations(
    tmp_path: Path, maximum: float
) -> None:
    """Removing aggregate authorization never removes each request's bounded admission."""
    budget = RequestBudget(ProjectStore(tmp_path, "budget-test"), identity="bounds")
    calls: list[str] = []
    with pytest.raises(ValueError, match="reservation must be finite and nonnegative"):
        _call(budget, key="invalid", calls=calls, maximum=maximum)
    assert calls == []
    assert budget.accounted_usd == 0


def test_uncapped_request_cannot_settle_above_its_reservation(tmp_path: Path) -> None:
    """A provider violating a finite request bound retains an unresolved receipt."""
    project = ProjectStore(tmp_path, "budget-test")
    budget = RequestBudget(project, identity="overcharge")
    with budget.scope("cell"), pytest.raises(ValueError, match="exceeds.*reservation"):
        budget.call(
            role="assistant",
            fingerprint="request",
            maximum_cost_usd=1,
            operation=lambda: "answer",
            encode=str,
            decode=str,
            charge=lambda result: 2,
        )
    assert budget.accounted_usd == 1
    reopened = RequestBudget(project, identity="overcharge")
    with reopened.scope("cell"), pytest.raises(ValueError, match="unresolved spend"):
        reopened.call(
            role="assistant",
            fingerprint="request",
            maximum_cost_usd=1,
            operation=lambda: "unexpected",
            encode=str,
            decode=str,
            charge=lambda result: 0,
        )


@pytest.mark.parametrize("limit", [2, None])
@pytest.mark.parametrize("maximum", [0, 1])
def test_unknown_dispatch_retains_its_reservation_and_never_replays(
    tmp_path: Path, maximum: float, limit: float | None
) -> None:
    """A provider failure is not evidence that a paid request was free."""
    budget = RequestBudget(
        ProjectStore(tmp_path, "budget-test"), identity="fixture", maximum_cost_usd=limit
    )

    def fail() -> str:
        """Simulate loss of a provider response after dispatch."""
        raise TimeoutError

    with budget.scope("cell"), pytest.raises(TimeoutError):
        budget.call(
            role="assistant",
            fingerprint="request",
            maximum_cost_usd=maximum,
            operation=fail,
            encode=str,
            decode=str,
            charge=lambda result: 0,
        )
    assert budget.accounted_usd == maximum
    resumed = RequestBudget(
        ProjectStore(tmp_path, "budget-test"), identity="fixture", maximum_cost_usd=limit
    )
    with resumed.scope("cell"), pytest.raises(ValueError, match="unresolved spend") as saved:
        resumed.call(
            role="assistant",
            fingerprint="request",
            maximum_cost_usd=maximum,
            operation=fail,
            encode=str,
            decode=str,
            charge=lambda result: 0,
        )
    assert not is_known_unbilled_failure(saved.value)


@pytest.mark.parametrize("unknown_first", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_terminal_admission_failure_settles_only_certified_unpaid_requests(
    tmp_path: Path, unknown_first: bool, cancel: bool
) -> None:
    """Timeout/cancel is durably zero only without unknown dispatch, and fresh attempts can run."""
    project = ProjectStore(tmp_path, "budget-test")
    budget = RequestBudget(project, identity="unpaid", maximum_cost_usd=2)
    attempts = 0

    async def operation(timeout: float) -> str:
        """Optionally lose one response, then receive only trusted admission refusals."""
        del timeout
        nonlocal attempts
        attempts += 1
        if unknown_first and attempts == 1:
            raise ProviderTransportError("unknown dispatch")
        raise ProviderTransportError("busy", status_code=429, known_unbilled=True)

    async def sleep(seconds: float) -> None:
        """Permit one unknown retry, then interrupt after the certified refusal."""
        del seconds
        if unknown_first and attempts == 1:
            return
        if cancel:
            raise asyncio.CancelledError
        raise ProviderDeadlineExceeded("provider request deadline exceeded")

    def dispatch() -> str:
        """Use the production owning retry loop to establish aggregate failure evidence."""
        return asyncio.run(
            run_with_retry_async(
                operation, policy=RetryPolicy(), deadline=RequestDeadline.after(10), sleep=sleep
            )
        )

    error_type = asyncio.CancelledError if cancel else ProviderDeadlineExceeded
    with budget.scope("cell-attempt-1"), pytest.raises(error_type):
        budget.call(
            role="assistant",
            fingerprint="request",
            maximum_cost_usd=1,
            operation=dispatch,
            encode=str,
            decode=str,
            charge=lambda result: 0,
        )
    expected = 1 if unknown_first else 0
    assert budget.accounted_usd == expected
    resumed = RequestBudget(project, identity="unpaid", maximum_cost_usd=2)
    assert resumed.accounted_usd == expected
    prior_attempts = attempts
    message = "unresolved spend" if unknown_first else "certified unpaid"
    with resumed.scope("cell-attempt-1"), pytest.raises(ValueError, match=message) as saved:
        resumed.call(
            role="assistant",
            fingerprint="request",
            maximum_cost_usd=1,
            operation=dispatch,
            encode=str,
            decode=str,
            charge=lambda result: 0,
        )
    assert attempts == prior_attempts
    assert known_unbilled_attempts(saved.value) == (0 if unknown_first else prior_attempts)
    with (
        resumed.scope("cell-attempt-1"),
        pytest.raises(ValueError, match="request changed") as changed,
    ):
        resumed.call(
            role="assistant",
            fingerprint="different-request",
            maximum_cost_usd=1,
            operation=dispatch,
            encode=str,
            decode=str,
            charge=lambda result: 0,
        )
    assert not is_known_unbilled_failure(changed.value)
    assert attempts == prior_attempts
    assert _call(resumed, key="cell-attempt-2", calls=[]) == "cell-attempt-2"
    assert resumed.accounted_usd == expected + 1


def test_changed_request_at_same_coordinate_fails_before_dispatch(tmp_path: Path) -> None:
    """Saved response reuse requires exact request identity, not merely matching text or model."""
    budget = RequestBudget(
        ProjectStore(tmp_path, "budget-test"), identity="fixture", maximum_cost_usd=2
    )
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
    budget = RequestBudget(
        ProjectStore(tmp_path, "budget-test"), identity="roles", maximum_cost_usd=10
    )
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
