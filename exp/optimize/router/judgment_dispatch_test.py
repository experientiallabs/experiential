"""Concurrency, ordering, and interruption contracts for judgment workers."""

import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from exp.common.evaluations import EvaluationCellEvidence
from exp.common.project import ProjectStore
from exp.optimize.router.judgment_dispatch import dispatch_judgments
from exp.runtime.models.budget import RequestBudget, SpendLimitReached


def _evidence(index: int) -> EvaluationCellEvidence:
    """Create minimal bound evidence without any provider or artifact access."""
    return EvaluationCellEvidence(
        cell_id=f"cell-{index}",
        protocol_id="protocol-a",
        rollout_artifact_id=f"rollout-{index}",
        judgment_artifact_id=f"judgment-{index}",
    )


@pytest.mark.parametrize("concurrency", [1, 3])
def test_judgments_overlap_with_one_total_limit_and_preserve_input_order(
    concurrency: int,
) -> None:
    """Two full windows overlap, complete out of order, and retain canonical output order."""
    barrier = threading.Barrier(3)
    finished_last = [threading.Event(), threading.Event()]
    lock = threading.Lock()
    active = 0
    peak = 0
    completion_order: list[int] = []
    progress: list[int] = []
    owner = threading.get_ident()

    def report_completed(count: int) -> None:
        """Reject background callbacks and retain each monotonic completion update."""
        assert threading.get_ident() == owner
        progress.append(count)

    def execute(index: int, cancelled: Callable[[], bool]) -> EvaluationCellEvidence:
        """Hold a complete window open, then finish each window's first item last."""
        nonlocal active, peak
        assert not cancelled()
        with lock:
            active += 1
            peak = max(peak, active)
        if concurrency > 1:
            barrier.wait(timeout=5)
        if concurrency > 1 and index % 3 == 0:
            assert finished_last[index // 3].wait(timeout=5)
        with lock:
            completion_order.append(index)
            active -= 1
        if index % 3 == 2:
            finished_last[index // 3].set()
        return _evidence(index)

    result = dispatch_judgments(
        tuple(range(6)), execute, maximum_concurrency=concurrency, on_completed=report_completed
    )
    assert result == tuple(_evidence(index) for index in range(6))
    assert peak == concurrency
    assert progress == list(range(1, 7))
    if concurrency > 1:
        assert completion_order.index(2) < completion_order.index(0)
        assert completion_order.index(5) < completion_order.index(3)


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_judgment_failure_stops_new_work_and_drains_started_work(
    failure: type[BaseException],
) -> None:
    """The original failure survives while a sibling persists its already admitted result."""
    barrier = threading.Barrier(2)
    started: list[int] = []
    persisted: list[int] = []

    def execute(index: int, cancelled: Callable[[], bool]) -> EvaluationCellEvidence:
        """Fail one worker after both start, while the other finishes durable cleanup."""
        started.append(index)
        barrier.wait(timeout=5)
        if index == 0:
            raise failure("injected judgment interruption")
        deadline = time.monotonic() + 5
        while not cancelled() and time.monotonic() < deadline:
            threading.Event().wait(0.001)
        assert cancelled()
        persisted.append(index)
        return _evidence(index)

    with pytest.raises(failure, match="injected judgment interruption"):
        dispatch_judgments(tuple(range(10)), execute, maximum_concurrency=2)
    assert sorted(started) == [0, 1]
    assert persisted == [1]


def test_invalid_concurrency_cannot_dispatch_a_judgment() -> None:
    """An invalid execution allowance fails before any worker runs."""

    def unexpected(index: int, cancelled: Callable[[], bool]) -> EvaluationCellEvidence:
        """Fail if admission dispatches despite a nonpositive allowance."""
        raise AssertionError("unexpected judgment dispatch")

    with pytest.raises(ValueError, match="concurrency must be positive"):
        dispatch_judgments((0,), unexpected, maximum_concurrency=0)


def test_parallel_judgment_spend_pause_drains_and_replays_paid_responses(
    tmp_path: Path,
) -> None:
    """A shared limit pauses the window, then a fresh ledger resumes without double payment."""
    project = ProjectStore(tmp_path, "parallel-judgments")
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    provider_calls: list[int] = []
    active = 0
    peak = 0

    def run(limit: float) -> tuple[EvaluationCellEvidence, ...]:
        """Reconstruct request admission as a new process would after a spending pause."""
        budget = RequestBudget(project, identity="judgments-a", maximum_cost_usd=limit)

        def execute(index: int, cancelled: Callable[[], bool]) -> EvaluationCellEvidence:
            """Save one exact response with a single dollar charged only on initial dispatch."""

            def operation() -> str:
                """Overlap the first paid pair before allowing either to settle."""
                nonlocal active, peak
                with lock:
                    provider_calls.append(index)
                    ordinal = len(provider_calls)
                    active += 1
                    peak = max(peak, active)
                try:
                    if ordinal <= 2:
                        barrier.wait(timeout=5)
                    return str(index)
                finally:
                    with lock:
                        active -= 1

            with budget.scope(f"judgment-{index}"):
                response = budget.call(
                    role="judge",
                    fingerprint=f"request-{index}",
                    maximum_cost_usd=1,
                    operation=operation,
                    encode=str,
                    decode=str,
                    charge=lambda result: 1,
                )
            assert response == str(index)
            return _evidence(index)

        try:
            return dispatch_judgments(tuple(range(5)), execute, maximum_concurrency=4)
        finally:
            assert active == 0
            assert budget.accounted_usd == len(provider_calls)
            assert budget.accounted_usd <= limit

    with pytest.raises(SpendLimitReached):
        run(3)
    assert len(provider_calls) == len(set(provider_calls)) == 3
    assert peak >= 2
    expected = tuple(_evidence(index) for index in range(5))
    assert run(5) == expected
    assert len(provider_calls) == len(set(provider_calls)) == 5
    assert run(5) == expected
    assert len(provider_calls) == 5
