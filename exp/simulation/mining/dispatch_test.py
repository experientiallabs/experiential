"""Tests for bounded parallel dispatch of paid tagging work."""

from __future__ import annotations

import threading
import time

import pytest

from exp.simulation.mining.dispatch import DEFAULT_TAGGING_CONCURRENCY, dispatch_tagging


def test_the_default_allowance_is_the_shared_one() -> None:
    """Concurrency vocabulary is shared across this project rather than invented per caller."""
    assert DEFAULT_TAGGING_CONCURRENCY == 8


def test_work_actually_overlaps_up_to_the_allowance() -> None:
    """A barrier is the proof: it can only trip if that many callers are inside at once."""
    arrived = threading.Barrier(4, timeout=10.0)

    def answer(item: int) -> int:
        """Block until every expected sibling is in flight, then return."""
        arrived.wait()
        return item * 10

    results = dispatch_tagging(tuple(range(4)), answer, maximum_concurrency=4)

    assert results == (0, 10, 20, 30)


def test_the_allowance_is_a_ceiling_nothing_exceeds() -> None:
    """A local run must stay a bounded load on the endpoint, not a burst of every item."""
    active = 0
    peak = 0
    guard = threading.Lock()

    def answer(item: int) -> int:
        """Record how many callers share this window at its busiest."""
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with guard:
            active -= 1
        return item

    dispatch_tagging(tuple(range(40)), answer, maximum_concurrency=3)

    assert peak <= 3
    assert peak > 1


def test_results_follow_input_order_not_completion_order() -> None:
    """Whatever the caller builds from the results cannot be allowed to depend on scheduling."""
    completed: list[int] = []
    guard = threading.Lock()

    def answer(item: int) -> str:
        """Finish in reverse order of admission."""
        time.sleep(0.02 * (5 - item))
        with guard:
            completed.append(item)
        return f"item-{item}"

    results = dispatch_tagging(tuple(range(5)), answer, maximum_concurrency=5)

    assert results == ("item-0", "item-1", "item-2", "item-3", "item-4")
    assert completed != list(range(5))


def test_one_call_in_flight_runs_inline_on_the_calling_thread() -> None:
    """Concurrency of one has to be a strictly serial pass, down to the thread it runs on."""
    threads: list[str] = []

    def answer(item: int) -> int:
        """Record the thread that answered."""
        threads.append(threading.current_thread().name)
        return item

    results = dispatch_tagging((1, 2, 3), answer, maximum_concurrency=1)

    assert results == (1, 2, 3)
    assert set(threads) == {threading.current_thread().name}


def test_no_items_needs_no_pool() -> None:
    """An empty pass is a no-op, not a reason to start threads."""

    def unexpected(item: int) -> int:
        """Fail if an empty item list reaches the answer boundary."""
        raise AssertionError("an empty dispatch answered something")

    assert dispatch_tagging((), unexpected, maximum_concurrency=8) == ()


def test_the_first_failure_by_item_order_is_the_one_raised() -> None:
    """A corpus-wide run must report the same first problem on every attempt."""

    def answer(item: int) -> int:
        """Fail two items, with the later one failing first in wall-clock order."""
        if item == 7:
            raise ValueError("later item failed first")
        if item == 2:
            time.sleep(0.05)
            raise ValueError("earliest failure")
        return item

    for _ in range(5):
        with pytest.raises(ValueError, match="earliest failure"):
            dispatch_tagging(tuple(range(8)), answer, maximum_concurrency=8)


def test_a_failure_stops_admitting_further_paid_work() -> None:
    """Every admission spends money, so nothing new starts once a pass is already failing."""
    attempted: set[int] = set()
    guard = threading.Lock()

    def answer(item: int) -> int:
        """Record the attempt and fail the first item."""
        with guard:
            attempted.add(item)
        if item == 0:
            raise ValueError("stop the pass")
        time.sleep(0.01)
        return item

    with pytest.raises(ValueError, match="stop the pass"):
        dispatch_tagging(tuple(range(100)), answer, maximum_concurrency=4)

    assert attempted <= {0, 1, 2, 3}


def test_work_already_in_flight_finishes_before_the_failure_surfaces() -> None:
    """An admitted item has been paid for, so it must reach its own commit path."""
    finished: list[int] = []
    guard = threading.Lock()
    released = threading.Event()

    def answer(item: int) -> int:
        """Fail the first item and let the siblings run to completion."""
        if item == 0:
            released.wait(5.0)
            raise ValueError("first item failed")
        time.sleep(0.02)
        with guard:
            finished.append(item)
        released.set()
        return item

    with pytest.raises(ValueError, match="first item failed"):
        dispatch_tagging(tuple(range(4)), answer, maximum_concurrency=4)

    assert sorted(finished) == [1, 2, 3]


def test_a_failed_pass_leaves_no_worker_threads_behind() -> None:
    """A long build makes many passes, so a leaked pool per failure would accumulate."""
    baseline = threading.active_count()

    def answer(item: int) -> int:
        """Fail every item."""
        raise RuntimeError("every item failed")

    with pytest.raises(RuntimeError, match="every item failed"):
        dispatch_tagging(tuple(range(16)), answer, maximum_concurrency=8)

    assert threading.active_count() == baseline


def test_a_nonpositive_allowance_is_refused() -> None:
    """Zero calls in flight would answer nothing while looking like a configured choice."""

    def unexpected(item: int) -> int:
        """Fail if a refused allowance still answered an item."""
        raise AssertionError("a refused allowance answered something")

    with pytest.raises(ValueError, match="at least one call in flight"):
        dispatch_tagging((1,), unexpected, maximum_concurrency=0)
