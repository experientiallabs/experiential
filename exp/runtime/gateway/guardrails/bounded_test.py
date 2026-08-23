"""Bounded classifier execution tests."""

from __future__ import annotations

import threading
import time

import pytest

from exp.runtime.gateway.guardrails.bounded import ClassifierTimeoutError, run_bounded


def test_run_bounded_returns_the_inspect_result() -> None:
    """A timely inspect result is returned to the caller."""
    assert run_bounded(lambda: 7, 0.5) == 7


def test_run_bounded_raises_without_waiting_for_a_blocking_inspect() -> None:
    """A hung inspect times out on the caller before the worker returns."""

    def hang() -> int:
        """Sleep longer than the allowed timeout."""
        time.sleep(0.25)
        return 1

    started = time.monotonic()
    with pytest.raises(ClassifierTimeoutError):
        run_bounded(hang, 0.05)
    assert time.monotonic() - started < 1.0


def test_run_bounded_propagates_inspect_errors() -> None:
    """Adapter exceptions surface after the worker finishes inside the budget."""

    def boom() -> int:
        """Fail immediately."""
        raise RuntimeError("classifier unavailable")

    with pytest.raises(RuntimeError, match="classifier unavailable"):
        run_bounded(boom, 0.5)


def test_run_bounded_caps_inflight_workers_instead_of_leaking_threads() -> None:
    """A full inflight cap fail-closes without starting another inspect thread."""
    slots = threading.BoundedSemaphore(2)
    block = threading.Event()
    entered = threading.Semaphore(0)
    started_inspects = 0
    lock = threading.Lock()

    def hang() -> int:
        """Hold a slot until the test releases every occupant."""
        nonlocal started_inspects
        with lock:
            started_inspects += 1
        entered.release()
        block.wait(2.0)
        return 1

    def occupy() -> None:
        """Hold one cap slot for the duration of the blocking inspect."""
        try:
            run_bounded(hang, 1.0, slots=slots)
        except ClassifierTimeoutError:
            return

    occupants = [threading.Thread(target=occupy) for _ in range(2)]
    for occupant in occupants:
        occupant.start()
    assert entered.acquire(timeout=1.0)
    assert entered.acquire(timeout=1.0)
    started = time.monotonic()
    with pytest.raises(ClassifierTimeoutError, match="capacity is exhausted"):
        run_bounded(hang, 0.05, slots=slots)
    assert time.monotonic() - started < 0.5
    assert started_inspects == 2
    block.set()
    for occupant in occupants:
        occupant.join(1.0)
    assert run_bounded(lambda: 3, 0.5, slots=slots) == 3
