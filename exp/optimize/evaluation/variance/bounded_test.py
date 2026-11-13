"""The bounded runner keeps order, overlaps calls and starts nothing after a failure."""

from __future__ import annotations

import threading
import time

import pytest

from exp.optimize.evaluation.variance.bounded import run_at_once, run_keyed


def test_calls_overlap_in_order_and_a_failure_cancels_the_unstarted_ones() -> None:
    """A stop (here an exception) must not keep dispatching queued work."""
    started: list[int] = []
    lock = threading.Lock()

    def call(index: int) -> int:
        with lock:
            started.append(index)
        time.sleep(0.05)
        if index == 1:
            raise RuntimeError("stopped")
        return index

    assert run_at_once([lambda i=i: i * 2 for i in range(5)], 3) == (0, 2, 4, 6, 8)
    with pytest.raises(RuntimeError, match="stopped"):
        run_at_once([lambda i=i: call(i) for i in range(20)], 2)
    assert len(started) < 20

    serial: list[int] = []
    with pytest.raises(RuntimeError):
        run_at_once([lambda i=i: serial.append(i) or call(i) for i in range(4)], 1)
    assert serial == [0, 1]


def test_keyed_calls_with_distinct_keys_overlap_and_shared_keys_never_do() -> None:
    """Distinct keys run at once; calls sharing a key run in order; results keep input order."""
    lock = threading.Lock()
    running: dict[str, int] = {}
    peak = {"all": 0, "shared": 0}
    order: list[int] = []

    def call(index: int, key: str) -> int:
        with lock:
            running[key] = running.get(key, 0) + 1
            peak["all"] = max(peak["all"], sum(running.values()))
            peak["shared"] = max(peak["shared"], max(running.values()))
        time.sleep(0.1)
        with lock:
            running[key] -= 1
            order.append(index)
        return index * 10

    keys = ["a", "b", "a", "c"]
    calls = [lambda i=i, k=k: call(i, k) for i, k in enumerate(keys)]
    assert run_keyed(calls, keys, 4) == (0, 10, 20, 30)
    assert peak["all"] == 3 and peak["shared"] == 1
    assert order.index(0) < order.index(2)
    order.clear()
    assert run_keyed(calls, keys, 1) == (0, 10, 20, 30) and order == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="one key"):
        run_keyed(calls, keys[:2], 2)
