"""The bounded runner keeps order, overlaps calls and starts nothing after a failure."""

from __future__ import annotations

import threading
import time

import pytest

from exp.optimize.evaluation.variance.bounded import run_at_once


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
