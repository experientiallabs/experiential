"""Tests for the injectable clock the run tracker measures durations against."""

from __future__ import annotations

import time

from wmo.common.observability.clock import Clock, SystemClock


class _ScriptedClock:
    """A test double that returns each tick in turn (the shape production tests inject)."""

    def __init__(self, ticks: list[float]) -> None:
        self._ticks = ticks
        self._index = 0

    def monotonic(self) -> float:
        value = self._ticks[self._index]
        self._index += 1
        return value


def test_system_clock_is_monotonic_and_reads_time_monotonic() -> None:
    clock = SystemClock()
    before = time.monotonic()

    first = clock.monotonic()
    second = clock.monotonic()

    assert before <= first <= second


def test_the_protocol_is_runtime_checkable_so_a_scripted_double_substitutes() -> None:
    # `Clock` is the seam cost/duration assertions depend on; a plain object with `monotonic`
    # must satisfy it without inheriting anything, or every test would need the real clock.
    assert isinstance(SystemClock(), Clock)
    assert isinstance(_ScriptedClock([0.0]), Clock)
    assert not isinstance(object(), Clock)


def test_only_differences_are_meaningful() -> None:
    clock: Clock = _ScriptedClock([10.5, 12.0])

    start = clock.monotonic()
    end = clock.monotonic()

    assert end - start == 1.5
