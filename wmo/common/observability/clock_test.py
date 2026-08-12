"""Tests for `SystemClock`, the production time source run durations are measured against.

The `Clock` protocol itself gets no test: asserting its one member restates the declaration, and
a scripted double defined here would only prove that the double returns what it was handed. What
matters about the seam is that real consumers can be pinned by injecting one, which
`wmo/common/observability/tracker_test.py` exercises with a scripted one.
"""

from __future__ import annotations

import time

from wmo.common.observability.clock import SystemClock


def test_system_clock_reads_time_monotonic_and_never_goes_backwards() -> None:
    # Durations are computed as differences of these readings, so a wall-clock source (which can
    # jump backwards over an NTP step or a DST change) would produce negative elapsed times.
    clock = SystemClock()
    before = time.monotonic()

    first = clock.monotonic()
    second = clock.monotonic()

    after = time.monotonic()

    assert before <= first <= second <= after
