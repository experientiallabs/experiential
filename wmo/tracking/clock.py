"""Injectable clock so run-tracking durations are deterministic in tests.

Production uses `SystemClock` (real `time.monotonic`). Tests pass a `FakeClock` with scripted ticks
so cost/time assertions don't depend on wall-clock timing.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def monotonic(self) -> float:
        """Seconds from an arbitrary epoch; only differences are meaningful (for durations)."""
        ...


class SystemClock:
    """Default clock backed by `time.monotonic` (monotonic, unaffected by wall-clock changes)."""

    def monotonic(self) -> float:
        return time.monotonic()
