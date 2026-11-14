"""Tests for the deferred crash reconcile runner."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest

from exp.runtime.gateway.ledger_crash_recovery import CrashReconcileFollowUp


class _FixedClock:
    """Gateway clock frozen at one wall time."""

    def __init__(self, wall: datetime) -> None:
        """Freeze the clock at ``wall``."""
        self.wall = wall

    def now(self) -> datetime:
        """Return the frozen wall time."""
        return self.wall

    def monotonic(self) -> float:
        """Return a constant monotonic time."""
        return 0.0


def test_follow_up_runs_once_when_already_due() -> None:
    """A pass whose bound has already passed reconciles exactly once and exits."""
    wall = datetime(2026, 8, 18, 20, 0, tzinfo=UTC)
    calls: list[int] = []

    def reconcile() -> tuple[int, int]:
        """Record one reconcile call."""
        calls.append(1)
        return (0, 1)

    follow_up = CrashReconcileFollowUp(
        due_at=wall - timedelta(seconds=1),
        reconcile=reconcile,
        clock=_FixedClock(wall),
        poll_seconds=0.01,
    )
    deadline = time.monotonic() + 10
    while not follow_up.finished and time.monotonic() < deadline:
        time.sleep(0.01)
    assert follow_up.finished
    assert calls == [1]


def test_follow_up_rejects_a_non_positive_poll_interval() -> None:
    """A zero poll interval would spin, so it is refused at construction."""
    wall = datetime(2026, 8, 18, 20, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="poll interval"):
        CrashReconcileFollowUp(
            due_at=wall, reconcile=lambda: (0, 0), clock=_FixedClock(wall), poll_seconds=0
        )
