"""Tests for the deferred crash reconcile runner."""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime, timedelta

import pytest

from exp.runtime.gateway.ledger_crash_recovery import (
    CrashReconcileFollowUp,
    pending_crash_follow_up,
)


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


def test_follow_up_re_arms_when_the_clock_steps_back_during_a_pass() -> None:
    """A row left open because the wall clock moved back is retried, never abandoned."""
    wall = datetime(2026, 8, 18, 20, 0, tzinfo=UTC)
    due = wall - timedelta(seconds=1)
    clock = _FixedClock(wall)
    calls: list[int] = []

    def reconcile() -> tuple[int, int]:
        """Step the clock back on the first pass, so that pass settles nothing."""
        calls.append(1)
        if len(calls) == 1:
            clock.wall = wall - timedelta(seconds=10)
        return (0, 0)

    def next_due() -> datetime | None:
        """The row stays open after the first pass and is gone after the second."""
        return due if len(calls) == 1 else None

    follow_up = CrashReconcileFollowUp(
        due_at=due, reconcile=reconcile, clock=clock, poll_seconds=0.01, next_due=next_due
    )
    deadline = time.monotonic() + 10
    while not calls and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.1)
    assert not follow_up.finished
    clock.wall = wall
    while not follow_up.finished and time.monotonic() < deadline:
        time.sleep(0.01)
    assert follow_up.finished
    assert calls == [1, 1]


def test_pre_dispatch_rows_are_due_at_their_deadline_and_attempted_rows_after_grace() -> None:
    """The follow-up uses the startup pass's bounds: grace applies only once an attempt exists."""
    deadline = datetime(2026, 8, 18, 20, 0, tzinfo=UTC)
    grace = timedelta(seconds=5)
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE gateway_requests (request_id TEXT, deadline_at TEXT, terminal_state TEXT);
        CREATE TABLE gateway_attempts (request_id TEXT, state TEXT);
        """
    )
    connection.execute(
        "INSERT INTO gateway_requests VALUES ('pre-dispatch', ?, NULL)", (deadline.isoformat(),)
    )
    follow_up = pending_crash_follow_up(connection, cleanup_grace=grace)
    assert follow_up is not None
    assert follow_up.due_at == deadline

    connection.execute("INSERT INTO gateway_attempts VALUES ('pre-dispatch', 'failed')")
    follow_up = pending_crash_follow_up(connection, cleanup_grace=grace)
    assert follow_up is not None
    assert follow_up.due_at == deadline + grace
