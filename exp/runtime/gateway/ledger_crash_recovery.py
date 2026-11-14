"""Settle requests and attempts that a stopped gateway process left open.

A gateway that dies mid-request leaves rows no live process will ever finish.
Startup settles every such row already past its deadline plus the cleanup
grace. Rows still inside that bound at startup are owned by no live process
either, so one follow-up pass settles exactly those rows once the latest of
their bounds has passed.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from exp.runtime.gateway.auth import utc_text
from exp.runtime.gateway.interfaces import GatewayClock

_logger = logging.getLogger(__name__)

# Longest real wait between two reads of the gateway clock while a follow-up
# pass is pending, so a wall-clock adjustment delays it by at most this long.
FOLLOW_UP_POLL_SECONDS = 1.0


@dataclass(frozen=True)
class CrashFollowUp:
    """Open rows a startup reconcile left for one later pass.

    Attributes:
        request_ids: Open requests present at startup; no other row is touched.
        due_at: Wall time when every listed request is past its deadline plus
            the cleanup grace.
    """

    request_ids: frozenset[str]
    due_at: datetime


def reconcile_crashed(
    connection: sqlite3.Connection,
    *,
    now: datetime,
    cleanup_grace: timedelta,
    request_ids: frozenset[str] | None = None,
) -> tuple[int, int]:
    """Settle open requests and dispatched attempts past their bounds.

    A request with no attempt expires at its deadline. A request whose
    attempts all settled (a failover settles each attempt without finishing
    the request) but that never finished becomes ``unknown_after_crash`` once
    its deadline plus ``cleanup_grace`` passes, as does every still dispatched
    attempt and its request. Both terminal states release the request's
    idempotency key for a retry.

    Args:
        connection: Open write transaction owned by the caller.
        now: Wall-clock reconcile time.
        cleanup_grace: Additional bound allowed for upstream cleanup after deadline.
        request_ids: When given, only these requests and their attempts are settled.

    Returns:
        Counts of reconciled open requests and unknown dispatched attempts.
    """
    settled_requests = 0
    unknown_attempts = 0
    request_rows = connection.execute(
        """
        SELECT r.request_id, r.deadline_at,
               EXISTS(
                   SELECT 1 FROM gateway_attempts AS a WHERE a.request_id = r.request_id
               ) AS has_attempt,
               EXISTS(
                   SELECT 1 FROM gateway_attempts AS a
                   WHERE a.request_id = r.request_id AND a.state = 'dispatched'
               ) AS has_dispatched
        FROM gateway_requests AS r WHERE r.terminal_state IS NULL
        """
    ).fetchall()
    for request in request_rows:
        request_id = str(request["request_id"])
        if request_ids is not None and request_id not in request_ids:
            continue
        deadline = datetime.fromisoformat(str(request["deadline_at"]))
        if int(request["has_attempt"]) == 0:
            if deadline > now:
                continue
            state = "expired_before_dispatch"
        elif int(request["has_dispatched"]) == 0:
            if deadline + cleanup_grace > now:
                continue
            state = "unknown_after_crash"
        else:
            continue
        connection.execute(
            """
            UPDATE gateway_requests SET terminal_state = ?, terminal_at = ?
            WHERE request_id = ? AND terminal_state IS NULL
            """,
            (state, utc_text(now), request_id),
        )
        settled_requests += 1
    attempt_rows = connection.execute(
        """
        SELECT a.attempt_id, a.request_id, r.deadline_at
        FROM gateway_attempts AS a
        JOIN gateway_requests AS r ON r.request_id = a.request_id
        WHERE a.state = 'dispatched'
        """
    ).fetchall()
    for attempt in attempt_rows:
        if request_ids is not None and str(attempt["request_id"]) not in request_ids:
            continue
        deadline = datetime.fromisoformat(str(attempt["deadline_at"]))
        if deadline + cleanup_grace > now:
            continue
        connection.execute(
            """
            UPDATE gateway_attempts
            SET state = 'unknown_after_crash', terminal_at = ?,
                usage_source = 'unknown'
            WHERE attempt_id = ? AND state = 'dispatched'
            """,
            (utc_text(now), str(attempt["attempt_id"])),
        )
        connection.execute(
            """
            UPDATE gateway_requests
            SET terminal_state = 'unknown_after_crash', terminal_at = ?
            WHERE request_id = ? AND terminal_state IS NULL
            """,
            (utc_text(now), str(attempt["request_id"])),
        )
        unknown_attempts += 1
    return settled_requests, unknown_attempts


def pending_crash_follow_up(
    connection: sqlite3.Connection, *, cleanup_grace: timedelta
) -> CrashFollowUp | None:
    """Return the open requests a reconcile left and when all of them are due.

    Args:
        connection: Open connection read before this process serves any request.
        cleanup_grace: The same bound the startup reconcile applied.

    Returns:
        The still open requests and their latest deadline plus grace, or
        ``None`` when nothing is open.
    """
    rows = connection.execute(
        "SELECT request_id, deadline_at FROM gateway_requests WHERE terminal_state IS NULL"
    ).fetchall()
    if not rows:
        return None
    return CrashFollowUp(
        request_ids=frozenset(str(row["request_id"]) for row in rows),
        due_at=max(datetime.fromisoformat(str(row["deadline_at"])) for row in rows) + cleanup_grace,
    )


class CrashReconcileFollowUp:
    """Run one deferred crash reconcile on a daemon thread, cancellable at shutdown."""

    def __init__(
        self,
        *,
        due_at: datetime,
        reconcile: Callable[[], tuple[int, int]],
        clock: GatewayClock,
        poll_seconds: float = FOLLOW_UP_POLL_SECONDS,
    ) -> None:
        """Start waiting for ``due_at`` on the gateway clock.

        Args:
            due_at: Wall time after which ``reconcile`` runs once.
            reconcile: The restricted reconcile pass; returns its two counts.
            clock: Gateway clock whose wall time decides when the pass is due.
            poll_seconds: Longest real wait between two clock reads.

        Raises:
            ValueError: ``poll_seconds`` is not positive.
        """
        if poll_seconds <= 0:
            raise ValueError("crash reconcile poll interval must be positive")
        self._due_at = due_at
        self._reconcile = reconcile
        self._clock = clock
        self._poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="exp-crash-reconcile-follow-up", daemon=True
        )
        self._thread.start()

    @property
    def finished(self) -> bool:
        """Whether the pass has run, failed, or been cancelled."""
        return not self._thread.is_alive()

    def close(self, timeout_seconds: float = 5.0) -> bool:
        """Cancel a pending pass and wait for a running one to finish.

        Args:
            timeout_seconds: Longest wait for an in-progress pass.

        Returns:
            Whether the thread has exited.
        """
        self._stop.set()
        self._thread.join(timeout_seconds)
        return not self._thread.is_alive()

    def _run(self) -> None:
        """Wait on the gateway clock until due, then reconcile once unless cancelled."""
        while not self._stop.is_set():
            remaining = (self._due_at - self._clock.now()).total_seconds()
            if remaining > 0:
                self._stop.wait(min(remaining, self._poll_seconds))
                continue
            try:
                requests, attempts = self._reconcile()
            except Exception:  # noqa: BLE001 - a failed pass leaves rows for the next startup.
                _logger.exception("gateway follow-up crash reconcile failed")
                return
            _logger.info(
                "gateway follow-up crash reconcile settled %d requests and %d attempts",
                requests,
                attempts,
            )
            return
