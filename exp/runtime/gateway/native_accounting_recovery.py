"""Fair background recovery selection, independent of the request admission path."""

from __future__ import annotations

# This companion owns the accounting registry's recovery loop and locked state.
# ruff: noqa: SLF001
import json
import logging
import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exp.runtime.gateway.native_accounting import NativeAttemptAccounting

_SWEEP_BATCH = 16
_SWEEP_GRACE_SECONDS = 5.0
_logger = logging.getLogger(__name__)


class RecoveryRotation[T]:
    """Rotate retained work by its stable identity across bounded sweeps."""

    def __init__(self, identity: Callable[[T], str]) -> None:
        """Bind the work identity extractor; callers serialize selection."""
        self._identity = identity
        self._order: deque[str] = deque()

    def select(self, items: list[T], limit: int) -> list[T]:
        """Visit each pending identity without retaining payloads or replaying one twice."""
        if limit < 0:
            raise ValueError("recovery selection limit must be nonnegative")
        pending = {self._identity(item): item for item in items}
        if len(pending) != len(items):
            raise ValueError("recovery work identities must be unique")
        self._order = deque(identity for identity in self._order if identity in pending)
        present = set(self._order)
        self._order.extend(identity for identity in pending if identity not in present)
        selected: list[T] = []
        for _ in range(min(limit, len(pending))):
            identity = self._order.popleft()
            selected.append(pending[identity])
            self._order.append(identity)
        return selected


def recovery_drained(accounting: NativeAttemptAccounting) -> bool:
    """Distinguish a healthy in-progress commit from a transport safe to close."""
    with accounting._lock:
        active = accounting._active_settlements > 0 or bool(accounting._inflight)
    return (
        not active
        and not accounting.request_settlements.outstanding
        and accounting.accounting_healthy
    )


def recover_until_stopped(accounting: NativeAttemptAccounting, interval_seconds: float) -> None:
    """Keep transport ownership until retained and in-progress terminal writes finish."""
    while True:
        closing = accounting._stop_sweeper.wait(interval_seconds)
        if closing and recovery_drained(accounting):
            return
        try:
            accounting.sweep_expired()
        except Exception as error:  # noqa: BLE001 - failed writes remain owned for retry.
            _logger.warning("gateway recovery pass failed: %s", type(error).__name__)
        if closing:
            if recovery_drained(accounting):
                return
            time.sleep(min(interval_seconds, 0.1))


def recover_expired(accounting: NativeAttemptAccounting) -> None:
    """Recover retained settlements and close abandoned requests.

    Retained writes replay the original outcome, usage, and finalize flag.
    Expired requests close as cancelled through their active attempt, or request row
    if none exists. This covers broken wire contracts and data-plane crashes short
    of process death. Failed writes retain this control plane's admission fence.
    """
    accounting.request_settlements.retry(_SWEEP_BATCH)
    now = time.monotonic()
    with accounting._lock:
        retained = accounting._retained_recovery.select(
            [
                (request_id, entry)
                for request_id, entry in accounting._inflight.items()
                if entry.pending_settlement is not None
            ],
            _SWEEP_BATCH,
        )
        abandoned = accounting._abandoned_recovery.select(
            [
                (request_id, entry)
                for request_id, entry in accounting._inflight.items()
                if entry.pending_settlement is None
                and (
                    entry.pending_abandon is not None
                    or entry.deadline_monotonic + _SWEEP_GRACE_SECONDS < now
                )
            ],
            _SWEEP_BATCH,
        )
    for request_id, entry in retained:
        settlement = entry.pending_settlement
        if settlement is None:
            continue
        try:
            finished = accounting._settle_swept(request_id, entry, settlement)
        except Exception as error:  # noqa: BLE001 - one poison item cannot stop later work.
            _logger.warning("retained gateway recovery failed: %s", type(error).__name__)
            continue
        if finished:
            with accounting._lock:
                if entry.pending_settlement == settlement:
                    entry.pending_settlement = None
                accounting._sweep_retained_replayed += 1
    for request_id, entry in abandoned:
        try:
            accounting.abandon(json.dumps({"request_id": request_id}))
        except Exception as error:  # noqa: BLE001 - retain cancellation and continue fair recovery.
            _logger.warning("abandoned gateway recovery failed: %s", type(error).__name__)
            continue
        with accounting._lock:
            if accounting._inflight.get(request_id) is not entry:
                accounting._sweep_abandoned_cancelled += 1
