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

from exp.runtime.gateway.disconnect_estimate import settled_terminal

if TYPE_CHECKING:
    from exp.runtime.gateway.native_accounting import NativeAttemptAccounting

_SWEEP_BATCH = 16
_SWEEP_GRACE_SECONDS = 5.0
_logger = logging.getLogger(__name__)


class FairRecovery[T]:
    """Rotate organizations and their retained writes across bounded sweeps."""

    def __init__(self, organization: Callable[[T], str]) -> None:
        """Bind the trusted scope extractor; callers serialize selection."""
        self._organization = organization
        self._organizations: deque[str] = deque()
        self._positions: dict[str, int] = {}

    def select(self, items: list[T], limit: int) -> list[T]:
        """Choose work fairly, retaining no request objects between sweeps."""
        groups: dict[str, list[T]] = {}
        for item in items:
            groups.setdefault(self._organization(item), []).append(item)
        self._organizations = deque(org for org in self._organizations if org in groups)
        present = set(self._organizations)
        self._organizations.extend(org for org in groups if org not in present)
        self._positions = {org: self._positions.get(org, 0) for org in groups}
        selected: list[T] = []
        counts: dict[str, int] = {}
        while self._organizations and len(selected) < min(limit, len(items)):
            org = self._organizations.popleft()
            values = groups[org]
            count = counts.get(org, 0)
            if count < len(values):
                position = self._positions[org] % len(values)
                selected.append(values[position])
                self._positions[org] = position + 1
                counts[org] = count + 1
            self._organizations.append(org)
        return selected


def recover_expired(accounting: NativeAttemptAccounting) -> None:
    """Recover retained settlements and close abandoned requests.

    Retained writes replay the original outcome, usage, and finalize flag.
    Expired requests close as cancelled through their active attempt, or request row
    if none exists. This covers broken wire contracts and data-plane crashes short
    of process death. Repeated failures retain only their organization's admission fence.
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
            terminal, failure = settled_terminal(settlement, entry, loads=accounting._loads)
            finished = accounting._settle_swept(
                request_id,
                entry,
                attempt_id=str(settlement["attempt_id"]),
                terminal=terminal,
                failure=failure,
                finalize=bool(settlement.get("finalize", True)),
                settlement=settlement,
            )
        except Exception as error:  # noqa: BLE001 - one poison item cannot stop another tenant.
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
