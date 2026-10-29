"""Retain request-only terminal writes, including paid prework, until they commit."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from inspect import signature

from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayFailure
from exp.runtime.gateway.native_accounting_errors import (
    NativeBridgeError,
    accounting_recovering_error,
)
from exp.runtime.gateway.native_accounting_recovery import RecoveryRotation
from exp.runtime.gateway.native_components import SyncWriteLedger


@dataclass
class _Settlement:
    """One request's original terminal write and its retry ownership.

    Attributes:
        authorization: Frozen authority for the accepted request.
        failure: Original terminal failure to preserve across retries.
        certify_no_effects: Whether accounting can certify no paid work occurred.
        web_search_requests: Completed search count to persist with the request.
        retained: Whether a failed write needs recovery, false initially.
        committed: Durable result shared by concurrent callers, initially unknown.
        lock: Per-request lock serializing terminal writes and recovery attempts.
    """

    authorization: AuthorizationSnapshot
    failure: GatewayFailure
    certify_no_effects: bool
    web_search_requests: int
    retained: bool = False
    committed: bool | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class RequestSettlements:
    """Request-only counterpart of the accounting registry's retained attempt settlements."""

    def __init__(self, ledger: SyncWriteLedger) -> None:
        """Bind the durable owner and retain no work until a terminal write begins."""
        try:
            signature(ledger.finish_request).bind(
                authorization=None, failure=None, certify_no_effects=False, web_search_requests=0
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "write ledger must implement finish_request(..., web_search_requests=...) "
                "and persist unattempted search expense before enabling this gateway release"
            ) from exc
        self._ledger = ledger
        self._pending: dict[str, _Settlement] = {}
        self._lock = threading.Lock()
        self._recovery = RecoveryRotation[_Settlement](lambda item: item.authorization.request_id)

    @property
    def pending(self) -> bool:
        """Whether a failed write remains uncommitted; ordinary in-progress commits are healthy."""
        with self._lock:
            return any(settlement.retained for settlement in self._pending.values())

    @property
    def outstanding(self) -> bool:
        """Whether any terminal write still owns the transport, including its first commit."""
        with self._lock:
            return bool(self._pending)

    def require_clear(self) -> None:
        """Refuse new paid work while this registry retains a failed meter write."""
        if self.pending:
            raise NativeBridgeError(accounting_recovering_error())

    def finish(
        self,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
        *,
        certify_no_effects: bool = False,
        web_search_requests: int = 0,
    ) -> bool:
        """Retain the original failure and meter before attempting an idempotent write."""
        with self._lock:
            settlement = self._pending.setdefault(
                authorization.request_id,
                _Settlement(
                    authorization,
                    failure,
                    certify_no_effects,
                    web_search_requests,
                ),
            )
        return self._commit(settlement)

    def _commit(self, settlement: _Settlement) -> bool:
        """Serialize duplicate finish/sweep writers without blocking other requests."""
        with settlement.lock:
            if settlement.committed is not None:
                return settlement.committed
            try:
                committed = self._ledger.finish_request(
                    authorization=settlement.authorization,
                    failure=settlement.failure,
                    certify_no_effects=settlement.certify_no_effects,
                    web_search_requests=settlement.web_search_requests,
                )
            except Exception:  # noqa: BLE001 - failed ownership must survive any ledger fault.
                with self._lock:
                    settlement.retained = True
                raise
            settlement.committed = committed is True
            with self._lock:
                if self._pending.get(settlement.authorization.request_id) is settlement:
                    del self._pending[settlement.authorization.request_id]
            return settlement.committed

    def retry(self, limit: int) -> None:
        """Replay bounded retained writes verbatim; a repeated failure keeps its owner."""
        with self._lock:
            pending = self._recovery.select(
                [item for item in self._pending.values() if item.retained], limit
            )
        for settlement in pending:
            try:
                self._commit(settlement)
            except Exception:  # noqa: BLE001 - retain until the durable writer recovers.
                continue
