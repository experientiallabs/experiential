"""Retain request-only terminal writes, including paid prework, until they commit.

A write the ledger refuses with a deterministic contract error
(:class:`~exp.runtime.gateway.ledger_errors.GatewayLedgerError`: the request is
unknown, owned by another organization or ledger, or already terminal with
another outcome) can never commit, so it is finalized and dropped with a
warning instead of being replayed forever. Every other failure stays retained
and fences only its own organization's new work.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from inspect import signature

from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayFailure
from exp.runtime.gateway.ledger_errors import (
    AccountingRecoverySupersededError,
    AccountingRecoveryUnresolvedTerminalError,
    GatewayLedgerError,
)
from exp.runtime.gateway.native_accounting_errors import (
    NativeBridgeError,
    accounting_recovering_error,
)
from exp.runtime.gateway.native_accounting_recovery import RecoveryRotation
from exp.runtime.gateway.native_components import SyncWriteLedger

_logger = logging.getLogger(__name__)


def _permanently_refused(error: Exception) -> bool:
    """Whether a terminal write failed a deterministic ledger contract and cannot commit.

    Recovery-owner outcomes are excluded: their owner must keep the waiver or
    liability until it independently verifies the durable result.
    """
    return isinstance(error, GatewayLedgerError) and not isinstance(
        error, (AccountingRecoverySupersededError, AccountingRecoveryUnresolvedTerminalError)
    )


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

    def require_clear(self, organization_id: str) -> None:
        """Refuse an organization's new paid work while it owns a retained failed write.

        Another organization's retained write never refuses this one.
        """
        with self._lock:
            pending = any(
                settlement.retained and settlement.authorization.organization_id == organization_id
                for settlement in self._pending.values()
            )
        if pending:
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
            except Exception as error:  # noqa: BLE001 - failed ownership must survive any ledger fault.
                if not _permanently_refused(error):
                    with self._lock:
                        settlement.retained = True
                    raise
                _logger.warning(
                    "abandoning a request terminal write the ledger permanently refused: %s",
                    type(error).__name__,
                    extra={"request_id": settlement.authorization.request_id},
                )
                committed = False
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
