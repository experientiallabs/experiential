"""Retain request-only terminal writes, including paid prework, until they commit.

A write the ledger refuses with a deterministic contract error (a plain
:class:`~exp.runtime.gateway.ledger_errors.GatewayLedgerError`: the request is
unknown, belongs to another organization, or is already terminal with another
outcome) can never commit, so it is finalized uncommitted and dropped with a
warning instead of being replayed on every sweep. Every other failure stays
retained: transport and storage faults, a deliberate
:class:`~exp.runtime.gateway.ledger_errors.AccountingRefusedError` (another
ledger owns the organization for now), a pre-dispatch
:class:`~exp.runtime.gateway.ledger_errors.AttemptRejectedError`, and the two
recovery-owner outcomes, whose owner must verify the durable result first.
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
    AccountingRefusedError,
    AttemptRejectedError,
    GatewayLedgerError,
)
from exp.runtime.gateway.native_accounting_fence import OrganizationFence, fences_after
from exp.runtime.gateway.native_accounting_recovery import RecoveryRotation
from exp.runtime.gateway.native_components import SyncWriteLedger

_logger = logging.getLogger(__name__)

_REPLAYABLE_LEDGER_ERRORS = (
    AccountingRecoverySupersededError,
    AccountingRecoveryUnresolvedTerminalError,
    AccountingRefusedError,
    AttemptRejectedError,
)
"""Ledger errors that may succeed on replay, so their write stays retained."""


def permanently_refused(error: BaseException) -> bool:
    """Return whether a terminal write failed a deterministic ledger contract.

    Args:
        error: The exception the durable write raised.

    Returns:
        True only for a plain contract violation that no replay can commit;
        False for every replayable ledger outcome and every other fault.
    """
    return isinstance(error, GatewayLedgerError) and not isinstance(
        error, _REPLAYABLE_LEDGER_ERRORS
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
        fences: Whether the latest failed write fences its organization's
            admission; a deliberate ledger refusal never does. False initially.
        committed: Durable result shared by concurrent callers, initially unknown.
        lock: Per-request lock serializing terminal writes and recovery attempts.
    """

    authorization: AuthorizationSnapshot
    failure: GatewayFailure
    certify_no_effects: bool
    web_search_requests: int
    retained: bool = False
    fences: bool = False
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
        self._fence = OrganizationFence()  # Fencing retained writes per org, under ``_lock``.
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

    def fenced_organizations(self) -> frozenset[str]:
        """Return the organizations owning a retained write that fences their admission."""
        with self._lock:
            return self._fence.organizations()

    def fenced(self, organization_id: str) -> bool:
        """Return, in constant time, whether one organization owns a fencing retained write."""
        with self._lock:
            return self._fence.fenced(organization_id)

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
            except Exception as exc:  # noqa: BLE001 - failed ownership must survive any ledger fault.
                if not permanently_refused(exc):
                    with self._lock:
                        settlement.retained = True
                        settlement.fences = fences_after(exc)
                        self._mark(settlement)
                    raise
                _logger.warning(
                    "dropping a request terminal write the ledger permanently refused: %s",
                    type(exc).__name__,
                    extra={"request_id": settlement.authorization.request_id},
                )
                committed = False
            settlement.committed = committed is True
            with self._lock:
                if self._pending.get(settlement.authorization.request_id) is settlement:
                    del self._pending[settlement.authorization.request_id]
                self._mark(settlement)
            return settlement.committed

    def _mark(self, settlement: _Settlement) -> None:
        """Re-derive one settlement's fence state; call under ``_lock``.

        Only a still-pending, retained, fencing write fences: a committed or
        dropped write releases its organization's entry.
        """
        authorization = settlement.authorization
        self._fence.mark(
            authorization.organization_id,
            authorization.request_id,
            fencing=settlement.retained
            and settlement.fences
            and self._pending.get(authorization.request_id) is settlement,
        )

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
