"""Typed content-free ledger transition and caller replay failures."""

from exp.runtime.gateway.stream_contracts import GatewayFailure, GatewayFailureClass


class GatewayLedgerError(ValueError):
    """A request or attempt transition violates the content-free ledger contract."""


class AccountingRecoverySupersededError(GatewayLedgerError):
    """An immutable provider terminal supersedes a trusted customer-waiver request.

    Accounting did not change. A recovery owner may release its pending waiver
    only after independently verifying the matching durable delivery outcome.
    """


class AccountingRecoveryUnresolvedTerminalError(GatewayLedgerError):
    """An ordinary terminal rejects recovery while its financial liability remains unresolved.

    No waiver or meter was applied. Only an independently verified matching
    settled job can retire the obsolete recovery owner; this outcome never
    authorizes provider reconstruction or infers a complete terminal meter.
    """


class AccountingRefusedError(GatewayLedgerError):
    """A deliberate, clean per-request refusal from the durable ledger.

    A host ledger raises this when it answers a write on purpose rather than
    failing to reach durable storage: for example when another ledger owns the
    organization's balance for now. The write did not land, but the ledger is
    healthy, so the refusal is a retryable error for this one request only and
    never counts as a settle failure: it fences no organization's admission.
    A refused terminal write stays owned by the recovery sweep and is replayed
    until it lands.
    """

    def __init__(self, message: str, *, retry_after_seconds: int | None = 1) -> None:
        """Retain the internal message and the advertised client wait.

        Args:
            message: Internal diagnostic message, never shown to callers.
            retry_after_seconds: Positive ``Retry-After`` for the caller, or
                ``None`` to advertise no wait.
        """
        if retry_after_seconds is not None and retry_after_seconds <= 0:
            raise ValueError("an accounting refusal's retry wait must be positive")
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class AttemptRejectedError(GatewayLedgerError):
    """A typed pre-dispatch rejection during authorization, acceptance or reservation.

    The rejected reservation wrote nothing durable, so the executor must not
    latch accounting health, must not dispatch a provider, and must not advance
    the fallback waterfall; the exception reaches the protocol boundary
    unchanged so the rejection keeps its own public error shape. ``failure`` is
    the sanitized failure that settles the already-accepted parent request.
    """

    def __init__(self, message: str, *, failure: GatewayFailure) -> None:
        """Retain the internal message and the sanitized settlement failure.

        Args:
            message: Internal diagnostic message, never shown to callers.
            failure: Sanitized failure persisted on the accepted request and,
                absent a more specific boundary mapping, shown to the caller.
        """
        super().__init__(message)
        self.failure = failure


class IdempotencyConflictError(AttemptRejectedError):
    """A caller operation key was reused with different canonical request content."""

    def __init__(self, message: str) -> None:
        """Retain the message with the canonical caller-error settlement shape."""
        super().__init__(
            message,
            failure=GatewayFailure(
                failure_class=GatewayFailureClass.INVALID_REQUEST,
                safe_message="caller operation key was reused with different request content",
            ),
        )


class IdempotencyReplayUnavailableError(AttemptRejectedError):
    """A completed or accepted keyed request exists but its content cannot be replayed."""

    def __init__(self, message: str) -> None:
        """Retain the message with the canonical replay-loss settlement shape."""
        super().__init__(
            message,
            failure=GatewayFailure(
                failure_class=GatewayFailureClass.INTERNAL,
                safe_message="completed keyed result is unavailable for durable replay",
            ),
        )
