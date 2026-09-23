"""Typed content-free ledger transition and caller replay failures."""

from exp.runtime.gateway.stream_contracts import GatewayFailure, GatewayFailureClass


class GatewayLedgerError(ValueError):
    """A request or attempt transition violates the content-free ledger contract."""


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
