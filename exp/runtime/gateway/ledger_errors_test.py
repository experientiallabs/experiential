"""Ledger error extraction keeps the existing public exception identities."""

from exp.runtime.gateway import ledger, ledger_errors
from exp.runtime.gateway.contracts import GatewayFailureClass


def test_public_ledger_errors_are_the_same_owned_classes() -> None:
    """Existing callers catch the exact classes now owned by the focused error module."""
    for name in (
        "GatewayLedgerError",
        "AttemptRejectedError",
        "IdempotencyConflictError",
        "IdempotencyReplayUnavailableError",
    ):
        assert getattr(ledger, name) is getattr(ledger_errors, name)
    conflict = ledger.IdempotencyConflictError("fixture")
    assert isinstance(conflict, ledger.AttemptRejectedError)
    assert conflict.failure.failure_class == GatewayFailureClass.INVALID_REQUEST
    unavailable = ledger.IdempotencyReplayUnavailableError("fixture")
    assert unavailable.failure.failure_class == GatewayFailureClass.INTERNAL
