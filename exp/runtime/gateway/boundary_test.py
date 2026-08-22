"""Boundary mapping for typed pre-dispatch ledger rejections."""

from __future__ import annotations

from exp.runtime.gateway.boundary import boundary_protocol_error
from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.ledger import (
    AttemptRejectedError,
    IdempotencyConflictError,
    IdempotencyReplayUnavailableError,
)


def test_idempotency_conflict_keeps_409_ahead_of_generic_rejection_mapping() -> None:
    """The specific conflict branch outranks the generic typed-rejection branch."""
    error = boundary_protocol_error(IdempotencyConflictError("reused operation"))
    assert error.status_code == 409
    assert error.detail.code == "idempotency_conflict"
    assert error.detail.param == "Idempotency-Key"


def test_idempotency_replay_unavailable_keeps_409() -> None:
    """The replay-loss branch stays 409 after re-parenting under the typed rejection."""
    error = boundary_protocol_error(IdempotencyReplayUnavailableError("replay lost"))
    assert error.status_code == 409
    assert error.detail.code == "idempotency_replay_unavailable"


def test_generic_typed_rejection_maps_through_its_own_failure_shape() -> None:
    """A ledger-assigned rejection shape reaches the caller unreshaped."""
    error = boundary_protocol_error(
        AttemptRejectedError(
            "virtual key was revoked between accept and dispatch",
            failure=GatewayFailure(
                failure_class=GatewayFailureClass.AUTHENTICATION,
                safe_message="the gateway key is invalid, expired, or revoked",
            ),
        )
    )
    assert error.status_code == 401
    assert error.detail.code == "invalid_key"
    assert error.detail.type == "authentication_error"
    assert error.detail.message == "the gateway key is invalid, expired, or revoked"
