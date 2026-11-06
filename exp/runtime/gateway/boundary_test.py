"""Boundary mapping for typed pre-dispatch ledger rejections."""

from __future__ import annotations

import logging

import pytest

from exp.runtime.gateway.boundary import boundary_protocol_error
from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.guardrails.contracts import (
    GuardrailAction,
    GuardrailRejected,
    guardrail_failure,
)
from exp.runtime.gateway.ledger import (
    AttemptRejectedError,
    IdempotencyConflictError,
    IdempotencyReplayUnavailableError,
)
from exp.runtime.gateway.sqlite.store import (
    AliasNotGrantedError,
    GatewayStoreError,
    InvalidVirtualKeyError,
)


def test_guardrail_rejection_maps_to_the_sanitized_public_failure() -> None:
    """A guardrail exception keeps its terminal content-filter shape."""
    error = boundary_protocol_error(
        GuardrailRejected(guardrail_failure(action=GuardrailAction.BLOCK, check_id="safety"))
    )
    assert error.status_code == 400
    assert error.detail.code == "content_filter"
    assert error.detail.message == "The request was blocked by a gateway guardrail."


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


def test_a_zdr_demand_the_local_gateway_cannot_judge_renders_as_a_403_on_provider_zdr() -> None:
    """The local store's refusal names the field and the reason, never a 500."""
    from exp.runtime.gateway.boundary import boundary_protocol_error
    from exp.runtime.gateway.sqlite.store import ZdrRoutingUnavailableError

    error = boundary_protocol_error(ZdrRoutingUnavailableError("no postures"))
    assert error.status_code == 403
    assert error.detail.code == "model_not_granted"
    assert error.detail.param == "provider.zdr"
    assert "provider.zdr" in error.detail.message


def test_bare_store_refusal_is_a_retryable_503_not_an_unnamed_400(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A control-store authority refusal names no caller field, so it is not a 400.

    Claude Code read these as "The gateway request is invalid" with no param
    (2026-10-17); the store's own reason belongs in the worker log.
    """
    with caplog.at_level(logging.WARNING, logger="exp.runtime.gateway.boundary"):
        error = boundary_protocol_error(
            GatewayStoreError("Model-chain request cannot cross a release or process generation")
        )
    assert error.status_code == 503
    assert error.detail.code == "gateway_authority_unavailable"
    assert error.detail.type == "api_error"
    assert error.retry_after_seconds == 5
    assert "release or process generation" not in error.detail.message
    assert "release or process generation" in caplog.text


def test_typed_store_refusals_keep_their_own_shapes() -> None:
    """Grant and key refusals still outrank the bare store branch."""
    assert boundary_protocol_error(AliasNotGrantedError("x")).status_code == 403
    assert boundary_protocol_error(InvalidVirtualKeyError("x")).status_code == 401


def test_an_unclassified_exception_is_a_logged_500(caplog: pytest.LogCaptureFixture) -> None:
    """The anonymous 500 names its exception type and raising frame in the worker log.

    2026-10-18: a host authorize raised OverflowError before admission; the
    caller saw internal_error with no request id and nothing was logged.
    """
    try:
        int(float("inf"))
    except OverflowError as overflow:
        with caplog.at_level(logging.ERROR, logger="exp.runtime.gateway.boundary"):
            error = boundary_protocol_error(overflow)
    assert error.status_code == 500
    assert error.detail.code == "internal_error"
    [record] = caplog.records
    assert "unclassified OverflowError raised at boundary_test.py:" in record.getMessage()
    assert record.__dict__["exception_type"] == "OverflowError"
    assert record.__dict__["raised_at"].endswith(
        "in test_an_unclassified_exception_is_a_logged_500"
    )
    # The exception's text may carry request content: never logged.
    assert record.exc_info is None
    assert "infinity" not in caplog.text


def test_a_typed_refusal_is_not_logged_as_unclassified(caplog: pytest.LogCaptureFixture) -> None:
    """Typed refusals keep their own shape and add no error log."""
    with caplog.at_level(logging.ERROR, logger="exp.runtime.gateway.boundary"):
        boundary_protocol_error(AliasNotGrantedError("not granted"))
    assert not caplog.records


def _psycopg_class(
    name: str, base: type[Exception], module: str, sqlstate: str | None
) -> type[Exception]:
    """A stand-in for a psycopg 3 error class (the engine does not depend on psycopg)."""
    return type(name, (base,), {"__module__": module, "sqlstate": sqlstate})


_PsycopgError = _psycopg_class("Error", Exception, "psycopg", None)
_OperationalError = _psycopg_class("OperationalError", _PsycopgError, "psycopg", None)
_PoolTimeout = _psycopg_class("PoolTimeout", _OperationalError, "psycopg_pool", None)


@pytest.mark.parametrize(
    "exception",
    [
        _psycopg_class("AdminShutdown", _OperationalError, "psycopg.errors", "57P01")(
            "terminating connection due to administrator command"
        ),
        _psycopg_class("CrashShutdown", _OperationalError, "psycopg.errors", "57P02")("crash"),
        _psycopg_class("CannotConnectNow", _OperationalError, "psycopg.errors", "57P03")("boot"),
        _psycopg_class("ConnectionFailure", _OperationalError, "psycopg.errors", "08006")("gone"),
        _OperationalError("consuming input failed: server closed the connection unexpectedly"),
        _PoolTimeout("couldn't get a connection after 30.00 sec"),
    ],
    ids=lambda exception: type(exception).__name__,
)
def test_a_lost_database_connection_is_a_retryable_503(
    exception: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    """A dropped Postgres connection is the retryable unavailable class, never a 500."""
    with caplog.at_level(logging.WARNING, logger="exp.runtime.gateway.boundary"):
        error = boundary_protocol_error(exception)
    assert error.status_code == 503
    assert error.detail.code == "gateway_unavailable"
    assert error.retry_after_seconds == 1
    assert "unclassified" not in caplog.text
    assert str(exception) not in caplog.text


def test_a_wrapped_lost_database_connection_is_a_retryable_503() -> None:
    """A host wrapper raised from a dropped connection keeps the retryable mapping."""
    shutdown = _psycopg_class("AdminShutdown", _OperationalError, "psycopg.errors", "57P01")()
    try:
        raise RuntimeError("authorize failed") from shutdown
    except RuntimeError as wrapped:
        error = boundary_protocol_error(wrapped)
    assert error.status_code == 503
    assert error.detail.code == "gateway_unavailable"


@pytest.mark.parametrize(
    ("name", "sqlstate"), [("QueryCanceled", "57014"), ("LockNotAvailable", "55P03")]
)
def test_other_operational_sqlstates_stay_unclassified(name: str, sqlstate: str) -> None:
    """A statement or lock timeout is not a lost connection and keeps the logged 500."""
    error = boundary_protocol_error(
        _psycopg_class(name, _OperationalError, "psycopg.errors", sqlstate)("timeout")
    )
    assert error.status_code == 500


def test_a_non_psycopg_operational_error_stays_unclassified() -> None:
    """Only psycopg's OperationalError is matched, not any class of that name."""
    foreign = _psycopg_class("OperationalError", Exception, "sqlite3", None)("disk I/O error")
    assert boundary_protocol_error(foreign).status_code == 500
