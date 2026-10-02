"""Tests for durable request certificates and migration of unknown historical work."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from exp.common.sqlite.connection import close_idle_connections
from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass, ProjectTarget
from exp.runtime.gateway.ledger import IdempotencyReplayUnavailableError, SQLiteAttemptLedger
from exp.runtime.gateway.ledger_test import FakeLedgerClock, _authority_fixture, _request


def test_historical_zero_attempt_failure_migrates_without_a_certificate(tmp_path: Path) -> None:
    """Schema migration never infers freedom from cost or absence of model attempts."""
    clock = FakeLedgerClock()
    store, ledger, key = _authority_fixture(tmp_path, clock)
    request = _request("same", idempotency_key="historical-work")
    original = store.authorize_request(
        raw_key=key,
        alias="coding",
        request=request,
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=original)
    failure = GatewayFailure(failure_class=GatewayFailureClass.THROTTLED, safe_message="busy")
    ledger.finish_request(authorization=original, failure=failure)
    close_idle_connections()
    with sqlite3.connect(tmp_path / "gateway.db") as connection:
        connection.execute("ALTER TABLE gateway_requests DROP COLUMN failed_without_effects")
        connection.execute("PRAGMA user_version = 25")
    restored = SQLiteAttemptLedger(tmp_path / "gateway.db", clock=clock)
    assert (
        restored.finish_request(authorization=original, failure=failure, certify_no_effects=True)
        is False
    )
    retry = store.authorize_request(
        raw_key=key,
        alias="coding",
        request=request,
        deadline_monotonic=clock.monotonic() + 30,
    )
    with pytest.raises(IdempotencyReplayUnavailableError):
        restored.accept_request(authorization=retry)
    with sqlite3.connect(tmp_path / "gateway.db") as connection:
        assert connection.execute(
            "SELECT failed_without_effects FROM gateway_requests"
        ).fetchone() == (0,)


@pytest.mark.parametrize("nonfailed", [False, True])
def test_certificate_requires_direct_target_and_failed_terminal_state(
    tmp_path: Path, nonfailed: bool
) -> None:
    """Even an explicit attestation cannot certify routing or another terminal outcome."""
    clock = FakeLedgerClock()
    store, ledger, key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=key,
        alias="coding",
        request=_request("same"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    if not nonfailed:
        authorization = authorization.model_copy(
            update={
                "target": ProjectTarget(
                    project_ref="saved-project",
                    activation_ref="activation",
                    catalog_sha256="a" * 64,
                )
            }
        )
    ledger.accept_request(authorization=authorization)
    assert (
        ledger.finish_request(
            authorization=authorization,
            failure=GatewayFailure(
                failure_class=GatewayFailureClass.CANCELLED
                if nonfailed
                else GatewayFailureClass.THROTTLED,
                safe_message="stopped",
            ),
            certify_no_effects=True,
        )
        is False
    )
