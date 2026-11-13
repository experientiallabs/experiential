"""Real SQLite lifecycle checks for the sole native financial owner.

Dispatch, refusal, valuation, recovery, concurrency and policy parity run inside
Rust beside the owner. These tests exercise its Python admission/lifecycle facade
against durable storage without substituting a Python financial callback.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.common.sqlite.native import connect
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting, NativeBridgeError
from exp.runtime.gateway.native_handoff_test import _prepared


@pytest.mark.parametrize("quoted", [False, True])
@pytest.mark.parametrize("recovery", ["retry", "sweep", "close"])
def test_failed_native_write_keeps_the_first_meter_until_durable_acknowledgement(
    tmp_path: Path, quoted: bool, recovery: str
) -> None:
    """Quote, cancellation and altered replays cannot replace a retained completed meter."""
    factory, entry, ledger = _prepared(tmp_path)
    owner = NativeAttemptAccounting(native_request_factory=factory, background_recovery=False)
    owner.register(entry)
    request_id = str(entry.authorization.request_id)
    started = json.loads(
        owner.start_attempt(json.dumps({"request_id": request_id, "attempt_ordinal": 0}))
    )
    terminal = {
        "request_id": request_id,
        "attempt_id": started["attempt_id"],
        "outcome": "completed",
        "finalize": True,
        "opened": True,
        "usage": {"input_tokens": 37, "output_tokens": 5, "cached_input_tokens": 3},
    }
    try:
        with connect(ledger.database_path) as db:
            assert db.execute("SELECT state FROM gateway_attempts").fetchone() == ("dispatched",)
            db.execute(
                "CREATE TRIGGER retain_terminal BEFORE UPDATE ON gateway_attempts "
                "BEGIN SELECT RAISE(ABORT, 'synthetic write failure'); END"
            )
        if quoted:
            owner.price(json.dumps(terminal))
            assert owner.accounting_healthy
            with connect(ledger.database_path) as db:
                assert db.execute("SELECT state FROM gateway_attempts").fetchone() == (
                    "dispatched",
                )
        with pytest.raises(NativeBridgeError):
            owner.settle(json.dumps(terminal))
        assert not owner.accounting_healthy
        assert owner.native.owns_request(request_id)
        assert not owner.close(0)
        with connect(ledger.database_path) as db:
            db.execute("DROP TRIGGER retain_terminal")
        match recovery:
            case "retry":
                changed = terminal | {"usage": {"input_tokens": 999, "output_tokens": 99}}
                owner.settle(json.dumps(changed))
            case "sweep":
                owner.sweep_expired()
            case "close":
                assert owner.close(2)
        assert owner.accounting_healthy
        with connect(ledger.database_path) as db:
            expected = ("completed", 37, 5, 3)
            assert (
                db.execute(
                    "SELECT state,input_tokens,output_tokens,cached_input_tokens "
                    "FROM gateway_attempts"
                ).fetchone()
                == expected
            )
            assert db.execute("SELECT terminal_state FROM gateway_requests").fetchone() == (
                "completed",
            )
        owner.settle(json.dumps(terminal))
        with connect(ledger.database_path) as db:
            assert db.execute("SELECT count(*) FROM gateway_attempts").fetchone() == (1,)
            assert (
                db.execute(
                    "SELECT state,input_tokens,output_tokens,cached_input_tokens "
                    "FROM gateway_attempts"
                ).fetchone()
                == expected
            )
    finally:
        with connect(ledger.database_path) as db:
            db.execute("DROP TRIGGER IF EXISTS retain_terminal")
        assert owner.close(2)
        ledger.close()


@pytest.mark.parametrize("failure", [None, "cancelled", "transport"])
@pytest.mark.parametrize("finalize", [False, True])
def test_native_terminal_metadata_survives_financial_retirement(
    tmp_path: Path, failure: str | None, finalize: bool
) -> None:
    """The immutable wire meter keeps its metadata across terminal state and request cleanup."""
    factory, entry, ledger = _prepared(tmp_path)
    owner = NativeAttemptAccounting(native_request_factory=factory, background_recovery=False)
    owner.register(entry)
    request_id = str(entry.authorization.request_id)
    started = json.loads(
        owner.start_attempt(json.dumps({"request_id": request_id, "attempt_ordinal": 0}))
    )
    terminal = {
        "request_id": request_id,
        "attempt_id": started["attempt_id"],
        "outcome": "completed" if failure is None else "failed",
        "failure": None
        if failure is None
        else {"failure_class": failure, "safe_message": "fixture terminal"},
        "finalize": finalize,
        "opened": True,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 5,
            "cached_input_tokens": 20,
            "cache_creation_input_tokens": 10,
            "cache_creation_1h_input_tokens": 2,
        },
        "first_token_at": "2026-09-18T01:02:03+00:00",
        "upstream_provider": "upstream",
    }
    try:
        owner.settle(json.dumps(terminal))
        owner.abandon(json.dumps({"request_id": request_id}))
        assert owner.close(2)
        with connect(ledger.database_path) as db:
            state = (
                "completed"
                if failure is None
                else "cancelled"
                if failure == "cancelled"
                else "failed"
            )
            assert db.execute(
                "SELECT state,input_tokens,output_tokens,cached_input_tokens,"
                "cache_creation_input_tokens,cache_creation_1h_input_tokens "
                "FROM gateway_attempts"
            ).fetchone() == (state, 100, 5, 20, 10, 2)
            assert db.execute("SELECT count(*) FROM gateway_attempts").fetchone() == (1,)
    finally:
        assert owner.close(2)
        ledger.close()


def test_native_local_price_defers_durable_terminal_until_settlement(
    tmp_path: Path,
) -> None:
    """A quote retains its meter without writing; later settlement and replays charge once."""
    factory, entry, ledger = _prepared(tmp_path)
    owner = NativeAttemptAccounting(native_request_factory=factory, background_recovery=False)
    owner.register(entry)
    request_id = str(entry.authorization.request_id)
    started = json.loads(
        owner.start_attempt(json.dumps({"request_id": request_id, "attempt_ordinal": 0}))
    )
    terminal = {
        "request_id": request_id,
        "attempt_id": started["attempt_id"],
        "outcome": "completed",
        "finalize": True,
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }
    try:
        owner.price(json.dumps(terminal))
        with connect(ledger.database_path) as db:
            assert db.execute(
                "SELECT state,input_tokens,output_tokens,budget_settled_nano_usd "
                "FROM gateway_attempts"
            ).fetchone() == ("dispatched", None, None, None)
            assert db.execute("SELECT terminal_state FROM gateway_requests").fetchone() == (None,)
        owner.settle(json.dumps(terminal))
        with connect(ledger.database_path) as db:
            row = db.execute(
                "SELECT state,input_tokens,output_tokens,budget_settled_nano_usd "
                "FROM gateway_attempts"
            ).fetchone()
            assert row is not None and row[:3] == ("completed", 100, 10)
        owner.settle(json.dumps(terminal))
        with connect(ledger.database_path) as db:
            assert (
                db.execute(
                    "SELECT state,input_tokens,output_tokens,budget_settled_nano_usd "
                    "FROM gateway_attempts"
                ).fetchone()
                == row
            )
            assert db.execute("SELECT count(*) FROM gateway_attempts").fetchone() == (1,)
    finally:
        assert owner.close(2)
        ledger.close()
