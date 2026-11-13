"""Native request ownership against the real local ledger schema and authority."""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from exp_gateway_native import NativeAccounting, NativeSQLiteLedger

from exp.common.core.artifacts import JsonObject
from exp.common.sqlite.native import connect
from exp.runtime.gateway.budgets import BudgetScope, BudgetScopeKind, SQLiteBudgetStore
from exp.runtime.gateway.contracts import AuthorizationSnapshot
from exp.runtime.gateway.ledger_test import (
    FakeLedgerClock,
    _authority_fixture,
    _deployment,
    _execution,
    _request,
)
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.native_registration import registration_json
from exp.runtime.gateway.native_sqlite import sqlite_request
from exp.runtime.gateway.routing import GatewayRoute


def _owner(
    tmp_path: Path, *, budget: int = 1_000_000
) -> tuple[NativeAccounting, AuthorizationSnapshot]:
    """Compose a native owner using real durable authority and a bounded monthly budget."""
    clock = FakeLedgerClock()
    clock.wall = datetime.now(UTC)
    clock.monotonic_value = time.monotonic()
    store, ledger, key = _authority_fixture(tmp_path, clock)
    budgets = SQLiteBudgetStore(store.database_path, clock=clock)
    budgets.set_limit(
        organization_id="org-one",
        period=clock.wall.strftime("%Y-%m"),
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_nano_usd=budget,
    )
    request = _request("private request content").model_copy(update={"maximum_output_tokens": 100})
    authorization = store.authorize_request(
        raw_key=key,
        alias="coding",
        request=request,
        deadline_monotonic=time.monotonic() + 60,
    )
    native = NativeSQLiteLedger(ledger.database_path)
    native.accept(authorization.model_dump_json(), 60)
    entry = InflightRequest(
        authorization=authorization,
        route=GatewayRoute(
            snapshot=_execution(authorization),
            deployment=_deployment(),
            fallback_deployments=(),
            route_reason="direct",
        ),
        request=request,
        deadline_monotonic=time.monotonic() + 60,
    )
    owner = NativeAccounting(background_recovery=False)
    owner.register(registration_json(entry, bind_sticky=True), sqlite_request(native, entry))
    return owner, authorization


def _call(
    owner: NativeAccounting, auth: AuthorizationSnapshot, method: str, **data: object
) -> JsonObject:
    """Drive the same native registry callback the serving bridge retains."""
    return json.loads(owner.call(method, json.dumps({"request_id": auth.request_id, **data})))


def _row(tmp_path: Path, table: str) -> sqlite3.Row:
    """Inspect committed state independently of the retained writer connection."""
    with connect(tmp_path / "gateway.db") as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(f"SELECT * FROM {table}").fetchone()


def test_native_sqlite_settles_exact_usage_and_releases_every_hold(tmp_path: Path) -> None:
    """One interpreter-independent request leaves a durable, content-free terminal."""
    owner, auth = _owner(tmp_path)
    try:
        started = _call(owner, auth, "start_attempt", attempt_ordinal=0)
        _call(
            owner,
            auth,
            "settle",
            attempt_id=started["attempt_id"],
            outcome="completed",
            finalize=True,
            usage={"input_tokens": 100, "output_tokens": 50, "cached_input_tokens": 20},
        )
        attempt = _row(tmp_path, "gateway_attempts")
        budget = _row(tmp_path, "gateway_monthly_budgets")
        assert attempt["state"] == "completed"
        assert attempt["estimated_cost_nano_usd"] == 380
        assert budget["reserved_nano_usd"] == 0
        assert budget["settled_nano_usd"] == 380
        assert budget["unknown_cost_attempts"] == 0
        assert "private request content" not in attempt["native_accounting_terminal"]
        assert "private request content" not in attempt["native_accounting_terms"]
        assert owner.drained()
    finally:
        assert owner.close()


def test_native_sqlite_price_is_read_only_until_the_terminal_is_settled(tmp_path: Path) -> None:
    """Quoting leaves durable delivery state and budget holds unchanged until settlement."""
    owner, auth = _owner(tmp_path)
    try:
        started = _call(owner, auth, "start_attempt", attempt_ordinal=0)
        terminal = {
            "attempt_id": started["attempt_id"],
            "outcome": "completed",
            "finalize": True,
            "usage": {"input_tokens": 100, "output_tokens": 50, "cached_input_tokens": 20},
        }
        tables = ("gateway_attempts", "gateway_requests", "gateway_monthly_budgets")
        before = {table: dict(_row(tmp_path, table)) for table in tables}
        assert before["gateway_monthly_budgets"]["reserved_nano_usd"] > 0

        _call(owner, auth, "price", **terminal)
        assert {table: dict(_row(tmp_path, table)) for table in tables} == before
        assert not owner.drained()
        with connect(tmp_path / "gateway.db") as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                _call(owner, auth, "price", **terminal)
            finally:
                connection.rollback()

        _call(owner, auth, "settle", **terminal)
        assert _row(tmp_path, "gateway_attempts")["state"] == "completed"
        assert _row(tmp_path, "gateway_requests")["terminal_state"] == "completed"
        budget = _row(tmp_path, "gateway_monthly_budgets")
        assert budget["reserved_nano_usd"] == 0
        assert budget["settled_nano_usd"] == 380
        assert owner.drained()
    finally:
        assert owner.close()


def test_native_sqlite_rejection_does_not_leave_an_attempt_or_budget_hold(tmp_path: Path) -> None:
    """Request-wide quota rejection leaves no partial reservation to recover."""
    owner, auth = _owner(tmp_path, budget=0)
    try:
        with pytest.raises(ValueError, match="[Bb]udget|[Aa]llocation"):
            _call(owner, auth, "start_attempt", attempt_ordinal=0)
        with connect(tmp_path / "gateway.db") as connection:
            assert connection.execute("SELECT count(*) FROM gateway_attempts").fetchone() == (0,)
        assert _row(tmp_path, "gateway_monthly_budgets")["reserved_nano_usd"] == 0
        assert _row(tmp_path, "gateway_requests")["terminal_state"] == "failed"
        assert owner.drained()
    finally:
        assert owner.close()


def test_native_sqlite_cancellation_retains_then_reconciles_late_meter(tmp_path: Path) -> None:
    """A late exact meter changes financial evidence without replacing cancellation."""
    owner, auth = _owner(tmp_path)
    try:
        started = _call(owner, auth, "start_attempt", attempt_ordinal=0)
        _call(owner, auth, "abandon")
        assert _row(tmp_path, "gateway_attempts")["state"] == "cancelled"
        recovered = NativeSQLiteLedger(tmp_path / "gateway.db")
        recovered.settle_existing(
            auth.model_dump_json(),
            str(started["attempt_id"]),
            json.dumps(
                {
                    "state": "completed",
                    "finalize": True,
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                }
            ),
        )
        attempt = _row(tmp_path, "gateway_attempts")
        assert attempt["state"] == "cancelled"
        assert attempt["usage_source"] == "observed"
        assert attempt["estimated_cost_nano_usd"] == 400
        assert _row(tmp_path, "gateway_monthly_budgets")["settled_nano_usd"] == 400
    finally:
        assert owner.close()


@pytest.mark.parametrize("change", ["populated", "symlink"])
def test_native_sqlite_rechecks_live_snapshot_before_reservation(
    tmp_path: Path, change: str
) -> None:
    """Post-admission file changes never bypass local model-chain enforcement."""
    owner, auth = _owner(tmp_path)
    try:
        if change == "populated":
            (tmp_path / "snapshot-one").write_text('{"model_chains":{"hidden":{}}}')
        else:
            (tmp_path / "outside.json").write_text("{}")
            (tmp_path / "snapshot-one").symlink_to(tmp_path / "outside.json")
        with pytest.raises(ValueError):
            _call(owner, auth, "start_attempt", attempt_ordinal=0)
        with connect(tmp_path / "gateway.db") as connection:
            assert connection.execute("SELECT count(*) FROM gateway_attempts").fetchone() == (0,)
    finally:
        assert owner.close()
