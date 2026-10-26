"""Real SQLite recovery of authoritative usage after provisional terminal writes."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pytest

from exp.common.models.catalog import GatewayDeploymentMetadata, GatewayTokenPrices
from exp.common.models.catalog_prices import BilledUnitKind, GatewayUnitPrices
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.budgets import BudgetScope, BudgetScopeKind, SQLiteBudgetStore
from exp.runtime.gateway.contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayUsage,
)
from exp.runtime.gateway.group_commit import GroupCommitAttemptLedger, SyncGroupCommitLedger
from exp.runtime.gateway.ledger import GatewayLedgerError, SQLiteAttemptLedger
from exp.runtime.gateway.ledger_test import (
    FakeLedgerClock,
    _attempt_row,
    _authority_fixture,
    _deployment,
    _execution,
    _request,
)
from exp.runtime.gateway.service_tiers import GatewayServiceTierAdmission
from exp.runtime.gateway.sqlite.store import SQLiteGatewayStore
from exp.runtime.gateway.stream_contracts import BilledUnits

_SCOPES = (
    BudgetScope(kind=BudgetScopeKind.TEAM),
    BudgetScope(kind=BudgetScopeKind.IDENTITY, identity_id="identity-one"),
)
_CANCELLED = GatewayFailure(
    failure_class=GatewayFailureClass.CANCELLED, safe_message="caller disconnected"
)


def _started(
    tmp_path: Path,
    *,
    maximum: int | None = 300,
    deployment: ExactModelDeployment | None = None,
    tier: GatewayServiceTierAdmission | None = None,
) -> tuple[FakeLedgerClock, SQLiteGatewayStore, SQLiteAttemptLedger, SQLiteBudgetStore, str]:
    clock = FakeLedgerClock()
    store, ledger, key = _authority_fixture(tmp_path, clock)
    budgets = SQLiteBudgetStore(store.database_path, clock=clock)
    for period in ("2026-08", "2026-09"):
        for scope in _SCOPES:
            budgets.set_limit(
                organization_id="org-one", period=period, scope=scope, limit_nano_usd=10_000
            )
    auth = store.authorize_request(
        raw_key=key,
        alias="coding",
        request=_request("late meter"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=auth)
    attempt = ledger.start_attempt(
        snapshot=_execution(auth),
        deployment=deployment or _deployment(),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_nano_usd=maximum,
        service_tier=tier,
    )
    return clock, store, ledger, budgets, attempt


def _cancelled(*, estimated: bool = False, usage: GatewayUsage | None = None) -> GatewayEvent:
    return GatewayEvent(
        kind=GatewayEventKind.FAILED,
        sequence_number=0,
        failure=_CANCELLED,
        usage=usage,
        usage_estimated=estimated,
        usage_incomplete_due_to_disconnect=True,
    )


def _observed(*, input_tokens: int = 10, output_tokens: int = 5) -> GatewayEvent:
    return GatewayEvent(
        kind=GatewayEventKind.COMPLETED,
        sequence_number=1,
        usage=GatewayUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _assert_allocations(budgets: SQLiteBudgetStore, *, settled: int, reserved: int = 0) -> None:
    original = budgets.remaining(organization_id="org-one", period="2026-08")
    assert len(original) == 2
    for allocation in original:
        assert allocation.settled_nano_usd == settled
        assert allocation.reserved_nano_usd == reserved
        assert allocation.unknown_cost_attempts == 0
    for allocation in budgets.remaining(organization_id="org-one", period="2026-09"):
        assert allocation.settled_nano_usd == 0
        assert allocation.reserved_nano_usd == 0
        assert allocation.unknown_cost_attempts == 0


@pytest.mark.parametrize("group_commit", [False, True])
@pytest.mark.parametrize("estimated", [False, True])
@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("tokens,expected", [((10, 5), 40), ((200, 50), 600)])
def test_late_observed_meter_replaces_provisional_billing_once(
    tmp_path: Path,
    group_commit: bool,
    estimated: bool,
    failed: bool,
    tokens: tuple[int, int],
    expected: int,
) -> None:
    """Unknown/estimated cancellation commits first; exact late usage changes only billing."""
    clock, store, ledger, budgets, attempt = _started(tmp_path)
    initial = _cancelled(
        estimated=estimated,
        usage=GatewayUsage(input_tokens=19, output_tokens=7) if estimated else None,
    )
    ledger.finish_attempt(attempt_id=attempt, terminal_event=initial, failure=_CANCELLED)
    before = dict(_attempt_row(tmp_path, attempt))
    _assert_allocations(budgets, settled=66 if estimated else 300)
    clock.advance(32 * 24 * 60 * 60)
    reopened = SQLiteAttemptLedger(store.database_path, clock=clock)
    terminal = _observed(input_tokens=tokens[0], output_tokens=tokens[1])
    if failed:
        terminal = terminal.model_copy(
            update={"kind": GatewayEventKind.FAILED, "failure": _CANCELLED}
        )
    writer = GroupCommitAttemptLedger(reopened) if group_commit else None
    sink = reopened if writer is None else SyncGroupCommitLedger(writer)
    try:
        for _ in range(2):
            sink.finish_attempt(
                attempt_id=attempt, terminal_event=terminal, failure=terminal.failure
            )
        # A delayed retry of the old estimate cannot downgrade the final meter.
        sink.finish_attempt(attempt_id=attempt, terminal_event=initial, failure=_CANCELLED)
        with pytest.raises(GatewayLedgerError, match="durable usage"):
            sink.finish_attempt(
                attempt_id=attempt,
                terminal_event=_observed(input_tokens=tokens[0] + 1, output_tokens=tokens[1]),
                failure=None,
            )
    finally:
        if writer is not None:
            writer.close()
    after = _attempt_row(tmp_path, attempt)
    for field in (
        "state",
        "terminal_at",
        "failure_class",
        "failure_message",
        "budget_period_start",
    ):
        assert after[field] == before[field]
    assert after["state"] == "cancelled"
    assert (after["input_tokens"], after["output_tokens"], after["usage_source"]) == (
        *tokens,
        "observed",
    )
    assert after["estimated_cost_nano_usd"] == expected
    assert after["budget_settled_nano_usd"] == expected
    _assert_allocations(budgets, settled=expected)
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute(
            "SELECT terminal_state FROM gateway_requests WHERE request_id = ?",
            (after["request_id"],),
        ).fetchone() == ("cancelled",)
        assert connection.execute(
            "SELECT reserved_nano_usd, settled_nano_usd FROM gateway_attempt_budget_charges "
            "WHERE attempt_id = ?",
            (attempt,),
        ).fetchall() == [(300, expected), (300, expected)]
    usage = reopened.usage(organization_id="org-one")
    assert usage[0].known_estimated_cost_nano_usd == expected
    assert usage[0].unknown_cost_attempts == 0


def test_crash_hold_reconciles_without_rewriting_delivery(tmp_path: Path) -> None:
    """Crash recovery keeps the bound until a complete meter commits to both allocations."""
    clock, store, ledger, budgets, attempt = _started(tmp_path)
    clock.advance(60)
    assert ledger.reconcile_crashed_requests(cleanup_grace=timedelta()) == (0, 1)
    before = dict(_attempt_row(tmp_path, attempt))
    _assert_allocations(budgets, settled=0, reserved=300)
    reopened = SQLiteAttemptLedger(store.database_path, clock=clock)
    for _ in range(2):
        reopened.finish_attempt(attempt_id=attempt, terminal_event=_observed(), failure=None)
    after = _attempt_row(tmp_path, attempt)
    assert after["state"] == "unknown_after_crash"
    assert after["terminal_at"] == before["terminal_at"]
    assert after["failure_class"] == before["failure_class"]
    assert after["usage_source"] == "observed"
    assert after["estimated_cost_nano_usd"] == 40
    _assert_allocations(budgets, settled=40)
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute(
            "SELECT terminal_state FROM gateway_requests WHERE request_id = ?",
            (after["request_id"],),
        ).fetchone() == ("unknown_after_crash",)


def test_unit_only_late_meter_reconciles_video_liability(tmp_path: Path) -> None:
    """A video's late exact seconds replace the provisional charge without invented tokens."""
    deployment = _deployment().model_copy(
        update={
            "gateway": GatewayDeploymentMetadata(
                prices=GatewayTokenPrices(
                    units=GatewayUnitPrices(
                        kind=BilledUnitKind.VIDEO_SECOND,
                        rates={"720p": 50},
                    )
                ),
                pricing_source="operator-authored",
            )
        }
    )
    _, _, ledger, budgets, attempt = _started(tmp_path, deployment=deployment)
    ledger.finish_attempt(attempt_id=attempt, terminal_event=_cancelled(), failure=_CANCELLED)
    terminal = GatewayEvent(
        kind=GatewayEventKind.COMPLETED,
        sequence_number=1,
        usage=GatewayUsage(
            billed_units=BilledUnits(
                kind=BilledUnitKind.VIDEO_SECOND,
                variant="720p",
                quantity_milli=5_000,
            )
        ),
    )
    for _ in range(2):
        ledger.finish_attempt(attempt_id=attempt, terminal_event=terminal, failure=None)
    row = _attempt_row(tmp_path, attempt)
    assert row["input_tokens"] is None
    assert row["output_tokens"] is None
    assert row["usage_source"] == "observed"
    assert row["billed_unit_kind"] == "video_second"
    assert row["billed_unit_variant"] == "720p"
    assert row["billed_quantity_milli"] == 5_000
    assert row["billed_unit_rate"] == 50
    assert row["estimated_cost_nano_usd"] == 250
    _assert_allocations(budgets, settled=250)


@pytest.mark.parametrize("zero_estimate", [False, True])
@pytest.mark.parametrize("refund", [False, True])
def test_late_meter_preserves_refunds_and_ambiguous_zero_charges(
    tmp_path: Path,
    zero_estimate: bool,
    refund: bool,
) -> None:
    """Legacy zero allocations lack refund provenance; retain them but record exact cost."""
    _, _, ledger, budgets, attempt = _started(tmp_path)
    failure = (
        _CANCELLED.model_copy(update={"safe_details": {"input_guardrail_denied": True}})
        if refund
        else _CANCELLED
    )
    initial = _cancelled(
        estimated=True,
        usage=GatewayUsage(
            input_tokens=0 if zero_estimate else 19,
            output_tokens=0 if zero_estimate else 7,
        ),
    )
    ledger.finish_attempt(attempt_id=attempt, terminal_event=initial, failure=failure)
    for _ in range(2):
        ledger.finish_attempt(attempt_id=attempt, terminal_event=_observed(), failure=None)
    row = _attempt_row(tmp_path, attempt)
    assert row["usage_source"] == "observed"
    assert row["estimated_cost_nano_usd"] == 40
    expected = 0 if refund or zero_estimate else 40
    assert row["budget_settled_nano_usd"] == expected
    _assert_allocations(budgets, settled=expected)


def test_scope_operator_assignment_survives_late_natural_settlement(tmp_path: Path) -> None:
    """An explicit unknown-cost assignment applies only to its chosen allocation."""
    _, _, ledger, budgets, attempt = _started(tmp_path, maximum=None)
    ledger.finish_attempt(attempt_id=attempt, terminal_event=_cancelled(), failure=_CANCELLED)
    count, _ = budgets.reconcile_unknown_costs(
        organization_id="org-one",
        period="2026-08",
        scope=_SCOPES[0],
        assigned_cost_nano_usd=7,
    )
    assert count == 1
    for _ in range(2):
        ledger.finish_attempt(attempt_id=attempt, terminal_event=_observed(), failure=None)
    row = _attempt_row(tmp_path, attempt)
    assert row["estimated_cost_nano_usd"] == 40
    assert row["budget_settled_nano_usd"] == 40
    with sqlite3.connect(tmp_path / "gateway.db") as connection:
        assert connection.execute(
            "SELECT b.scope_kind, c.settled_nano_usd, b.settled_nano_usd, "
            "b.unknown_cost_attempts FROM gateway_attempt_budget_charges AS c "
            "JOIN gateway_monthly_budgets AS b ON b.budget_id = c.budget_id "
            "WHERE c.attempt_id = ? ORDER BY b.scope_kind",
            (attempt,),
        ).fetchall() == [("identity", 40, 40, 0), ("team", 7, 7, 0)]


def test_reconciliation_failure_rolls_back_meter_and_all_budget_deltas(tmp_path: Path) -> None:
    """A counter invariant failure cannot acknowledge usage while dropping its money write."""
    _, store, ledger, _, attempt = _started(tmp_path)
    ledger.finish_attempt(attempt_id=attempt, terminal_event=_cancelled(), failure=_CANCELLED)
    before = dict(_attempt_row(tmp_path, attempt))
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE gateway_monthly_budgets SET settled_nano_usd = 0 WHERE scope_kind = 'identity'"
        )
        counters = connection.execute(
            "SELECT budget_id, settled_nano_usd, reserved_nano_usd FROM gateway_monthly_budgets "
            "ORDER BY budget_id"
        ).fetchall()
    with pytest.raises(GatewayLedgerError, match="integer capacity|inconsistent"):
        ledger.finish_attempt(attempt_id=attempt, terminal_event=_observed(), failure=None)
    assert dict(_attempt_row(tmp_path, attempt)) == before
    with sqlite3.connect(store.database_path) as connection:
        assert (
            connection.execute(
                "SELECT budget_id, settled_nano_usd, reserved_nano_usd "
                "FROM gateway_monthly_budgets "
                "ORDER BY budget_id"
            ).fetchall()
            == counters
        )
        assert connection.execute(
            "SELECT settled_nano_usd FROM gateway_attempt_budget_charges WHERE attempt_id = ?",
            (attempt,),
        ).fetchall() == [(300,), (300,)]


def test_two_sqlite_owners_commit_identical_late_meter_once(tmp_path: Path) -> None:
    """Independent recovery owners serialize the replacement instead of charging twice."""
    clock, store, ledger, budgets, attempt = _started(tmp_path)
    ledger.finish_attempt(attempt_id=attempt, terminal_event=_cancelled(), failure=_CANCELLED)
    owners = (ledger, SQLiteAttemptLedger(store.database_path, clock=clock))
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                owner.finish_attempt,
                attempt_id=attempt,
                terminal_event=_observed(),
                failure=None,
            )
            for owner in owners
        ]
        for future in futures:
            future.result(timeout=5)
    _assert_allocations(budgets, settled=40)


def test_late_estimate_cannot_replace_unknown_or_observed_evidence(tmp_path: Path) -> None:
    """Only the provider's complete observation can replace an already terminal liability."""
    _, _, ledger, budgets, attempt = _started(tmp_path)
    ledger.finish_attempt(attempt_id=attempt, terminal_event=_cancelled(), failure=_CANCELLED)
    before = dict(_attempt_row(tmp_path, attempt))
    estimate = _cancelled(estimated=True, usage=GatewayUsage(input_tokens=19, output_tokens=7))
    ledger.finish_attempt(attempt_id=attempt, terminal_event=estimate, failure=_CANCELLED)
    assert dict(_attempt_row(tmp_path, attempt)) == before
    _assert_allocations(budgets, settled=300)
    ledger.finish_attempt(attempt_id=attempt, terminal_event=_observed(), failure=None)
    observed = dict(_attempt_row(tmp_path, attempt))
    ledger.finish_attempt(attempt_id=attempt, terminal_event=estimate, failure=_CANCELLED)
    assert dict(_attempt_row(tmp_path, attempt)) == observed
    _assert_allocations(budgets, settled=40)


@pytest.mark.parametrize("refund", [False, True])
def test_late_meter_preserves_confirmed_tier_and_charge_disposition(
    tmp_path: Path,
    refund: bool,
) -> None:
    """A confirmed schedule survives an omitted replay receipt and a zero-charge refund."""
    tier = GatewayServiceTierAdmission(
        requested="priority",
        standard_prices=GatewayTokenPrices(
            input_nano_usd_per_million_tokens=1_000_000,
            output_nano_usd_per_million_tokens=2_000_000,
        ),
        requested_prices=GatewayTokenPrices(
            input_nano_usd_per_million_tokens=2_000_000,
            output_nano_usd_per_million_tokens=4_000_000,
        ),
    )
    _, _, ledger, budgets, attempt = _started(tmp_path, tier=tier)
    receipt = tier.settlement(served="default", resolution="confirmed")
    failure = (
        _CANCELLED.model_copy(update={"safe_details": {"input_guardrail_denied": True}})
        if refund
        else _CANCELLED
    )
    initial = _cancelled(estimated=True, usage=GatewayUsage(input_tokens=19, output_tokens=7))
    ledger.finish_attempt(
        attempt_id=attempt, terminal_event=initial, failure=failure, service_tier=receipt
    )
    for supplied in (None, receipt, None):
        ledger.finish_attempt(
            attempt_id=attempt, terminal_event=_observed(), failure=None, service_tier=supplied
        )
    row = _attempt_row(tmp_path, attempt)
    assert row["estimated_cost_nano_usd"] == 20
    assert row["service_tier_settlement"] == receipt.model_dump_json()
    _assert_allocations(budgets, settled=0 if refund else 20)
    with pytest.raises(GatewayLedgerError, match="settled tier"):
        ledger.finish_attempt(
            attempt_id=attempt,
            terminal_event=_observed(),
            failure=None,
            service_tier=tier.settlement(served="priority", resolution="confirmed"),
        )
