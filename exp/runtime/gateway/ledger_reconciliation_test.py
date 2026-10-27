"""Real SQLite recovery of authoritative usage after provisional terminal writes."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from unittest.mock import DEFAULT, Mock

import pytest
from pydantic import ValidationError

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
from exp.runtime.gateway.ledger_errors import AccountingRecoverySupersededError
from exp.runtime.gateway.ledger_test import (
    FakeLedgerClock,
    _attempt_row,
    _authority_fixture,
    _deployment,
    _execution,
    _request,
)
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.native_recovery_test import RecoveryHostFake
from exp.runtime.gateway.platform import AttemptSettlementRequest
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.gateway.service_tiers import GatewayServiceTierAdmission
from exp.runtime.gateway.sqlite import migrations
from exp.runtime.gateway.sqlite.platform import SQLiteGatewayPlatform
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
    """Reserve one priced attempt against its original team and identity allocations."""
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
    """Build the trusted incomplete-disconnect observation used by recovery."""
    return GatewayEvent(
        kind=GatewayEventKind.FAILED,
        sequence_number=0,
        failure=_CANCELLED,
        usage=usage,
        usage_estimated=estimated,
        usage_incomplete_due_to_disconnect=True,
    )


def _observed(*, input_tokens: int = 10, output_tokens: int = 5) -> GatewayEvent:
    """Build a complete provider meter without recovery-only flags."""
    return GatewayEvent(
        kind=GatewayEventKind.COMPLETED,
        sequence_number=1,
        usage=GatewayUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _assert_allocations(budgets: SQLiteBudgetStore, *, settled: int, reserved: int = 0) -> None:
    """Check both original allocations while proving the next month remains untouched."""
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


@pytest.mark.parametrize("kind", [GatewayEventKind.COMPLETED, GatewayEventKind.INCOMPLETE])
@pytest.mark.parametrize("tiered", [False, True])
def test_provider_terminal_without_usage_is_not_a_recovery_estimate(
    tmp_path: Path,
    kind: GatewayEventKind,
    tiered: bool,
) -> None:
    """Missing usage in a final provider observation stays immutable, including receipt omission."""
    tier = (
        GatewayServiceTierAdmission(
            requested="priority",
            standard_prices=GatewayTokenPrices(),
            requested_prices=GatewayTokenPrices(
                input_nano_usd_per_million_tokens=2_000_000,
                output_nano_usd_per_million_tokens=4_000_000,
            ),
        )
        if tiered
        else None
    )
    _, _, ledger, budgets, attempt = _started(tmp_path, tier=tier)
    receipt = None if tier is None else tier.settlement(served="priority", resolution="confirmed")
    terminal = GatewayEvent(kind=kind, sequence_number=0)
    ledger.finish_attempt(
        attempt_id=attempt,
        terminal_event=terminal,
        failure=None,
        service_tier=receipt,
    )
    before = dict(_attempt_row(tmp_path, attempt))
    for supplied in (receipt, None):
        ledger.finish_attempt(
            attempt_id=attempt,
            terminal_event=terminal,
            failure=None,
            service_tier=supplied,
        )
        with pytest.raises(GatewayLedgerError, match="durable usage"):
            ledger.finish_attempt(
                attempt_id=attempt,
                terminal_event=_observed(),
                failure=None,
                service_tier=supplied,
            )
    assert dict(_attempt_row(tmp_path, attempt)) == before
    _assert_allocations(budgets, settled=0 if tiered else 300, reserved=300 if tiered else 0)


def test_completed_explicit_estimate_can_accept_observed_usage(tmp_path: Path) -> None:
    """Explicit estimate provenance, unlike absent final-provider usage, permits replacement."""
    _, _, ledger, budgets, attempt = _started(tmp_path)
    terminal = _observed(input_tokens=19, output_tokens=7).model_copy(
        update={"usage_estimated": True}
    )
    ledger.finish_attempt(attempt_id=attempt, terminal_event=terminal, failure=None)
    _assert_allocations(budgets, settled=66)
    for _ in range(2):
        ledger.finish_attempt(attempt_id=attempt, terminal_event=_observed(), failure=None)
    row = _attempt_row(tmp_path, attempt)
    assert row["state"] == "completed"
    assert row["usage_source"] == "observed"
    assert row["estimated_cost_nano_usd"] == 40
    _assert_allocations(budgets, settled=40)


def test_recovered_usage_waits_for_missing_tier_receipt(tmp_path: Path) -> None:
    """A cancellation's meter and pricing receipt can arrive separately without freeing its hold."""
    tier = GatewayServiceTierAdmission(
        requested="priority",
        standard_prices=GatewayTokenPrices(),
        requested_prices=GatewayTokenPrices(
            input_nano_usd_per_million_tokens=2_000_000,
            output_nano_usd_per_million_tokens=4_000_000,
        ),
    )
    _, _, ledger, budgets, attempt = _started(tmp_path, tier=tier)
    ledger.finish_attempt(attempt_id=attempt, terminal_event=_cancelled(), failure=_CANCELLED)
    for _ in range(2):
        ledger.finish_attempt(attempt_id=attempt, terminal_event=_observed(), failure=None)
    row = _attempt_row(tmp_path, attempt)
    assert row["usage_source"] == "observed"
    assert row["estimated_cost_nano_usd"] is None
    assert row["budget_settled_nano_usd"] is None
    _assert_allocations(budgets, settled=0, reserved=300)
    receipt = tier.settlement(served="priority", resolution="confirmed")
    for _ in range(2):
        ledger.finish_attempt(
            attempt_id=attempt,
            terminal_event=_observed(),
            failure=None,
            service_tier=receipt,
        )
    _assert_allocations(budgets, settled=40)


@pytest.mark.parametrize("output_tokens", [0, 7])
@pytest.mark.parametrize("tiered", [False, True])
def test_serialized_disconnect_replay_preserves_durable_unknown_billing(
    tmp_path: Path,
    output_tokens: int,
    tiered: bool,
) -> None:
    """Losing the private disconnect marker on an exact replay cannot reprice its meter."""
    tier = (
        GatewayServiceTierAdmission(
            requested="priority",
            standard_prices=GatewayTokenPrices(),
            requested_prices=GatewayTokenPrices(
                input_nano_usd_per_million_tokens=2_000_000,
                output_nano_usd_per_million_tokens=4_000_000,
            ),
        )
        if tiered
        else None
    )
    _, _, ledger, budgets, attempt = _started(tmp_path, tier=tier)
    receipt = None if tier is None else tier.settlement(served="priority", resolution="confirmed")
    terminal = _cancelled(usage=GatewayUsage(input_tokens=19, output_tokens=output_tokens))
    ledger.finish_attempt(
        attempt_id=attempt,
        terminal_event=terminal,
        failure=_CANCELLED,
        service_tier=receipt,
    )
    before = dict(_attempt_row(tmp_path, attempt))
    public_replay = GatewayEvent.model_validate_json(terminal.model_dump_json())
    assert public_replay.usage_incomplete_due_to_disconnect is False
    assert public_replay.usage == terminal.usage
    for supplied in (receipt, None):
        ledger.finish_attempt(
            attempt_id=attempt,
            terminal_event=public_replay,
            failure=_CANCELLED,
            service_tier=supplied,
        )
        with pytest.raises(GatewayLedgerError, match="differs from durable accounting evidence"):
            ledger.finish_attempt(
                attempt_id=attempt,
                terminal_event=public_replay.model_copy(
                    update={
                        "usage": GatewayUsage(input_tokens=20, output_tokens=output_tokens),
                    }
                ),
                failure=_CANCELLED,
                service_tier=supplied,
            )
    assert dict(_attempt_row(tmp_path, attempt)) == before
    assert before["usage_source"] == "observed"
    assert before["estimated_cost_nano_usd"] is None
    _assert_allocations(budgets, settled=0 if tiered else 300, reserved=300 if tiered else 0)


@pytest.mark.parametrize(
    "failure_class",
    [
        GatewayFailureClass.PROVIDER_INTERNAL,
        GatewayFailureClass.TRANSPORT,
        GatewayFailureClass.INTERNAL,
    ],
)
@pytest.mark.parametrize("separate_failure", [False, True])
def test_failed_provider_observation_does_not_prove_recovery(
    tmp_path: Path,
    failure_class: GatewayFailureClass,
    separate_failure: bool,
) -> None:
    """A provider's unmetered failure is immutable even if its class resembles cleanup."""
    _, _, ledger, budgets, attempt = _started(tmp_path)
    failure = GatewayFailure(
        failure_class=failure_class,
        safe_message="provider failure",
    )
    terminal = GatewayEvent(kind=GatewayEventKind.FAILED, sequence_number=0, failure=failure)
    ledger.finish_attempt(
        attempt_id=attempt,
        terminal_event=terminal,
        failure=failure if separate_failure else None,
    )
    before = dict(_attempt_row(tmp_path, attempt))
    assert before["accounting_recovery"] == 0
    for _ in range(2):
        with pytest.raises(GatewayLedgerError, match="differs from durable accounting evidence"):
            ledger.finish_attempt(attempt_id=attempt, terminal_event=_observed(), failure=None)
    assert dict(_attempt_row(tmp_path, attempt)) == before
    _assert_allocations(budgets, settled=300)


@pytest.mark.parametrize("estimated", [False, True])
@pytest.mark.parametrize("same_outcome", [False, True])
def test_public_replay_refuses_internal_recovery_without_mutating_accounting(
    tmp_path: Path,
    estimated: bool,
    same_outcome: bool,
) -> None:
    """A rejected public replay cannot apply the internal ledger's otherwise valid upgrade."""
    _, store, ledger, budgets, attempt = _started(tmp_path)
    platform = SQLiteGatewayPlatform(store.database_path, attempts=ledger, budgets=budgets)
    initial = _cancelled(
        estimated=estimated,
        usage=GatewayUsage(input_tokens=19, output_tokens=7) if estimated else None,
    )
    request = AttemptSettlementRequest(
        organization_id="org-one",
        attempt_id=attempt,
        terminal_event=initial,
    )
    settlement = platform.settle_attempt(request)
    before = dict(_attempt_row(tmp_path, attempt))
    terminal = _observed()
    if same_outcome:
        terminal = terminal.model_copy(
            update={"kind": GatewayEventKind.FAILED, "failure": _CANCELLED}
        )
    with pytest.raises(ValueError, match="differs from durable accounting evidence"):
        platform.settle_attempt(request.model_copy(update={"terminal_event": terminal}))
    if estimated:
        # Public JSON cannot promote the private estimate provenance to observed.
        with pytest.raises(ValueError, match="differs from durable accounting evidence"):
            platform.settle_attempt(
                AttemptSettlementRequest.model_validate_json(request.model_dump_json())
            )
    assert platform.settle_attempt(request) == settlement
    assert dict(_attempt_row(tmp_path, attempt)) == before
    _assert_allocations(budgets, settled=66 if estimated else 300)
    ledger.finish_attempt(attempt_id=attempt, terminal_event=_observed(), failure=None)
    assert _attempt_row(tmp_path, attempt)["usage_source"] == "observed"
    _assert_allocations(budgets, settled=40)


def test_public_initial_settlement_validation_rolls_back_the_write(tmp_path: Path) -> None:
    """A public field the durable record cannot reproduce must fail before accounting commits."""
    _, store, ledger, budgets, attempt = _started(tmp_path)
    platform = SQLiteGatewayPlatform(store.database_path, attempts=ledger, budgets=budgets)
    before = dict(_attempt_row(tmp_path, attempt))
    request = AttemptSettlementRequest(
        organization_id="org-one",
        attempt_id=attempt,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=0,
            usage=GatewayUsage(input_tokens=10, output_tokens=5, tool_names=("tool",)),
        ),
    )
    with pytest.raises(ValueError, match="differs from durable accounting evidence"):
        platform.settle_attempt(request)
    assert dict(_attempt_row(tmp_path, attempt)) == before
    _assert_allocations(budgets, settled=0, reserved=300)
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute(
            "SELECT terminal_state, terminal_at FROM gateway_requests WHERE request_id = ?",
            (before["request_id"],),
        ).fetchone() == (None, None)


@pytest.mark.parametrize(
    "late_result", ["meter", "unmetered", "unmetered_tier", "unmetered_self_served"]
)
def test_real_sqlite_native_abandonment_acknowledges_the_late_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    late_result: str,
) -> None:
    """A late meter upgrades recovery; a weaker failure drains without changing liability."""
    monkeypatch.setattr(NativeAttemptAccounting, "_sweep_loop", lambda self: None)
    clock = FakeLedgerClock()
    store, ledger, key = _authority_fixture(tmp_path, clock)
    budgets = SQLiteBudgetStore(store.database_path, clock=clock)
    budgets.set_limit(
        organization_id="org-one",
        period="2026-08",
        scope=_SCOPES[0],
        limit_nano_usd=1_000,
    )
    request = _request("native late meter")
    authorization = store.authorize_request(
        raw_key=key,
        alias="coding",
        request=request,
        deadline_monotonic=clock.monotonic() + 30,
    )
    snapshot, deployment = _execution(authorization), _deployment()
    tier = (
        GatewayServiceTierAdmission(
            requested="priority",
            standard_prices=deployment.gateway.prices,
            requested_prices=deployment.gateway.prices,
        )
        if late_result in {"unmetered_tier", "unmetered_self_served"}
        else None
    )
    if late_result == "unmetered_self_served":
        monkeypatch.setattr("exp.runtime.gateway.class_throttles.fast_on_rung", lambda *_: True)
    ledger.accept_request(authorization=authorization)
    attempt = ledger.start_attempt(
        snapshot=snapshot,
        deployment=deployment,
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_nano_usd=300,
        service_tier=tier,
    )
    accounting = NativeAttemptAccounting(ledger, recovery_host=RecoveryHostFake())
    entry = InflightRequest(
        authorization=authorization,
        route=GatewayRoute(snapshot=snapshot, deployment=deployment, route_reason="direct"),
        request=request,
        deadline_monotonic=time.monotonic() + 30,
        total_attempts=1,
        attempt_depths={attempt: 0},
        active_attempt_id=attempt,
        attempt_service_tiers={} if tier is None else {attempt: tier},
    )
    accounting.register(entry)
    entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
    gate = entry.execution_lock

    class ObservedGate:
        """Signal that the exact callback retained its meter before waiting behind cleanup."""

        def __enter__(self) -> None:
            """Wait until the already-running abandonment releases the request gate."""
            waiting.set()
            gate.acquire()

        def __exit__(self, *_exception: object) -> None:
            """Release the same request gate after the observed write."""
            gate.release()

        def acquire(self, *, blocking: bool) -> bool:
            """Preserve abandonment's nonblocking gate acquisition."""
            return gate.acquire(blocking=blocking)

        def release(self) -> None:
            """Release abandonment's ownership of the gate."""
            gate.release()

    def delay_unknown(**fields: object) -> object:
        """Pause the actual unknown SQLite write until the exact meter is retained."""
        terminal = fields["terminal_event"]
        assert isinstance(terminal, GatewayEvent)
        if terminal.accounting_recovery:
            entered.set()
            assert release.wait(5)
        return DEFAULT

    monkeypatch.setattr(entry, "execution_lock", ObservedGate())
    finish = Mock(wraps=ledger.finish_attempt, side_effect=delay_unknown)
    monkeypatch.setattr(accounting, "_finish_attempt", finish)
    payload = json.dumps(
        {
            "request_id": authorization.request_id,
            "attempt_id": attempt,
            "outcome": "completed" if late_result == "meter" else "failed",
            **(
                {"usage": {"input_tokens": 10, "output_tokens": 5}}
                if late_result == "meter"
                else {"failure": {"failure_class": "transport", "safe_message": "open failed"}}
            ),
            "finalize": True,
            "opened": late_result == "meter",
        }
    )
    errors: list[BaseException] = []
    abandoned: list[object] = []

    def abandon() -> None:
        """Commit the cancellation before the provider's exact write can start."""
        try:
            abandoned.append(
                json.loads(accounting.abandon(json.dumps({"request_id": authorization.request_id})))
            )
        except BaseException as error:  # noqa: BLE001 - report thread failures to the test.
            errors.append(error)

    def observe() -> None:
        """Send the provider callback through the real native accounting path."""
        try:
            accounting.settle(payload)
        except BaseException as error:  # noqa: BLE001 - report thread failures to the test.
            errors.append(error)

    cleanup, observed = threading.Thread(target=abandon), threading.Thread(target=observe)
    cleanup.start()
    try:
        assert entered.wait(5)
        observed.start()
        assert waiting.wait(5)
        assert entry.pending_settlement is not None
    finally:
        release.set()
        cleanup.join(5)
        if observed.ident is not None:
            observed.join(5)
    assert not cleanup.is_alive()
    assert not observed.is_alive()
    assert not errors
    assert abandoned == [{"terminalized": False}]
    assert finish.call_count == 2
    assert accounting.entry(authorization.request_id) is None
    assert accounting.close(5)
    row = _attempt_row(tmp_path, attempt)
    assert row["state"] == "cancelled"
    assert row["accounting_recovery"] == 1
    assert row["usage_source"] == ("observed" if late_result == "meter" else "unknown")
    assert (row["input_tokens"], row["output_tokens"], row["estimated_cost_nano_usd"]) == (
        (10, 5, 40) if late_result == "meter" else (None, None, None)
    )
    if tier is not None:
        assert json.loads(row["service_tier_settlement"])["resolution"] == (
            "confirmed" if late_result == "unmetered_self_served" else "missing"
        )
    allocation = budgets.remaining(organization_id="org-one", period="2026-08")[0]
    assert allocation.settled_nano_usd == (
        40 if late_result == "meter" else 0 if tier is not None else 300
    )
    assert allocation.reserved_nano_usd == (300 if tier is not None else 0)
    assert allocation.unknown_cost_attempts == 0


@pytest.mark.parametrize(
    "field",
    [
        "accounting_recovery",
        "_accounting_recovery",
        "accounting_release_reservation",
        "_accounting_release_reservation",
    ],
)
@pytest.mark.parametrize("release_reservation", [False, True])
def test_recovery_provenance_is_internal_and_cannot_be_invented_by_public_payloads(
    tmp_path: Path,
    field: str,
    release_reservation: bool,
) -> None:
    """Only the trusted constructor can create a recovery-capable failed terminal."""
    _, store, ledger, budgets, attempt = _started(tmp_path)
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.INTERNAL,
        safe_message="local recovery",
    )
    recovery = GatewayEvent.for_accounting_recovery(
        failure=failure, release_reservation=release_reservation
    )
    assert recovery.accounting_recovery
    assert recovery.accounting_release_reservation is release_reservation
    serialized = GatewayEvent.model_validate_json(recovery.model_dump_json())
    assert not serialized.accounting_recovery
    assert not serialized.accounting_release_reservation
    request = AttemptSettlementRequest(
        organization_id="org-one",
        attempt_id=attempt,
        terminal_event=recovery,
    )
    payload = json.loads(request.model_dump_json())
    payload["terminal_event"][field] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AttemptSettlementRequest.model_validate_json(json.dumps(payload))
    platform = SQLiteGatewayPlatform(store.database_path, attempts=ledger, budgets=budgets)
    before = dict(_attempt_row(tmp_path, attempt))
    with pytest.raises(ValueError, match="cannot declare internal accounting recovery"):
        platform.settle_attempt(request)
    assert dict(_attempt_row(tmp_path, attempt)) == before
    _assert_allocations(budgets, settled=0, reserved=300)
    ledger.finish_attempt(attempt_id=attempt, terminal_event=recovery, failure=failure)
    assert _attempt_row(tmp_path, attempt)["accounting_recovery"] == 1
    _assert_allocations(budgets, settled=0 if release_reservation else 300)
    ledger = SQLiteAttemptLedger(store.database_path)
    for _ in range(2):
        ledger.finish_attempt(attempt_id=attempt, terminal_event=_observed(), failure=None)
    row = _attempt_row(tmp_path, attempt)
    assert row["state"] == "failed"
    assert row["failure_class"] == "internal"
    assert row["usage_source"] == "observed"
    assert row["estimated_cost_nano_usd"] == 40
    _assert_allocations(budgets, settled=0 if release_reservation else 40)


@pytest.mark.parametrize("maximum", [None, 300])
@pytest.mark.parametrize("crashed", [False, True])
@pytest.mark.parametrize("observed_first", [False, True])
def test_customer_waiver_can_follow_unknown_recovery_without_losing_provider_meter(
    tmp_path: Path, maximum: int | None, crashed: bool, observed_first: bool
) -> None:
    """A timeout releases provisional or crash-held liability while retaining late meters."""
    clock, store, ledger, budgets, attempt = _started(tmp_path, maximum=maximum)
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.INTERNAL, safe_message="dispatch unconfirmed"
    )
    if crashed:
        clock.advance(60)
        assert ledger.reconcile_crashed_requests(cleanup_grace=timedelta()) == (0, 1)
    else:
        ledger.finish_attempt(
            attempt_id=attempt,
            terminal_event=GatewayEvent.for_accounting_recovery(failure=failure),
            failure=failure,
        )
    before = dict(_attempt_row(tmp_path, attempt))
    if observed_first:
        ledger.finish_attempt(attempt_id=attempt, terminal_event=_observed(), failure=None)
        _assert_allocations(budgets, settled=40)
    timeout = GatewayFailure(
        failure_class=GatewayFailureClass.TIMEOUT, safe_message="settlement deadline reached"
    )
    waiver = GatewayEvent.for_accounting_recovery(failure=timeout, release_reservation=True)
    for _ in range(2):
        ledger.finish_attempt(attempt_id=attempt, terminal_event=waiver, failure=timeout)
    row = _attempt_row(tmp_path, attempt)
    assert row["usage_source"] == ("observed" if observed_first else "unknown")
    assert row["estimated_cost_nano_usd"] == (40 if observed_first else None)
    assert row["failure_class"] == before["failure_class"]
    assert row["state"] == before["state"]
    assert row["terminal_at"] == before["terminal_at"]
    assert row["budget_settled_nano_usd"] == 0
    _assert_allocations(budgets, settled=0)
    ledger = SQLiteAttemptLedger(store.database_path)
    for _ in range(2):
        ledger.finish_attempt(attempt_id=attempt, terminal_event=_observed(), failure=None)
        ledger.finish_attempt(attempt_id=attempt, terminal_event=waiver, failure=timeout)
    row = _attempt_row(tmp_path, attempt)
    assert row["usage_source"] == "observed"
    assert row["estimated_cost_nano_usd"] == 40
    assert row["budget_settled_nano_usd"] == 0
    _assert_allocations(budgets, settled=0)


@pytest.mark.parametrize("failed", [False, True])
def test_customer_waiver_does_not_replace_an_ordinary_exact_provider_terminal(
    tmp_path: Path, failed: bool
) -> None:
    """A timeout disposition cannot convert an ordinary observed terminal into recovery."""
    _, _, ledger, budgets, attempt = _started(tmp_path)
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.PROVIDER_INTERNAL, safe_message="provider failed"
    )
    event = _observed()
    if failed:
        event = event.model_copy(update={"kind": GatewayEventKind.FAILED, "failure": failure})
    ledger.finish_attempt(attempt_id=attempt, terminal_event=event, failure=event.failure)
    before = dict(_attempt_row(tmp_path, attempt))
    timeout = GatewayFailure(
        failure_class=GatewayFailureClass.TIMEOUT, safe_message="settlement deadline reached"
    )
    waiver = GatewayEvent.for_accounting_recovery(failure=timeout, release_reservation=True)
    for _ in range(2):
        with pytest.raises(AccountingRecoverySupersededError, match="immutable provider terminal"):
            ledger.finish_attempt(attempt_id=attempt, terminal_event=waiver, failure=timeout)
    assert dict(_attempt_row(tmp_path, attempt)) == before
    _assert_allocations(budgets, settled=40)


def test_schema32_preserves_legacy_failure_without_inventing_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The additive migration defaults old terminal rows to immutable and preserves money."""
    with monkeypatch.context() as prior:
        prior.setattr(migrations, "SCHEMA_VERSION", 31)
        _, store, ledger, budgets, attempt = _started(tmp_path)
        failure = GatewayFailure(
            failure_class=GatewayFailureClass.INTERNAL,
            safe_message="previous failure",
        )
        terminal = GatewayEvent(kind=GatewayEventKind.FAILED, sequence_number=0, failure=failure)
        ledger.finish_attempt(attempt_id=attempt, terminal_event=terminal, failure=failure)
        before = dict(_attempt_row(tmp_path, attempt))
    assert "accounting_recovery" not in before
    backup = migrations.initialize_database(store.database_path)
    assert backup is not None
    with sqlite3.connect(backup) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (31,)
        assert "accounting_recovery" not in {
            row[1] for row in connection.execute("PRAGMA table_info(gateway_attempts)")
        }
    after = dict(_attempt_row(tmp_path, attempt))
    assert after.pop("accounting_recovery") == 0
    assert after == before
    with pytest.raises(GatewayLedgerError, match="differs from durable accounting evidence"):
        ledger.finish_attempt(attempt_id=attempt, terminal_event=_observed(), failure=None)
    _assert_allocations(budgets, settled=300)


@pytest.mark.parametrize("crashed", [False, True])
@pytest.mark.parametrize("observed_first", [False, True])
@pytest.mark.parametrize("tiered", [False, True])
def test_unmetered_recovery_failure_replay_preserves_durable_evidence(
    tmp_path: Path, crashed: bool, observed_first: bool, tiered: bool
) -> None:
    """Weaker late failures survive reopen without changing meters, charges or delivery."""
    prices = _deployment().gateway.prices
    tier = (
        GatewayServiceTierAdmission(
            requested="priority", standard_prices=prices, requested_prices=prices
        )
        if tiered
        else None
    )
    clock, store, ledger, budgets, attempt = _started(tmp_path, tier=tier)
    receipt = None if tier is None else tier.settlement(served="priority", resolution="confirmed")
    if crashed:
        clock.advance(60)
        assert ledger.reconcile_crashed_requests(cleanup_grace=timedelta()) == (0, 1)
    else:
        ledger.finish_attempt(
            attempt_id=attempt,
            terminal_event=GatewayEvent.for_accounting_recovery(failure=_CANCELLED),
            failure=_CANCELLED,
        )
    if observed_first:
        ledger.finish_attempt(
            attempt_id=attempt, terminal_event=_observed(), failure=None, service_tier=receipt
        )
    before = dict(_attempt_row(tmp_path, attempt))
    allocations = budgets.remaining(organization_id="org-one", period="2026-08")
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.TRANSPORT, safe_message="late open failure"
    )
    event = GatewayEvent(kind=GatewayEventKind.FAILED, sequence_number=2, failure=failure)
    reopened = SQLiteAttemptLedger(store.database_path, clock=clock)
    missing = None if tier is None else tier.settlement(served=None, resolution="missing")
    for supplied in (None, missing, receipt, missing, receipt):
        reopened.finish_attempt(
            attempt_id=attempt, terminal_event=event, failure=None, service_tier=supplied
        )
    after = dict(_attempt_row(tmp_path, attempt))
    if receipt is not None:
        assert after["service_tier_settlement"] == receipt.model_dump_json()
        before.pop("service_tier_settlement")
        after.pop("service_tier_settlement")
    assert after == before
    assert budgets.remaining(organization_id="org-one", period="2026-08") == allocations


@pytest.mark.parametrize("incoming", ["partial", "estimated", "completed_without_usage"])
def test_recovery_acknowledgment_does_not_absorb_other_terminal_evidence(
    tmp_path: Path, incoming: str
) -> None:
    """Only an actual unmetered failure is weaker than the trusted recovery outcome."""
    _, _, ledger, budgets, attempt = _started(tmp_path)
    ledger.finish_attempt(
        attempt_id=attempt,
        terminal_event=GatewayEvent.for_accounting_recovery(failure=_CANCELLED),
        failure=_CANCELLED,
    )
    before = dict(_attempt_row(tmp_path, attempt))
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.TRANSPORT, safe_message="late open failure"
    )
    event = (
        GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=1)
        if incoming == "completed_without_usage"
        else GatewayEvent(
            kind=(
                GatewayEventKind.INCOMPLETE if incoming == "estimated" else GatewayEventKind.FAILED
            ),
            sequence_number=1,
            failure=None if incoming == "estimated" else failure,
            usage=GatewayUsage(
                input_tokens=10, output_tokens=5 if incoming == "estimated" else None
            ),
            usage_estimated=incoming == "estimated",
            usage_unreported_truncation=incoming == "estimated",
        )
    )
    with pytest.raises(GatewayLedgerError, match="another terminal state"):
        ledger.finish_attempt(attempt_id=attempt, terminal_event=event, failure=event.failure)
    assert dict(_attempt_row(tmp_path, attempt)) == before
    _assert_allocations(budgets, settled=300)


def test_unmetered_failure_cannot_replace_an_ordinary_completed_provider_terminal(
    tmp_path: Path,
) -> None:
    """The recovery acknowledgment does not relax immutable ordinary provider outcomes."""
    _, _, ledger, budgets, attempt = _started(tmp_path)
    ledger.finish_attempt(attempt_id=attempt, terminal_event=_observed(), failure=None)
    before = dict(_attempt_row(tmp_path, attempt))
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.TRANSPORT, safe_message="late open failure"
    )
    with pytest.raises(GatewayLedgerError, match="another terminal state"):
        ledger.finish_attempt(attempt_id=attempt, terminal_event=None, failure=failure)
    assert dict(_attempt_row(tmp_path, attempt)) == before
    _assert_allocations(budgets, settled=40)


@pytest.mark.parametrize("crashed", [False, True])
@pytest.mark.parametrize("waived", [False, True])
def test_complete_meter_after_unmetered_tier_confirmation_resolves_only_known_liability(
    tmp_path: Path, crashed: bool, waived: bool
) -> None:
    """Confirmed metadata remains usable after reopen and never reverses a customer waiver."""
    prices = _deployment().gateway.prices
    tier = GatewayServiceTierAdmission(
        requested="priority", standard_prices=prices, requested_prices=prices
    )
    clock, store, ledger, budgets, attempt = _started(tmp_path, tier=tier)
    if crashed:
        clock.advance(60)
        assert ledger.reconcile_crashed_requests(cleanup_grace=timedelta()) == (0, 1)
    else:
        ledger.finish_attempt(
            attempt_id=attempt,
            terminal_event=GatewayEvent.for_accounting_recovery(failure=_CANCELLED),
            failure=_CANCELLED,
        )
    if waived:
        ledger.finish_attempt(
            attempt_id=attempt,
            terminal_event=GatewayEvent.for_accounting_recovery(
                failure=_CANCELLED, release_reservation=True
            ),
            failure=_CANCELLED,
        )
    ledger.finish_attempt(attempt_id=attempt, terminal_event=_observed(), failure=None)
    before = dict(_attempt_row(tmp_path, attempt))
    assert before["usage_source"] == "observed"
    assert before["estimated_cost_nano_usd"] is None
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.TRANSPORT, safe_message="late open failure"
    )
    receipt = tier.settlement(served="priority", resolution="confirmed")
    ledger.finish_attempt(
        attempt_id=attempt, terminal_event=None, failure=failure, service_tier=receipt
    )
    confirmed = _attempt_row(tmp_path, attempt)
    assert confirmed["estimated_cost_nano_usd"] is None
    assert confirmed["budget_settled_nano_usd"] == (0 if waived else None)
    _assert_allocations(budgets, settled=0, reserved=0 if waived else 300)
    reopened = SQLiteAttemptLedger(store.database_path, clock=clock)
    for _ in range(2):
        reopened.finish_attempt(attempt_id=attempt, terminal_event=_observed(), failure=None)
    after = _attempt_row(tmp_path, attempt)
    for field in ("state", "terminal_at", "failure_class", "input_tokens", "output_tokens"):
        assert after[field] == before[field]
    assert after["estimated_cost_nano_usd"] == 40
    assert after["budget_settled_nano_usd"] == (0 if waived else 40)
    _assert_allocations(budgets, settled=0 if waived else 40)


@pytest.mark.parametrize("invalid", ["unadmitted", "forged_rates", "conflicting_confirmation"])
def test_unmetered_recovery_failure_still_rejects_invalid_tier_evidence(
    tmp_path: Path, invalid: str
) -> None:
    """Weaker failures cannot invent schedules or replace a confirmed served tier."""
    prices = _deployment().gateway.prices
    tier = GatewayServiceTierAdmission(
        requested="priority", standard_prices=prices, requested_prices=prices
    )
    _, _, ledger, budgets, attempt = _started(
        tmp_path, tier=None if invalid == "unadmitted" else tier
    )
    ledger.finish_attempt(
        attempt_id=attempt,
        terminal_event=GatewayEvent.for_accounting_recovery(failure=_CANCELLED),
        failure=_CANCELLED,
    )
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.TRANSPORT, safe_message="late open failure"
    )
    event = GatewayEvent(kind=GatewayEventKind.FAILED, sequence_number=1, failure=failure)
    receipt = tier.settlement(served="priority", resolution="confirmed")
    if invalid == "conflicting_confirmation":
        ledger.finish_attempt(
            attempt_id=attempt, terminal_event=event, failure=failure, service_tier=receipt
        )
        receipt = tier.settlement(served="default", resolution="confirmed")
    elif invalid == "forged_rates":
        receipt = receipt.model_copy(
            update={"prices": prices.model_copy(update={"input_nano_usd_per_million_tokens": 999})}
        )
    before = dict(_attempt_row(tmp_path, attempt))
    with pytest.raises(GatewayLedgerError, match="service-tier receipt"):
        ledger.finish_attempt(
            attempt_id=attempt, terminal_event=event, failure=failure, service_tier=receipt
        )
    assert dict(_attempt_row(tmp_path, attempt)) == before
    _assert_allocations(
        budgets,
        settled=300 if invalid == "unadmitted" else 0,
        reserved=0 if invalid == "unadmitted" else 300,
    )
