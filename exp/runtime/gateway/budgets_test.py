"""Tests for monthly hard limits, reservation concurrency, and UTC rollover."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import canonical_json_bytes
from exp.common.models import ModelCapabilities
from exp.common.models.catalog import GatewayDeploymentMetadata, GatewayTokenPrices
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.budgets import (
    BudgetReservationRejected,
    BudgetScope,
    BudgetScopeKind,
    SQLiteBudgetStore,
    maximum_attempt_cost_micro_usd,
)
from exp.runtime.gateway.contracts import (
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.sqlite.store import SQLiteGatewayStore

_CATALOG = "a" * 64


class _Clock:
    """Controllable aware wall and monotonic clock."""

    def __init__(self) -> None:
        """Start near the end of one UTC month."""
        self.wall = datetime(2026, 8, 31, 23, 59, tzinfo=UTC)
        self.monotonic_value = 100.0

    def now(self) -> datetime:
        """Return controlled wall time."""
        return self.wall

    def monotonic(self) -> float:
        """Return controlled monotonic time."""
        return self.monotonic_value

    def advance(self, duration: timedelta) -> None:
        """Advance both clocks by one duration."""
        self.wall += duration
        self.monotonic_value += duration.total_seconds()


def _deployment(*, priced: bool = True, deployment_id: str = "primary") -> ExactModelDeployment:
    """Return one exact deployment with optional hard-limit pricing."""
    prices = (
        GatewayTokenPrices(
            input_micro_usd_per_million_tokens=1_000_000,
            output_micro_usd_per_million_tokens=2_000_000,
        )
        if priced
        else GatewayTokenPrices()
    )
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=deployment_id,
        exact_model_id="exact-one",
        connection=f"connection-{deployment_id}",
        provider="openai-compatible",
        provider_model="provider-model",
        connection_sha256=("b" if deployment_id == "primary" else "c") * 64,
        capabilities_sha256="d" * 64,
        capabilities=ModelCapabilities(maximum_output_tokens=16),
        gateway=GatewayDeploymentMetadata(
            prices=prices,
            pricing_source="test",
        ),
    )


def _request(content: str) -> GatewayRequest:
    """Build one bounded request whose content is never persisted."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content=content),),
        maximum_output_tokens=16,
    )


def _authority(
    tmp_path: Path,
    clock: _Clock,
) -> tuple[SQLiteGatewayStore, SQLiteAttemptLedger, SQLiteBudgetStore, str]:
    """Create real SQLite authority, ledger, budget store, and one granted key."""
    path = tmp_path / "gateway.db"
    store = SQLiteGatewayStore(path, clock=clock)
    ledger = SQLiteAttemptLedger(path, clock=clock)
    budgets = SQLiteBudgetStore(path, clock=clock)
    store.create_organization(organization_id="org", slug="org", display_name="Org")
    store.create_identity(organization_id="org", identity_id="identity", display_name="Identity")
    store.register_catalog_snapshot(
        organization_id="org",
        snapshot_ref="snapshot",
        catalog_sha256=_CATALOG,
    )
    store.activate_alias_revision(
        organization_id="org",
        alias_id="coding",
        alias_name="coding",
        revision_id="revision",
        target=DirectTarget(pool_id="pool"),
        snapshot_ref="snapshot",
        catalog_sha256=_CATALOG,
    )
    store.grant_alias(organization_id="org", identity_id="identity", alias_id="coding")
    key = store.issue_virtual_key(
        organization_id="org",
        identity_id="identity",
        key_id="key",
    ).raw_key
    return store, ledger, budgets, key


def _accepted(
    store: SQLiteGatewayStore,
    ledger: SQLiteAttemptLedger,
    clock: _Clock,
    key: str,
    content: str,
) -> tuple[GatewayRequest, ExecutionSnapshot]:
    """Authorize and durably accept one unique request."""
    request = _request(content)
    authorization = store.authorize_request(
        raw_key=key,
        alias="coding",
        request=request,
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    return request, ExecutionSnapshot(
        authorization=authorization,
        exact_model_id="exact-one",
        pool_id="pool",
        deployment_ids=("primary", "secondary"),
    )


def test_maximum_attempt_cost_is_integer_conservative_and_unknown_prices_fail_closed() -> None:
    """Reservation pricing uses canonical bytes, output ceiling, and no float money."""
    request = _request("four bytes")
    known = maximum_attempt_cost_micro_usd(request, _deployment())

    assert known is not None and isinstance(known, int) and known > 32
    assert maximum_attempt_cost_micro_usd(request, _deployment(priced=False)) is None
    unrepresentable = _deployment().model_copy(
        update={
            "gateway": _deployment().gateway.model_copy(
                update={
                    "prices": GatewayTokenPrices(
                        input_micro_usd_per_million_tokens=10**30,
                        output_micro_usd_per_million_tokens=10**30,
                    )
                }
            )
        }
    )
    assert maximum_attempt_cost_micro_usd(request, unrepresentable) is None
    all_prices = _deployment().model_copy(
        update={
            "gateway": _deployment().gateway.model_copy(
                update={
                    "prices": GatewayTokenPrices(
                        input_micro_usd_per_million_tokens=1_000_000,
                        cached_input_micro_usd_per_million_tokens=3_000_000,
                        output_micro_usd_per_million_tokens=2_000_000,
                        reasoning_micro_usd_per_million_tokens=4_000_000,
                    )
                }
            )
        }
    )
    assert all_prices.gateway.capabilities.reports_cached_input_tokens is False
    assert all_prices.gateway.capabilities.reports_reasoning_tokens is False
    all_dimensions = maximum_attempt_cost_micro_usd(request, all_prices)
    assert all_dimensions is not None and all_dimensions > known
    assert (
        maximum_attempt_cost_micro_usd(
            request.model_copy(update={"maximum_output_tokens": None}),
            _deployment().model_copy(update={"capabilities": ModelCapabilities()}),
        )
        is None
    )


def test_concurrent_identity_reservations_never_exceed_hard_limit(tmp_path: Path) -> None:
    """BEGIN IMMEDIATE serializes competing reservations before attempt insertion."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.IDENTITY, identity_id="identity"),
        limit_micro_usd=500,
    )
    snapshots = [
        _accepted(store, ledger, clock, key, f"concurrent-{index}")[1] for index in range(10)
    ]

    def reserve(index: int) -> str:
        """Attempt one competing fixed-size reservation."""
        return ledger.start_attempt(
            snapshot=snapshots[index],
            deployment=_deployment(),
            attempt_ordinal=0,
            route_depth=0,
            maximum_cost_micro_usd=100,
        )

    accepted: list[str] = []
    rejected = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(reserve, index) for index in range(10)]
        for future in futures:
            try:
                accepted.append(future.result())
            except BudgetReservationRejected:
                rejected += 1

    assert len(accepted) == 5
    assert rejected == 5
    remaining = budgets.remaining(organization_id="org", period="2026-08")[0]
    assert remaining.reserved_micro_usd == 500
    assert remaining.remaining_micro_usd == 0


def test_configured_optional_prices_are_reserved_even_when_reporting_hints_are_false(
    tmp_path: Path,
) -> None:
    """Provider usage cannot settle cached or reasoning cost above its reservation."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    deployment = _deployment().model_copy(
        update={
            "gateway": _deployment().gateway.model_copy(
                update={
                    "prices": GatewayTokenPrices(
                        input_micro_usd_per_million_tokens=1_000_000,
                        cached_input_micro_usd_per_million_tokens=3_000_000,
                        output_micro_usd_per_million_tokens=2_000_000,
                        reasoning_micro_usd_per_million_tokens=4_000_000,
                    )
                }
            )
        }
    )
    request, snapshot = _accepted(store, ledger, clock, key, "all-price-dimensions")
    maximum = maximum_attempt_cost_micro_usd(request, deployment)
    assert maximum is not None
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_micro_usd=maximum,
    )
    attempt = ledger.start_attempt(
        snapshot=snapshot,
        deployment=deployment,
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=maximum,
    )
    input_ceiling = len(canonical_json_bytes(request))
    ledger.finish_attempt(
        attempt_id=attempt,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=0,
            usage=GatewayUsage(
                input_tokens=input_ceiling,
                cached_input_tokens=input_ceiling,
                output_tokens=16,
                reasoning_tokens=16,
            ),
        ),
        failure=None,
    )

    remaining = budgets.remaining(organization_id="org", period="2026-08")[0]
    assert remaining.reserved_micro_usd == 0
    assert remaining.settled_micro_usd == maximum
    assert remaining.remaining_micro_usd == 0


def test_settlement_counts_failures_retries_and_rollover_without_reset(tmp_path: Path) -> None:
    """Every physical attempt settles in its start month and old buckets remain unchanged."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_micro_usd=1_000,
    )
    _request_one, snapshot_one = _accepted(store, ledger, clock, key, "failed-attempt")
    failed = ledger.start_attempt(
        snapshot=snapshot_one,
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=300,
    )
    ledger.finish_attempt(
        attempt_id=failed,
        terminal_event=None,
        failure=GatewayFailure(
            failure_class=GatewayFailureClass.TRANSPORT,
            safe_message="provider transport failed",
        ),
    )

    # A failure without usage can still be billable, so the maximum reservation remains charged.
    august = budgets.remaining(organization_id="org", period="2026-08")[0]
    assert august.charged_micro_usd == 300
    clock.advance(timedelta(minutes=2))
    budgets.set_limit(
        organization_id="org",
        period="2026-09",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_micro_usd=1_000,
    )
    _request_two, snapshot_two = _accepted(store, ledger, clock, key, "completed-attempt")
    completed = ledger.start_attempt(
        snapshot=snapshot_two,
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=300,
    )
    ledger.finish_attempt(
        attempt_id=completed,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=0,
            usage=GatewayUsage(input_tokens=10, output_tokens=5),
        ),
        failure=None,
    )

    assert budgets.remaining(organization_id="org", period="2026-08")[0].charged_micro_usd == 300
    september = budgets.remaining(organization_id="org", period="2026-09")[0]
    assert september.reserved_micro_usd == 0
    assert september.settled_micro_usd == 20
    assert september.remaining_micro_usd == 980


def test_unknown_historical_cost_fails_closed_only_after_limit_exists(tmp_path: Path) -> None:
    """An unpriced attempt may run without limits but blocks a later hard cap in that month."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    _first_request, first_snapshot = _accepted(store, ledger, clock, key, "unknown-first")
    attempt = ledger.start_attempt(
        snapshot=first_snapshot,
        deployment=_deployment(priced=False),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=None,
    )
    ledger.finish_attempt(
        attempt_id=attempt,
        terminal_event=None,
        failure=GatewayFailure(
            failure_class=GatewayFailureClass.TRANSPORT,
            safe_message="provider transport failed",
        ),
    )
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_micro_usd=1_000,
    )
    _second_request, second_snapshot = _accepted(store, ledger, clock, key, "unknown-second")

    with pytest.raises(BudgetReservationRejected, match="prior attempts with unknown cost"):
        ledger.start_attempt(
            snapshot=second_snapshot,
            deployment=_deployment(),
            attempt_ordinal=0,
            route_depth=0,
            maximum_cost_micro_usd=100,
        )
    remaining = budgets.remaining(organization_id="org", period="2026-08")[0]
    assert remaining.unknown_cost_attempts == 1
    assert remaining.remaining_micro_usd == 0


def test_limit_created_midflight_adopts_and_settles_existing_reservation(
    tmp_path: Path,
) -> None:
    """A new limit binds an active attempt and later settles it exactly once."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    _request_value, snapshot = _accepted(store, ledger, clock, key, "midflight")
    attempt = ledger.start_attempt(
        snapshot=snapshot,
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=300,
    )

    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_micro_usd=1_000,
    )
    active = budgets.remaining(organization_id="org", period="2026-08")[0]
    assert active.reserved_micro_usd == 300
    assert active.settled_micro_usd == 0

    terminal = GatewayEvent(
        kind=GatewayEventKind.COMPLETED,
        sequence_number=0,
        usage=GatewayUsage(input_tokens=10, output_tokens=5),
    )
    ledger.finish_attempt(attempt_id=attempt, terminal_event=terminal, failure=None)
    ledger.finish_attempt(attempt_id=attempt, terminal_event=terminal, failure=None)

    settled = budgets.remaining(organization_id="org", period="2026-08")[0]
    assert settled.reserved_micro_usd == 0
    assert settled.settled_micro_usd == 20
    assert settled.remaining_micro_usd == 980


def test_crash_recovery_retains_reservation_and_idempotent_settlement_charges_once(
    tmp_path: Path,
) -> None:
    """Unknown dispatches keep their maximum while repeated settlement cannot double-charge."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_micro_usd=500,
    )
    _request_one, first_snapshot = _accepted(store, ledger, clock, key, "crash")
    ledger.start_attempt(
        snapshot=first_snapshot,
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=100,
    )
    clock.advance(timedelta(seconds=31))
    assert ledger.reconcile_crashed_requests(cleanup_grace=timedelta(0)) == (0, 1)
    after_crash = budgets.remaining(organization_id="org", period="2026-08")[0]
    assert after_crash.reserved_micro_usd == 100
    assert after_crash.remaining_micro_usd == 400

    _request_two, second_snapshot = _accepted(store, ledger, clock, key, "settle-once")
    settled = ledger.start_attempt(
        snapshot=second_snapshot,
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=200,
    )
    terminal = GatewayEvent(
        kind=GatewayEventKind.COMPLETED,
        sequence_number=0,
        usage=GatewayUsage(input_tokens=10, output_tokens=5),
    )
    ledger.finish_attempt(attempt_id=settled, terminal_event=terminal, failure=None)
    ledger.finish_attempt(attempt_id=settled, terminal_event=terminal, failure=None)

    remaining = budgets.remaining(organization_id="org", period="2026-08")[0]
    assert remaining.charged_micro_usd == 120
    assert remaining.reserved_micro_usd == 100
    assert remaining.settled_micro_usd == 20
    assert remaining.remaining_micro_usd == 380


def test_migrated_attempts_receive_explicit_period_and_cost_state(tmp_path: Path) -> None:
    """The current schema stores no floating monetary fields or resettable counters."""
    clock = _Clock()
    store, ledger, _budgets, key = _authority(tmp_path, clock)
    _request_value, snapshot = _accepted(store, ledger, clock, key, "schema")
    attempt = ledger.start_attempt(
        snapshot=snapshot,
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=123,
    )
    connection = sqlite3.connect(ledger.database_path)
    try:
        row = connection.execute(
            """
            SELECT budget_period_start, budget_reserved_micro_usd,
                   budget_settled_micro_usd
            FROM gateway_attempts WHERE attempt_id = ?
            """,
            (attempt,),
        ).fetchone()
    finally:
        connection.close()
    assert row == ("2026-08-01T00:00:00+00:00", 123, None)
