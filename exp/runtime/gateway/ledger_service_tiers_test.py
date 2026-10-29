# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Persistent tier authority, exact long-context pricing and conservative billing holds."""

import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from exp.common.models.catalog import BillingSource, GatewayLongContextTier, GatewayTokenPrices
from exp.runtime.gateway.budgets import BudgetScope, BudgetScopeKind, SQLiteBudgetStore
from exp.runtime.gateway.contracts import GatewayEvent, GatewayEventKind, GatewayUsage
from exp.runtime.gateway.ledger import GatewayLedgerError, SQLiteAttemptLedger
from exp.runtime.gateway.ledger_service_tiers import card_usage_cost
from exp.runtime.gateway.ledger_test import (
    FakeLedgerClock,
    _authority_fixture,
    _deployment,
    _execution,
    _request,
)
from exp.runtime.gateway.service_tiers import GatewayServiceTierAdmission


def test_long_context_threshold_and_both_cache_write_dimensions() -> None:
    """Every input subset is priced once on either side of the exact threshold."""
    prices = GatewayTokenPrices(
        input_nano_usd_per_million_tokens=1_000_000,
        cached_input_nano_usd_per_million_tokens=100_000,
        cache_creation_input_nano_usd_per_million_tokens=2_000_000,
        cache_creation_1h_input_nano_usd_per_million_tokens=3_000_000,
        output_nano_usd_per_million_tokens=4_000_000,
        long_context=GatewayLongContextTier(
            input_threshold_tokens=100,
            input_nano_usd_per_million_tokens=2_000_000,
            cached_input_nano_usd_per_million_tokens=200_000,
            cache_creation_input_nano_usd_per_million_tokens=4_000_000,
            cache_creation_1h_input_nano_usd_per_million_tokens=6_000_000,
            output_nano_usd_per_million_tokens=8_000_000,
        ),
    )
    usage = GatewayUsage(
        input_tokens=99,
        output_tokens=10,
        cached_input_tokens=10,
        cache_creation_input_tokens=20,
        cache_creation_1h_input_tokens=5,
    )
    assert card_usage_cost(prices, usage) == 155
    assert card_usage_cost(prices, usage.model_copy(update={"input_tokens": 100})) == 312
    assert (
        card_usage_cost(prices, usage.model_copy(update={"cache_creation_1h_input_tokens": None}))
        is None
    )


@pytest.mark.parametrize("hour_rate,expected", [(5_000_000, 408), (6_000_000, None), (None, None)])
def test_tier_long_context_without_ttl_prices_only_uniform_writes(
    hour_rate: int | None, expected: int | None
) -> None:
    """A service card's selected long-context schedule owns rate-invariance, not base prices."""
    prices = GatewayTokenPrices(
        input_nano_usd_per_million_tokens=2_000_000,
        output_nano_usd_per_million_tokens=4_000_000,
        cache_creation_input_nano_usd_per_million_tokens=2_500_000,
        cache_creation_1h_input_nano_usd_per_million_tokens=2_500_000,
        long_context=GatewayLongContextTier(
            input_threshold_tokens=100,
            input_nano_usd_per_million_tokens=4_000_000,
            cached_input_nano_usd_per_million_tokens=400_000,
            cache_creation_input_nano_usd_per_million_tokens=5_000_000,
            cache_creation_1h_input_nano_usd_per_million_tokens=hour_rate,
            output_nano_usd_per_million_tokens=2_000_000,
        ),
    )
    usage = GatewayUsage(
        input_tokens=100, output_tokens=10, cached_input_tokens=20, cache_creation_input_tokens=60
    )
    assert card_usage_cost(prices, usage) == expected
    assert usage.cache_creation_1h_input_tokens is None


@pytest.mark.parametrize("served", ["priority", "default", None])
def test_tier_receipt_survives_reopen_and_holds_unknown(tmp_path: Path, served: str | None) -> None:
    """A missing receipt keeps reserved funds, while witnessed fallback bills standard."""
    clock = FakeLedgerClock()
    store, ledger, key = _authority_fixture(tmp_path, clock)
    budgets = SQLiteBudgetStore(store.database_path, clock=clock)
    budgets.set_limit(
        organization_id="org-one",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_nano_usd=1000,
        strict_unknown_cost=True,
    )
    auth = store.authorize_request(
        raw_key=key,
        alias="coding",
        request=_request("tier"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=auth)
    admission = GatewayServiceTierAdmission(
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
    attempt = ledger.start_attempt(
        snapshot=_execution(auth),
        deployment=_deployment(billing_source=BillingSource.HOST_MANAGED),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_nano_usd=300,
        service_tier=admission,
    )
    reopened = SQLiteAttemptLedger(store.database_path, clock=clock)
    receipt = (
        None
        if served is None
        else admission.settlement(
            served="priority" if served == "priority" else "default", resolution="confirmed"
        )
    )
    terminal = GatewayEvent(
        kind=GatewayEventKind.COMPLETED,
        sequence_number=0,
        usage=GatewayUsage(input_tokens=10, output_tokens=5),
    )
    reopened.finish_attempt(
        attempt_id=attempt, terminal_event=terminal, failure=None, service_tier=receipt
    )
    reopened.finish_attempt(
        attempt_id=attempt, terminal_event=terminal, failure=None, service_tier=receipt
    )
    with sqlite3.connect(store.database_path) as connection:
        cost, settled, persisted = connection.execute(
            "SELECT estimated_cost_nano_usd, budget_settled_nano_usd, "
            "service_tier_admission FROM gateway_attempts"
        ).fetchone()
        reserved, charged = connection.execute(
            "SELECT reserved_nano_usd, settled_nano_usd FROM gateway_monthly_budgets"
        ).fetchone()
    assert GatewayServiceTierAdmission.model_validate_json(persisted) == admission
    expected = None if served is None else (40 if served == "priority" else 20)
    assert cost == settled == expected
    assert (reserved, charged) == ((300, 0) if served is None else (0, expected))
    if served is None:
        recovered = SQLiteAttemptLedger(store.database_path, clock=clock)
        confirmed = admission.settlement(served="default", resolution="confirmed")
        forged = confirmed.model_copy(update={"prices": admission.requested_prices})
        with pytest.raises(GatewayLedgerError, match="admitted schedules"):
            recovered.finish_attempt(
                attempt_id=attempt, terminal_event=terminal, failure=None, service_tier=forged
            )
        different = terminal.model_copy(
            update={"usage": GatewayUsage(input_tokens=11, output_tokens=5)}
        )
        with pytest.raises(GatewayLedgerError, match="durable usage"):
            recovered.finish_attempt(
                attempt_id=attempt, terminal_event=different, failure=None, service_tier=confirmed
            )
        for _ in range(2):
            recovered.finish_attempt(
                attempt_id=attempt, terminal_event=terminal, failure=None, service_tier=confirmed
            )
        with pytest.raises(GatewayLedgerError, match="settled tier"):
            recovered.finish_attempt(
                attempt_id=attempt,
                terminal_event=terminal,
                failure=None,
                service_tier=admission.settlement(served="priority", resolution="confirmed"),
            )
        with sqlite3.connect(store.database_path) as connection:
            assert connection.execute(
                "SELECT reserved_nano_usd, settled_nano_usd FROM gateway_monthly_budgets"
            ).fetchone() == (0, 20)
            assert connection.execute(
                "SELECT count(*), estimated_cost_nano_usd, budget_settled_nano_usd "
                "FROM gateway_attempts"
            ).fetchone() == (1, 20, 20)


@pytest.mark.parametrize("missing", ["usage", "rate", "ttl"])
def test_confirmed_unpriceable_receipt_replay_preserves_hold(tmp_path: Path, missing: str) -> None:
    """An acknowledged hold remains idempotent even when a confirmed tier cannot be priced."""
    clock = FakeLedgerClock()
    store, ledger, key = _authority_fixture(tmp_path, clock)
    budgets = SQLiteBudgetStore(store.database_path, clock=clock)
    budgets.set_limit(
        organization_id="org-one",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_nano_usd=1000,
        strict_unknown_cost=True,
    )
    auth = store.authorize_request(
        raw_key=key,
        alias="coding",
        request=_request("held receipt"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=auth)
    admission = GatewayServiceTierAdmission(
        requested="priority",
        standard_prices=GatewayTokenPrices(),
        requested_prices=GatewayTokenPrices(
            input_nano_usd_per_million_tokens=2_000_000,
            output_nano_usd_per_million_tokens=None if missing == "rate" else 4_000_000,
            cache_creation_input_nano_usd_per_million_tokens=2_500_000,
            cache_creation_1h_input_nano_usd_per_million_tokens=4_000_000,
        ),
    )
    attempt = ledger.start_attempt(
        snapshot=_execution(auth),
        deployment=_deployment(billing_source=BillingSource.HOST_MANAGED),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_nano_usd=300,
        service_tier=admission,
    )
    receipt = admission.settlement(served="priority", resolution="confirmed")
    usage = (
        None
        if missing == "usage"
        else GatewayUsage(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=5 if missing == "ttl" else None,
        )
    )
    terminal = GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=0, usage=usage)
    ledger.finish_attempt(
        attempt_id=attempt, terminal_event=terminal, failure=None, service_tier=receipt
    )
    ledger = SQLiteAttemptLedger(store.database_path, clock=clock)
    ledger.finish_attempt(
        attempt_id=attempt, terminal_event=terminal, failure=None, service_tier=receipt
    )
    with pytest.raises(GatewayLedgerError, match="durable usage"):
        ledger.finish_attempt(
            attempt_id=attempt,
            terminal_event=terminal.model_copy(
                update={"usage": GatewayUsage(input_tokens=11, output_tokens=5)}
            ),
            failure=None,
            service_tier=receipt,
        )
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute(
            "SELECT reserved_nano_usd, settled_nano_usd FROM gateway_monthly_budgets"
        ).fetchone() == (300, 0)
        assert connection.execute(
            "SELECT estimated_cost_nano_usd, budget_settled_nano_usd FROM gateway_attempts"
        ).fetchone() == (None, None)


@pytest.mark.parametrize("served", ["default", "priority"])
def test_lost_settlement_after_crash_recovers_billing_not_delivery(
    tmp_path: Path, served: str
) -> None:
    """An expired dispatched row accepts one observed receipt without claiming delivery."""
    clock = FakeLedgerClock()
    store, ledger, key = _authority_fixture(tmp_path, clock)
    budgets = SQLiteBudgetStore(store.database_path, clock=clock)
    budgets.set_limit(
        organization_id="org-one",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_nano_usd=1000,
        strict_unknown_cost=True,
    )
    auth = store.authorize_request(
        raw_key=key,
        alias="coding",
        request=_request("lost receipt"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=auth)
    admission = GatewayServiceTierAdmission(
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
    attempt = ledger.start_attempt(
        snapshot=_execution(auth),
        deployment=_deployment(billing_source=BillingSource.HOST_MANAGED),
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_nano_usd=300,
        service_tier=admission,
    )
    clock.advance(60)
    restarted = SQLiteAttemptLedger(store.database_path, clock=clock)
    assert restarted.reconcile_crashed_requests(cleanup_grace=timedelta()) == (0, 1)
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute(
            "SELECT reserved_nano_usd, settled_nano_usd FROM gateway_monthly_budgets"
        ).fetchone() == (300, 0)
    receipt = admission.settlement(
        served="default" if served == "default" else "priority", resolution="confirmed"
    )
    usage = GatewayUsage(input_tokens=10, output_tokens=5)
    terminal = GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=0, usage=usage)
    forged = receipt.model_copy(
        update={"prices": GatewayTokenPrices(input_nano_usd_per_million_tokens=99)}
    )
    with pytest.raises(GatewayLedgerError, match="admitted schedules"):
        restarted.finish_attempt(
            attempt_id=attempt, terminal_event=terminal, failure=None, service_tier=forged
        )
    for invalid in (
        terminal.model_copy(update={"usage_estimated": True}),
        terminal.model_copy(update={"usage": None}),
    ):
        with pytest.raises(GatewayLedgerError, match="complete observed usage"):
            restarted.finish_attempt(
                attempt_id=attempt, terminal_event=invalid, failure=None, service_tier=receipt
            )
    for _ in range(2):
        restarted.finish_attempt(
            attempt_id=attempt, terminal_event=terminal, failure=None, service_tier=receipt
        )
    with pytest.raises(GatewayLedgerError, match="durable usage"):
        restarted.finish_attempt(
            attempt_id=attempt,
            terminal_event=terminal.model_copy(
                update={"usage": GatewayUsage(input_tokens=11, output_tokens=5)}
            ),
            failure=None,
            service_tier=receipt,
        )
    expected = 20 if served == "default" else 40
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute(
            "SELECT count(*), state, input_tokens, output_tokens, usage_source, "
            "estimated_cost_nano_usd, budget_settled_nano_usd FROM gateway_attempts"
        ).fetchone() == (1, "unknown_after_crash", 10, 5, "observed", expected, expected)
        assert connection.execute("SELECT terminal_state FROM gateway_requests").fetchone() == (
            "unknown_after_crash",
        )
        assert connection.execute(
            "SELECT reserved_nano_usd, settled_nano_usd FROM gateway_monthly_budgets"
        ).fetchone() == (0, expected)


def test_receipt_cannot_inject_a_price(tmp_path: Path) -> None:
    """The durable admission, not a settlement caller, owns the accepted rates."""
    clock = FakeLedgerClock()
    store, ledger, key = _authority_fixture(tmp_path, clock)
    auth = store.authorize_request(
        raw_key=key,
        alias="coding",
        request=_request("tier"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=auth)
    admission = GatewayServiceTierAdmission(
        requested="priority",
        standard_prices=GatewayTokenPrices(),
        requested_prices=GatewayTokenPrices(input_nano_usd_per_million_tokens=2),
    )
    attempt = ledger.start_attempt(
        snapshot=_execution(auth),
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
        service_tier=admission,
    )
    receipt = admission.settlement(served="priority", resolution="confirmed").model_copy(
        update={"prices": GatewayTokenPrices(input_nano_usd_per_million_tokens=3)}
    )
    with pytest.raises(GatewayLedgerError, match="admitted schedules"):
        ledger.finish_attempt(
            attempt_id=attempt,
            terminal_event=GatewayEvent(
                kind=GatewayEventKind.COMPLETED,
                sequence_number=0,
                usage=GatewayUsage(input_tokens=1, output_tokens=0),
            ),
            failure=None,
            service_tier=receipt,
        )
