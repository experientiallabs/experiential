"""Focused tests for SQLite platform row conversion and replay checks."""

from pathlib import Path

import pytest

from exp.common.models.catalog import GatewayLongContextTier, GatewayTokenPrices
from exp.runtime.gateway.budgets import BudgetScope, BudgetScopeKind, SQLiteBudgetStore
from exp.runtime.gateway.contracts import GatewayEvent, GatewayEventKind, GatewayUsage
from exp.runtime.gateway.ledger_test import (
    FakeLedgerClock,
    _authority_fixture,
    _deployment,
    _execution,
    _request,
)
from exp.runtime.gateway.native_settlement import terminal_from_settlement
from exp.runtime.gateway.platform import (
    AttemptReservationRequest,
    AttemptSettlementRequest,
    AttemptUsageSource,
)
from exp.runtime.gateway.sqlite.platform import SQLiteGatewayPlatform


@pytest.mark.parametrize(
    "usage",
    [
        GatewayUsage(input_tokens=19, cached_input_tokens=3),
        GatewayUsage(output_tokens=7, reasoning_tokens=2),
        GatewayUsage(input_tokens=0),
        GatewayUsage(output_tokens=0),
        GatewayUsage(
            input_tokens=19, cache_creation_input_tokens=5, cache_creation_1h_input_tokens=2
        ),
    ],
)
def test_partial_usage_roundtrips_through_public_settlement(
    tmp_path: Path, usage: GatewayUsage
) -> None:
    """A missing primary leg stays unknown through initial settlement and exact replay."""
    clock = FakeLedgerClock()
    store, ledger, key = _authority_fixture(tmp_path, clock)
    authority = store.authorize_request(
        raw_key=key,
        alias="coding",
        request=_request("partial meter"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authority)
    platform = SQLiteGatewayPlatform(store.database_path, attempts=ledger)
    reservation = platform.reserve_attempt(
        AttemptReservationRequest(
            organization_id="org-one",
            snapshot=_execution(authority),
            deployment=_deployment(),
            attempt_ordinal=0,
            route_depth=0,
            maximum_cost_nano_usd=300,
        )
    )
    terminal = GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=0, usage=usage)
    request = AttemptSettlementRequest(
        organization_id="org-one", attempt_id=reservation.attempt_id, terminal_event=terminal
    )
    settlement = platform.settle_attempt(request)
    assert settlement.usage == usage
    assert settlement.usage_source is AttemptUsageSource.OBSERVED
    assert settlement.estimated_cost_nano_usd is None
    assert settlement.settled_nano_usd == 300
    assert platform.settle_attempt(request) == settlement
    assert (
        platform.settle_attempt(
            AttemptSettlementRequest.model_validate_json(request.model_dump_json())
        )
        == settlement
    )
    missing_leg = "input_tokens" if usage.input_tokens is None else "output_tokens"
    changed = terminal.model_copy(update={"usage": usage.model_copy(update={missing_leg: 0})})
    with pytest.raises(ValueError, match="differs from durable accounting evidence"):
        platform.settle_attempt(request.model_copy(update={"terminal_event": changed}))
    assert platform.settle_attempt(request) == settlement
    attributed = platform.usage_attribution(organization_id="org-one").identities[0]
    assert attributed.input_tokens == (usage.input_tokens or 0)
    assert attributed.output_tokens == (usage.output_tokens or 0)
    assert attributed.unknown_cost_attempts == 1
    assert attributed.known_estimated_cost_nano_usd == 0


@pytest.mark.parametrize("output_tokens", [0, 7])
def test_disconnect_settlement_replay_keeps_unknown_cost_and_budget_fallback(
    tmp_path: Path, output_tokens: int
) -> None:
    """Public replay cannot replace a persisted conservative bound with partial cost."""
    clock = FakeLedgerClock()
    store, ledger, key = _authority_fixture(tmp_path, clock)
    budgets = SQLiteBudgetStore(store.database_path, clock=clock)
    budgets.set_limit(
        organization_id="org-one",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_nano_usd=400,
        strict_unknown_cost=True,
    )
    authority = store.authorize_request(
        raw_key=key,
        alias="coding",
        request=_request("disconnect before terminal meter"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authority)
    platform = SQLiteGatewayPlatform(store.database_path, attempts=ledger, budgets=budgets)
    reservation = platform.reserve_attempt(
        AttemptReservationRequest(
            organization_id="org-one",
            snapshot=_execution(authority),
            deployment=_deployment(),
            attempt_ordinal=0,
            route_depth=0,
            maximum_cost_nano_usd=300,
        )
    )
    terminal, _ = terminal_from_settlement(
        {
            "outcome": "failed",
            "dispatched": True,
            "usage_incomplete_due_to_disconnect": True,
            "usage": {"input_tokens": 19, "output_tokens": output_tokens},
            "failure": {"failure_class": "cancelled", "safe_message": "caller disconnected"},
        }
    )
    request = AttemptSettlementRequest(
        organization_id="org-one", attempt_id=reservation.attempt_id, terminal_event=terminal
    )
    settlement = platform.settle_attempt(request)
    assert settlement.state == "cancelled"
    assert settlement.usage == GatewayUsage(input_tokens=19, output_tokens=output_tokens)
    assert settlement.usage_source is AttemptUsageSource.OBSERVED
    assert settlement.estimated_cost_nano_usd is None
    assert settlement.settled_nano_usd == reservation.reserved_nano_usd == 300
    assert platform.settle_attempt(request) == settlement
    # An estimate completing the same disconnect is a different settlement:
    # it prices the meter and labels itself estimated, never observed.
    estimated_request = request.model_copy(
        update={
            "terminal_event": terminal.model_copy(
                update={
                    "usage": GatewayUsage(input_tokens=19, output_tokens=max(output_tokens, 1)),
                    "usage_estimated": True,
                }
            )
        }
    )
    with pytest.raises(ValueError, match="differs from durable accounting evidence"):
        platform.settle_attempt(estimated_request)
    # Public event serialization deliberately omits the trusted native marker.
    # Once settled, even this replay must retain the durable unknown-cost outcome.
    public_replay = AttemptSettlementRequest.model_validate_json(request.model_dump_json())
    assert public_replay.terminal_event is not None
    assert public_replay.terminal_event.usage_incomplete_due_to_disconnect is False
    assert platform.settle_attempt(public_replay) == settlement
    assert budgets.remaining(organization_id="org-one", period="2026-08")[0].charged_nano_usd == 300
    attributed = platform.usage_attribution(organization_id="org-one").identities[0]
    assert attributed.unknown_cost_attempts == 1
    assert attributed.known_estimated_cost_nano_usd == 0


@pytest.mark.parametrize("long_context", [False, True])
@pytest.mark.parametrize(
    "field",
    [
        "cache_creation_input_nano_usd_per_million_tokens",
        "cache_creation_1h_input_nano_usd_per_million_tokens",
    ],
)
def test_reservation_replay_rejects_changed_frozen_cache_write_rate(
    tmp_path: Path, long_context: bool, field: str
) -> None:
    """Changing only one write price must not replay an existing reservation."""
    clock = FakeLedgerClock()
    store, ledger, key = _authority_fixture(tmp_path, clock)
    authority = store.authorize_request(
        raw_key=key,
        alias="coding",
        request=_request("hello"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authority)
    prices = GatewayTokenPrices(
        cache_creation_input_nano_usd_per_million_tokens=3750,
        cache_creation_1h_input_nano_usd_per_million_tokens=6000,
        long_context=GatewayLongContextTier(
            input_threshold_tokens=200000,
            cache_creation_input_nano_usd_per_million_tokens=7500,
            cache_creation_1h_input_nano_usd_per_million_tokens=12000,
        ),
    )
    deployment = _deployment()
    deployment = deployment.model_copy(
        update={"gateway": deployment.gateway.model_copy(update={"prices": prices})}
    )
    request = AttemptReservationRequest(
        organization_id="org-one",
        snapshot=_execution(authority),
        deployment=deployment,
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_nano_usd=10000,
    )
    platform = SQLiteGatewayPlatform(tmp_path / "gateway.db", attempts=ledger)
    first = platform.reserve_attempt(request)
    assert platform.reserve_attempt(request) == first
    if long_context:
        assert prices.long_context is not None
        changed = prices.model_copy(
            update={"long_context": prices.long_context.model_copy(update={field: 1})}
        )
    else:
        changed = prices.model_copy(update={field: 1})
    altered = deployment.model_copy(
        update={"gateway": deployment.gateway.model_copy(update={"prices": changed})}
    )
    with pytest.raises(ValueError, match="differs from durable accounting input"):
        platform.reserve_attempt(request.model_copy(update={"deployment": altered}))
    assert platform.reserve_attempt(request) == first
