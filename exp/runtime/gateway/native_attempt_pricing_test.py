"""Tests for the one card a physical attempt reserves, freezes, and settles against."""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

from exp.common.models.catalog import (
    GatewayDeploymentMetadata,
    GatewayServiceTierPrices,
    GatewayTokenPrices,
)
from exp.common.models.catalog_prices import GatewayOffPeakPrices, GatewayPriceWindow
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.common.models.model import BillingSource, ModelCapabilities
from exp.runtime.gateway.attempt_tokens import worst_case_output_tokens
from exp.runtime.gateway.budgets import BudgetScope, BudgetScopeKind, SQLiteBudgetStore
from exp.runtime.gateway.contracts import (
    AttemptId,
    GatewayServiceTierAdmission,
)
from exp.runtime.gateway.execution_snapshot import ExecutionSnapshot
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.ledger_test import (
    FakeLedgerClock,
    _authority_fixture,
    _deployment,
    _execution,
    _request,
)
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_attempt_pricing import deployment_priced_at, priced_attempt
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.native_recovery_test import RecoveryHostFake
from exp.runtime.gateway.native_service_tiers import tier_ceiling
from exp.runtime.gateway.routing import GatewayRoute

_PEAK = GatewayTokenPrices(
    input_nano_usd_per_million_tokens=300_000_000,
    cached_input_nano_usd_per_million_tokens=6_000_000,
    output_nano_usd_per_million_tokens=1_200_000_000,
)
_OFF_PEAK = GatewayServiceTierPrices(
    input_nano_usd_per_million_tokens=150_000_000,
    cached_input_nano_usd_per_million_tokens=3_000_000,
    output_nano_usd_per_million_tokens=600_000_000,
)


def _deepseek() -> ExactModelDeployment:
    """A host-funded DeepSeek rung on its published peak/off-peak schedule."""
    prices = GatewayTokenPrices(
        **_PEAK.model_dump(exclude_defaults=True),
        off_peak=GatewayOffPeakPrices(
            prices=_OFF_PEAK,
            peak_windows=(
                GatewayPriceWindow(days=(1, 2, 3, 4, 5), start_minute=60, end_minute=240),
                GatewayPriceWindow(days=(1, 2, 3, 4, 5), start_minute=360, end_minute=600),
            ),
        ),
    )
    base = _deployment(billing_source=BillingSource.HOST_MANAGED)
    return base.model_copy(
        update={
            "gateway": GatewayDeploymentMetadata(prices=prices),
            "capabilities": ModelCapabilities(maximum_output_tokens=8_192),
        }
    )


def test_deployment_priced_at_copies_only_a_scheduled_rung() -> None:
    """Schedule-free and absent rungs come back as the same object; scheduled ones lose it."""
    deployment = _deepseek()
    peak = datetime(2026, 10, 19, 7, tzinfo=UTC)
    off_peak = datetime(2026, 10, 19, 12, tzinfo=UTC)
    assert deployment_priced_at(deployment, peak).gateway.prices == _PEAK
    assert deployment_priced_at(None, off_peak) is None
    plain = _deployment()
    assert deployment_priced_at(plain, off_peak) is plain
    repriced = deployment_priced_at(deployment, off_peak)
    assert repriced.gateway.prices == GatewayTokenPrices(**_OFF_PEAK.model_dump())
    assert repriced.deployment_id == deployment.deployment_id
    assert deployment.gateway.prices.off_peak is not None  # the catalog object is untouched


def test_priced_attempt_admits_no_tier_on_a_scheduled_rung() -> None:
    """A scheduled card carries no tier cards, so a requested tier prices at the window card."""
    deployment, tier = priced_attempt(
        _deepseek(),
        "priority",
        forwards_tier=True,
        instant=datetime(2026, 10, 24, 12, tzinfo=UTC),
    )
    assert tier is None
    assert deployment.gateway.prices.input_nano_usd_per_million_tokens == 150_000_000


class _CapturingLedger(SQLiteAttemptLedger):
    """The real SQLite ledger, also recording each reservation's card and ceiling."""

    def __init__(self, path: Path, *, clock: FakeLedgerClock) -> None:
        """Open the ledger with no recorded reservations."""
        super().__init__(path, clock=clock)
        self.reservations: list[tuple[ExactModelDeployment, int | None, int | None]] = []

    def start_attempt(
        self,
        *,
        snapshot: ExecutionSnapshot,
        deployment: ExactModelDeployment,
        attempt_ordinal: int,
        route_depth: int,
        maximum_cost_nano_usd: int | None = None,
        reserved_input_tokens: int | None = None,
        reserved_output_tokens: int | None = None,
        route_reason: str | None = None,
        fallback_reason: str | None = None,
        dispatch_reason: str | None = None,
        preferred_deployment: ExactModelDeployment | None = None,
        service_tier: GatewayServiceTierAdmission | None = None,
    ) -> AttemptId:
        """Record the selected deployment and ceiling, then reserve for real."""
        self.reservations.append((deployment, maximum_cost_nano_usd, reserved_input_tokens))
        return super().start_attempt(
            snapshot=snapshot,
            deployment=deployment,
            attempt_ordinal=attempt_ordinal,
            route_depth=route_depth,
            maximum_cost_nano_usd=maximum_cost_nano_usd,
            reserved_input_tokens=reserved_input_tokens,
            reserved_output_tokens=reserved_output_tokens,
            route_reason=route_reason,
            fallback_reason=fallback_reason,
            dispatch_reason=dispatch_reason,
            preferred_deployment=preferred_deployment,
            service_tier=service_tier,
        )


def _priced_request(
    tmp_path: Path, clock: FakeLedgerClock, deployment: ExactModelDeployment
) -> tuple[NativeAttemptAccounting, _CapturingLedger, InflightRequest, Path]:
    """Register one funded request on ``deployment`` behind the real SQLite ledger."""
    store, inner, key = _authority_fixture(tmp_path, clock)
    del inner  # recording replaces the fixture's ledger on the same database
    SQLiteBudgetStore(store.database_path, clock=clock).set_limit(
        organization_id="org-one",
        period="2026-10",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_nano_usd=10**12,
    )
    ledger = _CapturingLedger(store.database_path, clock=clock)
    request = _request("price this attempt at its start")
    authorization = store.authorize_request(
        raw_key=key, alias="coding", request=request, deadline_monotonic=clock.monotonic() + 600
    )
    snapshot = _execution(authorization)
    ledger.accept_request(authorization=authorization)
    accounting = NativeAttemptAccounting(
        ledger,
        recovery_host=RecoveryHostFake(),
        clock=clock.now,
    )
    entry = InflightRequest(
        authorization=authorization,
        route=GatewayRoute(snapshot=snapshot, deployment=deployment, route_reason="direct"),
        request=request,
        deadline_monotonic=time.monotonic() + 600,
    )
    accounting.register(entry)
    return accounting, ledger, entry, store.database_path


def _start(accounting: NativeAttemptAccounting, entry: InflightRequest, ordinal: int) -> str:
    """Start one physical attempt through the data plane's wire shape."""
    started = json.loads(
        accounting.start_attempt(
            json.dumps(
                {
                    "request_id": entry.authorization.request_id,
                    "attempt_ordinal": ordinal,
                    "current_depth": None if ordinal == 0 else 0,
                    "failure": None
                    if ordinal == 0
                    else {
                        "failure_class": "provider_internal",
                        "safe_message": "provider failed",
                        "retryable_same_deployment": True,
                        "failover_eligible": True,
                    },
                }
            )
        )
    )
    return str(started["attempt_id"])


def _settle(
    accounting: NativeAttemptAccounting,
    entry: InflightRequest,
    attempt_id: str,
    *,
    succeeded: bool,
) -> None:
    """Settle one attempt with observed usage, finalizing only a success."""
    accounting.settle(
        json.dumps(
            {
                "request_id": entry.authorization.request_id,
                "attempt_id": attempt_id,
                "outcome": "completed" if succeeded else "failed",
                "usage": {"input_tokens": 10_000, "output_tokens": 2_000},
                "tool_names": [],
                "failure": None
                if succeeded
                else {"failure_class": "provider_internal", "safe_message": "provider failed"},
                "finalize": succeeded,
                "opened": True,
                "dispatched": True,
            }
        )
    )


def _frozen(database: Path, attempt_id: str) -> tuple[object, ...]:
    """Read one attempt's frozen rates, settled cost, and reservation."""
    with sqlite3.connect(database) as connection:
        rates = connection.execute(
            "SELECT input_rate, cached_input_rate, output_rate, estimated_cost_nano_usd "
            "FROM gateway_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        charge = connection.execute(
            "SELECT reserved_nano_usd, settled_nano_usd FROM gateway_attempt_budget_charges "
            "WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
    return (*rates, *charge)


def _assert_reserved_from(
    ledger: _CapturingLedger, index: int, entry: InflightRequest, card: GatewayTokenPrices
) -> None:
    """The reservation ceiling came from exactly the card frozen on the attempt."""
    deployment, ceiling, input_tokens = ledger.reservations[index]
    assert deployment.gateway.prices == card
    assert input_tokens is not None
    assert worst_case_output_tokens(entry.request, deployment) > 0
    assert ceiling == tier_ceiling(entry.request, deployment, None, input_tokens=input_tokens)


def test_a_stream_crossing_the_peak_end_settles_at_the_card_it_reserved(tmp_path: Path) -> None:
    """Started at 09:59:59 UTC Monday (peak) and settled at 10:02:30 (off-peak): all peak."""
    clock = FakeLedgerClock()
    clock.wall = datetime(2026, 10, 19, 9, 59, 59, tzinfo=UTC)
    accounting, ledger, entry, database = _priced_request(tmp_path, clock, _deepseek())
    attempt_id = _start(accounting, entry, 0)
    clock.advance(151)
    _settle(accounting, entry, attempt_id, succeeded=True)
    peak_cost = 10_000 * 300 + 2_000 * 1_200  # tokens x nano-USD per token
    reserved = ledger.reservations[0][1]
    assert _frozen(database, attempt_id) == (
        300_000_000,
        6_000_000,
        1_200_000_000,
        peak_cost,
        reserved,
        peak_cost,
    )
    _assert_reserved_from(ledger, 0, entry, _PEAK)


def test_a_retry_after_the_boundary_prices_its_own_start(tmp_path: Path) -> None:
    """Each physical attempt is its own upstream call: the redial at 10:01:01 is off-peak.

    The first attempt starts at 10:00:58, inside the one-minute guard band after
    the 10:00 peak end, so it still pays peak.
    """
    clock = FakeLedgerClock()
    clock.wall = datetime(2026, 10, 19, 10, 0, 58, tzinfo=UTC)
    accounting, ledger, entry, database = _priced_request(tmp_path, clock, _deepseek())
    first = _start(accounting, entry, 0)
    clock.advance(2)
    _settle(accounting, entry, first, succeeded=False)
    clock.advance(1)
    second = _start(accounting, entry, 1)
    clock.advance(30)
    _settle(accounting, entry, second, succeeded=True)
    off_peak = GatewayTokenPrices(**_OFF_PEAK.model_dump(exclude_defaults=True))
    _assert_reserved_from(ledger, 0, entry, _PEAK)
    _assert_reserved_from(ledger, 1, entry, off_peak)
    first_row, second_row = _frozen(database, first), _frozen(database, second)
    assert first_row[:4] == (300_000_000, 6_000_000, 1_200_000_000, 10_000 * 300 + 2_000 * 1_200)
    off_peak_cost = 10_000 * 150 + 2_000 * 600
    assert second_row == (
        150_000_000,
        3_000_000,
        600_000_000,
        off_peak_cost,
        ledger.reservations[1][1],
        off_peak_cost,
    )
    assert ledger.reservations[1][1] is not None
    assert ledger.reservations[0][1] is not None
    assert 2 * ledger.reservations[1][1] <= ledger.reservations[0][1] + 1
