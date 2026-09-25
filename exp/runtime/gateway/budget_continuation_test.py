"""Actual SQLite refusals carry scope authority without creating provider attempts."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.budget_continuation import denied_destination_pool
from exp.runtime.gateway.budgets import (
    BudgetRefusalBinding,
    BudgetReservationRejected,
    BudgetScope,
    BudgetScopeKind,
)
from exp.runtime.gateway.budgets_test import (
    _accepted_chain,
    _activate_chain,
    _authority,
    _Clock,
    _request,
    _unknown_attempt,
)
from exp.runtime.gateway.contracts import (
    ExecutionSnapshot,
    GatewayEvent,
    GatewayEventKind,
    GatewayUsage,
)
from exp.runtime.gateway.model_plan import model_execution_snapshot
from exp.runtime.gateway.model_plan_test import catalog as model_catalog
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting, NativeBridgeError
from exp.runtime.gateway.native_accounting_test import _RecordingLedger, _settle, _start
from exp.runtime.gateway.native_execution import InflightRequest, deployment_health_key
from exp.runtime.gateway.native_execution_test import _route
from exp.runtime.gateway.routing import GatewayRoute


class ScopedLedger(_RecordingLedger):
    """Return a precisely bound child refusal and record no rejected provider call."""

    def __init__(self) -> None:
        """Track reservation checks separately from successfully recorded attempts."""
        super().__init__()
        self.checked: list[str] = []

    def start_attempt(
        self,
        *,
        snapshot: object,
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
    ) -> str:
        """Deny each B provider from the same atomic-style immutable scope."""
        assert isinstance(snapshot, ExecutionSnapshot)
        self.checked.append(deployment.deployment_id)
        if snapshot.stage_for_depth(route_depth).pool_id == "pool-b":
            raise BudgetReservationRejected(
                scope_kind=BudgetScopeKind.POOL,
                reason="destination budget exhausted",
                binding=BudgetRefusalBinding(
                    request_id=snapshot.authorization.request_id,
                    organization_id=snapshot.authorization.organization_id,
                    alias_revision_id=snapshot.authorization.alias_revision_id,
                    scope=BudgetScope(
                        kind=BudgetScopeKind.POOL, alias_id="coding", pool_id="pool-b"
                    ),
                    application="destination",
                ),
            )
        attempt_id = super().start_attempt(
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
        )
        self.started[-1]["fallback_reason"] = fallback_reason
        return attempt_id


def test_denied_pool_skips_later_members_without_poisoning_parent() -> None:
    """A denied pool is local to this request and skips its later providers only."""
    catalog = model_catalog()
    auth = _route().snapshot.authorization
    snapshot = model_execution_snapshot(catalog, auth, catalog.pools[0])
    by_id = {d.deployment_id: d for d in catalog.deployments}
    b2 = by_id["b1"].model_copy(update={"deployment_id": "b2"})
    child = snapshot.model_stages[1].model_copy(
        update={"deployment_ids": ("b1", "b2"), "rung_positions": (0, 1)}
    )
    snapshot = snapshot.model_copy(
        update={
            "deployment_ids": ("a1", "b1", "b2", "a2"),
            "model_stages": (snapshot.model_stages[0], child, snapshot.model_stages[2]),
        }
    )
    route = GatewayRoute(
        snapshot=snapshot,
        deployment=by_id["a1"],
        fallback_deployments=(by_id["b1"], b2, by_id["a2"]),
        route_reason="direct",
    )
    ledger = ScopedLedger()
    accounting = NativeAttemptAccounting(ledger)
    entry = InflightRequest(auth, route, _request("test"), time.monotonic() + 30)
    accounting.register(entry)
    first = _start(accounting, ordinal=0, request_id=auth.request_id)
    _settle(
        accounting,
        attempt_id=str(first["attempt_id"]),
        outcome="failed",
        finalize=False,
        request_id=auth.request_id,
        failure={"failure_class": "transport", "safe_message": "failed"},
    )
    result = _start(
        accounting,
        ordinal=1,
        current_depth=0,
        request_id=auth.request_id,
        failure={"failure_class": "transport", "safe_message": "failed", "failover_eligible": True},
    )
    assert result["route_depth"] == 3
    assert ledger.checked == ["a1", "b1", "a2"]
    assert [row["deployment_id"] for row in ledger.started] == ["a1", "a2"]
    assert entry.denied_destination_pools == {"pool-b"}
    assert not accounting.health.suppressed(
        (auth.catalog_sha256, by_id["b1"].deployment_id, by_id["b1"].connection_sha256)
    )


@pytest.mark.parametrize("pre_denied", [False, True])
@pytest.mark.parametrize("allowed", [False, True])
def test_denied_child_pool_cannot_bypass_conditional_parent_successor(
    pre_denied: bool, allowed: bool
) -> None:
    """Every path skipping a denied child retains the same conditional-fallback ladder."""
    catalog = model_catalog()
    auth = _route().snapshot.authorization
    snapshot = model_execution_snapshot(catalog, auth, catalog.pools[0])
    by_id = {d.deployment_id: d for d in catalog.deployments}
    parent = by_id["a2"]
    restricted = parent.model_copy(
        update={
            "gateway": parent.gateway.model_copy(
                update={
                    "capabilities": parent.gateway.capabilities.model_copy(
                        update={"failover_only_on": ("transport",) if allowed else ("refusal",)}
                    )
                }
            )
        }
    )
    route = GatewayRoute(
        snapshot=snapshot,
        deployment=by_id["a1"],
        fallback_deployments=(by_id["b1"], restricted),
        route_reason="direct",
    )
    ledger = ScopedLedger()
    accounting = NativeAttemptAccounting(ledger)
    entry = InflightRequest(auth, route, _request("test"), time.monotonic() + 30)
    entry.recovery_reason = "scoped_recovery"
    if pre_denied:
        entry.denied_destination_pools.add("pool-b")
    accounting.register(entry)
    first = _start(accounting, ordinal=0, request_id=auth.request_id)
    assert ledger.started[0]["dispatch_reason"] == "scoped_recovery"
    _settle(
        accounting,
        attempt_id=str(first["attempt_id"]),
        outcome="failed",
        finalize=False,
        request_id=auth.request_id,
        failure={"failure_class": "transport", "safe_message": "failed"},
    )
    result = _start(
        accounting,
        ordinal=1,
        current_depth=0,
        request_id=auth.request_id,
        failure={"failure_class": "transport", "safe_message": "failed", "failover_eligible": True},
    )
    checked = ["a1"] if pre_denied else ["a1", "b1"]
    if allowed:
        assert result["route_depth"] == 2
        assert ledger.checked == [*checked, "a2"]
        assert [row["deployment_id"] for row in ledger.started] == ["a1", "a2"]
        assert ledger.started[-1]["fallback_reason"] == "failover_only_on:transport"
        assert ledger.started[-1]["dispatch_reason"] != "scoped_recovery"
    else:
        assert result["exhausted"] is True
        assert ledger.checked == checked
        assert [row["deployment_id"] for row in ledger.started] == ["a1"]


def test_child_pool_refusal_continues_to_funded_parent_suffix(tmp_path: Path) -> None:
    """B's actual atomic budget refusal skips B only; A's next provider reserves normally."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    catalog = _activate_chain(store, tmp_path)
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.POOL, alias_id="coding", pool_id="child-pool"),
        limit_nano_usd=0,
    )
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.POOL, alias_id="coding", pool_id="pool"),
        limit_nano_usd=1000,
    )
    snapshot = _accepted_chain(store, ledger, clock, key, catalog)
    by_id = {d.deployment_id: d for d in catalog.deployments}
    route = GatewayRoute(
        snapshot=snapshot,
        deployment=by_id["primary"],
        fallback_deployments=(by_id["child"], by_id["secondary"]),
        route_reason="direct",
    )
    accounting = NativeAttemptAccounting(ledger)
    entry = InflightRequest(
        snapshot.authorization,
        route,
        _request("chain-budget"),
        time.monotonic() + 30,
        total_attempts=1,
        attempt_counts=[1, 0, 0],
    )
    accounting.register(entry)
    result = json.loads(
        accounting.start_attempt(
            json.dumps(
                {
                    "request_id": snapshot.authorization.request_id,
                    "attempt_ordinal": 1,
                    "current_depth": 0,
                    "failure": {
                        "failure_class": "transport",
                        "safe_message": "failed",
                        "failover_eligible": True,
                        "retryable_same_deployment": False,
                    },
                }
            )
        )
    )
    assert result["route_depth"] == 2
    assert entry.denied_destination_pools == {"child-pool"}
    assert entry.total_attempts == 2
    with sqlite3.connect(ledger.database_path) as db:
        assert db.execute("SELECT deployment_id FROM gateway_attempts").fetchall() == [
            ("secondary",)
        ]
    accounting.abandon(json.dumps({"request_id": snapshot.authorization.request_id}))


@pytest.mark.parametrize(
    "mismatch",
    ["request_id", "organization_id", "alias_revision_id", "root", "shared", "unknown", "pool"],
)
def test_only_exact_destination_refusal_is_skippable(tmp_path: Path, mismatch: str) -> None:
    """Unknown or stale refusal authority cannot bypass request-wide budget gates."""
    clock = _Clock()
    store, ledger, _budgets, key = _authority(tmp_path, clock)
    catalog = _activate_chain(store, tmp_path)
    snapshot = _accepted_chain(store, ledger, clock, key, catalog)
    binding = BudgetRefusalBinding(
        request_id=snapshot.authorization.request_id,
        organization_id="org",
        alias_revision_id=snapshot.authorization.alias_revision_id,
        scope=BudgetScope(kind=BudgetScopeKind.POOL, alias_id="coding", pool_id="child-pool"),
        application="destination",
    )
    assert (
        denied_destination_pool(
            BudgetReservationRejected(
                scope_kind=BudgetScopeKind.POOL, reason="denied", binding=binding
            ),
            snapshot,
            1,
        )
        == "child-pool"
    )
    if mismatch in ("request_id", "organization_id", "alias_revision_id"):
        binding = binding.model_copy(update={mismatch: "different"})
    elif mismatch in ("root", "shared"):
        binding = binding.model_copy(update={"application": mismatch})
    elif mismatch == "pool":
        binding = binding.model_copy(
            update={
                "scope": BudgetScope(kind=BudgetScopeKind.POOL, alias_id="coding", pool_id="pool")
            }
        )
    rejection = BudgetReservationRejected(
        scope_kind=BudgetScopeKind.POOL,
        reason="denied",
        binding=None if mismatch == "unknown" else binding,
    )
    assert denied_destination_pool(rejection, snapshot, 1) is None


@pytest.mark.parametrize("terminal", ["root", "team", "identity", "unknown"])
@pytest.mark.parametrize("child_key", ["000-child", "zzz-child"])
def test_shared_denial_precedes_child_refusal(
    tmp_path: Path, terminal: str, child_key: str
) -> None:
    """A child-only binding requires every request-wide gate to pass atomically."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    if terminal == "unknown":
        _unknown_attempt(store, ledger, clock, key, "unknown-before-chain")
    catalog = _activate_chain(store, tmp_path)
    scope = (
        BudgetScope(kind=BudgetScopeKind.POOL, alias_id="coding", pool_id="pool")
        if terminal == "root"
        else BudgetScope(kind=BudgetScopeKind.IDENTITY, identity_id="identity")
        if terminal == "identity"
        else BudgetScope(kind=BudgetScopeKind.TEAM)
    )
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=scope,
        limit_nano_usd=1000 if terminal == "unknown" else 0,
        strict_unknown_cost=terminal == "unknown",
    )
    _, child_limit = budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.POOL, alias_id="coding", pool_id="child-pool"),
        limit_nano_usd=0,
    )
    with sqlite3.connect(ledger.database_path) as db:
        db.execute(
            "UPDATE gateway_monthly_budgets SET scope_key=? WHERE budget_id=?",
            (child_key, child_limit.budget_id),
        )
        before_attempts = db.execute("SELECT COUNT(*) FROM gateway_attempts").fetchone()
        before_charges = db.execute(
            "SELECT COUNT(*) FROM gateway_attempt_budget_charges"
        ).fetchone()
    snapshot = _accepted_chain(store, ledger, clock, key, catalog)
    with pytest.raises(BudgetReservationRejected) as caught:
        ledger.start_attempt(
            snapshot=snapshot,
            deployment=catalog.deployments[-1],
            attempt_ordinal=0,
            route_depth=1,
            maximum_cost_nano_usd=38,
        )
    assert caught.value.scope_kind == scope.kind
    assert caught.value.binding is not None
    assert caught.value.binding.application in ("root", "shared")
    assert denied_destination_pool(caught.value, snapshot, 1) is None
    if terminal == "unknown":
        assert "prior attempts with unknown cost" in str(caught.value)
    with sqlite3.connect(ledger.database_path) as db:
        assert db.execute("SELECT COUNT(*) FROM gateway_attempts").fetchone() == before_attempts
        assert (
            db.execute("SELECT COUNT(*) FROM gateway_attempt_budget_charges").fetchone()
            == before_charges
        )


def test_later_child_denial_keeps_root_failure_terminal(tmp_path: Path) -> None:
    """A cheaper parent suffix cannot evade a root refusal encountered at a child."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    catalog = _activate_chain(store, tmp_path)
    snapshot = _accepted_chain(store, ledger, clock, key, catalog)
    child = catalog.deployments[-1]
    first_child = ledger.start_attempt(
        snapshot=snapshot,
        deployment=child,
        attempt_ordinal=0,
        route_depth=1,
        maximum_cost_nano_usd=90,
    )
    ledger.finish_attempt(
        attempt_id=first_child,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=0,
            usage=GatewayUsage(input_tokens=80, output_tokens=5),
        ),
        failure=None,
        finalize_request=False,
    )
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.POOL, alias_id="coding", pool_id="pool"),
        limit_nano_usd=100,
    )
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.POOL, alias_id="coding", pool_id="child-pool"),
        limit_nano_usd=90,
    )
    # The remaining root allowance is ten; an inexpensive suffix would fit,
    # but the actual child checks reject the request's root allocation first.
    parent = catalog.deployments[1].model_copy(
        update={
            "gateway": catalog.deployments[1].gateway.model_copy(
                update={
                    "prices": catalog.deployments[1].gateway.prices.model_copy(
                        update={
                            "input_nano_usd_per_million_tokens": 1,
                            "output_nano_usd_per_million_tokens": 1,
                        }
                    )
                }
            )
        }
    )
    route = GatewayRoute(
        snapshot=snapshot,
        deployment=catalog.deployments[0],
        fallback_deployments=(child, parent),
        route_reason="direct",
    )
    accounting = NativeAttemptAccounting(ledger)
    entry = InflightRequest(
        snapshot.authorization,
        route,
        _request("chain-budget"),
        time.monotonic() + 30,
        total_attempts=1,
        attempt_counts=[0, 1, 0],
    )
    accounting.register(entry)
    with pytest.raises(NativeBridgeError) as caught:
        _start(
            accounting,
            ordinal=1,
            current_depth=0,
            request_id=snapshot.authorization.request_id,
            failure={
                "failure_class": "transport",
                "safe_message": "failed",
                "failover_eligible": True,
                "retryable_same_deployment": False,
            },
        )
    assert json.loads(caught.value.public_error_json)["status_code"] == 429
    assert not entry.denied_destination_pools
    with sqlite3.connect(ledger.database_path) as db:
        assert db.execute("SELECT deployment_id FROM gateway_attempts").fetchall() == [("child",)]


@pytest.mark.parametrize("suffix", ["parent", "same-pool", "other-pool"])
def test_denial_after_a_child_attempt_preserves_forward_pool_boundaries(
    tmp_path: Path, suffix: str
) -> None:
    """A later real child denial skips its remaining leaves, not unrelated funded pools."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    catalog = _activate_chain(store, tmp_path)
    snapshot = _accepted_chain(store, ledger, clock, key, catalog)
    primary, secondary, child = catalog.deployments
    child2 = child.model_copy(update={"deployment_id": "child2"})
    child3 = child.model_copy(update={"deployment_id": "child3"})
    target = secondary if suffix == "parent" else child3
    target_pool = "pool" if suffix == "parent" else "child-pool"
    if suffix == "other-pool":
        target = child3.model_copy(update={"exact_model_id": "exact-other"})
        target_pool = "other-pool"
    child_stage = snapshot.model_stages[1].model_copy(
        update={"deployment_ids": ("child", "child2"), "rung_positions": (0, 1)}
    )
    suffix_stage = snapshot.model_stages[2].model_copy(
        update={
            "pool_id": target_pool,
            "exact_model_id": target.exact_model_id,
            "deployment_ids": (target.deployment_id,),
            "rung_positions": (2,),
        }
    )
    snapshot = snapshot.model_copy(
        update={
            "deployment_ids": ("primary", "child", "child2", target.deployment_id),
            "model_stages": (snapshot.model_stages[0], child_stage, suffix_stage),
        }
    )
    route = GatewayRoute(
        snapshot=snapshot,
        deployment=primary,
        fallback_deployments=(child, child2, target),
        route_reason="direct",
    )
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.POOL, alias_id="coding", pool_id="pool"),
        limit_nano_usd=1000,
    )
    accounting = NativeAttemptAccounting(ledger)
    entry = InflightRequest(
        snapshot.authorization, route, _request("chain-budget"), time.monotonic() + 30
    )
    accounting.register(entry)
    failure: JsonObject = {
        "failure_class": "transport",
        "safe_message": "failed",
        "failover_eligible": True,
        "retryable_same_deployment": False,
    }
    a = _start(accounting, ordinal=0, request_id=snapshot.authorization.request_id)
    _settle(
        accounting,
        attempt_id=str(a["attempt_id"]),
        outcome="failed",
        finalize=False,
        request_id=snapshot.authorization.request_id,
        failure=failure,
    )
    b = _start(
        accounting,
        ordinal=1,
        current_depth=0,
        request_id=snapshot.authorization.request_id,
        failure=failure,
    )
    assert b["route_depth"] == 1
    _settle(
        accounting,
        attempt_id=str(b["attempt_id"]),
        outcome="failed",
        finalize=False,
        request_id=snapshot.authorization.request_id,
        failure=failure,
    )
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.POOL, alias_id="coding", pool_id="child-pool"),
        limit_nano_usd=0,
    )
    result = _start(
        accounting,
        ordinal=2,
        current_depth=1,
        request_id=snapshot.authorization.request_id,
        failure=failure,
    )
    assert entry.denied_destination_pools == {"child-pool"}
    assert not accounting.health.suppressed(deployment_health_key(entry.authorization, child2))
    with sqlite3.connect(ledger.database_path) as db:
        attempted = [
            row[0]
            for row in db.execute(
                "SELECT deployment_id FROM gateway_attempts ORDER BY attempt_ordinal"
            )
        ]
    if suffix == "same-pool":
        assert result["exhausted"] is True
        terminal_failure = result["failure"]
        assert isinstance(terminal_failure, dict)
        assert terminal_failure["failure_class"] == "quota_exceeded"
        assert attempted == ["primary", "child"]
        assert entry.total_attempts == 2
    else:
        assert result["route_depth"] == 3
        assert attempted == ["primary", "child", target.deployment_id]
        assert entry.total_attempts == 3
        accounting.abandon(json.dumps({"request_id": snapshot.authorization.request_id}))
    assert accounting.entry(snapshot.authorization.request_id) is None
