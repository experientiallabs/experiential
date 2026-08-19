"""Tests for content-free attempt accounting, recovery, and usage."""

from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from wmo.common.models.catalog import BillingSource, GatewayDeploymentMetadata, GatewayTokenPrices
from wmo.common.models.gateway_catalog import ExactModelDeployment
from wmo.runtime.gateway.contracts import (
    AuthorizationSnapshot,
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
from wmo.runtime.gateway.ledger import (
    GatewayLedgerError,
    IdempotencyConflictError,
    IdempotencyReplayUnavailableError,
    SQLiteAttemptLedger,
)
from wmo.runtime.gateway.sqlite.store import SQLiteGatewayStore

_CATALOG_DIGEST = "a" * 64


class FakeLedgerClock:
    """Controllable wall and monotonic clock for ledger tests."""

    def __init__(self) -> None:
        """Initialize fixed wall and monotonic times."""
        self.wall = datetime(2026, 8, 18, 20, 0, tzinfo=UTC)
        self.monotonic_value = 1_000.0

    def now(self) -> datetime:
        """Return the controlled wall time."""
        return self.wall

    def monotonic(self) -> float:
        """Return the controlled monotonic time."""
        return self.monotonic_value

    def advance(self, seconds: float) -> None:
        """Advance wall and monotonic time equally.

        Args:
            seconds: Elapsed seconds.
        """
        self.wall += timedelta(seconds=seconds)
        self.monotonic_value += seconds


def _deployment(
    *,
    priced: bool = True,
    billing_source: BillingSource = BillingSource.CUSTOMER_MANAGED,
) -> ExactModelDeployment:
    """Create one exact singleton deployment with optional known rates."""
    prices = (
        GatewayTokenPrices(
            input_micro_usd_per_million_tokens=2_000_000,
            cached_input_micro_usd_per_million_tokens=1_000_000,
            output_micro_usd_per_million_tokens=4_000_000,
            reasoning_micro_usd_per_million_tokens=5_000_000,
        )
        if priced
        else GatewayTokenPrices()
    )
    return ExactModelDeployment(
        deployment_id="deployment-one",
        source_alias="source-one",
        exact_model_id="exact-one",
        connection="connection-one",
        provider="openai",
        provider_model="provider-model-canary",
        billing_source=billing_source,
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        gateway=GatewayDeploymentMetadata(
            prices=prices,
            pricing_source="operator-authored",
            pricing_effective_at=datetime(2026, 8, 18, tzinfo=UTC),
        ),
    )


def _request(content: str, *, idempotency_key: str | None = None) -> GatewayRequest:
    """Create one request whose content must not enter SQLite."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content=content),),
        idempotency_key=idempotency_key,
    )


def _authority_fixture(
    tmp_path: Path,
    clock: FakeLedgerClock,
) -> tuple[SQLiteGatewayStore, SQLiteAttemptLedger, str]:
    """Create explicit authority and one granted key for ledger tests."""
    path = tmp_path / "gateway.db"
    store = SQLiteGatewayStore(path, clock=clock)
    ledger = SQLiteAttemptLedger(path, clock=clock)
    store.create_organization(organization_id="org-one", slug="one", display_name="One")
    store.create_identity(
        organization_id="org-one", identity_id="identity-one", display_name="Identity"
    )
    store.register_catalog_snapshot(
        organization_id="org-one",
        snapshot_ref="snapshot-one",
        catalog_sha256=_CATALOG_DIGEST,
    )
    store.activate_alias_revision(
        organization_id="org-one",
        alias_id="alias-one",
        alias_name="coding",
        revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        snapshot_ref="snapshot-one",
        catalog_sha256=_CATALOG_DIGEST,
    )
    store.grant_alias(organization_id="org-one", identity_id="identity-one", alias_id="alias-one")
    issued = store.issue_virtual_key(
        organization_id="org-one", identity_id="identity-one", key_id="key-one"
    )
    return store, ledger, issued.raw_key


def _execution(authorization: AuthorizationSnapshot) -> ExecutionSnapshot:
    """Bind a typed authorization snapshot to the singleton route."""
    return ExecutionSnapshot(
        authorization=authorization,
        exact_model_id="exact-one",
        pool_id="pool-one",
        deployment_ids=("deployment-one",),
    )


def test_attempt_usage_and_integer_cost_are_content_free(tmp_path: Path) -> None:
    """Attempt settlement preserves normalized usage and attributed integer cost only."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    request = _request("prompt-content-canary")
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=request,
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    attempt_id = ledger.start_attempt(
        snapshot=_execution(authorization), deployment=_deployment(), route_depth=0
    )
    ledger.record_route_context(
        attempt_id=attempt_id,
        route_reason="direct_alias",
        fallback_reason=None,
    )
    clock.advance(0.125)

    ledger.finish_attempt(
        attempt_id=attempt_id,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=3,
            usage=GatewayUsage(
                input_tokens=1_000,
                cached_input_tokens=100,
                output_tokens=500,
                reasoning_tokens=50,
            ),
        ),
        failure=None,
    )

    usage = ledger.usage(organization_id="org-one")
    assert len(usage) == 1
    assert usage[0].requests == 1
    assert usage[0].attempts == 1
    assert usage[0].known_estimated_cost_micro_usd == 4_350
    assert usage[0].unknown_cost_attempts == 0
    assert 124 <= usage[0].total_latency_ms <= 126
    assert usage[0].average_latency_ms is not None
    assert usage[0].terminal_counts[0].state == "completed"

    durable = (tmp_path / "gateway.db").read_bytes()
    wal = tmp_path / "gateway.db-wal"
    if wal.exists():
        durable += wal.read_bytes()
    assert b"prompt-content-canary" not in durable
    assert raw_key.encode() not in durable
    assert b"provider-model-canary" not in durable


def test_attempt_retains_dispatch_billing_source_after_catalog_change(tmp_path: Path) -> None:
    """Attempt attribution remains frozen when later catalog ownership changes."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("billing-freeze"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    dispatched = _deployment(billing_source=BillingSource.HOST_MANAGED)
    attempt_id = ledger.start_attempt(
        snapshot=_execution(authorization), deployment=dispatched, route_depth=0
    )

    authored_after_dispatch = dispatched.model_copy(
        update={"billing_source": BillingSource.CUSTOMER_MANAGED}
    )
    assert authored_after_dispatch.billing_source == BillingSource.CUSTOMER_MANAGED
    ledger.finish_attempt(
        attempt_id=attempt_id,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=1,
            usage=GatewayUsage(input_tokens=1, output_tokens=1),
        ),
        failure=None,
    )

    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        row = connection.execute(
            "SELECT billing_source, state FROM gateway_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
    finally:
        connection.close()
    assert row == (BillingSource.HOST_MANAGED.value, "completed")


def test_unknown_prices_remain_unknown_instead_of_zero(tmp_path: Path) -> None:
    """Observed token usage with absent rates increments unknown-cost accounting."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("unknown-price"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    attempt_id = ledger.start_attempt(
        snapshot=_execution(authorization), deployment=_deployment(priced=False), route_depth=0
    )
    ledger.finish_attempt(
        attempt_id=attempt_id,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=1,
            usage=GatewayUsage(input_tokens=1, output_tokens=1),
        ),
        failure=None,
    )

    usage = ledger.usage(organization_id="org-one")[0]
    assert usage.known_estimated_cost_micro_usd == 0
    assert usage.unknown_cost_attempts == 1


def test_cancelled_post_commit_attempt_keeps_observed_billable_usage(tmp_path: Path) -> None:
    """Cancellation does not erase observed provider usage or attributed cost."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("cancelled-stream"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    attempt_id = ledger.start_attempt(
        snapshot=_execution(authorization), deployment=_deployment(), route_depth=0
    )
    cancelled = GatewayFailure(
        failure_class=GatewayFailureClass.CANCELLED,
        safe_message="client disconnected",
    )
    ledger.finish_attempt(
        attempt_id=attempt_id,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.FAILED,
            sequence_number=2,
            failure=cancelled,
            usage=GatewayUsage(input_tokens=100, output_tokens=20),
        ),
        failure=cancelled,
    )

    usage = ledger.usage(organization_id="org-one")[0]
    assert usage.known_estimated_cost_micro_usd == 280
    assert usage.terminal_counts[0].attempts == 1
    assert usage.terminal_counts[0].state == "cancelled"


def test_failed_attempt_terminalizes_its_parent_request(tmp_path: Path) -> None:
    """An ordinary terminal provider failure closes both attempt and request state."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("failed-request"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    attempt_id = ledger.start_attempt(
        snapshot=_execution(authorization), deployment=_deployment(), route_depth=0
    )
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
        safe_message="provider request failed",
    )

    ledger.finish_attempt(attempt_id=attempt_id, terminal_event=None, failure=failure)

    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        request_state = connection.execute(
            "SELECT terminal_state FROM gateway_requests WHERE request_id = ?",
            (authorization.request_id,),
        ).fetchone()[0]
        attempt_state = connection.execute(
            "SELECT state FROM gateway_attempts WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()[0]
    finally:
        connection.close()
    assert request_state == "failed"
    assert attempt_state == "failed"


def test_predispatch_failure_terminalizes_real_sqlite_request(tmp_path: Path) -> None:
    """Accepted routing failures cannot remain unterminated without an attempt row."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("routing-failure"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)

    ledger.finish_request(
        authorization=authorization,
        failure=GatewayFailure(
            failure_class=GatewayFailureClass.INTERNAL,
            safe_message="route activation failed",
        ),
    )

    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        request_state = connection.execute(
            "SELECT terminal_state FROM gateway_requests WHERE request_id = ?",
            (authorization.request_id,),
        ).fetchone()[0]
        attempt_count = connection.execute(
            "SELECT COUNT(*) FROM gateway_attempts WHERE request_id = ?",
            (authorization.request_id,),
        ).fetchone()[0]
    finally:
        connection.close()
    assert request_state == "failed"
    assert attempt_count == 0


def test_intermediate_attempt_can_settle_without_finalizing_parent(tmp_path: Path) -> None:
    """The physical-attempt seam leaves parent finalization to a later route owner."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("future-waterfall"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    attempt_id = ledger.start_attempt(
        snapshot=_execution(authorization), deployment=_deployment(), route_depth=0
    )

    ledger.finish_attempt(
        attempt_id=attempt_id,
        terminal_event=None,
        failure=GatewayFailure(
            failure_class=GatewayFailureClass.TRANSPORT,
            safe_message="retry on sibling route",
        ),
        finalize_request=False,
    )

    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        states = connection.execute(
            """
            SELECT r.terminal_state, a.state
            FROM gateway_requests AS r
            JOIN gateway_attempts AS a ON a.request_id = r.request_id
            WHERE r.request_id = ?
            """,
            (authorization.request_id,),
        ).fetchone()
    finally:
        connection.close()
    assert states == (None, "failed")


def test_terminal_parent_rejects_late_attempt_dispatch(tmp_path: Path) -> None:
    """No retry path can dispatch after durable request terminalization."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    authorization = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("late-dispatch"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    ledger.finish_request(
        authorization=authorization,
        failure=GatewayFailure(
            failure_class=GatewayFailureClass.INTERNAL,
            safe_message="route failed before dispatch",
        ),
    )

    with pytest.raises(GatewayLedgerError, match="already terminal"):
        ledger.start_attempt(
            snapshot=_execution(authorization),
            deployment=_deployment(),
            route_depth=0,
        )


def test_idempotency_is_opt_in_and_restart_replay_fails_closed(tmp_path: Path) -> None:
    """A stored keyed request is never redispatched when response content is unavailable."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    first = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("same", idempotency_key="caller-operation"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=first)

    restarted = SQLiteAttemptLedger(tmp_path / "gateway.db", clock=clock)
    matching = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("same", idempotency_key="caller-operation"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    with pytest.raises(IdempotencyReplayUnavailableError, match="replay is unavailable"):
        restarted.accept_request(authorization=matching)

    conflicting = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("different", idempotency_key="caller-operation"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    with pytest.raises(IdempotencyConflictError, match="different request"):
        restarted.accept_request(authorization=conflicting)


def test_crash_reconciliation_waits_for_deadline_and_cleanup_bound(tmp_path: Path) -> None:
    """Expired accepted work is free while dispatched work becomes unknown after crash."""
    clock = FakeLedgerClock()
    store, ledger, raw_key = _authority_fixture(tmp_path, clock)
    accepted = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("accepted-only"),
        deadline_monotonic=clock.monotonic() + 10,
    )
    ledger.accept_request(authorization=accepted)
    dispatched = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("dispatched", idempotency_key="crash-operation"),
        deadline_monotonic=clock.monotonic() + 10,
    )
    ledger.accept_request(authorization=dispatched)
    ledger.start_attempt(snapshot=_execution(dispatched), deployment=_deployment(), route_depth=0)

    clock.advance(12)
    assert ledger.reconcile_crashed_requests(cleanup_grace=timedelta(seconds=5)) == (1, 0)
    clock.advance(4)
    assert ledger.reconcile_crashed_requests(cleanup_grace=timedelta(seconds=5)) == (0, 1)

    superseding = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request("dispatched", idempotency_key="crash-operation"),
        deadline_monotonic=clock.monotonic() + 10,
    )
    ledger.accept_request(authorization=superseding)
    ledger.start_attempt(snapshot=_execution(superseding), deployment=_deployment(), route_depth=0)

    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM gateway_attempts WHERE state = 'unknown_after_crash'"
            ).fetchone()[0]
            == 1
        )
        assert connection.execute("SELECT COUNT(*) FROM gateway_attempts").fetchone()[0] == 2
    finally:
        connection.close()


def test_concurrent_multi_identity_wal_preserves_receipts_grants_and_attempts(
    tmp_path: Path,
) -> None:
    """Concurrent writers retain each authority mutation and terminal attempt exactly once."""
    path = tmp_path / "gateway.db"
    store = SQLiteGatewayStore(path, busy_timeout_ms=10_000)
    ledger = SQLiteAttemptLedger(path, busy_timeout_ms=10_000)
    store.create_organization(organization_id="org-one", slug="one", display_name="One")
    store.register_catalog_snapshot(
        organization_id="org-one",
        snapshot_ref="snapshot-one",
        catalog_sha256=_CATALOG_DIGEST,
    )
    store.activate_alias_revision(
        organization_id="org-one",
        alias_id="alias-one",
        alias_name="coding",
        revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        snapshot_ref="snapshot-one",
        catalog_sha256=_CATALOG_DIGEST,
    )

    def run_identity(index: int) -> None:
        """Create one identity and account one completed request."""
        identity_id = f"identity-{index}"
        key_id = f"key-{index}"
        store.create_identity(
            organization_id="org-one",
            identity_id=identity_id,
            display_name=f"Identity {index}",
            operation_id=f"operation-identity-{index}",
        )
        store.grant_alias(organization_id="org-one", identity_id=identity_id, alias_id="alias-one")
        issued = store.issue_virtual_key(
            organization_id="org-one",
            identity_id=identity_id,
            key_id=key_id,
            operation_id=f"operation-key-{index}",
        )
        authorization = store.authorize_request(
            raw_key=issued.raw_key,
            alias="coding",
            request=_request(f"concurrent-content-{index}"),
            deadline_monotonic=time.monotonic() + 60,
        )
        ledger.accept_request(authorization=authorization)
        attempt_id = ledger.start_attempt(
            snapshot=_execution(authorization), deployment=_deployment(), route_depth=0
        )
        ledger.finish_attempt(
            attempt_id=attempt_id,
            terminal_event=GatewayEvent(
                kind=GatewayEventKind.COMPLETED,
                sequence_number=1,
                usage=GatewayUsage(input_tokens=1, output_tokens=1),
            ),
            failure=None,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(executor.map(run_identity, range(8)))

    usage = ledger.usage(organization_id="org-one")
    assert len(usage) == 8
    assert sum(item.requests for item in usage) == 8
    assert sum(item.attempts for item in usage) == 8
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM operation_receipts").fetchone()[0] == 16
        assert connection.execute("SELECT COUNT(*) FROM identity_alias_grants").fetchone()[0] == 8
    finally:
        connection.close()
