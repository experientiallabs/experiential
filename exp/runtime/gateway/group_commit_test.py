"""Tests for batched durable group commits over the synchronous attempt ledger."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.models.catalog import BillingSource, GatewayDeploymentMetadata, GatewayTokenPrices
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayMessage,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.gateway.group_commit import GroupCommitAttemptLedger, abandoned_write_outcome
from exp.runtime.gateway.ledger import GatewayLedgerError, SQLiteAttemptLedger
from exp.runtime.gateway.sqlite.store import SQLiteGatewayStore

_CATALOG_DIGEST = "a" * 64


class FakeLedgerClock:
    """Controllable wall and monotonic clock for group-commit tests."""

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


def _deployment() -> ExactModelDeployment:
    """Create one exact singleton deployment with known rates."""
    return ExactModelDeployment(
        deployment_id="deployment-one",
        source_alias="source-one",
        exact_model_id="exact-one",
        connection="connection-one",
        provider="openai",
        provider_model="provider-model-canary",
        billing_source=BillingSource.CUSTOMER_MANAGED,
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        gateway=GatewayDeploymentMetadata(
            prices=GatewayTokenPrices(
                input_micro_usd_per_million_tokens=2_000_000,
                cached_input_micro_usd_per_million_tokens=1_000_000,
                output_micro_usd_per_million_tokens=4_000_000,
                reasoning_micro_usd_per_million_tokens=5_000_000,
            ),
            pricing_source="operator-authored",
            pricing_effective_at=datetime(2026, 8, 18, tzinfo=UTC),
        ),
    )


def _request(content: str) -> GatewayRequest:
    """Create one request whose content must not enter SQLite."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content=content),),
    )


def _authority_fixture(
    tmp_path: Path,
    clock: FakeLedgerClock,
) -> tuple[SQLiteGatewayStore, SQLiteAttemptLedger, str]:
    """Create explicit authority and one granted key for group-commit tests."""
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


def _authorize(
    store: SQLiteGatewayStore,
    clock: FakeLedgerClock,
    raw_key: str,
    content: str,
) -> AuthorizationSnapshot:
    """Authorize one keyed request against the fixture authority."""
    return store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request(content),
        deadline_monotonic=clock.monotonic() + 30,
    )


def _execution(authorization: AuthorizationSnapshot) -> ExecutionSnapshot:
    """Bind a typed authorization snapshot to the singleton route."""
    return ExecutionSnapshot(
        authorization=authorization,
        exact_model_id="exact-one",
        pool_id="pool-one",
        deployment_ids=("deployment-one",),
    )


def test_full_request_lifecycle_commits_durably_through_group_writer(tmp_path: Path) -> None:
    """Acceptance, dispatch with route context, and settlement persist exactly."""
    clock = FakeLedgerClock()
    store, core, raw_key = _authority_fixture(tmp_path, clock)
    grouped = GroupCommitAttemptLedger(core)

    async def lifecycle() -> str:
        """Run one complete request lifecycle through the batching writer."""
        authorization = _authorize(store, clock, raw_key, "prompt-canary")
        await grouped.accept_request(authorization=authorization)
        attempt_id = await grouped.start_attempt(
            snapshot=_execution(authorization),
            deployment=_deployment(),
            attempt_ordinal=0,
            route_depth=0,
            route_reason="direct_alias",
            fallback_reason=None,
        )
        await grouped.finish_attempt(
            attempt_id=attempt_id,
            terminal_event=GatewayEvent(
                kind=GatewayEventKind.COMPLETED,
                sequence_number=1,
                usage=GatewayUsage(input_tokens=1_000, output_tokens=500),
            ),
            failure=None,
        )
        await grouped.flush()
        return attempt_id

    attempt_id = asyncio.run(lifecycle())
    grouped.close()
    connection = sqlite3.connect(tmp_path / "gateway.db")
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        "SELECT state, route_reason, input_tokens, output_tokens FROM gateway_attempts"
        " WHERE attempt_id = ?",
        (attempt_id,),
    ).fetchone()
    request = connection.execute("SELECT terminal_state FROM gateway_requests").fetchone()
    connection.close()
    assert row is not None
    assert str(row["state"]) == "completed"
    assert str(row["route_reason"]) == "direct_alias"
    assert int(row["input_tokens"]) == 1_000
    assert int(row["output_tokens"]) == 500
    assert str(request["terminal_state"]) == "completed"


def test_failed_operation_rolls_back_alone_and_batch_siblings_commit(tmp_path: Path) -> None:
    """One rejected write re-raises to its caller without harming batch siblings."""
    clock = FakeLedgerClock()
    store, core, raw_key = _authority_fixture(tmp_path, clock)
    grouped = GroupCommitAttemptLedger(core)

    async def mixed_batch() -> None:
        """Submit one valid acceptance alongside one write that must fail."""
        good = _authorize(store, clock, raw_key, "good-prompt")
        accepted = grouped.accept_request(authorization=good)
        rejected = grouped.finish_attempt(
            attempt_id="attempt-missing",
            terminal_event=None,
            failure=None,
        )
        results = await asyncio.gather(accepted, rejected, return_exceptions=True)
        assert results[0] is None
        assert isinstance(results[1], GatewayLedgerError)

    asyncio.run(mixed_batch())
    grouped.close()
    connection = sqlite3.connect(tmp_path / "gateway.db")
    count = connection.execute("SELECT COUNT(*) FROM gateway_requests").fetchone()[0]
    connection.close()
    assert int(count) == 1


def test_concurrent_writes_share_batches_and_all_become_durable(tmp_path: Path) -> None:
    """Many concurrent acceptances resolve only after each row is durable."""
    clock = FakeLedgerClock()
    store, core, raw_key = _authority_fixture(tmp_path, clock)
    grouped = GroupCommitAttemptLedger(core)

    async def accept_many() -> None:
        """Accept many independent requests concurrently through one writer."""
        authorizations = [
            _authorize(store, clock, raw_key, f"prompt-{index}") for index in range(64)
        ]
        await asyncio.gather(
            *(
                grouped.accept_request(authorization=authorization)
                for authorization in authorizations
            )
        )

    asyncio.run(accept_many())
    grouped.close()
    connection = sqlite3.connect(tmp_path / "gateway.db")
    count = connection.execute("SELECT COUNT(*) FROM gateway_requests").fetchone()[0]
    connection.close()
    assert int(count) == 64


def test_cancelled_caller_keeps_writer_running_and_write_durable(tmp_path: Path) -> None:
    """A cancelled awaiting task neither kills the writer nor loses its write."""
    clock = FakeLedgerClock()
    store, core, raw_key = _authority_fixture(tmp_path, clock)
    grouped = GroupCommitAttemptLedger(core)

    async def cancel_then_continue() -> None:
        """Cancel one submitting task mid-flight, then keep using the writer."""
        cancelled = _authorize(store, clock, raw_key, "cancelled-prompt")
        task = asyncio.ensure_future(grouped.accept_request(authorization=cancelled))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        survivor = _authorize(store, clock, raw_key, "survivor-prompt")
        await grouped.accept_request(authorization=survivor)
        await grouped.flush()

    asyncio.run(cancel_then_continue())
    grouped.close()
    connection = sqlite3.connect(tmp_path / "gateway.db")
    count = connection.execute("SELECT COUNT(*) FROM gateway_requests").fetchone()[0]
    connection.close()
    assert int(count) == 2


def test_closed_writer_rejects_new_operations(tmp_path: Path) -> None:
    """Submissions after close fail fast instead of queueing forever."""
    clock = FakeLedgerClock()
    _, core, _ = _authority_fixture(tmp_path, clock)
    grouped = GroupCommitAttemptLedger(core)
    grouped.close()

    async def submit() -> None:
        """Attempt one flush against the closed writer."""
        await grouped.flush()

    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(submit())


def test_max_batch_size_must_be_positive(tmp_path: Path) -> None:
    """A zero batch bound is rejected at construction."""
    clock = FakeLedgerClock()
    _, core, _ = _authority_fixture(tmp_path, clock)
    with pytest.raises(ValueError, match="at least one"):
        GroupCommitAttemptLedger(core, max_batch_size=0)


def test_abandoned_write_returns_committed_attempt_id() -> None:
    """A cancellation-abandoned write still yields its durable attempt ID."""

    async def scenario() -> None:
        """Cancel the waiter repeatedly while the write keeps running."""
        started = asyncio.Event()

        async def reserve() -> str:
            """Simulate a shielded durable write that outlives the caller."""
            started.set()
            await asyncio.sleep(0.01)
            return "attempt-durable"

        write = asyncio.ensure_future(reserve())
        await started.wait()

        async def waiter() -> str | None:
            """Recover the abandoned write outcome."""
            return await abandoned_write_outcome(write)

        waiting = asyncio.ensure_future(waiter())
        await asyncio.sleep(0)
        waiting.cancel()
        assert await waiting == "attempt-durable"

    asyncio.run(scenario())


def test_abandoned_write_returns_none_when_write_failed() -> None:
    """A write that raised committed nothing, so no attempt needs settling."""

    async def scenario() -> None:
        """Observe a failed write through the abandoned-outcome path."""

        async def reserve() -> str:
            """Simulate a rolled-back ledger write."""
            raise GatewayLedgerError("attempt write unavailable")

        write = asyncio.ensure_future(reserve())
        assert await abandoned_write_outcome(write) is None

    asyncio.run(scenario())


def test_abandoned_write_waits_out_pending_write() -> None:
    """The outcome helper blocks until the in-flight write actually resolves."""

    async def scenario() -> None:
        """Resolve the write only after the helper starts waiting."""
        gate: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        async def reserve() -> str:
            """Simulate a write pending on the group-commit batch."""
            return await gate

        write = asyncio.ensure_future(reserve())
        await asyncio.sleep(0)
        outcome = asyncio.ensure_future(abandoned_write_outcome(write))
        await asyncio.sleep(0)
        assert not outcome.done()
        gate.set_result("attempt-late")
        assert await outcome == "attempt-late"

    asyncio.run(scenario())


def test_abandoned_write_cancel_absorption_raises_nothing() -> None:
    """Cancelling the helper's waiter does not surface once the write resolves."""

    async def scenario() -> None:
        """Cancel the helper while it waits, then confirm the durable result."""
        gate: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        async def reserve() -> str:
            """Simulate a write pending on the group-commit batch."""
            return await gate

        write = asyncio.ensure_future(reserve())
        await asyncio.sleep(0)
        outcome = asyncio.ensure_future(abandoned_write_outcome(write))
        await asyncio.sleep(0)
        outcome.cancel()
        gate.set_result("attempt-after-cancel")
        assert await outcome == "attempt-after-cancel"

    asyncio.run(scenario())


def test_cancelled_write_task_yields_none() -> None:
    """A write task cancelled before running reports no durable attempt."""

    async def scenario() -> None:
        """Cancel the write itself and confirm a None outcome."""

        async def reserve() -> str:
            """Simulate a write that never starts."""
            await asyncio.sleep(60)
            return "attempt-unreachable"

        write = asyncio.ensure_future(reserve())
        write.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.gather(write)
        assert await abandoned_write_outcome(write) is None

    asyncio.run(scenario())
