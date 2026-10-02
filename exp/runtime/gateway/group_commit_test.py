"""Tests for batched durable group commits over the synchronous attempt ledger."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

import pytest

from exp.common.models.catalog import BillingSource, GatewayDeploymentMetadata, GatewayTokenPrices
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway import model_chain_authority as authority
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayMessage,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.gateway.group_commit import (
    GroupCommitAttemptLedger,
    SyncGroupCommitLedger,
    abandoned_write_outcome,
)
from exp.runtime.gateway.ledger import GatewayLedgerError, SQLiteAttemptLedger
from exp.runtime.gateway.model_chain_authority import (
    ChainOperation,
    ModelChainAuthorityError,
    SQLiteChainPreflight,
)
from exp.runtime.gateway.snapshot_file import PreparedSnapshotFile
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
                input_nano_usd_per_million_tokens=2_000_000,
                cached_input_nano_usd_per_million_tokens=1_000_000,
                output_nano_usd_per_million_tokens=4_000_000,
                reasoning_nano_usd_per_million_tokens=5_000_000,
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


@pytest.mark.parametrize("outcome", ["success", "refusal", "batch_failure", "cancelled"])
def test_group_preflight_handles_close_after_every_outcome(tmp_path: Path, outcome: str) -> None:
    """The writer owns all retained proof handles through cancellation, refusal and rollback."""
    clock = FakeLedgerClock()
    store, core, raw_key = _authority_fixture(tmp_path, clock)
    (tmp_path / "snapshot-one").write_text("{}")
    authorization = _authorize(store, clock, raw_key, "proof-lifetime")
    original = core.prepare_chain_authority
    proofs: list[SQLiteChainPreflight] = []
    entered, release = threading.Event(), threading.Event()

    @contextmanager
    def tracked(
        auth: AuthorizationSnapshot,
        operation: ChainOperation,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> Iterator[SQLiteChainPreflight | None]:
        """Pause only outside the transaction and retain each real production proof."""
        with original(auth, operation, connection=connection) as proof:
            assert proof is not None
            proofs.append(proof)
            entered.set()
            if outcome == "cancelled":
                assert release.wait(5)
            yield proof

    def fail_fence(prepared: PreparedSnapshotFile) -> None:
        """Refuse an actual apply fence after the real out-of-lock preparation completed."""
        if entered.is_set():
            raise ValueError("controlled generation change")

    def fail_batch(connection: sqlite3.Connection, batch: list[object]) -> None:
        """Simulate a writer failure after preflight but before a transaction starts."""
        raise sqlite3.OperationalError("controlled writer failure")

    grouped = GroupCommitAttemptLedger(core)
    try:
        with mock.patch.object(core, "prepare_chain_authority", tracked):
            if outcome == "batch_failure":
                with (
                    mock.patch.object(grouped, "_commit_batch", fail_batch),
                    pytest.raises(sqlite3.OperationalError),
                ):
                    SyncGroupCommitLedger(grouped).accept_request(authorization=authorization)
            elif outcome == "refusal":
                with (
                    mock.patch.object(PreparedSnapshotFile, "validate_current", fail_fence),
                    pytest.raises(ModelChainAuthorityError),
                ):
                    SyncGroupCommitLedger(grouped).accept_request(authorization=authorization)
            elif outcome == "cancelled":

                async def cancel_waiter() -> None:
                    """Cancel only the waiter, then let its queued operation finish durably."""
                    task = asyncio.create_task(grouped.accept_request(authorization=authorization))
                    assert await asyncio.to_thread(entered.wait, 5)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    release.set()
                    await grouped.flush()

                asyncio.run(cancel_waiter())
            else:
                SyncGroupCommitLedger(grouped).accept_request(authorization=authorization)
    finally:
        release.set()
        grouped.close()
    assert proofs and all(proof._closed for proof in proofs)
    assert all(
        file._closed and (file._stream is None or file._stream.closed)
        for proof in proofs
        for file in proof.files
    )
    with sqlite3.connect(core.database_path) as connection:
        count = connection.execute("SELECT count(*) FROM gateway_requests").fetchone()[0]
    assert count == (1 if outcome in ("success", "cancelled") else 0)
    with pytest.raises(RuntimeError, match="closed"):
        SyncGroupCommitLedger(grouped).accept_request(authorization=authorization)
    assert len(proofs) == 1


@pytest.mark.parametrize("batch_size", [1, 16])
def test_group_preflight_reuses_the_writer_database_connection(
    tmp_path: Path,
    batch_size: int,
) -> None:
    """Prepared batch size never allocates cached or concurrently held reader connections."""
    clock = FakeLedgerClock()
    store, core, raw_key = _authority_fixture(tmp_path, clock)
    authorizations = [
        _authorize(store, clock, raw_key, f"request-{index}") for index in range(batch_size)
    ]
    entered, release = threading.Event(), threading.Event()
    grouped = GroupCommitAttemptLedger(core, max_batch_size=batch_size)

    def pause(connection: sqlite3.Connection) -> None:
        """Hold a preceding writer operation while the next exact batch is queued."""
        entered.set()
        assert release.wait(5)

    blocker = grouped._enqueue(pause)
    assert entered.wait(5)
    try:
        with mock.patch.object(core, "_connect", wraps=core._connect) as reader_checkouts:
            writes = [
                grouped._enqueue_chain(
                    auth,
                    "accept",
                    lambda connection, proof, auth=auth: core.apply_accept_request(
                        connection, authorization=auth, chain_preflight=proof
                    ),
                )
                for auth in authorizations
            ]
            release.set()
            blocker.result(timeout=5)
            for write in writes:
                write.result(timeout=5)
            assert reader_checkouts.call_count == 0
    finally:
        release.set()
        grouped.close()


def test_group_writer_classifies_bytes_before_begin(tmp_path: Path) -> None:
    """Every JSON classification can obtain another write lock because BEGIN has not started."""
    clock = FakeLedgerClock()
    store, core, raw_key = _authority_fixture(tmp_path, clock)
    (tmp_path / "snapshot-one").write_text("{}")
    authorization = _authorize(store, clock, raw_key, "before-begin")
    original = authority.json.loads
    reads = 0

    def classify(content: bytes) -> dict[str, object]:
        """Probe actual writer-lock availability around each real catalog parse."""
        nonlocal reads
        with sqlite3.connect(core.database_path, timeout=0) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.rollback()
        reads += 1
        return original(content)

    grouped = GroupCommitAttemptLedger(core)
    try:
        with mock.patch.object(authority.json, "loads", classify):
            sync = SyncGroupCommitLedger(grouped)
            sync.accept_request(authorization=authorization)
            sync.start_attempt(
                snapshot=_execution(authorization),
                deployment=_deployment(),
                attempt_ordinal=0,
                route_depth=0,
            )
    finally:
        grouped.close()
    assert reads == 1  # Reservation reuses only the already classified, still-current plain pair.


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


def test_sync_facade_commits_full_lifecycle_durably_without_an_event_loop(
    tmp_path: Path,
) -> None:
    """The blocking facade lands acceptance, dispatch, and settlement exactly."""
    clock = FakeLedgerClock()
    store, core, raw_key = _authority_fixture(tmp_path, clock)
    grouped = GroupCommitAttemptLedger(core)
    facade = SyncGroupCommitLedger(grouped)
    authorization = _authorize(store, clock, raw_key, "sync-prompt")
    facade.accept_request(authorization=authorization)
    attempt_id = facade.start_attempt(
        snapshot=_execution(authorization),
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
        route_reason="direct_alias",
        fallback_reason=None,
    )
    facade.finish_attempt(
        attempt_id=attempt_id,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=1,
            usage=GatewayUsage(input_tokens=100, output_tokens=40),
        ),
        failure=None,
    )
    facade.flush()
    grouped.close()
    connection = sqlite3.connect(tmp_path / "gateway.db")
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        "SELECT state, input_tokens, output_tokens FROM gateway_attempts WHERE attempt_id = ?",
        (attempt_id,),
    ).fetchone()
    request = connection.execute("SELECT terminal_state FROM gateway_requests").fetchone()
    connection.close()
    assert row is not None
    assert str(row["state"]) == "completed"
    assert int(row["input_tokens"]) == 100
    assert int(row["output_tokens"]) == 40
    assert str(request["terminal_state"]) == "completed"


def test_sync_facade_raises_the_original_failure_and_stays_usable(tmp_path: Path) -> None:
    """A rolled-back sync operation re-raises to its caller; the writer keeps
    serving later operations."""
    clock = FakeLedgerClock()
    store, core, raw_key = _authority_fixture(tmp_path, clock)
    grouped = GroupCommitAttemptLedger(core)
    facade = SyncGroupCommitLedger(grouped)
    with pytest.raises(GatewayLedgerError):
        facade.finish_attempt(attempt_id="attempt-missing", terminal_event=None, failure=None)
    survivor = _authorize(store, clock, raw_key, "survivor-prompt")
    facade.accept_request(authorization=survivor)
    grouped.close()
    connection = sqlite3.connect(tmp_path / "gateway.db")
    count = connection.execute("SELECT COUNT(*) FROM gateway_requests").fetchone()[0]
    connection.close()
    assert int(count) == 1


def test_concurrent_sync_threads_all_commit_exactly_once(tmp_path: Path) -> None:
    """Blocking callers on many threads each observe exactly their own durable row."""
    clock = FakeLedgerClock()
    store, core, raw_key = _authority_fixture(tmp_path, clock)
    grouped = GroupCommitAttemptLedger(core)
    facade = SyncGroupCommitLedger(grouped)
    authorizations = [_authorize(store, clock, raw_key, f"thread-{index}") for index in range(16)]
    barrier = threading.Barrier(len(authorizations))
    errors: list[BaseException] = []

    def accept(authorization: AuthorizationSnapshot) -> None:
        """Block one thread on its own durable acceptance.

        Args:
            authorization: This thread's frozen authority snapshot.
        """
        try:
            barrier.wait(timeout=10)
            facade.accept_request(authorization=authorization)
        except BaseException as exc:  # noqa: BLE001 - the test asserts no error.
            errors.append(exc)

    threads = [
        threading.Thread(target=accept, args=(authorization,)) for authorization in authorizations
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
    grouped.close()
    assert errors == []
    connection = sqlite3.connect(tmp_path / "gateway.db")
    request_ids = [
        str(row[0])
        for row in connection.execute("SELECT request_id FROM gateway_requests").fetchall()
    ]
    connection.close()
    assert sorted(request_ids) == sorted(item.request_id for item in authorizations)


def test_closed_writer_rejects_sync_operations(tmp_path: Path) -> None:
    """Sync submissions after close fail fast with a clear closed-writer error."""
    clock = FakeLedgerClock()
    _, core, _ = _authority_fixture(tmp_path, clock)
    grouped = GroupCommitAttemptLedger(core)
    facade = SyncGroupCommitLedger(grouped)
    grouped.close()
    with pytest.raises(RuntimeError, match="closed"):
        facade.flush()


def test_batch_machinery_failure_fails_blocked_callers_and_writer_recovers(
    tmp_path: Path,
) -> None:
    """A batch that fails outside its own transaction still fails its callers
    instead of stranding them, and the writer keeps serving afterwards."""
    clock = FakeLedgerClock()
    store, core, raw_key = _authority_fixture(tmp_path, clock)
    grouped = GroupCommitAttemptLedger(core)
    facade = SyncGroupCommitLedger(grouped)
    with mock.patch.object(
        grouped,
        "_commit_batch",
        side_effect=RuntimeError("simulated savepoint machinery loss"),
    ):
        with pytest.raises(RuntimeError, match="savepoint machinery"):
            facade.flush()
    survivor = _authorize(store, clock, raw_key, "post-failure-prompt")
    facade.accept_request(authorization=survivor)
    grouped.close()
    connection = sqlite3.connect(tmp_path / "gateway.db")
    count = connection.execute("SELECT COUNT(*) FROM gateway_requests").fetchone()[0]
    connection.close()
    assert int(count) == 1


def test_writer_startup_failure_fails_sync_callers_instead_of_hanging(
    tmp_path: Path,
) -> None:
    """A writer thread that dies before serving rejects callers rather than
    stranding them on a queue nothing will ever drain."""
    clock = FakeLedgerClock()
    _, core, _ = _authority_fixture(tmp_path, clock)
    with mock.patch(
        "exp.runtime.gateway.group_commit.connect_database",
        side_effect=sqlite3.OperationalError("simulated writer connection loss"),
    ):
        grouped = GroupCommitAttemptLedger(core)
        grouped._thread.join(timeout=10)  # noqa: SLF001 - deterministic crash ordering.
    facade = SyncGroupCommitLedger(grouped)
    with pytest.raises(RuntimeError, match="closed"):
        facade.flush()


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


@pytest.mark.parametrize("writes", [None, 0, 3])
def test_facades_forward_upstream_provider_only_to_a_host_hook_that_accepts_it(
    tmp_path: Path,
    writes: int | None,
) -> None:
    """The hosted-ledger seam probes the host's apply hook, not the engine facade.

    A host whose ``apply_finish_attempt`` predates ``upstream_provider`` (the
    platform hook at the 0.7.88 repin) must settle cleanly with the keyword
    withheld; the engine's own core still persists the named upstream.
    """
    clock = FakeLedgerClock()
    store, core, raw_key = _authority_fixture(tmp_path, clock)
    recorded: list[dict[str, object]] = []
    original = core.apply_finish_attempt
    observed = datetime(2026, 9, 18, 1, 2, 3, tzinfo=UTC)
    usage = GatewayUsage(input_tokens=10, output_tokens=4, cache_creation_input_tokens=writes)

    def legacy_apply(
        connection: sqlite3.Connection,
        *,
        attempt_id: str,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
        first_token_at: datetime | None = None,
        retry_after_seconds: int | None = None,
        ratelimit_limit_requests: int | None = None,
        ratelimit_remaining_requests: int | None = None,
        ratelimit_limit_tokens: int | None = None,
        ratelimit_remaining_tokens: int | None = None,
    ) -> None:
        """The pre-keyword host hook shape: any extra keyword would TypeError here."""
        assert terminal_event is not None and terminal_event.usage == usage
        assert first_token_at == observed
        recorded.append({"attempt_id": attempt_id, "finalize": finalize_request})
        original(
            connection,
            attempt_id=attempt_id,
            terminal_event=terminal_event,
            failure=failure,
            finalize_request=finalize_request,
            first_token_at=first_token_at,
            retry_after_seconds=retry_after_seconds,
            ratelimit_limit_requests=ratelimit_limit_requests,
            ratelimit_remaining_requests=ratelimit_remaining_requests,
            ratelimit_limit_tokens=ratelimit_limit_tokens,
            ratelimit_remaining_tokens=ratelimit_remaining_tokens,
        )

    with mock.patch.object(core, "apply_finish_attempt", legacy_apply):
        grouped = GroupCommitAttemptLedger(core)
        facade = SyncGroupCommitLedger(grouped)
        authorization = _authorize(store, clock, raw_key, "legacy-host-hook")
        facade.accept_request(authorization=authorization)
        attempt_id = facade.start_attempt(
            snapshot=_execution(authorization),
            deployment=_deployment(),
            attempt_ordinal=0,
            route_depth=0,
            route_reason="direct_alias",
            fallback_reason=None,
        )
        facade.finish_attempt(
            attempt_id=attempt_id,
            terminal_event=GatewayEvent(
                kind=GatewayEventKind.COMPLETED,
                sequence_number=1,
                usage=usage,
            ),
            failure=None,
            first_token_at=observed,
            upstream_provider="Azure",
        )
        facade.flush()
        grouped.close()
    assert recorded == [{"attempt_id": attempt_id, "finalize": True}]
    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        assert connection.execute(
            "SELECT state, upstream_provider FROM gateway_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone() == ("completed", None)
    finally:
        connection.close()

    # The engine's own core accepts the keyword, so the label lands.
    grouped = GroupCommitAttemptLedger(core)
    facade = SyncGroupCommitLedger(grouped)
    authorization = _authorize(store, clock, raw_key, "current-host-hook")
    facade.accept_request(authorization=authorization)
    attempt_id = facade.start_attempt(
        snapshot=_execution(authorization),
        deployment=_deployment(),
        attempt_ordinal=0,
        route_depth=0,
        route_reason="direct_alias",
        fallback_reason=None,
    )
    facade.finish_attempt(
        attempt_id=attempt_id,
        terminal_event=GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=1,
            usage=usage,
        ),
        failure=None,
        first_token_at=observed,
        upstream_provider="Azure",
    )
    facade.flush()
    grouped.close()
    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        assert connection.execute(
            "SELECT upstream_provider FROM gateway_attempts WHERE attempt_id = ?", (attempt_id,)
        ).fetchone() == ("Azure",)
    finally:
        connection.close()


@pytest.mark.parametrize("writes", [None, 0, 3])
def test_async_facade_withholds_upstream_provider_from_a_legacy_host_hook(
    tmp_path: Path, writes: int | None
) -> None:
    """``GroupCommitAttemptLedger.finish_attempt`` probes the host hook the same way.

    The async facade captures the hook, probes it and queues its own lambda
    independently of the blocking facade, so it gets its own legacy-hook case:
    an old-signature ``apply_finish_attempt`` settles a full async lifecycle
    with ``upstream_provider`` withheld, the row completed and the column NULL.
    """
    clock = FakeLedgerClock()
    store, core, raw_key = _authority_fixture(tmp_path, clock)
    recorded: list[str] = []
    original = core.apply_finish_attempt
    usage = GatewayUsage(input_tokens=10, output_tokens=4, cache_creation_input_tokens=writes)
    observed = datetime(2026, 9, 18, 1, 2, 3, tzinfo=UTC)

    def legacy_apply(
        connection: sqlite3.Connection,
        *,
        attempt_id: str,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
        first_token_at: datetime | None = None,
        retry_after_seconds: int | None = None,
        ratelimit_limit_requests: int | None = None,
        ratelimit_remaining_requests: int | None = None,
        ratelimit_limit_tokens: int | None = None,
        ratelimit_remaining_tokens: int | None = None,
    ) -> None:
        """The pre-keyword host hook shape: any extra keyword would TypeError here."""
        assert terminal_event is not None and terminal_event.usage == usage
        assert first_token_at == observed
        recorded.append(attempt_id)
        original(
            connection,
            attempt_id=attempt_id,
            terminal_event=terminal_event,
            failure=failure,
            finalize_request=finalize_request,
            first_token_at=first_token_at,
            retry_after_seconds=retry_after_seconds,
            ratelimit_limit_requests=ratelimit_limit_requests,
            ratelimit_remaining_requests=ratelimit_remaining_requests,
            ratelimit_limit_tokens=ratelimit_limit_tokens,
            ratelimit_remaining_tokens=ratelimit_remaining_tokens,
        )

    with mock.patch.object(core, "apply_finish_attempt", legacy_apply):
        grouped = GroupCommitAttemptLedger(core)

        async def lifecycle() -> str:
            """Accept, dispatch and settle one request through the async facade."""
            authorization = _authorize(store, clock, raw_key, "async-legacy-host-hook")
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
                    usage=usage,
                ),
                failure=None,
                first_token_at=observed,
                upstream_provider="Azure",
            )
            await grouped.flush()
            return attempt_id

        attempt_id = asyncio.run(lifecycle())
        grouped.close()
    assert recorded == [attempt_id]
    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        assert connection.execute(
            "SELECT state, upstream_provider FROM gateway_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone() == ("completed", None)
    finally:
        connection.close()


def _pre_search_meter_hook(
    original: Callable[..., None], recorded: list[dict[str, object]]
) -> Callable[..., None]:
    """The host hook shape that learned ``upstream_provider`` but not ``web_search_requests``.

    Any further keyword would TypeError here, which is the drift under test.
    """

    def legacy_apply(
        connection: sqlite3.Connection,
        *,
        attempt_id: str,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
        first_token_at: datetime | None = None,
        retry_after_seconds: int | None = None,
        ratelimit_limit_requests: int | None = None,
        ratelimit_remaining_requests: int | None = None,
        ratelimit_limit_tokens: int | None = None,
        ratelimit_remaining_tokens: int | None = None,
        upstream_provider: str | None = None,
    ) -> None:
        """Record the settle and forward it to the engine's own core."""
        recorded.append({"attempt_id": attempt_id, "upstream_provider": upstream_provider})
        original(
            connection,
            attempt_id=attempt_id,
            terminal_event=terminal_event,
            failure=failure,
            finalize_request=finalize_request,
            first_token_at=first_token_at,
            retry_after_seconds=retry_after_seconds,
            ratelimit_limit_requests=ratelimit_limit_requests,
            ratelimit_remaining_requests=ratelimit_remaining_requests,
            ratelimit_limit_tokens=ratelimit_limit_tokens,
            ratelimit_remaining_tokens=ratelimit_remaining_tokens,
            upstream_provider=upstream_provider,
        )

    return legacy_apply


def _search_meter_hook(
    original: Callable[..., None], recorded: list[dict[str, object]]
) -> Callable[..., None]:
    """A host hook that accepts every settle keyword and records which ones arrived."""

    def current_apply(connection: sqlite3.Connection, **kwargs: object) -> None:
        """Record the keywords handed over, then forward them all."""
        recorded.append(dict(kwargs))
        original(connection, **kwargs)

    return current_apply


def _searching_terminal() -> GatewayEvent:
    """One completed terminal whose usage bills two gateway-executed searches."""
    return GatewayEvent(
        kind=GatewayEventKind.COMPLETED,
        sequence_number=1,
        usage=GatewayUsage(input_tokens=10, output_tokens=4, web_search_requests=2),
    )


def _pre_tool_search_meter_hook(
    original: Callable[..., None], recorded: list[dict[str, object]]
) -> Callable[..., None]:
    """The host hook shape that learned ``web_search_requests`` but not ``tool_search_requests``.

    Any further keyword would TypeError here, which is the drift under test.
    """

    def legacy_apply(
        connection: sqlite3.Connection,
        *,
        attempt_id: str,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
        first_token_at: datetime | None = None,
        retry_after_seconds: int | None = None,
        ratelimit_limit_requests: int | None = None,
        ratelimit_remaining_requests: int | None = None,
        ratelimit_limit_tokens: int | None = None,
        ratelimit_remaining_tokens: int | None = None,
        upstream_provider: str | None = None,
        web_search_requests: int = 0,
    ) -> None:
        """Record the settle and forward it to the engine's own core."""
        recorded.append(
            {
                "attempt_id": attempt_id,
                "upstream_provider": upstream_provider,
                "web_search_requests": web_search_requests,
            }
        )
        original(
            connection,
            attempt_id=attempt_id,
            terminal_event=terminal_event,
            failure=failure,
            finalize_request=finalize_request,
            first_token_at=first_token_at,
            retry_after_seconds=retry_after_seconds,
            ratelimit_limit_requests=ratelimit_limit_requests,
            ratelimit_remaining_requests=ratelimit_remaining_requests,
            ratelimit_limit_tokens=ratelimit_limit_tokens,
            ratelimit_remaining_tokens=ratelimit_remaining_tokens,
            upstream_provider=upstream_provider,
            web_search_requests=web_search_requests,
        )

    return legacy_apply


def _tool_searching_terminal() -> GatewayEvent:
    """One completed terminal whose usage bills one web search and two tool-search rounds."""
    return GatewayEvent(
        kind=GatewayEventKind.COMPLETED,
        sequence_number=1,
        usage=GatewayUsage(
            input_tokens=10, output_tokens=4, web_search_requests=1, tool_search_requests=2
        ),
    )


def test_sync_facade_forwards_web_search_requests_only_to_a_host_hook_that_accepts_it(
    tmp_path: Path,
) -> None:
    """Mirror of the ``upstream_provider`` seam for the search meter, on the blocking facade.

    A host hook predating ``web_search_requests`` settles cleanly with it
    withheld while still receiving the upstream label; a current hook gets the
    settled count; a zero count is withheld from the current hook too.
    """
    clock = FakeLedgerClock()
    store, core, raw_key = _authority_fixture(tmp_path, clock)
    original = core.apply_finish_attempt
    recorded: list[dict[str, object]] = []

    def lifecycle(label: str, terminal: GatewayEvent) -> str:
        """Accept, dispatch and settle one request through the blocking facade."""
        grouped = GroupCommitAttemptLedger(core)
        facade = SyncGroupCommitLedger(grouped)
        authorization = _authorize(store, clock, raw_key, label)
        facade.accept_request(authorization=authorization)
        attempt_id = facade.start_attempt(
            snapshot=_execution(authorization),
            deployment=_deployment(),
            attempt_ordinal=0,
            route_depth=0,
            route_reason="direct_alias",
            fallback_reason=None,
        )
        facade.finish_attempt(
            attempt_id=attempt_id,
            terminal_event=terminal,
            failure=None,
            upstream_provider="Azure",
            web_search_requests=terminal.usage.web_search_requests if terminal.usage else 0,
        )
        facade.flush()
        grouped.close()
        return attempt_id

    with mock.patch.object(
        core, "apply_finish_attempt", _pre_search_meter_hook(original, recorded)
    ):
        legacy_attempt = lifecycle("pre-search-meter-hook", _searching_terminal())
    assert recorded == [{"attempt_id": legacy_attempt, "upstream_provider": "Azure"}]

    recorded.clear()
    with mock.patch.object(core, "apply_finish_attempt", _search_meter_hook(original, recorded)):
        billed_attempt = lifecycle("search-meter-hook", _searching_terminal())
        unbilled_attempt = lifecycle(
            "no-search-hook",
            GatewayEvent(
                kind=GatewayEventKind.COMPLETED,
                sequence_number=1,
                usage=GatewayUsage(input_tokens=10, output_tokens=4),
            ),
        )
    assert [entry["attempt_id"] for entry in recorded] == [billed_attempt, unbilled_attempt]
    assert recorded[0]["web_search_requests"] == 2
    assert "web_search_requests" not in recorded[1]

    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        rows = connection.execute(
            "SELECT attempt_id, state, upstream_provider FROM gateway_attempts ORDER BY attempt_id"
        ).fetchall()
    finally:
        connection.close()
    assert sorted(rows) == sorted(
        [
            (attempt, "completed", "Azure")
            for attempt in (legacy_attempt, billed_attempt, unbilled_attempt)
        ]
    )


def test_async_facade_withholds_web_search_requests_from_a_pre_meter_host_hook(
    tmp_path: Path,
) -> None:
    """``GroupCommitAttemptLedger.finish_attempt`` probes the hook for the meter the same way."""
    clock = FakeLedgerClock()
    store, core, raw_key = _authority_fixture(tmp_path, clock)
    original = core.apply_finish_attempt
    recorded: list[dict[str, object]] = []

    with mock.patch.object(
        core, "apply_finish_attempt", _pre_search_meter_hook(original, recorded)
    ):
        grouped = GroupCommitAttemptLedger(core)

        async def lifecycle() -> str:
            """Accept, dispatch and settle one searching request through the async facade."""
            authorization = _authorize(store, clock, raw_key, "async-pre-search-meter-hook")
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
                terminal_event=_searching_terminal(),
                failure=None,
                upstream_provider="Azure",
                web_search_requests=2,
            )
            await grouped.flush()
            return attempt_id

        attempt_id = asyncio.run(lifecycle())
        grouped.close()
    assert recorded == [{"attempt_id": attempt_id, "upstream_provider": "Azure"}]
    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        assert connection.execute(
            "SELECT state, upstream_provider FROM gateway_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone() == ("completed", "Azure")
    finally:
        connection.close()


def test_sync_facade_forwards_tool_search_requests_only_to_a_host_hook_that_accepts_it(
    tmp_path: Path,
) -> None:
    """Mirror of the ``web_search_requests`` seam for the tool-search meter, on the blocking facade.

    A host hook predating ``tool_search_requests`` settles cleanly with it
    withheld while still receiving the upstream label and the web-search count;
    a current hook gets the settled count; a zero count is withheld from the
    current hook too.
    """
    clock = FakeLedgerClock()
    store, core, raw_key = _authority_fixture(tmp_path, clock)
    original = core.apply_finish_attempt
    recorded: list[dict[str, object]] = []

    def lifecycle(label: str, terminal: GatewayEvent) -> str:
        """Accept, dispatch and settle one request through the blocking facade."""
        grouped = GroupCommitAttemptLedger(core)
        facade = SyncGroupCommitLedger(grouped)
        authorization = _authorize(store, clock, raw_key, label)
        facade.accept_request(authorization=authorization)
        attempt_id = facade.start_attempt(
            snapshot=_execution(authorization),
            deployment=_deployment(),
            attempt_ordinal=0,
            route_depth=0,
            route_reason="direct_alias",
            fallback_reason=None,
        )
        usage = terminal.usage
        facade.finish_attempt(
            attempt_id=attempt_id,
            terminal_event=terminal,
            failure=None,
            upstream_provider="Azure",
            web_search_requests=usage.web_search_requests if usage else 0,
            tool_search_requests=usage.tool_search_requests if usage else 0,
        )
        facade.flush()
        grouped.close()
        return attempt_id

    with mock.patch.object(
        core, "apply_finish_attempt", _pre_tool_search_meter_hook(original, recorded)
    ):
        legacy_attempt = lifecycle("pre-tool-search-meter-hook", _tool_searching_terminal())
    assert recorded == [
        {"attempt_id": legacy_attempt, "upstream_provider": "Azure", "web_search_requests": 1}
    ]

    recorded.clear()
    with mock.patch.object(core, "apply_finish_attempt", _search_meter_hook(original, recorded)):
        billed_attempt = lifecycle("tool-search-meter-hook", _tool_searching_terminal())
        unbilled_attempt = lifecycle("no-tool-search-hook", _searching_terminal())
    assert [entry["attempt_id"] for entry in recorded] == [billed_attempt, unbilled_attempt]
    assert recorded[0]["tool_search_requests"] == 2
    assert recorded[0]["web_search_requests"] == 1
    assert "tool_search_requests" not in recorded[1]
    assert recorded[1]["web_search_requests"] == 2

    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        rows = connection.execute(
            "SELECT attempt_id, state, upstream_provider FROM gateway_attempts ORDER BY attempt_id"
        ).fetchall()
    finally:
        connection.close()
    assert sorted(rows) == sorted(
        [
            (attempt, "completed", "Azure")
            for attempt in (legacy_attempt, billed_attempt, unbilled_attempt)
        ]
    )


def test_async_facade_withholds_tool_search_requests_from_a_pre_meter_host_hook(
    tmp_path: Path,
) -> None:
    """``GroupCommitAttemptLedger.finish_attempt`` probes the hook for the tool-search meter too."""
    clock = FakeLedgerClock()
    store, core, raw_key = _authority_fixture(tmp_path, clock)
    original = core.apply_finish_attempt
    recorded: list[dict[str, object]] = []

    with mock.patch.object(
        core, "apply_finish_attempt", _pre_tool_search_meter_hook(original, recorded)
    ):
        grouped = GroupCommitAttemptLedger(core)

        async def lifecycle() -> str:
            """Accept, dispatch and settle one tool-searching request through the async facade."""
            authorization = _authorize(store, clock, raw_key, "async-pre-tool-search-meter-hook")
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
                terminal_event=_tool_searching_terminal(),
                failure=None,
                upstream_provider="Azure",
                web_search_requests=1,
                tool_search_requests=2,
            )
            await grouped.flush()
            return attempt_id

        attempt_id = asyncio.run(lifecycle())
        grouped.close()
    assert recorded == [
        {"attempt_id": attempt_id, "upstream_provider": "Azure", "web_search_requests": 1}
    ]
    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        assert connection.execute(
            "SELECT state, upstream_provider FROM gateway_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone() == ("completed", "Azure")
    finally:
        connection.close()
