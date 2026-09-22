"""Adversarial durability tests for text simulation paid-cell claims."""

import logging
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import canonical_json_bytes
from exp.common.core.locks import FileLockTimeout, file_write_lock
from exp.common.project import ArtifactStore, ProjectPaths
from exp.simulation.engines.text import leases
from exp.simulation.engines.text.leases import (
    TextCellLeaseClaim,
    TextCellLeaseError,
    TextCellLeaseState,
    TextCellLeaseStatus,
    TextCellLeaseStore,
)

_TIME = datetime(2026, 8, 12, tzinfo=UTC)
_DIGEST = "a" * 64


def test_completed_rollout_release_timeout_defers_to_safe_reaping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A cleanup lock timeout cannot discard a completed rollout or authorize replay."""
    store = TextCellLeaseStore(tmp_path / "projects" / "project-a", clock=lambda: _TIME)

    def acquire(*, completed: bool) -> TextCellLeaseClaim:
        """Admit or reap the same exact cell without changing its durable identity."""
        return store.acquire(
            lease_id="lease-a",
            resolution_id="resolution-a",
            simulation_id="simulation-a",
            rollout_id="rollout-a",
            binding_sha256=_DIGEST,
            maximum_cost_usd=1.0,
            observed_spend_usd=lambda: 0.0,
            rollout_completed=lambda _: completed,
        )

    claim = acquire(completed=False)
    assert claim.lease is not None

    @contextmanager
    def busy_lock(path: Path, *, what: str, timeout_s: float = 10.0) -> Iterator[None]:
        """Hold cleanup behind another metadata writer after evidence is durable."""
        raise FileLockTimeout("metadata writer is busy")
        yield

    with monkeypatch.context() as patch:
        patch.setattr(leases, "file_write_lock", busy_lock)
        with caplog.at_level(logging.WARNING):
            store.release(claim.lease)
    assert "after immutable rollout persistence" in caplog.text
    assert store._records.read("lease-a") is not None
    completed = acquire(completed=True)
    assert completed.state == TextCellLeaseState.COMPLETED
    assert store._records.read("lease-a") is None


@pytest.mark.parametrize("operation", ["release", "dispatch_intent"])
def test_local_workers_queue_before_the_cross_process_admission_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """Slow spend reconciliation cannot make local cleanup poll the cross-process lock."""
    store = TextCellLeaseStore(tmp_path / "projects" / "project-a", clock=lambda: _TIME)

    def acquire(suffix: str, spend: Callable[[], float]) -> TextCellLeaseClaim:
        """Reserve one independent bounded cell under the shared ledger."""
        return store.acquire(
            lease_id=f"lease-{suffix}",
            resolution_id="resolution-a",
            simulation_id="simulation-a",
            rollout_id=f"rollout-{suffix}",
            binding_sha256=_DIGEST,
            maximum_cost_usd=1.0,
            reservation_cost_usd=0.2,
            stop_on_overspend=True,
            rollout_completed=lambda _: False,
            observed_spend_usd=spend,
        )

    first = acquire("a", lambda: 0.0)
    assert first.lease is not None
    reconciling = threading.Event()
    finish_reconciliation = threading.Event()
    release_started = threading.Event()
    concurrent_file_lock = threading.Event()
    active = threading.Lock()

    @contextmanager
    def observed_lock(path: Path, *, what: str, timeout_s: float = 10.0) -> Iterator[None]:
        """Detect local contenders before taking the real durable lock."""
        if not active.acquire(blocking=False):
            concurrent_file_lock.set()
            raise AssertionError("local metadata operations must queue before the file lock")
        try:
            with file_write_lock(path, what=what, timeout_s=timeout_s):
                yield
        finally:
            active.release()

    def slow_spend() -> float:
        """Hold reconciliation until cleanup has begun waiting."""
        reconciling.set()
        assert finish_reconciliation.wait(timeout=5)
        return 0.0

    def update_owned_lease() -> None:
        """Update an owned claim while another local worker reconciles spend."""
        release_started.set()
        assert first.lease is not None
        if operation == "release":
            store.release(first.lease)
        else:
            assert store.record_dispatch_intent(first.lease).dispatch_intent_recorded

    monkeypatch.setattr(leases, "file_write_lock", observed_lock)
    with ThreadPoolExecutor(max_workers=2) as pool:
        admission = pool.submit(acquire, "b", slow_spend)
        try:
            assert reconciling.wait(timeout=5)
            cleanup = pool.submit(update_owned_lease)
            assert release_started.wait(timeout=5)
            assert not concurrent_file_lock.wait(timeout=0.1)
        finally:
            finish_reconciliation.set()
        assert admission.result(timeout=5).state == TextCellLeaseState.OWNED
        cleanup.result(timeout=5)
    assert (store._records.read("lease-a") is not None) == (operation == "dispatch_intent")


def test_dispatch_intent_blocks_replay_until_rollout_is_durable(
    tmp_path: Path,
) -> None:
    """A durable dispatch intent keeps the live claim until exact rollout evidence exists."""
    project = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    store = TextCellLeaseStore(project.project_directory, clock=lambda: _TIME)
    first = store.acquire(
        lease_id="lease-a",
        resolution_id="resolution-a",
        simulation_id="simulation-a",
        rollout_id="rollout-a",
        binding_sha256=_DIGEST,
        maximum_cost_usd=1.0,
        rollout_completed=lambda _rollout_id: False,
        observed_spend_usd=lambda: 0.0,
    )
    assert first.lease is not None

    intended = store.record_dispatch_intent(first.lease)
    assert intended.dispatch_intent_recorded
    assert intended.status == TextCellLeaseStatus.ACTIVE

    elapsed = [0.0]
    contender = TextCellLeaseStore(
        project.project_directory,
        clock=lambda: _TIME,
        sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
        monotonic=lambda: elapsed[0],
        wait_timeout_seconds=0.05,
    )
    blocked = contender.acquire(
        lease_id="lease-a",
        resolution_id="resolution-a",
        simulation_id="simulation-a",
        rollout_id="rollout-a",
        binding_sha256=_DIGEST,
        maximum_cost_usd=1.0,
        rollout_completed=lambda _rollout_id: False,
        observed_spend_usd=lambda: 0.0,
    )
    completed = store.acquire(
        lease_id="lease-a",
        resolution_id="resolution-a",
        simulation_id="simulation-a",
        rollout_id="rollout-a",
        binding_sha256=_DIGEST,
        maximum_cost_usd=1.0,
        rollout_completed=lambda rollout_id: rollout_id == "rollout-a",
        observed_spend_usd=lambda: 0.0,
    )

    assert blocked.state == TextCellLeaseState.CONTENDED
    assert blocked.retryable
    assert completed.state == TextCellLeaseState.COMPLETED
    assert store._records.list_ids() == ()


def test_expired_dead_paid_claim_is_recovered_as_stale_without_replay(tmp_path: Path) -> None:
    """A crash after claim creation is never silently replayed as a second paid provider call."""
    project = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    original = TextCellLeaseStore(project.project_directory, clock=lambda: _TIME)
    first = original.acquire(
        lease_id="lease-a",
        resolution_id="resolution-a",
        simulation_id="simulation-a",
        rollout_id="rollout-a",
        binding_sha256=_DIGEST,
        maximum_cost_usd=1.0,
        rollout_completed=lambda _rollout_id: False,
        observed_spend_usd=lambda: 0.0,
    )

    recovery_store = TextCellLeaseStore(
        project.project_directory,
        clock=lambda: _TIME + timedelta(minutes=16),
        owner_alive=lambda _pid: False,
    )
    recovered = recovery_store.acquire(
        lease_id="lease-a",
        resolution_id="resolution-a",
        simulation_id="simulation-a",
        rollout_id="rollout-a",
        binding_sha256=_DIGEST,
        maximum_cost_usd=1.0,
        rollout_completed=lambda _rollout_id: False,
        observed_spend_usd=lambda: 0.0,
    )
    elapsed = [0.0]
    blocked_store = TextCellLeaseStore(
        project.project_directory,
        clock=lambda: _TIME + timedelta(minutes=16),
        owner_alive=lambda _pid: False,
        sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
        monotonic=lambda: elapsed[0],
        wait_timeout_seconds=0.05,
    )
    allowed = blocked_store.acquire(
        lease_id="lease-b",
        resolution_id="resolution-a",
        simulation_id="simulation-a",
        rollout_id="rollout-b",
        binding_sha256="b" * 64,
        maximum_cost_usd=1.0,
        rollout_completed=lambda _rollout_id: False,
        observed_spend_usd=lambda: 0.1,
    )

    assert first.state == TextCellLeaseState.OWNED
    assert recovered.state == TextCellLeaseState.STALE
    assert recovered.lease is not None
    assert recovered.lease.status == TextCellLeaseStatus.STALE
    assert recovered.lease.unknown_spend_blocks_budget
    assert recovered.lease.reserved_cost_usd == 1.0
    assert allowed.state == TextCellLeaseState.CONTENDED
    assert allowed.retryable
    assert blocked_store.stale_recovery_pending("lease-a")
    assert not blocked_store.stale_recovery_pending("lease-b")


def test_live_paid_claim_returns_retryable_contention_at_finite_deadline(
    tmp_path: Path,
) -> None:
    """A live hung owner cannot wait forever or turn contention into permanent evidence."""
    project = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    original = TextCellLeaseStore(project.project_directory, clock=lambda: _TIME)
    first = original.acquire(
        lease_id="lease-a",
        resolution_id="resolution-a",
        simulation_id="simulation-a",
        rollout_id="rollout-a",
        binding_sha256=_DIGEST,
        maximum_cost_usd=1.0,
        rollout_completed=lambda _rollout_id: False,
        observed_spend_usd=lambda: 0.0,
    )
    elapsed = [0.0]
    contender = TextCellLeaseStore(
        project.project_directory,
        clock=lambda: _TIME,
        sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
        monotonic=lambda: elapsed[0],
        poll_interval_seconds=0.02,
        wait_timeout_seconds=0.05,
    )

    blocked = contender.acquire(
        lease_id="lease-a",
        resolution_id="resolution-a",
        simulation_id="simulation-a",
        rollout_id="rollout-a",
        binding_sha256=_DIGEST,
        maximum_cost_usd=1.0,
        rollout_completed=lambda _rollout_id: False,
        observed_spend_usd=lambda: 0.0,
    )

    assert first.state == TextCellLeaseState.OWNED
    assert blocked.state == TextCellLeaseState.CONTENDED
    assert blocked.retryable
    assert elapsed[0] == pytest.approx(0.05)
    assert len(contender._records.list_ids()) == 1


def test_cancelled_paid_claim_wait_returns_retryable_contention_without_a_lease(
    tmp_path: Path,
) -> None:
    """Cooperative cancellation stops admission before a provider-owning lease is created."""
    project = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    store = TextCellLeaseStore(project.project_directory, clock=lambda: _TIME)

    cancelled = store.acquire(
        lease_id="lease-a",
        resolution_id="resolution-a",
        simulation_id="simulation-a",
        rollout_id="rollout-a",
        binding_sha256=_DIGEST,
        maximum_cost_usd=1.0,
        rollout_completed=lambda _rollout_id: False,
        observed_spend_usd=lambda: 0.0,
        cancelled=lambda: True,
    )

    assert cancelled.state == TextCellLeaseState.CONTENDED
    assert cancelled.retryable
    assert store._records.list_ids() == ()


@pytest.mark.parametrize("local", [False, True])
def test_admission_lock_wait_obeys_the_same_finite_deadline(tmp_path: Path, local: bool) -> None:
    """A hung local or cross-process lock cannot bypass the lease acquisition deadline."""
    project = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    lease_directory = project.project_directory / "simulation-leases"
    lease_directory.mkdir(parents=True)
    store = TextCellLeaseStore(
        project.project_directory,
        clock=lambda: _TIME,
        poll_interval_seconds=0.01,
        wait_timeout_seconds=0.04,
    )

    started = time.monotonic()
    with (
        store._admission_lock
        if local
        else file_write_lock(lease_directory / "admission", what="test admission holder")
    ):
        blocked = store.acquire(
            lease_id="lease-a",
            resolution_id="resolution-a",
            simulation_id="simulation-a",
            rollout_id="rollout-a",
            binding_sha256=_DIGEST,
            maximum_cost_usd=1.0,
            rollout_completed=lambda _rollout_id: False,
            observed_spend_usd=lambda: 0.0,
        )
    elapsed = time.monotonic() - started

    assert blocked.state == TextCellLeaseState.CONTENDED
    assert elapsed < 0.5
    assert store._records.list_ids() == ()


def test_completed_one_dollar_reservation_reaps_before_actual_ten_cent_spend(
    tmp_path: Path,
) -> None:
    """A crash after a cheap artifact cannot leave its former whole-budget claim counted."""
    project = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    store = TextCellLeaseStore(project.project_directory, clock=lambda: _TIME)
    completed: set[str] = set()
    first = store.acquire(
        lease_id="lease-a",
        resolution_id="resolution-a",
        simulation_id="simulation-a",
        rollout_id="rollout-a",
        binding_sha256=_DIGEST,
        maximum_cost_usd=1.0,
        rollout_completed=completed.__contains__,
        observed_spend_usd=lambda: 0.0,
    )
    completed.add("rollout-a")

    second = store.acquire(
        lease_id="lease-b",
        resolution_id="resolution-a",
        simulation_id="simulation-a",
        rollout_id="rollout-b",
        binding_sha256="b" * 64,
        maximum_cost_usd=1.0,
        rollout_completed=completed.__contains__,
        observed_spend_usd=lambda: 0.1,
    )

    assert first.lease is not None
    assert first.lease.reserved_cost_usd == 1.0
    assert second.state == TextCellLeaseState.OWNED
    assert second.lease is not None
    assert second.lease.reserved_cost_usd == pytest.approx(0.9)
    assert store._records.list_ids() == ("lease-b",)


def test_finite_budget_contender_waits_until_whole_run_reservation_releases(
    tmp_path: Path,
) -> None:
    """Without a per-cell cost bound, a second paid cell cannot safely overlap the first."""
    project = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    completed: set[str] = set()
    spend = [0.0]
    store = TextCellLeaseStore(project.project_directory, clock=lambda: _TIME)

    first = store.acquire(
        lease_id="lease-a",
        resolution_id="resolution-a",
        simulation_id="simulation-a",
        rollout_id="rollout-a",
        binding_sha256=_DIGEST,
        maximum_cost_usd=1.0,
        rollout_completed=completed.__contains__,
        observed_spend_usd=lambda: spend[0],
    )
    elapsed = [0.0]

    def finish_first(seconds: float) -> None:
        elapsed[0] += seconds
        completed.add("rollout-a")
        spend[0] = 0.2

    contender = TextCellLeaseStore(
        project.project_directory,
        clock=lambda: _TIME,
        sleep=finish_first,
        monotonic=lambda: elapsed[0],
        wait_timeout_seconds=0.1,
    )
    second = contender.acquire(
        lease_id="lease-b",
        resolution_id="resolution-a",
        simulation_id="simulation-a",
        rollout_id="rollout-b",
        binding_sha256="b" * 64,
        maximum_cost_usd=1.0,
        rollout_completed=completed.__contains__,
        observed_spend_usd=lambda: spend[0],
    )

    assert first.lease is not None
    assert second.lease is not None
    assert first.lease.reserved_cost_usd == 1.0
    assert second.lease.reserved_cost_usd == pytest.approx(0.8)
    assert elapsed[0] > 0


def test_crash_after_rollout_artifact_recovers_completed_and_clears_reservation(
    tmp_path: Path,
) -> None:
    """A persisted rollout is authoritative when its owner crashes before lease release."""
    project = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    store = TextCellLeaseStore(project.project_directory, clock=lambda: _TIME)
    first = store.acquire(
        lease_id="lease-a",
        resolution_id="resolution-a",
        simulation_id="simulation-a",
        rollout_id="rollout-a",
        binding_sha256=_DIGEST,
        maximum_cost_usd=1.0,
        rollout_completed=lambda _rollout_id: False,
        observed_spend_usd=lambda: 0.0,
    )
    assert first.lease is not None

    recovered = store.acquire(
        lease_id="lease-a",
        resolution_id="resolution-a",
        simulation_id="simulation-a",
        rollout_id="rollout-a",
        binding_sha256=_DIGEST,
        maximum_cost_usd=1.0,
        rollout_completed=lambda rollout_id: rollout_id == "rollout-a",
        observed_spend_usd=lambda: 0.1,
    )

    assert recovered.state == TextCellLeaseState.COMPLETED
    assert store._records.list_ids() == ()


def test_budget_contender_waits_for_active_claim_before_proven_over_budget_block(
    tmp_path: Path,
) -> None:
    """In stop mode only recomputed committed spend after an active claim is a block."""
    project = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    completed: set[str] = set()
    spend = [0.0]
    original = TextCellLeaseStore(project.project_directory, clock=lambda: _TIME)
    original.acquire(
        lease_id="lease-a",
        resolution_id="resolution-a",
        simulation_id="simulation-a",
        rollout_id="rollout-a",
        binding_sha256=_DIGEST,
        maximum_cost_usd=1.0,
        rollout_completed=completed.__contains__,
        observed_spend_usd=lambda: spend[0],
    )
    elapsed = [0.0]

    def resolve_active_claim(seconds: float) -> None:
        elapsed[0] += seconds
        completed.add("rollout-a")
        spend[0] = 1.1

    contender = TextCellLeaseStore(
        project.project_directory,
        clock=lambda: _TIME,
        sleep=resolve_active_claim,
        monotonic=lambda: elapsed[0],
        wait_timeout_seconds=0.1,
    )
    blocked = contender.acquire(
        lease_id="lease-b",
        resolution_id="resolution-a",
        simulation_id="simulation-a",
        rollout_id="rollout-b",
        binding_sha256="b" * 64,
        maximum_cost_usd=1.0,
        rollout_completed=completed.__contains__,
        observed_spend_usd=lambda: spend[0],
        stop_on_overspend=True,
    )

    assert elapsed[0] > 0
    assert blocked.state == TextCellLeaseState.BUDGET_BLOCKED
    assert blocked.observed_spend_usd == 1.1
    assert contender._records.list_ids() == ()


def test_default_admission_warns_and_owns_after_spend_reaches_the_ceiling(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """By default an authorized run admits a cell after reconciled spend crosses the ceiling.

    Args:
        tmp_path: Isolated project root for durable admission leases.
        caplog: Captured lease-store log records.
    """
    project = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    store = TextCellLeaseStore(project.project_directory, clock=lambda: _TIME)

    with caplog.at_level(logging.WARNING, logger="exp.simulation.engines.text.leases"):
        claim = store.acquire(
            lease_id="lease-a",
            resolution_id="resolution-a",
            simulation_id="simulation-a",
            rollout_id="rollout-a",
            binding_sha256=_DIGEST,
            maximum_cost_usd=1.0,
            rollout_completed=lambda _rollout_id: False,
            observed_spend_usd=lambda: 1.5,
        )

    assert claim.state == TextCellLeaseState.OWNED
    assert claim.lease is not None
    assert claim.lease.reserved_cost_usd == 1.0
    assert any("authorized" in record.message for record in caplog.records)


def test_stale_tombstone_refuses_changed_database_record(tmp_path: Path) -> None:
    """Compare-and-replace refuses to overwrite changed evidence and rolls back the transaction."""
    project = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    original = TextCellLeaseStore(project.project_directory, clock=lambda: _TIME)
    original.acquire(
        lease_id="lease-a",
        resolution_id="resolution-a",
        simulation_id="simulation-a",
        rollout_id="rollout-a",
        binding_sha256=_DIGEST,
        maximum_cost_usd=1.0,
        rollout_completed=lambda _rollout_id: False,
        observed_spend_usd=lambda: 0.0,
    )
    original_bytes = original._records.read("lease-a")
    assert original_bytes is not None

    def change_claim(_pid: int) -> bool:
        """Inject a different exact claim between validation and conditional replacement."""
        changed = leases.TextCellLease.model_validate_json(original_bytes).model_copy(
            update={"binding_sha256": "b" * 64}
        )
        original._records.write("lease-a", canonical_json_bytes(changed))
        return False

    recovery = TextCellLeaseStore(
        project.project_directory,
        clock=lambda: _TIME + timedelta(minutes=16),
        owner_alive=change_claim,
    )

    with pytest.raises(TextCellLeaseError, match="changed before mutation"):
        recovery.acquire(
            lease_id="lease-a",
            resolution_id="resolution-a",
            simulation_id="simulation-a",
            rollout_id="rollout-a",
            binding_sha256=_DIGEST,
            maximum_cost_usd=1.0,
            rollout_completed=lambda _rollout_id: False,
            observed_spend_usd=lambda: 0.0,
        )

    assert original._records.read("lease-a") == original_bytes


def test_parallel_cell_reservations_share_but_never_duplicate_remaining_budget(
    tmp_path: Path,
) -> None:
    """Two bounded attempts may overlap; a third cannot claim already reserved dollars."""
    store = TextCellLeaseStore(
        tmp_path / "projects" / "project-a",
        clock=lambda: _TIME,
        wait_timeout_seconds=0.001,
        poll_interval_seconds=0.001,
    )

    def acquire(suffix: str) -> TextCellLeaseClaim:
        """Request forty cents from a shared one-dollar ceiling."""
        return store.acquire(
            lease_id=f"lease-{suffix}",
            resolution_id="resolution-a",
            simulation_id="simulation-a",
            rollout_id=f"rollout-{suffix}",
            binding_sha256=_DIGEST,
            maximum_cost_usd=1.0,
            reservation_cost_usd=0.4,
            stop_on_overspend=True,
            rollout_completed=lambda _: False,
            observed_spend_usd=lambda: 0.1,
        )

    first, second = acquire("a"), acquire("b")
    assert first.state == second.state == TextCellLeaseState.OWNED
    assert first.lease is not None and second.lease is not None
    assert first.lease.reserved_cost_usd == second.lease.reserved_cost_usd == 0.4
    third = acquire("c")
    assert third.state == TextCellLeaseState.CONTENDED
    store.release(first.lease)
    assert acquire("c").state == TextCellLeaseState.OWNED
