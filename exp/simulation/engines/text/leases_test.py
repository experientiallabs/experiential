"""Adversarial durability tests for text simulation paid-cell claims."""

import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.locks import file_write_lock
from exp.common.project import ArtifactStore, ProjectPaths
from exp.simulation.engines.text.leases import (
    TextCellLeaseError,
    TextCellLeaseState,
    TextCellLeaseStatus,
    TextCellLeaseStore,
)

_TIME = datetime(2026, 8, 12, tzinfo=UTC)
_DIGEST = "a" * 64


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
    assert tuple((project.project_directory / "simulation-leases").glob("*.json")) == ()


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
    assert len(tuple((project.project_directory / "simulation-leases").glob("*.json"))) == 1


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
    assert tuple((project.project_directory / "simulation-leases").glob("*.json")) == ()


def test_admission_lock_wait_obeys_the_same_finite_deadline(tmp_path: Path) -> None:
    """A hung admission lock cannot bypass the lease acquisition deadline."""
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
    with file_write_lock(lease_directory / "admission", what="test admission holder"):
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
    assert tuple(lease_directory.glob("*.json")) == ()


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
    lease_paths = tuple((project.project_directory / "simulation-leases").glob("*.json"))
    assert tuple(path.name for path in lease_paths) == ("lease-b.json",)


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
    assert tuple((project.project_directory / "simulation-leases").glob("*.json")) == ()


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
    assert tuple((project.project_directory / "simulation-leases").glob("*.json")) == ()


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


def test_stale_tombstone_rejects_symlink_swap_without_touching_victim(tmp_path: Path) -> None:
    """A lease swapped after safe read cannot redirect tombstone bytes outside the lease dir."""
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
    lease_path = project.project_directory / "simulation-leases" / "lease-a.json"
    victim = tmp_path / "victim.txt"
    victim.write_text("do not overwrite", encoding="utf-8")

    def swap_to_symlink(_pid: int) -> bool:
        lease_path.unlink()
        lease_path.symlink_to(victim)
        return False

    recovery = TextCellLeaseStore(
        project.project_directory,
        clock=lambda: _TIME + timedelta(minutes=16),
        owner_alive=swap_to_symlink,
    )

    with pytest.raises(TextCellLeaseError, match="cannot be mutated safely"):
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

    assert victim.read_text(encoding="utf-8") == "do not overwrite"
    assert lease_path.is_symlink()
