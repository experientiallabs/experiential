"""Adversarial durability tests for text simulation paid-cell claims."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from wmo.common.project import ArtifactStore, ProjectPaths
from wmo.simulation.engines.text.leases import TextCellLeaseState, TextCellLeaseStore

_TIME = datetime(2026, 8, 12, tzinfo=UTC)
_DIGEST = "a" * 64


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
        maximum_cost_usd=None,
        rollout_completed=lambda: False,
        observed_spend_usd=lambda: 0.0,
    )

    recovered = TextCellLeaseStore(
        project.project_directory,
        clock=lambda: _TIME + timedelta(minutes=16),
        owner_alive=lambda _pid: False,
    ).acquire(
        lease_id="lease-a",
        resolution_id="resolution-a",
        simulation_id="simulation-a",
        rollout_id="rollout-a",
        binding_sha256=_DIGEST,
        maximum_cost_usd=None,
        rollout_completed=lambda: False,
        observed_spend_usd=lambda: 0.0,
    )

    assert first.state == TextCellLeaseState.OWNED
    assert recovered.state == TextCellLeaseState.STALE
    assert recovered.lease == first.lease
