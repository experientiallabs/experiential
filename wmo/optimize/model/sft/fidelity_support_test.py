"""Shared fidelity fixtures for SFT builder regression tests."""

from wmo.common.core.artifacts import ArtifactId
from wmo.common.evaluations import FidelityPair


def usable_fidelity_pairs(
    cell_ids: tuple[ArtifactId, ...],
    observed_rollout_id: ArtifactId,
    simulated_rollout_id: ArtifactId,
) -> tuple[FidelityPair, ...]:
    """Build deterministic usable fidelity pairs.

    Args:
        cell_ids: Planned fidelity cell identities.
        observed_rollout_id: Observed rollout shared by the fixture pairs.
        simulated_rollout_id: Simulated rollout shared by the fixture pairs.

    Returns:
        One usable pair for each planned cell, in input order.
    """
    return tuple(
        FidelityPair(
            fidelity_cell_id=cell_id,
            observed_cell_id=f"observed-{index}",
            observed_rollout_id=observed_rollout_id,
            simulated_rollout_id=simulated_rollout_id,
            observed_score=1.0,
            simulated_score=0.95,
            absolute_error=0.05,
            status="usable",
        )
        for index, cell_id in enumerate(cell_ids)
    )
