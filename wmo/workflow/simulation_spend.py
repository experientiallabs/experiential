"""Finite-cost reconciliation for persisted simulation evidence."""

from __future__ import annotations

import math

from wmo.common.rollouts import RolloutArtifact, RolloutEventKind
from wmo.workflow.errors import RouterCompositionError


def observed_rollout_spend(rollout: RolloutArtifact) -> float:
    """Reconcile one rollout's observed spend and conservative retrieval estimates.

    Args:
        rollout: Completed simulation evidence whose provider economics are inspected.

    Returns:
        Candidate, retrieval, simulator, environment, and priced orchestration spend.

    Raises:
        RouterCompositionError: A dispatched operation is unknown, unpriced, or misclassified.
    """
    if rollout.failure is not None and (
        rollout.failure.details.get("provider_dispatch_unknown_spend") is True
        or rollout.failure.details.get("environment_dispatch_unknown_spend") is True
        or rollout.failure.details.get("phase") == "paid_cell_stale_lease"
    ):
        raise RouterCompositionError("simulation rollout has unknown dispatched spend")
    economics = []
    costs = []
    if any(span.kind == RolloutEventKind.AGENT_MODEL_CALL for span in rollout.spans):
        economics.append(rollout.candidate_economics)
    if rollout.evidence_source == "world_model":
        retrieval = rollout.retrieval_economics
        if retrieval is not None and retrieval.cost_usd is not None:
            retrieval_cost = retrieval.cost_usd
            if retrieval_cost.provenance != "estimated" or retrieval_cost.value < 0:
                raise RouterCompositionError(
                    "simulation retrieval spend lacks its conservative reservation estimate"
                )
            costs.append(retrieval_cost.value)
        world_dispatched = any(
            span.kind == RolloutEventKind.SIMULATOR_WORLD_MODEL_CALL for span in rollout.spans
        )
        if world_dispatched and not costs:
            raise RouterCompositionError(
                "simulation retrieval spend is missing before a world-model dispatch"
            )
        if world_dispatched:
            economics.append(rollout.world_model_economics)
    elif rollout.evidence_source == "sandbox":
        binding = rollout.sandbox_binding
        if binding is None:
            raise RouterCompositionError("sandbox rollout lacks its environment cost binding")
        if binding.environment_maximum_episode_cost_usd != 0:
            economics.append(rollout.sandbox_economics)
    else:
        raise RouterCompositionError("production evidence cannot count as simulation spend")
    if rollout.orchestration_economics is not None:
        orchestration_cost = rollout.orchestration_economics.cost_usd
        if orchestration_cost is not None:
            economics.append(rollout.orchestration_economics)
    for operation in economics:
        cost = operation.cost_usd if operation is not None else None
        if cost is None or cost.provenance != "observed" or cost.value < 0:
            raise RouterCompositionError("simulation rollout spend is not fully observed")
        costs.append(cost.value)
    return math.fsum(costs)
