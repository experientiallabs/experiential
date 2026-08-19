"""Finite-cost reconciliation for persisted simulation evidence."""

from __future__ import annotations

import math

from wmo.common.rollouts import (
    RolloutArtifact,
    RolloutEventKind,
    unknown_dispatch_reserved_cost_usd,
    unknown_spend_failure,
)
from wmo.optimize.router.errors import RouterCompositionError


def observed_rollout_spend(rollout: RolloutArtifact) -> float:
    """Reconcile observed spend and reservation-derived simulation estimates.

    A rollout whose failure left one dispatch's spend unknown is charged its exact persisted
    worst-case reservation on top of every priced operation, so one ambiguous cell counts
    conservatively against the ceiling instead of aborting reconciliation for every other
    valid rollout. Unknown-spend evidence with no persisted reservation stays fail-closed.

    Args:
        rollout: Completed simulation evidence whose provider economics are inspected.

    Returns:
        Candidate, retrieval, simulator, environment, and priced orchestration spend, plus
        any persisted worst-case reservation for an unknown-spend dispatch failure.

    Raises:
        RouterCompositionError: A dispatched operation is unknown, unpriced, or misclassified.
    """
    unknown_spend = unknown_spend_failure(rollout.failure)
    economics = []
    costs = []
    if any(span.kind == RolloutEventKind.AGENT_MODEL_CALL for span in rollout.spans):
        economics.append((rollout.candidate_economics, True))
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
            economics.append((rollout.world_model_economics, True))
    elif rollout.evidence_source == "sandbox":
        binding = rollout.sandbox_binding
        if binding is None:
            raise RouterCompositionError("sandbox rollout lacks its environment cost binding")
        if binding.environment_maximum_episode_cost_usd != 0:
            economics.append((rollout.sandbox_economics, False))
    else:
        raise RouterCompositionError("production evidence cannot count as simulation spend")
    if rollout.orchestration_economics is not None:
        orchestration_cost = rollout.orchestration_economics.cost_usd
        if orchestration_cost is not None:
            economics.append((rollout.orchestration_economics, False))
    for operation, allows_completion_estimate in economics:
        cost = operation.cost_usd if operation is not None else None
        allowed_provenance = (
            {"observed", "estimated"} if allows_completion_estimate else {"observed"}
        )
        if cost is None:
            if unknown_spend:
                continue
            raise RouterCompositionError("simulation rollout spend is not fully observed")
        if cost.provenance not in allowed_provenance or cost.value < 0:
            raise RouterCompositionError("simulation rollout spend is not fully observed")
        costs.append(cost.value)
    if unknown_spend:
        costs.append(_unknown_dispatch_charge(rollout))
    return math.fsum(costs)


def _unknown_dispatch_charge(rollout: RolloutArtifact) -> float:
    """Return the exact persisted worst-case charge for one unknown-spend failure.

    Args:
        rollout: Failed evidence whose dispatched spend is permanently ambiguous.

    Returns:
        The reservation persisted with the failure.

    Raises:
        RouterCompositionError: No durable worst-case reservation was persisted.
    """
    reserved = unknown_dispatch_reserved_cost_usd(rollout.failure)
    if reserved is None:
        raise RouterCompositionError(
            "simulation rollout has unknown dispatched spend and no persisted reservation"
        )
    return reserved
