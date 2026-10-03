"""Finite-cost reconciliation for persisted simulation evidence."""

from __future__ import annotations

import hashlib
import math
from typing import Literal, overload

from exp.common.core.artifacts import ArtifactInput
from exp.common.evaluations.evidence import read_rollout
from exp.common.project import ProjectStore
from exp.common.rollouts import (
    RolloutArtifact,
    RolloutEventKind,
    SimulationArtifactSet,
    unknown_dispatch_reservation_is_upper_bound,
    unknown_dispatch_reserved_cost_usd,
    unknown_spend_failure,
)
from exp.optimize.router.errors import RouterCompositionError
from exp.simulation.engines.text.errors import SimulationConfigurationError
from exp.simulation.engines.text.grounding import load_completion_contract
from exp.simulation.engines.text.lineage_spend import lineage_spend


def observed_rollout_spend(rollout: RolloutArtifact) -> float:
    """Reconcile observed spend and reservation-derived simulation estimates.

    A rollout whose failure left one dispatch's spend unknown is charged its exact persisted
    worst-case reservation on top of every priced operation, so one ambiguous cell counts
    conservatively against the ceiling instead of aborting reconciliation for every other
    valid rollout. Unknown-spend evidence without a proven reservation bound stays fail-closed.

    Args:
        rollout: Completed simulation evidence whose provider economics are inspected.

    Returns:
        Candidate, retrieval, simulator, environment, and priced orchestration spend, plus
        any persisted worst-case reservation for an unknown-spend dispatch failure.

    Raises:
        RouterCompositionError: A dispatched operation is unknown, unpriced, or misclassified.
    """
    amount = _observed_rollout_spend(rollout)
    if amount is None:
        raise RouterCompositionError(
            "simulation rollout has unknown dispatched spend and no persisted reservation "
            "proven as an upper bound; reconcile pricing before finite-budget execution"
        )
    return amount


def _observed_rollout_spend(rollout: RolloutArtifact) -> float | None:
    """Validate every recorded economy before preserving any unresolved dispatch liability.

    Args:
        rollout: Immutable simulation evidence with observed costs and retained dispatch bounds.

    Returns:
        The sum of verified costs and bounded reservations, or None when a dispatched charge
        remains unknown without a proven upper bound.

    Raises:
        RouterCompositionError: Evidence is not simulation output, lacks required bindings or
            costs, or contains invalid cost values or provenance.
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
        charge = _unknown_dispatch_charge(rollout)
        if charge is None:
            return None
        costs.append(charge)
    return math.fsum(costs)


def _unknown_dispatch_charge(rollout: RolloutArtifact) -> float | None:
    """Return a verified reservation bound for one unknown-spend failure, when available.

    Args:
        rollout: Failed evidence whose dispatched spend is permanently ambiguous.

    Returns:
        The reservation persisted with the failure, or the durable sandbox episode ceiling
        for environment dispatches that predate per-failure reservation persistence, or None
        when the retained estimate cannot bound the liability.
    """
    if not unknown_dispatch_reservation_is_upper_bound(rollout.failure):
        return None
    reserved = unknown_dispatch_reserved_cost_usd(rollout.failure)
    if reserved is None and rollout.evidence_source == "sandbox":
        binding = rollout.sandbox_binding
        if binding is not None:
            reserved = binding.environment_maximum_episode_cost_usd
    return reserved


@overload
def verified_simulation_spend(
    project: ProjectStore,
    expected: SimulationArtifactSet,
    completion_contract_input: ArtifactInput | None,
    *,
    allow_unknown_interrupted: Literal[False] = False,
) -> float:
    """Require a finite reconciled total for ordinary and finite-budget callers."""
    ...


@overload
def verified_simulation_spend(
    project: ProjectStore,
    expected: SimulationArtifactSet,
    completion_contract_input: ArtifactInput | None,
    *,
    allow_unknown_interrupted: Literal[True],
) -> float | None:
    """Preserve unresolved spend for shared-ledger execution or free replay."""
    ...


def verified_simulation_spend(
    project: ProjectStore,
    expected: SimulationArtifactSet,
    completion_contract_input: ArtifactInput | None,
    *,
    allow_unknown_interrupted: bool = False,
) -> float | None:
    """Recompute one phase's spend from verified immutable rollouts.

    Args:
        project: Project store containing the completed simulation artifacts.
        expected: Exact artifact set returned for the simulation phase.
        completion_contract_input: Reviewed completion reservation contract reference used to
            charge superseded retry attempts conservatively.
        allow_unknown_interrupted: Shared-ledger callers may retain unknown dispatched spend
            for uncapped execution or free replay, without fabricating whole-cell reservations.

    Returns:
        Reconciled total, or None when permitted dispatched liability has no proven cost bound.

    Raises:
        RouterCompositionError: The set, index, rollout, or economics cannot be verified.
    """
    stored = project.artifacts.read(expected.artifact_set_id)
    if stored.manifest.artifact_type != "simulation-artifact-set":
        raise RouterCompositionError("simulation spend source has the wrong artifact type")
    artifact_set = SimulationArtifactSet.model_validate_json(
        project.artifacts.read_bytes(expected.artifact_set_id, "artifact-set.json")
    )
    if artifact_set != expected:
        raise RouterCompositionError("simulation spend source differs from its completed set")
    index_payload = project.artifacts.read_bytes(
        expected.artifact_set_id, artifact_set.artifacts_path
    )
    if hashlib.sha256(index_payload).hexdigest() != artifact_set.artifacts_sha256:
        raise RouterCompositionError("simulation spend index digest has drifted")
    try:
        load_completion_contract(project.artifacts, completion_contract_input)
    except SimulationConfigurationError as exc:
        raise RouterCompositionError(str(exc)) from exc
    rollouts = tuple(
        read_rollout(project.artifacts, identity)[0] for identity in artifact_set.artifact_ids
    )

    total = lineage_spend(project.artifacts, rollouts, measure=_observed_rollout_spend)
    if total is None and not allow_unknown_interrupted:
        raise RouterCompositionError("simulation lineage spend is unknown")
    return total
