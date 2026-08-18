"""Finite-cost reconciliation for persisted simulation evidence."""

from __future__ import annotations

import hashlib
import math

from wmo.common.core.artifacts import ArtifactInput
from wmo.common.evaluations.evidence import read_rollout
from wmo.common.project import ProjectStore
from wmo.common.rollouts import (
    RolloutArtifact,
    RolloutEventKind,
    SimulationArtifactSet,
    unknown_dispatch_reserved_cost_usd,
    unknown_spend_failure,
)
from wmo.optimize.router.errors import RouterCompositionError
from wmo.simulation.engines.text.bindings import rollout_id_for_binding
from wmo.simulation.engines.text.errors import SimulationConfigurationError
from wmo.simulation.engines.text.grounding import (
    load_completion_contract,
    unknown_dispatch_worst_case_usd,
)
from wmo.simulation.engines.text.rollout_support import rollout_spend


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
        The reservation persisted with the failure, or the durable sandbox episode ceiling
        for environment dispatches that predate per-failure reservation persistence.

    Raises:
        RouterCompositionError: No durable worst-case reservation was persisted.
    """
    reserved = unknown_dispatch_reserved_cost_usd(rollout.failure)
    if reserved is None and rollout.evidence_source == "sandbox":
        binding = rollout.sandbox_binding
        if binding is not None:
            reserved = binding.environment_maximum_episode_cost_usd
    if reserved is None:
        raise RouterCompositionError(
            "simulation rollout has unknown dispatched spend and no persisted reservation"
        )
    return reserved


def verified_simulation_spend(
    project: ProjectStore,
    expected: SimulationArtifactSet,
    completion_contract_input: ArtifactInput | None,
) -> float:
    """Recompute one phase's spend from verified immutable rollouts.

    Args:
        project: Project store containing the completed simulation artifacts.
        expected: Exact artifact set returned for the simulation phase.
        completion_contract_input: Reviewed completion reservation contract reference used to
            charge superseded retry attempts conservatively.

    Returns:
        Finite total of candidate, world-model, and retrieval dispatch spend.

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
    values: list[float] = []
    for rollout_id in artifact_set.artifact_ids:
        rollout = read_rollout(project.artifacts, rollout_id)[0]
        values.append(observed_rollout_spend(rollout))
        values.extend(superseded_attempt_spend(project, rollout, completion_contract_input))
    return math.fsum(values)


def superseded_attempt_spend(
    project: ProjectStore,
    rollout: RolloutArtifact,
    completion_contract_input: ArtifactInput | None,
) -> tuple[float, ...]:
    """Return conservative charges for every superseded retry attempt behind one rollout.

    Args:
        project: Project store containing the immutable prior-attempt artifacts.
        rollout: Final rollout selected for its cell, possibly after retries.
        completion_contract_input: Reviewed completion reservation contract reference.

    Returns:
        One worst-case charge per superseded attempt, so retried dispatches with unknown
        spend still count against the phase ceiling.

    Raises:
        RouterCompositionError: A superseded attempt cannot be reconciled conservatively.
    """
    if rollout.retry_attempt == 0:
        return ()
    binding = rollout.simulation_binding
    if binding is None:
        raise RouterCompositionError("retried simulation rollout lacks its cell binding")
    try:
        contract = load_completion_contract(project.artifacts, completion_contract_input)
    except SimulationConfigurationError as exc:
        raise RouterCompositionError(str(exc)) from exc
    charges = []
    for attempt in range(rollout.retry_attempt):
        prior, _input = read_rollout(
            project.artifacts, rollout_id_for_binding(binding, attempt=attempt)
        )
        spend = rollout_spend(
            prior,
            unknown_dispatch_fallback_usd=lambda item: unknown_dispatch_worst_case_usd(
                contract,
                item.simulation_binding.candidate_alias
                if item.simulation_binding is not None
                else None,
            ),
        )
        if spend is None:
            raise RouterCompositionError(
                "superseded simulation attempt spend cannot be reconciled conservatively"
            )
        charges.append(spend)
    return tuple(charges)
