"""Post-consent immutable inputs for automatic router composition."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from wmo.common.core.artifacts import ArtifactInput
from wmo.common.evaluations import ObservedProductionCell
from wmo.common.models import (
    CandidateTokenPrice,
    PricingSnapshot,
    load_model_catalog,
    persist_pricing_snapshot,
    router_candidate_capabilities_sha256,
)
from wmo.common.project import ProjectStore, artifact_input
from wmo.common.routing import ReservedFrozenEmbeddingSet, persist_router_embeddings
from wmo.optimize.router.automatic_router_preflight import AutomaticRouterPreflight
from wmo.optimize.router.manual_judge_artifacts import write_production_rollout
from wmo.optimize.router.router_execution_contract import (
    CandidateExecutionBinding,
    RouterExecutionContract,
    persist_router_execution_contract,
)
from wmo.runtime.models import CapabilityRequirement, RuntimeModelCatalog
from wmo.runtime.router.capability import (
    RouterRuntimeCapabilityContract,
    RuntimeCandidateCapability,
    capability_contract_input,
    persist_router_runtime_capability_contract,
)
from wmo.simulation.specs import (
    SimulationCompletionContract,
    persist_simulation_completion_contract,
)


@dataclass(frozen=True)
class AutomaticRouterArtifacts:
    """Immutable evidence, pricing, embeddings, and execution reservations."""

    observed_cells: tuple[ObservedProductionCell, ...]
    attribution_input: ArtifactInput
    pricing: PricingSnapshot
    router_embeddings: ReservedFrozenEmbeddingSet
    router_embedding_input: ArtifactInput
    simulation_completion: SimulationCompletionContract
    simulation_completion_input: ArtifactInput
    runtime_capabilities: RouterRuntimeCapabilityContract
    runtime_capability_input: ArtifactInput
    execution_contract: RouterExecutionContract
    execution_contract_input: ArtifactInput


def materialize_automatic_router_artifacts(
    project: ProjectStore,
    preflight: AutomaticRouterPreflight,
    runtime_catalog: RuntimeModelCatalog,
    *,
    attribution_input: ArtifactInput,
    router_embedding_maximum_attempts: int,
    completion_maximum_attempts: int,
    maximum_provider_cost_usd: float,
    created_at: datetime,
    code_revision: str,
) -> AutomaticRouterArtifacts:
    """Persist exact automatic-router inputs after provider-spend consent.

    Args:
        project: Existing project whose completed build is being optimized.
        preflight: Aggregate read-only prerequisite and reservation result.
        runtime_catalog: Credential-capable resolver over the just-persisted catalog.
        attribution_input: Immutable real-overlap attribution persisted after consent.
        router_embedding_maximum_attempts: Active embedding retry ceiling.
        completion_maximum_attempts: Active completion retry ceiling.
        maximum_provider_cost_usd: Exact operator-approved shared provider ceiling.
        created_at: Materialization time for newly completed artifacts.
        code_revision: Exact producer revision.

    Returns:
        Complete immutable inputs for dependency-injected router composition.

    Raises:
        ValueError: Catalog state, credentials, model identity, or replay evidence differs.
    """
    active_catalog = load_model_catalog(project.model_catalog_path)
    if active_catalog != preflight.catalog:
        raise ValueError("persisted model catalog differs from the confirmed router preflight")
    resolved_embedder = runtime_catalog.preflight(
        preflight.embedder_alias,
        CapabilityRequirement(requires_embeddings=True),
    )
    if (
        resolved_embedder.snapshot != preflight.embedder
        or resolved_embedder.embedding_client is None
    ):
        raise ValueError("resolved embedder differs from the completed-build preflight")

    observed_cells = _persist_observed_cells(
        project,
        preflight,
        attribution_input,
        created_at,
        code_revision,
    )
    pricing = persist_pricing_snapshot(
        project.artifacts,
        _canonical_candidate_prices(preflight),
        created_at=created_at,
        code_revision=code_revision,
    )
    pricing_input = artifact_input(project.artifacts.read(pricing.pricing_snapshot_id).manifest)
    embeddings = persist_router_embeddings(
        project.artifacts,
        task_set_input=preflight.completed_build.task_set,
        tasks=preflight.tasks,
        embedder_alias=preflight.embedder_alias,
        embedder=preflight.embedder,
        client=resolved_embedder.embedding_client,
        reservation=preflight.router_embedding_reservation,
        active_input_usd_per_million_tokens=(
            preflight.router_embedding_reservation.input_usd_per_million_tokens
        ),
        active_maximum_attempts_per_feature=router_embedding_maximum_attempts,
        created_at=created_at,
        code_revision=code_revision,
    )
    embedding_input = artifact_input(project.artifacts.read(embeddings.embedding_set_id).manifest)
    completion_inputs = _sorted_inputs(
        preflight.completed_build.task_set,
        preflight.completed_build.fit_rag,
        preflight.completed_build.world_model,
        pricing_input,
    )
    completion, completion_input = persist_simulation_completion_contract(
        project.artifacts,
        inputs=completion_inputs,
        candidate_requests=preflight.candidate_completion_reservations,
        world_model_alias=preflight.world_model_alias,
        world_model_request=preflight.world_model_completion_reservation,
        maximum_attempts=completion_maximum_attempts,
        created_at=created_at,
        code_revision=code_revision,
    )
    runtime_capabilities = persist_router_runtime_capability_contract(
        project.artifacts,
        candidates=_runtime_capability_bindings(preflight),
        created_at=created_at,
        code_revision=code_revision,
    )
    runtime_capability_input = capability_contract_input(project.artifacts, runtime_capabilities)
    execution_inputs = _sorted_inputs(
        preflight.completed_build.trace_dataset,
        preflight.completed_build.task_set,
        preflight.completed_build.fit_rag,
        preflight.completed_build.world_model,
        preflight.setup_input,
        preflight.judge_audit_input,
        preflight.approved_calibration_input,
        attribution_input,
        pricing_input,
        embedding_input,
        completion_input,
        runtime_capability_input,
    )
    execution = persist_router_execution_contract(
        project.artifacts,
        inputs=execution_inputs,
        router_embedding_input=embedding_input,
        simulation_completion_input=completion_input,
        runtime_capability_input=runtime_capability_input,
        router_embedding_reservation=preflight.router_embedding_reservation,
        candidates=_candidate_bindings(preflight),
        incumbent_alias=preflight.incumbent_alias,
        agent_factory_sha256=preflight.agent_factory_sha256,
        simulation_configuration_sha256=preflight.simulation_configuration_sha256,
        preferred_fidelity_overlaps=preflight.preferred_fidelity_overlaps,
        fidelity_planned_overlaps=preflight.fidelity_overlap_count,
        fidelity_minimum_usable_overlaps=min(8, preflight.fidelity_overlap_count),
        world_model_alias=preflight.world_model_alias,
        world_model=preflight.world_model,
        world_model_request=preflight.world_model_completion_reservation,
        judge_alias=preflight.judge_alias,
        judge_model=preflight.judge_model,
        judge_request=preflight.judge_completion_reservation,
        maximum_judge_provider_calls=preflight.judge_provider_call_count,
        maximum_provider_cost_usd=maximum_provider_cost_usd,
        created_at=created_at,
        code_revision=code_revision,
    )
    execution_input = artifact_input(
        project.artifacts.read(execution.execution_contract_id).manifest
    )
    return AutomaticRouterArtifacts(
        observed_cells=observed_cells,
        attribution_input=attribution_input,
        pricing=pricing,
        router_embeddings=embeddings,
        router_embedding_input=embedding_input,
        simulation_completion=completion,
        simulation_completion_input=completion_input,
        runtime_capabilities=runtime_capabilities,
        runtime_capability_input=runtime_capability_input,
        execution_contract=execution,
        execution_contract_input=execution_input,
    )


def _canonical_candidate_prices(
    preflight: AutomaticRouterPreflight,
) -> tuple[CandidateTokenPrice, ...]:
    """Align selected prices with the canonical candidate snapshot order.

    Args:
        preflight: Verified candidate snapshots and complete catalog-derived prices.

    Returns:
        Exact price rows ordered by canonical candidate alias.

    Raises:
        ValueError: Prices repeat, omit, or add a selected candidate alias.
    """
    prices = {item.candidate_alias: item for item in preflight.candidate_prices}
    aliases = tuple(item.alias for item in preflight.candidates)
    if len(prices) != len(preflight.candidate_prices) or set(prices) != set(aliases):
        raise ValueError("candidate pricing differs from the canonical selected candidates")
    return tuple(prices[alias] for alias in aliases)


def _persist_observed_cells(
    project: ProjectStore,
    preflight: AutomaticRouterPreflight,
    attribution_input: ArtifactInput,
    created_at: datetime,
    code_revision: str,
) -> tuple[ObservedProductionCell, ...]:
    """Persist every selected real fit overlap as production rollout evidence.

    Args:
        project: Project-local artifact store.
        preflight: Verified real task and trace overlaps.
        attribution_input: Exact immutable attribution authorizing selected candidate snapshots.
        created_at: Artifact materialization time.
        code_revision: Exact producer revision.

    Returns:
        Deterministic observed-cell bindings for evaluation planning.
    """
    cells = []
    for item in preflight.observed_traces[: preflight.fidelity_overlap_count]:
        rollout_input = write_production_rollout(
            project,
            preflight.setup,
            item.task,
            item.trace,
            created_at,
            code_revision,
            attributed_candidate=item.attribution.candidate_model,
            attribution_input=attribution_input,
        )
        cells.append(
            ObservedProductionCell(
                task_id=item.task.task_id,
                candidate_alias=item.candidate_alias,
                repeat=0,
                rollout_artifact_id=rollout_input.artifact_id,
            )
        )
    return tuple(cells)


def _candidate_bindings(
    preflight: AutomaticRouterPreflight,
) -> tuple[CandidateExecutionBinding, ...]:
    """Bind each selected candidate to exact execution capabilities and reservation.

    Args:
        preflight: Verified catalog, candidates, and request reservations.

    Returns:
        Candidate bindings in operator-selected order.

    Raises:
        ValueError: A selected alias lacks the preflight-proven capability record.
    """
    requests = {
        item.candidate_alias: item.request for item in preflight.candidate_completion_reservations
    }
    bindings = []
    for candidate in preflight.candidates:
        record = preflight.catalog.models[candidate.alias]
        if record.capabilities is None:
            raise ValueError(f"candidate {candidate.alias!r} has no capability declaration")
        bindings.append(
            CandidateExecutionBinding(
                candidate_alias=candidate.alias,
                model=candidate.model,
                routing_capabilities_sha256=router_candidate_capabilities_sha256(
                    record.capabilities
                ),
                request=requests[candidate.alias],
            )
        )
    return tuple(bindings)


def _runtime_capability_bindings(
    preflight: AutomaticRouterPreflight,
) -> tuple[RuntimeCandidateCapability, ...]:
    """Build runtime-owned candidate bindings without extending the legacy policy.

    Args:
        preflight: Verified catalog and selected candidate identities.

    Returns:
        Exact candidate bindings in operator-selected order.

    Raises:
        ValueError: A selected candidate has no capability declaration.
    """
    bindings = []
    for candidate in preflight.candidates:
        capabilities = preflight.catalog.models[candidate.alias].capabilities
        if capabilities is None:
            raise ValueError(f"candidate {candidate.alias!r} has no capability declaration")
        bindings.append(
            RuntimeCandidateCapability(
                candidate_alias=candidate.alias,
                model=candidate.model,
                routing_capabilities_sha256=router_candidate_capabilities_sha256(capabilities),
            )
        )
    return tuple(bindings)


def _sorted_inputs(*values: ArtifactInput) -> tuple[ArtifactInput, ...]:
    """Return unique artifact pointers in canonical identifier order.

    Args:
        *values: Immutable artifact pointers to canonicalize.

    Returns:
        Sorted unique input pointers.

    Raises:
        ValueError: One artifact identifier appears with different manifest evidence.
    """
    by_id: dict[str, ArtifactInput] = {}
    for value in values:
        existing = by_id.get(value.artifact_id)
        if existing is not None and existing != value:
            raise ValueError(f"artifact input {value.artifact_id!r} has conflicting manifests")
        by_id[value.artifact_id] = value
    return tuple(by_id[key] for key in sorted(by_id))
