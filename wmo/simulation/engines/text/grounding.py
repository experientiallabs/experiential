"""Fit-only retrieval bindings and finite-cost preflight for text simulation."""

from __future__ import annotations

from wmo.common.core.artifacts import (
    ArtifactInput,
    FailureAttribution,
    FailureCode,
    StructuredFailure,
)
from wmo.common.models import (
    CompletionCostReservation,
    verify_completion_reservation,
)
from wmo.common.project import ArtifactCorruptionError, ArtifactStore, artifact_input
from wmo.common.tasks import LoadedTaskSet, load_task_set
from wmo.runtime.models import ResolvedModel
from wmo.simulation.engines.text.errors import SimulationConfigurationError
from wmo.simulation.retrieval import TraceRAGRetriever
from wmo.simulation.specs import (
    SimulationCompletionContract,
    SimulationSpec,
    WorldModelSettings,
    load_simulation_completion_contract,
)


def load_completion_contract(
    store: ArtifactStore, contract_input: ArtifactInput
) -> SimulationCompletionContract:
    """Load the exact completion reservation artifact.

    Args:
        store: Project-local immutable artifact store.
        contract_input: Expected immutable contract manifest.

    Returns:
        Verified completion contract.

    Raises:
        SimulationConfigurationError: The artifact is missing, corrupt, or hash-mismatched.
    """
    try:
        contract, persisted_input = load_simulation_completion_contract(
            store, contract_input.artifact_id
        )
    except (OSError, ValueError) as exc:
        raise SimulationConfigurationError(
            "simulation completion reservation artifact cannot be read safely"
        ) from exc
    if persisted_input != contract_input:
        raise SimulationConfigurationError(
            "simulation completion reservation manifest differs from its frozen input"
        )
    return contract


def load_simulation_task_set(store: ArtifactStore, task_set_input: ArtifactInput) -> LoadedTaskSet:
    """Load the exact manifest-bound task set selected for simulation.

    Args:
        store: Project-local immutable artifact store.
        task_set_input: Expected immutable task-set manifest.

    Returns:
        Verified task-set envelope and task rows.

    Raises:
        SimulationConfigurationError: Artifact type, digest, or task content differs.
    """
    try:
        stored = store.read(task_set_input.artifact_id)
    except ArtifactCorruptionError as exc:
        raise SimulationConfigurationError(
            f"simulation task set {task_set_input.artifact_id!r} cannot be read safely"
        ) from exc
    if (
        stored.manifest.artifact_type != "task-set"
        or artifact_input(stored.manifest) != task_set_input
    ):
        raise SimulationConfigurationError(
            "simulation task_set_input must name the exact persisted task-set manifest"
        )
    try:
        return load_task_set(store, task_set_input.artifact_id)
    except ArtifactCorruptionError as exc:
        raise SimulationConfigurationError(
            f"simulation task set {task_set_input.artifact_id!r} has invalid task content"
        ) from exc


def verify_fit_retriever(
    retriever: TraceRAGRetriever,
    fit_rag_input: ArtifactInput,
) -> None:
    """Require one exact fit-only artifact and its active embedding binding.

    Args:
        retriever: Read-only retriever selected for active simulation.
        fit_rag_input: Exact completed-build manifest pointer required by the simulation.

    Raises:
        SimulationConfigurationError: The artifact, partition, or frozen lineage scope differs.
    """
    if retriever.rag_input != fit_rag_input:
        raise SimulationConfigurationError(
            "fit retriever manifest differs from the simulation RAG input"
        )
    index = retriever.index
    if index.included_partitions != ("fit",):
        raise SimulationConfigurationError("text simulation requires a fit-only RAG index")
    if index.included_lineage_ids != index.fit_lineage_ids:
        raise SimulationConfigurationError(
            "fit retriever lineages differ from the frozen fit lineage set"
        )


def require_grounding_settings(
    spec: SimulationSpec,
    *,
    fit_rag_input: ArtifactInput,
    retriever: TraceRAGRetriever,
) -> WorldModelSettings:
    """Validate immutable retrieval identity and economics before any side effect.

    Args:
        spec: Finite simulation specification carrying the persisted reservation.
        fit_rag_input: Exact completed-build fit index required by the simulator.
        retriever: Active read-only retriever and catalog-backed embedding binding.

    Returns:
        Validated world-model settings from the specification.

    Raises:
        SimulationConfigurationError: An artifact, embedder, retry, price, or budget pin differs.
    """
    if fit_rag_input not in spec.inputs:
        raise SimulationConfigurationError(
            "simulation spec inputs must include the exact fit RAG manifest reference"
        )
    if spec.maximum_cost_usd is None:
        raise SimulationConfigurationError(
            "text simulation requires a finite provider spend ceiling"
        )
    settings = spec.world_model
    if settings is None:  # pragma: no cover - selected settings are validated by SimulationSpec
        raise SimulationConfigurationError("world-model simulation settings are missing")
    if settings.grounded_world_model_input not in spec.inputs:
        raise SimulationConfigurationError(
            "simulation spec inputs must include the grounded world-model artifact"
        )
    reservation = settings.query_embedding
    if reservation is None:
        raise SimulationConfigurationError(
            "text simulation requires a query-embedding cost reservation"
        )
    if reservation.model != retriever.index.embedder:
        raise SimulationConfigurationError(
            "query-embedding reservation differs from the fit RAG embedder"
        )
    if reservation.maximum_attempts != retriever.maximum_attempts:
        raise SimulationConfigurationError(
            "query-embedding retry reservation differs from the active client"
        )
    if reservation.input_usd_per_million_tokens != retriever.input_usd_per_million_tokens:
        raise SimulationConfigurationError(
            "query-embedding price reservation differs from the active catalog"
        )
    return settings


def completion_reservations(
    contract: SimulationCompletionContract,
    *,
    candidate_alias: str,
    candidate: ResolvedModel,
    world_model: ResolvedModel,
) -> tuple[CompletionCostReservation, CompletionCostReservation]:
    """Resolve and verify exact candidate and world request ceilings.

    Args:
        contract: Frozen automatic-simulation reservations.
        candidate_alias: Evaluation cell candidate alias.
        candidate: Active exact candidate model.
        world_model: Active exact world model.

    Returns:
        Candidate and world request reservations.

    Raises:
        SimulationConfigurationError: A reservation is absent or differs from active metadata.
    """
    if contract.world_model_alias != world_model.alias:
        raise SimulationConfigurationError(
            "completion reservation contract selects a different world-model alias"
        )
    attempts = contract.maximum_attempts
    world = contract.world_model_request
    candidates = {item.candidate_alias: item.request for item in contract.candidate_requests}
    selected = candidates.get(candidate_alias)
    if selected is None:
        raise SimulationConfigurationError(
            f"candidate {candidate_alias!r} lacks a completion cost reservation"
        )
    try:
        verify_completion_reservation(
            selected,
            model=candidate.snapshot,
            capabilities=candidate.capabilities,
            maximum_attempts=attempts,
        )
        verify_completion_reservation(
            world,
            model=world_model.snapshot,
            capabilities=world_model.capabilities,
            maximum_attempts=attempts,
        )
    except ValueError as exc:
        raise SimulationConfigurationError(str(exc)) from exc
    return selected, world


def unknown_dispatch_worst_case_usd(
    contract: SimulationCompletionContract,
    candidate_alias: str | None,
) -> float | None:
    """Return the persisted retry-inclusive worst case for one ambiguous provider dispatch.

    The failed dispatch is one candidate or one world-model call, but persisted evidence does
    not always record which, so the returned charge conservatively covers both request
    ceilings from the frozen completion contract.

    Args:
        contract: Frozen automatic-simulation reservations.
        candidate_alias: Candidate alias bound to the failed cell, or ``None`` when unknown.

    Returns:
        The combined candidate and world-model retry-inclusive maximum, or ``None`` when the
        alias reservation is unavailable.
    """
    if candidate_alias is None:
        return None
    candidates = {item.candidate_alias: item.request for item in contract.candidate_requests}
    candidate = candidates.get(candidate_alias)
    if candidate is None:
        return None
    return (
        candidate.estimated_maximum_call_cost_usd
        + contract.world_model_request.estimated_maximum_call_cost_usd
    )


def episode_reservation_failure(
    settings: WorldModelSettings,
    *,
    completion_contract: SimulationCompletionContract,
    remaining_cost_usd: float,
    stop_on_overspend: bool = True,
) -> StructuredFailure | None:
    """Reject an episode only when stop mode finds the reconciled remainder exhausted.

    Frozen completion reservations carry planning estimates only, so an episode is never
    rejected because an estimated full run looks too expensive. By default the authorized run
    also admits episodes after the remainder is exhausted; stop mode returns a structured
    zero-dispatch failure instead. Every actual request is still checked against the model's
    real context capacity immediately before dispatch.

    Args:
        settings: Frozen retrieval and completion reservations.
        completion_contract: Frozen completion reservations.
        remaining_cost_usd: Reconciled cell budget remaining under the durable lease.
        stop_on_overspend: When true, an exhausted remainder blocks the next episode.

    Returns:
        Structured zero-dispatch failure when stop mode finds no reconciled spend remaining.

    Raises:
        SimulationConfigurationError: A configured reservation is missing or unpriced.
    """
    query = settings.query_embedding
    if query is None:  # pragma: no cover - grounding settings require it
        raise SimulationConfigurationError("query-embedding reservation is missing")
    if remaining_cost_usd > 0 or not stop_on_overspend:
        return None
    return StructuredFailure(
        code=FailureCode.BUDGET,
        message="reconciled provider spend has exhausted the remaining simulation ceiling",
        attribution=FailureAttribution.MODEL,
        details={
            "phase": "episode_provider_spend",
            "remaining_cost_usd": remaining_cost_usd,
        },
    )
