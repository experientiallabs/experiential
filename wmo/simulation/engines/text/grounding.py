"""Fit-only retrieval bindings and finite-cost preflight for text simulation."""

from __future__ import annotations

from wmo.common.core.artifacts import (
    ArtifactInput,
    FailureAttribution,
    FailureCode,
    StructuredFailure,
)
from wmo.common.models import EmbeddingCostReservation, NumericMeasurement, OperationEconomics
from wmo.simulation.engines.text.errors import SimulationConfigurationError
from wmo.simulation.retrieval import TraceRAGRetriever
from wmo.simulation.specs import SimulationSpec, WorldModelSettings


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


def maximum_query_reservation(
    reservation: EmbeddingCostReservation,
) -> OperationEconomics:
    """Return the retry-inclusive maximum spend reserved for one retrieval query.

    Args:
        reservation: Persisted price, retry ceiling, and maximum query-token count.

    Returns:
        Estimated economics for the largest allowed query and every allowed retry.
    """
    cost = (
        reservation.maximum_input_tokens
        * reservation.maximum_attempts
        * reservation.input_usd_per_million_tokens
        / 1_000_000
    )
    return OperationEconomics(cost_usd=NumericMeasurement(value=cost, provenance="estimated"))


def query_reservation_failure(
    reservation: EmbeddingCostReservation,
    remaining_cost_usd: float,
) -> StructuredFailure | None:
    """Reject a query reservation that cannot fit before any provider dispatch.

    Args:
        reservation: Persisted worst-case query-embedding reservation.
        remaining_cost_usd: Reconciled provider budget available to the cell.

    Returns:
        Structured budget failure when the reservation exceeds the remainder, otherwise ``None``.

    Raises:
        SimulationConfigurationError: The reservation unexpectedly has no estimated cost.
    """
    economics = maximum_query_reservation(reservation)
    cost = economics.cost_usd
    if cost is None:  # pragma: no cover - maximum_query_reservation always prices the call
        raise SimulationConfigurationError("query-embedding reservation has unknown cost")
    if cost.value <= remaining_cost_usd:
        return None
    return StructuredFailure(
        code=FailureCode.BUDGET,
        message="query-embedding reservation exceeds remaining simulation spend",
        attribution=FailureAttribution.MODEL,
        details={
            "phase": "query_embedding_reservation",
            "reserved_cost_usd": cost.value,
            "remaining_cost_usd": remaining_cost_usd,
        },
    )
