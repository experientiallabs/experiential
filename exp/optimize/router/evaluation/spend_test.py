"""Tests for conservative finite-cost reconciliation of persisted simulation evidence."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from exp.common.core.artifacts import (
    ArtifactInput,
    FailureAttribution,
    FailureCode,
    JsonValue,
    StructuredFailure,
)
from exp.common.models import (
    BillingSource,
    EmbeddingCostReservation,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    Usage,
)
from exp.common.rollouts import (
    UNKNOWN_DISPATCH_RESERVED_COST_KEY,
    RolloutArtifact,
    RolloutEventKind,
    RolloutSpan,
    SimulationCellBinding,
    SimulationMode,
    StopReason,
    WorldModelSimulatorSnapshot,
)
from exp.optimize.router.errors import RouterCompositionError
from exp.optimize.router.evaluation.spend import observed_rollout_spend

_DIGEST = "a" * 64
_TIME = datetime(2026, 8, 11, tzinfo=UTC)


def _model() -> ModelSnapshot:
    """Return one pinned provider model snapshot fixture."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="openai",
        model_id="gpt-5.4",
        capabilities_sha256=_DIGEST,
        connection_sha256=_DIGEST,
    )


def _binding() -> SimulationCellBinding:
    """Return one complete immutable cell binding fixture."""
    return SimulationCellBinding(
        evaluation_plan_input=ArtifactInput(artifact_id="evaluation-plan", sha256=_DIGEST),
        task_set_input=ArtifactInput(artifact_id="task-set", sha256=_DIGEST),
        fit_rag_input=ArtifactInput(artifact_id="fit-rag", sha256=_DIGEST),
        grounded_world_model_input=ArtifactInput(
            artifact_id="grounded-world-model", sha256=_DIGEST
        ),
        task_set_tasks_sha256=_DIGEST,
        task_sha256=_DIGEST,
        candidate_alias="candidate-a",
        candidate=_model(),
        agent_id="customer-agent",
        repeat=0,
        world_model_alias="world-model-a",
        world_model=_model(),
        simulator_id="world-model-v1",
        prompt_id="world-prompt-v1",
        prompt_version="v1",
        prompt_sha256=_DIGEST,
        query_embedding=EmbeddingCostReservation(
            model=_model(),
            input_usd_per_million_tokens=0.0,
            maximum_attempts=1,
            maximum_input_tokens=1,
        ),
        simulation_spec_input=ArtifactInput(artifact_id="simulation-spec", sha256=_DIGEST),
        simulation_spec_sha256=_DIGEST,
        simulation_inputs_sha256=_DIGEST,
    )


def _rollout(
    *,
    candidate_economics: OperationEconomics,
    stop_reason: StopReason = StopReason.COMPLETED,
    failure: StructuredFailure | None = None,
) -> RolloutArtifact:
    """Build one world-model rollout with a single dispatched candidate call.

    Args:
        candidate_economics: Combined candidate operation economics to persist.
        stop_reason: Terminal reason recorded for the episode.
        failure: Optional structured failure recorded with the evidence.

    Returns:
        Canonical rollout fixture bound to one dispatched candidate call span.
    """
    return RolloutArtifact(
        schema_version=1,
        created_at=_TIME,
        inputs=(
            ArtifactInput(artifact_id="evaluation-plan", sha256=_DIGEST),
            ArtifactInput(artifact_id="fit-rag", sha256=_DIGEST),
            ArtifactInput(artifact_id="grounded-world-model", sha256=_DIGEST),
            ArtifactInput(artifact_id="simulation-spec", sha256=_DIGEST),
            ArtifactInput(artifact_id="task-set", sha256=_DIGEST),
        ),
        code_revision="test-revision",
        artifact_id="rollout-artifact-1",
        simulation_id="simulation-1",
        cell_id="cell-1",
        mode=SimulationMode.WORLD_MODEL,
        rollout_id="rollout-1",
        trace_id="0123456789abcdef0123456789abcdef",
        evidence_source="world_model",
        source_run_id="run-1",
        task_id="task-1",
        candidate=_model(),
        agent_id="customer-agent",
        simulator=WorldModelSimulatorSnapshot(
            simulator_id="world-model-v1",
            prompt_id="world-prompt-v1",
            prompt_version="v1",
            prompt_sha256=_DIGEST,
            world_model=_model(),
        ),
        world_model=_model(),
        seed=7,
        repeat=0,
        spans=(
            RolloutSpan(
                span_id="span-1",
                kind=RolloutEventKind.AGENT_MODEL_CALL,
                started_at=_TIME,
                ended_at=_TIME + timedelta(seconds=1),
                model=_model(),
            ),
        ),
        stop_reason=stop_reason,
        failure=failure,
        candidate_economics=candidate_economics,
        retrieval_economics=OperationEconomics(),
        simulation_spec_sha256=_DIGEST,
        simulation_binding=_binding(),
    )


def _unknown_spend_failure(
    *,
    reserved: float | None,
) -> StructuredFailure:
    """Return one persisted provider dispatch failure with permanently ambiguous spend.

    Args:
        reserved: Optional durable worst-case reservation persisted with the failure.

    Returns:
        Structured provider failure marking an unknown-spend dispatch window.
    """
    details: dict[str, JsonValue] = {
        "phase": "candidate_or_world_model",
        "provider_dispatch_unknown_spend": True,
        "retry_classification": "transport",
    }
    if reserved is not None:
        details[UNKNOWN_DISPATCH_RESERVED_COST_KEY] = reserved
    return StructuredFailure(
        code=FailureCode.PROVIDER,
        message="text simulation provider call failed with ProviderTransportError",
        retryable=True,
        exception_type="ProviderTransportError",
        attribution=FailureAttribution.MODEL,
        details=details,
    )


def _observed(cost: float) -> OperationEconomics:
    """Return complete observed economics for one priced operation."""
    return OperationEconomics(
        usage=Usage(input_tokens=8, output_tokens=4),
        cost_usd=NumericMeasurement(value=cost, provenance="observed"),
    )


def test_mixed_success_and_unknown_spend_rollouts_reconcile_conservatively() -> None:
    """One unknown-spend failure charges its reservation without aborting priced peers."""
    succeeded = _rollout(candidate_economics=_observed(0.10))
    failed = _rollout(
        candidate_economics=OperationEconomics(),
        stop_reason=StopReason.FAILURE,
        failure=_unknown_spend_failure(reserved=0.25),
    )

    total = math.fsum(observed_rollout_spend(item) for item in (succeeded, failed))

    assert observed_rollout_spend(succeeded) == 0.10
    assert observed_rollout_spend(failed) == 0.25
    assert total == pytest.approx(0.35)


def test_unknown_spend_failure_without_reservation_stays_fail_closed() -> None:
    """Ambiguous dispatch spend with no durable worst-case reservation cannot reconcile."""
    failed = _rollout(
        candidate_economics=OperationEconomics(),
        stop_reason=StopReason.FAILURE,
        failure=_unknown_spend_failure(reserved=None),
    )

    with pytest.raises(RouterCompositionError, match="no persisted reservation"):
        observed_rollout_spend(failed)


def test_ordinary_unpriced_evidence_still_fails_closed() -> None:
    """A dispatched call without unknown-spend classification must stay fully priced."""
    unpriced = _rollout(candidate_economics=OperationEconomics())

    with pytest.raises(RouterCompositionError, match="not fully observed"):
        observed_rollout_spend(unpriced)


def test_stale_lease_failure_charges_its_persisted_whole_ceiling_barrier() -> None:
    """A stale paid-cell tombstone reconciles to its exact durable reservation."""
    details: dict[str, JsonValue] = {
        "phase": "paid_cell_stale_lease",
        "lease_id": "lease-1",
        UNKNOWN_DISPATCH_RESERVED_COST_KEY: 1.0,
    }
    failed = _rollout(
        candidate_economics=OperationEconomics(),
        stop_reason=StopReason.FAILURE,
        failure=StructuredFailure(
            code=FailureCode.BUDGET,
            message="a prior paid-cell claim expired after its owner exited",
            attribution=FailureAttribution.MODEL,
            details=details,
        ),
    )

    assert observed_rollout_spend(failed) == 1.0
