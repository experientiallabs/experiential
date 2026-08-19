"""Tests for phase-scoped router simulation specifications."""

from __future__ import annotations

from datetime import UTC, datetime

from wmo.common.core.artifacts import ArtifactInput
from wmo.common.evaluations import EvaluationCell, EvaluationPlan, EvaluationProtocol
from wmo.common.models import ModelSnapshot, RoutedCandidateSnapshot
from wmo.common.routing import KnnGuard
from wmo.optimize.router.composition import RouterEvaluationSetup
from wmo.optimize.router.evaluation.simulation_spec import build_router_simulation_spec
from wmo.simulation.specs import WorldModelSettings

_DIGEST = "a" * 64
_TIME = datetime(2026, 8, 17, tzinfo=UTC)


def _snapshot(alias: str) -> ModelSnapshot:
    """Freeze one deterministic fixture model identity.

    Args:
        alias: Stable local model alias used as the model identifier.

    Returns:
        Snapshot with fixed capability and connection digests.
    """
    return ModelSnapshot(
        provider="test",
        model_id=alias,
        revision="fixture",
        capabilities_sha256=_DIGEST,
        connection_sha256="b" * 64,
    )


def _plan_and_setup() -> tuple[EvaluationPlan, EvaluationCell, RouterEvaluationSetup]:
    """Build one minimal frozen plan, its simulated fit cell, and a reviewed setup.

    Returns:
        Plan, the single simulated fit cell, and the matching evaluation setup.
    """
    cell = EvaluationCell(
        cell_id="cell-a",
        task_id="task-a",
        candidate_alias="candidate-a",
        repeat=0,
        purpose="fit",
        execution="simulate",
    )
    plan = EvaluationPlan(
        schema_version=1,
        created_at=_TIME,
        code_revision="test-revision",
        plan_id="plan-a",
        task_set_id="task-set-a",
        candidate_snapshots=(
            RoutedCandidateSnapshot(alias="candidate-a", model=_snapshot("candidate-a")),
        ),
        pricing_snapshot_id="pricing-a",
        pricing_snapshot_sha256=_DIGEST,
        cells=(cell,),
    )
    protocol = EvaluationProtocol(
        protocol_id="protocol-world",
        evidence_source="world_model",
        agent_id="agent-a",
        simulator_id="text-world-model-v1",
        world_model=_snapshot("world-model-a"),
        simulator_prompt_id="world-model-text-v1",
        rubric_id="rubric-a",
        judge_calibration_id="calibration-a",
        pricing_snapshot_id="pricing-a",
    )
    setup = RouterEvaluationSetup(
        candidates=(RoutedCandidateSnapshot(alias="candidate-a", model=_snapshot("candidate-a")),),
        observed_cells=(),
        production_protocol=protocol,
        simulation_protocol=protocol,
        embedding_set_id="embeddings-a",
        fit_rag_input=ArtifactInput(artifact_id="fit-rag-a", sha256=_DIGEST),
        pricing_snapshot_id="pricing-a",
        guard=KnnGuard(
            maximum_neighbors=10,
            minimum_paired_observations=8,
            relative_similarity_threshold=0.0,
            uncertainty_multiplier=0.5,
            quality_tolerance=0.0,
        ),
        judgment_status="provisional",
        world_model_settings=WorldModelSettings(
            world_model_alias="world-model-a",
            grounded_world_model_input=ArtifactInput(artifact_id="world-model-a", sha256=_DIGEST),
            prompt_version="text-world-model-v1",
        ),
        simulation_completion_input=ArtifactInput(
            artifact_id="completion-contract-a", sha256=_DIGEST
        ),
        agent_id="agent-a",
        seed=7,
        maximum_steps=2,
        maximum_concurrency=1,
    )
    return plan, cell, setup


def test_overspend_policy_changes_stable_simulation_identity() -> None:
    """Specs that differ only in overspend policy carry the flag and distinct identities."""
    plan, cell, setup = _plan_and_setup()
    plan_input = ArtifactInput(artifact_id="plan-a", sha256=_DIGEST)
    task_input = ArtifactInput(artifact_id="task-set-a", sha256=_DIGEST)

    default_spec, stop_spec = (
        build_router_simulation_spec(
            plan,
            plan_input,
            task_input,
            setup,
            10.0,
            _TIME,
            "test-revision",
            (cell,),
            phase="fit",
            stop_on_overspend=stop,
        )
        for stop in (False, True)
    )

    assert default_spec.stop_on_overspend is False
    assert stop_spec.stop_on_overspend is True
    assert default_spec.simulation_id != stop_spec.simulation_id
