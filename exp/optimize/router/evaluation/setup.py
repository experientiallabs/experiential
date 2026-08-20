"""Completed-build and review checks for router evaluation setup."""

from __future__ import annotations

from exp.common.core.artifacts import ArtifactId, ArtifactInput
from exp.common.evaluations import EvaluationProtocol
from exp.common.project import ProjectBuildArtifacts
from exp.optimize.router.errors import RouterCompositionError
from exp.optimize.router.evaluation.build import verify_completed_grounding_inputs


def verify_router_evaluation_setup(
    *,
    completed: ProjectBuildArtifacts,
    fit_rag_input: ArtifactInput,
    grounded_world_model_input: ArtifactInput,
    production_protocol: EvaluationProtocol,
    simulation_protocol: EvaluationProtocol,
    rubric_id: ArtifactId,
    calibration_id: ArtifactId,
) -> None:
    """Bind reviewed protocols and simulation grounding to completed artifacts.

    Args:
        completed: Exact completed-build retrieval and world-model pointers.
        fit_rag_input: Reviewed fit-only RAG input.
        grounded_world_model_input: Reviewed executable world-model input.
        production_protocol: Protocol selected for real production evidence.
        simulation_protocol: Protocol selected for world-model evidence.
        rubric_id: Approved rubric identity.
        calibration_id: Approved manual judge-calibration identity.

    Raises:
        RouterCompositionError: A review, role, or completed-build identity differs.
    """
    verify_completed_grounding_inputs(
        completed,
        fit_rag_input=fit_rag_input,
        grounded_world_model_input=grounded_world_model_input,
    )
    protocols = (production_protocol, simulation_protocol)
    if any(
        protocol.rubric_id != rubric_id or protocol.judge_calibration_id != calibration_id
        for protocol in protocols
    ):
        raise RouterCompositionError("evaluation protocols differ from approved review artifacts")
    if production_protocol.evidence_source != "production":
        raise RouterCompositionError("production_protocol must name production evidence")
    if simulation_protocol.evidence_source != "world_model":
        raise RouterCompositionError("simulation_protocol must name world-model evidence")
