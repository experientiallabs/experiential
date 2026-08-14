"""Completed-build verification for router composition."""

from __future__ import annotations

from datetime import datetime

from wmo.common.core.artifacts import ArtifactInput
from wmo.common.project import ProjectBuildArtifacts, ProjectStore, artifact_input
from wmo.optimize.router.errors import RouterCompositionError
from wmo.simulation.build import BuildReviewReadiness, ProjectBuild, build_project
from wmo.simulation.ingest.otlp import TraceNormalizationResult


def completed_project_build(project: ProjectStore) -> ProjectBuildArtifacts:
    """Load the exact completed project build before workflow side effects.

    Args:
        project: Initialized project whose immutable build graph must be selected.

    Returns:
        Exact trace, task, serving RAG, fit RAG, and world-model pointers.

    Raises:
        RouterCompositionError: The project has no completed build pointer.
    """
    completed = project.load_project().build
    if completed is None:
        raise RouterCompositionError("router composition requires a completed project build")
    return completed


def verify_completed_build_inputs(
    completed: ProjectBuildArtifacts,
    built: ProjectBuild,
) -> None:
    """Require composition traces and tasks to match the completed build graph.

    Args:
        completed: Exact build pointers selected in project configuration.
        built: Deterministically loaded trace and task artifacts for this composition.

    Raises:
        RouterCompositionError: Trace or task identities differ from the completed build.
    """
    trace_input = artifact_input(built.artifacts.trace_dataset.manifest)
    if trace_input != completed.trace_dataset or built.review.task_set != completed.task_set:
        raise RouterCompositionError(
            "router composition traces or tasks differ from the completed project build"
        )


def reconstruct_completed_project_build(
    project: ProjectStore,
    normalized: TraceNormalizationResult,
    *,
    created_at: datetime,
) -> ProjectBuild:
    """Reconstruct the exact selected build under its frozen local mining contract.

    Args:
        project: Project whose completed trace and task artifacts are selected.
        normalized: Canonical traces that must reproduce the selected immutable evidence.
        created_at: Materialization time used only if an immutable replay needs it.

    Returns:
        Verified trace, task, mining, and review objects for router composition.

    Raises:
        RouterCompositionError: Review state, artifact pointers, or deterministic replay differs.
    """
    completed = completed_project_build(project)
    review_state = project.read_review()
    if not isinstance(review_state, dict) or "build_review" not in review_state:
        raise RouterCompositionError("completed project build has no build review contract")
    try:
        review = BuildReviewReadiness.model_validate(review_state["build_review"])
    except ValueError as exc:
        raise RouterCompositionError("completed project build review is invalid") from exc
    if review.trace_dataset != completed.trace_dataset or review.task_set != completed.task_set:
        raise RouterCompositionError("completed project build review names different artifacts")
    try:
        trace_input = artifact_input(
            project.artifacts.read(completed.trace_dataset.artifact_id).manifest
        )
        task_input = artifact_input(project.artifacts.read(completed.task_set.artifact_id).manifest)
    except (FileNotFoundError, ValueError) as exc:
        raise RouterCompositionError("completed project build artifacts are unavailable") from exc
    if trace_input != completed.trace_dataset or task_input != completed.task_set:
        raise RouterCompositionError("completed project build artifact manifest changed")
    try:
        built = build_project(
            normalized,
            project,
            created_at=created_at,
            code_revision=review.code_revision,
            mining_spec=review.mining_spec,
        )
    except ValueError as exc:
        raise RouterCompositionError(f"completed project build cannot be replayed: {exc}") from exc
    verify_completed_build_inputs(completed, built)
    return built


def verify_completed_grounding_inputs(
    completed: ProjectBuildArtifacts,
    *,
    fit_rag_input: ArtifactInput,
    grounded_world_model_input: ArtifactInput,
) -> None:
    """Require simulation grounding to match the completed build graph.

    Args:
        completed: Exact completed-build pointers selected by the project.
        fit_rag_input: Reviewed fit-only RAG pointer supplied for optimization.
        grounded_world_model_input: Reviewed world-model artifact pointer supplied for simulation.

    Raises:
        RouterCompositionError: Either grounding input differs from the completed build.
    """
    if fit_rag_input != completed.fit_rag or grounded_world_model_input != completed.world_model:
        raise RouterCompositionError(
            "evaluation setup grounding differs from the completed project build"
        )
