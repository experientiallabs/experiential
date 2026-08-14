"""Completed-build verification for router composition."""

from __future__ import annotations

from wmo.common.core.artifacts import ArtifactInput
from wmo.common.project import ProjectBuildArtifacts, ProjectStore, artifact_input
from wmo.simulation.build import ProjectBuild
from wmo.workflow.errors import RouterCompositionError


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
