"""Verified build-owned lineage inputs for runtime retrieval refresh."""

from __future__ import annotations

from wmo.common.project import (
    ArtifactCorruptionError,
    ArtifactStore,
    ProjectBuildArtifacts,
    artifact_input,
)
from wmo.simulation.mining.bindings import load_task_set_lineage_bindings
from wmo.simulation.retrieval.contracts import RAGLineageBinding


def load_completed_build_rag_lineage_bindings(
    store: ArtifactStore,
    completed: ProjectBuildArtifacts,
) -> tuple[RAGLineageBinding, ...]:
    """Derive exact refresh bindings from a recursively verified completed build.

    Args:
        store: Project-local immutable artifact store.
        completed: Exact immutable outputs selected as the project's completed build.

    Returns:
        One sorted retrieval binding for every imported source trace.

    Raises:
        ArtifactCorruptionError: Build pointers, source inputs, or complete bindings are missing or
            disagree with their immutable artifacts.
    """
    task_stored = store.read(completed.task_set.artifact_id)
    if task_stored.manifest.artifact_type != "task-set":
        raise ArtifactCorruptionError(
            f"completed build task-set pointer names {task_stored.manifest.artifact_type}"
        )
    if artifact_input(task_stored.manifest) != completed.task_set:
        raise ArtifactCorruptionError("completed build task-set manifest digest differs")
    trace_stored = store.read(completed.trace_dataset.artifact_id)
    if trace_stored.manifest.artifact_type != "trace-dataset":
        raise ArtifactCorruptionError(
            f"completed build trace pointer names {trace_stored.manifest.artifact_type}"
        )
    if artifact_input(trace_stored.manifest) != completed.trace_dataset:
        raise ArtifactCorruptionError("completed build trace-dataset manifest digest differs")
    if task_stored.manifest.inputs != (completed.trace_dataset,):
        raise ArtifactCorruptionError(
            "completed build task set does not depend on its exact trace dataset"
        )
    payload = load_task_set_lineage_bindings(store, completed.task_set.artifact_id)
    if payload.trace_dataset != completed.trace_dataset:
        raise ArtifactCorruptionError(
            "completed build lineage bindings name a different trace dataset"
        )
    return tuple(
        RAGLineageBinding(
            trace_id=binding.trace_id,
            lineage_id=binding.lineage_id,
            partition=binding.partition,
        )
        for binding in payload.bindings
    )
