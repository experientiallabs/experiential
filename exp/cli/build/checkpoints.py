"""Completed grounded-build checkpoints survive interrupted project selection."""

from exp.common.core.artifacts import ArtifactInput, canonical_json_bytes, sha256_json
from exp.common.models import ModelSnapshot
from exp.common.project import ProjectBuildArtifacts, ProjectStore, artifact_input
from exp.common.project.records import ProjectRecords
from exp.simulation.build import ProjectBuild
from exp.simulation.retrieval import load_rag_index
from exp.simulation.world_model.artifact import (
    WORLD_MODEL_ARTIFACT_PATH,
    GroundedWorldModelArtifact,
)


def _records(store: ProjectStore) -> ProjectRecords:
    """Bind the durable completed-build checkpoint namespace."""
    return ProjectRecords(store.paths.root, store.paths.project_id, "grounded-build-checkpoints")


def _key(
    trace: ArtifactInput,
    task: ArtifactInput,
    world_alias: str,
    world_snapshot: ModelSnapshot,
    embedder_snapshot: ModelSnapshot,
    top_k: int,
) -> str:
    """Bind the checkpoint to every immutable grounded-build input."""
    return sha256_json(
        {
            "trace": trace.model_dump(mode="json"),
            "task": task.model_dump(mode="json"),
            "world_alias": world_alias,
            "world": world_snapshot.model_dump(mode="json"),
            "embedder": embedder_snapshot.model_dump(mode="json"),
            "top_k": top_k,
        }
    )


def save_grounded_checkpoint(store: ProjectStore, build: ProjectBuildArtifacts) -> None:
    """Save the fully persisted graph before its atomic configuration/review selection."""
    world = GroundedWorldModelArtifact.model_validate_json(
        store.artifacts.read_bytes(build.world_model.artifact_id, WORLD_MODEL_ARTIFACT_PATH)
    )
    serving = load_rag_index(store.artifacts, build.serving_rag.artifact_id)
    key = _key(
        build.trace_dataset,
        build.task_set,
        world.model_alias,
        world.model,
        serving.index.embedder,
        world.top_k,
    )
    records = _records(store)
    payload = canonical_json_bytes(build)
    with records.transaction():
        previous = records.read(key)
        if previous is not None:
            if previous != payload:
                raise ValueError("completed build checkpoint differs from its frozen inputs")
            return
        records.write(key, payload, exclusive=True)


def reuse_completed_grounded_artifacts(
    store: ProjectStore,
    completed: ProjectBuild,
    *,
    world_alias: str,
    world_snapshot: ModelSnapshot,
    embedder_snapshot: ModelSnapshot,
    top_k: int,
) -> ProjectBuildArtifacts | None:
    """Reuse a completely matching verified build without credentials or provider calls.

    Args:
        store: Project artifact store containing a possible completed build.
        completed: Current persisted trace and task build.
        world_alias: Configured world-model alias required by the artifact.
        world_snapshot: Secret-free world-model identity required by the artifact.
        embedder_snapshot: Secret-free embedder identity required by both indexes.
        top_k: Requested retrieval result count.

    Returns:
        Verified existing build pointers, or ``None`` when any identity differs.

    """
    trace_input = artifact_input(completed.artifacts.trace_dataset.manifest)
    task_input = artifact_input(
        store.artifacts.read(completed.artifacts.task_set.task_set_id).manifest
    )
    payload = _records(store).read(
        _key(trace_input, task_input, world_alias, world_snapshot, embedder_snapshot, top_k)
    )
    existing = (
        ProjectBuildArtifacts.model_validate_json(payload)
        if payload is not None
        else store.load_project().build
    )
    if existing is None:
        return None
    if existing.trace_dataset != trace_input or existing.task_set != task_input:
        return None
    serving = load_rag_index(store.artifacts, existing.serving_rag.artifact_id)
    fit = load_rag_index(store.artifacts, existing.fit_rag.artifact_id)
    if (
        serving.index.embedder != embedder_snapshot
        or fit.index.embedder != embedder_snapshot
        or serving.index.default_top_k != top_k
        or fit.index.default_top_k != top_k
        or serving.index.included_partitions != ("fit", "held_out")
        or fit.index.included_partitions != ("fit",)
    ):
        return None
    world = GroundedWorldModelArtifact.model_validate_json(
        store.artifacts.read_bytes(existing.world_model.artifact_id, WORLD_MODEL_ARTIFACT_PATH)
    )
    if (
        world.serving_rag != existing.serving_rag
        or world.model_alias != world_alias
        or world.model != world_snapshot
        or world.top_k != top_k
    ):
        return None
    return existing
