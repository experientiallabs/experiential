"""Completed-build verification for router composition."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from exp.common.core.artifacts import ArtifactInput
from exp.common.project import (
    ArtifactStoreError,
    ProjectBuildArtifacts,
    ProjectStore,
    artifact_input,
)
from exp.common.tasks import TaskCase, TaskSet, load_task_set
from exp.common.traces.ingest import PersistedTraceDataset, persist_trace_dataset
from exp.common.traces.ingest.otlp import TraceNormalizationResult
from exp.optimize.router.errors import RouterCompositionError
from exp.simulation.build import BuildReviewReadiness
from exp.simulation.mining.bindings import TaskSetLineageBindings, load_task_set_lineage_bindings


@dataclass(frozen=True)
class VerifiedProjectBuild:
    """The completed project build exactly as storage holds it, after verification.

    Every field is loaded from the project's immutable artifacts rather than recomputed from the
    evidence behind them, so this object states what verification proved: the selected build's
    pointers, the review contract that names them, and the stored evidence whose content-addressed
    identities still bind their own bytes.

    Attributes:
        completed: Exact immutable build pointers selected in project configuration.
        review: Validated build-review handoff, whose own identity binds the mining spec, the
            project configuration, the producer revision and both artifact pointers.
        trace_dataset: Stored normalized evidence, proven to be exactly what this composition's
            canonical traces normalize to.
        task_set: Stored representative task-set envelope.
        tasks: Ordered mined task cases the stored task set holds.
        lineage: Complete source-trace lineage and partition assignments the task set owns.
    """

    completed: ProjectBuildArtifacts
    review: BuildReviewReadiness
    trace_dataset: PersistedTraceDataset
    task_set: TaskSet
    tasks: tuple[TaskCase, ...]
    lineage: TaskSetLineageBindings


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


def reconstruct_completed_project_build(
    project: ProjectStore,
    normalized: TraceNormalizationResult,
    *,
    created_at: datetime,
) -> VerifiedProjectBuild:
    """Reconstruct the completed build from storage and verify it without re-deriving it.

    What this proves: the selected task set is the one the build stored, unaltered. Every artifact
    in the build graph is content-addressed, so loading one recomputes its identity from its own
    bytes. The task-set identity binds its tasks, its coverage report and its complete lineage
    bindings against the exact source trace dataset, and the dataset identity binds its canonical
    records, their source provenance and their model-identity evidence. An artifact that was
    edited, truncated, swapped or rebuilt under different inputs therefore no longer matches the
    identity the completed build points at, and is refused here.

    What this deliberately does not prove: that the task set follows from the traces. Deriving it
    again would mean mining again under the geometry the review names, and the decision-tag
    geometry mines by asking a decision model about every distinct case, which this path has no
    model, alias or bank root to ask. Verification is decoupled from the mining geometry instead,
    so a project mined by tagging optimizes a router exactly like one mined by hashing. That is a
    real reduction, from proving the stored task set derivable to proving it unaltered, and it is
    accepted on purpose: it is how the hosted path already reads the same build, and nothing
    downstream of here reads the mining evidence between a trace dataset and its task set.

    Args:
        project: Project whose completed trace and task artifacts are selected.
        normalized: Canonical traces for this composition. They must normalize to the exact stored
            trace dataset, which is everything they can still prove about it once mining is out of
            verification.
        created_at: Materialization time handed to normalization persistence, which requires one.
            Every replay adopts the stored artifact's own completion time, so nothing here is
            stamped with it.

    Returns:
        The verified stored trace dataset, task set, lineage bindings, and review contract.

    Raises:
        RouterCompositionError: Review state, artifact pointers, artifact content, task origin, or
            the supplied canonical traces differ from the completed project build.
    """
    completed = completed_project_build(project)
    review = _completed_build_review(project, completed)
    try:
        trace_manifest = project.artifacts.read(completed.trace_dataset.artifact_id).manifest
        task_manifest = project.artifacts.read(completed.task_set.artifact_id).manifest
        loaded = load_task_set(project.artifacts, completed.task_set.artifact_id)
        # One load does the strongest work available here: it recomputes the task-set identity
        # from the stored tasks, coverage and bindings against the stored dataset, and verifies
        # that dataset as a current content-addressed build artifact.
        lineage = load_task_set_lineage_bindings(project.artifacts, completed.task_set.artifact_id)
    except (ArtifactStoreError, FileNotFoundError, ValueError) as exc:
        raise RouterCompositionError(
            "completed project build artifacts no longer load against their stored content; "
            f"rebuild the project before optimizing a router over it: {exc}"
        ) from exc
    if (
        artifact_input(trace_manifest) != completed.trace_dataset
        or artifact_input(task_manifest) != completed.task_set
    ):
        raise RouterCompositionError(
            "completed project build artifact manifest changed; rebuild the project before "
            "optimizing a router over it"
        )
    if task_manifest.inputs != (completed.trace_dataset,):
        raise RouterCompositionError(
            "completed project build task set does not bind its completed trace dataset"
        )
    if any(task.origin == "authored" for task in loaded.tasks):
        raise RouterCompositionError(
            "router optimization replays mined scenarios only, and this build selects authored "
            "scenarios; restore the mined set with select_authored_scenarios(project, ()) first"
        )
    trace_dataset = _replayed_trace_dataset(
        project, normalized, completed, review, created_at=created_at
    )
    return VerifiedProjectBuild(
        completed=completed,
        review=review,
        trace_dataset=trace_dataset,
        task_set=loaded.task_set,
        tasks=loaded.tasks,
        lineage=lineage,
    )


def _completed_build_review(
    project: ProjectStore,
    completed: ProjectBuildArtifacts,
) -> BuildReviewReadiness:
    """Load the project's build-review contract and require it to name the completed build.

    Args:
        project: Project whose mutable review state holds the build handoff.
        completed: Exact build pointers selected in project configuration.

    Returns:
        The validated review contract, whose identity binds its own complete content.

    Raises:
        RouterCompositionError: The review is absent, invalid, or names other artifacts.
    """
    review_state = project.read_review()
    if not isinstance(review_state, dict) or "build_review" not in review_state:
        raise RouterCompositionError("completed project build has no build review contract")
    try:
        review = BuildReviewReadiness.model_validate(review_state["build_review"])
    except ValueError as exc:
        raise RouterCompositionError("completed project build review is invalid") from exc
    if review.trace_dataset != completed.trace_dataset or review.task_set != completed.task_set:
        raise RouterCompositionError("completed project build review names different artifacts")
    return review


def _replayed_trace_dataset(
    project: ProjectStore,
    normalized: TraceNormalizationResult,
    completed: ProjectBuildArtifacts,
    review: BuildReviewReadiness,
    *,
    created_at: datetime,
) -> PersistedTraceDataset:
    """Require this composition's canonical traces to replay the stored trace dataset exactly.

    Normalization persistence is deterministic and provider-free, so the supplied traces can still
    be held to the stored evidence even though mining cannot. Replaying under the completed
    dataset's own identity makes the store compare envelope and payload bytes against that exact
    artifact, so a divergent corpus is refused here instead of silently landing a second dataset
    beside the build.

    Args:
        project: Project owning the immutable trace-dataset artifact.
        normalized: Canonical traces supplied to this composition.
        completed: Exact build pointers selected in project configuration.
        review: Validated review contract naming the producer revision the build used.
        created_at: Materialization time required by normalization persistence. The stored
            artifact's own completion time is adopted on every replay.

    Returns:
        The stored dataset envelope and manifest, proven to hold these exact canonical traces.

    Raises:
        RouterCompositionError: The supplied traces do not reproduce the stored dataset.
    """
    refusal = (
        "router composition traces differ from the completed project build; optimize over the "
        "traces that build used, or run exp build again over the current ones"
    )
    try:
        dataset = persist_trace_dataset(
            normalized,
            project.artifacts,
            created_at=created_at,
            code_revision=review.code_revision,
            dataset_id=completed.trace_dataset.artifact_id,
        )
    except (ArtifactStoreError, ValueError) as exc:
        raise RouterCompositionError(f"{refusal}: {exc}") from exc
    if artifact_input(dataset.manifest) != completed.trace_dataset:
        raise RouterCompositionError(refusal)
    return dataset


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
