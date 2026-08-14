"""Direct local composition from normalized trace evidence to immutable representative tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    ContractModel,
    SourceIdentity,
    stable_id,
)
from wmo.common.project import ArtifactStore, ProjectConfig, ProjectStore, artifact_input
from wmo.common.tasks import TaskSet
from wmo.simulation.ingest.dataset import PersistedTraceDataset, persist_trace_dataset
from wmo.simulation.ingest.otlp import TraceNormalizationResult
from wmo.simulation.mining.descriptors import DescriptorEmbedder, HashingDescriptorEmbedder
from wmo.simulation.mining.service import MiningSpec, TaskMiningResult, mine_tasks, persist_task_set


@dataclass(frozen=True)
class TaskSetBuild:
    """Completed canonical trace-dataset and task-set artifacts from one normalized input.

    Args:
        trace_dataset: Immutable raw-source-normalized evidence and manifest.
        mining: Deterministic representative-task selection evidence.
        task_set: Immutable task-set artifact derived only from the trace-dataset manifest.
    """

    trace_dataset: PersistedTraceDataset
    mining: TaskMiningResult
    task_set: TaskSet


class BuildReviewReadiness(ContractModel):
    """Manifest-bound provider-free handoff from deterministic mining to rubric review."""

    schema_version: int = 1
    readiness_id: ArtifactId
    status: Literal["proposals_pending"] = "proposals_pending"
    trace_dataset: ArtifactInput
    task_set: ArtifactInput
    project_config: ProjectConfig
    source: SourceIdentity
    mining_spec: MiningSpec
    descriptor_embedder: Literal["hashing-descriptor-v1"] = "hashing-descriptor-v1"
    descriptor_dimensions: int = Field(ge=8)
    code_revision: str = Field(min_length=1, max_length=256)
    paid_calls_made: Literal[0] = 0

    @model_validator(mode="after")
    def _require_content_identity(self) -> BuildReviewReadiness:
        """Reject readiness state whose ID does not bind its complete deterministic content."""
        expected = _build_review_id(self.model_dump(mode="json", exclude={"readiness_id"}))
        if self.readiness_id != expected:
            raise ValueError("build review readiness ID differs from its complete binding")
        return self


@dataclass(frozen=True)
class ProjectBuild:
    """Completed local artifacts and their explicit pending-review handoff.

    Args:
        artifacts: Deterministic trace and representative-task artifacts.
        review: Local mutable handoff proving no rubric proposal or judge call occurred.
    """

    artifacts: TaskSetBuild
    review: BuildReviewReadiness


def build_task_set(
    normalized: TraceNormalizationResult,
    store: ArtifactStore,
    *,
    created_at: datetime,
    code_revision: str,
    mining_spec: MiningSpec | None = None,
    embedder: DescriptorEmbedder | None = None,
) -> TaskSetBuild:
    """Persist normalized evidence, mine representatives, and persist a dependent task set.

    This is the supported no-network Python composition path. It intentionally accepts a
    pre-normalized result rather than a file path, so the only raw-source read remains in one
    selected canonical OTLP or PostHog loader.

    Args:
        normalized: Canonical traces and explicit validation exclusions from one source loader.
        store: Project-local immutable artifact store for both completed artifacts.
        created_at: Shared completion timestamp for this local build.
        code_revision: Exact WMO revision producing the artifacts.
        mining_spec: Optional representative selection controls, defaulting to 50 fit and 20 held
            out tasks.
        embedder: Explicit descriptor embedder. The deterministic local hashing embedder is used
            when omitted.

    Returns:
        Immutable trace-dataset and task-set artifacts plus the mining evidence between them.
    """
    trace_dataset = persist_trace_dataset(
        normalized,
        store,
        created_at=created_at,
        code_revision=code_revision,
    )
    mining = mine_tasks(
        trace_dataset.traces,
        mining_spec,
        embedder=embedder or HashingDescriptorEmbedder(),
        input_trace_count=len(trace_dataset.traces) + trace_dataset.dataset.invalid_trace_count,
        invalid_trace_count=trace_dataset.dataset.invalid_trace_count,
    )
    dataset_input = artifact_input(trace_dataset.manifest)
    task_set = persist_task_set(
        mining,
        store,
        task_set_id=_task_set_id(dataset_input, mining),
        created_at=created_at,
        code_revision=code_revision,
        inputs=(dataset_input,),
    )
    return TaskSetBuild(trace_dataset=trace_dataset, mining=mining, task_set=task_set)


def build_project(
    normalized: TraceNormalizationResult,
    store: ProjectStore,
    *,
    created_at: datetime,
    code_revision: str,
    mining_spec: MiningSpec | None = None,
    embedder: DescriptorEmbedder | None = None,
) -> ProjectBuild:
    """Build local evidence and stop before any model proposal or judge operation.

    Args:
        normalized: Canonical local traces from one explicit loader.
        store: Initialized project store that owns artifacts and review state.
        created_at: Completion time used for newly written immutable artifacts.
        code_revision: Exact WMO revision producing the artifacts.
        mining_spec: Optional deterministic representative-selection controls.
        embedder: Optional explicit descriptor embedder.

    Returns:
        Completed artifacts and a local ``proposals_pending`` review handoff.

    Raises:
        ValueError: Existing review state binds a different deterministic build.
    """
    resolved_spec = mining_spec or MiningSpec()
    resolved_embedder = embedder or HashingDescriptorEmbedder()
    if not isinstance(resolved_embedder, HashingDescriptorEmbedder):
        raise ValueError(
            "the provider-free project build requires HashingDescriptorEmbedder; use "
            "build_task_set for an explicitly configured external descriptor embedder"
        )
    artifacts = build_task_set(
        normalized,
        store.artifacts,
        created_at=created_at,
        code_revision=code_revision,
        mining_spec=resolved_spec,
        embedder=resolved_embedder,
    )
    trace_stored = store.artifacts.read(artifacts.trace_dataset.dataset.dataset_id)
    task_stored = store.artifacts.read(artifacts.task_set.task_set_id)
    if trace_stored.manifest != artifacts.trace_dataset.manifest:
        raise ValueError("loaded trace dataset manifest differs from the completed build")
    trace_input = artifact_input(trace_stored.manifest)
    task_input = artifact_input(task_stored.manifest)
    if artifacts.task_set.inputs != (trace_input,):
        raise ValueError("loaded task set does not bind the exact trace dataset manifest")
    project_config = store.load_project().model_copy(update={"build": None})
    source = artifacts.trace_dataset.dataset.source
    if source is None:
        raise ValueError("completed trace dataset has no immutable source identity")
    descriptor_dimensions = len(artifacts.mining.analysis.candidates[0].vector)
    review_binding = {
        "schema_version": 1,
        "status": "proposals_pending",
        "trace_dataset": trace_input.model_dump(mode="json"),
        "task_set": task_input.model_dump(mode="json"),
        "project_config": project_config.model_dump(mode="json"),
        "source": source.model_dump(mode="json"),
        "mining_spec": resolved_spec.model_dump(mode="json"),
        "descriptor_embedder": "hashing-descriptor-v1",
        "descriptor_dimensions": descriptor_dimensions,
        "code_revision": code_revision,
        "paid_calls_made": 0,
    }
    review = BuildReviewReadiness(
        readiness_id=_build_review_id(review_binding),
        trace_dataset=trace_input,
        task_set=task_input,
        project_config=project_config,
        source=source,
        mining_spec=resolved_spec,
        descriptor_embedder="hashing-descriptor-v1",
        descriptor_dimensions=descriptor_dimensions,
        code_revision=code_revision,
    )
    current = store.read_review()
    if current is None:
        root_review: dict[str, object] = {}
    elif isinstance(current, dict):
        root_review = dict(current)
    else:
        raise ValueError("review.json must be an object before build readiness can be recorded")
    prior = root_review.get("build_review")
    serialized = review.model_dump(mode="json")
    if prior is not None:
        try:
            existing = BuildReviewReadiness.model_validate(prior)
        except ValueError as exc:
            raise ValueError("review.json contains invalid build readiness") from exc
        if existing != review:
            raise ValueError("review.json already binds a different completed build")
    root_review["build_review"] = serialized
    store.write_review(root_review)
    return ProjectBuild(artifacts=artifacts, review=review)


def _task_set_id(dataset_input: ArtifactInput, mining: TaskMiningResult) -> str:
    """Return the content-addressed task-set ID for one immutable dataset and selection result."""
    return stable_id(
        "task-set",
        {
            "trace_dataset": dataset_input.model_dump(mode="json"),
            "tasks": [task.model_dump(mode="json") for task in mining.tasks],
            "coverage": mining.coverage.model_dump(mode="json"),
        },
    )


def _build_review_id(binding: dict[str, object]) -> str:
    """Return the content ID for one complete manifest-bound build review handoff."""
    return stable_id("build-review", binding)
