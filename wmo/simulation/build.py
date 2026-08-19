"""Direct local composition from normalized trace evidence to immutable representative tasks."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    ContractModel,
    JsonObject,
    SourceIdentity,
    stable_id,
)
from wmo.common.project import (
    ArtifactStore,
    ProjectBuildArtifacts,
    ProjectConfig,
    ProjectProviderFreeStage,
    ProjectStore,
    ProjectStoreError,
    ProjectTracePreparationSettings,
    artifact_input,
    coordinate_completed_build_selection,
)
from wmo.common.project.project import require_durable_source_id
from wmo.common.release_revision import installed_release_revision
from wmo.common.tasks import TaskSet
from wmo.simulation.ingest.dataset import (
    PersistedTraceDataset,
    persist_trace_dataset,
)
from wmo.simulation.ingest.otlp import TraceNormalizationResult
from wmo.simulation.ingest.sources import load_trace_source
from wmo.simulation.mining.bindings import (
    bindings_for_mining,
    task_set_content_id,
)
from wmo.simulation.mining.descriptors import DescriptorEmbedder, HashingDescriptorEmbedder
from wmo.simulation.mining.service import MiningSpec, TaskMiningResult, mine_tasks, persist_task_set

_BUILD_SCOPED_REVIEW_KEYS = (
    "manual_judge",
    "rubric_review",
    "human_score_history",
    "human_score_submissions",
)
_MINIMUM_PROVIDER_FREE_TRACE_COUNT = 100
_MAXIMUM_PROVIDER_FREE_TRACE_COUNT = 1000


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


def prepare_project_traces(
    project: str,
    trace_file: Path,
    *,
    root: Path,
    source_id: str,
    settings: ProjectTracePreparationSettings,
) -> ProjectProviderFreeStage:
    """Prepare and select provider-free Project evidence from one acquired trace file.

    Args:
        project: Safe local Project identifier below ``root/projects``.
        trace_file: Worker-local path to the immutable acquired source bytes.
        root: Local WMO artifact root.
        source_id: Stable caller-owned source label that remains valid after the worker exits.
        settings: Declared source kind and deterministic representative-task controls.

    Returns:
        The exact selected trace and task manifest pointers for the completed provider-free stage.

    Raises:
        ValueError: The source label, source kind, normalized trace count, or evidence is invalid.
        ProjectStoreError: Existing Project settings or a selected stage conflict with this request.
    """
    durable_source_id = require_durable_source_id(source_id)
    normalized = load_trace_source(
        settings.source_kind,
        Path(trace_file),
        source_id=durable_source_id,
    )
    valid_trace_count = len(normalized.traces)
    if (
        not _MINIMUM_PROVIDER_FREE_TRACE_COUNT
        <= valid_trace_count
        <= (_MAXIMUM_PROVIDER_FREE_TRACE_COUNT)
    ):
        raise ValueError(
            "provider-free trace preparation requires "
            f"{_MINIMUM_PROVIDER_FREE_TRACE_COUNT} to {_MAXIMUM_PROVIDER_FREE_TRACE_COUNT} valid "
            f"normalized traces; got {valid_trace_count} after excluding "
            f"{len(normalized.issues)} records"
        )
    code_revision = installed_release_revision()
    store = _initialize_provider_free_project(root, project, settings)
    completed = build_project(
        normalized,
        store,
        created_at=datetime.now(UTC),
        code_revision=code_revision,
        mining_spec=MiningSpec(
            fit_task_budget=settings.fit_task_budget,
            held_out_task_budget=settings.held_out_task_budget,
        ),
        embedder=HashingDescriptorEmbedder(dimensions=settings.descriptor_dimensions),
    )
    trace_dataset = artifact_input(completed.artifacts.trace_dataset.manifest)
    task_set = artifact_input(
        store.artifacts.read(completed.artifacts.task_set.task_set_id).manifest
    )
    stage = ProjectProviderFreeStage(
        trace_dataset=trace_dataset,
        task_set=task_set,
    )
    store.bind_provider_free_stage(stage)
    return stage


def load_project_provider_free_stage(
    project: str,
    *,
    root: Path,
) -> ProjectProviderFreeStage:
    """Load and verify a Project's selected provider-free stage after process restart.

    Args:
        project: Safe local Project identifier below ``root/projects``.
        root: Local WMO artifact root.

    Returns:
        The selected provider-free stage with exact verified manifest pointers.

    Raises:
        ProjectStoreError: The Project has no selected stage or its pointer graph is invalid.
    """
    store = ProjectStore(root, project)
    stage = store.load_project().provider_free_stage
    if stage is None:
        raise ProjectStoreError(
            "project has no completed provider-free stage; call prepare_project_traces first"
        )
    store.bind_provider_free_stage(stage)
    return stage


def _initialize_provider_free_project(
    root: Path,
    project: str,
    settings: ProjectTracePreparationSettings,
) -> ProjectStore:
    """Initialize a minimal Project or verify its immutable trace preparation settings.

    Args:
        root: Local WMO artifact root.
        project: Safe local Project identifier.
        settings: Provider-free settings selected before evidence construction.

    Returns:
        Project store containing the requested provider-free settings.

    Raises:
        ProjectStoreError: Existing trace preparation settings differ from the request.
    """
    store = ProjectStore(root, project)
    proposed = ProjectConfig(
        project_id=project,
        trace_preparation=settings,
        retrieval=None,
        budgets=None,
    )
    if not store.paths.project_toml.exists():
        try:
            store.initialize(proposed)
        except ProjectStoreError:
            existing = store.load_project()
            if existing.trace_preparation != settings:
                raise
    existing = store.load_project()
    if existing.trace_preparation != settings:
        raise ProjectStoreError(
            "project already has different provider-free trace preparation settings"
        )
    return store


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
    selected canonical source loader.

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
    project_config = _provider_free_review_config(store.load_project())
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
    return ProjectBuild(artifacts=artifacts, review=review)


def _provider_free_review_config(config: ProjectConfig) -> ProjectConfig:
    """Remove completed hosted-stage pointers from deterministic build-review identity.

    Args:
        config: Current Project configuration before or after hosted stage selection.

    Returns:
        Valid Project configuration retaining setup while omitting provider-backed results.
    """
    return config.model_copy(
        update={
            "build": None,
            "build_spend_ledger": None,
            "hosted_judge": None,
            "router_policy": None,
            "router_report": None,
        }
    )


def select_completed_build(
    store: ProjectStore,
    build: ProjectBuildArtifacts,
    review: BuildReviewReadiness,
) -> None:
    """Select a completed graph before advancing its recoverable review handoff.

    Official build writers share one cross-process coordination lock. The immutable build is
    selected first, then review readiness advances. If execution stops between those writes, an
    exact restart verifies the selected graph and repairs the review handoff without provider work.

    Args:
        store: Project store receiving the completed-build selection.
        build: Verified immutable trace, task, RAG, and world-model graph.
        review: Exact readiness record derived from the graph's trace and task artifacts.

    Raises:
        ValueError: The proposed graph and review name different trace or task artifacts.
    """
    if build.trace_dataset != review.trace_dataset or build.task_set != review.task_set:
        raise ValueError("completed build does not match proposed build review")
    with _build_review_coordination(store):
        store.bind_completed_build(build)
        select_build_review(store, review)


@contextmanager
def coordinate_selected_build_review(
    store: ProjectStore,
    *,
    trace_dataset: ArtifactInput,
    task_set: ArtifactInput,
) -> Iterator[BuildReviewReadiness]:
    """Hold the build-review transaction while a dependent review mutation completes.

    Args:
        store: Project store whose selected build and review state must remain aligned.
        trace_dataset: Exact trace artifact required by the dependent mutation.
        task_set: Exact task artifact required by the dependent mutation.

    Yields:
        The verified selected-build readiness while replacement selection is excluded.

    Raises:
        ValueError: Review state is absent, malformed, stale, or names different evidence.
    """
    with _build_review_coordination(store):
        current = store.read_review()
        if not isinstance(current, dict) or "build_review" not in current:
            raise ValueError("project has no completed build review")
        try:
            review = BuildReviewReadiness.model_validate(current["build_review"])
        except ValueError as exc:
            raise ValueError("project build review is invalid") from exc
        if review.trace_dataset != trace_dataset or review.task_set != task_set:
            raise ValueError("manual judge evidence differs from the selected build review")
        if not _completed_build_matches_review(store, review):
            raise ValueError("selected completed build and review state are not synchronized")
        yield review


@contextmanager
def _build_review_coordination(store: ProjectStore) -> Iterator[None]:
    """Serialize completed-build selection with dependent mutable review transactions.

    Args:
        store: Project store whose build and review state share the coordination boundary.

    Yields:
        None while the project-local cross-process lock is held.
    """
    with coordinate_completed_build_selection(store):
        yield


def select_build_review(store: ProjectStore, review: BuildReviewReadiness) -> None:
    """Advance review readiness after the exact completed build has been selected.

    The completed-build selection is written first. If execution stops before this function
    finishes, a later identical build verifies the selected graph and repairs the review pointer.

    Args:
        store: Project store owning the selected build and mutable review handoff.
        review: Exact readiness record derived from the selected trace and task artifacts.

    Raises:
        ValueError: The selected completed build does not match the proposed review.
    """
    if not _completed_build_matches_review(store, review):
        raise ValueError("selected completed build does not match proposed build review")

    def update(current: JsonValue | None) -> JsonObject:
        """Advance build review and discard judge state bound to replaced evidence.

        Args:
            current: Current complete review JSON value.

        Returns:
            Updated object with the selected build-review handoff and compatible judge state.
        """
        root_review = _review_object(current)
        prior_value = root_review.get("build_review")
        try:
            prior = (
                BuildReviewReadiness.model_validate(prior_value)
                if prior_value is not None
                else None
            )
        except ValueError as exc:
            raise ValueError("review.json contains invalid build readiness") from exc
        return _advance_build_review(root_review, prior=prior, review=review)

    store.update_review(update)


def _advance_build_review(
    root_review: JsonObject,
    *,
    prior: BuildReviewReadiness | None,
    review: BuildReviewReadiness,
) -> JsonObject:
    """Advance the build binding and remove only namespaces owned by replaced evidence.

    Args:
        root_review: Mutable complete project review object.
        prior: Previous build binding, or None before the first build.
        review: Selected build binding that will replace it.

    Returns:
        Review state with the selected build binding, unrelated members preserved, and judge
        namespaces removed only when their prior build binding was replaced.
    """
    if prior != review:
        for key in _BUILD_SCOPED_REVIEW_KEYS:
            root_review.pop(key, None)
    root_review["build_review"] = review.model_dump(mode="json")
    return root_review


def _review_object(current: JsonValue | None) -> JsonObject:
    """Return mutable object-shaped review state.

    Args:
        current: Current parsed review value, or ``None`` before any review exists.

    Returns:
        Mutable copy of the current object-shaped review state.

    Raises:
        ValueError: Existing review state is not an object.
    """
    if current is None:
        return {}
    if isinstance(current, dict):
        return dict(current)
    raise ValueError("review.json must be an object before build readiness can be recorded")


def _completed_build_matches_review(store: ProjectStore, review: BuildReviewReadiness) -> bool:
    """Verify that the selected completed build authorizes advancing its review handoff.

    Args:
        store: Project store containing the mutable completed-build selection.
        review: Existing review handoff that a subsequent build would replace.

    Returns:
        Whether the selected build and verified immutable manifests match the review handoff.
    """
    completed = store.load_project().build
    if completed is None:
        return False
    if completed.trace_dataset != review.trace_dataset or completed.task_set != review.task_set:
        return False
    expected_types = {
        "trace_dataset": "trace-dataset",
        "task_set": "task-set",
        "serving_rag": "trace-rag-index",
        "fit_rag": "trace-rag-index",
        "world_model": "grounded-world-model",
    }
    manifests = {}
    for field_name, artifact_type in expected_types.items():
        pointer = getattr(completed, field_name)
        stored = store.artifacts.read(pointer.artifact_id)
        if stored.manifest.artifact_type != artifact_type:
            return False
        if artifact_input(stored.manifest) != pointer:
            return False
        manifests[field_name] = stored.manifest
    expected_inputs = {
        "task_set": (completed.trace_dataset,),
        "serving_rag": (completed.trace_dataset,),
        "fit_rag": (completed.trace_dataset,),
        "world_model": (completed.serving_rag,),
    }
    for field_name, inputs in expected_inputs.items():
        if manifests[field_name].inputs != inputs:
            return False
    return True


def _task_set_id(dataset_input: ArtifactInput, mining: TaskMiningResult) -> str:
    """Return the content-addressed task-set ID for one immutable dataset and selection result."""
    return task_set_content_id(
        dataset_input,
        mining.tasks,
        mining.coverage,
        bindings_for_mining(mining),
    )


def _build_review_id(binding: dict[str, object]) -> str:
    """Return the content ID for one complete manifest-bound build review handoff."""
    return stable_id("build-review", binding)
