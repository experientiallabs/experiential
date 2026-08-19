"""Behavioral tests for explicit manual judge setup and calibration."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, Thread
from typing import Literal, cast

import pytest

import wmo.optimize.router.judging.service as manual_judge_workflow
import wmo.simulation.build as simulation_build
from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    JsonObject,
    JsonValue,
    SourceIdentity,
    sha256_json,
    stable_id,
)
from wmo.common.judging import (
    HumanScore,
    HumanScoreHistory,
    HumanScoreReview,
    PromptDefinition,
    RubricDimension,
    RubricReview,
    scored_axis,
)
from wmo.common.judging.judgment import Judgment
from wmo.common.models import (
    AssistantAction,
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelRequest,
    ModelResponse,
    ModelRoles,
    ModelSnapshot,
    OperationEconomics,
    PricingSource,
)
from wmo.common.project import (
    ProjectBuildArtifacts,
    ProjectConfig,
    ProjectStore,
    ProjectStoreError,
    artifact_input,
)
from wmo.common.traces import Trace, TraceOutcome, TraceSource, TraceSpan
from wmo.optimize.router.judging.contracts import (
    JudgePromptTemplate,
    JudgeScoreProjection,
    ManualJudgeLabel,
    ManualJudgeReviewState,
    ManualJudgeSetupArtifact,
    judge_feedback_schema,
)
from wmo.optimize.router.judging.labels import calibration_sample_digest, read_label_draft
from wmo.optimize.router.judging.review import read_trace_reviews
from wmo.optimize.router.judging.service import (
    ManualJudgeError,
    calibrate_manual_judge,
    calibration_sample,
    commit_manual_judge_setup,
    estimate_manual_judge_budget,
    prepare_manual_judge_calibration,
    prepare_manual_judge_setup,
)
from wmo.optimize.router.judging.template_bind import DEFAULT_JUDGE_TEMPLATE
from wmo.runtime.models.registry import ResolvedModel, RuntimeModelCatalog
from wmo.simulation.build import ProjectBuild, build_project, select_completed_build
from wmo.simulation.ingest.otlp import TraceNormalizationResult
from wmo.simulation.mining.service import MiningSpec

_TIME = datetime(2026, 8, 13, tzinfo=UTC)
_DIGEST = "a" * 64
_FeedbackShape = Literal["scalar", "boolean", "categorical", "pairwise"]


def _wide_axes() -> tuple[RubricDimension, ...]:
    """Return a 0-5 task-success axis for custom projection contracts."""
    return (
        scored_axis(
            "task-success",
            "Task success",
            "Whether the customer received a correct outcome.",
        ),
    )


class _JudgeClient:
    """Return deterministic scalar scores while recording every provider call."""

    def __init__(self, model: ModelSnapshot) -> None:
        """Bind the exact configured model identity for deterministic responses.

        Args:
            model: Frozen configured judge snapshot.
        """
        self.model = model
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return one schema-valid score for the request.

        Args:
            request: Structured LM judge request.

        Returns:
            Deterministic model response with no observed provider cost.
        """
        self.requests.append(request)
        return ModelResponse(
            output=AssistantAction(
                content=json.dumps(
                    {
                        "dimensions": [
                            {
                                "dimension_id": "task-success",
                                "raw_score": 1,
                                "rationale": "The trace shows the task was handled.",
                            }
                        ]
                    }
                )
            ),
            model=self.model,
            economics=OperationEconomics(),
        )


class _RuntimeCatalog:
    """Inject one fake resolved judge while counting credential-resolution boundaries."""

    def __init__(self, resolved: ResolvedModel) -> None:
        """Store the resolved fake judge.

        Args:
            resolved: Exact configured model and fake client.
        """
        self.resolved = resolved
        self.preflight_calls = 0

    def preflight(self, alias: str) -> ResolvedModel:
        """Return the fake judge only for its configured alias.

        Args:
            alias: Alias requested by calibration.

        Returns:
            Injected resolved judge.
        """
        self.preflight_calls += 1
        assert alias == self.resolved.alias
        return self.resolved


class _StructuredJudgeClient:
    """Return one selected finalized feedback shape and record exact visible requests."""

    def __init__(
        self,
        model: ModelSnapshot,
        shape: _FeedbackShape,
        *,
        fail_after: int | None = None,
    ) -> None:
        """Bind the response shape and optional deterministic interruption boundary.

        Args:
            model: Exact configured judge snapshot.
            shape: Scalar, boolean, categorical, or pairwise response shape.
            fail_after: Raise before this zero-based provider dispatch when set.
        """
        self.model = model
        self.shape = shape
        self.fail_after = fail_after
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return schema-valid evidence or simulate an interrupted provider run.

        Args:
            request: Exact finalized judge request.

        Returns:
            Deterministic structured response for the configured shape.

        Raises:
            RuntimeError: The configured interruption boundary is reached.
        """
        if self.fail_after is not None and len(self.requests) >= self.fail_after:
            raise RuntimeError("simulated provider interruption")
        self.requests.append(request)
        common = {
            "dimension_id": "task-success",
            "rationale": "Structured evidence supports the verdict.",
        }
        if self.shape == "scalar":
            dimension = {**common, "raw_score": 1}
        elif self.shape == "boolean":
            dimension = {**common, "passed": True}
        elif self.shape == "categorical":
            dimension = {**common, "category": "good"}
        else:
            dimension = {**common, "winner": "winner_a"}
        return ModelResponse(
            output=AssistantAction(content=json.dumps({"dimensions": [dimension]})),
            model=self.model,
            economics=OperationEconomics(),
        )


def _model() -> ModelSnapshot:
    """Return the exact snapshot produced by the local catalog fixture."""
    return ModelSnapshot(
        provider="openai",
        model_id="judge-model",
        capabilities_sha256=_capabilities_digest(),
        connection_sha256=_connection_digest(),
    )


def _capabilities_digest() -> str:
    """Return the static capability digest used by the fixture catalog."""
    return sha256_json(ModelCapabilities())


def _connection_digest() -> str:
    """Return the static connection digest used by the fixture catalog."""
    return ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY").identity_sha256()


def _catalog() -> ModelCatalog:
    """Return a secret-free catalog with one explicit configured judge alias."""
    return ModelCatalog(
        connections={
            "openai-main": ConnectionConfig(
                provider="openai",
                api_key_env="OPENAI_API_KEY",
            )
        },
        models={
            "judge-main": ModelRecord(
                connection="openai-main",
                model="judge-model",
                capabilities=ModelCapabilities(),
            )
        },
        roles=ModelRoles(judge="judge-main"),
    )


def _trace(index: int) -> Trace:
    """Build one distinct real normalized trace with captured model output.

    Args:
        index: Unique trace and lineage fixture index.

    Returns:
        Complete normalized successful production trace.
    """
    started_at = _TIME + timedelta(minutes=index)
    return Trace(
        trace_id=f"trace-{index}",
        conversation_id=f"conversation-{index}",
        task=f"Resolve support case {index}",
        spans=(
            TraceSpan(
                span_id=f"span-{index}",
                name="agent.model_call",
                started_at=started_at,
                ended_at=started_at + timedelta(seconds=1),
                attributes={"output": f"Resolved case {index}."},
                model=_model(),
            ),
        ),
        outcome=TraceOutcome(status="success"),
        source=TraceSource(
            identity=SourceIdentity(kind="otlp", source_id="fixture.otlp", sha256=_DIGEST),
            semantic_convention_version="1.37.0",
        ),
    )


def _unique_task_trace(index: int) -> Trace:
    """Build a trace with a distinct alphabetic request descriptor.

    Args:
        index: Unique trace fixture index.

    Returns:
        Trace whose task tokens do not collapse numeric-only descriptors.
    """
    suffix = "".join(chr(ord("a") + digit) for digit in divmod(index, 26))
    return _trace(index).model_copy(update={"task": f"Handle unique request {suffix}"})


def _persist_grounded_build(
    store: ProjectStore,
    built: ProjectBuild,
    *,
    select: bool,
) -> ProjectBuildArtifacts:
    """Persist a complete immutable graph and optionally select it for judge mutations.

    Args:
        store: Initialized project receiving the grounded graph.
        built: Deterministic trace and task build awaiting grounded artifacts.
        select: Whether to bind the graph and advance its build review immediately.

    Returns:
        Complete grounded artifact pointers, selected when requested.
    """
    trace_input = artifact_input(built.artifacts.trace_dataset.manifest)
    created_at = built.artifacts.trace_dataset.dataset.created_at
    revision = built.review.code_revision
    serving_id = stable_id(
        "test-serving-rag",
        {"trace_dataset": trace_input.model_dump(mode="json")},
    )
    serving_manifest = store.artifacts.write_json(
        artifact_id=serving_id,
        artifact_type="trace-rag-index",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=created_at,
            inputs=(trace_input,),
            code_revision=revision,
        ),
        files={"fixture.json": {"partition": "serving"}},
    )
    fit_id = stable_id(
        "test-fit-rag",
        {"trace_dataset": trace_input.model_dump(mode="json")},
    )
    fit_manifest = store.artifacts.write_json(
        artifact_id=fit_id,
        artifact_type="trace-rag-index",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=created_at,
            inputs=(trace_input,),
            code_revision=revision,
        ),
        files={"fixture.json": {"partition": "fit"}},
    )
    serving_input = artifact_input(serving_manifest)
    world_id = stable_id(
        "test-grounded-world-model",
        {"serving_rag": serving_input.model_dump(mode="json")},
    )
    world_manifest = store.artifacts.write_json(
        artifact_id=world_id,
        artifact_type="grounded-world-model",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=created_at,
            inputs=(serving_input,),
            code_revision=revision,
        ),
        files={"fixture.json": {"model_alias": "judge-main"}},
    )
    completed = ProjectBuildArtifacts(
        trace_dataset=trace_input,
        task_set=built.review.task_set,
        serving_rag=serving_input,
        fit_rag=artifact_input(fit_manifest),
        world_model=artifact_input(world_manifest),
    )
    if select:
        select_completed_build(store, completed, built.review)
    return completed


def _built_store(tmp_path: Path) -> ProjectStore:
    """Create a completed build with three fit lineages and one held-out lineage.

    Args:
        tmp_path: Isolated test directory.

    Returns:
        Initialized project store with deterministic build readiness.
    """
    store = ProjectStore(tmp_path / ".wmo", "support")
    store.initialize(ProjectConfig(project_id="support"))
    built = build_project(
        TraceNormalizationResult(traces=tuple(_trace(index) for index in range(100)), issues=()),
        store,
        created_at=_TIME,
        code_revision="test-revision",
        mining_spec=MiningSpec(fit_task_budget=3, held_out_task_budget=1),
    )
    _persist_grounded_build(store, built, select=True)
    return store


def _unpaired_trace_store(tmp_path: Path) -> ProjectStore:
    """Create a completed build whose selected tasks each have one source trace.

    Args:
        tmp_path: Isolated test directory.

    Returns:
        Initialized project store containing only distinct task sources.
    """
    store = ProjectStore(tmp_path / ".wmo", "support")
    store.initialize(ProjectConfig(project_id="support"))
    built = build_project(
        TraceNormalizationResult(
            traces=tuple(_unique_task_trace(index) for index in range(100)), issues=()
        ),
        store,
        created_at=_TIME,
        code_revision="test-revision",
        mining_spec=MiningSpec(
            fit_task_budget=3,
            held_out_task_budget=1,
            semantic_duplicate_threshold=1.0,
        ),
    )
    _persist_grounded_build(store, built, select=True)
    return store


def _setup(store: ProjectStore) -> ManualJudgeSetupArtifact:
    """Prepare and explicitly confirm the default scalar judge setup.

    Args:
        store: Completed build fixture.

    Returns:
        Persisted manual judge setup.
    """
    plan = prepare_manual_judge_setup(
        store,
        _catalog(),
        created_at=_TIME,
        code_revision="test-revision",
    )
    return commit_manual_judge_setup(store, plan, confirmed=True)


def _template(shape: _FeedbackShape) -> JudgePromptTemplate:
    """Build one fully explicit supported structured-feedback setup contract.

    Args:
        shape: Scalar, boolean, categorical, or pairwise feedback shape.

    Returns:
        Versioned prompt, variable mapping, schema, and score projection.
    """
    prompt = PromptDefinition.from_text("custom-judge-v1", "Follow the saved contract exactly.")
    if shape == "boolean":
        projection = JudgeScoreProjection(boolean_scores={"false": 0, "true": 5})
    elif shape == "categorical":
        projection = JudgeScoreProjection(categorical_scores={"bad": 0, "good": 5})
    elif shape == "pairwise":
        projection = JudgeScoreProjection(
            pairwise_scores={"winner_a": 5, "winner_b": 0, "tie": 3},
            pairwise_aggregation="rounded_mean",
        )
    else:
        projection = JudgeScoreProjection()
    mapping: JsonObject = (
        {"rubric": "RULES_CUSTOM", "candidate_a": "OUTPUT_A", "candidate_b": "OUTPUT_B"}
        if shape == "pairwise"
        else {"rubric": "RULES_CUSTOM", "rollout": "TRACE_CUSTOM"}
    )
    return JudgePromptTemplate(
        response_shape=shape,
        prompt=prompt,
        variable_mapping=mapping,
        response_schema=judge_feedback_schema(
            shape,
            categories=tuple(sorted(projection.categorical_scores)),
        ),
        score_projection=projection,
    )


def _labels(store: ProjectStore) -> tuple[ManualJudgeLabel, ...]:
    """Return complete human labels for the frozen calibration sample.

    Args:
        store: Project with finalized manual judge setup.

    Returns:
        One score for every selected real trace.
    """
    plan = prepare_manual_judge_calibration(store, sample_size=3)
    return tuple(
        ManualJudgeLabel(
            trace_id=trace.trace_id,
            dimension_id="task-success",
            score=1,
        )
        for trace in plan.traces
    )


def test_setup_failure_is_read_only_and_setup_never_calls_a_model(tmp_path: Path) -> None:
    """Preview and refused confirmation leave review and immutable artifacts unchanged."""
    store = _built_store(tmp_path)
    before_review = store.read_review()
    before_artifacts = store.artifacts.list_ids()
    plan = prepare_manual_judge_setup(
        store,
        _catalog(),
        created_at=_TIME,
        code_revision="test-revision",
    )

    assert plan.previews
    assert [item.dimension_id for item in plan.dimensions] == ["task-success"]
    assert plan.dimensions[0].min_score == 0
    assert plan.dimensions[0].max_score == 1
    assert plan.dimensions[0].description == (
        "The agent successfully completed the task requested in the original user prompt"
    )
    assert store.read_review() == before_review
    assert store.artifacts.list_ids() == before_artifacts
    with pytest.raises(ManualJudgeError, match="explicit confirmation"):
        commit_manual_judge_setup(store, plan, confirmed=False)
    assert store.read_review() == before_review
    assert store.artifacts.list_ids() == before_artifacts


def test_build_replacement_crash_blocks_stale_judge_commit_and_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale judge setup cannot cross an interrupted replacement selection.

    Args:
        tmp_path: Isolated project and immutable artifact root.
        monkeypatch: Patch fixture controlling the bind-to-review interruption.
    """
    store = _built_store(tmp_path)
    selected_a = store.load_project().build
    assert selected_a is not None
    old_manifests = {
        pointer.artifact_id: store.artifacts.read(pointer.artifact_id).manifest
        for pointer in (
            selected_a.trace_dataset,
            selected_a.task_set,
            selected_a.serving_rag,
            selected_a.fit_rag,
            selected_a.world_model,
        )
    }
    stale_plan = prepare_manual_judge_setup(
        store,
        _catalog(),
        created_at=_TIME,
        code_revision="test-revision",
    )
    selected_setup = commit_manual_judge_setup(store, stale_plan, confirmed=True)
    selected_review = store.read_review()
    assert isinstance(selected_review, dict)
    assert selected_review["manual_judge"]["setup"]["artifact_id"] == selected_setup.setup_id
    assert "rubric_review" in selected_review
    replacement = build_project(
        TraceNormalizationResult(traces=tuple(_trace(index) for index in range(100)), issues=()),
        store,
        created_at=_TIME + timedelta(seconds=1),
        code_revision="replacement-revision",
        mining_spec=MiningSpec(fit_task_budget=3, held_out_task_budget=1),
    )
    completed_b = _persist_grounded_build(store, replacement, select=False)
    selection_paused = Event()
    release_selection = Event()
    judge_started = Event()
    builder_errors: list[BaseException] = []
    judge_errors: list[BaseException] = []
    original_select_review = simulation_build.select_build_review

    def interrupt_after_bind(*_args: object, **_kwargs: object) -> None:
        """Pause after selecting graph B, then simulate interruption before review B.

        Raises:
            RuntimeError: Always after the test releases the paused selection.
        """
        selection_paused.set()
        assert release_selection.wait(timeout=5)
        raise RuntimeError("injected interruption after completed-build selection")

    def run_builder() -> None:
        """Capture the expected interrupted replacement result."""
        try:
            select_completed_build(store, completed_b, replacement.review)
        except BaseException as exc:  # noqa: BLE001 - thread must report every terminal result
            builder_errors.append(exc)

    def run_stale_judge() -> None:
        """Capture the stale judge commit result without terminating the test process."""
        judge_started.set()
        try:
            commit_manual_judge_setup(store, stale_plan, confirmed=True)
        except BaseException as exc:  # noqa: BLE001 - thread must report every terminal result
            judge_errors.append(exc)

    monkeypatch.setattr(simulation_build, "select_build_review", interrupt_after_bind)
    builder = Thread(target=run_builder, daemon=True)
    builder.start()
    assert selection_paused.wait(timeout=5)
    assert store.load_project().build == completed_b
    stale_review = store.read_review()
    assert isinstance(stale_review, dict)
    assert stale_review["build_review"]["readiness_id"] == stale_plan.build.readiness_id

    judge = Thread(target=run_stale_judge, daemon=True)
    judge.start()
    assert judge_started.wait(timeout=5)
    judge.join(timeout=0.1)
    assert judge.is_alive()
    release_selection.set()
    builder.join(timeout=5)
    judge.join(timeout=5)
    assert not builder.is_alive()
    assert not judge.is_alive()
    assert len(builder_errors) == 1
    assert "injected interruption" in str(builder_errors[0])
    assert len(judge_errors) == 1
    assert isinstance(judge_errors[0], ManualJudgeError)
    assert "not synchronized" in str(judge_errors[0])
    assert store.load_project().build == completed_b
    assert store.read_review() == stale_review

    monkeypatch.setattr(simulation_build, "select_build_review", original_select_review)
    select_completed_build(store, completed_b, replacement.review)
    recovered_review = store.read_review()
    assert isinstance(recovered_review, dict)
    assert recovered_review["build_review"]["readiness_id"] == replacement.review.readiness_id
    for stale_key in (
        "manual_judge",
        "rubric_review",
        "human_score_history",
        "human_score_submissions",
    ):
        assert stale_key not in recovered_review
    for artifact_id, manifest in old_manifests.items():
        assert store.artifacts.read(artifact_id).manifest == manifest

    current_plan = prepare_manual_judge_setup(
        store,
        _catalog(),
        created_at=_TIME + timedelta(seconds=2),
        code_revision="replacement-revision",
    )
    current_setup = commit_manual_judge_setup(store, current_plan, confirmed=True)
    final_review = store.read_review()
    assert isinstance(final_review, dict)
    assert final_review["manual_judge"]["setup"]["artifact_id"] == current_setup.setup_id


def test_build_replacement_serializes_direct_rubric_writer_and_removes_stale_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A direct rubric writer finishes before selection cleanup and cannot restore A later.

    Args:
        tmp_path: Isolated project and immutable artifact root.
        monkeypatch: Patch fixture pausing the rubric transition inside both coordination locks.
    """
    store = _built_store(tmp_path)
    selected_a = store.load_project().build
    assert selected_a is not None
    plan = prepare_manual_judge_setup(
        store,
        _catalog(),
        created_at=_TIME,
        code_revision="test-revision",
    )
    stale = RubricReview.open(
        store,
        source_task_set_id=selected_a.task_set.artifact_id,
        code_revision="test-revision",
        clock=lambda: _TIME,
    )
    stale.replace_all(plan.dimensions)
    replacement = build_project(
        TraceNormalizationResult(traces=tuple(_trace(index) for index in range(100)), issues=()),
        store,
        created_at=_TIME + timedelta(seconds=1),
        code_revision="replacement-revision",
        mining_spec=MiningSpec(fit_task_budget=3, held_out_task_budget=1),
    )
    completed_b = _persist_grounded_build(store, replacement, select=False)
    writer_paused = Event()
    release_writer = Event()
    writer_errors: list[BaseException] = []
    selector_errors: list[BaseException] = []
    original_replace = RubricReview._replace

    def pause_replace(
        review: RubricReview,
        *,
        dimensions: tuple[RubricDimension, ...],
        event_kind: Literal["accept", "reject", "edit", "add", "replace_all", "order"],
        event_dimension_ids: tuple[str, ...],
        rejected_dimension_ids: tuple[str, ...] | None = None,
    ) -> None:
        """Pause one rubric mutation while it owns build and review coordination.

        Args:
            review: Working rubric service applying the locked transition.
            dimensions: Complete active rubric dimensions after the transition.
            event_kind: Exact transition kind recorded in the review history.
            event_dimension_ids: Dimension identities affected by the transition.
            rejected_dimension_ids: Optional complete rejected-dimension replacement.
        """
        writer_paused.set()
        assert release_writer.wait(timeout=5)
        original_replace(
            review,
            dimensions=dimensions,
            event_kind=event_kind,
            event_dimension_ids=event_dimension_ids,
            rejected_dimension_ids=rejected_dimension_ids,
        )

    def run_writer() -> None:
        """Record any unexpected direct rubric-writer failure."""
        try:
            stale.order(tuple(item.dimension_id for item in plan.dimensions))
        except BaseException as exc:  # noqa: BLE001 - thread reports its terminal result
            writer_errors.append(exc)

    def run_selector() -> None:
        """Select build B after the rubric writer releases coordination."""
        try:
            select_completed_build(store, completed_b, replacement.review)
        except BaseException as exc:  # noqa: BLE001 - thread reports its terminal result
            selector_errors.append(exc)

    monkeypatch.setattr(RubricReview, "_replace", pause_replace)
    writer = Thread(target=run_writer, daemon=True)
    writer.start()
    assert writer_paused.wait(timeout=5)
    selector = Thread(target=run_selector, daemon=True)
    selector.start()
    selector.join(timeout=0.1)
    assert selector.is_alive()
    release_writer.set()
    writer.join(timeout=5)
    selector.join(timeout=5)

    assert not writer.is_alive()
    assert not selector.is_alive()
    assert writer_errors == []
    assert selector_errors == []
    saved = store.read_review()
    assert isinstance(saved, dict)
    assert saved["build_review"]["readiness_id"] == replacement.review.readiness_id
    assert "rubric_review" not in saved
    with pytest.raises(ProjectStoreError, match="selected completed build"):
        stale.order(tuple(item.dimension_id for item in plan.dimensions))


def test_build_replacement_serializes_human_score_writer_and_removes_stale_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A direct score writer cannot race build cleanup or restore build-A labels afterward.

    Args:
        tmp_path: Isolated project and immutable artifact root.
        monkeypatch: Patch fixture pausing score history mutation inside both coordination locks.
    """
    store = _built_store(tmp_path)
    setup = _setup(store)
    selected_a = store.load_project().build
    assert selected_a is not None
    stale = HumanScoreReview.open(store)
    replacement = build_project(
        TraceNormalizationResult(traces=tuple(_trace(index) for index in range(100)), issues=()),
        store,
        created_at=_TIME + timedelta(seconds=1),
        code_revision="replacement-revision",
        mining_spec=MiningSpec(fit_task_budget=3, held_out_task_budget=1),
    )
    completed_b = _persist_grounded_build(store, replacement, select=False)
    writer_paused = Event()
    release_writer = Event()
    writer_errors: list[BaseException] = []
    selector_errors: list[BaseException] = []
    original_append = HumanScoreHistory.append

    def pause_append(history: HumanScoreHistory, score: HumanScore) -> HumanScoreHistory:
        """Pause one score mutation while it owns build and review coordination.

        Args:
            history: Latest persisted score history.
            score: Human score being appended.

        Returns:
            Updated score history after the selector has begun waiting.
        """
        writer_paused.set()
        assert release_writer.wait(timeout=5)
        return original_append(history, score)

    def run_writer() -> None:
        """Record any unexpected direct score-writer failure."""
        try:
            stale.upsert(
                rubric_id=setup.rubric.artifact_id,
                rollout_id="rollout-a",
                lineage_id="lineage-a",
                dimension_id="task-success",
                score=1,
                submission_id="submission-a",
                created_at=_TIME,
            )
        except BaseException as exc:  # noqa: BLE001 - thread reports its terminal result
            writer_errors.append(exc)

    def run_selector() -> None:
        """Select build B after the score writer releases coordination."""
        try:
            select_completed_build(store, completed_b, replacement.review)
        except BaseException as exc:  # noqa: BLE001 - thread reports its terminal result
            selector_errors.append(exc)

    monkeypatch.setattr(HumanScoreHistory, "append", pause_append)
    writer = Thread(target=run_writer, daemon=True)
    writer.start()
    assert writer_paused.wait(timeout=5)
    selector = Thread(target=run_selector, daemon=True)
    selector.start()
    selector.join(timeout=0.1)
    assert selector.is_alive()
    release_writer.set()
    writer.join(timeout=5)
    selector.join(timeout=5)

    assert not writer.is_alive()
    assert not selector.is_alive()
    assert writer_errors == []
    assert selector_errors == []
    saved = store.read_review()
    assert isinstance(saved, dict)
    assert saved["build_review"]["readiness_id"] == replacement.review.readiness_id
    assert "human_score_history" not in saved
    assert "human_score_submissions" not in saved
    with pytest.raises(ValueError, match="selected completed build"):
        stale.upsert(
            rubric_id=setup.rubric.artifact_id,
            rollout_id="rollout-a",
            lineage_id="lineage-a",
            dimension_id="task-success",
            score=1,
            submission_id="submission-b",
            created_at=_TIME + timedelta(seconds=2),
        )


def test_estimate_uses_catalog_prices_and_records_provenance(tmp_path: Path) -> None:
    """Catalog-resolved prices stay on the budget before any credential lookup."""
    store = _built_store(tmp_path)
    _setup(store)
    plan = prepare_manual_judge_calibration(store, sample_size=3)
    catalog = _catalog().model_copy(
        update={
            "models": {
                "judge-main": ModelRecord(
                    connection="openai-main",
                    model="judge-model",
                    capabilities=ModelCapabilities(
                        input_cost_per_million_tokens_usd=1.0,
                        output_cost_per_million_tokens_usd=2.0,
                    ),
                )
            }
        }
    )

    budget = estimate_manual_judge_budget(
        plan,
        catalog=catalog,
        maximum_input_tokens_per_call=4_096,
        maximum_cost_usd=1.0,
    )

    assert budget.input_usd_per_million_tokens == 1.0
    assert budget.output_usd_per_million_tokens == 2.0
    assert budget.pricing_source is PricingSource.CONFIGURED
    assert budget.call_count == 3
    assert budget.estimated_cost_usd == pytest.approx(0.331776)


def test_estimate_fails_when_the_ceiling_cannot_admit_the_sample(tmp_path: Path) -> None:
    """A conservative estimate that exceeds the operator ceiling fails closed."""
    store = _built_store(tmp_path)
    _setup(store)
    plan = prepare_manual_judge_calibration(store, sample_size=3)

    with pytest.raises(ValueError, match="exceeds --maximum-cost-usd"):
        estimate_manual_judge_budget(
            plan,
            input_usd_per_million_tokens=1.0,
            output_usd_per_million_tokens=2.0,
            maximum_input_tokens_per_call=4_096,
            maximum_cost_usd=0.000001,
        )


def test_estimate_requires_catalog_or_both_overrides(tmp_path: Path) -> None:
    """A budget cannot be invented from a sample count alone."""
    store = _built_store(tmp_path)
    _setup(store)
    plan = prepare_manual_judge_calibration(store, sample_size=3)

    with pytest.raises(ManualJudgeError, match="project model catalog"):
        estimate_manual_judge_budget(
            plan,
            maximum_input_tokens_per_call=4_096,
            maximum_cost_usd=1.0,
        )


def test_calibration_refuses_before_resolution_write_or_model_call(tmp_path: Path) -> None:
    """Missing spend consent blocks credentials, artifacts, and dispatch, but keeps labels."""
    store = _built_store(tmp_path)
    _setup(store)
    plan = prepare_manual_judge_calibration(store, sample_size=3)
    labels = _labels(store)
    client = _JudgeClient(plan.setup.judge_model)
    runtime = _RuntimeCatalog(
        ResolvedModel(
            alias="judge-main",
            snapshot=plan.setup.judge_model,
            capabilities=ModelCapabilities(),
            client=client,
            embedding_client=None,
        )
    )
    budget = estimate_manual_judge_budget(
        plan,
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
        maximum_input_tokens_per_call=4_096,
        maximum_cost_usd=1.0,
    )
    before_review = store.read_review()
    before_artifacts = store.artifacts.list_ids()

    with pytest.raises(ManualJudgeError, match="spend consent"):
        calibrate_manual_judge(
            store,
            cast(RuntimeModelCatalog, runtime),
            plan,
            labels,
            budget,
            spend_consented=False,
            approve=False,
            accept_insufficient_labels=False,
            created_at=_TIME,
            code_revision="test-revision",
        )

    assert runtime.preflight_calls == 0
    assert client.requests == []
    review = store.read_review()
    assert isinstance(review, dict)
    manual_judge = review["manual_judge"]
    assert isinstance(manual_judge, dict)
    drafts = cast(list[JsonValue], manual_judge["label_drafts"])
    assert len(drafts) == 1
    drafted = drafts[0]
    assert isinstance(drafted, dict)
    assert len(cast(list[JsonValue], drafted["labels"])) == len(labels)
    restored = {
        **review,
        "manual_judge": {**manual_judge, "label_drafts": []},
    }
    assert restored == before_review
    assert store.artifacts.list_ids() == before_artifacts


def test_calibration_reports_then_approves_and_replays_without_calls(tmp_path: Path) -> None:
    """Completed evidence stays unapproved until a second explicit approval and replays free."""
    store = _built_store(tmp_path)
    _setup(store)
    plan = prepare_manual_judge_calibration(store, sample_size=3)
    labels = _labels(store)
    client = _JudgeClient(plan.setup.judge_model)
    runtime = _RuntimeCatalog(
        ResolvedModel(
            alias="judge-main",
            snapshot=plan.setup.judge_model,
            capabilities=ModelCapabilities(),
            client=client,
            embedding_client=None,
        )
    )
    budget = estimate_manual_judge_budget(
        plan,
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
        maximum_input_tokens_per_call=4_096,
        maximum_cost_usd=1.0,
    )

    reviewed = calibrate_manual_judge(
        store,
        cast(RuntimeModelCatalog, runtime),
        plan,
        labels,
        budget,
        spend_consented=True,
        approve=False,
        accept_insufficient_labels=True,
        created_at=_TIME,
        code_revision="test-revision",
    )

    assert reviewed.approved_calibration is None
    assert reviewed.provider_calls_made == 3
    assert len(client.requests) == 3
    assert reviewed.report.worst_disagreements

    approved = calibrate_manual_judge(
        store,
        cast(RuntimeModelCatalog, runtime),
        plan,
        labels,
        budget,
        spend_consented=True,
        approve=True,
        accept_insufficient_labels=True,
        created_at=_TIME + timedelta(seconds=1),
        code_revision="test-revision",
    )

    assert approved.approved_calibration is not None
    assert approved.provider_calls_made == 0
    assert len(client.requests) == 3
    replay = calibrate_manual_judge(
        store,
        cast(RuntimeModelCatalog, runtime),
        plan,
        labels,
        budget,
        spend_consented=True,
        approve=True,
        accept_insufficient_labels=True,
        created_at=_TIME + timedelta(seconds=2),
        code_revision="test-revision",
    )
    assert replay.approved_calibration == approved.approved_calibration
    assert replay.provider_calls_made == 0
    assert len(client.requests) == 3


def test_completed_audit_tamper_fails_before_replay_or_approval(tmp_path: Path) -> None:
    """Audit payload changes invalidate the manifest before any replayed provider call."""
    store = _built_store(tmp_path)
    _setup(store)
    plan = prepare_manual_judge_calibration(store, sample_size=3)
    labels = _labels(store)
    client = _JudgeClient(plan.setup.judge_model)
    runtime = _RuntimeCatalog(
        ResolvedModel(
            alias="judge-main",
            snapshot=plan.setup.judge_model,
            capabilities=ModelCapabilities(),
            client=client,
            embedding_client=None,
        )
    )
    budget = estimate_manual_judge_budget(
        plan,
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
        maximum_input_tokens_per_call=4_096,
        maximum_cost_usd=1.0,
    )
    result = calibrate_manual_judge(
        store,
        cast(RuntimeModelCatalog, runtime),
        plan,
        labels,
        budget,
        spend_consented=True,
        approve=False,
        accept_insufficient_labels=True,
        created_at=_TIME,
        code_revision="test-revision",
    )
    audit_path = store.artifacts.read(result.audit.audit_id).directory / "audit.json"
    audit_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ManualJudgeError, match="audit is unavailable"):
        calibrate_manual_judge(
            store,
            cast(RuntimeModelCatalog, runtime),
            plan,
            labels,
            budget,
            spend_consented=True,
            approve=True,
            accept_insufficient_labels=True,
            created_at=_TIME + timedelta(seconds=1),
            code_revision="test-revision",
        )
    assert len(client.requests) == 3


def test_setup_rejects_stale_file_based_projection_without_editor(tmp_path: Path) -> None:
    """Custom projections are bound during prepare, including --approve and --rubric-file."""
    store = _built_store(tmp_path)

    with pytest.raises(ManualJudgeError, match="boolean score projections"):
        prepare_manual_judge_setup(
            store,
            _catalog(),
            prompt_template=_template("boolean"),
            created_at=_TIME,
            code_revision="test-revision",
        )


def test_setup_replay_rejects_changed_contract_with_same_model(tmp_path: Path) -> None:
    """A saved alias and model cannot mask a changed prompt, mapping, schema, or projection."""
    store = _built_store(tmp_path)
    plan = prepare_manual_judge_setup(
        store,
        _catalog(),
        created_at=_TIME,
        code_revision="test-revision",
    )
    saved = commit_manual_judge_setup(store, plan, confirmed=True)
    changed = replace(plan, prompt_template=_template("boolean"))

    with pytest.raises(ManualJudgeError, match="different finalized judge setup"):
        commit_manual_judge_setup(store, changed, confirmed=True)

    review = store.read_review()
    assert isinstance(review, dict)
    assert review["manual_judge"]["setup"]["artifact_id"] == saved.setup_id


def test_setup_commit_upgrades_template_version_and_restarts_calibration(
    tmp_path: Path,
) -> None:
    """A saved version 2 setup is replaced by its version 3 twin with fresh review state."""
    store = _built_store(tmp_path)
    legacy_plan = prepare_manual_judge_setup(
        store,
        _catalog(),
        prompt_template=DEFAULT_JUDGE_TEMPLATE.model_copy(update={"template_version": "2"}),
        preview_count=2,
        created_at=_TIME,
        code_revision="test-revision",
    )
    legacy = commit_manual_judge_setup(store, legacy_plan, confirmed=True)
    upgrade_plan = prepare_manual_judge_setup(
        store,
        _catalog(),
        created_at=_TIME,
        code_revision="test-revision",
    )
    upgraded = commit_manual_judge_setup(store, upgrade_plan, confirmed=True)

    assert legacy.prompt_template.template_version == "2"
    assert upgraded.prompt_template.template_version == "3"
    assert upgraded.setup_id != legacy.setup_id
    assert upgraded.rubric == legacy.rubric
    assert upgraded.inputs == legacy.inputs
    review = store.read_review()
    assert isinstance(review, dict)
    state = ManualJudgeReviewState.model_validate(review["manual_judge"])
    assert state.setup.artifact_id == upgraded.setup_id
    assert state.label_drafts == ()
    assert state.trace_reviews == ()
    assert state.audit is None
    assert state.approved_calibration is None
    assert store.artifacts.read(legacy.setup_id).manifest is not None


@pytest.mark.parametrize("shape", ["boolean", "categorical"])
def test_non_scalar_calibration_executes_saved_contract(
    tmp_path: Path, shape: _FeedbackShape
) -> None:
    """Boolean and categorical calls render and project the exact finalized contract."""
    store = _built_store(tmp_path)
    setup_plan = prepare_manual_judge_setup(
        store,
        _catalog(),
        dimensions=_wide_axes(),
        prompt_template=_template(shape),
        created_at=_TIME,
        code_revision="test-revision",
    )
    commit_manual_judge_setup(store, setup_plan, confirmed=True)
    plan = prepare_manual_judge_calibration(store, sample_size=1)
    labels = (
        ManualJudgeLabel(
            trace_id=plan.traces[0].trace_id,
            dimension_id="task-success",
            score=4,
        ),
    )
    client = _StructuredJudgeClient(plan.setup.judge_model, shape)
    runtime = _RuntimeCatalog(
        ResolvedModel(
            alias="judge-main",
            snapshot=plan.setup.judge_model,
            capabilities=ModelCapabilities(),
            client=client,
            embedding_client=None,
        )
    )
    budget = estimate_manual_judge_budget(
        plan,
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
        maximum_input_tokens_per_call=4_096,
        maximum_cost_usd=1.0,
    )

    result = calibrate_manual_judge(
        store,
        cast(RuntimeModelCatalog, runtime),
        plan,
        labels,
        budget,
        spend_consented=True,
        approve=False,
        accept_insufficient_labels=True,
        created_at=_TIME,
        code_revision="test-revision",
    )

    assert result.provider_calls_made == 1
    request = client.requests[0]
    assert request.messages[0].content == "Follow the saved contract exactly."
    visible = request.messages[1].content or ""
    assert "RULES_CUSTOM:" in visible
    assert "TRACE_CUSTOM:" in visible
    assert json.dumps(plan.setup.prompt_template.response_schema, sort_keys=True) in visible
    judgment = Judgment.model_validate_json(
        store.artifacts.read_bytes(result.audit.judgments[0].judgment.artifact_id, "judgment.json")
    )
    assert judgment.dimensions[0].raw_score == 5
    assert result.audit.positional_bias_comparisons is None


def test_pairwise_calibration_uses_same_task_and_counterbalances_order(tmp_path: Path) -> None:
    """Pairwise calibration freezes typed labels, both orders, and direct bias counts."""
    store = _built_store(tmp_path)
    setup_plan = prepare_manual_judge_setup(
        store,
        _catalog(),
        dimensions=_wide_axes(),
        prompt_template=_template("pairwise"),
        created_at=_TIME,
        code_revision="test-revision",
    )
    commit_manual_judge_setup(store, setup_plan, confirmed=True)
    plan = prepare_manual_judge_calibration(store, sample_size=1)
    reference = plan.reference_traces[0]
    assert reference is not None
    labels = (
        ManualJudgeLabel(
            trace_id=plan.traces[0].trace_id,
            reference_trace_id=reference.trace_id,
            dimension_id="task-success",
            winner="winner_a",
        ),
    )
    client = _StructuredJudgeClient(plan.setup.judge_model, "pairwise")
    runtime = _RuntimeCatalog(
        ResolvedModel(
            alias="judge-main",
            snapshot=plan.setup.judge_model,
            capabilities=ModelCapabilities(),
            client=client,
            embedding_client=None,
        )
    )
    budget = estimate_manual_judge_budget(
        plan,
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
        maximum_input_tokens_per_call=4_096,
        maximum_cost_usd=1.0,
    )

    result = calibrate_manual_judge(
        store,
        cast(RuntimeModelCatalog, runtime),
        plan,
        labels,
        budget,
        spend_consented=True,
        approve=False,
        accept_insufficient_labels=True,
        created_at=_TIME,
        code_revision="test-revision",
    )

    assert plan.tasks[0].task_id
    assert len(client.requests) == 2
    assert len(result.audit.judgments[0].probes) == 2
    assert result.audit.positional_bias_comparisons == 1
    assert result.audit.positional_bias_flips == 1
    sample_sha256 = calibration_sample_digest(plan.setup, calibration_sample(plan))
    review = read_trace_reviews(store, plan.setup, sample_sha256)[0]
    proposal = review.axes[0].judge_proposal
    accepted = review.axes[0].final_accepted_label
    assert proposal.cited_trace_evidence == ()
    assert proposal.cited_reference_trace_evidence == ()
    assert accepted.cited_trace_evidence == proposal.cited_trace_evidence
    assert accepted.cited_reference_trace_evidence == proposal.cited_reference_trace_evidence
    first = client.requests[0].messages[1].content or ""
    second = client.requests[1].messages[1].content or ""
    target_rollout = plan.previews[0].rollout_id
    reference_rollout = plan.previews[0].reference_rollout_id
    assert reference_rollout is not None
    assert first.index(target_rollout) < first.index(reference_rollout)
    assert second.index(reference_rollout) < second.index(target_rollout)


def test_pairwise_calibration_fails_before_labels_or_calls_without_same_task_pair(
    tmp_path: Path,
) -> None:
    """Pairwise selection refuses unrelated evidence before labels or provider resolution."""
    store = _unpaired_trace_store(tmp_path)
    setup_plan = prepare_manual_judge_setup(
        store,
        _catalog(),
        dimensions=_wide_axes(),
        prompt_template=_template("pairwise"),
        created_at=_TIME,
        code_revision="test-revision",
    )
    commit_manual_judge_setup(store, setup_plan, confirmed=True)
    before_review = store.read_review()
    before_artifacts = store.artifacts.list_ids()

    with pytest.raises(ManualJudgeError, match="two real outputs for the same canonical fit task"):
        prepare_manual_judge_calibration(store, sample_size=1)

    assert store.read_review() == before_review
    assert store.artifacts.list_ids() == before_artifacts


def test_interrupted_calibration_reuses_completed_probes_at_later_time(tmp_path: Path) -> None:
    """Retrying later reuses completed rows and dispatches only missing provider probes."""
    store = _built_store(tmp_path)
    _setup(store)
    plan = prepare_manual_judge_calibration(store, sample_size=3)
    labels = _labels(store)
    first_client = _StructuredJudgeClient(plan.setup.judge_model, "scalar", fail_after=1)
    first_runtime = _RuntimeCatalog(
        ResolvedModel(
            alias="judge-main",
            snapshot=plan.setup.judge_model,
            capabilities=ModelCapabilities(),
            client=first_client,
            embedding_client=None,
        )
    )
    budget = estimate_manual_judge_budget(
        plan,
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
        maximum_input_tokens_per_call=4_096,
        maximum_cost_usd=1.0,
    )
    with pytest.raises(RuntimeError, match="simulated provider interruption"):
        calibrate_manual_judge(
            store,
            cast(RuntimeModelCatalog, first_runtime),
            plan,
            labels,
            budget,
            spend_consented=True,
            approve=False,
            accept_insufficient_labels=True,
            created_at=_TIME,
            code_revision="test-revision",
        )
    assert len(first_client.requests) == 1

    retry_client = _StructuredJudgeClient(plan.setup.judge_model, "scalar")
    retry_runtime = _RuntimeCatalog(
        ResolvedModel(
            alias="judge-main",
            snapshot=plan.setup.judge_model,
            capabilities=ModelCapabilities(),
            client=retry_client,
            embedding_client=None,
        )
    )
    result = calibrate_manual_judge(
        store,
        cast(RuntimeModelCatalog, retry_runtime),
        plan,
        labels,
        budget,
        spend_consented=True,
        approve=False,
        accept_insufficient_labels=True,
        created_at=_TIME + timedelta(minutes=10),
        code_revision="test-revision",
    )

    assert result.provider_calls_made == 2
    assert len(retry_client.requests) == 2
    assert len(result.audit.judgments) == 3


def test_provider_failure_keeps_completed_human_labels_for_replay(tmp_path: Path) -> None:
    """A provider interruption leaves every completed rating durable and reusable.

    Raises:
        AssertionError: Completed human labels are lost or a resumed label differs.
    """
    store = _built_store(tmp_path)
    setup = _setup(store)
    plan = prepare_manual_judge_calibration(store, sample_size=3)
    labels = _labels(store)
    client = _StructuredJudgeClient(plan.setup.judge_model, "scalar", fail_after=0)
    runtime = _RuntimeCatalog(
        ResolvedModel(
            alias="judge-main",
            snapshot=plan.setup.judge_model,
            capabilities=ModelCapabilities(),
            client=client,
            embedding_client=None,
        )
    )
    budget = estimate_manual_judge_budget(
        plan,
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
        maximum_input_tokens_per_call=4_096,
        maximum_cost_usd=1.0,
    )
    with pytest.raises(RuntimeError, match="simulated provider interruption"):
        calibrate_manual_judge(
            store,
            cast(RuntimeModelCatalog, runtime),
            plan,
            labels,
            budget,
            spend_consented=True,
            approve=False,
            accept_insufficient_labels=True,
            created_at=_TIME,
            code_revision="test-revision",
        )

    digest = calibration_sample_digest(setup, calibration_sample(plan))
    assert read_label_draft(store, setup, digest) == labels


def test_retry_reuses_audit_when_review_pointer_write_was_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume a verified audit after interruption before its review pointer is saved.

    Args:
        tmp_path: Isolated project root for immutable calibration evidence.
        monkeypatch: Pytest patch service for the review-pointer crash boundary.
    """
    store = _built_store(tmp_path)
    _setup(store)
    plan = prepare_manual_judge_calibration(store, sample_size=1)
    labels = _labels(store)[:1]
    budget = estimate_manual_judge_budget(
        plan,
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
        maximum_input_tokens_per_call=4_096,
        maximum_cost_usd=1.0,
    )
    first_client = _StructuredJudgeClient(plan.setup.judge_model, "scalar")
    first_runtime = _RuntimeCatalog(
        ResolvedModel(
            alias="judge-main",
            snapshot=plan.setup.judge_model,
            capabilities=ModelCapabilities(),
            client=first_client,
            embedding_client=None,
        )
    )
    original_write_review_state = manual_judge_workflow.write_review_state

    def interrupt_review_pointer(*args: object, **kwargs: object) -> None:
        """Raise at the exact boundary after audit persistence and before state publication.

        Args:
            args: Positional review-state writer arguments.
            kwargs: Keyword review-state writer arguments.

        Raises:
            RuntimeError: Always, to model process interruption before the pointer write.
        """
        del args, kwargs
        raise RuntimeError("simulated review pointer interruption")

    monkeypatch.setattr(manual_judge_workflow, "write_review_state", interrupt_review_pointer)
    with pytest.raises(RuntimeError, match="simulated review pointer interruption"):
        calibrate_manual_judge(
            store,
            cast(RuntimeModelCatalog, first_runtime),
            plan,
            labels,
            budget,
            spend_consented=True,
            approve=False,
            accept_insufficient_labels=True,
            created_at=_TIME,
            code_revision="test-revision",
        )
    audits = tuple(
        artifact_id
        for artifact_id in store.artifacts.list_ids()
        if store.artifacts.read(artifact_id).manifest.artifact_type
        == "manual-judge-calibration-audit"
    )
    assert len(audits) == 1
    review = store.read_review()
    assert isinstance(review, dict)
    assert review["manual_judge"]["audit"] is None

    monkeypatch.setattr(
        manual_judge_workflow,
        "write_review_state",
        original_write_review_state,
    )
    retry_client = _StructuredJudgeClient(plan.setup.judge_model, "scalar")
    retry_runtime = _RuntimeCatalog(
        ResolvedModel(
            alias="judge-main",
            snapshot=plan.setup.judge_model,
            capabilities=ModelCapabilities(),
            client=retry_client,
            embedding_client=None,
        )
    )
    result = calibrate_manual_judge(
        store,
        cast(RuntimeModelCatalog, retry_runtime),
        plan,
        labels,
        budget,
        spend_consented=True,
        approve=False,
        accept_insufficient_labels=True,
        created_at=_TIME + timedelta(minutes=10),
        code_revision="test-revision",
    )

    assert result.audit.audit_id == audits[0]
    assert result.audit.created_at == _TIME
    assert result.provider_calls_made == 0
    assert retry_client.requests == []


def test_interrupted_pairwise_probe_reuses_forward_order(tmp_path: Path) -> None:
    """A reverse-order interruption reuses the frozen forward probe on retry."""
    store = _built_store(tmp_path)
    setup_plan = prepare_manual_judge_setup(
        store,
        _catalog(),
        dimensions=_wide_axes(),
        prompt_template=_template("pairwise"),
        created_at=_TIME,
        code_revision="test-revision",
    )
    commit_manual_judge_setup(store, setup_plan, confirmed=True)
    plan = prepare_manual_judge_calibration(store, sample_size=1)
    reference = plan.reference_traces[0]
    assert reference is not None
    labels = (
        ManualJudgeLabel(
            trace_id=plan.traces[0].trace_id,
            reference_trace_id=reference.trace_id,
            dimension_id="task-success",
            winner="winner_a",
        ),
    )
    budget = estimate_manual_judge_budget(
        plan,
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
        maximum_input_tokens_per_call=4_096,
        maximum_cost_usd=1.0,
    )
    first_client = _StructuredJudgeClient(plan.setup.judge_model, "pairwise", fail_after=1)
    first_runtime = _RuntimeCatalog(
        ResolvedModel(
            alias="judge-main",
            snapshot=plan.setup.judge_model,
            capabilities=ModelCapabilities(),
            client=first_client,
            embedding_client=None,
        )
    )
    with pytest.raises(RuntimeError, match="simulated provider interruption"):
        calibrate_manual_judge(
            store,
            cast(RuntimeModelCatalog, first_runtime),
            plan,
            labels,
            budget,
            spend_consented=True,
            approve=False,
            accept_insufficient_labels=True,
            created_at=_TIME,
            code_revision="test-revision",
        )
    assert len(first_client.requests) == 1

    retry_client = _StructuredJudgeClient(plan.setup.judge_model, "pairwise")
    retry_runtime = _RuntimeCatalog(
        ResolvedModel(
            alias="judge-main",
            snapshot=plan.setup.judge_model,
            capabilities=ModelCapabilities(),
            client=retry_client,
            embedding_client=None,
        )
    )
    result = calibrate_manual_judge(
        store,
        cast(RuntimeModelCatalog, retry_runtime),
        plan,
        labels,
        budget,
        spend_consented=True,
        approve=False,
        accept_insufficient_labels=True,
        created_at=_TIME + timedelta(minutes=10),
        code_revision="test-revision",
    )

    assert result.provider_calls_made == 1
    assert len(retry_client.requests) == 1
    assert len(result.audit.judgments[0].probes) == 2
