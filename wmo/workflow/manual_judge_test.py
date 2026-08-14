"""Behavioral tests for explicit manual judge setup and calibration."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

import pytest

import wmo.workflow.manual_judge as manual_judge_workflow
from wmo.common.core.artifacts import JsonObject, SourceIdentity, sha256_json
from wmo.common.judging import PromptDefinition
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
)
from wmo.common.project import ProjectConfig, ProjectStore
from wmo.common.traces import Trace, TraceOutcome, TraceSource, TraceSpan
from wmo.runtime.models.registry import ResolvedModel, RuntimeModelCatalog
from wmo.simulation.build import build_project
from wmo.simulation.ingest.otlp import TraceNormalizationResult
from wmo.simulation.mining.service import MiningSpec
from wmo.workflow.manual_judge import (
    ManualJudgeError,
    calibrate_manual_judge,
    commit_manual_judge_setup,
    estimate_manual_judge_budget,
    prepare_manual_judge_calibration,
    prepare_manual_judge_setup,
)
from wmo.workflow.manual_judge_contracts import (
    JudgePromptTemplate,
    JudgeScoreProjection,
    ManualJudgeLabel,
    ManualJudgeSetupArtifact,
    judge_feedback_schema,
)

_TIME = datetime(2026, 8, 13, tzinfo=UTC)
_DIGEST = "a" * 64
_FeedbackShape = Literal["scalar", "boolean", "categorical", "pairwise"]


class _JudgeClient:
    """Return deterministic cited scalar scores while recording every provider call."""

    def __init__(self, model: ModelSnapshot) -> None:
        """Bind the exact configured model identity for deterministic responses.

        Args:
            model: Frozen configured judge snapshot.
        """
        self.model = model
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return one schema-valid score citing a span present in the request.

        Args:
            request: Structured LM judge request.

        Returns:
            Deterministic model response with no observed provider cost.
        """
        self.requests.append(request)
        content = request.messages[1].content or ""
        match = re.search(r'"span_id":\s*"([^"]+)"', content)
        assert match is not None
        return ModelResponse(
            output=AssistantAction(
                content=json.dumps(
                    {
                        "dimensions": [
                            {
                                "dimension_id": "task-success",
                                "raw_score": 4,
                                "evidence_span_ids": [match.group(1)],
                                "feedback": "The trace shows the task was handled.",
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
        content = request.messages[1].content or ""
        span_ids = re.findall(r'"span_id":\s*"([^"]+)"', content)
        assert span_ids
        common = {
            "dimension_id": "task-success",
            "feedback": "Structured evidence supports the verdict.",
        }
        if self.shape == "scalar":
            dimension = {**common, "raw_score": 4, "evidence_span_ids": [span_ids[0]]}
        elif self.shape == "boolean":
            dimension = {**common, "passed": True, "evidence_span_ids": [span_ids[0]]}
        elif self.shape == "categorical":
            dimension = {**common, "category": "good", "evidence_span_ids": [span_ids[0]]}
        else:
            assert len(span_ids) >= 2
            dimension = {
                **common,
                "winner": "winner_a",
                "evidence_span_ids_a": [span_ids[0]],
                "evidence_span_ids_b": [span_ids[1]],
            }
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


def _built_store(tmp_path: Path) -> ProjectStore:
    """Create a completed build with three fit lineages and one held-out lineage.

    Args:
        tmp_path: Isolated test directory.

    Returns:
        Initialized project store with deterministic build readiness.
    """
    store = ProjectStore(tmp_path / ".wmo", "support")
    store.initialize(ProjectConfig(project_id="support"))
    build_project(
        TraceNormalizationResult(traces=tuple(_trace(index) for index in range(100)), issues=()),
        store,
        created_at=_TIME,
        code_revision="test-revision",
        mining_spec=MiningSpec(fit_task_budget=3, held_out_task_budget=1),
    )
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
    build_project(
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
        projection = JudgeScoreProjection(boolean_scores={"false": 1, "true": 4})
    elif shape == "categorical":
        projection = JudgeScoreProjection(categorical_scores={"bad": 1, "good": 4})
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
            score=4,
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
    assert store.read_review() == before_review
    assert store.artifacts.list_ids() == before_artifacts
    with pytest.raises(ManualJudgeError, match="explicit confirmation"):
        commit_manual_judge_setup(store, plan, confirmed=False)
    assert store.read_review() == before_review
    assert store.artifacts.list_ids() == before_artifacts


def test_calibration_refuses_before_resolution_write_or_model_call(tmp_path: Path) -> None:
    """Missing spend consent blocks credentials, artifacts, labels, and provider dispatch."""
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
    assert store.read_review() == before_review
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


@pytest.mark.parametrize("shape", ["boolean", "categorical"])
def test_non_scalar_calibration_executes_saved_contract(
    tmp_path: Path, shape: _FeedbackShape
) -> None:
    """Boolean and categorical calls render and project the exact finalized contract."""
    store = _built_store(tmp_path)
    setup_plan = prepare_manual_judge_setup(
        store,
        _catalog(),
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
    assert judgment.dimensions[0].raw_score == 4
    assert result.audit.positional_bias_comparisons is None


def test_pairwise_calibration_uses_same_task_and_counterbalances_order(tmp_path: Path) -> None:
    """Pairwise calibration freezes typed labels, both orders, and direct bias counts."""
    store = _built_store(tmp_path)
    setup_plan = prepare_manual_judge_setup(
        store,
        _catalog(),
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
