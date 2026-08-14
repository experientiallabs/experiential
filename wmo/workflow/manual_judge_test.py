"""Behavioral tests for explicit manual judge setup and calibration."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from wmo.common.core.artifacts import SourceIdentity, sha256_json
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
from wmo.workflow.manual_judge_contracts import ManualJudgeLabel, ManualJudgeSetupArtifact

_TIME = datetime(2026, 8, 13, tzinfo=UTC)
_DIGEST = "a" * 64


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
