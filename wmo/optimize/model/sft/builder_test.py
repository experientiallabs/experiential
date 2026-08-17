"""Store-backed regression tests for frozen, leakage-safe SFT dataset construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from wmo.common.core.artifacts import (
    ArtifactInput,
    FailureCode,
    SourceIdentity,
    StructuredFailure,
    canonical_json_bytes,
    sha256_json,
)
from wmo.common.judging import (
    HumanScore,
    HumanScoreReview,
    JudgeCalibration,
    JudgeCalibrationService,
    JudgeScoreObservation,
    Judgment,
    LMJudge,
    PromptDefinition,
    RouterLineageAssignment,
    RouterLineageSplit,
    Rubric,
    RubricDimension,
    ScoreAnchor,
    write_router_lineage_split,
)
from wmo.common.models import (
    AssistantAction,
    EmbeddingCostReservation,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
)
from wmo.common.project import ProjectStore, artifact_input
from wmo.common.project.store_test import _store
from wmo.common.rollouts import (
    RolloutArtifact,
    RolloutEventKind,
    RolloutSpan,
    SimulationCellBinding,
    SimulationMode,
    StopReason,
    WorldModelSimulatorSnapshot,
)
from wmo.common.tasks import TaskCase, TaskSet
from wmo.common.traces import Trace, TraceOutcome, TraceSource, TraceSpan
from wmo.optimize.model.sft.builder import (
    SFTBuildError,
    SFTBuildSpec,
    build_sft_dataset,
    ensure_no_cross_split_fingerprints,
    load_sft_dataset,
    write_sft_dataset,
)
from wmo.optimize.model.sft.builder_fidelity_fixture_test import approved_fidelity_report
from wmo.optimize.model.sft.contracts import (
    AssistantActionEvent,
    HumanApproval,
    InfrastructureFailureEvent,
    ProductionAcceptanceEvidence,
    ProductionAcceptanceRule,
    ProductionSFTSource,
    SFTDatasetArtifact,
    SFTMessage,
    SFTTranscript,
    TeacherAcceptanceEvidence,
    TeacherAcceptanceRule,
    TeacherSFTSource,
)
from wmo.simulation.ingest.dataset import persist_trace_dataset
from wmo.simulation.ingest.otlp import TraceNormalizationResult

_DIGEST = "a" * 64
_TIME = datetime(2026, 8, 12, tzinfo=UTC)


def _judge_content(span_id: str, feedback: str) -> str:
    """Build one deterministic fake-judge response."""
    dimension = {
        "dimension_id": "quality",
        "raw_score": 5,
        "evidence_span_ids": [span_id],
        "feedback": feedback,
    }
    return json.dumps({"dimensions": [dimension]})


class _FakeJudgeClient:
    """Return one fixed structured judgment without network access or provider credentials."""

    def __init__(self, model: ModelSnapshot, content: str) -> None:
        self._model = model
        self._content = content
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return the configured model identity and structured payload."""
        self.requests.append(request)
        return ModelResponse(
            output=AssistantAction(content=self._content),
            model=self._model,
            economics=OperationEconomics(),
        )


@dataclass(frozen=True)
class _TeacherFixture:
    """Verified teacher-evidence IDs retained for adversarial SFT consumption tests."""

    source: TeacherSFTSource
    evidence: TeacherAcceptanceEvidence
    rollout_input: ArtifactInput
    task_set_input: ArtifactInput
    judgment_input: ArtifactInput
    calibration_input: ArtifactInput
    fidelity_input: ArtifactInput
    rule_input: ArtifactInput
    transcript: SFTTranscript


def _inputs(*values: ArtifactInput) -> tuple[ArtifactInput, ...]:
    """Return exact artifact references in the canonical input ordering."""
    return tuple(sorted(values, key=lambda value: value.artifact_id))


def _model(model_id: str = "judge-model") -> ModelSnapshot:
    """Build one fixed resolved model identity for local immutable test evidence."""
    return ModelSnapshot(
        provider="test",
        model_id=model_id,
        capabilities_sha256=_DIGEST,
        connection_sha256=_DIGEST,
    )


def _transcript(tag: str) -> SFTTranscript:
    """Build one canonical source transcript with two trainable assistant actions."""
    return SFTTranscript(
        events=(
            SFTMessage(role="system", content="Follow the support policy."),
            SFTMessage(role="user", content=f"Resolve support request {tag}."),
            AssistantActionEvent(action=AssistantAction(content=f"Investigating {tag}.")),
            AssistantActionEvent(action=AssistantAction(content=f"Resolved {tag}.")),
        )
    )


def _trace(tag: str, *, task: str | None = None) -> Trace:
    """Build one normalized successful production trace eligible for trace-dataset storage."""
    return Trace(
        trace_id=f"trace-{tag}",
        conversation_id=f"conversation-{tag}",
        task=task or f"Resolve support request {tag}.",
        spans=(
            TraceSpan(
                span_id=f"trace-span-{tag}",
                name="agent.model_call",
                started_at=_TIME,
                ended_at=_TIME + timedelta(seconds=1),
            ),
        ),
        outcome=TraceOutcome(status="success", outcome_name="resolved"),
        source=TraceSource(
            identity=SourceIdentity(kind="otlp", source_id="fixture", sha256=_DIGEST),
            semantic_convention_version="1.37.0",
        ),
    )


def _write_trace_dataset(store: ProjectStore, trace: Trace) -> ArtifactInput:
    """Persist canonical W5-style trace evidence and return its manifest-derived input."""
    persisted = persist_trace_dataset(
        TraceNormalizationResult(traces=(trace,), issues=()),
        store.artifacts,
        created_at=_TIME,
        code_revision="w12-test",
    )
    return artifact_input(persisted.manifest)


def _write_production_source(
    store: ProjectStore,
    tag: str,
    *,
    human_approval: bool = False,
    task: str | None = None,
    transcript: SFTTranscript | None = None,
) -> ProductionSFTSource:
    """Persist production acceptance evidence whose transcript is owned by that artifact."""
    trace = _trace(tag, task=task)
    trace_input = _write_trace_dataset(store, trace)
    rule = ProductionAcceptanceRule(
        schema_version=1,
        created_at=_TIME,
        code_revision="w12-test",
        acceptance_rule_id=f"production-rule-{tag}",
        accepted_outcomes=("resolved",),
        allow_human_approval=True,
    )
    rule_input = artifact_input(
        store.artifacts.write_json(
            artifact_id=rule.acceptance_rule_id,
            artifact_type="sft-production-acceptance-rule",
            envelope=rule,
            files={"rule.json": rule},
        )
    )
    approval_input: ArtifactInput | None = None
    if human_approval:
        approval = HumanApproval(
            schema_version=1,
            created_at=_TIME,
            inputs=(trace_input,),
            code_revision="w12-test",
            approval_id=f"human-approval-{tag}",
            trace_dataset=trace_input,
            trace_id=trace.trace_id,
            approved_at=_TIME,
        )
        approval_input = artifact_input(
            store.artifacts.write_json(
                artifact_id=approval.approval_id,
                artifact_type="sft-human-approval",
                envelope=approval,
                files={"approval.json": approval},
            )
        )
    transcript = transcript or _transcript(tag)
    transcript_payload = canonical_json_bytes(transcript)
    evidence = ProductionAcceptanceEvidence(
        schema_version=1,
        created_at=_TIME,
        inputs=(
            _inputs(trace_input, rule_input)
            if approval_input is None
            else _inputs(trace_input, rule_input, approval_input)
        ),
        code_revision="w12-test",
        acceptance_evidence_id=f"production-evidence-{tag}",
        trace_dataset=trace_input,
        trace_id=trace.trace_id,
        trace_sha256=sha256_json(trace),
        acceptance_rule=rule_input,
        decision="human_approval" if approval_input is not None else "trusted_outcome",
        outcome_sha256=None if approval_input is not None else sha256_json(trace.outcome),
        human_approval=approval_input,
        transcript_path="transcript.json",
        transcript_sha256=hashlib.sha256(transcript_payload).hexdigest(),
        accepted_at=_TIME,
    )
    store.artifacts.write(
        artifact_id=evidence.acceptance_evidence_id,
        artifact_type="sft-production-acceptance",
        envelope=evidence,
        files={
            "evidence.json": canonical_json_bytes(evidence),
            "transcript.json": transcript_payload,
        },
    )
    return ProductionSFTSource(acceptance_evidence_id=evidence.acceptance_evidence_id)


def _task_set(store: ProjectStore) -> tuple[TaskCase, TaskSet, ArtifactInput]:
    """Persist a W5-shaped task set with trace-input lineage and exact task payload bytes."""
    source_trace = _trace("task-source")
    source_input = _write_trace_dataset(store, source_trace)
    task = TaskCase(
        task_id="task-teacher",
        lineage_group_id="task-lineage-teacher",
        partition="fit",
        instruction="Resolve the teacher support request.",
        workload_weight=1.0,
        source_trace_ids=(source_trace.trace_id,),
    )
    task_payload = canonical_json_bytes(task) + b"\n"
    task_set = TaskSet(
        schema_version=1,
        created_at=_TIME,
        inputs=(source_input,),
        code_revision="w12-test",
        task_set_id="task-set-teacher",
        task_ids=(task.task_id,),
        tasks_path="tasks.jsonl",
        tasks_sha256=hashlib.sha256(task_payload).hexdigest(),
    )
    task_set_input = artifact_input(
        store.artifacts.write(
            artifact_id=task_set.task_set_id,
            artifact_type="task-set",
            envelope=task_set,
            files={
                "task-set.json": canonical_json_bytes(task_set),
                "tasks.jsonl": task_payload,
            },
        )
    )
    return task, task_set, task_set_input


def _rubric(task_set_input: ArtifactInput) -> Rubric:
    """Build a one-dimensional, human-approved rubric tied to one canonical task set."""
    return Rubric(
        schema_version=1,
        created_at=_TIME,
        inputs=(task_set_input,),
        code_revision="w12-test",
        rubric_id="rubric-teacher",
        dimensions=(
            RubricDimension(
                dimension_id="quality",
                name="Quality",
                description="Whether the rollout resolved the requested support work.",
                min_score=0,
                max_score=5,
                anchors=(
                    ScoreAnchor(score=0, description="Quality level 0."),
                    ScoreAnchor(score=1, description="Quality level 1."),
                    ScoreAnchor(score=2, description="Quality level 2."),
                    ScoreAnchor(score=3, description="Quality level 3."),
                    ScoreAnchor(score=4, description="Quality level 4."),
                    ScoreAnchor(score=5, description="Quality level 5."),
                ),
            ),
        ),
        source_task_set_id=task_set_input.artifact_id,
        status="human_approved",
        approved_at=_TIME,
    )


def _rollout(
    tag: str,
    task: TaskCase,
    task_set: TaskSet,
    task_set_input: ArtifactInput,
) -> RolloutArtifact:
    """Build one successful world-model rollout suitable for authoritative judging.

    Args:
        tag: Stable suffix used to isolate fixture artifact identities.
        task: Exact task represented by the rollout.
        task_set: Immutable task set owning the task.
        task_set_input: Manifest pointer for the owning task set.

    Returns:
        Successful rollout with complete grounded simulation provenance.
    """
    model = _model("candidate-model")
    plan_input = ArtifactInput(artifact_id=f"evaluation-plan-{tag}", sha256=_DIGEST)
    spec_input = ArtifactInput(artifact_id=f"simulation-spec-{tag}", sha256=_DIGEST)
    fit_rag_input = ArtifactInput(artifact_id=f"fit-rag-{tag}", sha256=_DIGEST)
    grounded_world_model_input = ArtifactInput(artifact_id="grounded-world-model", sha256=_DIGEST)
    binding = SimulationCellBinding(
        evaluation_plan_input=plan_input,
        task_set_input=task_set_input,
        fit_rag_input=fit_rag_input,
        grounded_world_model_input=grounded_world_model_input,
        task_set_tasks_sha256=task_set.tasks_sha256,
        task_sha256=sha256_json(task),
        candidate_alias="candidate-a",
        candidate=model,
        agent_id="customer-agent",
        repeat=0,
        world_model_alias="world-model-a",
        world_model=model,
        simulator_id="world-model-v1",
        prompt_id="world-prompt-v1",
        prompt_version="v1",
        prompt_sha256=_DIGEST,
        query_embedding=EmbeddingCostReservation(
            model=model,
            input_usd_per_million_tokens=0.0,
            maximum_attempts=1,
            maximum_input_tokens=1,
        ),
        simulation_spec_input=spec_input,
        simulation_spec_sha256=_DIGEST,
        simulation_inputs_sha256=_DIGEST,
    )
    return RolloutArtifact(
        schema_version=1,
        created_at=_TIME,
        inputs=_inputs(
            plan_input,
            fit_rag_input,
            grounded_world_model_input,
            spec_input,
            task_set_input,
        ),
        code_revision="w12-test",
        artifact_id=f"rollout-artifact-{tag}",
        simulation_id=f"simulation-{tag}",
        cell_id=f"cell-{tag}",
        mode=SimulationMode.WORLD_MODEL,
        rollout_id=f"rollout-{tag}",
        trace_id=f"teacher-trace-{tag}",
        evidence_source="world_model",
        source_run_id=f"run-{tag}",
        task_id=task.task_id,
        candidate=model,
        agent_id="customer-agent",
        simulator=WorldModelSimulatorSnapshot(
            simulator_id="world-model-v1",
            prompt_id="world-prompt-v1",
            prompt_version="v1",
            prompt_sha256=_DIGEST,
            world_model=model,
        ),
        world_model=model,
        seed=1,
        repeat=0,
        spans=(
            RolloutSpan(
                span_id=f"rollout-span-{tag}",
                kind=RolloutEventKind.AGENT_MODEL_CALL,
                started_at=_TIME,
                ended_at=_TIME + timedelta(seconds=1),
                payload={"result": "resolved"},
                model=model,
            ),
        ),
        final_output=AssistantAction(content="Resolved support request."),
        stop_reason=StopReason.COMPLETED,
        candidate_economics=OperationEconomics(),
        retrieval_economics=OperationEconomics(),
        simulation_spec_sha256=_DIGEST,
        simulation_binding=binding,
    )


def _write_teacher_source(
    store: ProjectStore, *, bind_fidelity_to_teacher_rollout: bool = True
) -> _TeacherFixture:
    """Persist a complete W5, W6, rollout, fidelity, and acceptance chain for one teacher row."""
    task, task_set, task_set_input = _task_set(store)
    rubric = _rubric(task_set_input)
    store.artifacts.write_json(
        artifact_id=rubric.rubric_id,
        artifact_type="rubric",
        envelope=rubric,
        files={"rubric.json": rubric},
    )
    prompt = PromptDefinition.from_text("judge-prompt-v1", "Return structured judgment JSON.")
    model = _model()
    rollouts = tuple(
        _rollout(f"calibration-{index}", task, task_set, task_set_input) for index in range(2)
    )
    rollout_inputs: list[ArtifactInput] = []
    for rollout in rollouts:
        rollout_inputs.append(
            artifact_input(
                store.artifacts.write_json(
                    artifact_id=rollout.artifact_id,
                    artifact_type="rollout",
                    envelope=rollout,
                    files={"rollout.json": rollout},
                )
            )
        )
    split = write_router_lineage_split(
        store,
        RouterLineageSplit(
            schema_version=1,
            created_at=_TIME,
            inputs=(task_set_input,),
            code_revision="w12-test",
            split_id="router-lineage-split-teacher",
            source_task_set_id=task_set.task_set_id,
            fit_lineage_ids=("calibration-lineage-0", "calibration-lineage-1"),
            held_out_lineage_ids=(),
            assignments=tuple(
                RouterLineageAssignment(
                    rollout_id=rollout.rollout_id,
                    lineage_id=f"calibration-lineage-{index}",
                )
                for index, rollout in enumerate(rollouts)
            ),
        ),
    )
    review = HumanScoreReview.open(store)
    empty_labels = review.finalize(
        rubric_id=rubric.rubric_id,
        code_revision="w12-test",
        created_at=_TIME,
    )
    calibration_service = JudgeCalibrationService()
    provisional = calibration_service.bootstrap_provisional(
        store,
        rubric_id=rubric.rubric_id,
        label_set_id=empty_labels.label_set_id,
        router_lineage_split_id=split.split_id,
        judge_model=model,
        judge_prompt=prompt,
        created_at=_TIME,
        code_revision="w12-test",
    )
    observations: list[JudgeScoreObservation] = []
    for index, rollout in enumerate(rollouts):
        span_id = rollout.spans[0].span_id
        judgment = LMJudge(
            _FakeJudgeClient(
                model,
                _judge_content(span_id, "The rollout resolved the request."),
            ),
            prompt,
            code_revision="w12-test",
            clock=lambda: _TIME,
        ).judge_and_write(
            store,
            rollout_artifact_id=rollout.artifact_id,
            rubric_artifact_id=rubric.rubric_id,
            calibration_artifact_id=provisional.calibration_id,
        )
        judgment_input = artifact_input(store.artifacts.read(judgment.judgment_id).manifest)
        observations.append(
            JudgeScoreObservation(
                judgment=judgment_input,
                source_rollout=rollout_inputs[index],
                dimension_id="quality",
                raw_score=5,
                evidence_span_ids=(span_id,),
            )
        )
        review.append(
            HumanScore(
                label_id=f"human-label-{index}",
                rubric_id=rubric.rubric_id,
                rollout_id=rollout.rollout_id,
                lineage_id=f"calibration-lineage-{index}",
                dimension_id="quality",
                score=5,
                created_at=_TIME,
            )
        )
    labels = review.finalize(
        rubric_id=rubric.rubric_id,
        code_revision="w12-test",
        created_at=_TIME,
    )
    report = calibration_service.build_report(
        store,
        rubric_id=rubric.rubric_id,
        label_set_id=labels.label_set_id,
        router_lineage_split_id=split.split_id,
        observations=tuple(observations),
        created_at=_TIME,
        code_revision="w12-test",
    )
    calibration_service.write_report(store, report)
    calibration = calibration_service.write_calibration(
        store,
        report=report,
        calibration=calibration_service.approve(
            store,
            report,
            approved_at=_TIME,
            accept_insufficient_labels=True,
        ),
    )
    calibration_input = artifact_input(store.artifacts.read(calibration.calibration_id).manifest)
    final_rollout = _rollout("teacher", task, task_set, task_set_input)
    rollout_input = artifact_input(
        store.artifacts.write_json(
            artifact_id=final_rollout.artifact_id,
            artifact_type="rollout",
            envelope=final_rollout,
            files={"rollout.json": final_rollout},
        )
    )
    fidelity_rollout_input = (
        rollout_input if bind_fidelity_to_teacher_rollout else rollout_inputs[0]
    )
    final_judgment = LMJudge(
        _FakeJudgeClient(
            model,
            _judge_content(final_rollout.spans[0].span_id, "The teacher resolved the request."),
        ),
        prompt,
        code_revision="w12-test",
        clock=lambda: _TIME,
    ).judge_and_write(
        store,
        rollout_artifact_id=final_rollout.artifact_id,
        rubric_artifact_id=rubric.rubric_id,
        calibration_artifact_id=calibration.calibration_id,
    )
    judgment_input = artifact_input(store.artifacts.read(final_judgment.judgment_id).manifest)
    fidelity = approved_fidelity_report(
        inputs=_inputs(task_set_input, fidelity_rollout_input),
        created_at=_TIME,
        digest=_DIGEST,
    )
    fidelity_input = artifact_input(
        store.artifacts.write_json(
            artifact_id=fidelity.fidelity_report_id,
            artifact_type="fidelity-report",
            envelope=fidelity,
            files={"fidelity-report.json": fidelity},
        )
    )
    rule = TeacherAcceptanceRule(
        schema_version=1,
        created_at=_TIME,
        inputs=(calibration_input,),
        code_revision="w12-test",
        acceptance_rule_id="teacher-rule",
        minimum_overall_score=0.8,
        required_calibration=calibration_input,
    )
    rule_input = artifact_input(
        store.artifacts.write_json(
            artifact_id=rule.acceptance_rule_id,
            artifact_type="sft-teacher-acceptance-rule",
            envelope=rule,
            files={"rule.json": rule},
        )
    )
    transcript = _transcript("teacher")
    transcript_payload = canonical_json_bytes(transcript)
    evidence = TeacherAcceptanceEvidence(
        schema_version=1,
        created_at=_TIME,
        inputs=_inputs(
            rollout_input,
            task_set_input,
            judgment_input,
            calibration_input,
            fidelity_input,
            rule_input,
        ),
        code_revision="w12-test",
        acceptance_evidence_id="teacher-evidence",
        rollout=rollout_input,
        task_set=task_set_input,
        task_set_tasks_sha256=task_set.tasks_sha256,
        task_set_inputs=task_set.inputs,
        task_id=task.task_id,
        task_sha256=sha256_json(task),
        judgment=judgment_input,
        calibration=calibration_input,
        fidelity_report=fidelity_input,
        acceptance_rule=rule_input,
        transcript_path="transcript.json",
        transcript_sha256=hashlib.sha256(transcript_payload).hexdigest(),
        accepted_at=_TIME,
    )
    store.artifacts.write(
        artifact_id=evidence.acceptance_evidence_id,
        artifact_type="sft-teacher-acceptance",
        envelope=evidence,
        files={
            "evidence.json": canonical_json_bytes(evidence),
            "transcript.json": transcript_payload,
        },
    )
    return _TeacherFixture(
        source=TeacherSFTSource(acceptance_evidence_id=evidence.acceptance_evidence_id),
        evidence=evidence,
        rollout_input=rollout_input,
        task_set_input=task_set_input,
        judgment_input=judgment_input,
        calibration_input=calibration_input,
        fidelity_input=fidelity_input,
        rule_input=rule_input,
        transcript=transcript,
    )


def _build(
    store: ProjectStore,
    *,
    production: tuple[ProductionSFTSource, ...] = (),
    teacher: tuple[TeacherSFTSource, ...] = (),
) -> SFTDatasetArtifact:
    """Build one SFT dataset through the public store-backed composition boundary."""
    return build_sft_dataset(
        store=store,
        production_sources=production,
        teacher_sources=teacher,
        spec=SFTBuildSpec(held_out_fraction=0.5, representative_sample_count=2),
        created_at=_TIME,
        code_revision="w12-test",
    )


def test_default_build_spec_reloads_and_rebuilds_without_identity_drift(tmp_path: Path) -> None:
    """Preserve the canonical default W12 spec and artifact identity across all-train support.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    store = _store(tmp_path)
    source = _write_production_source(store, "default-spec-compatibility")
    pre_change_payload = {
        "held_out_fraction": 0.2,
        "representative_sample_count": 3,
        "split_salt": "wmo-sft-split-v1",
    }
    spec = SFTBuildSpec.model_validate(pre_change_payload)
    first = build_sft_dataset(
        store=store,
        production_sources=(source,),
        teacher_sources=(),
        spec=spec,
        created_at=_TIME,
        code_revision="w12-default-compatibility-test",
    )
    write_sft_dataset(store, first)
    loaded = load_sft_dataset(store, first.dataset.dataset_id)
    rebuilt = build_sft_dataset(
        store=store,
        production_sources=(source,),
        teacher_sources=(),
        spec=spec,
        created_at=_TIME,
        code_revision="w12-default-compatibility-test",
    )

    assert spec.model_dump(mode="json", exclude_none=False) == pre_change_payload
    assert loaded.build_spec == spec
    assert rebuilt.dataset.dataset_id == first.dataset.dataset_id
    assert rebuilt.dataset.build_sha256 == first.dataset.build_sha256


def test_store_backed_sources_hydrate_transcripts_and_preserve_full_actions(tmp_path: Path) -> None:
    """Production and teacher rows come from verified evidence, not caller-owned transcripts."""
    store = _store(tmp_path)
    production = _write_production_source(store, "production", human_approval=True)
    teacher = _write_teacher_source(store)

    artifact = _build(store, production=(production,), teacher=(teacher.source,))

    assert {row.example.source.kind for row in artifact.rows} == {
        "production_trace",
        "teacher_rollout",
    }
    assert {row.example.target.content for row in artifact.rows} == {
        "Investigating production.",
        "Resolved production.",
        "Investigating teacher.",
        "Resolved teacher.",
    }
    assert "transcript" not in ProductionSFTSource.model_fields
    assert "transcript" not in TeacherSFTSource.model_fields
    assert "observed_overall_score" not in TeacherAcceptanceEvidence.model_fields
    assert teacher.task_set_input in artifact.dataset.inputs
    assert teacher.calibration_input in artifact.dataset.inputs
    assert teacher.evidence.task_set_tasks_sha256
    with pytest.raises(ValidationError):
        ProductionSFTSource.model_validate(
            {
                "acceptance_evidence_id": production.acceptance_evidence_id,
                "transcript": _transcript("caller-controlled").model_dump(mode="json"),
            }
        )


def test_store_backed_source_rejects_corrupt_transcript_and_cross_store_pointer(
    tmp_path: Path,
) -> None:
    """An accepted pointer cannot inject arbitrary transcript bytes or cross project boundaries."""
    source_store = _store(tmp_path / "source")
    source = _write_production_source(source_store, "corrupt")
    transcript_path = (
        source_store.artifacts.read(source.acceptance_evidence_id).directory / "transcript.json"
    )
    transcript_path.write_text('{"events":[]}\n', encoding="utf-8")

    with pytest.raises(SFTBuildError, match="not accepted evidence"):
        _build(source_store, production=(source,))

    other_store = _store(tmp_path / "other")
    with pytest.raises(SFTBuildError, match="not accepted evidence"):
        _build(other_store, production=(source,))

    teacher_source = _write_teacher_source(source_store)
    with pytest.raises(SFTBuildError, match="not accepted evidence"):
        _build(other_store, teacher=(teacher_source.source,))


def test_teacher_rejects_forged_score_and_recursively_unverifiable_calibration(
    tmp_path: Path,
) -> None:
    """Teacher acceptance recomputes score and invokes W6 recursive calibration verification."""
    store = _store(tmp_path)
    fixture = _write_teacher_source(store)
    original = Judgment.model_validate_json(
        store.artifacts.read_bytes(fixture.judgment_input.artifact_id, "judgment.json")
    )
    forged_dimension = original.dimensions[0].model_copy(update={"calibrated_score": 4.0})
    forged_judgment = original.model_copy(
        update={
            "judgment_id": "forged-teacher-judgment",
            "dimensions": (forged_dimension,),
            "overall_score": 0.8,
        }
    )
    forged_judgment_input = artifact_input(
        store.artifacts.write_json(
            artifact_id=forged_judgment.judgment_id,
            artifact_type="judgment",
            envelope=forged_judgment,
            files={"judgment.json": forged_judgment},
        )
    )
    forged_evidence = fixture.evidence.model_copy(
        update={
            "acceptance_evidence_id": "teacher-evidence-forged-score",
            "inputs": _inputs(
                fixture.rollout_input,
                fixture.task_set_input,
                forged_judgment_input,
                fixture.calibration_input,
                fixture.fidelity_input,
                fixture.rule_input,
            ),
            "judgment": forged_judgment_input,
        }
    )
    store.artifacts.write(
        artifact_id=forged_evidence.acceptance_evidence_id,
        artifact_type="sft-teacher-acceptance",
        envelope=forged_evidence,
        files={
            "evidence.json": canonical_json_bytes(forged_evidence),
            "transcript.json": canonical_json_bytes(fixture.transcript),
        },
    )
    with pytest.raises(SFTBuildError, match="calibrated dimension"):
        _build(
            store,
            teacher=(
                TeacherSFTSource(acceptance_evidence_id=forged_evidence.acceptance_evidence_id),
            ),
        )

    calibration = JudgeCalibration.model_validate_json(
        store.artifacts.read_bytes(fixture.calibration_input.artifact_id, "calibration.json")
    )
    forged_calibration = calibration.model_copy(
        update={
            "calibration_id": "forged-teacher-calibration",
            "out_of_fold_report_sha256": "b" * 64,
        }
    )
    forged_calibration_input = artifact_input(
        store.artifacts.write_json(
            artifact_id=forged_calibration.calibration_id,
            artifact_type="judge-calibration",
            envelope=forged_calibration,
            files={"calibration.json": forged_calibration},
        )
    )
    forged_rule = TeacherAcceptanceRule(
        schema_version=1,
        created_at=_TIME,
        inputs=(forged_calibration_input,),
        code_revision="w12-test",
        acceptance_rule_id="teacher-rule-forged-calibration",
        minimum_overall_score=0.8,
        required_calibration=forged_calibration_input,
    )
    forged_rule_input = artifact_input(
        store.artifacts.write_json(
            artifact_id=forged_rule.acceptance_rule_id,
            artifact_type="sft-teacher-acceptance-rule",
            envelope=forged_rule,
            files={"rule.json": forged_rule},
        )
    )
    forged_calibration_evidence = fixture.evidence.model_copy(
        update={
            "acceptance_evidence_id": "teacher-evidence-forged-calibration",
            "inputs": _inputs(
                fixture.rollout_input,
                fixture.task_set_input,
                fixture.judgment_input,
                forged_calibration_input,
                fixture.fidelity_input,
                forged_rule_input,
            ),
            "calibration": forged_calibration_input,
            "acceptance_rule": forged_rule_input,
        }
    )
    store.artifacts.write(
        artifact_id=forged_calibration_evidence.acceptance_evidence_id,
        artifact_type="sft-teacher-acceptance",
        envelope=forged_calibration_evidence,
        files={
            "evidence.json": canonical_json_bytes(forged_calibration_evidence),
            "transcript.json": canonical_json_bytes(fixture.transcript),
        },
    )
    with pytest.raises(SFTBuildError, match="recursive W6 provenance"):
        _build(
            store,
            teacher=(
                TeacherSFTSource(
                    acceptance_evidence_id=forged_calibration_evidence.acceptance_evidence_id
                ),
            ),
        )


def test_teacher_rejects_fidelity_report_bound_to_another_rollout(tmp_path: Path) -> None:
    """Teacher evidence cannot reuse an approved fidelity report from another rollout."""
    store = _store(tmp_path)
    fixture = _write_teacher_source(store, bind_fidelity_to_teacher_rollout=False)

    with pytest.raises(SFTBuildError, match="fidelity report does not bind the stored rollout"):
        _build(store, teacher=(fixture.source,))


def test_teacher_rejects_corrupt_full_task_case_and_preserves_task_set_lineage(
    tmp_path: Path,
) -> None:
    """Task IDs alone never authorize teacher data, task bytes and task-set inputs are verified."""
    store = _store(tmp_path)
    fixture = _write_teacher_source(store)
    task_path = store.artifacts.read(fixture.task_set_input.artifact_id).directory / "tasks.jsonl"
    task_path.write_text('{"task_id":"task-teacher"}\n', encoding="utf-8")

    with pytest.raises(SFTBuildError, match="not accepted evidence"):
        _build(store, teacher=(fixture.source,))


def test_ineligible_actions_remain_context_but_never_become_sft_targets(tmp_path: Path) -> None:
    """Verified transcript events retain context while excluding failed and unapproved targets."""
    store = _store(tmp_path)
    unapproved = AssistantActionEvent(
        action=AssistantAction(content="This action was not approved."), approved=False
    )
    failed = AssistantActionEvent(action=AssistantAction(content="This action failed."))
    accepted = AssistantActionEvent(action=AssistantAction(content="This action is approved."))
    source = _write_production_source(
        store,
        "context",
        transcript=SFTTranscript(
            events=(
                SFTMessage(role="user", content="Complete the request."),
                unapproved,
                failed,
                InfrastructureFailureEvent(
                    action_index=2,
                    failure=StructuredFailure(
                        code=FailureCode.TIMEOUT,
                        message="tool transport timed out",
                    ),
                ),
                accepted,
            )
        ),
    )

    artifact = _build(store, production=(source,))

    assert [row.example.target for row in artifact.rows] == [accepted.action]
    assert artifact.rows[0].example.history[-1] == failed
    assert {item.reason for item in artifact.inspection.exclusions} == {
        "infrastructure_failure",
        "unapproved_action",
    }


def test_shared_fingerprints_union_lineages_and_build_digest_is_order_independent(
    tmp_path: Path,
) -> None:
    """Keep duplicate verified examples in one split and preserve source-order invariance."""
    store = _store(tmp_path)
    shared_transcript = SFTTranscript(
        events=(
            SFTMessage(role="user", content="Resolve the shared request."),
            AssistantActionEvent(action=AssistantAction(content="Resolved the shared request.")),
        )
    )
    first = _write_production_source(
        store,
        "shared-one",
        task="Resolve the shared request.",
        transcript=shared_transcript,
    )
    second = _write_production_source(
        store,
        "shared-two",
        task="Resolve the shared request.",
        transcript=shared_transcript,
    )
    distinct = _write_production_source(store, "distinct")

    forward = _build(store, production=(first, second, distinct))
    reversed_build = _build(store, production=(distinct, second, first))

    references = {item.source_id: item for item in forward.sources}
    shared_partition = next(
        item
        for item in forward.partitions
        if references["trace-shared-one"].leakage_group_id in item.leakage_group_ids
    )
    assert references["trace-shared-two"].leakage_group_id in shared_partition.leakage_group_ids
    assert any(
        item.reason == "duplicate_normalized_example" for item in forward.inspection.exclusions
    )
    assert forward.dataset.build_sha256 == reversed_build.dataset.build_sha256
    assert forward.rows == reversed_build.rows

    row = forward.rows[0]
    with pytest.raises(SFTBuildError, match="appears in both"):
        ensure_no_cross_split_fingerprints((row, row.model_copy(update={"partition": "held_out"})))


def test_frozen_dataset_round_trips_only_after_store_backed_build(tmp_path: Path) -> None:
    """A persisted SFT artifact retains verified source manifests and rejects later corruption."""
    store = _store(tmp_path)
    artifact = _build(
        store,
        production=(
            _write_production_source(store, "one"),
            _write_production_source(store, "two"),
        ),
    )
    written = write_sft_dataset(store, artifact)
    assert load_sft_dataset(store, written.dataset.dataset_id) == written
    assert all(reference.source_artifact in written.dataset.inputs for reference in written.sources)
    assert all(
        reference.acceptance_evidence in written.dataset.inputs for reference in written.sources
    )
