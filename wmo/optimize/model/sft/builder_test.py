"""Focused tests for accepted, leakage-safe frozen SFT dataset construction."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest

from wmo.common.core.artifacts import (
    ArtifactInput,
    FailureCode,
    SourceIdentity,
    StructuredFailure,
    sha256_json,
)
from wmo.common.evaluations import FidelityFailure, FidelityReport
from wmo.common.judging import DimensionJudgment, DimensionScoreMap, JudgeCalibration, Judgment
from wmo.common.models import AssistantAction, ModelSnapshot, OperationEconomics, ToolCall
from wmo.common.project import ProjectConfig, ProjectStore
from wmo.common.rollouts import (
    RolloutArtifact,
    RolloutEventKind,
    RolloutSpan,
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
from wmo.optimize.model.sft.contracts import (
    AssistantActionEvent,
    HumanApproval,
    InfrastructureFailureEvent,
    ProductionAcceptanceEvidence,
    ProductionAcceptanceRule,
    ProductionSFTSource,
    RolloutExampleSource,
    SFTDatasetArtifact,
    SFTMessage,
    SFTTranscript,
    TeacherAcceptanceEvidence,
    TeacherAcceptanceRule,
    TeacherSFTSource,
    ToolEvent,
    TraceExampleSource,
)

_DIGEST = "a" * 64
_TIME = datetime(2026, 8, 11, tzinfo=UTC)


def _inputs(*items: ArtifactInput) -> tuple[ArtifactInput, ...]:
    return tuple(sorted(items, key=lambda item: item.artifact_id))


def _model() -> ModelSnapshot:
    return ModelSnapshot(
        provider="test",
        model_id="teacher-v1",
        capabilities_sha256=_DIGEST,
        connection_sha256=_DIGEST,
    )


def _trace(trace_id: str, conversation_id: str, task: str) -> Trace:
    return Trace(
        trace_id=trace_id,
        conversation_id=conversation_id,
        task=task,
        spans=(
            TraceSpan(
                span_id=f"span-{trace_id}",
                name="agent.model_call",
                started_at=_TIME,
                ended_at=_TIME + timedelta(seconds=1),
            ),
        ),
        outcome=TraceOutcome(status="success", outcome_name="resolved"),
        source=TraceSource(
            identity=SourceIdentity(kind="production", source_id=f"source-{trace_id}"),
            semantic_convention_version="1.0",
        ),
    )


def _tool_action() -> AssistantAction:
    return AssistantAction(
        content="I will verify the order and issue the refund.",
        tool_calls=(
            ToolCall(call_id="lookup-1", name="lookup_order", arguments={"order_id": "o-17"}),
            ToolCall(call_id="refund-1", name="issue_refund", arguments={"order_id": "o-17"}),
        ),
    )


def _transcript(*, approved: bool = True, failed: bool = False) -> SFTTranscript:
    events = [
        SFTMessage(role="system", content="Follow the support policy."),
        SFTMessage(role="user", content="Please refund order o-17."),
        AssistantActionEvent(action=_tool_action(), approved=approved),
        ToolEvent(tool_call_id="lookup-1", tool_name="lookup_order", content="order is eligible"),
        ToolEvent(tool_call_id="refund-1", tool_name="issue_refund", content="refund issued"),
    ]
    if failed:
        events.append(
            InfrastructureFailureEvent(
                action_index=2,
                failure=StructuredFailure(
                    code=FailureCode.TIMEOUT, message="tool transport timed out"
                ),
            )
        )
    else:
        events.append(
            AssistantActionEvent(
                action=AssistantAction(content="Your refund is complete."), approved=approved
            )
        )
    return SFTTranscript(events=tuple(events))


def _production_source(
    tag: str,
    *,
    task: str = "Refund order o-17.",
    approved: bool = True,
    failed: bool = False,
) -> ProductionSFTSource:
    trace = _trace(f"trace-{tag}", f"conversation-{tag}", task)
    rule = ProductionAcceptanceRule(
        schema_version=1,
        created_at=_TIME,
        code_revision="w12-test",
        acceptance_rule_id=f"production-rule-{tag}",
        accepted_outcomes=("resolved",),
        allow_human_approval=True,
    )
    evidence = ProductionAcceptanceEvidence(
        schema_version=1,
        created_at=_TIME,
        inputs=_inputs(
            ArtifactInput(artifact_id=rule.acceptance_rule_id, sha256=sha256_json(rule))
        ),
        code_revision="w12-test",
        acceptance_evidence_id=f"production-evidence-{tag}",
        trace_id=trace.trace_id,
        trace_sha256=sha256_json(trace),
        acceptance_rule_id=rule.acceptance_rule_id,
        acceptance_rule_sha256=sha256_json(rule),
        decision="trusted_outcome",
        outcome_sha256=sha256_json(trace.outcome),
        accepted_at=_TIME,
    )
    return ProductionSFTSource(
        trace=trace,
        transcript=_transcript(approved=approved, failed=failed),
        acceptance_rule=rule,
        acceptance_evidence=evidence,
    )


def _human_approved_production_source(tag: str) -> ProductionSFTSource:
    """Build a locally approved production source with complete immutable references."""
    source = _production_source(tag)
    approval = HumanApproval(
        schema_version=1,
        created_at=_TIME,
        code_revision="w12-test",
        approval_id=f"human-approval-{tag}",
        trace_id=source.trace.trace_id,
        approved_at=_TIME,
    )
    evidence = ProductionAcceptanceEvidence(
        schema_version=1,
        created_at=_TIME,
        inputs=_inputs(
            ArtifactInput(
                artifact_id=source.acceptance_rule.acceptance_rule_id,
                sha256=sha256_json(source.acceptance_rule),
            ),
            ArtifactInput(artifact_id=approval.approval_id, sha256=sha256_json(approval)),
        ),
        code_revision="w12-test",
        acceptance_evidence_id=f"human-evidence-{tag}",
        trace_id=source.trace.trace_id,
        trace_sha256=sha256_json(source.trace),
        acceptance_rule_id=source.acceptance_rule.acceptance_rule_id,
        acceptance_rule_sha256=sha256_json(source.acceptance_rule),
        decision="human_approval",
        human_approval_id=approval.approval_id,
        human_approval_sha256=sha256_json(approval),
        accepted_at=_TIME,
    )
    return source.model_copy(update={"acceptance_evidence": evidence, "human_approval": approval})


def _task_case(tag: str, instruction: str) -> TaskCase:
    return TaskCase(
        task_id=f"task-{tag}",
        lineage_group_id=f"task-lineage-{tag}",
        partition="fit",
        instruction=instruction,
        workload_weight=1.0,
        source_trace_ids=(f"teacher-trace-{tag}",),
    )


def _task_set(task: TaskCase) -> TaskSet:
    return TaskSet(
        schema_version=1,
        created_at=_TIME,
        code_revision="w12-test",
        task_set_id=f"task-set-{task.task_id.removeprefix('task-')}",
        task_ids=(task.task_id,),
        tasks_path="tasks.jsonl",
        tasks_sha256=_DIGEST,
    )


def _rollout(tag: str, task_id: str) -> RolloutArtifact:
    model = _model()
    return RolloutArtifact(
        schema_version=1,
        created_at=_TIME,
        code_revision="w12-test",
        artifact_id=f"rollout-artifact-{tag}",
        simulation_id=f"simulation-{tag}",
        cell_id=f"cell-{tag}",
        mode=SimulationMode.WORLD_MODEL,
        rollout_id=f"rollout-{tag}",
        trace_id=f"teacher-trace-{tag}",
        evidence_source="world_model",
        source_run_id=f"run-{tag}",
        task_id=task_id,
        candidate=model,
        agent_id="customer-agent",
        simulator=WorldModelSimulatorSnapshot(
            simulator_id="world-model-v1",
            prompt_id="world-prompt-v1",
            world_model=model,
        ),
        world_model=model,
        seed=7,
        repeat=0,
        spans=(
            RolloutSpan(
                span_id=f"rollout-span-{tag}",
                kind=RolloutEventKind.AGENT_MODEL_CALL,
                started_at=_TIME,
                ended_at=_TIME + timedelta(seconds=1),
                model=model,
            ),
        ),
        stop_reason=StopReason.COMPLETED,
        candidate_economics=OperationEconomics(),
        simulation_spec_sha256=_DIGEST,
    )


def _calibration(
    tag: str, *, status: Literal["human_calibrated", "insufficient"] = "human_calibrated"
) -> JudgeCalibration:
    return JudgeCalibration(
        schema_version=1,
        created_at=_TIME,
        code_revision="w12-test",
        calibration_id=f"calibration-{tag}",
        rubric_id=f"rubric-{tag}",
        judge_model=_model(),
        judge_prompt_id="judge-prompt-v1",
        judge_prompt_sha256=_DIGEST,
        label_set_id=f"labels-{tag}",
        calibration_lineage_ids=(f"lineage-{tag}",),
        excluded_router_held_out_lineage_ids=(),
        validation_method="grouped_k_fold",
        out_of_fold_report_id=f"calibration-report-{tag}",
        out_of_fold_report_sha256=_DIGEST,
        score_maps=(
            DimensionScoreMap(
                dimension_id="quality",
                calibrated_scores=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
            ),
        ),
        label_count=1,
        status=status,
        approved_at=_TIME if status == "human_calibrated" else None,
    )


def _teacher_source(
    tag: str,
    *,
    task: str = "Refund order o-17.",
    score: float = 1.0,
    minimum_score: float = 0.8,
    calibration_status: Literal["human_calibrated", "insufficient"] = "human_calibrated",
    fidelity_status: Literal["approved", "rejected", "insufficient"] = "approved",
) -> TeacherSFTSource:
    task_case = _task_case(tag, task)
    task_set = _task_set(task_case)
    rollout = _rollout(tag, task_case.task_id)
    calibration = _calibration(tag, status=calibration_status)
    raw_score = 5 if score == 1.0 else 4
    judgment = Judgment(
        schema_version=1,
        created_at=_TIME,
        code_revision="w12-test",
        judgment_id=f"judgment-{tag}",
        rollout_id=rollout.rollout_id,
        rubric_id=calibration.rubric_id,
        calibration_id=calibration.calibration_id,
        judge_model=_model(),
        judge_prompt_id="judge-prompt-v1",
        judge_prompt_sha256=_DIGEST,
        dimensions=(
            DimensionJudgment(
                dimension_id="quality",
                raw_score=raw_score,
                calibrated_score=score * 5,
                evidence_span_ids=(f"rollout-span-{tag}",),
                feedback="The rollout completed the requested action.",
            ),
        ),
        overall_score=score,
    )
    if fidelity_status == "approved":
        usable_overlap_count, score_mae = 8, 0.08
    elif fidelity_status == "rejected":
        usable_overlap_count, score_mae = 8, 0.12
    else:
        usable_overlap_count, score_mae = 7, None
    failures = tuple(
        FidelityFailure(
            cell_id=f"fidelity-cell-{tag}-{index}",
            failure=StructuredFailure(code=FailureCode.TIMEOUT, message="overlap unavailable"),
        )
        for index in range(usable_overlap_count, 10)
    )
    fidelity = FidelityReport(
        schema_version=1,
        created_at=_TIME,
        code_revision="w12-test",
        fidelity_report_id=f"fidelity-{tag}",
        protocol_sha256=_DIGEST,
        overlap_cell_ids=tuple(f"fidelity-cell-{tag}-{index}" for index in range(10)),
        planned_overlap_count=10,
        usable_overlap_count=usable_overlap_count,
        failed_overlap_count=len(failures),
        score_mae=score_mae,
        failures=failures,
        gate_id=f"fidelity-gate-{tag}",
        gate_sha256=_DIGEST,
        status=fidelity_status,
        approved_at=_TIME if fidelity_status == "approved" else None,
    )
    rule = TeacherAcceptanceRule(
        schema_version=1,
        created_at=_TIME,
        code_revision="w12-test",
        acceptance_rule_id=f"teacher-rule-{tag}",
        minimum_overall_score=minimum_score,
        required_calibration_id=calibration.calibration_id,
    )
    evidence = TeacherAcceptanceEvidence(
        schema_version=1,
        created_at=_TIME,
        inputs=_inputs(
            ArtifactInput(artifact_id=rollout.rollout_id, sha256=sha256_json(rollout)),
            ArtifactInput(artifact_id=judgment.judgment_id, sha256=sha256_json(judgment)),
            ArtifactInput(artifact_id=calibration.calibration_id, sha256=sha256_json(calibration)),
            ArtifactInput(
                artifact_id=fidelity.fidelity_report_id,
                sha256=sha256_json(fidelity),
            ),
            ArtifactInput(artifact_id=rule.acceptance_rule_id, sha256=sha256_json(rule)),
        ),
        code_revision="w12-test",
        acceptance_evidence_id=f"teacher-evidence-{tag}",
        rollout_id=rollout.rollout_id,
        rollout_sha256=sha256_json(rollout),
        judgment_id=judgment.judgment_id,
        judgment_sha256=sha256_json(judgment),
        calibration_id=calibration.calibration_id,
        calibration_sha256=sha256_json(calibration),
        fidelity_report_id=fidelity.fidelity_report_id,
        fidelity_report_sha256=sha256_json(fidelity),
        acceptance_rule_id=rule.acceptance_rule_id,
        acceptance_rule_sha256=sha256_json(rule),
        observed_overall_score=score,
        accepted_at=_TIME,
    )
    return TeacherSFTSource(
        rollout=rollout,
        task=task_case,
        task_set=task_set,
        transcript=_transcript(),
        acceptance_rule=rule,
        acceptance_evidence=evidence,
        judgment=judgment,
        calibration=calibration,
        fidelity=fidelity,
    )


def _build(
    production: tuple[ProductionSFTSource, ...] = (),
    teacher: tuple[TeacherSFTSource, ...] = (),
    *,
    created_at: datetime = _TIME,
) -> SFTDatasetArtifact:
    return build_sft_dataset(
        production_sources=production,
        teacher_sources=teacher,
        spec=SFTBuildSpec(held_out_fraction=0.5, representative_sample_count=2),
        created_at=created_at,
        code_revision="w12-test",
    )


def test_production_acceptance_filters_unapproved_failed_and_observation_events() -> None:
    """Only accepted assistant actions become targets, never tools, failures, or unapproved turns.

    The source ledger still records why excluded events were not targets.
    """
    valid = _production_source("valid")
    invalid_evidence = _production_source("invalid")
    invalid_evidence = invalid_evidence.model_copy(
        update={
            "acceptance_evidence": invalid_evidence.acceptance_evidence.model_copy(
                update={"trace_sha256": "f" * 64}
            )
        }
    )
    unapproved = _production_source("unapproved", approved=False)
    failed = _production_source("failed", failed=True)

    artifact = _build((valid, invalid_evidence, unapproved, failed))

    assert {row.example.target.content for row in artifact.rows} == {
        "I will verify the order and issue the refund.",
        "Your refund is complete.",
    }
    reasons = {exclusion.reason for exclusion in artifact.inspection.exclusions}
    assert "invalid_production_acceptance" in reasons
    assert "unapproved_action" in reasons
    assert "infrastructure_failure" in reasons
    assert "observation_context_only" in reasons
    assert all(row.example.source.kind == "production_trace" for row in artifact.rows)


def test_ineligible_assistant_actions_remain_visible_context_for_later_targets() -> None:
    """An excluded assistant action is not learned, but later visible context remains faithful."""
    source = _production_source("context")
    excluded_action = AssistantActionEvent(
        action=AssistantAction(content="This action was not approved."), approved=False
    )
    accepted_action = AssistantActionEvent(
        action=AssistantAction(content="This approved action follows it.")
    )
    source = source.model_copy(
        update={
            "transcript": SFTTranscript(
                events=(
                    SFTMessage(role="user", content="Complete the request."),
                    excluded_action,
                    accepted_action,
                )
            )
        }
    )

    artifact = _build((source,))

    assert len(artifact.rows) == 1
    assert artifact.rows[0].example.target == accepted_action.action
    assert artifact.rows[0].example.history[-1] == excluded_action


def test_production_acceptance_requires_verified_outcome_or_human_references() -> None:
    """Every production evidence branch rejects missing or mismatched immutable references."""
    valid_human_approval = _human_approved_production_source("human-valid")
    missing_human_approval = _human_approved_production_source("human-missing")
    missing_human_approval = missing_human_approval.model_copy(update={"human_approval": None})
    mismatched_human_approval = _human_approved_production_source("human-hash")
    mismatched_human_approval = mismatched_human_approval.model_copy(
        update={
            "acceptance_evidence": mismatched_human_approval.acceptance_evidence.model_copy(
                update={"human_approval_sha256": "d" * 64}
            )
        }
    )
    mismatched_trace = _production_source("trace-hash")
    mismatched_trace = mismatched_trace.model_copy(
        update={
            "acceptance_evidence": mismatched_trace.acceptance_evidence.model_copy(
                update={"trace_sha256": "d" * 64}
            )
        }
    )
    mismatched_rule = _production_source("rule-hash")
    mismatched_rule = mismatched_rule.model_copy(
        update={
            "acceptance_evidence": mismatched_rule.acceptance_evidence.model_copy(
                update={"acceptance_rule_sha256": "d" * 64}
            )
        }
    )
    mismatched_outcome = _production_source("outcome-hash")
    mismatched_outcome = mismatched_outcome.model_copy(
        update={
            "acceptance_evidence": mismatched_outcome.acceptance_evidence.model_copy(
                update={"outcome_sha256": "d" * 64}
            )
        }
    )
    disallowed_human_approval = _human_approved_production_source("human-disallowed")
    disallowed_rule = disallowed_human_approval.acceptance_rule.model_copy(
        update={"allow_human_approval": False}
    )
    approval = disallowed_human_approval.human_approval
    assert approval is not None
    disallowed_human_approval = disallowed_human_approval.model_copy(
        update={
            "acceptance_rule": disallowed_rule,
            "acceptance_evidence": disallowed_human_approval.acceptance_evidence.model_copy(
                update={
                    "acceptance_rule_sha256": sha256_json(disallowed_rule),
                    "inputs": _inputs(
                        ArtifactInput(
                            artifact_id=disallowed_rule.acceptance_rule_id,
                            sha256=sha256_json(disallowed_rule),
                        ),
                        ArtifactInput(
                            artifact_id=approval.approval_id,
                            sha256=sha256_json(approval),
                        ),
                    ),
                }
            ),
        }
    )
    untrusted_outcome = _production_source("untrusted-outcome")
    alternate_outcome = TraceOutcome(status="success", outcome_name="declined")
    alternate_trace = untrusted_outcome.trace.model_copy(update={"outcome": alternate_outcome})
    untrusted_outcome = untrusted_outcome.model_copy(
        update={
            "trace": alternate_trace,
            "acceptance_evidence": untrusted_outcome.acceptance_evidence.model_copy(
                update={
                    "trace_sha256": sha256_json(alternate_trace),
                    "outcome_sha256": sha256_json(alternate_outcome),
                }
            ),
        }
    )

    artifact = _build(
        (
            valid_human_approval,
            missing_human_approval,
            mismatched_human_approval,
            mismatched_trace,
            mismatched_rule,
            mismatched_outcome,
            disallowed_human_approval,
            untrusted_outcome,
        )
    )

    assert {
        row.example.source.trace_id
        for row in artifact.rows
        if isinstance(row.example.source, TraceExampleSource)
    } == {"trace-human-valid"}
    excluded_ids = {
        exclusion.source_id
        for exclusion in artifact.inspection.exclusions
        if exclusion.reason == "invalid_production_acceptance"
    }
    assert excluded_ids == {
        "trace-human-missing",
        "trace-human-hash",
        "trace-trace-hash",
        "trace-rule-hash",
        "trace-outcome-hash",
        "trace-human-disallowed",
        "trace-untrusted-outcome",
    }


def test_teacher_acceptance_requires_every_immutable_prerequisite() -> None:
    """Every teacher evidence reference and acceptance gate must be proven before training."""
    valid = _teacher_source("valid")
    bad_rollout_hash = _teacher_source("rollout-hash")
    bad_rollout_hash = bad_rollout_hash.model_copy(
        update={
            "acceptance_evidence": bad_rollout_hash.acceptance_evidence.model_copy(
                update={"rollout_sha256": "c" * 64}
            )
        }
    )
    bad_judgment_hash = _teacher_source("judgment-hash")
    bad_judgment_hash = bad_judgment_hash.model_copy(
        update={
            "acceptance_evidence": bad_judgment_hash.acceptance_evidence.model_copy(
                update={"judgment_sha256": "c" * 64}
            )
        }
    )
    bad_calibration_hash = _teacher_source("calibration-hash")
    bad_calibration_hash = bad_calibration_hash.model_copy(
        update={
            "acceptance_evidence": bad_calibration_hash.acceptance_evidence.model_copy(
                update={"calibration_sha256": "c" * 64}
            )
        }
    )
    bad_fidelity_hash = _teacher_source("fidelity-hash")
    bad_fidelity_hash = bad_fidelity_hash.model_copy(
        update={
            "acceptance_evidence": bad_fidelity_hash.acceptance_evidence.model_copy(
                update={"fidelity_report_sha256": "c" * 64}
            )
        }
    )
    bad_rule_hash = _teacher_source("rule-hash")
    bad_rule_hash = bad_rule_hash.model_copy(
        update={
            "acceptance_evidence": bad_rule_hash.acceptance_evidence.model_copy(
                update={"acceptance_rule_sha256": "c" * 64}
            )
        }
    )
    insufficient_calibration = _teacher_source("calibration", calibration_status="insufficient")
    rejected_fidelity = _teacher_source("fidelity-rejected", fidelity_status="rejected")
    insufficient_fidelity = _teacher_source("fidelity-insufficient", fidelity_status="insufficient")
    low_score = _teacher_source("score", score=0.8, minimum_score=0.9)
    unfinished = _teacher_source("unfinished")
    unfinished = unfinished.model_copy(
        update={
            "rollout": unfinished.rollout.model_copy(
                update={"stop_reason": StopReason.MAXIMUM_STEPS}
            )
        }
    )
    mismatched_task = _teacher_source("task-mismatch")
    mismatched_task = mismatched_task.model_copy(
        update={"task": _task_case("different", "A different canonical task.")}
    )
    task_outside_set = _teacher_source("task-outside-set")
    task_outside_set = task_outside_set.model_copy(
        update={
            "task_set": task_outside_set.task_set.model_copy(
                update={"task_ids": ("task-not-this-one",)}
            )
        }
    )
    unproven = _teacher_source("unproven")
    unproven = unproven.model_copy(
        update={
            "acceptance_evidence": unproven.acceptance_evidence.model_copy(update={"inputs": ()})
        }
    )

    artifact = _build(
        teacher=(
            valid,
            bad_rollout_hash,
            bad_judgment_hash,
            bad_calibration_hash,
            bad_fidelity_hash,
            bad_rule_hash,
            insufficient_calibration,
            rejected_fidelity,
            insufficient_fidelity,
            low_score,
            unfinished,
            mismatched_task,
            task_outside_set,
            unproven,
        )
    )

    assert {
        row.example.source.rollout_id
        for row in artifact.rows
        if isinstance(row.example.source, RolloutExampleSource)
    } == {"rollout-valid"}
    assert "task-set-valid" in {item.artifact_id for item in artifact.dataset.inputs}
    excluded_ids = {
        exclusion.source_id
        for exclusion in artifact.inspection.exclusions
        if exclusion.reason == "invalid_teacher_acceptance"
    }
    assert excluded_ids == {
        "rollout-rollout-hash",
        "rollout-judgment-hash",
        "rollout-calibration-hash",
        "rollout-fidelity-hash",
        "rollout-rule-hash",
        "rollout-calibration",
        "rollout-fidelity-rejected",
        "rollout-fidelity-insufficient",
        "rollout-score",
        "rollout-unfinished",
        "rollout-task-mismatch",
        "rollout-task-outside-set",
        "rollout-unproven",
    }


def test_shared_fingerprints_union_lineages_before_split_and_deduplicate_globally() -> None:
    """Identical context-target content cannot leak across partitions and keeps one row globally."""
    first = _production_source("one")
    second = _production_source("two")
    distinct = _production_source("three", task="Cancel order o-17.")

    artifact = _build((first, second, distinct))

    references = {source.source_id: source for source in artifact.sources}
    first_group = references["trace-one"].leakage_group_id
    second_group = references["trace-two"].leakage_group_id
    shared_partition = next(
        partition for partition in artifact.partitions if first_group in partition.leakage_group_ids
    )
    assert second_group in shared_partition.leakage_group_ids
    assert any(
        exclusion.reason == "duplicate_normalized_example"
        for exclusion in artifact.inspection.exclusions
    )
    fingerprints_by_partition: dict[str, set[str]] = {"train": set(), "held_out": set()}
    for row in artifact.rows:
        fingerprints_by_partition[row.partition].add(row.fingerprint)
    assert not fingerprints_by_partition["train"].intersection(
        fingerprints_by_partition["held_out"]
    )


def test_cross_split_fingerprint_is_explicitly_rejected() -> None:
    """Corrupt rows cannot place one normalized example in both train and held-out data."""
    row = _build((_production_source("cross-split"),)).rows[0]
    held_out_copy = row.model_copy(update={"partition": "held_out"})

    with pytest.raises(SFTBuildError, match="appears in both"):
        ensure_no_cross_split_fingerprints((row, held_out_copy))


def test_same_sources_produce_the_same_digest_regardless_of_input_order_or_build_time() -> None:
    """The semantic digest is stable while materialization time remains provenance."""
    first = _production_source("first")
    second = _production_source("second", task="Cancel order o-17.")

    forward = _build((first, second), created_at=_TIME)
    reversed_build = _build((second, first), created_at=_TIME + timedelta(days=1))

    assert forward.dataset.build_sha256 == reversed_build.dataset.build_sha256
    assert forward.dataset.dataset_id == reversed_build.dataset.dataset_id
    assert forward.rows == reversed_build.rows


def test_frozen_artifact_contains_provenance_samples_exclusions_and_training_gate(
    tmp_path: Path,
) -> None:
    """Only a prior accepted immutable artifact can be loaded for later training consumption."""
    accepted = _build((_production_source("artifact"), _production_source("other", task="Other.")))
    store = ProjectStore(tmp_path / ".wmo", "sft-project")
    store.initialize(ProjectConfig(project_id="sft-project"))

    written = write_sft_dataset(store, accepted)
    loaded = load_sft_dataset(store, written.dataset.dataset_id)
    stored = store.artifacts.read(written.dataset.dataset_id)
    metadata = written.metadata()

    assert loaded == written
    assert {file.path for file in stored.manifest.files} == {"dataset.json", "examples.jsonl"}
    assert metadata.sources
    assert metadata.partitions
    assert metadata.inspection.exclusions
    assert metadata.representative_samples
    source_reference = next(source for source in metadata.sources if source.accepted)
    assert source_reference.acceptance_evidence_id == "production-evidence-artifact"
    assert source_reference.acceptance_evidence_sha256 == sha256_json(
        _production_source("artifact").acceptance_evidence
    )
    assert {item.artifact_id for item in metadata.dataset.inputs} == {
        "production-evidence-artifact",
        "production-evidence-other",
        "production-rule-artifact",
        "production-rule-other",
    }
    assert metadata.inspection.dataset_id == metadata.dataset.dataset_id
    assert all(sample in written.rows for sample in metadata.representative_samples)

    invalid = _production_source("nope")
    invalid = invalid.model_copy(
        update={
            "acceptance_evidence": invalid.acceptance_evidence.model_copy(
                update={"acceptance_rule_sha256": "d" * 64}
            )
        }
    )
    insufficient = _build((invalid,))
    write_sft_dataset(store, insufficient)
    with pytest.raises(ValueError, match="insufficient"):
        load_sft_dataset(store, insufficient.dataset.dataset_id)
    assert (
        load_sft_dataset(
            store, insufficient.dataset.dataset_id, require_accepted=False
        ).dataset.status
        == "insufficient"
    )
