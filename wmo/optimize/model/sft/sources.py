"""Store-backed acceptance-chain verification for frozen SFT source evidence."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ValidationError

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    Sha256,
    canonical_json_bytes,
    sha256_json,
    sorted_unique_inputs,
    stable_id,
)
from wmo.common.evaluations import FidelityReport
from wmo.common.judging import (
    CalibrationError,
    JudgeCalibration,
    Judgment,
    Rubric,
    verify_persisted_calibration,
)
from wmo.common.judging.provenance import JudgingProvenanceError, read_artifact_json
from wmo.common.project import ArtifactCorruptionError, ProjectStore, artifact_input
from wmo.common.rollouts import RolloutArtifact
from wmo.common.tasks import TaskCase, load_task_set
from wmo.common.traces import Trace, load_trace_dataset
from wmo.optimize.model.sft.contracts import (
    HumanApproval,
    InfrastructureFailureEvent,
    ProductionAcceptanceEvidence,
    ProductionAcceptanceRule,
    ProductionSFTSource,
    RolloutExampleSource,
    RuntimeInteractionExampleSource,
    SFTContextEvent,
    SFTSourceReference,
    SFTTranscript,
    TeacherAcceptanceEvidence,
    TeacherAcceptanceRule,
    TeacherSFTSource,
    TraceExampleSource,
)


class SFTSourceVerificationError(ValueError):
    """Raised when an SFT source cannot be proven from one project-local artifact store."""


@dataclass(frozen=True)
class PreparedSFTSource:
    """One accepted source hydrated exclusively from verified immutable artifact bytes."""

    kind: Literal["production_trace", "teacher_rollout", "runtime_interaction"]
    source_id: str
    source_artifact: ArtifactInput
    source_record_sha256: Sha256
    leakage_group_id: ArtifactId
    task: str
    transcript_events: tuple[
        SFTContextEvent | InfrastructureFailureEvent,
        ...,
    ]
    example_source: TraceExampleSource | RolloutExampleSource | RuntimeInteractionExampleSource
    score: float | None
    direct_inputs: tuple[ArtifactInput, ...]
    acceptance_rule_id: ArtifactId | None
    acceptance_evidence_id: ArtifactId | None
    acceptance_evidence: ArtifactInput | None
    target_action_indexes: tuple[int, ...] | None = None

    def reference(self) -> SFTSourceReference:
        """Return the auditable immutable-source record for a frozen dataset manifest."""
        return SFTSourceReference(
            kind=self.kind,
            source_id=self.source_id,
            source_artifact=self.source_artifact,
            source_record_sha256=self.source_record_sha256,
            leakage_group_id=self.leakage_group_id,
            acceptance_evidence=self.acceptance_evidence,
            accepted=True,
        )


def resolve_production_source(
    store: ProjectStore, source: ProductionSFTSource
) -> PreparedSFTSource:
    """Hydrate one accepted production source from its immutable acceptance artifact.

    Args:
        store: Project store that owns every acceptance and trace artifact.
        source: A caller-supplied pointer, never caller-supplied trace or transcript data.

    Returns:
        The verified source ready for leakage partitioning and example extraction.

    Raises:
        SFTSourceVerificationError: The pointer, evidence chain, source trace, or transcript is
            absent, corrupt, cross-bound, or not accepted.
    """
    evidence, evidence_input = _read_json(
        store,
        artifact_id=source.acceptance_evidence_id,
        artifact_type="sft-production-acceptance",
        relative_path="evidence.json",
        model_type=ProductionAcceptanceEvidence,
    )
    if evidence.acceptance_evidence_id != source.acceptance_evidence_id:
        raise SFTSourceVerificationError("production acceptance evidence has the wrong identity")
    trace, trace_input = _load_trace(store, evidence)
    rule, rule_input = _read_json(
        store,
        artifact_id=evidence.acceptance_rule.artifact_id,
        artifact_type="sft-production-acceptance-rule",
        relative_path="rule.json",
        model_type=ProductionAcceptanceRule,
        expected_input=evidence.acceptance_rule,
    )
    if rule.acceptance_rule_id != rule_input.artifact_id:
        raise SFTSourceVerificationError("production acceptance rule has the wrong identity")
    approval_input = _verify_production_decision(store, evidence, trace, trace_input, rule)
    transcript = _load_transcript(
        store,
        evidence_id=evidence_input.artifact_id,
        transcript_path=evidence.transcript_path,
        transcript_sha256=evidence.transcript_sha256,
    )
    lineage_key = trace.conversation_id or trace.trace_id
    source_artifact = trace_input
    return PreparedSFTSource(
        kind="production_trace",
        source_id=trace.trace_id,
        source_artifact=source_artifact,
        source_record_sha256=sha256_json(trace),
        leakage_group_id=stable_id(
            "sft-lineage",
            {"kind": "production_trace", "source_lineage": lineage_key},
        ),
        task=trace.task,
        transcript_events=transcript.events,
        example_source=TraceExampleSource(
            trace_id=trace.trace_id,
            acceptance_evidence=evidence_input,
        ),
        score=None,
        direct_inputs=_sorted_inputs(
            (trace_input, rule_input, evidence_input)
            if approval_input is None
            else (trace_input, rule_input, approval_input, evidence_input)
        ),
        acceptance_rule_id=rule.acceptance_rule_id,
        acceptance_evidence_id=evidence.acceptance_evidence_id,
        acceptance_evidence=evidence_input,
    )


def resolve_teacher_source(store: ProjectStore, source: TeacherSFTSource) -> PreparedSFTSource:
    """Hydrate one accepted teacher source from its immutable acceptance artifact.

    Args:
        store: Project store that owns every rollout, task, judge, and acceptance artifact.
        source: A caller-supplied pointer, never caller-supplied rollout, score, or transcript data.

    Returns:
        The verified source ready for leakage partitioning and example extraction.

    Raises:
        SFTSourceVerificationError: A required source is absent, corrupt, mismatched, or fails
            teacher acceptance policy.
    """
    evidence, evidence_input = _read_json(
        store,
        artifact_id=source.acceptance_evidence_id,
        artifact_type="sft-teacher-acceptance",
        relative_path="evidence.json",
        model_type=TeacherAcceptanceEvidence,
    )
    if evidence.acceptance_evidence_id != source.acceptance_evidence_id:
        raise SFTSourceVerificationError("teacher acceptance evidence has the wrong identity")
    rollout, rollout_input = _read_json(
        store,
        artifact_id=evidence.rollout.artifact_id,
        artifact_type="rollout",
        relative_path="rollout.json",
        model_type=RolloutArtifact,
        expected_input=evidence.rollout,
    )
    if rollout.artifact_id != rollout_input.artifact_id:
        raise SFTSourceVerificationError("teacher rollout has the wrong artifact identity")
    task, task_set_input, task_set_inputs = _load_task(store, evidence)
    rule, rule_input = _read_json(
        store,
        artifact_id=evidence.acceptance_rule.artifact_id,
        artifact_type="sft-teacher-acceptance-rule",
        relative_path="rule.json",
        model_type=TeacherAcceptanceRule,
        expected_input=evidence.acceptance_rule,
    )
    if rule.acceptance_rule_id != rule_input.artifact_id:
        raise SFTSourceVerificationError("teacher acceptance rule has the wrong identity")
    judgment, judgment_input = _read_json(
        store,
        artifact_id=evidence.judgment.artifact_id,
        artifact_type="judgment",
        relative_path="judgment.json",
        model_type=Judgment,
        expected_input=evidence.judgment,
    )
    if judgment.judgment_id != judgment_input.artifact_id:
        raise SFTSourceVerificationError("teacher judgment has the wrong identity")
    calibration, calibration_input = _verify_calibration(store, evidence)
    fidelity, fidelity_input = _read_json(
        store,
        artifact_id=evidence.fidelity_report.artifact_id,
        artifact_type="fidelity-report",
        relative_path="fidelity-report.json",
        model_type=FidelityReport,
        expected_input=evidence.fidelity_report,
    )
    if fidelity.fidelity_report_id != fidelity_input.artifact_id:
        raise SFTSourceVerificationError("teacher fidelity report has the wrong identity")
    transcript = _load_transcript(
        store,
        evidence_id=evidence_input.artifact_id,
        transcript_path=evidence.transcript_path,
        transcript_sha256=evidence.transcript_sha256,
    )
    score, rubric_input = _verify_teacher_decision(
        store,
        evidence=evidence,
        rollout=rollout,
        rollout_input=rollout_input,
        task=task,
        task_set_input=task_set_input,
        rule=rule,
        rule_input=rule_input,
        judgment=judgment,
        judgment_input=judgment_input,
        calibration=calibration,
        calibration_input=calibration_input,
        fidelity=fidelity,
        fidelity_input=fidelity_input,
    )
    return PreparedSFTSource(
        kind="teacher_rollout",
        source_id=rollout.rollout_id,
        source_artifact=rollout_input,
        source_record_sha256=sha256_json(rollout),
        leakage_group_id=task.lineage_group_id,
        task=task.instruction,
        transcript_events=transcript.events,
        example_source=RolloutExampleSource(
            rollout_id=rollout.rollout_id,
            acceptance_evidence=evidence_input,
        ),
        score=score,
        direct_inputs=_sorted_inputs(
            (
                rollout_input,
                task_set_input,
                *task_set_inputs,
                rule_input,
                judgment_input,
                calibration_input,
                fidelity_input,
                evidence_input,
                rubric_input,
                *calibration.inputs,
            )
        ),
        acceptance_rule_id=rule.acceptance_rule_id,
        acceptance_evidence_id=evidence.acceptance_evidence_id,
        acceptance_evidence=evidence_input,
    )


def _load_trace(
    store: ProjectStore, evidence: ProductionAcceptanceEvidence
) -> tuple[Trace, ArtifactInput]:
    """Load the exact persisted trace named by accepted production evidence."""
    try:
        loaded = load_trace_dataset(store.artifacts, evidence.trace_dataset.artifact_id)
        trace_manifest = store.artifacts.read(evidence.trace_dataset.artifact_id).manifest
    except (ArtifactCorruptionError, ValueError) as exc:
        raise SFTSourceVerificationError(
            "production trace dataset is unavailable or corrupt"
        ) from exc
    trace_input = artifact_input(trace_manifest)
    if trace_input != evidence.trace_dataset:
        raise SFTSourceVerificationError(
            "production evidence trace-dataset manifest does not match local evidence"
        )
    trace = next((item for item in loaded.traces if item.trace_id == evidence.trace_id), None)
    if trace is None:
        raise SFTSourceVerificationError(
            "production evidence names a trace absent from its dataset"
        )
    if sha256_json(trace) != evidence.trace_sha256:
        raise SFTSourceVerificationError(
            "production evidence trace digest does not match stored trace"
        )
    return trace, trace_input


def _verify_production_decision(
    store: ProjectStore,
    evidence: ProductionAcceptanceEvidence,
    trace: Trace,
    trace_input: ArtifactInput,
    rule: ProductionAcceptanceRule,
) -> ArtifactInput | None:
    """Verify a production decision against canonical trace, rule, and approval artifacts."""
    if trace.outcome is not None and trace.outcome.status == "failure":
        raise SFTSourceVerificationError("production trace has a terminal infrastructure failure")
    if any(span.failure is not None for span in trace.spans):
        raise SFTSourceVerificationError(
            "production trace has a recorded infrastructure span failure"
        )
    if evidence.decision == "trusted_outcome":
        outcome = trace.outcome
        if outcome is None or outcome.status != "success":
            raise SFTSourceVerificationError(
                "trusted production acceptance requires a successful stored outcome"
            )
        if sha256_json(outcome) != evidence.outcome_sha256:
            raise SFTSourceVerificationError(
                "production acceptance outcome digest does not match stored trace"
            )
        if outcome.outcome_name not in rule.accepted_outcomes:
            raise SFTSourceVerificationError("production outcome is not accepted by its rule")
        return None
    if not rule.allow_human_approval:
        raise SFTSourceVerificationError("production rule does not allow human approval")
    if evidence.human_approval is None:
        raise SFTSourceVerificationError("production approval evidence is missing")
    approval, approval_input = _read_json(
        store,
        artifact_id=evidence.human_approval.artifact_id,
        artifact_type="sft-human-approval",
        relative_path="approval.json",
        model_type=HumanApproval,
        expected_input=evidence.human_approval,
    )
    if approval.approval_id != approval_input.artifact_id:
        raise SFTSourceVerificationError("human approval has the wrong artifact identity")
    if approval.trace_dataset != trace_input or approval.trace_id != trace.trace_id:
        raise SFTSourceVerificationError("human approval does not bind the accepted stored trace")
    return approval_input


def _load_task(
    store: ProjectStore, evidence: TeacherAcceptanceEvidence
) -> tuple[TaskCase, ArtifactInput, tuple[ArtifactInput, ...]]:
    """Load the exact immutable task record and task-set lineage named by teacher evidence."""
    try:
        loaded = load_task_set(store.artifacts, evidence.task_set.artifact_id)
        task_set_manifest = store.artifacts.read(evidence.task_set.artifact_id).manifest
    except (ArtifactCorruptionError, ValueError) as exc:
        raise SFTSourceVerificationError("teacher task set is unavailable or corrupt") from exc
    task_set_input = artifact_input(task_set_manifest)
    if task_set_input != evidence.task_set:
        raise SFTSourceVerificationError(
            "teacher evidence task-set manifest does not match local immutable evidence"
        )
    if loaded.task_set.tasks_sha256 != evidence.task_set_tasks_sha256:
        raise SFTSourceVerificationError("teacher evidence task payload digest does not match")
    if loaded.task_set.inputs != evidence.task_set_inputs:
        raise SFTSourceVerificationError("teacher evidence task-set input lineage does not match")
    task = next((item for item in loaded.tasks if item.task_id == evidence.task_id), None)
    if task is None:
        raise SFTSourceVerificationError("teacher evidence task is absent from its task set")
    if sha256_json(task) != evidence.task_sha256:
        raise SFTSourceVerificationError("teacher evidence task content does not match stored task")
    return task, task_set_input, loaded.task_set.inputs


def _verify_calibration(
    store: ProjectStore, evidence: TeacherAcceptanceEvidence
) -> tuple[JudgeCalibration, ArtifactInput]:
    """Resolve a calibration only through W6's recursive persisted-evidence verifier."""
    try:
        calibration, calibration_input = verify_persisted_calibration(
            store, evidence.calibration.artifact_id
        )
    except CalibrationError as exc:
        raise SFTSourceVerificationError(
            "teacher calibration is unavailable or fails recursive W6 provenance verification"
        ) from exc
    if calibration_input != evidence.calibration:
        raise SFTSourceVerificationError(
            "teacher evidence calibration manifest does not match local immutable evidence"
        )
    if calibration.calibration_id != calibration_input.artifact_id:
        raise SFTSourceVerificationError("teacher calibration has the wrong artifact identity")
    return calibration, calibration_input


def _verify_teacher_decision(
    store: ProjectStore,
    *,
    evidence: TeacherAcceptanceEvidence,
    rollout: RolloutArtifact,
    rollout_input: ArtifactInput,
    task: TaskCase,
    task_set_input: ArtifactInput,
    rule: TeacherAcceptanceRule,
    rule_input: ArtifactInput,
    judgment: Judgment,
    judgment_input: ArtifactInput,
    calibration: JudgeCalibration,
    calibration_input: ArtifactInput,
    fidelity: FidelityReport,
    fidelity_input: ArtifactInput,
) -> tuple[float, ArtifactInput]:
    """Verify all teacher gates and recompute the usable score from persisted judgment maps."""
    if rule.required_calibration != calibration_input:
        raise SFTSourceVerificationError("teacher acceptance rule names a different calibration")
    if rollout.failure is not None or rollout.stop_reason != "completed":
        raise SFTSourceVerificationError(
            "teacher rollout has an infrastructure or execution failure"
        )
    if any(span.failure is not None for span in rollout.spans):
        raise SFTSourceVerificationError(
            "teacher rollout has a recorded infrastructure span failure"
        )
    if rollout.evidence_source not in {"world_model", "sandbox"}:
        raise SFTSourceVerificationError(
            "teacher rollout must come from a world-model or sandbox simulation"
        )
    if rollout.task_id != task.task_id:
        raise SFTSourceVerificationError("teacher rollout names a different canonical task")
    if calibration.status != "human_calibrated" or calibration.approved_at is None:
        raise SFTSourceVerificationError(
            "teacher rollout requires a recursively verified human-calibrated judge"
        )
    if fidelity.status != "approved" or fidelity.approved_at is None:
        raise SFTSourceVerificationError("teacher rollout requires an approved fidelity report")
    if rollout_input not in fidelity.inputs:
        raise SFTSourceVerificationError("teacher fidelity report does not bind the stored rollout")
    if judgment.rollout_id != rollout.rollout_id:
        raise SFTSourceVerificationError("teacher judgment does not belong to the stored rollout")
    if judgment.calibration_id != calibration.calibration_id:
        raise SFTSourceVerificationError("teacher judgment uses a different stored calibration")
    rubric, rubric_input = _read_json(
        store,
        artifact_id=calibration.rubric_id,
        artifact_type="rubric",
        relative_path="rubric.json",
        model_type=Rubric,
    )
    if rubric.rubric_id != rubric_input.artifact_id:
        raise SFTSourceVerificationError("teacher rubric has the wrong artifact identity")
    if rubric.source_task_set_id != task_set_input.artifact_id:
        raise SFTSourceVerificationError("teacher rubric belongs to a different task set")
    if task_set_input not in rubric.inputs:
        raise SFTSourceVerificationError("teacher rubric does not hash its canonical task set")
    if judgment.rubric_id != rubric.rubric_id:
        raise SFTSourceVerificationError("teacher judgment uses a different stored rubric")
    if (
        judgment.judge_model != calibration.judge_model
        or judgment.judge_prompt_id != calibration.judge_prompt_id
        or judgment.judge_prompt_sha256 != calibration.judge_prompt_sha256
    ):
        raise SFTSourceVerificationError(
            "teacher judgment model or prompt does not match its stored calibration"
        )
    required_judgment_inputs = {rollout_input, rubric_input, calibration_input}
    if not required_judgment_inputs.issubset(set(judgment.inputs)):
        raise SFTSourceVerificationError(
            "teacher judgment does not hash its rollout, rubric, and calibration artifacts"
        )
    score = _recompute_calibrated_score(judgment, rubric, calibration)
    if score < rule.minimum_overall_score:
        raise SFTSourceVerificationError("teacher judgment score does not meet acceptance policy")
    if evidence.inputs != tuple(
        sorted(
            (
                rollout_input,
                task_set_input,
                judgment_input,
                calibration_input,
                fidelity_input,
                rule_input,
            ),
            key=lambda item: item.artifact_id,
        )
    ):
        raise SFTSourceVerificationError("teacher evidence has noncanonical direct artifact inputs")
    return score, rubric_input


def _recompute_calibrated_score(
    judgment: Judgment,
    rubric: Rubric,
    calibration: JudgeCalibration,
) -> float:
    """Recompute one authoritative equal-weight score from persisted raw scores and maps."""
    rubric_ids = tuple(item.dimension_id for item in rubric.dimensions)
    maps = {item.dimension_id: item for item in calibration.score_maps}
    dimensions = {item.dimension_id: item for item in judgment.dimensions}
    if set(maps) != set(rubric_ids) or set(dimensions) != set(rubric_ids):
        raise SFTSourceVerificationError(
            "teacher judgment, rubric, and calibration must cover the same dimensions"
        )
    calibrated_scores: list[float] = []
    for dimension_id in rubric_ids:
        dimension = dimensions[dimension_id]
        expected = maps[dimension_id].apply(dimension.raw_score)
        if not math.isclose(
            dimension.calibrated_score,
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise SFTSourceVerificationError(
                "teacher judgment calibrated dimension does not match stored score map"
            )
        axis = next(item for item in rubric.dimensions if item.dimension_id == dimension_id)
        calibrated_scores.append(axis.normalize_score(expected))
    overall = sum(calibrated_scores) / len(calibrated_scores)
    if not math.isclose(judgment.overall_score, overall, rel_tol=1e-12, abs_tol=1e-12):
        raise SFTSourceVerificationError(
            "teacher judgment overall score does not match authoritative calibrated dimensions"
        )
    return overall


def _load_transcript(
    store: ProjectStore,
    *,
    evidence_id: ArtifactId,
    transcript_path: str,
    transcript_sha256: Sha256,
) -> SFTTranscript:
    """Read and immediately verify canonical transcript bytes owned by accepted evidence."""
    try:
        payload = store.artifacts.read_bytes(evidence_id, transcript_path)
    except (ArtifactCorruptionError, ValueError) as exc:
        raise SFTSourceVerificationError(
            "accepted SFT transcript is unavailable or corrupt"
        ) from exc
    if hashlib.sha256(payload).hexdigest() != transcript_sha256:
        raise SFTSourceVerificationError("accepted SFT transcript bytes do not match evidence")
    try:
        transcript = SFTTranscript.model_validate_json(payload)
    except ValidationError as exc:
        raise SFTSourceVerificationError(
            "accepted SFT transcript is not a valid canonical record"
        ) from exc
    if canonical_json_bytes(transcript) != payload:
        raise SFTSourceVerificationError("accepted SFT transcript bytes are not canonical")
    return transcript


def _read_json[ModelT: BaseModel](
    store: ProjectStore,
    *,
    artifact_id: ArtifactId,
    artifact_type: str,
    relative_path: str,
    model_type: type[ModelT],
    expected_input: ArtifactInput | None = None,
) -> tuple[ModelT, ArtifactInput]:
    """Load one typed immutable record and normalize W6/project-store errors for SFT callers."""
    try:
        return read_artifact_json(
            store,
            artifact_id=artifact_id,
            expected_artifact_type=artifact_type,
            relative_path=relative_path,
            model_type=model_type,
            expected_input=expected_input,
        )
    except JudgingProvenanceError as exc:
        raise SFTSourceVerificationError(
            f"required {artifact_type} artifact is unavailable or does not match evidence"
        ) from exc


def _sorted_inputs(inputs: Iterable[ArtifactInput]) -> tuple[ArtifactInput, ...]:
    """Return exact unique authoritative artifact inputs in stable identity order."""
    return sorted_unique_inputs(*inputs, error_type=SFTSourceVerificationError)
