"""Typed immutable contracts for frozen, leakage-safe SFT dataset construction."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ContractModel,
    Sha256,
    StructuredFailure,
    validate_artifact_file_path,
)
from wmo.common.evaluations import FidelityReport
from wmo.common.judging import JudgeCalibration, Judgment
from wmo.common.models import AssistantAction
from wmo.common.rollouts import RolloutArtifact
from wmo.common.tasks import TaskCase, TaskSet
from wmo.common.traces import Trace


def _require_timezone(value: datetime, *, label: str) -> datetime:
    """Require one explicit timezone-aware immutable record timestamp."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return value


class SFTMessage(ContractModel):
    """A non-assistant message retained verbatim in canonical SFT context."""

    kind: Literal["message"] = "message"
    role: Literal["system", "user", "observation"]
    content: str


class AssistantActionEvent(ContractModel):
    """A complete assistant turn and its explicit source-side approval state."""

    kind: Literal["assistant_action"] = "assistant_action"
    action: AssistantAction
    approved: bool = True


class ToolEvent(ContractModel):
    """One ordered tool result retained as SFT context and never used as a target."""

    kind: Literal["tool"] = "tool"
    tool_call_id: str = Field(min_length=1, max_length=256)
    content: str
    tool_name: str | None = Field(default=None, min_length=1, max_length=256)


class InfrastructureFailureEvent(ContractModel):
    """A source event that proves an associated action cannot be an SFT target."""

    kind: Literal["infrastructure_failure"] = "infrastructure_failure"
    action_index: int = Field(ge=0)
    failure: StructuredFailure


SFTContextEvent = Annotated[
    SFTMessage | AssistantActionEvent | ToolEvent,
    Field(discriminator="kind"),
]
SFTTranscriptEvent = Annotated[
    SFTMessage | AssistantActionEvent | ToolEvent | InfrastructureFailureEvent,
    Field(discriminator="kind"),
]


class SFTTranscript(ContractModel):
    """The ordered source transcript from which assistant turns can be extracted."""

    events: tuple[SFTTranscriptEvent, ...]

    @field_validator("events")
    @classmethod
    def _require_nonempty_linked_events(
        cls, value: tuple[SFTTranscriptEvent, ...]
    ) -> tuple[SFTTranscriptEvent, ...]:
        if not value:
            raise ValueError("SFT transcripts need at least one event")
        tool_names: dict[str, str] = {}
        for index, event in enumerate(value):
            if isinstance(event, AssistantActionEvent):
                for call in event.action.tool_calls:
                    if call.call_id in tool_names:
                        raise ValueError("SFT transcripts must not repeat a tool call ID")
                    tool_names[call.call_id] = call.name
            elif isinstance(event, ToolEvent):
                expected_name = tool_names.get(event.tool_call_id)
                if expected_name is None:
                    raise ValueError("SFT tool events must name an earlier assistant tool call")
                if event.tool_name is not None and event.tool_name != expected_name:
                    raise ValueError("SFT tool event name must match its assistant tool call")
            elif isinstance(event, InfrastructureFailureEvent):
                if event.action_index >= index or not isinstance(
                    value[event.action_index], AssistantActionEvent
                ):
                    raise ValueError(
                        "SFT infrastructure failures must name an earlier assistant action index"
                    )
        return value


class ProductionAcceptanceRule(ArtifactEnvelope):
    """Immutable policy that admits trusted production outcomes or human approvals."""

    acceptance_rule_id: ArtifactId
    accepted_outcomes: tuple[str, ...]
    allow_human_approval: bool

    @field_validator("accepted_outcomes")
    @classmethod
    def _require_sorted_unique_outcomes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("production acceptance rules require at least one trusted outcome")
        if any(not outcome for outcome in value):
            raise ValueError("production acceptance outcomes must be non-empty")
        if len(set(value)) != len(value):
            raise ValueError("production acceptance outcomes must not repeat")
        if value != tuple(sorted(value)):
            raise ValueError("production acceptance outcomes must be sorted")
        return value


class TeacherAcceptanceRule(ArtifactEnvelope):
    """Immutable score, calibration, and fidelity policy for teacher rollout acceptance."""

    acceptance_rule_id: ArtifactId
    minimum_overall_score: float = Field(ge=0, le=1)
    required_calibration_id: ArtifactId
    require_approved_fidelity: Literal[True] = True

    @field_validator("minimum_overall_score")
    @classmethod
    def _require_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("teacher acceptance score must be finite")
        return value


class HumanApproval(ArtifactEnvelope):
    """One explicit immutable human approval bound to a production trace."""

    approval_id: ArtifactId
    trace_id: str = Field(min_length=1, max_length=512)
    decision: Literal["approved"] = "approved"
    approved_at: datetime

    @field_validator("approved_at")
    @classmethod
    def _require_approved_at_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value, label="human approval time")


class ProductionAcceptanceEvidence(ArtifactEnvelope):
    """Immutable evidence that one production trace satisfies a production acceptance rule."""

    acceptance_evidence_id: ArtifactId
    trace_id: str = Field(min_length=1, max_length=512)
    trace_sha256: Sha256
    acceptance_rule_id: ArtifactId
    acceptance_rule_sha256: Sha256
    decision: Literal["trusted_outcome", "human_approval"]
    outcome_sha256: Sha256 | None = None
    human_approval_id: ArtifactId | None = None
    human_approval_sha256: Sha256 | None = None
    accepted_at: datetime

    @field_validator("accepted_at")
    @classmethod
    def _require_accepted_at_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value, label="production acceptance time")

    @model_validator(mode="after")
    def _require_complete_evidence_branch(self) -> ProductionAcceptanceEvidence:
        expected_inputs = {self.acceptance_rule_id: self.acceptance_rule_sha256}
        if self.decision == "trusted_outcome":
            if self.outcome_sha256 is None:
                raise ValueError("trusted-outcome evidence requires an outcome digest")
            if self.human_approval_id is not None or self.human_approval_sha256 is not None:
                raise ValueError("trusted-outcome evidence must not name human approval")
        else:
            if self.outcome_sha256 is not None:
                raise ValueError("human-approval evidence must not name an outcome digest")
            if self.human_approval_id is None or self.human_approval_sha256 is None:
                raise ValueError("human-approval evidence requires approval identity and digest")
            expected_inputs[self.human_approval_id] = self.human_approval_sha256
        actual_inputs = {item.artifact_id: item.sha256 for item in self.inputs}
        if actual_inputs != expected_inputs:
            raise ValueError("production acceptance evidence inputs must match its references")
        return self


class TeacherAcceptanceEvidence(ArtifactEnvelope):
    """Immutable evidence that one teacher rollout passed all required quality gates."""

    acceptance_evidence_id: ArtifactId
    rollout_id: ArtifactId
    rollout_sha256: Sha256
    judgment_id: ArtifactId
    judgment_sha256: Sha256
    calibration_id: ArtifactId
    calibration_sha256: Sha256
    fidelity_report_id: ArtifactId
    fidelity_report_sha256: Sha256
    acceptance_rule_id: ArtifactId
    acceptance_rule_sha256: Sha256
    observed_overall_score: float = Field(ge=0, le=1)
    accepted_at: datetime

    @field_validator("observed_overall_score")
    @classmethod
    def _require_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("teacher acceptance score must be finite")
        return value

    @field_validator("accepted_at")
    @classmethod
    def _require_accepted_at_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value, label="teacher acceptance time")

    @model_validator(mode="after")
    def _require_complete_references(self) -> TeacherAcceptanceEvidence:
        expected_inputs = {
            self.rollout_id: self.rollout_sha256,
            self.judgment_id: self.judgment_sha256,
            self.calibration_id: self.calibration_sha256,
            self.fidelity_report_id: self.fidelity_report_sha256,
            self.acceptance_rule_id: self.acceptance_rule_sha256,
        }
        actual_inputs = {item.artifact_id: item.sha256 for item in self.inputs}
        if actual_inputs != expected_inputs:
            raise ValueError(
                "teacher acceptance evidence inputs must match every required reference"
            )
        return self


class ProductionSFTSource(ContractModel):
    """One production trace, canonical transcript, and immutable acceptance chain."""

    trace: Trace
    transcript: SFTTranscript
    acceptance_rule: ProductionAcceptanceRule
    acceptance_evidence: ProductionAcceptanceEvidence
    human_approval: HumanApproval | None = None


class TeacherSFTSource(ContractModel):
    """One teacher rollout, canonical transcript, and immutable accepted quality evidence."""

    rollout: RolloutArtifact
    task: TaskCase
    task_set: TaskSet
    transcript: SFTTranscript
    acceptance_rule: TeacherAcceptanceRule
    acceptance_evidence: TeacherAcceptanceEvidence
    judgment: Judgment
    calibration: JudgeCalibration
    fidelity: FidelityReport


class TraceExampleSource(ContractModel):
    """Production trace provenance retained on an extracted SFT example."""

    kind: Literal["production_trace"] = "production_trace"
    trace_id: str = Field(min_length=1, max_length=512)
    acceptance_evidence_id: ArtifactId
    acceptance_evidence_sha256: Sha256


class RolloutExampleSource(ContractModel):
    """Teacher rollout provenance retained on an extracted SFT example."""

    kind: Literal["teacher_rollout"] = "teacher_rollout"
    rollout_id: ArtifactId
    acceptance_evidence_id: ArtifactId
    acceptance_evidence_sha256: Sha256


SFTExampleSource = Annotated[
    TraceExampleSource | RolloutExampleSource,
    Field(discriminator="kind"),
]


class SFTExample(ContractModel):
    """Canonical task context plus one complete assistant-action target."""

    example_id: ArtifactId
    leakage_group_id: ArtifactId
    task: str = Field(min_length=1)
    history: tuple[SFTContextEvent, ...]
    target: AssistantAction
    source: SFTExampleSource
    source_step_index: int = Field(ge=0)
    score: float | None = Field(default=None, ge=0, le=1)

    @field_validator("score")
    @classmethod
    def _require_finite_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("SFT example scores must be finite")
        return value


class PartitionedSFTExample(ContractModel):
    """One persisted SFT example assigned to train or held-out data with its fingerprint."""

    partition: Literal["train", "held_out"]
    fingerprint: Sha256
    example: SFTExample


class SFTPartition(ContractModel):
    """One connected leakage component and its deterministic frozen partition."""

    component_id: ArtifactId
    partition: Literal["train", "held_out"]
    leakage_group_ids: tuple[ArtifactId, ...]
    fingerprints: tuple[Sha256, ...]

    @field_validator("leakage_group_ids", "fingerprints")
    @classmethod
    def _require_sorted_unique_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("SFT leakage components need at least one lineage and fingerprint")
        if len(set(value)) != len(value):
            raise ValueError("SFT leakage component values must not repeat")
        if value != tuple(sorted(value)):
            raise ValueError("SFT leakage component values must be sorted")
        return value


class SFTSourceReference(ContractModel):
    """An auditable source and acceptance-evidence reference in a frozen dataset manifest."""

    kind: Literal["production_trace", "teacher_rollout"]
    source_id: str = Field(min_length=1, max_length=512)
    source_sha256: Sha256
    leakage_group_id: ArtifactId
    acceptance_evidence_id: ArtifactId
    acceptance_evidence_sha256: Sha256
    accepted: bool
    exclusion_reason: str | None = None

    @model_validator(mode="after")
    def _require_consistent_source_status(self) -> SFTSourceReference:
        if self.accepted and self.exclusion_reason is not None:
            raise ValueError("accepted SFT sources must not carry an exclusion reason")
        if not self.accepted and self.exclusion_reason is None:
            raise ValueError("excluded SFT sources must name an exclusion reason")
        return self


class SFTExclusion(ContractModel):
    """One source-level or action-level reason data was withheld from an SFT dataset."""

    source_kind: Literal["production_trace", "teacher_rollout"]
    source_id: str = Field(min_length=1, max_length=512)
    action_index: int | None = Field(default=None, ge=0)
    reason: Literal[
        "duplicate_normalized_example",
        "infrastructure_failure",
        "invalid_production_acceptance",
        "invalid_teacher_acceptance",
        "observation_context_only",
        "unapproved_action",
    ]
    detail: str = Field(min_length=1)


class SFTDataset(ArtifactEnvelope):
    """Frozen accepted SFT dataset envelope and paths to its exact normalized example rows."""

    dataset_id: ArtifactId
    build_sha256: Sha256
    status: Literal["accepted", "insufficient"]
    acceptance_rule_ids: tuple[ArtifactId, ...]
    acceptance_evidence_ids: tuple[ArtifactId, ...]
    train_leakage_group_ids: tuple[ArtifactId, ...]
    held_out_leakage_group_ids: tuple[ArtifactId, ...]
    train_example_ids: tuple[ArtifactId, ...]
    held_out_example_ids: tuple[ArtifactId, ...]
    examples_path: str = Field(min_length=1)
    examples_sha256: Sha256

    @field_validator("examples_path")
    @classmethod
    def _require_safe_examples_path(cls, value: str) -> str:
        return validate_artifact_file_path(value).as_posix()

    @field_validator(
        "acceptance_rule_ids",
        "acceptance_evidence_ids",
        "train_leakage_group_ids",
        "held_out_leakage_group_ids",
        "train_example_ids",
        "held_out_example_ids",
    )
    @classmethod
    def _require_sorted_unique_ids(cls, value: tuple[ArtifactId, ...]) -> tuple[ArtifactId, ...]:
        if len(set(value)) != len(value):
            raise ValueError("frozen SFT dataset IDs must not repeat")
        if value != tuple(sorted(value)):
            raise ValueError("frozen SFT dataset IDs must be sorted")
        return value

    @model_validator(mode="after")
    def _require_disjoint_partitions(self) -> SFTDataset:
        if set(self.train_leakage_group_ids).intersection(self.held_out_leakage_group_ids):
            raise ValueError("SFT train and held-out lineage groups must be disjoint")
        if set(self.train_example_ids).intersection(self.held_out_example_ids):
            raise ValueError("SFT train and held-out examples must be disjoint")
        return self


class SFTInspectionReport(ContractModel):
    """Deterministic provenance, coverage, sample, and exclusion report for a frozen dataset."""

    report_id: ArtifactId
    dataset_id: ArtifactId
    build_sha256: Sha256
    source_count: int = Field(ge=0)
    accepted_source_count: int = Field(ge=0)
    eligible_action_count: int = Field(ge=0)
    fingerprint_count: int = Field(ge=0)
    connected_component_count: int = Field(ge=0)
    train_example_count: int = Field(ge=0)
    held_out_example_count: int = Field(ge=0)
    exclusions: tuple[SFTExclusion, ...]
    representative_train_example_ids: tuple[ArtifactId, ...]
    representative_held_out_example_ids: tuple[ArtifactId, ...]

    @field_validator("representative_train_example_ids", "representative_held_out_example_ids")
    @classmethod
    def _require_unique_sample_ids(cls, value: tuple[ArtifactId, ...]) -> tuple[ArtifactId, ...]:
        if len(set(value)) != len(value):
            raise ValueError("SFT representative samples must not repeat")
        return value


class SFTDatasetMetadata(ContractModel):
    """All non-row frozen manifest data persisted beside canonical SFT JSONL examples."""

    dataset: SFTDataset
    sources: tuple[SFTSourceReference, ...]
    partitions: tuple[SFTPartition, ...]
    inspection: SFTInspectionReport
    representative_samples: tuple[PartitionedSFTExample, ...]


class SFTDatasetArtifact(ContractModel):
    """Materialized dataset rows plus the immutable metadata required to reload and inspect them."""

    dataset: SFTDataset
    sources: tuple[SFTSourceReference, ...]
    partitions: tuple[SFTPartition, ...]
    inspection: SFTInspectionReport
    representative_samples: tuple[PartitionedSFTExample, ...]
    rows: tuple[PartitionedSFTExample, ...]

    def metadata(self) -> SFTDatasetMetadata:
        """Return the immutable metadata file stored beside the canonical JSONL rows."""
        return SFTDatasetMetadata(
            dataset=self.dataset,
            sources=self.sources,
            partitions=self.partitions,
            inspection=self.inspection,
            representative_samples=self.representative_samples,
        )
