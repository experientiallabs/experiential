"""Immutable contracts for explicit local judge setup and calibration review."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    JsonObject,
)
from wmo.common.judging import (
    CalibrationReport,
    PromptDefinition,
)
from wmo.common.models import ModelSnapshot


class ManualJudgeError(ValueError):
    """Raised when manual judge setup or calibration violates a local contract."""


class ManualJudgeLabel(ContractModel):
    """One human score supplied for a real trace preview."""

    trace_id: str = Field(min_length=1, max_length=512)
    dimension_id: ArtifactId
    score: Literal[0, 1, 2, 3, 4, 5]


class JudgePromptTemplate(ContractModel):
    """Versioned prompt, variable mapping, and strict response schema."""

    template_id: Literal["wmo-judge-evidence-json"] = "wmo-judge-evidence-json"
    template_version: Literal["1"] = "1"
    prompt: PromptDefinition
    variable_mapping: JsonObject
    response_schema: JsonObject


class JudgeTracePreview(ContractModel):
    """Human-readable local trace selected for calibration labeling."""

    trace_id: str = Field(min_length=1, max_length=512)
    rollout_id: ArtifactId
    task_id: ArtifactId
    lineage_id: ArtifactId
    task: str = Field(min_length=1)
    outcome: str = Field(min_length=1, max_length=128)
    span_names: tuple[str, ...]


class ManualJudgeSetupArtifact(ArtifactEnvelope):
    """Frozen judge contract approved before any calibration work or model call."""

    setup_id: ArtifactId
    project_id: ArtifactId
    judge_alias: ArtifactId
    judge_model: ModelSnapshot
    prompt_template: JudgePromptTemplate
    trace_dataset: ArtifactInput
    task_set: ArtifactInput
    rubric: ArtifactInput
    previews: tuple[JudgeTracePreview, ...]

    @model_validator(mode="after")
    def _require_complete_inputs(self) -> ManualJudgeSetupArtifact:
        """Require every setup source to appear exactly once in envelope inputs.

        Returns:
            The setup after verifying its immutable input graph.

        Raises:
            ValueError: An input is absent, duplicated, or ordered noncanonically.
        """
        expected = tuple(
            sorted(
                (self.trace_dataset, self.task_set, self.rubric), key=lambda item: item.artifact_id
            )
        )
        if len({item.artifact_id for item in expected}) != len(expected):
            raise ValueError("manual judge setup inputs must have unique artifact IDs")
        if self.inputs != expected:
            raise ValueError("manual judge setup must hash its complete canonical input graph")
        if not self.previews:
            raise ValueError("manual judge setup requires at least one rendered real-trace preview")
        return self


class JudgeCalibrationBudget(ContractModel):
    """Conservative finite reservation for counterbalanced judge calls."""

    input_usd_per_million_tokens: float = Field(ge=0)
    output_usd_per_million_tokens: float = Field(ge=0)
    maximum_input_tokens_per_call: int = Field(gt=0)
    maximum_output_tokens_per_call: Literal[4096] = 4096
    maximum_attempts_per_call: int = Field(gt=0)
    call_count: int = Field(gt=0)
    estimated_cost_usd: float = Field(ge=0)
    maximum_cost_usd: float = Field(gt=0)

    @field_validator(
        "input_usd_per_million_tokens",
        "output_usd_per_million_tokens",
        "estimated_cost_usd",
        "maximum_cost_usd",
    )
    @classmethod
    def _require_finite(cls, value: float) -> float:
        """Reject non-finite price and budget values."""
        if not math.isfinite(value):
            raise ValueError("judge calibration economics must be finite")
        return value

    @model_validator(mode="after")
    def _require_estimate_within_budget(self) -> JudgeCalibrationBudget:
        """Require the conservative full-run estimate to fit the caller ceiling.

        Returns:
            The budget after validating its finite admission boundary.

        Raises:
            ValueError: The estimate exceeds the maximum allowed spend.
        """
        if self.estimated_cost_usd > self.maximum_cost_usd:
            raise ValueError(
                "judge calibration estimate exceeds --maximum-cost-usd; raise the ceiling or "
                "reduce the labeled sample"
            )
        return self


class JudgeRunEvidence(ContractModel):
    """One calibration rollout and its persisted structured judgment."""

    rollout: ArtifactInput
    judgment: ArtifactInput


class ManualJudgeCalibrationAudit(ArtifactEnvelope):
    """Immutable reviewed calibration evidence before the approval decision."""

    audit_id: ArtifactId
    setup: ArtifactInput
    human_labels: ArtifactInput
    lineage_split: ArtifactInput
    provisional_calibration: ArtifactInput
    report: ArtifactInput
    budget: JudgeCalibrationBudget
    judgments: tuple[JudgeRunEvidence, ...]

    @model_validator(mode="after")
    def _require_complete_audit_inputs(self) -> ManualJudgeCalibrationAudit:
        """Bind the audit to setup, report, rollouts, and forward judgments.

        Returns:
            The audit after verifying its exact immutable inputs.

        Raises:
            ValueError: The input graph or report identity differs from the audit content.
        """
        expected = tuple(
            sorted(
                (
                    self.setup,
                    self.human_labels,
                    self.lineage_split,
                    self.provisional_calibration,
                    self.report,
                    *(item.rollout for item in self.judgments),
                    *(item.judgment for item in self.judgments),
                ),
                key=lambda item: item.artifact_id,
            )
        )
        if self.inputs != expected:
            raise ValueError("manual judge audit must hash its complete canonical input graph")
        if not self.judgments:
            raise ValueError("manual judge audit requires at least one judge probe")
        return self


class ManualJudgeReviewState(ContractModel):
    """Mutable review pointers for resumable setup, audit, and explicit approval."""

    setup: ArtifactInput
    audit: ArtifactInput | None = None
    approved_calibration: ArtifactInput | None = None


class ManualJudgeCalibrationResult(ContractModel):
    """Completed audit and optional explicitly approved immutable calibration."""

    audit: ManualJudgeCalibrationAudit
    report: CalibrationReport
    approved_calibration: ArtifactInput | None = None
    provider_calls_made: int = Field(ge=0)
    completed_at: datetime
