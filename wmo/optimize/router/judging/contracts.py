"""Immutable contracts for explicit local judge setup and calibration review."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal, cast

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
)
from wmo.common.judging import (
    CalibrationReport,
    PromptDefinition,
)
from wmo.common.models import ModelSnapshot, OperationEconomics


class ManualJudgeError(ValueError):
    """Raised when manual judge setup or calibration violates a local contract."""


class ManualJudgeLabel(ContractModel):
    """One human score or typed pairwise preference for real trace evidence."""

    trace_id: str = Field(min_length=1, max_length=512)
    reference_trace_id: str | None = Field(default=None, min_length=1, max_length=512)
    dimension_id: ArtifactId
    score: Literal[0, 1, 2, 3, 4, 5] | None = None
    winner: Literal["winner_a", "winner_b", "tie"] | None = None

    @model_validator(mode="after")
    def _require_one_label_shape(self) -> ManualJudgeLabel:
        """Require either one scalar score or one fully identified pairwise preference.

        Returns:
            The validated human label.

        Raises:
            ValueError: Scalar and pairwise fields are mixed or incomplete.
        """
        scalar = self.score is not None and self.winner is None and self.reference_trace_id is None
        pairwise = (
            self.score is None and self.winner is not None and self.reference_trace_id is not None
        )
        if not scalar and not pairwise:
            raise ValueError("human labels must be either scalar or fully specified pairwise")
        if self.reference_trace_id == self.trace_id:
            raise ValueError("pairwise labels require two distinct traces")
        return self


class JudgeScoreProjection(ContractModel):
    """Versioned explicit projection from structured feedback to router scores."""

    projection_version: Literal["1"] = "1"
    boolean_scores: dict[Literal["false", "true"], Literal[0, 1, 2, 3, 4, 5]] = Field(
        default_factory=dict
    )
    categorical_scores: dict[str, Literal[0, 1, 2, 3, 4, 5]] = Field(default_factory=dict)
    pairwise_scores: dict[Literal["winner_a", "winner_b", "tie"], Literal[0, 1, 2, 3, 4, 5]] = (
        Field(default_factory=dict)
    )
    pairwise_aggregation: Literal["rounded_mean"] | None = None


class JudgePromptTemplate(ContractModel):
    """Versioned prompt, variable mapping, and strict response schema."""

    template_id: Literal["wmo-judge-evidence-json"] = "wmo-judge-evidence-json"
    template_version: Literal["1"] = "1"
    response_shape: Literal["scalar", "boolean", "categorical", "pairwise"] = "scalar"
    prompt: PromptDefinition
    variable_mapping: JsonObject
    response_schema: JsonObject
    score_projection: JudgeScoreProjection = Field(default_factory=JudgeScoreProjection)

    @model_validator(mode="after")
    def _require_executable_contract(self) -> JudgePromptTemplate:
        """Require an exact supported schema, mapping, and explicit numeric projection.

        Returns:
            The executable prompt contract.

        Raises:
            ValueError: The schema, variables, or projection cannot be executed exactly.
        """
        required = (
            {"rubric", "candidate_a", "candidate_b"}
            if self.response_shape == "pairwise"
            else {"rubric", "rollout"}
        )
        if set(self.variable_mapping) != required or any(
            not isinstance(value, str) or not value.strip()
            for value in self.variable_mapping.values()
        ):
            raise ValueError(
                "judge variable mapping must name every required canonical input exactly once"
            )
        if len(set(cast(str, value) for value in self.variable_mapping.values())) != len(required):
            raise ValueError("judge variable mapping values must be unique")
        projection = self.score_projection
        if self.response_shape == "scalar":
            valid_projection = not (
                projection.boolean_scores
                or projection.categorical_scores
                or projection.pairwise_scores
                or projection.pairwise_aggregation
            )
        elif self.response_shape == "boolean":
            valid_projection = (
                set(projection.boolean_scores) == {"false", "true"}
                and not projection.categorical_scores
                and not projection.pairwise_scores
                and projection.pairwise_aggregation is None
            )
        elif self.response_shape == "categorical":
            valid_projection = (
                bool(projection.categorical_scores)
                and not projection.boolean_scores
                and not projection.pairwise_scores
                and projection.pairwise_aggregation is None
            )
        else:
            valid_projection = (
                set(projection.pairwise_scores) == {"winner_a", "winner_b", "tie"}
                and projection.pairwise_aggregation == "rounded_mean"
                and not projection.boolean_scores
                and not projection.categorical_scores
            )
        if not valid_projection:
            raise ValueError(
                f"judge {self.response_shape} response shape requires its exact saved score map"
            )
        expected_schema = judge_feedback_schema(
            self.response_shape,
            categories=tuple(sorted(projection.categorical_scores)),
        )
        if self.response_schema != expected_schema:
            raise ValueError("judge response schema is not the supported canonical schema")
        return self


class JudgeTracePreview(ContractModel):
    """Human-readable local trace selected for calibration labeling."""

    trace_id: str = Field(min_length=1, max_length=512)
    rollout_id: ArtifactId
    task_id: ArtifactId
    lineage_id: ArtifactId
    task: str = Field(min_length=1)
    outcome: str = Field(min_length=1, max_length=128)
    span_names: tuple[str, ...]
    reference_trace_id: str | None = Field(default=None, min_length=1, max_length=512)
    reference_rollout_id: ArtifactId | None = None


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
    reference_rollout: ArtifactInput | None = None
    judgment: ArtifactInput
    probes: tuple[ArtifactInput, ...] = ()


class JudgeProtocolProbeArtifact(ArtifactEnvelope):
    """One immutable schema-valid provider probe used by manual calibration."""

    probe_id: ArtifactId
    setup: ArtifactInput
    rollout: ArtifactInput
    reference_rollout: ArtifactInput | None = None
    order: Literal["single", "forward", "reverse"]
    response: JsonObject
    model: ModelSnapshot
    economics: OperationEconomics

    @model_validator(mode="after")
    def _require_complete_probe_inputs(self) -> JudgeProtocolProbeArtifact:
        """Bind a provider probe to setup and every visible rollout.

        Returns:
            The probe after exact input-graph validation.

        Raises:
            ValueError: An input is missing, duplicated, or ordered incorrectly.
        """
        expected = tuple(
            sorted(
                (
                    self.setup,
                    self.rollout,
                    *((self.reference_rollout,) if self.reference_rollout is not None else ()),
                ),
                key=lambda item: item.artifact_id,
            )
        )
        if len({item.artifact_id for item in expected}) != len(expected):
            raise ValueError("manual judge probe inputs must be unique")
        if self.inputs != expected:
            raise ValueError("manual judge probe must hash its complete input graph")
        if self.order == "single" and self.reference_rollout is not None:
            raise ValueError("single judge probes cannot bind a reference rollout")
        if self.order != "single" and self.reference_rollout is None:
            raise ValueError("counterbalanced judge probes require a reference rollout")
        return self


def judge_feedback_schema(
    shape: Literal["scalar", "boolean", "categorical", "pairwise"],
    *,
    categories: tuple[str, ...] = (),
) -> JsonObject:
    """Build the exact supported structured-feedback schema for one response shape.

    Args:
        shape: Supported structured feedback shape.
        categories: Saved categorical values when ``shape`` is categorical.

    Returns:
        Canonical strict JSON schema persisted in judge setup.

    Raises:
        ValueError: Categorical feedback does not define at least one category.
    """
    dimension_properties: JsonObject = {
        "dimension_id": {"type": "string"},
        "feedback": {"type": "string", "minLength": 1},
    }
    required = ["dimension_id", "feedback"]
    if shape == "scalar":
        dimension_properties["raw_score"] = {
            "type": "integer",
            "minimum": 0,
            "maximum": 5,
        }
        dimension_properties["evidence_span_ids"] = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "uniqueItems": True,
        }
        required.extend(("raw_score", "evidence_span_ids"))
    elif shape == "boolean":
        dimension_properties["passed"] = {"type": "boolean"}
        dimension_properties["evidence_span_ids"] = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "uniqueItems": True,
        }
        required.extend(("passed", "evidence_span_ids"))
    elif shape == "categorical":
        if not categories:
            raise ValueError("categorical judge feedback requires at least one saved category")
        dimension_properties["category"] = {"type": "string", "enum": list(categories)}
        dimension_properties["evidence_span_ids"] = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "uniqueItems": True,
        }
        required.extend(("category", "evidence_span_ids"))
    else:
        dimension_properties["winner"] = {
            "type": "string",
            "enum": ["winner_a", "winner_b", "tie"],
        }
        for key in ("evidence_span_ids_a", "evidence_span_ids_b"):
            dimension_properties[key] = {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "uniqueItems": True,
            }
        required.extend(("winner", "evidence_span_ids_a", "evidence_span_ids_b"))
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "dimensions": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": dimension_properties,
                    "required": required,
                },
            }
        },
        "required": ["dimensions"],
    }


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
    positional_bias_comparisons: int | None = Field(default=None, ge=1)
    positional_bias_flips: int | None = Field(default=None, ge=0)

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
                    *(
                        item.reference_rollout
                        for item in self.judgments
                        if item.reference_rollout is not None
                    ),
                    *(item.judgment for item in self.judgments),
                    *(probe for item in self.judgments for probe in item.probes),
                ),
                key=lambda item: item.artifact_id,
            )
        )
        if self.inputs != expected:
            raise ValueError("manual judge audit must hash its complete canonical input graph")
        if not self.judgments:
            raise ValueError("manual judge audit requires at least one judge probe")
        if (self.positional_bias_comparisons is None) != (self.positional_bias_flips is None):
            raise ValueError("positional-bias counts must be both present or both absent")
        if (
            self.positional_bias_comparisons is not None
            and self.positional_bias_flips is not None
            and self.positional_bias_flips > self.positional_bias_comparisons
        ):
            raise ValueError("positional-bias flips cannot exceed comparisons")
        return self


class ManualJudgeLabelDraft(ContractModel):
    """Human labels persisted for one frozen calibration sample before provider work.

    Human rating is the expensive part of manual calibration, so completed ratings become durable
    local review state as soon as they exist. The sample digest binds a draft to one exact setup,
    trace selection, rubric, and response shape, so a later attempt resumes the same work and an
    unrelated sample never inherits stale scores.
    """

    draft_version: Literal["manual-judge-label-draft-v1"] = "manual-judge-label-draft-v1"
    setup_id: ArtifactId
    sample_sha256: Sha256
    labels: tuple[ManualJudgeLabel, ...]
    updated_at: datetime

    @field_validator("labels")
    @classmethod
    def _require_unique_label_keys(
        cls, value: tuple[ManualJudgeLabel, ...]
    ) -> tuple[ManualJudgeLabel, ...]:
        """Reject two drafted scores for one trace, reference, and dimension key."""
        keys = tuple(
            (label.trace_id, label.reference_trace_id, label.dimension_id) for label in value
        )
        if len(set(keys)) != len(keys):
            raise ValueError("a label draft must not repeat a trace dimension")
        return value


class ManualJudgeReviewState(ContractModel):
    """Mutable review pointers for resumable setup, labels, audit, and explicit approval."""

    setup: ArtifactInput
    label_draft: ManualJudgeLabelDraft | None = None
    audit: ArtifactInput | None = None
    approved_calibration: ArtifactInput | None = None


class ManualJudgeCalibrationResult(ContractModel):
    """Completed audit and optional explicitly approved immutable calibration."""

    audit: ManualJudgeCalibrationAudit
    report: CalibrationReport
    approved_calibration: ArtifactInput | None = None
    provider_calls_made: int = Field(ge=0)
    completed_at: datetime
