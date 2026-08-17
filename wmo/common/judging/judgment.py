"""Canonical immutable judgment contracts for existing rollout evidence."""

from __future__ import annotations

import math

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import ArtifactEnvelope, ArtifactId, ContractModel, Sha256
from wmo.common.models import ModelSnapshot, OperationEconomics


def accept_legacy_judge_output_fields(
    value: object, *, map_feedback_to_rationale: bool = False
) -> object:
    """Drop retired citation keys and optionally map feedback onto rationale.

    Args:
        value: Candidate model payload.
        map_feedback_to_rationale: When true, copy a string ``feedback`` value onto
            ``rationale`` if rationale is absent.

    Returns:
        The original payload, or a shallow copy without retired citation keys.
    """
    if not isinstance(value, dict):
        return value
    payload = dict(value)
    feedback = payload.pop("feedback", None)
    payload.pop("evidence_span_ids", None)
    payload.pop("evidence_span_ids_a", None)
    payload.pop("evidence_span_ids_b", None)
    if map_feedback_to_rationale and "rationale" not in payload and isinstance(feedback, str):
        payload["rationale"] = feedback
    return payload


class DimensionJudgment(ContractModel):
    """A judge's raw and calibrated assessment of one rubric axis."""

    dimension_id: ArtifactId
    raw_score: int = Field(ge=0)
    calibrated_score: float = Field(ge=0)
    min_score: int = Field(default=0, ge=0)
    max_score: int = Field(default=5, ge=0)
    rationale: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_citation_fields(cls, value: object) -> object:
        """Load retired citation payloads without treating them as current output."""
        return accept_legacy_judge_output_fields(value, map_feedback_to_rationale=True)

    @field_validator("calibrated_score")
    @classmethod
    def _require_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("calibrated_score must be finite")
        return value

    @model_validator(mode="after")
    def _require_score_inside_axis_range(self) -> DimensionJudgment:
        if self.min_score >= self.max_score:
            raise ValueError("dimension judgment range must have min_score below max_score")
        if self.raw_score < self.min_score or self.raw_score > self.max_score:
            raise ValueError("raw_score must be an integer inside the axis range")
        if self.calibrated_score < self.min_score or self.calibrated_score > self.max_score:
            raise ValueError("calibrated_score must stay inside the axis range")
        return self

    def normalize_score(self) -> float:
        """Map the calibrated axis score onto the unit interval."""
        return (self.calibrated_score - self.min_score) / (self.max_score - self.min_score)


class Judgment(ArtifactEnvelope):
    """A frozen score of one rollout under one rubric and calibration."""

    judgment_id: ArtifactId
    rollout_id: ArtifactId
    rubric_id: ArtifactId
    calibration_id: ArtifactId
    judge_model: ModelSnapshot
    judge_prompt_id: str = Field(min_length=1, max_length=256)
    judge_prompt_sha256: Sha256
    dimensions: tuple[DimensionJudgment, ...]
    overall_score: float = Field(ge=0, le=1)
    judge_economics: OperationEconomics | None = None

    @field_validator("dimensions")
    @classmethod
    def _require_unique_dimensions(
        cls, value: tuple[DimensionJudgment, ...]
    ) -> tuple[DimensionJudgment, ...]:
        if not value:
            raise ValueError("a judgment must score at least one rubric axis")
        dimension_ids = tuple(dimension.dimension_id for dimension in value)
        if len(set(dimension_ids)) != len(dimension_ids):
            raise ValueError("judgments must not repeat a rubric axis")
        return value

    @field_validator("overall_score")
    @classmethod
    def _require_finite_overall_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("overall_score must be finite")
        return value

    @model_validator(mode="after")
    def _require_equal_weight_overall_score(self) -> Judgment:
        expected_score = sum(dimension.normalize_score() for dimension in self.dimensions) / len(
            self.dimensions
        )
        if not math.isclose(self.overall_score, expected_score, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("overall_score must equal the equal-weight calibrated axis mean")
        return self
