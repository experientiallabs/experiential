"""Canonical immutable judgment contracts for existing rollout evidence."""

from __future__ import annotations

import math

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import ArtifactEnvelope, ArtifactId, ContractModel, Sha256
from wmo.common.models import ModelSnapshot, OperationEconomics


class DimensionJudgment(ContractModel):
    """A judge's raw and calibrated assessment of one rubric axis."""

    dimension_id: ArtifactId
    raw_score: int = Field(ge=0)
    calibrated_score: float = Field(ge=0)
    min_score: int = Field(default=0, ge=0)
    max_score: int = Field(default=5, ge=0)
    evidence_span_ids: tuple[str, ...]
    feedback: str = Field(min_length=1)

    @field_validator("evidence_span_ids")
    @classmethod
    def _require_unique_evidence_spans(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("dimension judgments require at least one cited evidence span")
        if any(not span_id for span_id in value):
            raise ValueError("dimension judgment evidence span IDs must be non-empty")
        if len(set(value)) != len(value):
            raise ValueError("dimension judgment evidence span IDs must not repeat")
        return value

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
