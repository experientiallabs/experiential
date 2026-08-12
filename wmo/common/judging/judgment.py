"""Canonical immutable judgment contracts for existing rollout evidence."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import ArtifactEnvelope, ArtifactId, ContractModel, Sha256
from wmo.common.models import ModelSnapshot, OperationEconomics


class DimensionJudgment(ContractModel):
    """A judge's raw and calibrated assessment of one rubric dimension."""

    dimension_id: ArtifactId
    raw_score: Literal[0, 1, 2, 3, 4, 5]
    calibrated_score: float = Field(ge=0, le=5)
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
            raise ValueError("a judgment must score at least one rubric dimension")
        dimension_ids = tuple(dimension.dimension_id for dimension in value)
        if len(set(dimension_ids)) != len(dimension_ids):
            raise ValueError("judgments must not repeat a rubric dimension")
        return value

    @field_validator("overall_score")
    @classmethod
    def _require_finite_overall_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("overall_score must be finite")
        return value

    @model_validator(mode="after")
    def _require_equal_weight_overall_score(self) -> Judgment:
        expected_score = sum(dimension.calibrated_score / 5 for dimension in self.dimensions) / len(
            self.dimensions
        )
        if not math.isclose(self.overall_score, expected_score, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("overall_score must equal the equal-weight calibrated dimension mean")
        return self
