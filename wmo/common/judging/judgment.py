"""Canonical immutable judgment contracts for existing rollout evidence."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator

from wmo.common.core.artifacts import ArtifactEnvelope, ArtifactId, ContractModel
from wmo.common.models import OperationEconomics


class DimensionJudgment(ContractModel):
    """A judge's raw and calibrated assessment of one rubric dimension."""

    dimension_id: ArtifactId
    raw_score: Literal[0, 1, 2, 3, 4, 5]
    calibrated_score: float = Field(ge=0, le=5)
    evidence_span_ids: tuple[str, ...]
    feedback: str = Field(min_length=1)

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
