"""Canonical rubric and judge-calibration contracts."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import ArtifactEnvelope, ArtifactId, ContractModel
from wmo.common.models import ModelSnapshot


class ScoreAnchor(ContractModel):
    """Plain-language anchor for one integer rubric score."""

    score: Literal[0, 1, 2, 3, 4, 5]
    description: str = Field(min_length=1)


class RubricDimension(ContractModel):
    """A zero-to-five quality dimension with complete scoring anchors."""

    dimension_id: ArtifactId
    name: str = Field(min_length=1, max_length=256)
    description: str = Field(min_length=1)
    anchors: tuple[ScoreAnchor, ...]

    @field_validator("anchors")
    @classmethod
    def _require_exact_score_anchors(
        cls, value: tuple[ScoreAnchor, ...]
    ) -> tuple[ScoreAnchor, ...]:
        expected_scores = (0, 1, 2, 3, 4, 5)
        scores = tuple(anchor.score for anchor in value)
        if scores != expected_scores:
            raise ValueError("rubric anchors must contain scores zero through five in order")
        return value


class Rubric(ArtifactEnvelope):
    """An immutable, versioned set of equal-weight quality dimensions."""

    rubric_id: ArtifactId
    dimensions: tuple[RubricDimension, ...]
    source_task_set_id: ArtifactId
    status: Literal["provisional", "human_approved"]
    approved_at: datetime | None = None

    @field_validator("dimensions")
    @classmethod
    def _require_unique_dimensions(
        cls, value: tuple[RubricDimension, ...]
    ) -> tuple[RubricDimension, ...]:
        if not value:
            raise ValueError("a rubric must contain at least one dimension")
        dimension_ids = tuple(dimension.dimension_id for dimension in value)
        if len(set(dimension_ids)) != len(dimension_ids):
            raise ValueError("rubric dimensions must have unique IDs")
        return value

    @model_validator(mode="after")
    def _require_consistent_approval(self) -> Rubric:
        if self.status == "human_approved" and self.approved_at is None:
            raise ValueError("human-approved rubrics require approved_at")
        if self.status == "provisional" and self.approved_at is not None:
            raise ValueError("provisional rubrics must not set approved_at")
        return self


class DimensionScoreMap(ContractModel):
    """A monotonic mapping from raw judge scores to expected human scores."""

    dimension_id: ArtifactId
    calibrated_scores: tuple[float, float, float, float, float, float]

    @field_validator("calibrated_scores")
    @classmethod
    def _require_monotonic_finite_scores(
        cls, value: tuple[float, float, float, float, float, float]
    ) -> tuple[float, float, float, float, float, float]:
        if any(not math.isfinite(score) or score < 0 or score > 5 for score in value):
            raise ValueError(
                "calibrated rubric scores must be finite values from zero through five"
            )
        if tuple(sorted(value)) != value:
            raise ValueError("calibrated rubric scores must be monotonic")
        return value


class JudgeCalibration(ArtifactEnvelope):
    """Frozen judge model, prompt, label lineage, and score maps for one rubric."""

    calibration_id: ArtifactId
    rubric_id: ArtifactId
    judge_model: ModelSnapshot
    judge_prompt_id: str = Field(min_length=1, max_length=256)
    label_set_id: ArtifactId
    calibration_lineage_ids: tuple[ArtifactId, ...]
    excluded_router_held_out_lineage_ids: tuple[ArtifactId, ...]
    validation_method: Literal["grouped_k_fold"]
    out_of_fold_report_id: ArtifactId
    score_maps: tuple[DimensionScoreMap, ...]
    approved_at: datetime | None = None

    @field_validator("score_maps")
    @classmethod
    def _require_unique_score_maps(
        cls, value: tuple[DimensionScoreMap, ...]
    ) -> tuple[DimensionScoreMap, ...]:
        dimension_ids = tuple(score_map.dimension_id for score_map in value)
        if not dimension_ids:
            raise ValueError("judge calibration needs a score map for every rubric dimension")
        if len(set(dimension_ids)) != len(dimension_ids):
            raise ValueError("judge calibration score maps must not repeat a dimension")
        return value
