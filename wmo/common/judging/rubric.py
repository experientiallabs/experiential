"""Canonical rubric and judge-calibration contracts."""

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
    Sha256,
)
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
    accepted_proposal_evidence_ids: tuple[ArtifactId, ...] = ()
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

    @field_validator("accepted_proposal_evidence_ids")
    @classmethod
    def _require_sorted_unique_proposal_evidence(
        cls, value: tuple[ArtifactId, ...]
    ) -> tuple[ArtifactId, ...]:
        if len(set(value)) != len(value):
            raise ValueError("rubric accepted proposal evidence IDs must not repeat")
        if value != tuple(sorted(value)):
            raise ValueError("rubric accepted proposal evidence IDs must be sorted")
        return value

    @model_validator(mode="after")
    def _require_consistent_approval(self) -> Rubric:
        expected_input_ids = tuple(
            sorted((self.source_task_set_id, *self.accepted_proposal_evidence_ids))
        )
        if tuple(item.artifact_id for item in self.inputs) != expected_input_ids:
            raise ValueError(
                "rubrics must hash their source task set and accepted proposal evidence"
            )
        if self.status == "human_approved" and self.approved_at is None:
            raise ValueError("human-approved rubrics require approved_at")
        if self.status == "provisional" and self.approved_at is not None:
            raise ValueError("provisional rubrics must not set approved_at")
        if self.approved_at is not None and (
            self.approved_at.tzinfo is None or self.approved_at.utcoffset() is None
        ):
            raise ValueError("rubric approval times must include a timezone")
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

    def apply(self, raw_score: int) -> float:
        """Return the calibrated value for one integer raw judge score.

        Args:
            raw_score: Raw zero-to-five score emitted by the structured judge.

        Returns:
            The frozen monotonic calibrated score.

        Raises:
            ValueError: The raw score is outside the supported zero-to-five range.
        """
        if raw_score not in range(6):
            raise ValueError("raw judge scores must be integers from zero through five")
        return self.calibrated_scores[raw_score]


class JudgeCalibration(ArtifactEnvelope):
    """Frozen judge model, prompt, label lineage, and score maps for one rubric."""

    calibration_id: ArtifactId
    rubric_id: ArtifactId
    judge_model: ModelSnapshot
    judge_prompt_id: str = Field(min_length=1, max_length=256)
    judge_prompt_sha256: Sha256
    label_set_id: ArtifactId
    calibration_lineage_ids: tuple[ArtifactId, ...]
    excluded_router_held_out_lineage_ids: tuple[ArtifactId, ...]
    validation_method: Literal["grouped_k_fold"]
    out_of_fold_report_id: ArtifactId
    out_of_fold_report_sha256: Sha256
    score_maps: tuple[DimensionScoreMap, ...]
    label_count: int = Field(default=0, ge=0)
    recommended_label_count: Literal[10] = 10
    status: Literal["provisional", "insufficient", "human_calibrated"] = "provisional"
    approved_at: datetime | None = None
    risk_acceptance: ArtifactInput | None = None

    @field_validator("calibration_lineage_ids", "excluded_router_held_out_lineage_ids")
    @classmethod
    def _require_unique_lineage_ids(cls, value: tuple[ArtifactId, ...]) -> tuple[ArtifactId, ...]:
        if len(set(value)) != len(value):
            raise ValueError("judge calibration lineage IDs must not repeat")
        return value

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

    @model_validator(mode="after")
    def _require_sealed_calibration_lineages(self) -> JudgeCalibration:
        if set(self.calibration_lineage_ids).intersection(
            self.excluded_router_held_out_lineage_ids
        ):
            raise ValueError("calibration lineages must exclude router-held-out lineages")
        if self.status == "human_calibrated" and self.approved_at is None:
            raise ValueError("human-calibrated judge calibrations require approved_at")
        if self.status != "human_calibrated" and self.approved_at is not None:
            raise ValueError("unapproved judge calibrations must not set approved_at")
        if self.status != "human_calibrated" and self.risk_acceptance is not None:
            raise ValueError("only human-calibrated judges can name risk acceptance evidence")
        if self.status == "provisional" and self.label_count != 0:
            raise ValueError("provisional judge calibrations require zero human labels")
        if self.status == "provisional" and any(
            score_map.calibrated_scores != (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
            for score_map in self.score_maps
        ):
            raise ValueError("provisional judge calibrations require identity score maps")
        if self.status == "insufficient" and self.label_count == 0:
            raise ValueError("insufficient judge calibrations require at least one human label")
        if self.status == "insufficient" and not self.calibration_lineage_ids:
            raise ValueError("insufficient judge calibrations require fit lineages")
        if self.status == "human_calibrated" and self.label_count == 0:
            raise ValueError("human-calibrated judge calibrations require human labels")
        if self.status == "human_calibrated" and not self.calibration_lineage_ids:
            raise ValueError("human-calibrated judge calibrations require fit lineages")
        if self.approved_at is not None and (
            self.approved_at.tzinfo is None or self.approved_at.utcoffset() is None
        ):
            raise ValueError("judge calibration approval times must include a timezone")
        return self
