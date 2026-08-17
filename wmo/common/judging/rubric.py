"""Canonical rubric and judge-calibration contracts."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import AwareDatetime, Field, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
)
from wmo.common.models import ModelSnapshot

MAX_AXIS_SCORE = 10


class ScoreAnchor(ContractModel):
    """Plain-language meaning for one integer score on a rubric axis."""

    score: int = Field(ge=0, le=MAX_AXIS_SCORE)
    description: str = Field(min_length=1)


class RubricDimension(ContractModel):
    """One scored axis with an inclusive integer range and score meanings."""

    dimension_id: ArtifactId
    name: str = Field(min_length=1, max_length=256)
    description: str = Field(min_length=1)
    min_score: int = Field(default=0, ge=0, le=MAX_AXIS_SCORE)
    max_score: int = Field(default=5, ge=0, le=MAX_AXIS_SCORE)
    anchors: tuple[ScoreAnchor, ...]

    @model_validator(mode="after")
    def _require_valid_range_and_anchors(self) -> RubricDimension:
        if self.min_score >= self.max_score:
            raise ValueError("rubric axis range must be inclusive with min_score below max_score")
        permitted = self.permitted_scores()
        scores = tuple(anchor.score for anchor in self.anchors)
        if not scores:
            raise ValueError("a rubric axis needs at least one score meaning")
        if len(set(scores)) != len(scores):
            raise ValueError("rubric axis anchors must have unique scores")
        if scores != tuple(sorted(scores)):
            raise ValueError("rubric axis anchors must be ordered by increasing score")
        unknown = tuple(score for score in scores if score not in permitted)
        if unknown:
            raise ValueError("rubric axis anchors must stay inside the inclusive score range")
        if scores[0] != self.min_score or scores[-1] != self.max_score:
            raise ValueError("rubric axis anchors must include the inclusive range endpoints")
        return self

    def permitted_scores(self) -> tuple[int, ...]:
        """Return every integer score inside this inclusive axis range."""
        return tuple(range(self.min_score, self.max_score + 1))

    def contains_score(self, score: int) -> bool:
        """Return whether ``score`` is an integer inside this inclusive range."""
        return score in self.permitted_scores()

    def normalize_score(self, score: float) -> float:
        """Map one axis-unit score onto the unit interval.

        Args:
            score: Raw or calibrated value on this axis scale.

        Returns:
            ``(score - min_score) / (max_score - min_score)``.
        """
        return (score - self.min_score) / (self.max_score - self.min_score)

    def prompt_payload(self) -> JsonObject:
        """Return the single prompt-facing payload for this axis."""
        return {
            "dimension_id": self.dimension_id,
            "name": self.name,
            "description": self.description,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "anchors": [anchor.model_dump(mode="json") for anchor in self.anchors],
        }

    def identity_calibrated_scores(self) -> tuple[float, ...]:
        """Return the identity map covering every permitted raw score."""
        return tuple(float(score) for score in self.permitted_scores())


def scored_axis(
    dimension_id: ArtifactId,
    name: str,
    description: str,
    *,
    min_score: int = 0,
    max_score: int = 5,
    anchor_stem: str | None = None,
) -> RubricDimension:
    """Build one axis with a complete meaning for every permitted score.

    Args:
        dimension_id: Stable axis identity.
        name: Human label shown in setup and calibration.
        description: Plain-language meaning of the axis.
        min_score: Inclusive lower bound.
        max_score: Inclusive upper bound.
        anchor_stem: Optional prefix for generated score meanings.

    Returns:
        A validated axis with one anchor per integer in the range.
    """
    stem = name if anchor_stem is None else anchor_stem
    return RubricDimension(
        dimension_id=dimension_id,
        name=name,
        description=description,
        min_score=min_score,
        max_score=max_score,
        anchors=tuple(
            ScoreAnchor(score=score, description=f"{stem} {score}.")
            for score in range(min_score, max_score + 1)
        ),
    )


def default_task_success_axis() -> RubricDimension:
    """Return the default 0-1 task-success axis."""
    return RubricDimension(
        dimension_id="task-success",
        name="Task success",
        description=(
            "The agent successfully completed the task requested in the original user prompt"
        ),
        min_score=0,
        max_score=1,
        anchors=(
            ScoreAnchor(score=0, description="The agent did not complete the requested task."),
            ScoreAnchor(
                score=1,
                description="The agent successfully completed the requested task.",
            ),
        ),
    )


def score_bounds(dimensions: tuple[RubricDimension, ...]) -> tuple[int, int]:
    """Return the shared inclusive axis range.

    Args:
        dimensions: Non-empty ordered rubric axes.

    Returns:
        The inclusive ``(min_score, max_score)`` shared by every axis.

    Raises:
        ValueError: The rubric has no axes, or axes use different ranges.
    """
    if not dimensions:
        raise ValueError("a rubric must contain at least one axis")
    ranges = {(item.min_score, item.max_score) for item in dimensions}
    if len(ranges) != 1:
        raise ValueError("every rubric axis must share the same inclusive score range")
    return next(iter(ranges))


class Rubric(ArtifactEnvelope):
    """An immutable, versioned ordered list of equal-weight scoring axes."""

    rubric_id: ArtifactId
    dimensions: tuple[RubricDimension, ...]
    source_task_set_id: ArtifactId
    accepted_proposal_evidence_ids: tuple[ArtifactId, ...] = ()
    status: Literal["provisional", "human_approved"]
    approved_at: AwareDatetime | None = None

    @field_validator("dimensions")
    @classmethod
    def _require_unique_dimensions(
        cls, value: tuple[RubricDimension, ...]
    ) -> tuple[RubricDimension, ...]:
        if not value:
            raise ValueError("a rubric must contain at least one axis")
        dimension_ids = tuple(dimension.dimension_id for dimension in value)
        if len(set(dimension_ids)) != len(dimension_ids):
            raise ValueError("rubric axes must have unique IDs")
        score_bounds(value)
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
        return self

    def axis(self, dimension_id: ArtifactId) -> RubricDimension:
        """Return the axis with ``dimension_id``.

        Args:
            dimension_id: Stable axis identity.

        Returns:
            The matching axis.

        Raises:
            ValueError: The rubric does not contain that axis.
        """
        for dimension in self.dimensions:
            if dimension.dimension_id == dimension_id:
                return dimension
        raise ValueError(f"rubric has no axis {dimension_id}")


class DimensionScoreMap(ContractModel):
    """A monotonic mapping from raw judge scores to expected human scores."""

    dimension_id: ArtifactId
    min_score: int = Field(default=0, ge=0, le=MAX_AXIS_SCORE)
    max_score: int = Field(default=5, ge=0, le=MAX_AXIS_SCORE)
    calibrated_scores: tuple[float, ...]

    @model_validator(mode="after")
    def _require_monotonic_range_scores(self) -> DimensionScoreMap:
        if self.min_score >= self.max_score:
            raise ValueError("score-map range must be inclusive with min_score below max_score")
        expected_length = self.max_score - self.min_score + 1
        if len(self.calibrated_scores) != expected_length:
            raise ValueError("calibrated rubric scores must cover every integer in the axis range")
        if any(
            not math.isfinite(score) or score < self.min_score or score > self.max_score
            for score in self.calibrated_scores
        ):
            raise ValueError("calibrated rubric scores must be finite values inside the axis range")
        if tuple(sorted(self.calibrated_scores)) != self.calibrated_scores:
            raise ValueError("calibrated rubric scores must be monotonic")
        return self

    def identity_scores(self) -> tuple[float, ...]:
        """Return the identity map for this score-map range."""
        return tuple(float(score) for score in range(self.min_score, self.max_score + 1))

    def is_identity(self) -> bool:
        """Return whether this map leaves every raw score unchanged."""
        return self.calibrated_scores == self.identity_scores()

    def apply(self, raw_score: int) -> float:
        """Return the calibrated value for one integer raw judge score.

        Args:
            raw_score: Raw integer score emitted by the structured judge.

        Returns:
            The frozen monotonic calibrated score.

        Raises:
            ValueError: The raw score is outside this map's inclusive range.
        """
        if raw_score < self.min_score or raw_score > self.max_score:
            raise ValueError("raw judge scores must be integers inside the axis range")
        return self.calibrated_scores[raw_score - self.min_score]


def identity_score_map(
    dimension_id: ArtifactId, min_score: int, max_score: int
) -> DimensionScoreMap:
    """Return the identity map for one axis range.

    Args:
        dimension_id: Axis receiving the map.
        min_score: Inclusive lower bound.
        max_score: Inclusive upper bound.

    Returns:
        A monotonic identity map covering every permitted raw score.
    """
    return DimensionScoreMap(
        dimension_id=dimension_id,
        min_score=min_score,
        max_score=max_score,
        calibrated_scores=tuple(float(score) for score in range(min_score, max_score + 1)),
    )


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
    approved_at: AwareDatetime | None = None
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
            raise ValueError("judge calibration needs a score map for every rubric axis")
        if len(set(dimension_ids)) != len(dimension_ids):
            raise ValueError("judge calibration score maps must not repeat an axis")
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
            not score_map.is_identity() for score_map in self.score_maps
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
        return self
