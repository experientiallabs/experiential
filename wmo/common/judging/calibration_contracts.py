"""Persisted report and observation contracts for common judge calibration."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    Sha256,
)
from wmo.common.judging.calibration_metrics import (
    DimensionCalibrationMetrics,
    OutOfFoldPrediction,
    WorstDisagreement,
)
from wmo.common.judging.lineage import RouterLineageSplit
from wmo.common.judging.rubric import DimensionScoreMap
from wmo.common.models import ModelSnapshot


class JudgeScoreObservation(ContractModel):
    """Raw dimension evidence bound to a stored judgment and rollout."""

    judgment: ArtifactInput
    source_rollout: ArtifactInput
    dimension_id: ArtifactId
    raw_score: Literal[0, 1, 2, 3, 4, 5]
    evidence_span_ids: tuple[str, ...]

    @field_validator("evidence_span_ids")
    @classmethod
    def _require_unique_evidence_spans(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("calibration observations require cited rollout spans")
        if len(set(value)) != len(value):
            raise ValueError("calibration observation citations must not repeat")
        return value


class CalibrationReport(ArtifactEnvelope):
    """Frozen per-dimension OOF evidence and eligible-label refit maps."""

    report_id: ArtifactId
    rubric_id: ArtifactId
    rubric_dimension_ids: tuple[ArtifactId, ...]
    judge_model: ModelSnapshot
    judge_prompt_id: str = Field(min_length=1, max_length=256)
    judge_prompt_sha256: Sha256
    label_set_id: ArtifactId
    router_lineage_split_id: ArtifactId
    router_lineages: RouterLineageSplit
    observations: tuple[JudgeScoreObservation, ...]
    eligible_label_count: int = Field(ge=0)
    eligible_rollout_count: int = Field(ge=0)
    eligible_lineage_ids: tuple[ArtifactId, ...]
    eligible_lineage_count: int = Field(ge=0)
    excluded_held_out_label_count: int = Field(ge=0)
    excluded_held_out_rollout_count: int = Field(ge=0)
    excluded_held_out_lineage_ids: tuple[ArtifactId, ...]
    excluded_held_out_lineage_count: int = Field(ge=0)
    recommended_label_count: Literal[10] = 10
    status: Literal["provisional", "insufficient", "ready_for_approval"]
    score_maps: tuple[DimensionScoreMap, ...]
    dimension_metrics: tuple[DimensionCalibrationMetrics, ...]
    out_of_fold_predictions: tuple[OutOfFoldPrediction, ...]
    worst_disagreements: tuple[WorstDisagreement, ...]

    @field_validator(
        "rubric_dimension_ids", "eligible_lineage_ids", "excluded_held_out_lineage_ids"
    )
    @classmethod
    def _require_unique_ids(cls, value: tuple[ArtifactId, ...]) -> tuple[ArtifactId, ...]:
        if len(set(value)) != len(value):
            raise ValueError("calibration report IDs must not repeat")
        return value

    @field_validator("eligible_lineage_ids", "excluded_held_out_lineage_ids")
    @classmethod
    def _require_sorted_lineages(cls, value: tuple[ArtifactId, ...]) -> tuple[ArtifactId, ...]:
        if value != tuple(sorted(value)):
            raise ValueError("calibration report eligible lineages must be sorted")
        return value

    @model_validator(mode="after")
    def _require_coherent_report(self) -> CalibrationReport:
        metric_ids = tuple(item.dimension_id for item in self.dimension_metrics)
        map_ids = tuple(item.dimension_id for item in self.score_maps)
        if not self.rubric_dimension_ids:
            raise ValueError("calibration reports require every rubric dimension")
        if set(metric_ids) != set(self.rubric_dimension_ids):
            raise ValueError("calibration report metrics must cover every rubric dimension")
        if set(map_ids) != set(self.rubric_dimension_ids):
            raise ValueError("calibration report score maps must cover every rubric dimension")
        if self.router_lineages.split_id != self.router_lineage_split_id:
            raise ValueError("calibration report must retain its exact lineage split")
        eligible = set(self.eligible_lineage_ids)
        fit = set(self.router_lineages.fit_lineage_ids)
        held_out = set(self.router_lineages.held_out_lineage_ids)
        if not eligible.issubset(fit) or eligible.intersection(held_out):
            raise ValueError("calibration report lineages must be eligible router fit lineages")
        if any(item.lineage_id not in eligible for item in self.out_of_fold_predictions):
            raise ValueError("out-of-fold predictions must use eligible calibration lineages")
        if self.eligible_lineage_count != len(self.eligible_lineage_ids):
            raise ValueError("calibration report eligible lineage denominator must be exact")
        if self.excluded_held_out_lineage_count != len(self.excluded_held_out_lineage_ids):
            raise ValueError("calibration report held-out lineage denominator must be exact")
        expected_input_ids = tuple(
            sorted(
                {
                    self.rubric_id,
                    self.label_set_id,
                    self.router_lineage_split_id,
                    *(item.judgment.artifact_id for item in self.observations),
                    *(item.source_rollout.artifact_id for item in self.observations),
                }
            )
        )
        if tuple(item.artifact_id for item in self.inputs) != expected_input_ids:
            raise ValueError("calibration reports must hash exactly their frozen source artifacts")
        if self.status == "provisional":
            self._require_provisional_shape()
        elif not self.observations:
            raise ValueError("non-provisional calibration reports require stored final judgments")
        return self

    def _require_provisional_shape(self) -> None:
        """Reject provisional records that claim evidence or a learned score map."""
        identity = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
        if self.observations or self.eligible_label_count or self.eligible_rollout_count:
            raise ValueError("provisional calibration reports require zero eligible labels")
        if self.eligible_lineage_ids or self.eligible_lineage_count:
            raise ValueError("provisional calibration reports require zero eligible lineages")
        if self.excluded_held_out_label_count or self.excluded_held_out_rollout_count:
            raise ValueError("provisional calibration reports require zero held-out labels")
        if self.excluded_held_out_lineage_ids or self.excluded_held_out_lineage_count:
            raise ValueError("provisional calibration reports require zero held-out lineages")
        if self.out_of_fold_predictions or self.worst_disagreements:
            raise ValueError("provisional calibration reports cannot claim OOF evidence")
        if any(score_map.calibrated_scores != identity for score_map in self.score_maps):
            raise ValueError("provisional calibration reports require identity score maps")
        if any(
            metric.label_count
            or metric.rollout_count
            or metric.lineage_count
            or metric.fold_count
            or metric.out_of_fold_prediction_count
            or metric.mae is not None
            or metric.rank_agreement is not None
            or metric.mean_optimistic_error is not None
            or metric.maximum_optimistic_error is not None
            for metric in self.dimension_metrics
        ):
            raise ValueError("provisional calibration reports require zero metric denominators")
