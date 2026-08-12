"""Per-dimension grouped out-of-fold fitting and metric calculations."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, field_validator

from wmo.common.core.artifacts import ArtifactId, ContractModel
from wmo.common.judging.labels import HumanScore
from wmo.common.judging.rubric import DimensionScoreMap, Rubric


class CalibrationDatum(ContractModel):
    """One active human score joined to one verified raw judge observation."""

    human_score: HumanScore
    raw_score: Literal[0, 1, 2, 3, 4, 5]


class OutOfFoldPrediction(ContractModel):
    """One valid lineage-grouped held-out calibration prediction."""

    label_id: ArtifactId
    rollout_id: ArtifactId
    lineage_id: ArtifactId
    dimension_id: ArtifactId
    fold_index: int = Field(ge=0)
    raw_score: Literal[0, 1, 2, 3, 4, 5]
    human_score: Literal[0, 1, 2, 3, 4, 5]
    calibrated_score: float = Field(ge=0, le=5)
    absolute_error: float = Field(ge=0)
    optimistic_error: float = Field(ge=0)

    @field_validator("calibrated_score", "absolute_error", "optimistic_error")
    @classmethod
    def _require_finite_metrics(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("calibration prediction metrics must be finite")
        return value


class DimensionCalibrationMetrics(ContractModel):
    """Per-dimension denominators and valid grouped out-of-fold error evidence."""

    dimension_id: ArtifactId
    label_count: int = Field(ge=0)
    rollout_count: int = Field(ge=0)
    lineage_count: int = Field(ge=0)
    fold_count: int = Field(ge=0)
    out_of_fold_prediction_count: int = Field(ge=0)
    mae: float | None = Field(default=None, ge=0)
    rank_agreement: float | None = Field(default=None, ge=-1, le=1)
    mean_optimistic_error: float | None = Field(default=None, ge=0)
    maximum_optimistic_error: float | None = Field(default=None, ge=0)

    @field_validator("mae", "rank_agreement", "mean_optimistic_error", "maximum_optimistic_error")
    @classmethod
    def _require_finite_optional_metrics(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("calibration metrics must be finite")
        return value


class WorstDisagreement(ContractModel):
    """One visible high-error prediction for human calibration review."""

    prediction: OutOfFoldPrediction
    direction: Literal["optimistic", "pessimistic", "exact"]


def fit_score_map(dimension_id: ArtifactId, data: Sequence[CalibrationDatum]) -> DimensionScoreMap:
    """Fit a monotonic score map with deterministic interpolation for absent raw scores.

    Args:
        dimension_id: Rubric dimension represented by the calibration observations.
        data: Active human scores paired with verified raw judge scores.

    Returns:
        Monotonic mapping from each raw score to a calibrated human-equivalent score.
    """
    by_raw_score: dict[int, list[int]] = defaultdict(list)
    for item in data:
        by_raw_score[item.raw_score].append(item.human_score.score)
    if not by_raw_score:
        return DimensionScoreMap(
            dimension_id=dimension_id,
            calibrated_scores=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
        )
    blocks = [
        _IsotonicBlock(
            start=raw_score,
            end=raw_score,
            weighted_total=float(sum(scores)),
            weight=len(scores),
        )
        for raw_score, scores in sorted(by_raw_score.items())
    ]
    fitted: list[_IsotonicBlock] = []
    for block in blocks:
        fitted.append(block)
        while len(fitted) >= 2 and fitted[-2].mean > fitted[-1].mean:
            right = fitted.pop()
            left = fitted.pop()
            fitted.append(
                _IsotonicBlock(
                    start=left.start,
                    end=right.end,
                    weighted_total=left.weighted_total + right.weighted_total,
                    weight=left.weight + right.weight,
                )
            )
    observed = {
        raw_score: block.mean
        for block in fitted
        for raw_score in range(block.start, block.end + 1)
        if raw_score in by_raw_score
    }
    return DimensionScoreMap(
        dimension_id=dimension_id,
        calibrated_scores=(
            _interpolate_score(0, observed),
            _interpolate_score(1, observed),
            _interpolate_score(2, observed),
            _interpolate_score(3, observed),
            _interpolate_score(4, observed),
            _interpolate_score(5, observed),
        ),
    )


def grouped_predictions_and_metrics(
    rubric: Rubric, data: Sequence[CalibrationDatum]
) -> tuple[tuple[OutOfFoldPrediction, ...], tuple[DimensionCalibrationMetrics, ...]]:
    """Fit valid lineage-grouped folds and summarize their held-out predictions.

    Args:
        rubric: Approved rubric whose dimensions define the calibration groups.
        data: Active human score and raw-judge observation pairs.

    Returns:
        Held-out predictions followed by metrics for every rubric dimension.
    """
    predictions: list[OutOfFoldPrediction] = []
    metrics: list[DimensionCalibrationMetrics] = []
    for dimension in rubric.dimensions:
        dimension_data = tuple(
            item for item in data if item.human_score.dimension_id == dimension.dimension_id
        )
        lineages = tuple(sorted({item.human_score.lineage_id for item in dimension_data}))
        dimension_predictions, fold_count = _dimension_predictions(
            dimension_id=dimension.dimension_id,
            data=dimension_data,
            lineages=lineages,
        )
        predictions.extend(dimension_predictions)
        metrics.append(
            _metrics_for_dimension(
                dimension_id=dimension.dimension_id,
                label_count=len(dimension_data),
                rollout_count=len({item.human_score.rollout_id for item in dimension_data}),
                lineage_count=len(lineages),
                fold_count=fold_count,
                predictions=dimension_predictions,
            )
        )
    return tuple(predictions), tuple(metrics)


def has_valid_grouped_oof(
    metrics: DimensionCalibrationMetrics,
    predictions: Sequence[OutOfFoldPrediction],
) -> bool:
    """Return whether one dimension has complete nonempty grouped held-out evidence.

    Args:
        metrics: Summary metrics for one rubric dimension.
        predictions: Held-out predictions for the same dimension.

    Returns:
        True when every expected grouped fold and label has valid held-out evidence.
    """
    if metrics.label_count == 0 or metrics.lineage_count < 2:
        return False
    expected_fold_count = min(5, metrics.lineage_count)
    if metrics.fold_count != expected_fold_count:
        return False
    if metrics.out_of_fold_prediction_count != metrics.label_count:
        return False
    if len(predictions) != metrics.label_count:
        return False
    expected_folds = set(range(expected_fold_count))
    if {item.fold_index for item in predictions} != expected_folds:
        return False
    return all(
        any(item.fold_index == fold_index for item in predictions) for fold_index in expected_folds
    )


def worst_disagreements(
    predictions: Sequence[OutOfFoldPrediction],
) -> tuple[WorstDisagreement, ...]:
    """Return the ten largest valid OOF disagreements with deterministic tie ordering.

    Args:
        predictions: Valid held-out calibrated predictions to inspect.

    Returns:
        Largest absolute-error disagreements, labeled by calibration direction.
    """
    disagreements: list[WorstDisagreement] = []
    for prediction in sorted(predictions, key=lambda item: (-item.absolute_error, item.label_id))[
        :10
    ]:
        error = prediction.calibrated_score - prediction.human_score
        direction: Literal["optimistic", "pessimistic", "exact"]
        if error > 0:
            direction = "optimistic"
        elif error < 0:
            direction = "pessimistic"
        else:
            direction = "exact"
        disagreements.append(WorstDisagreement(prediction=prediction, direction=direction))
    return tuple(disagreements)


class _IsotonicBlock:
    """One pooled-adjacent-violators block used only during a local monotonic fit."""

    def __init__(self, start: int, end: int, weighted_total: float, weight: int) -> None:
        self.start = start
        self.end = end
        self.weighted_total = weighted_total
        self.weight = weight

    @property
    def mean(self) -> float:
        """Return this block's weighted human-score mean."""
        return self.weighted_total / self.weight


def _dimension_predictions(
    *,
    dimension_id: ArtifactId,
    data: Sequence[CalibrationDatum],
    lineages: tuple[ArtifactId, ...],
) -> tuple[tuple[OutOfFoldPrediction, ...], int]:
    """Generate predictions only for folds with nonempty train and held-out partitions."""
    if len(lineages) < 2:
        return (), 0
    requested_fold_count = min(5, len(lineages))
    fold_by_lineage = {
        lineage_id: index % requested_fold_count for index, lineage_id in enumerate(lineages)
    }
    predictions: list[OutOfFoldPrediction] = []
    emitted_fold_count = 0
    for fold_index in range(requested_fold_count):
        held_out = tuple(
            item for item in data if fold_by_lineage[item.human_score.lineage_id] == fold_index
        )
        training = tuple(
            item for item in data if fold_by_lineage[item.human_score.lineage_id] != fold_index
        )
        if not held_out or not training:
            continue
        emitted_fold_count += 1
        score_map = fit_score_map(dimension_id, training)
        for datum in held_out:
            calibrated_score = score_map.apply(datum.raw_score)
            error = calibrated_score - datum.human_score.score
            predictions.append(
                OutOfFoldPrediction(
                    label_id=datum.human_score.label_id,
                    rollout_id=datum.human_score.rollout_id,
                    lineage_id=datum.human_score.lineage_id,
                    dimension_id=dimension_id,
                    fold_index=fold_index,
                    raw_score=datum.raw_score,
                    human_score=datum.human_score.score,
                    calibrated_score=calibrated_score,
                    absolute_error=abs(error),
                    optimistic_error=max(error, 0.0),
                )
            )
    return tuple(predictions), emitted_fold_count


def _metrics_for_dimension(
    *,
    dimension_id: ArtifactId,
    label_count: int,
    rollout_count: int,
    lineage_count: int,
    fold_count: int,
    predictions: Sequence[OutOfFoldPrediction],
) -> DimensionCalibrationMetrics:
    """Calculate dimension metrics exclusively from valid emitted OOF predictions."""
    if not predictions:
        return DimensionCalibrationMetrics(
            dimension_id=dimension_id,
            label_count=label_count,
            rollout_count=rollout_count,
            lineage_count=lineage_count,
            fold_count=fold_count,
            out_of_fold_prediction_count=0,
        )
    absolute_errors = tuple(item.absolute_error for item in predictions)
    optimistic_errors = tuple(item.optimistic_error for item in predictions)
    return DimensionCalibrationMetrics(
        dimension_id=dimension_id,
        label_count=label_count,
        rollout_count=rollout_count,
        lineage_count=lineage_count,
        fold_count=fold_count,
        out_of_fold_prediction_count=len(predictions),
        mae=sum(absolute_errors) / len(absolute_errors),
        rank_agreement=_spearman(
            tuple(item.calibrated_score for item in predictions),
            tuple(float(item.human_score) for item in predictions),
        ),
        mean_optimistic_error=sum(optimistic_errors) / len(optimistic_errors),
        maximum_optimistic_error=max(optimistic_errors),
    )


def _interpolate_score(raw_score: int, observed: dict[int, float]) -> float:
    """Fill a missing raw-score bucket without violating the fitted monotonic map."""
    if raw_score in observed:
        return observed[raw_score]
    lower = [score for score in observed if score < raw_score]
    upper = [score for score in observed if score > raw_score]
    if not lower:
        return observed[min(upper)]
    if not upper:
        return observed[max(lower)]
    left = max(lower)
    right = min(upper)
    return observed[left] + (raw_score - left) / (right - left) * (observed[right] - observed[left])


def _spearman(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    """Return tie-aware Spearman agreement or None when either side lacks variance."""
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (left_rank - left_mean) * (right_rank - right_mean)
        for left_rank, right_rank in zip(left_ranks, right_ranks, strict=True)
    )
    left_denominator = math.sqrt(sum((rank - left_mean) ** 2 for rank in left_ranks))
    right_denominator = math.sqrt(sum((rank - right_mean) ** 2 for rank in right_ranks))
    if left_denominator == 0 or right_denominator == 0:
        return None
    return max(-1.0, min(1.0, numerator / (left_denominator * right_denominator)))


def _average_ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    """Assign one-based average ranks while preserving deterministic equal-value ordering."""
    ranked = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    results = [0.0] * len(values)
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][1] == ranked[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for index, _value in ranked[start:end]:
            results[index] = average_rank
        start = end
    return tuple(results)
