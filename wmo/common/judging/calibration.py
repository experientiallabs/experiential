"""Leakage-safe grouped calibration and immutable approval services for LM judges."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import ArtifactEnvelope, ArtifactId, ContractModel, Sha256, stable_id
from wmo.common.judging.labels import HumanLabelSet, HumanScore
from wmo.common.judging.prompts import PromptDefinition
from wmo.common.judging.rubric import DimensionScoreMap, JudgeCalibration, Rubric
from wmo.common.models import ModelSnapshot
from wmo.common.project import ArtifactAlreadyExistsError, ProjectStore


class CalibrationError(ValueError):
    """Raised when calibration evidence, grouped validation, or approval is invalid."""


class RouterLineageSplit(ContractModel):
    """Frozen fit and router-held-out lineage partitions consumed by calibration."""

    fit_lineage_ids: tuple[ArtifactId, ...]
    held_out_lineage_ids: tuple[ArtifactId, ...]

    @field_validator("fit_lineage_ids", "held_out_lineage_ids")
    @classmethod
    def _require_unique_lineages(cls, value: tuple[ArtifactId, ...]) -> tuple[ArtifactId, ...]:
        if len(set(value)) != len(value):
            raise ValueError("router lineage IDs must not repeat")
        return value

    @model_validator(mode="after")
    def _require_disjoint_partitions(self) -> RouterLineageSplit:
        if set(self.fit_lineage_ids).intersection(self.held_out_lineage_ids):
            raise ValueError("router fit and held-out lineage IDs must be disjoint")
        return self


class JudgeScoreObservation(ContractModel):
    """Raw zero-to-five LM score shown to a human reviewer before correction."""

    rubric_id: ArtifactId
    rollout_id: ArtifactId
    lineage_id: ArtifactId
    dimension_id: ArtifactId
    raw_score: Literal[0, 1, 2, 3, 4, 5]


class CalibrationDatum(ContractModel):
    """One joined active human score and corresponding raw LM judge score."""

    human_score: HumanScore
    judge_score: JudgeScoreObservation

    @model_validator(mode="after")
    def _require_matching_human_and_judge_target(self) -> CalibrationDatum:
        human = self.human_score
        judge = self.judge_score
        if (human.rubric_id, human.rollout_id, human.lineage_id, human.dimension_id) != (
            judge.rubric_id,
            judge.rollout_id,
            judge.lineage_id,
            judge.dimension_id,
        ):
            raise ValueError("human and raw judge scores must name the same rollout and scale")
        return self


class OutOfFoldPrediction(ContractModel):
    """One grouped-held-out calibration prediction used for visible error reporting."""

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
    """Grouped out-of-fold accuracy and optimism metrics for one rubric dimension."""

    dimension_id: ArtifactId
    label_count: int = Field(ge=0)
    lineage_count: int = Field(ge=0)
    fold_count: int = Field(ge=0)
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
    """One visible high-error out-of-fold label for human calibration review."""

    prediction: OutOfFoldPrediction
    direction: Literal["optimistic", "pessimistic", "exact"]


class CalibrationReport(ArtifactEnvelope):
    """Frozen grouped out-of-fold evidence and full eligible-label refit maps."""

    report_id: ArtifactId
    rubric_id: ArtifactId
    judge_model: ModelSnapshot
    judge_prompt_id: str = Field(min_length=1, max_length=256)
    judge_prompt_sha256: Sha256
    label_set_id: ArtifactId
    router_lineages: RouterLineageSplit
    eligible_label_count: int = Field(ge=0)
    eligible_rollout_count: int = Field(ge=0)
    eligible_lineage_ids: tuple[ArtifactId, ...]
    excluded_held_out_label_count: int = Field(ge=0)
    excluded_held_out_rollout_count: int = Field(ge=0)
    recommended_label_count: Literal[10] = 10
    status: Literal["provisional", "insufficient", "ready_for_approval"]
    score_maps: tuple[DimensionScoreMap, ...]
    dimension_metrics: tuple[DimensionCalibrationMetrics, ...]
    out_of_fold_predictions: tuple[OutOfFoldPrediction, ...]
    worst_disagreements: tuple[WorstDisagreement, ...]

    @field_validator("score_maps", "dimension_metrics")
    @classmethod
    def _require_unique_dimension_records(
        cls,
        value: tuple[DimensionScoreMap, ...] | tuple[DimensionCalibrationMetrics, ...],
    ) -> tuple[DimensionScoreMap, ...] | tuple[DimensionCalibrationMetrics, ...]:
        dimension_ids = tuple(item.dimension_id for item in value)
        if len(set(dimension_ids)) != len(dimension_ids):
            raise ValueError("calibration report dimension records must not repeat")
        return value

    @model_validator(mode="after")
    def _require_lineage_safe_report(self) -> CalibrationReport:
        eligible = set(self.eligible_lineage_ids)
        fit = set(self.router_lineages.fit_lineage_ids)
        held_out = set(self.router_lineages.held_out_lineage_ids)
        if not eligible.issubset(fit):
            raise ValueError("calibration report lineages must come from router fit lineages")
        if eligible.intersection(held_out):
            raise ValueError("calibration report must exclude router-held-out lineages")
        prediction_lineages = {item.lineage_id for item in self.out_of_fold_predictions}
        if not prediction_lineages.issubset(eligible):
            raise ValueError("out-of-fold predictions must use eligible calibration lineages")
        return self


class JudgeCalibrationService:
    """Build provisional or grouped-calibrated score maps and require explicit approval."""

    def build_report(
        self,
        *,
        rubric: Rubric,
        judge_model: ModelSnapshot,
        prompt: PromptDefinition,
        label_set: HumanLabelSet,
        judge_scores: Sequence[JudgeScoreObservation],
        router_lineages: RouterLineageSplit,
        created_at: datetime,
        code_revision: str,
    ) -> CalibrationReport:
        """Fit leakage-safe score maps and expose grouped out-of-fold calibration evidence.

        Args:
            rubric: Frozen rubric whose dimensions need one calibrated score map each.
            judge_model: Exact resolved LM identity that produced the raw scores.
            prompt: Immutable prompt identity used for those raw scores.
            label_set: Frozen append-only human labels for this exact rubric version.
            judge_scores: Raw LM scores matched to every active human label.
            router_lineages: Frozen fit and held-out router partitions.
            created_at: Time this immutable report is materialized.
            code_revision: Exact code revision responsible for the report.

        Returns:
            A report with full eligible-label maps, grouped OOF metrics, and visible disagreements.

        Raises:
            CalibrationError: Labels or raw scores violate rubric or lineage boundaries.
        """
        _require_timezone(created_at)
        data, excluded = _join_eligible_data(
            rubric=rubric,
            label_set=label_set,
            judge_scores=judge_scores,
            router_lineages=router_lineages,
        )
        score_maps = tuple(
            _fit_score_map(dimension.dimension_id, _dimension_data(data, dimension.dimension_id))
            for dimension in rubric.dimensions
        )
        predictions, metrics = _grouped_predictions_and_metrics(rubric, data)
        status = _report_status(rubric, data, predictions)
        worst = _worst_disagreements(predictions)
        report_id = stable_id(
            "calibration-report",
            _CalibrationReportIdentity(
                rubric_id=rubric.rubric_id,
                judge_model=judge_model,
                judge_prompt_id=prompt.prompt_id,
                judge_prompt_sha256=prompt.sha256,
                label_set=label_set,
                router_lineages=router_lineages,
                eligible_data=data,
                excluded_data=excluded,
            ),
        )
        return CalibrationReport(
            schema_version=1,
            created_at=created_at,
            code_revision=code_revision,
            report_id=report_id,
            rubric_id=rubric.rubric_id,
            judge_model=judge_model,
            judge_prompt_id=prompt.prompt_id,
            judge_prompt_sha256=prompt.sha256,
            label_set_id=label_set.label_set_id,
            router_lineages=router_lineages,
            eligible_label_count=len(data),
            eligible_rollout_count=len({item.human_score.rollout_id for item in data}),
            eligible_lineage_ids=tuple(sorted({item.human_score.lineage_id for item in data})),
            excluded_held_out_label_count=len(excluded),
            excluded_held_out_rollout_count=len({item.human_score.rollout_id for item in excluded}),
            status=status,
            score_maps=score_maps,
            dimension_metrics=metrics,
            out_of_fold_predictions=predictions,
            worst_disagreements=worst,
        )

    def provisional_calibration(self, report: CalibrationReport) -> JudgeCalibration:
        """Return an identity-map provisional calibration when no human labels exist.

        Args:
            report: Empty-label calibration report for the rubric and fixed judge identity.

        Returns:
            A provisional identity-map calibration that remains visibly non-human-calibrated.

        Raises:
            CalibrationError: The report contains labels or is not provisional.
        """
        if report.status != "provisional" or report.eligible_label_count != 0:
            raise CalibrationError("provisional calibration requires a zero-label report")
        return JudgeCalibration(
            schema_version=1,
            created_at=report.created_at,
            code_revision=report.code_revision,
            calibration_id=stable_id("judge-calibration", report),
            rubric_id=report.rubric_id,
            judge_model=report.judge_model,
            judge_prompt_id=report.judge_prompt_id,
            judge_prompt_sha256=report.judge_prompt_sha256,
            label_set_id=report.label_set_id,
            calibration_lineage_ids=(),
            excluded_router_held_out_lineage_ids=report.router_lineages.held_out_lineage_ids,
            validation_method="grouped_k_fold",
            out_of_fold_report_id=report.report_id,
            score_maps=report.score_maps,
            label_count=0,
            status="provisional",
        )

    def approve(
        self,
        report: CalibrationReport,
        *,
        approved_at: datetime,
        accept_insufficient_labels: bool = False,
    ) -> JudgeCalibration:
        """Freeze an eligible-label refit after explicit customer approval.

        Args:
            report: Grouped cross-validation report being approved.
            approved_at: Time the customer accepted the visible calibration evidence.
            accept_insufficient_labels: Explicit risk acceptance required below the recommendation.

        Returns:
            A human-calibrated judge configuration with maps refit on all eligible labels.

        Raises:
            CalibrationError: The report is provisional or insufficient without explicit consent.
        """
        _require_timezone(approved_at)
        if report.status == "provisional":
            raise CalibrationError("zero-label provisional calibration cannot be human-approved")
        if report.status == "insufficient" and not accept_insufficient_labels:
            raise CalibrationError(
                "insufficient labels require explicit accept_insufficient_labels=True approval"
            )
        return JudgeCalibration(
            schema_version=1,
            created_at=approved_at,
            code_revision=report.code_revision,
            calibration_id=stable_id(
                "judge-calibration",
                _CalibrationIdentity(report_id=report.report_id, approved_at=approved_at),
            ),
            rubric_id=report.rubric_id,
            judge_model=report.judge_model,
            judge_prompt_id=report.judge_prompt_id,
            judge_prompt_sha256=report.judge_prompt_sha256,
            label_set_id=report.label_set_id,
            calibration_lineage_ids=report.eligible_lineage_ids,
            excluded_router_held_out_lineage_ids=report.router_lineages.held_out_lineage_ids,
            validation_method="grouped_k_fold",
            out_of_fold_report_id=report.report_id,
            score_maps=report.score_maps,
            label_count=report.eligible_label_count,
            status="human_calibrated",
            approved_at=approved_at,
        )

    def insufficient_calibration(self, report: CalibrationReport) -> JudgeCalibration:
        """Return a visibly unapproved calibration when labels are below the recommendation.

        Args:
            report: Nonzero-label grouped report that remains below the review recommendation.

        Returns:
            An insufficient calibration with the eligible-label refit and exact label denominator.

        Raises:
            CalibrationError: The report is provisional, ready for approval, or inconsistent.
        """
        if report.status != "insufficient" or report.eligible_label_count == 0:
            raise CalibrationError(
                "insufficient calibration requires a nonzero insufficient report"
            )
        return JudgeCalibration(
            schema_version=1,
            created_at=report.created_at,
            code_revision=report.code_revision,
            calibration_id=stable_id("judge-calibration", report),
            rubric_id=report.rubric_id,
            judge_model=report.judge_model,
            judge_prompt_id=report.judge_prompt_id,
            judge_prompt_sha256=report.judge_prompt_sha256,
            label_set_id=report.label_set_id,
            calibration_lineage_ids=report.eligible_lineage_ids,
            excluded_router_held_out_lineage_ids=report.router_lineages.held_out_lineage_ids,
            validation_method="grouped_k_fold",
            out_of_fold_report_id=report.report_id,
            score_maps=report.score_maps,
            label_count=report.eligible_label_count,
            status="insufficient",
        )

    def write_report(self, store: ProjectStore, report: CalibrationReport) -> CalibrationReport:
        """Persist one completed calibration report as an immutable project artifact.

        Args:
            store: Project-local artifact store that owns the completed report directory.
            report: Grouped out-of-fold report to freeze exactly as reviewed.

        Returns:
            The supplied report after its immutable artifact has been verified or written.

        Raises:
            CalibrationError: An existing artifact with the same ID differs from this report.
        """
        return _write_immutable_report(store, report)

    def write_calibration(
        self,
        store: ProjectStore,
        *,
        report: CalibrationReport,
        calibration: JudgeCalibration,
    ) -> JudgeCalibration:
        """Persist a calibration only when it is exactly bound to its reviewed report.

        Args:
            store: Project-local artifact store that owns the completed calibration directory.
            report: Immutable grouped evidence reviewed before approval.
            calibration: Provisional or approved score map to freeze for later judgment calls.

        Returns:
            The supplied calibration after its immutable artifact has been verified or written.

        Raises:
            CalibrationError: The calibration does not match the report or an existing artifact.
        """
        _require_calibration_report_binding(report, calibration)
        try:
            store.artifacts.write_json(
                artifact_id=calibration.calibration_id,
                artifact_type="judge-calibration",
                envelope=calibration,
                files={"calibration.json": calibration},
            )
        except ArtifactAlreadyExistsError:
            try:
                stored = JudgeCalibration.model_validate_json(
                    store.artifacts.read_bytes(calibration.calibration_id, "calibration.json")
                )
            except ValueError as exc:
                raise CalibrationError(
                    "existing judge-calibration artifact cannot be resumed safely"
                ) from exc
            if not _same_calibration_identity(stored, calibration):
                raise CalibrationError(
                    "existing judge-calibration artifact conflicts with this calibration"
                ) from None
            calibration = stored
        return calibration


class _CalibrationReportIdentity(ContractModel):
    """Content that fixes one deterministic calibration-report identity."""

    rubric_id: ArtifactId
    judge_model: ModelSnapshot
    judge_prompt_id: str
    judge_prompt_sha256: Sha256
    label_set: HumanLabelSet
    router_lineages: RouterLineageSplit
    eligible_data: tuple[CalibrationDatum, ...]
    excluded_data: tuple[CalibrationDatum, ...]


class _CalibrationIdentity(ContractModel):
    """Content that fixes one approved calibration identity."""

    report_id: ArtifactId
    approved_at: datetime


def _write_immutable_report(store: ProjectStore, report: CalibrationReport) -> CalibrationReport:
    """Write or verify one content-addressed calibration report artifact."""
    try:
        store.artifacts.write_json(
            artifact_id=report.report_id,
            artifact_type="judge-calibration-report",
            envelope=report,
            files={"report.json": report},
        )
    except ArtifactAlreadyExistsError:
        try:
            stored = CalibrationReport.model_validate_json(
                store.artifacts.read_bytes(report.report_id, "report.json")
            )
        except ValueError as exc:
            raise CalibrationError(
                "existing judge-calibration report artifact cannot be resumed safely"
            ) from exc
        if not _same_report_identity(stored, report):
            raise CalibrationError(
                "existing judge-calibration report artifact conflicts with this report"
            ) from None
        return stored
    return report


def _require_calibration_report_binding(
    report: CalibrationReport, calibration: JudgeCalibration
) -> None:
    """Require all report identities to agree before a calibration becomes immutable."""
    if calibration.rubric_id != report.rubric_id:
        raise CalibrationError("judge calibration belongs to a different rubric than its report")
    if calibration.judge_model != report.judge_model:
        raise CalibrationError("judge calibration model does not match its report")
    if (
        calibration.judge_prompt_id != report.judge_prompt_id
        or calibration.judge_prompt_sha256 != report.judge_prompt_sha256
    ):
        raise CalibrationError("judge calibration prompt does not match its report")
    if calibration.label_set_id != report.label_set_id:
        raise CalibrationError("judge calibration label set does not match its report")
    if calibration.out_of_fold_report_id != report.report_id:
        raise CalibrationError("judge calibration must name its out-of-fold report")
    if calibration.score_maps != report.score_maps:
        raise CalibrationError(
            "judge calibration score maps must be the report's eligible-label refit"
        )
    if calibration.calibration_lineage_ids != report.eligible_lineage_ids:
        raise CalibrationError("judge calibration lineages must match report eligible lineages")
    if (
        calibration.excluded_router_held_out_lineage_ids
        != report.router_lineages.held_out_lineage_ids
    ):
        raise CalibrationError("judge calibration held-out lineages must match its report")
    if calibration.label_count != report.eligible_label_count:
        raise CalibrationError("judge calibration label denominator must match its report")
    if calibration.status == "provisional" and report.status != "provisional":
        raise CalibrationError("only a zero-label report can produce provisional calibration")
    if calibration.status == "insufficient" and report.status != "insufficient":
        raise CalibrationError("only an insufficient report can produce insufficient calibration")
    if calibration.status == "human_calibrated" and report.status == "provisional":
        raise CalibrationError("a zero-label report cannot produce human calibration")


def _same_report_identity(left: CalibrationReport, right: CalibrationReport) -> bool:
    """Compare report evidence while permitting a safe retry with a later wall-clock time."""
    return (
        left.schema_version == right.schema_version
        and left.report_id == right.report_id
        and left.rubric_id == right.rubric_id
        and left.judge_model == right.judge_model
        and left.judge_prompt_id == right.judge_prompt_id
        and left.judge_prompt_sha256 == right.judge_prompt_sha256
        and left.label_set_id == right.label_set_id
        and left.router_lineages == right.router_lineages
        and left.eligible_label_count == right.eligible_label_count
        and left.eligible_rollout_count == right.eligible_rollout_count
        and left.eligible_lineage_ids == right.eligible_lineage_ids
        and left.excluded_held_out_label_count == right.excluded_held_out_label_count
        and left.excluded_held_out_rollout_count == right.excluded_held_out_rollout_count
        and left.recommended_label_count == right.recommended_label_count
        and left.status == right.status
        and left.score_maps == right.score_maps
        and left.dimension_metrics == right.dimension_metrics
        and left.out_of_fold_predictions == right.out_of_fold_predictions
        and left.worst_disagreements == right.worst_disagreements
        and left.code_revision == right.code_revision
        and left.inputs == right.inputs
        and left.source == right.source
    )


def _same_calibration_identity(left: JudgeCalibration, right: JudgeCalibration) -> bool:
    """Compare frozen calibration content without retry-time artifact timestamps."""
    return (
        left.schema_version == right.schema_version
        and left.calibration_id == right.calibration_id
        and left.rubric_id == right.rubric_id
        and left.judge_model == right.judge_model
        and left.judge_prompt_id == right.judge_prompt_id
        and left.judge_prompt_sha256 == right.judge_prompt_sha256
        and left.label_set_id == right.label_set_id
        and left.calibration_lineage_ids == right.calibration_lineage_ids
        and left.excluded_router_held_out_lineage_ids == right.excluded_router_held_out_lineage_ids
        and left.validation_method == right.validation_method
        and left.out_of_fold_report_id == right.out_of_fold_report_id
        and left.score_maps == right.score_maps
        and left.label_count == right.label_count
        and left.recommended_label_count == right.recommended_label_count
        and left.status == right.status
        and left.approved_at == right.approved_at
        and left.code_revision == right.code_revision
        and left.inputs == right.inputs
        and left.source == right.source
    )


def _join_eligible_data(
    *,
    rubric: Rubric,
    label_set: HumanLabelSet,
    judge_scores: Sequence[JudgeScoreObservation],
    router_lineages: RouterLineageSplit,
) -> tuple[tuple[CalibrationDatum, ...], tuple[CalibrationDatum, ...]]:
    """Join active labels to raw scores and remove router-held-out lineages."""
    if label_set.rubric_id != rubric.rubric_id:
        raise CalibrationError("human label set belongs to a different frozen rubric")
    rubric_dimension_ids = {dimension.dimension_id for dimension in rubric.dimensions}
    score_by_target: dict[tuple[str, str, str, str], JudgeScoreObservation] = {}
    for score in judge_scores:
        if score.rubric_id != rubric.rubric_id:
            raise CalibrationError("raw LM judge score belongs to a different frozen rubric")
        target = _target(
            score.rubric_id,
            score.rollout_id,
            score.lineage_id,
            score.dimension_id,
        )
        if target in score_by_target:
            raise CalibrationError("raw judge scores must not repeat a rollout and scale")
        score_by_target[target] = score
    fit_lineages = set(router_lineages.fit_lineage_ids)
    held_out_lineages = set(router_lineages.held_out_lineage_ids)
    eligible = []
    excluded = []
    for human_score in label_set.history.active_scores():
        if human_score.rubric_id != rubric.rubric_id:
            raise CalibrationError("human score belongs to a different frozen rubric")
        if human_score.dimension_id not in rubric_dimension_ids:
            raise CalibrationError("human score dimension is absent from the frozen rubric")
        if human_score.lineage_id not in fit_lineages | held_out_lineages:
            raise CalibrationError("human score lineage is absent from the frozen router split")
        target = _target(
            human_score.rubric_id,
            human_score.rollout_id,
            human_score.lineage_id,
            human_score.dimension_id,
        )
        judge_score = score_by_target.get(target)
        if judge_score is None:
            raise CalibrationError("every active human score needs a matching raw LM judge score")
        datum = CalibrationDatum(human_score=human_score, judge_score=judge_score)
        if human_score.lineage_id in held_out_lineages:
            excluded.append(datum)
        else:
            eligible.append(datum)
    active_targets = {
        _target(
            item.human_score.rubric_id,
            item.human_score.rollout_id,
            item.human_score.lineage_id,
            item.human_score.dimension_id,
        )
        for item in (*eligible, *excluded)
    }
    unused_scores = set(score_by_target) - active_targets
    if unused_scores:
        raise CalibrationError("raw LM judge scores must match active human labels exactly")
    return tuple(eligible), tuple(excluded)


def _dimension_data(
    data: Sequence[CalibrationDatum], dimension_id: ArtifactId
) -> tuple[CalibrationDatum, ...]:
    """Return the calibration data for one rubric dimension in stable input order."""
    return tuple(item for item in data if item.human_score.dimension_id == dimension_id)


def _fit_score_map(dimension_id: ArtifactId, data: Sequence[CalibrationDatum]) -> DimensionScoreMap:
    """Fit a monotonic pooled-adjacent-violators map with deterministic gap filling."""
    by_raw_score: dict[int, list[int]] = defaultdict(list)
    for item in data:
        by_raw_score[item.judge_score.raw_score].append(item.human_score.score)
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
    observed_values = {
        raw_score: block.mean
        for block in fitted
        for raw_score in range(block.start, block.end + 1)
        if raw_score in by_raw_score
    }
    return DimensionScoreMap(
        dimension_id=dimension_id,
        calibrated_scores=(
            _interpolate_score(0, observed_values),
            _interpolate_score(1, observed_values),
            _interpolate_score(2, observed_values),
            _interpolate_score(3, observed_values),
            _interpolate_score(4, observed_values),
            _interpolate_score(5, observed_values),
        ),
    )


class _IsotonicBlock:
    """Mutable-free pooled-adjacent-violators block used only during one local fit."""

    def __init__(self, start: int, end: int, weighted_total: float, weight: int) -> None:
        self.start = start
        self.end = end
        self.weighted_total = weighted_total
        self.weight = weight

    @property
    def mean(self) -> float:
        """Return the weighted average label score for this block."""
        return self.weighted_total / self.weight


def _interpolate_score(raw_score: int, observed: dict[int, float]) -> float:
    """Fill missing raw-score buckets without violating an isotonic fitted map."""
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
    left_value = observed[left]
    right_value = observed[right]
    fraction = (raw_score - left) / (right - left)
    return left_value + fraction * (right_value - left_value)


def _grouped_predictions_and_metrics(
    rubric: Rubric, data: Sequence[CalibrationDatum]
) -> tuple[tuple[OutOfFoldPrediction, ...], tuple[DimensionCalibrationMetrics, ...]]:
    """Run one frozen lineage-fold assignment and summarize OOF error per dimension."""
    predictions = []
    metrics = []
    all_lineages = tuple(sorted({item.human_score.lineage_id for item in data}))
    fold_count = min(5, len(all_lineages))
    fold_by_lineage = (
        {lineage: index % fold_count for index, lineage in enumerate(all_lineages)}
        if fold_count >= 2
        else {}
    )
    for dimension in rubric.dimensions:
        dimension_data = _dimension_data(data, dimension.dimension_id)
        lineages = tuple(sorted({item.human_score.lineage_id for item in dimension_data}))
        dimension_predictions = []
        if fold_by_lineage:
            for datum in dimension_data:
                fold_index = fold_by_lineage[datum.human_score.lineage_id]
                training = tuple(
                    item
                    for item in dimension_data
                    if fold_by_lineage[item.human_score.lineage_id] != fold_index
                )
                score_map = _fit_score_map(dimension.dimension_id, training)
                calibrated_score = score_map.apply(datum.judge_score.raw_score)
                error = calibrated_score - datum.human_score.score
                dimension_predictions.append(
                    OutOfFoldPrediction(
                        label_id=datum.human_score.label_id,
                        rollout_id=datum.human_score.rollout_id,
                        lineage_id=datum.human_score.lineage_id,
                        dimension_id=dimension.dimension_id,
                        fold_index=fold_index,
                        raw_score=datum.judge_score.raw_score,
                        human_score=datum.human_score.score,
                        calibrated_score=calibrated_score,
                        absolute_error=abs(error),
                        optimistic_error=max(error, 0.0),
                    )
                )
        predictions.extend(dimension_predictions)
        metrics.append(
            _metrics_for_dimension(
                dimension_id=dimension.dimension_id,
                label_count=len(dimension_data),
                lineage_count=len(lineages),
                fold_count=fold_count if fold_by_lineage else 0,
                predictions=dimension_predictions,
            )
        )
    return tuple(predictions), tuple(metrics)


def _metrics_for_dimension(
    *,
    dimension_id: ArtifactId,
    label_count: int,
    lineage_count: int,
    fold_count: int,
    predictions: Sequence[OutOfFoldPrediction],
) -> DimensionCalibrationMetrics:
    """Calculate MAE, rank agreement, and optimistic error from one OOF prediction set."""
    if not predictions:
        return DimensionCalibrationMetrics(
            dimension_id=dimension_id,
            label_count=label_count,
            lineage_count=lineage_count,
            fold_count=fold_count,
        )
    errors = tuple(item.absolute_error for item in predictions)
    optimistic_errors = tuple(item.optimistic_error for item in predictions)
    return DimensionCalibrationMetrics(
        dimension_id=dimension_id,
        label_count=label_count,
        lineage_count=lineage_count,
        fold_count=fold_count,
        mae=sum(errors) / len(errors),
        rank_agreement=_spearman(
            tuple(item.calibrated_score for item in predictions),
            tuple(float(item.human_score) for item in predictions),
        ),
        mean_optimistic_error=sum(optimistic_errors) / len(optimistic_errors),
        maximum_optimistic_error=max(optimistic_errors),
    )


def _report_status(
    rubric: Rubric,
    data: Sequence[CalibrationDatum],
    predictions: Sequence[OutOfFoldPrediction],
) -> Literal["provisional", "insufficient", "ready_for_approval"]:
    """Classify calibration readiness without silently treating scant labels as approved."""
    if not data:
        return "provisional"
    scored_dimensions = {item.human_score.dimension_id for item in data}
    if len({item.human_score.rollout_id for item in data}) < 10:
        return "insufficient"
    if scored_dimensions != {dimension.dimension_id for dimension in rubric.dimensions}:
        return "insufficient"
    if not predictions:
        return "insufficient"
    return "ready_for_approval"


def _worst_disagreements(
    predictions: Sequence[OutOfFoldPrediction],
) -> tuple[WorstDisagreement, ...]:
    """Return the ten largest visible OOF disagreements with stable tie ordering."""
    ordered = sorted(
        predictions,
        key=lambda item: (-item.absolute_error, item.label_id),
    )[:10]
    disagreements = []
    for prediction in ordered:
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


def _spearman(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    """Return tie-aware Spearman rank agreement or None when no variance exists."""
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
    """Assign one-based average ranks to values while preserving stable equal-value ordering."""
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


def _target(
    rubric_id: str, rollout_id: str, lineage_id: str, dimension_id: str
) -> tuple[str, str, str, str]:
    """Return the stable join key for human and raw judge score evidence."""
    return (rubric_id, rollout_id, lineage_id, dimension_id)


def _require_timezone(value: datetime) -> None:
    """Reject naive timestamps before they become immutable calibration provenance."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise CalibrationError("calibration timestamps must include a timezone")


def utc_now() -> datetime:
    """Return a UTC timestamp for callers that do not inject artifact creation time."""
    return datetime.now(UTC)
