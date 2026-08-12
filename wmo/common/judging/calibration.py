"""Manifest-bound, leakage-safe calibration services for common LM judging."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    Sha256,
    stable_id,
)
from wmo.common.judging.calibration_metrics import (
    CalibrationDatum,
    DimensionCalibrationMetrics,
    OutOfFoldPrediction,
    WorstDisagreement,
    fit_score_map,
    grouped_predictions_and_metrics,
    has_valid_grouped_oof,
    worst_disagreements,
)
from wmo.common.judging.judgment import Judgment
from wmo.common.judging.labels import HumanLabelSet, HumanScore
from wmo.common.judging.lineage import RouterLineageSplit
from wmo.common.judging.provenance import (
    JudgingProvenanceError,
    read_artifact_json,
    resolve_artifact,
    sorted_verified_inputs,
)
from wmo.common.judging.rubric import DimensionScoreMap, JudgeCalibration, Rubric
from wmo.common.models import ModelSnapshot
from wmo.common.project import ArtifactAlreadyExistsError, ProjectStore
from wmo.common.rollouts import RolloutArtifact


class CalibrationError(ValueError):
    """Raised when calibration evidence, validation, or approval is invalid."""


class JudgeScoreObservation(ContractModel):
    """Raw dimension evidence cryptographically bound to a stored judgment and rollout."""

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
    """Frozen per-dimension OOF evidence and full eligible-label refit maps."""

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
        input_ids = tuple(item.artifact_id for item in self.inputs)
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
        if input_ids != expected_input_ids:
            raise ValueError("calibration reports must hash exactly their frozen source artifacts")
        return self


@dataclass(frozen=True)
class _VerifiedObservation:
    """One resolved observation with its immutable source records and manifest references."""

    observation: JudgeScoreObservation
    judgment: Judgment
    rollout: RolloutArtifact
    lineage_id: ArtifactId
    judgment_input: ArtifactInput
    rollout_input: ArtifactInput


class JudgeCalibrationService:
    """Build and persist report-bound provisional, insufficient, or approved judge maps."""

    def build_report(
        self,
        store: ProjectStore,
        *,
        rubric_id: ArtifactId,
        label_set_id: ArtifactId,
        router_lineage_split_id: ArtifactId,
        observations: Sequence[JudgeScoreObservation],
        created_at: datetime,
        code_revision: str,
    ) -> CalibrationReport:
        """Build a report only from manifest-verified judgment and rollout evidence.

        Args:
            store: Project store containing all completed immutable calibration inputs.
            rubric_id: Completed rubric artifact to calibrate.
            label_set_id: Completed human-label-set artifact to join to raw scores.
            router_lineage_split_id: Completed rollout-to-lineage split artifact.
            observations: Raw dimension evidence that must match stored judgments exactly.
            created_at: Time the report is materialized.
            code_revision: Exact revision responsible for the report.

        Returns:
            A report whose model, prompt, inputs, maps, and OOF evidence come from verified data.

        Raises:
            CalibrationError: Any input is missing, corrupt, mismatched, or statistically invalid.
        """
        _require_timezone(created_at)
        rubric, rubric_input = _load_rubric(store, rubric_id)
        label_set, label_set_input = _load_label_set(store, label_set_id)
        split, split_input = _load_lineage_split(store, router_lineage_split_id)
        if label_set.rubric_id != rubric.rubric_id:
            raise CalibrationError("human label set belongs to a different frozen rubric")
        if label_set.inputs != (rubric_input,):
            raise CalibrationError(
                "human label set does not hash the exact finalized rubric manifest"
            )
        verified = _resolve_observations(
            store,
            rubric=rubric,
            rubric_input=rubric_input,
            split=split,
            observations=observations,
        )
        judge_model, prompt_id, prompt_sha256 = _single_judge_binding(verified)
        data, excluded = _join_eligible_data(
            rubric=rubric,
            label_set=label_set,
            split=split,
            observations=verified,
        )
        score_maps = tuple(
            fit_score_map(
                dimension.dimension_id,
                tuple(
                    item for item in data if item.human_score.dimension_id == dimension.dimension_id
                ),
            )
            for dimension in rubric.dimensions
        )
        predictions, metrics = grouped_predictions_and_metrics(rubric, data)
        status = _report_status(rubric, data, metrics, predictions)
        inputs = sorted_verified_inputs(
            (
                rubric_input,
                label_set_input,
                split_input,
                *(item.judgment_input for item in verified),
                *(item.rollout_input for item in verified),
            )
        )
        report_id = stable_id(
            "judge-calibration-report",
            {
                "rubric_id": rubric.rubric_id,
                "label_set_id": label_set.label_set_id,
                "router_lineage_split_id": split.split_id,
                "observations": [item.model_dump(mode="json") for item in observations],
                "inputs": [item.model_dump(mode="json") for item in inputs],
                "code_revision": code_revision,
            },
        )
        return CalibrationReport(
            schema_version=1,
            created_at=created_at,
            inputs=inputs,
            code_revision=code_revision,
            report_id=report_id,
            rubric_id=rubric.rubric_id,
            rubric_dimension_ids=tuple(dimension.dimension_id for dimension in rubric.dimensions),
            judge_model=judge_model,
            judge_prompt_id=prompt_id,
            judge_prompt_sha256=prompt_sha256,
            label_set_id=label_set.label_set_id,
            router_lineage_split_id=split.split_id,
            router_lineages=split,
            observations=tuple(observations),
            eligible_label_count=len(data),
            eligible_rollout_count=len({item.human_score.rollout_id for item in data}),
            eligible_lineage_ids=tuple(sorted({item.human_score.lineage_id for item in data})),
            eligible_lineage_count=len({item.human_score.lineage_id for item in data}),
            excluded_held_out_label_count=len(excluded),
            excluded_held_out_rollout_count=len({item.human_score.rollout_id for item in excluded}),
            excluded_held_out_lineage_ids=tuple(
                sorted({item.human_score.lineage_id for item in excluded})
            ),
            excluded_held_out_lineage_count=len({item.human_score.lineage_id for item in excluded}),
            status=status,
            score_maps=score_maps,
            dimension_metrics=metrics,
            out_of_fold_predictions=predictions,
            worst_disagreements=worst_disagreements(predictions),
        )

    def provisional_calibration(
        self, store: ProjectStore, report: CalibrationReport
    ) -> JudgeCalibration:
        """Create a persisted-report-bound identity calibration for a zero-label report."""
        stored_report, report_input = _require_persisted_report(store, report)
        if stored_report.status != "provisional" or stored_report.eligible_label_count != 0:
            raise CalibrationError("provisional calibration requires a zero-label report")
        return _calibration_from_report(
            stored_report,
            report_input=report_input,
            status="provisional",
            approved_at=None,
        )

    def insufficient_calibration(
        self, store: ProjectStore, report: CalibrationReport
    ) -> JudgeCalibration:
        """Create a persisted-report-bound calibration that remains visibly insufficient."""
        stored_report, report_input = _require_persisted_report(store, report)
        if stored_report.status != "insufficient" or stored_report.eligible_label_count == 0:
            raise CalibrationError(
                "insufficient calibration requires a nonzero insufficient report"
            )
        return _calibration_from_report(
            stored_report,
            report_input=report_input,
            status="insufficient",
            approved_at=None,
        )

    def approve(
        self,
        store: ProjectStore,
        report: CalibrationReport,
        *,
        approved_at: datetime,
        accept_insufficient_labels: bool = False,
    ) -> JudgeCalibration:
        """Freeze a human-calibrated map only after complete per-dimension OOF validation.

        Args:
            store: Project store containing the exact completed reviewed report artifact.
            report: Report whose persisted evidence is being approved.
            approved_at: Time the customer accepted the visible OOF evidence.
            accept_insufficient_labels: Explicit risk acceptance below ten rollouts when
                per-dimension OOF evidence is valid.

        Returns:
            A human-calibrated configuration bound to the exact persisted report manifest.

        Raises:
            CalibrationError: The report is provisional, incomplete, or lacks required consent.
        """
        _require_timezone(approved_at)
        stored_report, report_input = _require_persisted_report(store, report)
        if stored_report.status == "provisional":
            raise CalibrationError("zero-label provisional calibration cannot be human-approved")
        if not _report_has_complete_dimension_oof(stored_report):
            raise CalibrationError(
                "cannot approve calibration without complete per-dimension grouped OOF evidence"
            )
        if stored_report.status == "insufficient" and not accept_insufficient_labels:
            raise CalibrationError(
                "insufficient labels require explicit accept_insufficient_labels=True approval"
            )
        return _calibration_from_report(
            stored_report,
            report_input=report_input,
            status="human_calibrated",
            approved_at=approved_at,
        )

    def write_report(self, store: ProjectStore, report: CalibrationReport) -> CalibrationReport:
        """Persist a report only after re-verifying every claimed immutable input.

        Args:
            store: Project store that owns all report sources and output artifact storage.
            report: Report built from verified sources to persist or safely resume.

        Returns:
            The stored report, including a safe idempotent retry result.

        Raises:
            CalibrationError: A source input or existing output cannot be proven equivalent.
        """
        _verify_report_sources(store, report)
        try:
            store.artifacts.write_json(
                artifact_id=report.report_id,
                artifact_type="judge-calibration-report",
                envelope=report,
                files={"report.json": report},
            )
        except ArtifactAlreadyExistsError:
            stored, _input = _load_report(store, report.report_id)
            if not _same_report_identity(stored, report):
                raise CalibrationError(
                    "existing judge-calibration report artifact conflicts with this report"
                ) from None
            return stored
        return report

    def write_calibration(
        self,
        store: ProjectStore,
        *,
        report: CalibrationReport,
        calibration: JudgeCalibration,
    ) -> JudgeCalibration:
        """Persist a calibration only when its report already exists and hash matches exactly.

        Args:
            store: Project-local immutable artifact store.
            report: Exact completed grouped validation artifact reviewed for this calibration.
            calibration: Configuration whose report hash and all source inputs must match.

        Returns:
            The stored calibration, including safe idempotent retry recovery.

        Raises:
            CalibrationError: The report is unpersisted, corrupt, mismatched, or noncanonical.
        """
        stored_report, report_input = _require_persisted_report(store, report)
        _require_calibration_report_binding(stored_report, calibration, report_input)
        try:
            store.artifacts.write_json(
                artifact_id=calibration.calibration_id,
                artifact_type="judge-calibration",
                envelope=calibration,
                files={"calibration.json": calibration},
            )
        except ArtifactAlreadyExistsError:
            try:
                stored, _input = read_artifact_json(
                    store,
                    artifact_id=calibration.calibration_id,
                    expected_artifact_type="judge-calibration",
                    relative_path="calibration.json",
                    model_type=JudgeCalibration,
                )
            except JudgingProvenanceError as exc:
                raise CalibrationError(
                    "existing judge-calibration artifact cannot be resumed safely"
                ) from exc
            if not _same_calibration_identity(stored, calibration):
                raise CalibrationError(
                    "existing judge-calibration artifact conflicts with this calibration"
                ) from None
            return stored
        return calibration


def _load_rubric(store: ProjectStore, rubric_id: ArtifactId) -> tuple[Rubric, ArtifactInput]:
    """Load one completed rubric artifact as a verified report input."""
    try:
        rubric, rubric_input = read_artifact_json(
            store,
            artifact_id=rubric_id,
            expected_artifact_type="rubric",
            relative_path="rubric.json",
            model_type=Rubric,
        )
    except JudgingProvenanceError as exc:
        raise CalibrationError(
            "calibration requires a completed immutable rubric artifact"
        ) from exc
    if rubric.rubric_id != rubric_id:
        raise CalibrationError("stored rubric record does not match its artifact identity")
    return rubric, rubric_input


def _load_label_set(
    store: ProjectStore, label_set_id: ArtifactId
) -> tuple[HumanLabelSet, ArtifactInput]:
    """Load one completed human-label-set artifact as a verified report input."""
    try:
        label_set, label_set_input = read_artifact_json(
            store,
            artifact_id=label_set_id,
            expected_artifact_type="human-label-set",
            relative_path="labels.json",
            model_type=HumanLabelSet,
        )
    except JudgingProvenanceError as exc:
        raise CalibrationError(
            "calibration requires a completed immutable human label set"
        ) from exc
    if label_set.label_set_id != label_set_id:
        raise CalibrationError("stored human label-set record does not match its artifact identity")
    return label_set, label_set_input


def _load_lineage_split(
    store: ProjectStore, split_id: ArtifactId
) -> tuple[RouterLineageSplit, ArtifactInput]:
    """Load one completed router-lineage split artifact as a verified report input."""
    try:
        split, split_input = read_artifact_json(
            store,
            artifact_id=split_id,
            expected_artifact_type="router-lineage-split",
            relative_path="split.json",
            model_type=RouterLineageSplit,
        )
    except JudgingProvenanceError as exc:
        raise CalibrationError(
            "calibration requires a completed immutable router lineage split"
        ) from exc
    if split.split_id != split_id:
        raise CalibrationError("stored router split record does not match its artifact identity")
    try:
        _task_set, task_set_input = resolve_artifact(
            store,
            artifact_id=split.source_task_set_id,
            expected_artifact_type="task-set",
            expected_input=split.inputs[0],
        )
    except JudgingProvenanceError as exc:
        raise CalibrationError(
            "router lineage split source task-set evidence is unavailable or mismatched"
        ) from exc
    if split.inputs != (task_set_input,):
        raise CalibrationError(
            "router lineage split inputs are not the verified source task-set manifest"
        )
    return split, split_input


def _load_report(
    store: ProjectStore, report_id: ArtifactId
) -> tuple[CalibrationReport, ArtifactInput]:
    """Load one completed calibration report artifact and its canonical manifest reference."""
    try:
        report, report_input = read_artifact_json(
            store,
            artifact_id=report_id,
            expected_artifact_type="judge-calibration-report",
            relative_path="report.json",
            model_type=CalibrationReport,
        )
    except JudgingProvenanceError as exc:
        raise CalibrationError("completed calibration report is unavailable") from exc
    if report.report_id != report_id:
        raise CalibrationError(
            "stored calibration report record does not match its artifact identity"
        )
    return report, report_input


def _resolve_observations(
    store: ProjectStore,
    *,
    rubric: Rubric,
    rubric_input: ArtifactInput,
    split: RouterLineageSplit,
    observations: Sequence[JudgeScoreObservation],
) -> tuple[_VerifiedObservation, ...]:
    """Resolve every observation and prove its raw score and citations match source artifacts."""
    if not observations:
        raise CalibrationError(
            "calibration requires uncalibrated judgment evidence to bind judge model and prompt"
        )
    resolved: list[_VerifiedObservation] = []
    seen_targets: set[tuple[str, str, str]] = set()
    rubric_dimension_ids = {item.dimension_id for item in rubric.dimensions}
    for observation in observations:
        try:
            judgment, judgment_input = read_artifact_json(
                store,
                artifact_id=observation.judgment.artifact_id,
                expected_artifact_type="judgment",
                relative_path="judgment.json",
                model_type=Judgment,
                expected_input=observation.judgment,
            )
            rollout, rollout_input = read_artifact_json(
                store,
                artifact_id=observation.source_rollout.artifact_id,
                expected_artifact_type="rollout",
                relative_path="rollout.json",
                model_type=RolloutArtifact,
                expected_input=observation.source_rollout,
            )
        except JudgingProvenanceError as exc:
            raise CalibrationError(
                "calibration observation references missing or corrupt evidence"
            ) from exc
        if judgment.judgment_id != observation.judgment.artifact_id:
            raise CalibrationError("observation judgment ID does not match its immutable artifact")
        if rollout.artifact_id != observation.source_rollout.artifact_id:
            raise CalibrationError("observation rollout ID does not match its immutable artifact")
        if judgment.rollout_id != rollout.rollout_id:
            raise CalibrationError(
                "observation judgment and rollout do not name the same source rollout"
            )
        if judgment.rubric_id != rubric.rubric_id:
            raise CalibrationError("observation judgment belongs to a different frozen rubric")
        if observation.dimension_id not in rubric_dimension_ids:
            raise CalibrationError("observation dimension is absent from the frozen rubric")
        _require_final_judgment_provenance(
            store,
            judgment=judgment,
            rollout_input=rollout_input,
            rubric_input=rubric_input,
        )
        dimension = next(
            (item for item in judgment.dimensions if item.dimension_id == observation.dimension_id),
            None,
        )
        if dimension is None:
            raise CalibrationError("observation dimension is absent from its uncalibrated judgment")
        if dimension.raw_score != observation.raw_score:
            raise CalibrationError("observation raw score does not match its uncalibrated judgment")
        if dimension.evidence_span_ids != observation.evidence_span_ids:
            raise CalibrationError("observation citations do not match its uncalibrated judgment")
        known_spans = {span.span_id for span in rollout.spans}
        if not set(observation.evidence_span_ids).issubset(known_spans):
            raise CalibrationError(
                "calibration observation cites spans absent from its source rollout"
            )
        try:
            lineage_id = split.lineage_for_rollout(rollout.rollout_id)
        except ValueError as exc:
            raise CalibrationError(
                "source rollout is absent from the frozen router lineage split"
            ) from exc
        target = (rollout.rollout_id, lineage_id, observation.dimension_id)
        if target in seen_targets:
            raise CalibrationError(
                "calibration observations must not repeat a rollout, lineage, and dimension"
            )
        seen_targets.add(target)
        resolved.append(
            _VerifiedObservation(
                observation=observation,
                judgment=judgment,
                rollout=rollout,
                lineage_id=lineage_id,
                judgment_input=judgment_input,
                rollout_input=rollout_input,
            )
        )
    return tuple(resolved)


def _require_final_judgment_provenance(
    store: ProjectStore,
    *,
    judgment: Judgment,
    rollout_input: ArtifactInput,
    rubric_input: ArtifactInput,
) -> None:
    """Require a source judgment to already hash its rollout, rubric, and calibration inputs."""
    try:
        calibration, calibration_input = read_artifact_json(
            store,
            artifact_id=judgment.calibration_id,
            expected_artifact_type="judge-calibration",
            relative_path="calibration.json",
            model_type=JudgeCalibration,
        )
    except JudgingProvenanceError as exc:
        raise CalibrationError("uncalibrated judgment has no completed calibration input") from exc
    if calibration.calibration_id != judgment.calibration_id:
        raise CalibrationError("uncalibrated judgment calibration record has the wrong identity")
    if (
        calibration.rubric_id != judgment.rubric_id
        or calibration.judge_model != judgment.judge_model
        or calibration.judge_prompt_id != judgment.judge_prompt_id
        or calibration.judge_prompt_sha256 != judgment.judge_prompt_sha256
    ):
        raise CalibrationError("uncalibrated judgment does not match its frozen calibration")
    required = {rollout_input, rubric_input, calibration_input}
    if not required.issubset(set(judgment.inputs)):
        raise CalibrationError(
            "uncalibrated judgment must hash its source rollout, rubric, and calibration"
        )


def _single_judge_binding(
    observations: Sequence[_VerifiedObservation],
) -> tuple[ModelSnapshot, str, Sha256]:
    """Derive one report model and prompt only from resolved uncalibrated judgments."""
    first = observations[0].judgment
    binding = (first.judge_model, first.judge_prompt_id, first.judge_prompt_sha256)
    for observation in observations[1:]:
        candidate = observation.judgment
        if (
            candidate.judge_model,
            candidate.judge_prompt_id,
            candidate.judge_prompt_sha256,
        ) != binding:
            raise CalibrationError(
                "every calibration observation must come from one exact judge model and prompt"
            )
    return binding


def _join_eligible_data(
    *,
    rubric: Rubric,
    label_set: HumanLabelSet,
    split: RouterLineageSplit,
    observations: Sequence[_VerifiedObservation],
) -> tuple[tuple[CalibrationDatum, ...], tuple[CalibrationDatum, ...]]:
    """Join active human labels to verified judgment evidence and exclude held-out lineages."""
    by_target = {
        (item.rollout.rollout_id, item.lineage_id, item.observation.dimension_id): item
        for item in observations
    }
    rubric_dimensions = {item.dimension_id for item in rubric.dimensions}
    active = label_set.history.active_scores()
    eligible: list[CalibrationDatum] = []
    excluded: list[CalibrationDatum] = []
    matched_targets: set[tuple[str, str, str]] = set()
    fit_lineages = set(split.fit_lineage_ids)
    held_out_lineages = set(split.held_out_lineage_ids)
    for human_score in active:
        _validate_human_score_target(
            human_score,
            rubric=rubric,
            rubric_dimensions=rubric_dimensions,
            split=split,
        )
        target = (human_score.rollout_id, human_score.lineage_id, human_score.dimension_id)
        evidence = by_target.get(target)
        if evidence is None:
            raise CalibrationError(
                "every active human label needs matching manifest-bound raw judge evidence"
            )
        matched_targets.add(target)
        datum = CalibrationDatum(human_score=human_score, raw_score=evidence.observation.raw_score)
        if human_score.lineage_id in held_out_lineages:
            excluded.append(datum)
        elif human_score.lineage_id in fit_lineages:
            eligible.append(datum)
        else:
            raise CalibrationError("human score lineage is absent from the frozen router split")
    if active and set(by_target) != matched_targets:
        raise CalibrationError("raw judge evidence must match active human labels exactly")
    return tuple(eligible), tuple(excluded)


def _validate_human_score_target(
    human_score: HumanScore,
    *,
    rubric: Rubric,
    rubric_dimensions: set[ArtifactId],
    split: RouterLineageSplit,
) -> None:
    """Validate rubric, dimension, rollout, and lineage identities before calibration fitting."""
    if human_score.rubric_id != rubric.rubric_id:
        raise CalibrationError("human score belongs to a different frozen rubric")
    if human_score.dimension_id not in rubric_dimensions:
        raise CalibrationError("human score dimension is absent from the frozen rubric")
    try:
        expected_lineage = split.lineage_for_rollout(human_score.rollout_id)
    except ValueError as exc:
        raise CalibrationError(
            "human score rollout is absent from the frozen router split"
        ) from exc
    if expected_lineage != human_score.lineage_id:
        raise CalibrationError("human score lineage does not match the frozen router split")


def _report_status(
    rubric: Rubric,
    data: Sequence[CalibrationDatum],
    metrics: Sequence[DimensionCalibrationMetrics],
    predictions: Sequence[OutOfFoldPrediction],
) -> Literal["provisional", "insufficient", "ready_for_approval"]:
    """Classify readiness while treating missing per-dimension OOF evidence as insufficient."""
    if not data:
        return "provisional"
    if not _has_complete_dimension_oof(
        tuple(dimension.dimension_id for dimension in rubric.dimensions), metrics, predictions
    ):
        return "insufficient"
    if len({item.human_score.rollout_id for item in data}) < 10:
        return "insufficient"
    return "ready_for_approval"


def _has_complete_dimension_oof(
    rubric_dimension_ids: tuple[ArtifactId, ...],
    metrics: Sequence[DimensionCalibrationMetrics],
    predictions: Sequence[OutOfFoldPrediction],
) -> bool:
    """Return whether every rubric dimension has complete valid grouped OOF evidence."""
    by_dimension = {
        dimension_id: tuple(
            prediction for prediction in predictions if prediction.dimension_id == dimension_id
        )
        for dimension_id in rubric_dimension_ids
    }
    metrics_by_dimension = {item.dimension_id: item for item in metrics}
    if set(metrics_by_dimension) != set(rubric_dimension_ids):
        return False
    if any(item.dimension_id not in set(rubric_dimension_ids) for item in predictions):
        return False
    return all(
        has_valid_grouped_oof(metrics_by_dimension[dimension_id], by_dimension[dimension_id])
        for dimension_id in rubric_dimension_ids
    )


def _report_has_complete_dimension_oof(report: CalibrationReport) -> bool:
    """Check a persisted report's own per-dimension grouped OOF completeness before approval."""
    return _has_complete_dimension_oof(
        report.rubric_dimension_ids,
        report.dimension_metrics,
        report.out_of_fold_predictions,
    )


def _verify_report_sources(store: ProjectStore, report: CalibrationReport) -> None:
    """Re-resolve report sources and reject caller-supplied or stale digests before writing."""
    expected = JudgeCalibrationService().build_report(
        store,
        rubric_id=report.rubric_id,
        label_set_id=report.label_set_id,
        router_lineage_split_id=report.router_lineage_split_id,
        observations=report.observations,
        created_at=report.created_at,
        code_revision=report.code_revision,
    )
    if expected != report:
        raise CalibrationError(
            "calibration report content is not derived from its verified immutable evidence"
        )


def _require_persisted_report(
    store: ProjectStore, report: CalibrationReport
) -> tuple[CalibrationReport, ArtifactInput]:
    """Require a report to already exist as an unchanged completed immutable artifact."""
    stored, report_input = _load_report(store, report.report_id)
    if stored != report:
        raise CalibrationError("calibration report is not the exact persisted reviewed artifact")
    _verify_report_sources(store, stored)
    return stored, report_input


def _calibration_from_report(
    report: CalibrationReport,
    *,
    report_input: ArtifactInput,
    status: Literal["provisional", "insufficient", "human_calibrated"],
    approved_at: datetime | None,
) -> JudgeCalibration:
    """Build a calibration whose inputs include the report and every frozen report identity."""
    inputs = sorted_verified_inputs((report_input, *report.inputs))
    return JudgeCalibration(
        schema_version=1,
        created_at=report.created_at if approved_at is None else approved_at,
        inputs=inputs,
        code_revision=report.code_revision,
        calibration_id=stable_id(
            "judge-calibration",
            {
                "report": report_input.model_dump(mode="json"),
                "inputs": [item.model_dump(mode="json") for item in inputs],
                "status": status,
                "approved_at": approved_at.isoformat() if approved_at is not None else None,
            },
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
        out_of_fold_report_sha256=report_input.sha256,
        score_maps=report.score_maps,
        label_count=report.eligible_label_count,
        status=status,
        approved_at=approved_at,
    )


def _require_calibration_report_binding(
    report: CalibrationReport,
    calibration: JudgeCalibration,
    report_input: ArtifactInput,
) -> None:
    """Require a calibration to pin the exact persisted report hash and all frozen sources."""
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
    if (
        calibration.out_of_fold_report_id != report.report_id
        or calibration.out_of_fold_report_sha256 != report_input.sha256
    ):
        raise CalibrationError("judge calibration must name the exact persisted report manifest")
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
    expected_inputs = sorted_verified_inputs((report_input, *report.inputs))
    if calibration.inputs != expected_inputs:
        raise CalibrationError(
            "judge calibration inputs must be the report and all frozen report inputs"
        )
    if calibration.status == "provisional" and report.status != "provisional":
        raise CalibrationError("only a zero-label report can produce provisional calibration")
    if calibration.status == "insufficient" and report.status != "insufficient":
        raise CalibrationError("only an insufficient report can produce insufficient calibration")
    if calibration.status == "human_calibrated" and not _report_has_complete_dimension_oof(report):
        raise CalibrationError(
            "human calibration requires complete per-dimension grouped OOF evidence"
        )
    expected = _calibration_from_report(
        report,
        report_input=report_input,
        status=calibration.status,
        approved_at=calibration.approved_at,
    )
    if expected != calibration:
        raise CalibrationError(
            "judge calibration content is not derived from its exact persisted report"
        )


def _same_report_identity(left: CalibrationReport, right: CalibrationReport) -> bool:
    """Compare report evidence while permitting a safe retry with a later wall-clock time."""
    return left.model_dump(exclude={"created_at"}) == right.model_dump(exclude={"created_at"})


def _same_calibration_identity(left: JudgeCalibration, right: JudgeCalibration) -> bool:
    """Compare frozen calibration content without retry-time artifact timestamps."""
    return left.model_dump(exclude={"created_at"}) == right.model_dump(exclude={"created_at"})


def _require_timezone(value: datetime) -> None:
    """Reject naive timestamps before they become immutable calibration provenance."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise CalibrationError("calibration timestamps must include a timezone")


def utc_now() -> datetime:
    """Return a UTC timestamp for callers that do not inject artifact creation time."""
    return datetime.now(UTC)
