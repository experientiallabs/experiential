"""Manifest-bound, leakage-safe calibration services for common LM judging."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    Sha256,
    canonical_json_bytes,
    stable_id,
)
from wmo.common.judging.calibration_assembly import calibration_from_report
from wmo.common.judging.calibration_contracts import CalibrationReport, JudgeScoreObservation
from wmo.common.judging.calibration_metrics import (
    CalibrationDatum,
    DimensionCalibrationMetrics,
    OutOfFoldPrediction,
    fit_score_map,
    grouped_predictions_and_metrics,
    has_valid_grouped_oof,
    worst_disagreements,
)
from wmo.common.judging.judgment import Judgment
from wmo.common.judging.labels import HumanLabelSet, HumanScore
from wmo.common.judging.lineage import RouterLineageSplit
from wmo.common.judging.prompts import PromptDefinition
from wmo.common.judging.provenance import (
    JudgingProvenanceError,
    read_artifact_json,
    resolve_artifact,
    sorted_verified_inputs,
)
from wmo.common.judging.risk_acceptance import (
    RiskAcceptanceError,
    calibration_inputs,
    require_calibration_risk_acceptance,
    write_insufficient_calibration_risk_acceptance,
)
from wmo.common.judging.rubric import JudgeCalibration, Rubric
from wmo.common.models import ModelSnapshot
from wmo.common.project import ArtifactCorruptionError, ProjectStore
from wmo.common.rollouts import RolloutArtifact


class CalibrationError(ValueError):
    """Raised when calibration evidence, validation, or approval is invalid."""


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
        _require_label_set_rubric_binding(label_set, rubric, rubric_input)
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
                min_score=dimension.min_score,
                max_score=dimension.max_score,
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

    def bootstrap_provisional(
        self,
        store: ProjectStore,
        *,
        rubric_id: ArtifactId,
        label_set_id: ArtifactId,
        router_lineage_split_id: ArtifactId,
        judge_model: ModelSnapshot,
        judge_prompt: PromptDefinition,
        created_at: datetime,
        code_revision: str,
    ) -> JudgeCalibration:
        """Create and persist the canonical zero-label identity-map calibration.

        Args:
            store: Project store containing finalized immutable bootstrap artifacts.
            rubric_id: Finalized rubric artifact used for every judged dimension.
            label_set_id: Finalized empty human-label-set artifact for the rubric.
            router_lineage_split_id: Frozen router lineage split for later human calibration.
            judge_model: Exact snapshot of the configured LM judge.
            judge_prompt: Immutable prompt definition supplied to the configured LM judge.
            created_at: Time the provisional report and calibration are materialized.
            code_revision: Exact revision responsible for the bootstrap artifacts.

        Returns:
            A persisted provisional calibration with one identity map per rubric dimension.

        Raises:
            CalibrationError: A source is unavailable, has labels, or cannot form the bootstrap.
        """
        _require_timezone(created_at)
        rubric, rubric_input = _load_rubric(store, rubric_id)
        label_set, label_set_input = _load_label_set(store, label_set_id)
        split, split_input = _load_lineage_split(store, router_lineage_split_id)
        _require_label_set_rubric_binding(label_set, rubric, rubric_input)
        report = _build_provisional_report(
            rubric=rubric,
            rubric_input=rubric_input,
            label_set=label_set,
            label_set_input=label_set_input,
            split=split,
            split_input=split_input,
            judge_model=judge_model,
            judge_prompt_id=judge_prompt.prompt_id,
            judge_prompt_sha256=judge_prompt.sha256,
            created_at=created_at,
            code_revision=code_revision,
        )
        stored_report = self.write_report(store, report)
        stored_report, report_input = _require_persisted_report(store, stored_report)
        calibration = calibration_from_report(
            stored_report,
            report_input=report_input,
            status="provisional",
            approved_at=None,
        )
        return self.write_calibration(store, report=stored_report, calibration=calibration)

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
            accept_insufficient_labels: Explicit risk acceptance below five rollouts when
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
        risk_acceptance: ArtifactInput | None = None
        if stored_report.status == "insufficient":
            try:
                risk_acceptance = write_insufficient_calibration_risk_acceptance(
                    store,
                    report=stored_report,
                    report_input=report_input,
                    accepted_at=approved_at,
                )
            except RiskAcceptanceError as exc:
                raise CalibrationError(str(exc)) from exc
        return calibration_from_report(
            stored_report,
            report_input=report_input,
            status="human_calibrated",
            approved_at=approved_at,
            risk_acceptance=risk_acceptance,
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
            stored, _ = store.artifacts.write_or_replay(
                artifact_id=report.report_id,
                artifact_type="judge-calibration-report",
                envelope=report,
                envelope_path="report.json",
                envelope_type=CalibrationReport,
                files={"report.json": canonical_json_bytes(report)},
            )
        except ArtifactCorruptionError as exc:
            raise CalibrationError(
                "existing judge-calibration report artifact cannot be resumed safely"
            ) from exc
        except ValueError as exc:
            raise CalibrationError(
                "existing judge-calibration report artifact conflicts with this report"
            ) from exc
        return stored

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
        _require_calibration_report_binding(store, stored_report, calibration, report_input)
        try:
            stored, _ = store.artifacts.write_or_replay(
                artifact_id=calibration.calibration_id,
                artifact_type="judge-calibration",
                envelope=calibration,
                envelope_path="calibration.json",
                envelope_type=JudgeCalibration,
                files={"calibration.json": canonical_json_bytes(calibration)},
            )
        except ArtifactCorruptionError as exc:
            raise CalibrationError(
                "existing judge-calibration artifact cannot be resumed safely"
            ) from exc
        except ValueError as exc:
            raise CalibrationError(
                "existing judge-calibration artifact conflicts with this calibration"
            ) from exc
        return stored


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


def _require_label_set_rubric_binding(
    label_set: HumanLabelSet, rubric: Rubric, rubric_input: ArtifactInput
) -> None:
    """Require a frozen label set to name and hash the exact finalized rubric."""
    if label_set.rubric_id != rubric.rubric_id:
        raise CalibrationError("human label set belongs to a different frozen rubric")
    if label_set.inputs != (rubric_input,):
        raise CalibrationError("human label set does not hash the exact finalized rubric manifest")


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
    """Resolve every observation and prove its raw score matches source artifacts."""
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
    """Require a source judgment to use an eligible persisted calibration and exact inputs."""
    # The verifier rebuilds reports through this module, so importing it above would cycle.
    from wmo.common.judging.calibration_provenance import (
        _load_authoritative_persisted_calibration,
    )

    try:
        calibration, calibration_input = _load_authoritative_persisted_calibration(
            store, judgment.calibration_id
        )
    except CalibrationError as exc:
        raise CalibrationError(
            "source judgment has no eligible persisted calibration input"
        ) from exc
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
        return "insufficient"
    if not _has_complete_dimension_oof(
        tuple(dimension.dimension_id for dimension in rubric.dimensions), metrics, predictions
    ):
        return "insufficient"
    if len({item.human_score.rollout_id for item in data}) < 5:
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


def _build_provisional_report(
    *,
    rubric: Rubric,
    rubric_input: ArtifactInput,
    label_set: HumanLabelSet,
    label_set_input: ArtifactInput,
    split: RouterLineageSplit,
    split_input: ArtifactInput,
    judge_model: ModelSnapshot,
    judge_prompt_id: str,
    judge_prompt_sha256: Sha256,
    created_at: datetime,
    code_revision: str,
) -> CalibrationReport:
    """Derive the narrow zero-label report exception from frozen bootstrap evidence."""
    if label_set.history.scores:
        raise CalibrationError("provisional bootstrap requires a finalized zero-label set")
    inputs = sorted_verified_inputs((rubric_input, label_set_input, split_input))
    score_maps = tuple(
        fit_score_map(
            dimension.dimension_id,
            (),
            min_score=dimension.min_score,
            max_score=dimension.max_score,
        )
        for dimension in rubric.dimensions
    )
    predictions, metrics = grouped_predictions_and_metrics(rubric, ())
    report_id = stable_id(
        "judge-calibration-report",
        {
            "rubric_id": rubric.rubric_id,
            "label_set_id": label_set.label_set_id,
            "router_lineage_split_id": split.split_id,
            "judge_model": judge_model.model_dump(mode="json"),
            "judge_prompt_id": judge_prompt_id,
            "judge_prompt_sha256": judge_prompt_sha256,
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
        judge_prompt_id=judge_prompt_id,
        judge_prompt_sha256=judge_prompt_sha256,
        label_set_id=label_set.label_set_id,
        router_lineage_split_id=split.split_id,
        router_lineages=split,
        observations=(),
        eligible_label_count=0,
        eligible_rollout_count=0,
        eligible_lineage_ids=(),
        eligible_lineage_count=0,
        excluded_held_out_label_count=0,
        excluded_held_out_rollout_count=0,
        excluded_held_out_lineage_ids=(),
        excluded_held_out_lineage_count=0,
        status="provisional",
        score_maps=score_maps,
        dimension_metrics=metrics,
        out_of_fold_predictions=predictions,
        worst_disagreements=(),
    )


def _verify_report_sources(store: ProjectStore, report: CalibrationReport) -> None:
    """Re-resolve report sources and reject caller-supplied or stale digests before writing."""
    if report.status == "provisional":
        _verify_provisional_report_sources(store, report)
        return
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


def _verify_provisional_report_sources(store: ProjectStore, report: CalibrationReport) -> None:
    """Rebuild a zero-label bootstrap without requiring a nonexistent source judgment."""
    rubric, rubric_input = _load_rubric(store, report.rubric_id)
    label_set, label_set_input = _load_label_set(store, report.label_set_id)
    split, split_input = _load_lineage_split(store, report.router_lineage_split_id)
    _require_label_set_rubric_binding(label_set, rubric, rubric_input)
    expected = _build_provisional_report(
        rubric=rubric,
        rubric_input=rubric_input,
        label_set=label_set,
        label_set_input=label_set_input,
        split=split,
        split_input=split_input,
        judge_model=report.judge_model,
        judge_prompt_id=report.judge_prompt_id,
        judge_prompt_sha256=report.judge_prompt_sha256,
        created_at=report.created_at,
        code_revision=report.code_revision,
    )
    if expected != report:
        raise CalibrationError(
            "provisional calibration report is not derived from its verified bootstrap evidence"
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


def _require_calibration_report_binding(
    store: ProjectStore,
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
    try:
        require_calibration_risk_acceptance(
            store, report=report, report_input=report_input, calibration=calibration
        )
    except RiskAcceptanceError as exc:
        raise CalibrationError(str(exc)) from exc
    expected_inputs = calibration_inputs(report_input, report.inputs, calibration.risk_acceptance)
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
    expected = calibration_from_report(
        report,
        report_input=report_input,
        status=calibration.status,
        approved_at=calibration.approved_at,
        risk_acceptance=calibration.risk_acceptance,
    )
    if expected != calibration:
        raise CalibrationError(
            "judge calibration content is not derived from its exact persisted report"
        )


def _require_timezone(value: datetime) -> None:
    """Reject naive timestamps before they become immutable calibration provenance."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise CalibrationError("calibration timestamps must include a timezone")
