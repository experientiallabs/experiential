"""Tests for leakage-safe grouped judge calibration and approval states."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from wmo.common.judging import (
    CalibrationError,
    HumanLabelSet,
    HumanScore,
    HumanScoreHistory,
    JudgeCalibrationService,
    JudgeScoreObservation,
    PromptDefinition,
    RouterLineageSplit,
    Rubric,
    RubricDimension,
    ScoreAnchor,
)
from wmo.common.models import ModelSnapshot
from wmo.common.project import ProjectConfig, ProjectStore

_DIGEST = "a" * 64
_TIME = datetime(2026, 8, 11, tzinfo=UTC)

Score = Literal[0, 1, 2, 3, 4, 5]


def _rubric() -> Rubric:
    return Rubric(
        schema_version=1,
        created_at=_TIME,
        code_revision="w6-test",
        rubric_id="rubric-1",
        dimensions=(
            RubricDimension(
                dimension_id="task-success",
                name="Task success",
                description="Whether the task outcome was achieved.",
                anchors=(
                    ScoreAnchor(score=0, description="Anchor 0."),
                    ScoreAnchor(score=1, description="Anchor 1."),
                    ScoreAnchor(score=2, description="Anchor 2."),
                    ScoreAnchor(score=3, description="Anchor 3."),
                    ScoreAnchor(score=4, description="Anchor 4."),
                    ScoreAnchor(score=5, description="Anchor 5."),
                ),
            ),
        ),
        source_task_set_id="task-set-1",
        status="human_approved",
        approved_at=_TIME,
    )


def _two_dimension_rubric() -> Rubric:
    """Return a complete rubric that distinguishes rollout coverage from score-label count."""
    base = _rubric()
    return Rubric(
        schema_version=base.schema_version,
        created_at=base.created_at,
        code_revision=base.code_revision,
        rubric_id=base.rubric_id,
        dimensions=(
            *base.dimensions,
            RubricDimension(
                dimension_id="policy-compliance",
                name="Policy compliance",
                description="Whether the response follows the documented policy.",
                anchors=base.dimensions[0].anchors,
            ),
        ),
        source_task_set_id=base.source_task_set_id,
        status=base.status,
        approved_at=base.approved_at,
    )


def _model() -> ModelSnapshot:
    return ModelSnapshot(
        provider="fake",
        model_id="judge-model",
        capabilities_sha256=_DIGEST,
    )


def _score(label_id: str, rollout_id: str, lineage_id: str, score: Score) -> HumanScore:
    return HumanScore(
        label_id=label_id,
        rubric_id="rubric-1",
        rollout_id=rollout_id,
        lineage_id=lineage_id,
        dimension_id="task-success",
        score=score,
        created_at=_TIME,
    )


def _judge_score(rollout_id: str, lineage_id: str, raw_score: Score) -> JudgeScoreObservation:
    return JudgeScoreObservation(
        rubric_id="rubric-1",
        rollout_id=rollout_id,
        lineage_id=lineage_id,
        dimension_id="task-success",
        raw_score=raw_score,
    )


def _fit_history_and_scores(
    count: int = 10,
) -> tuple[HumanScoreHistory, tuple[JudgeScoreObservation, ...], RouterLineageSplit]:
    pairs: tuple[tuple[Score, Score], ...] = (
        (5, 0),
        (5, 0),
        (4, 0),
        (4, 1),
        (3, 1),
        (3, 2),
        (2, 2),
        (2, 3),
        (1, 3),
        (1, 4),
    )
    history = HumanScoreHistory()
    judge_scores: list[JudgeScoreObservation] = []
    fit_lineages: list[str] = []
    for index, (raw_score, human_score) in enumerate(pairs[:count], start=1):
        rollout_id = f"rollout-fit-{index}"
        lineage_id = f"lineage-fit-{index}"
        history = history.append(_score(f"label-fit-{index}", rollout_id, lineage_id, human_score))
        judge_scores.append(_judge_score(rollout_id, lineage_id, raw_score))
        fit_lineages.append(lineage_id)
    return (
        history,
        tuple(judge_scores),
        RouterLineageSplit(
            fit_lineage_ids=tuple(fit_lineages),
            held_out_lineage_ids=("lineage-held-out",),
        ),
    )


def _label_set(history: HumanScoreHistory) -> HumanLabelSet:
    return HumanLabelSet(
        schema_version=1,
        created_at=_TIME,
        code_revision="w6-test",
        label_set_id="label-set-1",
        rubric_id="rubric-1",
        history=history,
        active_label_ids=tuple(score.label_id for score in history.active_scores()),
    )


def _store(tmp_path: Path) -> ProjectStore:
    store = ProjectStore(tmp_path / ".wmo", "support-project")
    store.initialize(ProjectConfig(project_id="support-project"))
    return store


def test_grouped_calibration_excludes_router_held_out_labels_and_reports_oof_errors() -> None:
    """Held-out labels cannot affect score maps while biased optimism stays visible OOF."""
    history, judge_scores, lineages = _fit_history_and_scores()
    held_out_score = _score("label-held-out", "rollout-held-out", "lineage-held-out", 5)
    full_history = history.append(held_out_score)
    full_scores = (*judge_scores, _judge_score("rollout-held-out", "lineage-held-out", 5))
    prompt = PromptDefinition.from_text("judge-prompt-v1", "Return structured scores.")
    service = JudgeCalibrationService()

    report = service.build_report(
        rubric=_rubric(),
        judge_model=_model(),
        prompt=prompt,
        label_set=_label_set(full_history),
        judge_scores=full_scores,
        router_lineages=lineages,
        created_at=_TIME,
        code_revision="w6-test",
    )
    fit_only_report = service.build_report(
        rubric=_rubric(),
        judge_model=_model(),
        prompt=prompt,
        label_set=_label_set(history),
        judge_scores=judge_scores,
        router_lineages=lineages,
        created_at=_TIME,
        code_revision="w6-test",
    )

    metrics = report.dimension_metrics[0]
    assert report.status == "ready_for_approval"
    assert report.eligible_rollout_count == 10
    assert report.excluded_held_out_label_count == 1
    assert all(item.lineage_id != "lineage-held-out" for item in report.out_of_fold_predictions)
    assert report.score_maps == fit_only_report.score_maps
    assert metrics.mae is not None
    assert metrics.mean_optimistic_error is not None
    assert metrics.mean_optimistic_error > 0
    assert any(item.direction == "optimistic" for item in report.worst_disagreements)

    calibration = service.approve(report, approved_at=_TIME)
    assert calibration.status == "human_calibrated"
    assert calibration.label_count == 10
    assert calibration.excluded_router_held_out_lineage_ids == ("lineage-held-out",)


def test_insufficient_labels_require_explicit_risk_approval() -> None:
    """Nine rollout labels remain insufficient until a human explicitly accepts that risk."""
    history, judge_scores, lineages = _fit_history_and_scores(count=9)
    prompt = PromptDefinition.from_text("judge-prompt-v1", "Return structured scores.")
    service = JudgeCalibrationService()
    report = service.build_report(
        rubric=_rubric(),
        judge_model=_model(),
        prompt=prompt,
        label_set=_label_set(history),
        judge_scores=judge_scores,
        router_lineages=lineages,
        created_at=_TIME,
        code_revision="w6-test",
    )

    assert report.status == "insufficient"
    assert report.eligible_rollout_count == 9
    insufficient = service.insufficient_calibration(report)
    assert insufficient.status == "insufficient"
    assert insufficient.label_count == 9
    with pytest.raises(CalibrationError, match="explicit"):
        service.approve(report, approved_at=_TIME)
    calibration = service.approve(
        report,
        approved_at=_TIME,
        accept_insufficient_labels=True,
    )
    assert calibration.status == "human_calibrated"
    assert calibration.label_count == 9


def test_calibration_preserves_label_and_rollout_denominators_separately() -> None:
    """Ten reviewed rollouts with two dimensions preserve all twenty active score labels."""
    pairs: tuple[tuple[Score, Score], ...] = (
        (5, 0),
        (5, 0),
        (4, 1),
        (4, 1),
        (3, 2),
        (3, 2),
        (2, 3),
        (2, 3),
        (1, 4),
        (0, 5),
    )
    history = HumanScoreHistory()
    judge_scores: list[JudgeScoreObservation] = []
    for index, (raw_score, human_score) in enumerate(pairs, start=1):
        rollout_id = f"rollout-{index}"
        lineage_id = f"lineage-{index}"
        task_score = _score(f"task-label-{index}", rollout_id, lineage_id, human_score)
        policy_score = task_score.model_copy(
            update={
                "label_id": f"policy-label-{index}",
                "dimension_id": "policy-compliance",
            }
        )
        history = history.append(task_score).append(policy_score)
        judge_scores.extend(
            (
                _judge_score(rollout_id, lineage_id, raw_score),
                _judge_score(rollout_id, lineage_id, raw_score).model_copy(
                    update={"dimension_id": "policy-compliance"}
                ),
            )
        )
    service = JudgeCalibrationService()
    report = service.build_report(
        rubric=_two_dimension_rubric(),
        judge_model=_model(),
        prompt=PromptDefinition.from_text("judge-prompt-v1", "Return structured scores."),
        label_set=_label_set(history),
        judge_scores=tuple(judge_scores),
        router_lineages=RouterLineageSplit(
            fit_lineage_ids=tuple(f"lineage-{index}" for index in range(1, 11)),
            held_out_lineage_ids=("lineage-held-out",),
        ),
        created_at=_TIME,
        code_revision="w6-test",
    )

    calibration = service.approve(report, approved_at=_TIME)

    assert report.eligible_rollout_count == 10
    assert report.eligible_label_count == 20
    assert calibration.label_count == 20


def test_grouped_out_of_fold_predictions_keep_all_rollouts_in_one_lineage_together() -> None:
    """Conversation siblings receive one shared held-out fold rather than leaking into training."""
    history = HumanScoreHistory()
    judge_scores: list[JudgeScoreObservation] = []
    pairs: tuple[tuple[str, Score, Score], ...] = (
        ("lineage-a", 5, 1),
        ("lineage-a", 4, 1),
        ("lineage-b", 1, 4),
        ("lineage-b", 0, 5),
    )
    for index, (lineage_id, raw_score, human_score) in enumerate(pairs, start=1):
        rollout_id = f"rollout-{index}"
        history = history.append(
            _score(
                f"label-{index}",
                rollout_id,
                lineage_id,
                human_score,
            )
        )
        judge_scores.append(_judge_score(rollout_id, lineage_id, raw_score))
    report = JudgeCalibrationService().build_report(
        rubric=_rubric(),
        judge_model=_model(),
        prompt=PromptDefinition.from_text("judge-prompt-v1", "Return structured scores."),
        label_set=_label_set(history),
        judge_scores=tuple(judge_scores),
        router_lineages=RouterLineageSplit(
            fit_lineage_ids=("lineage-a", "lineage-b"),
            held_out_lineage_ids=("lineage-held-out",),
        ),
        created_at=_TIME,
        code_revision="w6-test",
    )

    fold_by_lineage = {
        lineage_id: {
            item.fold_index
            for item in report.out_of_fold_predictions
            if item.lineage_id == lineage_id
        }
        for lineage_id in ("lineage-a", "lineage-b")
    }

    assert report.status == "insufficient"
    assert fold_by_lineage == {"lineage-a": {0}, "lineage-b": {1}}


def test_no_human_labels_produces_a_visible_identity_map_provisional_judge() -> None:
    """An empty label set stays provisional rather than claiming human calibration."""
    prompt = PromptDefinition.from_text("judge-prompt-v1", "Return structured scores.")
    report = JudgeCalibrationService().build_report(
        rubric=_rubric(),
        judge_model=_model(),
        prompt=prompt,
        label_set=_label_set(HumanScoreHistory()),
        judge_scores=(),
        router_lineages=RouterLineageSplit(
            fit_lineage_ids=("lineage-fit-1",),
            held_out_lineage_ids=("lineage-held-out",),
        ),
        created_at=_TIME,
        code_revision="w6-test",
    )

    calibration = JudgeCalibrationService().provisional_calibration(report)

    assert report.status == "provisional"
    assert calibration.status == "provisional"
    assert calibration.score_maps[0].calibrated_scores == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)


def test_reviewed_report_and_approved_calibration_freeze_as_immutable_artifacts(
    tmp_path: Path,
) -> None:
    """A calibration cannot become a mutable side file after the human approval gate."""
    history, judge_scores, lineages = _fit_history_and_scores()
    service = JudgeCalibrationService()
    report = service.build_report(
        rubric=_rubric(),
        judge_model=_model(),
        prompt=PromptDefinition.from_text("judge-prompt-v1", "Return structured scores."),
        label_set=_label_set(history),
        judge_scores=judge_scores,
        router_lineages=lineages,
        created_at=_TIME,
        code_revision="w6-test",
    )
    calibration = service.approve(report, approved_at=_TIME)
    store = _store(tmp_path)

    assert service.write_report(store, report) == report
    assert service.write_report(store, report) == report
    assert service.write_calibration(store, report=report, calibration=calibration) == calibration
    assert service.write_calibration(store, report=report, calibration=calibration) == calibration
    assert (
        store.artifacts.read(report.report_id).manifest.artifact_type == "judge-calibration-report"
    )
    assert (
        store.artifacts.read(calibration.calibration_id).manifest.artifact_type
        == "judge-calibration"
    )
