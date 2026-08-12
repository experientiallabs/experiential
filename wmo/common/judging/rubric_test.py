"""Tests for immutable zero-to-five rubric and calibration contracts."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from wmo.common.judging import (
    DimensionScoreMap,
    JudgeCalibration,
    Rubric,
    RubricDimension,
    ScoreAnchor,
)
from wmo.common.models import ModelSnapshot

_DIGEST = "a" * 64


def _dimension() -> RubricDimension:
    return RubricDimension(
        dimension_id="task-success",
        name="Task success",
        description="Whether the customer received a correct outcome.",
        anchors=(
            ScoreAnchor(score=0, description="Score 0 outcome."),
            ScoreAnchor(score=1, description="Score 1 outcome."),
            ScoreAnchor(score=2, description="Score 2 outcome."),
            ScoreAnchor(score=3, description="Score 3 outcome."),
            ScoreAnchor(score=4, description="Score 4 outcome."),
            ScoreAnchor(score=5, description="Score 5 outcome."),
        ),
    )


def test_rubric_and_calibration_round_trip() -> None:
    """Approved rubric and calibration records retain their model and score-map provenance."""
    approved_at = datetime(2026, 8, 11, tzinfo=UTC)
    rubric = Rubric(
        schema_version=1,
        created_at=approved_at,
        code_revision="e7aad17",
        rubric_id="support-rubric-v1",
        dimensions=(_dimension(),),
        source_task_set_id="task-set-v1",
        status="human_approved",
        approved_at=approved_at,
    )
    calibration = JudgeCalibration(
        schema_version=1,
        created_at=approved_at,
        code_revision="e7aad17",
        calibration_id="judge-calibration-v1",
        rubric_id=rubric.rubric_id,
        judge_model=ModelSnapshot(
            provider="openai",
            model_id="gpt-5.4-mini",
            capabilities_sha256=_DIGEST,
            connection_sha256=_DIGEST,
        ),
        judge_prompt_id="judge-prompt-v1",
        label_set_id="label-set-v1",
        calibration_lineage_ids=("lineage-fit-1",),
        excluded_router_held_out_lineage_ids=("lineage-held-out-1",),
        validation_method="grouped_k_fold",
        out_of_fold_report_id="judge-report-v1",
        score_maps=(
            DimensionScoreMap(
                dimension_id="task-success",
                calibrated_scores=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
            ),
        ),
        approved_at=approved_at,
    )

    assert Rubric.model_validate_json(rubric.model_dump_json()) == rubric
    assert JudgeCalibration.model_validate_json(calibration.model_dump_json()) == calibration


def test_rubric_requires_ordered_complete_anchors_and_approval_time() -> None:
    """Incomplete anchors and unrecorded human approval fail before persistence."""
    with pytest.raises(ValidationError, match="zero through five"):
        RubricDimension(
            dimension_id="task-success",
            name="Task success",
            description="Whether the customer received a correct outcome.",
            anchors=(ScoreAnchor(score=0, description="Bad."),),
        )
    with pytest.raises(ValidationError, match="require approved_at"):
        Rubric(
            schema_version=1,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="e7aad17",
            rubric_id="support-rubric-v1",
            dimensions=(_dimension(),),
            source_task_set_id="task-set-v1",
            status="human_approved",
        )
    with pytest.raises(ValidationError, match="exclude router-held-out"):
        JudgeCalibration(
            schema_version=1,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="e7aad17",
            calibration_id="judge-calibration-v1",
            rubric_id="support-rubric-v1",
            judge_model=ModelSnapshot(
                provider="openai",
                model_id="gpt-5.4-mini",
                capabilities_sha256=_DIGEST,
                connection_sha256=_DIGEST,
            ),
            judge_prompt_id="judge-prompt-v1",
            label_set_id="label-set-v1",
            calibration_lineage_ids=("lineage-1",),
            excluded_router_held_out_lineage_ids=("lineage-1",),
            validation_method="grouped_k_fold",
            out_of_fold_report_id="judge-report-v1",
            score_maps=(
                DimensionScoreMap(
                    dimension_id="task-success",
                    calibrated_scores=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
                ),
            ),
        )
