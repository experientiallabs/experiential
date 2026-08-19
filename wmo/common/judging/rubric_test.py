"""Tests for the ordered-axis rubric contract and identity score maps."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from wmo.common.core.artifacts import ArtifactInput
from wmo.common.judging import (
    DimensionScoreMap,
    JudgeCalibration,
    Rubric,
    RubricDimension,
    ScoreAnchor,
    default_task_success_axis,
    identity_score_map,
    score_bounds,
    scored_axis,
)
from wmo.common.models import BillingSource, ModelSnapshot

_DIGEST = "a" * 64


def _axis(
    dimension_id: str = "task-success",
    *,
    min_score: int = 0,
    max_score: int = 1,
) -> RubricDimension:
    """Return one complete axis covering every integer in the requested range."""
    return scored_axis(
        dimension_id,
        "Task success",
        "Whether the customer received a correct outcome.",
        min_score=min_score,
        max_score=max_score,
    )


def test_default_task_success_axis_uses_zero_one_and_required_description() -> None:
    """The built-in rubric is one 0-1 task-success axis with the product meaning."""
    axis = default_task_success_axis()

    assert axis.dimension_id == "task-success"
    assert axis.name == "Task success"
    assert axis.description == (
        "The agent successfully completed the task requested in the original user prompt"
    )
    assert axis.min_score == 0
    assert axis.max_score == 1
    assert axis.permitted_scores() == (0, 1)
    assert axis.normalize_score(0) == 0.0
    assert axis.normalize_score(1) == 1.0
    payload = axis.prompt_payload()
    assert payload["min_score"] == 0
    assert payload["max_score"] == 1
    assert payload["description"] == axis.description


def test_rubric_rejects_empty_axes_duplicate_ids_and_invalid_ranges() -> None:
    """A rubric needs unique IDs, a real inclusive range, and endpoint anchors."""
    with pytest.raises(ValidationError, match="at least one axis"):
        Rubric(
            schema_version=1,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            inputs=(ArtifactInput(artifact_id="task-set-v1", sha256=_DIGEST),),
            code_revision="e7aad17",
            rubric_id="support-rubric-v1",
            dimensions=(),
            source_task_set_id="task-set-v1",
            status="provisional",
        )
    with pytest.raises(ValidationError, match="unique IDs"):
        Rubric(
            schema_version=1,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            inputs=(ArtifactInput(artifact_id="task-set-v1", sha256=_DIGEST),),
            code_revision="e7aad17",
            rubric_id="support-rubric-v1",
            dimensions=(_axis("task-success"), _axis("task-success")),
            source_task_set_id="task-set-v1",
            status="provisional",
        )
    with pytest.raises(ValidationError, match="min_score below max_score"):
        RubricDimension(
            dimension_id="task-success",
            name="Task success",
            description="Whether the customer received a correct outcome.",
            min_score=1,
            max_score=1,
            anchors=(ScoreAnchor(score=1, description="Only score."),),
        )
    with pytest.raises(ValidationError, match="inclusive range endpoints"):
        RubricDimension(
            dimension_id="task-success",
            name="Task success",
            description="Whether the customer received a correct outcome.",
            min_score=0,
            max_score=4,
            anchors=(ScoreAnchor(score=2, description="Middle only."),),
        )


def test_rubric_rejects_axes_that_do_not_share_one_range() -> None:
    """A shared schema cannot describe mixed inclusive ranges in one rubric."""
    with pytest.raises(ValidationError, match="same inclusive score range"):
        Rubric(
            schema_version=1,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            inputs=(ArtifactInput(artifact_id="task-set-v1", sha256=_DIGEST),),
            code_revision="e7aad17",
            rubric_id="support-rubric-v1",
            dimensions=(
                _axis(min_score=0, max_score=1),
                _axis("quality", min_score=0, max_score=4),
            ),
            source_task_set_id="task-set-v1",
            status="provisional",
        )


def test_zero_to_four_axis_allows_meaningful_interior_anchors() -> None:
    """A 0-4 axis may omit interior scores when endpoints and anchors stay valid."""
    axis = RubricDimension(
        dimension_id="quality",
        name="Quality",
        description="How completely the agent solved the requested work.",
        min_score=0,
        max_score=4,
        anchors=(
            ScoreAnchor(score=0, description="No useful progress."),
            ScoreAnchor(score=2, description="Partial completion with remaining gaps."),
            ScoreAnchor(score=4, description="Complete and correct."),
        ),
    )

    assert axis.permitted_scores() == (0, 1, 2, 3, 4)
    assert axis.contains_score(3)
    assert not axis.contains_score(5)
    assert score_bounds((axis,)) == (0, 4)
    assert identity_score_map(axis.dimension_id, 0, 4).is_identity()


def test_rubric_requires_approval_time_and_calibration_excludes_held_out() -> None:
    """Incomplete approval metadata and overlapping lineages fail before persistence."""
    with pytest.raises(ValidationError, match="require approved_at"):
        Rubric(
            schema_version=1,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            inputs=(ArtifactInput(artifact_id="task-set-v1", sha256=_DIGEST),),
            code_revision="e7aad17",
            rubric_id="support-rubric-v1",
            dimensions=(_axis(),),
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
                billing_source=BillingSource.CUSTOMER_MANAGED,
                provider="openai",
                model_id="gpt-5.4-mini",
                capabilities_sha256=_DIGEST,
                connection_sha256=_DIGEST,
            ),
            judge_prompt_id="judge-prompt-v1",
            judge_prompt_sha256=_DIGEST,
            label_set_id="label-set-v1",
            calibration_lineage_ids=("lineage-1",),
            excluded_router_held_out_lineage_ids=("lineage-1",),
            validation_method="grouped_k_fold",
            out_of_fold_report_id="judge-report-v1",
            out_of_fold_report_sha256=_DIGEST,
            score_maps=(
                DimensionScoreMap(
                    dimension_id="task-success",
                    min_score=0,
                    max_score=1,
                    calibrated_scores=(0.0, 1.0),
                ),
            ),
        )
