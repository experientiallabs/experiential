"""Tests for reusable and immutable user-authored judge criteria."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import exp
from exp.common.judging.definition import JudgeDefinition
from exp.common.judging.rubric import RubricDimension, ScoreAnchor, scored_axis


def test_definition_round_trip_and_revision_pin_every_edit() -> None:
    """Persistence preserves the version while changing the syllabus produces a new one."""
    judge = JudgeDefinition.task_success()
    assert exp.JudgeDefinition is JudgeDefinition
    assert callable(exp.prepare_manual_judge_setup)
    assert callable(exp.calibrate_manual_judge)
    assert JudgeDefinition.model_validate_json(judge.model_dump_json()).revision == judge.revision
    edited = JudgeDefinition(
        name=judge.name, syllabus="Also require cited evidence.", dimensions=judge.dimensions
    )
    assert edited.revision != judge.revision
    with pytest.raises(ValidationError, match="frozen"):
        judge.name = "Changed"


def test_definition_accepts_independent_signed_axis_ranges() -> None:
    """Each axis owns its units and has complete authored score meanings."""
    judge = JudgeDefinition(
        name="Support",
        syllabus="Assess resolution and whether the interaction helped the user.",
        dimensions=(
            *JudgeDefinition.task_success().dimensions,
            scored_axis(
                "helpfulness", "Helpfulness", "Effect on the user.", min_score=-2, max_score=2
            ),
        ),
    )
    assert judge.dimensions[1].normalize_score(-1) == 0.25


def test_definition_rejects_duplicate_blank_or_incomplete_criteria() -> None:
    """Authoring fails closed before an evaluation or inference client can be created."""
    default = JudgeDefinition.task_success()
    with pytest.raises(ValidationError, match="unique IDs"):
        JudgeDefinition(name="Judge", syllabus="Assess success.", dimensions=default.dimensions * 2)
    with pytest.raises(ValidationError, match="visible text"):
        JudgeDefinition(name=" ", syllabus="Assess success.", dimensions=default.dimensions)
    sparse = RubricDimension(
        dimension_id="quality",
        name="Quality",
        description="Outcome quality.",
        min_score=1,
        max_score=3,
        anchors=(
            ScoreAnchor(score=1, description="Failed."),
            ScoreAnchor(score=3, description="Done."),
        ),
    )
    with pytest.raises(ValidationError, match="every score"):
        JudgeDefinition(name="Judge", syllabus="Assess quality.", dimensions=(sparse,))
