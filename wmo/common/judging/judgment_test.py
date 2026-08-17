"""Tests for immutable judgments linked to rollout and rubric evidence."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from wmo.common.judging import DimensionJudgment, Judgment
from wmo.common.models import ModelSnapshot

_DIGEST = "a" * 64


def _judge_model() -> ModelSnapshot:
    return ModelSnapshot(
        provider="openai",
        model_id="gpt-5.4-mini",
        capabilities_sha256=_DIGEST,
        connection_sha256=_DIGEST,
    )


def _dimension_judgment() -> DimensionJudgment:
    return DimensionJudgment(
        dimension_id="task-success",
        raw_score=4,
        calibrated_score=3.8,
        rationale="The refund request includes the order and reason.",
    )


def test_judgment_rejects_duplicate_dimensions_and_nonfinite_scores() -> None:
    """A judgment cannot silently score one rubric dimension twice or store NaN."""
    with pytest.raises(ValidationError, match="repeat"):
        Judgment(
            schema_version=1,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="e7aad17",
            judgment_id="judgment-1",
            rollout_id="rollout-1",
            rubric_id="support-rubric-v1",
            calibration_id="judge-calibration-v1",
            judge_model=_judge_model(),
            judge_prompt_id="judge-prompt-v1",
            judge_prompt_sha256=_DIGEST,
            dimensions=(_dimension_judgment(), _dimension_judgment()),
            overall_score=0.76,
        )
    with pytest.raises(ValidationError):
        DimensionJudgment(
            dimension_id="task-success",
            raw_score=4,
            calibrated_score=float("nan"),
            rationale="Invalid score.",
        )
    with pytest.raises(ValidationError, match="equal-weight"):
        Judgment(
            schema_version=1,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="e7aad17",
            judgment_id="judgment-1",
            rollout_id="rollout-1",
            rubric_id="support-rubric-v1",
            calibration_id="judge-calibration-v1",
            judge_model=_judge_model(),
            judge_prompt_id="judge-prompt-v1",
            judge_prompt_sha256=_DIGEST,
            dimensions=(_dimension_judgment(),),
            overall_score=0.75,
        )


def test_dimension_judgment_accepts_missing_null_and_unbounded_rationale() -> None:
    """A required score may omit rationale, store null, or keep an arbitrarily long string."""
    missing = DimensionJudgment(
        dimension_id="task-success",
        raw_score=4,
        calibrated_score=3.8,
    )
    explicit_null = DimensionJudgment.model_validate(
        {
            "dimension_id": "task-success",
            "raw_score": 4,
            "calibrated_score": 3.8,
            "rationale": None,
        }
    )
    long_rationale = "x" * 10_000
    unbounded = DimensionJudgment(
        dimension_id="task-success",
        raw_score=4,
        calibrated_score=3.8,
        rationale=long_rationale,
    )

    assert missing.rationale is None
    assert explicit_null.rationale is None
    assert unbounded.rationale == long_rationale


def test_dimension_judgment_loads_retired_citation_fields() -> None:
    """Citation-era judgments keep their score and map feedback onto rationale."""
    loaded = DimensionJudgment.model_validate(
        {
            "dimension_id": "task-success",
            "raw_score": 4,
            "calibrated_score": 3.8,
            "min_score": 0,
            "max_score": 5,
            "evidence_span_ids": ["span-1"],
            "feedback": "The refund request includes the order and reason.",
        }
    )
    dumped = loaded.model_dump(mode="json")

    assert loaded.rationale == "The refund request includes the order and reason."
    assert "evidence_span_ids" not in dumped
    assert "feedback" not in dumped
