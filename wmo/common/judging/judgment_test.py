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
    )


def _dimension_judgment() -> DimensionJudgment:
    return DimensionJudgment(
        dimension_id="task-success",
        raw_score=4,
        calibrated_score=3.8,
        evidence_span_ids=("span-1",),
        feedback="The refund request includes the order and reason.",
    )


def test_judgment_round_trip() -> None:
    """A score stays linked to explicit rollout, rubric, and calibration identities."""
    judgment = Judgment(
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
        overall_score=0.76,
    )

    assert Judgment.model_validate_json(judgment.model_dump_json()) == judgment


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
            evidence_span_ids=("span-1",),
            feedback="Invalid score.",
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
    with pytest.raises(ValidationError, match="at least one cited"):
        DimensionJudgment(
            dimension_id="task-success",
            raw_score=4,
            calibrated_score=3.8,
            evidence_span_ids=(),
            feedback="The refund request includes the order and reason.",
        )
