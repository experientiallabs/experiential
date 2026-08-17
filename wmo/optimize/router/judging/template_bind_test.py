"""Tests for binding judge prompt contracts to one shared axis range."""

from __future__ import annotations

import pytest

from wmo.common.judging import PromptDefinition, default_task_success_axis
from wmo.optimize.router.judging.contracts import (
    JudgePromptTemplate,
    JudgeScoreProjection,
    judge_feedback_schema,
)
from wmo.optimize.router.judging.template_bind import bind_prompt_template


def test_bind_prompt_template_rejects_stale_boolean_projections() -> None:
    """File-based custom projections must span the selected axis, even without the editor."""
    template = JudgePromptTemplate(
        prompt=PromptDefinition.from_text("custom-bool-v1", "Return passed."),
        response_shape="boolean",
        variable_mapping={"rubric": "RULES_CUSTOM", "rollout": "TRACE_CUSTOM"},
        response_schema=judge_feedback_schema("boolean"),
        score_projection=JudgeScoreProjection(boolean_scores={"false": 1, "true": 4}),
    )

    with pytest.raises(ValueError, match="boolean score projections"):
        bind_prompt_template(template, (default_task_success_axis(),))
