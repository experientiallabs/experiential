"""Tests for binding judge prompt contracts to one shared axis range."""

from __future__ import annotations

import pytest

from wmo.common.judging import PromptDefinition, default_task_success_axis
from wmo.optimize.router.judging.contracts import (
    JudgePromptTemplate,
    JudgeScoreProjection,
    judge_feedback_schema,
)
from wmo.optimize.router.judging.template_bind import (
    DEFAULT_JUDGE_PROMPT,
    bind_prompt_template,
    default_judge_template,
)


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


def test_bind_prompt_template_keeps_custom_scalar_with_builtin_prompt_id() -> None:
    """A custom scalar contract is not replaced just because it reuses the built-in prompt ID."""
    template = JudgePromptTemplate(
        prompt=PromptDefinition.from_text(
            DEFAULT_JUDGE_PROMPT.prompt_id,
            "Score only the cited tool failures.",
        ),
        variable_mapping={"rubric": "RULES_CUSTOM", "rollout": "TRACE_CUSTOM"},
        response_schema=judge_feedback_schema("scalar", min_score=0, max_score=5),
    )

    bound = bind_prompt_template(template, (default_task_success_axis(),))

    assert bound.prompt.text == "Score only the cited tool failures."
    assert bound.prompt.sha256 != DEFAULT_JUDGE_PROMPT.sha256
    assert bound.variable_mapping == {"rubric": "RULES_CUSTOM", "rollout": "TRACE_CUSTOM"}
    assert (
        bound.response_schema
        == default_judge_template((default_task_success_axis(),)).response_schema
    )
