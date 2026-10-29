"""Tests for binding judge prompt contracts to one shared axis range."""

from __future__ import annotations

import pytest

from exp.common.judging import (
    JudgeDefinition,
    PromptDefinition,
    default_task_success_axis,
    scored_axis,
)
from exp.optimize.router.judging.contracts import (
    JudgePromptTemplate,
    JudgeScoreProjection,
    judge_feedback_schema,
)
from exp.optimize.router.judging.template_bind import (
    DEFAULT_JUDGE_PROMPT,
    bind_prompt_template,
    default_judge_template,
    judge_template,
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


def test_bind_prompt_template_rebinds_custom_scalar_and_checks_boolean_ranges() -> None:
    """A wide axis rebinds a custom scalar schema and re-checks boolean projections."""
    wide = scored_axis("quality", "Quality", "Completeness.", min_score=0, max_score=4)
    custom = JudgePromptTemplate(
        prompt=PromptDefinition.from_text("custom-judge-v1", "Follow the saved contract exactly."),
        variable_mapping={"rubric": "RULES_CUSTOM", "rollout": "TRACE_CUSTOM"},
        response_schema=judge_feedback_schema("scalar", min_score=0, max_score=1),
    )
    rebound = bind_prompt_template(custom, (wide,))
    assert rebound.prompt.prompt_id == "custom-judge-v1"
    assert rebound.response_schema == judge_feedback_schema("scalar", min_score=0, max_score=4)

    boolean = JudgePromptTemplate(
        prompt=PromptDefinition.from_text("custom-bool-v1", "Return passed."),
        response_shape="boolean",
        variable_mapping={"rubric": "RULES_CUSTOM", "rollout": "TRACE_CUSTOM"},
        response_schema=judge_feedback_schema("boolean"),
        score_projection=JudgeScoreProjection(boolean_scores={"false": 0, "true": 4}),
    )
    assert bind_prompt_template(boolean, (wide,)) is boolean
    stale = boolean.model_copy(
        update={"score_projection": JudgeScoreProjection(boolean_scores={"false": 0, "true": 1})}
    )
    with pytest.raises(ValueError, match="include 0 and 4"):
        bind_prompt_template(stale, (wide,))


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


def test_judge_template_pins_syllabus_and_independent_ranges() -> None:
    """A stored reusable definition has an executable schema and exact prompt provenance."""
    definition = JudgeDefinition(
        name="Support",
        syllabus="Penalize unsupported claims.",
        dimensions=(
            default_task_success_axis(),
            scored_axis("quality", "Quality", "Helpfulness.", min_score=-2, max_score=2),
        ),
    )
    template = judge_template(definition)
    assert definition.syllabus in template.prompt.text
    assert definition.revision in template.prompt.prompt_id
    assert template.response_schema == judge_feedback_schema("scalar", min_score=-2, max_score=2)
    assert JudgePromptTemplate.model_validate_json(template.model_dump_json()) == template
