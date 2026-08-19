"""Tests for rubric table editing helpers used by judge setup."""

from __future__ import annotations

from io import StringIO

import pytest
from pydantic import ValidationError
from rich.console import Console

from wmo.cli.judge.rubric import build_axis, edit_rubric_axes
from wmo.common.core.artifacts import JsonObject
from wmo.common.judging import PromptDefinition, default_task_success_axis
from wmo.optimize.router.judging.contracts import (
    JudgePromptTemplate,
    JudgeScoreProjection,
    judge_feedback_schema,
)
from wmo.optimize.router.judging.template_bind import bind_prompt_template, default_judge_template


def _raw_score_bounds(schema: JsonObject) -> tuple[int, int]:
    """Read inclusive scalar bounds from a generated judge response schema."""
    properties = schema["properties"]
    assert isinstance(properties, dict)
    dimensions = properties["dimensions"]
    assert isinstance(dimensions, dict)
    items = dimensions["items"]
    assert isinstance(items, dict)
    item_properties = items["properties"]
    assert isinstance(item_properties, dict)
    raw_score = item_properties["raw_score"]
    assert isinstance(raw_score, dict)
    minimum = raw_score["minimum"]
    maximum = raw_score["maximum"]
    assert isinstance(minimum, int)
    assert isinstance(maximum, int)
    return minimum, maximum


def test_build_axis_requires_unique_endpoint_meanings() -> None:
    """Editor fields produce the same canonical axis contract used everywhere else."""
    axis = build_axis(
        "task-success",
        "Task success",
        "The agent successfully completed the task requested in the original user prompt",
        0,
        1,
        {
            0: "The agent did not complete the requested task.",
            1: "The agent successfully completed the requested task.",
        },
    )

    assert axis == default_task_success_axis()
    with pytest.raises(ValidationError, match="inclusive range endpoints"):
        build_axis(
            "quality",
            "Quality",
            "How completely the agent solved the requested work.",
            0,
            4,
            {2: "Partial completion."},
        )


def test_default_template_schema_follows_selected_axis_bounds() -> None:
    """Prompt generation binds the scalar schema to the same axis ranges."""
    default_schema = default_judge_template((default_task_success_axis(),)).response_schema
    assert _raw_score_bounds(default_schema) == (0, 1)

    wide = build_axis(
        "quality",
        "Quality",
        "How completely the agent solved the requested work.",
        0,
        4,
        {
            0: "No useful progress.",
            2: "Partial completion with remaining gaps.",
            4: "Complete and correct.",
        },
    )
    wide_schema = default_judge_template((wide,)).response_schema
    assert _raw_score_bounds(wide_schema) == (0, 4)

    custom = JudgePromptTemplate(
        prompt=PromptDefinition.from_text("custom-judge-v1", "Follow the saved contract exactly."),
        variable_mapping={"rubric": "RULES_CUSTOM", "rollout": "TRACE_CUSTOM"},
        response_schema=judge_feedback_schema("scalar", min_score=0, max_score=1),
    )
    rebound = bind_prompt_template(custom, (wide,))
    assert rebound.prompt.prompt_id == "custom-judge-v1"
    assert _raw_score_bounds(rebound.response_schema) == (0, 4)

    boolean = JudgePromptTemplate(
        prompt=PromptDefinition.from_text("custom-bool-v1", "Return passed."),
        response_shape="boolean",
        variable_mapping={"rubric": "RULES_CUSTOM", "rollout": "TRACE_CUSTOM"},
        response_schema=judge_feedback_schema("boolean"),
        score_projection=JudgeScoreProjection(boolean_scores={"false": 0, "true": 4}),
    )
    with pytest.raises(ValueError, match="boolean score projections"):
        bind_prompt_template(boolean, (default_task_success_axis(),))
    assert bind_prompt_template(boolean, (wide,)) is boolean
    stale = boolean.model_copy(
        update={"score_projection": JudgeScoreProjection(boolean_scores={"false": 0, "true": 1})}
    )
    with pytest.raises(ValueError, match="include 0 and 4"):
        bind_prompt_template(stale, (wide,))


def test_edit_done_keeps_the_current_axes() -> None:
    """Choosing done leaves the displayed rubric unchanged."""
    current = (default_task_success_axis(),)

    result = edit_rubric_axes(
        current,
        console=Console(file=StringIO()),
        ask=lambda *_args, **_kwargs: "d",
        ask_int=lambda *_args, **_kwargs: 0,
    )

    assert result == current
