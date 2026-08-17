"""Bind a judge prompt contract to one shared rubric axis range."""

from __future__ import annotations

from wmo.common.judging import RubricDimension, default_task_success_axis, score_bounds
from wmo.common.judging.prompts import PromptDefinition
from wmo.optimize.router.judging.contracts import JudgePromptTemplate, judge_feedback_schema

_PROMPT_TEXT = (
    "Evaluate the supplied rollout against every rubric axis. Use only evidence in the "
    "rollout. Return strict JSON matching the supplied schema, with one integer score inside "
    "each axis inclusive range and an optional nullable rationale for every axis."
)
DEFAULT_JUDGE_PROMPT = PromptDefinition.from_text("wmo-judge-evidence-json-v3", _PROMPT_TEXT)


def default_judge_dimensions() -> tuple[RubricDimension, ...]:
    """Return the editable default task-success axis for first setup.

    Returns:
        One 0-1 axis whose meaning is completion of the original user prompt.
    """
    return (default_task_success_axis(),)


def default_judge_template(
    dimensions: tuple[RubricDimension, ...] | None = None,
) -> JudgePromptTemplate:
    """Return the built-in scalar prompt contract bound to one rubric's score bounds.

    Args:
        dimensions: Axes whose inclusive ranges set the scalar schema. Defaults to the
            built-in task-success axis.

    Returns:
        Versioned prompt, mapping, and response schema for those axes.
    """
    selected = dimensions or default_judge_dimensions()
    lowest, highest = score_bounds(selected)
    return JudgePromptTemplate(
        prompt=DEFAULT_JUDGE_PROMPT,
        variable_mapping={"rubric": "RUBRIC", "rollout": "ROLLOUT"},
        response_schema=judge_feedback_schema("scalar", min_score=lowest, max_score=highest),
    )


DEFAULT_JUDGE_TEMPLATE = default_judge_template()


def bind_prompt_template(
    template: JudgePromptTemplate,
    dimensions: tuple[RubricDimension, ...],
) -> JudgePromptTemplate:
    """Bind a prompt contract to the shared inclusive axis range.

    Args:
        template: Current prompt contract.
        dimensions: Replacement ordered axes.

    Returns:
        The same contract with scalar bounds updated to the selected range, or
        the unchanged non-scalar contract after projection checks.

    Raises:
        ValueError: The axes are empty, mixed, or a custom projection leaves the range.
    """
    lowest, highest = score_bounds(dimensions)
    _require_projection_in_range(template, lowest, highest)
    if template.response_shape != "scalar":
        return template
    return template.model_copy(
        update={
            "response_schema": judge_feedback_schema(
                "scalar",
                min_score=lowest,
                max_score=highest,
            )
        }
    )


def _require_projection_in_range(
    template: JudgePromptTemplate,
    min_score: int,
    max_score: int,
) -> None:
    """Reject custom projections that fall outside the selected axis range.

    Args:
        template: Current prompt contract.
        min_score: Inclusive lower bound of the selected axes.
        max_score: Inclusive upper bound of the selected axes.

    Raises:
        ValueError: A projected score is outside the shared axis range.
    """
    projection = template.score_projection
    if template.response_shape == "boolean":
        values = tuple(projection.boolean_scores.values())
    elif template.response_shape == "categorical":
        values = tuple(projection.categorical_scores.values())
    elif template.response_shape == "pairwise":
        values = tuple(projection.pairwise_scores.values())
    else:
        return
    if any(value < min_score or value > max_score for value in values) or (
        values and (min(values) != min_score or max(values) != max_score)
    ):
        raise ValueError(
            f"custom {template.response_shape} score projections must include "
            f"{min_score} and {max_score} and stay inside that range"
        )
