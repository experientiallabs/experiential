"""Interactive editing and human-readable rendering of one rubric contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt

from wmo.common.judging import (
    RubricDimension,
    ScoreAnchor,
    render_rubric_table,
    score_bounds,
)
from wmo.optimize.router.judging.contracts import JudgePromptTemplate
from wmo.optimize.router.judging.service import ManualJudgeSetupPlan
from wmo.optimize.router.judging.template_bind import bind_prompt_template


def render_setup_contract(plan: ManualJudgeSetupPlan, *, width: int = 80) -> str:
    """Render alias, model, prompt identity, and the human-readable rubric table.

    Args:
        plan: Read-only setup plan awaiting confirmation.
        width: Available terminal columns for the rubric table.

    Returns:
        Setup text with no schema dump and no raw trace preview.
    """
    model = f"{plan.judge_model.provider}/{plan.judge_model.model_id}"
    prompt = plan.prompt_template.prompt
    identity = (
        f"Judge alias: {plan.judge_alias}\n"
        f"Judge model: {model}\n"
        f"Prompt: {prompt.prompt_id} ({prompt.sha256})"
    )
    return f"{identity}\n\n{render_rubric_table(plan.dimensions, width=width)}"


def replace_setup_axes(
    plan: ManualJudgeSetupPlan,
    dimensions: tuple[RubricDimension, ...],
) -> ManualJudgeSetupPlan:
    """Return a plan that uses ``dimensions`` and rebinds the default prompt schema.

    Args:
        plan: Current setup plan.
        dimensions: Replacement ordered axes.

    Returns:
        A new plan whose default template matches the selected axis ranges.

    Raises:
        ValueError: The replacement has no axes.
    """
    if not dimensions:
        raise ValueError("a rubric must contain at least one axis")
    template = bind_prompt_template(plan.prompt_template, dimensions)
    return replace(plan, dimensions=dimensions, prompt_template=template)


def rebind_prompt_template(
    template: JudgePromptTemplate,
    dimensions: tuple[RubricDimension, ...],
) -> JudgePromptTemplate:
    """Bind a prompt contract to the shared inclusive axis range.

    Args:
        template: Current prompt contract.
        dimensions: Replacement ordered axes.

    Returns:
        The rebound prompt contract used by setup and the editor.

    Raises:
        ValueError: The axes are empty, mixed, or a custom projection leaves the range.
    """
    return bind_prompt_template(template, dimensions)


def maybe_edit_setup_plan(plan: ManualJudgeSetupPlan, *, console: Console) -> ManualJudgeSetupPlan:
    """Offer an interactive edit/save pass over the displayed rubric.

    Args:
        plan: Rendered setup plan.
        console: Local CLI console used for prompts and table reprints.

    Returns:
        The original plan, or a replacement after the operator edits axes.
    """
    if not Confirm.ask("Edit this rubric before saving?", default=False):
        return plan
    edited = edit_rubric_axes(plan.dimensions, console=console)
    updated = replace_setup_axes(plan, edited)
    console.print(render_setup_contract(updated, width=console.width))
    return updated


def edit_rubric_axes(
    dimensions: tuple[RubricDimension, ...],
    *,
    console: Console,
    ask: Callable[..., str] = Prompt.ask,
    ask_int: Callable[..., int] = IntPrompt.ask,
) -> tuple[RubricDimension, ...]:
    """Edit, add, or remove axes until the operator keeps a nonempty rubric.

    Args:
        dimensions: Current ordered axes.
        console: Local CLI console for status lines.
        ask: Text prompt used by tests and the live CLI.
        ask_int: Integer prompt used by tests and the live CLI.

    Returns:
        The edited nonempty axis tuple.

    Raises:
        ValueError: The operator removes the last axis or repeats an ID.
    """
    current = list(dimensions)
    while True:
        console.print(render_rubric_table(current, width=console.width))
        action = ask(
            "Edit [e], add [a], remove [r], or done [d]",
            choices=["e", "a", "r", "d"],
            default="d",
        )
        if action == "d":
            selected = tuple(current)
            score_bounds(selected)
            return selected
        if action == "a":
            current.append(_prompt_axis(console, ask=ask, ask_int=ask_int))
            ids = [item.dimension_id for item in current]
            if len(set(ids)) != len(ids):
                raise ValueError("rubric axes must have unique IDs")
            continue
        if not current:
            raise ValueError("a rubric must contain at least one axis")
        selected = _select_axis(current, ask=ask)
        if action == "r":
            if len(current) == 1:
                raise ValueError("a rubric must contain at least one axis")
            current = [item for item in current if item.dimension_id != selected.dimension_id]
            continue
        current = [
            _prompt_axis(console, existing=selected, ask=ask, ask_int=ask_int)
            if item.dimension_id == selected.dimension_id
            else item
            for item in current
        ]


def build_axis(
    dimension_id: str,
    name: str,
    description: str,
    min_score: int,
    max_score: int,
    meanings: dict[int, str],
) -> RubricDimension:
    """Build one canonical axis from editor fields.

    Args:
        dimension_id: Stable axis identity.
        name: Human label.
        description: Plain-language meaning of the axis.
        min_score: Inclusive lower bound.
        max_score: Inclusive upper bound.
        meanings: Score-to-description map. Endpoints are required.

    Returns:
        A validated rubric axis.

    Raises:
        ValueError: The range, IDs, or anchors are invalid.
    """
    anchors = tuple(
        ScoreAnchor(score=score, description=meanings[score]) for score in sorted(meanings)
    )
    return RubricDimension(
        dimension_id=dimension_id,
        name=name,
        description=description,
        min_score=min_score,
        max_score=max_score,
        anchors=anchors,
    )


def axis_score_choices(axis: RubricDimension) -> str:
    """Return the inclusive range shown next to a calibration score prompt."""
    return f"{axis.min_score}-{axis.max_score}"


def _select_axis(
    dimensions: list[RubricDimension],
    *,
    ask: Callable[..., str],
) -> RubricDimension:
    """Ask for one existing axis ID.

    Args:
        dimensions: Current axes.
        ask: Text prompt.

    Returns:
        The selected axis.

    Raises:
        ValueError: The ID is not in the current rubric.
    """
    known = {item.dimension_id: item for item in dimensions}
    selected = ask("Axis ID", choices=list(known), default=dimensions[0].dimension_id)
    return known[selected]


def _prompt_axis(
    console: Console,
    *,
    existing: RubricDimension | None = None,
    ask: Callable[..., str],
    ask_int: Callable[..., int],
) -> RubricDimension:
    """Collect one axis from the operator, including a meaning for every score.

    Args:
        console: Local CLI console reserved for future status lines.
        existing: Axis being edited, or ``None`` when adding.
        ask: Text prompt.
        ask_int: Integer prompt.

    Returns:
        A validated axis with a meaning for every permitted score.
    """
    del console
    dimension_id = existing.dimension_id if existing is not None else ask("Axis ID")
    name = ask("Label", default=existing.name if existing is not None else None)
    min_score = ask_int(
        "Lowest score",
        default=existing.min_score if existing is not None else 0,
    )
    max_score = ask_int(
        "Highest score",
        default=existing.max_score if existing is not None else 1,
    )
    description = ask(
        "Meaning",
        default=existing.description if existing is not None else None,
    )
    known = {anchor.score: anchor.description for anchor in existing.anchors} if existing else {}
    meanings = {
        score: ask(
            f"Score {score} meaning",
            default=known.get(score),
        )
        for score in range(min_score, max_score + 1)
    }
    return build_axis(dimension_id, name, description, min_score, max_score, meanings)
