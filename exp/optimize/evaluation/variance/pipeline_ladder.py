"""Upper-bound prices of the ``families`` stage's variant search and nudge ladder.

Each search round and each ladder round authors its own frozen world, rubric and judges for the
set it measures (:meth:`~exp.optimize.evaluation.variance.pipeline_families._Point.judge`), then
builds, witnesses and contrasts every variant and measures the set from judged free
continuations. The bounds below price every round as if it ran to its ceiling; a round whose set
the family later contrasts again replays, so the bound is never exceeded.
"""

from __future__ import annotations

from dataclasses import dataclass

from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.executable.costs import evaluation_costs, job_costs
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.variance.briefing_review import DEFAULT_REVIEW_REPAIRS
from exp.optimize.evaluation.variance.decision_measure import DecisionMeasureSpec
from exp.optimize.evaluation.variance.pipeline_inputs import library_draws, rubric_draws
from exp.optimize.evaluation.variance.pipeline_spec import AuthoringSettings


@dataclass(frozen=True)
class MeasuredSet:
    """One measured scenario set's size and what one of its variants costs.

    Attributes:
        states: Scenarios in the set.
        options: Options per decision.
    """

    states: int
    options: int


def chain_costs(project: ProjectStore, authoring: AuthoringSettings) -> list[CostComponent]:
    """Price one authored chain: the world author, the rubric author with its critic and library."""
    world = len(authoring.world_prompt_sets) * (2 + authoring.world_repairs)
    costs = [row.scaled(world) for row in evaluation_costs(project, authoring.world_author)]
    rubric = rubric_draws(authoring.rubric_profile) + library_draws(authoring)
    costs.extend(row.scaled(rubric) for row in evaluation_costs(project, authoring.rubric_author))
    return costs


def measured_set_costs(
    project: ProjectStore,
    authoring: AuthoringSettings,
    measure: DecisionMeasureSpec,
    episode: ContinuationJob,
    shape: MeasuredSet,
) -> tuple[list[CostComponent], int]:
    """Price contrasting and measuring one set; return its costs and its judged episodes.

    Args:
        project: Project whose catalog prices the roles.
        authoring: The families settings.
        measure: The joint measurement settings.
        episode: One actor episode's job.
        shape: The set's size.

    Returns:
        Actor, classifier and reviewer costs, and the episodes the frozen judge labels.
    """
    contrasted = shape.states * shape.options * authoring.repeats
    free = shape.states * measure.free_repeats
    runs = shape.states + contrasted + free
    costs = [row.scaled(runs) for row in job_costs(project, episode)]
    classify = free * measure.classifier_profile.budget.model_calls
    review = (
        shape.states * len(measure.reviewer_stances) * measure.reviewer_profile.budget.model_calls
    )
    for settings, draws in ((measure.classifier, classify), (measure.reviewer, review)):
        costs.extend(row.scaled(draws) for row in evaluation_costs(project, settings))
    return costs, contrasted + free


def search_ladder_costs(
    project: ProjectStore, authoring: AuthoringSettings, episode: ContinuationJob, options: int
) -> tuple[list[CostComponent], int, tuple[str, ...]]:
    """Price one family's variant search and nudge ladder as upper bounds.

    Args:
        project: Project whose catalog prices the roles.
        authoring: The families settings.
        episode: One actor episode's job.
        options: Most options per decision.

    Returns:
        The costs, the judged episodes they add, and plan lines describing both.
    """
    measure = authoring.decision_measure()
    search, ladder = authoring.search_spec(), authoring.ladder_spec()
    if measure is None or (search is None and ladder is None):
        return [], 0, ("Variant search and nudge ladder: off.",)
    costs: list[CostComponent] = []
    judged = 0
    lines: list[str] = []
    if search is not None:
        per_round = [
            row.scaled(search.profile.budget.model_calls)
            for row in evaluation_costs(project, search.author)
        ]
        per_round.extend(chain_costs(project, authoring))
        measured, episodes = measured_set_costs(
            project, authoring, measure, episode, MeasuredSet(authoring.maximum_variants, options)
        )
        per_round.extend(measured)
        costs.extend(row.scaled(search.maximum_rounds) for row in per_round)
        judged += search.maximum_rounds * episodes
        lines.append(
            f"Variant search: up to {search.maximum_rounds} rounds per family on "
            f"{search.author.model_alias} ({len(search.stances)} stances rotated), each set with "
            "its own frozen world, rubric and judges, contrasted and measured; only an admitted "
            "set is kept."
        )
    if ladder is not None:
        strengths = len(ladder.strengths)
        per_round = [
            row.scaled(ladder.profile.budget.model_calls)
            for row in evaluation_costs(project, ladder.author)
        ]
        leak = strengths * len(ladder.leak_variants) * (1 + DEFAULT_REVIEW_REPAIRS)
        realism = strengths * len(ladder.realism_stances) * ladder.profile.budget.model_calls
        for settings, draws in ((ladder.leak_reviewer, leak), (ladder.realism_reviewer, realism)):
            per_round.extend(row.scaled(draws) for row in evaluation_costs(project, settings))
        per_round.extend(chain_costs(project, authoring))
        measured, episodes = measured_set_costs(
            project, authoring, measure, episode, MeasuredSet(strengths + 1, options)
        )
        per_round.extend(measured)
        costs.extend(row.scaled(ladder.maximum_rounds) for row in per_round)
        judged += ladder.maximum_rounds * episodes
        lines.append(
            f"Nudge ladder: up to {ladder.maximum_rounds} rounds per admitted family on "
            f"{ladder.author.model_alias}, {strengths} nudges from "
            f"{ladder.strengths[0]} to {ladder.strengths[-1]} each leak-reviewed under "
            f"{len(ladder.leak_variants)} stances and realism-reviewed under "
            f"{len(ladder.realism_stances)}, measured with the un-nudged base under one rubric; "
            "every admitted rung and the un-nudged target become their own tasks."
        )
    return costs, judged, tuple(lines)
