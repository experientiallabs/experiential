"""A family's searched scenario set and its nudge ladder, as hooks of :func:`run_decision_family`.

Two optional stages replace or extend one family's single scenario draw, both measured by the
joint decision-variance measure under the family's frozen judges:

- **Variant search** (:func:`searched_scenarios`). With ``DecisionFamilySpec.search`` set, the
  family's scenarios come from
  :func:`~exp.optimize.evaluation.variance.variant_search.search_variants`: bounded author rounds,
  each set built, witnessed, contrasted and measured
  (:func:`~exp.optimize.evaluation.variance.variant_search.variant_measurer`) with every deficit
  fed back, accepting only a set the joint measure admits. The accepted set is the family's
  scenario set; every later family stage replays the measured variants.
- **Nudge ladder** (:func:`family_ladder`). With ``DecisionFamilySpec.ladder`` set, once the
  joint measure admits the family, the state whose free agents least often chose its declared
  correct option becomes the base of
  :func:`~exp.optimize.evaluation.variance.nudge_ladder.build_nudge_ladder`, measured by
  :func:`~exp.optimize.evaluation.variance.variant_search.ladder_measurer`. Every rung of the
  ladder's curriculum (the admitted nudged rungs, strongest first, then always the un-nudged
  base) becomes its own task: a :class:`RungTask` recording its strength, ladder position, judge
  binding and the replayed variant it runs from.

Both stages bind scenario sets through one :class:`FamilyBinder`, which asks the caller's judge
function for a whole set at a time, so a caller can author one frozen world, rubric and judge
per set; a binding is fixed before any contrast of the set runs and is never re-asked.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject
from exp.common.models import ToolCall
from exp.common.project import ProjectStore
from exp.common.tasks.facts import InstanceFacts
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.variance.admission import VariantJudging, frozen_start
from exp.optimize.evaluation.variance.authors import ScenarioDraft
from exp.optimize.evaluation.variance.contracts import (
    ActorView,
    DecisionState,
    Scenario,
    ScenarioLimits,
    StatePart,
)
from exp.optimize.evaluation.variance.decision_measure import DecisionMeasure, DecisionMeasureSpec
from exp.optimize.evaluation.variance.family_export import variant_job
from exp.optimize.evaluation.variance.family_stages import (
    FamilyVariant,
    failure_reason,
    run_variant,
)
from exp.optimize.evaluation.variance.nudge_ladder import (
    NudgeLadder,
    NudgeLadderSpec,
    build_nudge_ladder,
)
from exp.optimize.evaluation.variance.nudges import RungStrength
from exp.optimize.evaluation.variance.variant_search import (
    BoundVariant,
    VariantSearch,
    VariantSearchSpec,
    ladder_measurer,
    search_variants,
    variant_measurer,
)

SetJudges = Callable[[DecisionState, Sequence[Scenario]], Mapping[str, VariantJudging]]
"""Gives every scenario of one set its frozen judge, world freeze and compatibility review."""


class FamilyBinder:
    """Binds scenario sets to frozen judges, world-freeze starts and jobs, once per set.

    Attributes:
        project: Project holding the world freezes.
        job: The family's frozen job, rebound per frozen variant.
        fixed: The spec's judge binding per scenario identity, when given in advance.
        judges: The caller's set-level judge function, when the spec has no mapping.
    """

    def __init__(
        self,
        project: ProjectStore,
        job: ContinuationJob,
        fixed: Mapping[str, VariantJudging] | None,
        judges: SetJudges | None,
    ) -> None:
        """Bind the project, the family job and exactly one source of judges."""
        self.project, self.job, self.fixed, self.judges = project, job, fixed, judges

    def judging(
        self, decision: DecisionState, scenarios: Sequence[Scenario]
    ) -> dict[str, VariantJudging]:
        """Return every scenario's judge binding, asking the caller once for the whole set.

        Raises:
            KeyError: A scenario has no binding.
        """
        if self.fixed is not None:
            fixed = self.fixed
            found = {
                s.scenario_id: fixed[s.scenario_id] for s in scenarios if s.scenario_id in fixed
            }
        elif self.judges is not None:
            found = dict(self.judges(decision, tuple(scenarios)))
        else:
            found = {}
        missing = sorted({s.scenario_id for s in scenarios} - set(found))
        if missing:
            raise KeyError(f"no judge binding for scenarios {missing}")
        return found

    def bind(
        self, decision: DecisionState, scenarios: Sequence[Scenario]
    ) -> tuple[dict[str, BoundVariant], dict[str, VariantJudging]]:
        """Bind a set: each scenario's job, judge and frozen start, beside its judge binding.

        Raises:
            KeyError: A scenario has no binding.
            ValueError: A bound world freeze has no usable start for a scenario.
        """
        judging = self.judging(decision, scenarios)
        bound: dict[str, BoundVariant] = {}
        for sid, binding in judging.items():
            start = frozen_start(self.project, binding)
            bound[sid] = BoundVariant(
                job=variant_job(self.job, start),
                judge=binding.judge,
                frozen=None if start is None else start[0],
            )
        return bound, judging

    def __call__(
        self, decision: DecisionState, scenarios: Sequence[Scenario]
    ) -> dict[str, BoundVariant]:
        """Bind a set for the measurers: each scenario's job, judge and frozen start."""
        return self.bind(decision, scenarios)[0]


@dataclass(frozen=True)
class FamilyInputs:
    """What the measured hooks share with the family's own stages.

    Attributes:
        base: The decision's checkpoint in the picked trajectory.
        boundaries: Every saved boundary of the picked attempt, start first.
        actions: Concrete forced first action of every option.
        facts: Instance facts the world model was given, if any.
        repeats: Pairs per selection contrast.
        required_state: State parts every variant must hold beyond the baseline.
    """

    base: ArtifactInput
    boundaries: tuple[ArtifactInput, ...]
    actions: Mapping[str, ToolCall]
    repeats: int
    facts: InstanceFacts | None = None
    required_state: frozenset[StatePart] = field(default_factory=frozenset)


def searched_scenarios(
    project: ProjectStore,
    spec: VariantSearchSpec,
    measure: DecisionMeasureSpec,
    *,
    decision: DecisionState,
    before: JsonObject,
    limits: ScenarioLimits,
    view: ActorView | None,
    binder: FamilyBinder,
    inputs: FamilyInputs,
    identity: str,
    sources: tuple[ArtifactInput, ...],
    code_revision: str,
) -> tuple[ArtifactInput, VariantSearch, ScenarioDraft]:
    """Run, or replay, the family's variant search and return its result as a scenario draft.

    Args:
        project: Project holding every round, draw and measure.
        spec: The frozen search settings.
        measure: The family's joint measurement settings.
        decision: The family's decision state.
        before: World state just before the decision's step.
        limits: Tools and remaining calls available to the actor.
        view: What the actor had seen, needed to accept a fresh briefing.
        binder: Binds each round's set before its contrasts.
        inputs: The family's checkpoints, actions and contrast settings.
        identity: Stable identity of the search.
        sources: Saved artifacts the decision state and before-state came from.
        code_revision: Producing source identity.

    Returns:
        The saved search, the search, and a scenario draft that is accepted exactly when a round
        was admitted.
    """
    measurer = variant_measurer(
        project,
        measure,
        decision=decision,
        bind=binder,
        base=inputs.base,
        boundaries=inputs.boundaries,
        actions=inputs.actions,
        facts=inputs.facts,
        repeats=inputs.repeats,
        code_revision=code_revision,
        required_state=inputs.required_state,
    )
    reference, search = search_variants(
        project,
        spec,
        decision,
        before,
        limits,
        measure=measurer,
        identity=identity,
        code_revision=code_revision,
        view=view,
        sources=sources,
    )
    if search.scenarios is not None:
        draft = ScenarioDraft(identity=search.identity, scenarios=search.scenarios, accepted=True)
    else:
        last = search.rounds[-1].deficits if search.rounds else ()
        reason = f"variant search admitted no set in {len(search.rounds)} rounds"
        draft = ScenarioDraft(
            identity=search.identity,
            failure=(reason + ("; last deficits: " + "; ".join(last) if last else ""))[:2000],
        )
    return reference, search, draft


class RungTask(ContractModel):
    """One rung of a family's nudge-ladder curriculum, as a task of its own.

    Attributes:
        scenario_id: The rung's scenario, under the ladder's decision.
        position: Ladder position, from 0 (strongest); the un-nudged rung is last.
        strength: Nudge strength; ``none`` is the un-nudged target.
        nudge: The nudge message, None for the un-nudged rung.
        scenario: The rung's scenario.
        judging: The rung's frozen judge binding, shared by every rung of its measured set.
        variant: The rung's replayed variant, which its task starts from.
    """

    scenario_id: str
    position: int = Field(ge=0)
    strength: RungStrength
    nudge: str | None = None
    scenario: Scenario
    judging: VariantJudging
    variant: FamilyVariant


def ladder_base(measure: DecisionMeasure, scenarios: Sequence[Scenario]) -> Scenario | None:
    """Return the measured state whose free agents least often chose its correct option.

    Unclear and other choices count against the state; ties keep scenario order.

    Args:
        measure: The family's admitted joint measure.
        scenarios: The family's scenarios.

    Returns:
        The hardest measured state's scenario, or None when no state was measured.
    """
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    shares: list[tuple[float, int, Scenario]] = []
    for index, state in enumerate(measure.states):
        scenario = by_id.get(state.state_id)
        choices = state.evidence.choices
        if scenario is None or not choices:
            continue
        right = sum(choice == state.evidence.correct_option for choice in choices)
        shares.append((right / len(choices), index, scenario))
    return min(shares, key=lambda row: row[:2])[2] if shares else None


def family_ladder(
    project: ProjectStore,
    spec: NudgeLadderSpec,
    measure: DecisionMeasureSpec,
    *,
    decision: DecisionState,
    base: Scenario,
    instruction: str,
    view: ActorView,
    before: JsonObject,
    binder: FamilyBinder,
    inputs: FamilyInputs,
    identity: str,
    sources: tuple[ArtifactInput, ...],
    code_revision: str,
) -> tuple[ArtifactInput, NudgeLadder, tuple[RungTask, ...]] | str:
    """Build, or replay, the family's nudge ladder and turn its curriculum into rung tasks.

    Args:
        project: Project holding every draw, measure and the ladder.
        spec: The frozen ladder settings.
        measure: The family's joint measurement settings.
        decision: The family's decision state.
        base: The un-nudged scenario every rung keeps.
        instruction: The task's own instruction.
        view: What the actor had seen at the decision.
        before: World state just before the decision.
        binder: Binds each measured set of rungs before its contrasts.
        inputs: The family's checkpoints, actions and contrast settings.
        identity: Stable identity of the ladder.
        sources: Saved artifacts the decision and base scenario came from.
        code_revision: Producing source identity.

    Returns:
        The saved ladder, the ladder and one task per curriculum rung, or why no task could be
        built from it.
    """
    measurer = ladder_measurer(
        project,
        measure,
        bind=binder,
        base=inputs.base,
        boundaries=inputs.boundaries,
        actions=inputs.actions,
        facts=inputs.facts,
        repeats=inputs.repeats,
        code_revision=code_revision,
        required_state=inputs.required_state,
    )
    reference, ladder = build_nudge_ladder(
        project,
        spec,
        decision,
        base,
        instruction=instruction,
        view=view,
        before=before,
        measure=measurer,
        identity=identity,
        code_revision=code_revision,
        sources=sources,
    )
    chosen = next((row for row in ladder.rounds if row.round == ladder.chosen_round), None)
    if chosen is None or chosen.decision is None or chosen.measure is None:
        return "the ladder measured no round of rungs"
    laddered = chosen.decision
    measured = [
        rung.scenario
        for rung in chosen.rungs[:-1]
        if rung.review is not None and rung.review.passed
    ]
    try:
        bound, judging = binder.bind(laddered, (base, *measured))
    except (ValueError, KeyError) as error:
        return f"rung binding: {failure_reason(error)}"
    tasks: list[RungTask] = []
    for rung in ladder.curriculum:
        sid = rung.scenario.scenario_id
        if sid not in bound:
            continue
        variant = run_variant(
            project,
            job=bound[sid].job,
            base=inputs.base,
            boundaries=inputs.boundaries,
            decision=laddered,
            scenario=rung.scenario,
            actions=inputs.actions,
            facts=inputs.facts,
            repeats=inputs.repeats,
            judge=bound[sid].judge,
            code_revision=code_revision,
            required_state=inputs.required_state,
            frozen=bound[sid].frozen,
        )
        tasks.append(
            RungTask(
                scenario_id=sid,
                position=rung.position,
                strength=rung.strength,
                nudge=None if rung.nudge is None else rung.nudge.text,
                scenario=rung.scenario,
                judging=judging[sid],
                variant=variant,
            )
        )
    return reference, ladder, tuple(tasks)


__all__ = [
    "FamilyBinder",
    "FamilyInputs",
    "RungTask",
    "SetJudges",
    "family_ladder",
    "ladder_base",
    "searched_scenarios",
]
