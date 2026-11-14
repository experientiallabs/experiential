"""An LLM-driven search for scenario variants that carry decision variance and stay clear.

Maximizing uncertainty alone produces confusing tasks, so a variant set is accepted here only when
its measured decision variance holds on all three joint criteria
(:func:`~exp.optimize.evaluation.variance.decision_measure.joint_verdict`): the frozen actor's
choices change across the scenario states, those choices change eventual judged success, and
every state is clear (a separately best option that matches the declared one and an LLM clarity
review that it is decidable from observable facts).

:func:`search_variants` runs bounded rounds. Each round:

1. **Author.** The scenario author draws through
   :func:`~exp.optimize.evaluation.variance.family_stages.run_role` with one stance per round
   (:data:`SEARCH_STANCES`, rotated): it plans, reads the decision state, the before-state (as
   ``world:0``), the actor's limits and every earlier round's measured results, submits a draft
   that the host checks with the existing scenario checks
   (:func:`~exp.optimize.evaluation.variance.authors.review_scenarios`), self checks, and gives
   its final set through the existing ``author_scenarios`` answer schema.
2. **Measure.** A structurally sound set goes to the caller's :data:`Measurer`, which builds the
   variants, contrasts and free continuations and returns the saved decision measure;
   :func:`variant_measurer` composes it from the family stages.
3. **Feedback.** Structural problems or the measure's deficits, with per-state choice counts,
   consequences and clarity issues (:func:`measure_feedback`), become the next round's documents.

:func:`ladder_measurer` composes the same stages for a set measured under a decision it is
given, which the nudge ladder (:mod:`~exp.optimize.evaluation.variance.nudge_ladder`) needs
because its nudge values are declared only once the author has proposed them.

The search stops at the first admitted round or after ``maximum_rounds``. Every round is saved as
it completes, so an interrupted search resumes, and the finished search is saved as a
``variant-search`` artifact that replays without model calls or measurements.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import NamedTuple

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ToolCall
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import ToolSchema
from exp.common.tasks.facts import InstanceFacts
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.authors import (
    SCENARIO_PROMPT,
    SCENARIO_TOOL,
    review_scenarios,
    target_addendum,
)
from exp.optimize.evaluation.variance.contracts import (
    ActorView,
    DecisionState,
    Scenario,
    ScenarioLimits,
    ScenarioProblem,
    ScenarioSet,
    StatePart,
)
from exp.optimize.evaluation.variance.decision_measure import (
    DecisionMeasure,
    DecisionMeasureSpec,
    JointVerdict,
    load_decision_measure,
    measure_family,
)
from exp.optimize.evaluation.variance.evidence_tools import EvidenceToolProfile
from exp.optimize.evaluation.variance.family_stages import (
    MEASURE_PROFILE,
    failure_reason,
    run_role,
    run_variant,
)
from exp.optimize.evaluation.variance.judged import EpisodeJudge

VARIANT_SEARCH_PROTOCOL = "variant-search-v2"
"""Version of the search author's prompts, the round loop and its feedback."""

SEARCH_TYPE = "variant-search"
"""Artifact type of a saved :class:`VariantSearch`."""

ROUND_TYPE = "variant-search-round"
"""Artifact type of one saved :class:`SearchRound`."""

SEARCH_ADDENDUM = """

You are revising a variant set across measured rounds. The documents hold the decision state, the
world state before the decision (read it with read_world_state at point 0), the actor's limits,
what the actor had seen, and the measured results of every earlier round: per state, how free
agents chose, whether forcing different options changed judged success, and the clarity
reviewers' issues, with every deficit. A set is accepted only when all three hold:
(a) choices change across states: the deciding facts differ so that a careful agent chooses
    differently in different states;
(b) choices change success: in each state, forcing the right option succeeds clearly more often
    than forcing the wrong one;
(c) every state is clear: one option is clearly best, its deciding facts are observable within
    the remaining calls, and the task is solvable without guessing.
Never maximize uncertainty: a state where agents split because facts are hidden, ambiguous or
contradictory is refused. Fix every listed deficit and keep what already worked."""
"""Search contract appended to the scenario author's prompt."""

SEARCH_STANCES: dict[str, str] = {
    "contrast_first": """Start from the options. For each option, find the smallest set of
deciding facts that makes it clearly best, then build states that differ exactly in those facts
and nothing else.""",
    "observability_first": """Start from what the agent can observe. Keep only deciding paths
that a permitted witness call reveals plainly within the remaining calls, and make the right
option follow from them without an inference leap.""",
    "deficit_repair": """Start from the measured deficits of the earlier rounds. Change only what
each deficit requires: make an unclear state's deciding fact explicit, separate options whose
success did not differ, or move states apart where the choices did not change.""",
}
"""Author stances, rotated over rounds."""

Measurer = Callable[[ScenarioSet, int], tuple[ArtifactInput, DecisionMeasure] | str]
"""Measures one structurally sound set in one round: the saved measure, or why it could not."""


class BoundVariant(NamedTuple):
    """What one scenario's witness, contrasts and free continuations run under.

    Attributes:
        job: The frozen job, already rebound to the frozen runtime when a freeze is bound.
        judge: The scenario's frozen episode judge.
        frozen: The world freeze's saved start, or None when no freeze is bound.
    """

    job: ContinuationJob
    judge: EpisodeJudge
    frozen: ArtifactInput | None = None


Binder = Callable[[DecisionState, Sequence[Scenario]], Mapping[str, BoundVariant]]
"""Binds every scenario of one set, under the decision that declares it, before any contrast.

A binder sees the whole set at once, so a caller can author one frozen world, rubric and judge for
exactly that set; it raises ``ValueError`` or ``KeyError`` when a scenario cannot be bound."""

LadderMeasurer = Callable[[DecisionState, ScenarioSet], tuple[ArtifactInput, DecisionMeasure] | str]
"""Measures a set under the given decision: the saved measure, or why it could not."""


def search_prompts_sha256() -> str:
    """Return the identity of the search author's prompt, addendum and stances."""
    return sha256_json(
        {
            "protocol": VARIANT_SEARCH_PROTOCOL,
            "prompt": SCENARIO_PROMPT,
            "addendum": SEARCH_ADDENDUM,
            "stances": SEARCH_STANCES,
        }
    )


class VariantSearchSpec(ContractModel):
    """Frozen settings of one variant search.

    Attributes:
        author: Scenario author model and per-call allowance (an open-weight alias).
        profile: Evidence-tool profile and budget per author draw.
        stances: Author stances, rotated over rounds in this order.
        maximum_rounds: Most author rounds before the search stops unaccepted.
    """

    author: AuthorSettings
    profile: EvidenceToolProfile = MEASURE_PROFILE
    stances: tuple[str, ...] = tuple(SEARCH_STANCES)
    maximum_rounds: int = Field(default=3, ge=1, le=8, strict=True)

    @model_validator(mode="after")
    def _stances(self) -> VariantSearchSpec:
        """Refuse unknown or repeated stances."""
        given = self.stances
        if not given or len(set(given)) != len(given) or set(given) - set(SEARCH_STANCES):
            raise ValueError(f"stances must be distinct members of {sorted(SEARCH_STANCES)}")
        return self


class SearchRound(ContractModel):
    """One author round and its measurement.

    Attributes:
        round: Round number, from 0.
        stance: Author stance of the round.
        transcript: The author draw's saved transcript.
        scenarios: The authored set, when the answer parsed.
        problems: Structural problems of the set; a set with any is not measured.
        failure: Why the round produced no measured set, when it did not.
        measure: The saved decision measure, when one ran.
        verdict: The measure's joint verdict, when one ran.
        deficits: Everything the next round must fix.
        accepted: Whether the measured set was admitted on all three criteria.
    """

    round: int = Field(ge=0)
    stance: str
    transcript: ArtifactInput
    scenarios: ScenarioSet | None = None
    problems: tuple[ScenarioProblem, ...] = ()
    failure: str | None = Field(default=None, max_length=2000)
    measure: ArtifactInput | None = None
    verdict: JointVerdict | None = None
    deficits: tuple[str, ...] = ()
    accepted: bool = False


class VariantSearch(ContractModel):
    """The complete record of one variant search, every round kept.

    Attributes:
        protocol: Search protocol.
        identity: Identity naming the saved search.
        rounds: Every round, in order.
        accepted_round: The admitted round, or None when no round was admitted.
        scenarios: The admitted set, when one was.
        measure: The admitted set's saved decision measure, when one was.
    """

    protocol: str = VARIANT_SEARCH_PROTOCOL
    identity: str
    rounds: tuple[SearchRound, ...]
    accepted_round: int | None = None
    scenarios: ScenarioSet | None = None
    measure: ArtifactInput | None = None


def measure_feedback(measure: DecisionMeasure) -> JsonObject:
    """Render the measured results an author needs to revise its variants, per state.

    Args:
        measure: A saved decision measure.

    Returns:
        Per-state choice counts, consequence, clarity issues and problems, the choice shifts,
        and every deficit.
    """
    rows: list[JsonValue] = []
    for state, verdict in zip(measure.states, measure.verdict.states, strict=True):
        rows.append(
            {
                "state_id": state.state_id,
                "choice_counts": dict(verdict.counts),
                "unclear": verdict.unclear,
                "entropy": verdict.entropy,
                "consequence": None
                if verdict.consequence is None
                else verdict.consequence.model_dump(mode="json"),
                "clarity_issues": list(state.evidence.clarity_notes),
                "problems": list(verdict.problems),
            }
        )
    return {
        "admitted": measure.verdict.admitted,
        "states": rows,
        "shifts": [shift.model_dump(mode="json") for shift in measure.verdict.shifts],
        "deficits": list(measure.verdict.deficits),
    }


def _feedback(row: SearchRound, measure: DecisionMeasure | None) -> JsonObject:
    """Render one finished round for the next round's documents."""
    return {
        "round": row.round,
        "stance": row.stance,
        "scenarios": None if row.scenarios is None else row.scenarios.model_dump(mode="json"),
        "structural_problems": [problem.model_dump(mode="json") for problem in row.problems],
        "failure": row.failure,
        "measured": None if measure is None else measure_feedback(measure),
        "deficits": list(row.deficits),
    }


def _check(
    draft: JsonObject,
    decision: DecisionState,
    before: JsonObject,
    limits: ScenarioLimits,
    view: ActorView | None,
) -> str | None:
    """Return the existing scenario checks' problems with a draft set, or None when it is sound."""
    try:
        _, problems = review_scenarios(draft, decision, before, limits, view)
    except ValueError as error:
        return failure_reason(error)
    return "; ".join(f"{p.scenario_id or 'set'}: {p.reason}" for p in problems)[:4000] or None


def _round(
    project: ProjectStore,
    spec: VariantSearchSpec,
    decision: DecisionState,
    before: JsonObject,
    limits: ScenarioLimits,
    view: ActorView | None,
    earlier: list[JsonObject],
    number: int,
    measure: Measurer,
    sources: tuple[ArtifactInput, ...],
    code_revision: str,
) -> tuple[SearchRound, DecisionMeasure | None]:
    """Author one round over the earlier rounds' feedback and measure a sound set."""
    stance = spec.stances[number % len(spec.stances)]
    documents: dict[str, JsonValue] = {
        "decision": decision.model_dump(mode="json"),
        "world:0": before,
        "limits": {"tools": sorted(limits.tools), "remaining_calls": limits.remaining_calls},
        "actor_view": "not supplied" if view is None else view.model_dump(mode="json"),
        **{f"round:{row['round']}": row for row in earlier},
    }
    overview: JsonObject = {
        "decision": documents["decision"],
        "limits": documents["limits"],
        "actor_view": documents["actor_view"],
        "before_state_keys": sorted(before)[:64],
        "earlier_rounds": [{"round": r["round"], "deficits": r["deficits"]} for r in earlier],
    }
    answer = ToolSchema(
        name=SCENARIO_TOOL,
        description="Submit the scenario set.",
        input_schema=ScenarioSet.model_json_schema(),
    )
    reference, transcript = run_role(
        project,
        spec.author,
        spec.profile,
        role="search",
        instruction=SCENARIO_PROMPT + SEARCH_ADDENDUM + target_addendum("scenario"),
        stance=(stance, SEARCH_STANCES[stance]),
        documents=documents,
        overview=overview,
        answer=answer,
        sources=sources,
        code_revision=code_revision,
        check=lambda draft: _check(draft, decision, before, limits, view),
    )
    row = SearchRound(round=number, stance=stance, transcript=reference)
    if transcript.stop != "answered" or transcript.answer is None:
        reason = f"author: {transcript.failure or transcript.stop}"
        return row.model_copy(update={"failure": reason, "deficits": (reason,)}), None
    try:
        scenarios, problems = review_scenarios(transcript.answer, decision, before, limits, view)
    except ValueError as error:
        reason = f"author: {failure_reason(error)}"
        return row.model_copy(update={"failure": reason, "deficits": (reason,)}), None
    row = row.model_copy(update={"scenarios": scenarios, "problems": problems})
    if problems:
        deficits = tuple(f"structural: {p.scenario_id or 'set'}: {p.reason}" for p in problems)
        return row.model_copy(update={"deficits": deficits}), None
    measured = measure(scenarios, number)
    if isinstance(measured, str):
        reason = f"measure: {measured}"
        return row.model_copy(update={"failure": reason, "deficits": (reason,)}), None
    saved, result = measured
    verdict = result.verdict
    return (
        row.model_copy(
            update={
                "measure": saved,
                "verdict": verdict,
                "deficits": verdict.deficits,
                "accepted": verdict.admitted,
            }
        ),
        result,
    )


def search_variants(
    project: ProjectStore,
    spec: VariantSearchSpec,
    decision: DecisionState,
    before: JsonObject,
    limits: ScenarioLimits,
    *,
    measure: Measurer,
    identity: str,
    code_revision: str,
    view: ActorView | None = None,
    sources: tuple[ArtifactInput, ...] = (),
) -> tuple[ArtifactInput, VariantSearch]:
    """Run, resume or replay one bounded variant search; accept only an admitted set.

    Args:
        project: Project holding every round, draw and measure.
        spec: The frozen search settings.
        decision: The decision state to vary.
        before: World state just before the decision's step.
        limits: Tools and remaining calls available to the actor.
        measure: Measures one structurally sound set; never called for a saved round.
        identity: Stable identity of this search; its rounds derive from it.
        code_revision: Producing source identity.
        view: What the actor had seen, needed to accept a fresh briefing.
        sources: Saved artifacts the decision state and before-state came from.

    Returns:
        The saved ``variant-search`` reference and the search, with every round.
    """
    store = project.artifacts
    result_id = stable_id(
        SEARCH_TYPE,
        {
            "identity": identity,
            "prompts": search_prompts_sha256(),
            "spec": spec.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
            "before": sha256_json(before),
            "limits": limits.model_dump(mode="json"),
            "view": None if view is None else view.model_dump(mode="json"),
        },
    )
    with CoordinationLock(store.project_directory / (result_id + ".lock"), timeout=0):
        if store.exists(result_id):
            saved = VariantSearch.model_validate_json(store.read_bytes(result_id, "search.json"))
            return artifact_input(store.read(result_id).manifest), saved
        rounds: list[SearchRound] = []
        earlier: list[JsonObject] = []
        accepted: DecisionMeasure | None = None
        for number in range(spec.maximum_rounds):
            round_id = f"{result_id}-round-{number}"
            if store.exists(round_id):
                row = SearchRound.model_validate_json(store.read_bytes(round_id, "round.json"))
                result = (
                    None if row.measure is None else load_decision_measure(project, row.measure)
                )
            else:
                row, result = _round(
                    project,
                    spec,
                    decision,
                    before,
                    limits,
                    view,
                    earlier,
                    number,
                    measure,
                    sources,
                    code_revision,
                )
                _save(project, round_id, ROUND_TYPE, row, _inputs(row), code_revision)
            rounds.append(row)
            earlier.append(_feedback(row, result))
            if row.accepted:
                accepted = result
                break
        final = rounds[-1]
        search = VariantSearch(
            identity=result_id,
            rounds=tuple(rounds),
            accepted_round=final.round if accepted is not None else None,
            scenarios=final.scenarios if accepted is not None else None,
            measure=final.measure if accepted is not None else None,
        )
        inputs = tuple(ref for row in rounds for ref in _inputs(row))
        return _save(project, result_id, SEARCH_TYPE, search, inputs, code_revision), search


def _inputs(row: SearchRound) -> tuple[ArtifactInput, ...]:
    """Return the saved artifacts one round rests on."""
    return (row.transcript, *(() if row.measure is None else (row.measure,)))


def _save(
    project: ProjectStore,
    artifact_id: str,
    artifact_type: str,
    record: SearchRound | VariantSearch,
    inputs: tuple[ArtifactInput, ...],
    code_revision: str,
) -> ArtifactInput:
    """Persist one round or the finished search beside what it rests on."""
    name = "round.json" if isinstance(record, SearchRound) else "search.json"
    manifest = project.artifacts.write_json(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(sorted(set(inputs), key=lambda ref: ref.artifact_id)),
        ),
        files={name: record},
    )
    return artifact_input(manifest)


def variant_measurer(
    project: ProjectStore,
    spec: DecisionMeasureSpec,
    *,
    decision: DecisionState,
    bind: Binder,
    base: ArtifactInput,
    boundaries: tuple[ArtifactInput, ...],
    actions: Mapping[str, ToolCall],
    facts: InstanceFacts | None,
    repeats: int,
    code_revision: str,
    required_state: frozenset[StatePart] = frozenset(),
) -> Measurer:
    """Compose the measurer that builds, witnesses and contrasts each variant, then measures it.

    Args:
        project: Project holding every saved step.
        spec: The frozen measurement settings.
        decision: The decision state being varied.
        bind: Binds every scenario of a set (frozen job, judge and start) before its contrasts.
        base: The decision's checkpoint in the picked trajectory.
        boundaries: Every saved boundary of the picked attempt, start first.
        actions: Concrete forced first action of every option.
        facts: Instance facts the world model was given, if any.
        repeats: Pairs per selection contrast.
        code_revision: Producing source identity.
        required_state: State parts every variant must hold beyond the baseline.

    Returns:
        A measurer for :func:`search_variants`.
    """
    measure = ladder_measurer(
        project,
        spec,
        bind=bind,
        base=base,
        boundaries=boundaries,
        actions=actions,
        facts=facts,
        repeats=repeats,
        code_revision=code_revision,
        required_state=required_state,
    )
    return lambda scenarios, number: measure(decision, scenarios)


def ladder_measurer(
    project: ProjectStore,
    spec: DecisionMeasureSpec,
    *,
    bind: Binder,
    base: ArtifactInput,
    boundaries: tuple[ArtifactInput, ...],
    actions: Mapping[str, ToolCall],
    facts: InstanceFacts | None,
    repeats: int,
    code_revision: str,
    required_state: frozenset[StatePart] = frozenset(),
) -> LadderMeasurer:
    """Compose the measurer of a set under the decision it is given, as a nudge ladder needs.

    Each scenario becomes a variant whose witness and selection contrast run under the frozen
    job, judge and world-freeze start ``bind`` gives the whole set (one rubric for every rung of a
    ladder), and the contrasted variants are measured together by
    :func:`~exp.optimize.evaluation.variance.decision_measure.measure_family`. A set ``bind``
    refuses is reported as the reason it could not be measured.

    Args:
        project: Project holding every saved step.
        spec: The frozen measurement settings.
        bind: Binds every scenario of a set (frozen job, judge and start) before its contrasts.
        base: The decision's checkpoint in the picked trajectory.
        boundaries: Every saved boundary of the picked attempt, start first.
        actions: Concrete forced first action of every option.
        facts: Instance facts the world model was given, if any.
        repeats: Pairs per selection contrast.
        code_revision: Producing source identity.
        required_state: State parts every variant must hold beyond the baseline.

    Returns:
        A measurer for :func:`build_nudge_ladder`.
    """

    def measure(
        decision: DecisionState, scenarios: ScenarioSet
    ) -> tuple[ArtifactInput, DecisionMeasure] | str:
        """Run every family stage up to selection for one set and measure it."""
        try:
            bound = dict(bind(decision, scenarios.scenarios))
            missing = sorted({s.scenario_id for s in scenarios.scenarios} - set(bound))
            if missing:
                raise KeyError(f"no binding for scenarios {missing}")
        except (ValueError, KeyError) as error:
            return f"binding: {failure_reason(error)}"
        # Every variant's witness and contrasts start from its own frozen start: run them at once.
        def one(scenario: Scenario) -> object:
            return run_variant(
                project,
                job=bound[scenario.scenario_id].job,
                base=base,
                boundaries=boundaries,
                decision=decision,
                scenario=scenario,
                actions=actions,
                facts=facts,
                repeats=repeats,
                judge=bound[scenario.scenario_id].judge,
                code_revision=code_revision,
                required_state=required_state,
                frozen=bound[scenario.scenario_id].frozen,
            )

        from concurrent.futures import ThreadPoolExecutor
        from contextvars import copy_context

        with ThreadPoolExecutor(max(1, len(scenarios.scenarios))) as pool:
            futures = [pool.submit(copy_context().run, one, s) for s in scenarios.scenarios]
            variants = [future.result() for future in futures]
        reference, result, why = measure_family(
            project,
            spec,
            decision=decision,
            scenarios=scenarios.scenarios,
            variants=variants,
            jobs={sid: row.job for sid, row in bound.items()},
            judges={sid: row.judge for sid, row in bound.items()},
            code_revision=code_revision,
        )
        if reference is None or result is None:
            stops = "; ".join(
                f"{row.scenario_id}: {row.stopped}" for row in variants if row.stopped
            )
            return (why or "no measure") + (f" [variants stopped: {stops}]" if stops else "")
        return reference, result

    return measure
