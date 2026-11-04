"""The nudge ladder: one decision under instruction nudges of fading strength, ending un-nudged.

Where the frozen actor nearly always fails a decision, every attempt in a training group fails and
the group carries no gradient. A ladder keeps the decision's facts, obligations, correct option,
judge and rubric and only changes what the actor is told
(:mod:`~exp.optimize.evaluation.variance.nudges`): the author proposes nudges from strong to weak,
each is leak-reviewed and realism-reviewed under several prompt variants, and the base decision
and every surviving rung are measured together from judged episodes
(:mod:`~exp.optimize.evaluation.variance.decision_measure`, composed by
:func:`~exp.optimize.evaluation.variance.variant_search.ladder_measurer`).

A nudged rung is admitted as its own task (:func:`rung_verdict`) only when the nudge measurably
shifts the classified choice from the un-nudged decision (a choice shift that holds), the choice
still changes judged success in that version (consequence), the version is clear (a separately
best option matching the declared one and a passing LLM clarity review), and its own classified
free choices keep real variance, so its group has a gradient. Every rung records its strength
and ladder position. The last rung is always the un-nudged decision, the ladder's target;
:attr:`NudgeLadder.curriculum` lists the admitted rungs from strong to weak and then that target;
:func:`nudge_curriculum` orders the rungs of many ladders for training (strength first, the
un-nudged targets last, or only those targets for the comparison arm) and :func:`curriculum_tier`
gives the tier a training run has reached. Both are re-exported by
:mod:`~exp.optimize.evaluation.variance.schedule`.
Rounds rotate author stances and feed every refusal and measured result back, stopping at the
first round that admits a nudged rung. The ladder is saved as a ``nudge-ladder`` artifact and
replays without calls or measurements.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.briefing_review import (
    DEFAULT_NUDGE_VARIANTS,
    NUDGE_REVIEW_VARIANTS,
)
from exp.optimize.evaluation.variance.contracts import (
    ActorView,
    DecisionState,
    Scenario,
    ScenarioSet,
)
from exp.optimize.evaluation.variance.decision_measure import DecisionMeasure, StateVerdict
from exp.optimize.evaluation.variance.evidence_tools import EvidenceToolProfile
from exp.optimize.evaluation.variance.family_stages import MEASURE_PROFILE, failure_reason
from exp.optimize.evaluation.variance.nudges import (
    NUDGE_AUTHOR_STANCES,
    REALISM_STANCES,
    STRENGTH_RANK,
    NudgeReview,
    NudgeStrength,
    ProposedNudge,
    RungStrength,
    ladder_decision,
    nudge_prompts_sha256,
    propose_nudges,
    review_nudge,
    rung_scenario,
)
from exp.optimize.evaluation.variance.rewards import ChoiceShift
from exp.optimize.evaluation.variance.variant_search import LadderMeasurer

NUDGE_LADDER_PROTOCOL = "nudge-ladder-v1"
"""Version of the ladder's rounds, rung admission and curriculum."""

LADDER_TYPE = "nudge-ladder"
"""Artifact type of a saved :class:`NudgeLadder`."""


class NudgeLadderSpec(ContractModel):
    """Frozen settings of one nudge ladder.

    Attributes:
        author: Nudge author model and allowance (an open-weight alias).
        leak_reviewer: Nudge leak reviewer model and allowance (an open-weight alias).
        realism_reviewer: Realism reviewer model and allowance (an open-weight alias).
        profile: Evidence-tool profile of each author and realism draw.
        strengths: Nudge strengths, strongest first; the un-nudged rung always follows.
        author_stances: Author stances, rotated over rounds.
        leak_variants: Nudge reading stances of the leak reviewer.
        realism_stances: Realism reviewer stances, one draw per nudge each.
        minimum_realism: Fewest answered realism reviews a nudge needs; all must pass.
        maximum_rounds: Most author rounds; the ladder stops at the first round admitting a rung.
        minimum_minority: Smallest share of a nudged rung's classified free choices outside its
            most frequent choice, so its group has real choice variance.
    """

    author: AuthorSettings
    leak_reviewer: AuthorSettings
    realism_reviewer: AuthorSettings
    profile: EvidenceToolProfile = MEASURE_PROFILE
    strengths: tuple[NudgeStrength, ...] = ("strong", "medium", "weak")
    author_stances: tuple[str, ...] = tuple(NUDGE_AUTHOR_STANCES)
    leak_variants: tuple[str, ...] = DEFAULT_NUDGE_VARIANTS
    realism_stances: tuple[str, ...] = tuple(REALISM_STANCES)
    minimum_realism: int = Field(default=2, ge=1, le=8, strict=True)
    maximum_rounds: int = Field(default=2, ge=1, le=8, strict=True)
    minimum_minority: float = Field(default=0.125, gt=0.0, le=0.5)

    @model_validator(mode="after")
    def _ordered(self) -> NudgeLadderSpec:
        """Require strongest-first strengths, known distinct stances and a reachable quorum."""
        ranks = [STRENGTH_RANK[strength] for strength in self.strengths]
        if not ranks or ranks != sorted(set(ranks)):
            raise ValueError("strengths must be distinct and ordered from strong to weak")
        for given, known in (
            (self.author_stances, NUDGE_AUTHOR_STANCES),
            (self.leak_variants, NUDGE_REVIEW_VARIANTS),
            (self.realism_stances, REALISM_STANCES),
        ):
            if not given or len(set(given)) != len(given) or set(given) - set(known):
                raise ValueError(f"stances must be distinct members of {sorted(known)}")
        if self.minimum_realism > len(self.realism_stances):
            raise ValueError("minimum_realism exceeds the number of realism stances")
        return self


class NudgeRung(ContractModel):
    """One rung of a nudge ladder: one version of the decision, admitted as its own task or not.

    Attributes:
        position: Ladder position, from 0 (strongest); the un-nudged rung is last.
        strength: The nudge strength, ``none`` for the un-nudged decision.
        scenario: The rung's scenario: the base world and answer, with its nudge.
        nudge: The authored nudge, None for the un-nudged rung.
        review: The nudge's leak and realism reviews, None for the un-nudged rung.
        shift: Choice shift between this rung's and the un-nudged rung's free choices.
        state: The rung's measured state verdict (choices, consequence, clarity).
        minority: Share of classified free choices outside the most frequent one.
        admitted: Whether the rung becomes its own task. A nudged rung must pass both reviews,
            shift the choice, keep a consequential choice, stay clear and keep real choice
            variance; the un-nudged rung must be clear with a consequential choice.
        problems: Every reason it falls short.
    """

    position: int = Field(ge=0)
    strength: RungStrength
    scenario: Scenario
    nudge: ProposedNudge | None = None
    review: NudgeReview | None = None
    shift: ChoiceShift | None = None
    state: StateVerdict | None = None
    minority: float | None = None
    admitted: bool = False
    problems: tuple[str, ...] = ()


class LadderRound(ContractModel):
    """One author round of a nudge ladder.

    Attributes:
        round: Round number, from 0.
        stance: Author stance of the round.
        transcript: The author draw's saved transcript.
        failure: Why the round produced no rungs, when it did not.
        decision: The decision declaring the round's nudge values, when built.
        measure: The saved measure of the base and every reviewed rung, when one ran.
        rungs: Every rung, strongest first, the un-nudged rung last.
    """

    round: int = Field(ge=0)
    stance: str
    transcript: ArtifactInput
    failure: str | None = Field(default=None, max_length=2000)
    decision: DecisionState | None = None
    measure: ArtifactInput | None = None
    rungs: tuple[NudgeRung, ...] = ()


class NudgeLadder(ContractModel):
    """A saved nudge ladder over one decision, every round kept.

    Attributes:
        protocol: Ladder protocol.
        identity: Identity naming the saved ladder.
        base: Scenario identity of the un-nudged decision, the ladder's target.
        rounds: Every author round, in order.
        chosen_round: The round whose rungs are final: the first admitting a nudged rung, else the
            last round that produced rungs; None when none did.
        rungs: The final rungs, strongest first; the last is always the un-nudged decision.
    """

    protocol: str = NUDGE_LADDER_PROTOCOL
    identity: str
    base: str
    rounds: tuple[LadderRound, ...]
    chosen_round: int | None = None
    rungs: tuple[NudgeRung, ...]

    @model_validator(mode="after")
    def _ends_unnudged(self) -> NudgeLadder:
        """Require the last rung to be the un-nudged decision and ranks to fade."""
        if not self.rungs or self.rungs[-1].strength != "none" or self.rungs[-1].nudge:
            raise ValueError("a nudge ladder always ends with the un-nudged decision")
        ranks = [STRENGTH_RANK[rung.strength] for rung in self.rungs]
        if ranks != sorted(set(ranks)):
            raise ValueError("ladder rungs fade from the strongest nudge to none")
        return self

    @property
    def curriculum(self) -> tuple[NudgeRung, ...]:
        """Admitted nudged rungs from strong to weak, then always the un-nudged target."""
        return (*(rung for rung in self.rungs[:-1] if rung.admitted), self.rungs[-1])


def rung_verdict(
    rung: NudgeRung, base: str, measure: DecisionMeasure, minimum_minority: float
) -> NudgeRung:
    """Admit or refuse one rung from the joint measure of the base and the reviewed rungs.

    Args:
        rung: The rung, with its reviews when nudged.
        base: Scenario identity of the un-nudged rung.
        measure: The saved measure over the base and the rungs.
        minimum_minority: Smallest minority share of a nudged rung's classified choices.

    Returns:
        The rung with its shift, state verdict, minority share, admission and problems.
    """
    sid = rung.scenario.scenario_id
    states = {state.state_id: state for state in measure.verdict.states}
    state = states.get(sid)
    if state is None:
        return rung.model_copy(update={"problems": (*rung.problems, "the rung was not measured")})
    problems = list(rung.problems)
    if state.consequence is None or not state.consequence.holds:
        problems.append("the choice does not change judged success in this version")
    if not state.clear:
        problems.append("not clear: " + "; ".join(state.problems)[:800])
    total = sum(state.counts.values())
    minority = None if not total else 1.0 - max(state.counts.values()) / total
    shift = next((s for s in measure.verdict.shifts if set(s.states) == {sid, base}), None)
    if rung.nudge is not None:
        if shift is None or not shift.holds:
            problems.append(
                "the nudge does not shift the choice from the un-nudged decision"
                + ("" if shift is None else f" (divergence lower bound {shift.interval[0]:.2f})")
            )
        if minority is None or minority < minimum_minority:
            problems.append(
                f"no real choice variance: minority share {minority or 0.0:.2f}, below "
                f"{minimum_minority:.2f}"
            )
    return rung.model_copy(
        update={
            "shift": shift,
            "state": state,
            "minority": minority,
            "admitted": not problems,
            "problems": tuple(problems),
        }
    )


def _ladder_round(
    project: ProjectStore,
    spec: NudgeLadderSpec,
    subject: tuple[DecisionState, Scenario, str, ActorView, JsonObject],
    number: int,
    feedback: list[JsonObject],
    measure: LadderMeasurer,
    identity: str,
    sources: tuple[ArtifactInput, ...],
    code_revision: str,
) -> LadderRound:
    """Author, review and measure one round of rungs."""
    decision, base, instruction, view, before = subject
    stance = spec.author_stances[number % len(spec.author_stances)]
    transcript, proposal, failure = propose_nudges(
        project,
        spec.author,
        spec.profile,
        decision,
        base,
        view=view,
        before=before,
        strengths=spec.strengths,
        stance=stance,
        feedback=feedback,
        sources=sources,
        code_revision=code_revision,
    )
    row = LadderRound(round=number, stance=stance, transcript=transcript, failure=failure)
    if proposal is None:
        return row
    try:
        laddered = ladder_decision(decision, base, instruction, [n.text for n in proposal.nudges])
    except ValueError as error:
        return row.model_copy(update={"failure": f"nudge values: {failure_reason(error)}"})
    axis = laddered.instruction_axis
    assert axis is not None
    rungs: list[NudgeRung] = []
    for position, nudge in enumerate(proposal.nudges):
        value = next(v for v in axis.values if v.value_id == f"nudge-{position}")
        scenario = rung_scenario(base, value, f"{base.scenario_id[:40]}-nudge-{position}")
        review = review_nudge(
            project,
            (spec.leak_reviewer, spec.realism_reviewer),
            spec.profile,
            (laddered, scenario, nudge),
            view=view,
            before=before,
            identity=f"{identity}-round-{number}-nudge-{position}",
            leak_variants=spec.leak_variants,
            realism_stances=spec.realism_stances,
            minimum_realism=spec.minimum_realism,
            sources=(*sources, transcript),
            code_revision=code_revision,
        )
        rungs.append(
            NudgeRung(
                position=position,
                strength=nudge.strength,
                scenario=scenario,
                nudge=nudge,
                review=review,
                problems=review.problems,
            )
        )
    rungs.append(NudgeRung(position=len(rungs), strength="none", scenario=base))
    row = row.model_copy(update={"decision": laddered})
    reviewed = [rung for rung in rungs[:-1] if rung.review is not None and rung.review.passed]
    if not reviewed:
        target = rungs[-1].model_copy(update={"problems": ("no nudge passed both reviews",)})
        return row.model_copy(update={"rungs": (*rungs[:-1], target)})
    measured = measure(
        laddered,
        ScenarioSet(
            scenarios=(base, *(rung.scenario for rung in reviewed)),
            limitations=("versions of one decision that differ only by an instruction nudge",),
        ),
    )
    if isinstance(measured, str):
        rungs = [
            rung.model_copy(update={"problems": (*rung.problems, f"measure: {measured}")})
            for rung in rungs
        ]
        return row.model_copy(update={"rungs": tuple(rungs)})
    saved, result = measured
    final = tuple(
        rung_verdict(rung, base.scenario_id, result, spec.minimum_minority)
        if rung in reviewed or rung.nudge is None
        else rung
        for rung in rungs
    )
    return row.model_copy(update={"measure": saved, "rungs": final})


def _ladder_feedback(row: LadderRound) -> JsonObject:
    """Render one finished round for the next round's author."""
    return {
        "round": row.round,
        "stance": row.stance,
        "failure": row.failure,
        "rungs": [
            {
                "strength": rung.strength,
                "nudge": None if rung.nudge is None else rung.nudge.text,
                "admitted": rung.admitted,
                "choice_counts": None if rung.state is None else dict(rung.state.counts),
                "problems": list(rung.problems),
            }
            for rung in row.rungs
        ],
    }


def build_nudge_ladder(
    project: ProjectStore,
    spec: NudgeLadderSpec,
    decision: DecisionState,
    base: Scenario,
    *,
    instruction: str,
    view: ActorView,
    before: JsonObject,
    measure: LadderMeasurer,
    identity: str,
    code_revision: str,
    sources: tuple[ArtifactInput, ...] = (),
) -> tuple[ArtifactInput, NudgeLadder]:
    """Run or replay one nudge ladder over one decision; its last rung is always un-nudged.

    Each round, the author proposes one nudge per strength; every nudge is leak-reviewed and
    realism-reviewed; the base and every nudge that passed both are measured together; and each
    rung is admitted by :func:`rung_verdict`. The next round sees every refusal and measured
    result. Every draw and measure is saved by its owner, so an interrupted ladder resumes without
    buying anything twice, and the finished ladder replays without calls or measurements.

    Args:
        project: Project holding every draw, measure and the ladder.
        spec: The frozen ladder settings.
        decision: The decision state to nudge.
        base: The un-nudged scenario: the world and correct option every rung keeps.
        instruction: The task's own instruction.
        view: What the actor had seen at the decision.
        before: World state just before the decision, before the base scenario's edits.
        measure: Measures the base and the reviewed rungs under the ladder's decision.
        identity: Stable identity of this ladder; its draws derive from it.
        code_revision: Producing source identity.
        sources: Saved artifacts the decision and base scenario came from.

    Returns:
        The saved ``nudge-ladder`` reference and the ladder.
    """
    store = project.artifacts
    result_id = stable_id(
        LADDER_TYPE,
        {
            "identity": identity,
            "protocol": NUDGE_LADDER_PROTOCOL,
            "prompts": nudge_prompts_sha256(),
            "spec": spec.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
            "base": base.model_dump(mode="json"),
            "instruction": instruction,
            "view": view.model_dump(mode="json"),
            "before": sha256_json(before),
        },
    )
    with CoordinationLock(store.project_directory / (result_id + ".lock"), timeout=0):
        if store.exists(result_id):
            saved = NudgeLadder.model_validate_json(store.read_bytes(result_id, "ladder.json"))
            return artifact_input(store.read(result_id).manifest), saved
        rounds: list[LadderRound] = []
        feedback: list[JsonObject] = []
        for number in range(spec.maximum_rounds):
            row = _ladder_round(
                project,
                spec,
                (decision, base, instruction, view, before),
                number,
                feedback,
                measure,
                result_id,
                sources,
                code_revision,
            )
            rounds.append(row)
            feedback.append(_ladder_feedback(row))
            if any(rung.admitted for rung in row.rungs[:-1]):
                break
        built = [row for row in rounds if row.rungs]
        chosen = next(
            (row for row in built if any(r.admitted for r in row.rungs[:-1])),
            built[-1] if built else None,
        )
        ladder = NudgeLadder(
            identity=result_id,
            base=base.scenario_id,
            rounds=tuple(rounds),
            chosen_round=None if chosen is None else chosen.round,
            rungs=chosen.rungs
            if chosen is not None
            else (
                NudgeRung(
                    position=0,
                    strength="none",
                    scenario=base,
                    problems=("no round produced nudges to measure against",),
                ),
            ),
        )
        inputs = {
            ref
            for row in rounds
            for ref in (
                row.transcript,
                *(() if row.measure is None else (row.measure,)),
                *(
                    ref
                    for rung in row.rungs
                    if rung.review is not None
                    for ref in (
                        *(r.response for r in rung.review.leak if r.response is not None),
                        *(r.transcript for r in rung.review.realism),
                    )
                ),
            )
        }
        manifest = store.write_json(
            artifact_id=result_id,
            artifact_type=LADDER_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(sorted(inputs, key=lambda ref: ref.artifact_id)),
            ),
            files={"ladder.json": ladder},
        )
        return artifact_input(manifest), ladder


class CurriculumStage(ContractModel):
    """One admitted nudge-ladder rung, as a training stage.

    Attributes:
        ladder: Identity of the saved ladder.
        scenario_id: The rung's scenario, a task of its own.
        position: The rung's ladder position, from 0 (strongest).
        strength: Its nudge strength; ``none`` is the un-nudged target.
    """

    ladder: str
    scenario_id: str
    position: int = Field(ge=0)
    strength: RungStrength

    @property
    def nudged(self) -> bool:
        """Whether the stage carries a nudge."""
        return self.strength != "none"


def nudge_curriculum(
    ladders: Sequence[NudgeLadder], *, nudged: bool = True
) -> tuple[CurriculumStage, ...]:
    """Order the admitted rungs of nudge ladders from strong to weak, un-nudged targets last.

    Args:
        ladders: Saved nudge ladders, in the order their decisions should be trained.
        nudged: False keeps only the un-nudged targets, the comparison arm without nudges.

    Returns:
        Every admitted nudged rung by strength, then ladder order and position, followed by
        every ladder's un-nudged target.
    """
    stages = [
        (STRENGTH_RANK[rung.strength], index, rung.position, ladder.identity, rung)
        for index, ladder in enumerate(ladders)
        for rung in ladder.curriculum
        if nudged or rung.strength == "none"
    ]
    return tuple(
        CurriculumStage(
            ladder=identity,
            scenario_id=rung.scenario.scenario_id,
            position=rung.position,
            strength=rung.strength,
        )
        for _, _, _, identity, rung in sorted(stages, key=lambda row: row[:3])
    )


def curriculum_tier(
    stages: Sequence[CurriculumStage], completed: int, per_tier: int
) -> tuple[CurriculumStage, ...]:
    """Return the stages of the strength tier a number of completed rollouts has reached.

    Tiers run from the strongest nudge present to the un-nudged targets, each for ``per_tier``
    rollouts; once every nudged tier is spent the un-nudged tier stays current.

    Args:
        stages: An ordered curriculum from :func:`nudge_curriculum`.
        completed: Rollouts completed so far.
        per_tier: Rollouts per nudged strength tier.

    Returns:
        The current tier's stages, in curriculum order; empty for an empty curriculum.

    Raises:
        ValueError: ``completed`` is negative or ``per_tier`` is not positive.
    """
    if completed < 0 or per_tier < 1:
        raise ValueError("completed must be non-negative and per_tier positive")
    tiers = sorted({STRENGTH_RANK[stage.strength] for stage in stages})
    if not tiers:
        return ()
    current = tiers[min(completed // per_tier, len(tiers) - 1)]
    return tuple(stage for stage in stages if STRENGTH_RANK[stage.strength] == current)
