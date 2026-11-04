"""Audit dispositions, computed training eligibility and the records an eligible family yields.

A decision family may be trained on only after every condition below holds, and each is computed
from saved evidence rather than asserted by a caller:

- **family_flip**: the family qualified on its selection contrasts
  (:func:`~exp.optimize.evaluation.variance.contrast.family_flip`), whose cells are labelled only
  by each variant's frozen episode judge.
- **decision_variance**: when the registration requires it, the joint decision-variance measure
  (:mod:`~exp.optimize.evaluation.variance.decision_measure`) admitted the family: choices shift
  across states, the choice changes success and every state is clear. A family it blocks names
  this condition, and its confirmation is reported as not run because of it.
- **fresh_confirmation**: every variant that qualified on selection was contrasted again with
  independently drawn samples that took no part in selection, labelled by the same frozen judge,
  and the confirmation reproduced its separated winner.
- **audit_dispositions**: every audit finding (actor-critic flags, decision-state flags, scenario
  problems, variant checks marked for review) has a recorded disposition.
- **critic_scores**: no critic draw whose estimates were used is unscored.
- **tool_coverage**: the family's task declares no tool the learned rule book lacks a contract for.
- **rule_witnesses**: no rule the family relies on was contradicted without a witness, or lacks
  a supporting witness.
- **policy_consistency**: every estimate the records rest on shares one actor policy, harness,
  budget and sampling setting; only where it started (the checkpoint, the calls remaining there
  and the runtime world, which differs between frozen authored variants) and the judge that
  labels each variant differ. Each variant's world is checked by **world_provenance** instead.
- **judge_coverage**, **rubric_problems**, **rubric_compatibility**, **world_freeze**,
  **world_provenance**: every confirmed variant meets the family's frozen admission policy
  (:mod:`~exp.optimize.evaluation.variance.admission`); each failing variant is named with what
  was found.

Failed conditions are all listed, never only the first. Records are built only for an eligible
family, from its judged confirmation episodes under each variant's frozen judge: one
:class:`ContinuationGroup` per confirmed variant and forced option, referencing every saved
assessment, with group-relative advantages read back from those assessments; the imposed first
action is experiment-owned and never trainable as actor tokens. Discovery critic records are kept
only when the error-mining run's episode judge is every confirmed variant's judge (otherwise
``critics_refused`` says why) and stay non-trainable: they were scored under the discovery run, not
the variant's fresh predictions. Point-selector screening scores are kept for analysis.

Point-selector training records (:func:`selector_records`) reference each saved selector draw
(request settings, final response, full analysis transcript) and every judged slot of the points
it chose. Each draw's reward is its picks' decision relevance under the run's frozen episode
judge, scored by :func:`~exp.optimize.evaluation.variance.rewards.score_selector` under version 2
by default (each pick's classified choices or family joint score from ``decisions``) or version 1
(judged outcome variance). The ``J`` draws of one run are one group for group-relative advantages.
A draw is trainable only when the run's judge coverage passes the admission policy, every judged
study was made by the expected selector judge (within a family: the error-mining judge is every
confirmed variant's judge) and the group has at least two rewarded draws.

No optimizer runs here: an external trainer consumes these records and registers its revision
(:mod:`~exp.optimize.evaluation.variance.schedule`). Fresh confirmation draws come from
:func:`confirmation_job`; :func:`confirmation_episodes` renders them for a calibration grader.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, Self, get_args

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.action_contrast import ActionContrast
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport, CriticDraw
from exp.optimize.evaluation.variance.admission import FamilyAdmissionPolicy, coverage_problems
from exp.optimize.evaluation.variance.authors import DecisionStateAnswer
from exp.optimize.evaluation.variance.contracts import JudgeSelection
from exp.optimize.evaluation.variance.contrast import FlipVerdict, OptionContrast
from exp.optimize.evaluation.variance.exposure import EstimateMeaning
from exp.optimize.evaluation.variance.grader import EpisodeView
from exp.optimize.evaluation.variance.judged import (
    JUDGED_STUDY_TYPE,
    JudgeCoverage,
    JudgedSlot,
    JudgedStudy,
    slot_coverage,
    slot_rewards,
)
from exp.optimize.evaluation.variance.rewards import (
    DEFAULT_SELECTOR_OBJECTIVE,
    JudgeScore,
    PointDecision,
    PointOutcome,
    SelectorObjective,
    SelectorObjectiveV2,
    SelectorScore,
    group_advantages,
    score_selector,
)

FindingSource = Literal[
    "actor_critic", "decision_state", "scenario", "variant_review", "replay_divergence"
]
"""Where an audit finding was raised."""

Resolution = Literal["fixed", "accepted_with_reason", "not_applicable"]
"""How a finding was resolved."""

ConditionName = Literal[
    "family_flip",
    "decision_variance",
    "fresh_confirmation",
    "audit_dispositions",
    "critic_scores",
    "tool_coverage",
    "rule_witnesses",
    "policy_consistency",
    "judge_coverage",
    "rubric_problems",
    "rubric_compatibility",
    "world_freeze",
    "world_provenance",
]
"""Every condition training eligibility checks."""

CONDITIONS: tuple[ConditionName, ...] = get_args(ConditionName)
"""The conditions in the order they are reported."""

AnySelectorObjective = SelectorObjective | SelectorObjectiveV2
"""A point-selector objective of either version, told apart by its version literal."""

UNTIERED = "untiered"
"""Stratum of continuation groups whose actor has no recorded tier."""


class AuditFinding(ContractModel):
    """One audit finding, identified by a stable hash of where it came from and what it says.

    Attributes:
        finding_id: SHA-256 of ``source`` and ``detail``; recomputed on validation.
        source: Which stage raised it.
        detail: The finding exactly as that stage recorded it.
    """

    finding_id: str
    source: FindingSource
    detail: JsonObject

    @model_validator(mode="after")
    def _identity(self) -> Self:
        """Refuse an identity that is not the hash of the finding's content."""
        if self.finding_id != finding_identity(self.source, self.detail):
            raise ValueError("an audit finding's identity must hash its source and detail")
        return self


class AuditDisposition(ContractModel):
    """A recorded decision about one audit finding.

    Attributes:
        finding_id: Identity of the finding this resolves.
        resolution: Whether the problem was fixed, accepted with a reason, or does not apply.
        rationale: Why that resolution is right.
        resolver: Who decided, such as a reviewer's name or a named review process.
        evidence_refs: Saved artifacts or locators backing the decision.
    """

    finding_id: str = Field(min_length=1)
    resolution: Resolution
    rationale: str = Field(min_length=1, max_length=2000)
    resolver: str = Field(min_length=1, max_length=200)
    evidence_refs: tuple[str, ...] = ()


def finding_identity(source: FindingSource, detail: JsonObject) -> str:
    """Return the stable identity of a finding.

    Args:
        source: Which stage raised it.
        detail: The finding's content.

    Returns:
        SHA-256 over the source and canonical detail.
    """
    return sha256_json({"source": source, "detail": detail})


def audit_finding(source: FindingSource, detail: JsonObject) -> AuditFinding:
    """Build a finding with its identity.

    Args:
        source: Which stage raised it.
        detail: The finding's content.

    Returns:
        The finding.
    """
    return AuditFinding(finding_id=finding_identity(source, detail), source=source, detail=detail)


def pending_findings(
    findings: Iterable[AuditFinding], dispositions: Iterable[AuditDisposition]
) -> tuple[AuditFinding, ...]:
    """Return the findings no disposition resolves, in input order.

    Args:
        findings: Every finding a family carries.
        dispositions: Every recorded disposition.

    Returns:
        Findings without a disposition, repeated findings listed once.
    """
    resolved = {disposition.finding_id for disposition in dispositions}
    pending: dict[str, AuditFinding] = {}
    for finding in findings:
        if finding.finding_id not in resolved:
            pending.setdefault(finding.finding_id, finding)
    return tuple(pending.values())


class EligibilityCondition(ContractModel):
    """One computed eligibility condition.

    Attributes:
        name: Which condition.
        holds: Whether it holds.
        detail: What was found.
    """

    name: ConditionName
    holds: bool
    detail: str = Field(min_length=1)


class TrainingEligibility(ContractModel):
    """Every eligibility condition, computed; eligibility itself is derived, never stored.

    Attributes:
        conditions: Exactly one result per condition, in :data:`CONDITIONS` order.
    """

    conditions: tuple[EligibilityCondition, ...]

    @model_validator(mode="after")
    def _complete(self) -> Self:
        """Require every condition exactly once, so none can be left out to pass."""
        if tuple(condition.name for condition in self.conditions) != CONDITIONS:
            raise ValueError("training eligibility must report every condition once, in order")
        return self

    @property
    def eligible(self) -> bool:
        """Whether every condition holds."""
        return all(condition.holds for condition in self.conditions)

    @property
    def reasons(self) -> tuple[str, ...]:
        """The failed conditions, each with what was found."""
        return tuple(f"{c.name}: {c.detail}" for c in self.conditions if not c.holds)


def unreproduced(selection: FlipVerdict, confirmation: FlipVerdict | None) -> tuple[str, ...]:
    """Return the selection-qualifying variants whose confirmation did not reproduce the winner.

    Args:
        selection: The verdict over the selection contrasts.
        confirmation: The verdict over the fresh confirmation contrasts, or None when none ran.

    Returns:
        Scenario identities whose separated, matching selection winner was not again the
        separated, matching winner on fresh samples.
    """
    confirmed = {} if confirmation is None else {v.scenario_id: v for v in confirmation.variants}
    missing: list[str] = []
    for variant in selection.variants:
        if variant.status != "separated_matches":
            continue
        again = confirmed.get(variant.scenario_id)
        if again is None or again.status != "separated_matches" or again.leader != variant.leader:
            missing.append(variant.scenario_id)
    return tuple(missing)


def bound_job(job: ContinuationJob, checkpoint: ArtifactInput) -> ContinuationJob:
    """Start a frozen job at a checkpoint, with no carried prediction or selection plan.

    Args:
        job: The frozen job.
        checkpoint: Where its study starts.

    Returns:
        The rebound job; policy, budget and verifier are unchanged.
    """
    study = job.study.model_copy(
        update={
            "checkpoint": checkpoint,
            "critic_prediction": None,
            "predicted_success": None,
            "selection_plan": None,
        }
    )
    return ContinuationJob.model_validate_json(
        job.model_copy(update={"study": study}).model_dump_json()
    )


def confirmation_job(job: ContinuationJob, seed: int) -> ContinuationJob:
    """Return the job whose draws are independent of selection: its selector names the seed.

    Every intervention and suffix identity derives from the study selector, so the same policy,
    budget and verifier under this selector draws fresh samples that took no part in selection.

    Args:
        job: The frozen job selection ran under.
        seed: The confirmation seed.

    Returns:
        The re-keyed job.
    """
    selector = f"{job.study.selector} | fresh confirmation seed {seed}"[:2000]
    study = job.study.model_copy(update={"selector": selector})
    return ContinuationJob.model_validate_json(
        job.model_copy(update={"study": study}).model_dump_json()
    )


def confirmation_episodes(
    project: ProjectStore, scenario_id: str, contrast: OptionContrast
) -> tuple[EpisodeView, ...]:
    """Render every confirmation episode of one variant for the calibration criterion grader.

    Args:
        project: Project holding the saved action contrasts.
        scenario_id: The variant.
        contrast: Its confirmation option contrast.

    Returns:
        One view per cell with a saved suffix: the forced action and the graded outcome.

    Raises:
        ValueError: A saved action contrast differs from its reference.
    """
    store = project.artifacts
    views: list[EpisodeView] = []
    for reference in contrast.contrasts:
        if artifact_input(store.read(reference.artifact_id).manifest) != reference:
            raise ValueError(f"action contrast {reference.artifact_id} differs from its reference")
        saved = ActionContrast.model_validate_json(
            store.read_bytes(reference.artifact_id, "report.json")
        )
        for cell in saved.cells:
            if cell.study is None:
                continue
            branch = load_study(store, cell.study).branches[0]
            action = saved.spec.baseline if cell.arm == "baseline" else saved.spec.alternative
            text = json.dumps(
                {
                    "forced_action": action.model_dump(mode="json"),
                    "episode": branch.grading_outcome(),
                },
                sort_keys=True,
            )
            views.append(
                EpisodeView(
                    episode_id=f"{scenario_id}:{reference.artifact_id}:{cell.pair}:{cell.arm}",
                    text=text,
                )
            )
    return tuple(views)


def relied_rules(answer: DecisionStateAnswer | None) -> tuple[str, ...]:
    """Return the rule identities a decision-state answer cites, sorted.

    Args:
        answer: The validated answer, or None when the draw was unusable.

    Returns:
        Identities from every ``rule:`` locator in its evidence, audit and axis grounding.
    """
    if answer is None:
        return ()
    state = answer.state
    refs = (
        *state.evidence_refs,
        *(ref for flag in state.audit for ref in flag.evidence_refs),
        *(ref for grounding in answer.grounding for ref in grounding.refs),
    )
    return tuple(sorted({ref[len("rule:") :] for ref in refs if ref.startswith("rule:")}))


def policy_key(meaning: EstimateMeaning) -> str:
    """Return the identity of the policy an estimate was made under, ignoring where it started.

    Args:
        meaning: An estimate's frozen meaning.

    Returns:
        SHA-256 over every field except the checkpoint, the calls remaining there, the runtime
        world (frozen authored variants each run their own frozen definition, checked by the
        ``world_provenance`` condition) and the judge, which labels outcomes and is not part of
        the actor policy.
    """
    excluded = {"checkpoint", "remaining_calls", "world_sha256", "judge_sha256"}
    return sha256_json(meaning.model_dump(mode="json", exclude=excluded))


def unscored_critics(critics: Sequence[CriticDraw]) -> tuple[int, ...]:
    """Return the indices of critic draws with an assessment but no scored reward."""
    return tuple(
        draw.draw
        for draw in critics
        if draw.assessment is not None and (draw.score is None or draw.score.reward is None)
    )


def training_eligibility(
    *,
    selection: FlipVerdict | None,
    confirmation: FlipVerdict | None,
    findings: Sequence[AuditFinding],
    dispositions: Sequence[AuditDisposition],
    critics: Sequence[CriticDraw],
    uncovered_tools: Sequence[str],
    relied_rules: Sequence[str],
    contradicted_rules: Sequence[str],
    unwitnessed_rules: Sequence[str],
    meanings: Sequence[EstimateMeaning],
    judging: Mapping[str, dict[str, tuple[str, ...]]],
    decision_variance: tuple[bool, str] | None = None,
) -> TrainingEligibility:
    """Compute every eligibility condition from saved evidence.

    Args:
        selection: Verdict over the selection contrasts, or None when none ran.
        confirmation: Verdict over the fresh confirmation contrasts, or None when none ran.
        findings: Every audit finding the family carries.
        dispositions: Every recorded disposition.
        critics: Every critic draw of the actor-critic run.
        uncovered_tools: Tools the family's task declares without a rule-book contract.
        relied_rules: Rule identities the family's decision state cites.
        contradicted_rules: The rules stage's ``families_must_stay_diagnostic``.
        unwitnessed_rules: Rule identities without a supporting witness.
        meanings: Meanings of every estimate the records would rest on.
        judging: Admission-policy problems per condition and confirmed scenario, from
            :func:`~exp.optimize.evaluation.variance.admission.judging_problems`; a condition
            absent here was not checked and fails.
        decision_variance: Whether the joint decision-variance measure admitted the family and
            what it found, or None when the registration does not require it.

    Returns:
        The eligibility with every condition.
    """
    pending = pending_findings(findings, dispositions)
    unscored = unscored_critics(critics)
    contradicted = sorted(set(relied_rules) & set(contradicted_rules))
    unwitnessed = sorted(set(relied_rules) & set(unwitnessed_rules) - set(contradicted))
    policies = {policy_key(meaning) for meaning in meanings}
    missing = () if selection is None else unreproduced(selection, confirmation)
    qualified = selection is not None and selection.qualifies
    confirmed = qualified and confirmation is not None and not missing
    blocked = qualified and decision_variance is not None and not decision_variance[0]
    rule_problems = [
        *(f"{rule} was contradicted without a witness" for rule in contradicted),
        *(f"{rule} has no supporting witness" for rule in unwitnessed),
    ]
    return TrainingEligibility(
        conditions=(
            EligibilityCondition(
                name="family_flip",
                holds=qualified,
                detail="no option contrast ran"
                if selection is None
                else f"separated winners {list(selection.winners)}, triage "
                f"{list(selection.triage)}, undersampled {list(selection.undersampled)}",
            ),
            EligibilityCondition(
                name="decision_variance",
                holds=decision_variance is None or decision_variance[0],
                detail="not required by this registration"
                if decision_variance is None
                else decision_variance[1][:4000] or "no detail recorded",
            ),
            EligibilityCondition(
                name="fresh_confirmation",
                holds=confirmed,
                detail="selection did not qualify, so nothing was confirmed"
                if not qualified
                else "not run: the decision_variance condition blocked confirmation"
                if blocked and confirmation is None
                else "no confirmation contrast ran"
                if confirmation is None
                else f"confirmation did not reproduce the winner of {list(missing)}"
                if missing
                else "every qualifying variant reproduced its winner on fresh samples",
            ),
            EligibilityCondition(
                name="audit_dispositions",
                holds=not pending,
                detail=f"{len(pending)} of {len(findings)} audit findings have no disposition"
                if pending
                else f"all {len(findings)} audit findings have a disposition",
            ),
            EligibilityCondition(
                name="critic_scores",
                holds=not unscored,
                detail=f"critic draws {list(unscored)} have no scored estimate"
                if unscored
                else "every critic draw with estimates was scored",
            ),
            EligibilityCondition(
                name="tool_coverage",
                holds=not uncovered_tools,
                detail=f"tools {list(uncovered_tools)} have no learned contract"
                if uncovered_tools
                else "every declared tool has a learned contract",
            ),
            EligibilityCondition(
                name="rule_witnesses",
                holds=not rule_problems,
                detail="; ".join(rule_problems)
                if rule_problems
                else f"{len(set(relied_rules))} relied-on rules are witnessed",
            ),
            EligibilityCondition(
                name="policy_consistency",
                holds=len(policies) == 1,
                detail=f"estimates span {len(policies)} actor policies; train within one only"
                if len(policies) != 1
                else f"{len(meanings)} estimates share one actor policy",
            ),
            *(
                _admission(name, judging.get(name))
                for name in (
                    "judge_coverage",
                    "rubric_problems",
                    "rubric_compatibility",
                    "world_freeze",
                    "world_provenance",
                )
            ),
        )
    )


def _admission(
    name: ConditionName, problems: Mapping[str, tuple[str, ...]] | None
) -> EligibilityCondition:
    """Turn one admission-policy check into a condition naming every failing variant."""
    if problems is None:
        return EligibilityCondition(
            name=name, holds=False, detail="no confirmed variant was checked against the policy"
        )
    if problems:
        detail = "; ".join(f"{s}: {', '.join(found)}" for s, found in sorted(problems.items()))
        return EligibilityCondition(name=name, holds=False, detail=detail[:4000])
    return EligibilityCondition(name=name, holds=True, detail="every confirmed variant passes")


class CriticRecord(ContractModel):
    """One discovery critic estimate scored against the discovery-judged rate under one meaning.

    Attributes:
        draw: Critic draw index.
        point: Point number estimated.
        estimate: The critic's probability of success.
        measured_rate: Judged success rate of the point's scored continuations.
        graded: Judged continuations behind the measured rate.
        squared_error: The estimate's squared error against the measured rate.
        meaning: What both the estimate and the rate are conditional on, including the judge.
        reward: The draw's reward over every scored point (negative mean Brier score).
        trainable: Always False: the discovery judge is not a variant's frozen judge, so this
            is diagnostic until fresh variant-specific predictions are scored.
    """

    draw: int = Field(ge=0)
    point: int = Field(ge=0)
    estimate: float = Field(ge=0.0, le=1.0)
    measured_rate: float = Field(ge=0.0, le=1.0)
    graded: int = Field(ge=1)
    squared_error: float = Field(ge=0.0, le=1.0)
    meaning: EstimateMeaning
    reward: float = Field(ge=-1.0, le=0.0)
    trainable: Literal[False] = False


class ContinuationGroup(ContractModel):
    """One confirmed variant's episodes after one forced option, for group-relative training.

    Attributes:
        scenario_id: The confirmed variant.
        option_id: The option every episode in the group was forced to take first.
        tier: Recorded tier of the actor, or :data:`UNTIERED`.
        meaning: What the group's outcomes are conditional on, including the variant's judge.
        judge: Identity of the frozen judge whose saved verdicts label the group.
        episodes: Each planned episode's saved assessment or gap, in plan order.
        advantages: Group-relative advantage per episode, read from those assessments; None
            where unscored.
        imposed_action_trainable: Always False: the forced first action was imposed by the
            experiment and is not actor-sampled; only the actor's suffix may carry advantages.
    """

    scenario_id: str
    option_id: str
    tier: str = Field(min_length=1)
    meaning: EstimateMeaning
    judge: Sha256
    episodes: tuple[JudgedSlot, ...]
    advantages: tuple[float | None, ...]
    imposed_action_trainable: Literal[False] = False


class SelectorPoint(ContractModel):
    """One chosen point whose continuations ran, with every judged slot behind its outcome.

    Attributes:
        point: Point number within the picked attempt.
        checkpoint: The paired actor/world checkpoint the continuations started from.
        judged: Saved ``judged-continuation-study`` of the point's continuations.
        slots: Each continuation's saved assessment or gap, in branch order.
        outcome: Judged counts read back from those assessments.
    """

    point: int = Field(ge=0)
    checkpoint: ArtifactInput
    judged: ArtifactInput
    slots: tuple[JudgedSlot, ...]
    outcome: PointOutcome


class SelectorRecord(ContractModel):
    """One saved point-selector draw, its objective reward and group-relative advantage.

    Attributes:
        draw: Selector draw index.
        settings: Frozen request settings of the draw, including its model snapshot.
        response: Saved final response, or None when the draw produced none.
        transcript: Saved analysis transcript holding every request, tool result and the answer.
        selection: The validated selection, or None when the answer was refused.
        points: Chosen points whose continuations ran (see :attr:`SelectorRecords.points`).
        score: The objective score, or None when the draw has no saved answer to train on.
        advantage: Group-relative advantage across the run's draws; None when unrewarded.
        trainable: Whether an external trainer may use this draw; computed, never asserted.
        refused: Why it may not, when it may not.
    """

    draw: int = Field(ge=0)
    settings: AuthorSettings
    response: ArtifactInput | None
    transcript: ArtifactInput | None
    selection: JudgeSelection | None
    points: tuple[int, ...]
    score: SelectorScore | None
    advantage: float | None
    trainable: bool
    refused: tuple[str, ...]

    @model_validator(mode="after")
    def _computed(self) -> Self:
        """Require trainability to follow from its reasons and a rewarded, saved answer."""
        if self.trainable != (not self.refused):
            raise ValueError("a selector record is trainable exactly when nothing refuses it")
        if self.trainable and (self.advantage is None or self.transcript is None):
            raise ValueError("a trainable selector record needs a saved answer and an advantage")
        return self


class SelectorRecords(ContractModel):
    """Every point-selector draw of one error-mining run, as one group-relative training group.

    Attributes:
        objective: The frozen selector objective.
        judge: Selector judge identity the records were checked against.
        group: Group identity: the saved group of attempts and the picked attempt.
        coverage: The judge's coverage of every continuation from every chosen point.
        points: Chosen points whose continuations ran, in point order.
        draws: One record per selector draw, in draw order.
        refused: Why no draw of this group is trainable, when none is.
        optimizer: Always ``external``: these are records only; no optimizer runs here.
    """

    objective: AnySelectorObjective
    judge: Sha256
    group: str
    coverage: JudgeCoverage
    points: tuple[SelectorPoint, ...]
    draws: tuple[SelectorRecord, ...]
    refused: tuple[str, ...]
    optimizer: Literal["external"] = "external"

    @property
    def mean_reward(self) -> float | None:
        """Mean reward over rewarded draws, or None when none was rewarded."""
        rewards = [d.score.reward for d in self.draws if d.score and d.score.reward is not None]
        return sum(rewards) / len(rewards) if rewards else None


def load_judged_study(project: ProjectStore, reference: ArtifactInput) -> JudgedStudy:
    """Load a saved judged continuation study, refusing a changed or mistyped artifact.

    Args:
        project: Project holding the study.
        reference: Its saved reference.

    Returns:
        The judged study.

    Raises:
        ValueError: The artifact differs from its reference or is not a judged study.
    """
    stored = project.artifacts.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference:
        raise ValueError(f"judged study {reference.artifact_id} differs from its reference")
    if stored.manifest.artifact_type != JUDGED_STUDY_TYPE:
        raise ValueError(f"{reference.artifact_id} is not a {JUDGED_STUDY_TYPE}")
    return JudgedStudy.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "judged.json")
    )


def selector_records(
    project: ProjectStore,
    report: ActorCriticReport,
    *,
    judge: str,
    objective: SelectorObjective | SelectorObjectiveV2 = DEFAULT_SELECTOR_OBJECTIVE,
    admission: FamilyAdmissionPolicy | None = None,
    refusals: Sequence[str] = (),
    decisions: Mapping[str, PointDecision] | None = None,
) -> SelectorRecords:
    """Build every selector draw's training record from the run's saved judged continuations.

    Args:
        project: Project holding the run's judged studies and assessments.
        report: The saved actor-critic run.
        judge: Selector judge identity every judged study must carry.
        objective: The selector objective of either version; version 2 by default.
        admission: Coverage limits the run's judging must meet, or None for the defaults.
        refusals: Further group-level reasons the caller found, such as a judge mismatch.
        decisions: Classified choices or family joint score by point identity
            (``point-<step>``), read by version 2; a pick without one is unscored.

    Returns:
        The records; reading saved artifacts only, never calling a model.
    """
    problems = list(refusals)
    mining = report.spec.episode_judge.identity
    if mining != judge:
        problems.append(f"the run's episode judge {mining[:12]} is not the selector judge")
    coverage = JudgeCoverage(judge=mining)
    points: list[SelectorPoint] = []
    for row in report.points:
        if row.failure is not None or row.study is None or row.judged is None:
            continue
        judged = load_judged_study(project, row.judged)
        if judged.judge != mining or judged.study != row.study:
            problems.append(f"point {row.point} was judged by another judge or for another study")
            continue
        rewards = slot_rewards(project, judged.branches, mining)
        labels = [reward for reward in rewards if reward is not None]
        coverage = coverage.plus(slot_coverage(project, judged.branches, mining))
        passes = int(sum(labels))
        outcome = PointOutcome(
            point_id=f"point-{row.point}",
            step=row.point,
            successes=passes,
            failures=len(labels) - passes,
            unknown=len(rewards) - len(labels),
        )
        points.append(
            SelectorPoint(
                point=row.point,
                checkpoint=row.checkpoint,
                judged=row.judged,
                slots=judged.branches,
                outcome=outcome,
            )
        )
    problems.extend(coverage_problems(coverage, admission or FamilyAdmissionPolicy())[0])
    outcomes = {point.outcome.point_id: point.outcome for point in points}
    scores: list[SelectorScore | None] = []
    for draw in report.judges:
        if draw.response is None or draw.transcript is None:
            scores.append(None)
            continue
        picks = () if draw.selection is None else draw.selection.points
        points_chosen = [f"point-{p.step}" for p in picks]
        scores.append(score_selector(points_chosen, outcomes, objective, decisions))
    group = f"{report.group.artifact_id}:{report.picked}"
    rewards = [None if score is None else score.reward for score in scores]
    if sum(reward is not None for reward in rewards) < 2:
        problems.append("fewer than two selector draws were rewarded, so the group is empty")
    advantages = group_advantages(group, rewards).advantages
    draws = []
    for draw, score, advantage in zip(report.judges, scores, advantages, strict=True):
        reasons = list(problems)
        if score is None:
            reasons.append("the draw has no saved answer")
        elif score.reward is None:
            reasons.append("no pick of the draw was scored")
        chosen = set() if draw.selection is None else {p.step for p in draw.selection.points}
        draws.append(
            SelectorRecord(
                draw=draw.draw,
                settings=report.spec.judges[draw.draw],
                response=draw.response,
                transcript=draw.transcript,
                selection=draw.selection,
                points=tuple(point.point for point in points if point.point in chosen),
                score=score,
                advantage=advantage,
                trainable=not reasons,
                refused=tuple(reasons),
            )
        )
    return SelectorRecords(
        objective=objective,
        judge=judge,
        group=group,
        coverage=coverage,
        points=tuple(points),
        draws=tuple(draws),
        refused=tuple(problems),
    )


class JudgeScreening(ContractModel):
    """One point-selector draw's screening score, with its selector training record.

    Attributes:
        draw: Selector draw index.
        score: The screening score; analysis only, never a training reward.
        selector: The draw's selector-objective record, when built.
        trainable: Whether the draw's selector record is trainable; computed from it.
    """

    draw: int = Field(ge=0)
    score: JudgeScore
    selector: SelectorRecord | None = None
    trainable: bool = False

    @model_validator(mode="after")
    def _computed(self) -> Self:
        """Derive trainability from the selector record, so it cannot be asserted."""
        if self.trainable != (self.selector is not None and self.selector.trainable):
            raise ValueError("a screening entry is trainable exactly when its selector record is")
        return self


class TrainingRecords(ContractModel):
    """Every record an eligible family yields.

    Attributes:
        policy: Policy key every record shares (:func:`policy_key`).
        critics: Discovery critic records, by draw then point; none is trainable. Empty when
            the error-mining run's episode judge is not every confirmed variant's judge.
        critics_refused: Why discovery critic records were refused, when they were.
        groups: Judged continuation groups per actor tier, each by variant then option.
        judges: Point-selector screening scores, each with its selector record.
        coverage: The judge coverage of every confirmed variant behind the groups.
        selectors: Point-selector training records of the error-mining run.
    """

    policy: str
    critics: tuple[CriticRecord, ...]
    critics_refused: str | None = None
    groups: dict[str, tuple[ContinuationGroup, ...]]
    judges: tuple[JudgeScreening, ...]
    coverage: dict[str, JudgeCoverage] = Field(default_factory=dict)
    selectors: SelectorRecords | None = None


def training_records(
    project: ProjectStore,
    eligibility: TrainingEligibility,
    report: ActorCriticReport,
    confirmed: Mapping[str, tuple[OptionContrast, EstimateMeaning]],
    *,
    tiers: Mapping[str, str] | None = None,
    objective: SelectorObjective | SelectorObjectiveV2 = DEFAULT_SELECTOR_OBJECTIVE,
    admission: FamilyAdmissionPolicy | None = None,
    decisions: Mapping[str, PointDecision] | None = None,
) -> TrainingRecords | tuple[str, ...]:
    """Build the records of an eligible family from its judged confirmations, or say why not.

    Args:
        project: Project holding the saved assessments.
        eligibility: The family's computed eligibility.
        report: The actor-critic run the family was built from.
        confirmed: Confirmation contrast and its meaning per confirmed variant.
        tiers: Tier label per actor model identity, when tiers are recorded.
        objective: Point-selector objective of either version; version 2 by default.
        admission: Coverage limits the run's selector judging must meet, or None for defaults.
        decisions: Classified choices or family joint score by point identity, for version 2.

    Returns:
        The records, or the failed eligibility conditions.
    """
    if not eligibility.eligible:
        return eligibility.reasons
    if not confirmed:
        return ("no confirmed variant has judged episodes to build records from",)
    policy = policy_key(next(iter(confirmed.values()))[1])
    groups: dict[str, list[ContinuationGroup]] = {}
    for scenario, (contrast, meaning) in sorted(confirmed.items()):
        if policy_key(meaning) != policy:
            return (f"variant {scenario} was confirmed under a different actor policy",)
        tier = (tiers or {}).get(meaning.actor_model.model_id, UNTIERED)
        for option in contrast.options:
            slots = tuple(cell.slot for cell in option.cells)
            rewards = slot_rewards(project, slots, contrast.judge)
            groups.setdefault(tier, []).append(
                ContinuationGroup(
                    scenario_id=scenario,
                    option_id=option.option_id,
                    tier=tier,
                    meaning=meaning,
                    judge=contrast.judge,
                    episodes=slots,
                    advantages=group_advantages(
                        f"{scenario}:{option.option_id}", rewards
                    ).advantages,
                )
            )
    mining = report.spec.episode_judge.identity
    foreign = sorted(
        scenario for scenario, (contrast, _) in confirmed.items() if contrast.judge != mining
    )
    refused = (
        f"the error-mining judge {mining[:12]} is not the frozen judge of variants {foreign}, "
        "so its measured rates do not score critics for this family"
        if foreign
        else None
    )
    selectors = selector_records(
        project,
        report,
        judge=mining,
        objective=objective,
        admission=admission,
        refusals=(refused,) if refused else (),
        decisions=decisions,
    )
    by_draw = {record.draw: record for record in selectors.draws}
    return TrainingRecords(
        policy=policy,
        critics=() if refused else _discovery_critics(report),
        critics_refused=refused,
        groups={tier: tuple(items) for tier, items in sorted(groups.items())},
        judges=tuple(
            JudgeScreening(
                draw=draw.draw,
                score=draw.score,
                selector=by_draw.get(draw.draw),
                trainable=draw.draw in by_draw and by_draw[draw.draw].trainable,
            )
            for draw in report.judges
            if draw.score is not None
        ),
        coverage={scenario: contrast.coverage for scenario, (contrast, _) in confirmed.items()},
        selectors=selectors,
    )


def _discovery_critics(report: ActorCriticReport) -> tuple[CriticRecord, ...]:
    """Return every scored discovery critic estimate against its point's judged rate."""
    by_point = {
        row.point: row for row in report.points if row.failure is None and row.study is not None
    }
    critics: list[CriticRecord] = []
    for draw in report.critics:
        if draw.assessment is None or draw.score is None or draw.score.reward is None:
            continue
        for estimate in draw.assessment.estimates:
            row = by_point.get(estimate.step)
            rate = None if row is None else row.outcome.rate
            if row is None or rate is None:
                continue
            critics.append(
                CriticRecord(
                    draw=draw.draw,
                    point=row.point,
                    estimate=estimate.probability,
                    measured_rate=rate,
                    graded=row.outcome.graded,
                    squared_error=(estimate.probability - rate) ** 2,
                    meaning=row.meaning,
                    reward=draw.score.reward,
                )
            )
    return tuple(critics)
