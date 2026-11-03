"""Audit dispositions, computed training eligibility and the records an eligible family yields.

A decision family may be trained on only after every condition below holds, and each is computed
from saved evidence rather than asserted by a caller:

- **family_flip**: the family qualified on its selection contrasts
  (:func:`~exp.optimize.evaluation.variance.contrast.family_flip`).
- **fresh_confirmation**: every variant that qualified on selection was contrasted again with
  independently drawn samples that took no part in selection, and the confirmation reproduced its
  separated winner.
- **audit_dispositions**: every audit finding (actor-critic flags, decision-state flags, scenario
  problems, variant checks marked for review) has a recorded disposition.
- **critic_scores**: no critic draw whose estimates were used is unscored.
- **tool_coverage**: the family's task declares no tool the learned rule book lacks a contract for.
- **rule_witnesses**: no rule the family relies on was contradicted without a witness, or lacks
  a supporting witness.
- **policy_consistency**: every estimate the records rest on shares one actor policy, harness,
  budget, sampling setting, world and verifier; only the checkpoint and remaining calls differ.

Failed conditions are all listed, never only the first. Records are built only for an eligible
family: critic records (estimate against measured rate under one meaning, with the draw's reward),
one continuation group per point for group-relative training, stratified by actor tier when tiers
are recorded, and judge screening scores, which are always marked non-trainable.

Fresh confirmation draws come from :func:`confirmation_job`, and their episodes are rendered for
the rubric grader by :func:`confirmation_episodes`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject, sha256_json
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.action_contrast import ActionContrast
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport, CriticDraw
from exp.optimize.evaluation.variance.authors import DecisionStateAnswer
from exp.optimize.evaluation.variance.contrast import FlipVerdict, OptionContrast
from exp.optimize.evaluation.variance.exposure import EstimateMeaning
from exp.optimize.evaluation.variance.grader import EpisodeView
from exp.optimize.evaluation.variance.rewards import JudgeScore

FindingSource = Literal["actor_critic", "decision_state", "scenario", "variant_review"]
"""Where an audit finding was raised."""

Resolution = Literal["fixed", "accepted_with_reason", "not_applicable"]
"""How a finding was resolved."""

ConditionName = Literal[
    "family_flip",
    "fresh_confirmation",
    "audit_dispositions",
    "critic_scores",
    "tool_coverage",
    "rule_witnesses",
    "policy_consistency",
]
"""Every condition training eligibility checks."""

CONDITIONS: tuple[ConditionName, ...] = (
    "family_flip",
    "fresh_confirmation",
    "audit_dispositions",
    "critic_scores",
    "tool_coverage",
    "rule_witnesses",
    "policy_consistency",
)
"""The conditions in the order they are reported."""

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
    """Render every graded confirmation episode of one variant for the rubric grader.

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
        SHA-256 over every field except the checkpoint and the calls remaining there.
    """
    return sha256_json(meaning.model_dump(mode="json", exclude={"checkpoint", "remaining_calls"}))


def unscored_critics(critics: Sequence[CriticDraw]) -> tuple[int, ...]:
    """Return the draws whose estimates exist but earned no reward.

    Args:
        critics: Every critic draw of the run.

    Returns:
        Draw indices with an assessment and no scored reward.
    """
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
                name="fresh_confirmation",
                holds=confirmed,
                detail="selection did not qualify, so nothing was confirmed"
                if not qualified
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
        )
    )


class CriticRecord(ContractModel):
    """One critic estimate scored against the measured rate under one meaning.

    Attributes:
        draw: Critic draw index.
        point: Point number estimated.
        estimate: The critic's probability of success.
        measured_rate: Success rate of the point's graded continuations.
        graded: Continuations behind the measured rate.
        squared_error: The estimate's squared error against the measured rate.
        meaning: What both the estimate and the rate are conditional on.
        reward: The draw's reward over every scored point (negative mean Brier score).
    """

    draw: int = Field(ge=0)
    point: int = Field(ge=0)
    estimate: float = Field(ge=0.0, le=1.0)
    measured_rate: float = Field(ge=0.0, le=1.0)
    graded: int = Field(ge=1)
    squared_error: float = Field(ge=0.0, le=1.0)
    meaning: EstimateMeaning
    reward: float = Field(ge=-1.0, le=0.0)


class ContinuationGroup(ContractModel):
    """The continuations from one point as one group for group-relative training.

    Attributes:
        point: Point number the continuations started from.
        tier: Recorded tier of the actor, or :data:`UNTIERED`.
        meaning: What the group's outcomes are conditional on.
        study: Saved continuation study holding the episodes.
        advantages: Group-relative advantage per continuation; None where ungraded.
    """

    point: int = Field(ge=0)
    tier: str = Field(min_length=1)
    meaning: EstimateMeaning
    study: ArtifactInput
    advantages: tuple[float | None, ...]


class JudgeScreening(ContractModel):
    """One judge draw's screening score, kept for analysis and never trained on.

    Attributes:
        draw: Judge draw index.
        score: The screening score.
        trainable: Always False.
    """

    draw: int = Field(ge=0)
    score: JudgeScore
    trainable: Literal[False] = False


class TrainingRecords(ContractModel):
    """Every record an eligible family yields.

    Attributes:
        policy: Policy key every record shares (:func:`policy_key`).
        critics: Critic records, by draw then point.
        groups: Continuation groups per actor tier, each by point.
        judges: Judge screening scores; none of them is trainable.
    """

    policy: str
    critics: tuple[CriticRecord, ...]
    groups: dict[str, tuple[ContinuationGroup, ...]]
    judges: tuple[JudgeScreening, ...]


def training_records(
    eligibility: TrainingEligibility,
    report: ActorCriticReport,
    *,
    tiers: Mapping[str, str] | None = None,
) -> TrainingRecords | tuple[str, ...]:
    """Build the records of an eligible family, or return why there are none.

    Only points whose continuations ran and that share the family's single policy contribute.

    Args:
        eligibility: The family's computed eligibility.
        report: The actor-critic run the family was built from.
        tiers: Tier label per actor model identity, when tiers are recorded.

    Returns:
        The records, or the failed eligibility conditions.
    """
    if not eligibility.eligible:
        return eligibility.reasons
    rows = [row for row in report.points if row.failure is None and row.study is not None]
    if not rows:
        return ("no point has graded continuations to build records from",)
    policy = policy_key(rows[0].meaning)
    rows = [row for row in rows if policy_key(row.meaning) == policy]
    by_point = {row.point: row for row in rows}
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
    groups: dict[str, list[ContinuationGroup]] = {}
    for row in sorted(rows, key=lambda item: item.point):
        if row.study is None:
            continue
        tier = (tiers or {}).get(row.meaning.actor_model.model_id, UNTIERED)
        groups.setdefault(tier, []).append(
            ContinuationGroup(
                point=row.point,
                tier=tier,
                meaning=row.meaning,
                study=row.study,
                advantages=row.group.advantages,
            )
        )
    return TrainingRecords(
        policy=policy,
        critics=tuple(critics),
        groups={tier: tuple(items) for tier, items in sorted(groups.items())},
        judges=tuple(
            JudgeScreening(draw=draw.draw, score=draw.score)
            for draw in report.judges
            if draw.score is not None
        ),
    )
