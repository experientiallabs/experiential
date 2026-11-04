"""Selected estimation points, critic values and continuations over one group of attempts.

One error-mining run (protocol :data:`ERROR_MINING_PROTOCOL`) follows a fixed sequence over saved,
replayable owners. :meth:`ActorCriticSpec.error_mining` registers the reference configuration
G=16, J=8, C=8, K=4, M=8; every count, model setting, evidence-tool budget and seed stays an
explicit spec field.

1. **Group.** The registered initial job runs ``G`` attempts of one task from one starting
   checkpoint in the configured world (typically the grounded world model with learned rules),
   saving a paired actor/world checkpoint after every tool turn.
2. **Pick.** One attempt is chosen with a seed derived from the run identity. This is not an
   unconditional draw over every attempt: it is a seeded uniform choice over the *eligible*
   population (completed, at least one saved tool turn, and at least two boundaries with actor
   calls left). The rule, the seed material, the shuffled order and every attempt's eligibility
   are saved in :class:`PickRecord`.
3. **Point selectors.** ``J`` selector draws (saved as ``judges``; they are not the episode judge)
   each choose up to ``K`` points of that attempt, reading the privileged evidence packet.
4. **Critics.** ``C`` critic draws each estimate the probability of success from every distinct
   chosen point, from the same evidence, before any continuation from those points runs.
5. **Continuations.** ``M`` continuations run from each distinct point (up to ``J*K`` of them
   after deduplication) and are graded by the task verifier.
6. **Rewards.** Each critic is scored by squared error against the measured rates, each judge by
   how far its points' measured rates separate beyond sampling noise, and each point's
   continuations form their own group for group-relative advantages.

The report records the actual populations and costs: :class:`MiningCounts` holds planned and
actual counts, including distinct points after deduplication and continuations run, and every
selector draw, critic draw, the group and each point's continuation study carry their
:class:`~exp.optimize.evaluation.variance.evidence_tools.MeasuredUsage`, read from the saved
provider responses. No cost ratio between roles is assumed.

Every attempt group and continuation group carries its :class:`ActorTier`: the declared tier label
and the digest of the actor/harness/budget/world/verifier profile. Groups with different tiers are
never pooled into one policy group (:func:`policy_groups`).

The report ranks points by how informative they are (measured rate near one half, and a large drop
to the next point) so the best of them can become Harbor environments that start exactly there.
Unknown outcomes are kept explicit and never redrawn.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ModelCatalog, ModelSnapshot, load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
    continuation_result_reference,
    execute_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.continuations.study import ContinuationStudy, known_outcomes
from exp.optimize.evaluation.executable.package import export_executable_harbor
from exp.optimize.evaluation.executable.task import ExecutableTask, persist_executable_task
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.contracts import CriticAssessment, JudgeSelection
from exp.optimize.evaluation.variance.evidence import (
    EvidencePacket,
    LoadedReference,
    ReferenceSource,
    candidate_branches,
    continuable_points,
    evidence_packet,
    load_references,
)
from exp.optimize.evaluation.variance.evidence_tools import (
    EvidenceToolProfile,
    MeasuredUsage,
    study_usage,
    total_usage,
)
from exp.optimize.evaluation.variance.exposure import (
    EstimateMeaning,
    ExposurePlan,
    estimate_meaning,
)
from exp.optimize.evaluation.variance.rewards import (
    CriticScore,
    GroupAdvantage,
    JudgeScore,
    PointOutcome,
    critic_score,
    decision_drops,
    group_advantages,
    judge_score,
)
from exp.optimize.evaluation.variance.roles import draw_critic, draw_judge, role_protocol_sha256
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.continuations import ContinuationCheckpoint, load_continuation_checkpoint

ERROR_MINING_PROTOCOL = "error-mining-v1"
"""Version of the error-mining experiment configuration and its recorded populations."""

PICK_RULE = (
    "seeded uniform choice over the eligible population: shuffle the completed attempts with at "
    "least one saved tool turn using the seed material, then take the first that offers at least "
    "two boundaries with fewer actor calls spent than the actor's maximum; ineligible attempts "
    "are never picked, so this is not an unconditional draw over every attempted trajectory"
)
"""The trajectory pick's eligibility rule, saved verbatim in every report."""


class ActorCriticSpec(ContractModel):
    """One registered error-mining run over a single task.

    Attributes:
        initial: Job whose study runs the group of attempts; its ``repeats`` is ``G``.
        judges: One frozen model setting per point-selector draw (``J`` draws); repeating a
            setting draws again.
        critics: One frozen model setting per critic draw (``C`` draws).
        points: Most estimation points each selector may choose (``K``).
        continuations: Continuations run from each distinct chosen point (``M``).
        continuation_physical_attempts: Transport attempt ceiling for each point's continuations.
        seed: Seed mixed into the trajectory pick.
        protocol: Error-mining configuration version.
        selector_profile: Evidence-tool profile and per-draw budgets of every selector draw;
            the default is one forced response.
        critic_profile: Evidence-tool profile and per-draw budgets of every critic draw.
        references: Registered reference and specification material the roles may read; any
            kind not registered is named unavailable in the packet.
        actor_tier: Declared model-tier label of the actor policy; groups with different tiers
            are never pooled.
    """

    initial: ContinuationJob
    judges: tuple[AuthorSettings, ...] = Field(min_length=1, max_length=16)
    critics: tuple[AuthorSettings, ...] = Field(min_length=1, max_length=16)
    points: int = Field(default=4, ge=1, le=8, strict=True)
    continuations: int = Field(default=8, ge=1, le=100, strict=True)
    continuation_physical_attempts: int = Field(default=64, ge=1, le=10000, strict=True)
    seed: int = Field(default=0, ge=0, le=2**32 - 1, strict=True)
    protocol: Literal["error-mining-v1"] = ERROR_MINING_PROTOCOL
    selector_profile: EvidenceToolProfile = EvidenceToolProfile()
    critic_profile: EvidenceToolProfile = EvidenceToolProfile()
    references: tuple[ReferenceSource, ...] = Field(default=(), max_length=16)
    actor_tier: str = Field(default="default", pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")

    @model_validator(mode="after")
    def _distinct_references(self) -> Self:
        """Require every reference to have its own locator."""
        names = [reference.name for reference in self.references]
        if len(set(names)) != len(names):
            raise ValueError("reference names must be distinct")
        return self

    @classmethod
    def error_mining(
        cls,
        initial: ContinuationJob,
        *,
        selector: AuthorSettings,
        critic: AuthorSettings,
        attempts: int = 16,
        selector_draws: int = 8,
        critic_draws: int = 8,
        points: int = 4,
        continuations: int = 8,
        seed: int = 0,
        group_physical_attempts: int | None = None,
        continuation_physical_attempts: int = 64,
        selector_profile: EvidenceToolProfile | None = None,
        critic_profile: EvidenceToolProfile | None = None,
        references: tuple[ReferenceSource, ...] = (),
        actor_tier: str = "default",
    ) -> ActorCriticSpec:
        """Register an error-mining run; the defaults are G=16, J=8, C=8, K=4 and M=8.

        Args:
            initial: Job registering the actor, world, verifier and starting checkpoint.
            selector: Frozen model setting of every point-selector draw.
            critic: Frozen model setting of every critic draw.
            attempts: Group size ``G``.
            selector_draws: Point-selector draws ``J``.
            critic_draws: Critic draws ``C``.
            points: Most points per selector ``K``.
            continuations: Suffixes per distinct point ``M``.
            seed: Trajectory pick seed.
            group_physical_attempts: Transport ceiling of the group, or None to keep the job's.
            continuation_physical_attempts: Transport ceiling of each point's continuations.
            selector_profile: Selector evidence-tool profile, or None for one forced response.
            critic_profile: Critic evidence-tool profile, or None for one forced response.
            references: Registered reference material.
            actor_tier: Declared actor tier label.

        Returns:
            The registration.
        """
        group = initial.model_copy(
            update={
                "maximum_physical_attempts": group_physical_attempts
                or initial.maximum_physical_attempts,
                "study": initial.study.model_copy(update={"repeats": attempts}),
            }
        )
        return cls.model_validate(
            {
                "initial": group,
                "judges": (selector,) * selector_draws,
                "critics": (critic,) * critic_draws,
                "points": points,
                "continuations": continuations,
                "continuation_physical_attempts": continuation_physical_attempts,
                "seed": seed,
                "selector_profile": selector_profile or EvidenceToolProfile(),
                "critic_profile": critic_profile or EvidenceToolProfile(),
                "references": references,
                "actor_tier": actor_tier,
            }
        )

    def identity(self, code_revision: str) -> str:
        """Bind the registration, role prompts and producing code."""
        return stable_id(
            "actor-critic-run",
            {"spec": sha256_json(self), "roles": role_protocol_sha256(), "code": code_revision},
        )


class ActorTier(ContractModel):
    """The policy tier one group of attempts or continuations was sampled under.

    Attributes:
        label: Declared tier label.
        actor_model: Actor policy snapshot.
        profile_sha256: Digest of the actor, harness, budget, sampling, world and verifier,
            excluding the checkpoint and the calls left there.
    """

    label: str
    actor_model: ModelSnapshot
    profile_sha256: Sha256


def actor_tier(label: str, meaning: EstimateMeaning) -> ActorTier:
    """Return the tier of a group sampled under one estimate meaning."""
    profile = meaning.model_dump(mode="json", exclude={"checkpoint", "remaining_calls"})
    return ActorTier(
        label=label, actor_model=meaning.actor_model, profile_sha256=sha256_json(profile)
    )


class PickRecord(ContractModel):
    """How the estimated trajectory was picked, and from which population.

    Attributes:
        rule: The eligibility rule, verbatim.
        seed_material_sha256: Digest of the run identity and seed the shuffle used.
        population: Every attempt index in the group.
        order: Shuffled order of the candidate attempts.
        eligible: Attempts the rule admits, in attempt order.
        ineligible: Why each other attempt was not eligible.
        picked: The chosen attempt: the first eligible one in ``order``.
    """

    rule: str = PICK_RULE
    seed_material_sha256: Sha256
    population: tuple[int, ...]
    order: tuple[int, ...]
    eligible: tuple[int, ...]
    ineligible: dict[int, str]
    picked: int


class MiningCounts(ContractModel):
    """Planned and actual populations of one run.

    Attributes:
        attempts_planned: Group size ``G``.
        attempts_completed: Attempts whose execution completed.
        attempts_eligible: Attempts the pick rule admits.
        selector_draws: Planned selector draws ``J``.
        selector_answers: Draws with a usable selection.
        critic_draws: Planned critic draws ``C``.
        critic_answers: Draws with usable estimates.
        points_per_selector: Most points per selector ``K``.
        points_offered: Continuable points of the picked attempt.
        points_selected: Points across every usable selection, before deduplication.
        points_distinct: Distinct points after deduplication; at most ``J*K``.
        selections_per_point: How many usable selections chose each distinct point.
        suffixes_per_point: Planned continuations per distinct point ``M``.
        continuations_planned: ``points_distinct * M``.
        continuations_recorded: Continuation branches saved across every point study.
        continuations_graded: Recorded branches with a known pass or fail.
        continuations_unknown: Recorded branches without a known grade.
        point_failures: Points whose continuations could not run.
    """

    attempts_planned: int
    attempts_completed: int
    attempts_eligible: int
    selector_draws: int
    selector_answers: int
    critic_draws: int
    critic_answers: int
    points_per_selector: int
    points_offered: int
    points_selected: int
    points_distinct: int
    selections_per_point: dict[int, int]
    suffixes_per_point: int
    continuations_planned: int
    continuations_recorded: int
    continuations_graded: int
    continuations_unknown: int
    point_failures: int


class StudyCost(ContractModel):
    """Measured cost of one continuation study.

    Attributes:
        actor: Usage over the study's saved actor responses.
        physical_attempts: Provider attempts the study's dispatch ledger claimed, or None when
            the ledger is absent.
        unmeasured: Spend this record does not cover, named explicitly.
    """

    actor: MeasuredUsage
    physical_attempts: int | None
    unmeasured: tuple[str, ...] = (
        "world-model and cascade-decider provider calls are metered by their own owners",
    )


class JudgeDraw(ContractModel):
    """One judge draw and what it earned.

    Attributes:
        draw: Draw index.
        response: Saved final response reference, or None when no response exists.
        selection: The validated selection, or None when it was refused or unavailable.
        failure: Why the draw produced no usable selection.
        score: The draw's reward, once its points' continuations are graded.
        transcript: Saved analysis transcript: every request, tool result and the answer.
        usage: Measured usage across the draw's model calls.
    """

    draw: int
    response: ArtifactInput | None
    selection: JudgeSelection | None
    failure: str | None
    score: JudgeScore | None = None
    transcript: ArtifactInput | None = None
    usage: MeasuredUsage | None = None


class CriticDraw(ContractModel):
    """One critic draw and what it earned.

    Attributes:
        draw: Draw index.
        response: Saved final response reference, or None when no response exists.
        assessment: The validated estimates, or None when they were refused or unavailable.
        failure: Why the draw produced no usable estimates.
        score: The draw's reward, once the chosen points' continuations are graded.
        transcript: Saved analysis transcript: every request, tool result and the answer.
        usage: Measured usage across the draw's model calls.
    """

    draw: int
    response: ArtifactInput | None
    assessment: CriticAssessment | None
    failure: str | None
    score: CriticScore | None = None
    transcript: ArtifactInput | None = None
    usage: MeasuredUsage | None = None


class PointResult(ContractModel):
    """Continuations from one chosen point.

    Attributes:
        point: Point number within the picked attempt.
        checkpoint: The paired actor/world checkpoint continuations started from.
        meaning: What the measured rate is conditional on.
        study: Saved continuation study, or None when it could not run.
        outcome: Graded outcome counts.
        group: Group-relative advantages of the point's continuations.
        failure: Why continuations could not run.
        tier: Policy tier the continuations were sampled under.
        cost: Measured cost of the point's continuation study, when it ran.
    """

    point: int
    checkpoint: ArtifactInput
    meaning: EstimateMeaning
    study: ArtifactInput | None
    outcome: PointOutcome
    group: GroupAdvantage
    failure: str | None
    tier: ActorTier | None = None
    cost: StudyCost | None = None


class ActorCriticReport(ContractModel):
    """The complete record of one run, including failed and unknown parts.

    Attributes:
        spec: The registration.
        group: The saved group of attempts.
        picked: The attempt whose points were estimated.
        judges: Every judge draw.
        critics: Every critic draw.
        points: Continuation results per chosen point, in point order.
        drops: Rate drop from each scored point to the next.
        audit: Every audit flag any role raised, with its role and draw.
        ranking: Point numbers ordered as candidates for competing-action contrast.
        training_eligible: Whether any record of this run may be trained on.
        ineligible_reasons: Why it may not, when it may not.
        pick: The pick rule, its eligible population and the seeded order.
        counts: Planned and actual populations.
        group_tier: Policy tier of the group of attempts.
        group_cost: Measured cost of the group of attempts.
        usage: Measured usage per role: ``group``, ``selectors``, ``critics``, ``continuations``.
    """

    spec: ActorCriticSpec
    group: ArtifactInput
    picked: int
    judges: tuple[JudgeDraw, ...]
    critics: tuple[CriticDraw, ...]
    points: tuple[PointResult, ...]
    drops: dict[str, float]
    audit: tuple[JsonObject, ...]
    ranking: tuple[int, ...]
    training_eligible: bool
    ineligible_reasons: tuple[str, ...]
    pick: PickRecord | None = None
    counts: MiningCounts | None = None
    group_tier: ActorTier | None = None
    group_cost: StudyCost | None = None
    usage: dict[str, MeasuredUsage] = Field(default_factory=dict)


def policy_groups(points: Sequence[PointResult]) -> dict[tuple[str, int], GroupAdvantage]:
    """Return one policy group per tier and point, refusing to pool different tiers.

    Args:
        points: Continuation results; each one's tier must be recorded.

    Returns:
        Group-relative advantages keyed by ``(tier profile digest, point)``.

    Raises:
        ValueError: A result has no tier, or one key would pool two different tiers or two
            different groups.
    """
    groups: dict[tuple[str, int], GroupAdvantage] = {}
    tiers: dict[str, ActorTier] = {}
    for row in points:
        if row.tier is None:
            raise ValueError(f"point {row.point} has no recorded tier; it cannot join a group")
        known = tiers.setdefault(row.tier.profile_sha256, row.tier)
        if known != row.tier:
            raise ValueError("one profile digest carries two different tier labels or models")
        key = (row.tier.profile_sha256, row.point)
        if key in groups:
            raise ValueError(f"point {row.point} appears twice under one tier; never pool groups")
        groups[key] = row.group
    return groups


def run_actor_critic(
    project: ProjectStore,
    spec: ActorCriticSpec,
    *,
    exposure: ExposurePlan,
    code_revision: str,
) -> ActorCriticReport:
    """Run, or replay, one actor-critic registration over a mining-probe task.

    Args:
        project: Project holding every saved step.
        spec: The registration.
        exposure: The frozen data roles; the task must come from a mining-probe lineage, never
            from world learning or the final evaluation.
        code_revision: Producing source identity.

    Returns:
        The report, also saved as an ``actor-critic-report`` artifact.

    Raises:
        ValueError: The task is not a mining probe, the group has no attempt with enough
            continuable points, no judge produced a usable selection, or a continuation group
            was sampled under a different tier than the group.
    """
    spec = ActorCriticSpec.model_validate_json(spec.model_dump_json())
    store, identity = project.artifacts, spec.identity(code_revision)
    start = load_continuation_checkpoint(store, spec.initial.study.checkpoint)
    exposure.require_lineage(start.task.lineage_group_id, "mining_probes")
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        result_id = identity + "-result"
        if store.exists(result_id):
            return ActorCriticReport.model_validate_json(store.read_bytes(result_id, "report.json"))
        group = saved_continuation_job(project, spec.initial, code_revision=code_revision)
        if group is None:
            group = execute_continuation_job(project, spec.initial, code_revision=code_revision)
        source = continuation_result_reference(project, spec.initial, code_revision=code_revision)
        loader = cache(lambda ref: load_continuation_checkpoint(store, ref))
        references = load_references(store, spec.references)
        pick, packet = _pick(group, identity, spec, loader, references)
        sources = (
            source,
            *packet.pool.points.values(),
            *{reference.artifact for reference in spec.references},
        )
        judges = tuple(
            _judge(project, spec, packet, j, identity, sources, code_revision)
            for j in range(len(spec.judges))
        )
        chosen = tuple(
            sorted({p.step for d in judges if d.selection is not None for p in d.selection.points})
        )
        if not chosen:
            raise ValueError("no judge produced a usable selection; inspect the saved draws")
        critic_answers = tuple(
            _critic(project, spec, packet, chosen, c, identity, sources, code_revision)
            for c in range(len(spec.critics))
        )
        catalog = load_model_catalog(project.model_catalog_path)
        points = tuple(
            _continue(project, spec, packet, point, loader, catalog, code_revision)
            for point in chosen
        )
        group_tier = actor_tier(spec.actor_tier, estimate_meaning(spec.initial, start))
        policy_groups(points)
        if any(row.tier is not None and row.tier != group_tier for row in points):
            raise ValueError("a continuation group was sampled under a different tier")
        outcomes = tuple(result.outcome for result in points)
        judges = tuple(_reward_judge(draw, outcomes) for draw in judges)
        critics = tuple(_reward_critic(draw, outcomes) for draw in critic_answers)
        drops = decision_drops(outcomes)
        audit = _audit(judges, critics)
        group_cost = _study_cost(project, spec.initial, group, catalog, code_revision)
        report = ActorCriticReport(
            spec=spec,
            group=source,
            picked=packet.pool.branch,
            judges=judges,
            critics=critics,
            points=points,
            drops=drops,
            audit=audit,
            ranking=_rank(outcomes, drops),
            training_eligible=False,
            ineligible_reasons=_ineligible(audit, critics),
            pick=pick,
            counts=_counts(spec, group, pick, packet, judges, critics, points),
            group_tier=group_tier,
            group_cost=group_cost,
            usage={
                "group": group_cost.actor,
                "selectors": total_usage([d.usage for d in judges if d.usage is not None]),
                "critics": total_usage([d.usage for d in critics if d.usage is not None]),
                "continuations": total_usage([r.cost.actor for r in points if r.cost]),
            },
        )
        dependencies = {source, *(d.response for d in (*judges, *critics) if d.response)}
        dependencies |= {d.transcript for d in (*judges, *critics) if d.transcript}
        dependencies |= {r.study for r in points if r.study} | {r.checkpoint for r in points}
        store.write_json(
            artifact_id=result_id,
            artifact_type="actor-critic-report",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(sorted(dependencies, key=lambda ref: ref.artifact_id)),
            ),
            files={"report.json": report},
        )
        return report


def export_point_harbor(
    project: ProjectStore,
    report: ActorCriticReport,
    point: int,
    destination: Path,
    *,
    name: str,
    intent: str,
    code_revision: str,
    qualification: ArtifactInput | None = None,
) -> Path:
    """Package one chosen point as a Harbor environment that starts exactly there.

    The environment uses the run's own world runtime, so tool calls are answered by the same
    world model and learned rules the continuations ran against. Without a qualification for
    the exact task the package is diagnostic and awards no reward.

    Args:
        project: Project holding the run.
        report: The saved run.
        point: Chosen point number to start from.
        destination: New package directory.
        name: Harbor ``organization/name``.
        intent: The capability the environment measures.
        code_revision: Producing source identity.
        qualification: Qualification report for the exact task, when it has one.

    Returns:
        The published package directory.

    Raises:
        ValueError: The point was not chosen in this run.
    """
    rows = {row.point: row for row in report.points}
    if point not in rows:
        raise ValueError(f"point {point} was not chosen in this run; choose from {sorted(rows)}")
    job = _point_job(report.spec, rows[point].checkpoint)
    task = ExecutableTask(intent=intent, job=job)
    reference = persist_executable_task(
        project, task, created_at=datetime.now(UTC), code_revision=code_revision
    )
    return export_executable_harbor(
        project,
        reference,
        destination,
        name=name,
        qualification=qualification,
        diagnostic=qualification is None,
    )


def _pick(
    group: ContinuationStudy,
    identity: str,
    spec: ActorCriticSpec,
    loader: Callable[[ArtifactInput], ContinuationCheckpoint],
    references: tuple[LoadedReference, ...],
) -> tuple[PickRecord, EvidencePacket]:
    """Pick the first seeded-shuffled eligible attempt and record the eligible population.

    Args:
        group: The group of attempts.
        identity: Run identity mixed into the seed.
        spec: The registration.
        loader: Cached checkpoint loader.
        references: Registered references rendered into the packet.

    Returns:
        The pick record and the packet for the picked attempt.

    Raises:
        ValueError: No attempt offers two continuable points.
    """
    candidates = candidate_branches(group)
    order = list(candidates)
    material = f"{identity}:{spec.seed}"
    random.Random(material).shuffle(order)
    calls = spec.initial.maximum_model_calls
    ineligible: dict[int, str] = {}
    for row in group.branches:
        if row.status != "completed":
            ineligible[row.index] = f"status {row.status}"
        elif not row.checkpoints:
            ineligible[row.index] = "no saved tool turn"
        elif (
            len(continuable_points(group, row.index, maximum_model_calls=calls, checkpoint=loader))
            < 2
        ):
            ineligible[row.index] = "fewer than two boundaries with actor calls left"
    eligible = tuple(index for index in candidates if index not in ineligible)
    picked = next((index for index in order if index in eligible), None)
    if picked is None:
        raise ValueError("no attempt in the group offers two continuable points")
    record = PickRecord(
        seed_material_sha256=sha256_json({"material": material}),
        population=tuple(row.index for row in group.branches),
        order=tuple(order),
        eligible=eligible,
        ineligible=ineligible,
        picked=picked,
    )
    packet = evidence_packet(
        group, picked, maximum_model_calls=calls, checkpoint=loader, references=references
    )
    return record, packet


def _judge(
    project: ProjectStore,
    spec: ActorCriticSpec,
    packet: EvidencePacket,
    draw: int,
    identity: str,
    sources: tuple[ArtifactInput, ...],
    code_revision: str,
) -> JudgeDraw:
    """Make or replay one point-selector draw, keeping a refusal as a failed draw."""
    try:
        reference, transcript, selection, failure = draw_judge(
            project,
            spec.judges[draw],
            packet,
            points=spec.points,
            identity=f"{identity}-judge-{draw}",
            sources=sources,
            code_revision=code_revision,
            profile=spec.selector_profile,
        )
    except ValueError as error:
        return JudgeDraw(draw=draw, response=None, selection=None, failure=str(error)[:2000])
    return JudgeDraw(
        draw=draw,
        response=transcript.calls[-1].response if transcript.calls else None,
        selection=selection,
        failure=failure,
        transcript=reference,
        usage=transcript.usage,
    )


def _critic(
    project: ProjectStore,
    spec: ActorCriticSpec,
    packet: EvidencePacket,
    chosen: tuple[int, ...],
    draw: int,
    identity: str,
    sources: tuple[ArtifactInput, ...],
    code_revision: str,
) -> CriticDraw:
    """Make or replay one critic draw before any continuation from the chosen points runs."""
    try:
        reference, transcript, assessment, failure = draw_critic(
            project,
            spec.critics[draw],
            packet,
            chosen,
            identity=f"{identity}-critic-{draw}",
            sources=sources,
            code_revision=code_revision,
            profile=spec.critic_profile,
        )
    except ValueError as error:
        return CriticDraw(draw=draw, response=None, assessment=None, failure=str(error)[:2000])
    return CriticDraw(
        draw=draw,
        response=transcript.calls[-1].response if transcript.calls else None,
        assessment=assessment,
        failure=failure,
        transcript=reference,
        usage=transcript.usage,
    )


def _study_cost(
    project: ProjectStore,
    job: ContinuationJob,
    study: ContinuationStudy,
    catalog: ModelCatalog,
    code_revision: str,
) -> StudyCost:
    """Measure one study from its saved actor responses and its dispatch ledger."""
    revision = job.implementation(code_revision)
    path = project.artifacts.project_directory / (revision + "-physical.sqlite")
    attempts: int | None = None
    if path.exists():
        attempts = DispatchLedger(path, job.maximum_physical_attempts, revision).used()
    return StudyCost(
        actor=study_usage(project.artifacts, study, catalog, job.actor_alias),
        physical_attempts=attempts,
    )


def _counts(
    spec: ActorCriticSpec,
    group: ContinuationStudy,
    pick: PickRecord,
    packet: EvidencePacket,
    judges: tuple[JudgeDraw, ...],
    critics: tuple[CriticDraw, ...],
    points: tuple[PointResult, ...],
) -> MiningCounts:
    """Count the run's actual populations."""
    selections = [draw.selection for draw in judges if draw.selection is not None]
    per_point: dict[int, int] = {}
    for selection in selections:
        for point in selection.points:
            per_point[point.step] = per_point.get(point.step, 0) + 1
    recorded = sum(row.outcome.graded + row.outcome.unknown for row in points)
    graded = sum(row.outcome.graded for row in points)
    return MiningCounts(
        attempts_planned=len(group.branches),
        attempts_completed=sum(row.status == "completed" for row in group.branches),
        attempts_eligible=len(pick.eligible),
        selector_draws=len(spec.judges),
        selector_answers=len(selections),
        critic_draws=len(spec.critics),
        critic_answers=sum(draw.assessment is not None for draw in critics),
        points_per_selector=spec.points,
        points_offered=len(packet.pool.points),
        points_selected=sum(len(selection.points) for selection in selections),
        points_distinct=len(per_point),
        selections_per_point=dict(sorted(per_point.items())),
        suffixes_per_point=spec.continuations,
        continuations_planned=len(per_point) * spec.continuations,
        continuations_recorded=recorded,
        continuations_graded=graded,
        continuations_unknown=recorded - graded,
        point_failures=sum(row.failure is not None for row in points),
    )


def _point_job(spec: ActorCriticSpec, checkpoint: ArtifactInput) -> ContinuationJob:
    """Return the continuation job that starts at one point."""
    return spec.initial.model_copy(
        update={
            "maximum_physical_attempts": spec.continuation_physical_attempts,
            "study": spec.initial.study.model_copy(
                update={
                    "checkpoint": checkpoint,
                    "repeats": spec.continuations,
                    "critic_prediction": None,
                    "predicted_success": None,
                    "selection_plan": None,
                }
            ),
        }
    )


def _continue(
    project: ProjectStore,
    spec: ActorCriticSpec,
    packet: EvidencePacket,
    point: int,
    loader: Callable[[ArtifactInput], ContinuationCheckpoint],
    catalog: ModelCatalog,
    code_revision: str,
) -> PointResult:
    """Run or replay one point's continuations, grade them and measure their cost."""
    checkpoint = packet.pool.points[point]
    job = _point_job(spec, checkpoint)
    meaning = estimate_meaning(job, loader(checkpoint))
    tier = actor_tier(spec.actor_tier, meaning)
    point_id = f"point-{point}"
    try:
        study = saved_continuation_job(project, job, code_revision=code_revision)
        if study is None:
            study = execute_continuation_job(project, job, code_revision=code_revision)
        reference = continuation_result_reference(project, job, code_revision=code_revision)
    except (ValueError, RuntimeError, OSError) as error:
        empty = PointOutcome(point_id=point_id, step=point, successes=0, failures=0)
        return PointResult(
            point=point,
            checkpoint=checkpoint,
            meaning=meaning,
            study=None,
            outcome=empty,
            group=group_advantages(point_id, ()),
            failure=(type(error).__name__ + ": " + str(error))[:2000],
            tier=tier,
        )
    labels = known_outcomes(branch.assessment for branch in study.branches)
    rewards = tuple(
        None
        if branch.assessment is None or branch.assessment.status == "unassessable"
        else float(branch.assessment.status == "pass")
        for branch in study.branches
    )
    return PointResult(
        point=point,
        checkpoint=checkpoint,
        meaning=meaning,
        study=reference,
        outcome=PointOutcome(
            point_id=point_id,
            step=point,
            successes=sum(labels),
            failures=len(labels) - sum(labels),
            unknown=len(study.branches) - len(labels),
        ),
        group=group_advantages(point_id, rewards),
        failure=None,
        tier=tier,
        cost=_study_cost(project, job, study, catalog, code_revision),
    )


def _reward_judge(draw: JudgeDraw, outcomes: tuple[PointOutcome, ...]) -> JudgeDraw:
    """Attach the judge reward computed over the points this judge chose."""
    if draw.selection is None:
        return draw
    mine = {point.step for point in draw.selection.points}
    score = judge_score(tuple(o for o in outcomes if o.step in mine))
    return draw.model_copy(update={"score": score})


def _reward_critic(draw: CriticDraw, outcomes: tuple[PointOutcome, ...]) -> CriticDraw:
    """Attach the critic reward computed against every chosen point's measured rate."""
    if draw.assessment is None:
        return draw
    estimates = {f"point-{row.step}": row.probability for row in draw.assessment.estimates}
    return draw.model_copy(update={"score": critic_score(estimates, outcomes)})


def _audit(
    judges: tuple[JudgeDraw, ...], critics: tuple[CriticDraw, ...]
) -> tuple[JsonObject, ...]:
    """Collect every audit flag with the role and draw that raised it."""
    flags: list[JsonObject] = []
    for draw in judges:
        if draw.selection is not None:
            flags.extend(
                {"role": "judge", "draw": draw.draw, **flag.model_dump(mode="json")}
                for flag in draw.selection.audit
            )
    for draw in critics:
        if draw.assessment is not None:
            flags.extend(
                {"role": "critic", "draw": draw.draw, **flag.model_dump(mode="json")}
                for flag in draw.assessment.audit
            )
    return tuple(flags)


def _ineligible(audit: tuple[JsonObject, ...], critics: tuple[CriticDraw, ...]) -> tuple[str, ...]:
    """Explain why a run's records may not be trained on yet.

    A screening run never qualifies on its own: no point has been shown to be a decision until
    competing actions from the same checkpoint have been run under a frozen policy and budget.

    Args:
        audit: Every audit flag the run raised.
        critics: Every critic draw.

    Returns:
        The reasons, most fundamental first.
    """
    reasons = [
        "no chosen point has been tested with competing actions from the same checkpoint",
    ]
    if audit:
        reasons.append(f"{len(audit)} audit findings have no recorded disposition")
    unscored = [draw.draw for draw in critics if draw.score is None or draw.score.reward is None]
    if unscored:
        reasons.append(f"critic draws {unscored} have no scored estimate")
    return tuple(reasons)


def _rank(outcomes: tuple[PointOutcome, ...], drops: dict[str, float]) -> tuple[int, ...]:
    """Order points by uncertainty of the measured rate, then by the drop that follows them."""

    def key(outcome: PointOutcome) -> tuple[float, float, int]:
        """Most informative first; unscored points last."""
        rate = outcome.rate
        uncertainty = -1.0 if rate is None else rate * (1.0 - rate)
        return (-uncertainty, -drops.get(outcome.point_id, 0.0), outcome.step)

    return tuple(outcome.step for outcome in sorted(outcomes, key=key))
