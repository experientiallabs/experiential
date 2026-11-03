"""Judged estimation points, critic values and continuations over one group of attempts.

One run follows a fixed sequence over saved, replayable owners:

1. **Group.** The registered initial job runs ``G`` attempts of one task from one starting
   checkpoint in the configured world (typically the grounded world model with learned rules),
   saving a paired actor/world checkpoint after every tool turn.
2. **Pick.** One completed attempt is chosen with a seed derived from the run identity, so the
   choice is reproducible and not made after seeing which attempt is convenient.
3. **Judges.** ``J`` judge draws each choose up to ``K`` points of that attempt, reading every
   attempt, the group's success rate and the verifier.
4. **Critics.** ``C`` critic draws each estimate the probability of success from every chosen
   point, from the same evidence, before any continuation from those points runs.
5. **Continuations.** ``M`` continuations run from each chosen point and are graded by the task
   verifier.
6. **Rewards.** Each critic is scored by squared error against the measured rates, each judge by
   how far its points' measured rates separate beyond sampling noise, and each point's
   continuations form their own group for group-relative advantages.

The report ranks points by how informative they are (measured rate near one half, and a large drop
to the next point) so the best of them can become Harbor environments that start exactly there.
Unknown outcomes are kept explicit and never redrawn.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
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
    candidate_branches,
    evidence_packet,
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
from exp.simulation.continuations import ContinuationCheckpoint, load_continuation_checkpoint


class ActorCriticSpec(ContractModel):
    """One registered actor-critic run over a single task.

    Attributes:
        initial: Job whose study runs the group of attempts; its ``repeats`` is the group size.
        judges: One frozen model setting per judge draw; repeating a setting draws again.
        critics: One frozen model setting per critic draw.
        points: Most estimation points each judge may choose.
        continuations: Continuations run from each chosen point.
        continuation_physical_attempts: Transport attempt ceiling for each point's continuations.
        seed: Seed mixed into the trajectory pick.
    """

    initial: ContinuationJob
    judges: tuple[AuthorSettings, ...] = Field(min_length=1, max_length=16)
    critics: tuple[AuthorSettings, ...] = Field(min_length=1, max_length=16)
    points: int = Field(default=4, ge=1, le=8, strict=True)
    continuations: int = Field(default=8, ge=1, le=100, strict=True)
    continuation_physical_attempts: int = Field(default=64, ge=1, le=10000, strict=True)
    seed: int = Field(default=0, ge=0, le=2**32 - 1, strict=True)

    def identity(self, code_revision: str) -> str:
        """Bind the registration, role prompts and producing code."""
        return stable_id(
            "actor-critic-run",
            {"spec": sha256_json(self), "roles": role_protocol_sha256(), "code": code_revision},
        )


class JudgeDraw(ContractModel):
    """One judge draw and what it earned.

    Attributes:
        draw: Draw index.
        response: Saved response reference, or None when no response exists.
        selection: The validated selection, or None when it was refused or unavailable.
        failure: Why the draw produced no usable selection.
        score: The draw's reward, once its points' continuations are graded.
    """

    draw: int
    response: ArtifactInput | None
    selection: JudgeSelection | None
    failure: str | None
    score: JudgeScore | None = None


class CriticDraw(ContractModel):
    """One critic draw and what it earned.

    Attributes:
        draw: Draw index.
        response: Saved response reference, or None when no response exists.
        assessment: The validated estimates, or None when they were refused or unavailable.
        failure: Why the draw produced no usable estimates.
        score: The draw's reward, once the chosen points' continuations are graded.
    """

    draw: int
    response: ArtifactInput | None
    assessment: CriticAssessment | None
    failure: str | None
    score: CriticScore | None = None


class PointResult(ContractModel):
    """Continuations from one chosen point.

    Attributes:
        point: Point number within the picked attempt.
        checkpoint: The paired actor/world checkpoint continuations started from.
        study: Saved continuation study, or None when it could not run.
        outcome: Graded outcome counts.
        group: Group-relative advantages of the point's continuations.
        failure: Why continuations could not run.
    """

    point: int
    checkpoint: ArtifactInput
    study: ArtifactInput | None
    outcome: PointOutcome
    group: GroupAdvantage
    failure: str | None


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
        ranking: Point numbers ordered from most to least informative.
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


def run_actor_critic(
    project: ProjectStore, spec: ActorCriticSpec, *, code_revision: str
) -> ActorCriticReport:
    """Run, or replay, one actor-critic registration.

    Args:
        project: Project holding every saved step.
        spec: The registration.
        code_revision: Producing source identity.

    Returns:
        The report, also saved as an ``actor-critic-report`` artifact.

    Raises:
        ValueError: The group has no attempt with enough continuable points, or no judge produced
            a usable selection.
    """
    spec = ActorCriticSpec.model_validate_json(spec.model_dump_json())
    store, identity = project.artifacts, spec.identity(code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        result_id = identity + "-result"
        if store.exists(result_id):
            return ActorCriticReport.model_validate_json(store.read_bytes(result_id, "report.json"))
        group = saved_continuation_job(project, spec.initial, code_revision=code_revision)
        if group is None:
            group = execute_continuation_job(project, spec.initial, code_revision=code_revision)
        source = continuation_result_reference(project, spec.initial, code_revision=code_revision)
        loader = cache(lambda ref: load_continuation_checkpoint(store, ref))
        packet = _pick(group, identity, spec, loader)
        sources = (source, *packet.pool.points.values())
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
        points = tuple(_continue(project, spec, packet, point, code_revision) for point in chosen)
        outcomes = tuple(result.outcome for result in points)
        judges = tuple(_reward_judge(draw, outcomes) for draw in judges)
        critics = tuple(_reward_critic(draw, outcomes) for draw in critic_answers)
        drops = decision_drops(outcomes)
        report = ActorCriticReport(
            spec=spec,
            group=source,
            picked=packet.pool.branch,
            judges=judges,
            critics=critics,
            points=points,
            drops=drops,
            audit=_audit(judges, critics),
            ranking=_rank(outcomes, drops),
        )
        dependencies = {source, *(d.response for d in (*judges, *critics) if d.response)}
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
) -> EvidencePacket:
    """Pick the first seeded-shuffled attempt that offers at least two continuable points.

    Args:
        group: The group of attempts.
        identity: Run identity mixed into the seed.
        spec: The registration.
        loader: Cached checkpoint loader.

    Returns:
        The packet for the picked attempt.

    Raises:
        ValueError: No attempt offers two continuable points.
    """
    order = list(candidate_branches(group))
    random.Random(f"{identity}:{spec.seed}").shuffle(order)
    calls = spec.initial.maximum_model_calls
    for branch in order:
        try:
            return evidence_packet(group, branch, maximum_model_calls=calls, checkpoint=loader)
        except ValueError:
            continue
    raise ValueError("no attempt in the group offers two continuable points")


def _judge(
    project: ProjectStore,
    spec: ActorCriticSpec,
    packet: EvidencePacket,
    draw: int,
    identity: str,
    sources: tuple[ArtifactInput, ...],
    code_revision: str,
) -> JudgeDraw:
    """Make or replay one judge draw, keeping a refusal as a failed draw."""
    try:
        reference, selection = draw_judge(
            project,
            spec.judges[draw],
            packet,
            points=spec.points,
            identity=f"{identity}-judge-{draw}",
            sources=sources,
            code_revision=code_revision,
        )
    except ValueError as error:
        return JudgeDraw(draw=draw, response=None, selection=None, failure=str(error)[:2000])
    return JudgeDraw(draw=draw, response=reference, selection=selection, failure=None)


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
        reference, assessment = draw_critic(
            project,
            spec.critics[draw],
            packet,
            chosen,
            identity=f"{identity}-critic-{draw}",
            sources=sources,
            code_revision=code_revision,
        )
    except ValueError as error:
        return CriticDraw(draw=draw, response=None, assessment=None, failure=str(error)[:2000])
    return CriticDraw(draw=draw, response=reference, assessment=assessment, failure=None)


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
    code_revision: str,
) -> PointResult:
    """Run or replay one point's continuations and grade them."""
    checkpoint = packet.pool.points[point]
    job = _point_job(spec, checkpoint)
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
            study=None,
            outcome=empty,
            group=group_advantages(point_id, ()),
            failure=(type(error).__name__ + ": " + str(error))[:2000],
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


def _rank(outcomes: tuple[PointOutcome, ...], drops: dict[str, float]) -> tuple[int, ...]:
    """Order points by uncertainty of the measured rate, then by the drop that follows them."""

    def key(outcome: PointOutcome) -> tuple[float, float, int]:
        """Most informative first; unscored points last."""
        rate = outcome.rate
        uncertainty = -1.0 if rate is None else rate * (1.0 - rate)
        return (-uncertainty, -drops.get(outcome.point_id, 0.0), outcome.step)

    return tuple(outcome.step for outcome in sorted(outcomes, key=key))
