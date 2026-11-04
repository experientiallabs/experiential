"""A persisted, resumable rollout schedule for actor, point-selector and critic training data.

The schedule decides, one rollout at a time, which error-mining registration to run next. It never
trains anything: an external trainer consumes the records the rollouts produce
(:func:`~exp.optimize.evaluation.variance.records.selector_records`, critic and continuation
records) and registers each new frozen model revision it produces back here
(:func:`register_revision`).

- **Warm-up.** Every rollout runs the base registration's full selector and critic draws
  (``J`` and ``C`` of :attr:`TrainingScheduleSpec.base`).
- **Transition.** :class:`TransitionRule` moves to steady state after a number of completed
  warm-up rollouts, after the critic Brier score and the selector reward have both plateaued, or
  either or both (``combine``). Plateau is measured from saved rollout records of the current
  epoch only: the mean of the last ``window`` values against the mean of the ``window`` before.
- **Steady state.** Every ``steady_every``-th steady rollout (every other one by default) runs
  :attr:`TrainingScheduleSpec.steady_draws` (15) selector and critic draws. The rollouts in
  between are actor-only: the group of attempts still runs from the task's start and is judged by
  the frozen episode judge, so each still yields a judged continuation group for the actor.
- **Revisions and epochs.** The schedule tracks the frozen model snapshot of each role (actor,
  selector, critic). Registering a new revision starts a new epoch; rollouts planned in one epoch
  are recorded under it and never count toward, or mix records with, another epoch.

- **Selector objective.** Either version is accepted. Version 2
  (:class:`~exp.optimize.evaluation.variance.rewards.SelectorObjectiveV2`) reads each point's
  classified choices or family joint score from a caller's :data:`PointDecisions` function of the
  saved report; without one its picks are unscored. Version 1 stays the default here, so saved
  schedules keep their identity.
- **Nudge curriculum.** :func:`nudge_curriculum` (re-exported from
  :mod:`~exp.optimize.evaluation.variance.nudge_ladder`) orders the admitted rungs of nudge
  ladders from strong to weak with every un-nudged target last, and :func:`curriculum_tier` gives
  the strength tier a rollout count has reached, so training fades nudges and can compare the
  nudged and un-nudged arms. A schedule given a :class:`CurriculumPlan` (the records stage's
  curriculum) names the current tier's rungs in every rollout plan.

Every decision (a plan, a completion with its measured metrics, a registered revision) is saved
as an immutable event artifact chained to the one before it, so the schedule resumes from the
saved events and a replay reads them without calling any model. Each rollout re-keys the actor
job's study selector with the schedule and rollout number, so every rollout draws fresh attempts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ModelSnapshot
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
    continuation_result_reference,
    execute_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport, ActorCriticSpec
from exp.optimize.evaluation.variance.admission import FamilyAdmissionPolicy
from exp.optimize.evaluation.variance.judged import (
    JUDGED_LABELS_PROTOCOL,
    JUDGED_STUDY_TYPE,
    EpisodeJudge,
    judge_study,
    slot_rewards,
)
from exp.optimize.evaluation.variance.nudge_ladder import (
    CurriculumPlan as CurriculumPlan,
)
from exp.optimize.evaluation.variance.nudge_ladder import (
    CurriculumStage as CurriculumStage,
)
from exp.optimize.evaluation.variance.nudge_ladder import (
    curriculum_tier as curriculum_tier,
)
from exp.optimize.evaluation.variance.nudge_ladder import (
    nudge_curriculum as nudge_curriculum,
)
from exp.optimize.evaluation.variance.records import (
    AnySelectorObjective,
    SelectorRecords,
    load_judged_study,
    selector_records,
)
from exp.optimize.evaluation.variance.rewards import PointDecision, SelectorObjective

SCHEDULE_PROTOCOL = "variance-schedule-v1"
"""Version of the schedule's decision rules; bump when a plan for the same events changes."""

SCHEDULE_TYPE = "variance-schedule"
"""Artifact type of a saved :class:`TrainingScheduleSpec`."""

EVENT_TYPE = "variance-schedule-event"
"""Artifact type of a saved :class:`ScheduleEvent`."""

REPORT_TYPE = "actor-critic-report"
"""Artifact type of a saved actor-critic run."""

Role = Literal["actor", "selector", "critic"]
"""A role whose frozen model revision the schedule tracks."""

RolloutKind = Literal["warmup", "steady_roles", "actor_only"]
"""What one rollout runs."""

Phase = Literal["warmup", "steady"]
"""Where the schedule is."""

PointDecisions = Callable[[ActorCriticReport], Mapping[str, PointDecision]]
"""Gives a role rollout's classified choices or family joint score by point identity."""


class PlateauRule(ContractModel):
    """When the critic Brier score and the selector reward have stopped moving.

    Attributes:
        window: Rollouts per compared window; ``2 * window`` measured rollouts are needed.
        brier_tolerance: Largest change of the mean critic Brier score that counts as flat.
        reward_tolerance: Largest change of the mean selector reward that counts as flat.
    """

    window: int = Field(default=3, ge=1, le=100, strict=True)
    brier_tolerance: float = Field(default=0.01, ge=0.0, le=1.0)
    reward_tolerance: float = Field(default=0.01, ge=0.0, le=2.0)


class TransitionRule(ContractModel):
    """When warm-up ends.

    Attributes:
        warmup_rollouts: Completed warm-up rollouts after which the count condition holds.
        plateau: Plateau condition over the current epoch's saved warm-up records.
        combine: ``any`` moves on when either configured condition holds, ``all`` when both do.
    """

    warmup_rollouts: int | None = Field(default=8, ge=1, le=100000)
    plateau: PlateauRule | None = None
    combine: Literal["any", "all"] = "any"

    @model_validator(mode="after")
    def _configured(self) -> Self:
        """Require at least one condition, so warm-up cannot last forever by omission."""
        if self.warmup_rollouts is None and self.plateau is None:
            raise ValueError("a transition rule needs a warm-up count, a plateau, or both")
        return self


class TrainingScheduleSpec(ContractModel):
    """One frozen training schedule over a single error-mining task.

    Attributes:
        protocol: Schedule protocol.
        name: Readable schedule name.
        base: The warm-up registration; its selector and critic draws are the full draws and
            each role must use one frozen setting for every draw.
        steady_draws: Selector and critic draws of a steady-state role rollout.
        steady_every: Steady rollouts per role rollout; 2 runs roles on every other rollout.
        transition: When warm-up ends.
        objective: The point-selector objective scored for every role rollout, of either
            version; version 1 by default.
        admission: Judge-coverage limits the selector records must meet.
        rewarm_on_revision: Whether a new revision restarts warm-up.
        curriculum: Nudge curriculum each rollout plan names the current tier of, or None;
            left out of the identity while None.
    """

    protocol: Literal["variance-schedule-v1"] = SCHEDULE_PROTOCOL
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")
    base: ActorCriticSpec
    steady_draws: int = Field(default=15, ge=1, le=16, strict=True)
    steady_every: int = Field(default=2, ge=1, le=100, strict=True)
    transition: TransitionRule = TransitionRule()
    objective: AnySelectorObjective = SelectorObjective()
    admission: FamilyAdmissionPolicy = FamilyAdmissionPolicy()
    rewarm_on_revision: bool = False
    curriculum: CurriculumPlan | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def _one_revision_per_role(self) -> Self:
        """Require one frozen setting per role, so each role has exactly one revision."""
        if len(set(self.base.judges)) != 1 or len(set(self.base.critics)) != 1:
            raise ValueError("every selector draw and every critic draw must share one setting")
        return self

    @property
    def identity(self) -> str:
        """The schedule identity, binding the protocol and the complete spec."""
        return stable_id(
            "variance-schedule", {"protocol": self.protocol, "spec": sha256_json(self)}
        )


class RoleModels(ContractModel):
    """The frozen configuration every role runs under in one epoch.

    Attributes:
        actor: The actor's group job (its study names the actor model snapshot).
        selector: The point-selector request settings.
        critic: The critic request settings.
    """

    actor: ContinuationJob
    selector: AuthorSettings
    critic: AuthorSettings

    def snapshot(self, role: Role) -> ModelSnapshot:
        """Return one role's model snapshot."""
        if role == "actor":
            return self.actor.study.actor_model
        return self.selector.model if role == "selector" else self.critic.model

    def revisions(self) -> dict[str, str]:
        """Return the revision identity of every role: the digest of its model snapshot."""
        return {role: sha256_json(self.snapshot(role)) for role in ("actor", "selector", "critic")}


class RevisionRegistration(ContractModel):
    """A new frozen revision of one role, produced by an external trainer.

    Attributes:
        role: The role whose model changed.
        settings: New request settings, for the selector or the critic.
        actor: New group job, for the actor; it must start from the base task checkpoint.
        source: Where the revision came from, such as the external training run.
    """

    role: Role
    settings: AuthorSettings | None = None
    actor: ContinuationJob | None = None
    source: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def _matches_role(self) -> Self:
        """Require exactly the configuration the role uses."""
        if (self.role == "actor") != (self.actor is not None) or (self.role == "actor") == (
            self.settings is not None
        ):
            raise ValueError("an actor revision carries a job; other roles carry settings")
        return self


class RolloutPlan(ContractModel):
    """One planned rollout.

    Attributes:
        rollout: Rollout number across the whole schedule.
        epoch: Epoch the rollout belongs to.
        kind: ``warmup`` and ``steady_roles`` run the selector and critic; ``actor_only`` runs
            only the group of attempts and its judging.
        spec: The registration to run, for a role rollout; None for an actor-only rollout.
        actor_job: The group job; for a role rollout it is ``spec.initial``.
        episode_judge: The frozen judge labelling every episode.
        revisions: Revision identity of every role the rollout runs under.
        reason: Why this rollout was planned this way.
        curriculum: The current nudge-curriculum tier's rungs, when the schedule has one.
    """

    rollout: int = Field(ge=0)
    epoch: int = Field(ge=0)
    kind: RolloutKind
    spec: ActorCriticSpec | None
    actor_job: ContinuationJob
    episode_judge: EpisodeJudge
    revisions: dict[str, Sha256]
    reason: str = Field(min_length=1, max_length=2000)
    curriculum: tuple[CurriculumStage, ...] = Field(default=(), exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def _shape(self) -> Self:
        """Require a registration exactly for role rollouts, running the same group job."""
        if (self.kind == "actor_only") != (self.spec is None):
            raise ValueError("only an actor-only rollout has no registration")
        if self.spec is not None and self.spec.initial != self.actor_job:
            raise ValueError("a role rollout's group job is its registration's initial job")
        return self


class RolloutMetrics(ContractModel):
    """What one completed rollout measured, read from its saved records.

    Attributes:
        critic_brier: Mean Brier score over scored critic draws, or None.
        selector_reward: Mean selector-objective reward over rewarded draws, or None.
        selector_trainable: Selector draws whose record is trainable.
        actor_scored: Judged pass or fail episodes of the group of attempts.
        actor_passes: Judged passes among them.
    """

    critic_brier: float | None = None
    selector_reward: float | None = None
    selector_trainable: int = Field(default=0, ge=0)
    actor_scored: int = Field(default=0, ge=0)
    actor_passes: int = Field(default=0, ge=0)


class ScheduleEvent(ContractModel):
    """One saved schedule decision.

    Attributes:
        protocol: Schedule protocol.
        schedule: Schedule identity.
        sequence: Position in the schedule's event chain.
        previous: The event before it, or None for the first.
        kind: ``plan``, ``complete`` or ``revision``.
        plan: The planned rollout, for ``plan`` and ``complete``.
        result: The saved result, for ``complete``.
        metrics: What the result measured, for ``complete``.
        revision: The registered revision, for ``revision``.
        reason: Why the decision was made.
    """

    protocol: str = SCHEDULE_PROTOCOL
    schedule: str
    sequence: int = Field(ge=0)
    previous: ArtifactInput | None
    kind: Literal["plan", "complete", "revision"]
    plan: RolloutPlan | None = None
    result: ArtifactInput | None = None
    metrics: RolloutMetrics | None = None
    revision: RevisionRegistration | None = None
    reason: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def _payload(self) -> Self:
        """Require exactly the payload the event kind carries."""
        expected = {
            "plan": (True, False, False),
            "complete": (True, True, False),
            "revision": (False, False, True),
        }[self.kind]
        actual = (
            self.plan is not None,
            self.result is not None and self.metrics is not None,
            self.revision is not None,
        )
        if actual != expected:
            raise ValueError(f"a {self.kind} event carries the wrong payload")
        return self


class CompletedRollout(ContractModel):
    """A rollout and its saved result.

    Attributes:
        plan: The plan it ran.
        result: Its saved result: an actor-critic report or a judged group study.
        metrics: What it measured.
    """

    plan: RolloutPlan
    result: ArtifactInput
    metrics: RolloutMetrics


class ScheduleState(ContractModel):
    """The schedule folded from its saved events; derived, never stored.

    Attributes:
        schedule: Schedule identity.
        events: Saved events.
        epoch: Current epoch.
        phase: Current phase.
        models: Role configuration of the current epoch.
        revisions: Revision identity per role in the current epoch.
        pending: The planned rollout not yet completed, if any.
        planned: Rollouts planned so far.
        completed: Every completed rollout, in completion order.
        warmup_since: First epoch of the current warm-up.
        steady_planned: Steady rollouts planned since steady state began.
    """

    schedule: str
    events: int = Field(ge=0)
    epoch: int = Field(ge=0)
    phase: Phase
    models: RoleModels
    revisions: dict[str, Sha256]
    pending: RolloutPlan | None
    planned: int = Field(ge=0)
    completed: tuple[CompletedRollout, ...]
    warmup_since: int = Field(ge=0)
    steady_planned: int = Field(ge=0)

    def epoch_rollouts(self, epoch: int) -> tuple[CompletedRollout, ...]:
        """Return the completed rollouts of one epoch only."""
        return tuple(row for row in self.completed if row.plan.epoch == epoch)


def schedule_state(project: ProjectStore, spec: TrainingScheduleSpec) -> ScheduleState:
    """Fold the schedule's saved events into its state without writing or calling anything.

    Args:
        project: Project holding the schedule.
        spec: The schedule.

    Returns:
        The state; a schedule with no saved events starts in warm-up at epoch zero.

    Raises:
        ValueError: A saved spec or event belongs to another schedule or breaks the chain.
    """
    identity, store = spec.identity, project.artifacts
    if store.exists(identity):
        saved = TrainingScheduleSpec.model_validate_json(store.read_bytes(identity, "spec.json"))
        if saved != spec:
            raise ValueError(f"saved schedule {identity} has another spec")
    models = RoleModels(
        actor=spec.base.initial, selector=spec.base.judges[0], critic=spec.base.critics[0]
    )
    state = ScheduleState(
        schedule=identity,
        events=0,
        epoch=0,
        phase="warmup",
        models=models,
        revisions=models.revisions(),
        pending=None,
        planned=0,
        completed=(),
        warmup_since=0,
        steady_planned=0,
    )
    previous: ArtifactInput | None = None
    while store.exists(event_id := _event_id(identity, state.events)):
        event = ScheduleEvent.model_validate_json(store.read_bytes(event_id, "event.json"))
        if event.schedule != identity or event.sequence != state.events:
            raise ValueError(f"event {event_id} belongs to another schedule or position")
        if event.previous != previous:
            raise ValueError(f"event {event_id} does not follow the saved event before it")
        state = _apply(spec, state, event)
        previous = artifact_input(store.read(event_id).manifest)
    return state


def preview_rollout_plan(project: ProjectStore, spec: TrainingScheduleSpec) -> RolloutPlan:
    """Return the next rollout plan without saving it: the pending one, or the one to decide.

    Args:
        project: Project holding the schedule.
        spec: The schedule.

    Returns:
        The plan :func:`next_rollout_plan` would return.
    """
    state = schedule_state(project, spec)
    return state.pending or _decide(spec, state)


def next_rollout_plan(
    project: ProjectStore, spec: TrainingScheduleSpec, *, code_revision: str
) -> RolloutPlan:
    """Return the rollout to run next, saving the decision; a pending plan is returned again.

    Args:
        project: Project holding the schedule.
        spec: The schedule.
        code_revision: Producing source identity.

    Returns:
        The plan; ``plan.spec`` is the registration to run for a role rollout.
    """
    with _locked(project, spec, code_revision):
        state = schedule_state(project, spec)
        if state.pending is not None:
            return state.pending
        plan = _decide(spec, state)
        _save(project, spec, state, code_revision, kind="plan", plan=plan, reason=plan.reason)
        return plan


def complete_rollout(
    project: ProjectStore,
    spec: TrainingScheduleSpec,
    result: ArtifactInput,
    *,
    code_revision: str,
    decisions: PointDecisions | None = None,
) -> ScheduleState:
    """Record the pending rollout's saved result and what it measured.

    Args:
        project: Project holding the schedule and the result.
        spec: The schedule.
        result: The rollout's saved ``actor-critic-report`` (role rollout) or
            ``judged-continuation-study`` of its group job (actor-only rollout).
        code_revision: Producing source identity.
        decisions: Point decisions of a role rollout, read by a version 2 objective.

    Returns:
        The state after the completion.

    Raises:
        ValueError: No rollout is pending, or the result is not the pending rollout's.
    """
    with _locked(project, spec, code_revision):
        state = schedule_state(project, spec)
        plan = state.pending
        if plan is None:
            raise ValueError("no rollout is pending; plan one first")
        stored = project.artifacts.read(result.artifact_id)
        if artifact_input(stored.manifest) != result:
            raise ValueError(f"result {result.artifact_id} differs from its reference")
        if plan.spec is None:
            metrics = actor_result_metrics(project, plan, result)
        else:
            metrics = role_result_metrics(project, spec, plan, result, decisions)
        reason = f"rollout {plan.rollout} ({plan.kind}) completed in epoch {plan.epoch}"
        return _save(
            project,
            spec,
            state,
            code_revision,
            kind="complete",
            plan=plan,
            result=result,
            metrics=metrics,
            reason=reason,
        )


def register_revision(
    project: ProjectStore,
    spec: TrainingScheduleSpec,
    registration: RevisionRegistration,
    *,
    code_revision: str,
) -> ScheduleState:
    """Register a new frozen revision of one role and start a new epoch.

    Args:
        project: Project holding the schedule.
        spec: The schedule.
        registration: The new revision, produced by an external trainer.
        code_revision: Producing source identity.

    Returns:
        The state in the new epoch.

    Raises:
        ValueError: A rollout is still pending, the snapshot is not new, or a new actor job
            starts from another checkpoint or is judged differently.
    """
    with _locked(project, spec, code_revision):
        state = schedule_state(project, spec)
        if state.pending is not None:
            raise ValueError(
                f"rollout {state.pending.rollout} is pending under the current revisions; "
                "complete it before registering a revision"
            )
        models = _with_revision(state.models, registration)
        if models.revisions()[registration.role] == state.revisions[registration.role]:
            raise ValueError(f"the {registration.role} snapshot is not a new revision")
        if models.actor.study.checkpoint != spec.base.initial.study.checkpoint:
            raise ValueError("a new actor revision must start from the schedule's task checkpoint")
        reason = (
            f"new {registration.role} revision from {registration.source}; epoch "
            f"{state.epoch + 1} starts"
        )
        return _save(
            project,
            spec,
            state,
            code_revision,
            kind="revision",
            revision=registration,
            reason=reason,
        )


def epoch_selector_records(
    project: ProjectStore,
    spec: TrainingScheduleSpec,
    state: ScheduleState,
    epoch: int,
    decisions: PointDecisions | None = None,
) -> tuple[SelectorRecords, ...]:
    """Return the selector records of one epoch's role rollouts, never another epoch's.

    Args:
        project: Project holding the saved reports and assessments.
        spec: The schedule.
        state: The folded schedule.
        epoch: The epoch.
        decisions: Point decisions of each report, read by a version 2 objective.

    Returns:
        One record group per completed role rollout of the epoch, in completion order.
    """
    return tuple(
        _selectors(project, spec, _load_report(project, row.result), decisions)
        for row in state.epoch_rollouts(epoch)
        if row.plan.spec is not None
    )


def role_result_metrics(
    project: ProjectStore,
    spec: TrainingScheduleSpec,
    plan: RolloutPlan,
    result: ArtifactInput,
    decisions: PointDecisions | None = None,
) -> RolloutMetrics:
    """Measure a role rollout from its saved actor-critic report, without any call.

    Args:
        project: Project holding the report and its judged studies.
        spec: The schedule.
        plan: The rollout's plan.
        result: The saved report.
        decisions: Point decisions of the report, read by a version 2 objective.

    Returns:
        Critic Brier score, selector reward and trainable selector draws.

    Raises:
        ValueError: The result is not an actor-critic report of the planned registration.
    """
    report = _load_report(project, result)
    if report.spec != plan.spec:
        raise ValueError("the report ran another registration than the planned rollout")
    rewards = [
        draw.score.reward
        for draw in report.critics
        if draw.score is not None and draw.score.reward is not None
    ]
    selectors = _selectors(project, spec, report, decisions)
    return RolloutMetrics(
        critic_brier=-sum(rewards) / len(rewards) if rewards else None,
        selector_reward=selectors.mean_reward,
        selector_trainable=sum(record.trainable for record in selectors.draws),
    )


def actor_result_metrics(
    project: ProjectStore, plan: RolloutPlan, result: ArtifactInput
) -> RolloutMetrics:
    """Measure an actor-only rollout from its saved judged group study, without any call.

    Args:
        project: Project holding the judged study and its assessments.
        plan: The rollout's plan.
        result: The saved judged study of the group job.

    Returns:
        Judged passes among the scored episodes.

    Raises:
        ValueError: The result is not the planned group job judged by the planned judge.
    """
    judged = load_judged_study(project, result)
    if judged.judge != plan.episode_judge.identity:
        raise ValueError("the group was judged by another judge than the planned one")
    if load_study(project.artifacts, judged.study).spec != plan.actor_job.study:
        raise ValueError("the judged study is not the planned group job")
    labels = [r for r in slot_rewards(project, judged.branches, judged.judge) if r is not None]
    return RolloutMetrics(actor_scored=len(labels), actor_passes=int(sum(labels)))


def saved_actor_rollout(
    project: ProjectStore, plan: RolloutPlan, *, code_revision: str
) -> ArtifactInput | None:
    """Return an actor-only rollout's saved judged group study, when both steps are saved.

    Args:
        project: Project holding the work.
        plan: The actor-only plan.
        code_revision: Producing source identity.

    Returns:
        The saved judged study, or None when running it would still call a model.
    """
    if saved_continuation_job(project, plan.actor_job, code_revision=code_revision) is None:
        return None
    reference = continuation_result_reference(project, plan.actor_job, code_revision=code_revision)
    identity = stable_id(
        JUDGED_STUDY_TYPE,
        {
            "protocol": JUDGED_LABELS_PROTOCOL,
            "judge": plan.episode_judge.identity,
            "study": reference.model_dump(mode="json"),
        },
    )
    store = project.artifacts
    return artifact_input(store.read(identity).manifest) if store.exists(identity) else None


def run_actor_rollout(
    project: ProjectStore, plan: RolloutPlan, *, code_revision: str
) -> ArtifactInput:
    """Run, or replay, an actor-only rollout: the group job, then its judging.

    This buys actor and judge calls for unsaved work; callers ask spend consent first.

    Args:
        project: Project holding the work.
        plan: The actor-only plan.
        code_revision: Producing source identity.

    Returns:
        The saved judged group study, to pass to :func:`complete_rollout`.

    Raises:
        ValueError: The plan is a role rollout.
    """
    if plan.spec is not None:
        raise ValueError("a role rollout runs through run_actor_critic")
    job = plan.actor_job
    if saved_continuation_job(project, job, code_revision=code_revision) is None:
        execute_continuation_job(project, job, code_revision=code_revision)
    reference = continuation_result_reference(project, job, code_revision=code_revision)
    judged, _ = judge_study(
        project,
        reference,
        plan.episode_judge,
        configuration=job.configuration,
        code_revision=code_revision,
    )
    return judged


def plateaued(values: Sequence[float], window: int, tolerance: float) -> bool:
    """Return whether the last ``window`` values' mean moved at most ``tolerance``.

    Args:
        values: Measured values in completion order.
        window: Values per compared window.
        tolerance: Largest change of the window mean that counts as flat.

    Returns:
        False until ``2 * window`` values exist.
    """
    if len(values) < 2 * window:
        return False
    last = values[-window:]
    before = values[-2 * window : -window]
    return abs(sum(last) / window - sum(before) / window) <= tolerance


def _decide(spec: TrainingScheduleSpec, state: ScheduleState) -> RolloutPlan:
    """Decide the next rollout from the folded state."""
    phase, reason = state.phase, ""
    if phase == "warmup":
        moved, why = _transition(spec, state)
        if moved:
            phase, reason = "steady", f"warm-up ends: {why}; "
    if phase == "warmup":
        done = sum(
            row.plan.kind == "warmup" and row.plan.epoch >= state.warmup_since
            for row in state.completed
        )
        kind: RolloutKind = "warmup"
        selectors, critics = len(spec.base.judges), len(spec.base.critics)
        reason = f"warm-up rollout after {done} completed warm-up rollouts, full role draws"
    else:
        index = state.steady_planned if state.phase == "steady" else 0
        roles = index % spec.steady_every == 0
        kind = "steady_roles" if roles else "actor_only"
        selectors = critics = spec.steady_draws
        reason += f"steady rollout {index}: " + (
            f"{spec.steady_draws} selector and critic draws" if roles else "actor only"
        )
    job = _rekeyed(state.models.actor, state.schedule, state.planned)
    registration = None
    if kind != "actor_only":
        seed = int(sha256_json({"schedule": state.schedule, "rollout": state.planned})[:8], 16)
        registration = ActorCriticSpec.model_validate(
            {
                **spec.base.model_dump(mode="json"),
                "initial": job.model_dump(mode="json"),
                "judges": [state.models.selector.model_dump(mode="json")] * selectors,
                "critics": [state.models.critic.model_dump(mode="json")] * critics,
                "seed": seed,
            }
        )
        job = registration.initial
    stages = () if spec.curriculum is None else spec.curriculum.tier(state.planned)
    if stages:
        reason += f"; curriculum tier {stages[0].strength} ({len(stages)} rungs)"
    return RolloutPlan(
        rollout=state.planned,
        epoch=state.epoch,
        kind=kind,
        spec=registration,
        actor_job=job,
        episode_judge=spec.base.episode_judge,
        revisions=state.revisions,
        reason=reason[:2000],
        curriculum=stages,
    )


def _transition(spec: TrainingScheduleSpec, state: ScheduleState) -> tuple[bool, str]:
    """Evaluate the transition rule against saved warm-up records."""
    rule = spec.transition
    held: list[bool] = []
    reasons: list[str] = []
    if rule.warmup_rollouts is not None:
        done = sum(
            row.plan.kind == "warmup" and row.plan.epoch >= state.warmup_since
            for row in state.completed
        )
        held.append(done >= rule.warmup_rollouts)
        reasons.append(f"{done} of {rule.warmup_rollouts} warm-up rollouts completed")
    if rule.plateau is not None:
        rows = [
            row.metrics
            for row in state.epoch_rollouts(state.epoch)
            if row.plan.kind == "warmup" and row.plan.epoch >= state.warmup_since
        ]
        brier = [m.critic_brier for m in rows if m.critic_brier is not None]
        reward = [m.selector_reward for m in rows if m.selector_reward is not None]
        window = rule.plateau.window
        flat = plateaued(brier, window, rule.plateau.brier_tolerance) and plateaued(
            reward, window, rule.plateau.reward_tolerance
        )
        held.append(flat)
        reasons.append(
            f"critic Brier and selector reward {'plateaued' if flat else 'still moving'} over "
            f"{len(brier)} and {len(reward)} epoch-{state.epoch} records (window {window})"
        )
    moved = all(held) if rule.combine == "all" else any(held)
    return moved, "; ".join(reasons)


def _apply(spec: TrainingScheduleSpec, state: ScheduleState, event: ScheduleEvent) -> ScheduleState:
    """Fold one saved event into the state."""
    update: dict[str, object] = {"events": state.events + 1}
    if event.kind == "plan":
        plan = event.plan
        assert plan is not None
        if state.pending is not None or plan.rollout != state.planned or plan.epoch != state.epoch:
            raise ValueError(f"plan event {event.sequence} does not follow the saved state")
        steady = plan.kind != "warmup"
        update |= {
            "pending": plan,
            "planned": state.planned + 1,
            "phase": "steady" if steady else "warmup",
            "steady_planned": state.steady_planned + 1 if steady else 0,
        }
    elif event.kind == "complete":
        if state.pending is None or event.plan != state.pending:
            raise ValueError(f"completion event {event.sequence} completes no pending plan")
        assert event.result is not None and event.metrics is not None
        done = CompletedRollout(plan=state.pending, result=event.result, metrics=event.metrics)
        update |= {"pending": None, "completed": (*state.completed, done)}
    else:
        assert event.revision is not None
        models = _with_revision(state.models, event.revision)
        epoch = state.epoch + 1
        update |= {"epoch": epoch, "models": models, "revisions": models.revisions()}
        if spec.rewarm_on_revision:
            update |= {"phase": "warmup", "warmup_since": epoch, "steady_planned": 0}
    return state.model_copy(update=update)


def _with_revision(models: RoleModels, registration: RevisionRegistration) -> RoleModels:
    """Return the role configuration with one role replaced."""
    if registration.role == "actor":
        return models.model_copy(update={"actor": registration.actor})
    return models.model_copy(update={registration.role: registration.settings})


def _rekeyed(job: ContinuationJob, schedule: str, rollout: int) -> ContinuationJob:
    """Re-key the group job's study selector so the rollout draws fresh attempts."""
    selector = f"{job.study.selector} | schedule {schedule} rollout {rollout}"
    if len(selector) > 2000:
        raise ValueError("the actor job's selector is too long to re-key per rollout")
    study = job.study.model_copy(update={"selector": selector})
    return ContinuationJob.model_validate_json(
        job.model_copy(update={"study": study}).model_dump_json()
    )


def _selectors(
    project: ProjectStore,
    spec: TrainingScheduleSpec,
    report: ActorCriticReport,
    decisions: PointDecisions | None = None,
) -> SelectorRecords:
    """Build a role rollout's selector records under the schedule's objective."""
    return selector_records(
        project,
        report,
        judge=report.spec.episode_judge.identity,
        objective=spec.objective,
        admission=spec.admission,
        decisions=None if decisions is None else decisions(report),
    )


def _load_report(project: ProjectStore, result: ArtifactInput) -> ActorCriticReport:
    """Load a saved actor-critic report, refusing another artifact type or a changed one."""
    stored = project.artifacts.read(result.artifact_id)
    if artifact_input(stored.manifest) != result or stored.manifest.artifact_type != REPORT_TYPE:
        raise ValueError(f"{result.artifact_id} is not the referenced {REPORT_TYPE}")
    return ActorCriticReport.model_validate_json(
        project.artifacts.read_bytes(result.artifact_id, "report.json")
    )


def _event_id(schedule: str, sequence: int) -> str:
    """Return the artifact identity of one event."""
    return f"{schedule}-event-{sequence:06d}"


def _locked(
    project: ProjectStore, spec: TrainingScheduleSpec, code_revision: str
) -> CoordinationLock:
    """Save the spec once and return the schedule's mutation lock."""
    store, identity = project.artifacts, spec.identity
    if not store.exists(identity):
        store.write_json(
            artifact_id=identity,
            artifact_type=SCHEDULE_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=(),
            ),
            files={"spec.json": spec},
        )
    return CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0)


def _save(
    project: ProjectStore,
    spec: TrainingScheduleSpec,
    state: ScheduleState,
    code_revision: str,
    **fields: object,
) -> ScheduleState:
    """Save the next event, chained to the one before it, and return the folded state."""
    store = project.artifacts
    previous = (
        artifact_input(store.read(_event_id(state.schedule, state.events - 1)).manifest)
        if state.events
        else None
    )
    event = ScheduleEvent.model_validate(
        {"schedule": state.schedule, "sequence": state.events, "previous": previous, **fields}
    )
    after = _apply(spec, state, event)
    result = fields.get("result")
    inputs = tuple(ref for ref in (previous, result) if isinstance(ref, ArtifactInput))
    store.write_json(
        artifact_id=_event_id(state.schedule, state.events),
        artifact_type=EVENT_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(sorted(inputs, key=lambda ref: ref.artifact_id)),
        ),
        files={"event.json": event},
    )
    return after
