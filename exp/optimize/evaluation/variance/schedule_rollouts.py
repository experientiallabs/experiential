"""What a training schedule's rollouts run, and the decision evidence its records score.

:func:`run_actor_rollout` runs, or replays, an actor-only rollout (the group job, then its
judging by the frozen episode judge) and :func:`saved_actor_rollout` finds a saved one.

A :class:`~exp.optimize.evaluation.variance.nudge_ladder.CurriculumStage` bound to a qualified
rung task carries that task's job and frozen judge. :func:`curriculum_rollout` turns the current
strength tier's bound stages into the rollout's group job and episode judge: the actor revision
the schedule tracks, started at the rung's checkpoint under the rung's frozen runtime, rotating
over the tier's rungs by rollout number. A tier without bound stages runs the base task, so
diagnostic or unqualified rungs never become rollouts.

:func:`rollout_decisions` gives a version 2 schedule its point decisions: every continued point of
a role rollout's report gets options derived from its judged continuations and each continuation
classified (:func:`~exp.optimize.evaluation.variance.point_options.point_decisions`), the same
evidence the records stage scores selector records with. Each point is saved and replays without
calls, so measuring a completed rollout again buys nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

from exp.common.core.artifacts import ArtifactInput, stable_id
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
    continuation_result_reference,
    execute_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.executable.judged_acceptance import reward_qualification
from exp.optimize.evaluation.executable.task import load_executable_task
from exp.optimize.evaluation.variance import judge as episode_judgment
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport
from exp.optimize.evaluation.variance.judged import (
    JUDGED_LABELS_PROTOCOL,
    JUDGED_STUDY_TYPE,
    EpisodeJudge,
    judge_study,
)
from exp.optimize.evaluation.variance.nudge_ladder import CurriculumStage
from exp.optimize.evaluation.variance.point_options import PointOptionsSpec, point_decisions
from exp.optimize.evaluation.variance.rewards import PointDecision

if TYPE_CHECKING:
    from exp.optimize.evaluation.variance.schedule import RolloutPlan

PointDecisions = Callable[[ActorCriticReport], Mapping[str, PointDecision]]
"""Gives a role rollout's classified choices or family joint score by point identity."""


def validate_curriculum(project: ProjectStore, stages: Sequence[CurriculumStage]) -> None:
    """Reproduce each executable stage's exact reward gate before planning or resuming.

    A judged stage is gated by its judged acceptance, an unjudged one by its witnessed
    qualification without native replay (see
    :func:`~exp.optimize.evaluation.executable.judged_acceptance.reward_qualification`).
    """
    for stage in stages:
        if stage.job is None:
            continue
        if stage.task is None or stage.qualification is None:
            raise ValueError("executable curriculum needs its exact task and qualification")
        task = load_executable_task(project, stage.task)
        gated, profile, qualified, _ = reward_qualification(
            project, stage.qualification, judged=task.judge is not None, allow_native_replay=False
        )
        if (
            not qualified
            or gated != stage.task
            or profile != task.profile()
            or task.job != stage.job
            or task.judge != stage.judge
        ):
            raise ValueError("curriculum differs from its qualified task; regenerate the schedule")


def curriculum_rollout(
    actor: ContinuationJob, stages: Sequence[CurriculumStage], rollout: int
) -> tuple[ContinuationJob, EpisodeJudge, CurriculumStage] | None:
    """Return the rung task one rollout runs, when the current tier has a bound rung.

    Args:
        actor: The actor revision's group job.
        stages: The current strength tier's stages.
        rollout: The rollout number, which picks one bound rung in turn.

    Returns:
        The group job at the rung's checkpoint and runtime under the actor revision, the rung's
        frozen judge and the stage, or None when no stage of the tier is bound to a task.
    """
    bound = [stage for stage in stages if stage.job is not None and stage.judge is not None]
    if not bound:
        return None
    stage = bound[rollout % len(bound)]
    assert stage.job is not None and stage.judge is not None
    study = actor.study.model_copy(
        update={
            "checkpoint": stage.job.study.checkpoint,
            "probe_sha256": stage.job.study.probe_sha256,
            "verifier": stage.job.study.verifier,
            "training_specification": stage.job.study.training_specification,
            "training_evidence": stage.job.study.training_evidence,
            "predicted_success": None,
            "critic_prediction": None,
            "selection_plan": None,
        }
    )
    job = actor.model_copy(
        update={
            "study": study,
            "configuration": stage.job.configuration,
            "runtime_file": stage.job.runtime_file,
            "runtime_sha256": stage.job.runtime_sha256,
        }
    )
    return ContinuationJob.model_validate_json(job.model_dump_json()), stage.judge, stage


def rollout_decisions(
    project: ProjectStore, options: PointOptionsSpec | None, *, code_revision: str
) -> PointDecisions | None:
    """Return the point decisions a version 2 schedule scores its role rollouts with.

    Args:
        project: Project holding the reports and every derived point.
        options: The schedule's option proposer and classifier, or None.
        code_revision: Producing source identity.

    Returns:
        A function deriving, or replaying, every continued point's decision of a report, or
        None without option settings.
    """
    if options is None:
        return None

    def decide(report: ActorCriticReport) -> Mapping[str, PointDecision]:
        """Derive or replay the report's point decisions."""
        return point_decisions(project, options, report, code_revision=code_revision)[0]

    return decide


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
            "assessment_protocol": episode_judgment.JUDGE_PROTOCOL,
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


__all__ = [
    "PointDecisions",
    "curriculum_rollout",
    "rollout_decisions",
    "run_actor_rollout",
    "saved_actor_rollout",
]
