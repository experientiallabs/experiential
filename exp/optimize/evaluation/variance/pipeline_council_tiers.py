"""Prepare actual model-conditioned references before registering the complete council."""

from __future__ import annotations

from dataclasses import dataclass

from exp.common.core.artifacts import ArtifactInput
from exp.common.project import artifact_input
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.variance.actor_critic import ActorCriticSpec, run_actor_critic
from exp.optimize.evaluation.variance.council_prelude_hook import (
    CouncilPreludeCheckpoint,
    prelude_phase,
)
from exp.optimize.evaluation.variance.episode_evidence import run_task_sha256, run_world_sha256
from exp.optimize.evaluation.variance.episodes import JudgeProfile
from exp.optimize.evaluation.variance.evidence import ReferenceSource
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.judged import EpisodeJudge, judge_study
from exp.optimize.evaluation.variance.pipeline_behaviors import (
    behavior_reference,
    prepare_behavior_reference,
)
from exp.optimize.evaluation.variance.pipeline_errors import error_point_reference
from exp.optimize.evaluation.variance.pipeline_inputs import trace_search
from exp.optimize.evaluation.variance.pipeline_schedule import require_scheduled_plan
from exp.optimize.evaluation.variance.pipeline_spec import BuildOutputs, StageContext, TierMining
from exp.optimize.evaluation.variance.rubric_author import judge_definition
from exp.optimize.evaluation.variance.schedule import complete_rollout, next_rollout_plan
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.retrieval.embedding import RAGEmbedderBinding


@dataclass(frozen=True)
class TierInputs:
    """Exact real upstream evidence and model alias used to prepare one native study."""

    exposure: ExposurePlan
    job: ContinuationJob
    profile: JudgeProfile
    rubric: ArtifactInput
    alias: str
    labels: ArtifactInput | None
    error_points: ArtifactInput | None = None


@dataclass(frozen=True)
class PreparedTier:
    """Actual authored study and behavioral evidence, before any selector/critic execution."""

    registration: ActorCriticSpec
    exposure: ExposurePlan
    alias: str
    behavior: ArtifactInput | None


def prepare_tier(
    context: StageContext,
    tier_inputs: TierInputs,
    build: BuildOutputs,
) -> PreparedTier:
    """Author real behavioral references and then freeze the exact native model study."""
    from exp.optimize.evaluation.variance.pipeline_stages import group_job

    project, spec, revision = context.project, context.spec, context.code_revision
    mining, alias = spec.mining, tier_inputs.alias
    schedule = mining.schedule
    scheduled = schedule is not None and schedule.base.initial.actor_alias == alias
    plan = None
    if scheduled:
        assert schedule is not None
        plan = next_rollout_plan(project, schedule, code_revision=revision)
        require_scheduled_plan(context, plan)
    tier = group_job(project, spec, tier_inputs.job, alias)
    start = load_continuation_checkpoint(project.artifacts, tier.study.checkpoint)
    definition = judge_definition(
        project,
        tier_inputs.profile,
        tier_inputs.rubric,
        task_sha256=run_task_sha256(start),
        world_sha256=run_world_sha256(tier.configuration),
    )
    judge = EpisodeJudge(definition=definition, settings=mining.judge)
    with prelude_phase(
        context,
        CouncilPreludeCheckpoint(
            phase="behavior_reference",
            actor_alias=alias,
            job=tier,
            judge=judge,
            completed_rubric=tier_inputs.rubric,
        ),
    ):
        behavior = prepare_behavior_reference(context, tier, judge, tier_inputs.exposure)
    labels = tier_inputs.labels
    references = (
        *((behavior_reference(project, behavior),) if behavior is not None else ()),
        *(
            ()
            if labels is None
            else (
                ReferenceSource(
                    kind="traceml_labels", name="traceml", artifact=labels, file="labels.json"
                ),
            )
        ),
        *(
            ()
            if tier_inputs.error_points is None
            else (error_point_reference(tier_inputs.error_points),)
        ),
    )
    retrieval = (
        trace_search(project, build, start.task.task_id, mining.retrieval_top_k)
        if mining.retrieval
        else None
    )
    registration = ActorCriticSpec.error_mining(
        tier,
        episode_judge=judge,
        selector=mining.selector,
        critic=mining.critic,
        concurrency=mining.concurrency,
        attempts=mining.attempts,
        selector_draws=mining.selector_draws,
        critic_draws=mining.critic_draws,
        points=mining.points,
        continuations=mining.continuations,
        seed=mining.seed,
        continuation_physical_attempts=mining.continuation_physical_attempts,
        selector_profile=mining.selector_profile,
        critic_profile=mining.critic_profile,
        references=references,
        actor_tier=alias,
    ).model_copy(update={"retrieval": retrieval})
    if scheduled:
        assert plan is not None and plan.spec is not None
        registration = plan.spec
        judge = registration.episode_judge
        tier = registration.initial
    return PreparedTier(registration, tier_inputs.exposure, alias, behavior)


def run_tier(
    context: StageContext,
    prepared: PreparedTier,
    embedder: RAGEmbedderBinding | None,
) -> TierMining:
    """Execute one already registered study, preserving its original native role identities."""
    project, spec, revision = context.project, context.spec, context.code_revision
    registration, alias, behavior = prepared.registration, prepared.alias, prepared.behavior
    tier, judge = registration.initial, registration.episode_judge
    schedule = spec.mining.schedule
    scheduled = schedule is not None and schedule.base.initial.actor_alias == alias
    report = run_actor_critic(
        project,
        registration,
        exposure=prepared.exposure,
        code_revision=revision,
        embedder=embedder if registration.retrieval is not None else None,
        admission=context.dependencies.council_admission,
    )
    saved = artifact_input(
        project.artifacts.read(registration.identity(revision) + "-result").manifest
    )
    if scheduled:
        assert schedule is not None
        complete_rollout(project, schedule, saved, code_revision=revision)
    judged = report.initial_judged
    if judged is None:
        judged, _ = judge_study(
            project,
            report.group,
            judge,
            configuration=tier.configuration,
            code_revision=revision,
        )
    return TierMining(
        tier=alias,
        report=saved,
        judged_group=judged,
        judge=judge,
        job=tier,
        retrieval=registration.retrieval,
        behavior_differences=behavior,
    )
