"""Bind pipeline mining consent to the exact next scheduled role rollout."""

from exp.common.core.artifacts import sha256_json
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.pipeline_spec import DecisionVarianceSpec, StageContext
from exp.optimize.evaluation.variance.schedule import RolloutPlan, preview_rollout_plan


def scheduled_preview(project: ProjectStore, spec: DecisionVarianceSpec) -> RolloutPlan | None:
    """Preview the matching tier's role rollout, refusing actor-only work before paid stages."""
    schedule = spec.mining.schedule
    if schedule is None or schedule.base.initial.actor_alias not in spec.mining.actors:
        return None
    plan = preview_rollout_plan(project, schedule)
    if plan.spec is None:
        raise ValueError(
            "the schedule's next rollout is actor-only; execute it with the variance-schedule "
            "action, then review a fresh decision-variance plan"
        )
    return plan


def scheduled_line(plan: RolloutPlan) -> str:
    """Describe and bind the precise registration, model revisions and epoch in consent."""
    assert plan.spec is not None
    spec = plan.spec
    points = len(spec.judges) * spec.points
    episodes = spec.initial.study.repeats + points * spec.continuations
    return (
        f"Scheduled rollout {sha256_json(plan)}: epoch {plan.epoch}, rollout {plan.rollout}, "
        f"G={spec.initial.study.repeats}, J={len(spec.judges)}, C={len(spec.critics)}, "
        f"up to {points} points and {episodes} judged episodes; priced from this registration."
    )


def require_scheduled_plan(context: StageContext, plan: RolloutPlan) -> None:
    """Refuse schedule drift from the approved preview before any mining provider call."""
    if context.plan is None or scheduled_line(plan) not in context.plan.lines:
        raise ValueError(
            "the scheduled rollout changed since spend planning; review and approve a fresh plan"
        )
