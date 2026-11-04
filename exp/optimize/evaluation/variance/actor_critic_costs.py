"""Provider-free cost inventory for bounded actor, selector, critic and judge execution."""

from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable.costs import evaluation_costs, job_costs
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.variance.actor_critic import ActorCriticSpec
from exp.optimize.evaluation.variance.judge_prompt import judge_model_calls


def actor_critic_costs(project: ProjectStore, spec: ActorCriticSpec) -> tuple[CostComponent, ...]:
    """Price an upper bound for the group job, every role draw and every possible point's suffix.

    The group job of ``G`` attempts runs under its own physical attempt ceiling. Each of the
    ``J`` point-selector and ``C`` critic draws may make up to its evidence-tool budget's
    ``model_calls`` requests, each at the role's attempt ceiling, so every role row is scaled by
    that count. Each selector may choose ``K`` distinct points, so up to ``J * K`` points each run
    one suffix job of ``M`` continuations at the continuation attempt ceiling. The frozen episode
    judge assesses each episode with at most :func:`judge_model_calls` requests (its evidence-tool
    budget plus one, plus the plan, criterion-check and self-check draws when agentic), priced
    over ``G + J * K * M`` episodes: the ``J * K * M`` continuations it labels, plus the ``G``
    group attempts so the bound holds however the group is judged. Saved work and assessments
    that replay, unchosen points and draws that answer early make the real cost lower.

    Args:
        project: Project whose catalog prices the roles.
        spec: The registration.

    Returns:
        The complete upper-bound inventory.
    """
    suffix = spec.initial.model_copy(
        update={
            "maximum_physical_attempts": spec.continuation_physical_attempts,
            "study": spec.initial.study.model_copy(update={"repeats": spec.continuations}),
        }
    )
    selector_calls = spec.selector_profile.budget.model_calls
    critic_calls = spec.critic_profile.budget.model_calls
    count = len(spec.judges) * spec.points
    judge = spec.episode_judge
    episodes = spec.initial.study.repeats + count * spec.continuations
    judge_calls = judge_model_calls(judge.definition.profile) * episodes
    return (
        *job_costs(project, spec.initial),
        *(
            row.scaled(selector_calls)
            for role in spec.judges
            for row in evaluation_costs(project, role)
        ),
        *(
            row.scaled(critic_calls)
            for role in spec.critics
            for row in evaluation_costs(project, role)
        ),
        *(row.scaled(count) for row in job_costs(project, suffix)),
        *(row.scaled(judge_calls) for row in evaluation_costs(project, judge.settings)),
    )
