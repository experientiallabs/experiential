"""Closed-loop scoring: run the fixed agent on tasks against an environment, judge task success.

For each task, the agent loop runs to completion (submit or turn cap) and the `GoldJudge` scores the
transcript against the task's gold assertions. Per the repo's eval convention, every task runs
**k=3 passes** and metrics are means over the passes — never single-pass.

The environment is a factory parameter: `evaluate_closed_loop` binds it to the world model (the
`wmh eval closed-loop` path), and any real execution backend can bind the same core, producing a
directly comparable report (see `wmh.harness.agreement`).
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import fmean, pstdev

from pydantic import BaseModel, Field

from wmh.engine.world_model import WorldModel
from wmh.harness.environment import AgentEnvironment, WorldModelEnvironment
from wmh.harness.gold import GoldJudge, GoldVerdict
from wmh.harness.runtime import AgentRuntime, RunResult
from wmh.harness.tasks import TaskSpec
from wmh.providers.base import Provider

DEFAULT_K = 3  # eval-reporting convention: every metric is the mean of k passes, never single-pass

# Opens a fresh environment for one task. The world-model backend and any real backend both fit
# this shape, which is what lets the SAME scoring core measure simulation and reality.
EnvFactory = Callable[[TaskSpec], AgentEnvironment]


class TaskOutcome(BaseModel):
    """One task's closed-loop result across k passes."""

    task_id: str
    success_rate: float = 0.0  # fraction of k passes that fully passed gold
    mean_fraction: float = 0.0  # mean fraction-of-assertions across passes (partial credit)
    passes: int = 0
    verdicts: list[GoldVerdict] = Field(default_factory=list)


class ClosedLoopReport(BaseModel):
    """A closed-loop scorecard over a task suite.

    `label` names what produced the report (a world model name, or a real environment) so two
    reports compared by `compute_agreement` stay identifiable.
    """

    label: str = ""
    success_rate: float = 0.0  # mean over tasks of per-task pass rate
    mean_fraction: float = 0.0  # mean over tasks of mean assertion-fraction (denser signal)
    success_std: float = 0.0  # spread of per-task success rates
    k: int = DEFAULT_K
    per_task: dict[str, TaskOutcome] = Field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"success_rate={self.success_rate:.3f}±{self.success_std:.3f} "
            f"assertion_fraction={self.mean_fraction:.3f} "
            f"({len(self.per_task)} tasks, k={self.k})"
        )


def evaluate_with_env(
    tasks: list[TaskSpec],
    make_env: EnvFactory,
    runtime: AgentRuntime,
    judge: GoldJudge,
    *,
    label: str = "",
    k: int = DEFAULT_K,
    on_progress: Callable[[str, int, GoldVerdict], None] | None = None,
) -> ClosedLoopReport:
    """Score the agent on `tasks` against whatever env `make_env` opens, k passes per task."""
    if k < 1:
        raise ValueError("k must be >= 1 (metrics are means over k passes)")
    per_task: dict[str, TaskOutcome] = {}
    for task in tasks:
        verdicts: list[GoldVerdict] = []
        for attempt in range(k):
            result = _run_once(task, make_env, runtime)
            verdict = judge.score(task.instruction, result.answer, result.transcript(), task.gold)
            verdicts.append(verdict)
            if on_progress is not None:
                on_progress(task.task_id, attempt + 1, verdict)
        successes = [1.0 if v.passed else 0.0 for v in verdicts]
        per_task[task.task_id] = TaskOutcome(
            task_id=task.task_id,
            success_rate=fmean(successes),
            mean_fraction=fmean(v.fraction for v in verdicts),
            passes=k,
            verdicts=verdicts,
        )

    task_rates = [o.success_rate for o in per_task.values()]
    return ClosedLoopReport(
        label=label,
        success_rate=fmean(task_rates) if task_rates else 0.0,
        mean_fraction=fmean(o.mean_fraction for o in per_task.values()) if per_task else 0.0,
        success_std=pstdev(task_rates) if len(task_rates) > 1 else 0.0,
        k=k,
        per_task=per_task,
    )


def evaluate_closed_loop(
    tasks: list[TaskSpec],
    world_model: WorldModel,
    agent_provider: Provider,
    judge: GoldJudge,
    *,
    label: str = "world-model",
    k: int = DEFAULT_K,
    runtime: AgentRuntime | None = None,
    on_progress: Callable[[str, int, GoldVerdict], None] | None = None,
) -> ClosedLoopReport:
    """Score the fixed agent on `tasks` against `world_model` (`wmh eval closed-loop`)."""
    return evaluate_with_env(
        tasks,
        lambda task: WorldModelEnvironment(world_model, task=task.instruction),
        runtime if runtime is not None else AgentRuntime(agent_provider),
        judge,
        label=label,
        k=k,
        on_progress=on_progress,
    )


def _run_once(task: TaskSpec, make_env: EnvFactory, runtime: AgentRuntime) -> RunResult:
    """One rollout: a fresh environment per attempt, always closed."""
    env = make_env(task)
    try:
        return runtime.run(task.task_id, task.instruction, env)
    finally:
        env.close()
