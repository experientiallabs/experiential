"""Closed-loop evaluation: run a harness variant against the world model and score task success.

This is the fitness function the evolutionary manager optimizes. For each task we run the fixed
`AgentRuntime` (driven by one `HarnessSpec`) against a `WorldModelEnvironment` — the world model
answers every tool call instead of a real shell — then the `GoldJudge` scores the resulting run
against the task's gold assertions. Per the repo's eval convention we run **k=3 passes** and report
the mean success rate (never a single pass), which also gives a variance estimate for the manager.

The score is deliberately the same shape whether the underlying environment is real or simulated, so
"does the world model rank harnesses the same as reality?" (docs/sim_real_agreement.md) is a matter
of swapping the environment, not the eval.
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import fmean, pstdev

from pydantic import BaseModel, Field

from wmh.agent.environment import AgentEnvironment, WorldModelEnvironment
from wmh.agent.gold import GoldJudge, GoldVerdict
from wmh.agent.runtime import AgentRuntime, RunResult
from wmh.agent.skills import SkillLibrary
from wmh.agent.spec import HarnessSpec
from wmh.agent.tasks import TaskSpec
from wmh.engine.world_model import WorldModel
from wmh.providers.base import Provider

DEFAULT_K = 3  # eval-reporting convention: every metric is the mean of k passes, never single-pass

# A factory that opens a fresh `AgentEnvironment` for one task. The world-model backend and the real
# E2B backend both fit this shape, which is exactly what lets the SAME scoring core measure a
# harness in simulation and in reality — the substitution the sim-real agreement check turns on.
EnvFactory = Callable[[TaskSpec], AgentEnvironment]


class TaskOutcome(BaseModel):
    """One task's closed-loop result across k passes."""

    task_id: str
    success_rate: float = 0.0  # fraction of k passes that fully passed gold
    mean_fraction: float = 0.0  # mean fraction-of-assertions across passes (partial-credit signal)
    passes: int = 0
    verdicts: list[GoldVerdict] = Field(default_factory=list)


class ClosedLoopReport(BaseModel):
    """A harness variant's closed-loop scorecard over a task suite.

    `success_rate` is the headline fitness (mean over tasks of per-task success rate). `per_task`
    keeps the breakdown so the manager can build an instance-level Pareto frontier (GEPA): a variant
    that wins on some task subset survives even if its aggregate is not best.
    """

    harness: str
    success_rate: float = 0.0  # mean over tasks of per-task pass rate (the fitness scalar)
    mean_fraction: float = 0.0  # mean over tasks of mean assertion-fraction (denser signal)
    success_std: float = 0.0  # spread of per-task success rates (stability of the variant)
    k: int = DEFAULT_K
    per_task: dict[str, TaskOutcome] = Field(default_factory=dict)

    def score_vector(self) -> dict[str, float]:
        """Per-task success rates — the instance-level vector for Pareto comparison."""
        return {tid: o.success_rate for tid, o in self.per_task.items()}


def evaluate_with_env(
    spec: HarnessSpec,
    tasks: list[TaskSpec],
    make_env: EnvFactory,
    agent_provider: Provider,
    judge: GoldJudge,
    *,
    library: SkillLibrary | None = None,
    k: int = DEFAULT_K,
) -> ClosedLoopReport:
    """Score `spec` on `tasks` against whatever environment `make_env` opens, k passes per task.

    Environment-agnostic core shared by simulated (`evaluate_closed_loop`) and real
    (`wmh.agent.real_loop.evaluate_real`) evaluation: only `make_env` differs, so any score gap
    between the two is attributable to sim-vs-real, not to the scoring path.
    """
    per_task: dict[str, TaskOutcome] = {}
    for task in tasks:
        verdicts: list[GoldVerdict] = []
        for _ in range(k):
            result = _run_once(spec, task, make_env, agent_provider, library)
            transcript = result.transcript()
            verdicts.append(judge.score(task.instruction, result.answer, transcript, task.gold))
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
        harness=spec.name,
        success_rate=fmean(task_rates) if task_rates else 0.0,
        mean_fraction=fmean(o.mean_fraction for o in per_task.values()) if per_task else 0.0,
        success_std=pstdev(task_rates) if len(task_rates) > 1 else 0.0,
        k=k,
        per_task=per_task,
    )


def evaluate_closed_loop(
    spec: HarnessSpec,
    tasks: list[TaskSpec],
    world_model: WorldModel,
    agent_provider: Provider,
    judge: GoldJudge,
    *,
    library: SkillLibrary | None = None,
    k: int = DEFAULT_K,
) -> ClosedLoopReport:
    """Score `spec` on `tasks` against `world_model`, k passes per task (the sim fitness fn)."""
    return evaluate_with_env(
        spec,
        tasks,
        lambda task: WorldModelEnvironment(world_model, task=task.instruction),
        agent_provider,
        judge,
        library=library,
        k=k,
    )


def _run_once(
    spec: HarnessSpec,
    task: TaskSpec,
    make_env: EnvFactory,
    agent_provider: Provider,
    library: SkillLibrary | None,
) -> RunResult:
    """One rollout: fresh runtime + a fresh environment (sim or real) driving the agent loop."""
    runtime = AgentRuntime(spec, agent_provider, library=library)
    env = make_env(task)
    try:
        return runtime.run(task.task_id, task.instruction, env)
    finally:
        env.close()


def failing_transcripts(report: ClosedLoopReport, tasks: list[TaskSpec], limit: int = 3) -> str:
    """Summarize the worst tasks + their gold gaps, as reflection fuel for the mutation prompt.

    This is the OpenEvolve "artifacts side-channel" / GEPA reflection input: the mutation LLM reads
    *why* the variant failed (which assertions, on which tasks), not just the scalar delta.
    """
    by_id = {t.task_id: t for t in tasks}
    worst = sorted(report.per_task.values(), key=lambda o: o.success_rate)[:limit]
    blocks: list[str] = []
    for outcome in worst:
        task = by_id.get(outcome.task_id)
        instruction = task.instruction if task else outcome.task_id
        missed = _missed_assertions(outcome)
        blocks.append(
            f"- task {outcome.task_id} (success {outcome.success_rate:.2f}): {instruction}\n"
            f"  unmet: {missed or 'judge could not parse / no assertions'}"
        )
    return "\n".join(blocks)


def _missed_assertions(outcome: TaskOutcome) -> str:
    missed: list[str] = []
    for verdict in outcome.verdicts:
        for a in verdict.assertions:
            if not a.passed and a.assertion not in missed:
                missed.append(a.assertion)
    return "; ".join(missed)
