"""Closed-loop scoring: run a live agent on tasks against an environment, judge task success.

Open-loop eval (`wmh/engine/eval.py`, the default `wmh eval` mode) replays recorded steps
teacher-forced and scores per-step fidelity. This module is the closed-loop counterpart
(`wmh eval --mode closed-loop`): for each task, the agent loop runs to completion (submit or turn
cap) and the `GoldJudge` scores the transcript against the task's gold assertions. Per the repo's
eval convention, every task runs **k=3 passes** and metrics are means over the passes — never
single-pass.

The environment is a factory parameter: `evaluate_closed_loop` binds it to the world model
(`WorldModelEnvironment`), and any real execution backend can bind the same core through the
`AgentEnvironment` protocol, producing a directly comparable report (see
`wmh.evals.agreement`).
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import fmean, pstdev

from pydantic import BaseModel, Field

from wmh.core.types import Action, Observation
from wmh.engine.world_model import WorldModel
from wmh.evals.gold import GoldJudge, GoldVerdict
from wmh.evals.tasks import TaskSpec
from wmh.harness.environment import AgentEnvironment
from wmh.harness.runtime import AgentRuntime, RunResult, Runtime, TokenUsage, combine_usage
from wmh.providers.base import Provider

DEFAULT_K = 3  # eval-reporting convention: every metric is the mean of k passes, never single-pass

# Opens a fresh environment for one task. The world-model backend and any real backend both fit
# this shape, which is what lets the SAME scoring core measure simulation and reality.
EnvFactory = Callable[[TaskSpec], AgentEnvironment]


class WorldModelEnvironment:
    """A simulated environment: actions are answered by the world model, not a real shell.

    Wraps one `WorldModel` session, so the agent loop drives closed-loop eval exactly as it would
    drive a real environment. Sessions are explicitly ended on `close` so batch rollouts don't
    accumulate resident session state in the model.
    """

    def __init__(self, world_model: WorldModel, task: str) -> None:
        self._wm = world_model
        # enrich=False: this rollout's PREDICTED steps must not enter the retrieval buffer, or
        # k=2 retrieves k=1's hallucinations as demos and scores become order-dependent.
        self._session = world_model.new_session(task=task, enrich=False)
        self._closed = False

    @property
    def session_id(self) -> str:
        return self._session.id

    def execute(self, action: Action) -> Observation:
        return self._wm.step(self._session.id, action)

    def close(self) -> None:
        if not self._closed:
            self._wm.end_session(self._session.id)
            self._closed = True


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
    # Aggregate worker-LLM spend from runtimes that meter it themselves (the pi worker path);
    # None when every rollout came from a provider-wrapped runtime (metered upstream).
    worker_usage: TokenUsage | None = None

    @property
    def headline(self) -> float:
        """The `EvalResult` headline: end-to-end task success."""
        return self.success_rate

    def summary(self) -> str:
        return (
            f"success_rate={self.success_rate:.3f}±{self.success_std:.3f} "
            f"assertion_fraction={self.mean_fraction:.3f} "
            f"({len(self.per_task)} tasks, k={self.k})"
        )


def evaluate_with_env(
    tasks: list[TaskSpec],
    make_env: EnvFactory,
    runtime: Runtime,
    judge: GoldJudge,
    *,
    label: str = "",
    k: int = DEFAULT_K,
    concurrency: int = 1,
    on_progress: Callable[[str, int, GoldVerdict], None] | None = None,
) -> ClosedLoopReport:
    """Score the agent on `tasks` against whatever env `make_env` opens, k passes per task.

    `concurrency` is how many (task, attempt) cells run at once: the default 1 keeps the
    sequential loop (world-model behavior unchanged), 0 runs every cell simultaneously (the E2B
    one-sandbox-per-rollout backend), and N>1 caps the pool at N. The report is identical either
    way: verdicts are collected by cell index and aggregated per task in attempt order, and
    `on_progress` always fires from the calling thread so UI callbacks see a serial stream.
    """
    if k < 1:
        raise ValueError("k must be >= 1 (metrics are means over k passes)")
    per_task: dict[str, TaskOutcome] = {}
    usages: list[TokenUsage | None] = []
    if concurrency != 0 and concurrency <= 1:
        for task in tasks:
            verdicts: list[GoldVerdict] = []
            for attempt in range(k):
                result = _run_once(task, make_env, runtime)
                usages.append(result.worker_usage)
                verdict = judge.score(
                    task.instruction, result.answer, result.transcript(), task.gold
                )
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
    else:
        by_cell, usages = _run_cells_concurrently(
            tasks,
            make_env,
            runtime,
            judge,
            k=k,
            concurrency=concurrency,
            on_progress=on_progress,
        )
        for index, task in enumerate(tasks):
            verdicts = by_cell[index * k : (index + 1) * k]  # cells are task-major, attempt-minor
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
        worker_usage=combine_usage(usages),
    )


def evaluate_closed_loop(
    tasks: list[TaskSpec],
    world_model: WorldModel,
    agent_provider: Provider,
    judge: GoldJudge,
    *,
    label: str = "world-model",
    k: int = DEFAULT_K,
    concurrency: int = 1,
    runtime: Runtime | None = None,
    on_progress: Callable[[str, int, GoldVerdict], None] | None = None,
) -> ClosedLoopReport:
    """Score the fixed agent on `tasks` against `world_model` (`wmh eval --mode closed-loop`).

    With `concurrency != 1` the world model steps for many rollouts at once, so the whole eval
    runs under `world_model.frozen()` (the `scenario_fidelity.score_matrix` precedent): sessions
    are already independent (`enrich=False`), and freezing keeps parallel stepping from mutating
    the shared retrieval index mid-eval. Sequential behavior is unchanged.
    """

    def _evaluate() -> ClosedLoopReport:
        return evaluate_with_env(
            tasks,
            lambda task: WorldModelEnvironment(world_model, task=task.instruction),
            runtime if runtime is not None else AgentRuntime(agent_provider),
            judge,
            label=label,
            k=k,
            concurrency=concurrency,
            on_progress=on_progress,
        )

    if concurrency == 1:
        return _evaluate()
    with world_model.frozen():
        return _evaluate()


class ClosedLoopEval:
    """The closed-loop `Evaluation`: a live agent runs tasks with the world model as its env."""

    def __init__(
        self,
        tasks: list[TaskSpec],
        world_model: WorldModel,
        agent_provider: Provider,
        judge: GoldJudge,
        *,
        label: str = "world-model",
        k: int = DEFAULT_K,
        concurrency: int = 1,
        runtime: Runtime | None = None,
        on_progress: Callable[[str, int, GoldVerdict], None] | None = None,
    ) -> None:
        self._tasks = tasks
        self._world_model = world_model
        self._agent_provider = agent_provider
        self._judge = judge
        self._label = label
        self._k = k
        self._concurrency = concurrency
        self._runtime = runtime
        self._on_progress = on_progress

    def run(self) -> ClosedLoopReport:
        return evaluate_closed_loop(
            self._tasks,
            self._world_model,
            self._agent_provider,
            self._judge,
            label=self._label,
            k=self._k,
            concurrency=self._concurrency,
            runtime=self._runtime,
            on_progress=self._on_progress,
        )


def _run_once(task: TaskSpec, make_env: EnvFactory, runtime: Runtime) -> RunResult:
    """One rollout: a fresh environment per attempt, always closed."""
    env = make_env(task)
    try:
        return runtime.run(task.task_id, task.instruction, env)
    finally:
        env.close()


def _run_cells_concurrently(
    tasks: list[TaskSpec],
    make_env: EnvFactory,
    runtime: Runtime,
    judge: GoldJudge,
    *,
    k: int,
    concurrency: int,
    on_progress: Callable[[str, int, GoldVerdict], None] | None,
) -> tuple[list[GoldVerdict], list[TokenUsage | None]]:
    """Run every (task, attempt) cell on a thread pool; verdicts return in cell order.

    Cell order is task-major, attempt-minor — the exact order the sequential loop visits — so the
    caller can slice per task and aggregate deterministically. `on_progress` fires from THIS
    thread as futures land (gepa.py precedent: UI callbacks must be a serial stream). A rollout or
    judge call that raises is a real failure: pending cells are cancelled and the exception
    propagates IMMEDIATELY (fail-fast, scenario_fidelity precedent) — never swallowed into a
    verdict, and never held back while in-flight cells (minutes-long sandbox rollouts) drain:
    the pool is shut down with wait=False, so still-running cells finish on their own threads
    and release their environments through `_run_once`'s finally.
    """
    cells = [(task, attempt) for task in tasks for attempt in range(k)]
    if not cells:
        return [], []
    max_workers = len(cells) if concurrency == 0 else min(concurrency, len(cells))
    slots: list[GoldVerdict | None] = [None] * len(cells)
    usage_slots: list[TokenUsage | None] = [None] * len(cells)

    def run_cell(task: TaskSpec) -> tuple[GoldVerdict, TokenUsage | None]:
        result = _run_once(task, make_env, runtime)
        verdict = judge.score(task.instruction, result.answer, result.transcript(), task.gold)
        return verdict, result.worker_usage

    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {pool.submit(run_cell, task): i for i, (task, _attempt) in enumerate(cells)}
        for future in as_completed(futures):
            index = futures[future]
            verdict, usage = future.result()  # a raised rollout/judge exception propagates here
            slots[index] = verdict
            usage_slots[index] = usage
            if on_progress is not None:
                task, attempt = cells[index]
                on_progress(task.task_id, attempt + 1, verdict)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    verdicts: list[GoldVerdict] = []
    for slot in slots:
        if slot is None:  # pragma: no cover - every future completed, or we raised above
            raise RuntimeError("a cell completed without producing a verdict")
        verdicts.append(slot)
    return verdicts, usage_slots
