"""The benchmark adapter contract and the capture driver.

An adapter stands up ONE real benchmark: it lists tasks per split, opens a real environment for a
task (a workspace the agent's commands actually execute in), and grades a submission
deterministically. ``run_capture`` drives an agent through every task and assembles graded
Trajectories — the only sanctioned way to produce a trace corpus (real runs, never synthesized
observations).

The ``CommandEnv.execute`` seam is deliberately the smallest possible surface: swap in an
implementation backed by a world model and the same agent loop runs against the WM instead of the
real environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from environment_capture.trajectory import StepRecord, Task, Trajectory


@dataclass(frozen=True)
class ExecResult:
    """What the environment returned for one command."""

    output: str
    returncode: int


@runtime_checkable
class CommandEnv(Protocol):
    """A live environment for one task: execute commands, then release resources."""

    def execute(self, command: str) -> ExecResult: ...

    def close(self) -> None: ...


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """One real benchmark: tasks per split, a real env per task, a deterministic grader."""

    @property
    def name(self) -> str: ...

    def tasks(self, split: str) -> list[Task]: ...

    def open_env(self, task: Task) -> CommandEnv: ...

    def grade(self, task: Task, submission: str) -> float: ...


@dataclass(frozen=True)
class AgentRun:
    """What an agent produced on one task: the real steps taken and its final answer."""

    steps: list[StepRecord]
    final_answer: str
    model: str


class CaptureAgent(Protocol):
    """Anything that can drive a CommandEnv through one task."""

    def run(self, task: Task, env: CommandEnv) -> AgentRun: ...


@dataclass(frozen=True)
class TaskFailure:
    """A task the capture gave up on, with the last error it saw."""

    task_id: str
    error: str


@dataclass(frozen=True)
class CaptureResult:
    """What a capture run produced: graded trajectories plus the tasks it had to skip."""

    trajectories: list[Trajectory]
    failures: list[TaskFailure]


def run_capture(
    adapter: BenchmarkAdapter,
    agent: CaptureAgent,
    *,
    split: str,
    limit: int | None = None,
    tasks: list[Task] | None = None,
    attempts: int = 2,
) -> CaptureResult:
    """Run the agent over the split's tasks against the real environment; return graded runs.

    Pass ``tasks`` to run an explicit subset (e.g. one shard of a multi-model capture); it must
    come from ``adapter.tasks(split)`` for the split label to stay truthful.

    Failures are ISOLATED per task: each task gets up to ``attempts`` tries, and a task that
    still fails is recorded in ``failures`` instead of raising — a multi-hour capture run must
    never lose its completed trajectories to one transient provider/network error.
    """
    trajectories: list[Trajectory] = []
    failures: list[TaskFailure] = []
    for task in (tasks if tasks is not None else adapter.tasks(split))[:limit]:
        last_error = ""
        for _attempt in range(attempts):
            # open_env is part of the isolated unit: stateful backends boot subprocesses here
            # (readiness handshakes time out, venvs break) and file-based ones copy fixtures —
            # a boot failure on one task must not abort the whole capture.
            try:
                env = adapter.open_env(task)
            except Exception as error:  # noqa: BLE001 - isolation is the contract; error recorded
                last_error = f"{type(error).__name__}: {error}"
                continue
            try:
                run = agent.run(task, env)
            except Exception as error:  # noqa: BLE001 - isolation is the contract; error recorded
                last_error = f"{type(error).__name__}: {error}"
                continue
            finally:
                env.close()
            # grade() runs AFTER env.close(): out-of-process backends flush the state graders
            # read only on close. It stays inside the isolated unit — a grader edge case on one
            # submission must not discard the rest of the wave.
            try:
                reward = adapter.grade(task, run.final_answer)
            except Exception as error:  # noqa: BLE001 - isolation is the contract; error recorded
                last_error = f"{type(error).__name__}: {error}"
                continue
            trajectories.append(
                Trajectory(
                    task=task,
                    steps=run.steps,
                    final_answer=run.final_answer,
                    reward=reward,
                    model=run.model,
                    split=split,
                )
            )
            break
        else:
            failures.append(TaskFailure(task_id=task.task_id, error=last_error))
    return CaptureResult(trajectories=trajectories, failures=failures)
