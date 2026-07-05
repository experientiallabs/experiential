"""Tests for the BenchmarkAdapter protocol and the run_capture driver."""

from __future__ import annotations

from environment_capture.adapter import (
    AgentRun,
    BenchmarkAdapter,
    CommandEnv,
    ExecResult,
    run_capture,
)
from environment_capture.trajectory import StepRecord, Task, ToolCall


class _EchoEnv:
    """Deterministic env: echoes the command back; records close()."""

    def __init__(self) -> None:
        self.closed = False

    def execute(self, command: str) -> ExecResult:
        return ExecResult(output=f"ran:{command}", returncode=0)

    def close(self) -> None:
        self.closed = True


class _FakeAdapter:
    name = "echo-bench"

    def __init__(self) -> None:
        self.envs: list[_EchoEnv] = []

    def tasks(self, split: str) -> list[Task]:
        return [Task(task_id=f"{split}-{i}", prompt=f"say {i}", data={}) for i in range(3)]

    def open_env(self, task: Task) -> CommandEnv:
        env = _EchoEnv()
        self.envs.append(env)
        return env

    def grade(self, task: Task, submission: str) -> float:
        return 1.0 if submission == task.prompt.removeprefix("say ") else 0.0


class _OneShotAgent:
    """Runs one command through the env, then answers with the task's digit."""

    def run(self, task: Task, env: CommandEnv) -> AgentRun:
        result = env.execute(f"echo {task.task_id}")
        step = StepRecord(
            action=ToolCall(name="bash", arguments={"command": f"echo {task.task_id}"}),
            output=result.output,
            is_error=result.returncode != 0,
        )
        return AgentRun(steps=[step], final_answer=task.prompt.removeprefix("say "), model="fake")


def test_run_capture_grades_and_assembles_trajectories() -> None:
    adapter = _FakeAdapter()
    assert isinstance(adapter, BenchmarkAdapter)
    result = run_capture(adapter, _OneShotAgent(), split="train")
    assert result.failures == []
    trajectories = result.trajectories
    assert [t.task.task_id for t in trajectories] == ["train-0", "train-1", "train-2"]
    assert all(t.reward == 1.0 for t in trajectories)
    assert all(t.split == "train" for t in trajectories)
    assert trajectories[0].steps[0].output == "ran:echo train-0"
    assert all(env.closed for env in adapter.envs)


def test_run_capture_isolates_task_failures() -> None:
    """One task's crash is recorded as a failure, not raised — a multi-hour capture run must
    never lose completed trajectories to one transient error (this bit three benchmarks)."""
    adapter = _FakeAdapter()

    class _BoomOnMiddleAgent:
        def run(self, task: Task, env: CommandEnv) -> AgentRun:
            if task.task_id == "train-1":
                raise RuntimeError("transient network blip")
            return _OneShotAgent().run(task, env)

    result = run_capture(adapter, _BoomOnMiddleAgent(), split="train")
    assert [t.task.task_id for t in result.trajectories] == ["train-0", "train-2"]
    assert [f.task_id for f in result.failures] == ["train-1"]
    assert "transient network blip" in result.failures[0].error
    assert all(env.closed for env in adapter.envs)  # env released even when the agent dies


def test_run_capture_retries_transient_failures() -> None:
    adapter = _FakeAdapter()

    class _FlakyAgent:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, task: Task, env: CommandEnv) -> AgentRun:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("first call blip")
            return _OneShotAgent().run(task, env)

    result = run_capture(adapter, _FlakyAgent(), split="train", limit=1, attempts=2)
    assert result.failures == []
    assert len(result.trajectories) == 1


def test_run_capture_isolates_grader_failures() -> None:
    """grade() runs outside the agent loop but its crash must ALSO be a recorded failure —
    a grader edge case on one submission must never discard a whole capture wave."""
    adapter = _FakeAdapter()

    class _GraderBombAdapter:
        name = adapter.name

        def tasks(self, split: str) -> list[Task]:
            return adapter.tasks(split)

        def open_env(self, task: Task) -> CommandEnv:
            return adapter.open_env(task)

        def grade(self, task: Task, submission: str) -> float:
            if task.task_id == "train-1":
                raise ValueError("could not convert string to float: '- 1.500'")
            return 1.0

    result = run_capture(_GraderBombAdapter(), _OneShotAgent(), split="train")
    assert [t.task.task_id for t in result.trajectories] == ["train-0", "train-2"]
    assert [f.task_id for f in result.failures] == ["train-1"]
    assert "could not convert" in result.failures[0].error


def test_run_capture_limit() -> None:
    adapter = _FakeAdapter()
    result = run_capture(adapter, _OneShotAgent(), split="train", limit=2)
    assert len(result.trajectories) == 2


def test_run_capture_explicit_task_shard() -> None:
    adapter = _FakeAdapter()
    shard = adapter.tasks("train")[1:]
    result = run_capture(adapter, _OneShotAgent(), split="train", tasks=shard)
    assert [t.task.task_id for t in result.trajectories] == ["train-1", "train-2"]


def test_run_capture_grades_after_env_close() -> None:
    """Backends that flush state on close (out-of-process world servers) are graded only
    after the env has closed — grading a still-open env reads pre-finalization state."""
    adapter = _FakeAdapter()

    class _CloseSensingAdapter:
        name = adapter.name

        def __init__(self) -> None:
            self.env: _EchoEnv | None = None
            self.graded_after_close: list[bool] = []

        def tasks(self, split: str) -> list[Task]:
            return adapter.tasks(split)[:1]

        def open_env(self, task: Task) -> CommandEnv:
            self.env = _EchoEnv()
            return self.env

        def grade(self, task: Task, submission: str) -> float:
            assert self.env is not None
            self.graded_after_close.append(self.env.closed)
            return 1.0

    sensing = _CloseSensingAdapter()
    result = run_capture(sensing, _OneShotAgent(), split="train")
    assert len(result.trajectories) == 1
    assert sensing.graded_after_close == [True]


def test_run_capture_isolates_open_env_failures() -> None:
    """A backend that fails to BOOT for one task (readiness timeout, broken venv, missing
    fixture) is recorded as that task's failure — it must not abort the whole capture."""

    class _BootFailsAdapter(_FakeAdapter):
        def open_env(self, task: Task) -> CommandEnv:
            if task.task_id == "train-1":
                raise RuntimeError("world backend failed to boot")
            return super().open_env(task)

    result = run_capture(_BootFailsAdapter(), _OneShotAgent(), split="train", attempts=2)
    assert [t.task.task_id for t in result.trajectories] == ["train-0", "train-2"]
    assert [f.task_id for f in result.failures] == ["train-1"]
    assert "failed to boot" in result.failures[0].error
