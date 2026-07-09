"""Unit tests for the E2B environment backend — a scriptable `FakeSandbox`, no network.

The fakes implement the `SandboxHandle` slice exactly; the only SDK pieces used are the real
exception classes (`CommandExitException`, `NotFoundException`) so normalization is tested
against what the SDK actually raises.
"""

from __future__ import annotations

import time

import pytest
from e2b.exceptions import NotFoundException, RateLimitException
from e2b.sandbox.commands.command_handle import CommandExitException

from wmh.core.types import Action, ActionKind, Observation, Step
from wmh.evals.closed_loop import evaluate_with_env
from wmh.evals.gold import GoldJudge
from wmh.evals.tasks import TaskSpec
from wmh.harness.e2b_env import (
    COMMAND_TIMEOUT_S,
    E2BEnvironment,
    SandboxHandle,
    e2b_env_factory,
)
from wmh.harness.environment import AgentEnvironment
from wmh.harness.runtime import RunResult, StopReason
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind


class _Result:
    """A minimal CommandOutput: what a finished sandbox command reports."""

    def __init__(self, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class _FakeCommands:
    """Records every run; per-command outcomes are scriptable results or exceptions."""

    def __init__(self, outcomes: dict[str, _Result | Exception]) -> None:
        self.calls: list[tuple[str, float | None]] = []
        self._outcomes = outcomes

    def run(
        self,
        cmd: str,
        background: bool | None = None,
        *,
        stdin: bool | None = None,
        timeout: float | None = None,
    ) -> _Result:
        self.calls.append((cmd, timeout))
        outcome = self._outcomes.get(cmd)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome if outcome is not None else _Result(stdout=f"ran: {cmd}")

    def send_stdin(self, pid: int, data: str) -> None:  # pragma: no cover - slice completeness
        raise NotImplementedError


class _FakeFiles:
    """An in-memory filesystem raising the SDK's not-found error for missing paths."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def write(self, path: str, data: str) -> None:
        self.store[path] = data

    def read(self, path: str) -> str:
        if path not in self.store:
            raise NotFoundException(f"path not found: {path}")
        return self.store[path]


class FakeSandbox:
    """A scriptable `SandboxHandle` recording commands, file writes, and kills."""

    def __init__(self, outcomes: dict[str, _Result | Exception] | None = None) -> None:
        self.commands = _FakeCommands(outcomes or {})
        self.files = _FakeFiles()
        self.kills = 0

    def kill(self) -> bool:
        self.kills += 1
        return True


class _SetupTask(TaskSpec):
    """TaskSpec with the real-sandbox `setup` field (spec §1; harmless once TaskSpec has it)."""

    setup: list[str] = []


def _bash(command: str) -> Action:
    return Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": command})


def _tool(name: str, **arguments: str) -> Action:
    return Action(kind=ActionKind.TOOL_CALL, name=name, arguments=dict(arguments))


def _env(fake: FakeSandbox, setup: tuple[str, ...] = ()) -> E2BEnvironment:
    return E2BEnvironment(sandbox_factory=lambda: fake, setup=setup)


def test_bash_success_returns_output_and_exit_code() -> None:
    fake = FakeSandbox()
    obs = _env(fake).execute(_bash("echo hi"))
    assert obs == Observation(content="ran: echo hi", metadata={"exit_code": 0})
    assert fake.commands.calls == [("echo hi", COMMAND_TIMEOUT_S)]


def test_bash_nonzero_exit_normalized_into_error_observation() -> None:
    # The SDK raises on non-zero exit; the environment must observe it, not crash the rollout.
    exc = CommandExitException("err text", "out text", 2, None)
    obs = _env(FakeSandbox(outcomes={"boom": exc})).execute(_bash("boom"))
    assert obs.is_error
    assert obs.content == "out texterr text"
    assert obs.metadata == {"exit_code": 2}


def test_bash_unrelated_exception_propagates() -> None:
    with pytest.raises(TimeoutError):
        _env(FakeSandbox(outcomes={"slow": TimeoutError("120s")})).execute(_bash("slow"))


def test_bash_empty_command_is_error() -> None:
    obs = _env(FakeSandbox()).execute(Action(kind=ActionKind.TOOL_CALL, name="bash"))
    assert obs.is_error


def test_non_env_action_is_error_observation() -> None:
    obs = _env(FakeSandbox()).execute(Action(kind=ActionKind.MESSAGE, content="hello"))
    assert obs.is_error
    assert "unsupported action" in obs.content


def test_write_then_read_file_roundtrip() -> None:
    fake = FakeSandbox()
    env = _env(fake)
    wrote = env.execute(_tool("write_file", path="/home/user/a.txt", content="data"))
    assert not wrote.is_error
    assert fake.files.store == {"/home/user/a.txt": "data"}
    read = env.execute(_tool("read_file", path="/home/user/a.txt"))
    assert read == Observation(content="data")


def test_read_missing_file_is_error_observation() -> None:
    obs = _env(FakeSandbox()).execute(_tool("read_file", path="/nope.txt"))
    assert obs.is_error
    assert "/nope.txt" in obs.content


def test_setup_commands_run_at_construction() -> None:
    fake = FakeSandbox()
    _env(fake, setup=("apt-get install -y jq", "mkdir -p /work"))
    assert [cmd for cmd, _ in fake.commands.calls] == ["apt-get install -y jq", "mkdir -p /work"]


def test_setup_failure_closes_sandbox_and_raises_with_stderr() -> None:
    exc = CommandExitException("npm: not found", "", 127, None)
    fake = FakeSandbox(outcomes={"npm ci": exc})
    with pytest.raises(RuntimeError, match="npm: not found"):
        _env(fake, setup=("npm ci",))
    assert fake.kills == 1


def test_close_is_idempotent_and_best_effort() -> None:
    fake = FakeSandbox()
    env = _env(fake)
    env.close()
    env.close()
    assert fake.kills == 1


def test_factory_injects_sandbox_and_exposes_it() -> None:
    fake = FakeSandbox()
    assert _env(fake).sandbox is fake


def test_satisfies_agent_environment_protocol() -> None:
    assert isinstance(_env(FakeSandbox()), AgentEnvironment)


def test_create_retries_capacity_errors_with_fixed_delays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    fake = FakeSandbox()
    attempts: list[int] = []

    def factory() -> SandboxHandle:
        attempts.append(1)
        if len(attempts) <= 2:
            raise RateLimitException("429 too many requests")
        return fake

    env = E2BEnvironment(sandbox_factory=factory)
    assert env.sandbox is fake
    assert len(attempts) == 3
    assert sleeps == [1.0, 3.0]


def test_create_does_not_retry_non_capacity_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    def factory() -> SandboxHandle:
        raise ValueError("template does not exist")

    with pytest.raises(ValueError, match="template does not exist"):
        E2BEnvironment(sandbox_factory=factory)
    assert sleeps == []


def test_default_factory_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="E2B_API_KEY"):
        E2BEnvironment()


class _NeverProvider:
    """A Provider stub for `GoldJudge`: gold-less tasks pass trivially, so it is never called."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def verify(self):  # noqa: ANN202 - test stub never calls it
        raise NotImplementedError


class _EchoRuntime:
    """A fake Runtime: runs one bash action against whatever environment it is given."""

    def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
        action = _bash("echo hi")
        observation = environment.execute(action)
        step = Step(action=action, observation=observation, task=instruction)
        return RunResult(
            task_id=task_id,
            steps=[step],
            stop_reason=StopReason.SUBMITTED,
            answer=observation.content,
            turns=1,
        )


def test_env_factory_plugs_into_evaluate_with_env_and_maps_task_setup() -> None:
    created: list[FakeSandbox] = []

    def factory() -> SandboxHandle:
        fake = FakeSandbox()
        created.append(fake)
        return fake

    tasks: list[TaskSpec] = [
        _SetupTask(task_id="t1", instruction="say hi", setup=["mkdir -p /work"])
    ]
    report = evaluate_with_env(
        tasks,
        e2b_env_factory(sandbox_factory=factory),
        _EchoRuntime(),
        GoldJudge(_NeverProvider()),
        label="e2b",
        k=2,
    )
    assert report.success_rate == 1.0  # gold-less tasks pass trivially; the loop ran end-to-end
    assert report.per_task["t1"].passes == 2
    assert len(created) == 2  # one fresh sandbox per (task, attempt) cell
    for fake in created:
        assert [cmd for cmd, _ in fake.commands.calls] == ["mkdir -p /work", "echo hi"]
        assert fake.kills == 1  # every rollout closes its environment
