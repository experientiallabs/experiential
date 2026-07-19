"""Offline tests for the trusted WMH pi agent used by Harbor."""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext

import wmh.evals.harbor.agent as mod
from wmh.core.types import JsonObject
from wmh.harness.live_session import SessionEvent
from wmh.harness.pi_runner import (
    PiCandidateError,
    PiCandidateFailureReason,
    PiCandidateFailureStage,
    PiTurnResult,
    pi_node_baseline,
)
from wmh.harness.runner_link import TokenUsage
from wmh.providers.base import ProviderConfig, ProviderKind


class _Provider:
    """Structural tool-calling provider whose network method is never called here."""

    def complete_chat(self, request: object) -> object:
        raise AssertionError(f"unexpected completion request: {request}")


class _Environment:
    """Harbor environment slice used by the tool bridge and trace writer."""

    def __init__(self, results: list[ExecResult] | None = None, *, mounted: bool = True) -> None:
        self.results = list(results or [])
        self.calls: list[tuple[str, dict[str, str] | None, int | None]] = []
        self.uploads: list[tuple[Path | str, str]] = []
        self.capabilities = SimpleNamespace(mounted=mounted)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        _ = cwd, user
        self.calls.append((command, env, timeout_sec))
        return self.results.pop(0) if self.results else ExecResult(return_code=0)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.uploads.append((source_path, target_path))


def _agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> mod.WmhPiAgent:
    monkeypatch.setattr(mod, "get_provider", lambda _config: _Provider())
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="model")
    return mod.WmhPiAgent(
        logs_dir=tmp_path / "agent",
        model_name="bedrock/model",
        harness=cast("JsonObject", pi_node_baseline("candidate").model_dump(mode="json")),
        provider_config=cast("JsonObject", config.model_dump(mode="json")),
    )


def _deadline(seconds: float = 300.0) -> mod.ToolExecutionDeadline:
    return mod.ToolExecutionDeadline.after(seconds)


@pytest.mark.parametrize("turn_timeout_s", [float("nan"), float("inf"), float("-inf")])
def test_agent_rejects_non_finite_turn_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    turn_timeout_s: float,
) -> None:
    monkeypatch.setattr(mod, "get_provider", lambda _config: _Provider())
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="model")

    with pytest.raises(ValueError, match="finite and positive"):
        mod.WmhPiAgent(
            logs_dir=tmp_path / "agent",
            model_name="bedrock/model",
            harness=cast("JsonObject", pi_node_baseline().model_dump(mode="json")),
            provider_config=cast("JsonObject", config.model_dump(mode="json")),
            turn_timeout_s=turn_timeout_s,
        )


def test_agent_rejects_mutable_runner_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "get_provider", lambda _config: _Provider())
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="model")

    with pytest.raises(ValueError, match="digest-qualified"):
        mod.WmhPiAgent(
            logs_dir=tmp_path / "agent",
            model_name="bedrock/model",
            harness=cast("JsonObject", pi_node_baseline().model_dump(mode="json")),
            provider_config=cast("JsonObject", config.model_dump(mode="json")),
            runner_image="node:latest",
        )


def test_agent_rejects_environment_injection_before_task_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "get_provider", lambda _config: _Provider())
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="model")
    secret = "credential-secret-sentinel"

    with pytest.raises(ValueError, match="does not inject agent environment") as caught:
        mod.WmhPiAgent(
            logs_dir=tmp_path / "agent",
            model_name="bedrock/model",
            harness=cast("JsonObject", pi_node_baseline().model_dump(mode="json")),
            provider_config=cast("JsonObject", config.model_dump(mode="json")),
            extra_env={"AWS_SECRET_ACCESS_KEY": secret},
        )

    assert secret not in str(caught.value)


def test_tool_executor_bridges_bash_read_and_write_without_plaintext_content() -> None:
    async def scenario() -> None:
        environment = _Environment(
            [
                ExecResult(stdout="ok\n", return_code=0),
                ExecResult(stderr="missing", return_code=1),
                ExecResult(return_code=0),
            ]
        )
        executor = mod.HarborToolExecutor(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )
        output: list[tuple[str, str]] = []

        def emit(stream: str, text: str) -> None:
            output.append((stream, text))

        bash = await asyncio.to_thread(
            executor,
            "bash",
            cast("JsonObject", {"command": "pwd"}),
            emit,
            _deadline(),
        )
        read = await asyncio.to_thread(
            executor,
            "read_file",
            cast("JsonObject", {"path": "/workspace/missing"}),
            emit,
            _deadline(),
        )
        write = await asyncio.to_thread(
            executor,
            "write_file",
            cast(
                "JsonObject",
                {"path": "/workspace/file.txt", "content": "sensitive body"},
            ),
            emit,
            _deadline(),
        )

        assert bash == mod.ToolOutcome(content="ok\n")
        assert read.is_error is True
        assert write == mod.ToolOutcome(content="wrote /workspace/file.txt")
        write_command, write_env, timeout = environment.calls[2]
        assert "sensitive body" not in write_command
        assert write_env is not None
        assert write_env["BASH_ENV"] == "/dev/null"
        assert write_env["ENV"] == "/dev/null"
        assert write_env["BASHOPTS"] == ""
        assert write_env["SHELLOPTS"] == ""
        assert base64.b64decode(write_env["WMH_FILE_CONTENT_B64"]) == b"sensitive body"
        assert (
            base64.b64decode(write_env["WMH_TOOL_COMMAND_B64"])
            .decode()
            .endswith("> /workspace/file.txt")
        )
        assert all(call[0] == mod._BOUNDED_EXEC_COMMAND for call in environment.calls)
        bash_env = environment.calls[0][1]
        read_env = environment.calls[1][1]
        assert bash_env is not None
        assert read_env is not None
        assert bash_env["BASH_ENV"] == "/dev/null"
        assert bash_env["ENV"] == "/dev/null"
        assert bash_env["BASHOPTS"] == ""
        assert bash_env["SHELLOPTS"] == ""
        assert read_env["BASH_ENV"] == "/dev/null"
        assert read_env["ENV"] == "/dev/null"
        assert read_env["BASHOPTS"] == ""
        assert read_env["SHELLOPTS"] == ""
        assert base64.b64decode(bash_env["WMH_TOOL_COMMAND_B64"]) == b"pwd"
        assert base64.b64decode(read_env["WMH_TOOL_COMMAND_B64"]) == b"cat -- /workspace/missing"
        assert timeout is not None
        assert 0 < timeout <= 300

    asyncio.run(scenario())


def test_tool_executor_neutralizes_shell_startup_environment_before_exec() -> None:
    async def scenario() -> None:
        environment = _Environment([ExecResult(return_code=0)])
        executor = mod.HarborToolExecutor(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )

        await asyncio.to_thread(
            executor._exec,
            "printf safe",
            deadline=_deadline(),
            env={
                "BASH_ENV": "/workspace/poison.sh",
                "ENV": "/workspace/poison.sh",
                "BASHOPTS": "extdebug",
                "SHELLOPTS": "xtrace",
            },
        )

        command, env, _timeout = environment.calls[0]
        assert command.startswith("/bin/bash --noprofile --norc -p -c ")
        assert "/bin/bash --noprofile --norc -p" in command
        assert env is not None
        assert env["BASH_ENV"] == "/dev/null"
        assert env["ENV"] == "/dev/null"
        assert env["BASHOPTS"] == ""
        assert env["SHELLOPTS"] == ""

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("read_file", {"path": ""}, "path must not be empty"),
        ("write_file", {"path": "bad\x00path", "content": "x"}, "null bytes"),
        (
            "bash",
            {"command": "x" * (mod._TOOL_COMMAND_BYTES + 1)},
            "command exceeds",
        ),
        (
            "write_file",
            {"path": "/workspace/large", "content": "x" * (mod._TOOL_CONTENT_BYTES + 1)},
            "content exceeds",
        ),
    ],
)
def test_candidate_tool_argument_poison_is_a_bounded_observation(
    name: str,
    arguments: JsonObject,
    message: str,
) -> None:
    async def scenario() -> None:
        environment = _Environment()
        executor = mod.HarborToolExecutor(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )

        outcome = await asyncio.to_thread(
            executor,
            name,
            arguments,
            lambda *_args: None,
            _deadline(),
        )

        assert outcome.is_error is True
        assert len(outcome.content) <= mod._TOOL_OUTPUT_CHARS
        assert message in outcome.content
        assert environment.calls == []

    asyncio.run(scenario())


def test_tool_output_is_bounded_before_emission_and_structurally_at_exec() -> None:
    async def scenario() -> None:
        environment = _Environment(
            [
                ExecResult(
                    stdout="S" * (mod._TOOL_OUTPUT_CHARS * 10),
                    stderr="E" * (mod._TOOL_OUTPUT_CHARS * 10),
                    return_code=0,
                )
            ]
        )
        executor = mod.HarborToolExecutor(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )
        emitted: list[tuple[str, str]] = []

        outcome = await asyncio.to_thread(
            executor,
            "bash",
            cast("JsonObject", {"command": "printf poison-output"}),
            lambda stream, text: emitted.append((stream, text)),
            _deadline(),
        )

        assert outcome.truncated is True
        assert len(outcome.content) <= mod._TOOL_OUTPUT_CHARS
        assert sum(len(text) for _stream, text in emitted) <= mod._TOOL_OUTPUT_CHARS
        assert all(len(text) <= mod._TOOL_STREAM_CHARS for _stream, text in emitted)
        assert all(mod._TRUNCATION_MARKER in text for _stream, text in emitted)
        command, env, _timeout = environment.calls[0]
        assert command == mod._BOUNDED_EXEC_COMMAND
        assert "head -c" in command
        assert "wc -c" in command
        assert "poison-output" not in command
        assert env is not None
        assert base64.b64decode(env["WMH_TOOL_COMMAND_B64"]).decode() == "printf poison-output"

    asyncio.run(scenario())


def test_candidate_command_deadline_is_a_tool_observation_not_infrastructure() -> None:
    async def scenario() -> None:
        environment = _Environment([ExecResult(stderr="command timed out", return_code=124)])
        executor = mod.HarborToolExecutor(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )

        outcome = await asyncio.to_thread(
            executor,
            "bash",
            cast("JsonObject", {"command": "sleep 600"}),
            lambda *_args: None,
            _deadline(42.9),
        )

        assert outcome.is_error is True
        assert "command timed out" in outcome.content
        assert "[exit 124]" in outcome.content
        command, env, timeout = environment.calls[0]
        assert "timeout --signal=TERM --kill-after=" in command
        assert env is not None
        assert float(env["WMH_TOOL_KILL_AFTER_S"]) > 0
        assert float(env["WMH_TOOL_DEADLINE_S"]) < cast("int", timeout)
        assert cast("int", timeout) <= 42
        assert base64.b64decode(env["WMH_TOOL_COMMAND_B64"]) == b"sleep 600"

    asyncio.run(scenario())


def test_tool_executor_refuses_to_start_without_one_harbor_timeout_tick() -> None:
    async def scenario() -> None:
        environment = _Environment()
        executor = mod.HarborToolExecutor(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )

        with pytest.raises(mod.ToolExecutionDeadlineExceeded):
            await asyncio.to_thread(
                executor,
                "bash",
                cast("JsonObject", {"command": "pwd"}),
                lambda *_args: None,
                _deadline(0.1),
            )

        assert environment.calls == []

    asyncio.run(scenario())


def test_harbor_environment_timeout_before_turn_deadline_remains_infrastructure() -> None:
    class TimedOutEnvironment(_Environment):
        async def exec(
            self,
            command: str,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout_sec: int | None = None,
            user: str | int | None = None,
        ) -> ExecResult:
            await super().exec(command, cwd, env, timeout_sec, user)
            raise TimeoutError("Harbor environment timed out early")

    async def scenario() -> None:
        environment = TimedOutEnvironment()
        executor = mod.HarborToolExecutor(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )

        with pytest.raises(TimeoutError, match="Harbor environment timed out early"):
            await asyncio.to_thread(
                executor,
                "bash",
                cast("JsonObject", {"command": "pwd"}),
                lambda *_args: None,
                _deadline(42),
            )

        assert environment.calls[0][2] is not None
        assert environment.calls[0][2] <= 42

    asyncio.run(scenario())


def test_candidate_failure_returns_for_native_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    failure = PiCandidateError(
        "candidate failed",
        stage=PiCandidateFailureStage.MATERIALIZATION,
        reason=PiCandidateFailureReason.TIMEOUT,
        events=(SessionEvent(kind="error", payload={"message": "candidate failed"}),),
        worker_usage=TokenUsage(input_tokens=3, output_tokens=2, calls=1),
    )

    def fail_candidate(*_args: object, **_kwargs: object) -> PiTurnResult:
        raise failure

    monkeypatch.setattr(mod, "run_pi_turn", fail_candidate)
    context = AgentContext()
    environment = _Environment()

    asyncio.run(agent.run("task", cast("BaseEnvironment", environment), context))

    assert context.n_input_tokens == 3
    assert context.n_output_tokens == 2
    assert context.metadata is not None
    assert context.metadata["candidate_failure"] is True
    assert context.metadata["candidate_failure_stage"] == "materialization"
    assert context.metadata["candidate_failure_reason"] == "timeout"
    trace = (tmp_path / "wmh-events.jsonl").read_text(encoding="utf-8")
    assert "candidate failed" in trace


def test_success_populates_usage_and_backend_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    result = PiTurnResult(
        answer="done",
        terminal_reason="completed",
        events=(SessionEvent(kind="submit", payload={"answer": "done"}),),
        worker_usage=TokenUsage(input_tokens=11, output_tokens=5, calls=1),
    )
    monkeypatch.setattr(mod, "run_pi_turn", lambda *_args, **_kwargs: result)
    context = AgentContext()

    asyncio.run(agent.run("task", cast("BaseEnvironment", _Environment()), context))

    assert context.n_input_tokens == 11
    assert context.n_output_tokens == 5
    assert context.metadata is not None
    assert context.metadata["candidate_failure"] is False
    assert context.metadata["runner_image"] == mod.PI_CONTAINER_IMAGE
    assert context.metadata["harness_hash"] == agent._harness.execution_hash


def test_infrastructure_error_does_not_persist_raw_provider_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    secret = "provider-secret-sentinel"

    def fail(*_args: object, **_kwargs: object) -> PiTurnResult:
        raise RuntimeError(f"provider failed with {secret}")

    monkeypatch.setattr(mod, "run_pi_turn", fail)
    context = AgentContext()

    with pytest.raises(mod.WmhPiRunnerError, match="runner infrastructure failed") as caught:
        asyncio.run(
            agent.run(
                "task",
                cast("BaseEnvironment", _Environment()),
                context,
            )
        )

    assert secret not in str(caught.value)
    assert context.metadata is not None
    assert context.metadata["harness_hash"] == agent._harness.execution_hash
    assert context.metadata["runner_image"] == mod.PI_CONTAINER_IMAGE


def test_second_provider_call_failure_persists_partial_usage_and_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    failure = mod.PiInfrastructureError(
        mod.PiInfrastructureFailureKind.PROVIDER,
        events=(
            SessionEvent(kind="assistant_message", payload={"text": "first answer"}),
            SessionEvent(kind="error", payload={"message": "worker provider unavailable"}),
        ),
        worker_usage=TokenUsage(input_tokens=17, output_tokens=5, calls=1),
    )
    monkeypatch.setattr(
        mod,
        "run_pi_turn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )
    context = AgentContext()

    with pytest.raises(mod.WmhPiProviderError, match="worker provider failed"):
        asyncio.run(agent.run("task", cast("BaseEnvironment", _Environment()), context))

    assert context.n_input_tokens == 17
    assert context.n_output_tokens == 5
    assert context.metadata is not None
    assert context.metadata["infrastructure_failure"] is True
    assert context.metadata["infrastructure_failure_kind"] == "provider"
    trace = (tmp_path / "wmh-events.jsonl").read_text(encoding="utf-8")
    assert "first answer" in trace
    assert "worker provider unavailable" in trace


def test_trace_failure_cannot_skip_or_replace_runner_cleanup_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    secret = "trace-secret-sentinel"
    failure = mod.PiInfrastructureError(
        mod.PiInfrastructureFailureKind.PROVIDER,
        events=(SessionEvent(kind="error", payload={"message": "provider unavailable"}),),
    )

    class UnprovedCleanupFactory:
        cancel_calls = 0
        wait_calls = 0

        def __init__(self, *, image: str) -> None:
            _ = image

        def cancel(self) -> None:
            self.cancel_calls += 1

        def wait_closed(self, timeout_s: float) -> bool:
            _ = timeout_s
            self.wait_calls += 1
            return False

    factory = UnprovedCleanupFactory(image=mod.PI_CONTAINER_IMAGE)
    monkeypatch.setattr(mod, "LocalContainerRunnerFactory", lambda **_kwargs: factory)

    def fail_turn(*_args: object, **_kwargs: object) -> PiTurnResult:
        raise failure

    def fail_trace(_events: tuple[SessionEvent, ...]) -> None:
        raise RuntimeError(secret)

    monkeypatch.setattr(mod, "run_pi_turn", fail_turn)
    monkeypatch.setattr(agent, "_write_trace", fail_trace)

    with pytest.raises(mod.WmhPiCleanupError, match="cleanup was not proved") as caught:
        asyncio.run(agent.run("task", cast("BaseEnvironment", _Environment()), AgentContext()))

    assert secret not in str(caught.value)
    assert factory.cancel_calls == 1
    assert factory.wait_calls == 1


@pytest.mark.parametrize("branch", ["cancellation", "infrastructure"])
@pytest.mark.parametrize("cleanup_failure", ["cancel_exception", "cancel_timeout", "wait_timeout"])
def test_all_failure_branches_sanitize_cancel_and_wait_cleanup_failures(
    branch: str,
    cleanup_failure: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    secret = "cleanup-secret-sentinel"

    class FailingCleanupFactory:
        cancel_calls = 0
        wait_calls = 0

        def __init__(self, *, image: str) -> None:
            _ = image

        def cancel(self) -> None:
            self.cancel_calls += 1
            if cleanup_failure == "cancel_exception":
                raise RuntimeError(f"close failed with {secret}")
            if cleanup_failure == "cancel_timeout":
                time.sleep(0.03)

        def wait_closed(self, timeout_s: float) -> bool:
            _ = timeout_s
            self.wait_calls += 1
            return cleanup_failure != "wait_timeout"

    factory = FailingCleanupFactory(image=mod.PI_CONTAINER_IMAGE)
    monkeypatch.setattr(mod, "LocalContainerRunnerFactory", lambda **_kwargs: factory)
    monkeypatch.setattr(mod, "_RUNNER_CLEANUP_TIMEOUT_S", 0.005)

    def fail(*_args: object, **_kwargs: object) -> PiTurnResult:
        if branch == "cancellation":
            raise asyncio.CancelledError
        raise RuntimeError("runner infrastructure failed")

    monkeypatch.setattr(mod, "run_pi_turn", fail)

    with pytest.raises(mod.WmhPiCleanupError, match="runner cleanup was not proved") as caught:
        asyncio.run(agent.run("task", cast("BaseEnvironment", _Environment()), AgentContext()))

    assert secret not in str(caught.value)
    assert factory.cancel_calls == 1
    assert factory.wait_calls == 1


@pytest.mark.parametrize(
    ("message", "error_type"),
    [
        ("pi turn worker provider failed: secret", mod.WmhPiProviderError),
        ("pi turn tool executor failed for 'bash': secret", mod.WmhPiEnvironmentError),
        ("isolated pi cleanup failed: secret", mod.WmhPiCleanupError),
    ],
)
def test_infrastructure_taxonomy_is_preserved_without_raw_details(
    message: str, error_type: type[RuntimeError]
) -> None:
    classified = mod._typed_infrastructure_error(RuntimeError(message))

    assert isinstance(classified, error_type)
    assert "secret" not in str(classified)


def test_trace_stays_outside_task_mounted_logs_and_replaces_symlink_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")
    trace = tmp_path / "wmh-events.jsonl"
    trace.symlink_to(outside)
    monkeypatch.setattr(
        mod,
        "run_pi_turn",
        lambda *_args, **_kwargs: PiTurnResult(
            answer="",
            terminal_reason="max_turns",
            events=(),
            worker_usage=TokenUsage(),
        ),
    )
    environment = _Environment(mounted=False)

    asyncio.run(agent.run("task", cast("BaseEnvironment", environment), AgentContext()))

    assert environment.uploads == []
    assert outside.read_text(encoding="utf-8") == "keep"
    assert trace.is_symlink() is False
    assert trace.read_text(encoding="utf-8") == ""
    assert not (tmp_path / "agent" / "wmh-events.jsonl").exists()


def test_trace_persistence_failure_is_sanitized_infrastructure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    monkeypatch.setattr(
        mod,
        "run_pi_turn",
        lambda *_args, **_kwargs: PiTurnResult(
            answer="done",
            terminal_reason="completed",
            events=(),
            worker_usage=TokenUsage(),
        ),
    )
    secret = "filesystem-secret-sentinel"

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError(f"host write failed: {secret}")

    monkeypatch.setattr(mod.os, "replace", fail_replace)

    with pytest.raises(mod.WmhPiRunnerError, match="trace persistence") as caught:
        asyncio.run(agent.run("task", cast("BaseEnvironment", _Environment()), AgentContext()))

    assert secret not in str(caught.value)
