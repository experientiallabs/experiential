"""Tests for the Harbor bridge that runs an exact WMH harness document."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext

from wmh.core.types import Action, ActionKind
from wmh.evals.harbor.agent import HarborAgentEnvironment, WmhHarborAgent
from wmh.harness.doc import HarnessDoc
from wmh.harness.runtime import RunResult, RuntimeCancelled, StopReason, TokenUsage
from wmh.providers.base import ProviderConfig, ProviderKind


class _Environment:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    async def exec(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        **_kwargs: object,
    ) -> ExecResult:
        self.calls.append((command, env))
        if command == "false":
            return ExecResult(stdout="", stderr="failed\n", return_code=7)
        if command.startswith("cat --"):
            return ExecResult(stdout="contents\n", stderr="", return_code=0)
        return ExecResult(stdout="ok\n", stderr="", return_code=0)


def _provider_config() -> ProviderConfig:
    return ProviderConfig(kind=ProviderKind.BEDROCK, model="worker-model", region="us-west-2")


def test_harbor_environment_routes_supported_tools_and_preserves_command_failures() -> None:
    async def run() -> None:
        environment = _Environment()
        bridge = HarborAgentEnvironment(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )

        bash = await asyncio.to_thread(
            bridge.execute,
            Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "false"}),
        )
        read = await asyncio.to_thread(
            bridge.execute,
            Action(
                kind=ActionKind.TOOL_CALL,
                name="read_file",
                arguments={"path": "notes.txt"},
            ),
        )
        write = await asyncio.to_thread(
            bridge.execute,
            Action(
                kind=ActionKind.TOOL_CALL,
                name="write_file",
                arguments={"path": "out/data.txt", "content": "hello"},
            ),
        )

        assert bash.is_error is True
        assert bash.metadata["return_code"] == 7
        assert "failed" in bash.content
        assert read.content == "contents\n"
        assert write.content == "wrote out/data.txt"
        assert environment.calls[1][0] == "cat -- notes.txt"
        assert environment.calls[2][1] is not None
        assert environment.calls[2][1]["WMH_FILE_CONTENT_B64"] == "aGVsbG8="

    asyncio.run(run())


def test_repeated_cancellation_cannot_detach_abort_drain_or_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    abort_started = threading.Event()
    abort_finished = threading.Event()
    abort_release = threading.Event()
    run_release = threading.Event()
    close_started = threading.Event()
    close_release = threading.Event()
    events: list[str] = []

    class _Runtime:
        def __init__(self, should_cancel: Callable[[], bool]) -> None:
            self.should_cancel = should_cancel

        def run(
            self,
            _task_id: str,
            _instruction: str,
            _environment: HarborAgentEnvironment,
        ) -> RunResult:
            events.append("started")
            started.set()
            while not self.should_cancel():
                time.sleep(0.001)
            events.append("cancel-seen")
            run_release.wait(timeout=2)
            events.append("quiesced")
            raise RuntimeCancelled()

        def abort(self) -> None:
            events.append("abort-started")
            abort_started.set()
            abort_release.wait(timeout=2)
            events.append("abort-finished")
            abort_finished.set()

        def close(self) -> None:
            assert "quiesced" in events
            events.append("close-started")
            close_started.set()
            close_release.wait(timeout=2)
            events.append("closed")

    def runtime(
        _self: HarnessDoc,
        _provider: object,
        *,
        should_cancel: Callable[[], bool],
        **_kwargs: object,
    ) -> _Runtime:
        return _Runtime(should_cancel)

    monkeypatch.setattr("wmh.evals.harbor.agent.get_provider", lambda _config: object())
    monkeypatch.setattr(HarnessDoc, "runtime", runtime)
    agent = WmhHarborAgent(
        logs_dir=tmp_path,
        model_name="bedrock/worker-model",
        harness=HarnessDoc.baseline().model_dump(mode="json"),
        provider_config=_provider_config().model_dump(mode="json"),
    )

    async def run() -> None:
        task = asyncio.create_task(
            agent.run("solve it", cast("BaseEnvironment", _Environment()), AgentContext())
        )
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        assert await asyncio.to_thread(abort_started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        assert task.done() is False

        abort_release.set()
        assert await asyncio.to_thread(abort_finished.wait, 1)
        run_release.set()
        assert await asyncio.to_thread(close_started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        assert task.done() is False

        close_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert events.index("abort-finished") < events.index("quiesced")
        assert events.index("quiesced") < events.index("closed")

    asyncio.run(run())


def test_cancellation_during_close_is_re_raised_after_close_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_started = threading.Event()
    close_release = threading.Event()
    closed = threading.Event()

    class _Runtime:
        def run(
            self,
            task_id: str,
            _instruction: str,
            _environment: HarborAgentEnvironment,
        ) -> RunResult:
            return RunResult(
                task_id=task_id,
                stop_reason=StopReason.SUBMITTED,
                answer="done",
                turns=1,
            )

        def abort(self) -> None:
            raise AssertionError("completed runtime must not be aborted")

        def close(self) -> None:
            close_started.set()
            close_release.wait(timeout=2)
            closed.set()

    monkeypatch.setattr("wmh.evals.harbor.agent.get_provider", lambda _config: object())
    monkeypatch.setattr(HarnessDoc, "runtime", lambda *_args, **_kwargs: _Runtime())
    agent = WmhHarborAgent(
        logs_dir=tmp_path,
        model_name="bedrock/worker-model",
        harness=HarnessDoc.baseline().model_dump(mode="json"),
        provider_config=_provider_config().model_dump(mode="json"),
    )

    async def run() -> None:
        task = asyncio.create_task(
            agent.run("solve it", cast("BaseEnvironment", _Environment()), AgentContext())
        )
        assert await asyncio.to_thread(close_started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        assert task.done() is False
        close_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()

    asyncio.run(run())


def test_harbor_environment_rejects_malformed_or_unknown_tool_calls() -> None:
    async def run() -> None:
        environment = _Environment()
        bridge = HarborAgentEnvironment(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )
        malformed = await asyncio.to_thread(
            bridge.execute,
            Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": 4}),
        )
        unknown = await asyncio.to_thread(
            bridge.execute,
            Action(kind=ActionKind.TOOL_CALL, name="other", arguments={}),
        )

        assert malformed.is_error is True
        assert unknown.is_error is True
        assert environment.calls == []

    asyncio.run(run())


def test_agent_runs_the_exact_candidate_and_persists_its_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = HarnessDoc.baseline("candidate")
    environment = _Environment()
    provider = object()
    observed: dict[str, object] = {}

    class _Runtime:
        def run(
            self,
            task_id: str,
            instruction: str,
            task_environment: HarborAgentEnvironment,
        ) -> RunResult:
            observed["task_id"] = task_id
            observed["instruction"] = instruction
            observation = task_environment.execute(
                Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "pwd"})
            )
            assert observation.content == "ok\n"
            return RunResult(
                task_id=task_id,
                stop_reason=StopReason.SUBMITTED,
                answer="done",
                turns=3,
                worker_usage=TokenUsage(input_tokens=11, output_tokens=7, calls=2),
            )

        def close(self) -> None:
            observed["closed"] = True

    def runtime(
        self: HarnessDoc,
        actual_provider: object,
        *,
        backend: str = "local",
        e2b_template: str | None = None,
        transport_retries: int = 1,
        **_kwargs: object,
    ) -> _Runtime:
        observed["candidate_hash"] = self.doc_hash
        observed["provider"] = actual_provider
        observed["backend"] = backend
        observed["template"] = e2b_template
        observed["transport_retries"] = transport_retries
        return _Runtime()

    monkeypatch.setattr("wmh.evals.harbor.agent.get_provider", lambda _config: provider)
    monkeypatch.setattr(HarnessDoc, "runtime", runtime)
    agent = WmhHarborAgent(
        logs_dir=tmp_path,
        model_name="bedrock/worker-model",
        harness=candidate.model_dump(mode="json"),
        provider_config=_provider_config().model_dump(mode="json"),
        harness_backend="e2b",
        e2b_template="runner-template",
    )
    context = AgentContext()

    asyncio.run(
        agent.run(
            "solve it",
            cast("BaseEnvironment", environment),
            context,
        )
    )

    assert observed == {
        "candidate_hash": candidate.doc_hash,
        "provider": provider,
        "backend": "e2b",
        "template": "runner-template",
        "transport_retries": 0,
        "task_id": observed["task_id"],
        "instruction": "solve it",
        "closed": True,
    }
    assert context.n_input_tokens == 11
    assert context.n_output_tokens == 7
    assert context.metadata == {
        "candidate_doc_hash": candidate.doc_hash,
        "stop_reason": "submitted",
        "turns": 3,
    }
    trace = json.loads((tmp_path / "wmh-run.json").read_text())
    assert trace["task_id"] == observed["task_id"]
    assert trace["answer"] == "done"


def test_local_agent_forces_ssh_transport_despite_ambient_link_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _Runtime:
        def run(
            self,
            task_id: str,
            _instruction: str,
            _environment: HarborAgentEnvironment,
        ) -> RunResult:
            return RunResult(
                task_id=task_id,
                stop_reason=StopReason.SUBMITTED,
                answer="done",
                turns=1,
            )

        def close(self) -> None:
            return None

    def runtime(
        _self: HarnessDoc,
        _provider: object,
        *,
        pi_transport: str | None = None,
        **_kwargs: object,
    ) -> _Runtime:
        observed["pi_transport"] = pi_transport
        return _Runtime()

    monkeypatch.setenv("PI_TRANSPORT", "link")
    monkeypatch.setattr("wmh.evals.harbor.agent.get_provider", lambda _config: object())
    monkeypatch.setattr(HarnessDoc, "runtime", runtime)
    agent = WmhHarborAgent(
        logs_dir=tmp_path,
        model_name="bedrock/worker-model",
        harness=HarnessDoc.baseline().model_dump(mode="json"),
        provider_config=_provider_config().model_dump(mode="json"),
    )

    asyncio.run(
        agent.run(
            "solve it",
            cast("BaseEnvironment", _Environment()),
            AgentContext(),
        )
    )

    assert observed == {"pi_transport": "ssh"}


def test_agent_rejects_secret_environment_injection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="environment variables"):
        WmhHarborAgent(
            logs_dir=tmp_path,
            model_name="bedrock/worker-model",
            harness=HarnessDoc.baseline().model_dump(mode="json"),
            provider_config=_provider_config().model_dump(mode="json"),
            extra_env={"TOKEN": "secret"},
        )


@pytest.mark.parametrize("abort_raises", [False, True])
def test_agent_cancellation_waits_for_runtime_quiescence_before_returning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    abort_raises: bool,
) -> None:
    started = threading.Event()
    release = threading.Event()
    events: list[str] = []

    class _Runtime:
        def __init__(self, should_cancel: Callable[[], bool]) -> None:
            self.should_cancel = should_cancel

        def run(
            self,
            _task_id: str,
            _instruction: str,
            _environment: HarborAgentEnvironment,
        ) -> RunResult:
            events.append("started")
            started.set()
            while not self.should_cancel():
                time.sleep(0.001)
            events.append("cancel-seen")
            release.wait(timeout=2)
            events.append("quiesced")
            raise RuntimeCancelled()

        def abort(self) -> None:
            events.append("abort")
            if abort_raises:
                raise RuntimeError("abort failed")

        def close(self) -> None:
            assert "quiesced" in events
            events.append("closed")

    def runtime(
        _self: HarnessDoc,
        _provider: object,
        *,
        should_cancel: Callable[[], bool],
        **_kwargs: object,
    ) -> _Runtime:
        return _Runtime(should_cancel)

    monkeypatch.setattr("wmh.evals.harbor.agent.get_provider", lambda _config: object())
    monkeypatch.setattr(HarnessDoc, "runtime", runtime)
    agent = WmhHarborAgent(
        logs_dir=tmp_path,
        model_name="bedrock/worker-model",
        harness=HarnessDoc.baseline().model_dump(mode="json"),
        provider_config=_provider_config().model_dump(mode="json"),
    )

    async def run() -> None:
        task = asyncio.create_task(
            agent.run("solve it", cast("BaseEnvironment", _Environment()), AgentContext())
        )
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        for _ in range(1_000):
            if "abort" in events:
                break
            await asyncio.sleep(0.001)
        assert "abort" in events
        assert task.done() is False
        assert "closed" not in events
        release.set()
        if abort_raises:
            with pytest.raises(RuntimeError, match="abort failed"):
                await task
        else:
            with pytest.raises(asyncio.CancelledError):
                await task
        assert events.index("quiesced") < events.index("closed")

    asyncio.run(run())
