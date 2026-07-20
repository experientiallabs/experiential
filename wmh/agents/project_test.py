"""Tests for running ordinary agents inside a persistent filesystem project."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from llm_waterfall import ChatResponse

from wmh.agents import project as project_module
from wmh.agents.meta import meta_agent
from wmh.agents.project import AgentProject
from wmh.core.types import JsonObject
from wmh.harness import e2b_sandbox as e2b_sandbox_module
from wmh.harness.cost import (
    ProviderCostBinding,
    SearchComponentCostBinding,
    SearchComponentCostRuntime,
    SearchComponentRole,
    SearchCostBinding,
    SearchCostRuntime,
    TimedResourceCostBinding,
)
from wmh.harness.doc import TOOL_POLICY_ID, HarnessDoc
from wmh.harness.e2b_sandbox import SandboxCleanupError, SandboxHandle, SandboxUsage
from wmh.harness.live_session import SessionEvent
from wmh.harness.runtime import HarnessSearchCancelled
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.receipt import ProviderResponseIdentity
from wmh.tracking.budget import (
    BudgetBreachError,
    BudgetExceededError,
    BudgetPolicy,
    BudgetScope,
    ReservationStatus,
    SpendLedger,
    TimedResourceBudgetAccount,
    TimedResourceClass,
    TimedResourceCostMeter,
    TimedResourceRole,
    bind_budget_account,
    bind_timed_resource_account,
    bootstrap_budget_ledger,
)
from wmh.tracking.rate_limit import (
    E2B_SANDBOX_CREATE_RATE_POLICY,
    ExternalDispatchPermit,
    ExternalDispatchRateAuthority,
    ExternalDispatchRatePolicy,
)
from wmh.tracking.tariffs import catalog_provider_token_tariff, provider_cost_meter


class _Files:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def write(self, path: str, data: str) -> object:
        self.values[path] = data
        return None

    def read(
        self,
        path: str,
        *,
        request_timeout: float | None = None,
        gzip: bool = False,
    ) -> str:
        del request_timeout, gzip
        return self.values[path]


def _rate_authority(tmp_path: Path) -> ExternalDispatchRateAuthority:
    return ExternalDispatchRateAuthority.bootstrap(
        (tmp_path / "e2b-create-rate.json").resolve(),
        E2B_SANDBOX_CREATE_RATE_POLICY,
    )


class _Output:
    def __init__(self, *, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class _Commands:
    def __init__(self, files: _Files | None = None) -> None:
        self.runs: list[str] = []
        self.files = files

    def run(self, cmd: str, background: bool | None = None, **kwargs: object) -> _Output:
        del background, kwargs
        self.runs.append(cmd)
        if cmd.startswith("find ") and self.files is not None:
            paths = sorted(self.files.values)
            return _Output(stdout="\0".join(paths) + ("\0" if paths else ""))
        return _Output()

    def send_stdin(self, pid: int, data: str, request_timeout: float | None = None) -> object:
        del pid, data, request_timeout
        return None


class _Sandbox:
    def __init__(self) -> None:
        self.files = _Files()
        self.commands = _Commands(self.files)
        self.killed = False
        self.network_updates: list[dict[str, object]] = []

    def set_timeout(self, timeout: int) -> None:
        del timeout

    def update_network(self, network: dict[str, object]) -> None:
        self.network_updates.append(network)

    def kill(self, request_timeout: float | None = None) -> object:
        del request_timeout
        self.killed = True
        return None


class _FlakyKillSandbox(_Sandbox):
    def __init__(self, *, failures: int) -> None:
        super().__init__()
        self.failures = failures
        self.kill_attempts = 0

    def kill(self, request_timeout: float | None = None) -> object:
        del request_timeout
        self.kill_attempts += 1
        if self.kill_attempts <= self.failures:
            raise RuntimeError("control plane unavailable")
        self.killed = True
        return None


class _Channel:
    def __init__(self) -> None:
        self.inbound: list[JsonObject] = [
            {"type": "state", "status": "idle"},
            {"type": "state", "status": "running"},
            {
                "type": "tool_request",
                "req_id": 1,
                "name": "write_file",
                "arguments": {"path": "/home/user/project/result.txt", "content": "done"},
            },
            {
                "type": "tool_request",
                "req_id": 2,
                "name": "submit",
                "arguments": {"answer": "finished"},
            },
            {"type": "state", "status": "idle", "reason": "completed"},
        ]
        self.sent: list[JsonObject] = []
        self.closed = False

    def send(self, frame: JsonObject) -> None:
        self.sent.append(frame)

    def recv(self, timeout: float | None = None) -> JsonObject | None:
        del timeout
        return self.inbound.pop(0) if self.inbound else None

    def close(self) -> None:
        self.closed = True


class _Provider:
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="worker")

    def complete_chat(self, request: object) -> ChatResponse:
        del request
        raise AssertionError("scripted channel never requests the worker")


class _MeteredProvider(_Provider):
    def complete_chat(self, request: object) -> ChatResponse:
        del request
        return ChatResponse.model_validate(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "working"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            }
        )


class _FailingProvider(_Provider):
    def complete_chat(self, request: object) -> ChatResponse:
        del request
        raise RuntimeError("provider down")


def test_default_project_channel_enables_durable_outbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _Sandbox()
    channel = _Channel()
    captured: dict[str, object] = {}

    def start(sandbox_arg: object, **kwargs: object) -> _Channel:
        captured["sandbox"] = sandbox_arg
        captured.update(kwargs)
        return channel

    monkeypatch.setattr(project_module, "start_live_runner", start)

    assert project_module._start_channel(sandbox, "/project") is channel  # noqa: SLF001
    assert captured == {
        "sandbox": sandbox,
        "workspace": "/project",
        "durable_outbox": True,
    }


def test_project_preserves_files_and_runs_through_live_session() -> None:
    sandbox = _Sandbox()
    channel = _Channel()
    project = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: channel,
        owns_sandbox=False,
    )

    project.write_text("history/round-1.json", "{}")
    result = project.run(meta_agent(), _Provider(), "produce a result", timeout=1)

    assert project.read_text("history/round-1.json") == "{}"
    assert project.read_text("result.txt") == "done"
    assert result.answer == "finished"
    [session_start] = [frame for frame in channel.sent if frame["type"] == "session_start"]
    assert session_start["max_output_tokens"] == meta_agent().max_output_tokens() == 16384
    assert session_start["conversation_scope"] == "turn"
    assert project._session is not None  # noqa: SLF001 - runtime wiring contract
    assert project._session._actions_per_turn == meta_agent().max_turns() == 60  # noqa: SLF001
    assert any(frame["type"] == "user_message" for frame in channel.sent)
    assert channel.closed is False
    assert sandbox.commands.runs[:2] == [
        "mkdir -p /home/user/project /home/user/project.wmh-internal",
        "mkdir -p /home/user/project/history",
    ]
    project.close()
    assert channel.closed is True


def test_private_project_files_are_host_readable_but_agent_inaccessible() -> None:
    sandbox = _Sandbox()
    project = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: _Channel(),
        owns_sandbox=False,
    )

    project.write_private_text("score-archives/holdout.json", '{"secret": true}')

    assert project.read_private_text("score-archives/holdout.json") == '{"secret": true}'
    absolute = "/home/user/project.wmh-internal/score-archives/holdout.json"
    assert sandbox.files.values[absolute] == '{"secret": true}'
    outcome = project._execute_tool(  # noqa: SLF001 - verify the actual tool boundary
        "read_file",
        {"path": absolute},
        lambda stream, data: None,
    )
    assert outcome.is_error is True
    assert "escapes project workspace" in outcome.content


def test_private_project_files_are_reconstructed_by_a_fresh_host_project() -> None:
    sandbox = _Sandbox()
    first = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: _Channel(),
        owns_sandbox=False,
    )
    first.write_private_text("score-archives/record.json", '{"committed": true}')

    reconstructed = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: _Channel(),
        owns_sandbox=False,
    )

    assert reconstructed.read_private_text("score-archives/record.json") == '{"committed": true}'


def test_project_search_state_restores_visible_and_private_roots_without_disclosure() -> None:
    source = AgentProject(
        _Sandbox(),
        channel_factory=lambda sandbox, workspace: _Channel(),
        owns_sandbox=False,
    )
    source.write_text("proposals/iteration-0001/proposal-01.json", '{"public": true}')
    source.write_private_text(
        "score-archives/holdout/manifest.json",
        '{"hidden": true}',
    )
    state = source.export_search_state()

    restored_sandbox = _Sandbox()
    restored = AgentProject(
        restored_sandbox,
        channel_factory=lambda sandbox, workspace: _Channel(),
        owns_sandbox=False,
    )
    restored.restore_search_state(state)

    assert restored.read_text("proposals/iteration-0001/proposal-01.json") == ('{"public": true}')
    assert restored.read_private_text("score-archives/holdout/manifest.json") == (
        '{"hidden": true}'
    )
    assert (
        restored_sandbox.files.values[
            "/home/user/project/proposals/iteration-0001/proposal-01.json"
        ]
        == '{"public": true}'
    )
    assert (
        restored_sandbox.files.values[
            "/home/user/project.wmh-internal/score-archives/holdout/manifest.json"
        ]
        == '{"hidden": true}'
    )
    assert not any(
        value == '{"hidden": true}'
        for path, value in restored_sandbox.files.values.items()
        if path.startswith("/home/user/project/")
    )
    outcome = restored._execute_tool(  # noqa: SLF001 - enforce the actual visibility boundary
        "read_file",
        {"path": ("/home/user/project.wmh-internal/score-archives/holdout/manifest.json")},
        lambda stream, data: None,
    )
    assert outcome.is_error is True
    assert "escapes project workspace" in outcome.content


def test_project_search_state_rejects_noncanonical_paths() -> None:
    project = AgentProject(
        _Sandbox(),
        channel_factory=lambda sandbox, workspace: _Channel(),
        owns_sandbox=False,
    )

    with pytest.raises(ValueError, match="non-canonical"):
        project.restore_search_state(
            {
                "schema_version": "wmh.agent-project-state.v1",
                "visible_files": {"context//iteration.json": "{}"},
                "private_files": {},
            }
        )


def test_project_search_restore_rejects_preexisting_workspace_entries() -> None:
    sandbox = _Sandbox()
    sandbox.files.values["/home/user/project/stale-checkpoint.json"] = '{"stale": true}'
    project = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: _Channel(),
        owns_sandbox=False,
    )

    with pytest.raises(ValueError, match="must be empty"):
        project.restore_search_state(
            {
                "schema_version": "wmh.agent-project-state.v1",
                "visible_files": {},
                "private_files": {},
            }
        )


def test_project_grants_agent_writes_to_exact_files_only() -> None:
    """A turn grant contains agent writes without constraining trusted host writes."""
    sandbox = _Sandbox()
    channel = _Channel()
    channel.inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {
            "type": "tool_request",
            "req_id": 1,
            "name": "write_file",
            "arguments": {
                "path": "/home/user/project/context/round-1/parent.json",
                "content": "poisoned",
            },
        },
        {
            "type": "tool_request",
            "req_id": 2,
            "name": "write_file",
            "arguments": {
                "path": "/home/user/project/proposals/round-1/proposal-01.json",
                "content": "candidate",
            },
        },
        {
            "type": "tool_request",
            "req_id": 3,
            "name": "submit",
            "arguments": {"answer": "done"},
        },
        {"type": "state", "status": "idle", "reason": "completed"},
    ]
    project = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: channel,
        owns_sandbox=False,
    )
    project.write_text("context/round-1/parent.json", "trusted")
    host_write_pending = [True]

    def on_event(event: SessionEvent) -> None:
        if event.kind != "tool_call" or not host_write_pending:
            return
        host_write_pending.clear()
        project.write_text("context/round-1/host-note.txt", "host-authored")

    result = project.run(
        meta_agent(),
        _Provider(),
        "produce one proposal",
        timeout=1,
        on_event=on_event,
        writable_files=["proposals/round-1/proposal-01.json"],
    )

    assert result.answer == "done"
    assert project.read_text("context/round-1/parent.json") == "trusted"
    assert project.read_text("context/round-1/host-note.txt") == "host-authored"
    assert project.read_text("proposals/round-1/proposal-01.json") == "candidate"
    tool_results = [event for event in result.events if event.kind == "tool_result"]
    assert [event.payload["is_error"] for event in tool_results] == [True, False]
    assert "not writable in this project turn" in str(tool_results[0].payload["content"])


def test_project_write_grant_resets_between_turns_on_one_session() -> None:
    """A restricted turn cannot narrow a later backward-compatible unrestricted turn."""
    sandbox = _Sandbox()
    channel = _Channel()
    channel.inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {
            "type": "tool_request",
            "req_id": 1,
            "name": "write_file",
            "arguments": {"path": "memory.txt", "content": "blocked"},
        },
        {
            "type": "tool_request",
            "req_id": 2,
            "name": "submit",
            "arguments": {"answer": "restricted"},
        },
        {"type": "state", "status": "idle", "reason": "completed"},
        {"type": "state", "status": "running"},
        {
            "type": "tool_request",
            "req_id": 3,
            "name": "write_file",
            "arguments": {"path": "memory.txt", "content": "unrestricted"},
        },
        {
            "type": "tool_request",
            "req_id": 4,
            "name": "submit",
            "arguments": {"answer": "second"},
        },
        {"type": "state", "status": "idle", "reason": "completed"},
    ]
    project = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: channel,
        owns_sandbox=False,
    )
    project.write_text("memory.txt", "original")
    agent = meta_agent()
    provider = _Provider()

    first = project.run(agent, provider, "restricted", timeout=1, writable_files=[])
    assert first.answer == "restricted"
    assert project.read_text("memory.txt") == "original"

    second = project.run(agent, provider, "unrestricted", timeout=1)

    assert second.answer == "second"
    assert project.read_text("memory.txt") == "unrestricted"
    assert [frame["type"] for frame in channel.sent].count("session_start") == 1


def test_owned_project_disables_internet_before_the_agent_turn() -> None:
    order: list[str] = []

    class _OrderedSandbox(_Sandbox):
        def update_network(self, network: dict[str, object]) -> None:
            super().update_network(network)
            order.append("network-locked")

    class _OrderedChannel(_Channel):
        def send(self, frame: JsonObject) -> None:
            if frame["type"] == "session_start":
                order.append("session-start")
            super().send(frame)

    sandbox = _OrderedSandbox()
    channel = _OrderedChannel()
    project = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: channel,
    )

    project.run(meta_agent(), _Provider(), "produce a result", timeout=1)

    assert sandbox.network_updates == [{"allow_internet_access": False}]
    assert order[:2] == ["network-locked", "session-start"]
    project.close()


def test_project_retries_one_context_write_after_e2b_disconnect() -> None:
    """A transient context-file transport drop cannot fan out into a failed proposal batch."""

    class _DisconnectOnceCommands(_Commands):
        def __init__(self) -> None:
            super().__init__()
            self.disconnect_next = False
            self.attempts = 0

        def run(self, cmd: str, background: bool | None = None, **kwargs: object) -> _Output:
            self.attempts += 1
            if self.disconnect_next:
                self.disconnect_next = False
                raise RuntimeError("Server disconnected")
            return super().run(cmd, background=background, **kwargs)

    sandbox = _Sandbox()
    commands = _DisconnectOnceCommands()
    sandbox.commands = commands
    project = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: _Channel(),
        sandbox_factory=lambda: pytest.fail("an idempotent write must not replace the sandbox"),
    )
    attempts_before = commands.attempts
    commands.disconnect_next = True

    project.write_text("context/round-0003/parent.json", '{"round": 3}')

    assert commands.attempts - attempts_before == 2
    assert project.read_text("context/round-0003/parent.json") == '{"round": 3}'
    assert project.usage().count == 1
    assert sandbox.killed is False


def test_project_retries_one_context_write_after_closed_http2_connection() -> None:
    """A stale E2B HTTP/2 connection cannot invalidate an entire proposal batch."""

    class _ClosedHttp2OnceCommands(_Commands):
        def __init__(self) -> None:
            super().__init__()
            self.close_next = False
            self.attempts = 0

        def run(self, cmd: str, background: bool | None = None, **kwargs: object) -> _Output:
            self.attempts += 1
            if self.close_next:
                self.close_next = False
                raise RuntimeError(
                    "Invalid input ConnectionInputs.SEND_DATA in state ConnectionState.CLOSED"
                )
            return super().run(cmd, background=background, **kwargs)

    sandbox = _Sandbox()
    commands = _ClosedHttp2OnceCommands()
    sandbox.commands = commands
    project = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: _Channel(),
        sandbox_factory=lambda: pytest.fail("a fresh HTTP/2 request must reuse the project"),
    )
    attempts_before = commands.attempts
    commands.close_next = True

    project.write_text("context/round-0007/parent.json", '{"round": 7}')

    assert commands.attempts - attempts_before == 2
    assert project.read_text("context/round-0007/parent.json") == '{"round": 7}'
    assert project.usage().count == 1
    assert sandbox.killed is False


def test_project_replaces_owned_sandbox_after_repeated_closed_http2_writes() -> None:
    """An exhausted context-write retry reaches the project's bounded sandbox fallback."""

    class _ClosedHttp2Commands(_Commands):
        closed = False

        def run(self, cmd: str, background: bool | None = None, **kwargs: object) -> _Output:
            if self.closed:
                raise RuntimeError(
                    "Invalid input ConnectionInputs.SEND_DATA in state ConnectionState.CLOSED"
                )
            return super().run(cmd, background=background, **kwargs)

    original = _Sandbox()
    original_commands = _ClosedHttp2Commands()
    original.commands = original_commands
    replacement = _Sandbox()
    project = AgentProject(original, sandbox_factory=lambda: replacement)
    project.write_text("history/round-0006.json", '{"kept": true}')
    project.write_private_text("score-archives/holdout.json", '{"secret": true}')
    original_commands.closed = True

    project.write_text("context/round-0007/parent.json", '{"round": 7}')

    assert original.killed is True
    assert replacement.killed is False
    assert project.usage().count == 2
    assert replacement.files.values["/home/user/project/history/round-0006.json"] == (
        '{"kept": true}'
    )
    assert replacement.files.values["/home/user/project/context/round-0007/parent.json"] == (
        '{"round": 7}'
    )
    assert (
        replacement.files.values["/home/user/project.wmh-internal/score-archives/holdout.json"]
        == '{"secret": true}'
    )


def test_context_write_recovery_restarts_an_idle_project_session() -> None:
    """A poisoned next-round write preserves the archive and resumes on a fresh live session."""

    class _ClosedHttp2Commands(_Commands):
        closed = False

        def run(self, cmd: str, background: bool | None = None, **kwargs: object) -> _Output:
            if self.closed:
                raise RuntimeError(
                    "Invalid input ConnectionInputs.SEND_DATA in state ConnectionState.CLOSED"
                )
            return super().run(cmd, background=background, **kwargs)

    original = _Sandbox()
    original_commands = _ClosedHttp2Commands()
    original.commands = original_commands
    replacement = _Sandbox()
    original_channel = _Channel()
    replacement_channel = _Channel()
    project = AgentProject(
        original,
        channel_factory=lambda sandbox, workspace: (
            replacement_channel if sandbox is replacement else original_channel
        ),
        sandbox_factory=lambda: replacement,
    )
    agent = meta_agent()
    provider = _Provider()
    project.write_text("history/round-0006.json", '{"kept": true}')
    first = project.run(agent, provider, "round 6", timeout=1)
    original_commands.closed = True

    project.write_text("context/round-0007/parent.json", '{"round": 7}')
    second = project.run(agent, provider, "round 7", timeout=1)

    assert first.answer == second.answer == "finished"
    assert original_channel.closed is True
    assert replacement_channel.closed is False
    assert original.killed is True
    assert replacement.killed is False
    assert project.usage().count == 2
    assert project.read_text("history/round-0006.json") == '{"kept": true}'
    assert project.read_text("context/round-0007/parent.json") == '{"round": 7}'
    assert [frame["type"] for frame in original_channel.sent].count("user_message") == 1
    assert [frame["type"] for frame in replacement_channel.sent].count("user_message") == 1


def test_project_does_not_retry_non_transport_context_write_failure() -> None:
    """A real filesystem failure still propagates after exactly one attempt."""

    class _FailOnceCommands(_Commands):
        def __init__(self) -> None:
            super().__init__()
            self.fail_next = False
            self.attempts = 0

        def run(self, cmd: str, background: bool | None = None, **kwargs: object) -> _Output:
            self.attempts += 1
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("permission denied")
            return super().run(cmd, background=background, **kwargs)

    sandbox = _Sandbox()
    commands = _FailOnceCommands()
    sandbox.commands = commands
    project = AgentProject(sandbox, channel_factory=lambda sandbox, workspace: _Channel())
    attempts_before = commands.attempts
    commands.fail_next = True

    with pytest.raises(RuntimeError, match="permission denied"):
        project.write_text("context/round-0003/parent.json", '{"round": 3}')

    assert commands.attempts - attempts_before == 1


def test_project_reuses_one_agent_runner_across_fresh_project_turns() -> None:
    sandbox = _Sandbox()
    channel = _Channel()
    channel.inbound.extend(
        [
            {"type": "state", "status": "running"},
            {
                "type": "tool_request",
                "req_id": 3,
                "name": "submit",
                "arguments": {"answer": "second"},
            },
            {"type": "state", "status": "idle", "reason": "completed"},
        ]
    )
    starts = 0

    def channel_factory(sandbox: object, workspace: str) -> _Channel:
        nonlocal starts
        del sandbox, workspace
        starts += 1
        return channel

    project = AgentProject(sandbox, channel_factory=channel_factory, owns_sandbox=False)
    agent = meta_agent()
    provider = _Provider()

    first = project.run(agent, provider, "first turn", timeout=1)
    second = project.run(agent, provider, "second turn", timeout=1)

    assert first.answer == "finished"
    assert second.answer == "second"
    assert starts == 1
    assert [frame["type"] for frame in channel.sent].count("session_start") == 1
    assert [frame["type"] for frame in channel.sent].count("user_message") == 2
    assert (
        next(frame for frame in channel.sent if frame["type"] == "session_start")[
            "conversation_scope"
        ]
        == "turn"
    )


def test_project_surfaces_a_mid_turn_runner_error() -> None:
    sandbox = _Sandbox()
    channel = _Channel()
    channel.inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "episode_error", "note": "worker bridge disconnected"},
    ]
    project = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: channel,
        owns_sandbox=False,
    )

    with pytest.raises(RuntimeError, match="worker bridge disconnected"):
        project.run(meta_agent(), _Provider(), "produce a result", timeout=1)


def test_project_promotes_a_worker_error_after_the_runner_returns_idle() -> None:
    """A provider failure cannot look like a normally completed project turn."""
    sandbox = _Sandbox()
    channel = _Channel()
    channel.inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "llm_request", "req_id": 1, "openai_body": {"messages": []}},
        {"type": "state", "status": "idle", "reason": "completed"},
    ]
    events: list[SessionEvent] = []
    project = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: channel,
        owns_sandbox=False,
    )

    with pytest.raises(
        RuntimeError,
        match="project agent session failed: worker LLM error: provider down",
    ):
        project.run(
            meta_agent(),
            _FailingProvider(),
            "produce a result",
            timeout=1,
            on_event=events.append,
        )

    error = next(event for event in events if event.kind == "error")
    assert error.payload == {"message": "worker LLM error: provider down"}
    response = next(frame for frame in channel.sent if frame["type"] == "llm_response")
    assert response["error"] == "provider down"


def test_project_does_not_retry_a_provider_error_that_looks_like_transport() -> None:
    """Provider text cannot borrow the project transport's retry ownership."""

    class _TransportLookingProvider(_Provider):
        def __init__(self) -> None:
            self.calls = 0

        def complete_chat(self, request: object) -> ChatResponse:
            del request
            self.calls += 1
            raise RuntimeError("Server disconnected without sending a response")

    failed = _Channel()
    failed.inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "llm_request", "req_id": 1, "openai_body": {"messages": []}},
        {"type": "state", "status": "idle", "reason": "completed"},
    ]
    recovered = _Channel()
    channels = iter([failed, recovered])
    starts = 0

    def channel_factory(sandbox: object, workspace: str) -> _Channel:
        nonlocal starts
        del sandbox, workspace
        starts += 1
        return next(channels)

    provider = _TransportLookingProvider()
    project = AgentProject(
        _Sandbox(),
        channel_factory=channel_factory,
        owns_sandbox=False,
    )

    with pytest.raises(
        RuntimeError,
        match="worker LLM error: Server disconnected without sending a response",
    ):
        project.run(meta_agent(), provider, "produce a result", timeout=1)

    assert provider.calls == 1
    assert starts == 1
    assert failed.closed is False


def test_project_ignores_idle_until_the_new_turn_reports_running() -> None:
    """A stale idle frame cannot complete a newly queued project turn."""
    channel = _Channel()
    channel.inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "idle", "reason": "stale_abort"},
        {"type": "state", "status": "running"},
        {
            "type": "tool_request",
            "req_id": 1,
            "name": "submit",
            "arguments": {"answer": "fresh"},
        },
        {"type": "state", "status": "idle", "reason": "completed"},
    ]
    project = AgentProject(
        _Sandbox(),
        channel_factory=lambda sandbox, workspace: channel,
        owns_sandbox=False,
    )

    result = project.run(meta_agent(), _Provider(), "produce a result", timeout=1)

    assert result.answer == "fresh"


def test_project_timeout_retires_the_session_before_the_next_turn() -> None:
    """A late abort boundary cannot leak from a timed-out turn into its successor."""

    class _HangingChannel(_Channel):
        def __init__(self) -> None:
            super().__init__()
            self.inbound = [
                {"type": "state", "status": "idle"},
                {"type": "state", "status": "running"},
            ]

        def recv(self, timeout: float | None = None) -> JsonObject | None:
            if self.inbound:
                return super().recv(timeout)
            raise TimeoutError

    hanging = _HangingChannel()
    recovered = _Channel()
    channels = iter([hanging, recovered])
    starts = 0

    def channel_factory(sandbox: object, workspace: str) -> _Channel:
        nonlocal starts
        del sandbox, workspace
        starts += 1
        return next(channels)

    project = AgentProject(
        _Sandbox(),
        channel_factory=channel_factory,
        owns_sandbox=False,
    )
    agent = meta_agent()
    provider = _Provider()

    with pytest.raises(TimeoutError, match="project agent did not finish"):
        project.run(agent, provider, "timed-out turn", timeout=0.001)
    result = project.run(agent, provider, "next turn", timeout=1)

    assert hanging.closed is True
    assert result.answer == "finished"
    assert starts == 2


def test_project_cancellation_interrupts_and_retires_the_active_session() -> None:
    """Cancellation after one blocking provider pump cannot start a second model call."""

    class _CountingProvider(_MeteredProvider):
        def __init__(self) -> None:
            self.calls = 0

        def complete_chat(self, request: object) -> ChatResponse:
            self.calls += 1
            return super().complete_chat(request)

    channel = _Channel()
    channel.inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "llm_request", "req_id": 1, "openai_body": {"messages": []}},
        {"type": "llm_request", "req_id": 2, "openai_body": {"messages": []}},
    ]
    provider = _CountingProvider()
    project = AgentProject(
        _Sandbox(),
        channel_factory=lambda sandbox, workspace: channel,
        owns_sandbox=False,
    )

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        project.run(
            meta_agent(),
            provider,
            "produce a result",
            timeout=1,
            should_cancel=lambda: provider.calls >= 1,
        )

    assert provider.calls == 1
    assert channel.closed is True
    assert any(
        frame.get("type") == "abort" and frame.get("reason") == "harness_search_cancelled"
        for frame in channel.sent
    )


@pytest.mark.parametrize("reason", ["aborted", "turn_limit"])
def test_project_promotes_unsuccessful_terminal_turn_reasons(reason: str) -> None:
    """A bounded/aborted agent turn cannot masquerade as a completed proposal turn."""
    channel = _Channel()
    channel.inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
        {"type": "state", "status": "idle", "reason": reason},
    ]
    project = AgentProject(
        _Sandbox(),
        channel_factory=lambda sandbox, workspace: channel,
        owns_sandbox=False,
    )

    with pytest.raises(RuntimeError, match=f"turn ended with reason: {reason}"):
        project.run(meta_agent(), _Provider(), "produce a result", timeout=1)


def test_project_restarts_one_live_session_after_transport_disconnect() -> None:
    """A dropped runner stream retries once without replacing project storage."""

    class _DisconnectedChannel(_Channel):
        def __init__(self) -> None:
            super().__init__()
            self.inbound = [
                {"type": "state", "status": "idle"},
                {"type": "state", "status": "running"},
            ]

        def recv(self, timeout: float | None = None) -> JsonObject | None:
            if self.inbound:
                return super().recv(timeout)
            raise RuntimeError("Server disconnected")

    sandbox = _Sandbox()
    disconnected = _DisconnectedChannel()
    recovered = _Channel()
    channels = iter([disconnected, recovered])
    project = AgentProject(
        sandbox,
        channel_factory=lambda sandbox, workspace: next(channels),
        owns_sandbox=False,
    )
    project.write_text("history/round-1.json", '{"kept": true}')

    result = project.run(meta_agent(), _Provider(), "produce a result", timeout=1)

    assert result.answer == "finished"
    assert disconnected.closed is True
    assert recovered.closed is False
    assert project.read_text("history/round-1.json") == '{"kept": true}'
    assert [frame["type"] for frame in disconnected.sent].count("user_message") == 1
    assert [frame["type"] for frame in recovered.sent].count("user_message") == 1


def test_project_retries_a_transient_channel_send_failure() -> None:
    """E2B stdin timeouts are transport failures even after LiveSession stringifies them."""

    class _SendFailureChannel(_Channel):
        def send(self, frame: JsonObject) -> None:
            if frame.get("type") == "user_message":
                raise RuntimeError("request timed out")
            super().send(frame)

    failed = _SendFailureChannel()
    failed.inbound = [{"type": "state", "status": "idle"}]
    recovered = _Channel()
    channels = iter([failed, recovered])
    project = AgentProject(
        _Sandbox(),
        channel_factory=lambda sandbox, workspace: next(channels),
        owns_sandbox=False,
    )

    result = project.run(meta_agent(), _Provider(), "produce a result", timeout=1)

    assert result.answer == "finished"
    assert failed.closed is True
    assert recovered.closed is False


def test_project_retries_an_initial_session_start_socket_failure() -> None:
    """The direct LiveSession.start send reaches the same bounded recovery path."""

    class _StartFailureChannel(_Channel):
        def send(self, frame: JsonObject) -> None:
            if frame.get("type") == "session_start":
                raise RuntimeError("failed to send a frame to the E2B runner")
            super().send(frame)

    failed = _StartFailureChannel()
    recovered = _Channel()
    channels = iter([failed, recovered])
    project = AgentProject(
        _Sandbox(),
        channel_factory=lambda sandbox, workspace: next(channels),
        owns_sandbox=False,
    )

    result = project.run(meta_agent(), _Provider(), "produce a result", timeout=1)

    assert result.answer == "finished"
    assert failed.closed is True
    assert recovered.closed is False


def test_project_replaces_owned_sandbox_after_durable_outbox_failure() -> None:
    """A corrupt durable transport still reaches the bounded fresh-sandbox fallback."""

    class _CorruptOutboxChannel(_Channel):
        def __init__(self) -> None:
            super().__init__()
            self.inbound = [
                {"type": "state", "status": "idle"},
                {"type": "state", "status": "running"},
            ]

        def recv(self, timeout: float | None = None) -> JsonObject | None:
            if self.inbound:
                return super().recv(timeout)
            raise RuntimeError("durable outbox frame 4 unavailable after 5s")

    original = _Sandbox()
    replacement = _Sandbox()
    failed = _CorruptOutboxChannel()
    recovered = _Channel()
    project = AgentProject(
        original,
        channel_factory=lambda sandbox, workspace: recovered if sandbox is replacement else failed,
        sandbox_factory=lambda: replacement,
    )

    result = project.run(meta_agent(), _Provider(), "produce a result", timeout=1)

    assert result.answer == "finished"
    assert original.killed is True
    assert replacement.killed is False
    assert project.usage().count == 2


def test_denied_agent_write_never_enters_the_replayed_project_mirror() -> None:
    """A sandbox replacement replays trusted bytes, not a rejected agent overwrite."""

    class _DisconnectedAfterDeniedWrite(_Channel):
        def __init__(self) -> None:
            super().__init__()
            self.inbound = [
                {"type": "state", "status": "idle"},
                {"type": "state", "status": "running"},
                {
                    "type": "tool_request",
                    "req_id": 1,
                    "name": "write_file",
                    "arguments": {
                        "path": "context/round-1/parent.json",
                        "content": "poisoned",
                    },
                },
            ]

        def recv(self, timeout: float | None = None) -> JsonObject | None:
            if self.inbound:
                return super().recv(timeout)
            raise RuntimeError("Server disconnected")

    original = _Sandbox()
    replacement = _Sandbox()
    failed = _DisconnectedAfterDeniedWrite()
    recovered = _Channel()
    project = AgentProject(
        original,
        channel_factory=lambda sandbox, workspace: recovered if sandbox is replacement else failed,
        sandbox_factory=lambda: replacement,
    )
    parent_path = "context/round-1/parent.json"
    project.write_text(parent_path, "trusted")

    result = project.run(
        meta_agent(),
        _Provider(),
        "produce a result",
        timeout=1,
        writable_files=["result.txt"],
    )

    assert result.answer == "finished"
    assert original.killed is True
    assert replacement.files.values[f"/home/user/project/{parent_path}"] == "trusted"
    assert project.read_text(parent_path) == "trusted"
    assert project.read_text("result.txt") == "done"


def test_project_counts_worker_usage_from_failed_and_recovered_attempts() -> None:
    """A logical run reports tokens spent before and after its bounded recovery."""

    class _DisconnectedAfterLlmChannel(_Channel):
        def __init__(self) -> None:
            super().__init__()
            self.inbound = [
                {"type": "state", "status": "idle"},
                {"type": "state", "status": "running"},
                {"type": "llm_request", "req_id": 1, "openai_body": {"messages": []}},
            ]

        def recv(self, timeout: float | None = None) -> JsonObject | None:
            if self.inbound:
                return super().recv(timeout)
            raise RuntimeError("Server disconnected")

    failed = _DisconnectedAfterLlmChannel()
    recovered = _Channel()
    recovered.inbound.insert(
        2, {"type": "llm_request", "req_id": 3, "openai_body": {"messages": []}}
    )
    channels = iter([failed, recovered])
    project = AgentProject(
        _Sandbox(),
        channel_factory=lambda sandbox, workspace: next(channels),
        owns_sandbox=False,
    )

    result = project.run(meta_agent(), _MeteredProvider(), "produce a result", timeout=1)

    assert result.answer == "finished"
    assert result.worker_usage.calls == 2
    assert result.worker_usage.input_tokens == 10
    assert result.worker_usage.output_tokens == 14


def test_project_replaces_owned_sandbox_and_restores_files_after_disconnect() -> None:
    """A poisoned E2B transport is replaced without losing the project archive."""

    class _DisconnectedChannel(_Channel):
        def __init__(self) -> None:
            super().__init__()
            self.inbound = [
                {"type": "state", "status": "idle"},
                {"type": "state", "status": "running"},
            ]

        def recv(self, timeout: float | None = None) -> JsonObject | None:
            if self.inbound:
                return super().recv(timeout)
            raise RuntimeError("Server disconnected")

    original = _Sandbox()
    replacement = _Sandbox()
    disconnected = _DisconnectedChannel()
    recovered = _Channel()
    replacement_calls = 0

    def sandbox_factory() -> _Sandbox:
        nonlocal replacement_calls
        replacement_calls += 1
        return replacement

    project = AgentProject(
        original,
        channel_factory=lambda sandbox, workspace: (
            recovered if sandbox is replacement else disconnected
        ),
        sandbox_factory=sandbox_factory,
    )
    project.write_text("history/round-1.json", '{"kept": true}')

    result = project.run(meta_agent(), _Provider(), "produce a result", timeout=1)

    archived = "/home/user/project/history/round-1.json"
    assert result.answer == "finished"
    assert replacement_calls == 1
    assert original.killed is True
    assert replacement.killed is False
    assert replacement.files.values[archived] == '{"kept": true}'
    assert project.read_text("history/round-1.json") == '{"kept": true}'
    assert project.usage().count == 2


def test_project_meters_overlapping_replacement_sandbox_lifetimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacement bootstrap time bills both the old and new live sandboxes."""
    ticks = iter([0.0, 10.0, 15.0, 30.0])
    monkeypatch.setattr(project_module.time, "monotonic", lambda: next(ticks))
    original = _Sandbox()
    replacement = _Sandbox()
    project = AgentProject(original, sandbox_factory=lambda: replacement)

    project._replace_sandbox()  # noqa: SLF001
    usage = project.usage()

    assert usage.count == 2
    assert usage.seconds == 35.0  # old: 0..15 plus replacement: 10..30


def test_project_meters_a_replacement_that_fails_during_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A created sandbox is billable even if replay fails before it becomes active."""

    class _BrokenCommands(_Commands):
        def run(self, cmd: str, background: bool | None = None, **kwargs: object) -> _Output:
            del cmd, background, kwargs
            raise RuntimeError("restore failed")

    ticks = iter([0.0, 10.0, 20.0, 30.0])
    monkeypatch.setattr(project_module.time, "monotonic", lambda: next(ticks))
    original = _Sandbox()
    replacement = _Sandbox()
    replacement.commands = _BrokenCommands()
    project = AgentProject(original, sandbox_factory=lambda: replacement)

    with pytest.raises(RuntimeError, match="restore failed"):
        project._replace_sandbox()  # noqa: SLF001
    usage = project.usage()

    assert replacement.killed is True
    assert usage.count == 2
    assert usage.seconds == 40.0  # original: 0..30 plus failed replacement: 10..20


def test_project_initialization_failure_releases_only_an_owned_sandbox() -> None:
    """Constructor setup cannot orphan an owned project or kill a caller-owned lease."""

    class _BrokenCommands(_Commands):
        def run(self, cmd: str, background: bool | None = None, **kwargs: object) -> _Output:
            del cmd, background, kwargs
            raise RuntimeError("workspace setup failed")

    owned = _Sandbox()
    owned.commands = _BrokenCommands()
    with pytest.raises(RuntimeError, match="workspace setup failed"):
        AgentProject(owned)
    assert owned.killed is True

    caller_owned = _Sandbox()
    caller_owned.commands = _BrokenCommands()
    with pytest.raises(RuntimeError, match="workspace setup failed"):
        AgentProject(caller_owned, owns_sandbox=False)
    assert caller_owned.killed is False


def test_project_failed_close_keeps_usage_live_and_retries_every_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed meta-project kill remains billable and retryable instead of looking final."""
    now = [0.0]
    monkeypatch.setattr(project_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(e2b_sandbox_module.time, "sleep", lambda delay: None)
    sandbox = _FlakyKillSandbox(failures=3)
    project = AgentProject(sandbox)

    now[0] = 5.0
    with pytest.raises(SandboxCleanupError, match="1 of 1") as raised:
        project.close()
    assert sandbox.kill_attempts == 3
    assert raised.value.resource == "meta_project_sandbox"
    assert raised.value.sandbox_usage == SandboxUsage(count=1, seconds=5.0)
    assert project.usage().seconds == 5.0

    now[0] = 8.0
    assert project.usage().seconds == 8.0
    project.close()
    assert sandbox.kill_attempts == 4
    assert sandbox.killed is True
    assert project.usage().seconds == 8.0
    project.close()


def test_project_close_attempts_every_lease_after_accounting_failure() -> None:
    first = _Sandbox()
    second = _Sandbox()
    retired: list[SandboxHandle] = []
    fail_first = True

    def retire(sandbox: SandboxHandle) -> None:
        nonlocal fail_first
        retired.append(sandbox)
        cast("_Sandbox", sandbox).killed = True
        if sandbox is first and fail_first:
            fail_first = False
            raise RuntimeError("injected accounting failure")

    project = AgentProject(first, sandbox_retirer=retire)
    project._live_sandboxes[id(second)] = (  # noqa: SLF001 - multi-lease cleanup contract
        second,
        project_module.time.monotonic(),
    )
    project._sandbox_count += 1  # noqa: SLF001

    with pytest.raises(RuntimeError, match="injected accounting failure"):
        project.close()

    assert retired == [first, second]
    assert first.killed is True
    assert second.killed is True


def test_project_close_preserves_accounting_failure_behind_cleanup_retry() -> None:
    accounting = _Sandbox()
    cleanup = _Sandbox()
    cleanup_attempts = 0

    def retire(sandbox: SandboxHandle) -> None:
        nonlocal cleanup_attempts
        target = cast("_Sandbox", sandbox)
        if sandbox is accounting:
            target.killed = True
            raise BudgetBreachError("terminal settlement breach")
        cleanup_attempts += 1
        if cleanup_attempts == 1:
            raise SandboxCleanupError("cleanup still unproved")
        target.killed = True

    project = AgentProject(
        accounting,
        sandbox_retirer=retire,
        sandbox_retirement_proved=lambda sandbox: cast("_Sandbox", sandbox).killed,
    )
    project._live_sandboxes[id(cleanup)] = (  # noqa: SLF001 - multi-lease cleanup contract
        cleanup,
        project_module.time.monotonic(),
    )
    project._sandbox_count += 1  # noqa: SLF001

    with pytest.raises(SandboxCleanupError, match="1 of 2"):
        project.close()
    with pytest.raises(BudgetBreachError, match="terminal settlement breach"):
        project.close()
    project.close()

    assert accounting.killed is True
    assert cleanup.killed is True
    assert project._live_sandboxes == {}  # noqa: SLF001


def test_project_retains_old_and_replacement_leases_after_failed_retirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recovery cannot drop the old sandbox handle when its kill is unproven."""
    now = [0.0]
    monkeypatch.setattr(project_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(e2b_sandbox_module.time, "sleep", lambda delay: None)
    original = _FlakyKillSandbox(failures=3)
    replacement = _Sandbox()
    project = AgentProject(original, sandbox_factory=lambda: replacement)

    now[0] = 10.0
    with pytest.raises(SandboxCleanupError, match="cleanup failed"):
        project._replace_sandbox()  # noqa: SLF001

    now[0] = 20.0
    usage = project.usage()
    assert usage.count == 2
    assert usage.seconds == 30.0  # old: 0..20 plus replacement: 10..20

    project.close()
    assert original.kill_attempts == 4
    assert original.killed is True
    assert replacement.killed is True
    assert project.usage().seconds == 30.0


def test_project_retries_a_clean_premature_session_end() -> None:
    """A clean EOF before the turn boundary is a recoverable runner lifecycle loss."""

    ended = _Channel()
    ended.inbound = [
        {"type": "state", "status": "idle"},
        {"type": "state", "status": "running"},
    ]
    recovered = _Channel()
    channels = iter([ended, recovered])
    project = AgentProject(
        _Sandbox(),
        channel_factory=lambda sandbox, workspace: next(channels),
        owns_sandbox=False,
    )

    result = project.run(meta_agent(), _Provider(), "produce a result", timeout=1)

    assert result.answer == "finished"
    assert ended.closed is True
    assert recovered.closed is False


def test_project_rejects_paths_that_escape_its_workspace() -> None:
    project = AgentProject(_Sandbox(), channel_factory=lambda sandbox, workspace: _Channel())

    try:
        project.write_text("../escape", "no")
    except ValueError as error:
        assert "relative project path" in str(error)
    else:
        raise AssertionError("path traversal should fail")


def test_agent_file_tools_reject_paths_outside_the_project() -> None:
    """Absolute and traversing agent paths cannot reach runner or sibling files."""
    project = AgentProject(_Sandbox(), channel_factory=lambda sandbox, workspace: _Channel())

    for path in ("/home/user/runner.js", "../runner.js"):
        outcome = project._execute_tool("read_file", {"path": path}, lambda stream, data: None)
        assert outcome.is_error is True
        assert "escapes project workspace" in outcome.content


@pytest.mark.parametrize("tool", ["bash", "read_skill"])
def test_project_rejects_agents_with_uncontained_tools(tool: str) -> None:
    """The project still rejects capabilities outside its isolated tool allowlist."""
    base = meta_agent()
    uncontained = HarnessDoc(
        name="uncontained",
        surfaces=[
            surface.model_copy(update={"content": f"{tool}\nsubmit"})
            if surface.id == TOOL_POLICY_ID
            else surface
            for surface in base.surfaces
        ],
    )
    project = AgentProject(_Sandbox(), channel_factory=lambda sandbox, workspace: _Channel())

    with pytest.raises(ValueError, match=f"uncontained tools: {tool}"):
        project.run(uncontained, _Provider(), "escape", timeout=1)


class _AttestedProjectSandbox(_Sandbox):
    def __init__(
        self,
        create_kwargs: dict[str, object],
        *,
        sandbox_id: str,
        network_drift: bool = False,
        volume_drift: bool = False,
    ) -> None:
        super().__init__()
        self.sandbox_id = sandbox_id
        started_at = datetime.now(UTC)
        timeout = int(cast("float", create_kwargs["timeout"]))
        lifecycle = cast("dict[str, object]", create_kwargs["lifecycle"])
        self.info = SimpleNamespace(
            sandbox_id=sandbox_id,
            template_id=str(create_kwargs["template"]).split(":", 1)[0],
            cpu_count=2,
            memory_mb=2048,
            metadata=create_kwargs["metadata"],
            allow_internet_access=(
                True if network_drift else create_kwargs["allow_internet_access"]
            ),
            volume_mounts=([{"name": "unexpected"}] if volume_drift else []),
            lifecycle=dict(lifecycle),
            started_at=started_at,
            end_at=started_at + timedelta(seconds=timeout),
        )

    def get_info(self) -> SimpleNamespace:
        return self.info


class RateLimitException(RuntimeError):
    """Pinned E2B SDK 429 shape used without importing the optional SDK in tests."""

    __module__ = "e2b.exceptions"


_UNTRUSTED_RATE_LIMIT_EXCEPTION = type("RateLimitException", (RuntimeError,), {})


class _ProjectRateClock:
    def __init__(self) -> None:
        self.now_ns = 10_000_000_000

    def time_ns(self) -> int:
        return self.now_ns

    def sleep(self, seconds: float) -> None:
        self.now_ns += int(seconds * 1_000_000_000)


class _ProjectCreateClock:
    def __init__(self) -> None:
        self.now_s = 1_000.0

    def monotonic(self) -> float:
        return self.now_s

    def elapse(self, seconds: float) -> None:
        self.now_s += seconds


def _project_rate_authority(
    tmp_path: Path,
    *,
    policy: ExternalDispatchRatePolicy = E2B_SANDBOX_CREATE_RATE_POLICY,
    name: str = "project-create-rate.json",
) -> ExternalDispatchRateAuthority:
    clock = _ProjectRateClock()
    return ExternalDispatchRateAuthority.bootstrap(
        (tmp_path / name).resolve(),
        policy,
        clock_ns=clock.time_ns,
        sleeper=clock.sleep,
    )


_PROJECT_CONFIGURATION_ID = "project-proposer-v1"


def test_project_execution_commitment_binds_exact_path_free_launch_semantics(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_rate_authority = _rate_authority(first_dir)
    second_rate_authority = _rate_authority(second_dir)
    base = AgentProject.execution_commitment_for(
        timeout=60,
        template="template-immutable:build-one",
        cpu_count=2,
        memory_mb=2048,
        create_rate_authority=first_rate_authority,
    )

    variants = (
        AgentProject.execution_commitment_for(
            timeout=61,
            template="template-immutable:build-one",
            cpu_count=2,
            memory_mb=2048,
            create_rate_authority=first_rate_authority,
        ),
        AgentProject.execution_commitment_for(
            timeout=60,
            template="template-immutable:build-two",
            cpu_count=2,
            memory_mb=2048,
            create_rate_authority=first_rate_authority,
        ),
        AgentProject.execution_commitment_for(
            timeout=60,
            template="template-immutable:build-one",
            cpu_count=4,
            memory_mb=2048,
            create_rate_authority=first_rate_authority,
        ),
        AgentProject.execution_commitment_for(
            timeout=60,
            template="template-immutable:build-one",
            cpu_count=2,
            memory_mb=2048,
            create_rate_authority=second_rate_authority,
        ),
    )

    assert all(variant.digest != base.digest for variant in variants)
    assert base.secure is True
    assert base.internet_access_at_create is False
    assert base.timeout_action == "kill"
    assert base.auto_resume is False
    assert base.volume_mounts is False
    assert base.create_request_timeout_s == project_module.E2B_CREATE_REQUEST_TIMEOUT_S
    assert (
        base.resource_class.create_request_timeout_seconds
        == project_module._PROJECT_CREATE_HORIZON_S
    )
    assert base.resource_class.create_request_timeout_seconds > base.create_request_timeout_s
    assert str(tmp_path) not in base.model_dump_json()


def _project_cost_runtime(
    tmp_path: Path,
    *,
    component_configuration_id: str,
    timeout: int = 60,
    hard_limit: int | None = None,
    create_horizon: int | None = None,
) -> tuple[SearchComponentCostRuntime, TimedResourceBudgetAccount]:
    resource_class = TimedResourceClass(
        role=TimedResourceRole.PROPOSER_PROJECT,
        cpu_count=2,
        memory_mb=2048,
        provider_ttl_seconds=timeout,
        create_request_timeout_seconds=(
            project_module._PROJECT_CREATE_HORIZON_S if create_horizon is None else create_horizon
        ),
        cleanup_horizon_seconds=project_module.E2B_CLEANUP_HORIZON_S,
    )
    resource_meter = TimedResourceCostMeter(
        resource_type=resource_class.role.value,
        resource_class_digest=resource_class.digest,
        nano_usd_per_second=1,
        max_billing_seconds=resource_class.max_host_observation_seconds,
    )
    limit = resource_meter.maximum_charge_nano_usd() * 3 if hard_limit is None else hard_limit
    provider_config = ProviderConfig(
        kind=ProviderKind.BEDROCK,
        model_type="claude-haiku-4-5",
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region="us-east-1",
    )
    provider_meter = provider_cost_meter(catalog_provider_token_tariff(provider_config))
    policy = BudgetPolicy(
        study_id="project-cost-runtime-test",
        manifest_digest="sha256:" + hashlib.sha256(str(tmp_path).encode()).hexdigest(),
        hard_limit_nano_usd=limit,
        phase_limits_nano_usd={"search": limit},
        meters={
            "proposer-provider": provider_meter,
            "scorer-provider": provider_meter,
            "project": resource_meter,
        },
    )
    authority = bootstrap_budget_ledger((tmp_path / "cost-runtime.sqlite3").resolve(), policy)
    proposer_scope = BudgetScope(phase="search", category="proposer", run_id="test-run")
    scorer_scope = BudgetScope(phase="search", category="scorer", run_id="test-run")
    proposer_provider = authority.provider_account(
        scope=proposer_scope,
        meter_id="proposer-provider",
    )
    project_resource = authority.timed_resource_account(
        scope=proposer_scope,
        meter_id="project",
    )
    scorer_provider = authority.provider_account(
        scope=scorer_scope,
        meter_id="scorer-provider",
    )
    binding = SearchCostBinding(
        declared_hard_limit_nano_usd=policy.hard_limit_nano_usd,
        policy=policy,
        ledger_identity=authority.ledger_identity,
        phase="search",
        run_id="test-run",
        external_dispatch_rate_binding=_project_rate_authority(tmp_path).binding,
        proposer=SearchComponentCostBinding(
            role=SearchComponentRole.PROPOSER,
            configuration_id=component_configuration_id,
            scope_category="proposer",
            providers=(
                ProviderCostBinding(
                    component_configuration_id=component_configuration_id,
                    provider_config=provider_config,
                    response_identity=ProviderResponseIdentity(provider=ProviderKind.BEDROCK),
                    account=bind_budget_account(proposer_provider),
                ),
            ),
            timed_resources=(
                TimedResourceCostBinding(
                    component_configuration_id=component_configuration_id,
                    resource_type=resource_class.role.value,
                    resource_class_digest=resource_class.digest,
                    account=bind_timed_resource_account(project_resource),
                ),
            ),
        ),
        scorer=SearchComponentCostBinding(
            role=SearchComponentRole.SCORER,
            configuration_id="unused-scorer",
            scope_category="scorer",
            providers=(
                ProviderCostBinding(
                    component_configuration_id="unused-scorer",
                    provider_config=provider_config,
                    response_identity=ProviderResponseIdentity(provider=ProviderKind.BEDROCK),
                    account=bind_budget_account(scorer_provider),
                ),
            ),
        ),
    )
    return (
        SearchCostRuntime(authority=authority, binding=binding).for_component(
            SearchComponentRole.PROPOSER
        ),
        project_resource,
    )


def test_cost_bound_project_rejects_binding_drift_before_external_or_filesystem_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _account = _project_cost_runtime(
        tmp_path,
        component_configuration_id="expected-proposer",
    )
    factory_calls = 0

    def unexpected_factory(**_kwargs: object) -> Callable[[], _Sandbox]:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("cost binding drift must fail before sandbox factory construction")

    monkeypatch.setattr(project_module, "default_sandbox_factory", unexpected_factory)
    lease_dir = (tmp_path / "leases").resolve()

    with pytest.raises(ValueError, match="configuration_id differs"):
        AgentProject.create(
            timeout=60,
            template="template-immutable:build-immutable",
            cpu_count=2,
            memory_mb=2048,
            cost_runtime=runtime,
            component_configuration_id="drifted-proposer",
            lease_ledger_dir=lease_dir,
            create_rate_authority=_project_rate_authority(tmp_path),
        )

    assert factory_calls == 0
    assert not lease_dir.exists()


def test_cost_bound_project_requires_create_rate_authority_before_external_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _account = _project_cost_runtime(
        tmp_path,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
    )
    factory_calls = 0

    def unexpected_factory(**_kwargs: object) -> Callable[[], _Sandbox]:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("missing rate authority must fail before provider setup")

    monkeypatch.setattr(project_module, "default_sandbox_factory", unexpected_factory)
    lease_dir = (tmp_path / "leases").resolve()

    with pytest.raises(ValueError, match="create-rate authority"):
        AgentProject.create(
            timeout=60,
            template="template-immutable:build-immutable",
            cpu_count=2,
            memory_mb=2048,
            cost_runtime=runtime,
            component_configuration_id=_PROJECT_CONFIGURATION_ID,
            lease_ledger_dir=lease_dir,
            create_rate_authority=cast("ExternalDispatchRateAuthority", None),
        )

    assert factory_calls == 0
    assert not lease_dir.exists()


def test_cost_bound_project_defers_create_until_complete_search_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _account = _project_cost_runtime(
        tmp_path,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
    )
    creates = 0

    def default_factory(**kwargs: object) -> Callable[[], _AttestedProjectSandbox]:
        frozen = dict(kwargs)

        def create() -> _AttestedProjectSandbox:
            nonlocal creates
            creates += 1
            return _AttestedProjectSandbox(frozen, sandbox_id="deferred-project")

        return create

    monkeypatch.setattr(project_module, "default_sandbox_factory", default_factory)
    rate_authority = _project_rate_authority(tmp_path)
    execution_commitment = AgentProject.execution_commitment_for(
        timeout=60,
        template="template-immutable:build-immutable",
        cpu_count=2,
        memory_mb=2048,
        create_rate_authority=rate_authority,
    )
    project = AgentProject.create(
        timeout=60,
        template="template-immutable:build-immutable",
        cpu_count=2,
        memory_mb=2048,
        cost_runtime=runtime,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
        lease_ledger_dir=(tmp_path / "leases").resolve(),
        create_rate_authority=rate_authority,
    )

    assert creates == 0
    assert project.execution_configuration_id == execution_commitment.digest
    with pytest.raises(RuntimeError, match="complete search cost preflight"):
        project.write_text("context/input.json", "{}")
    assert creates == 0

    project.authorize_search_dispatch(runtime.search_binding)
    project.write_text("context/input.json", "{}")
    assert creates == 1
    project.close()


def test_budgeted_project_initial_and_replacement_sandboxes_are_offline_and_metered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, account = _project_cost_runtime(
        tmp_path,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
    )
    create_calls: list[dict[str, object]] = []
    sandboxes: list[_AttestedProjectSandbox] = []
    rate_authority = _project_rate_authority(tmp_path)
    acquired_sequences: list[int] = []
    acquire = rate_authority.acquire

    def acquire_rate_permit(*, timeout_seconds: float | None = None) -> object:
        permit = acquire(timeout_seconds=timeout_seconds)
        acquired_sequences.append(permit.sequence)
        return permit

    monkeypatch.setattr(rate_authority, "acquire", acquire_rate_permit)

    def default_factory(**kwargs: object) -> Callable[[], _AttestedProjectSandbox]:
        frozen = dict(kwargs)
        create_calls.append(frozen)

        def create() -> _AttestedProjectSandbox:
            sandbox = _AttestedProjectSandbox(
                frozen,
                sandbox_id=f"project-{len(sandboxes) + 1}",
            )
            sandboxes.append(sandbox)
            return sandbox

        return create

    monkeypatch.setattr(project_module, "default_sandbox_factory", default_factory)
    project = AgentProject.create(
        timeout=60,
        template="template-immutable:build-immutable",
        cpu_count=2,
        memory_mb=2048,
        cost_runtime=runtime,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
        lease_ledger_dir=(tmp_path / "leases").resolve(),
        create_rate_authority=rate_authority,
        api_key="explicit-secret-key",
    )

    project.authorize_search_dispatch(runtime.search_binding)
    project._replace_sandbox()  # noqa: SLF001 - verify the owned replacement contract
    project.close()

    assert len(create_calls) == len(sandboxes) == 2
    assert acquired_sequences == [1, 2]
    assert project.create_rate_binding == rate_authority.binding
    assert all(call["allow_internet_access"] is False for call in create_calls)
    assert all(call["secure"] is True for call in create_calls)
    assert all(call["volume_mounts"] is None for call in create_calls)
    assert all(
        call["lifecycle"] == {"on_timeout": "kill", "auto_resume": False} for call in create_calls
    )
    assert all(call["request_timeout"] == 30 for call in create_calls)
    assert all(sandbox.killed for sandbox in sandboxes)
    metadata = [cast("dict[str, str]", call["metadata"]) for call in create_calls]
    assert metadata[0]["wmh_runner_config"] == metadata[1]["wmh_runner_config"]
    assert metadata[0]["wmh_runner_lease"] != metadata[1]["wmh_runner_lease"]
    assert "explicit-secret-key" not in json.dumps(metadata)
    reservations = SpendLedger(account.ledger_path, account.policy).reservations()
    assert len(reservations) == 2
    assert all(item.status is ReservationStatus.SETTLED for item in reservations)


def test_budgeted_project_reused_handle_cannot_inherit_prior_retirement_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _account = _project_cost_runtime(
        tmp_path,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
    )
    shared: _AttestedProjectSandbox | None = None
    create_count = 0

    def default_factory(**kwargs: object) -> Callable[[], _AttestedProjectSandbox]:
        frozen = dict(kwargs)

        def create() -> _AttestedProjectSandbox:
            nonlocal shared, create_count
            create_count += 1
            replacement = _AttestedProjectSandbox(
                frozen,
                sandbox_id=f"reused-handle-{create_count}",
            )
            if shared is None:
                shared = replacement
            else:
                shared.sandbox_id = replacement.sandbox_id
                shared.info = replacement.info
                shared.killed = False
            return shared

        return create

    fail_retirement = False

    def retire(sandbox: SandboxHandle) -> None:
        if fail_retirement:
            raise SandboxCleanupError("second lease cleanup unproved")
        cast("_AttestedProjectSandbox", sandbox).killed = True

    monkeypatch.setattr(project_module, "default_sandbox_factory", default_factory)
    monkeypatch.setattr(project_module, "kill_sandbox", retire)
    factory = project_module._BudgetedProjectSandboxFactory(  # noqa: SLF001
        cost_runtime=runtime,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
        ledger_dir=(tmp_path / "leases").resolve(),
        timeout=60,
        template="template-immutable:build-immutable",
        api_key=None,
        cpu_count=2,
        memory_mb=2048,
        create_rate_authority=_project_rate_authority(tmp_path),
    )

    first = factory()
    factory.retire(first)
    assert factory.retirement_proved(first)

    second = factory()
    assert second is first
    assert not factory.retirement_proved(second)

    fail_retirement = True
    with pytest.raises(SandboxCleanupError, match="second lease cleanup unproved"):
        factory.retire(second)

    assert not factory.retirement_proved(second)
    assert id(second) in factory._live  # noqa: SLF001


def test_cost_bound_project_forgets_killed_lease_after_terminal_settlement_breach(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, account = _project_cost_runtime(
        tmp_path,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
    )
    created: list[_AttestedProjectSandbox] = []

    def default_factory(**kwargs: object) -> Callable[[], _AttestedProjectSandbox]:
        sandbox = _AttestedProjectSandbox(dict(kwargs), sandbox_id="terminal-breach")
        created.append(sandbox)
        return lambda: sandbox

    monkeypatch.setattr(project_module, "default_sandbox_factory", default_factory)
    project = AgentProject.create(
        timeout=60,
        template="template-immutable:build-immutable",
        cpu_count=2,
        memory_mb=2048,
        cost_runtime=runtime,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
        lease_ledger_dir=(tmp_path / "leases").resolve(),
        create_rate_authority=_project_rate_authority(tmp_path),
    )
    project.authorize_search_dispatch(runtime.search_binding)
    project.write_text("context/materialize.json", "{}")
    factory = cast("project_module._BudgetedProjectSandboxFactory", project._sandbox_factory)  # noqa: SLF001
    [lease] = factory._live.values()  # noqa: SLF001 - force a terminal horizon breach
    assert lease.reservation is not None
    lease.reservation._started_at_s -= 10_000  # noqa: SLF001

    with pytest.raises(BudgetBreachError, match="exceeded its"):
        project.close()

    [reservation] = SpendLedger(account.ledger_path, account.policy).reservations()
    assert reservation.status is ReservationStatus.BREACHED
    assert created[0].killed is True
    assert project._live_sandboxes == {}  # noqa: SLF001
    assert factory._live == {}  # noqa: SLF001
    project.close()


def test_cost_bound_project_forgets_killed_lease_after_accounting_integrity_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _account = _project_cost_runtime(
        tmp_path,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
    )
    created: list[_AttestedProjectSandbox] = []

    def default_factory(**kwargs: object) -> Callable[[], _AttestedProjectSandbox]:
        sandbox = _AttestedProjectSandbox(dict(kwargs), sandbox_id="accounting-error")
        created.append(sandbox)
        return lambda: sandbox

    monkeypatch.setattr(project_module, "default_sandbox_factory", default_factory)
    project = AgentProject.create(
        timeout=60,
        template="template-immutable:build-immutable",
        cpu_count=2,
        memory_mb=2048,
        cost_runtime=runtime,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
        lease_ledger_dir=(tmp_path / "leases").resolve(),
        create_rate_authority=_project_rate_authority(tmp_path),
    )
    project.authorize_search_dispatch(runtime.search_binding)
    project.write_text("context/materialize.json", "{}")
    factory = cast("project_module._BudgetedProjectSandboxFactory", project._sandbox_factory)  # noqa: SLF001
    [lease] = factory._live.values()  # noqa: SLF001 - inject post-kill accounting failure
    assert lease.reservation is not None

    def fail_settlement() -> None:
        raise RuntimeError("injected accounting integrity error")

    monkeypatch.setattr(lease.reservation, "settle", fail_settlement)

    with pytest.raises(RuntimeError, match="injected accounting integrity error"):
        project.close()

    assert created[0].killed is True
    assert project._live_sandboxes == {}  # noqa: SLF001
    assert factory._live == {}  # noqa: SLF001
    project.close()


def test_budgeted_project_accepts_sdk_lifecycle_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The E2B SDK exposes SandboxInfo.lifecycle as a TypedDict at runtime."""
    runtime, _account = _project_cost_runtime(
        tmp_path,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
    )

    def default_factory(**kwargs: object) -> Callable[[], _AttestedProjectSandbox]:
        sandbox = _AttestedProjectSandbox(dict(kwargs), sandbox_id="project-sdk-shape")
        sandbox.info.lifecycle = {"on_timeout": "kill", "auto_resume": False}
        return lambda: sandbox

    monkeypatch.setattr(project_module, "default_sandbox_factory", default_factory)
    project = AgentProject.create(
        timeout=60,
        template="template-immutable:build-immutable",
        cpu_count=2,
        memory_mb=2048,
        cost_runtime=runtime,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
        lease_ledger_dir=(tmp_path / "leases").resolve(),
        create_rate_authority=_project_rate_authority(tmp_path),
        api_key="explicit-secret-key",
    )

    project.authorize_search_dispatch(runtime.search_binding)
    project.write_text("context/materialize.json", "{}")
    project.close()


def test_budgeted_project_rejects_timeout_above_provider_maximum(tmp_path: Path) -> None:
    runtime, _account = _project_cost_runtime(
        tmp_path,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
        timeout=86_401,
    )

    with pytest.raises(ValueError, match="provider maximum"):
        AgentProject.create(
            timeout=86_401,
            template="template-immutable:build-immutable",
            cpu_count=2,
            memory_mb=2048,
            cost_runtime=runtime,
            component_configuration_id=_PROJECT_CONFIGURATION_ID,
            lease_ledger_dir=(tmp_path / "leases").resolve(),
            create_rate_authority=_project_rate_authority(tmp_path),
            api_key="explicit-secret-key",
        )


def test_agent_project_requires_host_rate_authority_before_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, account = _project_cost_runtime(
        tmp_path,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
    )
    dispatches = 0
    reaped: list[str] = []

    def unexpected_factory(**_kwargs: object) -> Callable[[], _Sandbox]:
        def create() -> _Sandbox:
            nonlocal dispatches
            dispatches += 1
            raise AssertionError("missing rate authority must precede provider dispatch")

        return create

    monkeypatch.setattr(project_module, "default_sandbox_factory", unexpected_factory)
    monkeypatch.setattr(
        project_module,
        "reap_e2b_runner_lease",
        lambda lease_id, *, api_key=None: (reaped.append(lease_id), (lease_id,))[1],
    )
    create_without_rate = cast("Callable[..., AgentProject]", AgentProject.create)

    with pytest.raises(TypeError, match="create_rate_authority"):
        create_without_rate(
            timeout=60,
            template="template-immutable:build-immutable",
            cpu_count=2,
            memory_mb=2048,
            cost_runtime=runtime,
            component_configuration_id=_PROJECT_CONFIGURATION_ID,
            lease_ledger_dir=(tmp_path / "leases").resolve(),
        )

    assert dispatches == 0
    assert reaped == []
    assert SpendLedger(account.ledger_path, account.policy).reservations() == []
    assert not (tmp_path / "leases").exists()


def test_budgeted_factory_rejects_missing_rate_authority_before_effects(tmp_path: Path) -> None:
    runtime, account = _project_cost_runtime(
        tmp_path,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
    )

    with pytest.raises(ValueError, match="create-rate authority"):
        project_module._BudgetedProjectSandboxFactory(
            cost_runtime=runtime,
            component_configuration_id=_PROJECT_CONFIGURATION_ID,
            ledger_dir=(tmp_path / "leases").resolve(),
            timeout=60,
            template="template-immutable:build-immutable",
            api_key=None,
            cpu_count=2,
            memory_mb=2048,
            create_rate_authority=cast("ExternalDispatchRateAuthority", None),
        )

    assert SpendLedger(account.ledger_path, account.policy).reservations() == []
    assert not (tmp_path / "leases").exists()


def test_agent_project_rejects_wrong_rate_policy_before_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, account = _project_cost_runtime(
        tmp_path,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
    )
    wrong_policy = ExternalDispatchRatePolicy(
        provider="e2b",
        operation="sandbox_create",
        maximum_dispatches=5,
        period_milliseconds=1000,
    )
    authority = _project_rate_authority(
        tmp_path,
        policy=wrong_policy,
        name="wrong-project-create-rate.json",
    )
    factories = 0

    def unexpected_factory(**_kwargs: object) -> Callable[[], _Sandbox]:
        nonlocal factories
        factories += 1
        raise AssertionError("wrong rate policy must precede provider setup")

    monkeypatch.setattr(project_module, "default_sandbox_factory", unexpected_factory)

    with pytest.raises(ValueError, match="frozen four-per-second policy"):
        AgentProject.create(
            timeout=60,
            template="template-immutable:build-immutable",
            cpu_count=2,
            memory_mb=2048,
            cost_runtime=runtime,
            component_configuration_id=_PROJECT_CONFIGURATION_ID,
            lease_ledger_dir=(tmp_path / "leases").resolve(),
            create_rate_authority=authority,
        )
    with pytest.raises(ValueError, match="frozen four-per-second policy"):
        project_module._BudgetedProjectSandboxFactory(
            cost_runtime=runtime,
            component_configuration_id=_PROJECT_CONFIGURATION_ID,
            ledger_dir=(tmp_path / "leases").resolve(),
            timeout=60,
            template="template-immutable:build-immutable",
            api_key=None,
            cpu_count=2,
            memory_mb=2048,
            create_rate_authority=authority,
        )

    assert factories == 0
    assert SpendLedger(account.ledger_path, account.policy).reservations() == []
    assert not (tmp_path / "leases").exists()


def test_budgeted_project_requires_budget_for_the_full_retry_horizon(tmp_path: Path) -> None:
    runtime, _account = _project_cost_runtime(
        tmp_path,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
        create_horizon=project_module.E2B_CREATE_REQUEST_TIMEOUT_S,
    )

    with pytest.raises(ValueError, match="resource class differs"):
        AgentProject.create(
            timeout=60,
            template="template-immutable:build-immutable",
            cpu_count=2,
            memory_mb=2048,
            cost_runtime=runtime,
            component_configuration_id=_PROJECT_CONFIGURATION_ID,
            lease_ledger_dir=(tmp_path / "leases").resolve(),
            create_rate_authority=_project_rate_authority(tmp_path),
        )

    assert not (tmp_path / "leases").exists()


def test_project_budget_denial_never_dispatches_or_reaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, account = _project_cost_runtime(
        tmp_path,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
        hard_limit=1,
    )
    creates = 0
    reaped: list[tuple[str, str | None]] = []

    def unexpected_factory(**_kwargs: object) -> Callable[[], _Sandbox]:
        nonlocal creates
        creates += 1
        raise AssertionError("budget denial must precede provider create")

    monkeypatch.setattr(project_module, "default_sandbox_factory", unexpected_factory)
    monkeypatch.setattr(
        project_module,
        "reap_e2b_runner_lease",
        lambda lease_id, *, api_key=None: (reaped.append((lease_id, api_key)), ())[1],
    )

    project = AgentProject.create(
        timeout=60,
        template="template-immutable:build-immutable",
        cpu_count=2,
        memory_mb=2048,
        cost_runtime=runtime,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
        lease_ledger_dir=(tmp_path / "leases").resolve(),
        create_rate_authority=_project_rate_authority(tmp_path),
        api_key="explicit-key",
    )
    project.authorize_search_dispatch(runtime.search_binding)

    with pytest.raises(BudgetExceededError, match="hard budget"):
        project.write_text("context/materialize.json", "{}")

    assert creates == 0
    assert reaped == []
    assert SpendLedger(account.ledger_path, account.policy).reservations() == []
    [lease_path] = (tmp_path / "leases").glob("*.json")
    assert json.loads(lease_path.read_text())["state"] == "retired"


def test_project_ambiguous_create_uses_explicit_key_and_forfeits_full_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, account = _project_cost_runtime(
        tmp_path,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
    )
    reaped: list[tuple[str, str | None]] = []

    def ambiguous_factory(**_kwargs: object) -> Callable[[], _Sandbox]:
        def create() -> _Sandbox:
            raise RuntimeError("ambiguous create")

        return create

    monkeypatch.setattr(project_module, "default_sandbox_factory", ambiguous_factory)
    monkeypatch.setattr(
        project_module,
        "reap_e2b_runner_lease",
        lambda lease_id, *, api_key=None: (reaped.append((lease_id, api_key)), ())[1],
    )

    project = AgentProject.create(
        timeout=60,
        template="template-immutable:build-immutable",
        cpu_count=2,
        memory_mb=2048,
        cost_runtime=runtime,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
        lease_ledger_dir=(tmp_path / "leases").resolve(),
        create_rate_authority=_project_rate_authority(tmp_path),
        api_key="explicit-key",
    )
    project.authorize_search_dispatch(runtime.search_binding)

    with pytest.raises(RuntimeError, match="ambiguous create"):
        project.write_text("context/materialize.json", "{}")

    [reservation] = SpendLedger(account.ledger_path, account.policy).reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.charged_nano_usd == reservation.max_nano_usd
    assert reaped == [(reservation.reservation_id, "explicit-key")]


def test_budgeted_project_retries_definitive_e2b_rate_refusal_under_one_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, account = _project_cost_runtime(
        tmp_path,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
    )
    create_calls = 0
    create_metadata: list[dict[str, str]] = []
    sleeps: list[float] = []
    reaped: list[str] = []
    dispatch_events: list[str] = []
    gate_timeouts: list[float | None] = []
    authority = _project_rate_authority(tmp_path)
    original_acquire = authority.acquire

    def acquire(*, timeout_seconds: float | None = None) -> ExternalDispatchPermit:
        dispatch_events.append("acquire")
        gate_timeouts.append(timeout_seconds)
        return original_acquire(timeout_seconds=timeout_seconds)

    def rate_limited_factory(**kwargs: object) -> Callable[[], _AttestedProjectSandbox]:
        frozen = dict(kwargs)

        def create() -> _AttestedProjectSandbox:
            nonlocal create_calls
            dispatch_events.append("dispatch")
            create_calls += 1
            create_metadata.append(dict(cast("dict[str, str]", frozen["metadata"])))
            if create_calls < 3:
                raise RateLimitException(
                    "429: Rate limit exceeded, please try again later. - capacity"
                )
            return _AttestedProjectSandbox(frozen, sandbox_id="rate-retry-success")

        return create

    monkeypatch.setattr(project_module, "default_sandbox_factory", rate_limited_factory)
    monkeypatch.setattr(project_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(authority, "acquire", acquire)
    monkeypatch.setattr(
        project_module,
        "reap_e2b_runner_lease",
        lambda lease_id, *, api_key=None: (reaped.append(lease_id), (lease_id,))[1],
    )

    project = AgentProject.create(
        timeout=60,
        template="template-immutable:build-immutable",
        cpu_count=2,
        memory_mb=2048,
        cost_runtime=runtime,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
        lease_ledger_dir=(tmp_path / "leases").resolve(),
        create_rate_authority=authority,
    )
    project.authorize_search_dispatch(runtime.search_binding)
    project.write_text("context/materialize.json", "{}")
    project.close()

    assert create_calls == 3
    assert dispatch_events == [
        "acquire",
        "dispatch",
        "acquire",
        "dispatch",
        "acquire",
        "dispatch",
    ]
    assert gate_timeouts == [project_module._PROJECT_RATE_GATE_TIMEOUT_S] * 3
    assert sleeps == [1.0, 3.0]
    assert len({metadata["wmh_runner_lease"] for metadata in create_metadata}) == 1
    assert len({metadata["wmh_runner_owner"] for metadata in create_metadata}) == 1
    assert reaped == []
    [reservation] = SpendLedger(account.ledger_path, account.policy).reservations()
    assert reservation.status is ReservationStatus.SETTLED
    [lease_path] = (tmp_path / "leases").glob("*.json")
    lease_record = json.loads(lease_path.read_text())
    assert lease_record["state"] == "retired"
    created_at = datetime.fromisoformat(lease_record["created_at"])
    provider_expiry_at = datetime.fromisoformat(lease_record["provider_expiry_at"])
    assert provider_expiry_at - created_at == timedelta(
        seconds=(
            project_module._PROJECT_CREATE_HORIZON_S
            + 60
            + project_module._PROJECT_PROVIDER_CLOCK_SKEW_S
        )
    )


def test_budgeted_project_exhausts_rate_refusals_without_orphan_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, account = _project_cost_runtime(
        tmp_path,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
    )
    create_calls = 0
    sleeps: list[float] = []
    reaped: list[str] = []
    authority = _project_rate_authority(tmp_path)

    def rate_limited_factory(**_kwargs: object) -> Callable[[], _Sandbox]:
        def create() -> _Sandbox:
            nonlocal create_calls
            create_calls += 1
            raise RateLimitException("429: Rate limit exceeded, please try again later.")

        return create

    monkeypatch.setattr(project_module, "default_sandbox_factory", rate_limited_factory)
    monkeypatch.setattr(project_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        project_module,
        "reap_e2b_runner_lease",
        lambda lease_id, *, api_key=None: (reaped.append(lease_id), (lease_id,))[1],
    )

    project = AgentProject.create(
        timeout=60,
        template="template-immutable:build-immutable",
        cpu_count=2,
        memory_mb=2048,
        cost_runtime=runtime,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
        lease_ledger_dir=(tmp_path / "leases").resolve(),
        create_rate_authority=authority,
    )
    project.authorize_search_dispatch(runtime.search_binding)

    with pytest.raises(RateLimitException, match="Rate limit exceeded"):
        project.write_text("context/materialize.json", "{}")

    assert create_calls == 4
    assert sleeps == [1.0, 3.0, 9.0]
    assert json.loads((tmp_path / "project-create-rate.json").read_text())["sequence"] == 4
    assert reaped == []
    [reservation] = SpendLedger(account.ledger_path, account.policy).reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.failure_type == "CreateRejected"
    [lease_path] = (tmp_path / "leases").glob("*.json")
    assert json.loads(lease_path.read_text())["state"] == "retired"


def test_budgeted_project_never_dispatches_a_retry_past_the_absolute_create_horizon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, account = _project_cost_runtime(
        tmp_path,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
    )
    authority = _project_rate_authority(tmp_path)
    clock = _ProjectCreateClock()
    gate_calls = 0
    create_calls = 0
    reaped: list[str] = []

    def acquire(*, timeout_seconds: float | None = None) -> ExternalDispatchPermit:
        nonlocal gate_calls
        gate_calls += 1
        assert timeout_seconds == project_module._PROJECT_RATE_GATE_TIMEOUT_S
        return ExternalDispatchPermit(
            policy_digest=authority.policy.digest,
            ledger_identity=authority.binding.ledger_identity,
            sequence=gate_calls,
            admitted_at_unix_ns=10_000_000_000 + gate_calls,
        )

    def delayed_refusal_factory(**_kwargs: object) -> Callable[[], _Sandbox]:
        def create() -> _Sandbox:
            nonlocal create_calls
            create_calls += 1
            if create_calls > 1:
                raise AssertionError("expired retry must not reach the provider")
            clock.elapse(
                project_module._PROJECT_CREATE_HORIZON_S
                - project_module.E2B_CREATE_REQUEST_TIMEOUT_S
            )
            raise RateLimitException("429: Rate limit exceeded, please try again later.")

        return create

    monkeypatch.setattr(authority, "acquire", acquire)
    monkeypatch.setattr(project_module, "default_sandbox_factory", delayed_refusal_factory)
    monkeypatch.setattr(project_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(project_module.time, "sleep", clock.elapse)
    monkeypatch.setattr(
        project_module,
        "reap_e2b_runner_lease",
        lambda lease_id, *, api_key=None: (reaped.append(lease_id), (lease_id,))[1],
    )

    project = AgentProject.create(
        timeout=60,
        template="template-immutable:build-immutable",
        cpu_count=2,
        memory_mb=2048,
        cost_runtime=runtime,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
        lease_ledger_dir=(tmp_path / "leases").resolve(),
        create_rate_authority=authority,
    )
    project.authorize_search_dispatch(runtime.search_binding)

    with pytest.raises(TimeoutError, match="create horizon"):
        project.write_text("context/materialize.json", "{}")

    assert gate_calls == create_calls == 1
    assert reaped == []
    [reservation] = SpendLedger(account.ledger_path, account.policy).reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.failure_type == "CreateRejected"
    [lease_path] = (tmp_path / "leases").glob("*.json")
    assert json.loads(lease_path.read_text())["state"] == "retired"


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("503 capacity unavailable"),
        RuntimeError("429 too many requests"),
        RateLimitException("429: Unauthorized"),
        _UNTRUSTED_RATE_LIMIT_EXCEPTION("429: Rate limit exceeded, please try again later."),
        RuntimeError("invalid template configuration"),
        TimeoutError("create request timed out"),
    ],
    ids=[
        "capacity-503",
        "untyped-429",
        "wrong-status",
        "wrong-module",
        "config",
        "timeout",
    ],
)
def test_budgeted_project_never_retries_unproven_create_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    runtime, account = _project_cost_runtime(
        tmp_path,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
    )
    create_calls = 0
    sleeps: list[float] = []
    reaped: list[str] = []

    def failed_factory(**_kwargs: object) -> Callable[[], _Sandbox]:
        def create() -> _Sandbox:
            nonlocal create_calls
            create_calls += 1
            raise error

        return create

    monkeypatch.setattr(project_module, "default_sandbox_factory", failed_factory)
    monkeypatch.setattr(project_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        project_module,
        "reap_e2b_runner_lease",
        lambda lease_id, *, api_key=None: (reaped.append(lease_id), (lease_id,))[1],
    )

    project = AgentProject.create(
        timeout=60,
        template="template-immutable:build-immutable",
        cpu_count=2,
        memory_mb=2048,
        cost_runtime=runtime,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
        lease_ledger_dir=(tmp_path / "leases").resolve(),
        create_rate_authority=_project_rate_authority(tmp_path),
    )
    project.authorize_search_dispatch(runtime.search_binding)

    with pytest.raises(type(error), match=re.escape(str(error))):
        project.write_text("context/materialize.json", "{}")

    assert create_calls == 1
    assert sleeps == []
    assert len(reaped) == 1
    [reservation] = SpendLedger(account.ledger_path, account.policy).reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.failure_type == "CreateUnknown"


@pytest.mark.parametrize("failed_activation", [1, 2])
def test_project_activation_failure_kills_and_terminates_budget_and_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_activation: int,
) -> None:
    runtime, account = _project_cost_runtime(
        tmp_path,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
    )
    created: list[_AttestedProjectSandbox] = []
    original_activate = project_module.RunnerLeaseLedger.activate
    activation_calls = 0

    def injected_activate(
        ledger: project_module.RunnerLeaseLedger,
        resource_id: str,
        *,
        expected_end_at: datetime | None = None,
    ) -> None:
        nonlocal activation_calls
        activation_calls += 1
        if activation_calls == failed_activation:
            raise RuntimeError("injected activation persistence failure")
        original_activate(ledger, resource_id, expected_end_at=expected_end_at)

    def default_factory(**kwargs: object) -> Callable[[], _AttestedProjectSandbox]:
        frozen = dict(kwargs)

        def create() -> _AttestedProjectSandbox:
            sandbox = _AttestedProjectSandbox(frozen, sandbox_id="activation-failure")
            created.append(sandbox)
            return sandbox

        return create

    monkeypatch.setattr(project_module.RunnerLeaseLedger, "activate", injected_activate)
    monkeypatch.setattr(project_module, "default_sandbox_factory", default_factory)

    project = AgentProject.create(
        timeout=60,
        template="template-immutable:build-immutable",
        cpu_count=2,
        memory_mb=2048,
        cost_runtime=runtime,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
        lease_ledger_dir=(tmp_path / "leases").resolve(),
        create_rate_authority=_project_rate_authority(tmp_path),
    )
    project.authorize_search_dispatch(runtime.search_binding)

    with pytest.raises(RuntimeError, match="activation persistence"):
        project.write_text("context/materialize.json", "{}")

    assert len(created) == 1
    assert created[0].killed is True
    [lease_path] = (tmp_path / "leases").glob("*.json")
    assert json.loads(lease_path.read_text())["state"] == "retired"
    [reservation] = SpendLedger(account.ledger_path, account.policy).reservations()
    assert reservation.status is ReservationStatus.SETTLED


@pytest.mark.parametrize(("network_drift", "volume_drift"), [(True, False), (False, True)])
def test_project_creation_fails_closed_on_network_or_volume_attestation_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    network_drift: bool,
    volume_drift: bool,
) -> None:
    runtime, _account = _project_cost_runtime(
        tmp_path,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
    )
    created: list[_AttestedProjectSandbox] = []

    def drifted_factory(**kwargs: object) -> Callable[[], _AttestedProjectSandbox]:
        frozen = dict(kwargs)

        def create() -> _AttestedProjectSandbox:
            sandbox = _AttestedProjectSandbox(
                frozen,
                sandbox_id="drifted-project",
                network_drift=network_drift,
                volume_drift=volume_drift,
            )
            created.append(sandbox)
            return sandbox

        return create

    monkeypatch.setattr(project_module, "default_sandbox_factory", drifted_factory)

    project = AgentProject.create(
        timeout=60,
        template="template-immutable:build-immutable",
        cpu_count=2,
        memory_mb=2048,
        cost_runtime=runtime,
        component_configuration_id=_PROJECT_CONFIGURATION_ID,
        lease_ledger_dir=(tmp_path / "leases").resolve(),
        create_rate_authority=_project_rate_authority(tmp_path),
    )
    project.authorize_search_dispatch(runtime.search_binding)

    with pytest.raises(RuntimeError, match="isolation differs"):
        project.write_text("context/materialize.json", "{}")

    assert len(created) == 1
    assert created[0].killed is True
