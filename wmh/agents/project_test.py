"""Tests for running ordinary agents inside a persistent filesystem project."""

from __future__ import annotations

import pytest
from llm_waterfall import ChatResponse

from wmh.agents.default import default_agent
from wmh.agents.meta import meta_agent
from wmh.agents.project import AgentProject
from wmh.core.types import JsonObject
from wmh.providers.base import ProviderConfig, ProviderKind


class _Files:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def write(self, path: str, data: str) -> object:
        self.values[path] = data
        return None

    def read(self, path: str) -> str:
        return self.values[path]


class _Output:
    stdout = ""
    stderr = ""
    exit_code = 0


class _Commands:
    def __init__(self) -> None:
        self.runs: list[str] = []

    def run(self, cmd: str, background: bool | None = None, **kwargs: object) -> _Output:
        del background, kwargs
        self.runs.append(cmd)
        return _Output()

    def send_stdin(self, pid: int, data: str) -> object:
        del pid, data
        return None


class _Sandbox:
    def __init__(self) -> None:
        self.files = _Files()
        self.commands = _Commands()
        self.killed = False

    def set_timeout(self, timeout: int) -> None:
        del timeout

    def kill(self) -> object:
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
    assert any(frame["type"] == "session_start" for frame in channel.sent)
    assert any(frame["type"] == "user_message" for frame in channel.sent)
    assert channel.closed is False
    assert sandbox.commands.runs[:2] == [
        "mkdir -p /home/user/project",
        "mkdir -p /home/user/project/history",
    ]
    project.close()
    assert channel.closed is True


def test_project_reuses_one_agent_session_across_turns() -> None:
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


def test_project_rejects_agents_with_uncontained_tools() -> None:
    """Tool policy is enforced before a shell-enabled project session can start."""
    project = AgentProject(_Sandbox(), channel_factory=lambda sandbox, workspace: _Channel())

    with pytest.raises(ValueError, match="uncontained tools: bash"):
        project.run(default_agent(), _Provider(), "escape", timeout=1)

    outcome = project._execute_tool(
        "bash",
        {"command": "cat /home/user/runner.js"},
        lambda stream, data: None,
    )
    assert outcome.is_error is True
    assert outcome.content == "tool 'bash' not available"
