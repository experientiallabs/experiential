"""Tests for running ordinary agents inside a persistent filesystem project."""

from __future__ import annotations

from typing import Literal, overload

import pytest
from llm_waterfall import ChatResponse

import wmh.agents.project as project_module
from wmh.agents.default import default_agent
from wmh.agents.meta import meta_agent
from wmh.agents.project import AgentProject
from wmh.core.types import JsonObject
from wmh.providers.base import ProviderConfig, ProviderKind


class _Files:
    def __init__(self) -> None:
        self.values: dict[str, str | bytes] = {}

    def write(self, path: str, data: str) -> object:
        self.values[path] = data
        return None

    @overload
    def read(self, path: str) -> str: ...

    @overload
    def read(self, path: str, *, format: Literal["bytes"]) -> bytes: ...

    def read(self, path: str, *, format: Literal["bytes"] | None = None) -> str | bytes:
        value = self.values[path]
        if format == "bytes":
            assert isinstance(value, bytes)
            return value
        assert isinstance(value, str)
        return value


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
    assert channel.closed is True
    assert sandbox.commands.runs[:2] == [
        "mkdir -p /home/user/project",
        "mkdir -p /home/user/project/history",
    ]


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


def test_project_exports_one_deterministic_regular_file_archive() -> None:
    """One sandbox command archives project files without following links."""
    sandbox = _Sandbox()
    sandbox.files.values["/home/user/.wmh-agent-project.tar.gz"] = b"archive-bytes"
    project = AgentProject(sandbox, channel_factory=lambda sandbox, workspace: _Channel())

    content = project.export_archive()

    assert content == b"archive-bytes"
    archive_command, cleanup_command = sandbox.commands.runs[-2:]
    assert archive_command.startswith(
        "set -eo pipefail; cd /home/user/project && find . -mindepth 1 -xdev"
    )
    assert "-type f -o -type d" in archive_command
    assert "sort -z" in archive_command
    assert "--no-recursion" in archive_command
    assert "--mtime=@0" in archive_command
    assert "gzip -n" in archive_command
    assert cleanup_command == "rm -f /home/user/.wmh-agent-project.tar.gz"


def test_project_archive_rejects_oversize_and_closed_projects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Archives stay within the portable limit and must precede teardown."""
    sandbox = _Sandbox()
    sandbox.files.values["/home/user/.wmh-agent-project.tar.gz"] = b"large"
    project = AgentProject(sandbox, channel_factory=lambda sandbox, workspace: _Channel())
    monkeypatch.setattr(project_module, "MAX_PROJECT_ARCHIVE_BYTES", 4)

    with pytest.raises(ValueError, match="archive exceeds 4 bytes"):
        project.export_archive()

    project.close()
    with pytest.raises(RuntimeError, match="closed project"):
        project.export_archive()
