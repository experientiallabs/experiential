"""Tests for running ordinary agents inside a persistent filesystem project."""

from __future__ import annotations

from llm_waterfall import ChatResponse

from wmh.agents.default import default_agent
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
    result = project.run(default_agent(), _Provider(), "produce a result", timeout=1)

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
