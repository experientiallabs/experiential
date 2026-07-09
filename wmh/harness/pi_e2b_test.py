"""Offline tests for the in-sandbox pi transport: fakes only, no E2B, no node, no provider.

A `_ScriptedHandle` plays the runner process (its stream events are scripted, stdin is recorded)
and a `FakeSandbox` implements the `SandboxHandle` slice, so `E2BStdioChannel` is exercised over
the real reader-thread/framing code path and `E2BPiRuntime.run` end-to-end — the runner script
speaks the same frames `runner_link_test.py`'s `_FakeChannel` does (hello → llm_request →
tool_request → done), and the environment answering tool calls is a real `E2BEnvironment` over the
same fake sandbox, proving env actions execute in the rollout's own VM.

No TS-side test: the repo has no TypeScript test precedent (no *_test.ts outside the untouchable
vendor tree, no root package.json), so runner_stdio.ts is covered by the frame-contract tests here.
"""

from __future__ import annotations

import base64
import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from wmh.core.types import Action, JsonObject, Observation
from wmh.harness.e2b_env import E2B_TEMPLATE_ENV, E2BEnvironment
from wmh.harness.pi_e2b import (
    NODE_INSTALL_CMD,
    PI_NPM_PACKAGES,
    RUNNER_WORKDIR,
    START_CMD,
    E2BPiRuntime,
    E2BStdioChannel,
)
from wmh.harness.runner_link import WorkerConfig
from wmh.harness.runtime import Runtime, StopReason
from wmh.harness.tools import SUBMIT, TOOL_REGISTRY, ToolSpec

_Event = tuple[str | None, str | None, str | None]
_PID = 4242


def _line(frame: JsonObject) -> str:
    """One frame as the runner would emit it: base64(JSON) + newline."""
    return base64.b64encode(json.dumps(frame).encode("utf-8")).decode("ascii") + "\n"


def _stdout_events(frames: list[JsonObject]) -> list[_Event]:
    return [(_line(f), None, None) for f in frames]


class _ScriptedHandle:
    """A fake background command handle: yields scripted (stdout, stderr, pty) events.

    `hold_open=True` keeps the stream open after the script (a live-but-silent runner, for the
    hello-timeout path); otherwise iteration ends, which the channel reads as process exit.
    """

    def __init__(self, events: list[_Event], *, hold_open: bool = False) -> None:
        self.pid = _PID
        self._events = list(events)
        self._hold_open = hold_open
        self._release = threading.Event()

    def __iter__(self) -> Iterator[_Event]:
        yield from self._events
        if self._hold_open:
            self._release.wait(2.0)  # self-releasing so the daemon reader never lingers


class _Result:
    """A minimal CommandOutput for foreground runs."""

    def __init__(self, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class _FakeCommands:
    """Foreground runs are recorded and echoed; background=True returns the scripted handle."""

    def __init__(self, handle: _ScriptedHandle) -> None:
        self._handle = handle
        self.calls: list[str] = []  # foreground commands, in order (installs, env bash, ...)
        self.background_cmds: list[str] = []
        self.stdin: list[tuple[int, str]] = []

    def run(
        self,
        cmd: str,
        background: bool | None = None,
        *,
        stdin: bool | None = None,
        timeout: float | None = None,
    ) -> object:
        if background:
            assert stdin is True  # the runner is useless without a writable stdin
            self.background_cmds.append(cmd)
            return self._handle
        self.calls.append(cmd)
        return _Result(stdout=f"ran: {cmd}")

    def send_stdin(self, pid: int, data: str) -> None:
        self.stdin.append((pid, data))


class _FakeFiles:
    """Records every write (path order preserved) and serves reads from the store."""

    def __init__(self) -> None:
        self.writes: list[str] = []
        self.store: dict[str, str] = {}

    def write(self, path: str, data: str) -> None:
        self.writes.append(path)
        self.store[path] = data

    def read(self, path: str) -> str:
        return self.store[path]


class FakeSandbox:
    """The `SandboxHandle` slice over a scripted runner process."""

    def __init__(self, handle: _ScriptedHandle) -> None:
        self.commands = _FakeCommands(handle)
        self.files = _FakeFiles()
        self.kills = 0

    def kill(self) -> bool:
        self.kills += 1
        return True


class _PlainEnv:
    """An AgentEnvironment that is not an E2BEnvironment (for the rejection test)."""

    def execute(self, action: Action) -> Observation:
        return Observation(content="nope")

    def close(self) -> None:
        pass


def _channel(fake: FakeSandbox, handle: _ScriptedHandle) -> E2BStdioChannel:
    return E2BStdioChannel(fake, handle)


def _tools() -> list[ToolSpec]:
    return [TOOL_REGISTRY["bash"], SUBMIT]


def _runtime(**kw: Any) -> E2BPiRuntime:  # noqa: ANN401 - test helper forwards ctor overrides
    defaults: dict[str, Any] = {
        "worker": WorkerConfig(),
        "files": {"src/agent.ts": "// a"},
        "tools": _tools(),
        "system_prompt": "sys",
        "template": None,
        "worker_fn": lambda body: {"choices": [{"message": {"content": "ok"}}]},
    }
    defaults.update(kw)
    return E2BPiRuntime(**defaults)


def _sent_frames(fake: FakeSandbox) -> list[Any]:
    """Decode every frame the host pushed into the runner's stdin (Any keeps deep asserts terse)."""
    lines = [data for _pid, data in fake.commands.stdin]
    return [json.loads(base64.b64decode(data.strip())) for data in lines]


def _of_kind(fake: FakeSandbox, kind: str) -> list[Any]:
    return [f for f in _sent_frames(fake) if f.get("type") == kind]


# --- E2BStdioChannel ---
def test_send_writes_base64_json_line_to_the_runner_pid() -> None:
    handle = _ScriptedHandle([], hold_open=True)
    fake = FakeSandbox(handle)
    channel = _channel(fake, handle)
    frame: JsonObject = {"type": "tool_response", "req_id": 1, "content": "café"}
    channel.send(frame)
    assert len(fake.commands.stdin) == 1
    pid, data = fake.commands.stdin[0]
    assert pid == _PID
    assert data.endswith("\n")
    assert json.loads(base64.b64decode(data.strip())) == frame


def test_recv_reassembles_partial_lines_and_collects_interleaved_stderr() -> None:
    hello: JsonObject = {"type": "hello", "n": 1}
    a: JsonObject = {"type": "llm_request", "req_id": 1}
    b: JsonObject = {"type": "done", "answer": "x"}
    line = _line(hello)
    events: list[_Event] = [
        (line[:10], None, None),  # partial line: no frame yet
        (None, "node warning one\n", None),  # stderr interleaved mid-frame
        (line[10:], None, None),  # completes the hello frame
        (None, "warn two\nwarn three", None),
        (_line(a) + _line(b), None, None),  # two frames in one event
    ]
    handle = _ScriptedHandle(events, hold_open=True)
    channel = _channel(FakeSandbox(handle), handle)
    assert channel.recv(timeout=2.0) == hello
    assert channel.recv(timeout=2.0) == a
    assert channel.recv(timeout=2.0) == b
    tail = channel.stderr_tail()
    assert "node warning one" in tail and "warn two" in tail and "warn three" in tail


def test_non_frame_stdout_noise_becomes_a_diagnostic_not_a_frame() -> None:
    ok: JsonObject = {"type": "hello"}
    events: list[_Event] = [
        ("stray print!!\n", None, None),  # not base64
        (base64.b64encode(b"[1, 2]").decode() + "\n", None, None),  # JSON but not an object
        (_line(ok), None, None),
    ]
    handle = _ScriptedHandle(events, hold_open=True)
    channel = _channel(FakeSandbox(handle), handle)
    assert channel.recv(timeout=2.0) == ok  # noise skipped, real frame delivered
    assert "stray print!!" in channel.stderr_tail()


def test_recv_after_process_exit_raises_with_recent_stderr() -> None:
    events: list[_Event] = [
        (_line({"type": "hello"}), None, None),
        (None, "Error: boom at agent.ts:7\n", None),
    ]
    handle = _ScriptedHandle(events)  # stream ends -> process exit
    channel = _channel(FakeSandbox(handle), handle)
    assert channel.recv(timeout=2.0) == {"type": "hello"}
    with pytest.raises(RuntimeError, match="exited mid-episode"):
        channel.recv()
    with pytest.raises(RuntimeError, match="boom at agent.ts:7"):  # sticky EOF, stderr included
        channel.recv()


def test_close_sends_shutdown_and_makes_the_stream_end_clean() -> None:
    handle = _ScriptedHandle([])
    fake = FakeSandbox(handle)
    channel = _channel(fake, handle)
    channel.close()
    channel.close()  # idempotent
    assert channel.recv() is None  # host-initiated shutdown reads as a clean channel close
    shutdowns = [f for f in _sent_frames(fake) if f["type"] == "shutdown"]
    assert len(shutdowns) == 1


# --- E2BPiRuntime ---
def test_run_rejects_non_e2b_environment() -> None:
    with pytest.raises(TypeError, match="E2BEnvironment"):
        _runtime().run("t1", "do it", _PlainEnv())


def test_satisfies_runtime_protocol() -> None:
    assert isinstance(_runtime(), Runtime)


def test_hello_timeout_error_includes_runner_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    handle = _ScriptedHandle([(None, "SyntaxError: unexpected token\n", None)], hold_open=True)
    fake = FakeSandbox(handle)
    env = E2BEnvironment(sandbox_factory=lambda: fake)
    with pytest.raises(RuntimeError, match="no hello") as excinfo:
        _runtime(hello_timeout=0.1).run("t1", "do it", env)
    assert "SyntaxError: unexpected token" in str(excinfo.value)


def test_end_to_end_fake_episode_delegates_to_runner_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    body: JsonObject = {"messages": [{"role": "user", "content": "hi"}]}
    script: list[JsonObject] = [
        {"type": "hello", "node_version": "v22.0.0"},
        {"type": "llm_request", "req_id": 1, "openai_body": body},
        {"type": "tool_request", "req_id": 2, "name": "bash", "arguments": {"command": "echo hi"}},
        {"type": "done", "answer": "finished"},
    ]
    handle = _ScriptedHandle(_stdout_events(script), hold_open=True)
    fake = FakeSandbox(handle)
    env = E2BEnvironment(sandbox_factory=lambda: fake)

    worker_calls: list[JsonObject] = []
    completion: JsonObject = {"choices": [{"message": {"content": "use bash"}}]}

    def worker(b: JsonObject) -> JsonObject:
        worker_calls.append(b)
        return completion

    result = _runtime(worker_fn=worker).run("t1", "do it", env)

    assert result.stop_reason is StopReason.SUBMITTED
    assert result.answer == "finished"
    assert len(result.steps) == 1  # the one brokered tool call
    step = result.steps[0]
    assert step.action == Action(
        kind=step.action.kind, name="bash", arguments={"command": "echo hi"}
    )
    assert step.observation.content == "ran: echo hi"  # produced by the sandbox, not a stub

    # Bootstrap ran against THIS sandbox: runner files up, node 22 + pinned pi deps installed.
    assert fake.files.store[f"{RUNNER_WORKDIR}/runner_stdio.ts"].startswith("/**")
    assert f"{RUNNER_WORKDIR}/runner_frames.ts" in fake.files.store
    assert fake.commands.calls[0] == NODE_INSTALL_CMD
    assert all(pkg in fake.commands.calls[1] for pkg in PI_NPM_PACKAGES)
    assert fake.commands.background_cmds == [START_CMD]

    # The tool_request went through environment.execute -> the same sandbox ran the command.
    assert fake.commands.calls[-1] == "echo hi"

    # Host -> runner frames: episode_start first, then the two answers, correlated by req_id.
    kinds = [f["type"] for f in _sent_frames(fake)]
    assert kinds == ["episode_start", "llm_response", "tool_response"]
    start = _of_kind(fake, "episode_start")[0]
    assert start["instruction"] == "do it" and start["system"] == "sys"
    assert start["files"] == {"src/agent.ts": "// a"}
    assert {t["name"] for t in start["tools"]} >= {"bash", "submit"}
    llm = _of_kind(fake, "llm_response")[0]
    assert llm["req_id"] == 1 and llm["completion"] == completion
    assert worker_calls == [body]  # answered host-side, by the injected worker
    tool = _of_kind(fake, "tool_response")[0]
    assert tool["req_id"] == 2 and tool["content"] == "ran: echo hi" and tool["is_error"] is False


def test_bootstrap_runs_once_for_two_episodes_on_the_same_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    script: list[JsonObject] = [
        {"type": "hello"},  # one hello: the runner process persists across episodes
        {"type": "done", "answer": "a1"},
        {"type": "done", "answer": "a2"},
    ]
    handle = _ScriptedHandle(_stdout_events(script), hold_open=True)
    fake = FakeSandbox(handle)
    env = E2BEnvironment(sandbox_factory=lambda: fake)
    runtime = _runtime()

    r1 = runtime.run("t1", "first", env)
    r2 = runtime.run("t2", "second", env)

    assert (r1.answer, r2.answer) == ("a1", "a2")
    assert fake.commands.calls.count(NODE_INSTALL_CMD) == 1  # installs once
    assert fake.commands.background_cmds == [START_CMD]  # one runner process
    assert fake.files.writes.count(f"{RUNNER_WORKDIR}/runner_stdio.ts") == 1  # files once
    starts = _of_kind(fake, "episode_start")
    assert len(starts) == 2
    assert starts[0]["episode_id"] != starts[1]["episode_id"]
    assert (starts[0]["instruction"], starts[1]["instruction"]) == ("first", "second")


def test_template_skips_node_and_npm_installs() -> None:
    script: list[JsonObject] = [{"type": "hello"}, {"type": "done", "answer": "ok"}]
    handle = _ScriptedHandle(_stdout_events(script), hold_open=True)
    fake = FakeSandbox(handle)
    env = E2BEnvironment(sandbox_factory=lambda: fake)

    result = _runtime(template="wmh-pi-node").run("t1", "do it", env)

    assert result.answer == "ok"
    assert fake.commands.calls == []  # no node upgrade, no npm install
    assert fake.commands.background_cmds == [START_CMD]  # the runner still starts
    assert f"{RUNNER_WORKDIR}/runner_stdio.ts" in fake.files.store  # repo files still refresh
    assert f"{RUNNER_WORKDIR}/package.json" not in fake.files.store  # template owns the layout
