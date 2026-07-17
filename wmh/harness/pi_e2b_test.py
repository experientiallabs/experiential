"""Offline tests for the in-sandbox pi transport: fakes only, no E2B, no node, no provider.

A `_ScriptedHandle` plays the runner process (its stream events are scripted, stdin is recorded)
and a `FakeSandbox` implements the `SandboxHandle` slice, so `E2BStdioChannel` is exercised over
the real reader-thread/framing code path, `E2BSandboxPool` over the real bootstrap/reuse/discard
lifecycle, and `E2BPiRuntime.run` end-to-end — the runner script speaks the same frames
`runner_link_test.py`'s `_FakeChannel` does (hello → llm_request → tool_request → done), and the
environment answering tool calls is a plain host-side `AgentEnvironment` fake: the sandbox is only
where the harness process lives, never where tool calls land.

The TypeScript runners have no package-level test harness here, so source-contract checks below
pin the host-provided model budget in addition to the Python frame-contract tests.
"""

from __future__ import annotations

import base64
import json
import shlex
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from llm_waterfall import ChatRequest, ChatResponse

from wmh.core.types import Action, JsonObject, Observation
from wmh.harness import e2b_sandbox as e2b_sandbox_module
from wmh.harness import pi_e2b as pi_e2b_module
from wmh.harness.e2b_sandbox import (
    E2B_TEMPLATE_ENV,
    SandboxCleanupError,
    SandboxFactory,
    SandboxUsage,
)
from wmh.harness.live_session import LiveSession, SessionEvent, ToolOutcome
from wmh.harness.pi_e2b import (
    LIVE_START_CMD,
    NODE_INSTALL_CMD,
    PI_NPM_PACKAGES,
    RUNNER_WORKDIR,
    START_CMD,
    TRANSPORT_KEEPALIVE_TYPE,
    E2BDurableChannel,
    E2BPiRuntime,
    E2BSandboxPool,
    E2BStdioChannel,
    session_entry_files,
    start_live_runner,
)
from wmh.harness.runtime import Runtime, RuntimeCancelled, StopReason
from wmh.harness.skills import Skill, SkillLibrary
from wmh.harness.tools import SUBMIT, TOOL_REGISTRY, ToolSpec

_Event = tuple[str | None, str | None, str | None]
_PID = 4242


class TimeoutException(Exception):
    """SDK-shaped timeout exception without importing the optional E2B package."""


TimeoutException.__module__ = "e2b.exceptions"


class RemoteProtocolError(Exception):
    """httpcore-shaped protocol exception without depending on its implementation."""


RemoteProtocolError.__module__ = "httpcore"


def test_all_pi_llm_bridges_forward_opaque_reasoning_details() -> None:
    """Every runner preserves stateless Responses reasoning through Pi's SSE parser."""
    entry = Path(pi_e2b_module.__file__).with_name("pi_entry")
    for filename in ("runner_stdio.ts", "runner_live.ts", "runner_service.ts"):
        source = (entry / filename).read_text(encoding="utf-8")
        assert "delta.reasoning_details = msg.reasoning_details" in source


def test_all_pi_llm_bridges_forward_usage_for_context_accounting() -> None:
    """Pi must see real usage instead of estimating an ever-growing transcript by characters."""
    entry = Path(pi_e2b_module.__file__).with_name("pi_entry")
    for filename in ("runner_stdio.ts", "runner_live.ts", "runner_service.ts"):
        source = (entry / filename).read_text(encoding="utf-8")
        assert "usage: reply.completion?.usage" in source


def test_all_pi_entrypoints_use_the_host_output_budget() -> None:
    """No execution mode may silently fall back to the old hard-coded 4k model ceiling."""
    entry = Path(pi_e2b_module.__file__).with_name("pi_entry")
    for filename in ("entry.ts", "runner_stdio.ts", "runner_live.ts", "runner_service.ts"):
        source = (entry / filename).read_text(encoding="utf-8")
        assert "max_output_tokens" in source
        assert "maxTokens: maxOutputTokens" in source
        assert "maxTokens: 4096" not in source


def _line(frame: JsonObject) -> str:
    """One frame as the runner would emit it: base64(JSON) + newline."""
    return base64.b64encode(json.dumps(frame).encode("utf-8")).decode("ascii") + "\n"


def _stdout_events(frames: list[JsonObject]) -> list[_Event]:
    return [(_line(f), None, None) for f in frames]


def _envelope(seq: int, frame: JsonObject) -> JsonObject:
    return {"transport_seq": seq, "frame": frame}


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
        self.disconnects = 0

    def __iter__(self) -> Iterator[_Event]:
        yield from self._events
        if self._hold_open:
            self._release.wait(2.0)  # self-releasing so the daemon reader never lingers

    def disconnect(self) -> None:
        self.disconnects += 1
        self._release.set()


class _DisconnectingHandle(_ScriptedHandle):
    """Yield a prefix, optionally wait for a race gate, then drop the output stream."""

    def __init__(self, events: list[_Event], *, gate: threading.Event | None = None) -> None:
        super().__init__(events)
        self._gate = gate

    def __iter__(self) -> Iterator[_Event]:
        yield from self._events
        if self._gate is not None:
            self._gate.wait(2.0)
        raise RuntimeError("Server disconnected")


class _GatedHandle(_ScriptedHandle):
    """Keep stdout silent until a test has consumed the same frame from disk."""

    def __init__(self, events: list[_Event], gate: threading.Event) -> None:
        super().__init__(events, hold_open=True)
        self._gate = gate

    def __iter__(self) -> Iterator[_Event]:
        self._gate.wait(2.0)
        yield from super().__iter__()


class _Result:
    """A minimal CommandOutput for foreground runs."""

    def __init__(self, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class _Process:
    def __init__(self, pid: int) -> None:
        self.pid = pid


class _FakeCommands:
    """Foreground runs are recorded and echoed; background=True returns the scripted handle."""

    def __init__(
        self,
        handle: _ScriptedHandle,
        *,
        reconnect_handles: Sequence[_ScriptedHandle] | None = None,
        on_stdin: Callable[[str], None] | None = None,
    ) -> None:
        self._handle = handle
        self._reconnect_handles = list(reconnect_handles or [])
        self.connect_started = threading.Event()
        self.connect_gate: threading.Event | None = None
        self.calls: list[str] = []  # foreground commands, in order (installs, ...)
        self.background_cmds: list[str] = []
        self.background_envs: list[dict[str, str] | None] = []
        self.connects: list[tuple[int, float | None]] = []
        self.stdin: list[tuple[int, str]] = []
        self.stdin_request_timeouts: list[float | None] = []
        self.running_pids = {_PID}
        self.killed: list[int] = []
        self.kill_request_timeouts: list[float | None] = []
        self.fail_sends_from: int | None = None
        self.fail_before_delivery: set[int] = set()
        self.send_error: Exception = TimeoutException("request timed out")
        self._on_stdin = on_stdin

    def run(
        self,
        cmd: str,
        background: bool | None = None,
        *,
        envs: dict[str, str] | None = None,
        stdin: bool | None = None,
        timeout: float | None = None,
    ) -> object:
        if background:
            assert stdin is True  # the runner is useless without a writable stdin
            self.background_cmds.append(cmd)
            self.background_envs.append(envs)
            return self._handle
        self.calls.append(cmd)
        return _Result(stdout=f"ran: {cmd}")

    def connect(self, pid: int, *, timeout: float | None = None) -> object:
        self.connects.append((pid, timeout))
        self.connect_started.set()
        if self.connect_gate is not None:
            self.connect_gate.wait(2.0)
        if not self._reconnect_handles:
            raise RuntimeError("process connection unavailable")
        return self._reconnect_handles.pop(0)

    def send_stdin(self, pid: int, data: str, request_timeout: float | None = None) -> None:
        self.stdin.append((pid, data))
        self.stdin_request_timeouts.append(request_timeout)
        if len(self.stdin) in self.fail_before_delivery:
            raise self.send_error
        if self._on_stdin is not None:
            self._on_stdin(data)
        if self.fail_sends_from is not None and len(self.stdin) >= self.fail_sends_from:
            raise self.send_error

    def list(self, request_timeout: float | None = None) -> Sequence[_Process]:
        del request_timeout
        return [_Process(pid) for pid in self.running_pids]

    def kill(self, pid: int, request_timeout: float | None = None) -> None:
        self.killed.append(pid)
        self.kill_request_timeouts.append(request_timeout)
        self.running_pids.discard(pid)


class _FakeFiles:
    """Records every write (path order preserved) and serves reads from the store."""

    def __init__(self) -> None:
        self.writes: list[str] = []
        self.store: dict[str, str] = {}

    def write(self, path: str, data: str) -> None:
        self.writes.append(path)
        self.store[path] = data

    def read(
        self,
        path: str,
        *,
        request_timeout: float | None = None,
        gzip: bool = False,
    ) -> str:
        del request_timeout, gzip
        return self.store[path]


class _TransientFiles(_FakeFiles):
    """Fail selected reads a bounded number of times to model E2B visibility/RPC races."""

    def __init__(self) -> None:
        super().__init__()
        self.failures: dict[str, int] = {}
        self.request_timeouts: list[float | None] = []
        self.gzip_values: list[bool] = []
        self.read_calls: list[tuple[str, float | None, bool]] = []

    def read(
        self,
        path: str,
        *,
        request_timeout: float | None = None,
        gzip: bool = False,
    ) -> str:
        self.request_timeouts.append(request_timeout)
        self.gzip_values.append(gzip)
        self.read_calls.append((path, request_timeout, gzip))
        remaining = self.failures.get(path, 0)
        if remaining:
            self.failures[path] = remaining - 1
            raise FileNotFoundError(path)
        return super().read(path, request_timeout=request_timeout, gzip=gzip)


class _LargeFrameFiles(_TransientFiles):
    """Require the durable frame path's long, gzip-enabled read contract."""

    def read(
        self,
        path: str,
        *,
        request_timeout: float | None = None,
        gzip: bool = False,
    ) -> str:
        if "/frames/" in path and ((request_timeout or 0) < 1.0 or not gzip):
            raise TimeoutError("large replay frame needs a compressed multi-second read")
        return super().read(path, request_timeout=request_timeout, gzip=gzip)


class _BlockingFrameFiles(_FakeFiles):
    """Hold one exact frame read until a cancellation test releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.frame_read_started = threading.Event()
        self.release_frame_read = threading.Event()

    def read(
        self,
        path: str,
        *,
        request_timeout: float | None = None,
        gzip: bool = False,
    ) -> str:
        if "/frames/" in path:
            self.frame_read_started.set()
            self.release_frame_read.wait(2.0)
        return super().read(path, request_timeout=request_timeout, gzip=gzip)


class _BlockingBootstrapFiles(_FakeFiles):
    """Hold the first bootstrap write so pool cancellation can race cold start."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def write(self, path: str, data: str) -> None:
        self.started.set()
        self.release.wait(2.0)
        super().write(path, data)


class FakeSandbox:
    """The `SandboxHandle` slice over a scripted runner process."""

    def __init__(
        self,
        handle: _ScriptedHandle,
        *,
        reconnect_handles: list[_ScriptedHandle] | None = None,
    ) -> None:
        self.files = _FakeFiles()
        self.durable_dispatches: list[JsonObject] = []
        self._last_durable_inbound_seq = 0
        self.drop_durable_acks = 0
        self.commands = _FakeCommands(
            handle,
            reconnect_handles=reconnect_handles,
            on_stdin=self._accept_durable_inbound,
        )
        self.kills = 0
        self.timeouts: list[int] = []
        self.dead = False

    def set_timeout(self, timeout: int) -> None:
        if self.dead:
            raise RuntimeError("sandbox not found")
        self.timeouts.append(timeout)

    def kill(self, request_timeout: float | None = None) -> bool:
        del request_timeout
        self.kills += 1
        return True

    def _accept_durable_inbound(self, data: str) -> None:
        """Model runner-side inbound dedupe and append its ack to the durable fake outbox."""
        try:
            value = json.loads(base64.b64decode(data.strip()))
        except (ValueError, TypeError):
            return
        if not isinstance(value, dict):
            return
        inbound_seq = value.get("transport_in_seq")
        frame = value.get("frame")
        if (
            isinstance(inbound_seq, bool)
            or not isinstance(inbound_seq, int)
            or inbound_seq <= 0
            or not isinstance(frame, dict)
        ):
            return
        head_paths = [path for path in self.files.store if path.endswith("/head")]
        if not head_paths:
            return  # legacy stdio runner: raw frames have no transport envelope anyway
        root = head_paths[-1].removesuffix("/head")
        output_seq = int(self.files.store[head_paths[-1]].strip()) + 1
        ack: JsonObject
        if inbound_seq == self._last_durable_inbound_seq + 1:
            self._last_durable_inbound_seq = inbound_seq
            self.durable_dispatches.append(cast("JsonObject", frame))
            ack = {"type": "transport_ack", "transport_in_seq": inbound_seq}
        elif inbound_seq <= self._last_durable_inbound_seq:
            ack = {"type": "transport_ack", "transport_in_seq": inbound_seq}
        else:
            ack = {
                "type": "transport_nack",
                "transport_in_seq": inbound_seq,
                "expected_transport_in_seq": self._last_durable_inbound_seq + 1,
            }
        if self.drop_durable_acks > 0:
            self.drop_durable_acks -= 1
            return
        self.files.store[f"{root}/frames/{output_seq:020d}.json"] = json.dumps(
            _envelope(output_seq, ack)
        )
        self.files.store[head_paths[-1]] = str(output_seq)


class _RecordingEnv:
    """ANY host-side `AgentEnvironment` (the world-model shape in real evals): records executes."""

    def __init__(self) -> None:
        self.actions: list[Action] = []

    def execute(self, action: Action) -> Observation:
        self.actions.append(action)
        return Observation(content="wm says ok")

    def close(self) -> None:
        pass


def _channel(
    fake: FakeSandbox,
    handle: _ScriptedHandle,
    *,
    sandbox_timeout_s: int | None = None,
    timeout_refresh_interval_s: float = 300.0,
    max_episode_lifetime_s: float = 3_600.0,
    reconnect_while_idle: bool = False,
) -> E2BStdioChannel:
    return E2BStdioChannel(
        fake,
        handle,
        sandbox_timeout_s=sandbox_timeout_s,
        timeout_refresh_interval_s=timeout_refresh_interval_s,
        max_episode_lifetime_s=max_episode_lifetime_s,
        reconnect_while_idle=reconnect_while_idle,
    )


def _tools() -> list[ToolSpec]:
    return [TOOL_REGISTRY["bash"], SUBMIT]


def _completion(content: str = "ok") -> ChatResponse:
    return ChatResponse.model_validate(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ]
        }
    )


class _Provider:
    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        del request
        return _completion()


def _factory_for(
    scripts: list[list[JsonObject]],
) -> tuple[Callable[[], FakeSandbox], list[FakeSandbox]]:
    """A sandbox factory: each call makes a FakeSandbox whose runner plays the next script."""
    made: list[FakeSandbox] = []
    remaining = [list(script) for script in scripts]

    def factory() -> FakeSandbox:
        frames = remaining.pop(0) if remaining else [{"type": "hello"}]
        fake = FakeSandbox(_ScriptedHandle(_stdout_events(frames), hold_open=True))
        made.append(fake)
        return fake

    return factory, made


def _runtime(
    *,
    pool: E2BSandboxPool | None = None,
    template: str | None = None,
    worker_fn: Callable[[ChatRequest], ChatResponse] | None = None,
    max_turns: int = 20,
    max_output_tokens: int = 4096,
    temperature: float = 0.7,
    skills: SkillLibrary | None = None,
    episode_timeout_s: float = 300.0,
    should_cancel: Callable[[], bool] | None = None,
) -> E2BPiRuntime:
    return E2BPiRuntime(
        provider=_Provider(),
        files={"src/agent.ts": "// a"},
        tools=_tools(),
        system_prompt="sys",
        template=template,
        pool=pool,
        worker_fn=worker_fn,
        max_turns=max_turns,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        skills=skills,
        episode_timeout_s=episode_timeout_s,
        should_cancel=should_cancel,
    )


def _sent_frames(fake: FakeSandbox) -> list[JsonObject]:
    """Decode every logical frame the host pushed, unwrapping durable inbound envelopes."""
    lines = [data for _pid, data in fake.commands.stdin]
    decoded = [cast("JsonObject", json.loads(base64.b64decode(data.strip()))) for data in lines]
    return [
        cast("JsonObject", value["frame"])
        if isinstance(value.get("transport_in_seq"), int) and isinstance(value.get("frame"), dict)
        else value
        for value in decoded
    ]


def _of_kind(fake: FakeSandbox, kind: str) -> list[JsonObject]:
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


def test_transport_keepalive_renews_a_pooled_lease_without_becoming_a_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Active runner heartbeats keep E2B alive but stay below RunnerLink's protocol."""
    now = [100.0]
    monkeypatch.setattr(pi_e2b_module.time, "monotonic", lambda: now[0])
    keepalive: JsonObject = {"type": TRANSPORT_KEEPALIVE_TYPE}
    hello: JsonObject = {"type": "hello"}
    handle = _ScriptedHandle([], hold_open=True)
    fake = FakeSandbox(handle)
    channel = _channel(fake, handle, sandbox_timeout_s=900)

    channel.send({"type": "episode_start"})
    channel._decode_line(_line(keepalive))  # noqa: SLF001 - exercise the reader's exact decoder
    assert fake.timeouts == []  # the pool's initial reset covers the first refresh window
    now[0] += 300
    channel._decode_line(_line(keepalive))  # noqa: SLF001
    channel._decode_line(_line(keepalive))  # noqa: SLF001 - throttled at the same timestamp
    channel._decode_line(_line(hello))  # noqa: SLF001
    assert channel.recv(timeout=2.0) == hello
    assert fake.timeouts == [900]


def test_transport_keepalive_stops_renewing_at_the_episode_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A yielding bad candidate cannot turn lease renewal into an unbounded sandbox leak."""
    now = [100.0]
    monkeypatch.setattr(pi_e2b_module.time, "monotonic", lambda: now[0])
    keepalive: JsonObject = {"type": TRANSPORT_KEEPALIVE_TYPE}
    handle = _ScriptedHandle([], hold_open=True)
    fake = FakeSandbox(handle)
    channel = _channel(
        fake,
        handle,
        sandbox_timeout_s=900,
        timeout_refresh_interval_s=300,
        max_episode_lifetime_s=600,
    )

    channel.send({"type": "episode_start"})
    now[0] += 300
    channel._decode_line(_line(keepalive))  # noqa: SLF001
    now[0] += 300
    channel._decode_line(_line(keepalive))  # noqa: SLF001

    assert fake.timeouts == [300]  # the final refresh expires exactly at the hard deadline


def test_transport_keepalive_refresh_failure_is_nonfatal_and_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A control-plane blip cannot poison the frame stream; the next heartbeat retries."""

    class FlakyTimeoutSandbox(FakeSandbox):
        def __init__(self, handle: _ScriptedHandle) -> None:
            super().__init__(handle)
            self.refresh_attempts = 0

        def set_timeout(self, timeout: int) -> None:
            self.refresh_attempts += 1
            if self.refresh_attempts == 1:
                raise RuntimeError("temporary control-plane failure")
            super().set_timeout(timeout)

    now = [100.0]
    monkeypatch.setattr(pi_e2b_module.time, "monotonic", lambda: now[0])
    keepalive: JsonObject = {"type": TRANSPORT_KEEPALIVE_TYPE}
    hello: JsonObject = {"type": "hello"}
    handle = _ScriptedHandle([], hold_open=True)
    fake = FlakyTimeoutSandbox(handle)
    channel = _channel(fake, handle, sandbox_timeout_s=900)

    channel.send({"type": "episode_start"})
    now[0] += 300
    channel._decode_line(_line(keepalive))  # noqa: SLF001
    channel._decode_line(_line(keepalive))  # noqa: SLF001
    channel._decode_line(_line(hello))  # noqa: SLF001
    assert channel.recv(timeout=2.0) == hello
    assert fake.refresh_attempts == 2
    assert fake.timeouts == [900]
    assert "sandbox timeout refresh failed" in channel.stderr_tail()


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


def test_live_channel_reconnects_same_pid_after_an_idle_stream_drop() -> None:
    """An idle reconnect preserves the runner transcript without surfacing a false process exit."""
    idle: JsonObject = {"type": "state", "status": "idle"}
    after_reconnect: JsonObject = {"type": "pong", "nonce": "still-here"}
    dropped = _DisconnectingHandle(_stdout_events([idle]))
    resumed = _ScriptedHandle(_stdout_events([after_reconnect]), hold_open=True)
    fake = FakeSandbox(dropped, reconnect_handles=[resumed])
    channel = _channel(fake, dropped, reconnect_while_idle=True)

    assert channel.recv(timeout=2.0) == idle
    assert channel.recv(timeout=2.0) == after_reconnect
    assert fake.commands.connects == [(_PID, 0)]
    assert "output stream failed" not in channel.stderr_tail()


def test_live_channel_reconnects_when_an_idle_stream_ends_without_an_exception() -> None:
    """The SDK can report a dropped HTTP stream as normal iterator exhaustion."""
    idle: JsonObject = {"type": "state", "status": "idle"}
    after_reconnect: JsonObject = {"type": "pong", "nonce": "still-here"}
    ended = _ScriptedHandle(_stdout_events([idle]))
    resumed = _ScriptedHandle(_stdout_events([after_reconnect]), hold_open=True)
    fake = FakeSandbox(ended, reconnect_handles=[resumed])
    channel = _channel(fake, ended, reconnect_while_idle=True)

    assert channel.recv(timeout=2.0) == idle
    assert channel.recv(timeout=2.0) == after_reconnect
    assert fake.commands.connects == [(_PID, 0)]


def test_live_channel_does_not_reconnect_a_busy_stream() -> None:
    """Mid-turn reconnect cannot prove whether a semantic frame was lost, so it fails closed."""
    running: JsonObject = {"type": "state", "status": "running"}
    dropped = _DisconnectingHandle(_stdout_events([running]))
    fake = FakeSandbox(
        dropped,
        reconnect_handles=[_ScriptedHandle([], hold_open=True)],
    )
    channel = _channel(fake, dropped, reconnect_while_idle=True)

    assert channel.recv(timeout=2.0) == running
    with pytest.raises(RuntimeError, match="Server disconnected"):
        channel.recv(timeout=2.0)
    assert fake.commands.connects == []


def test_user_message_wins_the_race_with_idle_reconnect() -> None:
    """Turn delivery clears the idle proof before stdin, so a concurrent drop is not resumed."""
    gate = threading.Event()
    idle: JsonObject = {"type": "state", "status": "idle"}
    dropped = _DisconnectingHandle(_stdout_events([idle]), gate=gate)
    fake = FakeSandbox(
        dropped,
        reconnect_handles=[_ScriptedHandle([], hold_open=True)],
    )
    channel = _channel(fake, dropped, reconnect_while_idle=True)

    assert channel.recv(timeout=2.0) == idle
    channel.send({"type": "user_message", "msg_id": "next", "text": "go"})
    gate.set()
    with pytest.raises(RuntimeError, match="Server disconnected"):
        channel.recv(timeout=2.0)
    assert fake.commands.connects == []


# --- E2BSandboxPool ---
def test_pool_acquire_creates_bootstraps_starts_runner_and_awaits_hello(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    factory, made = _factory_for([[{"type": "hello", "node_version": "v22.0.0"}]])
    pool = E2BSandboxPool(sandbox_factory=factory)
    sandbox, channel = pool.acquire()
    assert sandbox is made[0]
    fake = made[0]
    # Bootstrap: runner files up, node 22 + pinned pi deps installed, the runner started.
    assert fake.files.store[f"{RUNNER_WORKDIR}/runner_stdio.ts"].startswith("/**")
    assert TRANSPORT_KEEPALIVE_TYPE in fake.files.store[f"{RUNNER_WORKDIR}/runner_stdio.ts"]
    assert ".unref()" in fake.files.store[f"{RUNNER_WORKDIR}/runner_stdio.ts"]
    assert f"{RUNNER_WORKDIR}/runner_frames.ts" in fake.files.store
    assert fake.commands.calls[0] == NODE_INSTALL_CMD
    assert all(pkg in fake.commands.calls[1] for pkg in PI_NPM_PACKAGES)
    assert fake.commands.background_cmds == [START_CMD]
    # The hello was consumed by acquire: the channel is idle, ready for episode frames.
    with pytest.raises(TimeoutError):
        channel.recv(timeout=0.05)
    pool.close()


def test_pool_reuses_a_healthy_sandbox_without_rebootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    factory, made = _factory_for([[{"type": "hello"}]])
    pool = E2BSandboxPool(sandbox_factory=factory)
    sandbox, channel = pool.acquire()
    pool.release(sandbox, channel, healthy=True)
    again, channel_again = pool.acquire()
    assert again is sandbox and channel_again is channel  # the SAME warm sandbox came back
    assert len(made) == 1  # no second sandbox was ever created
    assert made[0].commands.calls.count(NODE_INSTALL_CMD) == 1  # bootstrap paid once
    assert made[0].commands.background_cmds == [START_CMD]  # one runner process
    pool.close()


def test_pool_retire_idle_rotates_only_free_sandboxes_and_preserves_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An iteration boundary kills idle streams and meters both runner lifetimes."""
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    now = [10.0]
    monkeypatch.setattr(pi_e2b_module.time, "monotonic", lambda: now[0])
    factory, made = _factory_for([[{"type": "hello"}], [{"type": "hello"}]])
    pool = E2BSandboxPool(sandbox_factory=factory)
    idle, idle_channel = pool.acquire()
    in_flight, in_flight_channel = pool.acquire()
    pool.release(idle, idle_channel, healthy=True)

    now[0] = 14.0
    assert pool.retire_idle() == 1
    assert made[0].kills == 1
    assert made[1].kills == 0  # an active episode is outside the idle rotation boundary
    assert pool.usage() == SandboxUsage(count=2, seconds=8.0)

    pool.release(in_flight, in_flight_channel, healthy=True)
    reused, _ = pool.acquire()
    assert reused is in_flight  # work that was active at the boundary remains safely reusable
    assert len(made) == 2
    now[0] = 16.0
    pool.close()
    assert pool.usage() == SandboxUsage(count=2, seconds=10.0)


def test_pool_discards_an_unhealthy_sandbox_and_creates_a_fresh_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    factory, made = _factory_for([[{"type": "hello"}], [{"type": "hello"}]])
    pool = E2BSandboxPool(sandbox_factory=factory)
    sandbox, channel = pool.acquire()
    pool.release(sandbox, channel, healthy=False)
    assert made[0].kills == 1  # a failed episode's runner state is unknown: never reused
    fresh, _fresh_channel = pool.acquire()
    assert fresh is made[1] and fresh is not sandbox
    assert made[1].commands.calls.count(NODE_INSTALL_CMD) == 1  # the fresh one bootstrapped
    pool.close()


def test_pool_default_factory_tags_initial_and_replacement_sandboxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    factory, made = _factory_for([[{"type": "hello"}], [{"type": "hello"}]])
    factory_calls: list[dict[str, object]] = []

    def recording_default_factory(**kwargs: object) -> SandboxFactory:
        factory_calls.append(kwargs)
        return factory

    monkeypatch.setattr(pi_e2b_module, "default_sandbox_factory", recording_default_factory)
    metadata = {"optimizer_run_id": "run-1", "purpose": "evaluation"}
    pool = E2BSandboxPool(template="tmpl", api_key="key", metadata=metadata)

    sandbox, channel = pool.acquire()
    pool.release(sandbox, channel, healthy=False)
    replacement, _ = pool.acquire()

    assert replacement is made[1]
    assert len(made) == 2
    assert factory_calls == [{"api_key": "key", "template": "tmpl", "metadata": metadata}]
    pool.close()


def test_pool_close_kills_everything_and_acquire_after_close_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    factory, made = _factory_for([[{"type": "hello"}], [{"type": "hello"}]])
    pool = E2BSandboxPool(sandbox_factory=factory)
    first, first_channel = pool.acquire()
    pool.acquire()  # a second, still-in-flight sandbox
    pool.release(first, first_channel, healthy=True)  # one idle, one in flight
    pool.close()
    assert [fake.kills for fake in made] == [1, 1]
    with pytest.raises(RuntimeError, match="closed"):
        pool.acquire()
    pool.close()  # idempotent


def test_pool_failed_kill_stays_live_in_usage_and_a_later_close_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unproven kill remains owned and billable until a retry succeeds."""
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    monkeypatch.setattr(e2b_sandbox_module.time, "sleep", lambda delay: None)
    now = [10.0]
    monkeypatch.setattr(pi_e2b_module.time, "monotonic", lambda: now[0])
    factory, made = _factory_for([[{"type": "hello"}]])
    pool = E2BSandboxPool(sandbox_factory=factory)
    pool.acquire()
    fake = made[0]
    attempts = 0

    def broken_kill(request_timeout: float | None = None) -> bool:
        nonlocal attempts
        del request_timeout
        attempts += 1
        raise RuntimeError("control plane unavailable")

    monkeypatch.setattr(fake, "kill", broken_kill)
    now[0] = 14.0
    with pytest.raises(SandboxCleanupError, match="failed to prove cleanup") as raised:
        pool.close()
    assert attempts == 3
    assert raised.value.resource == "evaluator_sandbox_pool"
    assert raised.value.sandbox_usage == SandboxUsage(count=1, seconds=4.0)
    assert pool.usage() == SandboxUsage(count=1, seconds=4.0)

    now[0] = 16.0
    assert pool.usage() == SandboxUsage(count=1, seconds=6.0)

    def successful_kill(request_timeout: float | None = None) -> bool:
        nonlocal attempts
        del request_timeout
        attempts += 1
        return True

    monkeypatch.setattr(fake, "kill", successful_kill)
    pool.close()
    assert attempts == 4
    assert pool.usage() == SandboxUsage(count=1, seconds=6.0)
    pool.close()  # proved cleanup remains idempotent


def test_pool_close_attempts_every_sandbox_when_one_kill_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One failed teardown cannot prevent sibling leases from being released."""
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    monkeypatch.setattr(e2b_sandbox_module.time, "sleep", lambda delay: None)
    factory, made = _factory_for([[{"type": "hello"}], [{"type": "hello"}]])
    pool = E2BSandboxPool(sandbox_factory=factory)
    pool.acquire()
    pool.acquire()

    def broken_kill(request_timeout: float | None = None) -> bool:
        del request_timeout
        made[0].kills += 1
        raise RuntimeError("control plane unavailable")

    monkeypatch.setattr(made[0], "kill", broken_kill)
    with pytest.raises(SandboxCleanupError, match="1 of 2"):
        pool.close()

    assert made[0].kills == 3
    assert made[1].kills == 1


def test_inflight_release_after_pool_close_does_not_kill_the_lease_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    factory, made = _factory_for([[{"type": "hello"}]])
    pool = E2BSandboxPool(sandbox_factory=factory)
    sandbox, channel = pool.acquire()

    pool.close()
    pool.release(sandbox, channel, healthy=False)  # in-flight episode unwinds after cancellation

    assert made[0].kills == 1


def test_pool_close_kills_a_sandbox_while_its_bootstrap_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation retires cold-start leases without waiting for their bootstrap RPC."""
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    handle = _ScriptedHandle(_stdout_events([{"type": "hello"}]), hold_open=True)
    fake = FakeSandbox(handle)
    files = _BlockingBootstrapFiles()
    fake.files = files
    pool = E2BSandboxPool(sandbox_factory=lambda: fake)
    errors: list[BaseException] = []

    def acquire() -> None:
        try:
            pool.acquire()
        except BaseException as exc:  # noqa: BLE001 - the worker hands its failure to the test
            errors.append(exc)

    worker = threading.Thread(target=acquire)
    worker.start()
    assert files.started.wait(1.0)

    pool.close()
    assert fake.kills == 1  # killed before the blocked bootstrap call is released

    files.release.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert len(errors) == 1
    assert "closed" in str(errors[0])
    assert fake.kills == 1  # bootstrap cleanup is idempotent after close already retired it


def test_pool_acquire_without_hello_raises_with_stderr_and_kills_the_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    handle = _ScriptedHandle([(None, "SyntaxError: unexpected token\n", None)], hold_open=True)
    fake = FakeSandbox(handle)
    pool = E2BSandboxPool(sandbox_factory=lambda: fake, hello_timeout=0.1)
    with pytest.raises(RuntimeError, match="no hello") as excinfo:
        pool.acquire()
    assert "SyntaxError: unexpected token" in str(excinfo.value)
    assert fake.kills == 1  # a sandbox that failed bootstrap never leaks


def test_pool_with_template_skips_installs_but_still_writes_runner_files() -> None:
    factory, made = _factory_for([[{"type": "hello"}]])
    pool = E2BSandboxPool(template="wmh-pi-node", sandbox_factory=factory)
    pool.acquire()
    fake = made[0]
    assert fake.commands.calls == []  # no node upgrade, no npm install
    assert fake.commands.background_cmds == [START_CMD]  # the runner still starts
    assert f"{RUNNER_WORKDIR}/runner_stdio.ts" in fake.files.store  # repo files still refresh
    assert f"{RUNNER_WORKDIR}/package.json" not in fake.files.store  # template owns the layout
    pool.close()


# --- E2BPiRuntime ---
def test_satisfies_runtime_protocol() -> None:
    assert isinstance(_runtime(), Runtime)


def test_end_to_end_fake_episode_answers_tools_via_the_host_side_environment(
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
    factory, made = _factory_for([script])
    env = _RecordingEnv()  # a plain AgentEnvironment: the world-model shape in real evals

    worker_calls: list[ChatRequest] = []
    completion = _completion("use bash")

    def worker(request: ChatRequest) -> ChatResponse:
        worker_calls.append(request)
        return completion

    with E2BSandboxPool(sandbox_factory=factory) as pool:
        result = _runtime(
            pool=pool,
            worker_fn=worker,
            max_turns=7,
            max_output_tokens=16384,
            temperature=0.35,
        ).run("t1", "do it", env)

    assert result.stop_reason is StopReason.SUBMITTED
    assert result.answer == "finished"
    assert len(result.steps) == 1  # the one brokered tool call
    step = result.steps[0]
    assert step.action == Action(
        kind=step.action.kind, name="bash", arguments={"command": "echo hi"}
    )
    # The tool_request went through environment.execute — answered HOST-side, not in the sandbox.
    assert [a.name for a in env.actions] == ["bash"]
    assert step.observation.content == "wm says ok"

    # Host -> runner frames: episode_start first, then the two answers, correlated by req_id.
    fake = made[0]
    kinds = [f["type"] for f in _sent_frames(fake)]
    assert kinds == ["episode_start", "llm_response", "tool_response"]
    start = _of_kind(fake, "episode_start")[0]
    assert start["instruction"] == "do it" and start["system"] == "sys"
    assert start["files"] == {"src/agent.ts": "// a"}
    assert start["max_turns"] == 7
    assert start["max_output_tokens"] == 16384
    assert start["temperature"] == 0.35
    assert start["episode_timeout_s"] == 300.0
    tool_names = {t["name"] for t in cast("list[JsonObject]", start["tools"])}
    assert tool_names >= {"bash", "submit"}
    llm = _of_kind(fake, "llm_response")[0]
    assert llm["req_id"] == 1 and llm["completion"] == completion.wire_payload()
    assert [request.messages[0].content for request in worker_calls] == ["hi"]
    assert [request.temperature for request in worker_calls] == [0.35]
    tool = _of_kind(fake, "tool_response")[0]
    assert tool["req_id"] == 2 and tool["content"] == "wm says ok" and tool["is_error"] is False


def test_skill_bodies_are_advertised_and_served_host_side_in_e2b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    script: list[JsonObject] = [
        {"type": "hello"},
        {
            "type": "tool_request",
            "req_id": 1,
            "name": "read_skill",
            "arguments": {"name": "count-words"},
        },
        {
            "type": "tool_request",
            "req_id": 2,
            "name": "read_skill",
            "arguments": {"name": "missing"},
        },
        {"type": "done", "answer": "finished"},
    ]
    factory, made = _factory_for([script])
    env = _RecordingEnv()
    skills = SkillLibrary(
        [Skill(name="count-words", description="count words", body="wc -w <path>")]
    )

    with E2BSandboxPool(sandbox_factory=factory) as pool:
        result = _runtime(pool=pool, skills=skills).run("t1", "do it", env)

    start = _of_kind(made[0], "episode_start")[0]
    assert "read_skill" in {tool["name"] for tool in cast("list[JsonObject]", start["tools"])}
    responses = _of_kind(made[0], "tool_response")
    assert responses[0]["content"] == "wc -w <path>" and responses[0]["is_error"] is False
    assert responses[1]["content"] == "no skill named 'missing'"
    assert responses[1]["is_error"] is True
    assert [step.action.name for step in result.steps] == ["read_skill", "read_skill"]
    assert env.actions == []


def test_two_runtimes_sharing_a_pool_reuse_warm_sandboxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    script: list[JsonObject] = [
        {"type": "hello"},  # one hello: the runner process persists across episodes and docs
        {"type": "done", "answer": "a1"},
        {"type": "done", "answer": "a2"},
    ]
    factory, made = _factory_for([script])
    with E2BSandboxPool(sandbox_factory=factory) as pool:
        first = _runtime(pool=pool)
        second = _runtime(pool=pool)
        r1 = first.run("t1", "first", _RecordingEnv())
        r2 = second.run("t2", "second", _RecordingEnv())

        assert (r1.answer, r2.answer) == ("a1", "a2")
        assert len(made) == 1  # both runtimes drew the SAME warm sandbox
        fake = made[0]
        assert fake.commands.calls.count(NODE_INSTALL_CMD) == 1  # bootstrap once, not per doc
        assert fake.commands.background_cmds == [START_CMD]  # one runner process
        assert fake.files.writes.count(f"{RUNNER_WORKDIR}/runner_stdio.ts") == 1  # files once
        starts = _of_kind(fake, "episode_start")
        assert len(starts) == 2
        assert starts[0]["episode_id"] != starts[1]["episode_id"]
        assert (starts[0]["instruction"], starts[1]["instruction"]) == ("first", "second")


def test_pi_episode_error_is_not_retried_and_keeps_the_sandbox_reusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runner-reported episode failure stays a result, not a transport retry."""
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    script: list[JsonObject] = [
        {"type": "hello"},
        {"type": "episode_error", "note": "pi failed"},
        {"type": "done", "answer": "next episode"},
    ]
    factory, made = _factory_for([script])
    pool = E2BSandboxPool(sandbox_factory=factory)
    runtime = _runtime(pool=pool)

    failed = runtime.run("t1", "first", _RecordingEnv())
    recovered = runtime.run("t2", "second", _RecordingEnv())

    assert failed.stop_reason is StopReason.ERROR
    assert "pi failed" in failed.steps[-1].observation.content
    assert recovered.answer == "next episode"
    assert len(made) == 1
    assert made[0].kills == 0
    pool.close()


def test_episode_wall_budget_retires_without_retry_or_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    factory, made = _factory_for(
        [
            [{"type": "hello"}],
            [{"type": "hello"}, {"type": "done", "answer": "fresh"}],
        ]
    )
    pool = E2BSandboxPool(sandbox_factory=factory)
    runtime = _runtime(pool=pool, episode_timeout_s=0.01)

    expired = runtime.run("t1", "first", _RecordingEnv())
    recovered = runtime.run("t2", "second", _RecordingEnv())

    assert expired.stop_reason is StopReason.BUDGET
    assert "wall budget" in expired.transcript()
    assert recovered.answer == "fresh"
    assert len(made) == 2  # no retry for the expired episode; the next episode gets a fresh lease
    assert made[0].kills == 1
    pool.close()


def test_runtime_cancellation_retires_the_active_sandbox_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    factory, made = _factory_for([[{"type": "hello"}]])
    pool = E2BSandboxPool(sandbox_factory=factory)
    checks = 0

    def should_cancel() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 3

    with pytest.raises(RuntimeCancelled, match="cancelled"):
        _runtime(pool=pool, should_cancel=should_cancel).run("t1", "first", _RecordingEnv())

    assert len(made) == 1
    assert made[0].kills == 1
    pool.close()


def test_close_kills_a_private_pool_but_never_a_shared_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    # Private pool: pool=None makes the runtime build its own from the (patched) default factory.
    factory, made = _factory_for([[{"type": "hello"}, {"type": "done", "answer": "a"}]])
    monkeypatch.setattr(pi_e2b_module, "default_sandbox_factory", lambda **_kw: factory)
    private = _runtime()
    assert private.run("t1", "go", _RecordingEnv()).answer == "a"
    private.close()
    assert made[0].kills == 1  # the runtime owned its pool, so close tore the sandbox down

    # Shared pool: the pool's owner (a whole search) outlives any one runtime.
    factory2, made2 = _factory_for([[{"type": "hello"}, {"type": "done", "answer": "b"}]])
    pool = E2BSandboxPool(sandbox_factory=factory2)
    shared = _runtime(pool=pool)
    assert shared.run("t1", "go", _RecordingEnv()).answer == "b"
    shared.close()
    assert made2[0].kills == 0  # closing the runtime must not kill the shared pool's sandboxes
    pool.close()
    assert made2[0].kills == 1


def test_acquire_extends_the_lifetime_of_a_reused_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse restarts E2B's lifetime countdown so long searches outlive the creation cap."""
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    factory, made = _factory_for([[{"type": "hello"}]])
    pool = E2BSandboxPool(sandbox_factory=factory)
    sandbox, channel = pool.acquire()
    [fake] = made
    assert fake.timeouts == [900]  # reset after bootstrap, immediately before the runner starts
    pool.release(sandbox, channel, healthy=True)
    again, _ = pool.acquire()
    assert again is sandbox
    assert fake.timeouts == [900, 900]  # the reuse extended the countdown again
    pool.close()


def test_acquire_replaces_a_dead_idle_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    """An idle sandbox past its lifetime fails the extension: retired, fresh one created."""
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    factory, made = _factory_for([[{"type": "hello"}], [{"type": "hello"}]])
    pool = E2BSandboxPool(sandbox_factory=factory)
    sandbox, channel = pool.acquire()
    pool.release(sandbox, channel, healthy=True)
    made[0].dead = True  # E2B killed it while idle

    fresh, _ = pool.acquire()

    assert fresh is not sandbox
    assert made[0].kills == 1  # the dead one was retired
    assert len(made) == 2
    assert pool.usage().count == 2
    pool.close()


def test_run_retries_once_on_a_fresh_sandbox_after_transport_death(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dropped runner stream fails the attempt's sandbox and replays on a fresh one."""
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    # First runner dies right after hello (stream ends mid-episode -> transport RuntimeError);
    # the second plays a full episode.
    factory, made = _factory_for(
        [
            [{"type": "hello"}],
            [
                {"type": "hello"},
                {"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {}},
                {"type": "done", "answer": "recovered"},
            ],
        ]
    )
    # hold_open applies to every scripted handle; kill the first stream by marking its
    # handle exhausted after hello (script end without hold -> EOF -> RuntimeError).
    original_factory = factory

    def dying_first_factory() -> FakeSandbox:
        fake = original_factory()
        if len(made) == 1:
            # First stream ends right after hello (no hold): the channel reads EOF mid-episode.
            fake.commands._handle._hold_open = False  # noqa: SLF001
        return fake

    pool = E2BSandboxPool(sandbox_factory=dying_first_factory)
    runtime = _runtime(pool=pool)
    env = _RecordingEnv()

    result = runtime.run("t1", "do it", env)

    assert result.stop_reason is StopReason.SUBMITTED
    assert result.answer == "recovered"
    assert len(made) == 2
    assert made[0].kills == 1  # the dead attempt's sandbox was discarded, not reused
    pool.close()


def test_run_retries_e2b_send_timeout_once_on_a_fresh_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uncertain send retires its sandbox and replays the episode on a fresh one."""
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    script: list[JsonObject] = [
        {"type": "hello"},
        {"type": "llm_request", "req_id": 1, "openai_body": {}},
        {"type": "done", "answer": "recovered"},
    ]
    factory, made = _factory_for([script, script])

    def timeout_first_factory() -> FakeSandbox:
        fake = factory()
        if len(made) == 1:
            # episode_start is send 1; fail the response send and every attempted resend.
            fake.commands.fail_sends_from = 2
        return fake

    pool = E2BSandboxPool(sandbox_factory=timeout_first_factory)
    result = _runtime(pool=pool).run("t1", "do it", _RecordingEnv())

    assert result.stop_reason is StopReason.SUBMITTED
    assert result.answer == "recovered"
    assert len(made) == 2
    assert len(_of_kind(made[0], "llm_response")) == 1
    assert made[0].kills == 1
    pool.close()


def test_run_retries_broken_pipe_once_on_a_fresh_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raw socket-style send failure also retires the uncertain sandbox."""
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    script: list[JsonObject] = [
        {"type": "hello"},
        {"type": "llm_request", "req_id": 1, "openai_body": {}},
        {"type": "done", "answer": "recovered"},
    ]
    factory, made = _factory_for([script, script])

    def broken_first_factory() -> FakeSandbox:
        fake = factory()
        if len(made) == 1:
            fake.commands.fail_sends_from = 2
            fake.commands.send_error = BrokenPipeError("broken pipe")
        return fake

    pool = E2BSandboxPool(sandbox_factory=broken_first_factory)
    result = _runtime(pool=pool).run("t1", "do it", _RecordingEnv())

    assert result.answer == "recovered"
    assert len(made) == 2
    assert made[0].kills == 1
    pool.close()


def test_run_retries_http2_remote_stream_reset_once_on_a_fresh_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E2B's exact HTTP/2 send reset retires the uncertain sandbox and replays once."""
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    script: list[JsonObject] = [
        {"type": "hello"},
        {"type": "llm_request", "req_id": 1, "openai_body": {}},
        {"type": "done", "answer": "recovered"},
    ]
    factory, made = _factory_for([script, script])

    def reset_first_factory() -> FakeSandbox:
        fake = factory()
        if len(made) == 1:
            fake.commands.fail_sends_from = 2
            fake.commands.send_error = RemoteProtocolError(
                "<StreamReset stream_id:13, error_code:1, remote_reset:True>"
            )
        return fake

    pool = E2BSandboxPool(sandbox_factory=reset_first_factory)
    result = _runtime(pool=pool).run("t1", "do it", _RecordingEnv())

    assert result.answer == "recovered"
    assert len(made) == 2
    assert [len(_of_kind(fake, "llm_response")) for fake in made] == [1, 1]
    assert made[0].kills == 1
    pool.close()


def test_arbitrary_httpcore_protocol_error_propagates_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A protocol error without the proven remote-reset shape remains ambiguous."""
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    script: list[JsonObject] = [
        {"type": "hello"},
        {"type": "llm_request", "req_id": 1, "openai_body": {}},
    ]
    factory, made = _factory_for([script, script])

    def protocol_error_factory() -> FakeSandbox:
        fake = factory()
        fake.commands.fail_sends_from = 2
        fake.commands.send_error = RemoteProtocolError("Server disconnected")
        return fake

    pool = E2BSandboxPool(sandbox_factory=protocol_error_factory)
    with pytest.raises(RemoteProtocolError, match="Server disconnected"):
        _runtime(pool=pool).run("t1", "do it", _RecordingEnv())

    assert len(made) == 1
    assert len(_of_kind(made[0], "llm_response")) == 1
    assert made[0].kills == 1
    pool.close()


def test_provider_http2_remote_stream_reset_is_not_a_transport_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same exception from the worker provider is reported to pi, not replayed."""
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    script: list[JsonObject] = [
        {"type": "hello"},
        {"type": "llm_request", "req_id": 1, "openai_body": {}},
        {"type": "done", "answer": "provider error handled"},
    ]
    factory, made = _factory_for([script, script])

    def worker(request: ChatRequest) -> ChatResponse:
        del request
        raise RemoteProtocolError("<StreamReset stream_id:13, error_code:1, remote_reset:True>")

    pool = E2BSandboxPool(sandbox_factory=factory)
    result = _runtime(pool=pool, worker_fn=worker).run("t1", "do it", _RecordingEnv())

    assert result.answer == "provider error handled"
    assert len(made) == 1
    responses = _of_kind(made[0], "llm_response")
    assert len(responses) == 1
    assert "StreamReset" in cast("str", responses[0]["error"])
    assert made[0].kills == 0
    pool.close()


def test_environment_os_error_propagates_without_episode_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool-side OS failure is not mistaken for an E2B transport failure."""
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    script: list[JsonObject] = [
        {"type": "hello"},
        {"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {}},
    ]
    factory, made = _factory_for([script, script])

    class FailingEnv(_RecordingEnv):
        def execute(self, action: Action) -> Observation:
            self.actions.append(action)
            raise OSError("tool filesystem failed")

    pool = E2BSandboxPool(sandbox_factory=factory)
    env = FailingEnv()
    with pytest.raises(OSError, match="tool filesystem failed"):
        _runtime(pool=pool).run("t1", "do it", env)

    assert len(env.actions) == 1
    assert len(made) == 1
    assert made[0].kills == 1
    pool.close()


def test_run_propagates_a_second_e2b_send_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fresh-sandbox recovery is attempted once, then the transport failure escapes."""
    monkeypatch.delenv(E2B_TEMPLATE_ENV, raising=False)
    script: list[JsonObject] = [
        {"type": "hello"},
        {"type": "llm_request", "req_id": 1, "openai_body": {}},
    ]
    factory, made = _factory_for([script, script])

    def timeout_factory() -> FakeSandbox:
        fake = factory()
        fake.commands.fail_sends_from = 2
        return fake

    pool = E2BSandboxPool(sandbox_factory=timeout_factory)
    with pytest.raises(TimeoutException, match="request timed out"):
        _runtime(pool=pool).run("t1", "do it", _RecordingEnv())

    assert len(made) == 2
    assert [len(_of_kind(fake, "llm_response")) for fake in made] == [1, 1]
    assert [fake.kills for fake in made] == [1, 1]
    pool.close()


# --- durable live-session transport ---
_OUTBOX = f"{RUNNER_WORKDIR}/test-outbox"
_OUTBOX_STDERR = f"{_OUTBOX}/stderr.log"


def _frame_path(seq: int) -> str:
    return f"{_OUTBOX}/frames/{seq:020d}.json"


def _store_outbox(fake: FakeSandbox, frames: list[JsonObject]) -> None:
    for seq, frame in enumerate(frames, start=1):
        fake.files.store[_frame_path(seq)] = json.dumps(_envelope(seq, frame))
    fake.files.store[f"{_OUTBOX}/head"] = str(len(frames))


def _durable_channel(
    fake: FakeSandbox,
    handle: _ScriptedHandle,
    *,
    frame_read_grace_s: float = 0.1,
    stdout_silence_grace_s: float = 45.0,
) -> E2BDurableChannel:
    return E2BDurableChannel(
        fake,
        handle,
        outbox_root=_OUTBOX,
        stderr_path=_OUTBOX_STDERR,
        poll_interval_s=0.005,
        stream_death_grace_s=0.005,
        frame_read_grace_s=frame_read_grace_s,
        pid_probe_interval_s=0.005,
        stdout_silence_grace_s=stdout_silence_grace_s,
    )


def test_durable_channel_backfills_a_stdout_gap_before_delivering_the_newer_frame() -> None:
    one: JsonObject = {"type": "hello"}
    two: JsonObject = {"type": "state", "status": "ready"}
    three: JsonObject = {"type": "state", "status": "idle"}
    handle = _ScriptedHandle(
        _stdout_events([_envelope(1, one), _envelope(3, three)]), hold_open=True
    )
    fake = FakeSandbox(handle)
    _store_outbox(fake, [one, two, three])
    channel = _durable_channel(fake, handle)

    assert [channel.recv(timeout=0.5) for _ in range(3)] == [one, two, three]


def test_durable_channel_dedupes_the_same_sequence_from_disk_and_stdout() -> None:
    hello: JsonObject = {"type": "hello"}
    gate = threading.Event()
    handle = _GatedHandle(_stdout_events([_envelope(1, hello), _envelope(1, hello)]), gate)
    fake = FakeSandbox(handle)
    _store_outbox(fake, [hello])
    channel = _durable_channel(fake, handle)

    assert channel.recv(timeout=0.5) == hello  # unary outbox wins the race
    gate.set()  # both duplicate stdout notifications arrive afterward
    with pytest.raises(TimeoutError):
        channel.recv(timeout=0.05)


def test_durable_channel_recovers_after_stdout_drops() -> None:
    hello: JsonObject = {"type": "hello"}
    idle: JsonObject = {"type": "state", "status": "idle"}
    handle = _DisconnectingHandle(_stdout_events([_envelope(1, hello)]))
    fake = FakeSandbox(handle)
    _store_outbox(fake, [hello, idle])
    channel = _durable_channel(fake, handle)

    assert channel.recv(timeout=0.5) == hello
    assert channel.recv(timeout=0.5) == idle
    assert "Server disconnected" in channel.stderr_tail()


def test_durable_channel_resume_preserves_both_sequence_cursors() -> None:
    """A memory-preserving sandbox resume reattaches without replaying either direction."""
    hello: JsonObject = {"type": "hello"}
    idle: JsonObject = {"type": "state", "status": "idle"}
    running: JsonObject = {"type": "state", "status": "running"}
    handle = _ScriptedHandle(
        _stdout_events([_envelope(1, hello), _envelope(2, idle)]),
        hold_open=True,
    )
    resumed = _ScriptedHandle(_stdout_events([_envelope(4, running)]), hold_open=True)
    fake = FakeSandbox(handle, reconnect_handles=[resumed])
    _store_outbox(fake, [hello, idle])
    channel = _durable_channel(fake, handle)

    assert channel.recv(timeout=0.5) == hello
    assert channel.recv(timeout=0.5) == idle
    first: JsonObject = {"type": "user_message", "text": "first"}
    channel.send(first)

    fake.files.store[_frame_path(4)] = json.dumps(_envelope(4, running))
    fake.files.store[f"{_OUTBOX}/head"] = "4"
    previous_reader = channel._reader  # noqa: SLF001
    channel.resume(fake)

    assert fake.commands.connects == [(_PID, 0)]
    previous_reader.join(timeout=0.5)
    assert not previous_reader.is_alive()
    assert channel._stream_dead_at is None  # noqa: SLF001
    assert channel.recv(timeout=0.5) == running
    second: JsonObject = {"type": "user_message", "text": "second"}
    channel.send(second)

    wires = [
        cast("JsonObject", json.loads(base64.b64decode(data.strip())))
        for _pid, data in fake.commands.stdin
    ]
    assert [wire["transport_in_seq"] for wire in wires] == [1, 2]
    assert fake.durable_dispatches == [first, second]


def test_durable_channel_resume_fences_a_concurrent_old_liveness_probe() -> None:
    """A recv racing reconnect cannot poison the resumed channel with the old PID state."""
    hello: JsonObject = {"type": "hello"}
    handle = _DisconnectingHandle(_stdout_events([_envelope(1, hello)]))
    resumed = _ScriptedHandle([], hold_open=True)
    fake = FakeSandbox(handle, reconnect_handles=[resumed])
    _store_outbox(fake, [hello])
    channel = _durable_channel(fake, handle, frame_read_grace_s=0.01)
    assert channel.recv(timeout=0.5) == hello
    deadline = time.monotonic() + 0.5
    while channel._stream_dead_at is None and time.monotonic() < deadline:  # noqa: SLF001
        time.sleep(0.001)
    assert channel._stream_dead_at is not None  # noqa: SLF001

    release_connect = threading.Event()
    fake.commands.connect_gate = release_connect
    fake.commands.running_pids.clear()
    resume_errors: list[Exception] = []
    recv_errors: list[Exception] = []

    def resume() -> None:
        try:
            channel.resume(fake)
        except Exception as error:  # noqa: BLE001 - asserted below
            resume_errors.append(error)

    def recv() -> None:
        try:
            channel.recv(timeout=0.1)
        except Exception as error:  # noqa: BLE001 - asserted below
            recv_errors.append(error)

    resume_thread = threading.Thread(target=resume)
    resume_thread.start()
    assert fake.commands.connect_started.wait(0.5)
    recv_thread = threading.Thread(target=recv)
    recv_thread.start()
    time.sleep(0.04)
    fake.commands.running_pids.add(_PID)
    release_connect.set()
    resume_thread.join(timeout=0.5)
    recv_thread.join(timeout=0.5)

    assert not resume_thread.is_alive()
    assert not recv_thread.is_alive()
    assert resume_errors == []
    assert len(recv_errors) == 1
    assert isinstance(recv_errors[0], TimeoutError)
    assert channel._fatal_error is None  # noqa: SLF001


def test_durable_channel_resume_rejects_a_fatal_state_set_during_connect() -> None:
    """The post-connect guard cannot return a newly attached but poisoned channel."""
    handle = _ScriptedHandle([], hold_open=True)
    resumed = _ScriptedHandle([], hold_open=True)
    fake = FakeSandbox(handle, reconnect_handles=[resumed])
    _store_outbox(fake, [])
    channel = _durable_channel(fake, handle)
    release_connect = threading.Event()
    fake.commands.connect_gate = release_connect
    resume_errors: list[Exception] = []

    def resume() -> None:
        try:
            channel.resume(fake)
        except Exception as error:  # noqa: BLE001 - asserted below
            resume_errors.append(error)

    resume_thread = threading.Thread(target=resume)
    resume_thread.start()
    assert fake.commands.connect_started.wait(0.5)
    with channel._state_lock:  # noqa: SLF001
        channel._mark_fatal_locked("simulated concurrent fatal state")  # noqa: SLF001
    release_connect.set()
    resume_thread.join(timeout=0.5)

    assert not resume_thread.is_alive()
    assert len(resume_errors) == 1
    assert "simulated concurrent fatal state" in str(resume_errors[0])
    assert resumed.disconnects == 1


def test_durable_channel_retries_a_transiently_missing_committed_frame() -> None:
    hello: JsonObject = {"type": "hello"}
    ready: JsonObject = {"type": "state", "status": "ready"}
    handle = _ScriptedHandle(_stdout_events([_envelope(2, ready)]), hold_open=True)
    fake = FakeSandbox(handle)
    transient = _TransientFiles()
    fake.files = transient
    _store_outbox(fake, [hello, ready])
    transient.failures[_frame_path(1)] = 2
    channel = _durable_channel(fake, handle)

    assert channel.recv(timeout=0.5) == hello
    assert channel.recv(timeout=0.5) == ready
    assert transient.failures[_frame_path(1)] == 0
    assert transient.request_timeouts
    head_calls = [call for call in transient.read_calls if call[0].endswith("/head")]
    frame_calls = [call for call in transient.read_calls if "/frames/" in call[0]]
    assert head_calls
    assert all(timeout is not None and timeout <= 0.25 for _, timeout, _ in head_calls)
    assert frame_calls
    assert all(timeout == 5.0 and gzip for _, timeout, gzip in frame_calls)


def test_durable_channel_replays_a_large_context_with_a_compressed_frame_budget() -> None:
    frame: JsonObject = {
        "type": "llm_request",
        "req_id": 1,
        "openai_body": {"messages": [{"role": "user", "content": "x" * 1_000_000}]},
    }
    handle = _ScriptedHandle([], hold_open=True)
    fake = FakeSandbox(handle)
    files = _LargeFrameFiles()
    fake.files = files
    _store_outbox(fake, [frame])
    channel = _durable_channel(fake, handle)

    assert channel.recv(timeout=0.5) == frame
    frame_calls = [call for call in files.read_calls if "/frames/" in call[0]]
    assert frame_calls == [(_frame_path(1), 5.0, True)]


def test_live_session_completes_exactly_once_across_a_durable_stdout_drop() -> None:
    """A dropped notification stream cannot duplicate model/tool work during one real turn."""
    frames: list[JsonObject] = [
        {"type": "hello"},
        {"type": "state", "status": "idle"},
        {"type": "llm_request", "req_id": 1, "openai_body": {"messages": []}},
        {
            "type": "tool_request",
            "req_id": 2,
            "name": "bash",
            "arguments": {"command": "pwd"},
        },
        {
            "type": "tool_request",
            "req_id": 3,
            "name": "submit",
            "arguments": {"answer": "done"},
        },
        {"type": "state", "status": "idle", "reason": "completed", "turns": 1},
    ]
    # stdout carries the opening frames, repeats the LLM request, then disconnects. The tool,
    # submit, and final state must continue from the exact filesystem frames.
    handle = _DisconnectingHandle(
        _stdout_events(
            [
                _envelope(1, frames[0]),
                _envelope(2, frames[1]),
                _envelope(3, frames[2]),
                _envelope(3, frames[2]),
            ]
        )
    )
    fake = FakeSandbox(handle)
    _store_outbox(fake, frames)
    channel = _durable_channel(fake, handle)
    assert channel.recv(timeout=0.5) == {"type": "hello"}

    worker_calls: list[ChatRequest] = []
    tool_calls: list[str] = []
    events: list[SessionEvent] = []

    def worker(request: ChatRequest) -> ChatResponse:
        worker_calls.append(request)
        return _completion("on it")

    def execute(name: str, arguments: JsonObject, emit) -> ToolOutcome:  # noqa: ANN001
        del arguments, emit
        tool_calls.append(name)
        return ToolOutcome(content="/home/user/project\n")

    session = LiveSession(
        channel,
        tools=[TOOL_REGISTRY["bash"], SUBMIT],
        execute_tool=execute,
        on_event=events.append,
        worker_fn=worker,
    )
    session.start()
    events.clear()
    session.send_user_message("finish once")
    for _ in range(20):
        session.pump(timeout=0.05)
        if any(
            event.kind == "state" and event.payload.get("reason") == "completed" for event in events
        ):
            break

    assert session.status == "idle"
    assert len(worker_calls) == 1
    assert tool_calls == ["bash"]
    dispatched_types = [frame["type"] for frame in fake.durable_dispatches]
    assert dispatched_types.count("llm_response") == 1
    assert dispatched_types.count("tool_response") == 2
    assert [event.kind for event in events].count("assistant_message") == 1
    assert [event.kind for event in events].count("tool_call") == 1
    assert [event.kind for event in events].count("submit") == 1


def test_durable_channel_fails_after_a_committed_frame_stays_corrupt() -> None:
    handle = _ScriptedHandle([], hold_open=True)
    fake = FakeSandbox(handle)
    fake.files.store[f"{_OUTBOX}/head"] = "1"
    fake.files.store[_frame_path(1)] = "{not-json"
    channel = _durable_channel(fake, handle, frame_read_grace_s=0.01)

    with pytest.raises(RuntimeError, match="durable outbox frame 1 unavailable"):
        channel.recv(timeout=0.5)


def test_durable_channel_fails_when_stream_and_head_reads_stay_unavailable() -> None:
    """A live PID cannot turn a combined output/filesystem outage into a six-hour spin."""
    handle = _DisconnectingHandle([])
    fake = FakeSandbox(handle)
    channel = _durable_channel(fake, handle, frame_read_grace_s=0.01)

    with pytest.raises(RuntimeError, match="durable outbox head unavailable"):
        channel.recv(timeout=0.5)


def test_durable_channel_rejects_an_unsequenced_semantic_stdout_frame() -> None:
    handle = _ScriptedHandle(_stdout_events([{"type": "llm_request", "req_id": 1}]))
    fake = FakeSandbox(handle)
    fake.files.store[f"{_OUTBOX}/head"] = "0"
    channel = _durable_channel(fake, handle)

    with pytest.raises(RuntimeError, match="unsequenced or malformed semantic frame"):
        channel.recv(timeout=0.5)


def test_durable_channel_fails_when_silent_stream_and_head_reads_stay_unavailable() -> None:
    handle = _ScriptedHandle([], hold_open=True)
    fake = FakeSandbox(handle)
    channel = _durable_channel(
        fake,
        handle,
        frame_read_grace_s=0.01,
        stdout_silence_grace_s=0.01,
    )

    with pytest.raises(RuntimeError, match="durable outbox head unavailable"):
        channel.recv(timeout=0.5)


def test_durable_channel_pid_exit_includes_the_durable_stderr_tail() -> None:
    handle = _DisconnectingHandle([])
    fake = FakeSandbox(handle)
    fake.files.store[f"{_OUTBOX}/head"] = "0"
    fake.files.store[_OUTBOX_STDERR] = "runner booted\nfatal durable detail\n"
    fake.commands.running_pids.clear()
    channel = _durable_channel(fake, handle, frame_read_grace_s=0.01)

    with pytest.raises(RuntimeError, match="fatal durable detail"):
        channel.recv(timeout=0.5)


def test_durable_channel_polls_disk_while_stdout_is_silently_open() -> None:
    hello: JsonObject = {"type": "hello"}
    handle = _ScriptedHandle([], hold_open=True)
    fake = FakeSandbox(handle)
    _store_outbox(fake, [hello])
    channel = _durable_channel(fake, handle)

    assert channel.recv(timeout=0.5) == hello


def test_start_live_runner_durable_handshake_survives_immediate_stdout_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pi_e2b_module.uuid, "uuid4", lambda: SimpleNamespace(hex="fixed-outbox-id"))
    root = f"{RUNNER_WORKDIR}/live-outbox-fixed-outbox-id"
    handle = _DisconnectingHandle([])
    fake = FakeSandbox(handle)
    fake.files.store[f"{root}/head"] = "1"
    fake.files.store[f"{root}/frames/{1:020d}.json"] = json.dumps(_envelope(1, {"type": "hello"}))

    channel = start_live_runner(
        fake, template="wmh-pi-node", durable_outbox=True, hello_timeout=0.5
    )

    assert isinstance(channel, E2BDurableChannel)
    assert fake.commands.background_envs == [{"WMH_LIVE_OUTBOX": root}]
    assert fake.commands.background_cmds == [f"{LIVE_START_CMD} 2>> {root}/stderr.log"]


def test_durable_channel_close_sends_shutdown_and_kills_the_runner_once() -> None:
    handle = _ScriptedHandle([], hold_open=True)
    fake = FakeSandbox(handle)
    fake.files.store[f"{_OUTBOX}/head"] = "0"
    channel = _durable_channel(fake, handle)

    channel.close()
    channel.close()

    assert channel._cleanup_done.wait(1.0)
    assert [f.get("type") for f in _sent_frames(fake)] == ["shutdown"]
    assert fake.commands.stdin_request_timeouts == [0.25]
    assert fake.commands.killed == [_PID]
    assert fake.commands.kill_request_timeouts == [0.25]
    assert handle.disconnects == 1
    assert channel.recv(timeout=0.01) is None


def test_durable_channel_close_does_not_wait_for_an_inflight_frame_read() -> None:
    handle = _ScriptedHandle([], hold_open=True)
    fake = FakeSandbox(handle)
    files = _BlockingFrameFiles()
    fake.files = files
    _store_outbox(fake, [{"type": "state", "status": "idle"}])
    channel = _durable_channel(fake, handle)
    received: list[JsonObject | None] = []

    receiver = threading.Thread(target=lambda: received.append(channel.recv(timeout=1.0)))
    receiver.start()
    assert files.frame_read_started.wait(0.5)

    started = time.monotonic()
    channel.close()
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert channel._cleanup_done.wait(1.0)
    assert fake.commands.killed == [_PID]
    files.release_frame_read.set()
    receiver.join(timeout=1.0)
    assert not receiver.is_alive()
    assert received == [None]


def test_durable_channel_close_discards_a_prequeued_tool_request() -> None:
    handle = _ScriptedHandle([], hold_open=True)
    fake = FakeSandbox(handle)
    fake.files.store[f"{_OUTBOX}/head"] = "0"
    channel = _durable_channel(fake, handle)
    channel._frames.put(
        {
            "type": "tool_request",
            "req_id": 1,
            "name": "bash",
            "arguments": {"command": "touch should-not-run"},
        }
    )

    channel.close()

    assert channel.recv(timeout=0.01) is None
    assert channel.recv(timeout=0.01) is None
    assert channel.recv(timeout=0.01) is None
    assert channel._cleanup_done.wait(1.0)


def test_durable_channel_bounds_normal_stdin_delivery() -> None:
    handle = _ScriptedHandle([], hold_open=True)
    fake = FakeSandbox(handle)
    fake.files.store[f"{_OUTBOX}/head"] = "0"
    channel = _durable_channel(fake, handle)

    channel.send({"type": "ping", "nonce": "n"})

    assert fake.commands.stdin_request_timeouts == [2.0]
    assert fake.durable_dispatches == [{"type": "ping", "nonce": "n"}]


def test_durable_channel_recovers_when_an_accepted_stdin_write_times_out() -> None:
    """A delivered-then-timeout RPC resolves through its durable ack without replaying."""
    handle = _ScriptedHandle([], hold_open=True)
    fake = FakeSandbox(handle)
    fake.files.store[f"{_OUTBOX}/head"] = "0"
    fake.commands.fail_sends_from = 1
    channel = _durable_channel(fake, handle)
    frame: JsonObject = {"type": "user_message", "msg_id": "m1", "text": "improve"}

    channel.send(frame)

    assert len(fake.commands.stdin) == 1
    assert fake.durable_dispatches == [frame]
    with pytest.raises(TimeoutError):
        channel.recv(timeout=0.02)  # the transport ack never leaks into LiveSession


def test_durable_channel_retries_the_same_sequence_after_a_lost_ack() -> None:
    """A duplicate physical write repairs ack loss without duplicate logical dispatch."""
    handle = _ScriptedHandle([], hold_open=True)
    fake = FakeSandbox(handle)
    fake.files.store[f"{_OUTBOX}/head"] = "0"
    fake.drop_durable_acks = 1
    channel = _durable_channel(fake, handle)
    frame: JsonObject = {"type": "tool_response", "req_id": 7, "content": "once"}

    channel.send(frame)

    assert len(fake.commands.stdin) == 2
    wires = [json.loads(base64.b64decode(data.strip())) for _pid, data in fake.commands.stdin]
    assert wires[0] == wires[1] == {"transport_in_seq": 1, "frame": frame}
    assert fake.durable_dispatches == [frame]


def test_durable_channel_retries_the_same_sequence_after_a_pre_delivery_timeout() -> None:
    """A failed physical write retries once without advancing the logical sequence."""
    handle = _ScriptedHandle([], hold_open=True)
    fake = FakeSandbox(handle)
    fake.files.store[f"{_OUTBOX}/head"] = "0"
    fake.commands.fail_before_delivery.add(1)
    channel = _durable_channel(fake, handle)
    frame: JsonObject = {"type": "session_start", "session_id": "s1"}

    channel.send(frame)

    assert len(fake.commands.stdin) == 2
    wires = [json.loads(base64.b64decode(data.strip())) for _pid, data in fake.commands.stdin]
    assert wires[0] == wires[1] == {"transport_in_seq": 1, "frame": frame}
    assert fake.durable_dispatches == [frame]


# --- live-session bootstrap (start_live_runner / session_entry_files) ---
def test_session_entry_files_returns_the_live_runner_source() -> None:
    files = session_entry_files()
    assert "runner_live.ts" in files
    assert files["runner_live.ts"].startswith("/**")
    assert TRANSPORT_KEEPALIVE_TYPE in files["runner_live.ts"]
    assert ".unref()" in files["runner_live.ts"]
    assert "conn.startTransportKeepalive();" in files["runner_live.ts"]
    assert 'frame.conversation_scope === "turn"' in files["runner_live.ts"]
    assert "this.agent.state.messages = [];" in files["runner_live.ts"]


def test_start_live_runner_bootstraps_starts_and_consumes_hello() -> None:
    fake = FakeSandbox(
        _ScriptedHandle(
            _stdout_events(
                [
                    {"type": TRANSPORT_KEEPALIVE_TYPE},
                    {"type": "hello"},
                ]
            ),
            hold_open=True,
        )
    )
    channel = start_live_runner(fake, template=None)
    # runner_live.ts uploaded; node 22 + pi deps installed; workspace ensured; runner started.
    assert f"{RUNNER_WORKDIR}/runner_live.ts" in fake.files.store
    assert fake.commands.calls[0] == NODE_INSTALL_CMD
    assert any("mkdir -p" in c for c in fake.commands.calls)
    assert fake.commands.background_cmds == [LIVE_START_CMD]
    assert fake.timeouts == []  # Platform owns live-session timeout and idle/suspend behavior
    # The hello was consumed; the channel is idle and ready for session frames.
    with pytest.raises(TimeoutError):
        channel.recv(timeout=0.05)


def test_start_live_runner_with_template_skips_installs() -> None:
    fake = FakeSandbox(_ScriptedHandle(_stdout_events([{"type": "hello"}]), hold_open=True))
    start_live_runner(fake, template="wmh-pi-node")
    assert all(c == c for c in fake.commands.calls)  # only the workspace mkdir, no installs
    assert NODE_INSTALL_CMD not in fake.commands.calls
    assert f"{RUNNER_WORKDIR}/package.json" not in fake.files.store
    assert fake.commands.background_cmds == [LIVE_START_CMD]


def test_start_live_runner_without_hello_raises_with_stderr() -> None:
    # Runner starts but never emits hello (stream held open): recv times out -> no-hello error.
    handle = _ScriptedHandle([(None, "boom: cannot start\n", None)], hold_open=True)
    fake = FakeSandbox(handle)
    with pytest.raises(RuntimeError, match="live runner sent no hello"):
        start_live_runner(fake, template="wmh-pi-node", hello_timeout=0.3)


def test_start_live_runner_quotes_the_workspace_path() -> None:
    """A caller-supplied workspace can't inject extra shell commands into the sandbox."""
    fake = FakeSandbox(_ScriptedHandle(_stdout_events([{"type": "hello"}]), hold_open=True))
    evil = "/tmp/x; touch /pwned"  # noqa: S108 - deliberately hostile input for the quoting test
    start_live_runner(fake, template="wmh-pi-node", workspace=evil)
    assert f"mkdir -p {shlex.quote(evil)}" in fake.commands.calls
    # Neutralized: no bare injected command runs.
    assert not any(c.startswith("mkdir -p /tmp/x;") for c in fake.commands.calls)


def test_start_live_runner_without_hello_closes_the_channel() -> None:
    """A failed handshake tears the runner channel down so no node process is orphaned."""
    handle = _ScriptedHandle([(None, "boom\n", None)], hold_open=True)
    fake = FakeSandbox(handle)
    with pytest.raises(RuntimeError, match="live runner sent no hello"):
        start_live_runner(fake, template="wmh-pi-node", hello_timeout=0.3)
    # close() asked the runner to exit (a shutdown frame reached its stdin).
    assert any(f.get("type") == "shutdown" for f in _sent_frames(fake))
