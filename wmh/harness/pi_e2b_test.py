"""Offline tests for the in-sandbox pi transport: fakes only, no E2B, no node, no provider.

A `_ScriptedHandle` plays the runner process (its stream events are scripted, stdin is recorded)
and a `FakeSandbox` implements the `SandboxHandle` slice, so `E2BStdioChannel` is exercised over
the real reader-thread/framing code path, `E2BSandboxPool` over the real bootstrap/reuse/discard
lifecycle, and `E2BPiRuntime.run` end-to-end — the runner script speaks the same frames
`runner_link_test.py`'s `_FakeChannel` does (hello → llm_request → tool_request → done), and the
environment answering tool calls is a plain host-side `AgentEnvironment` fake: the sandbox is only
where the harness process lives, never where tool calls land.

No TS-side test: the repo has no TypeScript test precedent (no *_test.ts outside the untouchable
vendor tree, no root package.json), so runner_stdio.ts is covered by the frame-contract tests here.
"""

from __future__ import annotations

import base64
import json
import threading
from collections.abc import Callable, Iterator
from typing import cast

import pytest

from wmh.core.types import Action, JsonObject, Observation
from wmh.harness import pi_e2b as pi_e2b_module
from wmh.harness.e2b_sandbox import E2B_TEMPLATE_ENV
from wmh.harness.pi_e2b import (
    NODE_INSTALL_CMD,
    PI_NPM_PACKAGES,
    RUNNER_WORKDIR,
    START_CMD,
    E2BPiRuntime,
    E2BSandboxPool,
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
        self.calls: list[str] = []  # foreground commands, in order (installs, ...)
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
        self.timeouts: list[int] = []
        self.dead = False

    def set_timeout(self, timeout: int) -> None:
        if self.dead:
            raise RuntimeError("sandbox not found")
        self.timeouts.append(timeout)

    def kill(self) -> bool:
        self.kills += 1
        return True


class _RecordingEnv:
    """ANY host-side `AgentEnvironment` (the world-model shape in real evals): records executes."""

    def __init__(self) -> None:
        self.actions: list[Action] = []

    def execute(self, action: Action) -> Observation:
        self.actions.append(action)
        return Observation(content="wm says ok")

    def close(self) -> None:
        pass


def _channel(fake: FakeSandbox, handle: _ScriptedHandle) -> E2BStdioChannel:
    return E2BStdioChannel(fake, handle)


def _tools() -> list[ToolSpec]:
    return [TOOL_REGISTRY["bash"], SUBMIT]


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
    worker_fn: Callable[[JsonObject], JsonObject] | None = None,
) -> E2BPiRuntime:
    return E2BPiRuntime(
        worker=WorkerConfig(),
        files={"src/agent.ts": "// a"},
        tools=_tools(),
        system_prompt="sys",
        template=template,
        pool=pool,
        worker_fn=worker_fn or (lambda body: {"choices": [{"message": {"content": "ok"}}]}),
    )


def _sent_frames(fake: FakeSandbox) -> list[JsonObject]:
    """Decode every frame the host pushed into the runner's stdin."""
    lines = [data for _pid, data in fake.commands.stdin]
    return [cast("JsonObject", json.loads(base64.b64decode(data.strip()))) for data in lines]


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

    worker_calls: list[JsonObject] = []
    completion: JsonObject = {"choices": [{"message": {"content": "use bash"}}]}

    def worker(b: JsonObject) -> JsonObject:
        worker_calls.append(b)
        return completion

    with E2BSandboxPool(sandbox_factory=factory) as pool:
        result = _runtime(pool=pool, worker_fn=worker).run("t1", "do it", env)

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
    tool_names = {t["name"] for t in cast("list[JsonObject]", start["tools"])}
    assert tool_names >= {"bash", "submit"}
    llm = _of_kind(fake, "llm_response")[0]
    assert llm["req_id"] == 1 and llm["completion"] == completion
    assert worker_calls == [body]  # answered host-side, by the injected worker
    tool = _of_kind(fake, "tool_response")[0]
    assert tool["req_id"] == 2 and tool["content"] == "wm says ok" and tool["is_error"] is False


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
    assert fake.timeouts == []  # fresh sandboxes carry their creation-time lifetime
    pool.release(sandbox, channel, healthy=True)
    again, _ = pool.acquire()
    assert again is sandbox
    assert fake.timeouts == [900]  # the reuse extended the countdown
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
