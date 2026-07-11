"""The pi agent inside E2B sandboxes: a stdio frame channel + the runtime that drives it.

For pi-node harnesses the agent *process* itself must run for real (its multi-file TypeScript
source — context forking, dropping, summarizing — is the thing under search), while the worker LLM
and the tool routing stay host-side. The ENVIRONMENT is whatever `AgentEnvironment` the eval
binds; in `wmh harness create` / `wmh eval` that is the world-model simulation — the sandbox is
purely the compute substrate for the harness process, and its filesystem is never an environment.

`E2BStdioChannel` carries the existing RunnerLink frame protocol over an E2B background command's
stdin/stdout — one base64(JSON) frame per line, because the sandbox command channel is a text
stream — and `E2BPiRuntime` composes it: acquire a sandbox from the runtime's own pool (create +
bootstrap on demand: upload `pi_entry/runner_stdio.ts`, install node 22 + the vendored pi's npm
deps unless a prebaked template supplies them), start the runner, await its `hello`, then delegate
the whole episode to `wmh.harness.runner_link.RunnerLink` — zero duplication of episode logic, and
the creds-stay-host-side invariant RunnerLink was built for holds (only frames enter the sandbox).

Sandboxes are pooled per runtime instance and reused across sequential episodes (bootstrap is
paid once per sandbox, not per rollout); concurrent episodes each acquire their own sandbox, so
rollouts parallelize naturally — no process-wide `_ACTIVE_CHANNEL` singleton, no max_concurrent:1
limit. The runner's stderr never carries frames; it is collected in a bounded deque and surfaced
in every transport error so a crashed node process diagnoses itself.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import queue
import shlex
import threading
import time
from collections import deque
from typing import cast

from wmh.core.types import JsonObject
from wmh.harness.e2b_sandbox import (
    DEFAULT_SANDBOX_TIMEOUT_S,
    E2B_TEMPLATE_ENV,
    CommandHandle,
    SandboxFactory,
    SandboxHandle,
    SandboxUsage,
    create_sandbox,
    default_sandbox_factory,
)
from wmh.harness.environment import AgentEnvironment
from wmh.harness.runner_link import RunnerLink, WorkerConfig, WorkerFn
from wmh.harness.runtime import RunResult
from wmh.harness.tools import ToolSpec

RUNNER_WORKDIR = "/home/user/pi-run"

# The npm packages the vendored pi source imports (verified against
# vendor/pi-agent/package.json [dependencies] and the import statements under vendor/pi-agent/src;
# same list the 65520da E2B backend installed, now version-pinned). Everything else is node:*.
PI_NPM_PACKAGES = (
    "@earendil-works/pi-ai@0.80.3",
    "ignore@7.0.5",
    "typebox@1.1.38",
    "yaml@2.9.0",
)

# The base image ships node 20.9; pi needs >= 22.6 for --experimental-strip-types (65520da
# precedent for the `n`-based upgrade). Installs are slow, hence the generous cap.
NODE_INSTALL_CMD = "npm install -g n && n 22"
INSTALL_TIMEOUT_S = 600.0
START_CMD = f"cd {RUNNER_WORKDIR} && node --experimental-strip-types runner_stdio.ts"
# The live-session runner (interactive multi-turn) vs. the episode runner (one-shot eval).
LIVE_START_CMD = f"cd {RUNNER_WORKDIR} && node --experimental-strip-types runner_live.ts"
_LIVE_RUNNER_FILES = ("runner_live.ts",)
# Default working directory a live session's real tools operate in (the agent's "cwd").
LIVE_WORKSPACE = "/home/user/workspace"
HELLO_TIMEOUT_S = 60.0

# ESM so node treats the runner's .ts files as modules regardless of syntax detection.
_PACKAGE_JSON = '{"name": "pi-run", "private": true, "type": "module"}\n'

_PI_ENTRY_DIR = os.path.join(os.path.dirname(__file__), "pi_entry")
# runner_stdio.ts is the entrypoint; runner_frames.ts rides along so the two runner transports
# stay deployed together (spec §4) even though the stdio runner is self-contained.
_RUNNER_FILES = ("runner_stdio.ts", "runner_frames.ts")

_STDERR_LINES = 50  # bounded diagnostics buffer: enough for a stack trace, never unbounded


class _Eof:
    """Reader-thread sentinel: the runner process's output stream ended."""


_EOF = _Eof()


class E2BStdioChannel:
    """A `runner_link.Channel` over an E2B background command's stdin/stdout.

    send: base64(JSON) + newline into the process's stdin (`commands.send_stdin`). recv: a blocking
    queue fed by a daemon reader thread that iterates the command handle's stream events,
    reassembles partial stdout lines, and decodes each complete line as one frame. stderr is never
    parsed as frames — it goes to a bounded deque surfaced in error messages. Once the process
    exits, recv raises a `RuntimeError` carrying that stderr tail (unless `close()` initiated the
    shutdown, in which case recv reports a clean end-of-channel `None`).
    """

    def __init__(
        self, sandbox: SandboxHandle, handle: CommandHandle, *, stderr_lines: int = _STDERR_LINES
    ) -> None:
        self._sandbox = sandbox
        self._handle = handle
        self._frames: queue.Queue[JsonObject | _Eof] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=stderr_lines)
        self._closed = False
        self._reader = threading.Thread(
            target=self._read_events, name="e2b-stdio-reader", daemon=True
        )
        self._reader.start()

    def send(self, frame: JsonObject) -> None:
        line = base64.b64encode(json.dumps(frame).encode("utf-8")).decode("ascii") + "\n"
        self._sandbox.commands.send_stdin(self._handle.pid, line)

    def recv(self, timeout: float | None = None) -> JsonObject | None:
        """The next frame from the runner; blocks (up to `timeout` seconds when given).

        The optional `timeout` is beyond the `Channel` protocol (which always blocks) — it exists
        for the hello handshake. Timing out raises `TimeoutError`; a dead runner process raises
        `RuntimeError`; both messages include the recent stderr.
        """
        try:
            item = self._frames.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(
                f"no frame from the pi runner within {timeout}s{self._stderr_suffix()}"
            ) from None
        if isinstance(item, _Eof):
            self._frames.put(item)  # keep EOF sticky so every later recv sees it too
            if self._closed:
                return None  # we asked it to shut down; a clean end-of-channel
            raise RuntimeError(f"pi runner process exited mid-episode{self._stderr_suffix()}")
        return item

    def close(self) -> None:
        """Ask the runner to exit (best-effort); marks the stream end as clean for recv."""
        if self._closed:
            return
        self._closed = True
        try:
            self.send({"type": "shutdown"})
        except Exception:  # noqa: BLE001 - the process/sandbox may already be gone; close is best-effort
            pass

    def stderr_tail(self) -> str:
        """The recent runner stderr (diagnostics; never part of the frame stream)."""
        return "\n".join(self._stderr)

    def _stderr_suffix(self) -> str:
        tail = self.stderr_tail()
        return f"; recent runner stderr:\n{tail}" if tail else ""

    def _read_events(self) -> None:
        pending = ""
        try:
            for stdout, stderr, _pty in self._handle:
                if stderr:
                    for line in stderr.splitlines():
                        if line.strip():
                            self._stderr.append(line)
                if not stdout:
                    continue
                pending += stdout
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    self._decode_line(line)
        except Exception as exc:  # noqa: BLE001 - a broken stream becomes EOF + a diagnostic, not a dead thread
            self._stderr.append(f"[channel] output stream failed: {exc}")
        finally:
            self._frames.put(_EOF)

    def _decode_line(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        try:
            frame = json.loads(base64.b64decode(text, validate=True))
        except ValueError:  # binascii.Error and JSONDecodeError are both ValueErrors
            self._stderr.append(f"[stdout] {text}")  # not a frame; keep it as a diagnostic
            return
        if isinstance(frame, dict):
            self._frames.put(cast("JsonObject", frame))
        else:
            self._stderr.append(f"[stdout] {text}")


class E2BSandboxPool:
    """Bootstrapped sandboxes with a running pi runner, reused across episodes and docs.

    The bootstrap (runner files + node 22 + pi's npm deps unless a template prebakes them) is
    doc-independent — a mutated harness's code surfaces travel per-episode in
    `episode_start.files` — so one pool serves a whole search: `create_harness` opens it once and
    every `_score` wave reuses warm sandboxes instead of re-paying installs. Concurrent episodes
    acquire distinct sandboxes; a sandbox whose episode raised is discarded (the runner process
    is in an unknown state — reuse could cross frames between episodes). `close()` kills
    everything; the pool lock guards only the free lists, so parallel bootstraps never serialize
    behind one sandbox's multi-minute npm install.
    """

    def __init__(
        self,
        *,
        template: str | None = None,
        api_key: str | None = None,
        sandbox_factory: SandboxFactory | None = None,
        hello_timeout: float = HELLO_TIMEOUT_S,
    ) -> None:
        self._template = template
        self._factory = sandbox_factory or default_sandbox_factory(
            api_key=api_key, template=template
        )
        self._hello_timeout = hello_timeout
        self._lock = threading.Lock()
        self._idle: list[tuple[SandboxHandle, E2BStdioChannel]] = []
        self._all: list[SandboxHandle] = []
        self._closed = False
        # Usage meter: lifetime seconds accumulate into _retired_seconds when a sandbox dies;
        # live sandboxes are counted from their _started stamp. Keyed by id() — SandboxHandle is
        # a protocol, not hashable-by-contract.
        self._started: dict[int, float] = {}
        self._created = 0
        self._retired_seconds = 0.0

    def acquire(self) -> tuple[SandboxHandle, E2BStdioChannel]:
        """An idle (bootstrapped, hello-verified) sandbox+channel; creates one when none is free.

        Reused sandboxes get their lifetime EXTENDED first (`set_timeout` restarts E2B's
        countdown): sandboxes created at a search's first wave must survive every later wave,
        and a long search otherwise outlives the fixed creation-time cap — the runner stream
        drops mid-episode ("Server disconnected"). A reused sandbox whose extension fails is
        already dead (idle past its cap); it is retired and the next one is tried.
        """
        while True:
            with self._lock:
                if self._closed:
                    raise RuntimeError("E2BSandboxPool is closed")
                if not self._idle:
                    break
                sandbox, channel = self._idle.pop()
            try:
                sandbox.set_timeout(int(DEFAULT_SANDBOX_TIMEOUT_S))
            except Exception:  # noqa: BLE001 - a dead idle sandbox is expected after long gaps
                self._retire(sandbox)
                continue
            return sandbox, channel
        sandbox = create_sandbox(self._factory)
        with self._lock:
            self._created += 1
            self._started[id(sandbox)] = time.monotonic()
        try:
            self._bootstrap(sandbox)
            channel = self._start_runner(sandbox)
        except BaseException:
            self._retire(sandbox)
            raise
        with self._lock:
            if not self._closed:
                self._all.append(sandbox)
                return sandbox, channel
        self._retire(sandbox)  # closed while we were bootstrapping: don't leak the sandbox
        raise RuntimeError("E2BSandboxPool is closed")

    def release(self, sandbox: SandboxHandle, channel: E2BStdioChannel, *, healthy: bool) -> None:
        """Return a sandbox for reuse, or discard it after a failed episode."""
        if healthy:
            with self._lock:
                if not self._closed:
                    self._idle.append((sandbox, channel))
                    return
        with self._lock:
            if sandbox in self._all:
                self._all.remove(sandbox)
        self._retire(sandbox)

    def close(self) -> None:
        """Kill every pooled sandbox; safe to call more than once."""
        with self._lock:
            self._closed = True
            sandboxes, self._all = self._all, []
            self._idle = []
        for sandbox in sandboxes:
            self._retire(sandbox)

    def usage(self) -> SandboxUsage:
        """The pool's spend meter so far: sandbox count and total lifetime seconds."""
        now = time.monotonic()
        with self._lock:
            live = sum(now - started for started in self._started.values())
            return SandboxUsage(count=self._created, seconds=self._retired_seconds + live)

    def _retire(self, sandbox: SandboxHandle) -> None:
        """Finalize the sandbox's lifetime on the meter, then kill it (best-effort)."""
        with self._lock:
            started = self._started.pop(id(sandbox), None)
            if started is not None:
                self._retired_seconds += time.monotonic() - started
        _kill_quietly(sandbox)

    def __enter__(self) -> E2BSandboxPool:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _bootstrap(self, sandbox: SandboxHandle) -> None:
        """Upload the runner files; on template-less sandboxes also install node 22 + pi's deps."""
        for name in _RUNNER_FILES:
            sandbox.files.write(f"{RUNNER_WORKDIR}/{name}", _read_entry(name))
        if self._template or os.environ.get(E2B_TEMPLATE_ENV):
            return  # the template prebakes node 22 + node_modules; only the runner files refresh
        sandbox.files.write(f"{RUNNER_WORKDIR}/package.json", _PACKAGE_JSON)
        sandbox.commands.run(NODE_INSTALL_CMD, timeout=INSTALL_TIMEOUT_S)
        sandbox.commands.run(
            f"cd {RUNNER_WORKDIR} && npm install {' '.join(PI_NPM_PACKAGES)}",
            timeout=INSTALL_TIMEOUT_S,
        )

    def _start_runner(self, sandbox: SandboxHandle) -> E2BStdioChannel:
        # timeout=0 = no command-connection limit (SDK-documented): the runner must outlive every
        # episode on this sandbox; the sandbox's own lifetime is the real bound.
        handle = sandbox.commands.run(START_CMD, background=True, stdin=True, timeout=0)
        # background=True always yields a handle; the union return type is the protocol's.
        channel = E2BStdioChannel(sandbox, cast("CommandHandle", handle))
        self._await_hello(channel)
        return channel

    def _await_hello(self, channel: E2BStdioChannel) -> None:
        """Block until the runner's `hello` frame (unknown frames are skipped, RunnerLink-style)."""
        deadline = time.monotonic() + self._hello_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(_no_hello(self._hello_timeout, channel))
            try:
                frame = channel.recv(timeout=remaining)
            except TimeoutError as exc:
                raise RuntimeError(_no_hello(self._hello_timeout, channel)) from exc
            if frame is None:
                raise RuntimeError(_no_hello(self._hello_timeout, channel))
            if frame.get("type") == "hello":
                return


class E2BPiRuntime:
    """The `Runtime` for pi-node harnesses on the e2b backend: pi runs inside a pooled sandbox.

    `run` accepts ANY `AgentEnvironment` — tool calls are answered host-side (the world-model
    simulation in closed-loop eval), while the harness process executes in an E2B sandbox drawn
    from the pool. Each episode acquires a sandbox, delegates wholly to `RunnerLink` (frame
    broker, worker-LLM answering, tool budget, transcript recording), and returns the sandbox for
    reuse. Pass a shared `pool` to amortize bootstrap across many runtimes (a whole harness
    search); without one, the runtime creates a private pool from `template` and owns its
    lifetime — call `close()` (or use as a context manager) when the eval finishes.
    """

    def __init__(
        self,
        *,
        worker: WorkerConfig,
        files: dict[str, str],
        tools: list[ToolSpec],
        system_prompt: str,
        template: str | None = None,
        api_key: str | None = None,
        pool: E2BSandboxPool | None = None,
        worker_fn: WorkerFn | None = None,
        hello_timeout: float = HELLO_TIMEOUT_S,
    ) -> None:
        self._worker = worker
        self._files = dict(files)
        self._tools = list(tools)
        self._system_prompt = system_prompt
        self._worker_fn = worker_fn  # test seam, exactly like RunnerLink's
        self._owns_pool = pool is None
        self._pool = pool or E2BSandboxPool(
            template=template, api_key=api_key, hello_timeout=hello_timeout
        )

    def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
        try:
            return self._run_episode(task_id, instruction, environment)
        except (RuntimeError, TimeoutError):
            # Transport death (stream drop, sandbox lifetime, dead runner) — the failed
            # attempt's sandbox was already discarded on release. Retry ONCE on a fresh
            # sandbox: the environment session may replay the dead attempt's opening steps,
            # which for the world-model sim beats failing a whole search wave over one
            # dropped connection. pi-level failures come back as RunResults (episode_error),
            # never as exceptions, so this retries infrastructure only.
            return self._run_episode(task_id, instruction, environment)

    def _run_episode(
        self, task_id: str, instruction: str, environment: AgentEnvironment
    ) -> RunResult:
        sandbox, channel = self._pool.acquire()
        healthy = False
        try:
            link = RunnerLink(
                channel,
                tools=self._tools,
                worker=self._worker,
                worker_fn=self._worker_fn,
                files=self._files,
                system_prompt=self._system_prompt,
            )
            result = link.run(task_id, instruction, environment)
            healthy = True
            return result
        finally:
            self._pool.release(sandbox, channel, healthy=healthy)

    def close(self) -> None:
        """Close the private pool; a no-op when the pool is shared (its owner closes it)."""
        if self._owns_pool:
            self._pool.close()

    def __enter__(self) -> E2BPiRuntime:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def session_entry_files() -> dict[str, str]:
    """The in-sandbox runner file(s) a live session uploads, as {filename: content}.

    Public accessor for consumers outside wmh (the platform's live-session driver reads these
    from the installed wmh package and writes them into the sandbox at start).
    """
    return {name: _read_entry(name) for name in _LIVE_RUNNER_FILES}


def start_live_runner(
    sandbox: SandboxHandle,
    *,
    template: str | None = None,
    workspace: str = LIVE_WORKSPACE,
    hello_timeout: float = HELLO_TIMEOUT_S,
) -> E2BStdioChannel:
    """Bootstrap and start `runner_live.ts` on an already-created sandbox; return its channel.

    A live session (unlike a pooled eval episode) is one dedicated sandbox whose lifecycle the
    caller owns — the caller creates it (setting its own timeout / metadata for reaping) and holds
    the handle for keepalive, PTY, and reconciliation. This function does only the runner
    bootstrap: upload the live runner file(s), install node 22 + pi's npm deps unless a template
    prebakes them, ensure the workspace dir, start the long-lived runner over a stdin-writable
    background command, and block until its `hello`. The returned `E2BStdioChannel` is what a
    `wmh.harness.live_session.LiveSession` drives.
    """
    for name in _LIVE_RUNNER_FILES:
        sandbox.files.write(f"{RUNNER_WORKDIR}/{name}", _read_entry(name))
    if not (template or os.environ.get(E2B_TEMPLATE_ENV)):
        sandbox.files.write(f"{RUNNER_WORKDIR}/package.json", _PACKAGE_JSON)
        sandbox.commands.run(NODE_INSTALL_CMD, timeout=INSTALL_TIMEOUT_S)
        sandbox.commands.run(
            f"cd {RUNNER_WORKDIR} && npm install {' '.join(PI_NPM_PACKAGES)}",
            timeout=INSTALL_TIMEOUT_S,
        )
    # `workspace` is a public parameter; quote it so a caller-supplied path can't
    # inject extra shell commands into the live sandbox.
    sandbox.commands.run(f"mkdir -p {shlex.quote(workspace)}", timeout=30)
    handle = sandbox.commands.run(LIVE_START_CMD, background=True, stdin=True, timeout=0)
    channel = E2BStdioChannel(sandbox, cast("CommandHandle", handle))
    try:
        deadline = time.monotonic() + hello_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(_no_hello_live(hello_timeout, channel))
            try:
                frame = channel.recv(timeout=remaining)
            except TimeoutError as exc:
                raise RuntimeError(_no_hello_live(hello_timeout, channel)) from exc
            if frame is None:
                raise RuntimeError(_no_hello_live(hello_timeout, channel))
            if frame.get("type") == "hello":
                return channel
    except Exception:
        # A failed handshake must not orphan the background node runner + its reader
        # thread in the caller-owned sandbox; close the channel before propagating.
        with contextlib.suppress(Exception):
            channel.close()
        raise


def _no_hello_live(timeout: float, channel: E2BStdioChannel) -> str:
    tail = channel.stderr_tail()
    suffix = f"; recent runner stderr:\n{tail}" if tail else ""
    return f"live runner sent no hello within {timeout:g}s ({LIVE_START_CMD!r}){suffix}"


def _kill_quietly(sandbox: SandboxHandle) -> None:
    """Best-effort sandbox teardown; a dead sandbox is already gone."""
    try:
        sandbox.kill()
    except Exception:  # noqa: BLE001 - teardown must never mask the original error
        pass


def _no_hello(timeout: float, channel: E2BStdioChannel) -> str:
    tail = channel.stderr_tail()
    suffix = f"; recent runner stderr:\n{tail}" if tail else ""
    return f"pi runner sent no hello within {timeout:g}s ({START_CMD!r}){suffix}"


def _read_entry(name: str) -> str:
    with open(os.path.join(_PI_ENTRY_DIR, name), encoding="utf-8") as fh:
        return fh.read()
