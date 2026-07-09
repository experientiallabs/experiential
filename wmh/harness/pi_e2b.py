"""The pi agent inside the E2B sandbox: a stdio frame channel + the runtime that drives it.

For pi-node harnesses the agent process itself must run in the rollout's sandbox (its multi-file
TypeScript source is the thing under search), while the worker LLM and the tool routing stay
host-side. `E2BStdioChannel` carries the existing RunnerLink frame protocol over an E2B background
command's stdin/stdout — one base64(JSON) frame per line, because the sandbox command channel is a
text stream — and `E2BPiRuntime` composes it: bootstrap the sandbox once (upload
`pi_entry/runner_stdio.ts`, install node 22 + the vendored pi's npm deps unless a prebaked
template supplies them), start the runner, await its `hello`, then delegate the whole episode to
`wmh.harness.runner_link.RunnerLink` — zero duplication of episode logic, and the
creds-stay-host-side invariant RunnerLink was built for holds (only frames enter the sandbox).

One channel per sandbox, one sandbox per rollout (`E2BEnvironment`): no process-wide
`_ACTIVE_CHANNEL` singleton, no max_concurrent:1 limit — rollouts parallelize naturally. The
runner's stderr never carries frames; it is collected in a bounded deque and surfaced in every
transport error so a crashed node process diagnoses itself.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import threading
import time
import weakref
from collections import deque
from typing import cast

from wmh.core.types import JsonObject
from wmh.harness.e2b_env import E2B_TEMPLATE_ENV, CommandHandle, E2BEnvironment, SandboxHandle
from wmh.harness.environment import AgentEnvironment
from wmh.harness.runner_link import RunnerLink, WorkerConfig, WorkerFn
from wmh.harness.runtime import RunResult
from wmh.harness.tools import ToolSpec

# Where the runner lives inside every sandbox. A prebaked template (WMH_E2B_TEMPLATE) must provide
# node >= 22.6 plus this directory containing node_modules and a package.json with type:"module";
# without a template, `_bootstrap` builds the same layout on the base image.
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


class E2BPiRuntime:
    """The `Runtime` for pi-node harnesses on the e2b backend: pi runs inside the rollout sandbox.

    `run` requires an `E2BEnvironment` (the runner shares its sandbox, so env tool calls and the
    agent process see the same filesystem), bootstraps that sandbox exactly once (idempotent per
    sandbox: runner files uploaded; node 22 + pi's npm deps installed unless a template prebakes
    them), starts `runner_stdio.ts` as a background command, awaits its `hello`, then delegates
    the episode to `RunnerLink` — the frame broker, worker-LLM answering, tool budget, and
    transcript recording all stay in that one implementation.
    """

    def __init__(
        self,
        *,
        worker: WorkerConfig,
        files: dict[str, str],
        tools: list[ToolSpec],
        system_prompt: str,
        template: str | None,
        api_key: str | None = None,
        worker_fn: WorkerFn | None = None,
        hello_timeout: float = HELLO_TIMEOUT_S,
    ) -> None:
        self._worker = worker
        self._files = dict(files)
        self._tools = list(tools)
        self._system_prompt = system_prompt
        self._template = template
        # The sandbox is created (and paid for) by E2BEnvironment; the key is accepted to mirror
        # the doc-runtime wiring surface but nothing here opens sandboxes today.
        self._api_key = api_key
        self._worker_fn = worker_fn  # test seam, exactly like RunnerLink's
        self._hello_timeout = hello_timeout
        self._lock = threading.Lock()
        # One live channel per sandbox: episodes on the same sandbox reuse the runner process, and
        # entries vanish with their sandbox objects (a rollout's env owns the sandbox lifetime).
        self._channels: weakref.WeakKeyDictionary[SandboxHandle, E2BStdioChannel] = (
            weakref.WeakKeyDictionary()
        )

    def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
        if not isinstance(environment, E2BEnvironment):
            raise TypeError(
                "E2BPiRuntime needs an E2BEnvironment (the pi agent runs inside its sandbox); "
                f"got {type(environment).__name__}"
            )
        channel = self._channel_for(environment.sandbox)
        link = RunnerLink(
            channel,
            tools=self._tools,
            worker=self._worker,
            worker_fn=self._worker_fn,
            files=self._files,
            system_prompt=self._system_prompt,
        )
        return link.run(task_id, instruction, environment)

    def _channel_for(self, sandbox: SandboxHandle) -> E2BStdioChannel:
        """The (bootstrapped, hello-verified) channel for this sandbox, created on first use.

        The lock guards only the map: parallel rollouts each own a distinct sandbox, so bootstraps
        run concurrently instead of serializing multi-minute npm installs behind one lock.
        """
        with self._lock:
            existing = self._channels.get(sandbox)
        if existing is not None:
            return existing
        self._bootstrap(sandbox)
        channel = self._start_runner(sandbox)
        with self._lock:
            self._channels[sandbox] = channel
        return channel

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


def _no_hello(timeout: float, channel: E2BStdioChannel) -> str:
    tail = channel.stderr_tail()
    suffix = f"; recent runner stderr:\n{tail}" if tail else ""
    return f"pi runner sent no hello within {timeout:g}s ({START_CMD!r}){suffix}"


def _read_entry(name: str) -> str:
    with open(os.path.join(_PI_ENTRY_DIR, name), encoding="utf-8") as fh:
        return fh.read()
