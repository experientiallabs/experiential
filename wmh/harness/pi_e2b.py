"""The pi agent inside E2B sandboxes: a stdio frame channel + the runtime that drives it.

For pi-node harnesses the agent *process* itself must run for real (its multi-file TypeScript
source — context forking, dropping, summarizing — is the thing under search), while the worker LLM
and the tool routing stay host-side. The ENVIRONMENT is whatever `AgentEnvironment` the eval
binds; in `wmh optimize` / `wmh eval` that is the world-model simulation. The sandbox is
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
import math
import os
import queue
import re
import shlex
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, cast, overload

from wmh.core.types import JsonObject
from wmh.harness.e2b_sandbox import (
    DEFAULT_SANDBOX_TIMEOUT_S,
    E2B_TEMPLATE_ENV,
    CommandHandle,
    SandboxCleanupError,
    SandboxFactory,
    SandboxHandle,
    SandboxUsage,
    create_sandbox,
    default_sandbox_factory,
    kill_sandbox,
)
from wmh.harness.environment import AgentEnvironment
from wmh.harness.runner_link import RunnerLink, WorkerFn
from wmh.harness.runtime import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_TURNS,
    RunResult,
    RuntimeCancelled,
    StopReason,
)
from wmh.harness.skills import SkillLibrary
from wmh.harness.tools import ToolSpec
from wmh.providers.base import ToolCallingProvider

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
_IDLE_RECONNECT_DELAYS_S = (0.0, 0.25, 1.0)

# Durable live sessions mirror every runner -> host semantic frame into a sequenced filesystem
# outbox. Stdout remains the low-latency path; these deliberately short unary RPC bounds keep a
# 500ms LiveSession pump (and cancellation) from inheriting the E2B SDK's much longer default
# request timeout when it polls the outbox.
_DURABLE_FILE_REQUEST_TIMEOUT_S = 0.25
_DURABLE_FRAME_REQUEST_TIMEOUT_S = 5.0
_DURABLE_POLL_INTERVAL_S = 0.25
_DURABLE_STDOUT_FAST_WAIT_S = 0.025
_DURABLE_STREAM_DEATH_GRACE_S = 0.5
_DURABLE_FRAME_READ_GRACE_S = 5.0
_DURABLE_PID_PROBE_INTERVAL_S = 5.0
_DURABLE_SEND_REQUEST_TIMEOUT_S = 2.0
_DURABLE_SEND_ACK_TIMEOUT_S = 5.0
_DURABLE_CLOSE_REQUEST_TIMEOUT_S = 0.25
_DURABLE_STDOUT_SILENCE_GRACE_S = 45.0

# A runner-originated heartbeat keeps the background command stream active while the host is
# synchronously waiting on a slow provider call. Pooled evaluation channels also use the same
# heartbeat to renew the sandbox lease; live-session callers own their own idle/suspend lifecycle
# and therefore leave renewal disabled.
TRANSPORT_KEEPALIVE_TYPE = "transport_keepalive"
_SANDBOX_TIMEOUT_REFRESH_S = 300.0
# Renewal must not turn mutated agent code into an unbounded cost leak. A legitimate slow episode
# may cross the base 15-minute lease, but one cell still gets a hard one-hour sandbox lifetime.
MAX_EVAL_EPISODE_LIFETIME_S = 3_600.0
# A score cell must finish much sooner than the E2B lease-renewal safety cap. This wall budget is
# host-enforced by RunnerLink and may be exceeded only by one already-running provider/tool call.
DEFAULT_EVAL_EPISODE_TIMEOUT_S = 300.0
_MAX_RETIRE_WORKERS = 16

_HTTPCORE_REMOTE_STREAM_RESET = re.compile(
    r"<StreamReset stream_id:\d+, error_code:\d+, remote_reset:True>"
)


class _Eof:
    """Reader-thread sentinel: the runner process's output stream ended."""


_EOF = _Eof()


class _E2BChannelSendError(RuntimeError):
    """A durable host frame could not be acknowledged by the E2B runner."""


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
        self,
        sandbox: SandboxHandle,
        handle: CommandHandle,
        *,
        stderr_lines: int = _STDERR_LINES,
        sandbox_timeout_s: int | None = None,
        timeout_refresh_interval_s: float = _SANDBOX_TIMEOUT_REFRESH_S,
        max_episode_lifetime_s: float = MAX_EVAL_EPISODE_LIFETIME_S,
        reconnect_while_idle: bool = False,
    ) -> None:
        self._sandbox = sandbox
        self._handle = handle
        self._pid = handle.pid
        self._frames: queue.Queue[JsonObject | _Eof] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=stderr_lines)
        self._sandbox_timeout_s = sandbox_timeout_s
        self._timeout_refresh_interval_s = timeout_refresh_interval_s
        self._max_episode_lifetime_s = max_episode_lifetime_s
        self._next_timeout_refresh_at = 0.0
        self._episode_renewal_deadline_at: float | None = None
        self._reconnect_while_idle = reconnect_while_idle
        # Guards the definitely-idle proof and the same-PID reconnect. A user message clears the
        # proof while holding this lock *before* stdin delivery, so the reader cannot reconnect
        # across a turn-start race where a semantic runner frame might be lost.
        self._transport_lock = threading.Lock()
        self._session_idle = False
        self._closed = False
        self._reader = threading.Thread(
            target=self._read_events, name="e2b-stdio-reader", daemon=True
        )
        self._reader.start()

    def send(self, frame: JsonObject) -> None:
        kind = frame.get("type")
        if kind == "episode_start" and self._sandbox_timeout_s is not None:
            now = time.monotonic()
            self._episode_renewal_deadline_at = now + self._max_episode_lifetime_s
            # The pool reset the lease immediately before this episode. Wait until the normal
            # refresh cadence instead of bursting one set_timeout call per sandbox at 30 seconds.
            self._next_timeout_refresh_at = now + self._timeout_refresh_interval_s
        line = base64.b64encode(json.dumps(frame).encode("utf-8")).decode("ascii") + "\n"
        with self._transport_lock:
            if kind in {"session_start", "user_message"}:
                self._session_idle = False
            try:
                self._sandbox.commands.send_stdin(self._pid, line)
            except OSError as exc:
                raise _E2BChannelSendError("failed to send a frame to the E2B runner") from exc
            except Exception as exc:  # noqa: BLE001 - classify one optional-SDK transport shape
                if _is_httpcore_remote_stream_reset(exc):
                    raise _E2BChannelSendError(
                        "failed to send a frame to the E2B runner after an HTTP/2 stream reset"
                    ) from exc
                raise

    def recv(self, timeout: float | None = None) -> JsonObject | None:
        """The next frame from the runner; blocks (up to `timeout` seconds when given).

        Timing out raises `TimeoutError`; a dead runner process raises `RuntimeError`; both
        messages include the recent stderr. RunnerLink uses bounded receives for evaluation wall
        budgets and cooperative cancellation; the hello handshake uses the same contract.
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
        with self._transport_lock:
            if self._closed:
                return
            self._closed = True
            self._session_idle = False
            self._episode_renewal_deadline_at = None
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
        handle = self._handle
        while True:
            try:
                for stdout, stderr, _pty in handle:
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
            except Exception as exc:  # noqa: BLE001 - classified by definitely-idle reconnect
                reconnected = self._reconnect_idle_stream()
                if reconnected is not None:
                    # A partially delivered line is not replay-safe. The runner writes one frame
                    # per line; dropping the fragment lets the next complete frame resynchronize.
                    pending = ""
                    handle = reconnected
                    continue
                self._stderr.append(f"[channel] output stream failed: {exc}")
            else:
                # E2B's stream generator may also end without an exception when the HTTP stream
                # disappears before a process-end event. Same-PID reconnect remains safe only at
                # the exact idle boundary; a real process exit simply makes connect fail.
                reconnected = self._reconnect_idle_stream()
                if reconnected is not None:
                    pending = ""
                    handle = reconnected
                    continue
            self._frames.put(_EOF)
            return

    def _reconnect_idle_stream(self) -> CommandHandle | None:
        """Reattach to the same live runner only while it is provably between turns.

        E2B command streams can suffer a transient HTTP/2 disconnect while their process and
        sandbox remain healthy. Reconnecting mid-turn is not safe because an LLM/tool/state frame
        may have been lost. Once ``state:idle`` was decoded, however, the runner emits only filtered
        transport heartbeats until the next host ``user_message``; same-PID reconnect is lossless.
        """
        if not self._reconnect_while_idle:
            return None
        for delay in _IDLE_RECONNECT_DELAYS_S:
            if delay:
                time.sleep(delay)
            # Hold the proof across connect. `send(user_message)` takes this same lock and clears
            # idle before stdin delivery, so it either happens entirely before or after reattach.
            with self._transport_lock:
                if self._closed or not self._session_idle:
                    return None
                try:
                    handle = self._sandbox.commands.connect(self._pid, timeout=0)
                except Exception:  # noqa: BLE001 - the next bounded attempt uses a fresh stream
                    continue
                self._handle = handle
                return handle
        return None

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
            if frame.get("type") == TRANSPORT_KEEPALIVE_TYPE:
                self._refresh_sandbox_timeout()
                return
            if frame.get("type") == "state":
                status = frame.get("status")
                with self._transport_lock:
                    # Only an exact idle acknowledgement is a replay-safety proof. Any new or
                    # malformed state value conservatively disables transparent reconnect.
                    self._session_idle = status == "idle"
            if frame.get("type") in {"done", "episode_error"}:
                self._episode_renewal_deadline_at = None
            self._frames.put(cast("JsonObject", frame))
        else:
            self._stderr.append(f"[stdout] {text}")

    def _refresh_sandbox_timeout(self) -> None:
        """Renew a pooled eval sandbox lease without surfacing the heartbeat as a protocol frame.

        A provider call can keep one episode active longer than E2B's 900-second sandbox timeout.
        The runner remains responsive during that host-side wait and emits transport heartbeats;
        refreshing here prevents E2B from killing an otherwise healthy in-flight episode. A
        transient refresh failure stays diagnostic-only and the next heartbeat retries it.
        """
        timeout = self._sandbox_timeout_s
        deadline = self._episode_renewal_deadline_at
        if timeout is None or deadline is None:
            return
        now = time.monotonic()
        if now < self._next_timeout_refresh_at or now >= deadline:
            return
        # Shorten the final lease so the sandbox still dies at the absolute episode deadline.
        refreshed_timeout = min(timeout, max(1, math.ceil(deadline - now)))
        try:
            self._sandbox.set_timeout(refreshed_timeout)
        except Exception as exc:  # noqa: BLE001 - retry on the next heartbeat; I/O remains authoritative
            self._stderr.append(f"[channel] sandbox timeout refresh failed: {exc}")
            return
        self._next_timeout_refresh_at = now + self._timeout_refresh_interval_s


class E2BDurableChannel:
    """A lossless live-runner channel backed by a sequenced E2B filesystem outbox.

    Host -> runner traffic uses the same unary ``commands.send_stdin`` operation as
    :class:`E2BStdioChannel`, wrapped in a monotonically sequenced envelope. The runner dispatches
    each inbound sequence once and emits a durable acknowledgement, so an RPC that delivers bytes
    and then times out can safely reuse the same sequence instead of replaying the whole agent
    turn. In the other direction, the durable runner publishes each semantic frame as
    ``{transport_seq, frame}`` to an exact per-sequence file, advances ``head``, and only then
    writes the same envelope to stdout. The stdout stream is therefore a fast notification path,
    not the source of truth: one expected-sequence cursor deduplicates both sources, fills gaps
    from exact files, and keeps polling ``head`` even when stdout is merely silent.

    A command-stream EOF or HTTP failure is not a runner failure. The channel continues over unary
    filesystem reads and only classifies process death after a grace period and an explicit
    ``commands.list`` result that omits the runner PID. This is what lets an ordinary
    ``LiveSession`` survive a dropped E2B output stream without changing its frame protocol.
    """

    def __init__(
        self,
        sandbox: SandboxHandle,
        handle: CommandHandle,
        *,
        outbox_root: str,
        stderr_path: str,
        stderr_lines: int = _STDERR_LINES,
        file_request_timeout_s: float = _DURABLE_FILE_REQUEST_TIMEOUT_S,
        frame_request_timeout_s: float = _DURABLE_FRAME_REQUEST_TIMEOUT_S,
        poll_interval_s: float = _DURABLE_POLL_INTERVAL_S,
        stdout_fast_wait_s: float = _DURABLE_STDOUT_FAST_WAIT_S,
        stream_death_grace_s: float = _DURABLE_STREAM_DEATH_GRACE_S,
        frame_read_grace_s: float = _DURABLE_FRAME_READ_GRACE_S,
        pid_probe_interval_s: float = _DURABLE_PID_PROBE_INTERVAL_S,
        send_request_timeout_s: float = _DURABLE_SEND_REQUEST_TIMEOUT_S,
        send_ack_timeout_s: float = _DURABLE_SEND_ACK_TIMEOUT_S,
        close_request_timeout_s: float = _DURABLE_CLOSE_REQUEST_TIMEOUT_S,
        stdout_silence_grace_s: float = _DURABLE_STDOUT_SILENCE_GRACE_S,
    ) -> None:
        self._sandbox = sandbox
        self._handle = handle
        self._pid = handle.pid
        self._outbox_root = outbox_root.rstrip("/")
        self._stderr_path = stderr_path
        self._stderr_lines = stderr_lines
        self._stderr: deque[str] = deque(maxlen=stderr_lines)
        self._durable_stderr = ""
        self._frames: queue.Queue[JsonObject | _Eof] = queue.Queue()
        self._state_lock = threading.Lock()
        self._resume_condition = threading.Condition(self._state_lock)
        self._send_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._ack_condition = threading.Condition()
        self._closed = threading.Event()
        self._cleanup_done = threading.Event()
        self._next_inbound_seq = 0
        self._acked_inbound_seq = 0
        self._last_seq = 0
        self._known_head = 0
        self._pending: dict[int, JsonObject] = {}
        self._missing_since: dict[int, float] = {}
        self._fatal_error: str | None = None
        self._eof_queued = False
        self._stream_dead_at: float | None = None
        self._stream_death_probe_done = False
        self._stream_error: str | None = None
        self._last_stdout_at = time.monotonic()
        self._head_failure_since: float | None = None
        self._last_head_failure_at: float | None = None
        self._next_pid_check_at = time.monotonic() + pid_probe_interval_s
        self._pid_dead_at: float | None = None
        self._last_outbox_error: str | None = None
        self._stream_generation = 0
        self._resuming = False
        self._file_request_timeout_s = file_request_timeout_s
        self._frame_request_timeout_s = frame_request_timeout_s
        self._poll_interval_s = poll_interval_s
        self._stdout_fast_wait_s = stdout_fast_wait_s
        self._stream_death_grace_s = stream_death_grace_s
        self._frame_read_grace_s = frame_read_grace_s
        self._pid_probe_interval_s = pid_probe_interval_s
        self._send_request_timeout_s = send_request_timeout_s
        self._send_ack_timeout_s = send_ack_timeout_s
        self._close_request_timeout_s = close_request_timeout_s
        self._stdout_silence_grace_s = stdout_silence_grace_s
        self._reader = self._new_reader(handle, generation=self._stream_generation)
        self._reader.start()

    def resume(self, sandbox: SandboxHandle, *, timeout: float | None = 0) -> None:
        """Reconnect this channel to its preserved runner after an E2B sandbox resume.

        E2B keeps the runner process and filesystem outbox in a memory-preserving pause, but the
        old command output stream is gone. Reusing this channel preserves both transport sequence
        cursors while replacing only that notification stream. The durable outbox remains the
        source of truth for frames produced while the stream was detached.

        Args:
            sandbox: The reconnected handle for the same memory-preserved sandbox.
            timeout: E2B command-stream timeout. Zero keeps the live stream attached indefinitely.
        """
        with self._send_lock, self._close_lock:
            if self._closed.is_set():
                raise RuntimeError("a closed durable runner channel cannot be resumed")
            with self._resume_condition:
                if self._fatal_error is not None:
                    raise RuntimeError(
                        f"a failed durable runner channel cannot be resumed: {self._fatal_error}"
                    )
                self._resuming = True
                self._stream_generation += 1
                generation = self._stream_generation
                previous_handle = self._handle
            try:
                handle = sandbox.commands.connect(self._pid, timeout=timeout)
            except BaseException:
                with self._resume_condition:
                    self._resuming = False
                    self._resume_condition.notify_all()
                raise
            now = time.monotonic()
            fatal_during_connect: str | None = None
            with self._resume_condition:
                fatal_during_connect = self._fatal_error
                if fatal_during_connect is None:
                    self._sandbox = sandbox
                    self._handle = handle
                    self._stream_dead_at = None
                    self._stream_death_probe_done = False
                    self._stream_error = None
                    self._last_stdout_at = now
                    self._head_failure_since = None
                    self._last_head_failure_at = None
                    self._next_pid_check_at = now + self._pid_probe_interval_s
                    self._pid_dead_at = None
                    self._last_outbox_error = None
                    self._missing_since.clear()
                self._resuming = False
                self._resume_condition.notify_all()
            if fatal_during_connect is not None:
                disconnect = getattr(handle, "disconnect", None)
                if callable(disconnect):
                    with contextlib.suppress(Exception):
                        disconnect()
                raise RuntimeError(
                    f"durable runner channel failed while resuming: {fatal_during_connect}"
                )
            reader = self._new_reader(handle, generation=generation)
            self._reader = reader

            disconnect = getattr(previous_handle, "disconnect", None)
            if callable(disconnect):
                with contextlib.suppress(Exception):
                    disconnect()
            reader.start()

    def send(self, frame: JsonObject) -> None:
        """Deliver one host frame once, retrying the same sequence until its durable ack."""
        with self._send_lock:
            if self._closed.is_set():
                raise _E2BChannelSendError("durable runner channel is closed")
            self._next_inbound_seq += 1
            inbound_seq = self._next_inbound_seq
            line = self._encode_inbound(inbound_seq, frame)
            deadline = time.monotonic() + self._send_ack_timeout_s
            last_error: Exception | None = None
            attempts = 0

            while True:
                if self._inbound_acknowledged(inbound_seq):
                    return
                if self._closed.is_set():
                    raise _E2BChannelSendError(
                        f"durable runner closed before acknowledging inbound frame {inbound_seq}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    error = _E2BChannelSendError(
                        "durable runner did not acknowledge inbound frame "
                        f"{inbound_seq} within {self._send_ack_timeout_s:g}s"
                    )
                    if last_error is not None:
                        raise error from last_error
                    raise error

                # At most two physical writes, always with the exact same sequence. If the first
                # write was accepted but its HTTP response or ack notification was lost, the
                # runner deduplicates the second and republishes the ack without dispatching.
                if attempts < 2:
                    attempts += 1
                    try:
                        self._sandbox.commands.send_stdin(
                            self._pid,
                            line,
                            request_timeout=min(self._send_request_timeout_s, remaining),
                        )
                    except Exception as exc:  # noqa: BLE001 - delivery may still have succeeded
                        last_error = exc

                remaining = deadline - time.monotonic()
                if self._wait_for_inbound_ack(
                    inbound_seq,
                    timeout=min(self._stdout_fast_wait_s, max(0.0, remaining)),
                ):
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    continue
                self._poll_outbox(
                    request_timeout=min(self._file_request_timeout_s, remaining),
                    frame_request_timeout=min(self._frame_request_timeout_s, remaining),
                )
                if self._inbound_acknowledged(inbound_seq):
                    return
                if self._fatal_error is not None:
                    raise _E2BChannelSendError(self._fatal_error)
                remaining = deadline - time.monotonic()
                self._wait_for_inbound_ack(
                    inbound_seq,
                    timeout=min(self._poll_interval_s, max(0.0, remaining)),
                )

    def recv(self, timeout: float | None = None) -> JsonObject | None:
        """Return the next semantic frame in transport-sequence order.

        Queue waits are capped by the outbox poll cadence, including during the hello handshake.
        This covers a silently hung stdout iterator as well as an explicit stream error. Every E2B
        head/liveness read gets a request timeout no larger than the caller's remaining deadline;
        exact committed frame reads get their own bounded budget because they can contain a full
        model context and are gzip-compressed by the E2B transport.
        """
        if self._closed.is_set():
            return None
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if not self._wait_for_resume(deadline):
                raise TimeoutError(
                    f"no frame from the pi runner within {timeout}s"
                    f"{self._stderr_suffix(include_durable=False)}"
                )
            item = self._get_queued_nowait()
            if item is not None:
                return self._unwrap_item(item)

            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(
                    f"no frame from the pi runner within {timeout}s"
                    f"{self._stderr_suffix(include_durable=False)}"
                )

            # Stdout is the common, low-latency path. Give its reader one short scheduling window
            # before issuing an E2B filesystem RPC; after an explicit stream failure, skip the
            # window and fall through to the durable source immediately.
            if self._stream_dead_at is None:
                fast_wait = self._stdout_fast_wait_s
                if remaining is not None:
                    fast_wait = min(fast_wait, remaining)
                if fast_wait > 0:
                    try:
                        item = self._frames.get(timeout=fast_wait)
                    except queue.Empty:
                        pass
                    else:
                        return self._unwrap_item(item)

            request_timeout = self._file_request_timeout_s
            if remaining is not None:
                request_timeout = min(request_timeout, max(0.001, remaining))
            self._poll_outbox(request_timeout=request_timeout)

            item = self._get_queued_nowait()
            if item is not None:
                return self._unwrap_item(item)

            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is None or remaining > 0:
                liveness_timeout = self._file_request_timeout_s
                if remaining is not None:
                    liveness_timeout = min(liveness_timeout, max(0.001, remaining))
                self._classify_runner_liveness(request_timeout=liveness_timeout)

            item = self._get_queued_nowait()
            if item is not None:
                return self._unwrap_item(item)

            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(
                    f"no frame from the pi runner within {timeout}s"
                    f"{self._stderr_suffix(include_durable=False)}"
                )
            wait = (
                self._poll_interval_s
                if remaining is None
                else min(self._poll_interval_s, remaining)
            )
            try:
                item = self._frames.get(timeout=max(0.001, wait))
            except queue.Empty:
                continue
            return self._unwrap_item(item)

    def _wait_for_resume(self, deadline: float | None) -> bool:
        """Block unary reads while a new sandbox attachment is being established."""
        with self._resume_condition:
            while self._resuming and not self._closed.is_set():
                if deadline is None:
                    self._resume_condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._resume_condition.wait(timeout=remaining)
            return not self._resuming

    def close(self) -> None:
        """Logically close immediately and retire the runner in bounded background cleanup."""
        with self._close_lock:
            if self._closed.is_set():
                return
            # Never wait for a filesystem replay that currently owns `_state_lock`: cancellation
            # must be able to retire the process even if a large exact-frame read is in flight.
            self._closed.set()
            if not self._eof_queued:
                self._eof_queued = True
                self._frames.put(_EOF)
            threading.Thread(
                target=self._cleanup_runner,
                name="e2b-durable-cleanup",
                daemon=True,
            ).start()

    def _cleanup_runner(self) -> None:
        """Best-effort remote teardown; isolated because the SDK may add a 5s health probe."""
        # Preserve a graceful shutdown when no logical send is in flight, but never wait behind
        # one: PID termination below is the authoritative bounded cleanup path.
        try:
            if self._send_lock.acquire(blocking=False):
                try:
                    self._next_inbound_seq += 1
                    line = self._encode_inbound(
                        self._next_inbound_seq,
                        {"type": "shutdown"},
                    )
                    try:
                        self._sandbox.commands.send_stdin(
                            self._pid,
                            line,
                            request_timeout=self._close_request_timeout_s,
                        )
                    except Exception:  # noqa: BLE001 - transport may already be dead
                        pass
                finally:
                    self._send_lock.release()
            try:
                self._sandbox.commands.kill(
                    self._pid,
                    request_timeout=self._close_request_timeout_s,
                )
            except Exception:  # noqa: BLE001 - the process may already have exited
                pass
            disconnect = getattr(self._handle, "disconnect", None)
            if callable(disconnect):
                with contextlib.suppress(Exception):
                    disconnect()
            if threading.current_thread() is not self._reader:
                self._reader.join(timeout=self._close_request_timeout_s)
        finally:
            self._cleanup_done.set()

    def stderr_tail(self) -> str:
        """Recent stream diagnostics plus the runner's durable stderr file."""
        self._refresh_durable_stderr(self._file_request_timeout_s)
        lines = [*self._stderr, *self._durable_stderr.splitlines()]
        return "\n".join(lines[-self._stderr_lines :])

    def _stderr_suffix(self, *, include_durable: bool = True) -> str:
        if include_durable:
            tail = self.stderr_tail()
        else:
            tail = "\n".join(self._stderr)
        return f"; recent runner stderr:\n{tail}" if tail else ""

    def _get_queued_nowait(self) -> JsonObject | _Eof | None:
        try:
            return self._frames.get_nowait()
        except queue.Empty:
            return None

    def _unwrap_item(self, item: JsonObject | _Eof) -> JsonObject | None:
        if self._closed.is_set():
            return None  # cancellation discards every semantic frame queued before teardown
        if not isinstance(item, _Eof):
            return item
        self._frames.put(item)  # sticky for every later recv
        if self._closed.is_set():
            return None
        message = self._fatal_error or "pi live runner process exited"
        raise RuntimeError(f"{message}{self._stderr_suffix()}")

    def _new_reader(self, handle: CommandHandle, *, generation: int) -> threading.Thread:
        return threading.Thread(
            target=self._read_events,
            args=(handle, generation),
            name="e2b-durable-reader",
            daemon=True,
        )

    def _read_events(self, handle: CommandHandle, generation: int) -> None:
        pending = ""
        try:
            for stdout, stderr, _pty in handle:
                if self._closed.is_set():
                    return
                with self._state_lock:
                    if generation != self._stream_generation:
                        return
                    if stderr:
                        for stderr_line in stderr.splitlines():
                            if stderr_line.strip():
                                self._stderr.append(stderr_line)
                    if stdout:
                        self._last_stdout_at = time.monotonic()
                if not stdout:
                    continue
                pending += stdout
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    self._decode_stdout_line(line, generation=generation)
        except Exception as exc:  # noqa: BLE001 - stdout is only a fallible notification path
            with self._state_lock:
                if generation == self._stream_generation and not self._closed.is_set():
                    self._stream_error = f"[channel] output stream failed: {exc}"
                    self._stderr.append(self._stream_error)
        else:
            with self._state_lock:
                if generation == self._stream_generation and not self._closed.is_set():
                    self._stream_error = "[channel] output stream ended"
        finally:
            with self._state_lock:
                if generation == self._stream_generation and not self._closed.is_set():
                    self._stream_dead_at = time.monotonic()

    def _decode_stdout_line(self, line: str, *, generation: int) -> None:
        text = line.strip()
        if not text:
            return
        try:
            value = json.loads(base64.b64decode(text, validate=True))
        except ValueError:
            with self._state_lock:
                if generation == self._stream_generation:
                    self._stderr.append(f"[stdout] {text}")
            return
        if isinstance(value, dict) and value.get("type") == TRANSPORT_KEEPALIVE_TYPE:
            return  # deliberately unsequenced and never persisted by the durable runner
        envelope = self._parse_envelope(value)
        if envelope is None:
            if isinstance(value, dict):
                with self._state_lock:
                    if generation != self._stream_generation:
                        return
                    self._mark_fatal_locked(
                        "durable runner emitted an unsequenced or malformed semantic frame"
                    )
                return
            with self._state_lock:
                if generation == self._stream_generation:
                    self._stderr.append(f"[stdout] invalid durable envelope: {text}")
            return
        seq, frame = envelope
        with self._state_lock:
            if generation != self._stream_generation:
                return
            if seq <= self._last_seq:
                return
            self._pending.setdefault(seq, frame)
            # The runner publishes the exact frame + head before stdout, so the notification is
            # proof that disk has committed through this sequence even if `head` is briefly stale.
            self._known_head = max(self._known_head, seq)
            # Never make the notification thread replay an unbounded disk gap while holding the
            # ordering lock. ``recv`` continues one exact file per bounded poll.
            self._drain_committed_locked(
                request_timeout=self._frame_request_timeout_s,
                max_file_reads=1,
            )

    def _poll_outbox(
        self,
        *,
        request_timeout: float,
        frame_request_timeout: float | None = None,
    ) -> None:
        if self._closed.is_set():
            return
        with self._state_lock:
            if self._resuming:
                return
            generation = self._stream_generation
            sandbox = self._sandbox
        # Head is tiny and latency-sensitive. Exact frames use a separate, gzip-enabled budget:
        # an LLM request can carry the agent's entire context and must not be misclassified as
        # unavailable merely because it cannot fit inside this short polling interval.
        unary_timeout = max(0.001, request_timeout)
        try:
            raw_head = sandbox.files.read(
                f"{self._outbox_root}/head", request_timeout=unary_timeout
            )
            head = int(raw_head.strip())
            if head < 0:
                raise ValueError("negative sequence")
        except Exception as exc:  # noqa: BLE001 - startup/momentary unary read misses are normal
            with self._state_lock:
                if generation != self._stream_generation or self._resuming:
                    return
                self._last_outbox_error = f"head read failed: {exc}"
                now = time.monotonic()
                # Treat only uninterrupted failures as one outage. AgentProject does not pump this
                # channel while it is idle between proposal iterations, so a long idle gap must not
                # make the next isolated read failure look ancient.
                if self._last_head_failure_at is None or now - self._last_head_failure_at > max(
                    1.0, self._poll_interval_s * 4
                ):
                    self._head_failure_since = now
                self._last_head_failure_at = now
                head_failure_since = self._head_failure_since or now
                output_unavailable = self._stream_dead_at is not None or (
                    now - self._last_stdout_at >= self._stdout_silence_grace_s
                )
                if output_unavailable and now - head_failure_since >= self._frame_read_grace_s:
                    self._mark_fatal_locked(
                        "durable outbox head unavailable after "
                        f"{self._frame_read_grace_s:g}s: {self._last_outbox_error}"
                    )
            return
        with self._state_lock:
            if generation != self._stream_generation or self._resuming:
                return
            self._head_failure_since = None
            self._last_head_failure_at = None
            self._known_head = max(self._known_head, head)
            self._drain_committed_locked(
                request_timeout=(
                    self._frame_request_timeout_s
                    if frame_request_timeout is None
                    else frame_request_timeout
                ),
                max_file_reads=1,
            )

    def _drain_committed_locked(
        self, *, request_timeout: float, max_file_reads: int | None = None
    ) -> None:
        """Advance the one delivery cursor; caller holds ``_state_lock``."""
        file_reads = 0
        while not self._closed.is_set() and self._fatal_error is None:
            expected = self._last_seq + 1
            frame = self._pending.pop(expected, None)
            if frame is None:
                if expected > self._known_head:
                    return
                if max_file_reads is not None and file_reads >= max_file_reads:
                    return
                file_reads += 1
                envelope = self._read_frame_file(expected, request_timeout=request_timeout)
                if self._closed.is_set():
                    return
                if envelope is None:
                    started = self._missing_since.setdefault(expected, time.monotonic())
                    if time.monotonic() - started >= self._frame_read_grace_s:
                        detail = f": {self._last_outbox_error}" if self._last_outbox_error else ""
                        self._mark_fatal_locked(
                            f"durable outbox frame {expected} unavailable after "
                            f"{self._frame_read_grace_s:g}s{detail}"
                        )
                    return
                _seq, frame = envelope
            self._missing_since.pop(expected, None)
            self._last_seq = expected
            if frame.get("type") != TRANSPORT_KEEPALIVE_TYPE:
                self._deliver_frame_locked(frame)

    def _deliver_frame_locked(self, frame: JsonObject) -> None:
        """Filter transport control frames; caller holds the outbound ordering lock."""
        kind = frame.get("type")
        if kind == "transport_ack":
            inbound_seq = frame.get("transport_in_seq")
            if (
                isinstance(inbound_seq, bool)
                or not isinstance(inbound_seq, int)
                or inbound_seq <= 0
                or inbound_seq > self._next_inbound_seq
            ):
                self._mark_fatal_locked("durable runner emitted an invalid inbound acknowledgement")
                return
            with self._ack_condition:
                self._acked_inbound_seq = max(self._acked_inbound_seq, inbound_seq)
                self._ack_condition.notify_all()
            return
        if kind == "transport_nack":
            expected = frame.get("expected_transport_in_seq")
            received = frame.get("transport_in_seq")
            self._mark_fatal_locked(
                f"durable runner rejected inbound sequence {received!r}; expected {expected!r}"
            )
            return
        self._frames.put(frame)

    @staticmethod
    def _encode_inbound(inbound_seq: int, frame: JsonObject) -> str:
        envelope: JsonObject = {
            "transport_in_seq": inbound_seq,
            "frame": frame,
        }
        return base64.b64encode(json.dumps(envelope).encode("utf-8")).decode("ascii") + "\n"

    def _inbound_acknowledged(self, inbound_seq: int) -> bool:
        with self._ack_condition:
            return self._acked_inbound_seq >= inbound_seq

    def _wait_for_inbound_ack(self, inbound_seq: int, *, timeout: float) -> bool:
        with self._ack_condition:
            if self._acked_inbound_seq >= inbound_seq:
                return True
            if timeout > 0:
                self._ack_condition.wait(timeout=timeout)
            return self._acked_inbound_seq >= inbound_seq

    def _read_frame_file(
        self, seq: int, *, request_timeout: float
    ) -> tuple[int, JsonObject] | None:
        path = f"{self._outbox_root}/frames/{seq:020d}.json"
        try:
            raw = self._sandbox.files.read(
                path,
                request_timeout=request_timeout,
                gzip=True,
            )
            value = json.loads(raw)
            envelope = self._parse_envelope(value)
            if envelope is None or envelope[0] != seq:
                raise ValueError(f"expected transport_seq {seq}")
            return envelope
        except Exception as exc:  # noqa: BLE001 - retried by subsequent bounded recv polls
            self._last_outbox_error = f"{path} read failed: {exc}"
            return None

    def _classify_runner_liveness(self, *, request_timeout: float) -> None:
        with self._state_lock:
            if self._closed.is_set() or self._resuming:
                return
            generation = self._stream_generation
            sandbox = self._sandbox
            dead_at = self._stream_dead_at
            now = time.monotonic()
            stream_failure_is_due = (
                dead_at is not None
                and not self._stream_death_probe_done
                and now - dead_at >= self._stream_death_grace_s
            )
            if not stream_failure_is_due and now < self._next_pid_check_at:
                return
            if stream_failure_is_due:
                self._stream_death_probe_done = True
            self._next_pid_check_at = now + self._pid_probe_interval_s
        try:
            processes = sandbox.commands.list(request_timeout=request_timeout)
        except Exception as exc:  # noqa: BLE001 - a failed probe is not proof of process death
            message = f"[channel] runner liveness probe failed: {exc}"
            with self._state_lock:
                if generation != self._stream_generation or self._resuming:
                    return
                if not self._stderr or self._stderr[-1] != message:
                    self._stderr.append(message)
            return
        with self._state_lock:
            if generation != self._stream_generation or self._resuming:
                return
            if any(process.pid == self._pid for process in processes):
                self._pid_dead_at = None
                return
            # `recv` polls disk before every liveness probe. Keep doing so for a bounded grace after
            # the PID disappears, allowing a final frame and head update to become visible before
            # EOF.
            if self._pid_dead_at is None:
                self._pid_dead_at = now
                return
            if now - self._pid_dead_at < self._frame_read_grace_s:
                return
        self._refresh_durable_stderr(
            request_timeout,
            generation=generation,
            sandbox=sandbox,
        )
        with self._state_lock:
            if generation != self._stream_generation or self._resuming:
                return
            detail = f" after {self._stream_error}" if self._stream_error else ""
            self._mark_fatal_locked(f"pi live runner process exited{detail}")

    def _refresh_durable_stderr(
        self,
        request_timeout: float,
        *,
        generation: int | None = None,
        sandbox: SandboxHandle | None = None,
    ) -> None:
        if generation is None or sandbox is None:
            with self._state_lock:
                if self._resuming:
                    return
                generation = self._stream_generation
                sandbox = self._sandbox
        try:
            stderr = sandbox.files.read(self._stderr_path, request_timeout=request_timeout)
        except Exception:  # noqa: BLE001 - diagnostics must never mask the transport error
            return
        with self._state_lock:
            if generation == self._stream_generation and not self._resuming:
                self._durable_stderr = "\n".join(stderr.splitlines()[-self._stderr_lines :])

    @staticmethod
    def _parse_envelope(value: object) -> tuple[int, JsonObject] | None:
        if not isinstance(value, dict):
            return None
        seq = value.get("transport_seq")
        frame = value.get("frame")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq <= 0:
            return None
        if not isinstance(frame, dict):
            return None
        return seq, cast("JsonObject", frame)

    def _mark_fatal_locked(self, message: str) -> None:
        if self._fatal_error is None:
            self._fatal_error = message
        self._queue_eof_locked()

    def _queue_eof_locked(self) -> None:
        if not self._eof_queued:
            self._eof_queued = True
            self._frames.put(_EOF)


class E2BSandboxPool:
    """Bootstrapped sandboxes with a running pi runner, reused within a proposal batch.

    The bootstrap (runner files + node 22 + pi's npm deps unless a template prebakes them) is
    doc-independent — a mutated harness's code surfaces travel per-episode in
    `episode_start.files` — so one pool serves a whole search. `create_harness` retires the idle
    pool at each iteration boundary before the proposer runs, preventing long
    proposer/evaluation gaps from leaving stale E2B command streams alive; score waves for
    sibling proposals in that iteration
    still reuse warm sandboxes instead of re-paying installs. Concurrent episodes acquire distinct
    sandboxes; a sandbox whose episode raised is discarded (the runner process is in an unknown
    state — reuse could cross frames between episodes). `close()` kills everything; the pool lock
    guards only the free lists, so parallel bootstraps never serialize behind one sandbox's
    multi-minute npm install.
    """

    def __init__(
        self,
        *,
        template: str | None = None,
        api_key: str | None = None,
        metadata: dict[str, str] | None = None,
        sandbox_factory: SandboxFactory | None = None,
        hello_timeout: float = HELLO_TIMEOUT_S,
    ) -> None:
        self._template = template
        self._factory = sandbox_factory or default_sandbox_factory(
            api_key=api_key,
            template=template,
            metadata=metadata,
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
        # An exact lease can be reached concurrently by pool.close() and an in-flight episode's
        # release(). The event makes one caller own each kill attempt while every other caller
        # waits for its proof before deciding whether a retry is still needed.
        self._retiring: dict[int, threading.Event] = {}
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
            if not self._closed:
                # Register before bootstrap so cancellation can kill cold-start sandboxes that
                # are still installing dependencies or waiting for the runner hello.
                self._all.append(sandbox)
                registered = True
            else:
                registered = False
        if not registered:
            self._retire(sandbox)
            raise RuntimeError("E2BSandboxPool is closed")
        try:
            self._bootstrap(sandbox)
            channel = self._start_runner(sandbox)
        except BaseException:
            self._retire(sandbox)
            raise
        with self._lock:
            if not self._closed:
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
        self._retire(sandbox)

    def retire_idle(self) -> int:
        """Retire every sandbox currently idle, preserving any episode still in flight.

        The idle set is detached atomically so a concurrent acquire cannot reclaim a stream that
        this call is about to kill. Teardown happens outside the pool lock and in a bounded worker
        set: a normal E2B evaluation wave can leave dozens of idle sandboxes, and serial network
        teardown would otherwise add material latency before every proposer call. Usage remains
        cumulative because each sandbox still passes through ``_retire`` exactly once.

        Returns the number retired, primarily for diagnostics and tests.
        """
        with self._lock:
            idle, self._idle = self._idle, []
        sandboxes = [sandbox for sandbox, _channel in idle]
        self._retire_many(sandboxes)
        return len(sandboxes)

    def close(self) -> None:
        """Kill every pooled sandbox; safe to call more than once."""
        with self._lock:
            self._closed = True
            sandboxes = list(self._all)
            self._idle = []
        self._retire_many(sandboxes)

    def usage(self) -> SandboxUsage:
        """The pool's spend meter so far: sandbox count and total lifetime seconds."""
        now = time.monotonic()
        with self._lock:
            live = sum(now - started for started in self._started.values())
            return SandboxUsage(count=self._created, seconds=self._retired_seconds + live)

    def _retire(self, sandbox: SandboxHandle) -> None:
        """Kill one lease, finalizing its meter only after teardown is proved."""
        sandbox_id = id(sandbox)
        while True:
            with self._lock:
                if sandbox_id not in self._started:
                    return  # another close/release path proved this exact lease gone
                owner_done = self._retiring.get(sandbox_id)
                if owner_done is None:
                    owner_done = threading.Event()
                    self._retiring[sandbox_id] = owner_done
                    # Never make a lease available again once retirement begins. Keep it in
                    # _all until kill succeeds so a later close() can retry a failed cleanup.
                    self._idle = [item for item in self._idle if item[0] is not sandbox]
                    break
            owner_done.wait()

        try:
            kill_sandbox(sandbox)
            retired_at = time.monotonic()
            with self._lock:
                started = self._started.pop(sandbox_id)
                self._all = [item for item in self._all if item is not sandbox]
                self._retired_seconds += retired_at - started
        finally:
            with self._lock:
                self._retiring.pop(sandbox_id, None)
                owner_done.set()

    def _retire_many(self, sandboxes: list[SandboxHandle]) -> None:
        """Retire every requested lease and report all unproven cleanup as one failure."""
        if not sandboxes:
            return

        def retire(sandbox: SandboxHandle) -> SandboxCleanupError | None:
            try:
                self._retire(sandbox)
            except SandboxCleanupError as error:
                return error
            return None

        if len(sandboxes) == 1:
            failures = [retire(sandboxes[0])]
        else:
            # Cancellation commonly lands during a full-concurrency eval wave. Retiring every
            # active lease in parallel keeps teardown latency bounded by one E2B retry sequence.
            with ThreadPoolExecutor(max_workers=min(_MAX_RETIRE_WORKERS, len(sandboxes))) as pool:
                failures = list(pool.map(retire, sandboxes))
        errors = [error for error in failures if error is not None]
        if errors:
            raise SandboxCleanupError(
                f"failed to prove cleanup for {len(errors)} of {len(sandboxes)} E2B sandboxes",
                resource="evaluator_sandbox_pool",
                sandbox_usage=self.usage(),
            ) from errors[0]

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
        # Bootstrap can consume much of the creation-time lease on a template-less sandbox. Reset
        # it immediately before the long-lived runner starts; subsequent in-flight heartbeats keep
        # extending it while an episode is active.
        timeout = int(DEFAULT_SANDBOX_TIMEOUT_S)
        sandbox.set_timeout(timeout)
        # timeout=0 = no command-connection limit (SDK-documented): the runner must outlive every
        # episode on this sandbox; the sandbox's own lifetime is the real bound.
        handle = sandbox.commands.run(START_CMD, background=True, stdin=True, timeout=0)
        # background=True always yields a handle; the union return type is the protocol's.
        channel = E2BStdioChannel(
            sandbox,
            cast("CommandHandle", handle),
            sandbox_timeout_s=timeout,
        )
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
        provider: ToolCallingProvider,
        files: dict[str, str],
        tools: list[ToolSpec],
        system_prompt: str,
        template: str | None = None,
        api_key: str | None = None,
        pool: E2BSandboxPool | None = None,
        worker_fn: WorkerFn | None = None,
        hello_timeout: float = HELLO_TIMEOUT_S,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.7,
        skills: SkillLibrary | None = None,
        episode_timeout_s: float = DEFAULT_EVAL_EPISODE_TIMEOUT_S,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if episode_timeout_s <= 0:
            raise ValueError("episode_timeout_s must be positive")
        self._provider = provider
        self._files = dict(files)
        self._tools = list(tools)
        self._system_prompt = system_prompt
        self._worker_fn = worker_fn  # test seam, exactly like RunnerLink's
        self._max_turns = max_turns
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._skills = skills if skills is not None else SkillLibrary()
        self._episode_timeout_s = episode_timeout_s
        self._should_cancel = should_cancel
        self._aborted = threading.Event()
        self._owns_pool = pool is None
        self._pool = pool or E2BSandboxPool(
            template=template, api_key=api_key, hello_timeout=hello_timeout
        )

    def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
        if self._cancel_requested():
            raise RuntimeCancelled("runtime episode cancelled")
        try:
            return self._run_episode(task_id, instruction, environment)
        except RuntimeCancelled:
            # RunnerLink owns the episode's authoritative partial worker meter.
            # Preserve it instead of replacing the cancellation at this wrapper.
            raise
        except Exception as exc:
            if self._cancel_requested():
                raise RuntimeCancelled("runtime episode cancelled") from exc
            if not _is_retryable_transport_error(exc):
                raise
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
                provider=self._provider,
                worker_fn=self._worker_fn,
                files=self._files,
                system_prompt=self._system_prompt,
                max_turns=self._max_turns,
                max_output_tokens=self._max_output_tokens,
                temperature=self._temperature,
                skills=self._skills,
                episode_timeout_s=self._episode_timeout_s,
                should_cancel=self._cancel_requested,
            )
            result = link.run(task_id, instruction, environment)
            # A wall-budget stop may leave the runner mid-turn. Never reuse or retry it: the
            # partial result is scoreable, but the sandbox's protocol state is not trustworthy.
            healthy = result.stop_reason is not StopReason.BUDGET
            return result
        finally:
            self._pool.release(sandbox, channel, healthy=healthy)

    def abort(self) -> None:
        """Stop every sibling episode and close the shared pool before its caller joins them."""
        self._aborted.set()
        self._pool.close()

    def _cancel_requested(self) -> bool:
        return self._aborted.is_set() or (self._should_cancel is not None and self._should_cancel())

    def close(self) -> None:
        """Close the private pool; a no-op when the pool is shared (its owner closes it)."""
        if self._owns_pool:
            self._pool.close()

    def __enter__(self) -> E2BPiRuntime:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _is_retryable_transport_error(exc: Exception) -> bool:
    """Whether an episode exception means its E2B transport is no longer trustworthy."""
    if isinstance(exc, RuntimeCancelled):
        return False
    if isinstance(exc, (RuntimeError, TimeoutError)):
        return True
    # Keep the E2B SDK optional at import time. Its TimeoutException is not a built-in
    # TimeoutError, but any instance means the sandbox/channel state is uncertain.
    exc_type = type(exc)
    return exc_type.__module__ == "e2b.exceptions" and exc_type.__name__ == "TimeoutException"


def _is_httpcore_remote_stream_reset(exc: Exception) -> bool:
    """Recognize the exact HTTP/2 reset emitted by E2B's httpcore control-plane call."""
    exc_type = type(exc)
    return (
        exc_type.__module__ == "httpcore"
        and exc_type.__name__ == "RemoteProtocolError"
        and _HTTPCORE_REMOTE_STREAM_RESET.fullmatch(str(exc)) is not None
    )


def session_entry_files() -> dict[str, str]:
    """The in-sandbox runner file(s) a live session uploads, as {filename: content}.

    Public accessor for consumers outside wmh (the platform's live-session driver reads these
    from the installed wmh package and writes them into the sandbox at start).
    """
    return {name: _read_entry(name) for name in _LIVE_RUNNER_FILES}


@overload
def start_live_runner(
    sandbox: SandboxHandle,
    *,
    template: str | None = None,
    workspace: str = LIVE_WORKSPACE,
    hello_timeout: float = HELLO_TIMEOUT_S,
    reconnect_while_idle: bool = False,
    durable_outbox: Literal[False] = False,
) -> E2BStdioChannel: ...


@overload
def start_live_runner(
    sandbox: SandboxHandle,
    *,
    template: str | None = None,
    workspace: str = LIVE_WORKSPACE,
    hello_timeout: float = HELLO_TIMEOUT_S,
    reconnect_while_idle: bool = False,
    durable_outbox: Literal[True],
) -> E2BDurableChannel: ...


def start_live_runner(
    sandbox: SandboxHandle,
    *,
    template: str | None = None,
    workspace: str = LIVE_WORKSPACE,
    hello_timeout: float = HELLO_TIMEOUT_S,
    reconnect_while_idle: bool = False,
    durable_outbox: bool = False,
) -> E2BStdioChannel | E2BDurableChannel:
    """Bootstrap and start `runner_live.ts` on an already-created sandbox; return its channel.

    A live session (unlike a pooled eval episode) is one dedicated sandbox whose lifecycle the
    caller owns — the caller creates it (setting its own timeout / metadata for reaping) and holds
    the handle for keepalive, PTY, and reconciliation. This function does only the runner
    bootstrap: upload the live runner file(s), install node 22 + pi's npm deps unless a template
    prebakes them, ensure the workspace dir, start the long-lived runner over a stdin-writable
    background command, and block until its `hello`. By default this returns the established
    ``E2BStdioChannel`` unchanged. ``durable_outbox=True`` instead gives the same ordinary runner a
    unique sequenced outbox below ``RUNNER_WORKDIR`` and returns an ``E2BDurableChannel``; project
    files and the agent-facing workspace remain separate from transport state.
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
    outbox_root: str | None = None
    stderr_path: str | None = None
    start_cmd = LIVE_START_CMD
    if durable_outbox:
        outbox_root = f"{RUNNER_WORKDIR}/live-outbox-{uuid.uuid4().hex}"
        stderr_path = f"{outbox_root}/stderr.log"
        sandbox.commands.run(
            f"mkdir -p {shlex.quote(workspace)} {shlex.quote(outbox_root + '/frames')}",
            timeout=30,
        )
        start_cmd = f"{LIVE_START_CMD} 2>> {shlex.quote(stderr_path)}"
        handle = sandbox.commands.run(
            start_cmd,
            background=True,
            envs={"WMH_LIVE_OUTBOX": outbox_root},
            stdin=True,
            timeout=0,
        )
        channel: E2BStdioChannel | E2BDurableChannel = E2BDurableChannel(
            sandbox,
            cast("CommandHandle", handle),
            outbox_root=outbox_root,
            stderr_path=stderr_path,
        )
    else:
        sandbox.commands.run(f"mkdir -p {shlex.quote(workspace)}", timeout=30)
        handle = sandbox.commands.run(LIVE_START_CMD, background=True, stdin=True, timeout=0)
        channel = E2BStdioChannel(
            sandbox,
            cast("CommandHandle", handle),
            reconnect_while_idle=reconnect_while_idle,
        )
    try:
        deadline = time.monotonic() + hello_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(_no_hello_live(hello_timeout, channel, start_cmd=start_cmd))
            try:
                frame = channel.recv(timeout=remaining)
            except TimeoutError as exc:
                raise RuntimeError(
                    _no_hello_live(hello_timeout, channel, start_cmd=start_cmd)
                ) from exc
            if frame is None:
                raise RuntimeError(_no_hello_live(hello_timeout, channel, start_cmd=start_cmd))
            if frame.get("type") == "hello":
                return channel
    except Exception:
        # A failed handshake must not orphan the background node runner + its reader
        # thread in the caller-owned sandbox; close the channel before propagating.
        with contextlib.suppress(Exception):
            channel.close()
        raise


def _no_hello_live(
    timeout: float,
    channel: E2BStdioChannel | E2BDurableChannel,
    *,
    start_cmd: str = LIVE_START_CMD,
) -> str:
    tail = channel.stderr_tail()
    suffix = f"; recent runner stderr:\n{tail}" if tail else ""
    return f"live runner sent no hello within {timeout:g}s ({start_cmd!r}){suffix}"


def _no_hello(timeout: float, channel: E2BStdioChannel) -> str:
    tail = channel.stderr_tail()
    suffix = f"; recent runner stderr:\n{tail}" if tail else ""
    return f"pi runner sent no hello within {timeout:g}s ({START_CMD!r}){suffix}"


def _read_entry(name: str) -> str:
    with open(os.path.join(_PI_ENTRY_DIR, name), encoding="utf-8") as fh:
        return fh.read()
