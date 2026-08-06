# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Run the built-in local pi harness or a hosted world model.

A bare run starts WMO's retained pi runtime as a local Node child. The host
answers worker-model requests and executes bash and file tools inside the
explicit working-directory boundary. Logged-in bare runs proxy worker calls
through the platform; logged-out runs use local provider credentials.

A platform target is resolved before execution. Hosted world models use their
interactive session API. Agent targets fail with an actionable message because
the platform no longer exposes the hosted agent-session API that the deleted
command used.
"""

from __future__ import annotations

import codecs
import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, BinaryIO, Protocol

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import wmo.common.providers.registry as provider_registry
import wmo.runtime.harness.live_session as live_session
import wmo.runtime.harness.pi_local as pi_local
import wmo.runtime.platform.client as platform_client
import wmo.runtime.platform.credentials as platform_credentials
from wmo.cli.model_roles import DEFAULT_AZURE_API_VERSION
from wmo.common.config import load_settings
from wmo.common.core.types import JsonObject
from wmo.common.providers.base import (
    PreparableProvider,
    ProviderConfig,
    ProviderKind,
    ToolCallingProvider,
)
from wmo.common.providers.models import model_types_for_provider, resolve_provider_model
from wmo.common.vendor.waterfall import ChatRequest, ChatResponse
from wmo.runtime.harness.doc import RUNTIME_KIND_ID, HarnessDoc, Surface, SurfaceKind
from wmo.runtime.harness.live_session import SessionEvent, ToolOutcome
from wmo.runtime.harness.pi_vendor import pi_agent_code_surfaces
from wmo.runtime.harness.runner_link import provider_context_window
from wmo.runtime.harness.tools import READ_SKILL, resolve_tools
from wmo.simulation.model.play import parse_action

_console = Console()

_TOOL_OUTPUT_CAP = 16_000
_BASH_TIMEOUT_S = 300.0
_PIPE_DRAIN_TIMEOUT_S = 0.25
_TICK_S = 5.0
_DEFAULT_PROVIDER = "bedrock"
_DEFAULT_MODEL = "claude-opus-4-8"


class _JailEscape(RuntimeError):
    """A tool path resolved outside the session's working directory."""


def _capped(content: str, *, is_error: bool = False, truncated: bool = False) -> ToolOutcome:
    """Cap tool output to the head+tail budget with a truncation marker."""
    if len(content) <= _TOOL_OUTPUT_CAP:
        return ToolOutcome(content=content, is_error=is_error, truncated=truncated)
    half = _TOOL_OUTPUT_CAP // 2
    dropped = len(content) - _TOOL_OUTPUT_CAP
    capped = f"{content[:half]}\n... [{dropped} chars truncated] ...\n{content[-half:]}"
    return ToolOutcome(content=capped, is_error=is_error, truncated=True)


def _assemble(doc: HarnessDoc) -> tuple[str, list, dict[str, str], dict[str, str]]:
    """Derive the LiveSession inputs from a HarnessDoc (mirrors the hosted driver).

    Returns the assembled system prompt (prompt + rendered tools + skills index),
    the resolved tool specs, the code surfaces as {path: content} (the agent's own
    code, materialized into the local runner), and skill bodies answered host-side.
    """
    tool_names = doc.tools()
    if doc.skills() and READ_SKILL.name not in tool_names:
        tool_names.append(READ_SKILL.name)
    tool_specs = resolve_tools(tool_names)
    system = doc.assembled_prompt()
    files = {surface.path: surface.content for surface in doc.code_files() if surface.path}
    skill_bodies = {skill.name: skill.body for skill in doc.skills()}
    return system, tool_specs, files, skill_bodies


def _pi_node_baseline() -> HarnessDoc:
    """A pi-node baseline: the default prompt/tools plus the vendored pi agent code.

    ``HarnessDoc.baseline`` is the in-process loop, which the live pi runner
    cannot host (it needs the pi agent's src/agent.ts). This grafts the vendored
    pi code surfaces on and pins ``param:runtime-kind = pi-node`` so a not-logged-in
    session has a runnable agent without fetching a champion.
    """
    base = HarnessDoc.baseline("local-session")
    surfaces = [
        *base.surfaces,
        *pi_agent_code_surfaces(),
        Surface(id=RUNTIME_KIND_ID, kind=SurfaceKind.PARAM, content="pi-node"),
    ]
    return HarnessDoc(name="local-session", surfaces=surfaces)


class LocalToolExecutor:
    """Jail file tools to one directory and start bash there without OS isolation."""

    def __init__(self, jail_root: Path) -> None:
        """Confine every tool path under ``jail_root`` (its resolved real path)."""
        self._jail = jail_root.resolve()
        self._cancelled = threading.Event()
        self._process_lock = threading.Lock()
        self._active_process: subprocess.Popen[bytes] | None = None

    def cancel(self) -> None:
        """Cancel the current command and reject racing tools until the turn reaches idle."""
        self._cancelled.set()
        with self._process_lock:
            process = self._active_process
        if process is not None and process.poll() is None:
            _kill_process_group(process)

    def reset_cancel(self) -> None:
        """Allow tools for the next turn after the runner reports the idle boundary."""
        self._cancelled.clear()

    def _resolve(self, path: str) -> Path:
        """Resolve a tool path under the jail, rejecting any escape."""
        target = (self._jail / path).resolve()
        try:
            target.relative_to(self._jail)
        except ValueError as error:
            raise _JailEscape(path) from error
        return target

    def __call__(
        self, name: str, args: JsonObject, emit: Callable[[str, str], None]
    ) -> ToolOutcome:
        """Execute one tool call locally; a failure is an observation, not a crash."""
        if self._cancelled.is_set():
            return ToolOutcome(content="interrupted", is_error=True)
        try:
            if name == "bash":
                return self._bash(str(args.get("command", "")), emit)
            if name == "read_file":
                target = self._resolve(str(args.get("path", "")))
                return _capped(target.read_text(encoding="utf-8", errors="replace"))
            if name == "write_file":
                path = str(args.get("path", ""))
                target = self._resolve(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(str(args.get("content", "")), encoding="utf-8")
                return ToolOutcome(content=f"wrote {path}")
        except _JailEscape as error:
            return ToolOutcome(content=f"path {error} escapes the session directory", is_error=True)
        except OSError as error:
            return ToolOutcome(content=f"{name} failed: {error}", is_error=True)
        return ToolOutcome(content=f"tool {name!r} not available", is_error=True)

    def _bash(self, command: str, emit: Callable[[str, str], None]) -> ToolOutcome:
        """Run a fresh ``bash -lc`` in the jail root, streaming output to ``emit``."""
        stdout_buffer = _BoundedTextBuffer(_TOOL_OUTPUT_CAP)
        stderr_buffer = _BoundedTextBuffer(_TOOL_OUTPUT_CAP)
        emit_lock = threading.Lock()
        process = subprocess.Popen(  # noqa: S603 - the agent's tool intentionally runs commands
            ["bash", "-lc", command],  # noqa: S607 - bash on PATH is the documented contract
            cwd=self._jail,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        # PIPE guarantees both handles. Keep the guard for type narrowing and defensive failure.
        if process.stdout is None or process.stderr is None:  # pragma: no cover
            process.kill()
            raise RuntimeError("bash process did not expose stdout and stderr")
        with self._process_lock:
            self._active_process = process
        if self._cancelled.is_set():
            _kill_process_group(process)
        reader_stop = threading.Event()
        readers = (
            threading.Thread(
                target=_drain_process_stream,
                args=(
                    process.stdout,
                    "stdout",
                    stdout_buffer,
                    emit,
                    emit_lock,
                    reader_stop,
                ),
                daemon=True,
            ),
            threading.Thread(
                target=_drain_process_stream,
                args=(
                    process.stderr,
                    "stderr",
                    stderr_buffer,
                    emit,
                    emit_lock,
                    reader_stop,
                ),
                daemon=True,
            ),
        )
        for reader in readers:
            reader.start()
        timed_out = False
        try:
            exit_code = process.wait(timeout=_BASH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_group(process)
            exit_code = 124
        finally:
            if process.poll() is None:
                _kill_process_group(process)
            process.wait()
            # A child can fork into the new group while the first signal is being delivered.
            # Re-signal after the leader exits so no just-forked descendant retains the pipes.
            if timed_out or self._cancelled.is_set():
                _kill_process_group(process)
            readers_stopped = _join_process_readers(readers, _PIPE_DRAIN_TIMEOUT_S)
            if not readers_stopped:
                # A background child can outlive bash while inheriting its stdout/stderr. Do not
                # let that keep the tool call open: kill the exact process group, then ask the
                # nonblocking readers to stop even if a detached descendant still owns a pipe.
                _kill_process_group(process)
                reader_stop.set()
                readers_stopped = _join_process_readers(readers, _PIPE_DRAIN_TIMEOUT_S)
            if readers_stopped:
                process.stdout.close()
                process.stderr.close()
            with self._process_lock:
                if self._active_process is process:
                    self._active_process = None

        if self._cancelled.is_set():
            return ToolOutcome(content="interrupted", is_error=True)
        if timed_out:
            note = f"\n[timed out after {_BASH_TIMEOUT_S:g}s]"
            stderr_buffer.append(note)
            _emit_safely(emit, emit_lock, "stderr", note)
        stdout = stdout_buffer.render()
        stderr = stderr_buffer.render()
        body = stdout + stderr
        if exit_code != 0:
            body = f"{body}\n[exit {exit_code}]"
        return _capped(
            body,
            is_error=exit_code != 0,
            truncated=stdout_buffer.truncated or stderr_buffer.truncated,
        )


class _BoundedTextBuffer:
    """Keep only the head and tail of one process stream in bounded memory."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._head_limit = limit // 2
        self._tail_limit = limit - self._head_limit
        self._head = ""
        self._tail = ""
        self._total = 0

    @property
    def truncated(self) -> bool:
        return self._total > self._limit

    def append(self, chunk: str) -> None:
        """Record a chunk without retaining more than ``limit`` characters."""
        if not chunk:
            return
        self._total += len(chunk)
        head_room = self._head_limit - len(self._head)
        if head_room > 0:
            self._head += chunk[:head_room]
            chunk = chunk[head_room:]
        if chunk and self._tail_limit:
            self._tail = (self._tail + chunk)[-self._tail_limit :]

    def render(self) -> str:
        """Return the complete stream or a fixed-size head/tail rendering."""
        if not self.truncated:
            return self._head + self._tail
        dropped = self._total - self._limit
        marker = f"\n... [{dropped} chars truncated] ...\n"
        payload = max(0, self._limit - len(marker))
        head_size = payload // 2
        tail_size = payload - head_size
        return self._head[:head_size] + marker + self._tail[-tail_size:]


def _drain_process_stream(
    stream: BinaryIO,
    stream_name: str,
    output: _BoundedTextBuffer,
    emit: Callable[[str, str], None],
    emit_lock: threading.Lock,
    stop: threading.Event,
) -> None:
    """Drain a byte pipe without blocking forever on a descendant that inherited it."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    os.set_blocking(stream.fileno(), False)
    while not stop.is_set():
        try:
            chunk = os.read(stream.fileno(), 4096)
        except BlockingIOError:
            stop.wait(0.01)
            continue
        except OSError:
            break
        if not chunk:
            break
        decoded = decoder.decode(chunk)
        output.append(decoded)
        _emit_safely(emit, emit_lock, stream_name, decoded)
    tail = decoder.decode(b"", final=True)
    output.append(tail)
    _emit_safely(emit, emit_lock, stream_name, tail)


def _join_process_readers(readers: tuple[threading.Thread, ...], timeout: float) -> bool:
    """Give all pipe readers one shared bounded interval to finish."""
    deadline = time.monotonic() + timeout
    for reader in readers:
        reader.join(timeout=max(0.0, deadline - time.monotonic()))
    return not any(reader.is_alive() for reader in readers)


def _emit_safely(
    emit: Callable[[str, str], None],
    lock: threading.Lock,
    stream: str,
    chunk: str,
) -> None:
    """Serialize the two pipe-reader callbacks and keep draining if a sink fails."""
    if not chunk:
        return
    with lock, contextlib.suppress(Exception):
        emit(stream, chunk)


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    """Kill bash and its descendants so a timed-out pipeline cannot retain the pipes."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Some app sandboxes allow setsid but deny killpg. Enumerate this exact process group so
        # foreground children are still terminated even after their leader exits.
        _kill_process_group_members(process.pid)


def _kill_process_group_members(group_id: int) -> None:
    """Kill one owned process group member-by-member when killpg is unavailable."""
    try:
        members = subprocess.run(  # noqa: S603, S607 - fixed pgrep with a numeric group ID
            ["pgrep", "-g", str(group_id)],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        ).stdout.splitlines() or [str(group_id)]
    except (OSError, subprocess.SubprocessError):
        members = [str(group_id)]
    for raw_pid in members:
        with contextlib.suppress(ValueError):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(int(raw_pid), signal.SIGKILL)


class RunRecorder(Protocol):
    """Recording slice consumed by the local driver and terminal event sink."""

    def record(self, event: SessionEvent) -> None: ...

    def flush(self) -> None: ...

    def finish(self, *, ended_reason: str, error: str | None) -> None: ...


class LocalPiRunRecorder:
    """Finish and close an org-scoped built-in pi run; it has no transcript."""

    def __init__(self, client: platform_client.PlatformClient, org_id: str, run_id: str) -> None:
        self._client = client
        self._org_id = org_id
        self._run_id = run_id

    def record(self, event: SessionEvent) -> None:
        """Ignore transcript events; this row exists only for usage accounting."""
        _ = event

    def flush(self) -> None:
        """There is no transcript buffer for a built-in run."""

    def finish(self, *, ended_reason: str, error: str | None) -> None:
        """Report the terminal state and release the HTTP client."""
        status = "failed" if error is not None else "ended"
        with contextlib.suppress(platform_client.PlatformError):
            self._client.finish_local_pi_run(
                self._org_id,
                self._run_id,
                status=status,
                ended_reason=ended_reason,
                error=error,
            )
        self._client.close()


class TerminalEventSink:
    """Render the SessionEvent stream to the terminal and mirror it to a recorder."""

    def __init__(
        self,
        *,
        recorder: RunRecorder | None,
        on_running: Callable[[bool], None],
    ) -> None:
        """Render to the console; ``on_running`` tracks turn state for keepalive."""
        self._recorder = recorder
        self._on_running = on_running

    def __call__(self, event: SessionEvent) -> None:
        """Render one event and mirror it (never raises: a sink must not stop the loop)."""
        with contextlib.suppress(Exception):
            self._render(event)
        if self._recorder is not None:
            self._recorder.record(event)

    def _render(self, event: SessionEvent) -> None:
        payload = event.payload
        if event.kind == "assistant_message":
            text = str(payload.get("text", ""))
            if text:
                _console.print(Text.assemble("\n", ("agent", "bold cyan"), " ", text))
        elif event.kind == "tool_call":
            call = f"$ {payload.get('name', '')} {payload.get('arguments', '')}"
            _console.print(Text(call, style="dim"))
        elif event.kind == "tool_output":
            _console.print(str(payload.get("text", "")), end="", markup=False, highlight=False)
        elif event.kind == "tool_result":
            if payload.get("is_error"):
                _console.print(Text(str(payload.get("content", "")), style="red"))
        elif event.kind == "submit":
            answer = str(payload.get("answer", ""))
            _console.print(Text.assemble("\n", ("submitted", "bold green"), " ", answer))
        elif event.kind == "state":
            status = str(payload.get("status", ""))
            self._on_running(status == "running")
            _console.print(Text(f"({status})", style="dim"))
        elif event.kind == "error":
            message = str(payload.get("message", ""))
            _console.print(Text(f"error: {message}", style="red"))


class StdinCommandReader(threading.Thread):
    """Feed typed stdin lines as steer/interrupt/end intents to the session."""

    def __init__(self, session: live_session.LiveSession) -> None:
        """Read stdin on a daemon thread; the session's intents are thread-safe."""
        super().__init__(daemon=True)
        self._session = session
        self.eof = threading.Event()

    def run(self) -> None:
        """Map each line to an intent until end-of-input or the session closes."""
        for raw in sys.stdin:
            if self._session.closed:
                return
            line = raw.strip()
            if line in {":quit", ":q", ":exit"}:
                self._session.end()
                return
            if line == ":stop":
                self._session.interrupt()
            elif line:
                self._session.send_user_message(line)
        # The driver owns EOF handling. For a one-shot ``--task`` it must wait
        # until the opening turn returns to idle before ending the session.
        self.eof.set()


class LocalLiveDriver:
    """Own one local pi process + LiveSession and drive it against the local directory."""

    def __init__(
        self,
        *,
        jail_root: Path,
        doc: HarnessDoc,
        provider: ToolCallingProvider | None,
        worker_fn: Callable[[ChatRequest], ChatResponse] | None,
        recorder: RunRecorder | None,
        instruction: str | None,
        context_window: int | None = None,
    ) -> None:
        """Configure the driver; ``run`` performs boot, loop, and teardown."""
        self._jail = jail_root
        self._doc = doc
        self._provider = provider
        self._worker_fn = worker_fn
        self._recorder = recorder
        self._instruction = instruction or None
        self._context_window = context_window
        self._executor = LocalToolExecutor(jail_root)
        self._channel: pi_local.LocalStdioChannel | None = None
        self._interrupts = 0

    def run(self) -> None:
        """Boot the local runner, drive the session, and always tear down."""
        system, tool_specs, files, skill_bodies = _assemble(self._doc)
        _console.print("[dim]starting the built-in pi harness locally...[/dim]")
        session: live_session.LiveSession | None = None
        reason = "user_ended"
        error: str | None = None
        try:
            channel = pi_local.start_local_live_runner()
            self._channel = channel
            session = live_session.LiveSession(
                channel,
                tools=tool_specs,
                execute_tool=self._execute,
                on_event=TerminalEventSink(recorder=self._recorder, on_running=self._on_running),
                files=files,
                system_prompt=system,
                skill_bodies=skill_bodies,
                provider=self._provider,
                worker_fn=self._worker_fn,
                max_output_tokens=self._doc.max_output_tokens(),
                temperature=self._doc.temperature(),
                context_window=self._context_window,
                cancel_active=self._executor.cancel,
                reset_cancel=self._executor.reset_cancel,
            )
            session.start()
            _console.print(
                "[green]session ready[/green] - type to steer, [bold]:stop[/bold] to interrupt, "
                "[bold]:quit[/bold] to end."
            )
            if self._instruction:
                session.send_user_message(self._instruction)
            reader = StdinCommandReader(session)
            reader.start()
            stdin_eof = getattr(reader, "eof", threading.Event())
            self._loop(session, stdin_eof)
            if session.status == "failed":
                error = "local live session runner failed"
                reason = "error"
                _console.print(f"[red]session failed: {error}[/red]")
        except Exception as exc:  # noqa: BLE001 - report any driver failure, then tear down
            error = str(exc)
            reason = "error"
            _console.print(Text(f"session failed: {exc}", style="red"))
        finally:
            self._teardown(session, reason=reason, error=error)
        if error is not None:
            raise typer.Exit(code=1)

    def _execute(
        self, name: str, args: JsonObject, emit: Callable[[str, str], None]
    ) -> ToolOutcome:
        """Run one tool locally (each tool blocks the session pump)."""
        return self._executor(name, args, emit)

    def _loop(self, session: live_session.LiveSession, stdin_eof: threading.Event) -> None:
        """Pump until closed, treating closed stdin as one-shot after ``--task``."""
        last_tick = 0.0
        saw_running = False
        end_sent = False
        while not session.closed:
            try:
                session.pump(timeout=0.5)
            except KeyboardInterrupt:
                self._handle_sigint(session)
            saw_running = saw_running or session.status == "running"
            if (
                stdin_eof.is_set()
                and not end_sent
                and (self._instruction is None or (saw_running and session.status == "idle"))
            ):
                session.end()
                end_sent = True
            now = time.monotonic()
            if now - last_tick >= _TICK_S:
                last_tick = now
                if self._recorder is not None:
                    self._recorder.flush()

    def _handle_sigint(self, session: live_session.LiveSession) -> None:
        """First Ctrl-C interrupts the current turn; a second ends the session."""
        if session.status != "running":
            self._interrupts = 0
            _console.print("\n[yellow]no active turn (:quit to end)[/yellow]")
            return
        self._interrupts += 1
        if self._interrupts == 1:
            _console.print("\n[yellow]interrupting (press Ctrl-C again to quit)[/yellow]")
            session.interrupt()
        else:
            _console.print("\n[yellow]ending session[/yellow]")
            session.end()

    def _on_running(self, running: bool) -> None:
        """Reset Ctrl-C escalation once the interrupted turn reaches a non-running state."""
        if not running:
            self._interrupts = 0

    def _teardown(
        self,
        session: live_session.LiveSession | None,
        *,
        reason: str,
        error: str | None,
    ) -> None:
        if session is not None and not session.closed:
            with contextlib.suppress(Exception):
                session.end()
        if self._recorder is not None:
            self._recorder.finish(ended_reason=reason, error=error)
        if self._channel is not None:
            with contextlib.suppress(Exception):
                self._channel.close()
        _console.print(f"[dim]session ended ({reason})[/dim]")


class RemoteWorldModelDriver:
    """Interactive terminal loop over the platform's world-model session API."""

    def __init__(
        self,
        client: platform_client.PlatformClient,
        target_id: str,
        name: str,
        task: str | None,
    ) -> None:
        """Store the resolved target and opening task."""
        self._client = client
        self._target_id = target_id
        self._name = name
        self._task = task

    def run(self) -> None:
        """Create one hosted session and step it until the user exits."""
        try:
            session = self._client.create_world_model_session(self._target_id, task=self._task)
            _console.print(
                Panel(
                    'Type an action such as [cyan]search {"q": "SFO"}[/cyan], '
                    "or a free-text message. Commands: [cyan]:help[/cyan], [cyan]:quit[/cyan].",
                    title=Text.assemble(("running world model", "bold"), " ", self._name),
                    subtitle=Text(f"task: {self._task}" if self._task else "no task set"),
                    border_style="cyan",
                )
            )
            self._loop(session.id)
        except platform_client.PlatformError as error:
            raise typer.BadParameter(str(error)) from error
        finally:
            self._client.close()

    def _loop(self, session_id: str) -> None:
        """Read actions and render hosted observations."""
        while True:
            try:
                line = _console.input("[bold]agent>[/bold] ").strip()
            except (EOFError, KeyboardInterrupt):
                _console.print("\n[dim]bye[/dim]")
                return
            if not line:
                continue
            if line in {":quit", ":q", ":exit"}:
                _console.print("[dim]bye[/dim]")
                return
            if line == ":detach":
                _console.print(
                    "[yellow]world-model sessions are interactive only; "
                    ":detach is unavailable. Use :quit to leave.[/yellow]"
                )
                continue
            if line in {":help", ":h"}:
                _console.print(
                    'Tool call: [cyan]name {"arg": "value"}[/cyan]. '
                    "Any other text is sent as a message."
                )
                continue
            try:
                action = parse_action(line)
            except ValueError as error:
                _console.print(Text(f"parse error: {error}", style="red"))
                continue
            try:
                with _console.status("[dim]world model thinking...[/dim]", spinner="dots"):
                    observation = self._client.step_world_model_session(session_id, action)
            except platform_client.PlatformError as error:
                _console.print(Text(f"step failed: {error}", style="red"))
                continue
            style = "red" if observation.is_error else "green"
            title = "error" if observation.is_error else "observation"
            _console.print(
                Panel(
                    Text(observation.content)
                    if observation.content
                    else Text("(empty)", style="dim"),
                    title=title,
                    border_style=style,
                )
            )


def _local_worker_provider(provider: str | None, model: str | None) -> ToolCallingProvider:
    """Build the logged-out worker provider from local environment credentials, and pre-flight it.

    The pre-flight is the point. Constructing a provider proves almost nothing (every backend
    builds its SDK client lazily), so a bare `wmo run` with nothing configured used to say
    nothing about the model it picked, download the ~130MB pi Node runtime, reach "session
    ready", and only THEN fail mid-session on a Bedrock configuration the user never chose. The
    chosen provider/model is now named up front and `PreparableProvider.prepare` runs before the
    harness boots, the same free, request-free check `wmo optimize route sweep` does over its
    roster (`wmo.common.providers.pool.prepare_pool_provider`). Backends keep their documented
    residual gaps: bedrock's AWS credentials and tinker's reachability are not locally knowable
    and stay first-call failures.

    Raises:
        typer.BadParameter: The provider name is unknown, the model cannot do tool calling, or
            the backend cannot be prepared. Every message names `wmo providers set`.
    """
    configured = load_settings().models.resolve("worker")
    provider_name = provider or (
        configured.provider if configured is not None else _DEFAULT_PROVIDER
    )
    try:
        kind = ProviderKind(provider_name)
    except ValueError:
        kinds = ", ".join(k.value for k in ProviderKind)
        msg = f"unknown provider {provider_name!r}; choose one of: {kinds}"
        raise typer.BadParameter(msg) from None
    if model is not None:
        model_name = model
    elif configured is not None and configured.provider == kind.value:
        model_name = configured.model
    elif provider is None:
        model_name = _DEFAULT_MODEL
    else:
        catalog = model_types_for_provider(kind)
        if not catalog:
            raise typer.BadParameter(
                f"provider {kind.value!r} has no default model; pass --model <model>, or run "
                f"`wmo providers set --provider {kind.value} --model <model>` to configure the "
                "worker role"
            )
        model_name = catalog[0]
    spec = resolve_provider_model(kind, model_name)
    use_configured_knobs = configured is not None and configured.provider == kind.value
    configured_spec = (
        resolve_provider_model(kind, configured.model) if use_configured_knobs else None
    )
    same_configured_model = (
        configured_spec is not None and configured_spec.model_id == spec.model_id
    )
    model_type = spec.model_type
    if (
        use_configured_knobs
        and configured is not None
        and configured.model_type is not None
        and (model is None or same_configured_model)
    ):
        # A tinker:// runtime ID does not encode the base model that selects its renderer,
        # tokenizer, and served context tier. Preserve that explicit identity unless this call
        # actually swapped to a different model.
        model_type = configured.model_type
    deployment = configured.deployment if use_configured_knobs else None
    api_version = configured.api_version if use_configured_knobs else None
    if kind is ProviderKind.AZURE_OPENAI:
        model_changed = (
            model is not None
            and configured_spec is not None
            and configured_spec.model_id != spec.model_id
        )
        if (
            model_changed
            and configured is not None
            and deployment is not None
            and deployment not in (spec.model_type, spec.model_id)
        ):
            raise typer.BadParameter(
                f"the configured azure worker serves {configured.model} from deployment "
                f"{deployment!r}, and on Azure the deployment name is what is actually invoked, "
                f"so --model {spec.model_type} needs the deployment that serves it. Run "
                f"`wmo providers set --provider azure --model {spec.model_type} "
                f"--deployment <deployment>` to point the worker role at it."
            )
        deployment = deployment or spec.model_type
        api_version = api_version or DEFAULT_AZURE_API_VERSION
    config = ProviderConfig(
        kind=kind,
        model_type=model_type,
        model=spec.model_id,
        region=configured.region if use_configured_knobs else None,
        endpoint=configured.endpoint if use_configured_knobs else None,
        deployment=deployment,
        api_version=api_version,
        reasoning_effort=configured.reasoning_effort if use_configured_knobs else None,
    )
    if (
        use_configured_knobs
        and configured is not None
        and configured.chat_max_tokens_field is not None
    ):
        config = config.model_copy(
            update={"chat_max_tokens_field": configured.chat_max_tokens_field}
        )
    built = provider_registry.get_provider(config)
    if not isinstance(built, ToolCallingProvider):
        msg = (
            f"provider {kind.value}/{spec.model_id} does not support structured tool calling; "
            "pick a tool-calling model with "
            f"`wmo providers set --provider {kind.value} --model <model>`"
        )
        raise typer.BadParameter(msg)
    source = "settings models.worker" if configured is not None else "built-in default"
    if provider is not None or model is not None:
        source = "--provider/--model"
    _console.print(Text(f"worker: {kind.value}/{spec.model_id} ({source})", style="dim"))
    if isinstance(built, PreparableProvider):
        try:
            built.prepare()
        except Exception as exc:  # noqa: BLE001 - every backend raises its own SDK's type here
            raise typer.BadParameter(
                f"worker provider {kind.value}/{spec.model_id} ({source}) cannot be used: {exc}\n"
                f"Fix that, or choose another worker with "
                f"`wmo providers set --provider <provider> --model <model>`"
            ) from exc
    return built


def _platform_worker_context_window(run: platform_client.LocalPiRunInfo) -> int | None:
    """Resolve the context guard for the platform-selected worker without using its credentials.

    A current platform may return the deployment-owned value directly. For older deployments,
    Tinker catalog identities can carry their served tier as the final numeric suffix; otherwise
    use the same provider capability probe as local pi execution. That probe is best-effort and
    degrades to the runner fallback when the local machine cannot inspect a platform-owned backend.
    """
    if run.context_window is not None:
        return run.context_window if run.context_window >= 1024 else None
    try:
        kind = ProviderKind(run.worker_provider)
    except ValueError:
        return None
    if kind is ProviderKind.TINKER:
        raw_tier = run.worker_model.rsplit(":", 1)[-1]
        with contextlib.suppress(ValueError):
            declared = int(raw_tier)
            if declared >= 1024:
                return declared
    spec = resolve_provider_model(kind, run.worker_model)
    try:
        worker = provider_registry.get_provider(
            ProviderConfig(
                kind=kind,
                model_type=spec.model_type,
                model=spec.model_id,
            )
        )
    except Exception:  # noqa: BLE001 - capability discovery must not block a proxied run
        return None
    return provider_context_window(worker)


_TARGET_ARG = typer.Argument(
    help="Platform run-target id. Omit it to run the built-in pi harness locally."
)
_DIR_OPT = typer.Option(
    "--dir",
    help="Working directory and file-tool jail for the built-in local pi harness.",
)
_PROVIDER_OPT = typer.Option(
    "--provider",
    help="Worker provider for a logged-out built-in local pi run.",
)
_MODEL_OPT = typer.Option(
    "--model",
    help="Worker model for a logged-out built-in local pi run.",
)
_TASK_OPT = typer.Option(
    "--task",
    "--instruction",
    help="Opening task for either execution kind.",
)
_YES_OPT = typer.Option(
    "--yes",
    help="Skip the local-execution consent prompt.",
)


def register(app: typer.Typer) -> None:
    """Register the root-level wmo run command."""

    @app.command("run")
    def run(
        target: Annotated[str | None, _TARGET_ARG] = None,
        directory: Annotated[str | None, _DIR_OPT] = None,
        provider: Annotated[str | None, _PROVIDER_OPT] = None,
        model: Annotated[str | None, _MODEL_OPT] = None,
        task: Annotated[str | None, _TASK_OPT] = None,
        yes: Annotated[bool, _YES_OPT] = False,
    ) -> None:
        """Run a hosted world model by id, or the built-in pi harness locally."""
        if target is not None and directory is not None:
            raise typer.BadParameter("--dir is only supported for a bare wmo run")

        jail_root = Path(directory or ".").resolve() if target is None else None
        if jail_root is not None and not jail_root.is_dir():
            raise typer.BadParameter(f"working directory does not exist: {jail_root}")

        confirm_local: Callable[[], None] | None = None
        if jail_root is not None:
            local_root = jail_root

            def confirm_execution() -> None:
                _confirm_local_execution(local_root, yes=yes)

            confirm_local = confirm_execution

        driver = _build_driver(
            target=target,
            jail_root=jail_root,
            provider=provider,
            model=model,
            task=task,
            confirm_local=confirm_local,
        )
        driver.run()


def _confirm_local_execution(jail_root: Path, *, yes: bool) -> None:
    """Warn that harness code and bash run with local user permissions."""
    _console.print(
        "[bold yellow]The built-in pi harness and its shell commands run on THIS machine"
        "[/bold yellow].\n"
        f"File tools stay under {jail_root}, and bash starts there, but bash is not "
        "OS-sandboxed and can access anything your user can."
    )
    if not yes and not typer.confirm("continue?"):
        raise typer.Exit(code=1)


def _build_driver(
    *,
    target: str | None,
    jail_root: Path | None,
    provider: str | None,
    model: str | None,
    task: str | None,
    confirm_local: Callable[[], None] | None = None,
) -> LocalLiveDriver | RemoteWorldModelDriver:
    """Resolve the execution kind once and assemble its driver."""
    credentials = platform_credentials.load_credentials()
    logged_in = credentials.is_complete()

    if target is None:
        if jail_root is None:
            raise typer.BadParameter("a working directory is required for the built-in pi harness")
        if not logged_in:
            if confirm_local is not None:
                confirm_local()
            _console.print(
                "[dim]not logged in: running the built-in baseline agent with local "
                "credentials[/dim]"
            )
            return LocalLiveDriver(
                jail_root=jail_root,
                doc=_pi_node_baseline(),
                provider=_local_worker_provider(provider, model),
                worker_fn=None,
                recorder=None,
                instruction=task,
            )
        if provider is not None or model is not None:
            raise typer.BadParameter(
                "logged-in runs use platform credentials; omit --provider/--model, "
                "or run wmo logout to use local credentials"
            )
        if confirm_local is not None:
            confirm_local()
        client = platform_client.PlatformClient(str(credentials.api_url), str(credentials.token))
        try:
            org_id = _default_org(client, credentials.default_org)
            run = client.create_local_pi_run(org_id)
        except typer.BadParameter:
            client.close()
            raise
        except platform_client.PlatformError as error:
            client.close()
            raise typer.BadParameter(str(error)) from error
        recorder = LocalPiRunRecorder(client, org_id, run.id)

        def built_in_worker(request: ChatRequest) -> ChatResponse:
            return client.complete_local_pi_worker(org_id, run.id, request)

        return LocalLiveDriver(
            jail_root=jail_root,
            doc=_pi_node_baseline(),
            provider=None,
            worker_fn=built_in_worker,
            recorder=recorder,
            instruction=task,
            context_window=_platform_worker_context_window(run),
        )

    if not logged_in:
        raise typer.BadParameter(
            "run wmo login to use a platform id, or omit the id to run the built-in pi harness"
        )
    if provider is not None or model is not None:
        raise typer.BadParameter(
            "platform target runs use platform credentials; --provider/--model are not accepted"
        )

    client = platform_client.PlatformClient(str(credentials.api_url), str(credentials.token))
    try:
        resolved = client.resolve_run_target(target)
        if resolved.kind == "world_model":
            return RemoteWorldModelDriver(client, resolved.id, resolved.name, task)
        client.close()
        raise typer.BadParameter(
            "hosted agent sessions are unavailable because the platform no longer exposes "
            "their session API; omit the id to run the built-in pi harness locally"
        )
    except platform_client.PlatformError as error:
        client.close()
        raise typer.BadParameter(str(error)) from error


def _default_org(client: platform_client.PlatformClient, configured: str | None) -> str:
    """Resolve the login's organization, auto-picking only an unambiguous sole org."""
    if configured is not None:
        return configured
    identity = client.whoami()
    if len(identity.orgs) == 1:
        return identity.orgs[0].id
    raise typer.BadParameter(
        "no default organization selected; run `wmo login` again and choose an organization"
    )
