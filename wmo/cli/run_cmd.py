# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Run WMO's built-in local pi harness.

The host answers worker-model requests and executes bash and file tools inside
the explicit working-directory boundary.
"""

from __future__ import annotations

import codecs
import contextlib
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, BinaryIO

import typer
from rich.console import Console
from rich.text import Text

import wmo.common.providers.registry as provider_registry
import wmo.runtime.harness.live_session as live_session
import wmo.runtime.harness.pi_local as pi_local
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
from wmo.runtime.harness.doc import RUNTIME_KIND_ID, HarnessDoc, Surface, SurfaceKind
from wmo.runtime.harness.live_session import SessionEvent, ToolOutcome
from wmo.runtime.harness.pi_vendor import pi_agent_code_surfaces
from wmo.runtime.harness.tools import READ_SKILL, ToolSpec, resolve_tools

_console = Console()

_TOOL_OUTPUT_CAP = 16_000
_BASH_TIMEOUT_S = 300.0
_PIPE_DRAIN_TIMEOUT_S = 0.25
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


def _assemble(doc: HarnessDoc) -> tuple[str, list[ToolSpec], dict[str, str], dict[str, str]]:
    """Derive the LiveSession inputs from a HarnessDoc.

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
    pi code surfaces on and pins ``param:runtime-kind = pi-node`` so a session
    has a runnable agent.
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
        self._operation_lock = threading.Lock()
        self._active_process: subprocess.Popen[bytes] | None = None

    def cancel(self) -> None:
        """Cancel the current command and reject racing tools until the turn reaches idle."""
        with self._operation_lock:
            self._cancelled.set()
            process = self._active_process
        if process is not None and process.poll() is None:
            _kill_process_group(process)

    def reset_cancel(self) -> None:
        """Allow tools for the next turn after the runner reports the idle boundary."""
        with self._operation_lock:
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
                return self._read_file(target)
            if name == "write_file":
                path = str(args.get("path", ""))
                target = self._resolve(path)
                # Cancellation and filesystem mutation share one linearization point. If cancel
                # acquires the lock first, no directory or file from the stale turn is created.
                with self._operation_lock:
                    if self._cancelled.is_set():
                        return ToolOutcome(content="interrupted", is_error=True)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(str(args.get("content", "")), encoding="utf-8")
                return ToolOutcome(content=f"wrote {path}")
        except _JailEscape as error:
            return ToolOutcome(content=f"path {error} escapes the session directory", is_error=True)
        except OSError as error:
            return ToolOutcome(content=f"{name} failed: {error}", is_error=True)
        return ToolOutcome(content=f"tool {name!r} not available", is_error=True)

    def _read_file(self, target: Path) -> ToolOutcome:
        """Read one regular file incrementally without retaining its full contents."""
        descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                return ToolOutcome(
                    content="read_file failed: path is not a regular file", is_error=True
                )
            output = _BoundedTextBuffer(_TOOL_OUTPUT_CAP)
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            while not self._cancelled.is_set():
                chunk = os.read(descriptor, 4096)
                if not chunk:
                    break
                output.append(decoder.decode(chunk))
            if self._cancelled.is_set():
                return ToolOutcome(content="interrupted", is_error=True)
            output.append(decoder.decode(b"", final=True))
            return ToolOutcome(content=output.render(), truncated=output.truncated)
        finally:
            os.close(descriptor)

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
        with self._operation_lock:
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
            with self._operation_lock:
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


class TerminalEventSink:
    """Render the SessionEvent stream to the terminal."""

    def __init__(
        self,
        *,
        on_running: Callable[[bool], None],
    ) -> None:
        """Render to the console; ``on_running`` tracks turn state for keepalive."""
        self._on_running = on_running

    def __call__(self, event: SessionEvent) -> None:
        """Render one event (never raises: a sink must not stop the loop)."""
        with contextlib.suppress(Exception):
            self._render(event)

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
        self.last_message_id: str | None = None
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
                self.last_message_id = self._session.send_user_message(line)
        # The driver owns EOF handling. It must wait until any submitted turn
        # returns to idle before ending the session.
        self.eof.set()


class LocalLiveDriver:
    """Own one local pi process + LiveSession and drive it against the local directory."""

    def __init__(
        self,
        *,
        jail_root: Path,
        doc: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str | None,
    ) -> None:
        """Configure the driver; ``run`` performs boot, loop, and teardown."""
        self._jail = jail_root
        self._doc = doc
        self._provider = provider
        self._instruction = instruction or None
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
                on_event=TerminalEventSink(on_running=self._on_running),
                files=files,
                system_prompt=system,
                skill_bodies=skill_bodies,
                provider=self._provider,
                max_output_tokens=self._doc.max_output_tokens(),
                temperature=self._doc.temperature(),
                cancel_active=self._executor.cancel,
                reset_cancel=self._executor.reset_cancel,
            )
            session.start()
            _console.print(
                "[green]session ready[/green] - type to steer, [bold]:stop[/bold] to interrupt, "
                "[bold]:quit[/bold] to end."
            )
            opening_message_id = None
            if self._instruction:
                opening_message_id = session.send_user_message(self._instruction)
            reader = StdinCommandReader(session)
            reader.start()
            self._loop(session, reader, opening_message_id)
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

    def _loop(
        self,
        session: live_session.LiveSession,
        reader: StdinCommandReader,
        opening_message_id: str | None,
    ) -> None:
        """Pump until closed, treating closed stdin as one-shot after the final turn."""
        end_sent = False
        while not session.closed:
            try:
                session.pump(timeout=0.5)
            except KeyboardInterrupt:
                self._handle_sigint(session)
            if reader.eof.is_set() and not end_sent:
                # EOF is published only after the reader queues its final message. Drain any
                # message that raced with the frame just received, then require the peer's idle
                # acknowledgement for the final message rather than accepting a stale idle frame.
                session.flush_pending_intents()
                final_message_id = reader.last_message_id or opening_message_id
                final_turn_completed = (
                    final_message_id is None
                    or session.last_completed_message_id == final_message_id
                )
                if session.status == "idle" and final_turn_completed:
                    session.end()
                    end_sent = True

    def _handle_sigint(self, session: live_session.LiveSession) -> None:
        """First Ctrl-C interrupts the current turn; a second ends the session."""
        if not session.turn_active:
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
        if self._channel is not None:
            with contextlib.suppress(Exception):
                self._channel.close()
        _console.print(f"[dim]session ended ({reason})[/dim]")


def _local_worker_provider(provider: str | None, model: str | None) -> ToolCallingProvider:
    """Build the local worker provider from environment credentials, and pre-flight it.

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


_DIR_OPT = typer.Option(
    "--dir",
    help="Working directory and file-tool jail for the built-in local pi harness.",
)
_PROVIDER_OPT = typer.Option(
    "--provider",
    help="Worker provider for the built-in local pi run.",
)
_MODEL_OPT = typer.Option(
    "--model",
    help="Worker model for the built-in local pi run.",
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
        directory: Annotated[str | None, _DIR_OPT] = None,
        provider: Annotated[str | None, _PROVIDER_OPT] = None,
        model: Annotated[str | None, _MODEL_OPT] = None,
        task: Annotated[str | None, _TASK_OPT] = None,
        yes: Annotated[bool, _YES_OPT] = False,
    ) -> None:
        """Run the built-in pi harness locally."""
        jail_root = Path(directory or ".").resolve()
        if not jail_root.is_dir():
            raise typer.BadParameter(f"working directory does not exist: {jail_root}")
        _confirm_local_execution(jail_root, yes=yes)
        driver = _build_driver(
            jail_root=jail_root,
            provider=provider,
            model=model,
            task=task,
        )
        driver.run()


def _confirm_local_execution(jail_root: Path, *, yes: bool) -> None:
    """Warn that harness code and bash run with local user permissions."""
    _console.print(
        Text.assemble(
            (
                "The built-in pi harness and its shell commands run on THIS machine",
                "bold yellow",
            ),
            ".\nFile tools stay under ",
            str(jail_root),
            ", and bash starts there, but bash is not OS-sandboxed and can access anything your "
            "user can.",
        )
    )
    if not yes and not typer.confirm("continue?"):
        raise typer.Exit(code=1)


def _build_driver(
    *,
    jail_root: Path,
    provider: str | None,
    model: str | None,
    task: str | None,
) -> LocalLiveDriver:
    """Assemble the local built-in pi driver."""
    return LocalLiveDriver(
        jail_root=jail_root,
        doc=_pi_node_baseline(),
        provider=_local_worker_provider(provider, model),
        instruction=task,
    )
