# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Run a platform target or the built-in pi harness from one CLI command.

Agent IDs run the champion pi harness in the platform's E2B sandbox. By
default no local files are sent. ``-u PATH`` opts into uploading a bounded
snapshot, live-syncing changes, and reconciling the final sandbox workspace
into that directory. Bare runs still launch the built-in vendored pi harness
as a local Node child.

The execution mode is chosen automatically (see :func:`register`):

* logged in + agent id: the platform owns E2B, provider credentials, metering,
  and the transcript; the CLI owns only workspace transport and terminal I/O.
* logged in + no id: the built-in harness runs locally with a platform-proxied
  worker and org-level usage record.
* logged out with no target: the built-in baseline pi agent can use the user's
  local provider credentials.

Only the bare built-in path executes harness code and bash on the user's real
machine, so that path retains the explicit local-execution consent prompt.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Protocol

import typer
from rich.console import Console
from rich.panel import Panel

from wmh.cli.workspace_sync import (
    WorkspaceSnapshot,
    WorkspaceSyncError,
    apply_workspace_patch,
    snapshot_workspace,
    sync_workspace,
    write_conflict_archive,
)
from wmh.engine.play import parse_action
from wmh.harness.doc import RUNTIME_KIND_ID, HarnessDoc, Surface, SurfaceKind
from wmh.harness.live_session import LiveSession, SessionEvent, ToolOutcome
from wmh.harness.pi_local import LocalStdioChannel, start_local_live_runner
from wmh.harness.pi_vendor import pi_agent_code_surfaces
from wmh.harness.skills import SkillLibrary
from wmh.harness.tools import render_tools, resolve_tools
from wmh.harness.workspace_patch import WorkspacePatchError, build_workspace_patch
from wmh.platform.client import PlatformClient, PlatformError, RemoteAgentSession
from wmh.platform.credentials import load_credentials
from wmh.providers.base import ProviderConfig, ProviderKind, ToolCallingProvider
from wmh.providers.models import resolve_provider_model
from wmh.providers.registry import get_provider

if TYPE_CHECKING:
    from collections.abc import Callable

    from llm_waterfall import ChatRequest, ChatResponse

    from wmh.core.types import JsonObject

_console = Console()

# Per-tool-call output cap (head+tail) reported to the transcript.
_TOOL_OUTPUT_CAP = 16_000
_BASH_TIMEOUT_S = 300.0
# Driver housekeeping cadence (event flush).
_TICK_S = 5.0
_WORKSPACE_SYNC_TICK_S = 1.0
# Default local worker when the user pins none.
_DEFAULT_PROVIDER = "bedrock"
_DEFAULT_MODEL = "claude-opus-4-8"


class _JailEscape(RuntimeError):
    """A tool path resolved outside the session's working directory."""


def _capped(content: str, *, is_error: bool = False) -> ToolOutcome:
    """Cap tool output to the head+tail budget with a truncation marker."""
    if len(content) <= _TOOL_OUTPUT_CAP:
        return ToolOutcome(content=content, is_error=is_error)
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
    tool_specs = resolve_tools(doc.tools())
    system = f"{doc.system_prompt()}\n\n## Tools\n{render_tools(tool_specs)}"
    skills = SkillLibrary(doc.skills())
    index = skills.render_index()
    if index:
        system += f"\n\n## Your skills (read a body with read_skill)\n{index}"
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
        try:
            result = subprocess.run(  # noqa: S603 - the agent's tool is meant to run shell commands
                ["bash", "-lc", command],  # noqa: S607 - bash on PATH is the documented contract
                cwd=self._jail,
                capture_output=True,
                text=True,
                timeout=_BASH_TIMEOUT_S,
                check=False,
            )
            stdout, stderr, exit_code = result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired as error:
            stdout = _as_text(error.stdout)
            stderr = _as_text(error.stderr) + f"\n[timed out after {int(_BASH_TIMEOUT_S)}s]"
            exit_code = 124
        if stdout:
            emit("stdout", stdout)
        if stderr:
            emit("stderr", stderr)
        body = stdout + stderr
        if exit_code != 0:
            body = f"{body}\n[exit {exit_code}]"
        return _capped(body, is_error=exit_code != 0)


def _as_text(value: object) -> str:
    """Coerce subprocess stdout/stderr (str | bytes | None) to text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else ""


class RunRecorder(Protocol):
    """Recording slice consumed by the local driver and terminal event sink."""

    def record(self, event: SessionEvent) -> None: ...

    def flush(self) -> None: ...

    def finish(self, *, ended_reason: str, error: str | None) -> None: ...


class LocalPiRunRecorder:
    """Finish and close an org-scoped built-in pi run; it has no transcript."""

    def __init__(self, client: PlatformClient, org_id: str, run_id: str) -> None:
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
        with contextlib.suppress(PlatformError):
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
                _console.print(f"\n[bold cyan]agent[/bold cyan] {text}")
        elif event.kind == "tool_call":
            _console.print(f"[dim]$ {payload.get('name', '')} {payload.get('arguments', '')}[/dim]")
        elif event.kind == "tool_output":
            _console.print(str(payload.get("text", "")), end="", markup=False, highlight=False)
        elif event.kind == "tool_result":
            if payload.get("is_error"):
                _console.print(f"[red]{payload.get('content', '')}[/red]")
        elif event.kind == "submit":
            _console.print(f"\n[bold green]submitted[/bold green] {payload.get('answer', '')}")
        elif event.kind == "state":
            status = str(payload.get("status", ""))
            self._on_running(status == "running")
            _console.print(f"[dim]({status})[/dim]")
        elif event.kind == "error":
            _console.print(f"[red]error: {payload.get('message', '')}[/red]")


class StdinCommandReader(threading.Thread):
    """Feed typed stdin lines as steer/interrupt/end intents to the session."""

    def __init__(self, session: LiveSession) -> None:
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
    ) -> None:
        """Configure the driver; ``run`` performs boot, loop, and teardown."""
        self._jail = jail_root
        self._doc = doc
        self._provider = provider
        self._worker_fn = worker_fn
        self._recorder = recorder
        self._instruction = instruction
        self._executor = LocalToolExecutor(jail_root)
        self._channel: LocalStdioChannel | None = None
        self._interrupts = 0

    def run(self) -> None:
        """Boot the local runner, drive the session, and always tear down."""
        system, tool_specs, files, skill_bodies = _assemble(self._doc)
        _console.print("[dim]starting the built-in pi harness locally...[/dim]")
        session: LiveSession | None = None
        reason = "user_ended"
        error: str | None = None
        try:
            channel = start_local_live_runner()
            self._channel = channel
            session = LiveSession(
                channel,
                tools=tool_specs,
                execute_tool=self._execute,
                on_event=TerminalEventSink(
                    recorder=self._recorder, on_running=lambda _running: None
                ),
                files=files,
                system_prompt=system,
                skill_bodies=skill_bodies,
                provider=self._provider,
                worker_fn=self._worker_fn,
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
            _console.print(f"[red]session failed: {exc}[/red]")
        finally:
            self._teardown(session, reason=reason, error=error)
        if error is not None:
            raise typer.Exit(code=1)

    def _execute(
        self, name: str, args: JsonObject, emit: Callable[[str, str], None]
    ) -> ToolOutcome:
        """Run one tool locally (each tool blocks the session pump)."""
        return self._executor(name, args, emit)

    def _loop(self, session: LiveSession, stdin_eof: threading.Event) -> None:
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

    def _handle_sigint(self, session: LiveSession) -> None:
        """First Ctrl-C interrupts the current turn; a second ends the session."""
        self._interrupts += 1
        if self._interrupts == 1:
            _console.print("\n[yellow]interrupting (press Ctrl-C again to quit)[/yellow]")
            session.interrupt()
        else:
            _console.print("\n[yellow]ending session[/yellow]")
            session.end()

    def _teardown(self, session: LiveSession | None, *, reason: str, error: str | None) -> None:
        if session is not None and not session.closed:
            with contextlib.suppress(Exception):
                session.end()
        if self._recorder is not None:
            self._recorder.finish(ended_reason=reason, error=error)
        if self._channel is not None:
            with contextlib.suppress(Exception):
                self._channel.close()
        _console.print(f"[dim]session ended ({reason})[/dim]")


class RemoteAgentCommandReader(threading.Thread):
    """Forward terminal lines to a platform-owned E2B agent session."""

    def __init__(self, client: PlatformClient, agent_id: str, session_id: str) -> None:
        """Store the hosted command-channel identity."""
        super().__init__(daemon=True)
        self._client = client
        self._agent_id = agent_id
        self._session_id = session_id
        self.eof = threading.Event()

    def run(self) -> None:
        """Map stdin lines to hosted steer, interrupt, and end commands."""
        try:
            for raw in sys.stdin:
                line = raw.strip()
                if line in {":quit", ":q", ":exit"}:
                    self._post("end")
                    return
                if line == ":stop":
                    self._post("interrupt")
                elif line:
                    self._post("user_message", text=line)
        except (OSError, PlatformError):
            pass
        finally:
            self.eof.set()

    def _post(self, kind: str, *, text: str | None = None) -> None:
        """Post one command through the authenticated platform client."""
        self._client.post_agent_session_command(self._agent_id, self._session_id, kind, text=text)


class RemoteAgentDriver:
    """Stream a hosted E2B agent, optionally syncing one local workspace."""

    def __init__(
        self,
        client: PlatformClient,
        target_id: str,
        name: str,
        jail_root: Path | None,
        task: str | None,
    ) -> None:
        """Store the resolved agent and optional local workspace transport root."""
        self._client = client
        self._target_id = target_id
        self._name = name
        self._jail = jail_root
        self._task = task
        self._interrupts = 0
        self._live_conflicts: set[str] = set()

    def run(self) -> None:
        """Run and stream one E2B session, syncing local files only when requested."""
        try:
            initial: WorkspaceSnapshot | None = None
            if self._jail is not None:
                with _console.status("[dim]snapshotting local workspace...[/dim]", spinner="dots"):
                    initial = snapshot_workspace(self._jail)
                _console.print(
                    f"[dim]uploading {len(initial.files)} files to the platform "
                    "E2B workspace...[/dim]"
                )
            session = self._client.create_agent_session(
                self._target_id,
                workspace=initial.archive if initial is not None else None,
                instruction=self._task,
            )
            quit_detail = "end and sync back" if initial is not None else "end"
            _console.print(
                f"[green]E2B session started[/green] for [bold]{self._name}[/bold]. "
                "Type to steer, [bold]:stop[/bold] to interrupt, "
                f"[bold]:quit[/bold] to {quit_detail}."
            )
            reader = RemoteAgentCommandReader(self._client, self._target_id, session.id)
            reader.start()
            stdin_eof = getattr(reader, "eof", threading.Event())
            terminal, synchronized = self._poll(session.id, initial, stdin_eof)
            jail_root = self._jail
            workspace_conflicts = False
            if jail_root is not None and synchronized is not None:
                with _console.status("[dim]syncing E2B workspace back...[/dim]", spinner="dots"):
                    final_archive = self._client.download_agent_workspace(
                        self._target_id, session.id
                    )
                    result = sync_workspace(
                        jail_root,
                        synchronized,
                        final_archive,
                        protected_paths=frozenset(self._live_conflicts),
                    )
                if result.conflicts:
                    workspace_conflicts = True
                    recovery = write_conflict_archive(jail_root, session.id, final_archive)
                    self._client.acknowledge_agent_workspace(self._target_id, session.id)
                    paths = ", ".join(result.conflicts)
                    _console.print(
                        f"[red]workspace conflicts preserved locally[/red]: {paths}\n"
                        f"The full E2B result is saved at [bold]{recovery}[/bold]."
                    )
                else:
                    self._client.acknowledge_agent_workspace(self._target_id, session.id)
                    _console.print(
                        f"[green]workspace synced[/green] ({len(result.applied)} changed paths)"
                    )
            if terminal.status == "failed":
                _console.print(f"[red]session failed: {terminal.error or 'unknown error'}[/red]")
                raise typer.Exit(code=1)
            if workspace_conflicts:
                raise typer.Exit(code=2)
        except (WorkspacePatchError, WorkspaceSyncError) as error:
            raise typer.BadParameter(str(error)) from error
        except PlatformError as error:
            raise typer.BadParameter(str(error)) from error
        finally:
            self._client.close()

    def _poll(
        self,
        session_id: str,
        synchronized: WorkspaceSnapshot | None,
        stdin_eof: threading.Event,
    ) -> tuple[RemoteAgentSession, WorkspaceSnapshot | None]:
        """Render new transcript events until output export makes the row terminal."""
        cursor = 0
        last_workspace_push = time.monotonic()
        sink = TerminalEventSink(recorder=None, on_running=lambda _running: None)
        saw_running = False
        end_sent = False
        while True:
            try:
                page = self._client.list_agent_session_events(
                    self._target_id, session_id, after=cursor
                )
                for event in page.events:
                    if event.kind == "workspace_patch":
                        if synchronized is None:
                            raise WorkspaceSyncError(
                                "received a workspace patch without --upload-dir"
                            )
                        synchronized = self._apply_remote_patch(session_id, event, synchronized)
                    elif event.kind == "status":
                        detail = event.payload.get("message") or event.payload.get("status")
                        if detail:
                            _console.print(f"[dim]({detail})[/dim]")
                    else:
                        sink(SessionEvent(kind=event.kind, payload=event.payload))
                        if event.kind == "state":
                            state = event.payload.get("status")
                            saw_running = saw_running or state == "running"
                            if (
                                stdin_eof.is_set()
                                and not end_sent
                                and saw_running
                                and state == "idle"
                            ):
                                self._client.post_agent_session_command(
                                    self._target_id, session_id, "end"
                                )
                                end_sent = True
                cursor = page.last_seq
                if page.status in {"ended", "failed"}:
                    return (
                        self._client.get_agent_session(self._target_id, session_id),
                        synchronized,
                    )
                now = time.monotonic()
                if stdin_eof.is_set() and self._task is None and not end_sent:
                    self._client.post_agent_session_command(self._target_id, session_id, "end")
                    end_sent = True
                if synchronized is not None and now - last_workspace_push >= _WORKSPACE_SYNC_TICK_S:
                    synchronized = self._push_local_patch(session_id, synchronized)
                    last_workspace_push = now
                time.sleep(0.5)
            except KeyboardInterrupt:
                self._interrupts += 1
                kind = "interrupt" if self._interrupts == 1 else "end"
                _console.print(
                    "\n[yellow]interrupting (press Ctrl-C again to end)[/yellow]"
                    if kind == "interrupt"
                    else "\n[yellow]ending session[/yellow]"
                )
                self._client.post_agent_session_command(self._target_id, session_id, kind)
                continue

    def _apply_remote_patch(
        self,
        session_id: str,
        event: object,
        synchronized: WorkspaceSnapshot,
    ) -> WorkspaceSnapshot:
        """Download and apply one announced E2B patch, then advance the local base."""
        jail_root = self._jail
        if jail_root is None:
            raise WorkspaceSyncError("workspace sync is not enabled")
        payload = getattr(event, "payload", {})
        revision_value = payload.get("revision") if isinstance(payload, dict) else None
        if not isinstance(revision_value, str) or not revision_value:
            raise WorkspaceSyncError("workspace patch event has no revision")
        content = self._client.download_agent_workspace_patch(
            self._target_id, session_id, revision_value
        )
        result = apply_workspace_patch(jail_root, content)
        self._live_conflicts.update(result.conflicts)
        self._client.acknowledge_agent_workspace_patch(self._target_id, session_id, revision_value)
        if result.applied:
            _console.print(f"[dim]workspace updated ({len(result.applied)} changed paths)[/dim]")
        if result.conflicts:
            paths = ", ".join(result.conflicts)
            _console.print(f"[yellow]workspace sync conflict[/yellow]: {paths}")
        return snapshot_workspace(jail_root)

    def _push_local_patch(
        self, session_id: str, synchronized: WorkspaceSnapshot
    ) -> WorkspaceSnapshot:
        """Send local edits made since the last synchronized snapshot."""
        jail_root = self._jail
        if jail_root is None:
            return synchronized
        try:
            current = snapshot_workspace(jail_root)
        except WorkspaceSyncError:
            return synchronized
        content = build_workspace_patch(synchronized.archive, current.archive)
        if content is None:
            return synchronized
        result = self._client.upload_agent_workspace_patch(self._target_id, session_id, content)
        self._live_conflicts.update(result.conflicts)
        if result.conflicts:
            paths = ", ".join(result.conflicts)
            _console.print(f"[yellow]workspace sync conflict[/yellow]: {paths}")
        # A conflicted path was rejected by E2B, so ``current`` cannot become
        # the synchronized base. Keep the prior base and conservatively retry
        # accepted sibling paths until the conflict is reconciled at teardown.
        return synchronized if result.conflicts else current


class RemoteWorldModelDriver:
    """Interactive terminal loop over the platform's world-model session API."""

    def __init__(self, client: PlatformClient, target_id: str, name: str, task: str | None) -> None:
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
                    title=f"[bold]running world model[/bold] {self._name}",
                    subtitle=f"task: {self._task}" if self._task else "no task set",
                    border_style="cyan",
                )
            )
            self._loop(session.id)
        except PlatformError as error:
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
            if line in {":help", ":h"}:
                _console.print(
                    'Tool call: [cyan]name {"arg": "value"}[/cyan]. '
                    "Any other text is sent as a message."
                )
                continue
            try:
                action = parse_action(line)
            except ValueError as error:
                _console.print(f"[red]parse error[/red]: {error}")
                continue
            try:
                with _console.status("[dim]world model thinking...[/dim]", spinner="dots"):
                    observation = self._client.step_world_model_session(session_id, action)
            except PlatformError as error:
                _console.print(f"[red]step failed[/red]: {error}")
                continue
            style = "red" if observation.is_error else "green"
            title = "error" if observation.is_error else "observation"
            _console.print(
                Panel(
                    observation.content or "[dim](empty)[/dim]",
                    title=title,
                    border_style=style,
                )
            )


def _local_worker_provider(provider: str | None, model: str | None) -> ToolCallingProvider:
    """Build the logged-out worker provider from local environment credentials."""
    try:
        kind = ProviderKind(provider or _DEFAULT_PROVIDER)
    except ValueError:
        kinds = ", ".join(k.value for k in ProviderKind)
        msg = f"unknown provider {provider!r}; choose one of: {kinds}"
        raise typer.BadParameter(msg) from None
    spec = resolve_provider_model(kind, model or _DEFAULT_MODEL)
    built = get_provider(ProviderConfig(kind=kind, model_type=spec.model_type, model=spec.model_id))
    if not isinstance(built, ToolCallingProvider):
        msg = f"provider {kind.value}/{spec.model_id} does not support structured tool calling"
        raise typer.BadParameter(msg)
    return built


_TARGET_ARG = typer.Argument(
    help="Platform world-model or agent id (omit to run the built-in pi harness locally)."
)
_DIR_OPT = typer.Option("--dir", help="Working directory for the built-in local pi harness.")
_UPLOAD_DIR_OPT = typer.Option(
    "-u", "--upload-dir", help="Directory to upload and live-sync for a hosted agent."
)
_PROVIDER_OPT = typer.Option(
    "--provider", help="Worker provider for the built-in local pi harness."
)
_MODEL_OPT = typer.Option("--model", help="Worker model for the built-in local pi harness.")
_TASK_OPT = typer.Option("--task", "--instruction", help="Opening task for either execution kind.")
_YES_OPT = typer.Option("--yes", help="Skip the local-execution consent prompt.")


def register(app: typer.Typer) -> None:
    """Register the unified ``wmh run`` command on the root app."""

    @app.command("run")
    def run(
        target: Annotated[str | None, _TARGET_ARG] = None,
        directory: Annotated[str | None, _DIR_OPT] = None,
        upload_directory: Annotated[str | None, _UPLOAD_DIR_OPT] = None,
        provider: Annotated[str | None, _PROVIDER_OPT] = None,
        model: Annotated[str | None, _MODEL_OPT] = None,
        task: Annotated[str | None, _TASK_OPT] = None,
        yes: Annotated[bool, _YES_OPT] = False,
    ) -> None:
        """Run a platform world model/agent by id, or the built-in pi harness."""
        if target is None and upload_directory is not None:
            raise typer.BadParameter("--upload-dir is only supported for platform agent ids")
        if target is not None and directory is not None:
            raise typer.BadParameter(
                "--dir is only supported for a bare `wmh run`; use --upload-dir for an agent"
            )
        if target is None:
            path = directory or "."
        else:
            path = upload_directory
        jail_root = Path(path).resolve() if path is not None else None
        if jail_root is not None and not jail_root.is_dir():
            msg = f"working directory does not exist: {jail_root}"
            raise typer.BadParameter(msg)
        confirm_local: Callable[[], None] | None = None
        if target is None and jail_root is not None:
            local_root = jail_root

            def confirm_execution() -> None:
                """Confirm the bare harness's local execution boundary."""
                _confirm_local_execution(local_root, target=target, yes=yes)

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


def _confirm_local_execution(jail_root: Path, *, target: str | None, yes: bool) -> None:
    """Warn that harness code and bash run with local user permissions."""
    label = f"agent {target}" if target else "the built-in pi harness"
    _console.print(
        f"[bold yellow]{label}, its harness code, and shell commands run on THIS machine"
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
) -> LocalLiveDriver | RemoteAgentDriver | RemoteWorldModelDriver:
    """Resolve the target kind once and assemble its execution driver."""
    credentials = load_credentials()
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
                "or run `wmh logout` to use local credentials"
            )
        if confirm_local is not None:
            confirm_local()
        client = PlatformClient(str(credentials.api_url), str(credentials.token))
        try:
            org_id = _default_org(client, credentials.default_org)
            run = client.create_local_pi_run(org_id)
        except typer.BadParameter:
            client.close()
            raise
        except PlatformError as error:
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
        )

    if not logged_in:
        msg = "run `wmh login` to run a platform id, or omit the id to run the built-in pi harness"
        raise typer.BadParameter(msg)

    if provider is not None or model is not None:
        raise typer.BadParameter(
            "platform target runs use platform credentials; --provider/--model are not accepted"
        )
    client = PlatformClient(str(credentials.api_url), str(credentials.token))
    try:
        resolved = client.resolve_run_target(target)
        if resolved.kind == "world_model":
            if jail_root is not None:
                client.close()
                raise typer.BadParameter("--upload-dir is only supported for agent ids")
            return RemoteWorldModelDriver(client, resolved.id, resolved.name, task)
        return RemoteAgentDriver(client, resolved.id, resolved.name, jail_root, task)
    except PlatformError as error:
        client.close()
        raise typer.BadParameter(str(error)) from error


def _default_org(client: PlatformClient, configured: str | None) -> str:
    """Resolve the login's organization, auto-picking only an unambiguous sole org."""
    if configured is not None:
        return configured
    identity = client.whoami()
    if len(identity.orgs) == 1:
        return identity.orgs[0].id
    raise typer.BadParameter(
        "no default organization selected; run `wmh login` again and choose an organization"
    )
