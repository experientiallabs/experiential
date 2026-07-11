# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""`wmh session start`: run an agent live against the LOCAL working directory.

The agent process (the real vendored pi harness) runs in a cloud E2B sandbox,
but every tool it calls is answered HERE by the CLI against a jailed local
directory: read_file / write_file / bash all hit the user's real disk, so the
agent's view IS the local tree (a bind-mount feel with no file sync). The
sandbox only hosts the pi runner process.

Three credential modes, chosen automatically (see :func:`register`):

* logged in, default: the worker LLM turn is answered by the platform through a
  metered proxy (platform keys, billed to the org), and the session is recorded
  on the platform (create -> append events -> finish) so it shows up in the UI.
* logged in, ``--local-provider``: the worker runs on the user's own local keys,
  but the session is still recorded.
* not logged in (or no agent given): a built-in baseline pi agent runs fully
  locally on the user's own keys and nothing is recorded.

The agent's bash runs on the user's real machine, so a consent prompt and a hard
path jail (every read/write resolves under the chosen directory) are mandatory.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console

from wmh.harness.doc import RUNTIME_KIND_ID, HarnessDoc, Surface, SurfaceKind
from wmh.harness.e2b_sandbox import default_sandbox_factory
from wmh.harness.live_session import LiveSession, SessionEvent, ToolOutcome
from wmh.harness.pi_e2b import start_live_runner
from wmh.harness.pi_vendor import pi_agent_code_surfaces
from wmh.harness.skills import SkillLibrary
from wmh.harness.tools import render_tools, resolve_tools
from wmh.platform.client import PlatformClient, PlatformError
from wmh.platform.credentials import load_credentials
from wmh.providers.base import ProviderConfig, ProviderKind, ToolCallingProvider
from wmh.providers.models import resolve_provider_model
from wmh.providers.registry import get_provider

if TYPE_CHECKING:
    from collections.abc import Callable

    from llm_waterfall import ChatRequest, ChatResponse

    from wmh.core.types import JsonObject, JsonValue
    from wmh.harness.e2b_sandbox import SandboxHandle

_console = Console()

# Per-tool-call output cap (head+tail) reported to the transcript.
_TOOL_OUTPUT_CAP = 16_000
_BASH_TIMEOUT_S = 300.0
# How far ahead each keepalive pushes the sandbox timeout while the agent runs.
_KEEPALIVE_S = 900
# Short initial ceiling: if the driver dies during boot the sandbox self-expires.
_STARTUP_TIMEOUT_S = 900
# Driver housekeeping cadence (event flush + keepalive).
_TICK_S = 5.0
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
    code, materialized into the sandbox), and skill bodies answered host-side.
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

    ``HarnessDoc.baseline`` is the in-process loop, which the live E2B runner
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
    """Answer read_file / write_file / bash against a jailed local directory."""

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
            return ToolOutcome(
                content=f"path {error} escapes the session directory", is_error=True
            )
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


class SessionRecorder:
    """Best-effort mirror of the local transcript to the platform (create done already)."""

    def __init__(self, client: PlatformClient, agent_id: str, session_id: str) -> None:
        """Buffer events for the given already-created platform session."""
        self._client = client
        self._agent_id = agent_id
        self._session_id = session_id
        self._buffer: list[dict[str, JsonValue]] = []

    def record(self, event: SessionEvent) -> None:
        """Queue one event for the next flush."""
        self._buffer.append({"kind": event.kind, "payload": event.payload})

    def flush(self) -> None:
        """Report queued events; drop the batch on a transport error (best-effort)."""
        if not self._buffer:
            return
        batch, self._buffer = self._buffer, []
        try:
            self._client.append_local_events(self._agent_id, self._session_id, batch)
        except PlatformError as error:
            _console.print(f"[yellow]could not report {len(batch)} events: {error}[/yellow]")

    def finish(self, *, ended_reason: str, sandbox_seconds: int, error: str | None) -> None:
        """Flush the tail and post the terminal transition."""
        self.flush()
        status = "failed" if error is not None else "ended"
        with contextlib.suppress(PlatformError):
            self._client.finish_local_session(
                self._agent_id,
                self._session_id,
                status=status,
                ended_reason=ended_reason,
                sandbox_seconds=sandbox_seconds,
                error=error,
            )


class TerminalEventSink:
    """Render the SessionEvent stream to the terminal and mirror it to a recorder."""

    def __init__(
        self,
        *,
        recorder: SessionRecorder | None,
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
        # EOF (Ctrl-D): end the session gracefully.
        with contextlib.suppress(Exception):
            self._session.end()


class LocalLiveDriver:
    """Own one E2B sandbox + LiveSession and drive it against the local directory."""

    def __init__(
        self,
        *,
        jail_root: Path,
        doc: HarnessDoc,
        provider: ToolCallingProvider | None,
        worker_fn: Callable[[ChatRequest], ChatResponse] | None,
        recorder: SessionRecorder | None,
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
        self._sandbox: SandboxHandle | None = None
        self._sandbox_started = 0.0
        self._agent_running = False
        self._interrupts = 0

    def run(self) -> None:
        """Boot the sandbox + runner, drive the session, and always tear down."""
        system, tool_specs, files, skill_bodies = _assemble(self._doc)
        factory = default_sandbox_factory(
            timeout=_STARTUP_TIMEOUT_S, metadata={"kind": "wmh-local-session"}
        )
        _console.print("[dim]creating sandbox and starting the agent...[/dim]")
        sandbox = factory()
        self._sandbox = sandbox
        self._sandbox_started = time.time()
        session: LiveSession | None = None
        reason = "user_ended"
        error: str | None = None
        try:
            channel = start_live_runner(sandbox)
            session = LiveSession(
                channel,
                tools=tool_specs,
                execute_tool=self._execute,
                on_event=TerminalEventSink(
                    recorder=self._recorder, on_running=self._set_running
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
            StdinCommandReader(session).start()
            self._loop(session)
        except Exception as exc:  # noqa: BLE001 - report any driver failure, then tear down
            error = str(exc)
            reason = "error"
            _console.print(f"[red]session failed: {exc}[/red]")
        finally:
            self._teardown(session, reason=reason, error=error)

    def _execute(
        self, name: str, args: JsonObject, emit: Callable[[str, str], None]
    ) -> ToolOutcome:
        """Extend the sandbox, then run the tool locally (each tool blocks the pump)."""
        self._extend_sandbox()
        return self._executor(name, args, emit)

    def _loop(self, session: LiveSession) -> None:
        last_tick = 0.0
        while not session.closed:
            try:
                session.pump(timeout=0.5)
            except KeyboardInterrupt:
                self._handle_sigint(session)
            now = time.monotonic()
            if now - last_tick >= _TICK_S:
                last_tick = now
                if self._recorder is not None:
                    self._recorder.flush()
                if self._agent_running:
                    self._extend_sandbox()

    def _handle_sigint(self, session: LiveSession) -> None:
        """First Ctrl-C interrupts the current turn; a second ends the session."""
        self._interrupts += 1
        if self._interrupts == 1:
            _console.print("\n[yellow]interrupting (press Ctrl-C again to quit)[/yellow]")
            session.interrupt()
        else:
            _console.print("\n[yellow]ending session[/yellow]")
            session.end()

    def _set_running(self, running: bool) -> None:
        self._agent_running = running

    def _extend_sandbox(self) -> None:
        if self._sandbox is not None:
            with contextlib.suppress(Exception):
                self._sandbox.set_timeout(_KEEPALIVE_S)

    def _teardown(self, session: LiveSession | None, *, reason: str, error: str | None) -> None:
        if session is not None and not session.closed:
            with contextlib.suppress(Exception):
                session.end()
        seconds = int(time.time() - self._sandbox_started) if self._sandbox_started else 0
        if self._recorder is not None:
            self._recorder.finish(ended_reason=reason, sandbox_seconds=seconds, error=error)
        if self._sandbox is not None:
            with contextlib.suppress(Exception):
                self._sandbox.kill()
        _console.print(f"[dim]session ended ({reason}), sandbox ran {seconds}s[/dim]")


def _local_worker_provider(provider: str | None, model: str | None) -> ToolCallingProvider:
    """Build a local worker provider from env creds (the --local-provider path)."""
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


_AGENT_ARG = typer.Argument(
    help="Platform agent id to run (omit to run the built-in baseline agent locally)."
)
_DIR_OPT = typer.Option("--dir", help="Local working directory the agent's tools act on.")
_LOCAL_PROVIDER_OPT = typer.Option(
    "--local-provider", help="Answer the worker LLM with your own local keys, not the platform."
)
_PROVIDER_OPT = typer.Option(
    "--provider", help="Local worker provider kind (with --local-provider)."
)
_MODEL_OPT = typer.Option("--model", help="Local worker model type (with --local-provider).")
_INSTRUCTION_OPT = typer.Option("--instruction", help="Opening instruction to send on start.")
_YES_OPT = typer.Option("--yes", help="Skip the local-execution consent prompt.")


def register(app: typer.Typer) -> None:
    """Register the ``wmh session`` command group on the root app."""
    session_app = typer.Typer(
        help="Run an agent live against your local working directory.", no_args_is_help=True
    )

    @session_app.command("start")
    def start(  # noqa: PLR0913 - a CLI entrypoint's options are its surface
        agent: Annotated[str | None, _AGENT_ARG] = None,
        directory: Annotated[str, _DIR_OPT] = ".",
        local_provider: Annotated[bool, _LOCAL_PROVIDER_OPT] = False,
        provider: Annotated[str | None, _PROVIDER_OPT] = None,
        model: Annotated[str | None, _MODEL_OPT] = None,
        instruction: Annotated[str | None, _INSTRUCTION_OPT] = None,
        yes: Annotated[bool, _YES_OPT] = False,
    ) -> None:
        """Start an interactive local session (agent in E2B, tools on your disk)."""
        jail_root = Path(directory).resolve()
        if not jail_root.is_dir():
            msg = f"working directory does not exist: {jail_root}"
            raise typer.BadParameter(msg)
        _confirm_local_execution(jail_root, agent=agent, yes=yes)
        driver = _build_driver(
            agent=agent,
            jail_root=jail_root,
            local_provider=local_provider,
            provider=provider,
            model=model,
            instruction=instruction,
        )
        driver.run()

    app.add_typer(session_app, name="session")


def _confirm_local_execution(jail_root: Path, *, agent: str | None, yes: bool) -> None:
    """Warn that the agent's bash runs on the real machine; require consent."""
    target = f"agent {agent}" if agent else "the built-in baseline agent"
    _console.print(
        f"[bold yellow]{target} will run shell commands on THIS machine[/bold yellow], "
        f"confined to:\n  {jail_root}"
    )
    if not yes and not typer.confirm("continue?"):
        raise typer.Exit(code=1)


def _build_driver(
    *,
    agent: str | None,
    jail_root: Path,
    local_provider: bool,
    provider: str | None,
    model: str | None,
    instruction: str | None,
) -> LocalLiveDriver:
    """Resolve the credential mode (A/B/C) and assemble the driver."""
    credentials = load_credentials()
    logged_in = credentials.is_complete()

    if agent is None:
        # Built-in baseline, fully local + ephemeral (no recording).
        if not logged_in:
            _console.print(
                "[dim]not logged in: running the built-in baseline agent, no recording[/dim]"
            )
        return LocalLiveDriver(
            jail_root=jail_root,
            doc=_pi_node_baseline(),
            provider=_local_worker_provider(provider, model),
            worker_fn=None,
            recorder=None,
            instruction=instruction,
        )

    if not logged_in:
        msg = (
            "run `wmh login` to run a platform agent, or omit the agent id "
            "to run the built-in baseline agent locally"
        )
        raise typer.BadParameter(msg)

    client = PlatformClient(str(credentials.api_url), str(credentials.token))
    try:
        champion = client.fetch_champion_harness(agent)
        doc = HarnessDoc.model_validate(champion.doc)
        session = client.create_local_session(agent, title=instruction)
    except PlatformError as error:
        raise typer.BadParameter(str(error)) from error

    recorder = SessionRecorder(client, agent, session.id)
    if local_provider:
        return LocalLiveDriver(
            jail_root=jail_root,
            doc=doc,
            provider=_local_worker_provider(provider, model),
            worker_fn=None,
            recorder=recorder,
            instruction=instruction,
        )

    def worker_fn(request: ChatRequest) -> ChatResponse:
        return client.complete_worker(agent, session.id, request)

    return LocalLiveDriver(
        jail_root=jail_root,
        doc=doc,
        provider=None,
        worker_fn=worker_fn,
        recorder=recorder,
        instruction=instruction,
    )
