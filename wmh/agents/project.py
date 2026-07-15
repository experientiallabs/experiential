"""Persistent E2B filesystem projects driven by the shared pi session runtime."""

from __future__ import annotations

import contextlib
import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from wmh.core.types import JsonObject
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_sandbox import (
    SandboxHandle,
    SandboxUsage,
    create_sandbox,
    default_sandbox_factory,
)
from wmh.harness.live_session import LiveSession, SessionEvent, ToolOutcome
from wmh.harness.pi_e2b import start_live_runner
from wmh.harness.runner_link import Channel, TokenUsage
from wmh.harness.tools import resolve_tools
from wmh.providers.base import ToolCallingProvider

PROJECT_WORKSPACE = "/home/user/project"
DEFAULT_PROJECT_TIMEOUT_S = 21_600
_OUTPUT_CAP = 16_000
_PROJECT_TOOLS = frozenset({"read_file", "write_file", "submit"})


class ChannelFactory(Protocol):
    """Start one fresh runner channel in a project's sandbox."""

    def __call__(self, sandbox: SandboxHandle, workspace: str) -> Channel: ...


@dataclass(frozen=True)
class AgentProjectRun:
    """Result of one agent turn inside a project."""

    answer: str
    events: tuple[SessionEvent, ...]
    worker_usage: TokenUsage


class AgentProject:
    """A persistent filesystem that can run project-scoped pi agents.

    The project owns environment state, while :class:`LiveSession` owns ordinary agent execution.
    Repeated ``run`` calls for the same agent and provider reuse one live session and its
    transcript.
    Changing the agent harness or provider starts a new session against the same filesystem.
    """

    def __init__(
        self,
        sandbox: SandboxHandle,
        *,
        workspace: str = PROJECT_WORKSPACE,
        channel_factory: ChannelFactory | None = None,
        owns_sandbox: bool = True,
    ) -> None:
        self._sandbox = sandbox
        self.workspace = workspace.rstrip("/")
        self._channel_factory = channel_factory or _start_channel
        self._owns_sandbox = owns_sandbox
        self._started_at = time.monotonic()
        self._finished_at: float | None = None
        self._channel: Channel | None = None
        self._session: LiveSession | None = None
        self._session_agent_hash: str | None = None
        self._session_provider: ToolCallingProvider | None = None
        self._active_event_sink: Callable[[SessionEvent], None] | None = None
        self._sandbox.commands.run(f"mkdir -p {shlex.quote(self.workspace)}", timeout=30)

    @classmethod
    def create(
        cls,
        *,
        timeout: float = DEFAULT_PROJECT_TIMEOUT_S,
        template: str | None = None,
        api_key: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> AgentProject:
        """Create one owned E2B project sandbox."""
        sandbox = create_sandbox(
            default_sandbox_factory(
                timeout=timeout,
                template=template,
                api_key=api_key,
                metadata=metadata,
            )
        )
        return cls(sandbox)

    def write_text(self, path: str, content: str) -> None:
        """Write one project-relative file without allowing path traversal."""
        absolute = self._absolute_path(path)
        directory = str(PurePosixPath(absolute).parent)
        self._sandbox.commands.run(f"mkdir -p {shlex.quote(directory)}", timeout=30)
        self._sandbox.files.write(absolute, content)

    def read_text(self, path: str) -> str:
        """Read one project-relative file."""
        return self._sandbox.files.read(self._absolute_path(path))

    def run(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str,
        *,
        timeout: float = DEFAULT_PROJECT_TIMEOUT_S,
        on_event: Callable[[SessionEvent], None] | None = None,
    ) -> AgentProjectRun:
        """Run one turn of an ordinary agent against this persistent project."""
        if self._finished_at is not None:
            raise RuntimeError("cannot run an agent in a closed project")
        if self._active_event_sink is not None:
            raise RuntimeError("a project agent turn is already running")
        unsupported = set(agent.tools()) - _PROJECT_TOOLS
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"project agents cannot use uncontained tools: {names}")
        session = self._ensure_session(agent, provider)
        events: list[SessionEvent] = []
        answer = ""
        turn_started = False
        turn_finished = False

        def sink(event: SessionEvent) -> None:
            nonlocal answer, turn_finished
            events.append(event)
            if event.kind == "submit":
                submitted = event.payload.get("answer")
                answer = submitted if isinstance(submitted, str) else ""
            elif turn_started and event.kind == "state" and event.payload.get("status") == "idle":
                turn_finished = True
            if on_event is not None:
                on_event(event)

        usage_before = TokenUsage(
            input_tokens=session.worker_usage.input_tokens,
            output_tokens=session.worker_usage.output_tokens,
            calls=session.worker_usage.calls,
        )
        self._active_event_sink = sink
        try:
            session.send_user_message(instruction)
            turn_started = True
            deadline = time.monotonic() + timeout
            while not turn_finished:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    session.interrupt("project_run_timeout")
                    session.pump(timeout=0)
                    raise TimeoutError(f"project agent did not finish within {timeout:g}s")
                if not session.pump(timeout=min(0.5, remaining)) and not turn_finished:
                    if session.failure_message is not None:
                        raise RuntimeError(
                            f"project agent session failed: {session.failure_message}"
                        )
                    raise RuntimeError("project agent session ended before completing its turn")
            usage = TokenUsage(
                input_tokens=session.worker_usage.input_tokens - usage_before.input_tokens,
                output_tokens=session.worker_usage.output_tokens - usage_before.output_tokens,
                calls=session.worker_usage.calls - usage_before.calls,
            )
        finally:
            self._active_event_sink = None
        return AgentProjectRun(answer=answer, events=tuple(events), worker_usage=usage)

    def _ensure_session(self, agent: HarnessDoc, provider: ToolCallingProvider) -> LiveSession:
        """Return the compatible live session, starting one when the harness changed."""
        if (
            self._session is not None
            and not self._session.closed
            and self._session_agent_hash == agent.doc_hash
            and self._session_provider is provider
        ):
            return self._session
        self._close_agent_session()
        channel = self._channel_factory(self._sandbox, self.workspace)
        skills = agent.skills()
        session = LiveSession(
            channel,
            tools=resolve_tools(agent.tools()),
            execute_tool=self._execute_tool,
            on_event=self._emit_session_event,
            files={surface.path: surface.content for surface in agent.code_files() if surface.path},
            system_prompt=agent.assembled_prompt(),
            skill_bodies={skill.name: skill.body for skill in skills},
            provider=provider,
            turn_cap=agent.max_turns(),
        )
        try:
            session.start()
        except Exception:
            close = getattr(channel, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
            raise
        self._channel = channel
        self._session = session
        self._session_agent_hash = agent.doc_hash
        self._session_provider = provider
        return session

    def _emit_session_event(self, event: SessionEvent) -> None:
        """Route session events to the currently active project turn."""
        if self._active_event_sink is not None:
            self._active_event_sink(event)

    def _close_agent_session(self) -> None:
        """Close the current agent session without touching the project filesystem."""
        session = self._session
        channel = self._channel
        self._session = None
        self._channel = None
        self._session_agent_hash = None
        self._session_provider = None
        if session is not None and not session.closed:
            session.end()
            session.pump(timeout=0)
        close = getattr(channel, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()

    def usage(self) -> SandboxUsage:
        """Return this project's sandbox lifetime meter."""
        ended = self._finished_at if self._finished_at is not None else time.monotonic()
        return SandboxUsage(count=1, seconds=max(0.0, ended - self._started_at))

    def close(self) -> None:
        """Kill an owned sandbox and finalize its usage meter."""
        if self._finished_at is not None:
            return
        self._close_agent_session()
        self._finished_at = time.monotonic()
        if self._owns_sandbox:
            with contextlib.suppress(Exception):
                self._sandbox.kill()

    def __enter__(self) -> AgentProject:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _absolute_path(self, path: str) -> str:
        candidate = PurePosixPath(path)
        if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
            raise ValueError(f"expected a relative project path, got {path!r}")
        return f"{self.workspace}/{candidate.as_posix()}"

    def _execute_tool(
        self,
        name: str,
        arguments: JsonObject,
        emit: Callable[[str, str], None],
    ) -> ToolOutcome:
        del emit
        try:
            if name == "read_file":
                path = self._tool_path(str(arguments.get("path", "")))
                return _capped(self._sandbox.files.read(path))
            if name == "write_file":
                path = self._tool_path(str(arguments.get("path", "")))
                self._sandbox.files.write(path, str(arguments.get("content", "")))
                return ToolOutcome(content=f"wrote {path}")
        except Exception as error:  # noqa: BLE001 - tool errors are agent observations
            return ToolOutcome(content=f"{name} failed: {error}", is_error=True)
        return ToolOutcome(content=f"tool {name!r} not available", is_error=True)

    def _tool_path(self, path: str) -> str:
        """Resolve an agent-supplied path while containing it to the project."""
        candidate = PurePosixPath(path)
        workspace = PurePosixPath(self.workspace)
        if candidate.is_absolute():
            try:
                candidate = candidate.relative_to(workspace)
            except ValueError as error:
                raise ValueError(f"path escapes project workspace: {path!r}") from error
        if not candidate.parts or ".." in candidate.parts:
            raise ValueError(f"path escapes project workspace: {path!r}")
        return str(workspace / candidate)


def _start_channel(sandbox: SandboxHandle, workspace: str) -> Channel:
    return start_live_runner(sandbox, workspace=workspace)


def _capped(content: str, *, is_error: bool = False) -> ToolOutcome:
    if len(content) <= _OUTPUT_CAP:
        return ToolOutcome(content=content, is_error=is_error)
    half = _OUTPUT_CAP // 2
    marker = f"\n... {len(content) - _OUTPUT_CAP} characters truncated ...\n"
    return ToolOutcome(
        content=f"{content[:half]}{marker}{content[-half:]}",
        is_error=is_error,
        truncated=True,
    )
