"""Trusted Harbor agent that evaluates a serialized WMH pi harness."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import shlex
import tempfile
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path, PurePosixPath
from typing import Final, cast

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig

from wmh.core.types import JsonObject
from wmh.harness.doc import HarnessDoc
from wmh.harness.live_session import OutputEmitter, SessionEvent, ToolOutcome
from wmh.harness.pi_local import (
    PI_CONTAINER_IMAGE,
    DockerStdioChannel,
    start_container_live_runner,
    validate_pi_container_image,
)
from wmh.harness.pi_runner import (
    PiCandidateError,
    PiInfrastructureError,
    PiInfrastructureFailureKind,
    PiTurnResult,
    ToolExecutionDeadline,
    ToolExecutionDeadlineExceeded,
    run_pi_turn,
)
from wmh.harness.runner_link import Channel, TokenUsage
from wmh.providers.base import ProviderConfig, ToolCallingProvider
from wmh.providers.registry import get_provider

_TOOL_OUTPUT_CHARS = 16_000
_TOOL_STREAM_BYTES = _TOOL_OUTPUT_CHARS // 2
_TOOL_COMMAND_BYTES = 32 * 1024
_TOOL_PATH_BYTES = 4 * 1024
_TOOL_CONTENT_BYTES = 64 * 1024
_RUNNER_CLEANUP_TIMEOUT_S = 30.0
_TRACE_FILE = "wmh-events.jsonl"
WMH_PI_AGENT_VERSION: Final = "0.1.0"
_TRUNCATION_MARKER = "\n... [output truncated] ...\n"
_TOOL_EXEC_HEAD_BYTES = _TOOL_OUTPUT_CHARS - len(_TRUNCATION_MARKER.encode())
_BOUNDED_EXEC_SCRIPT = f"""\
set -o pipefail
encoded_command=$WMH_TOOL_COMMAND_B64
command_deadline=$WMH_TOOL_DEADLINE_S
command_kill_after=$WMH_TOOL_KILL_AFTER_S
unset WMH_TOOL_COMMAND_B64 WMH_TOOL_DEADLINE_S WMH_TOOL_KILL_AFTER_S BASH_ENV ENV
cap_stream() {{
  local prefix remainder
  prefix=$(head -c {_TOOL_EXEC_HEAD_BYTES})
  remainder=$(wc -c)
  printf '%s' "$prefix"
  if [ "$remainder" -gt 0 ]; then
    printf '%s' {shlex.quote(_TRUNCATION_MARKER)}
  fi
}}
(
    printf '%s' "$encoded_command" |
    base64 -d |
    timeout --signal=TERM --kill-after="${{command_kill_after}}s" "${{command_deadline}}s" \
      /bin/bash --noprofile --norc -p
) 2>&1 | cap_stream
status=${{PIPESTATUS[0]}}
exit "$status"
"""
_BOUNDED_EXEC_COMMAND = f"/bin/bash --noprofile --norc -p -c {shlex.quote(_BOUNDED_EXEC_SCRIPT)}"


class WmhPiProviderError(RuntimeError):
    """The host-side worker provider failed during a benchmark trial."""


class WmhPiEnvironmentError(RuntimeError):
    """Harbor's task environment failed while executing a pi tool request."""


class WmhPiRunnerError(RuntimeError):
    """The isolated pi runner failed outside candidate-controlled execution."""


class WmhPiCleanupError(RuntimeError):
    """Candidate runner cleanup could not be proved."""


class _CandidateToolArgumentError(ValueError):
    """Candidate-supplied tool arguments failed a bounded validation rule."""


class HarborToolExecutor:
    """Synchronously expose one Harbor task environment to pi's host-side tool loop."""

    def __init__(
        self,
        event_loop: asyncio.AbstractEventLoop,
        environment: BaseEnvironment,
    ) -> None:
        self._event_loop = event_loop
        self._environment = environment

    def __call__(
        self,
        name: str,
        arguments: JsonObject,
        emit: OutputEmitter,
        deadline: ToolExecutionDeadline,
    ) -> ToolOutcome:
        """Execute a supported task tool and preserve command failures as observations."""
        try:
            if name == "bash":
                command = _tool_string_argument(
                    arguments,
                    "command",
                    max_bytes=_TOOL_COMMAND_BYTES,
                    allow_empty=False,
                    allow_null=False,
                )
                result = self._exec(command, deadline=deadline)
                return _command_outcome(result, emit)
            if name == "read_file":
                path = _task_path(arguments)
                result = self._exec(f"cat -- {shlex.quote(path)}", deadline=deadline)
                return _command_outcome(result, emit)
            if name == "write_file":
                path = _task_path(arguments)
                content = _tool_string_argument(
                    arguments,
                    "content",
                    max_bytes=_TOOL_CONTENT_BYTES,
                    allow_empty=True,
                    allow_null=True,
                )
                parent = str(PurePosixPath(path).parent)
                encoded = base64.b64encode(content.encode()).decode()
                command = (
                    f"mkdir -p -- {shlex.quote(parent)} && "
                    f"printf '%s' \"$WMH_FILE_CONTENT_B64\" | base64 -d > {shlex.quote(path)}"
                )
                result = self._exec(
                    command,
                    deadline=deadline,
                    env={"WMH_FILE_CONTENT_B64": encoded},
                )
                outcome = _command_outcome(result, emit)
                if not outcome.is_error:
                    return ToolOutcome(content=f"wrote {path}")
                return outcome
        except _CandidateToolArgumentError as exc:
            return _capped(f"invalid {name} arguments: {exc}", is_error=True)
        return ToolOutcome(content=f"tool {name!r} not available", is_error=True)

    def _exec(
        self,
        command: str,
        *,
        deadline: ToolExecutionDeadline,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        remaining_s = deadline.remaining_s()
        environment_timeout_s = math.floor(remaining_s)
        if environment_timeout_s < 1:
            raise ToolExecutionDeadlineExceeded
        timeout_headroom_s = min(
            30.0,
            max(0.1, environment_timeout_s / 10),
            environment_timeout_s / 2,
        )
        candidate_timeout_s = environment_timeout_s - timeout_headroom_s
        kill_after_s = min(5.0, timeout_headroom_s / 2)
        bounded_env = dict(env or {})
        # Harbor merges the task's persistent environment into every exec. Bash
        # evaluates BASH_ENV and imported option variables before the first command
        # in a non-interactive shell, so clearing them inside the script would be
        # too late. Privileged mode also refuses exported shell functions.
        bounded_env["BASH_ENV"] = "/dev/null"
        bounded_env["ENV"] = "/dev/null"
        bounded_env["BASHOPTS"] = ""
        bounded_env["SHELLOPTS"] = ""
        bounded_env["WMH_TOOL_COMMAND_B64"] = base64.b64encode(command.encode()).decode()
        bounded_env["WMH_TOOL_DEADLINE_S"] = f"{candidate_timeout_s:.6f}"
        bounded_env["WMH_TOOL_KILL_AFTER_S"] = f"{kill_after_s:.6f}"
        future = asyncio.run_coroutine_threadsafe(
            self._environment.exec(
                _BOUNDED_EXEC_COMMAND,
                env=bounded_env,
                timeout_sec=environment_timeout_s,
            ),
            self._event_loop,
        )
        try:
            wait_timeout_s = deadline.remaining_s()
            if wait_timeout_s <= 0:
                raise ToolExecutionDeadlineExceeded
            return future.result(timeout=wait_timeout_s)
        except TimeoutError:
            if deadline.remaining_s() <= 0:
                raise ToolExecutionDeadlineExceeded from None
            raise
        finally:
            if not future.done():
                future.cancel()


class LocalContainerRunnerFactory:
    """Open one isolated local pi container and support fail-closed cancellation."""

    def __init__(self, *, image: str = PI_CONTAINER_IMAGE) -> None:
        self._image = image
        self._lock = threading.Lock()
        self._active: DockerStdioChannel | None = None
        self._cancelled = False
        self._opening = False
        self._closed = threading.Event()

    def __call__(self) -> AbstractContextManager[Channel]:
        return self._open()

    @contextmanager
    def _open(self) -> Iterator[Channel]:
        with self._lock:
            self._opening = True
        try:
            channel = cast("DockerStdioChannel", start_container_live_runner(image=self._image))
        except BaseException:
            with self._lock:
                self._opening = False
            self._closed.set()
            raise
        with self._lock:
            self._opening = False
            cancelled = self._cancelled
            if not cancelled:
                self._active = channel
        if cancelled:
            try:
                channel.close()
            finally:
                self._closed.set()
            raise RuntimeError("pi runner was cancelled before startup completed")
        try:
            yield channel
        finally:
            try:
                channel.close()
            finally:
                with self._lock:
                    self._active = None
                self._closed.set()

    def cancel(self) -> None:
        """Stop an active runner, or arrange to stop one still being created."""
        with self._lock:
            self._cancelled = True
            channel = self._active
            opening = self._opening
        if channel is None and not opening:
            self._closed.set()
        if channel is not None:
            channel.close()

    def wait_closed(self, timeout_s: float) -> bool:
        """Wait until the runner context proves cleanup."""
        return self._closed.wait(timeout_s)


class WmhPiAgent(BaseAgent):
    """Run serialized pi harness source while Harbor owns the task and verifier lifecycle."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        skills_dir: str | None = None,
        *,
        extra_env: dict[str, str] | None = None,
        harness: JsonObject,
        provider_config: JsonObject,
        runner_image: str = PI_CONTAINER_IMAGE,
        turn_timeout_s: float = 300.0,
    ) -> None:
        if extra_env:
            raise ValueError(
                "WMH pi evaluation does not inject agent environment variables into tasks"
            )
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            logger=logger,
            mcp_servers=mcp_servers,
            skills_dir=skills_dir,
            extra_env=extra_env,
        )
        self._harness = HarnessDoc.model_validate(harness)
        provider = get_provider(ProviderConfig.model_validate(provider_config))
        if not isinstance(provider, ToolCallingProvider):
            raise TypeError("WMH pi benchmark evaluation needs a ToolCallingProvider")
        if self._harness.runtime_kind() != "pi-node":
            raise ValueError(
                "WMH pi benchmark evaluation requires runtime kind 'pi-node', got "
                f"{self._harness.runtime_kind()!r}"
            )
        validate_pi_container_image(runner_image)
        if not math.isfinite(turn_timeout_s) or turn_timeout_s <= 0:
            raise ValueError("turn_timeout_s must be finite and positive")
        self._provider = provider
        self._runner_image = runner_image
        self._turn_timeout_s = turn_timeout_s

    @staticmethod
    def name() -> str:
        return "wmh-pi"

    def version(self) -> str:
        return WMH_PI_AGENT_VERSION

    async def setup(self, environment: BaseEnvironment) -> None:
        """No task-side setup is required because pi runs in its own isolated container."""
        _ = environment

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run one pi turn, preserving candidate failures for Harbor's native verifier."""
        identity_metadata = cast(
            "JsonObject",
            {
                "harness_hash": self._harness.execution_hash,
                "runner_image": self._runner_image,
            },
        )
        _populate_context(context, TokenUsage(), identity_metadata)
        event_loop = asyncio.get_running_loop()
        executor = HarborToolExecutor(event_loop, environment)
        runner_factory = LocalContainerRunnerFactory(image=self._runner_image)
        candidate_error: PiCandidateError | None = None
        result: PiTurnResult | None = None
        try:
            result = await asyncio.to_thread(
                run_pi_turn,
                self._harness,
                instruction,
                execute_tool=executor,
                provider=self._provider,
                runner_factory=runner_factory,
                timeout_s=self._turn_timeout_s,
            )
        except PiCandidateError as exc:
            candidate_error = exc
        except asyncio.CancelledError:
            await _cancel_and_wait_runner(runner_factory)
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize every infrastructure failure uniformly
            trace_error: WmhPiRunnerError | None = None
            if isinstance(exc, PiInfrastructureError):
                metadata = cast(
                    "JsonObject",
                    {
                        **identity_metadata,
                        "infrastructure_failure": True,
                        "infrastructure_failure_kind": exc.kind.value,
                    },
                )
                _populate_context(context, exc.worker_usage, metadata)
                try:
                    self._write_trace(exc.events)
                except Exception:  # noqa: BLE001 - retain only a stable trace failure
                    trace_error = WmhPiRunnerError("WMH pi trace persistence failed")
            # Cleanup proof is independent of evidence persistence and always wins: a
            # surviving runner cannot be downgraded to an ordinary trace/provider error.
            await _cancel_and_wait_runner(runner_factory)
            if trace_error is not None:
                raise trace_error from None
            raise _typed_infrastructure_error(exc) from None

        if result is not None:
            events = result.events
            usage = result.worker_usage
            metadata = cast(
                "JsonObject",
                {
                    **identity_metadata,
                    "terminal_reason": result.terminal_reason,
                    "candidate_failure": False,
                },
            )
        else:
            assert candidate_error is not None
            events = candidate_error.events
            usage = candidate_error.worker_usage
            metadata = cast(
                "JsonObject",
                {
                    **identity_metadata,
                    "candidate_failure": True,
                    "candidate_failure_stage": candidate_error.stage.value,
                },
            )
        _populate_context(context, usage, metadata)
        self._write_trace(events)

    def _write_trace(self, events: tuple[SessionEvent, ...]) -> None:
        """Atomically write trace evidence outside task-mounted log directories."""
        trial_dir = self.logs_dir.parent
        if trial_dir.is_symlink():
            raise WmhPiRunnerError("WMH pi trace persistence failed")
        trial_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trial_dir / _TRACE_FILE
        lines = [
            json.dumps(
                {"kind": event.kind, "payload": event.payload},
                sort_keys=True,
                separators=(",", ":"),
            )
            for event in events
        ]
        payload = ("\n".join(lines) + ("\n" if lines else "")).encode()
        try:
            descriptor, temporary = tempfile.mkstemp(prefix=".wmh-events-", dir=trial_dir)
            temporary_path = Path(temporary)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, trace_path)
            finally:
                temporary_path.unlink(missing_ok=True)
        except OSError:
            raise WmhPiRunnerError("WMH pi trace persistence failed") from None


async def _cancel_and_wait_runner(runner_factory: LocalContainerRunnerFactory) -> None:
    """Attempt cancellation and wait independently, then publish only sanitized failure text."""
    cleanup_failed = False
    try:
        await asyncio.wait_for(
            asyncio.to_thread(runner_factory.cancel),
            timeout=_RUNNER_CLEANUP_TIMEOUT_S,
        )
    except (Exception, asyncio.CancelledError):  # noqa: BLE001
        # Cancellation, timeout, and close errors all require independent cleanup proof.
        cleanup_failed = True
    try:
        closed = await asyncio.wait_for(
            asyncio.to_thread(
                runner_factory.wait_closed,
                _RUNNER_CLEANUP_TIMEOUT_S,
            ),
            timeout=_RUNNER_CLEANUP_TIMEOUT_S + 1.0,
        )
    except (Exception, asyncio.CancelledError):  # noqa: BLE001
        # A second cancellation must not expose a partial cleanup as success.
        cleanup_failed = True
        closed = False
    if cleanup_failed or not closed:
        raise WmhPiCleanupError("WMH pi runner cleanup was not proved") from None


def _tool_string_argument(
    arguments: JsonObject,
    name: str,
    *,
    max_bytes: int,
    allow_empty: bool,
    allow_null: bool,
) -> str:
    """Return one bounded candidate string or raise a safe observation-only error."""
    value = arguments.get(name)
    if not isinstance(value, str):
        raise _CandidateToolArgumentError(f"{name} must be a string")
    if not allow_empty and not value:
        raise _CandidateToolArgumentError(f"{name} must not be empty")
    if not allow_null and "\x00" in value:
        raise _CandidateToolArgumentError(f"{name} must not contain null bytes")
    try:
        encoded = value.encode()
    except UnicodeEncodeError:
        raise _CandidateToolArgumentError(f"{name} must contain valid UTF-8 text") from None
    if len(encoded) > max_bytes:
        raise _CandidateToolArgumentError(f"{name} exceeds the {max_bytes}-byte limit")
    return value


def _task_path(arguments: JsonObject) -> str:
    return _tool_string_argument(
        arguments,
        "path",
        max_bytes=_TOOL_PATH_BYTES,
        allow_empty=False,
        allow_null=False,
    )


def _command_outcome(result: ExecResult, emit: OutputEmitter) -> ToolOutcome:
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    bounded_stdout, stdout_truncated = _bounded_text(stdout, _TOOL_STREAM_BYTES)
    bounded_stderr, stderr_truncated = _bounded_text(stderr, _TOOL_STREAM_BYTES)
    if bounded_stdout:
        emit("stdout", bounded_stdout)
    if bounded_stderr:
        emit("stderr", bounded_stderr)
    body = bounded_stdout + bounded_stderr
    if result.return_code != 0:
        body = f"{body}\n[exit {result.return_code}]"
    return _capped(
        body,
        is_error=result.return_code != 0,
        force_truncated=stdout_truncated or stderr_truncated,
    )


def _bounded_text(content: str, limit: int) -> tuple[str, bool]:
    """Return at most ``limit`` characters with a fixed-size truncation marker."""
    if len(content) <= limit:
        return content, False
    room = limit - len(_TRUNCATION_MARKER)
    head = room // 2
    tail = room - head
    return f"{content[:head]}{_TRUNCATION_MARKER}{content[-tail:]}", True


def _capped(
    content: str,
    *,
    is_error: bool,
    force_truncated: bool = False,
) -> ToolOutcome:
    text, truncated = _bounded_text(content, _TOOL_OUTPUT_CHARS)
    return ToolOutcome(
        content=text,
        is_error=is_error,
        truncated=force_truncated or truncated,
    )


def _populate_context(
    context: AgentContext,
    usage: TokenUsage,
    metadata: JsonObject,
) -> None:
    context.n_input_tokens = usage.input_tokens
    context.n_output_tokens = usage.output_tokens
    context.metadata = dict(metadata)


def _typed_infrastructure_error(error: Exception) -> RuntimeError:
    """Classify and sanitize a failure before Harbor persists its exception text."""
    if isinstance(error, PiInfrastructureError):
        if error.kind is PiInfrastructureFailureKind.PROVIDER:
            return WmhPiProviderError("WMH pi worker provider failed")
        return WmhPiEnvironmentError("WMH pi task environment failed")
    message = str(error)
    if message.startswith("pi turn worker provider failed"):
        return WmhPiProviderError("WMH pi worker provider failed")
    if message.startswith("pi turn tool executor failed"):
        return WmhPiEnvironmentError("WMH pi task environment failed")
    if "cleanup" in message.lower():
        return WmhPiCleanupError("WMH pi runner cleanup was not proved")
    return WmhPiRunnerError(f"WMH pi runner infrastructure failed ({type(error).__name__})")
