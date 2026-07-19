"""Trusted Harbor agent that evaluates a serialized WMH pi harness."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import re
import shlex
import tempfile
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Final, Protocol, cast

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig
from llm_waterfall import ChatResponse

from wmh.core.types import JsonObject
from wmh.evals.harbor.receipt_trace import validate_provider_receipt_trace
from wmh.harness.doc import HarnessDoc
from wmh.harness.live_session import OutputEmitter, SessionEvent, ToolOutcome
from wmh.harness.pi_local import (
    PI_CONTAINER_IMAGE,
    DockerStdioChannel,
    start_container_live_runner,
    validate_pi_container_image,
)
from wmh.harness.pi_runner import (
    AmbiguousTaskEnvironmentError,
    PiCandidateError,
    PiInfrastructureError,
    PiInfrastructureFailureKind,
    PiRunHealth,
    PiTurnResult,
    TurnDeadline,
    TurnDeadlineExceeded,
    run_pi_turn,
)
from wmh.harness.runner_link import Channel, TokenUsage
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.process_worker import (
    ProviderProcessWorker,
    ProviderWorkerCleanupError,
    ProviderWorkerDeadlineExceeded,
    ProviderWorkerFailure,
    ProviderWorkerUnavailable,
)
from wmh.providers.receipt import validate_chat_provider_receipt
from wmh.tracking.budget import BudgetAccount

_TOOL_OUTPUT_CHARS = 16_000
_TOOL_STREAM_CHARS = _TOOL_OUTPUT_CHARS // 2
_TOOL_COMMAND_BYTES = 32 * 1024
_TOOL_PATH_BYTES = 4 * 1024
_TOOL_CONTENT_BYTES = 64 * 1024
_EXECUTION_CLEANUP_TIMEOUT_S = 30.0
_PROVIDER_WORKER_START_TIMEOUT_S = 30.0
_ENVIRONMENT_ATTESTATION_TIMEOUT_S = 30
_MAX_ENVIRONMENT_ATTESTATION_OUTPUT_BYTES = 64 * 1024
_MAX_ENVIRONMENT_CONTAINERS = 64
_TASK_DISK_HEALTH_TIMEOUT_S = 10
_MIN_TASK_FREE_DISK_KIB = 128 * 1024
_TRACE_FILE = "wmh-events.jsonl"
# Bump whenever trusted host/runtime semantics change. Harbor run identity binds this value, so
# completed artifacts cannot be reused across evaluator behavior changes.
WMH_PI_AGENT_VERSION: Final = "0.5.0"
_TRUNCATION_MARKER = "\n... [output truncated] ...\n"
_TOOL_EXEC_RETAINED_BYTES = _TOOL_OUTPUT_CHARS - len(_TRUNCATION_MARKER.encode())
_TOOL_EXEC_HEAD_BYTES = _TOOL_EXEC_RETAINED_BYTES // 2
_TOOL_EXEC_TAIL_BYTES = _TOOL_EXEC_RETAINED_BYTES - _TOOL_EXEC_HEAD_BYTES
_BOUNDED_EXEC_SCRIPT = f"""\
set -o pipefail
encoded_command=$WMH_TOOL_COMMAND_B64
command_deadline=$WMH_TOOL_DEADLINE_S
command_kill_after=$WMH_TOOL_KILL_AFTER_S
unset WMH_TOOL_COMMAND_B64 WMH_TOOL_DEADLINE_S WMH_TOOL_KILL_AFTER_S BASH_ENV ENV
cap_stream() {{
  local prefix suffix
  prefix=$(head -c {_TOOL_EXEC_HEAD_BYTES})
  suffix=$(tail -c {_TOOL_EXEC_TAIL_BYTES})
  printf '%s' "$prefix"
  if [ -n "$suffix" ]; then
    printf '%s' {shlex.quote(_TRUNCATION_MARKER)}
    printf '%s' "$suffix"
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
_TASK_DISK_HEALTH_SCRIPT = """\
set -o pipefail
unset BASH_ENV ENV
probe=.wmh-evaluator-disk-health-4f8c31f2
if [ -e "$probe" ]; then
  exit 70
fi
mkdir "$probe" || exit 71
rmdir "$probe" || exit 72
LC_ALL=C df -Pk .
"""
_TASK_DISK_HEALTH_COMMAND = (
    f"/bin/bash --noprofile --norc -p -c {shlex.quote(_TASK_DISK_HEALTH_SCRIPT)}"
)


class WmhPiProviderError(RuntimeError):
    """The host-side worker provider failed during a benchmark trial."""


class WmhPiProviderDeadlineError(RuntimeError):
    """The host-side provider exceeded its evaluator-owned hard deadline."""


class WmhPiProviderReceiptError(RuntimeError):
    """Provider evidence was missing or inconsistent with the frozen configuration."""


class WmhPiEnvironmentError(RuntimeError):
    """Harbor's task environment failed while executing a pi tool request."""


class WmhPiEnvironmentConfirmationRequiredError(RuntimeError):
    """Harbor lost the task environment without enough evidence to assign ownership."""


class WmhPiRunnerError(RuntimeError):
    """The isolated pi runner failed outside candidate-controlled execution."""


class WmhPiCleanupError(RuntimeError):
    """Trusted pi execution cleanup could not be proved."""


class _CandidateToolArgumentError(ValueError):
    """Candidate-supplied tool arguments failed a bounded validation rule."""


class _TaskEnvironmentUnavailableError(RuntimeError):
    """The task environment failed a trusted readiness check before candidate execution."""


@dataclass(frozen=True)
class _TaskEnvironmentAttestation:
    digest: str
    evidence: JsonObject


@dataclass
class _EnvironmentCall:
    result: Future[ExecResult]
    closed: threading.Event
    cleanup_failed: threading.Event
    task: asyncio.Task[ExecResult] | None = None
    cancel_requested: bool = False


# Harbor 0.18 exposes no public API for the exact Docker Compose project or the E2B sandbox that
# actually ran. WMH pins Harbor to that exact version and centralizes the two private members in
# these protocol views. Missing members fail agent setup, before any provider request is admitted,
# rather than silently producing an unattested score. Replace these views when Harbor adds a public
# immutable environment-identity API.
class _DockerEnvironmentView(Protocol):
    async def _run_docker_compose_command(
        self,
        command: list[str],
        check: bool = True,
        timeout_sec: int | None = None,
        stdin_data: bytes | None = None,
        on_output: object | None = None,
    ) -> ExecResult: ...


class _E2BSandboxInfoView(Protocol):
    template_id: str
    cpu_count: int
    memory_mb: int
    started_at: datetime


class _E2BSandboxView(Protocol):
    async def get_info(self) -> _E2BSandboxInfoView: ...


class _E2BEnvironmentView(Protocol):
    _sandbox: _E2BSandboxView | None


class HarborToolExecutor:
    """Synchronously expose one Harbor task environment to pi's host-side tool loop."""

    def __init__(
        self,
        event_loop: asyncio.AbstractEventLoop,
        environment: BaseEnvironment,
    ) -> None:
        self._event_loop = event_loop
        self._environment = environment
        self._disk_health_initialized = False
        self._lock = threading.Lock()
        self._current_call: _EnvironmentCall | None = None
        self._cancelled = False

    def cancel(self) -> None:
        """Cancel any active Harbor environment call from the host event loop."""
        with self._lock:
            self._cancelled = True
            call = self._current_call
        if call is not None:
            self._cancel_call(call)

    def wait_closed(self, timeout_s: float) -> bool:
        """Wait until the latest Harbor coroutine has finished cancellation cleanup."""
        with self._lock:
            call = self._current_call
        if call is None:
            return True
        return call.closed.wait(timeout_s) and not call.cleanup_failed.is_set()

    def join_closed(self, timeout_s: float) -> bool:
        """Join the latest Harbor coroutine without abandoning it after the proof timeout."""
        with self._lock:
            call = self._current_call
        if call is None:
            return True
        closed_in_time = call.closed.wait(timeout_s)
        if not closed_in_time:
            call.closed.wait()
        return closed_in_time and not call.cleanup_failed.is_set()

    def __call__(
        self,
        name: str,
        arguments: JsonObject,
        emit: OutputEmitter,
        deadline: TurnDeadline,
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
        deadline: TurnDeadline,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        if not self._disk_health_initialized:
            self._check_task_disk_health(deadline=deadline, after_candidate=False)
            self._disk_health_initialized = True
        remaining_s = deadline.remaining_s()
        environment_timeout_s = math.floor(remaining_s)
        if environment_timeout_s < 1:
            raise TurnDeadlineExceeded
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
        result = self._environment_exec(
            _BOUNDED_EXEC_COMMAND,
            deadline=deadline,
            timeout_s=environment_timeout_s,
            env=bounded_env,
        )
        if _task_environment_resource_loss_requires_confirmation(result):
            raise AmbiguousTaskEnvironmentError
        self._check_task_disk_health(deadline=deadline, after_candidate=True)
        return result

    def _check_task_disk_health(
        self,
        *,
        deadline: TurnDeadline,
        after_candidate: bool,
    ) -> None:
        """Require fixed free space before execution and attribute a post-command loss."""
        remaining_s = math.floor(deadline.remaining_s())
        if remaining_s < 1:
            raise TurnDeadlineExceeded
        result = self._environment_exec(
            _TASK_DISK_HEALTH_COMMAND,
            deadline=deadline,
            timeout_s=min(_TASK_DISK_HEALTH_TIMEOUT_S, remaining_s),
            env={
                "BASH_ENV": "/dev/null",
                "ENV": "/dev/null",
                "BASHOPTS": "",
                "SHELLOPTS": "",
            },
        )
        available_kib = _task_free_disk_kib(result)
        if available_kib is not None and available_kib >= _MIN_TASK_FREE_DISK_KIB:
            return
        if after_candidate:
            # Local Docker tasks share host memory and backing storage with parallel cells.
            # A post-command loss therefore cannot prove candidate ownership without one fresh,
            # same-cell confirmation attempt in an isolated environment.
            raise AmbiguousTaskEnvironmentError
        raise _TaskEnvironmentUnavailableError

    def _environment_exec(
        self,
        command: str,
        *,
        deadline: TurnDeadline,
        timeout_s: int,
        env: dict[str, str],
    ) -> ExecResult:
        """Call Harbor's backend within one deadline and sanitize backend ambiguity."""
        call = self._submit_environment_call(command, timeout_s=timeout_s, env=env)
        try:
            wait_timeout_s = deadline.remaining_s()
            if wait_timeout_s <= 0:
                raise TurnDeadlineExceeded
            return call.result.result(timeout=wait_timeout_s)
        except TimeoutError:
            if deadline.remaining_s() <= 0:
                raise TurnDeadlineExceeded from None
            raise AmbiguousTaskEnvironmentError from None
        except TurnDeadlineExceeded:
            raise
        except Exception:  # noqa: BLE001 - backend exceptions lack one portable base class
            # An API exception cannot prove whether the backend or candidate destroyed the task
            # environment. Keep that boundary retryable instead of silently assigning ownership.
            raise AmbiguousTaskEnvironmentError from None
        finally:
            if not call.closed.is_set():
                self._cancel_call(call)

    def _submit_environment_call(
        self,
        command: str,
        *,
        timeout_s: int,
        env: dict[str, str],
    ) -> _EnvironmentCall:
        call = _EnvironmentCall(
            result=Future(),
            closed=threading.Event(),
            cleanup_failed=threading.Event(),
        )
        with self._lock:
            if self._cancelled:
                call.result.cancel()
                call.closed.set()
                raise AmbiguousTaskEnvironmentError
            self._current_call = call

        def complete(task: asyncio.Task[ExecResult]) -> None:
            try:
                result = task.result()
            except asyncio.CancelledError:
                outcome: ExecResult | BaseException = asyncio.CancelledError()
            except BaseException as exc:  # noqa: BLE001 - forwarded only to the trusted bridge
                outcome = exc
            else:
                outcome = result
            with self._lock:
                cancelled = call.cancel_requested
            if (
                cancelled
                and isinstance(outcome, BaseException)
                and not isinstance(outcome, asyncio.CancelledError)
            ):
                call.cleanup_failed.set()
            call.closed.set()
            if call.result.done():
                return
            if isinstance(outcome, asyncio.CancelledError):
                call.result.cancel()
            elif isinstance(outcome, BaseException):
                call.result.set_exception(outcome)
            else:
                call.result.set_result(outcome)

        def schedule() -> None:
            with self._lock:
                cancelled = call.cancel_requested
            if cancelled:
                call.result.cancel()
                call.closed.set()
                return
            task = self._event_loop.create_task(
                self._environment.exec(command, env=env, timeout_sec=timeout_s)
            )
            task.add_done_callback(complete)
            with self._lock:
                call.task = task
                cancelled = call.cancel_requested
            if cancelled:
                task.cancel()

        try:
            self._event_loop.call_soon_threadsafe(schedule)
        except RuntimeError:
            call.result.cancel()
            call.closed.set()
            raise AmbiguousTaskEnvironmentError from None
        return call

    def _cancel_call(self, call: _EnvironmentCall) -> None:
        with self._lock:
            if call.cancel_requested:
                return
            call.cancel_requested = True
            task = call.task
        call.result.cancel()
        if task is not None:
            try:
                self._event_loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass


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
        budget_account: JsonObject | None = None,
        runner_image: str = PI_CONTAINER_IMAGE,
        turn_timeout_s: float = 300.0,
        require_provider_receipts: bool = False,
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
        config = ProviderConfig.model_validate(provider_config)
        account = (
            BudgetAccount.model_validate(budget_account) if budget_account is not None else None
        )
        if account is not None:
            meter = account.policy.meters[account.meter_id]
            if meter.provider_config != config:
                raise ValueError("budget account provider config must match the Harbor agent")
        if self._harness.runtime_kind() != "pi-node":
            raise ValueError(
                "WMH pi benchmark evaluation requires runtime kind 'pi-node', got "
                f"{self._harness.runtime_kind()!r}"
            )
        validate_pi_container_image(runner_image)
        if not math.isfinite(turn_timeout_s) or turn_timeout_s <= 0:
            raise ValueError("turn_timeout_s must be finite and positive")
        if not isinstance(require_provider_receipts, bool):
            raise ValueError("require_provider_receipts must be a boolean")
        self._provider_config = config.model_copy(deep=True)
        self._budget_account = account.model_copy(deep=True) if account is not None else None
        self._runner_image = runner_image
        self._turn_timeout_s = turn_timeout_s
        self._require_provider_receipts = require_provider_receipts
        self._task_environment_attestation: _TaskEnvironmentAttestation | None = None

    @staticmethod
    def name() -> str:
        return "wmh-pi"

    def version(self) -> str:
        return WMH_PI_AGENT_VERSION

    async def setup(self, environment: BaseEnvironment) -> None:
        """Freeze the immutable environment actually started for this trial."""
        try:
            self._task_environment_attestation = await _attest_task_environment(environment)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - environment details may contain credentials or paths
            raise WmhPiEnvironmentError("task environment attestation failed") from None

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run one pi turn, preserving candidate failures for Harbor's native verifier."""
        attestation = self._task_environment_attestation
        if attestation is None:
            raise WmhPiEnvironmentError("task environment attestation is unavailable")
        identity_metadata = cast(
            "JsonObject",
            {
                "harness_hash": self._harness.execution_hash,
                "runner_image": self._runner_image,
                "task_environment_digest": attestation.digest,
                "task_environment_attestation": attestation.evidence,
                "run_health": PiRunHealth.VALID.value,
            },
        )
        _populate_context(context, TokenUsage(), identity_metadata)
        try:
            self._prepare_trace()
        except Exception:  # noqa: BLE001 - never expose host filesystem details
            _populate_context(
                context,
                TokenUsage(),
                cast(
                    "JsonObject",
                    {
                        **identity_metadata,
                        "infrastructure_failure": True,
                        "run_health": PiRunHealth.INFRASTRUCTURE_FAILURE.value,
                    },
                ),
            )
            raise WmhPiRunnerError("WMH pi trace persistence failed") from None
        event_loop = asyncio.get_running_loop()
        executor = HarborToolExecutor(event_loop, environment)
        runner_factory = LocalContainerRunnerFactory(image=self._runner_image)
        provider_worker = ProviderProcessWorker(
            self._provider_config,
            budget_account=self._budget_account,
        )
        candidate_error: PiCandidateError | None = None
        result: PiTurnResult | None = None
        turn_task = asyncio.create_task(
            asyncio.to_thread(
                _run_isolated_turn,
                provider_worker,
                runner_factory,
                executor,
                self._harness,
                instruction,
                self._turn_timeout_s,
                self._provider_config,
                self._require_provider_receipts,
            )
        )
        try:
            result = await asyncio.shield(turn_task)
        except PiCandidateError as exc:
            candidate_error = exc
            try:
                await _cancel_and_wait_execution(
                    executor,
                    provider_worker,
                    runner_factory,
                    turn_task,
                )
            except WmhPiCleanupError:
                _populate_context(
                    context,
                    exc.worker_usage,
                    cast(
                        "JsonObject",
                        {
                            **identity_metadata,
                            "infrastructure_failure": True,
                            "run_health": PiRunHealth.INFRASTRUCTURE_FAILURE.value,
                        },
                    ),
                )
                raise
        except asyncio.CancelledError:
            await _cancel_and_wait_execution(
                executor,
                provider_worker,
                runner_factory,
                turn_task,
            )
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize every infrastructure failure uniformly
            trace_error: WmhPiRunnerError | None = None
            run_health = PiRunHealth.INFRASTRUCTURE_FAILURE
            if isinstance(exc, PiInfrastructureError):
                run_health = exc.run_health
                metadata = cast(
                    "JsonObject",
                    {
                        **identity_metadata,
                        "infrastructure_failure": True,
                        "infrastructure_failure_kind": exc.kind.value,
                        "run_health": run_health.value,
                    },
                )
                _populate_context(context, exc.worker_usage, metadata)
                try:
                    self._write_trace(exc.events)
                except Exception:  # noqa: BLE001 - retain only a stable trace failure
                    trace_error = WmhPiRunnerError("WMH pi trace persistence failed")
            else:
                _populate_context(
                    context,
                    TokenUsage(),
                    cast(
                        "JsonObject",
                        {
                            **identity_metadata,
                            "infrastructure_failure": True,
                            "run_health": run_health.value,
                        },
                    ),
                )
            # Cleanup proof is independent of evidence persistence and always wins: surviving
            # trusted execution cannot be downgraded to an ordinary trace or provider error.
            await _cancel_and_wait_execution(
                executor,
                provider_worker,
                runner_factory,
                turn_task,
            )
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
                    "run_health": result.run_health.value,
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
                    "candidate_failure_reason": candidate_error.reason.value,
                    "run_health": candidate_error.run_health.value,
                },
            )
        _populate_context(context, usage, metadata)
        try:
            self._write_trace(events)
        except Exception:  # noqa: BLE001 - never admit candidate evidence without its fresh trace
            _populate_context(
                context,
                usage,
                cast(
                    "JsonObject",
                    {
                        **identity_metadata,
                        "infrastructure_failure": True,
                        "run_health": PiRunHealth.INFRASTRUCTURE_FAILURE.value,
                    },
                ),
            )
            raise WmhPiRunnerError("WMH pi trace persistence failed") from None

    def _prepare_trace(self) -> None:
        """Remove deterministic-path evidence from an earlier resumed attempt."""
        trial_dir = self.logs_dir.parent
        if trial_dir.is_symlink():
            raise WmhPiRunnerError("WMH pi trace persistence failed")
        try:
            trial_dir.mkdir(parents=True, exist_ok=True)
            (trial_dir / _TRACE_FILE).unlink(missing_ok=True)
        except OSError:
            raise WmhPiRunnerError("WMH pi trace persistence failed") from None

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


def _run_isolated_turn(
    provider_worker: ProviderProcessWorker,
    runner_factory: LocalContainerRunnerFactory,
    executor: HarborToolExecutor,
    harness: HarnessDoc,
    instruction: str,
    timeout_s: float,
    provider_config: ProviderConfig,
    require_provider_receipts: bool,
) -> PiTurnResult:
    """Own the disposable provider lifecycle for one synchronous pi turn."""
    try:
        provider_worker.start(TurnDeadline.after(_PROVIDER_WORKER_START_TIMEOUT_S))
    except ProviderWorkerDeadlineExceeded:
        raise PiInfrastructureError(PiInfrastructureFailureKind.PROVIDER_DEADLINE) from None
    except (ProviderWorkerFailure, ProviderWorkerUnavailable):
        raise PiInfrastructureError(PiInfrastructureFailureKind.PROVIDER) from None
    try:
        try:
            result = run_pi_turn(
                harness,
                instruction,
                execute_tool=executor,
                worker_fn=provider_worker.complete_chat,
                runner_factory=runner_factory,
                timeout_s=timeout_s,
                response_validator=(
                    partial(
                        _validate_provider_response_receipt,
                        provider_config=provider_config,
                        requested_temperature=harness.temperature(),
                        max_tokens=harness.max_output_tokens(),
                    )
                    if require_provider_receipts
                    else None
                ),
            )
        except (PiCandidateError, PiInfrastructureError) as exc:
            if require_provider_receipts:
                _require_complete_provider_receipt_trace(
                    exc.events,
                    exc.worker_usage,
                    provider_config,
                    requested_temperature=harness.temperature(),
                    max_tokens=harness.max_output_tokens(),
                )
            raise
        if require_provider_receipts:
            _require_complete_provider_receipt_trace(
                result.events,
                result.worker_usage,
                provider_config,
                requested_temperature=harness.temperature(),
                max_tokens=harness.max_output_tokens(),
            )
        return result
    finally:
        provider_worker.close()


def _validate_provider_response_receipt(
    response: ChatResponse,
    provider_config: ProviderConfig,
    *,
    requested_temperature: float,
    max_tokens: int,
) -> None:
    """Require response evidence to agree with the immutable provider target."""
    receipt = response.provider_receipt
    if receipt is None:
        raise ValueError("provider receipt is missing")
    validate_chat_provider_receipt(
        receipt,
        provider_config=provider_config,
        requested_temperature=requested_temperature,
        max_tokens=max_tokens,
    )
    if provider_config.kind is ProviderKind.BEDROCK:
        if response.model != provider_config.model:
            raise ValueError("Bedrock response model disagrees with the frozen provider model")
        return
    if receipt.response_id != response.id:
        raise ValueError("provider receipt response id disagrees with the completion")
    if receipt.response_model != response.model:
        raise ValueError("provider receipt response model disagrees with the completion")
    if receipt.system_fingerprint != response.system_fingerprint:
        raise ValueError("provider receipt fingerprint disagrees with the completion")


def _require_complete_provider_receipt_trace(
    events: tuple[SessionEvent, ...],
    usage: TokenUsage,
    provider_config: ProviderConfig,
    *,
    requested_temperature: float,
    max_tokens: int,
) -> None:
    """Reconcile one receipt event with every successfully metered provider call."""
    try:
        validate_provider_receipt_trace(
            (event.payload for event in events if event.kind == "provider_receipt"),
            expected_calls=usage.calls,
            provider_config=provider_config,
            requested_temperature=requested_temperature,
            max_tokens=max_tokens,
        )
    except Exception:  # noqa: BLE001 - trace contents never enter the persisted error text
        raise PiInfrastructureError(
            PiInfrastructureFailureKind.PROVIDER_RECEIPT,
            events=events,
            worker_usage=usage,
        ) from None


async def _attest_task_environment(
    environment: BaseEnvironment,
) -> _TaskEnvironmentAttestation:
    environment_type = environment.type()
    backend = getattr(environment_type, "value", environment_type)
    if backend == "docker":
        evidence = await _attest_docker_environment(cast("_DockerEnvironmentView", environment))
    elif backend == "e2b":
        evidence = await _attest_e2b_environment(
            environment,
            cast("_E2BEnvironmentView", environment),
        )
    else:
        raise RuntimeError("unsupported task environment backend")
    canonical = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    if len(canonical) > _MAX_ENVIRONMENT_ATTESTATION_OUTPUT_BYTES:
        raise RuntimeError("task environment attestation exceeds its evidence limit")
    return _TaskEnvironmentAttestation(
        digest="sha256:" + hashlib.sha256(canonical).hexdigest(),
        evidence=evidence,
    )


async def _attest_docker_environment(
    environment: _DockerEnvironmentView,
) -> JsonObject:
    result = await environment._run_docker_compose_command(
        ["ps", "--all", "--quiet"],
        timeout_sec=_ENVIRONMENT_ATTESTATION_TIMEOUT_S,
    )
    container_ids = _bounded_lines(result.stdout, label="Docker Compose container identities")
    if not container_ids or len(container_ids) > _MAX_ENVIRONMENT_CONTAINERS:
        raise RuntimeError("Docker Compose returned an invalid container set")
    if len(set(container_ids)) != len(container_ids):
        raise RuntimeError("Docker Compose returned duplicate container identities")
    if any(re.fullmatch(r"[0-9a-f]{12,64}", value) is None for value in container_ids):
        raise RuntimeError("Docker Compose returned an invalid container identity")

    daemon_platform = _single_platform(
        await _run_host_command(
            "docker",
            "info",
            "--format",
            "{{.OSType}}/{{.Architecture}}",
        ),
        label="Docker daemon platform",
    )
    services: list[tuple[str, int, str, str]] = []
    replicas: set[tuple[str, int]] = set()
    for container_id in container_ids:
        identity = await _run_host_command(
            "docker",
            "container",
            "inspect",
            "--format",
            '{{index .Config.Labels "com.docker.compose.service"}}\t'
            '{{index .Config.Labels "com.docker.compose.container-number"}}\t'
            "{{.Image}}\t{{.State.Status}}\t"
            "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
            container_id,
        )
        parts = identity.strip().split("\t")
        if len(parts) != 5:
            raise RuntimeError("Docker returned malformed container identity evidence")
        service, raw_replica, image_id, status, health = parts
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", service) is None:
            raise RuntimeError("Docker returned an invalid Compose service identity")
        if re.fullmatch(r"[1-9][0-9]{0,5}", raw_replica) is None:
            raise RuntimeError("Docker returned an invalid Compose replica identity")
        replica = int(raw_replica)
        replica_key = (service, replica)
        if replica_key in replicas:
            raise RuntimeError("Docker returned a duplicate Compose replica identity")
        replicas.add(replica_key)
        if re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
            raise RuntimeError("Docker returned a mutable or invalid image identity")
        if status != "running" or health not in {"none", "healthy"}:
            raise RuntimeError("Docker Compose service is not running and healthy")
        image_platform = _single_platform(
            await _run_host_command(
                "docker",
                "image",
                "inspect",
                "--format",
                "{{.Os}}/{{.Architecture}}",
                image_id,
            ),
            label="Docker image platform",
        )
        services.append((service, replica, image_id, image_platform))

    return cast(
        "JsonObject",
        {
            "schema_version": 1,
            "backend": "docker",
            "daemon_platform": daemon_platform,
            "services": [
                {
                    "service": service,
                    "replica": replica,
                    "image_id": image_id,
                    "image_platform": image_platform,
                }
                for service, replica, image_id, image_platform in sorted(services)
            ],
        },
    )


async def _attest_e2b_environment(
    environment: BaseEnvironment,
    view: _E2BEnvironmentView,
) -> JsonObject:
    sandbox = view._sandbox
    if sandbox is None:
        raise RuntimeError("E2B sandbox is unavailable")
    info = await sandbox.get_info()
    template_id = _bounded_identity(info.template_id, label="E2B template identity")
    cpu_count = info.cpu_count
    memory_mb = info.memory_mb
    if (
        isinstance(cpu_count, bool)
        or not isinstance(cpu_count, int)
        or cpu_count < 1
        or isinstance(memory_mb, bool)
        or not isinstance(memory_mb, int)
        or memory_mb < 1
    ):
        raise RuntimeError("E2B returned invalid sandbox resource evidence")
    tags = await _get_e2b_template_tags(template_id)
    default_tags = [
        (build_id, created_at) for tag, build_id, created_at in tags if tag == "default"
    ]
    if len(default_tags) != 1:
        raise RuntimeError("E2B template default tag did not resolve to one immutable build")
    raw_build_id, tag_created_at = default_tags[0]
    if (
        not isinstance(info.started_at, datetime)
        or info.started_at.tzinfo is None
        or not isinstance(tag_created_at, datetime)
        or tag_created_at.tzinfo is None
        or tag_created_at > info.started_at
    ):
        raise RuntimeError("E2B template tag changed after the sandbox started")
    build_id = _bounded_identity(raw_build_id, label="E2B build identity")
    platform_result = await environment.exec(
        "/bin/uname -s && /bin/uname -m",
        timeout_sec=_ENVIRONMENT_ATTESTATION_TIMEOUT_S,
    )
    if platform_result.return_code != 0:
        raise RuntimeError("E2B sandbox platform attestation failed")
    platform_lines = _bounded_lines(
        platform_result.stdout,
        label="E2B sandbox platform",
    )
    if len(platform_lines) != 2:
        raise RuntimeError("E2B returned malformed sandbox platform evidence")
    platform = "/".join(part.lower() for part in platform_lines)
    if re.fullmatch(r"[a-z0-9_.-]{1,64}/[a-z0-9_.-]{1,64}", platform) is None:
        raise RuntimeError("E2B returned invalid sandbox platform evidence")
    return cast(
        "JsonObject",
        {
            "schema_version": 1,
            "backend": "e2b",
            "template_id": template_id,
            "build_id": build_id,
            "platform": platform,
            "cpu_count": cpu_count,
            "memory_mb": memory_mb,
        },
    )


async def _get_e2b_template_tags(
    template_id: str,
) -> tuple[tuple[str, str, datetime], ...]:
    from e2b import AsyncTemplate

    tags = await AsyncTemplate.get_tags(template_id)
    return tuple((tag.tag, tag.build_id, tag.created_at) for tag in tags)


async def _run_host_command(*command: str) -> str:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=_ENVIRONMENT_ATTESTATION_TIMEOUT_S,
        )
    except TimeoutError:
        await _kill_and_wait_process(process)
        raise RuntimeError("task environment host attestation command timed out") from None
    except asyncio.CancelledError:
        await _kill_and_wait_process(process)
        raise
    if process.returncode != 0:
        raise RuntimeError("task environment host attestation command failed")
    if len(stdout) > _MAX_ENVIRONMENT_ATTESTATION_OUTPUT_BYTES:
        raise RuntimeError("task environment host attestation output exceeds its limit")
    try:
        return stdout.decode("utf-8")
    except UnicodeDecodeError:
        raise RuntimeError("task environment host attestation output is not UTF-8") from None


async def _kill_and_wait_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    cleanup = asyncio.create_task(process.wait())
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            continue
    await cleanup


def _bounded_lines(value: str | None, *, label: str) -> list[str]:
    if value is None:
        return []
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise RuntimeError(f"{label} is not valid UTF-8") from None
    if len(encoded) > _MAX_ENVIRONMENT_ATTESTATION_OUTPUT_BYTES:
        raise RuntimeError(f"{label} exceeds its evidence limit")
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if any(len(line.encode()) > 512 for line in lines):
        raise RuntimeError(f"{label} contains an oversized value")
    return lines


def _single_platform(value: str, *, label: str) -> str:
    lines = _bounded_lines(value, label=label)
    if len(lines) != 1:
        raise RuntimeError(f"{label} must contain one value")
    platform = lines[0].lower()
    if re.fullmatch(r"[a-z0-9_.-]{1,64}/[a-z0-9_.-]{1,64}", platform) is None:
        raise RuntimeError(f"{label} is invalid")
    return platform


def _bounded_identity(value: object, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.:-]{1,512}", value) is None:
        raise RuntimeError(f"{label} is invalid")
    return value


async def _cancel_and_wait_execution(
    executor: HarborToolExecutor,
    provider_worker: ProviderProcessWorker,
    runner_factory: LocalContainerRunnerFactory,
    turn_task: asyncio.Task[PiTurnResult],
) -> None:
    """Cancel all trusted execution and do not abandon its threads or subprocesses."""
    cleanup = asyncio.create_task(
        _prove_execution_cleanup(executor, provider_worker, runner_factory, turn_task)
    )
    cancelled_during_cleanup = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            # Repeated host cancellation cannot interrupt evaluator cleanup proof.
            cancelled_during_cleanup = True
            continue
    if not cleanup.result():
        raise WmhPiCleanupError("WMH pi execution cleanup was not proved") from None
    if cancelled_during_cleanup:
        raise asyncio.CancelledError


async def _prove_execution_cleanup(
    executor: HarborToolExecutor,
    provider_worker: ProviderProcessWorker,
    runner_factory: LocalContainerRunnerFactory,
    turn_task: asyncio.Task[PiTurnResult],
) -> bool:
    cleanup_proved = True
    try:
        executor.cancel()
    except BaseException:  # noqa: BLE001 - cleanup must continue across every component
        cleanup_proved = False

    provider_cancelled, runner_cancelled = await asyncio.gather(
        _run_cleanup_call(provider_worker.cancel),
        _run_cleanup_call(runner_factory.cancel),
    )
    cleanup_proved = cleanup_proved and provider_cancelled and runner_cancelled

    environment_closed, provider_closed, runner_closed = await asyncio.gather(
        _run_cleanup_call(
            lambda: executor.join_closed(_EXECUTION_CLEANUP_TIMEOUT_S),
            require_truthy=True,
        ),
        _run_cleanup_call(
            lambda: provider_worker.wait_closed(_EXECUTION_CLEANUP_TIMEOUT_S),
            require_truthy=True,
        ),
        _run_cleanup_call(
            lambda: runner_factory.wait_closed(_EXECUTION_CLEANUP_TIMEOUT_S),
            require_truthy=True,
        ),
    )
    cleanup_proved = cleanup_proved and environment_closed and provider_closed and runner_closed
    turn_closed = await _join_turn_task(turn_task)
    return cleanup_proved and turn_closed


async def _run_cleanup_call(
    operation: Callable[[], bool | None],
    *,
    require_truthy: bool = False,
) -> bool:
    """Run and join cleanup without depending on the possibly saturated turn executor."""
    completed = threading.Event()
    failed = threading.Event()
    results: list[bool | None] = []

    def run() -> None:
        try:
            results.append(operation())
        except BaseException:  # noqa: BLE001 - publish only one stable cleanup error
            failed.set()
        finally:
            completed.set()

    thread = threading.Thread(target=run, name="wmh-execution-cleanup")
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    try:
        thread.start()
    except RuntimeError:
        return False
    while not completed.is_set():
        await asyncio.sleep(0.01)
    elapsed_s = loop.time() - started_at
    thread.join()
    if failed.is_set() or len(results) != 1:
        return False
    if require_truthy:
        return bool(results[0])
    return elapsed_s <= _EXECUTION_CLEANUP_TIMEOUT_S


async def _join_turn_task(turn_task: asyncio.Task[PiTurnResult]) -> bool:
    """Consume one turn task and prove its backing thread exited, even after timeout."""
    completed_in_time = True
    try:
        await asyncio.wait_for(
            asyncio.shield(turn_task),
            timeout=_EXECUTION_CLEANUP_TIMEOUT_S,
        )
    except TimeoutError:
        completed_in_time = False
    except BaseException:  # noqa: BLE001 - task outcome is handled by the caller
        pass
    while not turn_task.done():
        try:
            await asyncio.shield(turn_task)
        except asyncio.CancelledError:
            continue
        except BaseException:  # noqa: BLE001 - task is joined, not reclassified here
            break
    try:
        turn_task.result()
    except BaseException:  # noqa: BLE001 - consume the joined task outcome
        pass
    return completed_in_time


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
    bounded_stdout, stdout_truncated = _bounded_text(stdout, _TOOL_STREAM_CHARS)
    bounded_stderr, stderr_truncated = _bounded_text(stderr, _TOOL_STREAM_CHARS)
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


def _task_environment_resource_loss_requires_confirmation(result: ExecResult) -> bool:
    """Return whether raw resource-loss evidence needs a fresh same-cell confirmation.

    Exit 137 is ambiguous between an evaluator-owned hard deadline, a container OOM, and host
    pressure. Local Docker cells also share backing storage, so ENOSPC cannot prove candidate
    ownership. Candidate-authored strings can trigger only a retryable confirmation, never a
    score or an infrastructure-cost exemption.
    """
    if result.return_code in {-9, 137}:
        return True
    output = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
    return "no space left on device" in output or "disk quota exceeded" in output


def _task_free_disk_kib(result: ExecResult) -> int | None:
    """Parse the POSIX ``df -Pk`` record emitted by the task-disk health command."""
    if result.return_code != 0 or result.stderr not in {None, ""}:
        return None
    lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
    if len(lines) != 2:
        return None
    header = lines[0].split()
    fields = lines[1].split()
    if len(header) < 6 or "Available" not in header or len(fields) < 6:
        return None
    value = fields[3]
    if not value.isascii() or not value.isdecimal():
        return None
    return int(value)


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
    context.metadata = {**metadata, "model_calls": usage.calls}


def _typed_infrastructure_error(error: Exception) -> RuntimeError:
    """Classify and sanitize a failure before Harbor persists its exception text."""
    if isinstance(error, ProviderWorkerCleanupError):
        return WmhPiCleanupError("WMH pi execution cleanup was not proved")
    if isinstance(error, PiInfrastructureError):
        if error.kind is PiInfrastructureFailureKind.PROVIDER:
            return WmhPiProviderError("WMH pi worker provider failed")
        if error.kind is PiInfrastructureFailureKind.PROVIDER_DEADLINE:
            return WmhPiProviderDeadlineError("WMH pi worker provider deadline expired")
        if error.kind is PiInfrastructureFailureKind.PROVIDER_RECEIPT:
            return WmhPiProviderReceiptError("WMH pi worker provider receipt is invalid")
        if error.kind is PiInfrastructureFailureKind.TASK_ENVIRONMENT_CONFIRMATION_REQUIRED:
            return WmhPiEnvironmentConfirmationRequiredError(
                "WMH pi task environment needs a fresh confirmation attempt"
            )
        return WmhPiEnvironmentError("WMH pi task environment failed")
    message = str(error)
    if message.startswith("pi turn worker provider failed"):
        return WmhPiProviderError("WMH pi worker provider failed")
    if message.startswith("pi turn tool executor failed"):
        return WmhPiEnvironmentError("WMH pi task environment failed")
    if "cleanup" in message.lower():
        return WmhPiCleanupError("WMH pi execution cleanup was not proved")
    return WmhPiRunnerError(f"WMH pi runner infrastructure failed ({type(error).__name__})")
