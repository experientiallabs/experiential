"""Harbor agent bridge for running an exact WMH harness document.

Harbor instantiates `WmhHarborAgent` from an import path plus JSON kwargs, so one serialized
candidate (the `HarnessDoc`), one provider config, and the execution knobs travel through
harbor's own trial machinery unchanged. The agent rebuilds the runtime host-side and drives it
against `HarborAgentEnvironment`, a synchronous adapter over harbor's async task environment —
the real container is the environment; the worker LLM and tool routing stay host-side.

Two properties here are load-bearing for the optimizer:

- The WMH transcript (`wmh-run.json` in harbor's logs_dir) is written in a ``finally``, so a
  trial cancelled by harbor's agent timeout still leaves whatever steps executed plus a
  ``stop_reason: "cancelled-by-harbor-timeout"`` marker. Timeout trials are the most informative
  failures a proposer sees; losing their transcripts would blind it.
- Episodes and cleanup run on a dedicated process-wide `ThreadPoolExecutor` sized at least as
  large as agent concurrency, never on ``asyncio.to_thread``'s default executor: with high agent
  concurrency the default pool (min(32, cpus + 4)) would queue episodes whose harbor timeout
  clocks are already running.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import shlex
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig

from wmh.core.types import Action, ActionKind, JsonObject, Observation
from wmh.harness.doc import HarnessDoc
from wmh.harness.environment import is_env_action
from wmh.harness.runtime import (
    DEFAULT_EVAL_EPISODE_TIMEOUT_S,
    RunResult,
    validate_episode_timeout_s,
)
from wmh.providers.base import ProviderConfig
from wmh.providers.registry import get_provider
from wmh.providers.retry import wrap_provider_with_retries

WMH_HARBOR_AGENT_VERSION = "1"
WMH_HARBOR_AGENT_IMPORT_PATH = "wmh.evals.harbor.agent:WmhHarborAgent"
# Keep every task command finite and leave cleanup headroom beneath the local Pi process cap. The
# local shim also joins active request handlers, so a late-starting command cannot outlive runtime
# return; E2B Pi already blocks synchronously on the same bridge.
MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC = 240
# Default size of the dedicated episode executor. It must stay >= the harbor agent concurrency
# plus in-flight cleanup, or queued episodes burn their harbor timeout budget before starting.
DEFAULT_EPISODE_WORKERS = 64
_TRACE_FILENAME = "wmh-run.json"
_CANCELLED_STOP_REASON = "cancelled-by-harbor-timeout"
_WRITE_COMMAND = (
    'mkdir -p -- "$(dirname -- "$WMH_FILE_PATH")" && '
    'printf \'%s\' "$WMH_FILE_CONTENT_B64" | base64 -d > "$WMH_FILE_PATH"'
)

_EXECUTOR_LOCK = threading.Lock()
_EPISODE_EXECUTOR: ThreadPoolExecutor | None = None
_EPISODE_EXECUTOR_WORKERS = 0


def _episode_executor(min_workers: int) -> ThreadPoolExecutor:
    """The process-wide executor for episode and cleanup work, grown to `min_workers`.

    One executor serves every concurrently running WmhHarborAgent in this process. When a job
    asks for more workers than the current pool has, a larger pool replaces it; the old pool
    keeps draining its in-flight work and is dropped without shutdown (process-lifetime infra).
    """
    global _EPISODE_EXECUTOR, _EPISODE_EXECUTOR_WORKERS
    with _EXECUTOR_LOCK:
        if _EPISODE_EXECUTOR is None or _EPISODE_EXECUTOR_WORKERS < min_workers:
            _EPISODE_EXECUTOR = ThreadPoolExecutor(
                max_workers=min_workers,
                thread_name_prefix="wmh-harbor-episode",
            )
            _EPISODE_EXECUTOR_WORKERS = min_workers
        return _EPISODE_EXECUTOR


class HarborAgentEnvironment:
    """Expose Harbor's async task environment through WMH's synchronous protocol.

    Every executed step is also recorded so a cancelled episode can still persist the partial
    transcript the optimizer's proposer feeds on.
    """

    def __init__(
        self,
        event_loop: asyncio.AbstractEventLoop,
        environment: BaseEnvironment,
        *,
        command_timeout_sec: int = MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
    ) -> None:
        self._event_loop = event_loop
        self._environment = environment
        self._command_timeout_sec = _validate_command_timeout_sec(command_timeout_sec)
        self._recorded_steps: list[JsonObject] = []

    def execute(self, action: Action) -> Observation:
        """Execute one supported WMH tool in Harbor's owned task environment."""
        observation = self._execute(action)
        self._recorded_steps.append(
            {
                "action": action.model_dump(mode="json"),
                "observation": observation.model_dump(mode="json"),
            }
        )
        return observation

    def recorded_steps(self) -> list[JsonObject]:
        """The steps executed so far (the partial-transcript source on cancellation)."""
        return list(self._recorded_steps)

    def close(self) -> None:
        """Leave lifecycle ownership with Harbor."""

    def _execute(self, action: Action) -> Observation:
        if action.kind is not ActionKind.TOOL_CALL or not is_env_action(action):
            return Observation(content=f"tool {action.name!r} not available", is_error=True)
        arguments = action.arguments or {}
        if action.name == "bash":
            command = _string_argument(arguments, "command")
            if command is None:
                return _invalid_arguments("bash", "command must be a string")
            return _command_observation(self._exec(command))
        if action.name == "read_file":
            path = _string_argument(arguments, "path", nonempty=True)
            if path is None:
                return _invalid_arguments("read_file", "path must be a nonempty string")
            return _command_observation(self._exec(f"cat -- {shlex.quote(path)}"))
        if action.name == "write_file":
            path = _string_argument(arguments, "path", nonempty=True)
            content = _string_argument(arguments, "content")
            if path is None or content is None:
                return _invalid_arguments(
                    "write_file", "path must be nonempty and content must be a string"
                )
            result = self._exec(
                _WRITE_COMMAND,
                env={
                    "WMH_FILE_PATH": path,
                    "WMH_FILE_CONTENT_B64": base64.b64encode(content.encode()).decode(),
                },
            )
            observation = _command_observation(result)
            if not observation.is_error:
                return Observation(
                    content=f"wrote {path}",
                    metadata=observation.metadata,
                )
            return observation
        return Observation(content=f"tool {action.name!r} not available", is_error=True)

    def _exec(self, command: str, *, env: dict[str, str] | None = None) -> ExecResult:
        future = asyncio.run_coroutine_threadsafe(
            self._environment.exec(
                command,
                env=env,
                timeout_sec=self._command_timeout_sec,
            ),
            self._event_loop,
        )
        return future.result()


class WmhHarborAgent(BaseAgent):
    """Run the serialized WMH candidate while Harbor owns tasks and verification."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        skills_dir: str | None = None,
        *,
        command_timeout_sec: int = MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
        extra_env: dict[str, str] | None = None,
        harness: JsonObject,
        provider_config: JsonObject,
        harness_backend: Literal["local", "e2b"] = "local",
        e2b_template: str | None = None,
        episode_timeout_sec: float = DEFAULT_EVAL_EPISODE_TIMEOUT_S,
        episode_workers: int = DEFAULT_EPISODE_WORKERS,
    ) -> None:
        if extra_env:
            raise ValueError("WMH Harbor evaluation does not inject agent environment variables")
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            logger=logger,
            mcp_servers=mcp_servers,
            skills_dir=skills_dir,
            extra_env=extra_env,
        )
        if harness_backend not in ("local", "e2b"):
            raise ValueError("harness_backend must be local or e2b")
        if harness_backend == "local" and e2b_template is not None:
            raise ValueError("e2b_template requires harness_backend='e2b'")
        try:
            self._episode_timeout_sec = validate_episode_timeout_s(episode_timeout_sec)
        except ValueError as error:
            raise ValueError("episode_timeout_sec must be a finite positive number") from error
        if (
            harness_backend == "local"
            and self._episode_timeout_sec != DEFAULT_EVAL_EPISODE_TIMEOUT_S
        ):
            raise ValueError("episode_timeout_sec requires harness_backend='e2b'")
        if isinstance(episode_workers, bool) or not isinstance(episode_workers, int):
            raise ValueError("episode_workers must be a positive integer")
        if episode_workers < 1:
            raise ValueError("episode_workers must be a positive integer")
        self._harness = HarnessDoc.model_validate(harness)
        self._provider_config = ProviderConfig.model_validate(provider_config)
        expected_model_name = f"{self._provider_config.kind.value}/{self._provider_config.model}"
        if model_name != expected_model_name:
            raise ValueError(
                f"Harbor model identity must be {expected_model_name!r}, got {model_name!r}"
            )
        # Retry-wrap the worker provider: Bedrock disables botocore's own retries, so one
        # unwrapped ThrottlingException would otherwise kill a whole trial.
        self._provider = wrap_provider_with_retries(get_provider(self._provider_config))
        self._command_timeout_sec = _validate_command_timeout_sec(command_timeout_sec)
        self._harness_backend = harness_backend
        self._e2b_template = e2b_template
        self._episode_workers = episode_workers

    @staticmethod
    def name() -> str:
        return "wmh-harness"

    def version(self) -> str:
        return WMH_HARBOR_AGENT_VERSION

    async def setup(self, environment: BaseEnvironment) -> None:
        """Use Harbor's already-started task environment without installing another agent."""
        del environment

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run the candidate in a dedicated worker thread and always persist its WMH trace."""
        context.metadata = {"candidate_doc_hash": self._harness.doc_hash}
        cancel_requested = threading.Event()
        runtime = self._harness.runtime(
            self._provider,
            backend=self._harness_backend,
            e2b_template=self._e2b_template,
            episode_timeout_s=(
                self._episode_timeout_sec if self._harness_backend == "e2b" else None
            ),
            # A real task environment is mutable, so an E2B transport failure must not replay
            # the whole episode against already-mutated state. Local Pi has no replay wrapper.
            transport_retries=0 if self._harness_backend == "e2b" else None,
            should_cancel=cancel_requested.is_set,
        )
        bridge = HarborAgentEnvironment(
            asyncio.get_running_loop(),
            environment,
            command_timeout_sec=self._command_timeout_sec,
        )
        task_id = str(self.context_id or self.session_id or "harbor-task")
        loop = asyncio.get_running_loop()
        executor = _episode_executor(self._episode_workers)
        run_task = asyncio.ensure_future(
            loop.run_in_executor(executor, lambda: runtime.run(task_id, instruction, bridge))
        )
        result: RunResult | None = None
        try:
            # Harbor enforces its agent timeout by cancelling this coroutine. Shield the
            # worker so cancellation cannot detach a still-running harness from the task
            # environment that Harbor is about to verify.
            result = await asyncio.shield(run_task)
        except asyncio.CancelledError:
            cancel_requested.set()
            abort = getattr(runtime, "abort", None)
            try:
                if callable(abort):
                    await _run_uncancellable(abort, executor)
            finally:
                await _wait_for_quiescence(run_task)
            raise
        finally:
            close = getattr(runtime, "close", None)
            if callable(close):
                await _run_uncancellable(close, executor)
            bridge.close()
            # The trace write lives inside this finally: a harbor-timeout cancellation (the most
            # informative failure class) must still leave the partial transcript for the
            # proposer. Synchronous small-file I/O, so re-cancellation cannot interrupt it.
            self._write_trace(task_id, run_task, bridge, cancelled=cancel_requested.is_set())
        _populate_context(context, result)

    def _write_trace(
        self,
        task_id: str,
        run_task: asyncio.Future[RunResult],
        bridge: HarborAgentEnvironment,
        *,
        cancelled: bool,
    ) -> None:
        """Persist the full RunResult when one exists, else the partial episode evidence."""
        payload = _trace_payload(task_id, run_task, bridge, cancelled=cancelled)
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            (self.logs_dir / _TRACE_FILENAME).write_text(payload, encoding="utf-8")
        except OSError:
            if not cancelled and not run_task.cancelled() and run_task.exception() is None:
                raise  # a healthy trial must not silently lose its transcript
            self.logger.warning("failed to persist the partial WMH trace", exc_info=True)


def _trace_payload(
    task_id: str,
    run_task: asyncio.Future[RunResult],
    bridge: HarborAgentEnvironment,
    *,
    cancelled: bool,
) -> str:
    if run_task.done() and not run_task.cancelled() and run_task.exception() is None:
        return run_task.result().model_dump_json(indent=2)
    error = None if not run_task.done() or run_task.cancelled() else run_task.exception()
    stop_reason = (
        _CANCELLED_STOP_REASON
        if cancelled
        else f"agent-exception:{type(error).__name__}"
        if error is not None
        else _CANCELLED_STOP_REASON
    )
    steps = bridge.recorded_steps()
    partial: JsonObject = {
        "task_id": task_id,
        "steps": steps,
        "stop_reason": stop_reason,
        "answer": "",
        "turns": len(steps),
        "partial": True,
    }
    if error is not None:
        partial["error"] = f"{type(error).__name__}: {error}"
    usage = getattr(error, "worker_usage", None)
    if usage is not None:
        partial["worker_usage"] = usage.model_dump(mode="json")
    return json.dumps(partial, indent=2, ensure_ascii=False)


async def _wait_for_quiescence(run_task: asyncio.Future[RunResult]) -> None:
    """Drain a shielded runtime even if the owning Harbor task is cancelled again."""
    await _wait_until_done(run_task)
    if not run_task.cancelled():
        run_task.exception()


async def _run_uncancellable[T](
    call: Callable[[], T],
    executor: ThreadPoolExecutor,
) -> T:
    """Run blocking cleanup to completion despite repeated coroutine cancellation."""
    loop = asyncio.get_running_loop()
    cleanup_task = asyncio.ensure_future(loop.run_in_executor(executor, call))
    cancelled = await _wait_until_done(cleanup_task)
    result = cleanup_task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


async def _wait_until_done[T](task: asyncio.Future[T]) -> bool:
    """Wait without propagating the child result or cancelling it with the waiter."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.wait({task})
        except asyncio.CancelledError:
            cancelled = True
    return cancelled


def _populate_context(context: AgentContext, result: RunResult | None) -> None:
    if result is None:
        return
    usage = result.worker_usage
    if usage is not None:
        context.n_input_tokens = usage.input_tokens
        context.n_output_tokens = usage.output_tokens
    metadata = dict(context.metadata or {})
    metadata.update(
        {
            "stop_reason": result.stop_reason.value,
            "turns": result.turns,
        }
    )
    context.metadata = metadata


def _string_argument(
    arguments: Mapping[str, object],
    name: str,
    *,
    nonempty: bool = False,
) -> str | None:
    value = arguments.get(name)
    if not isinstance(value, str) or (nonempty and not value):
        return None
    return value


def _invalid_arguments(tool: str, message: str) -> Observation:
    return Observation(content=f"invalid {tool} arguments: {message}", is_error=True)


def _command_observation(result: ExecResult) -> Observation:
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    content = stdout + stderr
    if result.return_code != 0:
        content += f"\n[exit {result.return_code}]"
    return Observation(
        content=content,
        is_error=result.return_code != 0,
        metadata={"return_code": result.return_code},
    )


def _validate_command_timeout_sec(value: int) -> int:
    """Validate the evaluator-owned finite task-command policy."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC
    ):
        raise ValueError(
            f"command_timeout_sec must be an integer in [1, {MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC}]"
        )
    return value
