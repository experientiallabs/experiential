"""Harbor agent bridge for running an exact WMH harness document."""

from __future__ import annotations

import asyncio
import base64
import logging
import shlex
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal, TypeVar

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig

from wmh.core.types import Action, ActionKind, Observation
from wmh.harness.doc import HarnessDoc
from wmh.harness.environment import is_env_action
from wmh.harness.runtime import RunResult
from wmh.providers.base import ProviderConfig
from wmh.providers.registry import get_provider

WMH_HARBOR_AGENT_VERSION = "1"
WMH_HARBOR_AGENT_IMPORT_PATH = "wmh.evals.harbor.agent:WmhHarborAgent"
_TRACE_FILENAME = "wmh-run.json"
_TaskResultT = TypeVar("_TaskResultT")
_WRITE_COMMAND = (
    'mkdir -p -- "$(dirname -- "$WMH_FILE_PATH")" && '
    'printf \'%s\' "$WMH_FILE_CONTENT_B64" | base64 -d > "$WMH_FILE_PATH"'
)


class HarborAgentEnvironment:
    """Expose Harbor's async task environment through WMH's synchronous protocol."""

    def __init__(
        self,
        event_loop: asyncio.AbstractEventLoop,
        environment: BaseEnvironment,
    ) -> None:
        self._event_loop = event_loop
        self._environment = environment

    def execute(self, action: Action) -> Observation:
        """Execute one supported WMH tool in Harbor's owned task environment."""
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
            return _command_observation(self._exec(f"cat -- {_shell_quote(path)}"))
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

    def close(self) -> None:
        """Leave lifecycle ownership with Harbor."""

    def _exec(self, command: str, *, env: dict[str, str] | None = None) -> ExecResult:
        future = asyncio.run_coroutine_threadsafe(
            self._environment.exec(command, env=env),
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
        extra_env: dict[str, str] | None = None,
        harness: dict[str, object],
        provider_config: dict[str, object],
        harness_backend: Literal["local", "e2b"] = "local",
        e2b_template: str | None = None,
        pi_transport: Literal["ssh"] | None = None,
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
        if harness_backend == "local" and pi_transport not in (None, "ssh"):
            raise ValueError("local Harbor harness execution requires pi_transport='ssh'")
        if harness_backend == "e2b" and pi_transport is not None:
            raise ValueError("pi_transport applies only to local harness execution")
        self._harness = HarnessDoc.model_validate(harness)
        self._provider_config = ProviderConfig.model_validate(provider_config)
        expected_model_name = f"{self._provider_config.kind.value}/{self._provider_config.model}"
        if model_name != expected_model_name:
            raise ValueError(
                f"Harbor model identity must be {expected_model_name!r}, got {model_name!r}"
            )
        self._provider = get_provider(self._provider_config)
        self._harness_backend = harness_backend
        self._e2b_template = e2b_template
        self._pi_transport: Literal["ssh"] | None = "ssh" if harness_backend == "local" else None

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
        """Run the candidate in a worker thread and preserve its exact WMH trace."""
        context.metadata = {"candidate_doc_hash": self._harness.doc_hash}
        cancel_requested = threading.Event()
        runtime = self._harness.runtime(
            self._provider,
            backend=self._harness_backend,
            e2b_template=self._e2b_template,
            pi_transport=self._pi_transport,
            # A real task environment is mutable, so an E2B transport failure must not replay
            # the whole episode against already-mutated state. Local Pi has no replay wrapper.
            transport_retries=0 if self._harness_backend == "e2b" else None,
            should_cancel=cancel_requested.is_set,
        )
        bridge = HarborAgentEnvironment(asyncio.get_running_loop(), environment)
        task_id = str(self.context_id or self.session_id or "harbor-task")
        run_task = asyncio.create_task(asyncio.to_thread(runtime.run, task_id, instruction, bridge))
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
                    await _run_uncancellable(abort)
            finally:
                await _wait_for_quiescence(run_task)
            raise
        finally:
            close = getattr(runtime, "close", None)
            if callable(close):
                await _run_uncancellable(close)
            bridge.close()
        _populate_context(context, result)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / _TRACE_FILENAME).write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )


async def _wait_for_quiescence(run_task: asyncio.Task[RunResult]) -> None:
    """Drain a shielded runtime even if the owning Harbor task is cancelled again."""
    await _wait_until_done(run_task)
    if not run_task.cancelled():
        run_task.exception()


async def _run_uncancellable(call: Callable[[], _TaskResultT]) -> _TaskResultT:
    """Run blocking cleanup to completion despite repeated coroutine cancellation."""
    cleanup_task = asyncio.create_task(asyncio.to_thread(call))
    cancelled = await _wait_until_done(cleanup_task)
    result = cleanup_task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


async def _wait_until_done(task: asyncio.Task[_TaskResultT]) -> bool:
    """Wait without propagating the child result or cancelling it with the waiter."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.wait({task})
        except asyncio.CancelledError:
            cancelled = True
    return cancelled


def _populate_context(context: AgentContext, result: RunResult) -> None:
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


def _shell_quote(value: str) -> str:
    return shlex.quote(value)
