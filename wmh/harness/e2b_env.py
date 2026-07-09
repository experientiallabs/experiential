"""`E2BEnvironment`: the real execution backend behind the `AgentEnvironment` seam.

Where `wmh.evals.closed_loop.WorldModelEnvironment` answers every tool call with a model
prediction, this module answers them for real: one fresh E2B microVM per rollout, `bash` running
in its shell and `read_file`/`write_file` going through the sandbox filesystem API. Because both
backends implement the same protocol, the identical agent loop and scoring core produce directly
comparable reports (`wmh.evals.agreement`).

The e2b SDK stays an optional extra (`uv sync --extra e2b`) imported lazily inside the default
sandbox factory; tests (and any caller) can inject a `SandboxFactory` returning any
`SandboxHandle`, the exact slice of `e2b.Sandbox` this module and `E2BPiRuntime` use. Sandbox
creation retries capacity/rate/5xx failures with fixed (1, 3, 9) s delays — the
`wmh.providers.retry.RetryingProvider` precedent, no RNG in scoring paths.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Protocol, cast, runtime_checkable

from wmh.core.types import Action, ActionKind, Observation
from wmh.evals.closed_loop import EnvFactory
from wmh.evals.tasks import TaskSpec
from wmh.harness.environment import ENV_TOOLS

E2B_API_KEY_ENV = "E2B_API_KEY"
E2B_TEMPLATE_ENV = "WMH_E2B_TEMPLATE"

# Per-command cap inside a rollout. Generous for agent steps (a stuck command should fail the
# step, not hang the eval); setup commands (package installs) share it.
COMMAND_TIMEOUT_S = 120.0

# Fixed delays before each retry of sandbox creation (RetryingProvider precedent: 1s, 3s, 9s).
_CREATE_DELAYS = (1.0, 3.0, 9.0)


@runtime_checkable
class CommandOutput(Protocol):
    """The result slice of a finished sandbox command (e2b's `CommandResult` shape).

    `runtime_checkable` because e2b's `CommandExitException` *is* a `CommandResult` (non-zero
    exits raise instead of returning) — an isinstance check against this protocol is how we
    recognize and normalize it without importing the SDK.
    """

    stdout: str
    stderr: str
    exit_code: int


class CommandHandle(Protocol):
    """A background sandbox command (e2b's handle): stdin by pid, iteration yields stream events.

    Iteration events are `(stdout, stderr, pty)` chunks; unused here but part of the frozen
    `SandboxHandle` slice because `E2BPiRuntime` drives the RunnerLink frame stream over one.
    """

    @property
    def pid(self) -> int: ...

    def __iter__(self) -> Iterator[tuple[str | None, str | None, str | None]]: ...


class SandboxCommands(Protocol):
    """The `sandbox.commands` slice: run (foreground or background) and stdin injection."""

    def run(
        self,
        cmd: str,
        background: bool | None = None,
        *,
        stdin: bool | None = None,
        timeout: float | None = None,
    ) -> CommandOutput | CommandHandle: ...

    def send_stdin(self, pid: int, data: str) -> object: ...


class SandboxFiles(Protocol):
    """The `sandbox.files` slice: whole-file read and write."""

    def write(self, path: str, data: str) -> object: ...

    def read(self, path: str) -> str: ...


@runtime_checkable
class SandboxHandle(Protocol):
    """The exact slice of `e2b.Sandbox` the harness uses, so tests substitute fakes."""

    @property
    def commands(self) -> SandboxCommands: ...

    @property
    def files(self) -> SandboxFiles: ...

    def kill(self) -> object: ...


# Opens one sandbox. The default factory calls the real SDK; tests inject fakes.
SandboxFactory = Callable[[], SandboxHandle]


def _default_sandbox_factory(
    *, api_key: str | None, template: str | None, timeout: float
) -> SandboxHandle:
    """Create a real E2B sandbox (lazy SDK import; key from arg or $E2B_API_KEY)."""
    try:
        from e2b import Sandbox
    except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
        raise ImportError(
            "the e2b SDK is not installed; run `uv sync --extra e2b` to use the e2b env backend"
        ) from exc
    key = api_key or os.environ.get(E2B_API_KEY_ENV)
    if not key:
        raise RuntimeError(f"set ${E2B_API_KEY_ENV} to run rollouts in E2B sandboxes")
    chosen = template or os.environ.get(E2B_TEMPLATE_ENV) or None
    sandbox = Sandbox.create(template=chosen, timeout=int(timeout), api_key=key)
    # The SDK object satisfies the protocol slice structurally; cast rather than pin the SDK's
    # full (much wider) signatures into the protocol.
    return cast("SandboxHandle", sandbox)


def _is_retryable_create_error(exc: Exception) -> bool:
    """True for capacity-shaped creation failures (rate limit / no capacity / 5xx).

    Matched by exception name and message so fakes need no SDK import; anything else (auth,
    bad template, missing key) fails immediately — retrying those only hides real bugs.
    """
    if type(exc).__name__ == "RateLimitException":  # e2b's 429
        return True
    text = str(exc).lower()
    if "rate limit" in text or "capacity" in text or "too many requests" in text:
        return True
    return any(code in text for code in ("429", "500", "502", "503", "504"))


def _create_with_retry(factory: SandboxFactory) -> SandboxHandle:
    """Call `factory` retrying capacity errors with fixed (1, 3, 9) s delays."""
    for delay in _CREATE_DELAYS:
        try:
            return factory()
        except Exception as exc:  # noqa: BLE001 - classified below; non-capacity re-raises
            if not _is_retryable_create_error(exc):
                raise
            time.sleep(delay)
    return factory()  # final attempt: let any error propagate


def _is_missing_file_error(exc: Exception) -> bool:
    """True when the SDK says the path does not exist (defensive import: no SDK, no match)."""
    try:
        from e2b.exceptions import NotFoundException
    except ImportError:  # pragma: no cover - injected fakes imply the SDK may be absent
        return False
    return isinstance(exc, NotFoundException)


def _to_observation(result: CommandOutput) -> Observation:
    """Normalize a command result (or a raised `CommandExitException`) into an Observation.

    Streams are concatenated as a real terminal would show them; a non-zero exit is an *error
    observation*, not a harness failure — agents run failing commands routinely.
    """
    exit_code = int(result.exit_code or 0)
    content = "".join(part for part in (result.stdout or "", result.stderr or "") if part).strip()
    return Observation(content=content, is_error=exit_code != 0, metadata={"exit_code": exit_code})


class E2BEnvironment:
    """A real sandbox behind the `AgentEnvironment` protocol: one E2B microVM per rollout.

    The sandbox is created eagerly in the constructor (with capacity retries) and `setup`
    commands run before the episode starts — a failed setup kills the sandbox and raises, so a
    broken task definition surfaces as a hard error instead of a mis-scored rollout.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        template: str | None = None,
        timeout: float = 600.0,
        setup: Sequence[str] = (),
        sandbox_factory: SandboxFactory | None = None,
    ) -> None:
        factory = sandbox_factory or (
            lambda: _default_sandbox_factory(api_key=api_key, template=template, timeout=timeout)
        )
        self._sandbox = _create_with_retry(factory)
        self._closed = False
        for command in setup:
            try:
                self._sandbox.commands.run(command, timeout=COMMAND_TIMEOUT_S)
            except Exception as exc:  # noqa: BLE001 - any setup failure must not leak the sandbox
                self.close()
                detail = f": {exc.stderr}" if isinstance(exc, CommandOutput) and exc.stderr else ""
                raise RuntimeError(f"sandbox setup command failed ({command!r}){detail}") from exc

    @property
    def sandbox(self) -> SandboxHandle:
        """The underlying sandbox — `E2BPiRuntime` runs the pi agent inside this same VM."""
        return self._sandbox

    def execute(self, action: Action) -> Observation:
        if action.kind != ActionKind.TOOL_CALL or action.name not in ENV_TOOLS:
            # A non-env tool reached the environment (shouldn't happen: the runtime routes those).
            return Observation(content=f"unsupported action: {action.name}", is_error=True)
        if action.name == "bash":
            return self._bash(action)
        if action.name == "read_file":
            return self._read_file(action)
        return self._write_file(action)

    def close(self) -> None:
        """Best-effort teardown; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        try:
            self._sandbox.kill()
        except Exception:  # noqa: BLE001 - best-effort cleanup; a dead sandbox is already gone
            pass

    def _bash(self, action: Action) -> Observation:
        command = action.arguments.get("command")
        if not isinstance(command, str) or not command:
            return Observation(content="empty command", is_error=True)
        try:
            result = self._sandbox.commands.run(command, timeout=COMMAND_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 - classified below; non-exit errors re-raise
            # The SDK *raises* `CommandExitException` on non-zero exit; it subclasses
            # `CommandResult`, so recognize it structurally (no SDK import needed) and
            # normalize it into an error observation. Anything else is a real failure.
            if not isinstance(exc, CommandOutput):
                raise
            return _to_observation(exc)
        return _to_observation(cast("CommandOutput", result))

    def _read_file(self, action: Action) -> Observation:
        path = action.arguments.get("path")
        if not isinstance(path, str) or not path:
            return Observation(content="read_file needs a path", is_error=True)
        try:
            return Observation(content=self._sandbox.files.read(path))
        except Exception as exc:  # noqa: BLE001 - classified below; non-missing errors re-raise
            if not _is_missing_file_error(exc):
                raise
            return Observation(content=f"no such file: {path}", is_error=True)

    def _write_file(self, action: Action) -> Observation:
        path = action.arguments.get("path")
        content = action.arguments.get("content")
        if not isinstance(path, str) or not path:
            return Observation(content="write_file needs a path", is_error=True)
        if not isinstance(content, str):
            return Observation(content="write_file needs string content", is_error=True)
        self._sandbox.files.write(path, content)
        return Observation(content=f"wrote {len(content)} characters to {path}")


def e2b_env_factory(
    *,
    api_key: str | None = None,
    template: str | None = None,
    timeout: float = 600.0,
    sandbox_factory: SandboxFactory | None = None,
) -> EnvFactory:
    """An `EnvFactory` opening one fresh E2B sandbox per rollout, seeded with the task's setup."""

    def make(task: TaskSpec) -> E2BEnvironment:
        return E2BEnvironment(
            api_key=api_key,
            template=template,
            timeout=timeout,
            setup=task.setup,
            sandbox_factory=sandbox_factory,
        )

    return make
