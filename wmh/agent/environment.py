"""The environment seam: one interface, a real backend and a simulated backend.

The whole point of the build-agent path is that the *same* agent loop runs two ways:

  - `E2BEnvironment` executes actions for real in a Firecracker microVM (E2B) — this is where we
    collect ground-truth traces to build/extend the world model.
  - `WorldModelEnvironment` routes the identical actions to `WorldModel.step`, so the frontier LLM
    predicts the observation instead of a real shell running it — this is closed-loop eval ("Docker
    as an LLM"), the decoupled-simulator pattern (Qwen-AgentWorld/WMA): a swappable env backend
    behind a fixed API.

Both return a normalized `Observation`, so `AgentRuntime` is oblivious to which one it drives.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from wmh.core.types import Action, ActionKind, Observation
from wmh.engine.world_model import WorldModel

E2B_API_KEY_ENV = "E2B_API_KEY"


@runtime_checkable
class AgentEnvironment(Protocol):
    """Executes an agent Action and returns what the environment observed."""

    def execute(self, action: Action) -> Observation:
        """Run one action; return the resulting observation."""
        ...

    def close(self) -> None:
        """Release any underlying resources (kill the sandbox, end the session)."""
        ...


def _render_command(action: Action) -> str | None:
    """Turn a bash/read_file/write_file action into a shell command, or None if not an env tool.

    The three env tools all reduce to shell, so the real and simulated backends share this mapping —
    a real sandbox runs the command; the world model is asked to predict its output.
    """
    args = action.arguments
    if action.name == "bash":
        cmd = args.get("command")
        return cmd if isinstance(cmd, str) else ""
    if action.name == "read_file":
        path = args.get("path")
        return f"cat {_shquote(path)}" if isinstance(path, str) else ""
    if action.name == "write_file":
        path, content = args.get("path"), args.get("content")
        if isinstance(path, str) and isinstance(content, str):
            heredoc = _heredoc(content)
            return f'mkdir -p "$(dirname {_shquote(path)})" && cat > {_shquote(path)} {heredoc}'
        return ""
    return None


def _shquote(value: str) -> str:
    """Single-quote a string for POSIX shell."""
    return "'" + value.replace("'", "'\\''") + "'"


def _heredoc(content: str) -> str:
    """A quoted heredoc writing `content` verbatim (no expansion); terminator chosen to be safe."""
    term = "WMH_EOF"
    while term in content:
        term += "_"
    body = content if content.endswith("\n") else content + "\n"
    return f"<<'{term}'\n{body}{term}"


class E2BEnvironment:
    """A real E2B sandbox: actions run as shell commands in a Firecracker microVM.

    Lazily imports the SDK so the dependency is optional (only trace collection needs it). Uses the
    v2 `Sandbox.create()` classmethod and `commands.run` / `files` APIs.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        template: str | None = None,
        timeout: int = 300,
        setup: list[str] | None = None,
    ) -> None:
        try:
            from e2b import Sandbox
        except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
            raise RuntimeError(
                "the E2B SDK is not installed; run `uv sync --extra e2b` to collect real traces"
            ) from exc
        key = api_key or os.environ.get(E2B_API_KEY_ENV)
        if not key:
            raise RuntimeError(f"set ${E2B_API_KEY_ENV} to collect traces in an E2B sandbox")
        # Pass template only when set (None would select a non-default overload path); the SDK reads
        # the key from `api_key=` explicitly so a caller-supplied key overrides the env var.
        if template is not None:
            self._sandbox = Sandbox.create(template=template, timeout=timeout, api_key=key)
        else:
            self._sandbox = Sandbox.create(timeout=timeout, api_key=key)
        # Run task setup non-fatally: a failing setup command (e.g. writing outside the sandbox
        # user's home) should surface as a hard error with the sandbox cleaned up, not leak the VM.
        try:
            for command in setup or []:
                self._sandbox.commands.run(command)
        except Exception:
            self.close()
            raise

    def execute(self, action: Action) -> Observation:
        command = _render_command(action)
        if command is None:
            # A non-env tool reached the environment (shouldn't happen: the runtime handles those).
            return Observation(content=f"unsupported action: {action.name}", is_error=True)
        if not command:
            return Observation(content="empty command", is_error=True)
        # The SDK *raises* `CommandExitException` on a non-zero exit rather than returning a result.
        # Agents run failing commands routinely (a failed grep, a missing file), and a failure is a
        # legitimate observation the world model must learn from — so we normalize both the success
        # result and the exception (which subclasses CommandResult with the same fields) into an
        # Observation instead of letting the exception abort the run.
        from e2b.sandbox.commands.command_handle import CommandExitException

        try:
            result: object = self._sandbox.commands.run(command, timeout=120)
        except CommandExitException as exc:
            result = exc
        return _result_to_observation(result)

    def close(self) -> None:
        try:
            self._sandbox.kill()
        except Exception:  # noqa: BLE001 - best-effort cleanup; a dead sandbox is already gone
            pass


class WorldModelEnvironment:
    """A simulated environment: actions are answered by the world model, not a real shell.

    Wraps one `WorldModel` session, so `AgentRuntime` drives closed-loop eval exactly as it drives a
    real run. `transcript` accumulates the (action, observation) pairs for the gold judge.
    """

    def __init__(self, world_model: WorldModel, task: str) -> None:
        self._wm = world_model
        self._session = world_model.new_session(task=task)

    def execute(self, action: Action) -> Observation:
        return self._wm.step(self._session.id, action)

    @property
    def session_id(self) -> str:
        return self._session.id

    def close(self) -> None:
        # The world model keeps sessions in memory for the life of the model; nothing to release.
        pass


def _result_to_observation(result: object) -> Observation:
    """Normalize an E2B `CommandResult` (or a raised `CommandExitException`) into an Observation.

    Both carry `stdout`/`stderr`/`exit_code`; we concatenate the streams (as a real terminal shows)
    and flag `is_error` on a non-zero exit.
    """
    exit_code = int(getattr(result, "exit_code", 0) or 0)
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    content = "".join(p for p in (stdout, stderr) if p).strip()
    return Observation(content=content, is_error=exit_code != 0, metadata={"exit_code": exit_code})


def message_observation(text: str) -> Observation:
    """A trivial observation for a `message` action (the agent talking, not acting)."""
    return Observation(content=text)


def is_env_action(action: Action) -> bool:
    """True when the action is one the environment executes (a tool call to an env tool)."""
    return action.kind == ActionKind.TOOL_CALL and _render_command(action) is not None
