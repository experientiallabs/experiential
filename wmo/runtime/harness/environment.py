"""The environment seam: the agent loop talks to an interface, not to any backend directly.

The `AgentEnvironment` protocol is the substitution point between the fixed agent loop and an
execution backend. The retired text-model environment implemented the same two methods as real
backends, so one loop and scoring contract can target either without importing backend mechanics.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from wmo.common.core.types import Action, ActionKind, Observation

# The tool names the environment answers (everything except the runtime-handled `submit`).
ENV_TOOLS = frozenset({"bash", "read_file", "write_file"})


@runtime_checkable
class AgentEnvironment(Protocol):
    """Executes an agent Action and returns what the environment observed."""

    def execute(self, action: Action) -> Observation:
        """Run one action; return the resulting observation."""
        ...

    def close(self) -> None:
        """Release any underlying resources (end the session)."""
        ...


def is_env_action(action: Action) -> bool:
    """True when the action is one the environment answers (a tool call to an env tool)."""
    return action.kind == ActionKind.TOOL_CALL and action.name in ENV_TOOLS
