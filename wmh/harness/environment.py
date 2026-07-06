"""The environment seam: the agent loop talks to an interface, not to any backend directly.

The `AgentEnvironment` protocol is the substitution point: closed-loop eval binds it to the world
model (`wmh.evals.closed_loop.WorldModelEnvironment` — every tool call answered by
`WorldModel.step`), and a real execution backend (a managed sandbox) implements the same two
methods, so the *same* agent loop and scoring can run against reality when one is available. That
symmetry is what makes a simulated report comparable to a real one (`wmh.evals.agreement`).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from wmh.core.types import Action, ActionKind, Observation

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
