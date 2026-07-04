"""The environment seam: the agent loop talks to an interface, not to the world model directly.

Closed-loop eval runs the agent against `WorldModelEnvironment` — every tool call is answered by
`WorldModel.step` (the frontier LLM predicting the observation) instead of a real shell. The
`AgentEnvironment` protocol is the substitution point: a real execution backend (a managed sandbox)
implements the same two methods, so the *same* agent loop and scoring can run against reality when
one is available. That symmetry is what makes a simulated report comparable to a real one
(`wmh.harness.agreement`).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from wmh.core.types import Action, ActionKind, Observation
from wmh.engine.world_model import WorldModel

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


class WorldModelEnvironment:
    """A simulated environment: actions are answered by the world model, not a real shell.

    Wraps one `WorldModel` session, so the agent loop drives closed-loop eval exactly as it would
    drive a real environment. Sessions are explicitly ended on `close` so batch rollouts don't
    accumulate resident session state in the model.
    """

    def __init__(self, world_model: WorldModel, task: str) -> None:
        self._wm = world_model
        self._session = world_model.new_session(task=task)
        self._closed = False

    @property
    def session_id(self) -> str:
        return self._session.id

    def execute(self, action: Action) -> Observation:
        return self._wm.step(self._session.id, action)

    def close(self) -> None:
        if not self._closed:
            self._wm.end_session(self._session.id)
            self._closed = True
