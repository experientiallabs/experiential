"""The environment contract consumed by the agent episode runner."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from wmo.common.core.types import Action, EnvState, Observation


@runtime_checkable
class Env(Protocol):
    """One episode of an environment an agent steps against."""

    def reset(self, task: str | None = None, seed_state: EnvState | None = None) -> EnvState:
        """Start an episode and return its live state view.

        The returned object is updated in place as the episode advances. Callers that need a
        point-in-time snapshot must copy it, as ``run_episode`` does for recorded steps.
        """
        ...

    def step(self, action: Action) -> Observation:
        """Apply an action and return the environment response."""
        ...

    def close(self) -> None:
        """Release episode resources. This operation is idempotent."""
        ...
