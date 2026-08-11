"""Tests for the Env protocol the episode runner steps against."""

from __future__ import annotations

from wmo.common.core.types import Action, ActionKind, EnvState, Observation
from wmo.runtime.environment import Env


class _MinimalEnv:
    """The smallest thing that is an Env: a live state view, a step, and an idempotent close."""

    def __init__(self) -> None:
        self.closes = 0
        self._state = EnvState()

    def reset(self, task: str | None = None, seed_state: EnvState | None = None) -> EnvState:
        self._state = seed_state.model_copy(deep=True) if seed_state else EnvState()
        return self._state

    def step(self, action: Action) -> Observation:
        self._state.scratchpad += f"ran {action.name}\n"  # the live view mutates in place
        return Observation(content="ok")

    def close(self) -> None:
        self.closes += 1


def test_the_protocol_is_reset_step_close() -> None:
    declared = sorted(name for name in vars(Env) if not name.startswith("_"))

    assert declared == ["close", "reset", "step"]


def test_a_structural_env_satisfies_the_protocol_without_inheriting_it() -> None:
    # Backends implement this by shape (the world-model env, sandbox envs, test doubles); nothing
    # subclasses `Env`, so the runtime check is the only conformance signal callers get.
    assert isinstance(_MinimalEnv(), Env)


def test_an_incomplete_env_is_not_an_env() -> None:
    class _NoClose:
        def reset(self, task: str | None = None, seed_state: EnvState | None = None) -> EnvState:
            return EnvState()

        def step(self, action: Action) -> Observation:
            return Observation(content="ok")

    assert not isinstance(_NoClose(), Env)


def test_reset_returns_a_live_view_that_step_updates_in_place() -> None:
    # The documented contract `run_episode` depends on: the object reset handed back keeps
    # reflecting the episode, which is why recorded steps snapshot it.
    env: Env = _MinimalEnv()
    state = env.reset("do the thing")

    env.step(Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"cmd": "ls"}))

    assert state.scratchpad == "ran bash\n"


def test_reset_seeds_from_a_supplied_state() -> None:
    env: Env = _MinimalEnv()

    state = env.reset(seed_state=EnvState(structured={"cwd": "/tmp"}, scratchpad="seeded\n"))

    assert state.structured == {"cwd": "/tmp"}
    assert state.scratchpad == "seeded\n"


def test_close_is_idempotent() -> None:
    env = _MinimalEnv()

    env.close()
    env.close()

    assert env.closes == 2  # calling twice is allowed; the env, not the caller, guards resources
