"""Tests for the harness environment seam: which actions the environment answers."""

from __future__ import annotations

from wmo.common.core.types import Action, ActionKind, Observation
from wmo.runtime.harness.environment import ENV_TOOLS, AgentEnvironment, is_env_action


class _RecordingEnvironment:
    """The smallest AgentEnvironment: executes an action, records it, and closes."""

    def __init__(self) -> None:
        self.executed: list[Action] = []
        self.closed = False

    def execute(self, action: Action) -> Observation:
        self.executed.append(action)
        return Observation(content="ok")

    def close(self) -> None:
        self.closed = True


def test_the_protocol_is_execute_plus_close() -> None:
    declared = sorted(name for name in vars(AgentEnvironment) if not name.startswith("_"))

    assert declared == ["close", "execute"]


def test_a_structural_backend_satisfies_the_protocol() -> None:
    # This is the substitution point between the world model and a real execution backend, and
    # neither side subclasses the protocol, so shape is the whole contract.
    assert isinstance(_RecordingEnvironment(), AgentEnvironment)

    class _ExecuteOnly:
        def execute(self, action: Action) -> Observation:
            return Observation(content="ok")

    assert not isinstance(_ExecuteOnly(), AgentEnvironment)


def test_env_tools_are_the_three_the_environment_answers() -> None:
    # `submit` is deliberately absent: the runtime handles it, so routing it to the environment
    # would ask the world model to predict an observation for ending the episode.
    assert ENV_TOOLS == frozenset({"bash", "read_file", "write_file"})
    assert "submit" not in ENV_TOOLS


def test_is_env_action_accepts_every_env_tool_call() -> None:
    for name in sorted(ENV_TOOLS):
        assert is_env_action(Action(kind=ActionKind.TOOL_CALL, name=name))


def test_is_env_action_rejects_submit_unknown_tools_and_messages() -> None:
    assert not is_env_action(Action(kind=ActionKind.TOOL_CALL, name="submit"))
    assert not is_env_action(Action(kind=ActionKind.TOOL_CALL, name="not_a_tool"))
    assert not is_env_action(Action(kind=ActionKind.TOOL_CALL, name=None))
    # A message that merely mentions a tool name is not a call to it.
    assert not is_env_action(Action(kind=ActionKind.MESSAGE, content="bash"))
