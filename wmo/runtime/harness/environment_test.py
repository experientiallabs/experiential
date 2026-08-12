"""Tests for `is_env_action`: which actions the environment answers, and which it must not.

The `AgentEnvironment` protocol gets no test of its own. Asserting it declares `execute` and
`close` restates the declaration, and no production code does `isinstance(x, AgentEnvironment)`,
so a structural conformance check here would pass for reasons unrelated to whether either side of
the seam works. The implementers are covered where they live
(`wmo/simulation/evaluation/closed_loop_test.py`), and `ty` binds them to the protocol.
"""

from __future__ import annotations

from wmo.common.core.types import Action, ActionKind
from wmo.runtime.harness.environment import ENV_TOOLS, is_env_action


def test_every_env_tool_call_routes_to_the_environment() -> None:
    for name in sorted(ENV_TOOLS):
        assert is_env_action(Action(kind=ActionKind.TOOL_CALL, name=name))


def test_submit_never_routes_to_the_environment() -> None:
    # `submit` ends the episode and is handled by the runtime. Routing it here would ask the world
    # model to invent an observation for the agent finishing, which scoring would then read.
    assert "submit" not in ENV_TOOLS
    assert not is_env_action(Action(kind=ActionKind.TOOL_CALL, name="submit"))


def test_unknown_tools_and_messages_are_not_env_actions() -> None:
    assert not is_env_action(Action(kind=ActionKind.TOOL_CALL, name="not_a_tool"))
    assert not is_env_action(Action(kind=ActionKind.TOOL_CALL, name=None))
    # A message that merely mentions a tool name is not a call to it.
    assert not is_env_action(Action(kind=ActionKind.MESSAGE, content="bash"))
