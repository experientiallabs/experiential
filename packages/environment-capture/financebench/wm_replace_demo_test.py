"""Tests for the WorldModel-as-CommandEnv bridge."""

from __future__ import annotations

from wm_replace_demo import WorldModelCommandEnv

from wmh.core.types import Action, ActionKind, EnvState, Observation


class _StubWorldModelEnv:
    """Duck-typed WorldModelEnv: records the actions it was stepped with."""

    def __init__(self) -> None:
        self.actions: list[Action] = []
        self.reset_task: str | None = None
        self.closed = False

    def reset(self, task: str | None = None, seed_state: EnvState | None = None) -> EnvState:
        self.reset_task = task
        return EnvState()

    def step(self, action: Action) -> Observation:
        self.actions.append(action)
        if action.arguments.get("command") == "boom":
            return Observation(content="kaboom", is_error=True)
        return Observation(content="42 rows", is_error=False)

    def close(self) -> None:
        self.closed = True


def test_bridge_translates_commands_to_tool_call_steps() -> None:
    stub = _StubWorldModelEnv()
    env = WorldModelCommandEnv(stub, task="What is capex?")
    assert stub.reset_task == "What is capex?"

    ok = env.execute("grep capex docs/*.txt")
    assert ok.output == "42 rows"
    assert ok.returncode == 0
    assert stub.actions[0].kind == ActionKind.TOOL_CALL
    assert stub.actions[0].name == "bash"
    assert stub.actions[0].arguments == {"command": "grep capex docs/*.txt"}

    err = env.execute("boom")
    assert err.returncode == 1
    assert err.output == "kaboom"

    env.close()
    assert stub.closed
