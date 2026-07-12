"""Tests for the canonical rendering helpers."""

from __future__ import annotations

from wmh.core.render import (
    build_env_prompt,
    encode_action,
    encode_state_action,
    render_action,
    render_demo,
    render_json,
)
from wmh.core.types import Action, ActionKind, EnvState, Observation, Step


def test_render_json_is_order_independent() -> None:
    a = render_json({"b": 2, "a": 1})
    b = render_json({"a": 1, "b": 2})
    assert a == b == '{"a":1,"b":2}'
    assert render_json({}) == "{}"


def test_render_action_tool_call_vs_message() -> None:
    tool = Action(kind=ActionKind.TOOL_CALL, name="buy", arguments={"sku": "A1"})
    assert render_action(tool) == 'tool_call buy({"sku":"A1"})'
    msg = Action(kind=ActionKind.MESSAGE, content="hello")
    assert render_action(msg) == "message: hello"


def test_encode_state_action_is_stable_and_structured() -> None:
    state = EnvState(structured={"b": 2, "a": 1}, scratchpad="logged in")
    action = Action(kind=ActionKind.TOOL_CALL, name="buy", arguments={"sku": "A1"})
    text = encode_state_action(state, action)
    assert "STATE:" in text and "ACTION kind=tool_call" in text
    assert "tool: buy" in text and '"a":1' in text
    # Insertion order must not change the rendering.
    other = EnvState(structured={"a": 1, "b": 2}, scratchpad="logged in")
    assert encode_state_action(other, action) == text


def test_render_demo_includes_observation() -> None:
    step = Step(
        action=Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "u1"}),
        observation=Observation(content="not found", is_error=True),
    )
    demo = render_demo(step)
    assert "get_user" in demo
    assert "OBSERVATION (is_error=True): not found" in demo


def test_render_demo_caps_observation() -> None:
    step = Step(
        action=Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"cmd": "cat big.log"}),
        observation=Observation(content="X" * 5000),
    )
    uncapped = render_demo(step)
    assert "X" * 5000 in uncapped  # no cap by default

    capped = render_demo(step, max_observation_chars=2000)
    assert "X" * 2000 in capped
    assert "X" * 2001 not in capped  # truncated at the cap
    assert "[+3000 chars]" in capped  # marker reports how much was dropped
    assert "bash" in capped  # the (state, action) head is preserved


def test_build_env_prompt_composes_all_parts() -> None:
    state = EnvState(structured={"cart": []})
    action = Action(kind=ActionKind.TOOL_CALL, name="add", arguments={"sku": "A1"})
    demo = Step(
        action=Action(kind=ActionKind.TOOL_CALL, name="add", arguments={"sku": "B2"}),
        observation=Observation(content="added B2"),
    )
    system, user = build_env_prompt("BASE", "buy stuff", state, action, demos=[demo])
    assert system == "BASE"
    assert "TASK:\nbuy stuff" in user
    assert "SIMILAR PAST EXAMPLES:" in user and "added B2" in user
    assert "AGENT ACTION:" in user and "add" in user
    assert "(start of session)" in user  # no history given


def test_build_env_prompt_handles_empty_optional_blocks() -> None:
    system, user = build_env_prompt(
        "BASE", None, EnvState(), Action(kind=ActionKind.MESSAGE, content="hi")
    )
    assert "TASK:\n(none)" in user
    assert "(no similar past examples)" in user
    assert "(start of session)" in user
    assert "scratchpad: (empty)" in user


def test_encode_action_is_command_only() -> None:
    action = Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "pytest -q"})
    key = encode_action(action)
    assert "bash" in key and "pytest -q" in key
    assert "STATE:" not in key and "ACTION kind=" not in key
