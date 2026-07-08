"""Tests for the canonical rendering helpers."""

from __future__ import annotations

from wmh.core.render import (
    build_env_prompt,
    encode_state_action,
    render_action,
    render_agent_messages,
    render_demo,
    render_harness,
    render_json,
)
from wmh.core.types import (
    Action,
    ActionKind,
    EnvState,
    HarnessContext,
    Observation,
    Step,
    ToolDefinition,
)


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


_HARNESS = HarnessContext(
    system_prompt="You are an expert coding assistant operating inside pi.",
    tools=[
        ToolDefinition(
            name="bash",
            description="Run a shell command",
            parameters={"type": "object", "properties": {"command": {"type": "string"}}},
        ),
        ToolDefinition(name="submit"),
    ],
)


def test_render_harness_lists_system_prompt_and_tools() -> None:
    text = render_harness(_HARNESS)
    assert "You are an expert coding assistant operating inside pi." in text
    assert "bash" in text and "Run a shell command" in text
    assert '"command"' in text  # tool parameter schemas are preserved verbatim
    assert "submit" in text


def test_env_prompt_never_includes_the_harness() -> None:
    # The world model simulates the ENVIRONMENT's response to an action; the agent's context
    # assembly (system prompt, tools) is the harness's job and stays agent-side.
    system, user = build_env_prompt(
        "BASE",
        "build an app",
        EnvState(),
        Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "ls"}),
    )
    assert system == "BASE"
    assert "AGENT HARNESS" not in user and "SYSTEM PROMPT" not in user


def test_render_harness_labels_inferred_contexts() -> None:
    from wmh.core.types import HarnessSource

    inferred = _HARNESS.model_copy(update={"source": HarnessSource.INFERRED})
    assert "reconstructed from the corpus' observed behavior" in render_harness(inferred)
    assert "reconstructed" not in render_harness(_HARNESS)  # captured stays unlabeled


def test_render_agent_messages_is_verbatim() -> None:
    system, user = render_agent_messages(_HARNESS, "build a python airbnb clone")
    assert system.startswith("You are an expert coding assistant operating inside pi.")
    assert '"name": "bash"' in system or '"name":"bash"' in system
    assert user == "build a python airbnb clone"  # the user message is the task, untouched
