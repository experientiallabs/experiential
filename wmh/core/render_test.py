"""Tests for the canonical rendering helpers."""

from __future__ import annotations

from wmh.core.render import (
    OUTPUT_CONTRACT,
    build_env_prompt,
    encode_action,
    encode_state_action,
    output_contract,
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


def test_build_env_prompt_defaults_are_byte_identical_to_v1() -> None:
    """Pin the default rendering: prebuilt models must keep serving unchanged.

    `knowledge`/`reasoning` are opt-in; with both off, the prompt must be byte-for-byte what it
    was before they existed. If this test breaks, existing `examples/*/models/` artifacts (and
    GEPA-optimized prompts evolved against this shape) silently degrade — change with care and
    regenerate/version the shipped models.
    """
    system, user = build_env_prompt(
        "BASE", "t", EnvState(), Action(kind=ActionKind.MESSAGE, content="hi")
    )
    assert system == "BASE"
    assert user == (
        "TASK:\nt\n\n"
        "INTERACTION HISTORY:\n(start of session)\n\n"
        "SIMILAR PAST EXAMPLES:\n(no similar past examples)\n\n"
        "CURRENT ENV STATE:\n  structured: {}\n  scratchpad: (empty)\n\n"
        "AGENT ACTION:\nmessage: hi\n\n" + OUTPUT_CONTRACT
    )


def test_build_env_prompt_knowledge_section_sits_between_task_and_history() -> None:
    _, user = build_env_prompt(
        "BASE",
        "t",
        EnvState(),
        Action(kind=ActionKind.MESSAGE, content="hi"),
        knowledge="- gate: modifying a booking requires auth",
    )
    knowledge_at = user.find("KNOWLEDGE BASE")
    assert user.find("TASK:") < knowledge_at < user.find("INTERACTION HISTORY:")
    assert "- gate: modifying a booking requires auth" in user


def test_build_env_prompt_reasoning_contract_deliberates_first() -> None:
    _, plain = build_env_prompt(
        "BASE", "t", EnvState(), Action(kind=ActionKind.MESSAGE, content="hi")
    )
    _, reasoned = build_env_prompt(
        "BASE", "t", EnvState(), Action(kind=ActionKind.MESSAGE, content="hi"), reasoning=True
    )
    assert '"reasoning"' not in plain and '"kb_note"' not in plain
    assert reasoned.find('"reasoning"') < reasoned.find('"output"')  # deliberation decoded first
    assert '"kb_note"' in reasoned
    assert '"ground_query"' not in reasoned  # only offered when a grounder is active


def test_output_contract_grounding_adds_ground_query() -> None:
    assert '"ground_query"' not in output_contract(reasoning=True)
    assert '"ground_query"' in output_contract(reasoning=True, grounding=True)
    assert output_contract() == OUTPUT_CONTRACT  # base contract is untouched


def test_build_env_prompt_confidence_off_is_byte_identical() -> None:
    # The lever must be invisible when off: explicit False renders exactly the default prompt.
    args = ("BASE", "t", EnvState(), Action(kind=ActionKind.MESSAGE, content="hi"))
    assert build_env_prompt(*args, confidence=False, confidence_why=False) == build_env_prompt(
        *args
    )
    assert '"confidence"' not in build_env_prompt(*args)[1]


def test_output_contract_confidence_sits_after_is_error() -> None:
    # Ordered decoding: the self-assessment must be conditioned on the emitted answer (D75).
    for contract in (
        output_contract(confidence=True),
        output_contract(reasoning=True, confidence=True),
        output_contract(reasoning=True, grounding=True, confidence=True),
    ):
        assert (
            contract.find('"is_error"')
            < contract.find('"confidence"')
            < contract.find('"state_note"')
        )
    assert '"confidence"' not in output_contract(reasoning=True)


def test_output_contract_confidence_why_justifies_before_the_number() -> None:
    contract = output_contract(confidence=True, confidence_why=True)
    assert contract.find('"confidence_why"') < contract.find('"confidence":')
    assert '"confidence_why"' not in output_contract(confidence=True)
    # The justification is meaningless without the number: why alone is a no-op.
    assert output_contract(confidence_why=True) == OUTPUT_CONTRACT


def test_render_demo_caps_observation_with_honest_marker() -> None:
    step = Step(
        task="t",
        state_before=EnvState(),
        action=Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "ls"}),
        observation=Observation(content="x" * 3000, is_error=False),
    )
    capped = render_demo(step, max_observation_chars=2000)
    assert "x" * 2000 in capped and "… [+1000 chars]" in capped
    assert render_demo(step) == render_demo(step, max_observation_chars=None)  # default unchanged


def test_build_env_prompt_threads_the_demo_cap() -> None:
    step = Step(
        task="t",
        state_before=EnvState(),
        action=Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "ls"}),
        observation=Observation(content="y" * 500, is_error=False),
    )
    _, user = build_env_prompt(
        "BASE",
        "t",
        EnvState(),
        step.action,
        demos=[step],
        max_retrieved_observation_chars=100,
    )
    assert "… [+400 chars]" in user
