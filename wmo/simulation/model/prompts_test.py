"""Tests for the engine-facing prompt entry point and the shipped base prompt.

The rendering itself is pinned in `wmo/common/core/render_test.py`; what matters here is that this
adapter feeds the shared renderer the live session's task, state, and history unchanged, so GEPA
evolves prompts against exactly what serving assembles.
"""

from __future__ import annotations

from wmo.common.core.render import build_env_prompt as render_env_prompt
from wmo.common.core.types import Action, ActionKind, EnvState, Observation, Session, Step
from wmo.simulation.model.prompts import BASE_ENV_PROMPT, build_env_prompt


def _session() -> Session:
    return Session(
        id="s1",
        task="delete the temp file",
        state=EnvState(structured={"cwd": "/tmp"}, scratchpad="created foo.txt"),
        history=[
            Step(
                state_before=EnvState(structured={"cwd": "/tmp"}),
                action=Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"cmd": "ls"}),
                observation=Observation(content="foo.txt"),
            )
        ],
    )


def _action() -> Action:
    return Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"cmd": "rm foo.txt"})


def test_the_base_prompt_is_the_system_message() -> None:
    system, _user = build_env_prompt(BASE_ENV_PROMPT, _session(), _action(), demos=[])

    assert system == BASE_ENV_PROMPT


def test_the_user_message_carries_the_task_state_and_history() -> None:
    _system, user = build_env_prompt("base", _session(), _action(), demos=[])

    assert "delete the temp file" in user
    assert "/tmp" in user
    assert "created foo.txt" in user
    assert "foo.txt" in user  # the prior observation
    assert "rm foo.txt" in user  # the incoming action
    assert "(no similar past examples)" in user


def test_it_renders_exactly_what_the_shared_renderer_does() -> None:
    # This module is an adapter, not a second assembly: any divergence would mean the optimizer
    # and the serving engine were tuning and running different prompts.
    session = _session()
    action = _action()
    demos = [
        Step(
            state_before=EnvState(),
            action=Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"cmd": "rm bar.txt"}),
            observation=Observation(content=""),
        )
    ]

    assert build_env_prompt("base", session, action, demos, knowledge="rm prints nothing") == (
        render_env_prompt(
            "base",
            session.task,
            session.state,
            action,
            history=session.history,
            demos=demos,
            knowledge="rm prints nothing",
        )
    )


def test_the_agentic_flags_pass_through() -> None:
    session, action = _session(), _action()

    plain = build_env_prompt("base", session, action, demos=[])
    reasoning = build_env_prompt("base", session, action, demos=[], reasoning=True)
    confident = build_env_prompt("base", session, action, demos=[], confidence=True)

    assert reasoning != plain
    assert confident != plain
    assert reasoning == render_env_prompt(
        "base",
        session.task,
        session.state,
        action,
        history=session.history,
        demos=[],
        reasoning=True,
    )


def test_the_base_prompt_targets_the_three_measured_failure_modes() -> None:
    # The prompt is a tuned artifact, and these are the behaviors it was written to fix; a rewrite
    # that drops one should have to say so here.
    prompt = BASE_ENV_PROMPT.lower()

    assert "you are the environment" in prompt  # never answer as an assistant
    assert "print nothing" in prompt  # do not narrate success for silent commands
    assert "success vs. error" in prompt  # decide the outcome from the state
    assert "verbatim" in prompt  # reuse concrete values from state and history
