"""Which per-message effort governs the reply."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.anthropic_protocol.per_message_effort import (
    PER_MESSAGE_EFFORT_DISCLOSURE,
    TurnFacts,
    governing_effort,
    take_system_turn_effort,
    with_governing_effort,
)

_USER = TurnFacts(role="user", new_input=True, effort=None)
_TOOL_RESULTS = TurnFacts(role="user", new_input=False, effort=None)
_ASSISTANT = TurnFacts(role="assistant", new_input=False, effort=None)


def _effort(value: str) -> TurnFacts:
    """Return a system turn that sets ``value``."""
    return TurnFacts(role="system", new_input=False, effort=value)


def test_no_effort_turn_governs_nothing() -> None:
    """Without a system turn that sets one, no per-message level governs."""
    assert governing_effort([_USER, _ASSISTANT, _USER]) is None


def test_a_turn_after_new_user_input_governs_that_reply() -> None:
    """A level set right after new user input governs the reply to it."""
    assert governing_effort([_USER, _effort("low")]) == "low"


def test_a_turn_after_an_assistant_starts_at_the_next_user_input() -> None:
    """A level set after an assistant turn waits for the next user input."""
    assert governing_effort([_USER, _ASSISTANT, _effort("low")]) is None
    assert governing_effort([_USER, _ASSISTANT, _effort("low"), _USER]) == "low"


def test_tool_results_are_not_new_input() -> None:
    """A user turn of tool results only does not start a pending level."""
    assert governing_effort([_USER, _ASSISTANT, _TOOL_RESULTS, _effort("low")]) is None


def test_a_level_stays_until_a_later_one_replaces_it() -> None:
    """A level stays in effect until a later level takes effect."""
    turns = [_USER, _effort("low"), _ASSISTANT, _USER, _ASSISTANT, _TOOL_RESULTS]
    assert governing_effort(turns) == "low"
    assert governing_effort([*turns[:4], _effort("max"), *turns[4:]]) == "max"


def test_the_later_turn_wins_when_two_take_effect_together() -> None:
    """Of two levels starting at the same user turn, the later turn wins."""
    turns = [_USER, _ASSISTANT, _effort("low"), _USER, _effort("high")]
    assert governing_effort(turns) == "high"


def test_a_pending_level_does_not_override_one_in_effect() -> None:
    """A level that has not taken effect leaves the current one governing."""
    turns = [_USER, _effort("low"), _ASSISTANT, _TOOL_RESULTS, _effort("high")]
    assert governing_effort(turns) == "low"


def test_consecutive_system_turns_are_judged_together() -> None:
    """A text system turn between the user turn and the level does not delay it."""
    text_turn = TurnFacts(role="system", new_input=False, effort=None)
    assert governing_effort([_USER, text_turn, _effort("low")]) == "low"


def test_folding_replaces_only_the_effort() -> None:
    """Folding swaps the effort and keeps every other top-level key."""
    assert with_governing_effort({"effort": "high", "format": {"x": 1}}, "low") == (
        {"effort": "low", "format": {"x": 1}},
        (PER_MESSAGE_EFFORT_DISCLOSURE,),
    )
    assert with_governing_effort(None, "low") == (
        {"effort": "low"},
        (PER_MESSAGE_EFFORT_DISCLOSURE,),
    )


def test_folding_without_a_change_discloses_nothing() -> None:
    """An unchanged or absent level folds to the caller's object, undisclosed."""
    assert with_governing_effort({"effort": "low"}, "low") == ({"effort": "low"}, ())
    assert with_governing_effort(None, None) == (None, ())


def test_a_body_without_system_turn_effort_is_returned_unchanged() -> None:
    """A body with no system-turn output_config is returned as the same object."""
    payload: JsonObject = {"messages": [{"role": "user", "content": "hi", "output_config": {}}]}
    taken, turns = take_system_turn_effort(payload)
    assert taken is payload
    assert turns.empty == frozenset()


def test_system_turn_effort_is_consumed_and_empty_turns_are_named() -> None:
    """A system turn's output_config is consumed; only unmarked empty turns drop."""
    payload: JsonObject = {
        "messages": [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t"}]},
            {"role": "system", "content": [], "output_config": {"effort": "low"}},
            {"role": "system", "content": [{"type": "text", "text": "", "cache_control": {}}]},
        ]
    }
    payload_messages = payload["messages"]
    assert isinstance(payload_messages, list)
    taken, turns = take_system_turn_effort(payload)
    assert taken["messages"] == [
        payload_messages[0],
        {"role": "system", "content": []},
        payload_messages[2],
    ]
    # A marked empty block is left to its typed refusal, not dropped.
    assert turns.empty == frozenset({1})
    # Tool results alone are not new user input.
    assert turns.facts[0].new_input is False
    assert turns.facts[1].effort == "low"


def test_resolution_is_one_pass_over_a_long_run_of_system_turns() -> None:
    """Many consecutive effort turns resolve in linear time (no rescans)."""
    turns = [_USER, _ASSISTANT, *[_effort("low")] * 200_000, _effort("max"), _USER]
    assert governing_effort(turns) == "max"
