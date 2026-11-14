"""Which per-message effort governs the reply."""

from __future__ import annotations

from exp.runtime.anthropic_protocol.per_message_effort import (
    PER_MESSAGE_EFFORT_DISCLOSURE,
    TurnFacts,
    governing_effort,
    with_governing_effort,
)

_USER = TurnFacts(role="user", new_input=True, effort=None)
_TOOL_RESULTS = TurnFacts(role="user", new_input=False, effort=None)
_ASSISTANT = TurnFacts(role="assistant", new_input=False, effort=None)


def _effort(value: str) -> TurnFacts:
    return TurnFacts(role="system", new_input=False, effort=value)


def test_no_effort_turn_governs_nothing() -> None:
    assert governing_effort([_USER, _ASSISTANT, _USER]) is None


def test_a_turn_after_new_user_input_governs_that_reply() -> None:
    assert governing_effort([_USER, _effort("low")]) == "low"


def test_a_turn_after_an_assistant_starts_at_the_next_user_input() -> None:
    assert governing_effort([_USER, _ASSISTANT, _effort("low")]) is None
    assert governing_effort([_USER, _ASSISTANT, _effort("low"), _USER]) == "low"


def test_tool_results_are_not_new_input() -> None:
    assert governing_effort([_USER, _ASSISTANT, _TOOL_RESULTS, _effort("low")]) is None


def test_a_level_stays_until_a_later_one_replaces_it() -> None:
    turns = [_USER, _effort("low"), _ASSISTANT, _USER, _ASSISTANT, _TOOL_RESULTS]
    assert governing_effort(turns) == "low"
    assert governing_effort([*turns[:4], _effort("max"), *turns[4:]]) == "max"


def test_the_later_turn_wins_when_two_take_effect_together() -> None:
    turns = [_USER, _ASSISTANT, _effort("low"), _USER, _effort("high")]
    assert governing_effort(turns) == "high"


def test_a_pending_level_does_not_override_one_in_effect() -> None:
    turns = [_USER, _effort("low"), _ASSISTANT, _TOOL_RESULTS, _effort("high")]
    assert governing_effort(turns) == "low"


def test_consecutive_system_turns_are_judged_together() -> None:
    text_turn = TurnFacts(role="system", new_input=False, effort=None)
    assert governing_effort([_USER, text_turn, _effort("low")]) == "low"


def test_folding_replaces_only_the_effort() -> None:
    assert with_governing_effort({"effort": "high", "format": {"x": 1}}, "low") == (
        {"effort": "low", "format": {"x": 1}},
        (PER_MESSAGE_EFFORT_DISCLOSURE,),
    )
    assert with_governing_effort(None, "low") == (
        {"effort": "low"},
        (PER_MESSAGE_EFFORT_DISCLOSURE,),
    )


def test_folding_without_a_change_discloses_nothing() -> None:
    assert with_governing_effort({"effort": "low"}, "low") == ({"effort": "low"}, ())
    assert with_governing_effort(None, None) == (None, ())
