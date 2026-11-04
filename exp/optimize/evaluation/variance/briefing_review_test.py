"""Offline tests for the model briefing leak reviewer, wired through the scenario author."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.variance.authors import SCENARIO_TOOL, ScenarioDraft, author_scenarios
from exp.optimize.evaluation.variance.authors_test import (
    _BEFORE,
    _scenario,
    _setup,
    _state,
    _tool_reply,
    _wire,
)
from exp.optimize.evaluation.variance.briefing_review import (
    DEFAULT_REVIEW_VARIANTS,
    REVIEW_TOOL,
    REVIEW_VARIANTS,
    BriefingReviewContext,
    review_payload,
    validate_review,
)
from exp.optimize.evaluation.variance.contracts import (
    ActorView,
    DecisionState,
    Scenario,
    ScenarioLimits,
)
from exp.runtime.models.providers.transport import ScriptedJsonTransport

_TASK = "Please refund order o-1."
_LIMITS = ScenarioLimits(tools=frozenset({"get_order", "refund"}), remaining_calls=4)


def _three_word_phrases(text: str) -> set[str]:
    """The three-word phrases the removed lexical check compared, kept to show what it missed."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[index : index + 3]) for index in range(len(words) - 2)}


def _briefed(text: str) -> JsonObject:
    """The late-delivery scenario (deny is right) started by a fresh briefing."""
    return {
        **_scenario("outside", 45, "deny"),
        "initialization": "fresh_briefing",
        "briefing": {"text": text, "entitled_paths": []},
    }


def _set(text: str) -> JsonObject:
    """A two-scenario set whose second scenario carries the briefing."""
    return {
        "scenarios": [_scenario("inside", 10, "refund"), _briefed(text)],
        "limitations": ["Clock coherence with the conversation is unchecked."],
    }


def _clean(quote: str, refs: list[str]) -> JsonObject:
    """A clean review grounding one claim."""
    return {
        "verdict": "clean",
        "grounded": [{"briefing_quote": quote, "view_refs": refs}],
        "limitations": ["Only the supplied view was read."],
    }


def _leak(quote: str, kind: str = "intended_option") -> JsonObject:
    """A review reporting one finding."""
    return {
        "verdict": "leaks",
        "findings": [
            {
                "kind": kind,
                "briefing_quote": quote,
                "view_refs": ["observed:0"],
                "explanation": "Telling the actor the request is out of time tells it to deny.",
            }
        ],
        "limitations": ["Only the supplied view was read."],
    }


def _sent(transport: ScriptedJsonTransport, index: int) -> JsonObject:
    """The parsed user payload of one recorded request."""
    messages = transport.requests[index].payload["messages"]
    assert isinstance(messages, list) and isinstance(messages[1], dict)
    parsed = json.loads(str(messages[1]["content"]))
    assert isinstance(parsed, dict)
    return parsed


def _view(*observed: str) -> ActorView:
    """The actor's view at the start."""
    return ActorView(instruction=_TASK, observed=(_TASK, *observed))


def test_a_paraphrased_leak_the_word_check_missed_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A briefing hinting at the denial in other words is caught by the reviewer, then replays."""
    project, settings = _setup(tmp_path, monkeypatch)
    decision = DecisionState.model_validate(_state())
    text = "The customer is asking well after the period in which money can be returned."
    deny = next(option.action for option in decision.options if option.option_id == "deny")
    # The removed check compared three-word phrases; this paraphrase shares none with the action.
    assert not _three_word_phrases(deny) & _three_word_phrases(text)
    quote = "well after the period in which money can be returned"
    transport = _wire(
        monkeypatch,
        _tool_reply(SCENARIO_TOOL, _set(text)),
        _tool_reply(REVIEW_TOOL, _leak(quote)),
        _tool_reply(REVIEW_TOOL, _leak(quote, "edited_value")),
        _tool_reply(REVIEW_TOOL, _clean("The customer is asking", ["instruction"])),
    )

    def draw() -> ScenarioDraft:
        """Buy or replay the set."""
        return author_scenarios(
            project,
            settings,
            decision,
            _BEFORE,
            _LIMITS,
            identity="briefed-paraphrase",
            code_revision="test",
            view=_view(),
        )

    draft = draw()
    assert not draft.accepted
    assert [review.variant for review in draft.reviews] == list(DEFAULT_REVIEW_VARIANTS)
    reasons = [problem.reason for problem in draft.problems]
    assert len(reasons) == 2 and all(p.scenario_id == "outside" for p in draft.problems)
    assert "found intended_option" in reasons[0] and "found edited_value" in reasons[1]
    assert len(transport.requests) == 4
    review = _sent(transport, 1)
    assert review["briefing"] == {"text": text, "entitled_paths": []}
    options = review["options"]
    assert isinstance(options, list)
    assert [isinstance(row, dict) and row["intended"] for row in options] == [False, True]
    assert review["edited_paths"] == [
        {"path": "/clock/days_since_delivery", "before": 10, "after": 45}
    ]
    assert draw() == draft and len(transport.requests) == 4


def test_shared_words_the_actor_had_seen_are_not_a_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observed wording that overlaps the option is accepted when every stance is clean."""
    project, settings = _setup(tmp_path, monkeypatch)
    decision = DecisionState.model_validate(_state())
    seen = "The customer wrote: reply that the refund window has closed is what the shop said."
    text = "The customer wrote that the refund window has closed is what the shop said."
    deny = next(option.action for option in decision.options if option.option_id == "deny")
    # The removed check would have refused this: it shares phrases with the intended action that
    # the instruction does not, although the actor had read them.
    assert _three_word_phrases(deny) & _three_word_phrases(text) - _three_word_phrases(_TASK)
    clean = _clean("the refund window has closed is what the shop said", ["observed:1"])
    transport = _wire(
        monkeypatch,
        _tool_reply(SCENARIO_TOOL, _set(text)),
        *(_tool_reply(REVIEW_TOOL, clean) for _ in REVIEW_VARIANTS),
    )

    draft = author_scenarios(
        project,
        settings,
        decision,
        _BEFORE,
        _LIMITS,
        identity="briefed-shared",
        code_revision="test",
        view=_view(seen),
    )

    assert draft.accepted and draft.problems == (), draft.problems
    assert all(review.answer is not None for review in draft.reviews)
    assert len(transport.requests) == 1 + len(REVIEW_VARIANTS)


def test_a_miscited_review_is_repaired_and_an_unusable_one_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A quote outside the briefing is shown back once; a stance with no usable answer refuses."""
    project, settings = _setup(tmp_path, monkeypatch)
    decision = DecisionState.model_validate(_state())
    text = "Pick up the order request where it stands."
    invented = _clean("The order was delivered ten days ago", ["observed:9"])
    fixed = _clean("Pick up the order request", ["instruction"])
    transport = _wire(
        monkeypatch,
        _tool_reply(SCENARIO_TOOL, _set(text)),
        _tool_reply(REVIEW_TOOL, invented),
        _tool_reply(REVIEW_TOOL, fixed),
        _tool_reply(REVIEW_TOOL, fixed),
        _tool_reply(REVIEW_TOOL, invented),
        _tool_reply(REVIEW_TOOL, invented),
    )

    draft = author_scenarios(
        project,
        settings,
        decision,
        _BEFORE,
        _LIMITS,
        identity="briefed-repair",
        code_revision="test",
        view=_view(),
    )

    first, second, third = draft.reviews
    assert first.answer is not None and first.repairs == 1
    assert second.answer is not None and second.repairs == 0
    assert third.answer is None and third.failure is not None
    assert "is not text of the briefing" in third.failure and "observed:9" in third.failure
    repair = _sent(transport, 2)
    assert "is not text of the briefing" in str(repair["previous_draft_problems"])
    assert not draft.accepted
    assert [p.reason.split(":")[0] for p in draft.problems] == [
        "briefing leak review adversarial-actor is unusable"
    ]
    assert len(transport.requests) == 6


def test_a_resumed_set_and_a_structurally_refused_set_buy_no_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reviews run only for fresh briefings of a set that passed every structural check."""
    project, settings = _setup(tmp_path, monkeypatch)
    decision = DecisionState.model_validate(_state())
    flat: JsonObject = {
        "scenarios": [
            _scenario("inside", 10, "deny"),
            _briefed("Pick up the order request where it stands."),
        ],
        "limitations": ["x"],
    }
    transport = _wire(monkeypatch, _tool_reply(SCENARIO_TOOL, flat))

    draft = author_scenarios(
        project,
        settings,
        decision,
        _BEFORE,
        _LIMITS,
        identity="briefed-flat",
        code_revision="test",
        view=_view(),
    )

    assert not draft.accepted and draft.reviews == ()
    assert len(transport.requests) == 1


def test_citations_are_the_only_mechanical_check() -> None:
    """Quotes must be briefing text and locators supplied; findings themselves are not judged."""
    decision = DecisionState.model_validate(_state())
    scenario = Scenario.model_validate(_briefed("Handle the\nlate   request."))
    context = BriefingReviewContext(
        scenario=scenario, decision=decision, view=_view("seen"), before=_BEFORE
    )
    payload = review_payload(context)
    assert payload["citable_view_locators"] == ["instruction", "observed:0", "observed:1"]
    assert payload["citable_state_paths"] == [
        "/clock/days_since_delivery",
        "/clock",
        "/policy/refund_window_days",
    ]
    assert validate_review(_clean("handle the late request", ["observed:1"]), context)
    with pytest.raises(ValueError, match="state paths"):
        validate_review(
            {
                **_leak("late request"),
                "findings": [
                    {
                        "kind": "steering",
                        "briefing_quote": "late request",
                        "state_paths": ["/order/secret"],
                        "explanation": "x",
                    }
                ],
            },
            context,
        )
    with pytest.raises(ValueError, match="verdict"):
        validate_review({"verdict": "leaks", "limitations": ["x"]}, context)
