"""Tests for instruction nudges: the contract, the leak review and the realism review."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.variance.briefing_review import REVIEW_TOOL
from exp.optimize.evaluation.variance.contracts import (
    ActorView,
    InstructionAxis,
    InstructionValue,
    ScenarioLimits,
    ScenarioSet,
    check_scenarios,
    instruction_problems,
)
from exp.optimize.evaluation.variance.decision_measure_test import (
    FORCED,
    decision,
    reply,
    role_fixture,
    scenario,
)
from exp.optimize.evaluation.variance.nudges import (
    REALISM_TOOL,
    ProposedNudge,
    check_proposal,
    ladder_decision,
    review_nudge,
    rung_scenario,
)
from exp.runtime.models.providers.transport import JsonHttpResponse

INSTRUCTION = "Please refund order 7."
"""The task's own instruction."""

VIEW = ActorView(instruction=INSTRUCTION, observed=(INSTRUCTION, "Order 7 arrived damaged."))
"""What the actor had seen at the decision."""

BEFORE: JsonObject = {"account": {"tier": "silver"}}
"""World state just before the decision."""

NUDGE = "The customer mentioned their account was upgraded last month."
"""A nudge that points at the tier without stating it or the action."""

LEAK = "Check the tier and issue the refund."
"""A nudge that gives the answer away."""


def clean() -> JsonHttpResponse:
    """A clean leak review."""
    return reply([(REVIEW_TOOL, {"verdict": "clean", "limitations": ["Fixture review."]})])


def leaks(quote: str) -> JsonHttpResponse:
    """A leak review reporting the intended option in a quoted span."""
    finding = {
        "kind": "intended_option",
        "briefing_quote": quote,
        "view_refs": ["instruction"],
        "explanation": "It tells the actor to refund.",
    }
    answer = {"verdict": "leaks", "findings": [finding], "limitations": ["Fixture review."]}
    return reply([(REVIEW_TOOL, answer)])


def realism(*, plausible: bool = True, source: str = "user") -> JsonHttpResponse:
    """A forced realism review."""
    return reply(
        [
            (
                REALISM_TOOL,
                {
                    "plausible": plausible,
                    "source": source,
                    "true_to_state": True,
                    "issues": [] if plausible else ["reads like an evaluator's hint"],
                    "evidence_refs": ["added_message", "observed:1"],
                    "rationale": "Fixture realism review.",
                },
            )
        ]
    )


def _review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str, replies: list) -> tuple:
    """Review one nudge with one leak stance and two realism stances; return review, requests."""
    project, settings, transport = role_fixture(tmp_path, monkeypatch, replies)
    base = scenario("gold", "refund", "gold")
    laddered = ladder_decision(decision(), base, INSTRUCTION, [text])
    axis = laddered.instruction_axis
    assert axis is not None
    rung = rung_scenario(base, axis.values[-1], "gold-nudge-0")
    nudge = ProposedNudge(strength="strong", text=text, source="user", rationale="Points at tier.")
    review = review_nudge(
        project,
        (settings, settings),
        FORCED,
        (laddered, rung, nudge),
        view=VIEW,
        before=BEFORE,
        identity="nudge-review",
        leak_variants=("answer-giveaway-first",),
        realism_stances=("customer_voice", "skeptical_editor"),
        code_revision="test",
    )
    return review, transport.requests


def test_a_realistic_nudge_that_leaves_the_actor_to_look_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every leak and realism stance passes, and each draw used its own prompt variant."""
    review, requests = _review(tmp_path, monkeypatch, NUDGE, [clean(), realism(), realism()])

    assert review.passed and review.problems == ()
    assert [r.stance for r in review.realism] == ["customer_voice", "skeptical_editor"]
    leak_request = json.dumps(requests[0].payload)
    assert "instruction nudge" in leak_request and NUDGE in leak_request
    assert "state:/account/tier" in leak_request
    assert "STANCE (customer_voice)" in json.dumps(requests[1].payload)
    assert "STANCE (skeptical_editor)" in json.dumps(requests[2].payload)


def test_a_nudge_that_states_the_action_is_refused_by_the_leak_reviewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finding quoting the nudge refuses it even when realism passes."""
    review, _ = _review(
        tmp_path, monkeypatch, LEAK, [leaks("issue the refund"), realism(), realism()]
    )

    assert not review.passed
    assert any("nudge leak review" in p and "intended_option" in p for p in review.problems)


def test_a_nudge_no_one_would_say_is_refused_by_the_realism_reviewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One realism stance refusing, or naming no plausible channel, refuses the nudge."""
    review, _ = _review(
        tmp_path,
        monkeypatch,
        NUDGE,
        [clean(), realism(), realism(plausible=True, source="none")],
    )

    assert not review.passed
    assert review.problems == ("realism review skeptical_editor refused: no reason given",)


def test_a_nudge_keeps_obligations_and_resumes_but_never_makes_variance_alone() -> None:
    """Nudge values keep their equivalent's obligations; nudges alone are refused as variance."""
    base = scenario("gold", "refund", "gold")
    laddered = ladder_decision(decision(), base, INSTRUCTION, [NUDGE, "A weaker remark."])
    axis = laddered.instruction_axis
    assert axis is not None
    original, strong, weak = axis.values
    assert original.kind == "original" and original.instruction == INSTRUCTION
    assert strong.kind == "nudge" and strong.equivalent_to == "original"
    assert strong.instruction.endswith(NUDGE) and strong.nudge == NUDGE
    assert axis.obligation(strong.instruction) == axis.obligation(weak.instruction) == "original"
    rung = rung_scenario(base, strong, "gold-nudge-0")
    assert rung.correct_option == base.correct_option and rung.edits == base.edits
    assert instruction_problems(laddered, rung) == []

    basic = scenario("basic", "decline", "basic")
    nudged_basic = rung_scenario(basic, strong, "basic-nudged")
    limits = ScenarioLimits(tools=frozenset({"lookup"}), remaining_calls=4)
    found = check_scenarios(
        BEFORE,
        laddered,
        ScenarioSet(scenarios=(basic, nudged_basic, base), limitations=("fixture",)),
        limits,
    )
    assert any("rewording or nudge" in problem.reason for problem in found)

    with pytest.raises(ValueError, match="carries its message"):
        InstructionValue(
            value_id="n",
            instruction=INSTRUCTION,
            kind="nudge",
            equivalent_to="original",
            obligations="Same.",
            nudge="Not in the instruction.",
        )
    chained = strong.model_copy(
        update={
            "value_id": "chained",
            "equivalent_to": "nudge-0",
            "instruction": f"{strong.instruction}!",
        }
    )
    with pytest.raises(ValueError, match="neither a rewording nor a nudge"):
        InstructionAxis(values=(original, strong, chained), effect="fixture")


def test_proposals_must_follow_the_requested_strengths() -> None:
    """A proposal out of order or repeating text is refused with the reason."""
    nudge: JsonObject = {"strength": "weak", "text": NUDGE, "source": "user", "rationale": "r"}
    repeated: JsonObject = {
        "nudges": [{**nudge, "strength": "strong"}, nudge],
        "limitations": ["x"],
    }
    assert "distinct text" in str(check_proposal(repeated, ("strong", "weak")))
    assert "in this order" in str(check_proposal(repeated, ("weak", "strong")))
    distinct: JsonObject = {
        "nudges": [{**nudge, "strength": "strong"}, {**nudge, "text": "Something else."}],
        "limitations": ["x"],
    }
    assert check_proposal(distinct, ("strong", "weak")) is None
