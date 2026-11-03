"""Tests for authored decision states, scenarios and their structural gates."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.optimize.evaluation.variance.contracts import (
    Criterion,
    DecisionOption,
    DecisionState,
    JudgeSelection,
    Scenario,
    ScenarioLimits,
    ScenarioSet,
    StateEdit,
    VariationAxis,
    WitnessStep,
    apply_edits,
    check_scenarios,
    resolve_pointer,
)

_BEFORE: JsonObject = {
    "policy": {"fixed": True, "refundable": True},
    "order": {"amount": 420},
    "audit": [],
}
_LIMITS = ScenarioLimits(tools=frozenset({"read_policy", "read_order"}), remaining_calls=3)


def _decision(**changes: JsonValue) -> DecisionState:
    """A refund decision whose right option depends on the refund policy."""
    base: JsonObject = {
        "step": 3,
        "decision": "Resolve the customer's cancellation",
        "options": [
            {"option_id": "refund", "action": "Issue a refund"},
            {"option_id": "credit", "action": "Issue store credit"},
        ],
        "relevant_paths": ["/policy/refundable"],
        "axes": [
            {
                "path": "/policy/refundable",
                "values": [True, False],
                "effect": "Only a refundable order may be refunded.",
            }
        ],
        "invariants": ["/policy/fixed", "/audit"],
        "criteria": [{"criterion_id": "right_action", "text": "Was the right option taken?"}],
        "evidence_refs": ["point:3"],
    }
    return DecisionState.model_validate({**base, **changes})


def _scenario(identity: str, value: JsonValue, option: str, **changes: JsonValue) -> Scenario:
    """One scenario editing the refund axis, witnessed by reading the policy."""
    base: JsonObject = {
        "scenario_id": identity,
        "edits": [{"path": "/policy/refundable", "value": value}],
        "correct_option": option,
        "deciding_paths": ["/policy/refundable"],
        "witness": [{"tool": "read_policy", "arguments": {}, "reveals": ["/policy/refundable"]}],
        "rationale": "The policy decides.",
        "criteria": ["right_action"],
    }
    return Scenario.model_validate({**base, **changes})


def _reasons(*scenarios: Scenario, decision: DecisionState | None = None) -> list[str]:
    """Run the gates and return every problem reason."""
    found = check_scenarios(
        _BEFORE,
        decision or _decision(),
        ScenarioSet(scenarios=scenarios, limitations=("simulated",)),
        _LIMITS,
    )
    return [problem.reason for problem in found]


def test_pointers_resolve_and_edits_copy() -> None:
    """Pointers read nested values; edits never mutate the source state."""
    assert resolve_pointer(_BEFORE, "/policy/fixed") == (True, True)
    assert resolve_pointer(_BEFORE, "/policy/missing") == (False, None)
    edited = apply_edits(_BEFORE, (StateEdit(path="/policy/refundable", value=False),))
    assert edited["policy"] == {"fixed": True, "refundable": False}
    assert _BEFORE["policy"] == {"fixed": True, "refundable": True}
    with pytest.raises(ValueError, match="no existing parent"):
        apply_edits(_BEFORE, (StateEdit(path="/nowhere/deep", value=1),))


def test_a_varied_witnessed_set_passes_the_structural_gates() -> None:
    """Distinct states, distinct options and a budgeted witness pass; the original may be one."""
    assert _reasons(_scenario("locked", False, "credit"), _scenario("open", True, "refund")) == []


def test_identical_states_are_not_distinct_variants() -> None:
    """Two scenarios that produce the same world state are one variant, whatever they claim."""
    reasons = _reasons(_scenario("locked", False, "credit"), _scenario("again", False, "refund"))
    assert "its world state is identical to scenario locked" in reasons


def test_paraphrased_choices_are_not_decision_variance() -> None:
    """Options are compared by identity, so prose rewording cannot fake variance."""
    same = _reasons(_scenario("a", False, "credit"), _scenario("b", True, "credit"))
    assert any("no decision variance" in reason for reason in same)
    prose = _reasons(_scenario("a", False, "issue refund"), _scenario("b", True, "credit"))
    assert any("unknown option 'issue refund'" in reason for reason in prose)


def test_replacing_a_parent_cannot_change_an_invariant() -> None:
    """An axis that contains an invariant is refused when the decision state is built."""
    with pytest.raises(ValueError, match="contain or sit inside an invariant"):
        _decision(
            axes=[
                {
                    "path": "/policy",
                    "values": [{"fixed": False, "refundable": True}, {"refundable": False}],
                    "effect": "e",
                }
            ]
        )


def test_values_outside_the_axis_are_refused() -> None:
    """An edit must use one of the axis's permitted values, compared exactly."""
    reasons = _reasons(_scenario("a", "maybe", "credit"), _scenario("b", 1, "refund"))
    assert sum("the axis does not permit" in reason for reason in reasons) == 2


def test_hidden_deciding_facts_and_unaffordable_witnesses_are_refused() -> None:
    """A fact no permitted call reveals within budget would have to be guessed."""
    hidden = _scenario(
        "hidden",
        False,
        "credit",
        witness=[{"tool": "read_order", "arguments": {}, "reveals": ["/order/amount"]}],
    )
    forbidden = _scenario(
        "forbidden",
        True,
        "refund",
        witness=[{"tool": "read_database", "arguments": {}, "reveals": ["/policy/refundable"]}],
    )
    reasons = _reasons(hidden, forbidden)
    assert any("could only be guessed" in reason for reason in reasons)
    assert any("cannot call: ['read_database']" in reason for reason in reasons)
    costly = _scenario(
        "costly",
        False,
        "credit",
        witness=[
            {"tool": "read_order", "arguments": {}, "reveals": ["/order/amount"]} for _ in range(3)
        ]
        + [{"tool": "read_policy", "arguments": {}, "reveals": ["/policy/refundable"]}],
    )
    assert any("only 3 remain" in reason for reason in _reasons(costly, forbidden))


def test_the_instruction_is_frozen_unless_declared() -> None:
    """A causal variant changes the world, not what the actor is told."""
    told = _scenario("told", False, "credit", instruction="Give store credit.")
    assert any(
        "instruction, which is frozen" in r for r in _reasons(told, _scenario("b", True, "refund"))
    )
    assert not any(
        "frozen" in r
        for r in _reasons(
            told, _scenario("b", True, "refund"), decision=_decision(vary_instruction=True)
        )
    )


def test_judges_and_decision_states_refuse_incoherent_answers() -> None:
    """A judge cannot choose one step twice, and options must be distinct."""
    point: JsonObject = {
        "step": 2,
        "decision": "d",
        "alternatives": ["a"],
        "deciding_state": [],
        "rationale": "r",
        "evidence_refs": ["point:2"],
    }
    with pytest.raises(ValueError, match="at most once"):
        JudgeSelection.model_validate({"points": [point, point], "limitations": ["none"]})
    with pytest.raises(ValueError, match="option identities"):
        _decision(options=[{"option_id": "a", "action": "x"}, {"option_id": "a", "action": "y"}])
    assert DecisionOption(option_id="a", action="x").option_id == "a"
    assert Criterion(criterion_id="c", text="t").text == "t"
    assert VariationAxis(path="/a", values=(1, 2), effect="e").values == (1, 2)
    assert WitnessStep(tool="t", arguments={}, reveals=("/a",)).tool == "t"
