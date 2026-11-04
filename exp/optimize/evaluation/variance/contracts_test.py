"""Tests for authored decision states, scenarios and their structural gates."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.optimize.evaluation.variance.contracts import (
    BASELINE_STATE,
    ActorView,
    Criterion,
    DecisionOption,
    DecisionState,
    FrozenPart,
    FrozenState,
    JudgeSelection,
    Scenario,
    ScenarioLimits,
    ScenarioSet,
    StateEdit,
    StatePart,
    VariationAxis,
    WitnessStep,
    apply_edits,
    briefing_leaks,
    check_scenarios,
    resolve_pointer,
)

_BEFORE: JsonObject = {
    "policy": {"fixed": True, "refundable": True},
    "order": {"amount": 420},
    "audit": [],
}
_LIMITS = ScenarioLimits(tools=frozenset({"read_policy", "read_order"}), remaining_calls=3)
_TASK = "Please cancel my order and refund it."
_AXIS: dict[str, JsonValue] = {
    "instruction_axis": {
        "values": [
            {
                "value_id": "original",
                "instruction": _TASK,
                "kind": "original",
                "obligations": "Resolve the cancellation as policy allows.",
            },
            {
                "value_id": "polite",
                "instruction": "Kindly cancel my order and refund it, please.",
                "kind": "reworded",
                "equivalent_to": "original",
                "obligations": "Resolve the cancellation as policy allows.",
            },
            {
                "value_id": "credit-only",
                "instruction": "Cancel my order; I only want store credit, never a refund.",
                "kind": "goal_change",
                "obligations": "Give store credit and move no money back.",
                "authorized_by": ["prefix:0"],
                "criteria": ["right_action"],
            },
        ],
        "effect": "A customer who declines a refund must get store credit.",
    }
}
_VIEW = ActorView(instruction=_TASK, observed=(_TASK, '{"order": {"amount": 420}}'))


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


def _reasons(
    *scenarios: Scenario, decision: DecisionState | None = None, view: ActorView | None = None
) -> list[str]:
    """Run the gates and return every problem reason."""
    found = check_scenarios(
        _BEFORE,
        decision or _decision(),
        ScenarioSet(scenarios=scenarios, limitations=("simulated",)),
        _LIMITS,
        view,
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
    assert "its world state and instruction are identical to scenario locked" in reasons


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
    """A state-only decision cannot vary wording; a declared axis admits only its own values."""
    told = _scenario("told", False, "credit", instruction="Give store credit.")
    assert any(
        "instruction, which is frozen" in r for r in _reasons(told, _scenario("b", True, "refund"))
    )
    undeclared = _reasons(told, _scenario("b", True, "refund"), decision=_decision(**_AXIS))
    assert "its instruction is not a declared instruction value" in undeclared
    assert not any("frozen" in r for r in undeclared)


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


def _briefed(identity: str, option: str, text: str, **changes: JsonValue) -> Scenario:
    """A fresh-briefing variant keeping the refundable world and changing only the instruction."""
    return _scenario(
        identity,
        True,
        option,
        instruction="Cancel my order; I only want store credit, never a refund.",
        initialization="fresh_briefing",
        briefing={"text": text, "entitled_paths": ["/order/amount"]},
        **changes,
    )


def test_an_instruction_only_variant_may_share_the_world_state() -> None:
    """Distinctness compares (state, obligations): a consent change keeps the state but differs."""
    briefed = _briefed("credit", "credit", "Your order of 420 is being cancelled.")
    accepted = _reasons(
        _scenario("open", True, "refund"), briefed, decision=_decision(**_AXIS), view=_VIEW
    )
    assert accepted == []


def test_an_equivalent_rewording_cannot_establish_decision_variance() -> None:
    """A rewording changes no obligation, so with the same state it is the same variant."""
    reworded = _scenario(
        "polite",
        True,
        "credit",
        instruction="Kindly cancel my order and refund it, please.",
        initialization="fresh_briefing",
        briefing={"text": "Your order is being cancelled."},
    )
    reasons = _reasons(
        _scenario("open", True, "refund"), reworded, decision=_decision(**_AXIS), view=_VIEW
    )
    assert any("only by an equivalent rewording" in reason for reason in reasons)


def test_a_changed_instruction_needs_a_fresh_briefing_bound_to_its_criteria() -> None:
    """A resumed prefix already stated the original; a goal change binds its own criteria."""
    resumed = _scenario(
        "credit",
        True,
        "credit",
        instruction="Cancel my order; I only want store credit, never a refund.",
    )
    decision = _decision(
        **_AXIS,
        criteria=[
            {"criterion_id": "right_action", "text": "Was the right option taken?"},
            {"criterion_id": "refund_sent", "text": "Was the refund sent?"},
        ],
    )
    reasons = _reasons(_scenario("open", True, "refund"), resumed, decision=decision, view=_VIEW)
    assert any("start it by fresh_briefing" in reason for reason in reasons)
    unbound = _briefed("credit", "credit", "Cancelling.", criteria=["right_action", "refund_sent"])
    reasons = _reasons(_scenario("open", True, "refund"), unbound, decision=decision, view=_VIEW)
    assert any("does not bind" in reason for reason in reasons)
    blind = _reasons(_scenario("open", True, "refund"), unbound, decision=_decision(**_AXIS))
    assert any("without the actor's view" in reason for reason in blind)


def test_a_fresh_briefing_that_leaks_or_imports_knowledge_is_refused() -> None:
    """The intended action, edited values and unobserved facts never reach a briefing."""
    decision = _decision(**_AXIS, relevant_paths=["/policy/refundable", "/policy/fixed"])
    leaky = _briefed("credit", "credit", "You should issue store credit for this order.")
    assert any(
        "intended option's action" in reason
        for reason in briefing_leaks(_BEFORE, decision, leaky, _VIEW)
    )
    secret = _BEFORE | {"order": {"amount": 420, "code": "Z9"}}
    hidden = _briefed("credit", "credit", "Your voucher code is Z9.")
    unobserved = briefing_leaks(
        secret, decision.model_copy(update={"relevant_paths": ("/order/code",)}), hidden, _VIEW
    )
    assert any("which was not observed" in reason for reason in unobserved)
    entitled = _briefed("credit", "credit", "Cancelling.")
    entitled = entitled.model_copy(
        update={"briefing": entitled.briefing.model_copy(update={"entitled_paths": ("/order",)})}
        if entitled.briefing
        else {}
    )
    assert any(
        "was not observable" in reason
        for reason in briefing_leaks(secret, decision, entitled, _VIEW)
    )
    edited = _scenario(
        "locked",
        False,
        "credit",
        initialization="fresh_briefing",
        briefing={
            "text": "Policy says refundable is 'false' now; amount 420.",
            "entitled_paths": [],
        },
    )
    numeric = _decision(
        axes=[{"path": "/order/amount", "values": [420, 99], "effect": "e"}],
        relevant_paths=["/order/amount"],
    )
    stale = edited.model_copy(update={"edits": (StateEdit(path="/order/amount", value=99),)})
    found = briefing_leaks(_BEFORE, numeric, stale, _VIEW)
    assert "briefing states the old value at edited path /order/amount" in found
    assert briefing_leaks(_BEFORE, _decision(), _scenario("a", True, "refund"), _VIEW) == []


def test_a_frozen_state_names_every_part_and_what_is_not_restorable() -> None:
    """A required part that is absent or external is reported; held parts carry a digest."""
    held = {"part": "clock", "status": "held", "sha256": "0" * 64, "detail": "t"}
    rows = [
        {**held, "part": part}
        if part != "files"
        else {"part": "files", "status": "absent", "detail": "no file image"}
        for part in (
            "domain_state",
            "files",
            "identity_permissions",
            "clock",
            "pending_events",
            "scheduler_random",
            "event_cursor",
            "conditioning_facts",
            "tool_session",
            "conversation",
        )
    ]
    frozen = FrozenState.model_validate({"parts": rows})
    assert frozen.unrestorable(BASELINE_STATE) == ()
    files: frozenset[StatePart] = frozenset({"files"})
    assert frozen.unrestorable(BASELINE_STATE | files) == ("files is absent: no file image",)
    with pytest.raises(ValueError, match="every state part once"):
        FrozenState.model_validate({"parts": rows[1:]})
    with pytest.raises(ValueError, match="digest exactly when"):
        FrozenPart.model_validate({**held, "status": "external"})
