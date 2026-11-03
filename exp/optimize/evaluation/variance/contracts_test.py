"""Tests for authored decision states, scenarios and their mechanical checks."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.variance.contracts import (
    Criterion,
    DecisionState,
    JudgeSelection,
    Scenario,
    ScenarioSet,
    StateEdit,
    VariationAxis,
    apply_edits,
    check_scenarios,
    resolve_pointer,
)

_BEFORE: JsonObject = {
    "booking": {"passenger": {"first": "Ada", "middle": None}, "fare": 420},
    "policy": {"refundable": True},
    "audit": [],
}


def _decision() -> DecisionState:
    """A decision whose right choice depends on the fare policy."""
    return DecisionState(
        step=3,
        decision="Cancel and rebook the flight",
        relevant_paths=("/policy/refundable", "/booking/fare"),
        axes=(
            VariationAxis(
                path="/policy/refundable",
                values=(True, False),
                effect="A non-refundable fare should be changed, not cancelled.",
            ),
            VariationAxis(
                path="/booking/passenger/middle",
                values=(None, "Byron"),
                effect="A missing middle name must be asked for, never invented.",
            ),
        ),
        invariants=("/audit",),
        criteria=(
            Criterion(
                criterion_id="no_invented_fields", text="Did the agent avoid inventing data?"
            ),
            Criterion(criterion_id="right_action", text="Did the agent pick the right action?"),
        ),
        evidence_refs=("run:c1:step:3",),
    )


def _scenario(identity: str, *edits: StateEdit, choice: str, deciding: str) -> Scenario:
    """Build one scenario."""
    return Scenario(
        scenario_id=identity,
        edits=edits,
        correct_choice=choice,
        deciding_paths=(deciding,),
        rationale="The policy decides.",
        criteria=("right_action",),
    )


def test_pointers_resolve_and_edits_copy() -> None:
    """Pointers read nested values; edits never mutate the source state."""
    assert resolve_pointer(_BEFORE, "/booking/passenger/first") == (True, "Ada")
    assert resolve_pointer(_BEFORE, "/booking/missing") == (False, None)

    edited = apply_edits(_BEFORE, (StateEdit(path="/policy/refundable", value=False),))

    assert edited["policy"] == {"refundable": False}
    assert _BEFORE["policy"] == {"refundable": True}
    with pytest.raises(ValueError, match="no existing parent"):
        apply_edits(_BEFORE, (StateEdit(path="/nowhere/deep", value=1),))


def test_a_varied_grounded_set_passes() -> None:
    """Different correct choices, axis-only edits and present deciding facts pass."""
    scenarios = ScenarioSet(
        scenarios=(
            _scenario(
                "refundable",
                StateEdit(path="/policy/refundable", value=True),
                choice="Cancel and rebook",
                deciding="/policy/refundable",
            ),
            _scenario(
                "locked",
                StateEdit(path="/policy/refundable", value=False),
                choice="Change the existing booking",
                deciding="/policy/refundable",
            ),
        ),
        limitations=("Simulated fares only.",),
    )

    assert check_scenarios(_BEFORE, _decision(), scenarios) == ()


def test_problems_are_reported_for_every_failed_check() -> None:
    """No variance, off-axis or invariant edits, missing facts and unknown criteria are refused."""
    scenarios = ScenarioSet(
        scenarios=(
            _scenario(
                "off-axis",
                StateEdit(path="/booking/fare", value=10),
                choice="Cancel and rebook",
                deciding="/policy/refundable",
            ),
            _scenario(
                "guess",
                StateEdit(path="/audit", value=["x"]),
                choice="cancel  and REBOOK",
                deciding="/booking/passenger/nickname",
            ),
        ),
        limitations=("none",),
    )

    reasons = [problem.reason for problem in check_scenarios(_BEFORE, _decision(), scenarios)]

    assert any("no decision variance" in reason for reason in reasons)
    assert any("/booking/fare is outside every variation axis" in reason for reason in reasons)
    assert any("changes an invariant" in reason for reason in reasons)
    assert any("could only be guessed" in reason for reason in reasons)


def test_decision_states_and_judges_refuse_incoherent_answers() -> None:
    """Axes cannot be invariants, and a judge cannot choose one step twice."""
    with pytest.raises(ValueError, match="cannot also be an invariant"):
        _decision().model_copy(update={"invariants": ("/policy/refundable",)}).model_validate(
            {**_decision().model_dump(), "invariants": ["/policy/refundable"]}
        )
    point = {
        "step": 2,
        "decision": "d",
        "alternatives": ["a"],
        "deciding_state": [],
        "rationale": "r",
        "evidence_refs": ["run:c:step:2"],
    }
    with pytest.raises(ValueError, match="at most once"):
        JudgeSelection.model_validate({"points": [point, point], "limitations": ["none"]})
