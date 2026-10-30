"""Each dimension is observed on its own evidence and never passes when it does not apply."""

import pytest

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models import ToolCall
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.continuations.study import ContinuationBranch
from exp.optimize.evaluation.equivalence.dimensions import (
    DimensionObservation,
    observe_trajectory,
    pair_dimensions,
    step_owner,
)
from exp.simulation.evaluation.tasks import OutcomeRule, TaskVerifier

TOOLS = (
    ToolSchema(name="edit", description="Edit rows.", input_schema={"type": "object"}),
    ToolSchema(name="read", description="Read rows.", input_schema={"type": "object"}),
)
GOAL = TaskVerifier(
    description="Six rows remain.",
    outcome_schema={},
    rules=(OutcomeRule(path="/final_state/rows", operator="equals", value=6),),
)
GOVERNED = GOAL.model_copy(
    update={"allowed_tools": ("edit", "read"), "preserved_paths": ("/input",)}
)


def step(
    name: str,
    *,
    owner: str | None = "native",
    status: str = "valid",
    error: bool = False,
    rows: int = 6,
    effects: str = "committed",
    action_owner: str | None = None,
) -> JsonObject:
    """One saved step with an exact world receipt unless the owner is None.

    Args:
        name: Called tool.
        owner: Receipt owner; None saves no receipt.
        status: Saved step status.
        error: Whether the reply reported an error.
        rows: Observed row count after the step.
        effects: Receipt effect disposition.
        action_owner: Tool name the receipt was issued for, when it differs from the call.

    Returns:
        The saved step.
    """
    action = ToolCall(call_id=f"call-{name}", name=name, arguments={})
    row: JsonObject = {
        "action": action.model_dump(mode="json"),
        "status": status,
        "after": {"rows": rows, "input": "original"},
    }
    if status == "valid":
        row["reply"] = {"content": "error" if error else "ok", "is_error": error}
    if owner is not None:
        receipted = action.model_copy(update={"name": action_owner or name})
        row["execution"] = {
            "owner": owner,
            "action_sha256": sha256_json(receipted),
            "phase": "observed",
            "dispatched": True,
            "effects": effects,
            "before_sha256": None,
            "after_sha256": None,
        }
    return row


def branch(*steps: JsonObject, rows: int = 6, status: str = "completed") -> ContinuationBranch:
    """A saved branch with the given steps and final row count."""
    return ContinuationBranch.model_validate(
        {
            "index": 0,
            "status": status,
            "outcome": {
                "initial_state": {"rows": 7, "input": "original"},
                "final_state": {"rows": rows, "input": "original"},
                "steps": list(steps),
                "final_answer": "done",
                "stop_reason": "completed",
            },
        }
    )


def observed(value: ContinuationBranch, verifier: TaskVerifier = GOVERNED) -> dict[str, str]:
    """Observed status by dimension, with inapplicable dimensions named as such."""
    return {
        row.dimension: row.status if row.applicable else "inapplicable"
        for row in observe_trajectory(value, verifier, TOOLS)
    }


def test_clean_native_trajectory_admits_applicable_dimensions_only() -> None:
    """Recovery does not apply without an error, and is not silently passed."""
    assert observed(branch(step("edit"))) == {
        "goal_success": "admitted",
        "policy_compliance": "admitted",
        "forbidden_effects": "admitted",
        "state_fidelity": "admitted",
        "tool_validity": "admitted",
        "recovery": "inapplicable",
    }
    ungoverned = observe_trajectory(branch(step("edit")), GOAL, TOOLS)
    assert [row.dimension for row in ungoverned if not row.applicable] == [
        "policy_compliance",
        "forbidden_effects",
        "recovery",
    ]
    assert all(row.status == "unassessable" for row in ungoverned if not row.applicable)
    shapeless = GOAL.model_copy(update={"rules": ()})
    assert observed(branch(step("edit")), shapeless)["goal_success"] == "inapplicable"
    assert observed(branch(), GOAL)["state_fidelity"] == "inapplicable"
    with pytest.raises(ValueError, match="never passed"):
        DimensionObservation(dimension="recovery", applicable=False, status="admitted", reason="x")


def test_tool_policy_and_protected_state_are_separate_dimensions() -> None:
    """A disallowed tool fails policy only; a protected-state change fails effects only."""
    narrow = GOVERNED.model_copy(update={"allowed_tools": ("edit",)})
    looked = observed(branch(step("read"), step("edit")), narrow)
    assert (looked["policy_compliance"], looked["forbidden_effects"]) == ("refused", "admitted")
    clobbered = step("edit")
    clobbered["after"] = {"rows": 6, "input": "overwritten"}
    changed = observed(branch(clobbered))
    assert (changed["policy_compliance"], changed["forbidden_effects"]) == ("admitted", "refused")
    assert changed["goal_success"] == "admitted"


def test_world_authority_comes_only_from_an_exact_receipt() -> None:
    """Simulated or unproven replies leave state fidelity for calibration or unassessable."""
    rows = observe_trajectory(branch(step("edit", owner="simulated")), GOVERNED, TOOLS)
    fidelity = rows[3]
    assert (fidelity.applicable, fidelity.status) == (True, "unassessable")
    assert "simulated owner" in fidelity.reason
    for unproven in (step("edit", owner=None), step("edit", action_owner="read")):
        assert step_owner(unproven) == "unknown"
        reason = observe_trajectory(branch(unproven), GOVERNED, TOOLS)[3].reason
        assert "unproven" in reason


@pytest.mark.parametrize(
    "steps,expected",
    [
        ((step("delete"),), "refused"),
        ((step("edit", status="blocked", owner=None),), "refused"),
        ((step("edit", status="failure", effects="rolled_back"),), "refused"),
        ((step("edit", status="failure", effects="unknown"),), "unassessable"),
        ((step("edit", status="failure", owner=None),), "unassessable"),
        ((step("edit", error=True), step("edit")), "admitted"),
    ],
)
def test_tool_validity_refuses_known_misuse_and_keeps_unknown_effects_unknown(
    steps: tuple[JsonObject, ...], expected: str
) -> None:
    """Undeclared, blocked or rolled-back calls refuse; unknown effects stay unassessable."""
    assert observed(branch(*steps), GOAL)["tool_validity"] == expected


def test_interrupted_execution_cannot_admit_tool_validity() -> None:
    """Calls after an interruption are unobserved."""
    assert observed(branch(step("edit"), status="outcome_unknown"))["tool_validity"] == (
        "unassessable"
    )


@pytest.mark.parametrize(
    "steps,rows,expected",
    [
        ((step("edit", error=True), step("edit")), 6, "admitted"),
        ((step("edit"), step("read", error=True)), 6, "refused"),
        ((step("edit", error=True), step("edit")), 5, "refused"),
        # The verifier treats a failed dispatch as an unobserved step, so the goal is unknown.
        ((step("edit", status="failure", effects="rolled_back"), step("edit")), 6, "unassessable"),
    ],
)
def test_recovery_needs_a_successful_retry_and_a_passing_goal(
    steps: tuple[JsonObject, ...], rows: int, expected: str
) -> None:
    """An error left unanswered, or followed by a failed goal, is not a recovery."""
    assert observed(branch(*steps, rows=rows), GOAL)["recovery"] == expected


def test_pairs_compare_labels_and_flag_simulator_only_admissions() -> None:
    """A candidate admitting what the reference refuses is a false pass, not agreement."""
    reference = branch(step("edit"), rows=5)
    candidate = branch(step("edit", owner="simulated"))
    rows = {
        row.dimension: row
        for row in pair_dimensions(
            reference, candidate, verifier=GOAL, tools=TOOLS, material_agreement="different"
        )
    }
    goal = rows["goal_success"]
    assert (goal.agreement, goal.other_only_admitted) == ("different", True)
    assert rows["policy_compliance"].agreement == "inapplicable"
    assert rows["tool_validity"].agreement == "match"
    assert rows["state_fidelity"].agreement == "different"
    assert not rows["state_fidelity"].other_only_admitted
    unknown = pair_dimensions(
        reference,
        branch(step("edit", status="failure", owner=None)),
        verifier=GOAL,
        tools=TOOLS,
        material_agreement="unassessable",
    )
    assert {row.dimension: row.agreement for row in unknown}["tool_validity"] == "unassessable"


def test_state_fidelity_pairs_need_an_executed_tool_and_a_native_reference() -> None:
    """No-tool pairs are inapplicable; an unproven reference reply is not truth to compare to."""
    candidate = branch(step("edit", owner="simulated"))
    cases = (
        (branch(), branch(), "inapplicable"),
        (branch(step("edit", owner=None)), candidate, "unassessable"),
        (branch(step("edit", owner="simulated")), candidate, "unassessable"),
        (branch(step("edit")), candidate, "match"),
        (branch(), candidate, "match"),
    )
    for reference, other, expected in cases:
        rows = pair_dimensions(
            reference, other, verifier=GOAL, tools=TOOLS, material_agreement="match"
        )
        assert {row.dimension: row.agreement for row in rows}["state_fidelity"] == expected


def test_argument_schema_policy_is_applicable_and_separate_from_state() -> None:
    """A constrained argument cannot be ignored or contaminate protected-state scoring."""
    schema_only = GOAL.model_copy(
        update={
            "argument_schemas": {"edit": {"properties": {"rows": {"maxItems": 0}}}},
            "preserved_paths": ("/input",),
        }
    )
    changed = step("edit")
    assert isinstance(changed["action"], dict)
    changed["action"]["arguments"] = {"rows": [1]}
    # Updating arguments also changes the truthful receipt's action identity.
    assert isinstance(changed["execution"], dict)
    changed["execution"]["action_sha256"] = sha256_json(ToolCall.model_validate(changed["action"]))
    values = observed(branch(changed), schema_only)
    assert values["policy_compliance"] == "refused"
    assert values["forbidden_effects"] == "admitted"
    assert values["goal_success"] == "admitted"
