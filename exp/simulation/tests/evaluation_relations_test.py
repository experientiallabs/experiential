"""Behavioral regression cases for task outcome relations and action policies."""

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject
from exp.simulation.evaluation_tasks import FieldRelation, TaskVerifier, grade


def outcome() -> JsonObject:
    """Return a declared structured reporting outcome, not native evidence."""
    return {
        "initial_state": {"order": {"status": "pending"}},
        "final_state": {"order": {"status": "pending"}},
        "final_answer": {"status": "pending"},
        "stop_reason": "completed",
        "steps": [{"action": {"name": "inspect"}, "reply": {"status": "pending"}}],
    }


def verifier() -> TaskVerifier:
    """Require observed status agreement and a genuinely read-only task."""
    return TaskVerifier(
        description="Report the observed status without mutation",
        outcome_schema={"type": "object"},
        relations=(FieldRelation(left="/final_answer/status", right="/steps/0/reply/status"),),
        allowed_tools=("inspect",),
        preserve_state=True,
    )


def test_status_contradiction_and_unrelated_state_are_not_rewarded() -> None:
    """A valid envelope must not conceal a wrong report or an unauthorized mutation."""
    value = outcome()
    assert grade(verifier(), value).status == "pass"
    value["final_answer"] = {"status": "delivered"}
    assert grade(verifier(), value).status == "fail"
    value = outcome()
    value["final_state"] = {"order": {"status": "cancelled"}}
    assert grade(verifier(), value).status == "fail"


def test_additional_valid_reads_pass_but_prohibited_action_fails() -> None:
    """Call count is not a proxy for the declared read-only requirement."""
    value = outcome()
    steps = value["steps"]
    assert isinstance(steps, list)
    steps *= 2
    assert grade(verifier(), value).status == "pass"
    steps.append({"action": {"name": "mutate"}})
    assert grade(verifier(), value).status == "fail"


def test_relation_requires_both_observations_and_strict_json_equality() -> None:
    """Missing inputs are unavailable, and booleans are not numeric evidence."""
    value = outcome()
    value["final_answer"] = {}
    assert grade(verifier(), value).status == "unassessable"
    value = outcome()
    value["final_answer"] = {"status": True}
    steps = value["steps"]
    assert isinstance(steps, list)
    first = steps[0]
    assert isinstance(first, dict)
    first["reply"] = {"status": 1}
    assert grade(verifier(), value).status == "fail"


def test_new_constraints_are_omitted_when_unused() -> None:
    """Unchanged task content retains the same serialized contract."""
    value = TaskVerifier(description="Existing", outcome_schema={}).model_dump(mode="json")
    assert "relations" not in value
    assert "allowed_tools" not in value
    assert "preserve_state" not in value


def test_relations_reject_non_pointer_paths() -> None:
    """A comparison cannot execute a generated expression."""
    with pytest.raises(ValidationError):
        FieldRelation(left="eval(answer)", right="/reply")
