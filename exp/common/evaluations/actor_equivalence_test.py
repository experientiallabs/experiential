"""Actor equivalence does not confuse different paths, missing evidence and native truth."""

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.evaluations.actor_equivalence import (
    ActorComparison,
    evaluate_actor_equivalence,
    rubric_score_100,
)
from exp.common.evaluations.reconstruction import AxisOutcome, WorkflowEvidence, WorkflowStep


def _workflow(identity: str, tool: str) -> WorkflowEvidence:
    """Create paired tasks whose legitimate action paths differ."""
    return WorkflowEvidence(
        workflow_id=identity,
        task="Find my order",
        task_context={"user": "a"},
        initial_state={},
        steps=(
            WorkflowStep(step=0, action={"name": tool}, status="valid", reply={"id": "order-a"}),
        ),
        final_answer="Order a",
        stop_reason="completed",
    )


def test_actor_paths_can_differ_without_forcing_a_mismatch() -> None:
    """The paired metric judges material equivalence instead of sequence equality."""
    pair = ActorComparison(
        comparison_id="pair",
        actor_sha256="a" * 64,
        reference=_workflow("native", "search"),
        candidate=_workflow("candidate", "get"),
    )

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """Rate differing but equivalent paths using the declared rubric inventory."""
        assert "identical tool sequences" in str(body["messages"])
        assert identity == "pair"
        return {
            name: {
                "dimension_id": name,
                "status": "assessed",
                "score": 4,
                "rationale": "Same supported order",
                "evidence_refs": ["A:step-0", "B:step-0"],
                "critical_flags": [],
            }
            for name in ("goal", "decisions", "facts", "consequences", "completion")
        }

    report = evaluate_actor_equivalence(pair, model_identity="frozen", complete=complete)
    assert report.status == "completed" and report.score_100 == 75


def test_unavailable_or_failing_judge_never_becomes_a_passing_score() -> None:
    """All rubric dimensions remain visible when the provider fails."""
    pair = ActorComparison(
        comparison_id="pair",
        actor_sha256="a" * 64,
        reference=_workflow("native", "get"),
        candidate=_workflow("sim", "get"),
    )

    def fail(body: JsonObject, identity: str) -> JsonObject:
        """Simulate an exhausted route without manufacturing an assessment."""
        raise TimeoutError

    report = evaluate_actor_equivalence(pair, model_identity="frozen", complete=fail)
    assert report.status == "failed" and report.score_100 is None
    assert len(report.axes) == 5
    assert (
        rubric_score_100(
            (AxisOutcome(dimension_id="goal", status="unassessable", rationale="No truth"),)
        )
        is None
    )


def test_actor_comparison_rejects_changed_initial_information() -> None:
    """Full native state cannot be supplied only to one paired actor."""
    with pytest.raises(ValueError, match="same task"):
        ActorComparison(
            comparison_id="pair",
            actor_sha256="a" * 64,
            reference=_workflow("native", "get"),
            candidate=_workflow("sim", "get").model_copy(update={"task_context": {}}),
        )
