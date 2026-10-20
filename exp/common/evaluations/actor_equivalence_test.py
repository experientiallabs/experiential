"""Actor equivalence does not confuse different paths, missing evidence and native truth."""

import json

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.evaluations.actor_equivalence import (
    ActorComparison,
    ActorExecutionAssessment,
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
        reference_assessment=ActorExecutionAssessment(execution="completed", task_result="pass"),
        candidate_assessment=ActorExecutionAssessment(execution="completed", task_result="pass"),
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
        reference_assessment=ActorExecutionAssessment(execution="completed", task_result="pass"),
        candidate_assessment=ActorExecutionAssessment(execution="completed", task_result="pass"),
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
            reference_assessment=ActorExecutionAssessment(
                execution="completed", task_result="pass"
            ),
            candidate_assessment=ActorExecutionAssessment(
                execution="completed", task_result="pass"
            ),
            reference=_workflow("native", "get"),
            candidate=_workflow("sim", "get").model_copy(update={"task_context": {}}),
        )


@pytest.mark.parametrize("identity", ["pair-1", "pair-2", "pair-3", "pair-4"])
def test_blinded_checks_follow_their_execution(identity: str) -> None:
    """Counterbalancing moves evidence and checks together without native/candidate labels."""
    pair = ActorComparison(
        comparison_id=identity,
        actor_sha256="a" * 64,
        reference=_workflow("native", "good"),
        candidate=_workflow("sim", "bad"),
        reference_assessment=ActorExecutionAssessment(execution="completed", task_result="pass"),
        candidate_assessment=ActorExecutionAssessment(execution="completed", task_result="fail"),
    )
    seen = []

    def inspect(body: JsonObject, identity: str) -> JsonObject:
        """Inspect the exact judge payload, with no provider call."""
        seen.append(body)
        messages = body["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        content = messages[1]["content"]
        assert isinstance(content, str)
        payload = json.loads(content)
        assert "task_checks" not in payload
        for label in ("A", "B"):
            side = payload[label]
            tool = side["steps"][0]["action"]["name"]
            assert side["task_checks"]["task_result"] == ("pass" if tool == "good" else "fail")
        return {}

    evaluate_actor_equivalence(pair, model_identity="frozen", complete=inspect)
    assert len(seen) == 1


@pytest.mark.parametrize("execution", ["incomplete", "unknown"])
def test_infrastructure_failure_is_unassessable_without_judge_dispatch(execution: str) -> None:
    """Matching simulator crashes cannot be turned into a high equivalence score."""
    pair = ActorComparison(
        comparison_id="blocked",
        actor_sha256="a" * 64,
        reference=_workflow("native", "get"),
        candidate=_workflow("sim", "get"),
        reference_assessment=ActorExecutionAssessment(execution="completed", task_result="pass"),
        candidate_assessment=ActorExecutionAssessment.model_validate(
            {"execution": execution, "task_result": "unassessable"}
        ),
    )

    def forbidden(body: JsonObject, identity: str) -> JsonObject:
        """Fail if an invalid pair purchases a judge call."""
        pytest.fail("invalid paired execution reached the judge")

    report = evaluate_actor_equivalence(pair, model_identity="frozen", complete=forbidden)
    assert report.status == "unassessable" and report.score_100 is None


def test_completed_attestation_cannot_override_a_failed_trace_slot() -> None:
    """The saved chronology takes precedence over an inconsistent caller's validity label."""
    trace = _workflow("native", "get")
    pair = ActorComparison(
        comparison_id="invalid-slot",
        actor_sha256="a" * 64,
        reference=trace,
        candidate=trace.model_copy(
            update={"steps": (trace.steps[0].model_copy(update={"status": "failure"}),)}
        ),
        reference_assessment=ActorExecutionAssessment(execution="completed", task_result="pass"),
        candidate_assessment=ActorExecutionAssessment(execution="completed", task_result="pass"),
    )

    def forbidden(body: JsonObject, identity: str) -> JsonObject:
        """No judge may rescue the invalid measurement."""
        pytest.fail("failed trace reached judge")

    assert evaluate_actor_equivalence(pair, model_identity="frozen", complete=forbidden).status == (
        "unassessable"
    )
