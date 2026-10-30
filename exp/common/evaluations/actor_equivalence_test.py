"""Actor equivalence does not confuse different paths, missing evidence and native truth."""

import json

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.evaluations.actor_equivalence import (
    ActorComparison,
    ActorExecutionAssessment,
    EquivalenceCriterion,
    EquivalenceRubric,
    evaluate_actor_equivalence,
    rubric_score_100,
)
from exp.common.evaluations.reconstruction import AxisOutcome, WorkflowEvidence, WorkflowStep
from exp.common.judging.definition import JudgeDefinition
from exp.common.judging.rubric import RubricDimension, ScoreAnchor


def equivalence_rubric() -> EquivalenceRubric:
    """Declare fixture criteria with observed recovery, alternatives and unknown limits."""
    ids = ("goal", "decisions", "facts", "consequences", "completion")
    return EquivalenceRubric(
        definition=JudgeDefinition(
            name="Recorded recovery",
            syllabus="A package refusal requires supported recovery, not invented success.",
            dimensions=tuple(
                RubricDimension(
                    dimension_id=key,
                    name=key,
                    description="Preserve supported error and recovery relationships.",
                    min_score=1,
                    max_score=5,
                    anchors=tuple(
                        ScoreAnchor(score=i, description=f"Recovery level {i}") for i in range(1, 6)
                    ),
                )
                for key in ids
            ),
        ),
        state_coverage="partial",
        criteria=tuple(
            EquivalenceCriterion(
                dimension_id=key,
                valid_variations=("Instance identifiers may differ.",),
                unassessable_when="The relevant effect is not observed.",
            )
            for key in ids
        ),
    )


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

    report = evaluate_actor_equivalence(
        pair, model_identity="frozen", rubric=equivalence_rubric(), complete=complete
    )
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

    report = evaluate_actor_equivalence(
        pair, model_identity="frozen", rubric=equivalence_rubric(), complete=fail
    )
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

    evaluate_actor_equivalence(
        pair, model_identity="frozen", rubric=equivalence_rubric(), complete=inspect
    )
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

    report = evaluate_actor_equivalence(
        pair, model_identity="frozen", rubric=equivalence_rubric(), complete=forbidden
    )
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

    assert evaluate_actor_equivalence(
        pair, model_identity="frozen", rubric=equivalence_rubric(), complete=forbidden
    ).status == ("unassessable")


def test_judge_receives_frozen_environment_criteria() -> None:
    """Environment-specific criteria, alternatives and uncertainty limits reach the judge."""
    rubric = equivalence_rubric()
    pair = ActorComparison(
        comparison_id="derived",
        actor_sha256="a" * 64,
        reference=_workflow("native", "read"),
        candidate=_workflow("sim", "read"),
        reference_assessment=ActorExecutionAssessment(
            execution="completed", task_result="unassessable"
        ),
        candidate_assessment=ActorExecutionAssessment(
            execution="completed", task_result="unassessable"
        ),
    )
    seen = []

    def inspect(body: JsonObject, identity: str) -> JsonObject:
        """Inspect the bound criteria without granting a passing assessment."""
        seen.append(body)
        assert "package refusal" in json.dumps(body)
        assert "Instance identifiers" in json.dumps(body)
        return {}

    report = evaluate_actor_equivalence(
        pair, model_identity="fixture", complete=inspect, rubric=rubric
    )
    assert len(seen) == 1
    assert report.score_100 is None


@pytest.mark.parametrize("combined_result", ["unassessable", "fail"])
def test_unknown_goal_preserves_behavior_ratings_without_claiming_success(
    combined_result: str,
) -> None:
    """A favorable judge cannot manufacture outcome truth from observable behavior."""
    pair = ActorComparison(
        comparison_id="unknown-result",
        actor_sha256="a" * 64,
        reference=_workflow("native", "read"),
        candidate=_workflow("sim", "inspect"),
        reference_assessment=ActorExecutionAssessment.model_validate(
            {
                "execution": "completed",
                "task_result": combined_result,
                "checks": {"goal": {"status": "unassessable"}},
            }
        ),
        candidate_assessment=ActorExecutionAssessment.model_validate(
            {
                "execution": "completed",
                "task_result": combined_result,
                "checks": {"goal": {"status": "unassessable"}},
            }
        ),
    )

    def favorable(body: JsonObject, identity: str) -> JsonObject:
        """Return supported behavior scores and an unsupported favorable goal score."""
        return {
            key: {
                "dimension_id": key,
                "status": "assessed",
                "score": 5,
                "rationale": "Observed equivalent behavior",
                "evidence_refs": ["A:step-0", "B:step-0"],
                "critical_flags": [],
            }
            for key in ("goal", "decisions", "facts", "consequences", "completion")
        }

    report = evaluate_actor_equivalence(
        pair, model_identity="fixture", complete=favorable, rubric=equivalence_rubric()
    )
    assert report.status == "completed" and report.score_100 is None
    assert report.axes[0].status == "unassessable" and report.axes[0].score is None
    assert all(axis.status == "assessed" and axis.score == 5 for axis in report.axes[1:])
