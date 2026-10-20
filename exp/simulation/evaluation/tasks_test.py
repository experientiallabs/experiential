"""Data-only task verifiers, independent qualification and immutable revisions."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.evaluations.reconstruction import WorkflowEvidence, WorkflowStep
from exp.common.project import ProjectStore
from exp.common.tasks import TaskCase
from exp.simulation.evaluation.admission import (
    EvaluationSpecification,
    persist_specification,
    record_label_witness,
    verify_label_witness,
)
from exp.simulation.evaluation.tasks import (
    EvaluationTask,
    OutcomeRule,
    TaskRevisionProposal,
    TaskVerifier,
    VerifierChallenge,
    apply_task_revision,
    grade,
    load_evaluation_task,
    persist_evaluation_task,
    qualify_verifier,
)


def verifier(status: str | None) -> TaskVerifier:
    """Check a declared final-world status, with explicit availability requirements."""
    schema: JsonObject = (
        {}
        if status is None
        else {
            "type": "object",
            "properties": {
                "final_state": {
                    "type": "object",
                    "properties": {"status": {"const": status}},
                    "required": ["status"],
                }
            },
            "required": ["final_state"],
        }
    )
    return TaskVerifier(
        description="Check completed status.",
        outcome_schema=schema,
        required_paths=("/final_state/status",),
    )


def outcome(status: str) -> JsonObject:
    """Supply a complete normalized candidate outcome."""
    return {
        "initial_state": {},
        "final_state": {"status": status},
        "final_answer": status,
        "steps": [],
        "stop_reason": "completed",
    }


def specification(project: ProjectStore) -> ArtifactInput:
    """Persist an independent task specification, distinct from every compared predicate."""
    return persist_specification(
        project.artifacts,
        EvaluationSpecification(
            description="Reference: an order is complete only when its final status is done.",
            predicate=TaskVerifier(
                description="Reference completion specification.",
                outcome_schema={"type": "object"},
                rules=(OutcomeRule(path="/final_state/status", operator="equals", value="done"),),
            ),
            subject="outcome",
        ),
        created_at=datetime.now(UTC),
        code_revision="test",
    )


def challenges(project: ProjectStore) -> tuple[VerifierChallenge, ...]:
    """Bind positive and negative labels to saved checks of a separate specification."""
    reference = specification(project)
    return tuple(
        VerifierChallenge(
            id=status,
            outcome=outcome(status),
            expected=expected,
            authority="synthetic_specification",
            rationale="Declared independent task specification.",
            witness=record_label_witness(
                project.artifacts,
                specification=reference,
                outcome=outcome(status),
                created_at=datetime.now(UTC),
                code_revision="test",
            ),
        )
        for status, expected in (("done", "pass"), ("pending", "fail"))
    )


def reverified(project: ProjectStore, cases: tuple[VerifierChallenge, ...]) -> frozenset[str]:
    """IDs whose saved witness re-derives from the store, as admission computes them."""
    return frozenset(
        case.id
        for case in cases
        if verify_label_witness(project.artifacts, case, label=case.expected, compared=()) is None
    )


def test_grade_uses_last_valid_state_and_preserves_unknown_outcomes() -> None:
    """Stateful grading uses observed candidate effects; missing evidence is not a pass."""
    evidence = WorkflowEvidence(
        workflow_id="case",
        task="Complete the task",
        initial_state={},
        steps=(
            WorkflowStep(
                step=0, action={"name": "finish"}, status="valid", after={"status": "done"}
            ),
        ),
        final_answer="Done",
        stop_reason="completed",
    )
    assert grade(verifier("done"), evidence).status == "pass"
    assert grade(verifier("done"), outcome("pending")).status == "fail"
    assert grade(verifier("done"), {"final_state": {}}).status == "unassessable"
    failed = evidence.model_copy(
        update={
            "steps": (
                WorkflowStep(
                    step=0, action={"name": "finish"}, status="failure", error_type="TimeoutError"
                ),
            )
        }
    )
    assert grade(verifier("done"), failed).status == "unassessable"


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "https://example.test/secret"},
        {"$ref": "#"},
        {"$defs": {"loop": {"$ref": "#/$defs/loop"}}, "$ref": "#/$defs/loop"},
    ],
)
def test_verifier_refuses_external_and_recursive_schema_resolution(schema: JsonObject) -> None:
    """A data verifier never fetches external resources or follows recursive references."""
    with pytest.raises(ValueError):
        TaskVerifier(description="Unsafe", outcome_schema=schema)


def test_qualification_requires_independent_labels_and_preserves_regressions(
    tmp_path: Path,
) -> None:
    """A blanket accepting verifier loses the protected negative and is refused."""
    project = ProjectStore(tmp_path, "task-test")
    cases = challenges(project)
    verified = reverified(project, cases)
    assert verified == {"done", "pending"}
    result = qualify_verifier(verifier(None), verifier("done"), cases, verified=verified)
    assert result.disposition == "accepted_development"
    assert result.gains == ("pending",)
    assert result.losses == ()
    bad = qualify_verifier(verifier("done"), verifier(None), cases, verified=verified)
    assert bad.disposition == "rejected"
    assert bad.losses == ("pending",)
    assert (
        qualify_verifier(verifier("done"), verifier("done"), cases, verified=verified).disposition
        == "rejected"
    )
    assert (
        qualify_verifier(verifier(None), verifier("done"), cases[:1], verified=verified).disposition
        == "incomplete"
    )
    # A witness that was not re-verified admits nothing, whatever it claims.
    assert qualify_verifier(verifier(None), verifier("done"), cases).disposition == "incomplete"
    generated = tuple(
        VerifierChallenge.model_validate(
            {
                **case.model_dump(exclude={"witness"}),
                "authority": "generated_hypothesis",
            }
        )
        for case in cases
    )
    assert (
        qualify_verifier(verifier(None), verifier("done"), generated, verified=verified).disposition
        == "incomplete"
    )


def test_generated_challenge_cannot_claim_independent_approval() -> None:
    """The hacker's proposed label cannot grant its own acceptance authority."""
    with pytest.raises(ValueError, match="generated"):
        VerifierChallenge(
            id="guess",
            outcome=outcome("done"),
            expected="pass",
            authority="generated_hypothesis",
            approved=True,
            rationale="Model says so.",
        )


def test_persisted_task_revision_retains_parent_and_rejects_changed_world(tmp_path: Path) -> None:
    """Prompt/verifier children preserve intent, world and independently labeled cases."""
    project = ProjectStore(tmp_path, "task-test")
    cases = challenges(project)
    ref = cases[0].witness
    assert ref is not None
    task = EvaluationTask(
        intent="Complete the order",
        task=TaskCase(
            task_id="task",
            origin="authored",
            lineage_group_id="customer",
            partition="fit",
            instruction="Complete it.",
            workload_weight=1,
        ),
        world_model=ref,
        checkpoint=ref,
        verifier=verifier(None),
        challenges=cases,
        known_limits=("Synthetic specification; no native equivalence claim.",),
    )
    parent = persist_evaluation_task(
        project.artifacts, task, created_at=datetime.now(UTC), code_revision="test"
    )
    proposal = TaskRevisionProposal(
        prompt="Complete it and report status.",
        verifier=verifier("done"),
        evidence_refs=("challenge:pending",),
        reason="Do not reward an incomplete order.",
    )
    child = apply_task_revision(task, proposal, parent_ref=parent)
    child_ref = persist_evaluation_task(
        project.artifacts, child, created_at=datetime.now(UTC), code_revision="test"
    )
    assert load_evaluation_task(project.artifacts, parent) == task
    assert load_evaluation_task(project.artifacts, child_ref) == child
    assert child.parent == parent
    assert child.intent == task.intent
    assert child.world_model == task.world_model
    assert child_ref != parent
    changed = child.model_copy(update={"intent": "A different goal"})
    with pytest.raises(ValueError, match="intent"):
        persist_evaluation_task(
            project.artifacts, changed, created_at=datetime.now(UTC), code_revision="test"
        )
    wrong = parent.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="digest"):
        load_evaluation_task(project.artifacts, wrong)


def test_revision_proposal_forbids_code_and_world_changes() -> None:
    """Author output is a restricted data patch, never arbitrary verifier execution."""
    with pytest.raises(ValueError):
        TaskRevisionProposal.model_validate(
            {
                "python": "raise SystemExit",
                "world_model": {},
                "evidence_refs": ["challenge:a"],
                "reason": "Change everything",
            }
        )
    with pytest.raises(ValueError, match="change"):
        TaskRevisionProposal(evidence_refs=("challenge:a",), reason="No change")


@pytest.mark.parametrize(
    "stop_reason", ["actor_failure", "simulator_failure", "failure", "maximum_steps", "unknown"]
)
def test_unavailable_or_empty_workflow_cannot_satisfy_a_weak_verifier(stop_reason: str) -> None:
    """A matching initial value cannot turn a stopped or nonexistent execution into success."""
    evidence = WorkflowEvidence(
        workflow_id="empty",
        task="Complete it",
        initial_state={"status": "done"},
        steps=(),
        stop_reason=stop_reason,
    )
    assert grade(verifier(None), evidence).status == "unassessable"
    assert (
        grade(verifier(None), {**outcome("done"), "stop_reason": stop_reason}).status
        == "unassessable"
    )
    assert grade(verifier(None), {}).status == "unassessable"
    assert (
        grade(verifier(None), evidence.model_copy(update={"stop_reason": "completed"})).status
        == "unassessable"
    )


def test_answer_only_completed_workflow_remains_assessable() -> None:
    """A real final answer can satisfy an answer task without fabricated tool activity."""
    evidence = WorkflowEvidence(
        workflow_id="answer",
        task="State the answer",
        initial_state={},
        steps=(),
        final_answer="Known",
        stop_reason="completed",
    )
    answer = TaskVerifier(
        description="Check a final answer",
        outcome_schema={"properties": {"final_answer": {"const": "Known"}}},
        required_paths=("/final_answer",),
    )
    assert grade(answer, evidence).status == "pass"


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "pattern": "(a+)+$"},
        {"properties": {"answer": {"pattern": "a"}}},
        {"$defs": {"text": {"pattern": "a"}}, "properties": {"answer": {"$ref": "#/$defs/text"}}},
        {"patternProperties": {".*": {"type": "string"}}},
    ],
)
def test_verifier_refuses_interpreted_regular_expressions(schema: JsonObject) -> None:
    """Untrusted data predicates cannot run unbounded backtracking expressions."""
    with pytest.raises(ValueError, match="regular expressions"):
        TaskVerifier(description="Untrusted predicate", outcome_schema=schema)


def test_verifier_preserves_literal_pattern_keys() -> None:
    """Instance data and property names do not become interpreted schema keywords."""
    literal: JsonObject = {"pattern": "(a+)+$", "patternProperties": {".*": True}}
    schema: JsonObject = {
        "properties": {
            "literal": {"const": literal},
            "choices": {"enum": [literal]},
            "pattern": {"type": "string"},
        },
        "required": ["literal", "choices", "pattern"],
    }
    predicate = TaskVerifier(description="Literal keys", outcome_schema=schema)
    assert (
        grade(predicate, {"literal": literal, "choices": literal, "pattern": "text"}).status
        == "pass"
    )


def test_rule_draft_grades_nested_arrays_and_numeric_object_keys() -> None:
    """A constrained authored predicate handles paths without inventing tool behavior."""
    from exp.simulation.evaluation.tasks import OutcomeRule, RuleVerifierDraft

    draft = RuleVerifierDraft(
        description="Require an observed read with the declared structure.",
        rules=(
            OutcomeRule(path="/steps", operator="type", value="array"),
            OutcomeRule(path="/steps", operator="min_items", value=1),
            OutcomeRule(path="/steps", operator="max_items", value=2),
            OutcomeRule(path="/steps/0/action/name", operator="equals", value="read"),
            OutcomeRule(path="/final_state/1/name", operator="type", value="string"),
            OutcomeRule(path="/final_state/1/name", operator="exists"),
        ),
    )
    predicate = draft.compile()
    good: JsonObject = {
        "steps": [{"action": {"name": "read"}}],
        "final_state": {"1": {"name": "first"}},
        "final_answer": "Read completed.",
    }
    assert grade(predicate, good).status == "pass"
    assert grade(predicate, {**good, "steps": []}).status == "fail"
    assert grade(predicate, {**good, "final_state": []}).status == "fail"
    assert grade(predicate, {**good, "final_state": {"1": {}}}).status == "fail"


@pytest.mark.parametrize(
    ("operator", "value"),
    [
        ("type", "dict"),
        ("type", 1),
        ("min_items", True),
        ("max_items", -1),
        ("min_items", 1.5),
        ("exists", True),
    ],
)
def test_rules_refuse_wrong_operator_value_types(operator: str, value: object) -> None:
    """Operator arguments are validated rather than coerced or silently ignored."""
    from exp.simulation.evaluation.tasks import OutcomeRule

    with pytest.raises(ValueError):
        OutcomeRule.model_validate({"path": "/steps", "operator": operator, "value": value})


def test_rules_preserve_boolean_number_distinctions_and_unknown_observations() -> None:
    """Strict equality and JSON types avoid bool-as-int accidental acceptance."""
    from exp.simulation.evaluation.tasks import OutcomeRule, RuleVerifierDraft

    for operator, expected in (("equals", 1), ("type", "integer"), ("type", "number")):
        predicate = RuleVerifierDraft(
            description="Typed field",
            rules=(OutcomeRule(path="/value", operator=operator, value=expected),),
        ).compile()
        assert grade(predicate, {"value": True}).status == "fail"
        assert grade(predicate, {"value": 1}).status == "pass"
    missing = RuleVerifierDraft(
        description="Required observation",
        required_paths=("/value",),
        rules=(OutcomeRule(path="/value", operator="equals", value=None),),
    ).compile()
    assert grade(missing, {"other": 1}).status == "unassessable"
    assert grade(missing, {"value": None}).status == "pass"
