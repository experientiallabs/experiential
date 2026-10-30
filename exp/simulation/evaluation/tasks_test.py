"""Data-only task verifiers, independent qualification and immutable revisions."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactInput, JsonObject, JsonValue
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
    LIMIT_STOP_REASONS,
    EvaluationTask,
    FieldRelation,
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
    "stop_reason", ["actor_failure", "simulator_failure", "failure", "cancelled", "unknown"]
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


@pytest.mark.parametrize("stop_reason", sorted(LIMIT_STOP_REASONS))
def test_declared_limit_stop_is_a_known_goal_failure(stop_reason: str) -> None:
    """Running out of budget fails the goal and stays known, even over a satisfying state."""
    limited = {**outcome("done"), "stop_reason": stop_reason}
    graded = grade(verifier("done"), limited)
    assert (graded.status, graded.goal.status, graded.policy.status) == ("fail", "fail", "pass")
    assert not graded.complete
    assert stop_reason in graded.reason
    violating = TaskVerifier(
        description="Only reads are allowed.",
        outcome_schema={},
        required_paths=("/final_state/status",),
        allowed_tools=("read",),
    )
    step = {"action": {"name": "write", "arguments": {}}, "status": "valid", "after": {}}
    assert grade(violating, {**limited, "steps": [step]}).policy.status == "fail"
    unobserved = {**limited, "steps": [{**step, "status": "failure"}]}
    assert grade(verifier("done"), unobserved).status == "unassessable"


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


@pytest.mark.parametrize(
    ("schema", "accepted", "refused"),
    [
        ({"type": "integer"}, (1, 0), (1.0, True)),
        ({"type": "number"}, (1, 1.5), (True,)),
        ({"const": 1}, (1,), (1.0, True)),
        ({"const": 1.0}, (1.0,), (1, True)),
        ({"enum": [0, "x"]}, (0, "x"), (0.0, False)),
        ({"not": {"type": "integer"}}, (1.0, True), (1,)),
        ({"oneOf": [{"const": 1}, {"const": 1.0}]}, (1, 1.0), (True,)),
        ({"if": {"type": "integer"}, "then": {"minimum": 5}}, (5, 1.0, True), (1,)),
        ({"properties": {"rows": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}}, (), ()),
    ],
)
def test_schema_checks_keep_booleans_integers_and_floats_distinct(
    schema: JsonObject, accepted: tuple[JsonValue, ...], refused: tuple[JsonValue, ...]
) -> None:
    """A schema-valid 1.0 or true cannot satisfy an integer the verifier requires."""
    predicate = TaskVerifier(
        description="Typed field",
        outcome_schema={"properties": {"value": schema}, "required": ["value"]},
    )
    for value in accepted:
        assert grade(predicate, {"value": value, "final_answer": "done"}).status == "pass"
    for value in refused:
        assert grade(predicate, {"value": value, "final_answer": "done"}).status == "fail"
    nested = TaskVerifier(description="Nested", outcome_schema=schema)
    if "properties" in schema:
        assert grade(nested, {"rows": 6, "final_answer": "done"}).status == "pass"
        assert grade(nested, {"rows": None, "final_answer": "done"}).status == "pass"
        assert grade(nested, {"rows": 6.0, "final_answer": "done"}).status == "fail"


@pytest.mark.parametrize(
    ("arguments", "status"),
    [({"count": 1.0}, "pass"), ({"count": True}, "pass"), ({"count": 1}, "fail")],
)
def test_argument_schema_negation_uses_strict_numeric_types(
    arguments: JsonObject, status: str
) -> None:
    """An argument policy refusing integers admits 1.0 and true, which are not integers."""
    check = TaskVerifier(
        description="The count is never an integer.",
        outcome_schema={},
        argument_schemas={"tally": {"properties": {"count": {"not": {"type": "integer"}}}}},
    )
    value: JsonObject = {
        "steps": [{"action": {"name": "tally", "arguments": arguments}, "status": "valid"}],
        "stop_reason": "completed",
        "final_answer": "done",
    }
    assert grade(check, value).policy.status == status


def test_observed_policy_violation_survives_later_missing_evidence() -> None:
    """An interrupted workflow cannot erase an earlier prohibited action."""
    check = TaskVerifier(
        description="Finish without sending mail.", outcome_schema={}, allowed_tools=("read",)
    )
    result = grade(
        check,
        {
            "steps": [
                {"action": {"name": "send_email", "arguments": {}}, "status": "valid"},
                {"action": None, "status": "unavailable"},
            ],
            "stop_reason": "failure",
        },
    )
    assert result.status == "fail"
    assert result.policy.status == "fail"
    assert result.goal.status == "unassessable"
    assert not result.complete


def test_argument_policy_and_protected_state_are_independent_of_goal() -> None:
    """An otherwise correct result cannot hide a forbidden recipient or unrelated edit."""
    check = TaskVerifier(
        description="Send only to the declared recipient; preserve the ledger.",
        outcome_schema={},
        allowed_tools=("send_email",),
        argument_rules={
            "send_email": (
                OutcomeRule(path="/to", operator="equals", value="permitted@example.invalid"),
            )
        },
        preserved_paths=("/ledger",),
    )
    value: JsonObject = {
        "initial_state": {"ledger": 1, "draft": "old"},
        "final_state": {"ledger": 1, "draft": "sent"},
        "steps": [
            {
                "action": {
                    "name": "send_email",
                    "arguments": {
                        "to": "permitted@example.invalid",
                    },
                },
                "status": "valid",
                "after": {"ledger": 1, "draft": "sent"},
            }
        ],
        "stop_reason": "completed",
        "final_answer": "done",
    }
    assert grade(check, value).status == "pass"
    wrong_recipient = {
        **value,
        "steps": [
            {
                "action": {
                    "name": "send_email",
                    "arguments": {"to": "forbidden@example.invalid"},
                },
                "status": "valid",
            }
        ],
    }
    assert grade(check, wrong_recipient).status == "fail"
    changed_then_restored = {
        **value,
        "steps": [
            {
                "action": {
                    "name": "send_email",
                    "arguments": {"to": "permitted@example.invalid"},
                },
                "status": "valid",
                "after": {"ledger": 2, "draft": "sent"},
            }
        ],
    }
    result = grade(check, changed_then_restored)
    assert result.goal.status == "pass"
    assert result.policy.status == "fail"
    deleted = {
        **value,
        "steps": [{**value["steps"][0], "after": {"draft": "sent"}}],
    }
    result = grade(check, deleted)
    assert result.policy.status == "fail"
    assert result.policy.reason == "protected state removed: /ledger"


def test_incomplete_action_inventory_cannot_prove_compliance() -> None:
    """Known allowed actions are not proof that unobserved actions were allowed."""
    check = TaskVerifier(description="Only read.", outcome_schema={}, allowed_tools=("read",))
    result = grade(check, {"steps": [{"action": None}], "final_answer": "done"})
    assert result.status == "unassessable"
    assert result.policy.status == "unassessable"


@pytest.mark.parametrize("steps", [None, "unavailable", [None]])
def test_matching_endpoints_do_not_prove_preservation_without_transition_inventory(
    steps: JsonValue,
) -> None:
    """Lost transitions could hide an edit then undo; equal endpoints cannot clear it."""
    check = TaskVerifier(description="Preserve state.", outcome_schema={}, preserve_state=True)
    result = grade(
        check,
        {
            "initial_state": {"balance": 5},
            "final_state": {"balance": 5},
            "steps": steps,
            "final_answer": "done",
        },
    )
    assert result.goal.status == "pass"
    assert result.policy.status == "unassessable"


def test_table_relations_compose_with_policy_and_incomplete_execution() -> None:
    """Correct rows cannot hide forbidden effects, and missing state cannot become a pass."""
    from exp.simulation.evaluation.table_relations import TableRelation
    from exp.simulation.evaluation.tasks import RuleVerifierDraft

    relation = TableRelation(
        source="/initial_state/rows",
        target="/final_state/rows",
        operation="last_per_key",
        key_columns=(0,),
    )
    draft = RuleVerifierDraft(
        description="Keep last row and preserve the protected field.",
        rules=(OutcomeRule(path="/final_state/output", operator="exists"),),
        table_relations=(relation,),
        preserved_paths=("/protected",),
    )
    predicate = draft.compile()
    assert predicate.table_relations == (relation,)
    initial: JsonObject = {"rows": [[1, "old"], [2, "keep"], [1, "new"]], "protected": 42}
    final: JsonObject = {"rows": [[2, "keep"], [1, "new"]], "protected": 42, "output": True}
    value: JsonObject = {
        "initial_state": initial,
        "final_state": final,
        "steps": [{"after": final, "status": "valid"}],
        "final_answer": "done",
        "stop_reason": "completed",
    }
    assert grade(predicate, value).status == "pass"
    changed = grade(predicate, {**value, "final_state": {**final, "protected": 0}})
    assert changed.goal.status == "pass" and changed.policy.status == "fail"
    assert (
        grade(predicate, {**value, "final_state": {"protected": 42}}).goal.status == "unassessable"
    )
    assert grade(predicate, {**value, "stop_reason": "failure"}).goal.status == "unassessable"
    assert (
        grade(predicate, {**value, "final_state": {**final, "rows": initial["rows"]}}).goal.status
        == "fail"
    )


def relation_outcome() -> JsonObject:
    """Return a declared structured reporting outcome for relation cases, not native evidence."""
    return {
        "initial_state": {"order": {"status": "pending"}},
        "final_state": {"order": {"status": "pending"}},
        "final_answer": {"status": "pending"},
        "stop_reason": "completed",
        "steps": [
            {
                "action": {"name": "inspect"},
                "reply": {"status": "pending"},
                "status": "valid",
                "after": {"order": {"status": "pending"}},
            }
        ],
    }


def relation_verifier() -> TaskVerifier:
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
    value = relation_outcome()
    assert grade(relation_verifier(), value).status == "pass"
    value["final_answer"] = {"status": "delivered"}
    assert grade(relation_verifier(), value).status == "fail"
    value = relation_outcome()
    value["final_state"] = {"order": {"status": "cancelled"}}
    assert grade(relation_verifier(), value).status == "fail"


def test_additional_valid_reads_pass_but_prohibited_action_fails() -> None:
    """Call count is not a proxy for the declared read-only requirement."""
    value = relation_outcome()
    steps = value["steps"]
    assert isinstance(steps, list)
    steps *= 2
    assert grade(relation_verifier(), value).status == "pass"
    steps.append({"action": {"name": "mutate"}})
    assert grade(relation_verifier(), value).status == "fail"


def test_relation_requires_both_observations_and_strict_json_equality() -> None:
    """Missing inputs are unavailable, and booleans are not numeric evidence."""
    value = relation_outcome()
    value["final_answer"] = {}
    assert grade(relation_verifier(), value).status == "unassessable"
    value = relation_outcome()
    value["final_answer"] = {"status": True}
    steps = value["steps"]
    assert isinstance(steps, list)
    first = steps[0]
    assert isinstance(first, dict)
    first["reply"] = {"status": 1}
    assert grade(relation_verifier(), value).status == "fail"


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


def test_matching_endpoints_do_not_prove_unobserved_intermediate_preservation() -> None:
    """A reverted side effect could hide between equal endpoints when step state is missing."""
    value = relation_outcome()
    steps = value["steps"]
    assert isinstance(steps, list) and isinstance(steps[0], dict)
    del steps[0]["after"]
    result = grade(relation_verifier(), value)
    assert result.goal.status == "pass" and result.policy.status == "unassessable"
    assert result.status == "unassessable"


@pytest.mark.parametrize("arguments", [{}, {"updates": []}])
def test_optional_argument_schema_accepts_absent_or_empty_field(arguments: JsonObject) -> None:
    """A properties constraint allows omission without weakening a present value."""
    check = TaskVerifier(
        description="No cell writes; optional updates must be empty.",
        outcome_schema={},
        argument_schemas={
            "edit": {"type": "object", "properties": {"updates": {"type": "array", "maxItems": 0}}}
        },
    )
    result = grade(
        check,
        {
            "steps": [{"action": {"name": "edit", "arguments": arguments}, "status": "valid"}],
            "stop_reason": "completed",
            "final_answer": "done",
        },
    )
    assert result.policy.status == "pass"


def test_argument_schema_preserves_header_policy_despite_restored_goal() -> None:
    """A generic observed delete then recovery cannot conceal a forbidden first start."""
    check = TaskVerifier(
        description="Preserve header throughout; delete data rows only.",
        outcome_schema={},
        rules=(OutcomeRule(path="/final_state/rows", operator="equals", value=45),),
        argument_schemas={
            "edit_workbook": {
                "type": "object",
                "properties": {
                    "deletions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"start": {"type": "integer", "minimum": 2}},
                        },
                    }
                },
            }
        },
    )
    steps: list[JsonValue] = [
        {
            "action": {
                "name": "edit_workbook",
                "arguments": {"deletions": [{"start": 1, "count": 1}]},
            },
            "status": "valid",
        },
        {
            "action": {
                "name": "edit_workbook",
                "arguments": {"deletions": [{"start": 53, "count": 1}]},
            },
            "status": "valid",
        },
    ]
    outcome: JsonObject = {"steps": steps, "final_state": {"rows": 45}, "stop_reason": "completed"}
    graded = grade(check, outcome)
    assert graded.goal.status == "pass" and graded.policy.status == "fail"
    assert grade(check, {**outcome, "steps": steps[1:]}).status == "pass"


@pytest.mark.parametrize(
    "action",
    [
        {"name": "edit"},
        {"name": "edit", "arguments": None},
        {"name": "edit", "arguments": "unknown"},
    ],
)
def test_argument_schema_missing_observation_is_unassessable(action: JsonObject) -> None:
    """Unavailable arguments never count as compliant with even an empty schema."""
    check = TaskVerifier(
        description="Observe arguments.", outcome_schema={}, argument_schemas={"edit": {}}
    )
    result = grade(
        check, {"steps": [{"action": action, "status": "valid"}], "stop_reason": "completed"}
    )
    assert result.policy.status == "unassessable"


def test_incomplete_argument_schema_inventory_cannot_pass() -> None:
    """One compliant prefix does not establish the policy for unobserved future calls."""
    check = TaskVerifier(
        description="Observe all arguments.", outcome_schema={}, argument_schemas={"edit": {}}
    )
    result = grade(
        check,
        {
            "steps": [{"action": {"name": "edit", "arguments": {}}, "status": "valid"}],
            "stop_reason": "failure",
        },
    )
    assert result.policy.status == "unassessable"


@pytest.mark.parametrize(
    "schema",
    [
        {"pattern": ".*"},
        {"properties": {"x": {"pattern": "x"}}},
        {"$ref": "https://example.invalid/schema"},
        {"$ref": "#"},
        {"type": "made-up"},
    ],
)
def test_argument_schemas_refuse_unsupported_predicates(schema: JsonObject) -> None:
    """Argument predicates reuse the bounded schema profile, not a new validator dialect."""
    from exp.simulation.evaluation.tasks import RuleVerifierDraft

    with pytest.raises(ValueError):
        TaskVerifier(
            description="Invalid policy.", outcome_schema={}, argument_schemas={"edit": schema}
        )
    with pytest.raises(ValueError):
        RuleVerifierDraft(
            description="Invalid draft.",
            rules=(OutcomeRule(path="", operator="exists"),),
            argument_schemas={"edit": schema},
        )


def test_argument_schema_draft_compiles_and_legacy_serialization_stays_identical() -> None:
    """The new predicate is opt-in, compiled and present in exact exported verifier bytes."""
    from exp.common.core.artifacts import sha256_json
    from exp.simulation.evaluation.tasks import RuleVerifierDraft

    legacy = TaskVerifier(description="Legacy predicate.", outcome_schema={"type": "object"})
    assert "argument_schemas" not in legacy.model_dump(mode="json")
    assert sha256_json(legacy) == "cdd0a7c546aaa0df817664c7b4452d7566d3b3cdd09b1c234432cc76df36c8a9"
    draft = RuleVerifierDraft(
        description="Constrain optional updates.",
        rules=(OutcomeRule(path="", operator="exists"),),
        argument_schemas={"edit": {"properties": {"updates": {"maxItems": 0}}}},
    )
    compiled = draft.compile()
    assert compiled.argument_schemas == draft.argument_schemas
    assert TaskVerifier.model_validate_json(compiled.model_dump_json()) == compiled
