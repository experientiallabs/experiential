"""Immutable evaluation tasks and independently qualified data-only verifiers."""

from collections.abc import Collection, Iterator, Sequence
from datetime import datetime
from operator import not_
from typing import Literal, Self

from jsonschema import Draft202012Validator, SchemaError, ValidationError, validators
from jsonschema.protocols import Validator
from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    sha256_json,
    stable_id,
    strict_equal,
)
from exp.common.core.json_schema import expand_local_references
from exp.common.evaluations.reconstruction import WorkflowEvidence
from exp.common.project import ArtifactStore, artifact_input
from exp.common.rollouts.artifact import StopReason
from exp.common.tasks import TaskCase
from exp.simulation.actor_context import ActorContextFragment
from exp.simulation.capability_briefs import load_capability_brief
from exp.simulation.evaluation.table_relations import TableRelation, assess_table_relation
from exp.simulation.reconstruction.models import check_schema, path_parts, resolve


def _strict_const(
    validator: Validator, const: JsonValue, instance: JsonValue, schema: JsonObject
) -> Iterator[ValidationError]:
    """Require the exact JSON type of a constant, so 1.0 and true never equal 1."""
    if not strict_equal(instance, const):
        yield ValidationError(f"{instance!r} is not strictly {const!r}")


def _strict_enum(
    validator: Validator, enum: list[JsonValue], instance: JsonValue, schema: JsonObject
) -> Iterator[ValidationError]:
    """Require one enumerated member with its exact JSON type."""
    if not any(strict_equal(instance, member) for member in enum):
        yield ValidationError(f"{instance!r} is not strictly one of {enum!r}")


# Draft 2020-12 accepts 1.0 as an integer and compares 1 equal to 1.0. Verifiers distinguish
# booleans, integers and floats the way strict_equal does, so every verifier schema is checked
# by this validator alone: a standard pass in front of it would reject values that strict
# semantics accept under `not`, `oneOf` or `if`.
_StrictTypes = validators.extend(
    Draft202012Validator,
    type_checker=Draft202012Validator.TYPE_CHECKER.redefine_many(
        {
            "integer": lambda _, value: type(value) is int,
            "number": lambda _, value: type(value) in (int, float),
        }
    ),
    validators={"const": _strict_const, "enum": _strict_enum},
)

Verdict = Literal["pass", "fail", "unassessable"]
LabelAuthority = Literal[
    "human",
    "recorded_reference",
    "native_execution",
    "synthetic_specification",
    "generated_hypothesis",
]
# Labels admitted by a caller's attestation over digest-checked source references.
ATTESTED_AUTHORITIES: frozenset[LabelAuthority] = frozenset({"human", "recorded_reference"})
# Labels admitted only after their saved witness is re-derived from the artifact store.
WITNESSED_AUTHORITIES: frozenset[LabelAuthority] = frozenset(
    {"native_execution", "synthetic_specification"}
)
# Declared resource limits that end an episode with its whole action inventory observed. An
# actor that reaches one before finishing has failed the goal within its budget, so the stop is a
# known failure that stays in success-rate denominators. Dropping it as unassessable would let an
# arm that exhausts its steps more often report a higher success rate. Runtime failures,
# cancellations and unknown dispatch outcomes remain unassessable.
LIMIT_STOP_REASONS: frozenset[str] = frozenset(
    {
        StopReason.MAXIMUM_STEPS.value,
        StopReason.MAXIMUM_OUTPUT_TOKENS.value,
        StopReason.MAXIMUM_TIME.value,
        StopReason.MAXIMUM_COST.value,
        StopReason.CONTEXT_OVERFLOW.value,
        StopReason.LENGTH.value,
    }
)


class OutcomeRule(ContractModel):
    """One constrained predicate over an observed outcome field.

    Attributes:
        path: Absolute JSON pointer, or empty for the entire outcome.
        operator: Existence, strict equality, JSON type, or array length comparison.
        value: Operator argument; existence has no argument.
    """

    path: str
    operator: Literal["exists", "equals", "type", "min_items", "max_items"]
    value: JsonValue = None

    @model_validator(mode="after")
    def arguments(self) -> Self:
        """Validate operator arguments without interpreting arbitrary expressions."""
        path_parts(self.path)
        if self.operator == "exists" and self.value is not None:
            raise ValueError("exists does not accept a value")
        if self.operator == "type" and self.value not in (
            "null",
            "boolean",
            "object",
            "array",
            "number",
            "integer",
            "string",
        ):
            raise ValueError("type requires a JSON type name")
        if self.operator in ("min_items", "max_items") and (
            type(self.value) is not int or self.value < 0
        ):
            raise ValueError("array length requires a nonnegative integer")
        return self


class FieldRelation(ContractModel):
    """A strict equality comparison between two observed JSON fields.

    Attributes:
        left: Required observation pointer on the candidate outcome.
        right: Required observation pointer to compare, never an executable expression.
    """

    left: str
    right: str

    @model_validator(mode="after")
    def pointers(self) -> Self:
        """Validate both pointers before any workflow is assessed."""
        path_parts(self.left)
        path_parts(self.right)
        return self


class TaskVerifier(ContractModel):
    """A finite JSON Schema predicate over a normalized candidate workflow.

    Attributes:
        description: What this predicate establishes and its scope.
        outcome_schema: Draft 2020-12 schema; remote and recursive refs are refused.
        required_paths: Workflow-rooted JSON pointers, such as /final_state/files/input.xlsx,
            whose unavailable observations prevent assessment.
        rules: Constrained authored field predicates, all required alongside the schema.
        relations: Equality requirements over observed fields, omitted when unused.
        table_relations: Bounded transformations over complete observed tables.
        allowed_tools: Optional explicit operation policy, independent of call count.
        preserve_state: Require every observed state to equal the initial state.
        preserved_paths: State-relative pointers, such as /files/input.xlsx, that must stay
            unchanged through the workflow. Do not prepend /initial_state or /final_state.
        argument_rules: Per-tool predicates over every observed call's arguments.
        argument_schemas: Optional bounded JSON Schemas over each named tool's arguments.
    """

    description: str = Field(min_length=1)
    outcome_schema: JsonObject
    required_paths: tuple[str, ...] = ()
    rules: tuple[OutcomeRule, ...] = ()
    relations: tuple[FieldRelation, ...] = Field(default=(), exclude_if=not_)
    table_relations: tuple[TableRelation, ...] = Field(default=(), max_length=32, exclude_if=not_)
    allowed_tools: tuple[str, ...] | None = Field(default=None, exclude_if=lambda v: v is None)
    preserve_state: bool = Field(default=False, strict=True, exclude_if=not_)
    preserved_paths: tuple[str, ...] = Field(default=(), exclude_if=not_)
    argument_rules: dict[str, tuple[OutcomeRule, ...]] = Field(
        default_factory=dict, exclude_if=not_
    )
    argument_schemas: dict[str, JsonObject] = Field(
        default_factory=dict, max_length=32, exclude_if=not_
    )

    @model_validator(mode="after")
    def supported_schema(self) -> Self:
        """Check the data predicate without fetching resources or executing generated code."""
        try:
            check_schema(self.outcome_schema)
        except SchemaError as error:
            raise ValueError("invalid verifier JSON Schema") from error
        _bounded_predicate(expand_local_references(self.outcome_schema))
        _check_argument_schemas(self.argument_schemas)
        for pointer in (*self.required_paths, *self.preserved_paths):
            path_parts(pointer)
        return self


class RuleVerifierDraft(ContractModel):
    """Model-authored verifier with a fixed structural envelope.

    Attributes:
        description: Predicate intent and limits.
        rules: Nonempty conjunction of typed field predicates.
        required_paths: Observations whose absence prevents assessment.
        relations: Strict comparisons of observed fields.
        table_relations: Bounded whole-row transformations without executable expressions.
        allowed_tools: Explicit operation policy independent of action count.
        preserve_state: Whether complete observed state must remain unchanged.
        preserved_paths: State subtrees protected across observations.
        argument_rules: Per-tool predicates over observed arguments.
        argument_schemas: Optional bounded JSON Schemas over named tools' arguments.
    """

    description: str = Field(min_length=1)
    rules: tuple[OutcomeRule, ...] = Field(min_length=1)
    required_paths: tuple[str, ...] = ()
    relations: tuple[FieldRelation, ...] = Field(default=(), exclude_if=not_)
    table_relations: tuple[TableRelation, ...] = Field(default=(), max_length=32, exclude_if=not_)
    allowed_tools: tuple[str, ...] | None = Field(default=None, exclude_if=lambda v: v is None)
    preserve_state: bool = Field(default=False, strict=True, exclude_if=not_)
    preserved_paths: tuple[str, ...] = Field(default=(), exclude_if=not_)
    argument_rules: dict[str, tuple[OutcomeRule, ...]] = Field(
        default_factory=dict, exclude_if=not_
    )
    argument_schemas: dict[str, JsonObject] = Field(
        default_factory=dict, max_length=32, exclude_if=not_
    )

    @model_validator(mode="after")
    def supported_argument_schemas(self) -> Self:
        """Refuse unsupported policy schemas before a model-authored draft is accepted."""
        _check_argument_schemas(self.argument_schemas)
        return self

    def compile(self) -> TaskVerifier:
        """Compile typed rules and bounded argument schemas into one data verifier."""
        return TaskVerifier(
            description=self.description,
            outcome_schema={"type": "object"},
            rules=self.rules,
            required_paths=self.required_paths,
            relations=self.relations,
            table_relations=self.table_relations,
            allowed_tools=self.allowed_tools,
            preserve_state=self.preserve_state,
            preserved_paths=self.preserved_paths,
            argument_rules=self.argument_rules,
            argument_schemas=self.argument_schemas,
        )


def _check_argument_schemas(schemas: dict[str, JsonObject]) -> None:
    """Apply the existing finite verifier schema rules to every declared argument policy."""
    for name, schema in schemas.items():
        if not name.strip():
            raise ValueError("argument schema requires a nonempty tool name")
        try:
            check_schema(schema)
        except SchemaError as error:
            raise ValueError("invalid argument policy JSON Schema") from error
        _bounded_predicate(expand_local_references(schema))


def _bounded_predicate(schema: JsonValue) -> None:
    """Reject executable regular expressions only in actual subschema positions."""
    if not isinstance(schema, dict):
        return
    if "pattern" in schema or "patternProperties" in schema:
        raise ValueError("verifier regular expressions are unsupported in this bounded profile")
    for keyword in (
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    ):
        child = schema.get(keyword)
        if child is not None:
            _bounded_predicate(child)
    for keyword in ("properties", "dependentSchemas"):
        children = schema.get(keyword)
        if isinstance(children, dict):
            for child in children.values():
                _bounded_predicate(child)
    for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
        children = schema.get(keyword)
        if isinstance(children, list):
            for child in children:
                _bounded_predicate(child)


class Assessment(ContractModel):
    """One independent assessment of an observed workflow.

    Attributes:
        status: Pass, fail or unavailable evidence within this assessment's scope.
        reason: Concrete mechanical evidence for the status.
    """

    status: Verdict
    reason: str = Field(min_length=1)


class VerifierGrade(Assessment):
    """Combined verdict retaining goal, policy and execution assessments independently.

    Attributes:
        goal: Declared outcome predicate. A stop at a declared resource limit
            (``LIMIT_STOP_REASONS``) with every step observed is a known goal failure; other
            incomplete execution leaves the goal unassessable.
        policy: Observed prohibited behavior survives missing later observations.
        complete: Whether the workflow completed with no unavailable steps. A declared-limit
            stop is not complete, even though its goal and policy are known.
    """

    goal: Assessment
    policy: Assessment
    complete: bool


class VerifierChallenge(ContractModel):
    """A labeled outcome, with independent authority separated from a model hypothesis.

    Attributes:
        id: Stable case identity within this task.
        outcome: Recorded workflow or its explicit normalized JSON projection.
        expected: Independently declared expected verdict, or a diagnostic hypothesis.
        authority: Source of the label; a generated hypothesis cannot approve itself.
            ``native_execution`` and ``synthetic_specification`` labels are admitted only by
            re-verifying ``witness``; ``human`` and ``recorded_reference`` labels by ``approved``.
        approved: Caller attestation admitting a human or recorded-reference label. Witnessed
            authorities leave it false: verification, not approval, admits them.
        rationale: Why the expected verdict follows from the declared specification.
        source_refs: Immutable witnesses for an attested label, required for approval.
        witness: Saved ``label-witness`` artifact re-derived at admission; required for a
            native-execution label and the only admission path for a specification label.
    """

    id: str = Field(min_length=1)
    outcome: WorkflowEvidence | JsonObject
    expected: Verdict
    authority: LabelAuthority
    approved: bool = Field(default=False, strict=True)
    rationale: str = Field(min_length=1)
    source_refs: tuple[ArtifactInput, ...] = ()
    witness: ArtifactInput | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def independent_approval(self) -> Self:
        """Refuse model-generated admission, approval without a witness and asserted witnesses."""
        if self.approved and self.authority == "generated_hypothesis":
            raise ValueError("a generated hypothesis cannot independently approve its label")
        if self.approved and self.authority in WITNESSED_AUTHORITIES:
            raise ValueError(
                f"{self.authority} labels are admitted by re-verifying their witness, not by "
                "approval; set approved=false and reference a saved label witness"
            )
        if self.approved and not self.source_refs:
            raise ValueError("approved verifier labels require immutable source witnesses")
        if self.witness is not None and self.authority not in WITNESSED_AUTHORITIES:
            raise ValueError(
                "only native_execution or synthetic_specification labels carry a label witness"
            )
        if self.authority == "native_execution" and self.witness is None:
            raise ValueError("a native_execution label requires a saved native label witness")
        return self


def label_admitted(challenge: VerifierChallenge, verified: Collection[str]) -> bool:
    """Whether one label is independent evidence: attested, or witnessed and re-verified.

    Args:
        challenge: Label to admit.
        verified: Challenge IDs whose saved witness was re-derived from the artifact store.

    Returns:
        True for an approved attested label or a witnessed label in ``verified``.
    """
    if challenge.authority in ATTESTED_AUTHORITIES:
        return challenge.approved
    return (
        challenge.authority in WITNESSED_AUTHORITIES
        and challenge.witness is not None
        and challenge.id in verified
    )


class ChallengeComparison(ContractModel):
    """One paired check; unapproved model labels remain diagnostics.

    Attributes:
        id: Challenge identity.
        independent: Whether an attested or re-verified witnessed label is available.
        expected: Declared expected verdict.
        parent: Mechanical parent verdict.
        child: Mechanical child verdict.
    """

    id: str
    independent: bool
    expected: Verdict
    parent: VerifierGrade
    child: VerifierGrade


class VerifierQualification(ContractModel):
    """Development selection from paired, independently labeled challenge outcomes.

    Attributes:
        disposition: Development acceptance, rejection, or insufficient evidence.
        gains: Independent cases corrected by the child.
        losses: Previously correct independent cases broken by the child.
        comparisons: All diagnostics, including unapproved generated hypotheses.
        reason: Admission rule that determined the disposition.
    """

    disposition: Literal["accepted_development", "rejected", "incomplete"]
    gains: tuple[str, ...]
    losses: tuple[str, ...]
    comparisons: tuple[ChallengeComparison, ...]
    reason: str


class EvaluationTask(ContractModel):
    """Versioned intent, actor view, frozen candidate world and verifier challenges.

    Attributes:
        intent: Original goal, immutable across refinements of the same task.
        task: Public actor inputs; private reference answers are prohibited.
        world_model: Exact simulator artifact, immutable across refinements.
        checkpoint: Exact candidate-world checkpoint, never a native-state rescue.
        verifier: Data-only mechanical predicate.
        challenges: Independent labels and explicitly unapproved diagnostics.
        parent: Prior task artifact, retained even for a rejected child.
        context_fragments: Source-annotated actor instructions, fixed in this first slice.
        known_limits: Preserved qualifications about what the artifact establishes.
        capability_brief: Optional private future-user intent and assumption provenance.
    """

    intent: str = Field(min_length=1)
    task: TaskCase
    world_model: ArtifactInput
    checkpoint: ArtifactInput
    verifier: TaskVerifier
    challenges: tuple[VerifierChallenge, ...]
    parent: ArtifactInput | None = None
    context_fragments: tuple[ActorContextFragment, ...] = ()
    known_limits: tuple[str, ...] = ()
    capability_brief: ArtifactInput | None = Field(default=None, exclude_if=lambda v: v is None)

    @model_validator(mode="after")
    def public_inputs(self) -> Self:
        """Keep labels outside actor inputs and require unique challenge identities."""
        if not self.intent.strip():
            raise ValueError("evaluation task intent must not be blank")
        if self.task.reference_answer is not None:
            raise ValueError("evaluation task actor inputs cannot contain a private reference")
        if len({case.id for case in self.challenges}) != len(self.challenges):
            raise ValueError("verifier challenge IDs must be unique")
        if len({fragment.fragment_id for fragment in self.context_fragments}) != len(
            self.context_fragments
        ):
            raise ValueError("actor context fragment IDs must be unique")
        return self


class TaskRevisionProposal(ContractModel):
    """A restricted author/verifier proposal that cannot mutate intent or world facts.

    Attributes:
        prompt: Optional replacement public task instruction.
        verifier: Optional replacement data predicate.
        evidence_refs: Actual supplied locators; the caller validates its inventory.
        reason: Source-backed explanation for the proposed revision.
    """

    prompt: str | None = Field(default=None, min_length=1)
    verifier: TaskVerifier | None = None
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def changed(self) -> Self:
        """Refuse empty patches without granting any model acceptance authority."""
        if self.prompt is None and self.verifier is None:
            raise ValueError("revision proposal must change prompt or verifier")
        if self.prompt is not None and not self.prompt.strip():
            raise ValueError("revision prompt must not be blank")
        return self


def normalized_outcome(evidence: WorkflowEvidence) -> JsonObject:
    """Project candidate-owned values without adding native references.

    Args:
        evidence: Actual workflow, including unavailable or blocked slots.

    Returns:
        Stable predicate input with the last observed valid poststate.
    """
    final_state = evidence.initial_state if not evidence.steps else None
    for step in evidence.steps:
        if step.status == "valid":
            final_state = step.after
    return {
        "initial_state": evidence.initial_state,
        "final_state": final_state,
        "final_answer": evidence.final_answer,
        "steps": [step.model_dump(mode="json") for step in evidence.steps],
        "stop_reason": evidence.stop_reason,
    }


def grade(verifier: TaskVerifier, outcome: WorkflowEvidence | JsonObject) -> VerifierGrade:
    """Assess goal and policy independently, retaining violations after an interruption.

    Args:
        verifier: Finite local predicates, argument constraints and preservation policy.
        outcome: Observed workflow or declared normalized observation.

    Returns:
        Combined verdict and its independent dimensions. A known violation is always a fail;
        absent evidence cannot establish success or complete policy compliance. A fully
        observed stop at a declared resource limit is a known goal failure, so it stays in
        every success-rate denominator.
    """
    value = normalized_outcome(outcome) if isinstance(outcome, WorkflowEvidence) else outcome
    steps = value.get("steps")
    stop_reason = value.get("stop_reason")
    observed = not (
        isinstance(steps, list)
        and any(
            isinstance(step, dict) and step.get("status") not in (None, "valid") for step in steps
        )
    )
    complete = stop_reason in (None, "completed", "agent_stop") and observed
    limited = stop_reason in LIMIT_STOP_REASONS and observed
    if complete:
        goal = _grade_goal(verifier, value)
    elif limited:
        goal = Assessment(
            status="fail", reason=f"workflow reached its declared {stop_reason} limit unfinished"
        )
    else:
        goal = Assessment(
            status="unassessable", reason="workflow did not complete or has unavailable steps"
        )
    # A limit stop ends the episode, so its observed action inventory is final.
    policy = _grade_policy(verifier, value, complete=complete or limited)
    if policy.status == "fail":
        combined = policy
    elif goal.status == "fail":
        combined = goal
    elif goal.status == "unassessable":
        combined = goal
    elif policy.status == "unassessable":
        combined = policy
    else:
        combined = Assessment(
            status="pass", reason="observed outcome satisfies the declared predicates"
        )
    return VerifierGrade(**combined.model_dump(), goal=goal, policy=policy, complete=complete)


def _grade_goal(verifier: TaskVerifier, value: JsonObject) -> Assessment:
    """Assess only the declared outcome, with no assumption about behavioral compliance."""
    if not value or (value.get("steps") == [] and value.get("final_answer") is None):
        return Assessment(status="unassessable", reason="workflow has no observed action or answer")
    table_paths = tuple(
        path for rule in verifier.table_relations for path in (rule.source, rule.target)
    )
    for pointer in (*verifier.required_paths, *table_paths):
        try:
            resolve(value, pointer)
        except ValueError:
            return Assessment(status="unassessable", reason=f"missing observation {pointer}")
    schema = expand_local_references(verifier.outcome_schema)
    error = next(_StrictTypes(schema).iter_errors(value), None)
    if error is not None:
        return Assessment(status="fail", reason=error.message)
    for rule in verifier.rules:
        try:
            observed = resolve(value, rule.path)
        except ValueError:
            return Assessment(status="fail", reason=f"rule field missing at {rule.path}")
        if not _rule_matches(rule, observed):
            return Assessment(status="fail", reason=f"{rule.operator} rule failed at {rule.path}")
    for relation in verifier.relations:
        try:
            left, right = resolve(value, relation.left), resolve(value, relation.right)
        except ValueError:
            return Assessment(status="unassessable", reason="required relation field unavailable")
        if not strict_equal(left, right):
            return Assessment(
                status="fail", reason=f"observations differ at {relation.left} and {relation.right}"
            )
    table_results = [assess_table_relation(rule, value) for rule in verifier.table_relations]
    for disposition in ("fail", "unassessable"):
        for status, reason in table_results:
            if status == disposition:
                return Assessment(status=status, reason=reason)
    return Assessment(status="pass", reason="observed outcome satisfies the declared predicates")


def _grade_policy(verifier: TaskVerifier, value: JsonObject, *, complete: bool) -> Assessment:
    """Collect policy evidence without letting one unknown hide a later known violation."""
    missing: list[str] = []
    violations: list[str] = []
    steps = value.get("steps")
    if verifier.allowed_tools is not None or verifier.argument_rules or verifier.argument_schemas:
        if not isinstance(steps, list):
            missing.append("action inventory unavailable")
        else:
            for step in steps:
                action = step.get("action") if isinstance(step, dict) else None
                if not isinstance(action, dict) or not isinstance(action.get("name"), str):
                    missing.append("action identity unavailable")
                    continue
                name = action["name"]
                if verifier.allowed_tools is not None and name not in verifier.allowed_tools:
                    violations.append("operation violates allowed tool policy")
                if (schema := verifier.argument_schemas.get(name)) is not None:
                    arguments = action.get("arguments")
                    if not isinstance(arguments, dict):
                        missing.append(f"argument observation unavailable: {name}")
                    else:
                        expanded = expand_local_references(schema)
                        error = next(_StrictTypes(expanded).iter_errors(arguments), None)
                        if error is not None:
                            violations.append(f"operation violates argument schema: {name}")
                for rule in verifier.argument_rules.get(name, ()):
                    try:
                        argument = resolve(action.get("arguments"), rule.path)
                    except ValueError:
                        missing.append(f"argument observation unavailable: {name}{rule.path}")
                        continue
                    if not _rule_matches(rule, argument):
                        violations.append(f"operation violates argument policy: {name}{rule.path}")
        if not complete:
            missing.append("incomplete workflow cannot establish complete action compliance")
    paths = ("",) if verifier.preserve_state else verifier.preserved_paths
    states = [value.get("final_state")]
    if isinstance(steps, list):
        states.extend(step.get("after") for step in steps if isinstance(step, dict))
        if paths and any(not isinstance(step, dict) for step in steps):
            missing.append("state transition observation unavailable")
    elif paths:
        missing.append("state transition inventory unavailable")
    for pointer in paths:
        try:
            initial = resolve(value.get("initial_state"), pointer)
            if value.get("initial_state") is None:
                raise ValueError("missing initial state")
        except ValueError:
            missing.append(f"initial state comparison unavailable: {pointer}")
            continue
        for state in states:
            if state is None:
                missing.append(f"state comparison unavailable: {pointer}")
                continue
            try:
                observed = resolve(state, pointer)
            except ValueError:
                violations.append(f"protected state removed: {pointer}")
                continue
            if not strict_equal(initial, observed):
                violations.append(f"protected state changed: {pointer}")
        if not complete:
            missing.append("incomplete workflow cannot establish state preservation")
    if violations:
        return Assessment(status="fail", reason="; ".join(dict.fromkeys(violations)))
    if missing:
        return Assessment(status="unassessable", reason="; ".join(dict.fromkeys(missing)))
    return Assessment(status="pass", reason="observed workflow satisfies the declared policy")


def _rule_matches(rule: OutcomeRule, observed: JsonValue) -> bool:
    """Interpret only the finite operator vocabulary, preserving JSON type distinctions."""
    match rule.operator:
        case "exists":
            return True
        case "equals":
            return strict_equal(observed, rule.value)
        case "type":
            match rule.value:
                case "null":
                    return observed is None
                case "boolean":
                    return isinstance(observed, bool)
                case "object":
                    return isinstance(observed, dict)
                case "array":
                    return isinstance(observed, list)
                case "number":
                    return type(observed) in (int, float)
                case "integer":
                    return type(observed) is int
                case "string":
                    return isinstance(observed, str)
            return False
        case "min_items" | "max_items":
            if not isinstance(observed, list) or type(rule.value) is not int:
                return False
            return (
                len(observed) >= rule.value
                if rule.operator == "min_items"
                else len(observed) <= rule.value
            )


def qualify_verifier(
    parent: TaskVerifier,
    child: TaskVerifier,
    challenges: Sequence[VerifierChallenge],
    *,
    verified: Collection[str] = frozenset(),
) -> VerifierQualification:
    """Select one development child only on independently admitted positive and negative labels.

    Args:
        parent: Frozen current verifier.
        child: Proposed replacement predicate.
        challenges: Same protected observations and labels for both verifiers.
        verified: Challenge IDs whose saved witness was re-verified; other witnessed labels
            remain diagnostics.

    Returns:
        Gains, losses and a disposition; generated hypotheses cannot establish acceptance.
    """
    if len({case.id for case in challenges}) != len(challenges):
        raise ValueError("verifier challenge IDs must be unique")
    comparisons = tuple(
        ChallengeComparison(
            id=case.id,
            independent=label_admitted(case, verified),
            expected=case.expected,
            parent=grade(parent, case.outcome),
            child=grade(child, case.outcome),
        )
        for case in challenges
    )
    independent = tuple(row for row in comparisons if row.independent)
    gains = tuple(
        row.id
        for row in independent
        if row.parent.status != row.expected and row.child.status == row.expected
    )
    losses = tuple(
        row.id
        for row in independent
        if row.parent.status == row.expected and row.child.status != row.expected
    )
    disposition: Literal["accepted_development", "rejected", "incomplete"]
    if not {"pass", "fail"} <= {row.expected for row in independent}:
        disposition, reason = "incomplete", "approved positive and negative labels are required"
    elif losses:
        disposition, reason = "rejected", "child regresses a protected independent label"
    elif any(
        row.child.status == "unassessable" and row.expected != "unassessable" for row in independent
    ):
        disposition, reason = "incomplete", "child cannot assess an independently labeled case"
    elif not gains or any(row.child.status != row.expected for row in independent):
        disposition, reason = "rejected", "child must improve and satisfy every independent label"
    else:
        disposition, reason = "accepted_development", "child improves without protected regressions"
    return VerifierQualification(
        disposition=disposition, gains=gains, losses=losses, comparisons=comparisons, reason=reason
    )


def apply_task_revision(
    parent: EvaluationTask, proposal: TaskRevisionProposal, *, parent_ref: ArtifactInput
) -> EvaluationTask:
    """Construct an immutable child restricted to public prompt and verifier changes.

    Args:
        parent: Current task whose intent, state and labels remain protected.
        proposal: Validated data-only author/verifier change.
        parent_ref: Exact parent artifact, checked during persistence.

    Returns:
        Proposed child; constructing it does not approve it.
    """
    task = parent.task
    if proposal.prompt is not None:
        task = task.model_copy(update={"instruction": proposal.prompt})
    return EvaluationTask(
        **{
            **parent.model_dump(mode="json"),
            "task": task.model_dump(mode="json"),
            "verifier": (proposal.verifier or parent.verifier).model_dump(mode="json"),
            "parent": parent_ref.model_dump(mode="json"),
        }
    )


def _references(task: EvaluationTask) -> tuple[ArtifactInput, ...]:
    """Collect one canonical reference for every immutable task dependency."""
    refs = [task.world_model, task.checkpoint]
    if task.capability_brief is not None:
        refs.append(task.capability_brief)
    if task.parent is not None:
        refs.append(task.parent)
    refs.extend(reference for case in task.challenges for reference in case.source_refs)
    refs.extend(case.witness for case in task.challenges if case.witness is not None)
    found: dict[str, ArtifactInput] = {}
    for reference in refs:
        if reference.artifact_id in found and found[reference.artifact_id] != reference:
            raise ValueError("task dependency has conflicting manifest digests")
        found[reference.artifact_id] = reference
    return tuple(found[key] for key in sorted(found))


def _verify_graph(store: ArtifactStore, references: Sequence[ArtifactInput]) -> None:
    """Verify dependency manifest digests and reject a cyclic artifact graph."""
    complete: set[str] = set()

    def visit(reference: ArtifactInput, active: frozenset[str]) -> None:
        """Read one verified node and its immutable dependencies."""
        if reference.artifact_id in active:
            raise ValueError("evaluation task artifact dependency cycle")
        stored = store.read(reference.artifact_id)
        if artifact_input(stored.manifest) != reference:
            raise ValueError("evaluation task dependency manifest digest differs")
        if reference.artifact_id not in complete:
            for child in stored.manifest.inputs:
                visit(child, active | {reference.artifact_id})
            complete.add(reference.artifact_id)

    for reference in references:
        visit(reference, frozenset())


def _protect_parent(parent: EvaluationTask, child: EvaluationTask) -> None:
    """Enforce task identity rather than allowing a revision to redefine its goal."""
    if parent.intent != child.intent:
        raise ValueError("task revision cannot change original intent")
    if parent.capability_brief != child.capability_brief:
        raise ValueError("task revision cannot change its capability brief")
    if parent.world_model != child.world_model or parent.checkpoint != child.checkpoint:
        raise ValueError("task revision cannot change the frozen world or checkpoint")
    if child.task.model_copy(update={"instruction": parent.task.instruction}) != parent.task:
        raise ValueError("task revision may change only the public instruction")
    if parent.challenges != child.challenges or parent.context_fragments != child.context_fragments:
        raise ValueError("task revision cannot rewrite labels or actor context sources")
    if not set(parent.known_limits) <= set(child.known_limits):
        raise ValueError("task revision cannot remove declared limitations")


def persist_evaluation_task(
    store: ArtifactStore, task: EvaluationTask, *, created_at: datetime, code_revision: str
) -> ArtifactInput:
    """Persist a proposed task with verified dependency closure and immutable parent.

    Args:
        store: Existing immutable artifact owner.
        task: Task data; persistence does not grant acceptance or release authority.
        created_at: Publication timestamp.
        code_revision: Producing source identity.

    Returns:
        Manifest-bound artifact reference, reusable by project selection/export.
    """
    checked = EvaluationTask.model_validate(task.model_dump(mode="json"))
    if checked.capability_brief is not None:
        load_capability_brief(store, checked.capability_brief).require_task(
            checked.task, checked.intent
        )
    references = _references(checked)
    _verify_graph(store, references)
    if checked.parent is not None:
        _protect_parent(load_evaluation_task(store, checked.parent), checked)
    identity = stable_id("evaluation-task", {"task_sha256": sha256_json(checked)})
    if store.exists(identity):
        reference = artifact_input(store.read(identity).manifest)
        if load_evaluation_task(store, reference) != checked:
            raise ValueError("evaluation task identity conflicts with saved content")
        return reference
    manifest = store.write_json(
        artifact_id=identity,
        artifact_type="evaluation-task",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=created_at,
            inputs=references,
            code_revision=code_revision,
        ),
        files={"task.json": checked.model_dump(mode="json")},
    )
    return artifact_input(manifest)


def load_evaluation_task(store: ArtifactStore, reference: ArtifactInput) -> EvaluationTask:
    """Load a digest-bound task and verify its dependency and parent constraints.

    Args:
        store: Existing artifact owner; no external source or provider is contacted.
        reference: Exact task manifest reference.

    Returns:
        Validated immutable task artifact data.
    """
    _verify_graph(store, (reference,))
    manifest = store.read(reference.artifact_id).manifest
    if manifest.artifact_type != "evaluation-task":
        raise ValueError("artifact is not an evaluation task")
    task = EvaluationTask.model_validate_json(store.read_bytes(reference.artifact_id, "task.json"))
    if task.capability_brief is not None:
        load_capability_brief(store, task.capability_brief).require_task(task.task, task.intent)
    if manifest.inputs != _references(task):
        raise ValueError("evaluation task manifest dependency inventory differs")
    if task.parent is not None:
        _protect_parent(load_evaluation_task(store, task.parent), task)
    return task
