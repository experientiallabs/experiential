"""Shared vocabulary for learned environment rules, kept apart from environment state.

A simulated environment is treated as a state machine with two separate stores:

- **Rules** describe how the environment behaves and are learned from many traces: what each tool
  requires (preconditions), what it changes (effects), what it shows (observations), and what must
  always hold (invariants). They change only when trace evidence changes them, and every rule keeps
  the evidence for and against it.
- **State** is the environment's current condition, a JSON object. Every action is a transition
  that rules permit or refuse, and that leaves state changed or unchanged.

Rules may carry a :class:`Predicate`, a small closed language over one transition: the state
before, the state after, the action and its observation. A predicate is data, never code, so a
rule proposed by a model can be checked without executing anything it wrote.

This module holds only types and the predicate evaluator. Parsing tool contracts from traces,
proposing rules, merging them, deriving properties and checking transitions live beside it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue

PredicateOp = Literal[
    "eq", "ne", "lt", "le", "gt", "ge", "in", "exists", "absent", "and", "or", "not", "implies"
]
"""Comparison, presence and connective operators of the predicate language."""

Scope = Literal["before", "after", "action", "observation"]
"""Which part of a transition a predicate path reads."""

RuleKind = Literal["precondition", "effect", "observation", "invariant"]
"""What a rule constrains."""

RuleStatus = Literal["supported", "contradicted", "unknown"]
"""Whether evidence supports a rule, contradicts it, or leaves it open."""


class Operand(ContractModel):
    """A value read from a transition, or a literal.

    Attributes:
        scope: Which part of the transition to read; None for a literal.
        path: JSON pointer within that part; empty string reads the whole part.
        value: Literal value, used when scope is None.
    """

    scope: Scope | None = None
    path: str = Field(default="", pattern=r"^(/[^/]*)*$")
    value: JsonValue = None


class Predicate(ContractModel):
    """A side-effect-free condition over one transition.

    Attributes:
        op: Operator.
        left: First operand of a comparison, or the operand tested by ``exists``/``absent``.
        right: Second operand of a comparison; for ``in``, a list literal or list-valued read.
        args: Sub-predicates of ``and``, ``or``, ``not`` (one) and ``implies`` (two).
    """

    op: PredicateOp
    left: Operand | None = None
    right: Operand | None = None
    args: tuple[Predicate, ...] = ()

    @model_validator(mode="after")
    def _shape(self) -> Predicate:
        """Require exactly the operands each operator uses."""
        if self.op in ("and", "or"):
            ok = len(self.args) >= 1 and self.left is None and self.right is None
        elif self.op == "not":
            ok = len(self.args) == 1 and self.left is None and self.right is None
        elif self.op == "implies":
            ok = len(self.args) == 2 and self.left is None and self.right is None
        elif self.op in ("exists", "absent"):
            ok = self.left is not None and self.left.scope is not None and self.right is None
            ok = ok and not self.args
        else:
            ok = self.left is not None and self.right is not None and not self.args
        if not ok:
            raise ValueError(f"predicate {self.op!r} has the wrong operands")
        return self


class Transition(ContractModel):
    """One step of an environment, as predicates read it.

    Attributes:
        before: State before the action.
        after: State after the action.
        action: The action, ``{"name": ..., "arguments": {...}}``.
        observation: What the actor was shown.
    """

    before: JsonObject
    after: JsonObject
    action: JsonObject
    observation: JsonValue = None


def evaluate(predicate: Predicate, transition: Transition) -> bool | None:
    """Evaluate a predicate, returning None when it reads a value that is absent.

    Absence is not falsity: a comparison over a missing path is undetermined, so a rule is never
    counted as violated or satisfied because a value was not recorded. ``exists`` and ``absent``
    are the operators that ask about presence. Connectives use three-valued logic.

    Args:
        predicate: The condition.
        transition: The step it is evaluated on.

    Returns:
        True, False, or None when undetermined.
    """
    op = predicate.op
    if op in ("and", "or", "not", "implies"):
        values = [evaluate(arg, transition) for arg in predicate.args]
        if op == "not":
            return None if values[0] is None else not values[0]
        if op == "implies":
            antecedent, consequent = values
            if antecedent is False or consequent is True:
                return True
            return None if antecedent is None or consequent is None else False
        if op == "and":
            if any(value is False for value in values):
                return False
            return None if any(value is None for value in values) else True
        if any(value is True for value in values):
            return True
        return None if any(value is None for value in values) else False
    if predicate.left is None:
        return None
    found, left = _read(predicate.left, transition)
    if op == "exists":
        return found
    if op == "absent":
        return not found
    if predicate.right is None:
        return None
    found_right, right = _read(predicate.right, transition)
    if not found or not found_right:
        return None
    if op == "eq":
        return _same(left, right)
    if op == "ne":
        return not _same(left, right)
    if op == "in":
        return isinstance(right, list) and any(_same(left, item) for item in right)
    return _order(op, left, right)


class ToolContract(ContractModel):
    """What one tool accepts and returns, parsed from traces without any model call.

    Attributes:
        name: Tool name.
        declared_schema: Input schema the tool was declared with, when any trace declared it.
        argument_types: Observed JSON type names per top-level argument.
        required_arguments: Arguments present in every observed call.
        result_shapes: Distinct observed result shapes (type and top-level keys), as strings.
        error_forms: Distinct observed failure forms, as short strings.
        calls: Observed call count.
        failures: Observed failed call count.
        examples: Up to a few ``trace:<trace_id>:span:<span_id>`` locators of observed calls.
    """

    name: str = Field(min_length=1)
    declared_schema: JsonObject | None = None
    argument_types: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    required_arguments: tuple[str, ...] = ()
    result_shapes: tuple[str, ...] = ()
    error_forms: tuple[str, ...] = ()
    calls: int = Field(default=0, ge=0)
    failures: int = Field(default=0, ge=0)
    examples: tuple[str, ...] = ()


class RuleProposal(ContractModel):
    """One rule proposed from one trajectory, with the evidence it rests on.

    Attributes:
        tool: Tool the rule is about, or None for an environment-wide invariant.
        kind: What the rule constrains.
        statement: The rule in one plain sentence.
        predicate: Optional checkable form over a transition.
        evidence: Locators of supporting observations.
        counter_evidence: Locators of observations that contradict it.
        confidence: Proposer's confidence in [0, 1].
    """

    tool: str | None = None
    kind: RuleKind
    statement: str = Field(min_length=1, max_length=600)
    predicate: Predicate | None = None
    evidence: tuple[str, ...] = Field(min_length=1)
    counter_evidence: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)


class ResearchQuestion(ContractModel):
    """An open question about environment behavior that the evidence does not settle.

    Attributes:
        tool: Tool the question is about, or None.
        question: The question, answerable by a witness.
        why_it_matters: Which decisions or rules depend on the answer.
        evidence: Locators that raised it.
    """

    tool: str | None = None
    question: str = Field(min_length=1, max_length=600)
    why_it_matters: str = Field(min_length=1, max_length=600)
    evidence: tuple[str, ...] = ()


class TrajectoryRules(ContractModel):
    """Everything one per-trajectory analysis proposed.

    Attributes:
        trace_id: The trajectory analyzed.
        rules: Proposed rules.
        questions: Open questions.
        limitations: What the analysis could not determine.
    """

    trace_id: str
    rules: tuple[RuleProposal, ...] = ()
    questions: tuple[ResearchQuestion, ...] = ()
    limitations: tuple[str, ...] = Field(min_length=1)


class Witness(ContractModel):
    """Independent evidence about a rule, outside the traces it was learned from.

    Attributes:
        kind: ``native`` for an observed execution of the real environment, ``specification``
            for an independently written specification.
        reference: Locator of the witness.
        supports: Whether the witness agrees with the rule.
    """

    kind: Literal["native", "specification"]
    reference: str = Field(min_length=1)
    supports: bool


class RuleWeighing(ContractModel):
    """A model's weighing of one merged rule's evidence, citing what it weighed.

    The status is the weigher's judgment, not a count: one strong, direct observation may
    support a rule, and several correlated weak ones may leave it unknown. Code checks only the
    structure (the rule exists, every citation is one of the rule's own locators, no citation
    reaches an excluded trace, and a decided status cites evidence on its side).

    Attributes:
        rule_id: The weighed rule.
        status: Supported, contradicted, or unknown.
        rationale: Why the cited evidence decides, or fails to decide, the rule.
        supporting: Cited supporting locators, each among the rule's evidence.
        contradicting: Cited contradicting locators, each among the rule's counter-evidence or
            evidence.
        independence: How independent the cited observations are of one another.
    """

    rule_id: str = Field(min_length=1, max_length=128)
    status: RuleStatus
    rationale: str = Field(min_length=1, max_length=4000)
    supporting: tuple[str, ...] = ()
    contradicting: tuple[str, ...] = ()
    independence: str = Field(default="", max_length=2000)


class Rule(ContractModel):
    """A merged rule with all evidence kept.

    Attributes:
        rule_id: Stable identity.
        tool: Tool the rule is about, or None.
        kind: What the rule constrains.
        statement: The rule in one plain sentence.
        predicate: Checkable form, when every proposal that had one agreed on it.
        status: Supported, contradicted, or unknown.
        support: Distinct supporting trajectories.
        contradiction: Distinct contradicting trajectories.
        evidence: Supporting locators.
        counter_evidence: Contradicting locators.
        witnesses: Independent witnesses; a decision boundary without one stays diagnostic.
        weighing: The evidence weighing that decided the status, when one was applied.
    """

    rule_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,127}$")
    tool: str | None = None
    kind: RuleKind
    statement: str
    predicate: Predicate | None = None
    status: RuleStatus
    support: int = Field(ge=0)
    contradiction: int = Field(ge=0)
    evidence: tuple[str, ...] = ()
    counter_evidence: tuple[str, ...] = ()
    witnesses: tuple[Witness, ...] = ()
    weighing: RuleWeighing | None = Field(default=None, exclude_if=lambda value: value is None)


class RuleBook(ContractModel):
    """The merged rule store, separate from any environment state.

    Attributes:
        contracts: Tool contracts by tool name.
        rules: Merged rules in stable order.
        questions: Open questions still unanswered.
        sources: Trace identities the book was learned from.
    """

    contracts: dict[str, ToolContract]
    rules: tuple[Rule, ...] = ()
    questions: tuple[ResearchQuestion, ...] = ()
    sources: tuple[str, ...] = ()


Predicate.model_rebuild()


def _read(operand: Operand, transition: Transition) -> tuple[bool, JsonValue]:
    """Read an operand from a transition."""
    if operand.scope is None:
        return True, operand.value
    current: JsonValue = getattr(transition, operand.scope)
    if operand.path == "":
        return current is not None or operand.scope != "observation", current
    for raw in operand.path.split("/")[1:]:
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            return False, None
    return True, current


def _same(left: JsonValue, right: JsonValue) -> bool:
    """Exact JSON equality that keeps booleans and numbers distinct."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


def _order(op: str, left: JsonValue, right: JsonValue) -> bool | None:
    """Order numbers with numbers and strings with strings; anything else is undetermined."""
    if isinstance(left, bool) or isinstance(right, bool):
        return None
    if isinstance(left, int | float) and isinstance(right, int | float):
        sign = (left > right) - (left < right)
    elif isinstance(left, str) and isinstance(right, str):
        sign = (left > right) - (left < right)
    else:
        return None
    return {"lt": sign < 0, "le": sign <= 0, "gt": sign > 0}.get(op, sign >= 0)
