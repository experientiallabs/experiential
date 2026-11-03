"""Decide whether a proposed transition is allowed by a rule book, without any model call.

The world model proposes each transition: an observation plus edits to state. This module is the
deterministic judge of that proposal. It reads only the rule book and the transition, executes
nothing a rule's author wrote, and returns a verdict a caller can enforce or record.

How each rule kind is read:

- A **precondition** that evaluates False means the world must refuse the action. If the
  proposal refused it, the rule explains the refusal; if the proposal accepted it, the world did
  something it must not, and that is a violation.
- An **effect** or **observation** rule must hold when the action succeeds. A refused action is
  not expected to show its effects, so these rules are not checked for it.
- An **invariant** must hold on every resulting state, refused or not.

Only rules with status ``supported`` and a predicate are enforced. Rules with status ``unknown``
and a predicate are evaluated and reported as advisory results, never enforced. Rules with status
``contradicted`` are ignored. A rule names a tool, or None to apply to every action.
"""

from __future__ import annotations

from exp.common.core.artifacts import ContractModel
from exp.simulation.rules.model import Rule, RuleBook, Transition, evaluate


class AdvisoryResult(ContractModel):
    """One unknown-status rule evaluated on a transition, reported but never enforced.

    Attributes:
        rule_id: The rule evaluated.
        result: True, False, or None when the rule read a value that is absent.
    """

    rule_id: str
    result: bool | None


class TransitionVerdict(ContractModel):
    """Whether a proposed transition is allowed, and which rules decided it.

    Attributes:
        allowed: True when no enforced rule is violated.
        refused_by: Supported preconditions that evaluated False, so the world must refuse the
            action. With a refusing proposal they explain the refusal; with an accepting one
            each is also listed in ``violations``.
        violations: Supported rules the proposal breaks: a precondition requiring refusal of an
            accepted action, or an effect, observation or invariant that evaluated False.
        undetermined: Supported rules that applied but evaluated None.
        advisory: Unknown-status rules that applied, with their results.
        unexplained_refusal: The proposal refused the action although every applicable
            supported precondition evaluated True. Not a violation: rules may be incomplete.
        reasons: One plain sentence per violation, quoting the rule, for feedback to the
            proposer.
    """

    allowed: bool
    refused_by: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()
    undetermined: tuple[str, ...] = ()
    advisory: tuple[AdvisoryResult, ...] = ()
    unexplained_refusal: bool = False
    reasons: tuple[str, ...] = ()


def check_proposed_transition(
    book: RuleBook, transition: Transition, *, refused: bool
) -> TransitionVerdict:
    """Check one proposed transition against every applicable enforced rule.

    Args:
        book: Merged rules; their contracts, questions and sources are not consulted.
        transition: The proposal: state before, proposed state after, action and observation.
        refused: Whether the proposal itself refused the action (an error observation).

    Returns:
        The verdict. ``allowed`` is False exactly when ``violations`` is nonempty.
    """
    name = transition.action.get("name")
    refused_by: list[str] = []
    violations: list[str] = []
    undetermined: list[str] = []
    advisory: list[AdvisoryResult] = []
    reasons: list[str] = []
    preconditions_hold = True
    for rule in book.rules:
        if rule.predicate is None or rule.status == "contradicted":
            continue
        if rule.tool is not None and rule.tool != name:
            continue
        if rule.kind in ("effect", "observation") and refused:
            continue
        result = evaluate(rule.predicate, transition)
        if rule.status == "unknown":
            advisory.append(AdvisoryResult(rule_id=rule.rule_id, result=result))
            continue
        if result is None:
            undetermined.append(rule.rule_id)
            if rule.kind == "precondition":
                preconditions_hold = False
            continue
        if result:
            continue
        if rule.kind == "precondition":
            refused_by.append(rule.rule_id)
            preconditions_hold = False
            if refused:
                continue
        violations.append(rule.rule_id)
        reasons.append(_reason(rule))
    return TransitionVerdict(
        allowed=not violations,
        refused_by=tuple(refused_by),
        violations=tuple(violations),
        undetermined=tuple(undetermined),
        advisory=tuple(advisory),
        unexplained_refusal=refused and preconditions_hold,
        reasons=tuple(reasons),
    )


def _reason(rule: Rule) -> str:
    """One feedback sentence for a broken rule.

    Args:
        rule: The violated rule.

    Returns:
        The rule identity, what the proposal did wrong and the rule's statement.
    """
    if rule.kind == "precondition":
        problem = "the world must refuse this action, but the proposal accepted it"
    elif rule.kind == "invariant":
        problem = "the proposed state breaks an invariant"
    elif rule.kind == "effect":
        problem = "the proposed state change lacks a required effect"
    else:
        problem = "the proposed observation breaks an observation rule"
    return f"rule {rule.rule_id} ({rule.kind}): {problem}. Rule: {rule.statement}"
