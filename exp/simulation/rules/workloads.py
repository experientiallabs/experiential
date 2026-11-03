"""Boundary challenge workloads for learned rules, and triage of their observed outcomes.

Exact replay of observed traces says nothing about states nobody observed, so a rule that
reproduces every trace can still place a refund limit or a permission boundary wrongly. Each
non-contradicted precondition or effect rule whose predicate compares a read against a literal
gets challenge cases on both sides of that boundary. A workload is a state edit, an action edit,
or both, applied to a copy of a base state, plus the triggering action and the outcome the rule
predicts. Executing workloads is out of scope here: :func:`triage` takes observed transitions.

Generation assumes the base state and example arguments describe a step the environment
permitted, so a challenge changes only the compared value. Under that assumption a refusal
where the rule predicts a permitted action contradicts the rule's boundary.

A contradiction on a rule without a supporting witness means the rule or the simulator is wrong.
Nothing here decides which: such rules need a native or specification witness, and any
environment family built on them stays diagnostic until one exists.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue
from exp.simulation.rules.model import (
    Operand,
    Predicate,
    Rule,
    RuleBook,
    Transition,
    Witness,
    evaluate,
)
from exp.simulation.rules.properties import DEFAULT_REFUSAL, RefusalCriteria, is_refusal

Target = Literal["state", "action"]
"""What an edit changes: the state before the action, or the action itself."""

Expectation = Literal["permitted", "refused", "effect_holds"]
"""What the rule predicts for a workload."""

Verdict = Literal["consistent_with_rule", "contradicts_rule", "undetermined"]
"""How an observed outcome relates to the rule's prediction."""

_FLIP = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq"}
_SCOPES: dict[str, Target] = {"before": "state", "action": "action"}


class Edit(ContractModel):
    """One value set by a challenge.

    Attributes:
        target: Whether the edit changes the state or the action.
        path: JSON pointer within the target; its parent must already exist.
        value: Value written at the path.
    """

    target: Target
    path: str = Field(pattern=r"^(/[^/]*)+$")
    value: JsonValue = None


class Workload(ContractModel):
    """One boundary challenge for one rule.

    Attributes:
        workload_id: Stable identity, ``<rule_id>/<n>`` in generation order.
        rule_id: The rule challenged; a workload targets only this rule.
        kind: The rule kind, precondition or effect.
        predicate: The rule predicate at generation time.
        witnesses: The rule witnesses at generation time.
        boundary: Human-readable description of the comparison challenged.
        side: ``inside`` when the challenged comparison holds for the edited value, otherwise
            ``outside``.
        edits: Edits applied to produce ``state`` and ``action``.
        state: Base state with state edits applied.
        action: Triggering action, ``{"name": tool, "arguments": {...}}``, with action edits.
        expected: ``permitted`` or ``refused`` for a precondition, from the whole predicate on
            the edited state and action; ``effect_holds`` for an effect, whose predicate must
            hold after any permitted action on either side.
    """

    workload_id: str = Field(min_length=1)
    rule_id: str = Field(min_length=1)
    kind: Literal["precondition", "effect"]
    predicate: Predicate
    witnesses: tuple[Witness, ...] = ()
    boundary: str
    side: Literal["inside", "outside"]
    edits: tuple[Edit, ...]
    state: JsonObject
    action: JsonObject
    expected: Expectation


class Triage(ContractModel):
    """Classification of one executed workload.

    Attributes:
        workload_id: The workload.
        rule_id: The rule it challenged.
        side: Which side of the boundary the workload targeted.
        expected: What the rule predicted.
        verdict: Whether the observation is consistent with the rule, contradicts it, or is
            undetermined.
        reason: One short sentence explaining the verdict.
        witnessed: Whether the rule has a supporting native or specification witness.
        needs_witness: True when the verdict contradicts the rule and no supporting witness
            exists, so the rule or the simulator is wrong and nothing here decides which.
    """

    workload_id: str
    rule_id: str
    side: Literal["inside", "outside"]
    expected: Expectation
    verdict: Verdict
    reason: str
    witnessed: bool
    needs_witness: bool


class _Boundary(ContractModel):
    """A comparison of one read against a literal, normalized so the read is on the left.

    Attributes:
        op: Comparison operator.
        read: The operand read from the transition.
        literal: The literal it is compared with.
    """

    op: Literal["lt", "le", "gt", "ge", "eq", "in"]
    read: Operand
    literal: JsonValue


def boundary_workloads(
    book: RuleBook,
    base_state: JsonObject,
    *,
    examples: Mapping[str, JsonObject] | None = None,
    observed_values: Mapping[str, Sequence[JsonValue]] | None = None,
    epsilon: float = 1e-6,
) -> tuple[Workload, ...]:
    """Generate challenge cases on both sides of every challengeable boundary.

    Integers get threshold-1, threshold and threshold+1. Floats get threshold minus and plus
    ``epsilon * max(1, |threshold|)`` and the threshold itself. ``in`` gets one member of the
    literal list and one non-member taken from observed values: the value at the read path in
    the base state or example, then ``observed_values`` for that path in list order. When no
    observed value is a non-member, only the member case is generated.

    A case whose edit has no existing parent is not generated. A precondition case whose whole
    predicate is undetermined on the edited state and action is not generated.

    Args:
        book: The merged rule store.
        base_state: State the challenges edit; never mutated.
        examples: Example arguments per tool name, copied into the triggering action.
        observed_values: Extra observed values per read, keyed ``before:<pointer>`` or
            ``action:<pointer>``, used for ``in`` non-members.
        epsilon: Relative step for float thresholds.

    Returns:
        Workloads in rule order, then boundary order, then value order.
    """
    found: list[Workload] = []
    for rule in book.rules:
        if not _challengeable(rule) or rule.tool is None or rule.predicate is None:
            continue
        arguments = copy.deepcopy((examples or {}).get(rule.tool, {}))
        base_action: JsonObject = {"name": rule.tool, "arguments": arguments}
        count = 0
        for boundary in _boundaries(rule.predicate):
            target = _SCOPES[boundary.read.scope or "before"]
            current = _lookup(base_state if target == "state" else base_action, boundary.read.path)
            extra = (observed_values or {}).get(f"{boundary.read.scope}:{boundary.read.path}", ())
            for value, inside in _values(boundary, current, tuple(extra), epsilon):
                edit = Edit(target=target, path=boundary.read.path, value=value)
                state = copy.deepcopy(base_state)
                action = copy.deepcopy(base_action)
                if not _apply(state if target == "state" else action, edit):
                    continue
                expected = _expectation(rule, state, action)
                if expected is None:
                    continue
                found.append(
                    Workload(
                        workload_id=f"{rule.rule_id}/{count}",
                        rule_id=rule.rule_id,
                        kind="precondition" if rule.kind == "precondition" else "effect",
                        predicate=rule.predicate,
                        witnesses=rule.witnesses,
                        boundary=_describe(boundary),
                        side="inside" if inside else "outside",
                        edits=(edit,),
                        state=state,
                        action=action,
                        expected=expected,
                    )
                )
                count += 1
    return tuple(found)


def triage(
    workloads: Sequence[Workload],
    observed: Mapping[str, Transition],
    refusal: RefusalCriteria = DEFAULT_REFUSAL,
) -> tuple[Triage, ...]:
    """Classify each executed workload against the rule it challenged.

    The predicate is evaluated on the observed transition, not on the planned edit, so a
    simulator that did not reach the planned state is judged on the state it did reach. An
    observed action for a different tool is undetermined.

    Args:
        workloads: Generated workloads.
        observed: Observed transition per executed workload id.
        refusal: How a refusal is recognized.

    Returns:
        One triage per workload present in ``observed``, in workload order.

    Raises:
        ValueError: When ``observed`` names a workload that is not supplied.
    """
    known = {workload.workload_id for workload in workloads}
    unknown = sorted(set(observed) - known)
    if unknown:
        raise ValueError(f"observed transitions for unknown workloads: {unknown}")
    found: list[Triage] = []
    for workload in workloads:
        transition = observed.get(workload.workload_id)
        if transition is None:
            continue
        verdict, reason = _classify(workload, transition, refusal)
        witnessed = any(witness.supports for witness in workload.witnesses)
        found.append(
            Triage(
                workload_id=workload.workload_id,
                rule_id=workload.rule_id,
                side=workload.side,
                expected=workload.expected,
                verdict=verdict,
                reason=reason,
                witnessed=witnessed,
                needs_witness=verdict == "contradicts_rule" and not witnessed,
            )
        )
    return tuple(found)


def requires_witness(book: RuleBook) -> tuple[str, ...]:
    """List rules whose boundaries are challenged but that have no supporting witness.

    Args:
        book: The merged rule store.

    Returns:
        Rule ids in rule order.
    """
    return tuple(
        rule.rule_id
        for rule in book.rules
        if _challengeable(rule)
        and rule.predicate is not None
        and _boundaries(rule.predicate)
        and not any(witness.supports for witness in rule.witnesses)
    )


def _challengeable(rule: Rule) -> bool:
    """Return whether a rule's kind and status admit boundary challenges."""
    return (
        rule.kind in ("precondition", "effect")
        and rule.status != "contradicted"
        and rule.tool is not None
        and rule.predicate is not None
    )


def _boundaries(predicate: Predicate) -> tuple[_Boundary, ...]:
    """Collect distinct literal comparisons over state or action reads, in predicate order."""
    found: list[_Boundary] = []
    pending = [predicate]
    while pending:
        node = pending.pop(0)
        pending[0:0] = list(node.args)
        boundary = _boundary(node)
        if boundary is not None and boundary not in found:
            found.append(boundary)
    return tuple(found)


def _boundary(node: Predicate) -> _Boundary | None:
    """Normalize one node into a boundary when it compares a read with a usable literal."""
    left, right = node.left, node.right
    if left is None or right is None:
        return None
    if node.op == "in":
        literal = right.value
        if _readable(left) and right.scope is None and isinstance(literal, list) and literal:
            return _Boundary(op="in", read=left, literal=literal)
        return None
    if node.op not in _FLIP:
        return None
    if _readable(left) and right.scope is None and _number(right.value):
        return _Boundary.model_validate({"op": node.op, "read": left, "literal": right.value})
    if _readable(right) and left.scope is None and _number(left.value):
        flipped = _FLIP[node.op]
        return _Boundary.model_validate({"op": flipped, "read": right, "literal": left.value})
    return None


def _describe(boundary: _Boundary) -> str:
    """Render a boundary as ``<scope>:<path> <op> <literal>``."""
    return f"{boundary.read.scope}:{boundary.read.path} {boundary.op} {boundary.literal!r}"


def _readable(operand: Operand) -> bool:
    """Return whether an operand reads an editable, non-root path."""
    return operand.scope in _SCOPES and operand.path != ""


def _number(value: JsonValue) -> bool:
    """Return whether a JSON value is a number and not a boolean."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _values(
    boundary: _Boundary, current: JsonValue, extra: tuple[JsonValue, ...], epsilon: float
) -> tuple[tuple[JsonValue, bool], ...]:
    """Return challenge values and whether the comparison holds for each."""
    literal = boundary.literal
    if boundary.op == "in":
        members = literal if isinstance(literal, list) else []
        member = current if any(_same(current, item) for item in members) else members[0]
        cases: list[tuple[JsonValue, bool]] = [(member, True)]
        for candidate in (current, *extra):
            if candidate is not None and not any(_same(candidate, item) for item in members):
                cases.append((candidate, False))
                break
        return tuple(cases)
    if isinstance(literal, int) and not isinstance(literal, bool):
        candidates: tuple[int | float, ...] = (literal - 1, literal, literal + 1)
    elif isinstance(literal, float):
        step = epsilon * max(1.0, abs(literal))
        candidates = (literal - step, literal, literal + step)
    else:
        return ()
    return tuple((value, _holds(boundary.op, value, literal)) for value in candidates)


def _holds(op: str, value: float, threshold: float) -> bool:
    """Evaluate a numeric comparison."""
    return {
        "lt": value < threshold,
        "le": value <= threshold,
        "gt": value > threshold,
        "ge": value >= threshold,
    }.get(op, value == threshold)


def _same(left: JsonValue, right: JsonValue) -> bool:
    """Exact JSON equality that keeps booleans and numbers distinct."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


def _tokens(path: str) -> list[str]:
    """Split a JSON pointer into unescaped tokens."""
    return [raw.replace("~1", "/").replace("~0", "~") for raw in path.split("/")[1:]]


def _lookup(document: JsonObject, path: str) -> JsonValue:
    """Read a JSON pointer, returning None when it is absent."""
    current: JsonValue = document
    for token in _tokens(path):
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            return None
    return current


def _apply(document: JsonObject, edit: Edit) -> bool:
    """Set a value in place when the pointer's parent exists; return whether it was set."""
    tokens = _tokens(edit.path)
    parent: JsonValue = document
    for token in tokens[:-1]:
        if isinstance(parent, dict) and token in parent:
            parent = parent[token]
        elif isinstance(parent, list) and token.isdigit() and int(token) < len(parent):
            parent = parent[int(token)]
        else:
            return False
    last = tokens[-1]
    if isinstance(parent, dict):
        parent[last] = edit.value
        return True
    if isinstance(parent, list) and last.isdigit() and int(last) < len(parent):
        parent[int(last)] = edit.value
        return True
    return False


def _expectation(rule: Rule, state: JsonObject, action: JsonObject) -> Expectation | None:
    """Predict a workload's outcome under its rule, or None when the rule cannot say."""
    if rule.kind == "effect":
        return "effect_holds"
    if rule.predicate is None:
        return None
    probe = Transition(before=state, after=state, action=action)
    value = evaluate(rule.predicate, probe)
    return None if value is None else "permitted" if value else "refused"


def _classify(
    workload: Workload, transition: Transition, refusal: RefusalCriteria
) -> tuple[Verdict, str]:
    """Classify one observed transition against its workload's rule."""
    if transition.action.get("name") != workload.action.get("name"):
        return "undetermined", "The observed action is for a different tool."
    refused = is_refusal(transition, refusal)
    if refused is None:
        return "undetermined", "Whether the environment refused the action is unknown."
    value = evaluate(workload.predicate, transition)
    if workload.kind == "effect":
        if refused:
            return "undetermined", "The action was refused, so the effect was not exercised."
        if value is None:
            return "undetermined", "The effect predicate is undetermined on the observation."
        if value:
            return "consistent_with_rule", "The action succeeded and the effect held."
        return "contradicts_rule", "The action succeeded but the effect did not hold."
    if value is None:
        return "undetermined", "The precondition is undetermined on the observed state."
    if value and not refused:
        return "consistent_with_rule", "The precondition held and the action was permitted."
    if not value and refused:
        return "consistent_with_rule", "The precondition failed and the action was refused."
    if value:
        return "contradicts_rule", "The precondition held but the action was refused."
    return "contradicts_rule", "The precondition failed but the action was permitted."
