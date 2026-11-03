"""Checkable properties derived from learned rules, and their checking over transitions.

A learned rule that reproduces every observed trace can still place a decision boundary wrongly,
so rules are turned into properties and checked against transitions that the rules were not
learned from, such as simulated sessions or boundary challenge workloads.

Each rule kind has one checking semantics on a :class:`Transition`:

- **precondition**: when the action is the rule's tool and the predicate is False, the
  environment must have refused the action. When the predicate is True the property holds only
  vacuously and is counted as ``vacuous``, never as ``holds``.
- **effect** and **observation**: when the action is the rule's tool and the environment did not
  refuse it, the predicate must hold. A refused action does not test the property.
- **invariant**: the predicate must hold on every transition, or on every transition of the
  rule's tool when the rule names one.

A predicate that evaluates to None, or an action whose refusal cannot be recognized, is
undetermined and never counted as a violation.

Refusal is recognized only from the observation, by :class:`RefusalCriteria`. Absent
observation is undetermined, not success.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonValue
from exp.common.evaluations.reconstruction import WorkflowEvidence
from exp.simulation.rules.model import Predicate, RuleBook, RuleKind, Transition, evaluate

Outcome = Literal["holds", "violated", "undetermined", "vacuous"]
"""Result of checking one property on one applicable transition."""


def _default_flags() -> dict[str, JsonValue]:
    """Return the default refusal flags, ``ok`` or ``success`` exactly False."""
    return {"/ok": False, "/success": False}


class RefusalCriteria(ContractModel):
    """How a refused action is recognized from its observation.

    An observation is a refusal when any marker below matches. An observation that is None is
    undetermined. Any other observation is a success.

    Attributes:
        error_keys: Top-level keys of an object observation whose presence with a non-empty,
            non-false value marks a refusal. Defaults to ``error`` and ``errors``.
        flags: JSON pointers into the observation mapped to the exact value that marks a
            refusal, for example ``{"/ok": False}``. Booleans and numbers are kept distinct.
        text_prefixes: Case-insensitive prefixes that mark a string observation, after leading
            whitespace is removed, as a refusal.
        require_unchanged_state: When True, an observation that matches a marker counts as a
            refusal only if the state after equals the state before; a matching observation
            with changed state is undetermined.
    """

    error_keys: tuple[str, ...] = ("error", "errors")
    flags: dict[str, JsonValue] = Field(default_factory=_default_flags)
    text_prefixes: tuple[str, ...] = ("error",)
    require_unchanged_state: bool = False


DEFAULT_REFUSAL = RefusalCriteria()
"""Refusal criteria used when a caller supplies none."""


class Property(ContractModel):
    """One checkable property derived from one rule.

    Attributes:
        rule_id: The rule it was derived from.
        kind: The rule kind, which selects the checking semantics.
        predicate: The rule's predicate.
        applies_to: Tool whose actions it constrains; None only for an environment-wide
            invariant, which applies to every transition.
    """

    rule_id: str = Field(min_length=1)
    kind: RuleKind
    predicate: Predicate
    applies_to: str | None = None


class PropertyResult(ContractModel):
    """Counts for one property over a sequence of transitions.

    Attributes:
        rule_id: The rule the property came from.
        kind: The rule kind.
        holds: Applicable transitions on which the property was tested and held.
        violated: Applicable transitions on which it was tested and failed.
        undetermined: Applicable transitions on which the predicate or the refusal could not
            be determined. Never counted as a violation.
        vacuous: Precondition transitions whose predicate was True, so nothing was tested.
        violations: Locators of violating transitions, in input order.
        undetermined_at: Locators of undetermined transitions, in input order.
    """

    rule_id: str
    kind: RuleKind
    holds: int = Field(default=0, ge=0)
    violated: int = Field(default=0, ge=0)
    undetermined: int = Field(default=0, ge=0)
    vacuous: int = Field(default=0, ge=0)
    violations: tuple[str, ...] = ()
    undetermined_at: tuple[str, ...] = ()


class PropertyReport(ContractModel):
    """Results of checking properties over transitions.

    Attributes:
        transitions: Number of transitions checked.
        refusal: Refusal criteria the check used.
        results: One result per property, in property order.
    """

    transitions: int = Field(ge=0)
    refusal: RefusalCriteria
    results: tuple[PropertyResult, ...]


def properties_from(book: RuleBook) -> tuple[Property, ...]:
    """Derive one property per checkable rule.

    A rule is checkable when it has a predicate and is not contradicted. Precondition, effect
    and observation rules without a tool constrain no action and are left out.

    Args:
        book: The merged rule store.

    Returns:
        Properties in rule order.
    """
    found: list[Property] = []
    for rule in book.rules:
        if rule.predicate is None or rule.status == "contradicted":
            continue
        if rule.kind != "invariant" and rule.tool is None:
            continue
        found.append(
            Property(
                rule_id=rule.rule_id, kind=rule.kind, predicate=rule.predicate, applies_to=rule.tool
            )
        )
    return tuple(found)


def action_name(transition: Transition) -> str | None:
    """Return the tool name of a transition's action, or None when it has none.

    Args:
        transition: The step.

    Returns:
        The ``name`` field when it is a string.
    """
    name = transition.action.get("name")
    return name if isinstance(name, str) else None


def is_refusal(transition: Transition, refusal: RefusalCriteria = DEFAULT_REFUSAL) -> bool | None:
    """Decide whether the environment refused a transition's action.

    Args:
        transition: The step.
        refusal: How a refusal is recognized.

    Returns:
        True for a refusal, False for a success, None when undetermined.
    """
    observation = transition.observation
    if observation is None:
        return None
    if not _marks_refusal(observation, refusal):
        return False
    if refusal.require_unchanged_state and transition.after != transition.before:
        return None
    return True


def check_transition(
    prop: Property, transition: Transition, refusal: RefusalCriteria = DEFAULT_REFUSAL
) -> Outcome | None:
    """Check one property on one transition.

    Args:
        prop: The property.
        transition: The step.
        refusal: How a refusal is recognized.

    Returns:
        The outcome, or None when the property does not apply to the transition.
    """
    if prop.applies_to is not None and action_name(transition) != prop.applies_to:
        return None
    if prop.kind == "invariant":
        return _verdict(evaluate(prop.predicate, transition))
    if prop.kind == "precondition":
        value = evaluate(prop.predicate, transition)
        if value is None:
            return "undetermined"
        if value:
            return "vacuous"
        refused = is_refusal(transition, refusal)
        return "undetermined" if refused is None else "holds" if refused else "violated"
    refused = is_refusal(transition, refusal)
    if refused is None:
        return "undetermined"
    if refused:
        return None
    return _verdict(evaluate(prop.predicate, transition))


def check_transitions(
    properties: Sequence[Property],
    transitions: Sequence[tuple[str, Transition]],
    refusal: RefusalCriteria = DEFAULT_REFUSAL,
) -> PropertyReport:
    """Check every property on every transition.

    Args:
        properties: Properties to check.
        transitions: Pairs of a step locator and the transition at that step.
        refusal: How a refusal is recognized.

    Returns:
        Per-property counts and locators.
    """
    results: list[PropertyResult] = []
    for prop in properties:
        counts: dict[Outcome, list[str]] = {
            "holds": [],
            "violated": [],
            "undetermined": [],
            "vacuous": [],
        }
        for locator, transition in transitions:
            outcome = check_transition(prop, transition, refusal)
            if outcome is not None:
                counts[outcome].append(locator)
        results.append(
            PropertyResult(
                rule_id=prop.rule_id,
                kind=prop.kind,
                holds=len(counts["holds"]),
                violated=len(counts["violated"]),
                undetermined=len(counts["undetermined"]),
                vacuous=len(counts["vacuous"]),
                violations=tuple(counts["violated"]),
                undetermined_at=tuple(counts["undetermined"]),
            )
        )
    return PropertyReport(transitions=len(transitions), refusal=refusal, results=tuple(results))


def transitions_from_workflow(evidence: WorkflowEvidence) -> tuple[tuple[str, Transition], ...]:
    """Convert a simulated workflow record into located transitions.

    Only steps with status ``valid`` and both states recorded become transitions; their reply is
    the observation. Failed, blocked, unattempted, unknown and unsupported steps are execution
    outcomes, not environment behavior, and are left out rather than read as refusals.

    Args:
        evidence: One workflow chronology.

    Returns:
        Pairs of ``workflow:<workflow_id>:step:<step>`` and the transition, in step order.
    """
    found: list[tuple[str, Transition]] = []
    for step in evidence.steps:
        if step.status != "valid" or step.before is None or step.after is None:
            continue
        locator = f"workflow:{evidence.workflow_id}:step:{step.step}"
        transition = Transition(
            before=step.before, after=step.after, action=step.action, observation=step.reply
        )
        found.append((locator, transition))
    return tuple(found)


def _verdict(value: bool | None) -> Outcome:
    """Map a predicate value to an outcome."""
    return "undetermined" if value is None else "holds" if value else "violated"


def _marks_refusal(observation: JsonValue, refusal: RefusalCriteria) -> bool:
    """Return whether an observation matches any refusal marker."""
    if isinstance(observation, str):
        text = observation.lstrip().lower()
        return any(text.startswith(prefix.lower()) for prefix in refusal.text_prefixes)
    if isinstance(observation, dict):
        for key in refusal.error_keys:
            value = observation.get(key)
            if value is not None and value is not False and value not in ("", [], {}):
                return True
    for pointer, marker in refusal.flags.items():
        found = Transition(before={}, after={}, action={}, observation=observation)
        probe = Predicate.model_validate(
            {
                "op": "eq",
                "left": {"scope": "observation", "path": pointer},
                "right": {"value": marker},
            }
        )
        if evaluate(probe, found) is True:
            return True
    return False
