"""Tests for deciding whether a proposed transition is allowed by a rule book."""

from __future__ import annotations

from exp.common.core.artifacts import JsonValue
from exp.simulation.rules.checker import AdvisoryResult, check_proposed_transition
from exp.simulation.rules.model import (
    Predicate,
    Rule,
    RuleBook,
    RuleKind,
    RuleStatus,
    Scope,
    Transition,
)


def _cmp(op: str, scope: Scope, path: str, value: JsonValue) -> Predicate:
    """A comparison of one transition read against a literal."""
    return Predicate.model_validate(
        {"op": op, "left": {"scope": scope, "path": path}, "right": {"value": value}}
    )


def _rule(
    rule_id: str,
    kind: RuleKind,
    predicate: Predicate | None,
    *,
    status: RuleStatus = "supported",
    tool: str | None = "refund",
) -> Rule:
    """A merged rule with placeholder evidence counts."""
    return Rule(
        rule_id=rule_id,
        tool=tool,
        kind=kind,
        statement=f"{rule_id} statement",
        predicate=predicate,
        status=status,
        support=1,
        contradiction=0,
    )


def _book(*rules: Rule) -> RuleBook:
    """A rule book with no contracts."""
    return RuleBook(contracts={}, rules=rules)


def _step(*, status: str = "paid", after: str = "refunded", ok: bool = True) -> Transition:
    """A refund step over an order with the given before and after status."""
    return Transition(
        before={"order": {"status": status}},
        after={"order": {"status": after}},
        action={"name": "refund", "arguments": {"amount": 40}},
        observation={"ok": ok},
    )


PAID = _cmp("eq", "before", "/order/status", "paid")
REFUNDED = _cmp("eq", "after", "/order/status", "refunded")
SHOWS_OK = _cmp("eq", "observation", "/ok", True)
KNOWN_STATUS = _cmp("in", "after", "/order/status", ["paid", "refunded"])


def test_satisfied_rules_allow_the_transition() -> None:
    """Every kind holding on an accepted action leaves nothing to report."""
    book = _book(
        _rule("pre", "precondition", PAID),
        _rule("eff", "effect", REFUNDED),
        _rule("obs", "observation", SHOWS_OK),
        _rule("inv", "invariant", KNOWN_STATUS, tool=None),
    )
    verdict = check_proposed_transition(book, _step(), refused=False)
    assert verdict.allowed and not verdict.violations and not verdict.refused_by
    assert not verdict.unexplained_refusal and verdict.reasons == ()


def test_failed_precondition_on_an_accepted_action_is_a_violation() -> None:
    """The world accepted an action it must refuse."""
    book = _book(_rule("pre", "precondition", PAID))
    verdict = check_proposed_transition(book, _step(status="open"), refused=False)
    assert not verdict.allowed
    assert verdict.refused_by == ("pre",) and verdict.violations == ("pre",)
    assert "must refuse" in verdict.reasons[0] and "pre statement" in verdict.reasons[0]


def test_failed_precondition_explains_a_refusal() -> None:
    """A refusal the rules require is allowed and explained, and skips effect rules."""
    book = _book(_rule("pre", "precondition", PAID), _rule("eff", "effect", REFUNDED))
    step = _step(status="open", after="open", ok=False)
    verdict = check_proposed_transition(book, step, refused=True)
    assert verdict.allowed and verdict.refused_by == ("pre",) and verdict.violations == ()
    assert not verdict.unexplained_refusal


def test_refusal_with_every_precondition_true_is_unexplained_not_violating() -> None:
    """Rules may be incomplete, so an unexplained refusal is reported but allowed."""
    book = _book(_rule("pre", "precondition", PAID), _rule("eff", "effect", REFUNDED))
    verdict = check_proposed_transition(book, _step(after="paid", ok=False), refused=True)
    assert verdict.allowed and verdict.unexplained_refusal and verdict.violations == ()
    vacuous = check_proposed_transition(_book(), _step(ok=False), refused=True)
    assert vacuous.allowed and vacuous.unexplained_refusal


def test_undetermined_precondition_does_not_make_a_refusal_unexplained() -> None:
    """A precondition over an absent value neither explains nor contradicts the refusal."""
    book = _book(_rule("pre", "precondition", _cmp("eq", "before", "/missing", 1)))
    verdict = check_proposed_transition(book, _step(ok=False), refused=True)
    assert verdict.allowed and verdict.undetermined == ("pre",)
    assert not verdict.unexplained_refusal


def test_effect_and_observation_rules_must_hold_on_success() -> None:
    """A missing effect or a wrong observation on an accepted action is a violation."""
    book = _book(_rule("eff", "effect", REFUNDED), _rule("obs", "observation", SHOWS_OK))
    verdict = check_proposed_transition(book, _step(after="paid", ok=False), refused=False)
    assert not verdict.allowed and verdict.violations == ("eff", "obs")
    assert len(verdict.reasons) == 2


def test_invariants_hold_on_refused_actions_too() -> None:
    """An invariant is checked on every resulting state."""
    book = _book(_rule("inv", "invariant", KNOWN_STATUS, tool=None))
    verdict = check_proposed_transition(book, _step(after="lost", ok=False), refused=True)
    assert not verdict.allowed and verdict.violations == ("inv",)
    assert "invariant" in verdict.reasons[0]


def test_absent_values_are_undetermined_not_violations() -> None:
    """A supported rule over a value that was not recorded is reported as undetermined."""
    book = _book(_rule("eff", "effect", _cmp("eq", "after", "/missing", 1)))
    verdict = check_proposed_transition(book, _step(), refused=False)
    assert verdict.allowed and verdict.undetermined == ("eff",) and verdict.violations == ()


def test_unknown_rules_are_advisory_only() -> None:
    """Unknown-status rules are evaluated and reported, never enforced."""
    book = _book(
        _rule("pre", "precondition", PAID, status="unknown"),
        _rule("eff", "effect", _cmp("eq", "after", "/missing", 1), status="unknown"),
    )
    verdict = check_proposed_transition(book, _step(status="open"), refused=False)
    assert verdict.allowed and verdict.violations == () and verdict.refused_by == ()
    assert verdict.advisory == (
        AdvisoryResult(rule_id="pre", result=False),
        AdvisoryResult(rule_id="eff", result=None),
    )


def test_contradicted_rules_and_rules_without_predicates_are_ignored() -> None:
    """Neither a contradicted rule nor an uncheckable one affects the verdict."""
    book = _book(
        _rule("pre", "precondition", PAID, status="contradicted"),
        _rule("eff", "effect", None),
        _rule("unk", "effect", None, status="unknown"),
    )
    verdict = check_proposed_transition(book, _step(status="open", after="x"), refused=False)
    assert verdict.allowed and verdict == type(verdict)(allowed=True)


def test_rules_for_other_tools_do_not_apply() -> None:
    """A rule naming another tool is skipped; a tool-free rule applies to every action."""
    book = _book(
        _rule("other", "precondition", PAID, tool="cancel"),
        _rule("any", "precondition", PAID, tool=None),
    )
    verdict = check_proposed_transition(book, _step(status="open"), refused=False)
    assert verdict.violations == ("any",)
