"""Tests for the rule vocabulary and the predicate evaluator."""

from __future__ import annotations

import pytest

from exp.simulation.rules.model import Operand, Predicate, Transition, evaluate

_STEP = Transition(
    before={"order": {"status": "paid", "amount": 40}, "flag": True},
    after={"order": {"status": "refunded", "amount": 40}, "flag": True},
    action={"name": "refund", "arguments": {"amount": 40}},
    observation={"ok": True},
)


def _cmp(op: str, scope: str, path: str, value: object) -> Predicate:
    """A comparison of one read against a literal."""
    return Predicate.model_validate(
        {"op": op, "left": {"scope": scope, "path": path}, "right": {"value": value}}
    )


def test_comparisons_read_every_part_of_a_transition() -> None:
    """Paths read before, after, action and observation."""
    assert evaluate(_cmp("eq", "before", "/order/status", "paid"), _STEP) is True
    assert evaluate(_cmp("eq", "after", "/order/status", "refunded"), _STEP) is True
    assert evaluate(_cmp("le", "action", "/arguments/amount", 40), _STEP) is True
    assert evaluate(_cmp("eq", "observation", "/ok", True), _STEP) is True
    assert evaluate(_cmp("in", "before", "/order/status", ["paid", "open"]), _STEP) is True


def test_absence_is_undetermined_not_false() -> None:
    """A missing value never makes a rule look violated or satisfied."""
    assert evaluate(_cmp("eq", "before", "/order/missing", 1), _STEP) is None
    exists = Predicate(op="exists", left=Operand(scope="before", path="/order/missing"))
    assert evaluate(exists, _STEP) is False
    absent = Predicate(op="absent", left=Operand(scope="before", path="/order/missing"))
    assert evaluate(absent, _STEP) is True


def test_booleans_are_never_numbers() -> None:
    """True does not equal 1 and cannot be ordered."""
    assert evaluate(_cmp("eq", "before", "/flag", 1), _STEP) is False
    assert evaluate(_cmp("ge", "before", "/flag", 0), _STEP) is None


def test_connectives_use_three_valued_logic() -> None:
    """Implication holds when its antecedent is false, whatever the consequent."""
    paid = _cmp("eq", "before", "/order/status", "paid")
    unknown = _cmp("eq", "before", "/nowhere", 1)
    refunded = _cmp("eq", "after", "/order/status", "refunded")
    assert evaluate(Predicate(op="implies", args=(paid, refunded)), _STEP) is True
    assert evaluate(Predicate(op="and", args=(paid, unknown)), _STEP) is None
    assert evaluate(Predicate(op="or", args=(paid, unknown)), _STEP) is True
    assert evaluate(Predicate(op="not", args=(paid,)), _STEP) is False
    not_paid = Predicate(op="not", args=(paid,))
    assert evaluate(Predicate(op="implies", args=(not_paid, unknown)), _STEP) is True


def test_malformed_predicates_are_refused() -> None:
    """Operators accept exactly their operands."""
    with pytest.raises(ValueError, match="wrong operands"):
        Predicate(op="eq", left=Operand(scope="before", path="/a"))
    with pytest.raises(ValueError, match="wrong operands"):
        Predicate(op="not", args=())
    with pytest.raises(ValueError, match="wrong operands"):
        Predicate(op="exists", left=Operand(value=1))
