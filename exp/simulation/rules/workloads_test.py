"""Tests for boundary challenge workloads and triage."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.simulation.rules.model import Predicate, Rule, RuleBook, Transition, Witness
from exp.simulation.rules.workloads import (
    Workload,
    boundary_workloads,
    requires_witness,
    triage,
)

_BASE: JsonObject = {"account": {"balance": 100, "role": "agent", "rate": 0.5}}


def _cmp(op: str, scope: str, path: str, value: JsonValue) -> JsonObject:
    """A comparison of one read against a literal, as predicate data."""
    return {"op": op, "left": {"scope": scope, "path": path}, "right": {"value": value}}


def _rule(rule_id: str, predicate: JsonObject, **fields: JsonValue) -> Rule:
    """A supported refund rule with defaults overridden."""
    data: JsonObject = {
        "rule_id": rule_id,
        "tool": "refund",
        "kind": "precondition",
        "statement": "s",
        "predicate": predicate,
        "status": "supported",
        "support": 2,
        "contradiction": 0,
    }
    data.update(fields)
    return Rule.model_validate(data)


def _book(*rules: Rule) -> RuleBook:
    """A rule book holding the given rules."""
    return RuleBook(contracts={}, rules=rules)


def _cases(workloads: tuple[Workload, ...]) -> list[tuple[JsonValue, str, str]]:
    """Edited value, side and expectation of each workload."""
    return [(w.edits[0].value, w.side, w.expected) for w in workloads]


def test_integer_boundary_on_action_argument() -> None:
    """An integer limit yields threshold-1, threshold and threshold+1 with predictions."""
    book = _book(_rule("limit", _cmp("le", "action", "/arguments/amount", 50)))
    workloads = boundary_workloads(book, _BASE, examples={"refund": {"amount": 10, "id": "o1"}})
    assert _cases(workloads) == [
        (49, "inside", "permitted"),
        (50, "inside", "permitted"),
        (51, "outside", "refused"),
    ]
    assert [w.workload_id for w in workloads] == ["limit/0", "limit/1", "limit/2"]
    assert workloads[2].action == {"name": "refund", "arguments": {"amount": 51, "id": "o1"}}
    assert workloads[0].state == _BASE
    assert {w.rule_id for w in workloads} == {"limit"}


def test_reversed_literal_and_float_epsilon_on_state() -> None:
    """A literal on the left is normalized; floats step by a small epsilon."""
    reversed_rule = _rule(
        "rate",
        {
            "op": "lt",
            "left": {"value": 0.25},
            "right": {"scope": "before", "path": "/account/rate"},
        },
    )
    workloads = boundary_workloads(_book(reversed_rule), _BASE, epsilon=1e-3)
    values = [w.edits[0].value for w in workloads]
    assert values == [pytest.approx(0.249), 0.25, pytest.approx(0.251)]
    assert [w.side for w in workloads] == ["outside", "outside", "inside"]
    assert workloads[2].state["account"] == {"balance": 100, "role": "agent", "rate": values[2]}
    assert _BASE["account"] == {"balance": 100, "role": "agent", "rate": 0.5}


def test_membership_uses_observed_non_member() -> None:
    """``in`` yields a member and a non-member drawn from observed values."""
    rule = _rule("role", _cmp("in", "before", "/account/role", ["admin", "owner"]))
    workloads = boundary_workloads(_book(rule), _BASE)
    assert _cases(workloads) == [("admin", "inside", "permitted"), ("agent", "outside", "refused")]
    admin_base: JsonObject = {"account": {"role": "admin"}}
    none_seen = boundary_workloads(_book(rule), admin_base)
    assert _cases(none_seen) == [("admin", "inside", "permitted")]
    observed = {"before:/account/role": ["owner", "viewer"]}
    extra = boundary_workloads(_book(rule), admin_base, observed_values=observed)
    assert _cases(extra) == [("admin", "inside", "permitted"), ("viewer", "outside", "refused")]


def test_edits_only_where_parent_exists() -> None:
    """A state path whose parent is missing produces no workload."""
    rule = _rule("deep", _cmp("ge", "before", "/missing/balance", 0))
    assert boundary_workloads(_book(rule), _BASE) == ()
    present = _rule("present", _cmp("ge", "before", "/account/credit", 0))
    workloads = boundary_workloads(_book(present), _BASE)
    assert [w.state["account"] for w in workloads][1] == {
        "balance": 100,
        "role": "agent",
        "rate": 0.5,
        "credit": 0,
    }


def test_only_challengeable_rules_generate_workloads() -> None:
    """Contradicted, invariant, string and boolean comparisons are not challenged."""
    book = _book(
        _rule("bad", _cmp("le", "action", "/arguments/amount", 5), status="contradicted"),
        _rule("inv", _cmp("ge", "before", "/account/balance", 0), kind="invariant"),
        _rule("text", _cmp("lt", "before", "/account/role", "m")),
        _rule("flag", _cmp("eq", "before", "/account/rate", True)),
    )
    assert boundary_workloads(book, _BASE) == ()
    assert requires_witness(book) == ()


def test_effect_workloads_expect_effect_on_both_sides() -> None:
    """An effect boundary inside an implication predicts the effect holds on each side."""
    predicate: JsonObject = {
        "op": "implies",
        "args": [
            _cmp("ge", "before", "/account/balance", 100),
            _cmp("eq", "after", "/account/tier", "gold"),
        ],
    }
    workloads = boundary_workloads(_book(_rule("tier", predicate, kind="effect")), _BASE)
    assert _cases(workloads) == [
        (99, "outside", "effect_holds"),
        (100, "inside", "effect_holds"),
        (101, "inside", "effect_holds"),
    ]


def _observe(
    workload: Workload, observation: JsonValue, after: JsonObject | None = None
) -> Transition:
    """An observed transition for a workload's planned state and action."""
    return Transition(
        before=workload.state,
        after=after if after is not None else workload.state,
        action=workload.action,
        observation=observation,
    )


def test_triage_classifies_and_flags_missing_witness() -> None:
    """Agreement, contradiction and undetermined outcomes are classified per workload."""
    rule = _rule("limit", _cmp("le", "action", "/arguments/amount", 50))
    inside, edge, outside = boundary_workloads(_book(rule), _BASE)
    observed = {
        inside.workload_id: _observe(inside, {"ok": True}),
        edge.workload_id: _observe(edge, {"error": "over limit"}),
        outside.workload_id: _observe(outside, None),
    }
    results = triage((inside, edge, outside), observed)
    assert [(t.verdict, t.needs_witness) for t in results] == [
        ("consistent_with_rule", False),
        ("contradicts_rule", True),
        ("undetermined", False),
    ]
    assert all(not t.witnessed for t in results)
    assert requires_witness(_book(rule)) == ("limit",)
    with pytest.raises(ValueError, match="unknown workloads"):
        triage((inside,), {"nope": _observe(inside, None)})


def test_triage_with_supporting_witness_does_not_require_one() -> None:
    """A supporting witness is recorded; contradiction still never resolves automatically."""
    witness = Witness(kind="native", reference="run:1", supports=True)
    rule = _rule(
        "limit",
        _cmp("le", "action", "/arguments/amount", 50),
        witnesses=[witness.model_dump()],
    )
    _, _, outside = boundary_workloads(_book(rule), _BASE)
    (result,) = triage((outside,), {outside.workload_id: _observe(outside, {"ok": True})})
    assert (result.verdict, result.witnessed, result.needs_witness) == (
        "contradicts_rule",
        True,
        False,
    )
    assert requires_witness(_book(rule)) == ()


def test_triage_effect_refusal_and_other_tool_are_undetermined() -> None:
    """A refused effect action and a different observed tool test nothing."""
    predicate: JsonObject = {
        "op": "implies",
        "args": [
            _cmp("ge", "before", "/account/balance", 100),
            _cmp("eq", "after", "/account/tier", "gold"),
        ],
    }
    _, at, _ = boundary_workloads(_book(_rule("tier", predicate, kind="effect")), _BASE)
    gold: JsonObject = {"account": {"balance": 100, "tier": "gold"}}
    silver: JsonObject = {"account": {"balance": 100, "tier": "silver"}}
    other = Transition(before=at.state, after=at.state, action={"name": "x"}, observation={})
    verdicts = [
        triage((at,), {at.workload_id: transition})[0].verdict
        for transition in (
            _observe(at, {"ok": True}, gold),
            _observe(at, {"ok": True}, silver),
            _observe(at, {"error": "no"}),
            other,
        )
    ]
    assert verdicts == [
        "consistent_with_rule",
        "contradicts_rule",
        "undetermined",
        "undetermined",
    ]


def test_boundaries_inside_connectives_are_found_once() -> None:
    """Nested comparisons are challenged once each; undetermined predictions are dropped."""
    nested: JsonObject = {
        "op": "and",
        "args": [
            _cmp("le", "action", "/arguments/amount", 50),
            {"op": "not", "args": [_cmp("le", "action", "/arguments/amount", 50)]},
            _cmp("in", "before", "/account/role", ["agent"]),
        ],
    }
    examples: dict[str, JsonObject] = {"refund": {"amount": 10}}
    workloads = boundary_workloads(_book(_rule("mix", nested)), _BASE, examples=examples)
    assert {w.boundary for w in workloads} == {
        "action:/arguments/amount le 50",
        "before:/account/role in ['agent']",
    }
    assert Predicate.model_validate(nested) == workloads[0].predicate
    assert {w.expected for w in workloads} == {"refused"}
    bare = boundary_workloads(_book(_rule("mix", nested)), _BASE)
    assert {w.boundary for w in bare} == {"action:/arguments/amount le 50"}
