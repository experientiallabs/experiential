"""Tests for rule-derived properties and their checking over transitions."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.evaluations.reconstruction import WorkflowEvidence, WorkflowStep
from exp.simulation.rules.model import Predicate, Rule, RuleBook, Transition
from exp.simulation.rules.properties import (
    Property,
    RefusalCriteria,
    check_transitions,
    is_refusal,
    properties_from,
    transitions_from_workflow,
)


def _cmp(op: str, scope: str, path: str, value: JsonValue) -> Predicate:
    """A comparison of one read against a literal."""
    return Predicate.model_validate(
        {"op": op, "left": {"scope": scope, "path": path}, "right": {"value": value}}
    )


def _step(
    amount: JsonValue, observation: JsonValue, *, after: JsonObject | None = None
) -> Transition:
    """A refund transition with the given amount and observation."""
    before: JsonObject = {"balance": 100, "status": "paid"}
    return Transition(
        before=before,
        after=after if after is not None else before,
        action={"name": "refund", "arguments": {"amount": amount}},
        observation=observation,
    )


_LIMIT = Property(
    rule_id="refund-limit",
    kind="precondition",
    predicate=_cmp("le", "action", "/arguments/amount", 50),
    applies_to="refund",
)


def _only(properties: list[Property], transitions: list[tuple[str, Transition]]) -> dict[str, int]:
    """Counts of the single result of a check."""
    (result,) = check_transitions(properties, transitions).results
    return {
        "holds": result.holds,
        "violated": result.violated,
        "undetermined": result.undetermined,
        "vacuous": result.vacuous,
    }


def test_properties_skip_contradicted_and_predicate_free_rules() -> None:
    """Only non-contradicted rules with a predicate and a target become properties."""
    predicate = _cmp("ge", "before", "/balance", 0)

    def rule(rule_id: str, **fields: object) -> Rule:
        """A rule with defaults overridden."""
        data: dict[str, object] = {
            "rule_id": rule_id,
            "tool": "refund",
            "kind": "precondition",
            "statement": "s",
            "predicate": predicate.model_dump(),
            "status": "supported",
            "support": 1,
            "contradiction": 0,
        }
        data.update(fields)
        return Rule.model_validate(data)

    book = RuleBook(
        contracts={},
        rules=(
            rule("keep"),
            rule("bad", status="contradicted"),
            rule("bare", predicate=None),
            rule("untargeted", tool=None),
            rule("inv", tool=None, kind="invariant", status="unknown"),
        ),
    )
    found = properties_from(book)
    assert [(p.rule_id, p.applies_to) for p in found] == [("keep", "refund"), ("inv", None)]


def test_refusal_recognition_markers_and_parameters() -> None:
    """Error keys, flags and text prefixes mark refusals; None is undetermined."""
    assert is_refusal(_step(1, {"error": "limit"})) is True
    assert is_refusal(_step(1, {"error": None, "ok": True})) is False
    assert is_refusal(_step(1, {"ok": False})) is True
    assert is_refusal(_step(1, {"ok": 0})) is False
    assert is_refusal(_step(1, "  Error: denied")) is True
    assert is_refusal(_step(1, "done")) is False
    assert is_refusal(_step(1, None)) is None
    custom = RefusalCriteria(error_keys=(), flags={"/code": 403}, text_prefixes=("denied",))
    assert is_refusal(_step(1, {"error": "x"}), custom) is False
    assert is_refusal(_step(1, {"code": 403}), custom) is True
    assert is_refusal(_step(1, "Denied"), custom) is True
    strict = RefusalCriteria(require_unchanged_state=True)
    changed = _step(1, {"error": "x"}, after={"balance": 99, "status": "paid"})
    assert is_refusal(changed, strict) is None
    assert is_refusal(_step(1, {"error": "x"}), strict) is True


def test_precondition_requires_refusal_when_predicate_false() -> None:
    """A false precondition must be refused; a true one is only vacuous."""
    transitions = [
        ("a", _step(80, {"error": "over limit"})),
        ("b", _step(80, {"ok": True})),
        ("c", _step(10, {"ok": True})),
        ("d", _step(80, None)),
        ("e", _step(None, {"ok": True})),
        ("f", Transition(before={}, after={}, action={"name": "other"}, observation=None)),
    ]
    assert _only([_LIMIT], transitions) == {
        "holds": 1,
        "violated": 1,
        "undetermined": 2,
        "vacuous": 1,
    }
    (result,) = check_transitions([_LIMIT], transitions).results
    assert result.violations == ("b",)
    assert result.undetermined_at == ("d", "e")


def test_effect_and_observation_checked_only_on_success() -> None:
    """Effects and observations must hold after a permitted action; refusals do not test them."""
    effect = Property(
        rule_id="refund-marks",
        kind="effect",
        predicate=_cmp("eq", "after", "/status", "refunded"),
        applies_to="refund",
    )
    done: JsonObject = {"balance": 100, "status": "refunded"}
    transitions = [
        ("ok", _step(10, {"ok": True}, after=done)),
        ("bad", _step(10, {"ok": True})),
        ("refused", _step(10, {"error": "no"})),
        ("unknown", _step(10, None, after=done)),
    ]
    assert _only([effect], transitions) == {
        "holds": 1,
        "violated": 1,
        "undetermined": 1,
        "vacuous": 0,
    }
    shown = Property(
        rule_id="refund-shows",
        kind="observation",
        predicate=_cmp("eq", "observation", "/ok", True),
        applies_to="refund",
    )
    assert _only([shown], [("x", _step(1, {"ok": True})), ("y", _step(1, {"ok": "yes"}))]) == {
        "holds": 1,
        "violated": 1,
        "undetermined": 0,
        "vacuous": 0,
    }


def test_invariant_checked_on_every_transition_and_absence_is_undetermined() -> None:
    """An invariant reads every transition; a missing value is never a violation."""
    invariant = Property(
        rule_id="non-negative", kind="invariant", predicate=_cmp("ge", "after", "/balance", 0)
    )
    transitions = [
        ("a", _step(1, None)),
        ("b", Transition(before={}, after={"balance": -1}, action={"name": "x"})),
        ("c", Transition(before={}, after={}, action={"name": "y"})),
    ]
    report = check_transitions([invariant], transitions)
    assert report.transitions == 3
    (result,) = report.results
    assert (result.holds, result.violated, result.undetermined) == (1, 1, 1)
    assert result.violations == ("b",)


def test_workflow_conversion_keeps_only_valid_recorded_steps() -> None:
    """Valid steps become located transitions; failed and blocked steps are left out."""
    evidence = WorkflowEvidence(
        workflow_id="wf-1",
        task="Refund the order.",
        initial_state={"balance": 100},
        steps=(
            WorkflowStep(
                step=0,
                action={"name": "refund", "arguments": {"amount": 80}},
                status="valid",
                reply={"error": "over limit"},
                before={"balance": 100},
                after={"balance": 100},
            ),
            WorkflowStep(
                step=1,
                action={"name": "refund", "arguments": {"amount": 10}},
                status="failure",
                before={"balance": 100},
                after={"balance": 100},
                error_type="TimeoutError",
            ),
            WorkflowStep(step=2, action={"name": "refund", "arguments": {}}, status="blocked"),
        ),
    )
    transitions = transitions_from_workflow(evidence)
    assert [locator for locator, _ in transitions] == ["workflow:wf-1:step:0"]
    _, transition = transitions[0]
    assert transition.observation == {"error": "over limit"}
    assert _only([_LIMIT], list(transitions))["holds"] == 1
