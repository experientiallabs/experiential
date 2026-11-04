"""Tests for merging per-trajectory rule proposals into a rule book."""

from __future__ import annotations

import pytest

from exp.simulation.rules.merge import (
    apply_weighings,
    attach_witnesses,
    check_weighing,
    diagnostic_rules,
    locator_trace,
    merge_rules,
)
from exp.simulation.rules.model import (
    Operand,
    Predicate,
    ResearchQuestion,
    RuleBook,
    RuleKind,
    RuleProposal,
    RuleWeighing,
    ToolContract,
    TrajectoryRules,
    Witness,
)

_PAID = Predicate(
    op="eq", left=Operand(scope="before", path="/order/status"), right=Operand(value="paid")
)
_NOT_PAID = Predicate(op="not", args=(_PAID,))
_UNPAID = Predicate(
    op="ne", left=Operand(scope="before", path="/order/status"), right=Operand(value="paid")
)
_CONTRACTS = {"refund": ToolContract(name="refund", calls=3)}


def _proposal(
    statement: str,
    trace: str,
    *,
    tool: str | None = "refund",
    kind: RuleKind = "precondition",
    predicate: Predicate | None = None,
    counter: tuple[str, ...] = (),
) -> RuleProposal:
    """A proposal whose evidence points into the given trace."""
    return RuleProposal(
        tool=tool,
        kind=kind,
        statement=statement,
        predicate=predicate,
        evidence=(f"trace:{trace}:span:1",),
        counter_evidence=counter,
        confidence=0.7,
    )


def _analysis(
    trace: str, *rules: RuleProposal, questions: tuple[ResearchQuestion, ...] = ()
) -> TrajectoryRules:
    """One trajectory's analysis."""
    return TrajectoryRules(trace_id=trace, rules=rules, questions=questions, limitations=("x",))


def test_same_statement_groups_and_counts_distinct_trajectories() -> None:
    """Normalized statements match exactly; repeats in one trace count once."""
    book = merge_rules(
        _CONTRACTS,
        [
            _analysis(
                "t1", _proposal("Order must be paid.", "t1"), _proposal("order MUST be  paid", "t1")
            ),
            _analysis("t2", _proposal("Order must be paid!", "t2")),
            _analysis("t3", _proposal("Order must be settled.", "t3")),
        ],
    )
    assert len(book.rules) == 2
    paid = next(rule for rule in book.rules if rule.support == 2)
    assert paid.status == "unknown" and paid.weighing is None
    assert paid.evidence == ("trace:t1:span:1", "trace:t2:span:1")
    settled = next(rule for rule in book.rules if rule.support == 1)
    assert settled.status == "unknown"
    assert book.sources == ("t1", "t2", "t3")
    assert book.contracts == _CONTRACTS


def test_tool_and_kind_keep_rules_apart() -> None:
    """The same sentence about a different tool or kind is a different rule."""
    book = merge_rules(
        {},
        [
            _analysis("t1", _proposal("Needs a paid order", "t1")),
            _analysis("t2", _proposal("Needs a paid order", "t2", tool="charge")),
            _analysis("t3", _proposal("Needs a paid order", "t3", kind="effect")),
        ],
    )
    assert len(book.rules) == 3
    assert len({rule.rule_id for rule in book.rules}) == 3


def test_identical_predicates_group_despite_different_wording() -> None:
    """Canonical predicate equality groups proposals whatever their statements say."""
    book = merge_rules(
        _CONTRACTS,
        [
            _analysis("t1", _proposal("Order must be paid", "t1", predicate=_PAID)),
            _analysis("t2", _proposal("Refund requires payment", "t2", predicate=_PAID)),
        ],
    )
    (rule,) = book.rules
    assert rule.predicate == _PAID
    assert rule.status == "unknown"
    assert rule.support == 2


def test_predicate_is_dropped_when_grouped_proposals_disagree() -> None:
    """A statement-only proposal joins differing predicates, and no predicate survives."""
    other = Predicate(op="exists", left=Operand(scope="before", path="/order/paid_at"))
    book = merge_rules(
        _CONTRACTS,
        [
            _analysis("t1", _proposal("Order must be paid", "t1", predicate=_PAID)),
            _analysis("t2", _proposal("Order must be paid", "t2", predicate=other)),
            _analysis("t3", _proposal("Order must be paid", "t3")),
        ],
    )
    (rule,) = book.rules
    assert rule.predicate is None
    assert rule.support == 3


def test_predicate_is_kept_when_statement_only_proposals_join() -> None:
    """Proposals without a predicate do not veto the one predicate the group agreed on."""
    book = merge_rules(
        _CONTRACTS,
        [
            _analysis("t1", _proposal("Order must be paid", "t1", predicate=_PAID)),
            _analysis("t2", _proposal("Order must be paid", "t2")),
        ],
    )
    (rule,) = book.rules
    assert rule.predicate == _PAID


def test_counter_evidence_contradicts_and_is_preserved() -> None:
    """Counter-evidence from any trajectory keeps the rule contradicted with all locators."""
    book = merge_rules(
        _CONTRACTS,
        [
            _analysis("t1", _proposal("Order must be paid", "t1")),
            _analysis("t2", _proposal("Order must be paid", "t2")),
            _analysis("t3", _proposal("Order must be paid", "t3", counter=("trace:t3:span:9",))),
        ],
    )
    (rule,) = book.rules
    assert rule.status == "contradicted"
    assert rule.support == 3
    assert rule.contradiction == 1
    assert rule.counter_evidence == ("trace:t3:span:9",)


@pytest.mark.parametrize("opposite", [_NOT_PAID, _UNPAID])
def test_opposite_predicates_contradict_each_other(opposite: Predicate) -> None:
    """A rule and its simple negation both stay, each contradicted by the other's evidence."""
    book = merge_rules(
        _CONTRACTS,
        [
            _analysis("t1", _proposal("Order must be paid", "t1", predicate=_PAID)),
            _analysis("t2", _proposal("Order must be paid", "t2", predicate=_PAID)),
            _analysis("t3", _proposal("Unpaid orders refund", "t3", predicate=opposite)),
        ],
    )
    paid = next(rule for rule in book.rules if rule.predicate == _PAID)
    other = next(rule for rule in book.rules if rule.predicate == opposite)
    assert paid.status == other.status == "contradicted"
    assert paid.contradiction == 1
    assert paid.counter_evidence == ("trace:t3:span:1",)
    assert other.contradiction == 2
    assert other.counter_evidence == ("trace:t1:span:1", "trace:t2:span:1")


def test_opposites_across_kinds_do_not_contradict() -> None:
    """Opposition is only detected within one tool and kind."""
    book = merge_rules(
        _CONTRACTS,
        [
            _analysis("t1", _proposal("a", "t1", predicate=_PAID)),
            _analysis("t2", _proposal("b", "t2", predicate=_NOT_PAID, kind="effect")),
        ],
    )
    assert {rule.status for rule in book.rules} == {"unknown"}


def test_input_order_does_not_change_the_book() -> None:
    """Shuffling analyses and proposals gives an identical book."""
    analyses = [
        _analysis(
            "t1",
            _proposal("Order must be paid", "t1", predicate=_PAID),
            _proposal("Writes an audit row", "t1", kind="effect", tool=None),
        ),
        _analysis("t2", _proposal("order must be paid.", "t2")),
        _analysis(
            "t3",
            _proposal("Writes an audit row", "t3", kind="effect", tool=None),
            _proposal("Unpaid", "t3", predicate=_NOT_PAID),
            questions=(ResearchQuestion(question="Is zero refundable?", why_it_matters="limits"),),
        ),
    ]
    forward = merge_rules(_CONTRACTS, analyses)
    shuffled = [
        _analysis(a.trace_id, *reversed(a.rules), questions=a.questions) for a in reversed(analyses)
    ]
    assert merge_rules(_CONTRACTS, shuffled) == forward
    assert all(rule.rule_id == rule.rule_id.lower() for rule in forward.rules)


def test_rule_ids_are_stable_across_added_evidence() -> None:
    """More supporting trajectories with the same content keep the rule identity."""
    one = merge_rules(_CONTRACTS, [_analysis("t1", _proposal("Order must be paid", "t1"))])
    two = merge_rules(
        _CONTRACTS,
        [
            _analysis("t1", _proposal("Order must be paid", "t1")),
            _analysis("t2", _proposal("ORDER must be paid", "t2")),
        ],
    )
    assert one.rules[0].rule_id == two.rules[0].rule_id


def test_questions_are_deduplicated_and_keep_evidence() -> None:
    """Questions with the same normalized text merge their evidence and reasons."""
    first = ResearchQuestion(
        tool="refund", question="Can amount be zero?", why_it_matters="a", evidence=("e1",)
    )
    second = ResearchQuestion(
        tool="refund", question="can amount be ZERO", why_it_matters="b", evidence=("e2",)
    )
    book = merge_rules(
        {}, [_analysis("t1", questions=(first,)), _analysis("t2", questions=(second,))]
    )
    (question,) = book.questions
    assert question.evidence == ("e1", "e2")
    assert question.why_it_matters == "a; b"


def _book() -> RuleBook:
    """A book with one rule seen in two trajectories, not yet weighed."""
    return merge_rules(
        _CONTRACTS,
        [
            _analysis("t1", _proposal("Order must be paid", "t1")),
            _analysis("t2", _proposal("Order must be paid", "t2")),
        ],
    )


def test_refuting_witness_contradicts_and_supporting_one_clears_diagnostic() -> None:
    """Witnesses attach by rule id; only a supporting witness takes a rule out of diagnostics."""
    book = _book()
    (rule,) = book.rules
    assert diagnostic_rules(book) == (rule,)
    native = Witness(kind="native", reference="run:1", supports=True)
    supported = attach_witnesses(book, {rule.rule_id: [native, native]})
    assert supported.rules[0].witnesses == (native,)
    assert supported.rules[0].status == "unknown"
    assert diagnostic_rules(supported) == ()
    refuting = Witness(kind="specification", reference="spec:2", supports=False)
    refuted = attach_witnesses(supported, {rule.rule_id: [refuting]})
    assert refuted.rules[0].status == "contradicted"
    assert len(refuted.rules[0].witnesses) == 2


def test_witnesses_for_unknown_rules_are_refused() -> None:
    """A witness keyed by a missing rule id is an error, not silently dropped."""
    with pytest.raises(ValueError, match="not in the rule book"):
        attach_witnesses(
            _book(), {"missing": [Witness(kind="native", reference="r", supports=True)]}
        )


def _weighing(rule_id: str, **fields: object) -> RuleWeighing:
    """A weighing of one rule with overrides."""
    values: dict[str, object] = {
        "rule_id": rule_id,
        "status": "supported",
        "rationale": "Directly observed.",
        "supporting": ("trace:t1:span:1",),
        **fields,
    }
    return RuleWeighing.model_validate(values)


def test_no_count_decides_support_and_a_valid_weighing_does() -> None:
    """One witness is enough when the weighing says so; two stay unknown when it says so."""
    book = merge_rules(
        _CONTRACTS,
        [
            _analysis("t1", _proposal("Order must be paid", "t1")),
            _analysis("t2", _proposal("Order must be settled", "t2")),
            _analysis("t3", _proposal("Order must be settled", "t3")),
        ],
    )
    paid = next(rule for rule in book.rules if rule.support == 1)
    settled = next(rule for rule in book.rules if rule.support == 2)
    assert {rule.status for rule in book.rules} == {"unknown"}
    weighed = apply_weighings(
        book,
        [
            _weighing(paid.rule_id),
            _weighing(
                settled.rule_id,
                status="unknown",
                rationale="Correlated.",
                supporting=settled.evidence,
            ),
        ],
    )
    after = {rule.rule_id: rule for rule in weighed.rules}
    assert after[paid.rule_id].status == "supported"
    assert after[settled.rule_id].status == "unknown"
    assert after[settled.rule_id].evidence == settled.evidence


def test_weighing_structure_is_checked_and_held_out_traces_are_excluded() -> None:
    """Citations stay within the rule's own evidence, never reach excluded traces, and back the
    decided side; merging refuses an analysis that reaches an excluded trace."""
    book = _book()
    (rule,) = book.rules
    with pytest.raises(ValueError, match="outside the rule's evidence"):
        check_weighing(rule, _weighing(rule.rule_id, supporting=("trace:t9:span:1",)), frozenset())
    with pytest.raises(ValueError, match="excluded traces"):
        check_weighing(rule, _weighing(rule.rule_id), frozenset({"t1"}))
    with pytest.raises(ValueError, match="at least one supporting"):
        check_weighing(rule, _weighing(rule.rule_id, supporting=()), frozenset())
    with pytest.raises(ValueError, match="at least one contradicting"):
        check_weighing(rule, _weighing(rule.rule_id, status="contradicted"), frozenset())
    with pytest.raises(ValueError, match="not in the book"):
        apply_weighings(book, [_weighing("missing")])
    with pytest.raises(ValueError, match="weighed twice"):
        apply_weighings(book, [_weighing(rule.rule_id), _weighing(rule.rule_id)])
    with pytest.raises(ValueError, match="held-out evidence"):
        merge_rules(_CONTRACTS, [_analysis("t1", _proposal("x", "t1"))], excluded=frozenset({"t1"}))
    with pytest.raises(ValueError, match="held-out evidence"):
        merge_rules(
            _CONTRACTS,
            [_analysis("t1", _proposal("x", "t1", counter=("trace:held:span:2",)))],
            excluded=frozenset({"held"}),
        )
    assert locator_trace("trace:a:b:span:c") == "a:b"
    assert locator_trace("trace:a") == "a" and locator_trace("contract:x") is None
