"""Offline tests for tag admission: reading, lineage independence, contradiction, guard tables."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from exp.common.core.artifacts import sha256_json
from exp.simulation.rules.admission import (
    AdmissionPolicy,
    AdmissionVerdict,
    QuestionRole,
    RuleQuestion,
    RuleTagMatrix,
    TrajectoryTags,
    admit,
    guard_rulebook,
    guard_table,
    question_id_for,
    read_role,
)
from exp.simulation.rules.checker import check_proposed_transition
from exp.simulation.rules.memory_set import RuleFile
from exp.simulation.rules.model import Predicate, Transition

RULE = "precondition.refund.abc"
POLICY = AdmissionPolicy()

YES, NO, MID = 0.9, 0.1, 0.5
SUPPORTS = (YES, YES, NO)
CONTRADICTS = (YES, NO, YES)
ABSENT = (NO, NO, NO)
UNSURE = (YES, MID, NO)


def questions(*roles: QuestionRole) -> tuple[RuleQuestion, ...]:
    """One question per role, in order."""
    texts = {role: f"Does the trajectory {role} the refund limit?" for role in roles}
    return tuple(
        RuleQuestion(question_id=question_id_for(RULE, role, texts[role]), role=role, text=text)
        for role, text in texts.items()
    )


QUESTIONS = questions("applies", "consistent", "contradicts")


def matrix(rows: Mapping[str, tuple[str, tuple[float, ...]]]) -> RuleTagMatrix:
    """A matrix from ``trace -> (lineage, values)``."""
    return RuleTagMatrix(
        rule_id=RULE,
        question_ids=tuple(q.question_id for q in QUESTIONS),
        rows=tuple(
            TrajectoryTags(trace_id=trace, lineage_id=lineage, values=values)
            for trace, (lineage, values) in rows.items()
        ),
    )


def test_role_reading_needs_every_answer_confident() -> None:
    """Disagreeing variants leave a role uncertain; a role with no question is uncertain."""
    assert read_role((0.8, 0.95), POLICY) == "yes"
    assert read_role((0.1, 0.2), POLICY) == "no"
    assert read_role((0.9, 0.2), POLICY) == "uncertain"
    assert read_role((0.6,), POLICY) == "uncertain"
    assert read_role((), POLICY) == "uncertain"


def test_two_independent_agreements_admit() -> None:
    """Two supporting trajectories of distinct lineages, nothing against: admitted, supported."""
    verdict = admit(
        QUESTIONS,
        matrix({"t1": ("a", SUPPORTS), "t2": ("b", SUPPORTS), "t3": ("c", ABSENT)}),
        POLICY,
    )
    assert verdict.admitted and verdict.status == "supported"
    assert verdict.supporting == ("t1", "t2") and verdict.supporting_lineages == ("a", "b")
    assert verdict.not_applicable == ("t3",)
    assert "t1" in verdict.summary() and "2 independent" in verdict.summary()


def test_one_agreement_is_not_enough() -> None:
    """A single supporting trajectory leaves the rule unknown."""
    verdict = admit(QUESTIONS, matrix({"t1": ("a", SUPPORTS), "t2": ("b", ABSENT)}), POLICY)
    assert not verdict.admitted and verdict.status == "unknown"


def test_same_lineage_agreements_count_once() -> None:
    """Near-duplicates share a lineage, so two of them are one independent agreement."""
    verdict = admit(QUESTIONS, matrix({"t1": ("a", SUPPORTS), "t2": ("a", SUPPORTS)}), POLICY)
    assert not verdict.admitted and verdict.status == "unknown"
    assert verdict.supporting == ("t1", "t2") and verdict.supporting_lineages == ("a",)
    assert verdict.duplicate_support == ("t2",)
    assert "not counted" in verdict.summary()


def test_confident_contradiction_blocks_admission() -> None:
    """One confident contradiction blocks admission however much support there is."""
    verdict = admit(
        QUESTIONS,
        matrix({"t1": ("a", SUPPORTS), "t2": ("b", SUPPORTS), "t3": ("c", CONTRADICTS)}),
        POLICY,
    )
    assert not verdict.admitted and verdict.status == "contradicted"
    assert verdict.contradicting == ("t3",)


def test_uncertain_answers_are_kept_and_never_counted() -> None:
    """An uncertain trajectory neither supports nor contradicts; it is recorded as uncertain."""
    unsure_contradiction = (YES, MID, YES)
    verdict = admit(
        QUESTIONS,
        matrix({"t1": ("a", SUPPORTS), "t2": ("b", UNSURE), "t3": ("c", unsure_contradiction)}),
        POLICY,
    )
    assert not verdict.admitted and verdict.status == "unknown"
    assert verdict.uncertain == ("t2", "t3") and verdict.contradicting == ()
    readings = {v.trace_id: v.reading for v in verdict.trajectories}
    assert readings == {"t1": "supports", "t2": "uncertain", "t3": "uncertain"}


def test_missing_role_and_mismatched_columns() -> None:
    """A rule without every role is never admitted; columns must match the questions."""
    partial = questions("applies", "consistent")
    empty = RuleTagMatrix(rule_id=RULE, question_ids=tuple(q.question_id for q in partial))
    verdict = admit(partial, empty, POLICY)
    assert not verdict.admitted and "contradicts" in verdict.reason
    with pytest.raises(ValueError, match="do not match"):
        admit(QUESTIONS, empty, POLICY)
    with pytest.raises(ValueError, match="every question"):
        RuleTagMatrix(
            rule_id=RULE,
            question_ids=("a", "b"),
            rows=(TrajectoryTags(trace_id="t", lineage_id="l", values=(0.1,)),),
        )
    with pytest.raises(ValueError):
        AdmissionPolicy(minimum_independent=1)


def _rule(rule_id: str, truism: str, **extra: object) -> RuleFile:
    """A minimal rule file."""
    return RuleFile.model_validate(
        {
            "rule_id": rule_id,
            "topic": "refund",
            "tool": "refund",
            "kind": "precondition",
            "truism": truism,
            "reasoning": "Because money cannot be returned twice.",
            "status": "supported",
            **extra,
        }
    )


def test_guard_table_from_admitted_rules_feeds_the_runtime_checker() -> None:
    """Admitted rules become guards; edited, demoted and removed ones are listed as withdrawn."""
    limit = Predicate.model_validate(
        {
            "op": "le",
            "left": {"scope": "action", "path": "/arguments/amount"},
            "right": {"value": 40},
        }
    )
    rows = {"t1": ("a", SUPPORTS), "t2": ("b", SUPPORTS)}
    admitted = admit(QUESTIONS, matrix(rows), POLICY)

    def verdict(rule_id: str) -> AdmissionVerdict:
        return admitted.model_copy(update={"rule_id": rule_id})

    rules = (
        _rule("guard.a", "Refund at most the paid amount.", predicate=limit),
        _rule("guard.b", "A refund needs an order id."),
        _rule("guard.c", "Edited after admission."),
        _rule("guard.d", "Demoted later.", status="unknown"),
        _rule("guard.e", "Never admitted."),
    )
    verdicts = [verdict(f"guard.{x}") for x in "abcdf"]
    verdicts.append(admit(QUESTIONS, matrix({"t1": ("a", SUPPORTS)}), POLICY))
    statements = {
        "guard.a": "Refund at most the paid amount.",
        "guard.b": "A refund needs an order id.",
        "guard.c": "The original truism.",
        "guard.d": "Demoted later.",
        "guard.f": "Removed by governance.",
    }
    table = guard_table(
        "refunds",
        ("refund", "lookup"),
        rules,
        verdicts,
        {"guard.a": QUESTIONS},
        statements,
        POLICY,
    )
    assert [e.rule_id for e in table.action_guards] == ["guard.a", "guard.b"]
    assert table.tools == ("lookup", "refund") and table.policy == POLICY.identity
    assert table.action_guards[0].questions == QUESTIONS
    assert table.action_guards[0].statement_sha256 == sha256_json(rules[0].truism)
    assert table.action_guards[0].supporting_lineages == ("a", "b")
    assert {(w.rule_id, w.reason) for w in table.withdrawn} == {
        ("guard.c", "truism edited after admission"),
        ("guard.d", "governance set it unknown"),
        ("guard.f", "removed"),
    }
    assert table.model_validate_json(table.model_dump_json()) == table

    book = guard_rulebook(table)
    assert [r.rule_id for r in book.rules] == ["guard.a"]
    over = Transition(before={}, after={}, action={"name": "refund", "arguments": {"amount": 400}})
    verdict_over = check_proposed_transition(book, over, refused=False)
    assert not verdict_over.allowed and verdict_over.violations == ("guard.a",)
    within = over.model_copy(update={"action": {"name": "refund", "arguments": {"amount": 10}}})
    assert check_proposed_transition(book, within, refused=False).allowed
