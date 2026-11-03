"""Tests for selecting traces by fired decision-model propositions."""

from __future__ import annotations

import pytest

from exp.simulation.mining.questions import Question, QuestionBank
from exp.simulation.mining.tag_selection import TagRule, fired_tags, select_traces
from exp.simulation.mining.tagging import TagMatrix, TagRow, TraceTaggingError

_BANK = QuestionBank(
    bank_id="shape-v1",
    version=1,
    questions=(
        Question(question_id="goal.repair", family="goal", text="Is the goal to repair?"),
        Question(question_id="goal.lookup", family="goal", text="Is it one lookup?"),
        Question(question_id="env.gui", family="env", text="Is a GUI driven?"),
    ),
)


def _matrix(**rows: tuple[float, float, float]) -> TagMatrix:
    """Build a matrix over the test bank with one row per keyword trace."""
    return TagMatrix(
        bank_id="shape-v1",
        bank_version=1,
        question_ids=_BANK.question_ids(),
        view="task",
        rows=tuple(
            TagRow(trace_id=trace, unit_hash=f"h-{trace}", values=values)
            for trace, values in rows.items()
        ),
    )


def test_fired_tags_use_a_strict_threshold() -> None:
    """A proposition exactly at the threshold has not fired."""
    matrix = _matrix(a=(0.5, 0.51, 0.0), b=(0.9, 0.1, 0.8))

    assert fired_tags(matrix) == {
        "a": frozenset({"goal.lookup"}),
        "b": frozenset({"goal.repair", "env.gui"}),
    }
    assert fired_tags(matrix, question_ids=("env.gui",)) == {
        "a": frozenset(),
        "b": frozenset({"env.gui"}),
    }


def test_a_rule_combines_required_any_and_excluded_propositions() -> None:
    """Only traces meeting every clause are selected, in row order."""
    matrix = _matrix(
        a=(0.9, 0.1, 0.1),
        b=(0.9, 0.1, 0.9),
        c=(0.2, 0.9, 0.1),
        d=(0.95, 0.6, 0.0),
    )
    rule = TagRule(
        name="repairs-without-gui",
        require_any=("goal.repair", "goal.lookup"),
        exclude=("env.gui",),
    )

    selection = select_traces(matrix, rule, bank=_BANK)

    assert selection.trace_ids == ("a", "c", "d")
    assert selection.uncertain == {"d": ("goal.lookup",)}


def test_a_matrix_for_another_bank_revision_is_refused() -> None:
    """Answers to a reworded bank never satisfy a rule written for the original."""
    matrix = _matrix(a=(0.9, 0.1, 0.1)).model_copy(update={"bank_version": 2})

    with pytest.raises(TraceTaggingError, match="v2"):
        select_traces(matrix, TagRule(name="r", require_all=("goal.repair",)), bank=_BANK)


def test_a_rule_naming_an_unknown_proposition_is_refused() -> None:
    """A typo in a rule fails loudly instead of matching nothing."""
    with pytest.raises(TraceTaggingError, match="goal.unknown"):
        select_traces(
            _matrix(a=(0.9, 0.1, 0.1)),
            TagRule(name="r", require_all=("goal.unknown",)),
            bank=_BANK,
        )


def test_a_rule_must_be_consistent() -> None:
    """A rule cannot be empty or require and exclude the same proposition."""
    with pytest.raises(ValueError, match="names no proposition"):
        TagRule(name="empty")
    with pytest.raises(ValueError, match="both requires and excludes"):
        TagRule(name="clash", require_all=("env.gui",), exclude=("env.gui",))
