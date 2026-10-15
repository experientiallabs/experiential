"""Tests for tag rule parsing, the JSON rule tree and rule evaluation."""

from __future__ import annotations

import pytest

from exp.common.tags.bank import DEFAULT_TAG_BANK, TagBank, TagDefinition, TagScope
from exp.common.tags.predicates import (
    MAX_RULE_DEPTH,
    AndRule,
    NotRule,
    OrRule,
    TagCondition,
    TagRule,
    TagRuleError,
    evaluate_rule,
    parse_rule,
)
from exp.common.tags.records import TagSummary

TURN_COUNT = len(DEFAULT_TAG_BANK.for_scope(TagScope.TURN))
CONVERSATION_COUNT = len(DEFAULT_TAG_BANK.for_scope(TagScope.CONVERSATION))


def _summary(
    high: dict[str, float] | None = None, low: dict[str, float] | None = None
) -> TagSummary:
    """Build a default-bank summary with chosen turn scores.

    Args:
        high: Turn maximum per tag id; unspecified tags score 0.
        low: Turn minimum per tag id; unspecified tags copy the maximum.

    Returns:
        The summary.
    """
    high = high or {}
    low = low or {}
    turn_max = [0.0] * TURN_COUNT
    for tag_id, score in high.items():
        turn_max[DEFAULT_TAG_BANK.index(tag_id)] = score
    turn_min = list(turn_max)
    for tag_id, score in low.items():
        turn_min[DEFAULT_TAG_BANK.index(tag_id)] = score
    return TagSummary(
        bank_id=DEFAULT_TAG_BANK.bank_id,
        bank_version=DEFAULT_TAG_BANK.version,
        bank_digest=DEFAULT_TAG_BANK.digest,
        turn_max=tuple(turn_max),
        turn_min=tuple(turn_min),
        turn_windows=2,
    )


def _conversation(**scores: float) -> tuple[float, ...]:
    """Build default-bank conversation scores.

    Args:
        **scores: Score per conversation tag id; unspecified tags score 0.

    Returns:
        Scores in bank conversation order.
    """
    values = [0.0] * CONVERSATION_COUNT
    for tag_id, score in scores.items():
        values[DEFAULT_TAG_BANK.index(tag_id)] = score
    return tuple(values)


def _matches(rule: str, summary: TagSummary, conversation: tuple[float, ...] | None) -> bool:
    """Parse and evaluate a rule against the default bank.

    Args:
        rule: Rule text.
        summary: Turn summary.
        conversation: Conversation scores.

    Returns:
        Whether the rule matches.
    """
    return evaluate_rule(parse_rule(rule), summary, conversation)


def test_bare_tag_defaults_to_max_at_least_half() -> None:
    """A bare tag compiles to max >= 0.5 with its scope and index."""
    rule = parse_rule("tool_error")
    assert rule.root == TagCondition(
        tag="tool_error", scope=TagScope.TURN, index=0, aggregate="max", op=">=", threshold=0.5
    )
    assert _matches("tool_error", _summary({"tool_error": 0.5}), None)
    assert not _matches("tool_error", _summary({"tool_error": 0.49}), None)


def test_and_binds_tighter_than_or() -> None:
    """``a OR b AND c`` groups as ``a OR (b AND c)``."""
    root = parse_rule("tool_error OR model_mistake AND model_refused").root
    assert isinstance(root, OrRule)
    assert isinstance(root.operands[1], AndRule)
    summary = _summary({"tool_error": 0.9})
    assert _matches("tool_error OR model_mistake AND model_refused", summary, None)
    assert not _matches("(tool_error OR model_mistake) AND model_refused", summary, None)


def test_not_binds_tighter_than_and_and_stacks() -> None:
    """NOT applies to the next unary term and may repeat."""
    root = parse_rule("NOT tool_error AND model_refused").root
    assert isinstance(root, AndRule)
    assert isinstance(root.operands[0], NotRule)
    summary = _summary({"model_refused": 0.8})
    assert _matches("NOT tool_error AND model_refused", summary, None)
    assert not _matches("NOT (tool_error OR model_refused)", summary, None)
    assert _matches("NOT NOT model_refused", summary, None)


def test_chains_flatten_into_one_node() -> None:
    """Repeated AND or OR produce one node with every operand in source order."""
    root = parse_rule("tool_error AND model_mistake AND task_finished").root
    assert isinstance(root, AndRule)
    assert [node.tag for node in root.operands if isinstance(node, TagCondition)] == [
        "tool_error",
        "model_mistake",
        "task_finished",
    ]


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("tool_error > 0.7", False),
        ("tool_error >= 0.7", True),
        ("tool_error < 0.7", False),
        ("tool_error <= 0.7", True),
        ("tool_error>0.69", True),
        ("tool_error < .8", True),
        ("tool_error >= 1", False),
        ("tool_error >= 0", True),
    ],
)
def test_thresholds_and_comparisons(rule: str, expected: bool) -> None:
    """Each operator compares the score with the written threshold.

    Args:
        rule: Rule text.
        expected: Whether a 0.7 score matches.
    """
    assert _matches(rule, _summary({"tool_error": 0.7}), None) is expected


def test_all_reads_the_minimum_across_windows() -> None:
    """``all(tag)`` holds only when every window clears the threshold."""
    summary = _summary({"model_mistake": 0.9}, {"model_mistake": 0.3})
    assert _matches("model_mistake", summary, None)
    assert not _matches("all(model_mistake)", summary, None)
    assert _matches("all(model_mistake) >= 0.3", summary, None)
    assert _matches("NOT all(model_mistake)", summary, None)
    condition = parse_rule("ALL( model_mistake ) < 0.4").root
    assert isinstance(condition, TagCondition)
    assert (condition.aggregate, condition.op, condition.threshold) == ("min", "<", 0.4)


def test_conversation_tags_read_the_opening_scores() -> None:
    """Conversation tags read the supplied opening scores, with or without all()."""
    summary = _summary({"tool_error": 0.9})
    conversation = _conversation(coding=0.8)
    assert _matches("coding AND tool_error", summary, conversation)
    assert _matches("all(coding) > 0.75", summary, conversation)
    assert not _matches("cybersecurity OR NOT tool_error", summary, conversation)


def test_conversation_tag_without_opening_scores_fails_even_when_short_circuited() -> None:
    """A rule using a conversation tag always needs the opening scores."""
    with pytest.raises(TagRuleError, match="no conversation scores"):
        _matches("tool_error OR coding", _summary({"tool_error": 1.0}), None)


def test_keywords_are_case_insensitive() -> None:
    """AND, OR, NOT and all may be written in any letter case."""
    upper = parse_rule("NOT tool_error AND model_refused OR task_finished").root
    lower = parse_rule("not tool_error and model_refused or task_finished").root
    assert upper == lower


def test_rule_tree_round_trips_through_json() -> None:
    """The parsed tree is plain JSON another engine can read back exactly."""
    rule = parse_rule("coding AND (tool_error OR all(model_mistake) > 0.8) AND NOT model_refused")
    payload = rule.model_dump(mode="json")
    assert payload["root"]["kind"] == "and"
    assert payload["root"]["operands"][1]["operands"][1] == {
        "kind": "tag",
        "tag": "model_mistake",
        "scope": "turn",
        "index": 4,
        "aggregate": "min",
        "op": ">",
        "threshold": 0.8,
    }
    assert TagRule.model_validate_json(rule.model_dump_json()) == rule
    assert [condition.tag for condition in rule.conditions()] == [
        "coding",
        "tool_error",
        "model_mistake",
        "model_refused",
    ]


def test_unknown_tag_suggests_and_lists_valid_ids() -> None:
    """An unknown id names the closest match and every valid id."""
    with pytest.raises(TagRuleError) as caught:
        parse_rule("coding AND tool_eror")
    message = str(caught.value)
    assert message.startswith("position 11: unknown tag 'tool_eror' (did you mean 'tool_error'?)")
    assert "valid tags in experiential-default v1: coding, cybersecurity" in message
    assert "took_external_action" in message
    with pytest.raises(TagRuleError, match="did you mean 'coding'"):
        parse_rule("CODING")


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        ("", "at least one tag"),
        ("   ", "at least one tag"),
        ("coding AND", "position 10: expected a tag id, found end of rule"),
        ("coding tool_error", "expected AND, OR or the end of the rule, found 'tool_error'"),
        ("(coding OR tool_error", "expected '\\)' to match the '\\(' at position 0"),
        ("coding)", "found '\\)'"),
        ("all tool_error", "write all\\(tag\\)"),
        ("all(tool_error > 0.5)", "expected '\\)' after the tag in all"),
        ("tool_error > high", "expected a threshold between 0 and 1"),
        ("tool_error >= 1.5", "threshold 1.5 must be between 0 and 1"),
        ("tool_error = 0.5", "unexpected character '='"),
        ("AND tool_error", "expected a tag id, found 'AND'"),
        ("(coding) > 0.5", "expected AND, OR or the end of the rule, found '>'"),
    ],
)
def test_malformed_rules_explain_the_problem(rule: str, message: str) -> None:
    """Syntax errors say what was expected and where.

    Args:
        rule: Malformed rule text.
        message: Expected error fragment.
    """
    with pytest.raises(TagRuleError, match=message):
        parse_rule(rule)


def test_error_points_a_caret_at_the_position() -> None:
    """The error repeats the rule with a caret under the problem."""
    with pytest.raises(TagRuleError) as caught:
        parse_rule("coding AND")
    assert str(caught.value).splitlines()[1:] == ["  coding AND", "            ^"]


def test_nesting_is_bounded() -> None:
    """Deeply nested rules fail with a clear limit instead of exhausting the stack."""
    within = "(" * MAX_RULE_DEPTH + "coding" + ")" * MAX_RULE_DEPTH
    assert isinstance(parse_rule(within).root, TagCondition)
    with pytest.raises(TagRuleError, match="nests deeper than 32 levels"):
        parse_rule("(" + within + ")")
    with pytest.raises(TagRuleError, match="nests deeper"):
        parse_rule("NOT " * (MAX_RULE_DEPTH + 1) + "coding")


def test_evaluation_rejects_scores_from_another_bank() -> None:
    """A rule only evaluates against scores of the bank it was parsed for."""
    other = TagBank(
        bank_id="custom",
        version=2,
        tags=(
            TagDefinition(
                id="tool_error",
                scope=TagScope.TURN,
                question="Did a tool fail?",
                true_when="A tool result reports a failure.",
                false_when="No tool result reports a failure.",
            ),
        ),
    )
    rule = parse_rule("tool_error", other)
    assert (rule.bank_id, rule.bank_version) == ("custom", 2)
    with pytest.raises(TagRuleError, match="parse the rule against that bank"):
        evaluate_rule(rule, _summary(), None)


def test_evaluation_rejects_edited_banks_under_the_same_version() -> None:
    """Scores from reordered or reworded tags under one id and version are rejected."""
    rule = parse_rule("tool_error")
    edited = _summary().model_copy(update={"bank_digest": "0" * 64})
    with pytest.raises(TagRuleError, match="different experiential-default v1 wording or order"):
        evaluate_rule(rule, edited, None)


def test_evaluation_requires_one_score_per_tag() -> None:
    """Short or long score arrays are rejected before any index is read."""
    with pytest.raises(TagRuleError, match="expected 15 conversation scores.*got 2"):
        _matches("tool_error", _summary(), (0.0, 0.0))
    short = _summary().model_copy(update={"turn_max": (0.9,), "turn_min": (0.9,)})
    with pytest.raises(TagRuleError, match="expected 10 turn scores.*got 1"):
        evaluate_rule(parse_rule("tool_error"), short, None)
