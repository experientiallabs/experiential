"""Tests for tag definitions, bank validation and the default bank."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exp.common.tags.bank import (
    DEFAULT_TAG_BANK,
    MAX_TAG_TEXT_CHARS,
    MAX_TAGS_PER_SCOPE,
    TagBank,
    TagDefinition,
    TagScope,
)

CONVERSATION_IDS = (
    "coding",
    "cybersecurity",
    "customer_support",
    "personal_assistant",
    "research",
    "web_lookup",
    "data_analysis",
    "writing_editing",
    "tutoring",
    "roleplay_companion",
    "math_science",
    "finance_legal_compliance",
    "health_medical",
    "autonomous_agent",
    "harmful_request",
)
TURN_IDS = (
    "tool_error",
    "user_frustrated",
    "positive_feedback",
    "user_corrected_model",
    "model_mistake",
    "model_refused",
    "model_asked_clarification",
    "task_finished",
    "claimed_success_without_evidence",
    "took_external_action",
)


def _definition(tag_id: str, scope: TagScope = TagScope.TURN) -> TagDefinition:
    """Build a minimal valid definition.

    Args:
        tag_id: Tag id.
        scope: Tag scope.

    Returns:
        The definition.
    """
    return TagDefinition(
        id=tag_id, scope=scope, question="Is it so?", true_when="It is.", false_when="It is not."
    )


def test_default_bank_has_the_exact_ids_in_score_order() -> None:
    """The default bank pins 15 conversation then 10 turn tags in the documented order."""
    assert DEFAULT_TAG_BANK.bank_id == "experiential-default"
    assert DEFAULT_TAG_BANK.version == 1
    assert DEFAULT_TAG_BANK.ids == CONVERSATION_IDS + TURN_IDS
    conversation = DEFAULT_TAG_BANK.for_scope(TagScope.CONVERSATION)
    turn = DEFAULT_TAG_BANK.for_scope(TagScope.TURN)
    assert tuple(tag.id for tag in conversation) == CONVERSATION_IDS
    assert tuple(tag.id for tag in turn) == TURN_IDS


def test_default_wording_is_complete_and_bounded() -> None:
    """Every default tag asks a question and words both outcomes within the length bound."""
    for tag in DEFAULT_TAG_BANK.tags:
        assert tag.question.endswith("?"), tag.id
        for text in (tag.question, tag.true_when, tag.false_when):
            assert 20 <= len(text) <= MAX_TAG_TEXT_CHARS, (tag.id, text)
            assert "\u2014" not in text
        assert tag.true_when != tag.false_when


def test_turn_questions_point_at_the_current_turn() -> None:
    """Turn wording says which slice to judge so a window is not read as the whole chat."""
    for tag in DEFAULT_TAG_BANK.for_scope(TagScope.TURN):
        wording = f"{tag.question} {tag.true_when} {tag.false_when}".lower()
        assert "turn" in wording or "new user message" in wording, tag.id


def test_index_is_the_position_in_the_scope_array() -> None:
    """Indexes count within each scope, not across the whole bank."""
    assert DEFAULT_TAG_BANK.index("coding") == 0
    assert DEFAULT_TAG_BANK.index("harmful_request") == 14
    assert DEFAULT_TAG_BANK.index("tool_error") == 0
    assert DEFAULT_TAG_BANK.index("took_external_action") == 9
    assert DEFAULT_TAG_BANK.get("task_finished").scope is TagScope.TURN


def test_unknown_tag_lists_valid_ids() -> None:
    """Looking up an unknown id names every valid id."""
    with pytest.raises(KeyError, match="valid tags: coding, cybersecurity"):
        DEFAULT_TAG_BANK.index("codeing")


def test_digest_tracks_wording_and_order() -> None:
    """The digest is stable for equal banks and changes with wording or order."""
    first = TagBank(bank_id="b", version=1, tags=(_definition("a"), _definition("b")))
    same = TagBank.model_validate_json(first.model_dump_json())
    reordered = TagBank(bank_id="b", version=1, tags=(_definition("b"), _definition("a")))
    reworded = TagBank(
        bank_id="b",
        version=1,
        tags=(_definition("a"), _definition("b").model_copy(update={"question": "Is it?"})),
    )
    assert first.digest == same.digest
    assert len({first.digest, reordered.digest, reworded.digest}) == 3
    assert (
        DEFAULT_TAG_BANK.digest
        == TagBank.model_validate_json(DEFAULT_TAG_BANK.model_dump_json()).digest
    )


def test_bank_rejects_duplicate_ids_across_scopes() -> None:
    """One id cannot appear twice, even in different scopes."""
    with pytest.raises(ValidationError, match="more than once"):
        TagBank(
            bank_id="b",
            version=1,
            tags=(_definition("x", TagScope.CONVERSATION), _definition("x", TagScope.TURN)),
        )


def test_bank_allows_the_scope_limit_and_rejects_one_more() -> None:
    """Each scope holds at most 32 tags, independently of the other scope."""
    full = tuple(_definition(f"t{i}") for i in range(MAX_TAGS_PER_SCOPE))
    other = tuple(_definition(f"c{i}", TagScope.CONVERSATION) for i in range(MAX_TAGS_PER_SCOPE))
    bank = TagBank(bank_id="b", version=1, tags=full + other)
    assert len(bank.for_scope(TagScope.TURN)) == MAX_TAGS_PER_SCOPE
    with pytest.raises(ValidationError, match="the turn scope has 33 tags"):
        TagBank(bank_id="b", version=1, tags=(*full, _definition("extra")))


@pytest.mark.parametrize("tag_id", ["Coding", "1st", "two__underscores", "trailing_", "a-b", ""])
def test_definition_rejects_non_snake_case_ids(tag_id: str) -> None:
    """Ids must be lowercase snake_case starting with a letter.

    Args:
        tag_id: Invalid id.
    """
    with pytest.raises(ValidationError):
        _definition(tag_id)


@pytest.mark.parametrize("keyword", ["and", "or", "not", "all"])
def test_definition_rejects_rule_keywords(keyword: str) -> None:
    """Rule keywords cannot name a tag.

    Args:
        keyword: Reserved word.
    """
    with pytest.raises(ValidationError, match="rule keyword"):
        _definition(keyword)


def test_definition_rejects_blank_and_overlong_wording() -> None:
    """Wording must be visible and within the length bound."""
    with pytest.raises(ValidationError, match="visible text"):
        _definition("a").model_validate({**_definition("a").model_dump(), "true_when": "   "})
    with pytest.raises(ValidationError):
        TagDefinition(
            id="a",
            scope=TagScope.TURN,
            question="x" * (MAX_TAG_TEXT_CHARS + 1),
            true_when="t",
            false_when="f",
        )


def test_bank_requires_a_tag_and_a_positive_version() -> None:
    """Empty banks and version zero are rejected."""
    with pytest.raises(ValidationError):
        TagBank(bank_id="b", version=1, tags=())
    with pytest.raises(ValidationError):
        TagBank(bank_id="b", version=0, tags=(_definition("a"),))


@pytest.mark.parametrize(
    ("tag_id", "fragment"),
    [
        ("cybersecurity", "whether defensive, authorized or malicious"),
        ("cybersecurity", "Whether the request is legitimate does not matter"),
        ("math_science", "Other topics may apply too"),
        ("research", "more than one source or step"),
        ("research", "A single lookup"),
        ("web_lookup", "search or browse the web"),
        ("tutoring", "even in a school subject"),
        ("positive_feedback", "thanks in advance"),
        ("positive_feedback", "thanking someone else"),
        ("took_external_action", "files in its local workspace or repository"),
        ("took_external_action", "drafting something for the user to send"),
        ("claimed_success_without_evidence", "can finish the task and still be true"),
        ("task_finished", "whether or not anything confirms it worked"),
        ("user_frustrated", "a boss, a vendor or a situation"),
    ],
)
def test_default_wording_keeps_the_labeling_rules(tag_id: str, fragment: str) -> None:
    """Default wording states the boundary cases the labeled test set settled.

    Args:
        tag_id: Tag whose wording is checked.
        fragment: Text the question or criteria must contain.
    """
    tag = DEFAULT_TAG_BANK.get(tag_id)
    assert fragment in f"{tag.question} {tag.true_when} {tag.false_when}"
