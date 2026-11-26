"""Tests for tag definitions and bank validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exp.common.tags.bank import (
    DEFAULT_TAG_BANK,
    MAX_TAGS_PER_SCOPE,
    TagBank,
    TagDefinition,
    TagScope,
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


def test_index_is_the_position_in_the_scope_array() -> None:
    """Indexes count within each scope, not across the whole bank."""
    bank = TagBank(
        bank_id="b",
        version=1,
        tags=(
            _definition("first", TagScope.CONVERSATION),
            _definition("turn_a"),
            _definition("second", TagScope.CONVERSATION),
            _definition("turn_b"),
        ),
    )
    assert [bank.index(tag_id) for tag_id in ("first", "second", "turn_a", "turn_b")] == [
        0,
        1,
        0,
        1,
    ]
    assert bank.get("second").scope is TagScope.CONVERSATION
    with pytest.raises(KeyError, match="valid tags: first, turn_a, second, turn_b"):
        bank.index("thrid")


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


def test_definition_rejects_blank_wording() -> None:
    """Whitespace-only wording would leave the decision model guessing."""
    with pytest.raises(ValidationError, match="visible text"):
        _definition("a").model_validate({**_definition("a").model_dump(), "true_when": "   "})
