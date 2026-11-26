"""Tests for tag result records, their JSON wire format and the turn summary."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from exp.common.tags.bank import DEFAULT_TAG_BANK
from exp.common.tags.records import ExchangeTags, TagFacts, TagSummary, WindowScores, summarize


def _facts() -> TagFacts:
    """Build ordinary exchange facts.

    Returns:
        Facts for a two-turn exchange with one tool call.
    """
    return TagFacts(
        tool_calls=1, turns=2, truncated=False, has_images=True, structured_output=False
    )


def _record(turn: tuple[WindowScores, ...], conversation: WindowScores | None) -> ExchangeTags:
    """Build a record with the given windows.

    Args:
        turn: Turn windows.
        conversation: Opening window or None.

    Returns:
        The record.
    """
    return ExchangeTags(
        exchange_id="req_123",
        conversation_key="conv_abc",
        bank_id=DEFAULT_TAG_BANK.bank_id,
        bank_version=DEFAULT_TAG_BANK.version,
        bank_digest=DEFAULT_TAG_BANK.digest,
        model="clef-27b",
        tagged_at=datetime(2026, 10, 14, 12, 30, 1, 123456, tzinfo=timezone(timedelta(hours=-7))),
        conversation=conversation,
        turn=turn,
        facts=_facts(),
    )


def test_json_round_trip_is_exact() -> None:
    """Serializing and parsing yields an equal record and identical JSON."""
    record = _record(
        turn=(
            WindowScores(window_index=0, start_message=3, end_message=6, scores=(0.1, 0.9)),
            WindowScores(window_index=1, start_message=5, end_message=8, scores=(1 / 3, 0.0)),
        ),
        conversation=WindowScores(
            window_index=0, start_message=0, end_message=2, scores=(0.123456789012345, 1.0)
        ),
    )
    text = record.model_dump_json()
    parsed = ExchangeTags.model_validate_json(text)
    assert parsed == record
    assert parsed.model_dump_json() == text
    assert parsed.turn[1].scores[0] == 1 / 3
    assert parsed.tagged_at.utcoffset() == timedelta(hours=-7)


def test_record_without_opening_round_trips() -> None:
    """An already-tagged conversation leaves the opening as null on the wire."""
    record = _record(
        turn=(WindowScores(window_index=0, start_message=0, end_message=1, scores=(0.5,)),),
        conversation=None,
    )
    assert '"conversation":null' in record.model_dump_json()
    assert ExchangeTags.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan"), float("inf")])
def test_window_rejects_non_probabilities(score: float) -> None:
    """Scores must be finite and within [0, 1].

    Args:
        score: Invalid score.
    """
    with pytest.raises(ValidationError, match="probabilities"):
        WindowScores(window_index=0, start_message=0, end_message=1, scores=(score,))


def test_window_rejects_an_empty_range() -> None:
    """A window must cover at least one message."""
    with pytest.raises(ValidationError, match="at least one message"):
        WindowScores(window_index=0, start_message=4, end_message=4, scores=())


def test_record_requires_ordered_aligned_turn_windows() -> None:
    """Turn windows are numbered in order and share one score count."""
    first = WindowScores(window_index=0, start_message=0, end_message=1, scores=(0.1,))
    with pytest.raises(ValidationError, match="numbered"):
        _record(turn=(first.model_copy(update={"window_index": 1}),), conversation=None)
    with pytest.raises(ValidationError, match="one score per turn tag"):
        _record(
            turn=(
                first,
                WindowScores(window_index=1, start_message=1, end_message=2, scores=(0.1, 0.2)),
            ),
            conversation=None,
        )
    with pytest.raises(ValidationError):
        _record(turn=(), conversation=None)
    with pytest.raises(ValidationError, match="index 0"):
        _record(turn=(first,), conversation=first.model_copy(update={"window_index": 2}))


def test_summarize_takes_per_tag_max_and_min() -> None:
    """Each tag keeps its highest and lowest score across turn windows."""
    record = _record(
        turn=(
            WindowScores(window_index=0, start_message=2, end_message=5, scores=(0.2, 0.9, 0.0)),
            WindowScores(window_index=1, start_message=4, end_message=7, scores=(0.7, 0.4, 0.0)),
            WindowScores(window_index=2, start_message=6, end_message=9, scores=(0.5, 0.6, 1.0)),
        ),
        conversation=None,
    )
    summary = summarize(record)
    assert summary == TagSummary(
        bank_id=DEFAULT_TAG_BANK.bank_id,
        bank_version=1,
        bank_digest=DEFAULT_TAG_BANK.digest,
        turn_max=(0.7, 0.9, 1.0),
        turn_min=(0.2, 0.4, 0.0),
        turn_windows=3,
    )


def test_summarize_single_window_has_equal_bounds() -> None:
    """One window gives identical max and min arrays."""
    window = WindowScores(window_index=0, start_message=0, end_message=1, scores=(0.3, 0.8))
    summary = summarize(_record(turn=(window,), conversation=None))
    assert summary.turn_max == summary.turn_min == (0.3, 0.8)
    assert summary.turn_windows == 1


def test_summary_rejects_inverted_or_misaligned_bounds() -> None:
    """A minimum above its maximum or arrays of different lengths are rejected."""
    with pytest.raises(ValidationError, match="cannot exceed"):
        TagSummary(
            bank_id="b",
            bank_version=1,
            bank_digest="0" * 64,
            turn_max=(0.1,),
            turn_min=(0.2,),
            turn_windows=1,
        )
    with pytest.raises(ValidationError, match="one score per turn tag"):
        TagSummary(
            bank_id="b",
            bank_version=1,
            bank_digest="0" * 64,
            turn_max=(0.1, 0.2),
            turn_min=(0.1,),
            turn_windows=1,
        )


def test_summarize_reads_opening_scores_from_the_tagged_opening() -> None:
    """A later exchange borrows its conversation's opening scores only from the same bank."""
    turn = (WindowScores(window_index=0, start_message=4, end_message=6, scores=(0.4,)),)
    opening_window = WindowScores(window_index=0, start_message=0, end_message=2, scores=(0.9,))
    first = _record(turn=turn, conversation=opening_window)
    later = _record(turn=turn, conversation=None)
    assert summarize(first).conversation == (0.9,)
    assert summarize(later).conversation is None
    assert summarize(later, first).conversation == (0.9,)
    own = _record(
        turn=turn,
        conversation=opening_window.model_copy(update={"scores": (0.1,)}),
    )
    assert summarize(own, first).conversation == (0.1,)
    with pytest.raises(ValueError, match="different bank"):
        summarize(later, first.model_copy(update={"bank_digest": "0" * 64}))
    with pytest.raises(ValueError, match="another conversation"):
        summarize(later, first.model_copy(update={"conversation_key": "other"}))
