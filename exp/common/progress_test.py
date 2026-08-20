"""Tests for the shared observational progress contract."""

from __future__ import annotations

import pytest

from exp.common.progress import ProgressEvent, report


def test_report_emits_one_frozen_event() -> None:
    """One call delivers one immutable event carrying exact counts."""
    seen: list[ProgressEvent] = []
    report(seen.append, "embeddings", completed=3, total=7, detail="serving index")
    assert seen == [ProgressEvent(stage="embeddings", completed=3, total=7, detail="serving index")]


def test_report_without_observer_is_a_no_op() -> None:
    """A missing hook never constructs or delivers an event."""
    report(None, "finalization")


def test_stage_must_be_nonempty() -> None:
    """Blank stage names are rejected."""
    with pytest.raises(ValueError, match="nonempty"):
        ProgressEvent(stage="   ")


@pytest.mark.parametrize(
    ("completed", "total"),
    [(1, None), (None, 1), (-1, 2), (2, -1), (3, 2)],
)
def test_counts_must_be_paired_and_possible(completed: int | None, total: int | None) -> None:
    """Partial, negative, or impossible counts are rejected."""
    with pytest.raises(ValueError, match="counts"):
        ProgressEvent(stage="judgments", completed=completed, total=total)


def test_counts_are_optional_together() -> None:
    """A stage without countable units carries no counts at all."""
    event = ProgressEvent(stage="fitting")
    assert event.completed is None
    assert event.total is None
