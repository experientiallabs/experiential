"""Tests for the engine-shared aware-clock helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from wmo.simulation.engines.clock import timestamp, utc_now


def test_timestamp_rejects_a_naive_clock() -> None:
    """Evidence timestamps must be timezone-aware or later ordering comparisons lie."""
    with pytest.raises(ValueError, match="timezone-aware"):
        timestamp(lambda: datetime(2026, 1, 1))  # noqa: DTZ001 - the naive value IS the case


def test_timestamp_never_precedes_the_prior_event() -> None:
    """A deterministic test clock that moves backwards must clamp to the prior event time."""
    earlier = datetime(2026, 1, 1, tzinfo=UTC)
    later = earlier + timedelta(seconds=5)

    assert timestamp(lambda: earlier, not_before=later) == later
    assert timestamp(lambda: later, not_before=earlier) == later


def test_utc_now_is_aware() -> None:
    """The default clock returns an aware UTC datetime usable in evidence ordering."""
    assert utc_now().utcoffset() is not None
