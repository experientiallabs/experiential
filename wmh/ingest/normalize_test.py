"""Tests for the shared normalizer helpers (the pieces every adapter relies on)."""

from __future__ import annotations

from wmh.ingest.normalize import iso_to_ordinal


def test_iso_to_ordinal_naive_timestamp_is_utc_not_local() -> None:
    """A naive timestamp (no tz) must be read as UTC, so ordinals are machine-independent.

    Regression: `datetime.fromisoformat("...").timestamp()` on a naive value uses the machine's
    local timezone, which would reorder spans differently across machines. `iso_to_ordinal` pins
    naive timestamps to UTC, so a naive value and its explicit-UTC form map to the same ordinal.
    """
    naive = iso_to_ordinal("2026-01-01T00:00:00", fallback=-1)
    explicit_utc = iso_to_ordinal("2026-01-01T00:00:00+00:00", fallback=-1)
    zulu = iso_to_ordinal("2026-01-01T00:00:00Z", fallback=-1)
    assert naive == explicit_utc == zulu
    assert naive > 0


def test_iso_to_ordinal_orders_and_falls_back() -> None:
    earlier = iso_to_ordinal("2026-01-01T00:00:00Z", fallback=0)
    later = iso_to_ordinal("2026-01-01T00:00:01Z", fallback=0)
    assert earlier < later
    # Absent/unparseable/non-string -> the caller's fallback (a list index), not a crash.
    assert iso_to_ordinal(None, fallback=7) == 7
    assert iso_to_ordinal("not-a-date", fallback=9) == 9
    assert iso_to_ordinal("", fallback=3) == 3
