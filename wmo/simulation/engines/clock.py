"""Aware-clock helpers shared by every simulation engine's evidence path."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime


def timestamp(clock: Callable[[], datetime], *, not_before: datetime | None = None) -> datetime:
    """Return an aware timestamp and prevent a deterministic test clock from moving backwards.

    Args:
        clock: Time source expected to return an aware UTC-comparable datetime.
        not_before: Optional prior event time this timestamp may not precede.

    Returns:
        The current clock value, or ``not_before`` when the clock moved backwards.

    Raises:
        ValueError: The clock returns a naive datetime.
    """
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("simulation clock must return timezone-aware datetimes")
    if not_before is not None and value < not_before:
        return not_before
    return value


def utc_now() -> datetime:
    """Return a timezone-aware default timestamp without importing provider state."""
    return datetime.now(UTC)
