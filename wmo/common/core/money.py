"""Exact fixed-point USD values shared by Project authorization boundaries."""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal, InvalidOperation
from math import inf, nextafter

type UsdAmount = Decimal

USD_QUANTUM = Decimal("0.000001")
USD_ZERO = Decimal("0.000000")
USD_MAXIMUM = Decimal("99999999999999.999999")


def exact_usd(value: object, *, allow_zero: bool = False) -> UsdAmount:
    """Parse one exact six-place USD value without binary-float persistence.

    Args:
        value: Decimal, canonical decimal text, integer, or boundary float input.
        allow_zero: Whether the exact zero value is accepted.

    Returns:
        Canonical six-place Decimal suitable for persisted contracts.

    Raises:
        ValueError: The value is non-finite, negative, over-wide, or over-precise.
    """
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
        canonical = parsed.quantize(USD_QUANTUM)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("USD value must be an exact decimal with at most six places") from exc
    if not parsed.is_finite() or canonical != parsed:
        raise ValueError("USD value must be finite with at most six decimal places")
    if canonical < USD_ZERO or (canonical == USD_ZERO and not allow_zero):
        raise ValueError("USD value must be positive")
    if canonical > USD_MAXIMUM:
        raise ValueError("USD value exceeds numeric(20,6)")
    return canonical


def reserve_usd(value: object) -> UsdAmount:
    """Round a nonnegative estimated provider cost up to the fixed-point boundary.

    Args:
        value: Estimated provider cost from an existing numeric calculation.

    Returns:
        Conservative six-place Decimal reservation.

    Raises:
        ValueError: The estimate is negative, non-finite, or over-wide.
    """
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
        reserved = parsed.quantize(USD_QUANTUM, rounding=ROUND_CEILING)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("provider reservation must be a finite decimal value") from exc
    if not parsed.is_finite() or parsed < USD_ZERO:
        raise ValueError("provider reservation must be finite and nonnegative")
    if reserved > USD_MAXIMUM:
        raise ValueError("provider reservation exceeds numeric(20,6)")
    return reserved


def nonincreasing_float_usd(value: object) -> float:
    """Convert exact USD to a positive float without increasing authorization.

    Args:
        value: Exact positive numeric(20,6) value at a legacy float boundary.

    Returns:
        The nearest representable positive float no greater than the exact value.

    Raises:
        ValueError: The exact value is invalid or cannot remain positive as a float.
    """
    exact = exact_usd(value)
    candidate = float(exact)
    while Decimal.from_float(candidate) > exact:
        candidate = nextafter(candidate, -inf)
    if candidate <= 0:
        raise ValueError("USD value is too small for a positive float boundary")
    return candidate
