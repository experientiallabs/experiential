"""Pure cost attribution helpers for the content-free attempt ledger.

Money is integer nano-USD everywhere in the engine (one nano-USD is a
billionth of a dollar; rates are nano-USD per MILLION tokens). Every cost is
rounded half-up at one nano-USD, and every amount must fit the signed 64-bit
ledger column, which :class:`NanoUsdOverflowError` guards explicitly so an
unrepresentable amount is refused rather than wrapped or coerced.
"""

from __future__ import annotations

from exp.runtime.gateway.contracts import GatewayUsage

MAXIMUM_NANO_USD = 9_223_372_036_854_775_807
"""Largest nano-USD amount the signed 64-bit ledger columns (SQLite INTEGER,
Postgres int8) can hold. Every cost, ceiling, reservation, and settlement is
checked against it by :func:`require_representable_nano_usd`."""


class NanoUsdOverflowError(ValueError):
    """A nano-USD amount does not fit the signed 64-bit ledger column.

    Raised instead of returning a wrapped, coerced, or silently unpriced value:
    with rates bounded at ``MAXIMUM_RATE_NANO_USD_PER_MILLION_TOKENS`` this is
    unreachable for any real request, so hitting it means a corrupt rate or a
    corrupt token count, both of which must fail closed by name.
    """


def require_representable_nano_usd(amount: int, *, what: str) -> int:
    """Return ``amount`` unless it exceeds the int8 ledger column, then raise.

    Args:
        amount: Nonnegative integer nano-USD amount.
        what: Short noun for the error message (``"attempt cost"``).

    Raises:
        NanoUsdOverflowError: The amount does not fit a signed 64-bit integer.
    """
    if amount > MAXIMUM_NANO_USD:
        raise NanoUsdOverflowError(
            f"{what} of {amount} nano-USD exceeds the signed 64-bit ledger column"
        )
    return amount


def estimated_cost_nano_usd(
    usage: GatewayUsage | None,
    *,
    input_rate: int | None,
    cached_input_rate: int | None,
    output_rate: int | None,
    reasoning_rate: int | None,
) -> int | None:
    """Compute attributed integer nano-USD or preserve unknown pricing.

    Cached-input and reasoning counts are subsets of their total token counts. Price the
    differently priced subsets at their configured rates and the fresh remainders at the base
    rates, clamping malformed detail counts to the corresponding total. A missing rate for a
    reported subset preserves unknown pricing rather than silently falling back to the base rate.

    Rates are nano-USD per million tokens, so the sum of ``tokens * rate`` is divided by one
    million and rounded half-up at one nano-USD. This is the ONE rounding rule of the ledger:
    a figure that the former micro-USD ledger rounded to a whole micro-USD is now carried at
    three more digits, so the two differ by at most half a micro-USD (500 nano-USD) and agree
    exactly whenever the micro figure was exact.

    Raises:
        NanoUsdOverflowError: The cost does not fit the signed 64-bit ledger column.
    """
    if usage is None or not usage.has_token_counts:
        return None
    assert usage.input_tokens is not None
    assert usage.output_tokens is not None
    cached_input_tokens = min(usage.cached_input_tokens or 0, usage.input_tokens)
    reasoning_tokens = min(usage.reasoning_tokens or 0, usage.output_tokens)
    dimensions = (
        (usage.input_tokens - cached_input_tokens, input_rate),
        (cached_input_tokens, cached_input_rate),
        (usage.output_tokens - reasoning_tokens, output_rate),
        (reasoning_tokens, reasoning_rate),
    )
    if any(tokens > 0 and rate is None for tokens, rate in dimensions):
        return None
    numerator = sum(tokens * (rate or 0) for tokens, rate in dimensions)
    return require_representable_nano_usd((numerator + 500_000) // 1_000_000, what="attempt cost")


def optional_int(value: int | None) -> int | None:
    """Convert one nullable SQLite integer value to its precise type."""
    return None if value is None else int(value)
