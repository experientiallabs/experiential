"""Pure cost attribution helpers for the content-free attempt ledger.

Money is integer nano-USD everywhere in the engine (one nano-USD is a
billionth of a dollar; rates are nano-USD per MILLION tokens). Every cost is
rounded half-up at one nano-USD, and every amount must fit the signed 64-bit
ledger column, which :class:`NanoUsdOverflowError` guards explicitly so an
unrepresentable amount is refused rather than wrapped or coerced.
"""

from __future__ import annotations

import sqlite3

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
    cache_creation_input_rate: int | None = None,
    cache_creation_1h_input_rate: int | None = None,
    output_rate: int | None,
    reasoning_rate: int | None,
) -> int | None:
    """Compute attributed integer nano-USD or preserve unknown pricing.

    Cache reads and writes are disjoint input subsets; one-hour writes are a
    subset of all writes. Clamp reads, then writes, to remaining input. A missing
    rate or TTL breakdown for observed writes preserves unknown cost. Reasoning
    is an output subset. Price each remainder at its base rate exactly once.

    Rates are nano-USD per million tokens, so the sum of ``tokens * rate`` is divided by one
    million and rounded half-up at one nano-USD. This is the ONE rounding rule of the ledger:
    a figure that the former micro-USD ledger rounded to a whole micro-USD is now carried at
    three more digits, so the two differ by at most half a micro-USD (500 nano-USD) and agree
    exactly whenever the micro figure was exact.

    Args:
        usage: Provider-observed totals and subsets, or None.
        input_rate: Nano-USD per million fresh input tokens.
        cached_input_rate: Nano-USD per million cache-read tokens.
        cache_creation_input_rate: Nano-USD per million observed five-minute writes.
        cache_creation_1h_input_rate: Nano-USD per million observed one-hour writes.
        output_rate: Nano-USD per million non-reasoning output tokens.
        reasoning_rate: Nano-USD per million reasoning tokens.

    Returns:
        Rounded nano-USD, or None when usage, TTL evidence, or a required rate is missing.

    Raises:
        NanoUsdOverflowError: The cost does not fit the signed 64-bit ledger column.
    """
    if usage is None or not usage.has_token_counts:
        return None
    assert usage.input_tokens is not None
    assert usage.output_tokens is not None
    cached_input_tokens = min(usage.cached_input_tokens or 0, usage.input_tokens)
    cache_creation = min(
        usage.cache_creation_input_tokens or 0, usage.input_tokens - cached_input_tokens
    )
    if cache_creation and usage.cache_creation_1h_input_tokens is None:
        return None
    hour_creation = min(usage.cache_creation_1h_input_tokens or 0, cache_creation)
    reasoning_tokens = min(usage.reasoning_tokens or 0, usage.output_tokens)
    dimensions = (
        (usage.input_tokens - cached_input_tokens - cache_creation, input_rate),
        (cache_creation - hour_creation, cache_creation_input_rate),
        (hour_creation, cache_creation_1h_input_rate),
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


def frozen_usage_cost(
    row: sqlite3.Row, usage: GatewayUsage | None, *, prefix: str = ""
) -> int | None:
    """Price usage with a frozen base, long-context, or preferred schedule.

    Args:
        row: Attempt row containing every rate of the selected schedule.
        usage: Provider-observed usage, or None when not reported.
        prefix: Column namespace of the frozen schedule.

    Returns:
        Attributed nano-USD, or None when evidence or a required rate is missing.
    """
    return estimated_cost_nano_usd(
        usage,
        input_rate=optional_int(row[f"{prefix}input_rate"]),
        cached_input_rate=optional_int(row[f"{prefix}cached_input_rate"]),
        cache_creation_input_rate=optional_int(row[f"{prefix}cache_creation_input_rate"]),
        cache_creation_1h_input_rate=optional_int(row[f"{prefix}cache_creation_1h_input_rate"]),
        output_rate=optional_int(row[f"{prefix}output_rate"]),
        reasoning_rate=optional_int(row[f"{prefix}reasoning_rate"]),
    )
