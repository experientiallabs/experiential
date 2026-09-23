"""Pure cost attribution helpers for the content-free attempt ledger.

Money is integer nano-USD everywhere in the engine (one nano-USD is a
billionth of a dollar; rates are nano-USD per MILLION tokens). Every cost is
rounded half-up at one nano-USD, and every amount must fit the signed 64-bit
ledger column, which :class:`NanoUsdOverflowError` guards explicitly so an
unrepresentable amount is refused rather than wrapped or coerced.
"""

from __future__ import annotations

import sqlite3

from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayUsage,
)
from exp.runtime.gateway.ledger_errors import GatewayLedgerError

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


def terminal_values(
    terminal_event: GatewayEvent | None,
    failure: GatewayFailure | None,
) -> tuple[str, str | None, str | None, GatewayUsage | None]:
    """Normalize one finish call to state, failure class, message, and usage.

    The failure message is the provider's own sanitized explanation
    (``provider_detail``); it is present only for a client-error rejection and
    is the same bounded, credential-free sentence the caller already receives.
    """
    event_failure = None if terminal_event is None else terminal_event.failure
    normalized = failure or event_failure
    if terminal_event is None and normalized is None:
        raise GatewayLedgerError("attempt finish needs a terminal event or failure")
    if terminal_event is not None and terminal_event.kind not in {
        GatewayEventKind.COMPLETED,
        GatewayEventKind.INCOMPLETE,
        GatewayEventKind.FAILED,
    }:
        raise GatewayLedgerError("attempt finish event must be terminal")
    if normalized is not None:
        state = (
            "cancelled" if normalized.failure_class == GatewayFailureClass.CANCELLED else "failed"
        )
        return (
            state,
            normalized.failure_class.value,
            normalized.provider_detail,
            (None if terminal_event is None else terminal_event.usage),
        )
    assert terminal_event is not None
    return terminal_event.kind.value, None, None, terminal_event.usage


def observed_usage_cost(
    row: sqlite3.Row,
    usage: GatewayUsage | None,
    terminal_event: GatewayEvent | None,
) -> int | None:
    """Price provider or explicitly estimated usage; unestimated disconnects remain unknown."""
    threshold = optional_int(row["long_context_threshold_tokens"])
    long_context = (
        threshold is not None
        and usage is not None
        and usage.input_tokens is not None
        and usage.input_tokens >= threshold
    )
    prefix = "long_context_" if long_context else ""
    partial = (
        terminal_event is not None
        and terminal_event.usage_incomplete_due_to_disconnect
        and not terminal_event.usage_estimated
    )
    return None if partial else frozen_usage_cost(row, usage, prefix=prefix)


def budget_settlement_nano_usd(
    row: sqlite3.Row,
    cost: int | None,
    usage: GatewayUsage | None,
    terminal_event: GatewayEvent | None,
) -> int | None:
    """Project the conservative settlement while retaining unmetered decision liability."""
    budget_settlement = cost if cost is not None else optional_int(row["budget_reserved_nano_usd"])
    if row["api_surface"] == GatewayApiSurface.DECISIONS.value and cost is None:
        # An unmetered decision can still have executed upstream. Keep the
        # reservation held without inventing usage or a settled charge.
        # Only a witnessed HTTP rejection proves that this hold can release.
        rejected = (
            terminal_event is not None
            and terminal_event.kind is GatewayEventKind.FAILED
            and terminal_event.decision_provider_rejected
            and usage is None
        )
        budget_settlement = 0 if rejected else None
    if budget_settlement is not None and budget_settlement > MAXIMUM_NANO_USD:
        raise GatewayLedgerError("attempt cost exceeds SQLite integer capacity")
    return budget_settlement


def usage_source_label(usage: GatewayUsage | None, *, estimated: bool) -> str:
    """The ledger's usage provenance for one settlement.

    ``unknown`` without usage, ``estimated`` when the registry completed a
    disconnect's meter with the gateway tokenizer, ``observed`` otherwise.

    Args:
        usage: The usage the settlement carries, if any.
        estimated: Whether that usage is the gateway's disconnect estimate.

    Returns:
        One of the three ``usage_source`` labels every ledger schema accepts.
    """
    if usage is None:
        return "unknown"
    return "estimated" if estimated else "observed"


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
