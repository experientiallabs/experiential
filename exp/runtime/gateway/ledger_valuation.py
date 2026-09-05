"""Pure cost attribution helpers for the content-free attempt ledger."""

from __future__ import annotations

from exp.runtime.gateway.contracts import GatewayUsage


def estimated_cost_micro_usd(
    usage: GatewayUsage | None,
    *,
    input_rate: int | None,
    cached_input_rate: int | None,
    cache_creation_input_rate: int | None = None,
    output_rate: int | None,
    reasoning_rate: int | None,
) -> int | None:
    """Compute attributed integer micro-USD or preserve unknown pricing.

    Cached-input, cache-write, and reasoning counts are disjoint subsets of
    their total token counts. Price each differently priced subset at its
    configured rate and the fresh remainders at the base rates, clamping
    malformed detail counts so the subsets never exceed their totals. A missing
    rate for a reported subset preserves unknown pricing rather than silently
    falling back to the base rate. ``cache_creation_input_tokens`` is the
    Anthropic cache-write surcharge leg; when absent it prices at zero.
    """
    if usage is None or not usage.has_token_counts:
        return None
    assert usage.input_tokens is not None
    assert usage.output_tokens is not None
    cached_input_tokens = min(usage.cached_input_tokens or 0, usage.input_tokens)
    remaining_input = usage.input_tokens - cached_input_tokens
    cache_creation_input_tokens = min(usage.cache_creation_input_tokens or 0, remaining_input)
    reasoning_tokens = min(usage.reasoning_tokens or 0, usage.output_tokens)
    dimensions = (
        (
            usage.input_tokens - cached_input_tokens - cache_creation_input_tokens,
            input_rate,
        ),
        (cached_input_tokens, cached_input_rate),
        (cache_creation_input_tokens, cache_creation_input_rate),
        (usage.output_tokens - reasoning_tokens, output_rate),
        (reasoning_tokens, reasoning_rate),
    )
    if any(tokens > 0 and rate is None for tokens, rate in dimensions):
        return None
    numerator = sum(tokens * (rate or 0) for tokens, rate in dimensions)
    return (numerator + 500_000) // 1_000_000


def optional_int(value: int | None) -> int | None:
    """Convert one nullable SQLite integer value to its precise type."""
    return None if value is None else int(value)
