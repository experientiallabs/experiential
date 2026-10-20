"""Pure integer valuation of complete token schedules and observed token subsets."""

from __future__ import annotations

from exp.common.models.catalog_prices import GatewayLongContextTier, GatewayTokenPrices
from exp.common.models.model import Usage


def twelve_hour_write_rate(explicit_rate: int | None, input_rate: int | None) -> int | None:
    """Return the rate for OpenAI 12-hour cache writes on one schedule.

    OpenAI bills a 12-hour write at 2.5 times the uncached input rate, a total
    rate rather than a surcharge
    (https://developers.openai.com/api/docs/guides/prompt-caching). An authored
    rate wins; without one the published multiple of the schedule's own input
    rate applies, rounded up so a derived rate never undercharges. It is never
    the shorter-lifetime write rate.

    Args:
        explicit_rate: The card's authored 12-hour write rate, or None.
        input_rate: The same card's uncached input rate, or None when unknown.

    Returns:
        Nano-USD per million 12-hour write tokens, or None when neither is known.
    """
    if explicit_rate is not None:
        return explicit_rate
    if input_rate is None:
        return None
    return (input_rate * 5 + 1) // 2


def token_cost_nano_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int | None,
    cache_write_input_tokens: int | None,
    cache_write_1h_input_tokens: int | None,
    reasoning_tokens: int | None,
    input_rate: int | None,
    cached_input_rate: int | None,
    cache_creation_input_rate: int | None,
    cache_creation_1h_input_rate: int | None,
    output_rate: int | None,
    reasoning_rate: int | None,
    image_output_tokens: int | None = None,
    image_output_rate: int | None = None,
    cache_write_12h_input_tokens: int | None = None,
    cache_creation_12h_input_rate: int | None = None,
) -> int | None:
    """Price disjoint input/output subsets with half-up nano-USD rounding.

    Missing subset counts never invent a discount. A positive cache-write count
    without its one-hour split is priceable only when both write rates are known
    and equal. Missing rates for positive dimensions preserve unknown cost.
    Gateway-normalized counts are clamped to their containing totals.

    OpenAI's 12-hour writes are a third disjoint write subset, reported only by a
    provider that wrote at that lifetime; absent means none. They price at
    :func:`twelve_hour_write_rate`, so a card without an authored rate still
    charges 2.5 times its input rate for them.

    With an image-output rate, generated-image tokens are carved out of the
    output first and the reasoning subset is clamped to what remains (Gemini 3
    Pro Image counts its image inside its thoughts). Without one, image tokens
    stay in the output exactly as before the rate existed. A rate that differs
    from the output rate with no reported image count leaves cost unknown.
    """
    cached = min(cached_input_tokens or 0, input_tokens)
    written = min(cache_write_input_tokens or 0, input_tokens - cached)
    if (
        written
        and cache_write_1h_input_tokens is None
        and (
            cache_creation_input_rate is None
            or cache_creation_input_rate != cache_creation_1h_input_rate
        )
    ):
        return None
    hour = min(cache_write_1h_input_tokens or 0, written)
    half_day = min(cache_write_12h_input_tokens or 0, written - hour)
    if (
        image_output_tokens is None
        and output_tokens > 0
        and image_output_rate is not None
        and image_output_rate != output_rate
    ):
        # Every split of the output between text and image prices differently.
        return None
    image = min(image_output_tokens or 0, output_tokens) if image_output_rate is not None else 0
    reasoning = min(reasoning_tokens or 0, output_tokens - image)
    dimensions = (
        (input_tokens - cached - written, input_rate),
        (cached, cached_input_rate),
        (written - hour - half_day, cache_creation_input_rate),
        (hour, cache_creation_1h_input_rate),
        (half_day, twelve_hour_write_rate(cache_creation_12h_input_rate, input_rate)),
        (output_tokens - image - reasoning, output_rate),
        (reasoning, reasoning_rate),
        (image, image_output_rate),
    )
    if any(tokens > 0 and rate is None for tokens, rate in dimensions):
        return None
    numerator = sum(tokens * (rate or 0) for tokens, rate in dimensions)
    return (numerator + 500_000) // 1_000_000


def schedule_usage_cost_nano_usd(prices: GatewayTokenPrices, usage: Usage) -> int | None:
    """Apply the observed processing tier and whole-request context tier once.

    An omitted, default, or auto service tier uses the ordinary request path. A tier
    without an authored schedule is unknown, never a fallback to base prices.
    """
    if (
        usage.service_tier not in (None, "default", "auto")
        and prices.service_tier(usage.service_tier) is None
    ):
        return None
    selected = prices.for_service_tier(usage.service_tier)
    schedule = selected
    if (
        selected.long_context is not None
        and usage.input_tokens >= selected.long_context.input_threshold_tokens
    ):
        schedule = selected.long_context
    if (
        (usage.cached_input_tokens or 0) + (usage.cache_write_input_tokens or 0)
        > usage.input_tokens
        or (usage.cache_write_1h_input_tokens or 0) + (usage.cache_write_12h_input_tokens or 0)
        > (usage.cache_write_input_tokens or 0)
        or (usage.reasoning_tokens or 0) > usage.output_tokens
    ):
        return None
    # Omitted meters are unknown, not zero. They can be ignored only when
    # every possible allocation of the remaining tokens has the same price.
    if (
        usage.cached_input_tokens is None
        and usage.input_tokens > (usage.cache_write_input_tokens or 0)
        and schedule.cached_input_nano_usd_per_million_tokens
        != schedule.input_nano_usd_per_million_tokens
    ):
        return None
    if (
        usage.cache_write_input_tokens is None
        and usage.input_tokens > (usage.cached_input_tokens or 0)
        and (
            schedule.cache_creation_input_nano_usd_per_million_tokens
            != schedule.input_nano_usd_per_million_tokens
            or schedule.cache_creation_1h_input_nano_usd_per_million_tokens
            != schedule.input_nano_usd_per_million_tokens
        )
    ):
        return None
    if (
        usage.reasoning_tokens is None
        and usage.output_tokens > 0
        and schedule.reasoning_nano_usd_per_million_tokens
        != schedule.output_nano_usd_per_million_tokens
    ):
        return None
    # The model-client usage carries no image subset, so a card that prices
    # images apart from text cannot value any output.
    if (
        usage.output_tokens > 0
        and schedule.image_output_nano_usd_per_million_tokens is not None
        and schedule.image_output_nano_usd_per_million_tokens
        != schedule.output_nano_usd_per_million_tokens
    ):
        return None
    return token_cost_nano_usd(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_write_input_tokens=usage.cache_write_input_tokens,
        cache_write_1h_input_tokens=usage.cache_write_1h_input_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        input_rate=schedule.input_nano_usd_per_million_tokens,
        cached_input_rate=schedule.cached_input_nano_usd_per_million_tokens,
        cache_creation_input_rate=schedule.cache_creation_input_nano_usd_per_million_tokens,
        cache_creation_1h_input_rate=schedule.cache_creation_1h_input_nano_usd_per_million_tokens,
        output_rate=schedule.output_nano_usd_per_million_tokens,
        reasoning_rate=schedule.reasoning_nano_usd_per_million_tokens,
        cache_write_12h_input_tokens=usage.cache_write_12h_input_tokens,
        cache_creation_12h_input_rate=schedule.cache_creation_12h_input_nano_usd_per_million_tokens,
    )


def schedule_maximum_cost_nano_usd(
    prices: GatewayTokenPrices, *, input_tokens: int, output_tokens: int
) -> int:
    """Estimate the maximum known charge for an ordinary ModelRequest.

    ModelRequest cannot select flex or priority. Those authored cards remain
    available for actual returned-tier valuation, but are not reachable here.

    Each conditional schedule replaces the entire request. This prices the
    maximum over schedules, not an invented combination of their dimensions.
    Unpriced dimensions contribute no known amount, not a claim of free usage.
    Callers must separately prove ``schedule_prices_complete`` before treating
    this estimate as a strict bound. Actual unpriced usage remains unknown.
    """
    schedules: list[GatewayTokenPrices | GatewayLongContextTier] = [prices]
    if (
        prices.long_context is not None
        and input_tokens >= prices.long_context.input_threshold_tokens
    ):
        schedules.append(prices.long_context)
    candidates = []
    for schedule in schedules:
        input_rate = max(
            (
                rate
                for rate in (
                    schedule.input_nano_usd_per_million_tokens,
                    schedule.cached_input_nano_usd_per_million_tokens,
                    schedule.cache_creation_input_nano_usd_per_million_tokens,
                    schedule.cache_creation_1h_input_nano_usd_per_million_tokens,
                )
                if rate is not None
            ),
            default=0,
        )
        output_rate = max(
            (
                rate
                for rate in (
                    schedule.output_nano_usd_per_million_tokens,
                    schedule.reasoning_nano_usd_per_million_tokens,
                    schedule.image_output_nano_usd_per_million_tokens,
                )
                if rate is not None
            ),
            default=0,
        )
        candidates.append(
            (input_tokens * input_rate + output_tokens * output_rate + 999_999) // 1_000_000
        )
    return max(candidates)


def schedule_prices_complete(prices: GatewayTokenPrices, *, maximum_input_tokens: int) -> bool:
    """Prove ordinary-request subsets and reachable context tiers have complete prices.

    Unrequested flex and priority cards do not constrain ModelRequest admission.
    Actual responses still select their explicitly returned processing tier.
    """
    schedules: list[GatewayTokenPrices | GatewayLongContextTier] = [prices]
    if (
        prices.long_context is not None
        and maximum_input_tokens >= prices.long_context.input_threshold_tokens
    ):
        schedules.append(prices.long_context)
    for schedule in schedules:
        if any(
            rate is None
            for rate in (
                schedule.input_nano_usd_per_million_tokens,
                schedule.cached_input_nano_usd_per_million_tokens,
                schedule.cache_creation_input_nano_usd_per_million_tokens,
                schedule.cache_creation_1h_input_nano_usd_per_million_tokens,
                schedule.output_nano_usd_per_million_tokens,
                schedule.reasoning_nano_usd_per_million_tokens,
            )
        ):
            return False
    return True
