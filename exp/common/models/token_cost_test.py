"""Complete schedules retain whole-request tiers and unknown token subsets."""

import pytest

from exp.common.models.catalog_prices import (
    GatewayLongContextTier,
    GatewayServiceTierPrices,
    GatewayTokenPrices,
)
from exp.common.models.model import Usage
from exp.common.models.token_cost import (
    schedule_maximum_cost_nano_usd,
    schedule_usage_cost_nano_usd,
    token_cost_nano_usd,
    twelve_hour_write_rate,
)


def prices() -> GatewayTokenPrices:
    """Return deliberately different rates so every billed subset is observable."""
    return GatewayTokenPrices(
        input_nano_usd_per_million_tokens=1_000_000_000,
        cached_input_nano_usd_per_million_tokens=100_000_000,
        cache_creation_input_nano_usd_per_million_tokens=2_000_000_000,
        cache_creation_1h_input_nano_usd_per_million_tokens=3_000_000_000,
        output_nano_usd_per_million_tokens=4_000_000_000,
        reasoning_nano_usd_per_million_tokens=5_000_000_000,
    )


def test_disjoint_subsets_and_unknown_ttl() -> None:
    """One-hour writes and reasoning are subsets, and an omitted split is not zero."""
    usage = Usage(
        input_tokens=100,
        output_tokens=20,
        cached_input_tokens=10,
        cache_write_input_tokens=30,
        cache_write_1h_input_tokens=5,
        reasoning_tokens=7,
    )
    assert schedule_usage_cost_nano_usd(prices(), usage) == 213_000
    assert (
        schedule_usage_cost_nano_usd(
            prices(), usage.model_copy(update={"cache_write_1h_input_tokens": None})
        )
        is None
    )
    assert (
        schedule_usage_cost_nano_usd(
            prices(), usage.model_copy(update={"cache_write_1h_input_tokens": 0})
        )
        == 208_000
    )
    assert (
        schedule_usage_cost_nano_usd(
            prices().model_copy(update={"reasoning_nano_usd_per_million_tokens": None}), usage
        )
        is None
    )


@pytest.mark.parametrize("base_tier", [None, "default", "auto"])
def test_whole_request_threshold_and_selected_service_tier(base_tier: str | None) -> None:
    """Threshold applies to each whole request; only an observed flex tier uses flex."""
    card = prices().model_copy(
        update={
            "long_context": GatewayLongContextTier(
                input_threshold_tokens=100,
                input_nano_usd_per_million_tokens=2_000_000_000,
                output_nano_usd_per_million_tokens=8_000_000_000,
            ),
            "flex": GatewayServiceTierPrices(
                input_nano_usd_per_million_tokens=500_000_000,
                output_nano_usd_per_million_tokens=2_000_000_000,
            ),
        }
    )
    ordinary = Usage(
        input_tokens=99,
        output_tokens=10,
        service_tier=base_tier,
        cached_input_tokens=0,
        cache_write_input_tokens=0,
        reasoning_tokens=0,
    )
    assert schedule_usage_cost_nano_usd(card, ordinary) == 139_000
    longer = ordinary.model_copy(update={"input_tokens": 100})
    assert schedule_usage_cost_nano_usd(card, longer) == 280_000
    assert (
        schedule_usage_cost_nano_usd(card, longer.model_copy(update={"service_tier": "flex"}))
        == 70_000
    )
    assert (
        schedule_usage_cost_nano_usd(card, longer.model_copy(update={"service_tier": "priority"}))
        is None
    )
    assert schedule_maximum_cost_nano_usd(card, input_tokens=100, output_tokens=10) >= 280_000


@pytest.mark.parametrize(
    "missing", ["cached_input_tokens", "cache_write_input_tokens", "reasoning_tokens"]
)
def test_absent_subset_is_not_a_known_zero_with_different_tariffs(missing: str) -> None:
    """Complete prices cannot replace the evidence needed to pick among unequal rates."""
    known = Usage(
        input_tokens=100,
        output_tokens=20,
        cached_input_tokens=0,
        cache_write_input_tokens=0,
        reasoning_tokens=0,
    )
    assert schedule_usage_cost_nano_usd(prices(), known) == 180_000
    assert schedule_usage_cost_nano_usd(prices(), known.model_copy(update={missing: None})) is None
    uniform = prices().model_copy(
        update={
            "cached_input_nano_usd_per_million_tokens": 1_000_000_000,
            "cache_creation_input_nano_usd_per_million_tokens": 1_000_000_000,
            "cache_creation_1h_input_nano_usd_per_million_tokens": 1_000_000_000,
            "reasoning_nano_usd_per_million_tokens": 4_000_000_000,
        }
    )
    assert (
        schedule_usage_cost_nano_usd(uniform, Usage(input_tokens=100, output_tokens=20)) == 180_000
    )


@pytest.mark.parametrize(
    "field",
    [
        "cached_input_tokens",
        "cache_write_input_tokens",
        "cache_write_1h_input_tokens",
        "reasoning_tokens",
    ],
)
def test_invalid_subsets_are_not_silently_underpriced(field: str) -> None:
    """Report valuation refuses impossible usage instead of clamping away paid evidence."""
    usage = Usage(input_tokens=10, output_tokens=10).model_copy(update={field: 11})
    assert schedule_usage_cost_nano_usd(prices(), usage) is None


def _image_cost(
    image_output_tokens: int | None, image_output_rate: int | None, reasoning_tokens: int = 0
) -> int | None:
    """Price 1,334 output tokens (Gemini 3 Pro Image's shape) under one image rule."""
    return token_cost_nano_usd(
        input_tokens=0,
        output_tokens=1_334,
        cached_input_tokens=0,
        cache_write_input_tokens=0,
        cache_write_1h_input_tokens=0,
        reasoning_tokens=reasoning_tokens,
        input_rate=0,
        cached_input_rate=0,
        cache_creation_input_rate=0,
        cache_creation_1h_input_rate=0,
        output_rate=12_000_000_000,
        reasoning_rate=12_000_000_000,
        image_output_tokens=image_output_tokens,
        image_output_rate=image_output_rate,
    )


def test_image_output_tokens_price_at_their_own_rate_carved_before_reasoning() -> None:
    """Image tokens leave the output first; reasoning is clamped to what remains."""
    # 1,120 image tokens at $120/M plus 214 text/thinking tokens at $12/M.
    expected = (1_120 * 120_000_000_000 + 214 * 12_000_000_000 + 500_000) // 1_000_000
    assert _image_cost(1_120, 120_000_000_000, reasoning_tokens=1_263) == expected
    assert _image_cost(1_120, 120_000_000_000) == expected


def test_an_unset_image_rate_keeps_todays_output_pricing() -> None:
    """No image rate: image tokens stay ordinary output, counted or not."""
    plain = (1_334 * 12_000_000_000 + 500_000) // 1_000_000
    assert _image_cost(1_120, None) == plain
    assert _image_cost(None, None) == plain


def test_a_distinct_image_rate_without_an_image_count_is_unknown() -> None:
    """Every text/image split prices differently, so the cost stays unknown."""
    assert _image_cost(None, 120_000_000_000) is None
    # A rate equal to the output rate prices every split the same.
    assert _image_cost(None, 12_000_000_000) == (1_334 * 12_000_000_000 + 500_000) // 1_000_000


def test_a_model_client_usage_cannot_value_a_card_with_a_distinct_image_rate() -> None:
    """Model-client usage has no image subset, so such a card's output is unknown."""
    prices = GatewayTokenPrices(
        input_nano_usd_per_million_tokens=1,
        cached_input_nano_usd_per_million_tokens=1,
        cache_creation_input_nano_usd_per_million_tokens=1,
        cache_creation_1h_input_nano_usd_per_million_tokens=1,
        output_nano_usd_per_million_tokens=12,
        reasoning_nano_usd_per_million_tokens=12,
        image_output_nano_usd_per_million_tokens=120,
    )
    usage = Usage(input_tokens=1, output_tokens=1, cached_input_tokens=0, reasoning_tokens=0)
    assert schedule_usage_cost_nano_usd(prices, usage) is None
    assert schedule_maximum_cost_nano_usd(prices, input_tokens=0, output_tokens=1_000_000) == 120


def test_an_unset_image_rate_leaves_the_card_identity_unchanged() -> None:
    """The new rate is omitted while unset, so existing catalog digests hold."""
    card = GatewayTokenPrices(output_nano_usd_per_million_tokens=1)
    assert "image_output_nano_usd_per_million_tokens" not in card.model_dump(exclude_defaults=True)
    flex = GatewayTokenPrices(
        flex=GatewayServiceTierPrices(image_output_nano_usd_per_million_tokens=7)
    ).for_service_tier("flex")
    assert flex.image_output_nano_usd_per_million_tokens == 7


def test_a_reported_zero_image_count_prices_all_output_at_the_output_rate() -> None:
    """A text answer on an image lane is billed as text, never at the image rate."""
    assert _image_cost(0, 120_000_000_000) == (1_334 * 12_000_000_000 + 500_000) // 1_000_000


def openai_twelve_hour_card(twelve_hour: int | None = None) -> GatewayTokenPrices:
    """Return a GPT-5.6 Luna shaped card: $2 input, 0.1x reads, 1.25x 30-minute writes."""
    return GatewayTokenPrices(
        input_nano_usd_per_million_tokens=2_000_000_000,
        cached_input_nano_usd_per_million_tokens=200_000_000,
        cache_creation_input_nano_usd_per_million_tokens=2_500_000_000,
        cache_creation_1h_input_nano_usd_per_million_tokens=2_500_000_000,
        cache_creation_12h_input_nano_usd_per_million_tokens=twelve_hour,
        output_nano_usd_per_million_tokens=12_000_000_000,
        reasoning_nano_usd_per_million_tokens=12_000_000_000,
    )


def test_twelve_hour_writes_bill_at_two_and_a_half_times_input_without_an_authored_rate() -> None:
    """OpenAI: a "12h" write is "2.5x the uncached input-token rate", a total rate.

    The live gpt-5.6-luna usage (2026-10-20): 3,250 input, 3,247 written at 12h.
    """
    usage = Usage(
        input_tokens=3_250,
        output_tokens=5,
        cached_input_tokens=0,
        cache_write_input_tokens=3_247,
        cache_write_12h_input_tokens=3_247,
        reasoning_tokens=0,
    )
    # 3 fresh x $2 + 3,247 x $5 (2.5 x $2) + 5 x $12, per million tokens.
    expected = (3 * 2_000_000_000 + 3_247 * 5_000_000_000 + 5 * 12_000_000_000 + 500_000) // (
        1_000_000
    )
    assert schedule_usage_cost_nano_usd(openai_twelve_hour_card(), usage) == expected
    # Never the 30-minute rate: the same writes at 30 minutes cost less.
    thirty = usage.model_copy(update={"cache_write_12h_input_tokens": 0})
    thirty_cost = schedule_usage_cost_nano_usd(openai_twelve_hour_card(), thirty)
    assert thirty_cost is not None
    assert thirty_cost < expected


def test_an_authored_twelve_hour_rate_wins_and_mixed_lifetimes_split() -> None:
    """An authored rate prices the 12-hour subset; the rest stay at the 30-minute rate."""
    usage = Usage(
        input_tokens=5_000,
        output_tokens=0,
        cached_input_tokens=0,
        cache_write_input_tokens=5_000,
        cache_write_12h_input_tokens=2_000,
        reasoning_tokens=0,
    )
    card = openai_twelve_hour_card(twelve_hour=6_000_000_000)
    assert (
        schedule_usage_cost_nano_usd(card, usage)
        == (3_000 * 2_500_000_000 + 2_000 * 6_000_000_000) // 1_000_000
    )


def test_a_derived_twelve_hour_rate_rounds_up_and_needs_the_input_rate() -> None:
    """An odd input rate derives a rate that never undercharges; no input rate is unknown."""
    assert twelve_hour_write_rate(None, 3) == 8
    assert twelve_hour_write_rate(None, 2_000_000_000) == 5_000_000_000
    assert twelve_hour_write_rate(7, 2_000_000_000) == 7
    assert twelve_hour_write_rate(None, None) is None
    assert (
        token_cost_nano_usd(
            input_tokens=10,
            output_tokens=0,
            cached_input_tokens=0,
            cache_write_input_tokens=10,
            cache_write_1h_input_tokens=0,
            cache_write_12h_input_tokens=10,
            reasoning_tokens=None,
            input_rate=None,
            cached_input_rate=1,
            cache_creation_input_rate=1,
            cache_creation_1h_input_rate=1,
            output_rate=1,
            reasoning_rate=1,
        )
        is None
    )


def test_timed_writes_beyond_total_writes_are_unknown_not_underpriced() -> None:
    """A one-hour plus 12-hour split larger than the writes contradicts the report."""
    usage = Usage(
        input_tokens=100,
        output_tokens=0,
        cached_input_tokens=0,
        cache_write_input_tokens=50,
        cache_write_1h_input_tokens=30,
        cache_write_12h_input_tokens=30,
    )
    assert schedule_usage_cost_nano_usd(openai_twelve_hour_card(), usage) is None


def test_an_unset_twelve_hour_rate_leaves_the_card_identity_unchanged() -> None:
    """The 12-hour rate is omitted while unset and rides every tier card when set."""
    key = "cache_creation_12h_input_nano_usd_per_million_tokens"
    assert key not in prices().model_dump(exclude_defaults=True)
    priority = GatewayTokenPrices(
        priority=GatewayServiceTierPrices(cache_creation_12h_input_nano_usd_per_million_tokens=9)
    ).for_service_tier("priority")
    assert priority.cache_creation_12h_input_nano_usd_per_million_tokens == 9
