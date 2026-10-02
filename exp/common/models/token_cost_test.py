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
