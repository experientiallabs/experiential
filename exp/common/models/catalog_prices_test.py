"""Tests for cache-write pricing across service-tier schedule selection."""

from exp.common.models.catalog import (
    GatewayLongContextTier,
    GatewayServiceTierPrices,
    GatewayTokenPrices,
)


def test_requested_service_tier_keeps_its_cache_write_schedule() -> None:
    """Tier selection uses the requested rates without falling back to base writes."""
    prices = GatewayTokenPrices(
        cache_creation_input_nano_usd_per_million_tokens=99,
        priority=GatewayServiceTierPrices(
            cache_creation_input_nano_usd_per_million_tokens=10,
            cache_creation_1h_input_nano_usd_per_million_tokens=20,
        ),
    )
    selected = prices.for_service_tier("priority")
    assert selected.cache_creation_input_nano_usd_per_million_tokens == 10
    assert selected.cache_creation_1h_input_nano_usd_per_million_tokens == 20
    assert selected.long_context is None


def test_service_tier_keeps_its_own_complete_long_context_schedule() -> None:
    """Tier selection never erases or inherits a different processing schedule."""
    long = GatewayLongContextTier(
        input_threshold_tokens=272_001,
        input_nano_usd_per_million_tokens=8_000_000_000,
        cache_creation_input_nano_usd_per_million_tokens=10_000_000_000,
        cache_creation_1h_input_nano_usd_per_million_tokens=16_000_000_000,
    )
    prices = GatewayTokenPrices(
        long_context=long.model_copy(update={"input_threshold_tokens": 100}),
        priority=GatewayServiceTierPrices(long_context=long),
    )
    assert prices.for_service_tier("priority").long_context == long
    assert prices.for_service_tier("default") is prices


def test_unset_long_context_does_not_change_snapshot_identity_projection() -> None:
    """Schema-five snapshots exclude declared defaults recursively, including this new None."""
    card = GatewayServiceTierPrices(input_nano_usd_per_million_tokens=7)
    assert card.model_dump(mode="json", exclude_defaults=True) == {
        "input_nano_usd_per_million_tokens": 7
    }
    prices = GatewayTokenPrices(priority=card)
    assert prices.model_dump(mode="json", exclude_defaults=True) == {
        "priority": {"input_nano_usd_per_million_tokens": 7}
    }
