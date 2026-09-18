"""Tests for cache-write pricing across service-tier schedule selection."""

from exp.common.models.catalog import GatewayServiceTierPrices, GatewayTokenPrices


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
