"""Tests for cache-write pricing, service-tier selection, and per-unit media cards."""

import pytest
from pydantic import ValidationError

from exp.common.models.catalog import (
    GatewayLongContextTier,
    GatewayServiceTierPrices,
    GatewayTokenPrices,
)
from exp.common.models.catalog_prices import (
    MAXIMUM_RATE_NANO_USD_PER_UNIT,
    MAXIMUM_UNIT_VARIANT_CHARACTERS,
    BilledUnitKind,
    GatewayUnitPrices,
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


def test_unit_card_prices_a_variant_or_falls_back_to_the_flat_rate() -> None:
    """A named variant prices itself; an unauthored one uses the flat rate, else nothing."""
    card = GatewayUnitPrices(
        kind=BilledUnitKind.VIDEO_SECOND, rates={"": 80_000_000, "1080p": 120_000_000}
    )
    assert card.rate_for("1080p") == 120_000_000
    assert card.rate_for("4k") == 80_000_000
    assert card.rate_for("") == 80_000_000
    flatless = GatewayUnitPrices(kind=BilledUnitKind.VIDEO_SECOND, rates={"720p": 1})
    assert flatless.rate_for("1080p") is None


@pytest.mark.parametrize(
    "rates",
    [
        {},
        {"": -1},
        {"": MAXIMUM_RATE_NANO_USD_PER_UNIT + 1},
        {"v" * (MAXIMUM_UNIT_VARIANT_CHARACTERS + 1): 1},
    ],
)
def test_unit_card_rejects_empty_negative_oversized_or_unbounded_rates(
    rates: dict[str, int],
) -> None:
    """A unit card fails closed at authoring instead of pricing at a corrupt rate."""
    with pytest.raises(ValidationError):
        GatewayUnitPrices(kind=BilledUnitKind.CHARACTER, rates=rates)


def test_unit_card_is_tier_independent_and_absent_by_default() -> None:
    """Processing tiers reprice tokens only, and a token card serializes unchanged."""
    units = GatewayUnitPrices(kind=BilledUnitKind.CHARACTER, rates={"": 15_000})
    card = GatewayTokenPrices(
        flex=GatewayServiceTierPrices(input_nano_usd_per_million_tokens=1), units=units
    )
    assert card.for_service_tier("flex").units == units
    assert "units" not in GatewayTokenPrices().model_dump(exclude_defaults=True)
