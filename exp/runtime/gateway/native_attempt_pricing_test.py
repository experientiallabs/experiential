"""Tests for the one card a physical attempt reserves, freezes, and settles against."""

from __future__ import annotations

from datetime import UTC, datetime

from exp.common.models.catalog import (
    GatewayDeploymentMetadata,
    GatewayServiceTierPrices,
    GatewayTokenPrices,
)
from exp.common.models.catalog_prices import GatewayOffPeakPrices, GatewayPriceWindow
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.common.models.model import BillingSource, ModelCapabilities
from exp.runtime.gateway.ledger_test import (
    _deployment,
)
from exp.runtime.gateway.native_attempt_pricing import deployment_priced_at, priced_attempt

_PEAK = GatewayTokenPrices(
    input_nano_usd_per_million_tokens=300_000_000,
    cached_input_nano_usd_per_million_tokens=6_000_000,
    output_nano_usd_per_million_tokens=1_200_000_000,
)
_OFF_PEAK = GatewayServiceTierPrices(
    input_nano_usd_per_million_tokens=150_000_000,
    cached_input_nano_usd_per_million_tokens=3_000_000,
    output_nano_usd_per_million_tokens=600_000_000,
)


def _deepseek() -> ExactModelDeployment:
    """A host-funded DeepSeek rung on its published peak/off-peak schedule."""
    prices = GatewayTokenPrices(
        **_PEAK.model_dump(exclude_defaults=True),
        off_peak=GatewayOffPeakPrices(
            prices=_OFF_PEAK,
            peak_windows=(
                GatewayPriceWindow(days=(1, 2, 3, 4, 5), start_minute=60, end_minute=240),
                GatewayPriceWindow(days=(1, 2, 3, 4, 5), start_minute=360, end_minute=600),
            ),
        ),
    )
    base = _deployment(billing_source=BillingSource.HOST_MANAGED)
    return base.model_copy(
        update={
            "gateway": GatewayDeploymentMetadata(prices=prices),
            "capabilities": ModelCapabilities(maximum_output_tokens=8_192),
        }
    )


def test_deployment_priced_at_copies_only_a_scheduled_rung() -> None:
    """Schedule-free and absent rungs come back as the same object; scheduled ones lose it."""
    deployment = _deepseek()
    peak = datetime(2026, 10, 19, 7, tzinfo=UTC)
    off_peak = datetime(2026, 10, 19, 12, tzinfo=UTC)
    assert deployment_priced_at(deployment, peak).gateway.prices == _PEAK
    assert deployment_priced_at(None, off_peak) is None
    plain = _deployment()
    assert deployment_priced_at(plain, off_peak) is plain
    repriced = deployment_priced_at(deployment, off_peak)
    assert repriced.gateway.prices == GatewayTokenPrices(**_OFF_PEAK.model_dump())
    assert repriced.deployment_id == deployment.deployment_id
    assert deployment.gateway.prices.off_peak is not None  # the catalog object is untouched


def test_priced_attempt_admits_no_tier_on_a_scheduled_rung() -> None:
    """A scheduled card carries no tier cards, so a requested tier prices at the window card."""
    deployment, tier = priced_attempt(
        _deepseek(),
        "priority",
        forwards_tier=True,
        instant=datetime(2026, 10, 24, 12, tzinfo=UTC),
    )
    assert tier is None
    assert deployment.gateway.prices.input_nano_usd_per_million_tokens == 150_000_000
