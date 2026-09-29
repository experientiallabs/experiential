# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Conservative ceilings price both immutable schedules, not only requested flex."""

from exp.common.models.catalog import GatewayLongContextTier, GatewayTokenPrices
from exp.runtime.gateway.ledger_test import _deployment, _request
from exp.runtime.gateway.native_service_tiers import settlement_kwarg, tier_ceiling
from exp.runtime.gateway.service_tiers import GatewayServiceTierAdmission


def test_flex_reserves_standard_fallback_and_reachable_long_context() -> None:
    """A cheaper requested tier cannot under-reserve a fallback standard completion."""
    standard = GatewayTokenPrices(
        input_nano_usd_per_million_tokens=2_000_000,
        output_nano_usd_per_million_tokens=4_000_000,
        long_context=GatewayLongContextTier(
            input_threshold_tokens=100,
            input_nano_usd_per_million_tokens=4_000_000,
            output_nano_usd_per_million_tokens=8_000_000,
        ),
    )
    requested = GatewayTokenPrices(
        input_nano_usd_per_million_tokens=1_000_000, output_nano_usd_per_million_tokens=2_000_000
    )
    admission = GatewayServiceTierAdmission(
        requested="flex", standard_prices=standard, requested_prices=requested
    )
    deployment = _deployment()
    deployment = deployment.model_copy(
        update={"gateway": deployment.gateway.model_copy(update={"prices": requested})}
    )
    request = _request("tokens").model_copy(update={"maximum_output_tokens": 10})
    assert tier_ceiling(request, deployment, admission, input_tokens=79) == 198
    assert tier_ceiling(request, deployment, admission, input_tokens=80) == 400
    assert settlement_kwarg(admission, {})["service_tier"].resolution == "missing"
    assert (
        settlement_kwarg(
            admission, {"service_tier": {"resolution": "confirmed", "served": "default"}}
        )["service_tier"].prices
        == standard
    )
    assert settlement_kwarg(None, {}) == {}
