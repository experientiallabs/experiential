# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Fast aliases and immutable tier receipt contracts."""

import pytest
from pydantic import ValidationError

from exp.common.models.catalog_prices import GatewayServiceTierPrices, GatewayTokenPrices
from exp.runtime.gateway.service_tiers import GatewayServiceTierAdmission, tier_admission
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses


@pytest.mark.parametrize("surface", ["chat", "responses"])
def test_fast_normalizes_before_replay_and_wire(surface: str) -> None:
    """Fast and priority are one canonical request, not alternate ledger spellings."""
    if surface == "chat":
        fast = decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "x"}],
                "service_tier": "fast",
            }
        )
        priority = decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "x"}],
                "service_tier": "priority",
            }
        )
    else:
        fast = decode_responses({"model": "coding", "input": "x", "service_tier": "fast"})
        priority = decode_responses({"model": "coding", "input": "x", "service_tier": "priority"})
    assert fast.request.service_tier == "priority"
    assert fast.request == priority.request


def test_receipt_can_select_only_admitted_cards() -> None:
    """Wrong served tiers, absence and conflicts hold rather than invent standard."""
    admission = GatewayServiceTierAdmission(
        requested="priority",
        standard_prices=GatewayTokenPrices(input_nano_usd_per_million_tokens=1),
        requested_prices=GatewayTokenPrices(input_nano_usd_per_million_tokens=2),
    )
    assert (
        admission.settlement(served="default", resolution="confirmed").prices
        == admission.standard_prices
    )
    assert (
        admission.settlement(served="priority", resolution="confirmed").prices
        == admission.requested_prices
    )
    assert admission.settlement(served="flex", resolution="confirmed").resolution == "unknown"
    for resolution in ("missing", "unknown", "conflicting"):
        receipt = admission.settlement(served=None, resolution=resolution)
        assert receipt.prices is None and receipt.served is None
    with pytest.raises(ValidationError):
        GatewayServiceTierAdmission(
            requested="priority",
            standard_prices=GatewayTokenPrices(priority=GatewayServiceTierPrices()),
            requested_prices=GatewayTokenPrices(),
        )


def test_tier_admission_does_not_change_byok_or_standard() -> None:
    """Only explicitly forwarded priced house requests need served-tier accounting."""
    prices = GatewayTokenPrices(
        priority=GatewayServiceTierPrices(input_nano_usd_per_million_tokens=2)
    )
    assert tier_admission(prices, "priority", forwards_tier=True, customer_managed=True) is None
    for tier in (None, "auto", "default", "scale"):
        assert tier_admission(prices, tier, forwards_tier=True, customer_managed=False) is None
    assert tier_admission(prices, "priority", forwards_tier=False, customer_managed=False) is None
