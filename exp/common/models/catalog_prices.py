"""Bounded integer nano-USD gateway pricing cards and schedule selection."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from exp.common.core.artifacts import ContractModel

MAXIMUM_RATE_NANO_USD_PER_MILLION_TOKENS = 1_000_000_000_000
"""Upper bound on any authored rate: $1,000 per million tokens in nano-USD.

Every published price today is far below it (the highest authored rate is
$600 per million, 6e11 nano-USD), and at this ceiling on every dimension a
1M-context request with the full output ceiling still sums to well under the
signed 64-bit ledger column before the per-million division, so no authored
catalog can produce an attempt cost the ledger cannot hold.
"""

NanoUsdRatePerMillionTokens = Annotated[
    int | None, Field(ge=0, le=MAXIMUM_RATE_NANO_USD_PER_MILLION_TOKENS)
]
"""One optional integer nano-USD-per-million-tokens rate; ``None`` is unknown, never zero."""


class GatewayLongContextTier(ContractModel):
    """Premium rates a provider applies to whole long-context requests.

    When ``usage.input_tokens >= input_threshold_tokens``, tier rates replace
    base rates for every dimension of the whole request, not just excess tokens.
    This models Gemini and Anthropic's long-context premium schedules. A ``None``
    tier rate stays unknown; it never inherits the base rate.
    """

    input_threshold_tokens: int = Field(gt=0)
    input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cached_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cache_creation_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cache_creation_1h_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    output_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    reasoning_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None


class GatewayServiceTierPrices(ContractModel):
    """PASS-THROUGH rates for one provider processing tier (flex / priority).

    OpenAI's ``service_tier`` reprices the WHOLE request (``flex`` discounted,
    ``priority`` premium): these rates replace the base schedule for every
    dimension at cost, no markup. ``None`` on a dimension is unknown exactly as
    on the base schedule (never the base rate). v1 bills the REQUESTED tier.
    """

    input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cached_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cache_creation_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cache_creation_1h_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    output_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    reasoning_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None


class GatewayTokenPrices(ContractModel):
    """Integer gateway attribution rates for one provider deployment.

    Values are integer nano-USD per million provider-reported tokens (one nano-USD is a
    billionth of a dollar: $1.25 per million is ``1_250_000_000``), bounded above by
    ``MAXIMUM_RATE_NANO_USD_PER_MILLION_TOKENS``. ``None`` means the rate is unknown; it must
    never be interpreted as zero. Existing optimizer float pricing remains unchanged.
    Cache-creation rates price disjoint 5-minute and 1-hour write tokens. The
    unqualified rate prices the remainder after observed 1-hour writes; missing
    TTL evidence leaves write cost unknown, never inferred from requested TTL.
    """

    input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cached_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cache_creation_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cache_creation_1h_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    output_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    reasoning_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    long_context: GatewayLongContextTier | None = None
    """Whole-request premium schedule for long-context input, when one exists.

    Verified against the providers' published schedules (2026-08-30):
    Gemini prices ``prompts > 200k tokens`` at a higher whole-request rate
    for input, output, and cache reads; Anthropic's Claude 4.6+ models serve
    the full 1M window at standard pricing (no tier), so current Anthropic
    deployments leave this ``None``.
    """
    flex: GatewayServiceTierPrices | None = None
    """Pass-through rates when the caller requests ``service_tier='flex'``."""
    priority: GatewayServiceTierPrices | None = None
    """Pass-through rates when the caller requests ``service_tier='priority'``."""

    def service_tier(self, tier: str | None) -> GatewayServiceTierPrices | None:
        """Find the pass-through card for a requested processing tier.

        Args:
            tier: Requested tier; only flex and priority have separate cards.

        Returns:
            The configured card, or None for base pricing or an unconfigured tier.
        """
        if tier == "flex":
            return self.flex
        if tier == "priority":
            return self.priority
        return None

    def for_service_tier(self, tier: str | None) -> GatewayTokenPrices:
        """Select the complete effective schedule without inheriting missing tier prices.

        Args:
            tier: Requested provider processing tier.

        Returns:
            The tier card as a whole-request schedule without long-context pricing,
            or this schedule when no configured tier card applies.
        """
        card = self.service_tier(tier)
        if card is None:
            return self
        return GatewayTokenPrices(
            input_nano_usd_per_million_tokens=card.input_nano_usd_per_million_tokens,
            cached_input_nano_usd_per_million_tokens=card.cached_input_nano_usd_per_million_tokens,
            cache_creation_input_nano_usd_per_million_tokens=card.cache_creation_input_nano_usd_per_million_tokens,
            cache_creation_1h_input_nano_usd_per_million_tokens=card.cache_creation_1h_input_nano_usd_per_million_tokens,
            output_nano_usd_per_million_tokens=card.output_nano_usd_per_million_tokens,
            reasoning_nano_usd_per_million_tokens=card.reasoning_nano_usd_per_million_tokens,
            long_context=None,
        )
