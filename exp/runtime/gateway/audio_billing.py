"""How an audio rung bills: the one meter its price card names.

Shared by admission (which excludes a rung with no valid meter), the startup
servability check (which excludes an alias whose audio claims cannot bill), and
the reservation ceilings (which price a rung's flat unit rate).
"""

from __future__ import annotations

from typing import Literal

from exp.common.models.catalog_prices import BilledUnitKind, GatewayTokenPrices
from exp.common.models.gateway_catalog import ExactModelDeployment

AudioSurface = Literal["speech", "transcription"]

BillingMode = Literal["units", "tokens"]
"""How the data plane bills one rung: ``units`` from the lane's unit card
(characters for speech, metered seconds for transcription), ``tokens`` from the
provider's reported token usage."""


def flat_unit_rate(prices: GatewayTokenPrices, kind: BilledUnitKind) -> int | None:
    """The lane's flat per-unit rate for one unit kind, or None when it has none.

    Speech and transcription bill one flat SKU per lane (``tts-1-hd`` is its own
    model, not a variant), so the reservation and the data plane's billed
    units both use the flat ``""`` variant.

    Args:
        prices: The deployment's frozen price card.
        kind: The unit kind the surface bills in.

    Returns:
        Integer nano-USD per unit, or None.
    """
    units = prices.units
    if units is None or units.kind != kind:
        return None
    return units.rate_for("")


def audio_token_rates(prices: GatewayTokenPrices) -> tuple[int | None, int | None]:
    """The input and output token rates an audio reservation must price at.

    An audio lane's input bound is a byte ceiling that can sit far above its
    real count, so a long-context schedule is always treated as reachable:
    each rate is the higher of the base and premium schedules. A rate either
    schedule leaves unknown leaves the lane unpriced (None).

    Args:
        prices: The lane's price card.

    Returns:
        The input and output rates, each None when a schedule leaves it unknown.
    """
    tier = prices.long_context
    schedules = [prices] if tier is None else [prices, tier]
    input_rates = [schedule.input_nano_usd_per_million_tokens for schedule in schedules]
    output_rates = [schedule.output_nano_usd_per_million_tokens for schedule in schedules]
    return (
        None if None in input_rates else max(rate for rate in input_rates if rate is not None),
        None if None in output_rates else max(rate for rate in output_rates if rate is not None),
    )


def billing_mode(deployment: ExactModelDeployment, surface: AudioSurface) -> BillingMode | None:
    """Choose how one rung bills: its unit card or complete token rates, never both.

    The meter is read off the lane's price card, which must name exactly one.
    An audio answer carries one meter, so a lane priced both ways (a unit card
    of the surface's kind beside any token rate) could never settle both legs;
    a lane priced neither way, or with only one of the two token rates, could
    not price the usage it would receive. Each returns None and the rung is
    excluded (fail-closed), as is a token lane whose long-context schedule
    leaves a rate unknown (its reservation could not be priced). A token-priced
    lane must also declare its provider-enforced output ceiling: neither a
    transcript's length nor generated audio's is bounded by the request, and
    the ceiling is both the reservation and the settle bound. A BYOK audio lane
    is authored with the rates its own provider charges, which also price the
    attributed cost.

    Args:
        deployment: The rung's certified deployment.
        surface: The audio surface being served.

    Returns:
        ``units``, ``tokens``, or None when the lane cannot bill this surface.
    """
    kind = BilledUnitKind.CHARACTER if surface == "speech" else BilledUnitKind.AUDIO_SECOND
    prices = deployment.gateway.prices
    unit_priced = flat_unit_rate(prices, kind) is not None
    # A long-context schedule is a second meter too: audio treats it as reachable.
    schedules = [prices] if prices.long_context is None else [prices, prices.long_context]
    any_token_rate = any(
        schedule.input_nano_usd_per_million_tokens is not None
        or schedule.output_nano_usd_per_million_tokens is not None
        for schedule in schedules
    )
    if unit_priced:
        return None if any_token_rate else "units"
    if None in audio_token_rates(prices):
        return None
    if deployment.capabilities is None or deployment.capabilities.maximum_output_tokens is None:
        return None
    return "tokens"
