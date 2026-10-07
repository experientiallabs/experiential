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


def billing_mode(deployment: ExactModelDeployment, surface: AudioSurface) -> BillingMode | None:
    """Choose how one rung bills: its unit card or complete token rates, never both.

    The meter is read off the lane's price card, which must name exactly one.
    An audio answer carries one meter, so a lane priced both ways (a unit card
    of the surface's kind beside any token rate) could never settle both legs;
    a lane priced neither way, or with only one of the two token rates, could
    not price the usage it would receive. Each returns None and the rung is
    excluded (fail-closed). A token-priced transcription lane must also declare
    its provider-enforced output ceiling, since a transcript's length is not
    bounded by the recording. A BYOK audio lane is authored with the rates its
    own provider charges, which also price the attributed cost.

    Args:
        deployment: The rung's certified deployment.
        surface: The audio surface being served.

    Returns:
        ``units``, ``tokens``, or None when the lane cannot bill this surface.
    """
    kind = BilledUnitKind.CHARACTER if surface == "speech" else BilledUnitKind.AUDIO_SECOND
    prices = deployment.gateway.prices
    unit_priced = flat_unit_rate(prices, kind) is not None
    rates = (
        prices.input_nano_usd_per_million_tokens,
        prices.output_nano_usd_per_million_tokens,
    )
    any_token_rate = any(rate is not None for rate in rates)
    if unit_priced:
        return None if any_token_rate else "units"
    if None in rates:
        return None
    if surface == "transcription" and (
        deployment.capabilities is None or deployment.capabilities.maximum_output_tokens is None
    ):
        return None
    return "tokens"
