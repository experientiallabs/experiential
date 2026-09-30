# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Frozen processing-tier cards and evidence-based settlement, never provider prices."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BeforeValidator, model_validator

from exp.common.core.artifacts import ContractModel
from exp.common.models.catalog_prices import GatewayTokenPrices


def normalize_service_tier(value: object) -> object:
    """Normalize the public Fast spelling before validation and replay identity."""
    return "priority" if value == "fast" else value


RequestedServiceTier = Literal["flex", "priority"]
ServedServiceTier = Literal["default", "flex", "priority"]
ServiceTierResolution = Literal["confirmed", "missing", "unknown", "conflicting"]
ServiceTier = Annotated[
    Literal["auto", "default", "flex", "scale", "priority"],
    BeforeValidator(normalize_service_tier),
]


class GatewayServiceTierAdmission(ContractModel):
    """The only schedules a tiered physical attempt can settle against.

    Attributes:
        requested: Canonical processing tier actually forwarded to the provider.
        standard_prices: Complete immutable standard schedule, including long context.
        requested_prices: Complete immutable requested-tier schedule, including writes.
    """

    requested: RequestedServiceTier
    standard_prices: GatewayTokenPrices
    requested_prices: GatewayTokenPrices

    @model_validator(mode="after")
    def require_flat_cards(self) -> Self:
        """Reject recursive tier tables; an admission freezes exactly two schedules."""
        for card in (self.standard_prices, self.requested_prices):
            if card.flex is not None or card.priority is not None:
                raise ValueError("service-tier admission schedules cannot contain other tier cards")
        return self

    def settlement(
        self, *, served: ServedServiceTier | None, resolution: ServiceTierResolution
    ) -> GatewayServiceTierSettlement:
        """Select only a witnessed, admitted card; anything else retains billing liability."""
        prices = None
        if resolution == "confirmed":
            if served == "default":
                prices = self.standard_prices
            elif served == self.requested:
                prices = self.requested_prices
            else:
                resolution = "unknown"
        return GatewayServiceTierSettlement(
            requested=self.requested,
            served=served if prices is not None else None,
            prices=prices,
            resolution=resolution,
        )


class GatewayServiceTierSettlement(ContractModel):
    """A witnessed tier and its admitted card, or an explicit unresolved billing hold.

    Attributes:
        requested: Canonical processing tier from admission.
        served: Witnessed canonical tier; standard normalizes to default.
        prices: Admitted schedule, or None to retain the reservation without charging it.
        resolution: Evidence verdict; missing, unknown and conflicting never imply standard.
    """

    requested: RequestedServiceTier
    served: ServedServiceTier | None
    prices: GatewayTokenPrices | None
    resolution: ServiceTierResolution

    @model_validator(mode="after")
    def require_evidence(self) -> Self:
        """Keep the billing hold distinct from a confirmed, possibly zero-priced card."""
        if self.resolution == "confirmed":
            if self.served not in ("default", self.requested) or self.prices is None:
                raise ValueError(
                    "confirmed service-tier settlement needs an admitted tier and card"
                )
            if self.prices.flex is not None or self.prices.priority is not None:
                raise ValueError("settlement prices cannot contain other tier cards")
        elif self.prices is not None or self.served is not None:
            raise ValueError("unconfirmed service-tier settlement cannot claim a tier or prices")
        return self


def tier_admission(
    prices: GatewayTokenPrices, tier: str | None, *, forwards_tier: bool, customer_managed: bool
) -> GatewayServiceTierAdmission | None:
    """Freeze admitted house-tier cards, leaving standard and BYOK accounting unchanged."""
    if customer_managed or not forwards_tier or tier not in ("flex", "priority"):
        return None
    if prices.service_tier(tier) is None:
        return None
    return GatewayServiceTierAdmission(
        requested="flex" if tier == "flex" else "priority",
        standard_prices=prices.model_copy(update={"flex": None, "priority": None}),
        requested_prices=prices.for_service_tier(tier),
    )
