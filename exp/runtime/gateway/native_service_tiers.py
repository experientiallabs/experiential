# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Tier admission ceilings and native evidence bound to immutable attempt cards."""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, ConfigDict

from exp.common.core.artifacts import JsonObject
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.attempt_costs import maximum_attempt_cost_nano_usd
from exp.runtime.gateway.embeddings_contracts import ServingRequest
from exp.runtime.gateway.service_tiers import (
    GatewayServiceTierAdmission,
    GatewayServiceTierSettlement,
    ServedServiceTier,
    ServiceTierResolution,
)


class TierObservation(BaseModel):
    """The native normalizer's bounded evidence, with no provider-authored price fields.

    Attributes:
        served: Canonical witnessed processing tier.
        resolution: Whether all supplied evidence agrees on a known tier.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    served: ServedServiceTier | None = None
    resolution: ServiceTierResolution = "missing"


class AdmissionKwarg(TypedDict, total=False):
    """Optional admission keyword, omitted on every untiered attempt."""

    service_tier: GatewayServiceTierAdmission


class SettlementKwarg(TypedDict, total=False):
    """Optional receipt keyword, omitted on every untiered attempt."""

    service_tier: GatewayServiceTierSettlement


def admission_kwarg(tier: GatewayServiceTierAdmission | None) -> AdmissionKwarg:
    """Never send a new keyword to untiered ledger callers."""
    return {} if tier is None else {"service_tier": tier}


def settlement_kwarg(tier: GatewayServiceTierAdmission | None, data: JsonObject) -> SettlementKwarg:
    """Bind native evidence to the exact cards accepted for this physical attempt."""
    if tier is None:
        return {}
    observation = TierObservation.model_validate(data.get("service_tier") or {})
    return {
        "service_tier": tier.settlement(
            served=observation.served, resolution=observation.resolution
        )
    }


def tier_ceiling(
    request: ServingRequest,
    deployment: ExactModelDeployment,
    tier: GatewayServiceTierAdmission | None,
    *,
    input_tokens: int,
) -> int | None:
    """Reserve the largest possible admitted schedule, including standard fallback."""
    requested = maximum_attempt_cost_nano_usd(request, deployment, input_tokens=input_tokens)
    if tier is None:
        return requested
    standard = deployment.model_copy(
        update={"gateway": deployment.gateway.model_copy(update={"prices": tier.standard_prices})}
    )
    fallback = maximum_attempt_cost_nano_usd(request, standard, input_tokens=input_tokens)
    return None if requested is None or fallback is None else max(requested, fallback)
