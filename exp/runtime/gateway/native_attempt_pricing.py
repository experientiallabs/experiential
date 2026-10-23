# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""The one card each physical attempt reserves, freezes, and settles against."""

from __future__ import annotations

from datetime import datetime

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.native_execution import deployment_priced_for_service_tier
from exp.runtime.gateway.service_tiers import GatewayServiceTierAdmission, tier_admission


def deployment_priced_at[DeploymentT: ExactModelDeployment | None](
    deployment: DeploymentT, instant: datetime
) -> DeploymentT:
    """Resolve one deployment's off-peak schedule for an attempt starting at ``instant``.

    No deployment or no schedule returns the argument itself; a scheduled rung
    always comes back schedule-free, at its peak or off-peak card. The copy
    stays Python-side and never crosses the native boundary.
    """
    if deployment is None:
        return deployment
    prices = deployment.gateway.prices
    effective = prices.effective_at(instant)
    if effective is prices:
        return deployment
    return deployment.model_copy(
        update={"gateway": deployment.gateway.model_copy(update={"prices": effective})}
    )


def priced_attempt(
    deployment: ExactModelDeployment,
    service_tier: str | None,
    *,
    forwards_tier: bool,
    instant: datetime,
) -> tuple[ExactModelDeployment, GatewayServiceTierAdmission | None]:
    """Select the one card an attempt reserves, freezes, and settles against.

    The off-peak window resolves first, once, at the attempt's start; every
    later derivation (tier admission, the reservation ceiling, the ledger's
    frozen card) reads the returned copy, so a request that crosses a window
    boundary settles at the card it reserved. A schedule never coexists with
    tier cards (the card validator refuses it), so the two never compose.

    Args:
        deployment: The rung's catalog deployment.
        service_tier: The caller's requested processing tier, if any.
        forwards_tier: Whether this rung forwards the requested tier upstream.
        instant: Timezone-aware attempt start.

    Returns:
        The repriced deployment and its tier admission, or None when untiered.
    """
    timed = deployment_priced_at(deployment, instant)
    tier = tier_admission(
        timed.gateway.prices,
        service_tier,
        forwards_tier=forwards_tier,
        customer_managed=timed.billing_source == "customer_managed",
    )
    return deployment_priced_for_service_tier(
        timed, service_tier, forwards_tier=forwards_tier
    ), tier
