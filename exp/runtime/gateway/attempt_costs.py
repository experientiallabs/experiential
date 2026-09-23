"""Conservative integer nano-USD reservation ceilings for each serving surface."""

from __future__ import annotations

from typing import assert_never

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.attempt_tokens import worst_case_input_tokens, worst_case_output_tokens
from exp.runtime.gateway.cache_write import requests_hour_cache
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.runtime.gateway.embeddings_contracts import (
    EmbeddingsRequest,
    ServingRequest,
    embeddings_input_ceiling_nano_usd,
)
from exp.runtime.gateway.images_contracts import ImagesRequest, images_ceiling_nano_usd
from exp.runtime.gateway.ledger_valuation import require_representable_nano_usd

LONG_CONTEXT_TIER_MARGIN_PERCENT = 20
"""How far below a long-context threshold the input estimate may sit and still
reserve at the premium schedule.

The input reservation is a realistic estimate with headroom, not an upper
bound, so an estimate just under the threshold can settle just over it and
be repriced for the WHOLE request (the tier doubles Gemini's rates above
200k). The reservation therefore treats the tier as reachable inside this
band below the threshold: a request estimated at 160k+ tokens against a 200k
tier reserves at premium rates and settles at whatever schedule the provider
actually applied. The cost of the rule is a one-attempt over-reservation of
roughly the tier multiple inside the band; without it a hard monthly budget
could be overdrawn by the same multiple on a threshold-straddling request.
"""


def maximum_attempt_cost_nano_usd(
    request: ServingRequest,
    deployment: ExactModelDeployment,
    *,
    input_tokens: int | None = None,
) -> int | None:
    """Return a conservative nano-USD ceiling for one physical call (per surface).

    ``input_tokens`` is the request's :func:`worst_case_input_tokens` when the
    caller already computed it (a ladder walk prices every candidate from one
    tokenizer pass); it is computed here otherwise. Both the platform's token
    reservation and this money ceiling price the same estimate.
    """
    if input_tokens is None:
        input_tokens = worst_case_input_tokens(request)
    match request:
        case EmbeddingsRequest():
            return embeddings_input_ceiling_nano_usd(
                input_tokens=input_tokens,
                input_rate=deployment.gateway.prices.input_nano_usd_per_million_tokens,
            )
        case ImagesRequest():
            return images_ceiling_nano_usd(
                request,
                input_tokens=input_tokens,
                input_rate=deployment.gateway.prices.input_nano_usd_per_million_tokens,
                output_rate=deployment.gateway.prices.output_nano_usd_per_million_tokens,
            )
        case GatewayRequest() | DecisionRequest():
            return _token_attempt_cost_nano_usd(request, deployment, input_tokens)
        case _:  # pragma: no cover - exhaustive over the ServingRequest union.
            assert_never(request)


def _token_attempt_cost_nano_usd(
    request: GatewayRequest | DecisionRequest,
    deployment: ExactModelDeployment,
    input_tokens: int,
) -> int | None:
    """Reserve the maximum applicable rate for each surface-specific token bound.

    Args:
        request: Canonical request including forwarded cache TTL markers.
        deployment: Frozen capabilities and base, tier, and cache-write prices.
        input_tokens: Input estimate including the configured headroom.

    Returns:
        Nano-USD ceiling, or None when an applicable rate is unknown.
    """
    output_tokens = worst_case_output_tokens(request, deployment)
    prices = deployment.gateway.prices
    capabilities = deployment.gateway.capabilities
    # The tier reprices the whole request once actual input reaches its
    # threshold, and the estimate can land under a threshold the provider's
    # count then crosses, so the tier is treated as reachable from
    # LONG_CONTEXT_TIER_MARGIN_PERCENT below it; a reachable tier must survive
    # the whole-request premium schedule.
    tier = prices.long_context
    if tier is not None and input_tokens * 100 < tier.input_threshold_tokens * (
        100 - LONG_CONTEXT_TIER_MARGIN_PERCENT
    ):
        tier = None
    schedules = [prices] if tier is None else [prices, tier]
    for schedule in schedules:
        required_rates = [
            schedule.input_nano_usd_per_million_tokens,
            schedule.output_nano_usd_per_million_tokens,
        ]
        if capabilities.reports_cached_input_tokens:
            required_rates.append(schedule.cached_input_nano_usd_per_million_tokens)
        if capabilities.reports_cache_creation_input_tokens:
            required_rates.append(schedule.cache_creation_input_nano_usd_per_million_tokens)
            if requests_hour_cache(request):
                required_rates.append(schedule.cache_creation_1h_input_nano_usd_per_million_tokens)
        if capabilities.reports_reasoning_tokens:
            required_rates.append(schedule.reasoning_nano_usd_per_million_tokens)
        if any(rate is None for rate in required_rates):
            return None
    input_rate = max(
        rate
        for schedule in schedules
        for rate in (
            schedule.input_nano_usd_per_million_tokens,
            schedule.cached_input_nano_usd_per_million_tokens,
            schedule.cache_creation_input_nano_usd_per_million_tokens
            if capabilities.reports_cache_creation_input_tokens
            else None,
            schedule.cache_creation_1h_input_nano_usd_per_million_tokens
            if capabilities.reports_cache_creation_input_tokens and requests_hour_cache(request)
            else None,
        )
        if rate is not None
    )
    output_rate = max(
        rate
        for schedule in schedules
        for rate in (
            schedule.output_nano_usd_per_million_tokens,
            schedule.reasoning_nano_usd_per_million_tokens,
        )
        if rate is not None
    )
    numerator = input_tokens * input_rate
    numerator += output_tokens * output_rate
    return require_representable_nano_usd(
        (numerator + 999_999) // 1_000_000, what="attempt reservation ceiling"
    )
