"""Per-dialect provider wire payload dispatch for canonical gateway requests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayRequest
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.fireworks import (
    require_responses_continuation_channel,
)
from exp.runtime.models.providers.messages_payloads import (
    anthropic_messages_stream_payload,
    bedrock_converse_stream_payload,
    gemini_generate_content_stream_payload,
)
from exp.runtime.models.providers.openai_payloads import (
    openai_compatible_stream_payload,
    openai_responses_stream_payload,
)

if TYPE_CHECKING:
    from exp.runtime.models.providers.base import GatewayWireProfile

SERVICE_TIER_DIALECTS = frozenset({"openai_responses", "openai_compatible"})
"""Wire dialects with a request field that preserves the caller's service tier."""


def _fireworks_continuation_required(profile: GatewayWireProfile, request: GatewayRequest) -> bool:
    """Return whether a Fireworks Responses turn can emit an unretained tool call."""
    return (
        request.surface == GatewayApiSurface.RESPONSES
        and (urlsplit(profile.url).hostname or "").lower() == "api.fireworks.ai"
        and bool(request.tools)
        and request.tool_choice != "none"
    )


def dialect_stream_payload(
    profile: GatewayWireProfile,
    provider_request: GatewayRequest,
) -> JsonObject:
    """Build the provider wire payload for one resolved wire profile.

    Args:
        profile: The resolved connection's wire profile.
        provider_request: Canonical request forced into streaming mode.

    Returns:
        The exact JSON payload the gateway sends upstream for this dialect.

    Raises:
        ProviderCapabilityError: The request uses a capability this dialect
            cannot preserve.
    """
    if _fireworks_continuation_required(profile, provider_request):
        require_responses_continuation_channel(provider_request)
    if provider_request.service_tier is not None and profile.dialect not in SERVICE_TIER_DIALECTS:
        # A processing-tier hint changes pricing and latency semantics, so a
        # dialect with no wire field for it declines instead of dropping the
        # field silently: admission then prefers a tier-preserving rung and
        # otherwise retries with the disclosed drop in capability_policy.
        raise ProviderCapabilityError(capability="service_tier")
    required_reasoning_effort = (
        profile.reasoning_effort if profile.reasoning_effort_required else None
    )
    if profile.dialect == "openai_responses":
        return openai_responses_stream_payload(
            profile.model_id,
            provider_request,
            supports_temperature=profile.supports_temperature,
            supports_top_p=(
                profile.supports_temperature
                if profile.supports_top_p is None
                else profile.supports_top_p
            ),
            supports_top_k=profile.supports_top_k,
            supports_logprobs=profile.supports_logprobs,
            supports_reasoning=profile.supports_reasoning,
            reasoning_effort=required_reasoning_effort,
            sampling_requires_reasoning_none=profile.sampling_requires_reasoning_none,
            forwards_service_tier=profile.billing_customer_managed,
        )
    if profile.dialect == "anthropic_messages":
        return anthropic_messages_stream_payload(
            profile.model_id,
            provider_request,
            supports_temperature=profile.supports_temperature,
            supports_top_p=(
                profile.supports_temperature
                if profile.supports_top_p is None
                else profile.supports_top_p
            ),
            supports_top_k=profile.supports_top_k,
            supports_logprobs=profile.supports_logprobs,
            supports_reasoning=profile.supports_reasoning,
            reasoning_effort=required_reasoning_effort,
        )
    if profile.dialect == "gemini_generate_content":
        return gemini_generate_content_stream_payload(
            profile.model_id,
            provider_request,
            supports_temperature=profile.supports_temperature,
            supports_top_p=(
                profile.supports_temperature
                if profile.supports_top_p is None
                else profile.supports_top_p
            ),
            supports_top_k=profile.supports_top_k,
            supports_logprobs=profile.supports_logprobs,
            supports_reasoning=profile.supports_reasoning,
            reasoning_effort=required_reasoning_effort,
        )
    if profile.dialect == "bedrock_converse_stream":
        return bedrock_converse_stream_payload(
            profile.model_id,
            provider_request,
            supports_temperature=profile.supports_temperature,
            supports_top_p=(
                profile.supports_temperature
                if profile.supports_top_p is None
                else profile.supports_top_p
            ),
            supports_top_k=profile.supports_top_k,
            supports_logprobs=profile.supports_logprobs,
        )
    if profile.dialect == "openai_compatible":
        if profile.fireworks_reasoning_route_sha256 is not None:
            require_responses_continuation_channel(provider_request)
        return openai_compatible_stream_payload(
            profile.model_id,
            provider_request,
            token_limit_key=profile.token_limit_key,
            supports_temperature=profile.supports_temperature,
            supports_top_p=(
                profile.supports_temperature
                if profile.supports_top_p is None
                else profile.supports_top_p
            ),
            supports_top_k=profile.supports_top_k,
            supports_frequency_penalty=profile.supports_frequency_penalty,
            supports_presence_penalty=profile.supports_presence_penalty,
            supports_logprobs=profile.supports_logprobs,
            supports_reasoning=profile.supports_reasoning,
            reasoning_wire_format=profile.reasoning_wire_format,
            reasoning_effort=required_reasoning_effort,
            sampling_requires_reasoning_none=profile.sampling_requires_reasoning_none,
            fireworks_reasoning_route_sha256=profile.fireworks_reasoning_route_sha256,
            hunyuan_reasoning_route_sha256=profile.hunyuan_reasoning_route_sha256,
            forwards_service_tier=profile.billing_customer_managed,
        )
    raise ProviderCapabilityError(capability=f"wire_dialect:{profile.dialect}")
