"""Per-rung dispatch construction for the native bridge.

Admission hands the bridge an ordered route; this module turns one rung of
it into everything the data plane needs for that deployment: the frozen wire
entry, the dispatch signer, the frozen-body binding and the reasoning-carrier
authority. Rungs are shaped independently so a control (``parallel_tool_calls``
today) can be honored natively on one rung and emulated on the next without
the route-level request changing.

A rung the host flagged in ``snapshot.zdr_constrained_deployment_ids`` is
frozen with OpenRouter's zero-data-retention routing constraint on its payload
and the metadata opt-in header on its dispatch; a flagged rung on any other
wire fails closed here, never dispatching unconstrained.
"""

from __future__ import annotations

from dataclasses import dataclass

from exp.common.core.artifacts import JsonObject, sha256_bytes
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayApiSurface, GatewayRequest
from exp.runtime.gateway.native_admission import shape_parallel_tool_calls
from exp.runtime.gateway.native_dispatch import frozen_dispatch
from exp.runtime.gateway.native_execution import FrozenDispatchBinding, deployment_wire_entry
from exp.runtime.gateway.native_image_output import image_aware_stream_payload
from exp.runtime.gateway.reasoning_carrier import (
    ReasoningCarrierAuthority,
    reasoning_carrier_authority,
    scheme_for_profile,
)
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models.providers import (
    emulated_gateway_capabilities,
    emulated_stop_sequences,
    preflight_gateway_request,
    require_gateway_provider,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.generation_parameter_validation import bounded_output_request
from exp.runtime.models.providers.logprobs import require_chat_logprobs, require_responses_logprobs
from exp.runtime.models.providers.openrouter_routing import (
    OPENROUTER_PROVIDER_ID,
    constrain_openrouter_zero_data_retention,
    forward_provider_preferences,
    openrouter_metadata_headers,
)
from exp.runtime.models.providers.protocol import GatewayDispatchSigner, NativeWireClient
from exp.runtime.models.providers.wire_messages import anthropic_request_headers

ZDR_CONSTRAINT_CAPABILITY = "zero_data_retention_constraint"
"""The capability a flagged rung's wire must express, named on the refusal."""


@dataclass(frozen=True, slots=True)
class RungDispatch:
    """One deployment's frozen dispatch material plus its shaping disclosure."""

    wire_entry: JsonObject
    signer: GatewayDispatchSigner | None
    binding: FrozenDispatchBinding | None
    carrier_authority: ReasoningCarrierAuthority | None
    parallel_disclosure: str | None
    output_disclosure: str | None
    reserved_output_tokens: int


def build_rung_dispatch(
    route: GatewayRoute,
    deployment: ExactModelDeployment,
    profile: GatewayWireProfile,
    client: NativeWireClient,
    *,
    provider_request: GatewayRequest,
    public_request: GatewayRequest,
    authorization: AuthorizationSnapshot,
    throttle_redial_budget: int = 0,
) -> RungDispatch:
    """Preflight, shape and freeze ``provider_request`` for one route rung.

    ``throttle_redial_budget`` rides onto the wire entry unchanged: it is
    the admission-time count of post-backoff redials a throttle on this rung
    is worth under the pool's ``throttle_redial`` schedule for this request.
    """
    require_gateway_provider(deployment.provider)
    preflight_gateway_request(
        provider_request,
        deployment.gateway.capabilities,
        model_capabilities=deployment.capabilities,
        public_stream=public_request.stream,
        route_provider=deployment.provider,
        emulated_capabilities=emulated_gateway_capabilities(
            profile.dialect, emulate_parallel_tool_calls=True
        ),
    )
    rung_request, parallel_disclosure = shape_parallel_tool_calls(
        provider_request, deployment.gateway.capabilities
    )
    capabilities = deployment.capabilities
    rung_request, output_bound = bounded_output_request(
        profile,
        rung_request,
        model_maximum_output_tokens=(
            capabilities.maximum_output_tokens if capabilities is not None else None
        ),
        context_window_tokens=(
            capabilities.context_window_tokens if capabilities is not None else None
        ),
    )
    output_disclosure = (
        f"max_tokens->default({output_bound};{profile.dialect};declared_bound)"
        if provider_request.maximum_output_tokens is None
        and rung_request.maximum_output_tokens is not None
        else None
    )
    if rung_request.surface == GatewayApiSurface.CHAT_COMPLETIONS:
        require_chat_logprobs((profile,), rung_request)
    require_responses_logprobs((profile,), rung_request)
    upstream_payload = image_aware_stream_payload(
        profile, rung_request, capabilities, deployment.provider
    )
    if rung_request.provider_preferences is not None and _openrouter_wire(deployment, profile):
        # The caller's routing preferences reach the one wire that defines
        # them; every other dialect's builder never emits the field.
        upstream_payload = forward_provider_preferences(
            upstream_payload, rung_request.provider_preferences
        )
    request_headers = (
        anthropic_request_headers(dict(profile.headers), rung_request)
        if profile.dialect == "anthropic_messages"
        else None
    )
    zdr_constrained = deployment.deployment_id in route.snapshot.zdr_constrained_deployment_ids
    if zdr_constrained:
        upstream_payload, request_headers = zdr_constrained_dispatch(
            deployment, profile, upstream_payload
        )
    upstream_body, signer = frozen_dispatch(profile, client, upstream_payload)
    wire_entry = deployment_wire_entry(
        route,
        deployment,
        profile,
        upstream_payload,
        upstream_body,
        headers=request_headers,
        stop_sequences=emulated_stop_sequences(profile.dialect, rung_request),
        serialize_tool_calls=rung_request.serialize_tool_calls,
        throttle_redial_budget=throttle_redial_budget,
        native_tool_translation=rung_request.native_tool_translation,
        zdr_constrained=zdr_constrained,
    )
    binding = (
        None
        if signer is None or upstream_body is None
        else FrozenDispatchBinding(
            url=profile.url,
            body_sha256=sha256_bytes(upstream_body.encode("utf-8")),
        )
    )
    carrier_scheme = scheme_for_profile(profile)
    carrier_authority = (
        None
        if carrier_scheme is None
        else reasoning_carrier_authority(
            authorization=authorization,
            exact_model_id=deployment.exact_model_id,
            pool_id=route.snapshot.stage_for_depth(
                route.snapshot.deployment_ids.index(deployment.deployment_id)
            ).pool_id,
            deployment=deployment,
            profile=profile,
            scheme=carrier_scheme,
        )
    )
    return RungDispatch(
        wire_entry=wire_entry,
        signer=signer,
        binding=binding,
        carrier_authority=carrier_authority,
        parallel_disclosure=parallel_disclosure,
        output_disclosure=output_disclosure,
        reserved_output_tokens=output_bound,
    )


def zdr_constrained_dispatch(
    deployment: ExactModelDeployment,
    profile: GatewayWireProfile,
    upstream_payload: JsonObject,
) -> tuple[JsonObject, dict[str, str]]:
    """Tighten one flagged rung's payload and headers to OpenRouter's ZDR constraint.

    Args:
        deployment: The flagged rung.
        profile: Its resolved wire profile.
        upstream_payload: The payload the dialect built for it.

    Returns:
        The payload with ``provider.zdr`` / ``provider.data_collection`` forced
        strict and the rung's headers plus the OpenRouter metadata opt-in.

    Raises:
        ProviderCapabilityError: The rung is not an OpenRouter Chat Completions
            wire, so no request field can express the constraint; the request
            fails closed rather than dispatching to a retaining upstream.
    """
    if not _openrouter_wire(deployment, profile):
        raise ProviderCapabilityError(capability=ZDR_CONSTRAINT_CAPABILITY)
    return (
        constrain_openrouter_zero_data_retention(upstream_payload),
        openrouter_metadata_headers(dict(profile.headers)),
    )


def _openrouter_wire(deployment: ExactModelDeployment, profile: GatewayWireProfile) -> bool:
    """Whether this rung is OpenRouter's Chat Completions wire (the only ``provider`` field)."""
    return deployment.provider == OPENROUTER_PROVIDER_ID and profile.dialect == "openai_compatible"
