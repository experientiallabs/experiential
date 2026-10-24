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

A rung whose authored dispatch policy sets ``upstream_priority`` (a self-hosted
vLLM lane) carries the caller's admission level as vLLM's ``priority`` field.

OpenRouter's sampling extras (``repetition_penalty``, ``min_p``, ``top_a``) are
set on each rung whose wire carries them and disclosed as dropped on the rest
(``rung_sampling_extras``), so the same request is honored on an OpenRouter or
self-hosted vLLM rung and served without them on a strict one.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from exp.common.core.artifacts import JsonObject, sha256_bytes
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.class_throttles import fast_on_rung, upstream_priority
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
from exp.runtime.gateway.sampling_extras import (
    OPENROUTER_SAMPLING_EXTRAS,
    VLLM_SAMPLING_EXTRAS,
    SamplingExtraName,
    forward_sampling_extras,
)
from exp.runtime.models.providers import (
    emulated_gateway_capabilities,
    emulated_stop_sequences,
    preflight_gateway_request,
    require_gateway_provider,
)
from exp.runtime.models.providers.anthropic import safeguards_for_upstream
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.generation_parameter_validation import bounded_output_request
from exp.runtime.models.providers.logprobs import (
    capture_logprobs_eligible,
    require_chat_logprobs,
    require_responses_logprobs,
)
from exp.runtime.models.providers.openrouter_routing import (
    constrain_openrouter_reasoning_off,
    constrain_openrouter_zero_data_retention,
    forward_provider_preferences,
    openrouter_chat_wire,
    openrouter_metadata_headers,
)
from exp.runtime.models.providers.protocol import (
    GatewayDispatchSigner,
    NativeWireClient,
    developer_role_rewrite_disclosure,
)
from exp.runtime.models.providers.wire_messages import anthropic_request_headers

NOVITA_HOST = "api.novita.ai"
"""Novita's origin host: its rungs pin a conversation by ``x-session-id``."""

NOVITA_SESSION_HEADER = "x-session-id"
"""The header Novita routes cache affinity by (configured on the house account)."""

ZDR_CONSTRAINT_CAPABILITY = "zero_data_retention_constraint"
"""The capability a flagged rung's wire must express, named on the refusal."""

UPSTREAM_PRIORITY_DIALECTS = frozenset({"openai_compatible", "openai_responses"})
"""The wires a vLLM server answers, where its ``priority`` body field exists."""

UPSTREAM_PRIORITY = {2: 0, 1: 1, 0: 2}
"""vLLM ``priority`` per ``AuthorizationSnapshot.priority_admission`` level.

Pro 0, paying 1, free 2: vLLM's priority scheduler runs the lowest value
first and, when KV runs out, preempts the highest value first.
"""


@dataclass(frozen=True, slots=True)
class RungDispatch:
    """One deployment's frozen dispatch material plus its shaping disclosure."""

    wire_entry: JsonObject
    signer: GatewayDispatchSigner | None
    binding: FrozenDispatchBinding | None
    carrier_authority: ReasoningCarrierAuthority | None
    parallel_disclosure: str | None
    output_disclosure: str | None
    developer_disclosure: str | None
    reserved_output_tokens: int
    sampling_disclosures: tuple[str, ...] = ()

    @property
    def disclosures(self) -> tuple[str, ...]:
        """Every per-rung shaping disclosure this dispatch publishes."""
        return tuple(
            disclosure
            for disclosure in (
                self.parallel_disclosure,
                self.output_disclosure,
                self.developer_disclosure,
                *self.sampling_disclosures,
            )
            if disclosure is not None
        )


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
    developer_disclosure = developer_role_rewrite_disclosure(
        profile.dialect, provider_request, deployment.gateway.capabilities
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
    # Auto-mode safeguards ride only an attempt on Anthropic's own API; the
    # payload and the beta header below both read this per-rung request.
    rung_request = safeguards_for_upstream(profile.url, rung_request)
    upstream_payload = image_aware_stream_payload(
        profile, rung_request, capabilities, deployment.provider
    )
    upstream_payload, sampling_disclosures = forward_sampling_extras(
        upstream_payload,
        rung_request.sampling_extras,
        rung_sampling_extras(deployment, profile.dialect),
    )
    if rung_request.provider_preferences is not None and openrouter_chat_wire(
        deployment.provider, profile.dialect
    ):
        # Only the caller's data-policy keys reach OpenRouter; routing keys
        # were resolved against our own rungs at admission.
        upstream_payload = forward_provider_preferences(
            upstream_payload, rung_request.provider_preferences
        )
    if openrouter_chat_wire(deployment.provider, profile.dialect):
        # After the caller's preferences, which replace the provider object.
        upstream_payload = constrain_openrouter_reasoning_off(upstream_payload)
    dispatch_policy = deployment.gateway.dispatch
    if (
        dispatch_policy is not None
        and dispatch_policy.upstream_priority
        and profile.dialect in UPSTREAM_PRIORITY_DIALECTS
    ):
        # Only a rung the host authored as self-hosted vLLM: every other
        # provider would receive an unknown field.
        upstream_payload = {
            **upstream_payload,
            "priority": upstream_priority(authorization, deployment, UPSTREAM_PRIORITY),
        }
        if fast_on_rung(authorization, deployment):
            # vLLM bills no processing tier: a :fast request's tier is ours
            # to serve (class_throttles), so the field never reaches it.
            upstream_payload.pop("service_tier", None)
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
    request_headers = with_session_affinity(profile, rung_request, request_headers)
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
        capture_logprobs=capture_logprobs_eligible(profile, rung_request, caller=public_request),
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
        developer_disclosure=developer_disclosure,
        reserved_output_tokens=output_bound,
        sampling_disclosures=sampling_disclosures,
    )


def rung_sampling_extras(
    deployment: ExactModelDeployment, dialect: str
) -> frozenset[SamplingExtraName]:
    """The OpenRouter sampling extras one rung's wire carries.

    OpenRouter's Chat Completions wire documents all three. A self-hosted vLLM
    rung (the host authored ``upstream_priority`` on it, the same signal that
    sends vLLM's ``priority`` field) takes ``repetition_penalty`` and ``min_p``
    but has no ``top_a``. Every other wire carries none: the catalog's generic
    ``openai-compatible`` provider also covers resellers (Groq, Mistral, xAI,
    Cerebras and others) that refuse unknown body fields with a 400, and a
    provider 400 does not fail over, so the extras are dropped there with
    disclosure rather than sent.

    Args:
        deployment: The rung.
        dialect: The rung's resolved wire dialect.

    Returns:
        The extras to set on this rung's payload.
    """
    if openrouter_chat_wire(deployment.provider, dialect):
        return OPENROUTER_SAMPLING_EXTRAS
    policy = deployment.gateway.dispatch
    if dialect == "openai_compatible" and policy is not None and policy.upstream_priority:
        return VLLM_SAMPLING_EXTRAS
    return frozenset()


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
    if not openrouter_chat_wire(deployment.provider, profile.dialect):
        raise ProviderCapabilityError(capability=ZDR_CONSTRAINT_CAPABILITY)
    return (
        constrain_openrouter_zero_data_retention(upstream_payload),
        openrouter_metadata_headers(dict(profile.headers)),
    )


def with_session_affinity(
    profile: GatewayWireProfile,
    rung_request: GatewayRequest,
    request_headers: dict[str, str] | None,
) -> dict[str, str] | None:
    """Name the conversation in ``x-session-id`` on a rung served from Novita's origin.

    Novita pins a session to the server holding its prefix cache when the
    request carries this header; its value is the tenant-namespaced
    ``provider_session_id`` admission derived, never the caller's raw key. A
    request with no session id (prompt caching off, or no user turn) and every
    other origin keep their headers unchanged.

    Args:
        profile: The rung's resolved wire profile.
        rung_request: The request this rung dispatches.
        request_headers: Per-request headers already composed, or ``None`` for
            the profile's static headers.

    Returns:
        The headers to dispatch with, the profile's static ones merged in when
        the session header is added.
    """
    session_id = rung_request.provider_session_id
    if session_id is None or (urlsplit(profile.url).hostname or "").lower() != NOVITA_HOST:
        return request_headers
    base = dict(profile.headers) if request_headers is None else request_headers
    return {**base, NOVITA_SESSION_HEADER: session_id}
