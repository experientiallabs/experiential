"""Route admission with disclosed coercion for the native control plane.

Admission prefers rungs that preserve every caller semantic verbatim: the
generation-control narrowing and the per-deployment capability preflight plus
payload build both run here. When no rung preserves the request verbatim,
this module applies the capability-preservation policy's minimal disclosed
coercion exactly once per layer and re-selects; when nothing coercible
remains, the first rung's own field-scoped rejection stays the answer. A
coercion is never silent: every substitution is disclosed through ``ignored_parameters`` in
``path->effective`` form, warn-logged for operators, and counted in the
admission metrics.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from exp.common.models.catalog import GatewayDeploymentCapabilities
from exp.runtime.gateway import prompt_size
from exp.runtime.gateway.affinity import (
    affinity_fingerprint,
    affinity_seed_material,
    affinity_weighted_rungs,
    house_aggregator_last,
    rendezvous_order,
)
from exp.runtime.gateway.capability_rungs import narrow_to_capable_rungs
from exp.runtime.gateway.class_throttles import fast_request, require_fast_path
from exp.runtime.gateway.contracts import AuthorizationSnapshot, DirectTarget, GatewayRequest
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_components import NativeGatewayComponents
from exp.runtime.gateway.native_dead_rungs import DeadRung, log_dead_rungs
from exp.runtime.gateway.native_execution import (
    deployment_health_key,
    reorder_route_deployments,
    request_carries_cache_markers,
    select_route_deployments,
)
from exp.runtime.gateway.native_fallback_rules import require_unrestricted_rung
from exp.runtime.gateway.native_image_output import image_aware_stream_payload
from exp.runtime.gateway.native_reasoning import rung_provider_request
from exp.runtime.gateway.native_request_policy import restrict_fallbacks
from exp.runtime.gateway.native_responses import ContinuationContext
from exp.runtime.gateway.native_stage_admission import (
    require_native_model_stage_contract,
    stage_affinity_ordered_rungs,
)
from exp.runtime.gateway.prompt_caching import (
    disclose_best_effort_caching,
    with_cache_affinity,
    without_prompt_caching,
)
from exp.runtime.gateway.recovery_binding import bind_recovery_profiles
from exp.runtime.gateway.routing import GatewayRoute, GatewayRoutingError
from exp.runtime.gateway.sticky_affinity import AffinityPlacement, sticky_first_order
from exp.runtime.gateway.tool_search.plan import plan_tool_search
from exp.runtime.gateway.web_search.plan import strip_search_carriers
from exp.runtime.models.providers import (
    emulated_gateway_capabilities,
    preflight_gateway_request,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.capability_policy import (
    coerce_generation_parameters,
    coerce_route_rejections,
    coerce_strict_tool_schemas,
    coerce_structured_text_schema,
    reserve_thinking_headroom,
)
from exp.runtime.models.providers.dialect_dispatch import CACHE_CONTROL_NOT_FORWARDED_SUFFIX
from exp.runtime.models.providers.errors import (
    ProviderCapabilityError,
    ProviderParameterError,
    ProviderResponseError,
)
from exp.runtime.models.providers.generation_parameter_validation import bounded_output_request
from exp.runtime.models.providers.generation_route_compat import (
    compatible_generation_parameter_profile_indexes,
    plaintext_reasoning_lead_order,
)
from exp.runtime.models.providers.prompt_cache_options import honors_markers
from exp.runtime.models.providers.protocol import NativeWireClient
from exp.runtime.models.providers.streaming_requests import route_generation_parameter_requests
from exp.runtime.openai_protocol.state import ProtocolNamespace, episode_namespace

_logger = logging.getLogger(__name__)

_ResolvedWires = tuple[tuple[GatewayWireProfile, NativeWireClient], ...]


def admitted_route_requests(
    route: GatewayRoute,
    resolved_wires: _ResolvedWires,
    request: GatewayRequest,
    *,
    accounting: NativeAttemptAccounting,
    authorization: AuthorizationSnapshot,
    continuation: ContinuationContext | None = None,
) -> tuple[GatewayRoute, _ResolvedWires, GatewayRequest, GatewayRequest, AffinityPlacement]:
    """Narrow one certified route to rungs that serve the admitted request.

    Args:
        route: Frozen route aligned with ``resolved_wires``.
        resolved_wires: Ordered wire profiles and clients per deployment.
        request: Canonical request produced by the public protocol decoder.
        accounting: Shared accounting owning the coercion counter.
        authorization: Frozen authority for the accepted request.
        continuation: Responses continuation context for a stored response; its
            original episode key keeps the conversation's cache-affinity placement.

    Returns:
        The narrowed route and wires, the public request (carrying any
        coercion disclosures), the streaming-forced provider request, and the
        resolved affinity placement.

    Raises:
        ProviderParameterError: No rung preserves a generation control and no
            disclosed coercion applies, or rungs declined for different
            field-specific reasons.
        ProviderCapabilityError: The first rung's capability rejection when no rung
            is protocol-compatible, or a ``:fast`` request no rung serves fast.
        GatewayRoutingError: No rung is protocol-compatible and none named a
            rejection.
    """
    request = fast_request(without_prompt_caching(request, authorization), authorization)
    route, resolved_wires, public_request, provider_request, disclosures = prepare_route_requests(
        route, resolved_wires, request, authorization.fast
    )
    require_fast_path(provider_request, authorization)
    if disclosures:
        record_admission_coercions(accounting, authorization, disclosures)
    provider_request = with_cache_affinity(provider_request, authorization)
    resolved_wires = bind_recovery_profiles(
        route.deployments,
        resolved_wires,
        authorization.organization_id,
        accounting.recovery_host,
        request_region=provider_request.inference_geo,
    )
    route, resolved_wires = _prefer_cache_capable_rungs(route, resolved_wires, provider_request)
    scheduled_count = len(route.deployments)
    route, resolved_wires, placement = _affinity_ordered_rungs(
        route,
        resolved_wires,
        provider_request,
        accounting=accounting,
        authorization=authorization,
        continuation=continuation,
    )
    if len(route.deployments) != scheduled_count and request_carries_cache_markers(
        provider_request
    ):
        # Only disclosures are projected after recovery narrows the route. Keep
        # already-shaped provider input and its tool identities exactly frozen.
        selected_public, _ = route_generation_parameter_requests(
            tuple(profile for profile, _client in resolved_wires), public_request
        )
        public_request = public_request.model_copy(
            update={
                "ignored_parameters": tuple(
                    dict.fromkeys(
                        (
                            *public_request.ignored_parameters,
                            *(
                                item
                                for item in selected_public.ignored_parameters
                                if item.endswith(CACHE_CONTROL_NOT_FORWARDED_SUFFIX)
                            ),
                        )
                    )
                )
            }
        )
    # Every surviving rung failover-only would leave nothing to dial first:
    # fail closed here, named, instead of exhausting a ladder that dialed nothing.
    require_unrestricted_rung(route)
    public_request = disclose_best_effort_caching(
        public_request, route.deployments, [profile for profile, _ in resolved_wires], authorization
    )
    return route, resolved_wires, public_request, provider_request, placement


def select_single_route_before_search(
    route: GatewayRoute,
    resolved_wires: _ResolvedWires,
    request: GatewayRequest,
    *,
    accounting: NativeAttemptAccounting,
    authorization: AuthorizationSnapshot,
    continuation: ContinuationContext | None,
) -> tuple[GatewayRoute, _ResolvedWires, AffinityPlacement | None]:
    """Select a no-fallback route before search can perform work or drop native tools.

    A pure preview removes web-search carriers and shapes emulated tool discovery
    solely for capability eligibility. Its disclosures never escape and it calls no
    backend. After filtering and the normal route ordering, only the winner reaches
    effectful search planning with the original request. Final admission runs once
    on that singleton, recording only its actual coercions.
    """
    routing = None if request.gateway is None else request.gateway.routing
    if routing is None or routing.allow_fallbacks:
        return route, resolved_wires, None
    if route.snapshot.model_stages:
        root_indexes = tuple(
            index
            for index in range(len(route.deployments))
            if route.snapshot.stage_for_depth(index).exact_model_id == route.snapshot.exact_model_id
        )
        if not root_indexes:
            raise GatewayRoutingError("no-fallback routing requires an eligible root-model route")
        route = select_route_deployments(route, root_indexes)
        resolved_wires = tuple(resolved_wires[index] for index in root_indexes)
    if route.resolved_route_id is not None:
        return restrict_fallbacks(request, route), resolved_wires[:1], None
    preview = fast_request(without_prompt_caching(request, authorization), authorization)
    preview = strip_search_carriers(preview) if preview.web_search is not None else preview
    preview = plan_tool_search(preview, [profile.dialect for profile, _ in resolved_wires]).request
    route, resolved_wires, _, prepared, _ = prepare_route_requests(
        route, resolved_wires, preview, authorization.fast
    )
    route, resolved_wires = _prefer_cache_capable_rungs(route, resolved_wires, prepared)
    route, resolved_wires, placement = _affinity_ordered_rungs(
        route,
        resolved_wires,
        prepared,
        accounting=accounting,
        authorization=authorization,
        continuation=continuation,
    )
    require_unrestricted_rung(route)
    restricted = restrict_fallbacks(request, route)
    index = route.deployments.index(restricted.deployment)
    return restricted, (resolved_wires[index],), placement


PRIORITY_TIER_UNAVAILABLE_DISCLOSURE = "service_tier->default(priority_unavailable)"
"""Disclosure recorded when no rung offers the priority tier and the request runs at standard."""


def prepare_route_requests(
    route: GatewayRoute,
    resolved_wires: _ResolvedWires,
    request: GatewayRequest,
    fast: bool = False,
) -> tuple[GatewayRoute, _ResolvedWires, GatewayRequest, GatewayRequest, tuple[str, ...]]:
    """Pure capability filtering and shaping, with disclosures returned but never recorded.

    This is the shared selection primitive for final admission and no-fallback
    preselection. It executes no provider or search work and mutates no accounting,
    affinity, health or request state.
    """
    require_native_model_stage_contract(route)
    # Priced house tiers require a card for that exact tier; BYOK needs only
    # wire support (``fast``, our card, never takes a BYOK rung). Explicit
    # priority keeps those candidates ahead of standard lanes. Flex and
    # BYOK-only routes retain their disclosed coercion policy.
    # Narrow impossible context windows before reserving or calling a provider.
    chat_indexes = tuple(
        index
        for index, (profile, _client) in enumerate(resolved_wires)
        if profile.dialect != "typesafe_systemone"
    )
    if not chat_indexes:
        raise ProviderCapabilityError(
            capability="completions",
            detail="This model serves typed decisions. Send state and questions to /v1/systemone.",
        )
    if len(chat_indexes) != len(route.deployments):
        route = select_route_deployments(route, chat_indexes)
        resolved_wires = tuple(resolved_wires[index] for index in chat_indexes)
    request, admission_disclosures = prompt_size.clamp_output_to_context_window(route, request)
    window_indexes = prompt_size.context_window_compatible_indexes(route, request)
    if len(window_indexes) != len(route.deployments):
        route = select_route_deployments(route, window_indexes)
        resolved_wires = tuple(resolved_wires[index] for index in window_indexes)
    route, resolved_wires, capable = narrow_to_capable_rungs(route, resolved_wires, request)
    admission_disclosures = (*admission_disclosures, *capable)
    if request.service_tier in ("flex", "priority"):
        tier = request.service_tier
        tier_indexes = tuple(
            index
            for index, (profile, _client) in enumerate(resolved_wires)
            if profile.forwards_tier(tier) and not (fast and profile.billing_customer_managed)
        )
        if not tier_indexes and tier == "priority":
            # Priority unavailable dispatches at standard with disclosure (what
            # OpenAI serves; never costs more; ``:fast`` refuses later). Flex
            # asks for a CHEAPER tier, so it keeps the typed refusal below.
            request = request.model_copy(update={"service_tier": None})
            admission_disclosures = (*admission_disclosures, PRIORITY_TIER_UNAVAILABLE_DISCLOSURE)
        elif not tier_indexes:
            raise ProviderCapabilityError(
                capability="service_tier",
                detail=(
                    "This model does not offer a flex processing tier. "
                    "Remove service_tier, or choose a model with tiered pricing enabled."
                ),
            )

        if tier == "priority" and any(
            not resolved_wires[i][0].billing_customer_managed for i in tier_indexes
        ):
            route = select_route_deployments(route, tier_indexes)
            resolved_wires = tuple(resolved_wires[index] for index in tier_indexes)
    # Rung prompt caps after window, capability and tier narrowing (prompt_cap_indexes).
    if (uncapped := prompt_size.prompt_cap_indexes(route, request)) is not None:
        route = select_route_deployments(route, uncapped)
        resolved_wires = tuple(resolved_wires[index] for index in uncapped)

    admitted_request = request
    coercion_disclosures: tuple[str, ...] = admission_disclosures
    full_route = route
    full_wires = resolved_wires

    def candidate_serves(candidate: GatewayRequest) -> bool:
        return _candidate_serves(full_route, full_wires, candidate)

    try:
        compatible_indexes = compatible_generation_parameter_profile_indexes(
            tuple(profile for profile, _client in resolved_wires),
            admitted_request,
        )
    except ProviderParameterError:
        # No rung preserves the request verbatim; retry once with the
        # minimal disclosed coercion when semantics allow, otherwise
        # keep the named rejection.
        coercion = coerce_generation_parameters(
            tuple(profile for profile, _client in resolved_wires),
            admitted_request,
            admits=candidate_serves,
        )
        if coercion is None:
            # No coercion SERVES, but one may still APPLY: the probe refused
            # every candidate because the coerced request dies one layer
            # later, on a blocker that has nothing to do with the coerced
            # field. Re-raising the verbatim rejection here would name that
            # field (an image on a text-only route read as "thinking is not
            # supported by this model route", because shaping rejects the
            # foreign-wire thinking config before preflight ever sees the
            # image; 25 such 400s on 2026-09-11). Carry the unprobed
            # coercion forward instead and let the stage that actually
            # refuses it name the caller's remedy. Nothing is recorded for a
            # request that never serves.
            coercion = coerce_generation_parameters(
                tuple(profile for profile, _client in resolved_wires),
                admitted_request,
            )
        if coercion is None:
            raise
        compatible_indexes = compatible_generation_parameter_profile_indexes(
            tuple(profile for profile, _client in resolved_wires),
            coercion.request,
        )
        admitted_request = coercion.request
        coercion_disclosures = (*coercion_disclosures, *coercion.disclosures)
    route = select_route_deployments(route, compatible_indexes)
    resolved_wires = tuple(resolved_wires[index] for index in compatible_indexes)
    if route.resolved_route_id is None and route.reasoning_pinned_deployment_id is None:
        # Plaintext-reasoning carriers lead; the rungs that drop it stay as fallbacks.
        order = plaintext_reasoning_lead_order(
            tuple(profile for profile, _client in resolved_wires),
            admitted_request,
            [route.snapshot.stage_for_depth(i).stage_index for i in range(len(resolved_wires))],
        )
        route = reorder_route_deployments(route, order)
        resolved_wires = tuple(resolved_wires[index] for index in order)
    # The headroom rule reads the rungs that SURVIVED narrowing: a rung with
    # no reasoning default (or no ``none`` tier) that narrowing has already
    # removed must not veto the coercion for the default-on rung that will
    # actually serve. Every surviving rung accepted the request verbatim and
    # offers ``none``, so the coerced request narrows to the same set.
    headroom = reserve_thinking_headroom(
        tuple(profile for profile, _client in resolved_wires), admitted_request
    )
    if headroom is not None:
        admitted_request = headroom.request
        coercion_disclosures = (*coercion_disclosures, *headroom.disclosures)
    public_request, provider_request = route_generation_parameter_requests(
        tuple(profile for profile, _client in resolved_wires),
        admitted_request,
    )
    # Re-check after semantic shaping. Output ceilings stay caller-owned;
    # omitted wire-required caps are resolved independently at payload build.
    shaped_indexes = prompt_size.context_window_compatible_indexes(route, provider_request)
    if len(shaped_indexes) != len(route.deployments):
        route = select_route_deployments(route, shaped_indexes)
        resolved_wires = tuple(resolved_wires[index] for index in shaped_indexes)
        public_request, provider_request = route_generation_parameter_requests(
            tuple(profile for profile, _client in resolved_wires),
            admitted_request,
        )
    provider_request = provider_request.model_copy(update={"stream": True, "include_usage": True})
    protocol_indexes, protocol_errors = protocol_compatible_indexes(
        route,
        resolved_wires,
        provider_request,
        public_stream=public_request.stream,
    )
    if not protocol_indexes:
        # Degrade once with disclosure where the rejection set allows it:
        # a unanimous capability rejection coerces any coercible capability,
        # mixed rejections only the service-tier hint.
        coercion = coerce_route_rejections(
            protocol_errors, len(route.deployments), admitted_request
        )
        if coercion is not None:
            admitted_request = coercion.request
            coercion_disclosures = (*coercion_disclosures, *coercion.disclosures)
            public_request, provider_request = route_generation_parameter_requests(
                tuple(profile for profile, _client in resolved_wires),
                admitted_request,
            )
            provider_request = provider_request.model_copy(
                update={"stream": True, "include_usage": True}
            )
            protocol_indexes, protocol_errors = protocol_compatible_indexes(
                route,
                resolved_wires,
                provider_request,
                public_stream=public_request.stream,
            )
    if not protocol_indexes and provider_request.parallel_tool_calls is not None:
        # LAST resort for parallel_tool_calls: no rung honours the control
        # natively (and no other coercion freed one), so admit the rungs whose
        # only objection is that control. The data plane then drops `true`
        # (the provider's default) or serializes `false` per rung, disclosed
        # (native_bridge's per-rung shaping). A native rung is always
        # preferred, which is why this pass runs after everything else.
        protocol_indexes, protocol_errors = protocol_compatible_indexes(
            route,
            resolved_wires,
            provider_request,
            public_stream=public_request.stream,
            emulate_parallel_tool_calls=True,
        )
    if not protocol_indexes:
        if not protocol_errors:
            raise GatewayRoutingError("authorized route has no compatible deployment")
        # Nothing coercible remains; the shared admit handler scopes
        # capability rejections to their exact public request field.
        raise route_rejection(protocol_errors)
    if len(protocol_indexes) != len(route.deployments):
        selected_indexes = tuple(protocol_indexes)
        route = select_route_deployments(route, selected_indexes)
        resolved_wires = tuple(resolved_wires[index] for index in selected_indexes)
        public_request, provider_request = route_generation_parameter_requests(
            tuple(profile for profile, _client in resolved_wires),
            admitted_request,
        )
        provider_request = provider_request.model_copy(
            update={"stream": True, "include_usage": True}
        )
    # The surviving rungs decide whether the structured-output schema and the
    # strict tool schemas need their objects closed; a route that lost every
    # Anthropic rung above is dispatched with the caller's schemas verbatim.
    surviving_profiles = tuple(profile for profile, _client in resolved_wires)
    for coerce_schema in (coerce_structured_text_schema, coerce_strict_tool_schemas):
        coercion = coerce_schema(surviving_profiles, admitted_request)
        if coercion is None:
            continue
        admitted_request = coercion.request
        coercion_disclosures = (*coercion_disclosures, *coercion.disclosures)
        public_request, provider_request = route_generation_parameter_requests(
            surviving_profiles,
            admitted_request,
        )
        provider_request = provider_request.model_copy(
            update={"stream": True, "include_usage": True}
        )
    if coercion_disclosures:
        public_request = public_request.model_copy(
            update={
                "ignored_parameters": tuple(
                    dict.fromkeys((*public_request.ignored_parameters, *coercion_disclosures))
                )
            }
        )
    return route, resolved_wires, public_request, provider_request, coercion_disclosures


def route_rejection(
    errors: Sequence[ProviderParameterError | ProviderCapabilityError],
) -> ProviderParameterError | ProviderCapabilityError:
    """Choose the one rejection the caller can act on when no rung serves.

    Rungs decline for their own reasons, and the first rung's reason is not
    always the caller's remedy. A ladder whose text-only rung refuses any
    image while an inline-only rung refuses just the remote URL can still
    carry the picture: the caller inlines the bytes. Reporting the text-only
    rung's refusal would tell them to drop the image instead. The URL
    rejection therefore wins whenever some rung raised it; otherwise the
    first rung's own rejection stays the answer. A provider-scoped media
    handle is the same story one step further: the rejection that names the
    provider holding the upload beats a rung that merely declares no handle
    support, since only the named provider can ever resolve the handle.

    Args:
        errors: One rejection per declined deployment, in route order.

    Returns:
        The rejection to surface to the caller.
    """
    for preferred in ("media_handle_provider", "image_url_input"):
        for error in errors:
            if isinstance(error, ProviderCapabilityError) and error.capability == preferred:
                return error
    return errors[0]


def _prefer_cache_capable_rungs(
    route: GatewayRoute,
    resolved_wires: _ResolvedWires,
    provider_request: GatewayRequest,
) -> tuple[GatewayRoute, _ResolvedWires]:
    """Dispatch marker-honoring rungs first on cache-preserving pools.

    Under ``maximize_cache`` a cache-marked request must never start on a
    wire that structurally drops its markers while a marker-honoring rung
    stands ready: the pool's whole policy is prefix-cache preservation, and
    a marker-dropping first rung silently bills every turn's full context
    uncached (measured ~10x on a large system prompt). The reorder is
    stable per group (house OpenRouter never leads; an order it leaves unmoved stands), and a
    route that narrowing left with NO marker-honoring rung is unchanged
    here: the dropped markers are already disclosed through the
    ``cache_control`` ``ignored_parameters`` entries. ``maximize_availability``
    pools keep their certified order untouched.
    """
    if _keeps_issuing_rung_first(route):
        return route, resolved_wires
    if route.snapshot.model_stages:
        # The stage scheduler ranks markers once, after stage-local affinity.
        return route, resolved_wires
    if route.snapshot.failover_mode != "maximize_cache":
        return route, resolved_wires
    if len(resolved_wires) < 2 or not request_carries_cache_markers(provider_request):
        return route, resolved_wires
    marker_capable = tuple(
        i for i, (w, _) in enumerate(resolved_wires) if honors_markers(w, provider_request)
    )
    if not marker_capable or len(marker_capable) == len(resolved_wires):
        return route, resolved_wires
    order = (*marker_capable, *(i for i in range(len(resolved_wires)) if i not in marker_capable))
    if order == tuple(range(len(resolved_wires))):
        return route, resolved_wires
    order = house_aggregator_last(order, route.deployments, [p for p, _ in resolved_wires])
    return (
        reorder_route_deployments(route, order),
        tuple(resolved_wires[index] for index in order),
    )


def _keeps_issuing_rung_first(route: GatewayRoute) -> bool:
    """Whether a reasoning continuation's issuing rung is on the route and must lead it.

    The issuing rung alone can replay the request's thinking, so while it is
    still dispatchable no cache-marker or affinity reorder may demote it and
    its fallbacks stay in pool order. Once admission has narrowed it out as
    dead the pin is stale: every surviving rung runs without the reasoning,
    so the pool's normal ordering applies to them.
    """
    if route.resolved_route_id is not None:
        return True
    pinned = route.reasoning_pinned_deployment_id
    return pinned is not None and any(
        deployment.deployment_id == pinned for deployment in route.deployments
    )


def _affinity_ordered_rungs(
    route: GatewayRoute,
    resolved_wires: _ResolvedWires,
    provider_request: GatewayRequest,
    *,
    accounting: NativeAttemptAccounting,
    authorization: AuthorizationSnapshot,
    continuation: ContinuationContext | None,
) -> tuple[GatewayRoute, _ResolvedWires, AffinityPlacement]:
    """Dispatch rungs in sticky-then-rendezvous order on affinity pools.

    Under ``maximize_cache_affinity`` the certified order is replaced by the
    request fingerprint's rendezvous permutation over the surviving rungs, so
    every worker sends one conversation to the same rung and, when that rung
    sheds or dies, to the same deterministic alternate. Weights come from each
    deployment's authored ``GatewayRungDispatchPolicy.affinity_weight``
    (default 1.0). A live worker-local sticky binding is honored AHEAD of
    rendezvous order (its rung holds the conversation's warm cache after a
    spill), except when its rung is suppressed (throttled or circuit-open)
    right now, in which case the binding is cleared so stickiness can never
    pin a conversation to a dead lane. The cache-marker guarantee composes: a
    cache-marked request on a route mixing marker-honoring and marker-dropping
    wires still dispatches the marker-honoring group first, ordered within
    each group. The other failover modes are untouched; prompt caching off keeps
    the certified order. A ``:fast`` request is placed like any other: its
    dedicated replicas keep their prefix cache warm per conversation.
    """
    if authorization.prompt_caching_disabled:
        return route, resolved_wires, AffinityPlacement()
    if route.snapshot.model_stages or accounting.recovery_host is not None:
        return stage_affinity_ordered_rungs(
            route,
            resolved_wires,
            provider_request,
            accounting=accounting,
            authorization=authorization,
            continuation=continuation,
        )
    if route.snapshot.failover_mode != "maximize_cache_affinity":
        return route, resolved_wires, AffinityPlacement()
    material = affinity_seed_material(
        provider_request,
        continuation_episode_key=None if continuation is None else continuation.episode_key,
        request_id=authorization.request_id,
    )
    fingerprint = affinity_fingerprint(
        organization_id=authorization.organization_id,
        identity_id=authorization.identity_id,
        material=material,
    )
    if len(resolved_wires) < 2 or _keeps_issuing_rung_first(route):
        return route, resolved_wires, AffinityPlacement(fingerprint=fingerprint)
    order = rendezvous_order(fingerprint, affinity_weighted_rungs(route.deployments))
    order, sticky_index, sticky_deployment_id = sticky_first_order(
        order,
        route,
        fingerprint=fingerprint,
        sticky=accounting.sticky,
        health=accounting.health,
        authorization=authorization,
    )
    if request_carries_cache_markers(provider_request):
        marker_capable = frozenset(
            index
            for index, (profile, _client) in enumerate(resolved_wires)
            if honors_markers(profile, provider_request)
        )
        if marker_capable and len(marker_capable) < len(resolved_wires):
            order = (
                *(index for index in order if index in marker_capable),
                *(index for index in order if index not in marker_capable),
            )
    # Last, so neither a sticky binding nor the marker grouping can lead with it.
    order = house_aggregator_last(order, route.deployments, [p for p, _ in resolved_wires])
    placement = AffinityPlacement(
        fingerprint=fingerprint,
        sticky_preferred=sticky_index is not None and order[0] == sticky_index,
        sticky_deployment_id=sticky_deployment_id,
    )
    return (
        reorder_route_deployments(route, order),
        tuple(resolved_wires[index] for index in order),
        placement,
    )


def _candidate_serves(
    route: GatewayRoute,
    resolved_wires: _ResolvedWires,
    candidate: GatewayRequest,
) -> bool:
    """Probe the full admission pipeline for one coercion candidate.

    The policy layer sees only wire profiles, so without this probe a
    candidate could pass generation narrowing yet land on rungs that all fail
    deployment capability preflight, blocking a farther candidate whose rungs
    serve. The probe mirrors the real pipeline exactly, including the single
    capability coercion admission may run afterwards: clearing one capability
    can merely expose the next, so the coerced candidate must itself pass
    preflight before the snap counts as servable.

    Args:
        route: Frozen full route aligned with ``resolved_wires``.
        resolved_wires: Ordered wire profiles and clients per deployment.
        candidate: One coercion candidate request.

    Returns:
        Whether admission would serve the candidate.
    """
    try:
        candidate_indexes = compatible_generation_parameter_profile_indexes(
            tuple(profile for profile, _client in resolved_wires),
            candidate,
        )
        candidate_route = select_route_deployments(route, candidate_indexes)
        candidate_wires = tuple(resolved_wires[index] for index in candidate_indexes)
        candidate_public, candidate_provider = route_generation_parameter_requests(
            tuple(profile for profile, _client in candidate_wires),
            candidate,
        )
    except (ProviderParameterError, ProviderCapabilityError):
        return False
    candidate_provider = candidate_provider.model_copy(
        update={"stream": True, "include_usage": True}
    )
    indexes, errors = protocol_compatible_indexes(
        candidate_route,
        candidate_wires,
        candidate_provider,
        public_stream=candidate_public.stream,
    )
    if indexes:
        return True
    capability_coercion = coerce_route_rejections(
        errors, len(candidate_route.deployments), candidate
    )
    if capability_coercion is None:
        return False
    try:
        _coerced_public, coerced_provider = route_generation_parameter_requests(
            tuple(profile for profile, _client in candidate_wires),
            capability_coercion.request,
        )
    except (ProviderParameterError, ProviderCapabilityError):
        return False
    coerced_provider = coerced_provider.model_copy(update={"stream": True, "include_usage": True})
    coerced_indexes, _coerced_errors = protocol_compatible_indexes(
        candidate_route,
        candidate_wires,
        coerced_provider,
        public_stream=candidate_public.stream,
    )
    return bool(coerced_indexes)


def protocol_compatible_indexes(
    route: GatewayRoute,
    resolved_wires: _ResolvedWires,
    provider_request: GatewayRequest,
    *,
    public_stream: bool | None,
    emulate_parallel_tool_calls: bool = False,
) -> tuple[tuple[int, ...], tuple[ProviderParameterError | ProviderCapabilityError, ...]]:
    """Select rungs that pass capability preflight and payload build.

    Args:
        route: Frozen route aligned with ``resolved_wires``.
        resolved_wires: Ordered wire profiles and clients per deployment.
        provider_request: Streaming-forced request to validate.
        public_stream: The caller's declared streaming intent.

    Returns:
        Ordered compatible indexes and every rung's rejection in route
        order, so the caller can distinguish a route-wide capability gap
        from rungs declining for different reasons. A payload builder that
        cannot represent the conversation as sent (for example an assistant
        turn with neither text nor a tool call) counts as a rung rejection
        on ``messages``, so a caller-shaped transcript is refused with a
        field-specific 400 instead of escaping admission as an internal
        failure.
    """
    indexes: list[int] = []
    errors: list[ProviderParameterError | ProviderCapabilityError] = []
    for index, (deployment, (profile, _client)) in enumerate(
        zip(route.deployments, resolved_wires, strict=True)
    ):
        # Each rung is probed with the request IT would dispatch: a
        # reasoning-pinned route's fallback rung sees the request without the
        # pinned provider's sealed reasoning, exactly as the dispatch build
        # shapes it, so the builder's foreign-block rejection never narrows a
        # legitimate failover rung out of the ladder.
        rung_request = rung_provider_request(route, deployment, provider_request)
        try:
            capabilities = deployment.capabilities
            rung_request, _bound = bounded_output_request(
                profile,
                rung_request,
                model_maximum_output_tokens=(
                    capabilities.maximum_output_tokens if capabilities is not None else None
                ),
                context_window_tokens=(
                    capabilities.context_window_tokens if capabilities is not None else None
                ),
            )
            preflight_gateway_request(
                rung_request,
                deployment.gateway.capabilities,
                model_capabilities=deployment.capabilities,
                public_stream=public_stream,
                route_provider=deployment.provider,
                emulated_capabilities=emulated_gateway_capabilities(
                    profile.dialect, emulate_parallel_tool_calls=emulate_parallel_tool_calls
                ),
            )
            image_aware_stream_payload(profile, rung_request, capabilities, deployment.provider)
        except (ProviderParameterError, ProviderCapabilityError) as exc:
            errors.append(exc)
            continue
        except ProviderResponseError as exc:
            errors.append(_unrepresentable_messages_error(exc))
            continue
        indexes.append(index)
    return tuple(indexes), tuple(errors)


def _unrepresentable_messages_error(exc: ProviderResponseError) -> ProviderParameterError:
    """Turn a payload builder's transcript rejection into a ``messages`` parameter error.

    Args:
        exc: The builder's gateway-authored reason the conversation cannot be
            encoded on this wire.

    Returns:
        A field-specific pre-dispatch rejection the shared admit handler maps
        to a caller-facing invalid request.
    """
    return ProviderParameterError(
        message=(
            f"This model route cannot encode the conversation as sent: {exc}. "
            "Fix the message history or choose a different model."
        ),
        param="messages",
        code="invalid_parameter",
    )


def shape_parallel_tool_calls(
    request: GatewayRequest,
    capabilities: GatewayDeploymentCapabilities,
) -> tuple[GatewayRequest, str | None]:
    """Shape ``parallel_tool_calls`` for one rung that may lack the control.

    A rung whose wire carries the control forwards it verbatim. One that does
    not gets ``true`` dropped (parallel calls are the provider's own default)
    or ``false`` emulated: the data plane serializes that rung's stream to one
    tool call per turn (``serialize_tool_calls``). Either way the caller reads
    the disclosure in ``ignored_parameters``.

    Args:
        request: The streaming-forced provider request.
        capabilities: The rung's deployment capability declaration.

    Returns:
        The request to build this rung's payload from, and the disclosure to
        publish (``None`` when nothing changed).
    """
    if request.parallel_tool_calls is None or capabilities.supports_parallel_tool_calls:
        return request, None
    if request.parallel_tool_calls:
        return (
            request.model_copy(update={"parallel_tool_calls": None}),
            "parallel_tool_calls->dropped(provider_default)",
        )
    return (
        request.model_copy(update={"parallel_tool_calls": None, "serialize_tool_calls": True}),
        "parallel_tool_calls->emulated(serialized_by_gateway)",
    )


def fold_parallel_tool_call_disclosures(
    public_request: GatewayRequest,
    disclosures: set[str],
    *,
    accounting: NativeAttemptAccounting,
    authorization: AuthorizationSnapshot,
) -> GatewayRequest:
    """Publish per-rung parallel-tool shaping like any other admission coercion.

    Args:
        public_request: The public request the admission answer carries.
        disclosures: Distinct disclosures the per-rung shaping produced.
        accounting: Shared accounting owning the coercion counter.
        authorization: Frozen authority for the accepted request.

    Returns:
        The public request with the disclosures folded into
        ``ignored_parameters`` (unchanged when there are none).
    """
    if not disclosures:
        return public_request
    ordered = tuple(sorted(disclosures))
    record_admission_coercions(accounting, authorization, ordered)
    return public_request.model_copy(
        update={
            "ignored_parameters": tuple(
                dict.fromkeys((*public_request.ignored_parameters, *ordered))
            )
        }
    )


def record_admission_coercions(
    accounting: NativeAttemptAccounting,
    authorization: AuthorizationSnapshot,
    disclosures: tuple[str, ...],
) -> None:
    """Log and count one admission's disclosed request coercions.

    A coercion is never silent: the caller sees it in
    ``ignored_parameters``, the log names it for operators, and the
    metrics snapshot counts it so a persistently coerced alias reaches a
    human instead of quietly serving degraded semantics forever.

    Args:
        accounting: Shared accounting owning the coercion counter.
        authorization: Frozen authority for the accepted request.
        disclosures: Path->effective disclosure strings applied.
    """
    accounting.record_admission_coercions(len(disclosures))
    _logger.warning(
        "gateway admission coerced request semantics for alias %r: %s "
        "(disclosed through ignored_parameters)",
        authorization.alias,
        ", ".join(disclosures),
    )


def resolve_admission_route(
    components: NativeGatewayComponents,
    authorization: AuthorizationSnapshot,
    request: GatewayRequest,
    *,
    continuation: ContinuationContext | None = None,
) -> GatewayRoute:
    """Resolve one direct or project route without an event loop.

    Direct pools resolve entirely inside frozen in-memory catalogs. Project
    targets run frozen learned selection synchronously on this worker thread
    through the shared selection seam and episode identity derivation, so
    there is exactly one policy execution path. A Responses continuation
    carries its original turn's episode key, so a continued request joins the
    same selection episode instead of re-running request-time embedding for a
    fresh one. Request-time embedding failure falls back to the frozen
    conservative baseline inside the shared runtime, and neither path mutates
    policy or evidence.
    """
    if isinstance(authorization.target, DirectTarget):
        return components.routes.resolve_direct(authorization)
    if continuation is not None:
        episode = (
            authorization.organization_id,
            authorization.identity_id,
            authorization.alias_revision_id,
            continuation.episode_key,
        )
    else:
        episode = episode_namespace(
            namespace=ProtocolNamespace(
                organization_id=authorization.organization_id,
                identity_id=authorization.identity_id,
                alias_revision_id=authorization.alias_revision_id,
            ),
            # The session-scoped correlation id is the stronger affinity
            # scope; a per-operation idempotency key only pins retries.
            caller_episode_key=request.client_request_id or request.idempotency_key,
            request_id=authorization.request_id,
        )
    return components.routes.resolve_project_blocking(
        authorization=authorization,
        request=request,
        episode_namespace=episode,
    )


def record_dead_admission_rungs(
    accounting: NativeAttemptAccounting,
    authorization: AuthorizationSnapshot,
    dead: tuple[DeadRung, ...],
    *,
    fallback_available: bool,
) -> None:
    """Record admission-dead rungs and surface a lead masked by fallback."""
    if not dead:
        return
    for rung in dead:
        accounting.health.failed(
            deployment_health_key(authorization, rung.deployment),
            rung.failure,
        )
    lead_masked = fallback_available and any(rung.index == 0 for rung in dead)
    accounting.record_admission_rung_skips(len(dead), lead_skipped=lead_masked)
    log_dead_rungs(authorization, dead, fallback_available=fallback_available)


def log_reasoning_continuation_rejection(
    authorization: AuthorizationSnapshot, stage: str, reason: object
) -> None:
    """Record why a reasoning-carrier continuation failed, for operators only.

    The caller sees one opaque 400 (naming the differing bound claim would be an
    authentic-continuation oracle), but an operator needs the exact reason to tell
    a genuine tamper from a benign authority drift. Nothing here carries a
    credential or the plaintext reasoning; the catalog-generation fields make a
    cross-worker or post-republish drift obvious when diffed against the issuing
    turn's admission log.
    """
    _logger.warning(
        "reasoning carrier continuation rejected",
        extra={
            "operation": "native_reasoning_continuation",
            "stage": stage,
            "reason": str(reason),
            "request_id": authorization.request_id,
            "alias": authorization.alias,
            "alias_revision_id": authorization.alias_revision_id,
            "catalog_sha256": authorization.catalog_sha256,
        },
    )
