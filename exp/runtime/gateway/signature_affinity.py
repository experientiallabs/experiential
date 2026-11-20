"""Route a Gemini continuation to the rung that minted its thought signature.

The Gemini API and Vertex AI each verify only the thought signatures they
issued. A conversation whose later turn routes to the other backend (latency
demotion, ordering, failover) has its signatures refused; the data plane then
replaces them with Google's bypass value, which keeps the turn alive but drops
the model's signed reasoning state, and long agent conversations repaired that
way end on a malformed function call far more often than ones kept on one
backend.

The data plane remembers, per worker and in memory only, which deployment
minted each signature it relayed (``exp_gateway_native.thought_signature_issuer``).
Admission looks up the newest replayed signature, records its issuer on the
request (``GatewayRequest.thought_signature_issuer``) so a host resolver can keep
that rung through speed demotion, and leads the admitted route with it. The
caller's own routing choices win: a named route, ``provider.order``, and a
reasoning-pinned continuation are left alone. A request with no remembered
issuer, or whose issuer is no longer in its route, routes and repairs exactly
as before.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import replace
from functools import cache

from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.native_execution import reorder_route_deployments
from exp.runtime.gateway.native_fallback_rules import rung_rules
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.gateway.sticky_affinity import AffinityPlacement

THOUGHT_SIGNATURE_BYPASS = "skip_thought_signature_validator"
"""Google's placeholder signature; it names no issuer."""

SIGNATURE_AFFINITY_DISCLOSURE = "thought_signature_issuer"
"""``fallback_reason`` token of a route led by its signature's issuing rung."""

type IssuerLookup = Callable[[str], str | None]


@cache
def native_issuer_lookup() -> IssuerLookup:
    """Return the data plane's in-memory signature issuer lookup."""
    native = importlib.import_module("exp_gateway_native")
    lookup: IssuerLookup = native.thought_signature_issuer
    return lookup


def newest_thought_signature(request: GatewayRequest) -> str | None:
    """Return the most recent replayed Gemini thought signature, if any."""
    for message in reversed(request.messages):
        for call in reversed(message.tool_calls):
            signature = call.provider_thought_signature
            if signature is not None and signature != THOUGHT_SIGNATURE_BYPASS:
                return signature
    return None


def with_thought_signature_issuer(
    request: GatewayRequest, lookup: IssuerLookup | None = None
) -> GatewayRequest:
    """Record the deployment that minted the request's newest signature.

    Args:
        request: Decoded request, after any stored continuation is applied.
        lookup: Signature to issuing deployment id; defaults to the data
            plane's memory (:func:`native_issuer_lookup`).

    Returns:
        The request with ``thought_signature_issuer`` set when known.
    """
    signature = newest_thought_signature(request)
    if signature is None:
        return request
    issuer = (lookup or native_issuer_lookup())(signature)
    if issuer is None or issuer == request.thought_signature_issuer:
        return request
    return request.model_copy(update={"thought_signature_issuer": issuer})


def lead[WireT](
    route: GatewayRoute,
    wires: tuple[WireT, ...],
    request: GatewayRequest,
    placement: AffinityPlacement,
) -> tuple[GatewayRoute, tuple[WireT, ...], AffinityPlacement]:
    """Move the signature's issuing rung to the front of the admitted route.

    Only within the leading model stage, and never against the caller's own
    routing (a named route, ``provider.order``) or a reasoning-pinned route.
    A ``failover_only_on`` rung is never moved: it may not take a first dial.
    The moved route discloses ``thought_signature_issuer``, and the placement
    no longer claims a sticky binding leads (``sticky_deployment_id`` still
    names the warm rung as cache evidence).

    Args:
        route: Admitted route after the engine's own scheduling.
        wires: Resolved wires in route order; reordered in step.
        request: Request carrying ``thought_signature_issuer``.
        placement: The admission's affinity placement facts.

    Returns:
        The route, wires and placement, led by the issuing rung when it applies.
    """
    issuer = request.thought_signature_issuer
    if (
        issuer is None
        or route.resolved_route_id is not None
        or route.reasoning_pinned_deployment_id is not None
        or (request.provider_preferences or {}).get("order")
    ):
        return route, wires, placement
    ids = [deployment.deployment_id for deployment in route.deployments]
    if issuer not in ids or ids.index(issuer) == 0:
        return route, wires, placement
    depth = ids.index(issuer)
    snapshot = route.snapshot
    if rung_rules(route.deployments[depth]) is not None or (
        snapshot.model_stages
        and snapshot.stage_for_depth(depth).stage_index != snapshot.stage_for_depth(0).stage_index
    ):
        return route, wires, placement
    order = (depth, *(index for index in range(len(ids)) if index != depth))
    reordered = reorder_route_deployments(route, order)
    disclosed = ",".join(
        token for token in (reordered.fallback_reason, SIGNATURE_AFFINITY_DISCLOSURE) if token
    )
    return (
        reordered.model_copy(update={"fallback_reason": disclosed}),
        tuple(wires[index] for index in order),
        replace(placement, sticky_preferred=False),
    )
