"""Capability-preservation policy for one certified gateway route.

Admission prefers a rung that preserves every caller semantic verbatim; that
preference already exists in three layers (operational deadness skipping in
``native_execution.dispatchable_route_profiles``, per-rung generation-control
narrowing in ``generation_route_compat``, and the per-deployment capability
preflight plus payload build in the control plane's admit loop). This module
owns the step AFTER all three fail: the minimal COERCE-WITH-DISCLOSURE that
keeps a request servable when semantics allow, and the enriched fail-closed
message when they do not. A coercion is never silent: every substitution is
disclosed through ``ignored_parameters`` in ``path->effective`` form, logged,
and counted by the control plane's admission metrics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.models.providers.reasoning_compat import nearest_supported_effort
from exp.runtime.models.providers.generation_parameter_validation import profile_reasoning_efforts

if TYPE_CHECKING:
    from exp.runtime.models.providers.base import GatewayWireProfile

STRICT_TOOLS_DISCLOSURE = "tools.strict->false"
"""Disclosure recorded when strict tools degrade to best-effort schemas."""

EFFORT_DROP_DISCLOSURE = "reasoning_effort"
"""Disclosure recorded when a no-reasoning route drops an explicit 'none'."""


@dataclass(frozen=True)
class RequestCoercion:
    """One disclosed request substitution admission may retry with."""

    request: GatewayRequest
    disclosures: tuple[str, ...]


def coerce_generation_parameters(
    profiles: Sequence[GatewayWireProfile],
    request: GatewayRequest,
) -> RequestCoercion | None:
    """Build the minimal disclosed coercion after verbatim narrowing failed.

    Only substitutions whose semantics survive are offered: a reasoning
    effort snaps to the nearest level any rung supports on the canonical
    ladder (ties prefer the lower level, so a coercion never spends more
    than requested), and an explicit ``none`` on a route with no reasoning
    support at all is dropped, because a non-reasoning model already delivers
    exactly what ``none`` asks for. A request whose effort exceeds a
    zero-reasoning route stays a named rejection: deleting the feature is
    not a nearest level.

    Args:
        profiles: Ordered wire profiles for every live route deployment.
        request: Decoded public request that no rung accepted verbatim.

    Returns:
        The disclosed substitution to retry with, or ``None`` when nothing
        coercible applies.
    """
    if request.reasoning_effort is None:
        return None
    ladder: set[str] = set()
    for profile in profiles:
        ladder.update(profile_reasoning_efforts(profile))
    if not ladder:
        if request.reasoning_effort == "none":
            return RequestCoercion(
                request=request.model_copy(update={"reasoning_effort": None}),
                disclosures=(EFFORT_DROP_DISCLOSURE,),
            )
        return None
    if request.reasoning_effort in ladder:
        # The effort itself is portable; the verbatim failure lies elsewhere
        # and a snap would change semantics for nothing.
        return None
    snapped = nearest_supported_effort(request.reasoning_effort, ladder)
    if snapped is None:
        return None
    return RequestCoercion(
        request=request.model_copy(update={"reasoning_effort": snapped}),
        disclosures=(f"reasoning_effort->{snapped}",),
    )


def coerce_capability(capability: str, request: GatewayRequest) -> RequestCoercion | None:
    """Build the disclosed coercion for one preflight capability rejection.

    Strict tools are the one coercible capability: degrading ``strict: true``
    to best-effort schemas weakens a correctness guarantee, so it happens
    only here, after every rung declined the verbatim request, and only as a
    disclosed drop. Every other capability names a feature with no
    approximation and stays fail-closed.

    Args:
        capability: Stable capability literal from the preflight rejection.
        request: Decoded request no rung could preserve.

    Returns:
        The disclosed substitution to retry with, or ``None`` when the
        capability cannot be coerced.
    """
    if capability != "strict_tools" or not any(tool.strict for tool in request.tools):
        return None
    return RequestCoercion(
        request=request.model_copy(
            update={
                "tools": tuple(
                    tool.model_copy(update={"strict": False}) if tool.strict else tool
                    for tool in request.tools
                )
            }
        ),
        disclosures=(STRICT_TOOLS_DISCLOSURE,),
    )


def route_capability_failure_message(capability: str, deployment_count: int) -> str:
    """Name the capability no rung declares and how to resolve it.

    The engine sees only the resolved route, so alias-level alternatives are
    the catalog's job (see ``capability_parity``); this message states the
    exact gap so an operator can declare the capability or the caller can
    switch aliases.

    Args:
        capability: Stable capability literal from the final rejection.
        deployment_count: Number of live deployments that declined it.

    Returns:
        The sanitized fail-closed message.
    """
    deployments = "deployment" if deployment_count == 1 else "deployments"
    return (
        f"the requested capability '{capability}' is not declared by any of the "
        f"{deployment_count} {deployments} in this model route; choose an alias "
        "whose route supports it or remove the field"
    )
