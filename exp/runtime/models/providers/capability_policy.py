"""Capability-preservation policy for one certified gateway route.

Admission prefers a rung that preserves every caller semantic verbatim; that
preference already exists in three layers (operational deadness skipping in
``native_execution.dispatchable_route_profiles``, per-rung generation-control
narrowing in ``generation_route_compat``, and the per-deployment capability
preflight plus payload build in the control plane's admit loop). This module
owns the step AFTER all three fail: the minimal COERCE-WITH-DISCLOSURE that
keeps a request servable when semantics allow; when they do not, the rung's
own field-scoped rejection stays the answer. A coercion is never silent: every substitution is
disclosed through ``ignored_parameters`` in ``path->effective`` form, logged,
and counted by the control plane's admission metrics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.models.providers.errors import (
    ProviderCapabilityError,
    ProviderParameterError,
)
from exp.runtime.models.providers.generation_parameter_validation import (
    profile_reasoning_efforts,
)
from exp.runtime.models.providers.generation_route_compat import (
    compatible_generation_parameter_profile_indexes,
)
from exp.runtime.models.providers.reasoning_compat import efforts_by_nearness

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
    # A heterogeneous waterfall can carry a nearby effort only on rungs that
    # reject some other control, so candidates are tried in nearness order
    # and the snap is the closest level that actually admits a rung.
    for candidate in efforts_by_nearness(request.reasoning_effort, ladder):
        snapped_request = request.model_copy(update={"reasoning_effort": candidate})
        try:
            compatible_generation_parameter_profile_indexes(profiles, snapped_request)
        except (ProviderParameterError, ProviderCapabilityError):
            continue
        return RequestCoercion(
            request=snapped_request,
            disclosures=(f"reasoning_effort->{candidate}",),
        )
    return None


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


def route_wide_capability(
    errors: Sequence[ProviderParameterError | ProviderCapabilityError],
    deployment_count: int,
) -> str | None:
    """Return the one capability EVERY route deployment rejected, if any.

    Deployments can decline different requirements; a capability coercion is
    honest only when a single capability was rejected by every rung, so mixed
    rejections keep the first rung's own field-specific error instead of
    degrading a field some rung could have preserved.

    Args:
        errors: One rejection per declined deployment, in route order.
        deployment_count: Number of deployments the route offered.

    Returns:
        The universally rejected capability literal, or ``None``.
    """
    if len(errors) != deployment_count:
        return None
    capabilities = {
        error.capability for error in errors if isinstance(error, ProviderCapabilityError)
    }
    if len(capabilities) == 1 and all(
        isinstance(error, ProviderCapabilityError) for error in errors
    ):
        return next(iter(capabilities))
    return None
