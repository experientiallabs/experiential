"""Which rungs may take a request a provider's own content filter refused.

Azure's content management system (and other account-side moderation layers)
can refuse a request, or the output it would have returned, that another
provider serving the same model applies no filter to. The data plane files
that verdict as a ``REFUSAL`` with ``provider_content_filter`` and makes it
failover-eligible once per request, only while nothing reached the caller; a
model declining in its own words and the gateway's own guardrail class are
never marked. The control plane then admits only an untried rung served by a
DIFFERENT provider: the same provider's filter (another deployment of it
included) would refuse the same content, so it is never dialed for it, and
the caller receives the refusal unchanged when no other provider serves the
alias. The walk stays inside the request's own route, so it can never move a
request to another model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.native_fallback_rules import FallbackRules

if TYPE_CHECKING:
    from exp.runtime.gateway.routing import GatewayRoute


def route_providers(route: GatewayRoute) -> tuple[str, ...]:
    """Return each depth's provider, in route order.

    Args:
        route: The admitted route.

    Returns:
        One provider id per route depth.
    """
    return tuple(deployment.provider for deployment in route.deployments)


def other_providers(
    depths: tuple[int, ...],
    failure: GatewayFailure,
    current_depth: int,
    providers: Sequence[str],
    refusal_failover: bool,
    rules: FallbackRules = (),
) -> tuple[int, ...]:
    """Keep the depths a provider content-filter refusal may advance to.

    Every walk that chooses a successor for the refusal (the candidate choice,
    and the failover pass whose sideways spill and forced pass follow a shed)
    reads this one filter. An alias revision that opted into refusal failover
    keeps its own unrestricted walk, and a ``failover_only_on`` rung that
    ``untried_depths`` matched for the refusal keeps its own opt-in.

    Args:
        depths: The untried depths the failure may advance to, in route order.
        failure: The failure that ended the previous dispatch.
        current_depth: Route position of the refusing rung.
        providers: Each depth's provider, or empty when the route carries
            no provider facts.
        refusal_failover: Whether the alias revision opted every typed
            refusal into failover.
        rules: Each depth's ``failover_only_on`` set (``None`` for an
            unrestricted rung); empty when no rung authored one.

    Returns:
        The depths, in route order, served by a provider other than the
        refusing rung's; ``depths`` unchanged for any other failure.
    """
    if (
        not failure.provider_content_filter
        or failure.failure_class != GatewayFailureClass.REFUSAL
        or refusal_failover
        or not providers
    ):
        return depths
    refusing = providers[current_depth]
    return tuple(
        depth
        for depth in depths
        if providers[depth] != refusing or (rules and rules[depth] is not None)
    )
