"""Which rungs may take a request a provider refused as longer than its window.

A provider's context-length 400 (``context_length_exceeded``, vLLM's "maximum
context length is N tokens") is the caller's error on a lane that serves the
window the model advertises, but a lane can serve less than that (a reseller
deploying a model at a fraction of its window). The data plane files such a
refusal with ``context_window_rejected`` and keeps it failover-eligible once
per request; the control plane then admits only an untried rung whose window
could fit, so the refusal reaches the caller unchanged when no rung could.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from exp.runtime.gateway.contracts import GatewayFailure

if TYPE_CHECKING:
    from exp.runtime.gateway.routing import GatewayRoute


def route_context_windows(route: GatewayRoute) -> tuple[int | None, ...]:
    """Return each depth's advertised context window (``None`` where unstated).

    Args:
        route: The admitted route.

    Returns:
        One window per route depth, in route order.
    """
    return tuple(
        None if deployment.capabilities is None else deployment.capabilities.context_window_tokens
        for deployment in route.deployments
    )


def wider_windows(
    depths: tuple[int, ...],
    failure: GatewayFailure,
    current_depth: int,
    context_windows: Sequence[int | None],
) -> tuple[int, ...]:
    """Keep the depths whose window could fit a request a narrower lane refused.

    Every walk that chooses a successor for the refusal (the candidate choice,
    and the failover pass whose sideways spill and forced pass follow a shed)
    reads this one filter. The bar is the window the rejecting provider
    stated, else the window the failed rung advertises. A rung whose
    advertised window is not larger than the bar would refuse the same
    request, so it is never dialed for it; a rung (or a bar) the catalog
    states no window for stays eligible, exactly as admission treats it.

    Args:
        depths: The untried depths the failure may advance to, in route order.
        failure: The provider's context-window rejection.
        current_depth: Route position of the rejecting rung.
        context_windows: Each depth's advertised window, or empty when the
            route carries no window facts.

    Returns:
        The depths, in route order, that may take the request; ``depths``
        unchanged for any failure that is not a context-window refusal.
    """
    if not failure.context_window_rejected or not context_windows:
        return depths
    bar = failure.context_window_limit_tokens or context_windows[current_depth]
    if bar is None:
        return depths
    return tuple(
        depth for depth in depths if (window := context_windows[depth]) is None or window > bar
    )
