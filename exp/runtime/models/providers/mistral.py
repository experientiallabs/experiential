"""Mistral's greedy-sampling rule on its own wire and every host that relays it.

Mistral refuses ``temperature: 0`` beside any ``top_p`` other than 1 with
"top_p must be 1 when using greedy sampling" (mistral-large-4, 2026-10-18:
nine 400s in one day). Greedy decoding always takes the single most likely
token, so the nucleus bound cannot change the answer: the gateway sends
``top_p: 1`` instead of refusing, and discloses it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from exp.runtime.gateway.contracts import GatewayRequest

if TYPE_CHECKING:
    from exp.runtime.models.providers.base import GatewayWireProfile

MISTRAL_GREEDY_TOP_P = "top_p->1(greedy_sampling)"
"""Disclosure recorded when ``top_p`` is sent as 1 because ``temperature`` is 0."""

_MISTRAL_FAMILY_TOKENS = ("mistral", "magistral", "codestral", "devstral", "ministral", "pixtral")
"""Tokens that name Mistral weights in every host's model id spelling."""


def is_mistral_model_id(model_id: str) -> bool:
    """Return whether one provider model identifier names a Mistral model.

    Hosts spell the weights as ``mistral-large-4`` (Mistral's own origin),
    ``mistralai/mistral-large-4`` (OpenRouter) or ``Mistral-Large-4`` (Azure),
    and the coding and reasoning lines carry their own family names, so the
    match is case-insensitive on those tokens and never on the host.
    """
    lowered = model_id.lower()
    return any(token in lowered for token in _MISTRAL_FAMILY_TOKENS)


def pin_greedy_top_p(
    profiles: Sequence[GatewayWireProfile],
    request: GatewayRequest,
    provider_updates: dict[str, object],
    ignored: list[str],
) -> None:
    """Send ``top_p: 1`` with ``temperature: 0`` when a Mistral rung is on the route.

    The value is set route-wide: at temperature 0 every rung decodes greedily,
    so ``top_p`` is a no-op on the others too and one provider request serves
    the whole waterfall. Nothing changes when an earlier rule already dropped
    ``temperature`` or ``top_p`` for the route.

    Args:
        profiles: Ordered wire profiles for every live route deployment.
        request: Canonical request being shaped.
        provider_updates: Provider-request field updates collected so far.
        ignored: Public disclosures collected so far, appended in place.
    """
    if (
        request.temperature != 0
        or request.top_p is None
        or request.top_p == 1
        or "temperature" in provider_updates
        or "top_p" in provider_updates
        or not any(is_mistral_model_id(profile.model_id) for profile in profiles)
    ):
        return
    provider_updates["top_p"] = 1.0
    if MISTRAL_GREEDY_TOP_P not in ignored:
        ignored.append(MISTRAL_GREEDY_TOP_P)
