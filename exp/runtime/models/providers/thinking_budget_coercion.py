"""Translate a numeric thinking budget onto a route's effort ladder, disclosed.

A numeric budget (``thinking.budget_tokens`` on Messages, ``thinking_budget``
on Chat) is preserved verbatim only on qualified Anthropic, Gemini 2.5 and
Qwen Cloud wires (:mod:`exp.runtime.models.providers.thinking_budget`). A route
that reasons through an effort ladder instead used to refuse the budget by
name, which ended every session of a client that pins one
(qwen3.8-flash-next-uncensored on Experiential Cloud, 27 refusals across two
organizations in two hours, production 2026-10-19). The budget now names the
depth it asks for, that depth snaps to the nearest tier the route serves, and
the substitution is disclosed. A route with no reasoning tier at all keeps the
refusal.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

from exp.common.models.model import ReasoningEffort
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderCapabilityError, ProviderParameterError
from exp.runtime.models.providers.generation_parameter_validation import (
    lane_default_reasoning_effort,
    profile_reasoning_efforts,
)
from exp.runtime.models.providers.generation_route_compat import (
    compatible_generation_parameter_profile_indexes,
)
from exp.runtime.models.providers.reasoning_compat import efforts_by_nearness
from exp.runtime.models.providers.streaming_requests import route_generation_parameter_requests
from exp.runtime.models.providers.thinking_budget import (
    require_thinking_budget_support,
    thinking_budget_parameter,
    thinking_budget_value,
)

BUDGET_EFFORT_DISCLOSURE_SOURCE = "route_effort_ladder"
"""Reason named in ``<field>->reasoning_effort:<tier>(route_effort_ladder)``."""

_BUDGET_TIER_CEILINGS: tuple[tuple[int, ReasoningEffort], ...] = (
    (2_048, "low"),
    (8_192, "medium"),
    (24_576, "high"),
)
"""Exclusive budget ceilings per tier; anything above the last is ``xhigh``.

A gateway choice, not a provider contract: under 2k tokens reads as light
thinking, under 8k as standard, under 24k as deep, and anything larger as
the deepest active tier before the ladder snap.
"""


def budget_effort_tier(budget: int, profiles: Sequence[GatewayWireProfile]) -> ReasoningEffort:
    """Return the effort depth one numeric budget asks for.

    Args:
        budget: Caller budget; ``0`` disables thinking and ``-1`` (Gemini's
            dynamic thinking) asks the model to pick its own depth.
        profiles: Route profiles, whose lane default names the dynamic depth.

    Returns:
        The requested tier before snapping onto the route's ladder.
    """
    if budget == 0:
        return "none"
    if budget < 0:
        default = lane_default_reasoning_effort(profiles)
        return cast("ReasoningEffort", default) if default is not None else "medium"
    for ceiling, tier in _BUDGET_TIER_CEILINGS:
        if budget < ceiling:
            return tier
    return "xhigh"


def coerce_budget_to_effort(
    profiles: Sequence[GatewayWireProfile],
    request: GatewayRequest,
    *,
    admits: Callable[[GatewayRequest], bool] | None = None,
) -> tuple[GatewayRequest, str] | None:
    """Replace a numeric budget no rung preserves with the nearest servable tier.

    Candidates are tried nearest first (a tie prefers the lower tier), and only
    one that survives full route construction and the caller's probe is
    offered, mirroring the explicit-effort snap. An active budget never snaps
    to ``none``; a zero budget snaps only to an exact off tier.

    Args:
        profiles: Ordered wire profiles for every live route deployment.
        request: Decoded request that no rung accepted verbatim.
        admits: Optional caller probe a candidate must pass.

    Returns:
        The translated request and its disclosure, or ``None`` when the request
        carries no budget, also names an effort, or the route has no tier that
        serves it (the named refusal stands).
    """
    budget = thinking_budget_value(request)
    if budget is None:
        return None
    if (
        request.reasoning_effort is not None
        or (request.provider_output_config or {}).get("effort") is not None
    ):
        # Two depth controls at once is the caller's contradiction to resolve.
        return None
    config = request.provider_thinking_config
    if request.thinking_budget is not None and config is not None:
        return None
    if config is not None and (
        config.get("type") != "enabled" or set(config) - {"type", "budget_tokens"}
    ):
        # Adaptive budgets translate for budgeted Anthropic rungs elsewhere, and a
        # config carrying more than the budget is not this translation's to drop.
        return None
    if any(_carries_budgets(profile, request) for profile in profiles):
        # A budget-capable rung decides: it serves the budget verbatim, or the
        # value itself is invalid there, and either way no effort may replace it.
        return None
    ladder: set[str] = set()
    for profile in profiles:
        ladder.update(profile_reasoning_efforts(profile))
    requested = budget_effort_tier(budget, profiles)
    if requested == "none":
        candidates = ladder & {"none"}
    else:
        candidates = ladder - {"none"}
    if not candidates:
        return None
    field = thinking_budget_parameter(request)
    for candidate in efforts_by_nearness(requested, candidates):
        translated = request.model_copy(
            update={
                "thinking_budget": None,
                "provider_thinking_config": None,
                "reasoning_effort": candidate,
            }
        )
        try:
            indexes = compatible_generation_parameter_profile_indexes(profiles, translated)
            route_generation_parameter_requests(
                tuple(profiles[index] for index in indexes), translated
            )
        except (ProviderParameterError, ProviderCapabilityError):
            continue
        if admits is not None and not admits(translated):
            continue
        return (
            translated,
            f"{field}->reasoning_effort:{candidate}({BUDGET_EFFORT_DISCLOSURE_SOURCE})",
        )
    return None


def _carries_budgets(profile: GatewayWireProfile, request: GatewayRequest) -> bool:
    """Whether one rung is budget-capable for this request.

    A refusal of the VALUE (out of range, above the output limit, combined
    with another depth control) names a rung that carries budgets; only an
    unsupported-parameter refusal says the rung cannot carry one at all.
    """
    try:
        require_thinking_budget_support(profile, request)
    except ProviderParameterError as exc:
        return exc.code != "unsupported_parameter"
    return True
