"""Disclosed request repairs for controls no rung of a route can carry verbatim.

Admission applies these only after every rung declined the verbatim request
(see ``capability_policy.coerce_generation_parameters``), so a rung that
honors the caller's exact value is always preferred. Each repair answers the
one question the refusal asked with the nearest request the route can serve,
and every repair is named in ``ignored_parameters``:

* an output ceiling outside the route's accepted range is clamped onto it
  (Claude Code sends ``max_tokens`` 64000 or 128000 to every model; a probe
  sends a ceiling below a reasoning lane's floor);
* a ``temperature`` or ``top_p`` outside the route's accepted range is clamped
  onto it the same way (Claude's thinking routes accept only 1.0; a caller's
  0.1 would otherwise be refused);
* a level-less enable-thinking toggle on a route whose rungs cannot reason at
  any depth is dropped, because the model can only answer without it;
* a request to RECEIVE encrypted reasoning on a route with no reasoning
  carrier is dropped (the answer arrives without those items, exactly as if
  the selector had been omitted), and replayed native encrypted reasoning
  that no rung can read is removed from history.

Numerical thinking budgets and explicit efforts are not handled
here; their substitutions live beside the rest of the effort ladder logic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from exp.runtime.gateway.contracts import GatewayMessage, GatewayRequest
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.generation_parameter_validation import (
    resolve_level_less_enable,
)

if TYPE_CHECKING:
    from exp.runtime.models.providers.base import GatewayWireProfile

OPENAI_RESPONSES_MINIMUM_OUTPUT_TOKENS = 16
"""Smallest ``max_output_tokens`` the native OpenAI Responses wire accepts."""

ENABLE_THINKING_DROPPED = "enable_thinking->dropped(no_reasoning_rung)"
"""Disclosure recorded when a level-less thinking toggle meets a route that cannot reason."""

ENCRYPTED_REASONING_INCLUDE_DROPPED = (
    "include.reasoning.encrypted_content->dropped(no_reasoning_carrier)"
)
"""Disclosure recorded when encrypted reasoning output cannot be produced on the route."""

ENCRYPTED_REASONING_HISTORY_DROPPED = "input.reasoning.encrypted_content->dropped(foreign_route)"
"""Disclosure recorded when replayed native encrypted reasoning no rung can read is removed."""


@dataclass(frozen=True)
class RouteRepair:
    """One disclosed repair and the request it produces.

    Attributes:
        request: Request after the repair.
        disclosures: ``ignored_parameters`` entries naming the repair.
    """

    request: GatewayRequest
    disclosures: tuple[str, ...]


def output_ceiling_range(profile: GatewayWireProfile) -> tuple[int, int | None]:
    """Return one rung's accepted ``[floor, maximum]`` output ceiling range."""
    floor = max(
        profile.minimum_output_tokens or 0,
        OPENAI_RESPONSES_MINIMUM_OUTPUT_TOKENS if profile.dialect == "openai_responses" else 0,
    )
    return floor, profile.maximum_output_tokens


def clamp_output_ceiling(
    profiles: Sequence[GatewayWireProfile], request: GatewayRequest
) -> RouteRepair | None:
    """Clamp an explicit output ceiling onto the lead rung's accepted range.

    A ceiling above the rung's maximum is lowered to that maximum (the caller
    can never be billed for more than they allowed); a ceiling below the
    rung's floor is raised to that floor, the smallest generation the
    provider will run. The rung that leads the route decides the target,
    since it is the one that serves first. A rung whose range is empty, a
    plan account that refuses any ceiling, or a route where some rung already
    honors the exact value is left alone.

    Args:
        profiles: Ordered wire profiles for every live route deployment.
        request: Decoded public request that no rung accepted verbatim.

    Returns:
        The clamped request, or ``None`` when the ceiling is not the blocker.
    """
    value = request.maximum_output_tokens
    if value is None or not profiles:
        return None
    if any(profile.omits_output_token_limit for profile in profiles):
        return None
    ranges = [output_ceiling_range(profile) for profile in profiles]
    if any(floor <= value and (top is None or value <= top) for floor, top in ranges):
        return None
    floor, top = ranges[0]
    if top is not None and floor > top:
        return None
    parameter = request.maximum_output_tokens_parameter or "max_tokens"
    if top is not None and value > top:
        target, reason = top, "route_maximum"
    else:
        target, reason = floor, "route_minimum"
    return RouteRepair(
        request=request.model_copy(update={"maximum_output_tokens": target}),
        disclosures=(f"{parameter}->{target}({reason})",),
    )


def _sampling_range(profile: GatewayWireProfile, control: str) -> tuple[float, float] | None:
    """Return one rung's accepted range for a sampling control, ``None`` if undeclared."""
    if control == "top_p":
        if profile.supports_top_p is not True:
            return None
        return profile.minimum_top_p, profile.maximum_top_p
    if not profile.supports_temperature:
        return None
    return profile.minimum_temperature, profile.maximum_temperature


def clamp_sampling_range(
    profiles: Sequence[GatewayWireProfile], request: GatewayRequest
) -> RouteRepair | None:
    """Clamp ``temperature`` and ``top_p`` onto the lead rung's accepted range.

    Sampling is a preference, so a value the route cannot carry verbatim
    dispatches at the nearest value the leading rung accepts rather than
    being refused. Each control is repaired only when it is a blocker: some
    rung declares it, and every rung that declares it refuses the value on
    range. A rung that does not declare the control serves by dropping it
    (disclosed in route shaping), so its presence leaves the value alone.
    Values outside the public bounds never reach here: the request decoder
    refuses them.

    Args:
        profiles: Ordered wire profiles for every live route deployment.
        request: Decoded public request that no rung accepted verbatim.

    Returns:
        The clamped request, or ``None`` when no sampling value is the blocker.
    """
    updates: dict[str, object] = {}
    disclosures: list[str] = []
    for control, value in (("temperature", request.temperature), ("top_p", request.top_p)):
        if value is None or not profiles:
            continue
        ranges = [_sampling_range(profile, control) for profile in profiles]
        if any(bounds is None or bounds[0] <= value <= bounds[1] for bounds in ranges):
            continue
        floor, top = ranges[0] or (value, value)
        if floor > top:
            continue
        target, reason = (top, "route_maximum") if value > top else (floor, "route_minimum")
        updates[control] = target
        disclosures.append(f"{control}->{target}({reason})")
    if not updates:
        return None
    return RouteRepair(request=request.model_copy(update=updates), disclosures=tuple(disclosures))


def drop_unservable_enable(
    profiles: Sequence[GatewayWireProfile], request: GatewayRequest
) -> RouteRepair | None:
    """Drop a level-less enable-thinking toggle no rung can honor at any depth.

    Applies only when EVERY rung on its own refuses the toggle (no on-tier and
    either no reasoning channel or an off switch). A numerical budget or an
    explicit effort is never touched here.

    Args:
        profiles: Ordered wire profiles for every live route deployment.
        request: Decoded public request that no rung accepted verbatim.

    Returns:
        The request without the toggle, or ``None`` when some rung honors it.
    """
    if not request.thinking_default_enable or request.reasoning_effort is not None:
        return None
    if request.thinking_budget is not None:
        return None
    for profile in profiles:
        try:
            resolve_level_less_enable((profile,), effort_path=request.caller_effort_parameter)
        except ProviderParameterError:
            continue
        return None
    return RouteRepair(
        request=request.model_copy(update={"thinking_default_enable": False}),
        disclosures=(ENABLE_THINKING_DROPPED,),
    )


def _carries_encrypted_reasoning(profile: GatewayWireProfile) -> bool:
    """Whether one rung alone forms an encrypted-reasoning channel."""
    return (
        profile.dialect == "openai_responses"
        or profile.fireworks_reasoning_route_sha256 is not None
        or profile.hunyuan_reasoning_route_sha256 is not None
    )


def drop_unservable_encrypted_reasoning(
    profiles: Sequence[GatewayWireProfile], request: GatewayRequest
) -> RouteRepair | None:
    """Drop encrypted reasoning a route can neither produce nor read.

    The ``include`` selector drops only when no rung alone could seal
    reasoning (no native Responses rung and no gateway carrier). Replayed
    native encrypted reasoning is removed only when no rung speaks native
    Responses: that payload is the issuing provider's own state, and a route
    that cannot read it serves the turn from its visible history, just as a
    model switch would. A message left with nothing to carry is removed.

    Args:
        profiles: Ordered wire profiles for every live route deployment.
        request: Decoded public request that no rung accepted verbatim.

    Returns:
        The repaired request, or ``None`` when nothing needed repair.
    """
    updates: dict[str, object] = {}
    disclosures: list[str] = []
    if request.include_encrypted_reasoning and not any(
        _carries_encrypted_reasoning(profile) for profile in profiles
    ):
        updates["include_encrypted_reasoning"] = False
        disclosures.append(ENCRYPTED_REASONING_INCLUDE_DROPPED)
    replayed = any(
        block.kind == "encrypted_reasoning"
        for message in request.messages
        for block in message.provider_reasoning
    )
    if replayed and not any(profile.dialect == "openai_responses" for profile in profiles):
        messages: list[GatewayMessage] = []
        for message in request.messages:
            kept = tuple(
                block for block in message.provider_reasoning if block.kind != "encrypted_reasoning"
            )
            if len(kept) == len(message.provider_reasoning):
                messages.append(message)
                continue
            if (
                message.content is None
                and not message.tool_calls
                and not kept
                and message.provider_item_id is None
            ):
                continue
            messages.append(message.model_copy(update={"provider_reasoning": kept}))
        if messages:
            updates["messages"] = tuple(messages)
            disclosures.append(ENCRYPTED_REASONING_HISTORY_DROPPED)
    if not updates:
        return None
    return RouteRepair(request=request.model_copy(update=updates), disclosures=tuple(disclosures))


ROUTE_REPAIRS: tuple[
    Callable[[Sequence[GatewayWireProfile], GatewayRequest], RouteRepair | None], ...
] = (
    clamp_output_ceiling,
    clamp_sampling_range,
    drop_unservable_enable,
    drop_unservable_encrypted_reasoning,
)
"""Independent repairs admission composes before any effort substitution."""


def apply_route_repairs(
    profiles: Sequence[GatewayWireProfile], request: GatewayRequest
) -> RouteRepair | None:
    """Apply every applicable independent repair in order.

    Args:
        profiles: Ordered wire profiles for every live route deployment.
        request: Decoded public request that no rung accepted verbatim.

    Returns:
        The cumulative repair, or ``None`` when none applies.
    """
    disclosures: list[str] = []
    for repair in ROUTE_REPAIRS:
        applied = repair(profiles, request)
        if applied is not None:
            request = applied.request
            disclosures.extend(applied.disclosures)
    if not disclosures:
        return None
    return RouteRepair(request=request, disclosures=tuple(disclosures))
