"""Shared generation-parameter validation helpers."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.models.providers.anthropic_tool_compat import (
    anthropic_input_schema_reshaping,
    anthropic_rejects_assistant_prefill,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.reasoning_compat import (
    REASONING_EFFORTS,
    supported_reasoning_efforts,
)


def effective_profile_reasoning_effort(
    profile: GatewayWireProfile,
    requested_effort: str | None,
) -> str | None:
    """Return an explicit caller effort or one wire's required default."""
    if requested_effort is not None:
        return requested_effort
    return profile.reasoning_effort if profile.reasoning_effort_required else None


def profile_reasoning_efforts(profile: GatewayWireProfile) -> tuple[str, ...]:
    """Return exact accepted efforts for one deployment wire profile."""
    if not profile.supports_reasoning or profile.reasoning_wire_format == "none":
        return ()
    return supported_reasoning_efforts(
        profile.model_id,
        profile.reasoning_wire_format,
        configured_effort=profile.reasoning_effort,
        explicit_efforts=profile.supported_reasoning_efforts or None,
    )


def lane_default_reasoning_effort(profiles: Sequence[GatewayWireProfile]) -> str | None:
    """Return the depth a level-less "think" asks for on a route of effort rungs.

    A budget-less thinking config (``adaptive``, or the bare ``enabled`` Claude
    Code sends) asks the MODEL to pick its depth, and on an effort rung the
    model's own depth is its catalog default (``reasoning_default_effort``,
    carried on the wire profile as ``reasoning_effort``): the first rung in
    route order that pins an active default it can serve names the tier, so an
    operator sets a lane's think-mode depth by catalog, not by code. A ``none``
    default is not a depth (that rung reasons only when asked) and is skipped.

    Args:
        profiles: Ordered wire profiles for every live route deployment.

    Returns:
        The lane's default tier, or ``None`` when no rung pins a servable one.
    """
    for profile in profiles:
        default = profile.reasoning_effort
        if (
            default is not None
            and default in REASONING_EFFORTS
            and default != "none"
            and default in profile_reasoning_efforts(profile)
        ):
            return default
    return None


def route_numeric_bounds(
    profiles: Sequence[GatewayWireProfile],
    *,
    minimum: Callable[[GatewayWireProfile], float | int],
    maximum: Callable[[GatewayWireProfile], float | int | None],
) -> tuple[float | int, float | int | None]:
    """Return the interval every rung of the route accepts for one control.

    The route's floor is the largest rung floor and its ceiling the smallest
    declared rung ceiling (``None`` when no rung declares one), so a value in
    the interval is accepted by every deployment the waterfall may reach.
    """
    route_minimum = max(minimum(profile) for profile in profiles)
    maxima = tuple(bound for profile in profiles if (bound := maximum(profile)) is not None)
    return route_minimum, (min(maxima) if maxima else None)


def clamp_route_numeric_parameter(
    profiles: Sequence[GatewayWireProfile],
    *,
    value: float | int,
    minimum: Callable[[GatewayWireProfile], float | int],
    maximum: Callable[[GatewayWireProfile], float | int | None],
) -> float | int:
    """Return the caller value clamped to the interval every rung accepts.

    Used only on routes whose every rung declares ``clamps_sampling_to_range``;
    an in-range value comes back unchanged so the caller sees no disclosure.
    """
    route_minimum, route_maximum = route_numeric_bounds(profiles, minimum=minimum, maximum=maximum)
    if value < route_minimum:
        return route_minimum
    if route_maximum is not None and value > route_maximum:
        return route_maximum
    return value


def require_route_numeric_parameter(
    profiles: Sequence[GatewayWireProfile],
    *,
    param: str,
    value: float | int,
    supported: Callable[[GatewayWireProfile], bool],
    minimum: Callable[[GatewayWireProfile], float | int],
    maximum: Callable[[GatewayWireProfile], float | int | None],
) -> None:
    """Require every waterfall rung to accept one exact numeric control."""
    if not all(supported(profile) for profile in profiles):
        raise ProviderParameterError(
            message=(
                f"The parameter {param!r} is not supported by this model route. "
                "Remove the field or choose a different model."
            ),
            param=param,
            code="unsupported_parameter",
        )
    route_minimum, route_maximum = route_numeric_bounds(profiles, minimum=minimum, maximum=maximum)
    if value >= route_minimum and (route_maximum is None or value <= route_maximum):
        return
    range_text = (
        f"{route_minimum} or greater"
        if route_maximum is None
        else f"between {route_minimum} and {route_maximum}"
    )
    raise ProviderParameterError(
        message=(
            f"The value {value!r} for {param!r} is not supported by this model route. "
            f"Supported values are {range_text}."
        ),
        param=param,
        code="invalid_parameter",
    )


REASONING_SUMMARY_DIALECTS = frozenset({"openai_responses", "anthropic_messages"})


def serves_reasoning_summary(profile: GatewayWireProfile) -> bool:
    """Return whether one rung's reasoning reaches Responses summary parts.

    Native Responses deployments carry summary parts on the wire, and
    Anthropic thinking text is projected onto the same parts by the
    Responses encoder. Every other dialect either has no reasoning text or
    surfaces a reasoning item the summary channel cannot carry.

    Args:
        profile: One certified deployment wire profile from the route.

    Returns:
        Whether this deployment can serve a requested reasoning summary.
    """
    return profile.supports_reasoning and profile.dialect in REASONING_SUMMARY_DIALECTS


def anthropic_reasoning_disengaged(request: GatewayRequest) -> bool:
    """Whether an Anthropic dispatch will send no extended-thinking budget.

    On the native Messages wire the model reasons only when the caller asks:
    a ``thinking`` config of type ``enabled``/``adaptive`` or a reasoning
    effort turns it on, and their absence leaves thinking OFF. This is the
    inverse of the OpenAI effort-native models, whose default IS reasoning, so
    it governs the srn sampling hatch ONLY for the anthropic_adaptive wire
    (a budgeted-enabled route such as haiku-4-5): with thinking off, Anthropic
    accepts an ordinary temperature, so srn must not drop it.
    """
    config = request.provider_thinking_config
    thinking_on = config is not None and config.get("type") in {"enabled", "adaptive"}
    effort_on = request.reasoning_effort is not None and request.reasoning_effort != "none"
    return not thinking_on and not effort_on


def require_assistant_prefill_supported(
    profiles: Sequence[GatewayWireProfile], request: GatewayRequest
) -> None:
    """Refuse a trailing assistant turn before dispatch on rungs whose model rejects it.

    Anthropic's 4.6+ and 5-generation releases answer assistant prefill with a
    400 after the request was dispatched and billed for admission. The rungs
    that carry such a model narrow out here with the same fact stated for the
    caller; a route with no other rung surfaces it as the request's 400. The
    check keys on the MODEL, not the wire: relays (OpenRouter's
    ``anthropic/claude-opus-5``, Azure Foundry's Claude deployments) forward
    the same rejection (live 2026-09-07: "Azure: This model does not support
    assistant message prefill" through OpenRouter), and the release matcher
    only ever matches a Claude release id.

    Raises:
        ProviderParameterError: The final message is an assistant turn and a
            profile's model refuses prefill.
    """
    if not request.messages or request.messages[-1].role != "assistant":
        return
    if request.messages[-1].provider_native_item is not None:
        return
    for profile in profiles:
        if not anthropic_rejects_assistant_prefill(profile.model_id):
            continue
        raise ProviderParameterError(
            message=(
                f"{profile.model_id} does not accept an assistant message as the final "
                "turn (assistant prefill). End the conversation with a user message, or "
                "choose a model alias that supports prefill."
            ),
            param="messages",
            code="unsupported_parameter",
        )


_ANTHROPIC_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
_ANTHROPIC_TOOL_NAME_DIALECTS = frozenset({"anthropic_messages", "bedrock_converse_stream"})


def require_tool_names_supported(
    profiles: Sequence[GatewayWireProfile], request: GatewayRequest
) -> None:
    """Refuse a tool name the Anthropic wire will 400 by name, before dispatch.

    Anthropic (and Bedrock, which relays the same rule) accepts tool names
    matching ``^[a-zA-Z0-9_-]{1,128}$``; a client sending dots, spaces, or
    a longer name learned that only from the provider's 400 after dispatch
    ("tools.0.custom.name: String should match pattern"). The rung narrows
    out with the index named; the name itself is caller content and stays
    out of the message.

    Raises:
        ProviderParameterError: A profile speaks an Anthropic wire and a tool
            name does not match.
    """
    if not request.tools or not any(
        profile.dialect in _ANTHROPIC_TOOL_NAME_DIALECTS for profile in profiles
    ):
        return
    for index, tool in enumerate(request.tools):
        if _ANTHROPIC_TOOL_NAME.fullmatch(tool.name) is None:
            raise ProviderParameterError(
                message=(
                    f"tools[{index}].name is not accepted by this model route: tool names "
                    "must match ^[a-zA-Z0-9_-]{1,128} (letters, digits, underscore, "
                    "hyphen). Rename the tool or choose a different model alias."
                ),
                param=f"tools[{index}].name",
                code="invalid_parameter",
            )


_ANTHROPIC_SCHEMA_DIALECTS = frozenset({"anthropic_messages", "bedrock_converse_stream"})


def disclose_anthropic_tool_schemas(
    profiles: Sequence[GatewayWireProfile], request: GatewayRequest, ignored: list[str]
) -> None:
    """Record every tool schema an Anthropic-family rung will reshape at dispatch.

    ``anthropic_input_schema`` flattens a root oneOf/anyOf/allOf into one
    object and adds a missing root ``type``; the caller reads the change in
    ``ignored_parameters`` as ``tools[i].parameters->reshaped(<kind>)``.
    """
    if not request.tools or not any(
        profile.dialect in _ANTHROPIC_SCHEMA_DIALECTS for profile in profiles
    ):
        return
    for index, tool in enumerate(request.tools):
        kind = anthropic_input_schema_reshaping(tool.parameters)
        if kind is not None:
            note = f"tools[{index}].parameters->reshaped({kind})"
            if note not in ignored:
                ignored.append(note)


def mid_conversation_system_present(request: GatewayRequest) -> bool:
    """Whether a system turn appears after the conversation has begun."""
    conversation_started = False
    for message in request.messages:
        if message.role in {"system", "developer"} and message.provider_native_item is None:
            if conversation_started:
                return True
        else:
            conversation_started = True
    return False
