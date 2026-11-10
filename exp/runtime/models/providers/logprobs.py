"""Lossless Chat token-probability admission for exact provider deployments."""

from collections.abc import Sequence

from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.generation_parameter_validation import (
    profile_reasoning_efforts,
)
from exp.runtime.models.providers.protocol import emulated_stop_sequences
from exp.runtime.models.providers.thinking_budget import (
    thinking_budget_parameter,
    thinking_budget_value,
)

_ROUTE_REFUSAL = "This model route cannot preserve the requested token probabilities."


def _caller_named_effort(request: GatewayRequest) -> bool:
    """Whether the effort on the request came from an effort field the caller sent.

    An effort translated from an enable or disable control, or resolved by the
    gateway for a rung, is not the caller's field to change.
    """
    return request.reasoning_effort is not None and request.reasoning_effort_parameter is not None


def _effort_refusal(
    effort: str | None, qualified: Sequence[str], request: GatewayRequest, parameter: str
) -> str:
    """Name the effort that blocked probabilities and the efforts that would serve them.

    The gateway never changes effort to admit probabilities, so without this a
    caller sending e.g. ``medium`` (which a plain request would coerce) sees only
    a generic route refusal and cannot tell what to change. ``qualified`` is the
    refusing rung's own set: route narrowing checks each rung alone, so an
    effort that rung qualifies admits the request on it.
    """
    field = request.caller_effort_parameter
    remove = "logprobs and top_logprobs" if parameter == "top_logprobs" else "logprobs"
    choices = ", ".join(qualified)
    if _caller_named_effort(request):
        return (
            f"Token probabilities are not available with {field} {effort!r} on this model. "
            f"Use {field} one of: {choices} (with any thinking on/off control set to agree), "
            f"or remove {remove}."
        )
    if request.reasoning_effort is not None:
        return (
            f"Token probabilities are not available with the reasoning effort this request "
            f"resolves to ({effort!r}) on this model. Send {field} one of: {choices} in place "
            f"of the other thinking controls, or remove {remove}."
        )
    requested = (
        "the model's default reasoning effort"
        if effort is None
        else f"the model's default reasoning effort {effort!r}"
    )
    return (
        f"Token probabilities are not available with {requested} on this model. "
        f"Use {field} one of: {choices} (with any thinking on/off control set to agree), "
        f"or remove {remove}."
    )


def require_chat_logprobs(profiles: Sequence[GatewayWireProfile], request: GatewayRequest) -> None:
    """Reject probability controls that any remaining deployment cannot preserve.

    Route narrowing calls this for each candidate before the final frozen rung
    repeats it. Probabilities are never dropped or admitted by changing effort.
    """
    if request.logprobs is not True and request.top_logprobs is None:
        return
    parameter = "top_logprobs" if request.top_logprobs is not None else "logprobs"
    if request.surface != GatewayApiSurface.CHAT_COMPLETIONS:
        raise ProviderParameterError(
            message="Token probabilities are supported only on Chat Completions.",
            param=parameter,
            code="unsupported_parameter",
        )
    if request.top_logprobs is not None and request.logprobs is not True:
        raise ProviderParameterError(
            message="top_logprobs requires logprobs=true.",
            param="top_logprobs",
            code="invalid_parameter",
        )
    if thinking_budget_value(request) is not None:
        raise ProviderParameterError(
            message=(
                "Token probabilities are not qualified with a numeric thinking_budget. "
                "Remove thinking_budget and select a probability-capable reasoning_effort, "
                "or disable logprobs."
            ),
            param=thinking_budget_parameter(request),
            code="unsupported_parameter",
        )
    # Capability first: an incapable rung anywhere makes every effort change futile.
    for profile in profiles:
        if not (profile.dialect == "openai_compatible" and profile.supports_logprobs is True):
            raise ProviderParameterError(
                message=_ROUTE_REFUSAL, param=parameter, code="unsupported_parameter"
            )
    for profile in profiles:
        if not profile.supports_reasoning:
            continue
        effort = (
            request.reasoning_effort
            if request.reasoning_effort is not None
            else profile.reasoning_effort
        )
        if effort not in profile.logprobs_reasoning_efforts:
            # Suggest only qualified efforts the rung can also carry, or the
            # retry is refused; with none, the caller's effort field is not named.
            ladder = profile_reasoning_efforts(profile)
            qualified = [level for level in profile.logprobs_reasoning_efforts if level in ladder]
            if not qualified:
                raise ProviderParameterError(
                    message=_ROUTE_REFUSAL, param=parameter, code="unsupported_parameter"
                )
            raise ProviderParameterError(
                message=_effort_refusal(effort, qualified, request, parameter),
                param=(
                    request.caller_effort_parameter if _caller_named_effort(request) else parameter
                ),
                code="unsupported_parameter",
            )
    for profile in profiles:
        if emulated_stop_sequences(profile.dialect, request):
            raise ProviderParameterError(
                message="Chat logprobs cannot be combined with gateway-emulated stop sequences.",
                param="stop",
                code="unsupported_parameter",
            )


def capture_logprobs_eligible(
    profile: GatewayWireProfile, request: GatewayRequest, *, caller: GatewayRequest | None = None
) -> bool:
    """Whether a rung may return Chat probabilities the caller did not request.

    The host's capture may then ask for them on this rung and keep them out of
    the caller's answer. The rung must satisfy exactly what an explicit Chat
    probability request would need, on the platform's own credential, and the
    caller must not have asked for probabilities on any surface (their own
    request already returns them in the captured response).

    Args:
        profile: The rung's wire profile.
        request: The effective rung request (capability and effort checks).
        caller: The caller's public request, whose probability selectors decide
            (an explicit ``logprobs: false`` is cleared from the rung request).

    Returns:
        Whether capture may request probabilities on this rung.
    """
    selectors = request if caller is None else caller
    if (
        selectors.logprobs is not None
        or selectors.top_logprobs is not None
        or selectors.include_output_text_logprobs
        or profile.billing_customer_managed
        or profile.dialect != "openai_compatible"
        or profile.supports_logprobs is not True
        or thinking_budget_value(request) is not None
        or emulated_stop_sequences(profile.dialect, request)
    ):
        return False
    if profile.supports_reasoning:
        effort = (
            request.reasoning_effort
            if request.reasoning_effort is not None
            else profile.reasoning_effort
        )
        return effort in profile.logprobs_reasoning_efforts
    return True


def require_responses_logprobs(
    profiles: Sequence[GatewayWireProfile], request: GatewayRequest
) -> None:
    """Require native Responses routes for an active output probability request."""
    if request.surface != GatewayApiSurface.RESPONSES:
        return
    if not request.include_output_text_logprobs and request.top_logprobs is None:
        return
    for profile in profiles:
        if profile.dialect != "openai_responses" or not profile.supports_responses_logprobs:
            raise ProviderParameterError(
                message="This model route cannot preserve Responses output text probabilities.",
                param="top_logprobs" if request.top_logprobs is not None else "include",
                code="unsupported_parameter",
            )


def require_unmodified_probability_output(request: GatewayRequest, output_checks: bool) -> None:
    """Reject output rewriting until its token alignment can be preserved."""
    active = (
        request.logprobs is True
        or request.include_output_text_logprobs
        or request.top_logprobs is not None
    )
    if active and output_checks:
        parameter = (
            "logprobs"
            if request.surface == GatewayApiSurface.CHAT_COMPLETIONS
            else ("top_logprobs" if request.top_logprobs is not None else "include")
        )
        surface = "Chat" if request.surface == GatewayApiSurface.CHAT_COMPLETIONS else "Responses"
        raise ProviderParameterError(
            message=f"{surface} probabilities cannot be combined with output guardrails.",
            param=parameter,
            code="unsupported_parameter",
        )
