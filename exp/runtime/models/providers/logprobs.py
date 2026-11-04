"""Lossless Chat token-probability admission for exact provider deployments."""

from collections.abc import Sequence

from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.protocol import emulated_stop_sequences
from exp.runtime.models.providers.thinking_budget import (
    thinking_budget_parameter,
    thinking_budget_value,
)


def _effort_refusal(effort: str | None, qualified: Sequence[str]) -> str:
    """Name the effort that blocked probabilities and the efforts that would serve them.

    The gateway never changes effort to admit probabilities, so without this a
    caller sending e.g. ``medium`` (which a plain request would coerce) sees only
    a generic route refusal and cannot tell what to change.
    """
    if not qualified:
        return "This model route cannot preserve the requested token probabilities."
    choices = ", ".join(qualified)
    requested = (
        "the model's default reasoning effort" if effort is None else f"reasoning_effort {effort!r}"
    )
    return (
        f"Token probabilities are not available with {requested} on this model. "
        f"Use reasoning_effort one of: {choices}, or disable logprobs."
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
    for profile in profiles:
        if not (profile.dialect == "openai_compatible" and profile.supports_logprobs is True):
            raise ProviderParameterError(
                message="This model route cannot preserve the requested token probabilities.",
                param=parameter,
                code="unsupported_parameter",
            )
        if profile.supports_reasoning:
            effort = (
                request.reasoning_effort
                if request.reasoning_effort is not None
                else profile.reasoning_effort
            )
            if effort not in profile.logprobs_reasoning_efforts:
                raise ProviderParameterError(
                    message=_effort_refusal(effort, profile.logprobs_reasoning_efforts),
                    param=(
                        "reasoning_effort" if request.reasoning_effort is not None else parameter
                    ),
                    code="unsupported_parameter",
                )
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
