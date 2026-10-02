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
        compatible = profile.dialect == "openai_compatible" and profile.supports_logprobs is True
        if profile.supports_reasoning:
            compatible = (
                compatible
                and (
                    request.reasoning_effort
                    if request.reasoning_effort is not None
                    else profile.reasoning_effort
                )
                in profile.logprobs_reasoning_efforts
            )
        if not compatible:
            raise ProviderParameterError(
                message="This model route cannot preserve the requested token probabilities.",
                param=parameter,
                code="unsupported_parameter",
            )
        if emulated_stop_sequences(profile.dialect, request):
            raise ProviderParameterError(
                message="Chat logprobs cannot be combined with gateway-emulated stop sequences.",
                param="stop",
                code="unsupported_parameter",
            )


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
