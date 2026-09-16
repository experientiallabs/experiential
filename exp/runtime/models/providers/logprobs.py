"""Lossless Chat token-probability admission for exact provider deployments."""

from collections.abc import Sequence

from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.generation_parameter_validation import (
    effective_profile_reasoning_effort,
)
from exp.runtime.models.providers.protocol import emulated_stop_sequences


def require_chat_logprobs(profiles: Sequence[GatewayWireProfile], request: GatewayRequest) -> None:
    """Reject probability controls that any remaining deployment cannot preserve.

    Route narrowing calls this for each candidate before the final frozen rung
    repeats it. Probabilities are never dropped or admitted by changing effort.
    """
    if request.logprobs is not True and request.top_logprobs is None:
        return
    parameter = "top_logprobs" if request.top_logprobs is not None else "logprobs"
    if request.top_logprobs is not None and request.logprobs is not True:
        raise ProviderParameterError(
            message="top_logprobs requires logprobs=true.",
            param="top_logprobs",
            code="invalid_parameter",
        )
    if request.surface != GatewayApiSurface.CHAT_COMPLETIONS:
        raise ProviderParameterError(
            message="Token probabilities are supported only on Chat Completions.",
            param=parameter,
            code="unsupported_parameter",
        )
    for profile in profiles:
        compatible = profile.dialect == "openai_compatible" and profile.supports_logprobs is True
        if profile.supports_reasoning:
            compatible = (
                compatible
                and effective_profile_reasoning_effort(profile, request.reasoning_effort)
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


def require_unmodified_probability_output(request: GatewayRequest, output_checks: bool) -> None:
    """Reject output rewriting until its token alignment can be preserved."""
    if request.logprobs is True and output_checks:
        raise ProviderParameterError(
            message="Chat logprobs cannot be combined with output guardrails.",
            param="logprobs",
            code="unsupported_parameter",
        )
