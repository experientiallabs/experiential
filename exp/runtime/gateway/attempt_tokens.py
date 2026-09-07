# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Worst-case token estimation for one physical gateway attempt.

The conservative ``(input, output)`` token bound a single dispatch can consume,
per API surface. It is what :func:`exp.runtime.gateway.budgets.maximum_attempt_cost_micro_usd`
prices, and what the platform's free-tier and token-rate-limit windows reserve
in flight (released to the settled truth on finish). Lives in its own module so
``budgets`` stays under the repo's hand-authored line ceiling.
"""

from typing import assert_never

from exp.common.core.artifacts import canonical_json_bytes
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.embeddings_contracts import EmbeddingsRequest, ServingRequest
from exp.runtime.gateway.images_contracts import ImagesRequest
from exp.runtime.gateway.replay_identity import provider_replay_authority

# Output tokens reserved when neither the caller nor the deployment bounds output
# (an output bound is not a price, so its absence must not unprice a route).
DEFAULT_RESERVATION_OUTPUT_TOKENS = 32_768


def worst_case_attempt_tokens(
    request: ServingRequest,
    deployment: ExactModelDeployment,
) -> tuple[int, int]:
    """Conservative worst-case ``(input, output)`` tokens for one call.

    Embeddings and image requests consume no completion output, so they reserve
    their byte-bounded input and zero output; a completion reserves its
    byte-bounded input (plus replayed-carrier bytes) and its clamped max output.
    """
    return worst_case_input_tokens(request), worst_case_output_tokens(request, deployment)


def worst_case_input_tokens(request: ServingRequest) -> int:
    """Byte-bounded worst-case input tokens for one request.

    Deployment-independent, and the only expensive half of the estimate (it
    serializes the whole request), so a ladder walk computes it once and pairs
    it with each candidate's cheap output clamp.
    """
    input_tokens = len(canonical_json_bytes(request))
    match request:
        case EmbeddingsRequest() | ImagesRequest():
            return input_tokens
        case GatewayRequest():
            # Excluded provider carriers (replayed reasoning, native items,
            # verbatim configs) are provider-read input the plain serialization
            # misses; their envelope bytes keep the bound an upper bound.
            replay_envelope = provider_replay_authority(request)
            if replay_envelope is not None:
                input_tokens += len(canonical_json_bytes(replay_envelope))
            return input_tokens
        case _:  # pragma: no cover - exhaustive over the ServingRequest union.
            assert_never(request)


def worst_case_output_tokens(
    request: ServingRequest,
    deployment: ExactModelDeployment,
) -> int:
    """Worst-case output tokens one deployment could emit for this request."""
    match request:
        case EmbeddingsRequest() | ImagesRequest():
            return 0
        case GatewayRequest():
            output_tokens = request.maximum_output_tokens
            deployment_ceiling = (
                deployment.capabilities.maximum_output_tokens
                if deployment.capabilities is not None
                else None
            )
            # Clamp caller output to the deployment ceiling: an unbounded value
            # would inflate the estimate past MAXIMUM_MICRO_USD and mis-refuse a
            # fundable request. Settlement charges actual tokens, not this bound.
            if output_tokens is None:
                output_tokens = deployment_ceiling
            elif deployment_ceiling is not None:
                output_tokens = min(output_tokens, deployment_ceiling)
            if output_tokens is None:
                # No caller value and no ceiling: a realistic default, bounded by
                # any known context window (the model cannot emit past it).
                context_window = (
                    deployment.capabilities.context_window_tokens
                    if deployment.capabilities is not None
                    else None
                )
                output_tokens = (
                    min(DEFAULT_RESERVATION_OUTPUT_TOKENS, context_window)
                    if context_window is not None
                    else DEFAULT_RESERVATION_OUTPUT_TOKENS
                )
            return output_tokens
        case _:  # pragma: no cover - exhaustive over the ServingRequest union.
            assert_never(request)
