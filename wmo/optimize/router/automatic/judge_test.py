"""Automatic-router approved-judge execution tests."""

from __future__ import annotations

import pytest

from wmo.common.models import (
    BillingSource,
    ModelCapabilities,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    completion_cost_reservation,
)
from wmo.optimize.router.automatic.judge import ReservedJudgeClient
from wmo.optimize.router.errors import (
    JudgeDispatchExhaustedError,
    JudgeTranscriptAdmissionError,
)
from wmo.runtime.models.providers.errors import ProviderRetryableResponseError
from wmo.runtime.models.providers.openai import openai_responses_response
from wmo.simulation.engines.text.recording import Utf8UpperBoundTokenCounter


class _Client:
    """Return one production-shaped provider response."""

    def __init__(self, response: ModelResponse) -> None:
        """Retain the response returned by the reservation boundary.

        Args:
            response: Parsed provider response with usage and no direct cost.
        """
        self.response = response

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return the retained response.

        Args:
            request: Exact judge request accepted by the wrapper.

        Returns:
            Production-shaped provider response.
        """
        del request
        return self.response


def test_reserved_judge_prices_openai_usage_without_observed_cost() -> None:
    """Reconcile a native OpenAI response under the frozen retry and cache rates.

    The retry allowance charges the observed request input at the highest input rate plus the
    full reserved output budget, not the hard admission ceiling.
    """
    model = ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="openai",
        model_id="judge-model",
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )
    response = openai_responses_response(
        {
            "id": "resp_judge",
            "object": "response",
            "created_at": 1.0,
            "status": "completed",
            "model": "judge-model",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "type": "message",
                    "id": "msg_judge",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": '{"score":4}', "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "input_tokens_details": {"cached_tokens": 25, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
        configured_model=model,
        latency_seconds=0.1,
    )
    reservation = completion_cost_reservation(
        model=model,
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=4.0,
        cached_input_usd_per_million_tokens=0.5,
        cache_write_usd_per_million_tokens=2.0,
        maximum_attempts=2,
        maximum_input_tokens=1_000,
        maximum_output_tokens=500,
    )
    client = ReservedJudgeClient(
        _Client(response),
        reservation=reservation,
        model=model,
        capabilities=ModelCapabilities(
            supports_completions=True,
            context_window_tokens=2_000,
            maximum_output_tokens=500,
            input_cost_per_million_tokens_usd=1.0,
            output_cost_per_million_tokens_usd=4.0,
            cached_input_cost_per_million_tokens_usd=0.5,
            cache_write_cost_per_million_tokens_usd=2.0,
        ),
        maximum_attempts=2,
        maximum_provider_calls=1,
    )

    result = client.complete(
        ModelRequest(
            messages=(ModelMessage(role="user", content="Judge this."),),
            maximum_output_tokens=500,
        )
    )

    assert result.economics.cost_usd is not None
    assert result.economics.cost_usd.provenance == "estimated"
    assert result.economics.cost_usd.value == 0.0024025


def test_reserved_judge_rejects_an_over_ceiling_transcript_without_a_provider_call() -> None:
    """An over-ceiling request raises the typed admission error before any provider dispatch."""
    model = ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="openai",
        model_id="judge-model",
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )
    reservation = completion_cost_reservation(
        model=model,
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=4.0,
        cached_input_usd_per_million_tokens=0.5,
        cache_write_usd_per_million_tokens=2.0,
        maximum_attempts=2,
        maximum_input_tokens=1_000,
        maximum_output_tokens=500,
    )
    client = ReservedJudgeClient(
        _Client(_unused_response(model)),
        reservation=reservation,
        model=model,
        capabilities=ModelCapabilities(
            supports_completions=True,
            context_window_tokens=2_000,
            maximum_output_tokens=500,
            input_cost_per_million_tokens_usd=1.0,
            output_cost_per_million_tokens_usd=4.0,
            cached_input_cost_per_million_tokens_usd=0.5,
            cache_write_cost_per_million_tokens_usd=2.0,
        ),
        maximum_attempts=2,
        maximum_provider_calls=1,
    )
    oversized = ModelRequest(
        messages=(ModelMessage(role="user", content="x" * 100_000),),
        maximum_output_tokens=500,
    )

    with pytest.raises(JudgeTranscriptAdmissionError):
        client.complete(oversized)

    assert client.calls == 0


def test_reserved_judge_prices_an_exhausted_empty_output_dispatch_conservatively() -> None:
    """An exhausted empty-output dispatch surfaces its conservative retry-bound spend."""
    model = ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="openai",
        model_id="judge-model",
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )
    reservation = completion_cost_reservation(
        model=model,
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=4.0,
        cached_input_usd_per_million_tokens=0.5,
        cache_write_usd_per_million_tokens=2.0,
        maximum_attempts=2,
        maximum_input_tokens=1_000,
        maximum_output_tokens=500,
    )
    client = ReservedJudgeClient(
        _ExhaustedClient(),
        reservation=reservation,
        model=model,
        capabilities=ModelCapabilities(
            supports_completions=True,
            context_window_tokens=2_000,
            maximum_output_tokens=500,
            input_cost_per_million_tokens_usd=1.0,
            output_cost_per_million_tokens_usd=4.0,
            cached_input_cost_per_million_tokens_usd=0.5,
            cache_write_cost_per_million_tokens_usd=2.0,
        ),
        maximum_attempts=2,
        maximum_provider_calls=1,
    )
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="Judge this."),),
        maximum_output_tokens=500,
    )

    with pytest.raises(JudgeDispatchExhaustedError) as excinfo:
        client.complete(request)

    counted_input = Utf8UpperBoundTokenCounter().count(request)
    expected = 2 * (counted_input * 2.0 + 500 * 4.0) / 1_000_000
    assert excinfo.value.conservative_cost_usd == expected
    assert client.calls == 1


class _ExhaustedClient:
    """Raise the retryable empty-output error for every dispatched request."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Raise the exhausted empty-output signal.

        Args:
            request: Exact judge request accepted by the wrapper.

        Raises:
            ProviderRetryableResponseError: Always, modeling exhausted bounded retries.
        """
        del request
        raise ProviderRetryableResponseError("OpenAI Responses output has no text or tool call")


def _unused_response(model: ModelSnapshot) -> ModelResponse:
    """Return one minimal parsed response that the admission test never dispatches.

    Args:
        model: Configured judge model identity.

    Returns:
        Production-shaped provider response.
    """
    return openai_responses_response(
        {
            "id": "resp_unused",
            "object": "response",
            "created_at": 1.0,
            "status": "completed",
            "model": "judge-model",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "type": "message",
                    "id": "msg_unused",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "unused", "annotations": []}],
                }
            ],
        },
        configured_model=model,
        latency_seconds=0.1,
    )
