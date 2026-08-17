"""Automatic-router approved-judge execution tests."""

from __future__ import annotations

from wmo.common.models import (
    ModelCapabilities,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    completion_cost_reservation,
)
from wmo.optimize.router.automatic.judge import ReservedJudgeClient
from wmo.runtime.models.providers.openai import openai_responses_response


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
