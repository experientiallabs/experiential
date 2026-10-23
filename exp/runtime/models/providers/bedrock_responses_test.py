"""Converse response normalization preserves output, stop reasons and cache accounting."""

import pytest

from exp.common.models import BillingSource, ModelFinishReason, ModelSnapshot
from exp.runtime.models.providers.bedrock_responses import converse_response
from exp.runtime.models.providers.errors import (
    ProviderRefusalError,
    ProviderRefusalSignal,
    ProviderResponseError,
)


def _snapshot() -> ModelSnapshot:
    """Build an immutable Bedrock identity fixture."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="bedrock",
        model_id="us.anthropic.claude-sonnet-4-5",
        capabilities_sha256="a" * 64,
        connection_sha256="a" * 64,
    )


def test_converse_response_normalizes_cache_legs_without_double_counting() -> None:
    """Converse inputTokens exclude cache legs, so read and write are added once."""
    response = converse_response(
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "done"},
                        {
                            "toolUse": {
                                "toolUseId": "call-new",
                                "name": "create_ticket",
                                "input": {"priority": "urgent"},
                            }
                        },
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 4,
                "cacheReadInputTokens": 6,
                "cacheWriteInputTokens": 2,
            },
        },
        configured_model=_snapshot(),
        latency_seconds=0.5,
    )

    assert response.finish_reason == ModelFinishReason.COMPLETED
    assert response.output.content == "done"
    assert response.output.tool_calls[0].call_id == "call-new"
    assert response.economics.usage is not None
    assert response.economics.usage.input_tokens == 18
    assert response.economics.usage.cached_input_tokens == 6
    assert response.economics.usage.cache_write_input_tokens == 2


def test_converse_response_reads_past_a_reasoning_block() -> None:
    """A reasoning model's thinking leads its turn and never fails the response."""
    response = converse_response(
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"reasoningContent": {"reasoningText": {"text": "a circle"}}},
                        {"text": "Circle"},
                    ],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 9, "outputTokens": 2},
        },
        configured_model=_snapshot(),
        latency_seconds=0.2,
    )

    assert response.output.content == "Circle"
    assert response.finish_reason == ModelFinishReason.COMPLETED


def test_converse_response_maps_length_and_rejects_unsupported_blocks() -> None:
    """max_tokens becomes length, and unknown content blocks fail closed."""
    length = converse_response(
        {
            "output": {"message": {"content": [{"text": "partial"}]}},
            "stopReason": "max_tokens",
            "usage": {"inputTokens": 3, "outputTokens": 2},
        },
        configured_model=_snapshot(),
        latency_seconds=0.1,
    )
    assert length.finish_reason == ModelFinishReason.LENGTH
    with pytest.raises(ProviderResponseError, match="unsupported block"):
        converse_response(
            {
                "output": {"message": {"content": [{"image": {"format": "png"}}]}},
                "stopReason": "end_turn",
            },
            configured_model=_snapshot(),
            latency_seconds=0.1,
        )
    with pytest.raises(ProviderRefusalError) as refusal_error:
        converse_response(
            {
                "output": {"message": {"content": [{"text": "blocked"}]}},
                "stopReason": "content_filtered",
            },
            configured_model=_snapshot(),
            latency_seconds=0.1,
        )
    assert refusal_error.value.signal is ProviderRefusalSignal.GUARDRAIL
    assert "blocked" not in str(refusal_error.value)
