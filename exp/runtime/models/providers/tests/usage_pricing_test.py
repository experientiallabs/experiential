"""SDK adapters retain the pricing subsets emitted by direct providers and gateway relays."""

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import Usage
from exp.runtime.models.providers.anthropic import anthropic_messages_response
from exp.runtime.models.providers.errors import ProviderResponseError
from exp.runtime.models.providers.openai import openai_responses_response
from exp.runtime.models.providers.openai_compatible import openai_compatible_response
from exp.runtime.models.providers.openai_compatible_test import _snapshot


def _relay_payload(*, responses: bool, marked_count: int = 0) -> JsonObject:
    """Build a compatible completed response with one explicitly unobserved meter."""
    details: JsonObject = {
        "cached_tokens": 0,
        "cache_write_tokens": marked_count,
        "cache_write_1h_tokens": 0,
    }
    usage: JsonObject = {
        "input_tokens" if responses else "prompt_tokens": 100,
        "output_tokens" if responses else "completion_tokens": 20,
        "total_tokens": 120,
        "input_tokens_details" if responses else "prompt_tokens_details": details,
        "output_tokens_details" if responses else "completion_tokens_details": {
            "reasoning_tokens": 0
        },
        "unreported_token_details": [
            "cached_tokens",
            "cache_write_tokens",
            "cache_write_1h_tokens",
            "reasoning_tokens",
        ],
    }
    if not responses:
        return {
            "model": "fixture",
            "choices": [{"message": {"content": "done"}}],
            "usage": usage,
        }
    return {
        "id": "resp_fixture",
        "object": "response",
        "created_at": 1.0,
        "status": "completed",
        "model": "fixture",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "type": "message",
                "id": "msg_fixture",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "done", "annotations": []}],
            }
        ],
        "usage": usage,
    }


@pytest.mark.parametrize("responses", [False, True])
def test_compatible_zero_placeholders_restore_unknown_sdk_usage(responses: bool) -> None:
    """An explicit relay marker survives typed SDK parsing instead of certifying free subsets."""
    parse = openai_responses_response if responses else openai_compatible_response
    response = parse(
        _relay_payload(responses=responses), configured_model=_snapshot(), latency_seconds=1
    )
    assert response.economics.usage == Usage(input_tokens=100, output_tokens=20)


@pytest.mark.parametrize("responses", [False, True])
def test_relay_unknown_marker_cannot_hide_an_observed_positive_subset(responses: bool) -> None:
    """A contradictory count fails closed before entering report economics."""
    parse = openai_responses_response if responses else openai_compatible_response
    with pytest.raises(ProviderResponseError, match="unreported token detail|observability"):
        parse(
            _relay_payload(responses=responses, marked_count=1),
            configured_model=_snapshot(),
            latency_seconds=1,
        )


@pytest.mark.parametrize("hour", [None, 0, 5])
@pytest.mark.parametrize("responses", [False, True])
def test_openai_usage_subsets_survive_sdk_conversion(hour: int | None, responses: bool) -> None:
    """Both public wire dialects distinguish absent TTL allocation from actual zero."""
    details: JsonObject = {"cached_tokens": 10, "cache_write_tokens": 30}
    if hour is not None:
        details["cache_write_1h_tokens"] = hour
    if responses:
        payload: JsonObject = {
            "id": "resp_fixture",
            "object": "response",
            "created_at": 1.0,
            "status": "completed",
            "model": "fixture",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "service_tier": "flex",
            "output": [
                {
                    "type": "message",
                    "id": "msg_fixture",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "done", "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_tokens_details": details,
                "output_tokens_details": {"reasoning_tokens": 7},
            },
        }
        response = openai_responses_response(
            payload, configured_model=_snapshot(), latency_seconds=1
        )
    else:
        response = openai_compatible_response(
            {
                "model": "fixture",
                "choices": [{"message": {"content": "done"}}],
                "service_tier": "flex",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_tokens_details": details,
                    "completion_tokens_details": {"reasoning_tokens": 7},
                },
            },
            configured_model=_snapshot(),
            latency_seconds=1,
        )
    assert response.economics.usage == Usage(
        input_tokens=100,
        output_tokens=20,
        cached_input_tokens=10,
        cache_write_input_tokens=30,
        cache_write_1h_input_tokens=hour,
        reasoning_tokens=7,
        service_tier="flex",
    )


@pytest.mark.parametrize("hour", [None, 0, 5])
def test_messages_usage_keeps_known_and_unknown_one_hour_writes(hour: int | None) -> None:
    """Anthropic writes are additive to ordinary input, while TTL stays within writes."""
    usage: JsonObject = {
        "input_tokens": 60,
        "output_tokens": 20,
        "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 30,
    }
    if hour is not None:
        usage["cache_creation"] = {"ephemeral_1h_input_tokens": hour}
    response = anthropic_messages_response(
        {
            "model": "fixture",
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
            "usage": usage,
        },
        configured_model=_snapshot("anthropic"),
        latency_seconds=1,
    )
    assert response.economics.usage == Usage(
        input_tokens=100,
        output_tokens=20,
        cached_input_tokens=10,
        cache_write_input_tokens=30,
        cache_write_1h_input_tokens=hour,
    )
