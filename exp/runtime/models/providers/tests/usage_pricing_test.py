"""SDK adapters retain the pricing subsets emitted by direct providers and gateway relays."""

import json

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
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
        usage["cache_creation"] = {
            "ephemeral_5m_input_tokens": 30 - hour,
            "ephemeral_1h_input_tokens": hour,
        }
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


@pytest.mark.parametrize("hour", [True, False, "2", 2.0, -1])
def test_responses_ttl_extra_requires_an_actual_nonnegative_integer(hour: JsonValue) -> None:
    """An untyped SDK extra cannot coerce a boolean, string or float into billed usage."""
    payload = _relay_payload(responses=True)
    usage = payload["usage"]
    assert isinstance(usage, dict)
    usage.pop("unreported_token_details")
    details = usage["input_tokens_details"]
    assert isinstance(details, dict)
    details["cache_write_tokens"] = 4
    details["cache_write_1h_tokens"] = hour
    with pytest.raises(ProviderResponseError, match="observability"):
        openai_responses_response(payload, configured_model=_snapshot(), latency_seconds=1)


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize(
    ("outputs", "reasoning", "total", "expected"),
    [(900, 300, 1214, 1200), (900, 300, 914, 900), (2, 3, 0, 5), (900, 300, 0, 900), (2, 3, 16, 2)],
)
def test_openai_shaped_additive_reasoning_matches_native(
    responses: bool, outputs: int, reasoning: int, total: int | None, expected: int
) -> None:
    """Provider totals disambiguate additive reasoning; native and completed paths agree."""
    payload = _relay_payload(responses=responses)
    usage = payload["usage"]
    assert isinstance(usage, dict)
    usage.pop("unreported_token_details")
    usage["input_tokens" if responses else "prompt_tokens"] = 14
    usage["output_tokens" if responses else "completion_tokens"] = outputs
    if total is None:
        usage.pop("total_tokens")
    else:
        usage["total_tokens"] = total
    details = usage["output_tokens_details" if responses else "completion_tokens_details"]
    assert isinstance(details, dict)
    details["reasoning_tokens"] = reasoning
    parse = openai_responses_response if responses else openai_compatible_response
    observed = parse(payload, configured_model=_snapshot(), latency_seconds=1).economics.usage
    assert observed is not None and observed.output_tokens == expected
    assert observed.reasoning_tokens == reasoning
    native = pytest.importorskip("exp_gateway_native")
    if responses:
        event = {"type": "response.completed", "response": payload}
        dialect = "openai_responses"
    else:
        event = {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}], "usage": usage}
        dialect = "openai_compatible"
    frame = "data: " + json.dumps(event) + "\n\n"
    normalized = json.loads(native.normalize_stream_fixture(dialect, json.dumps([frame])))
    assert normalized["failure"] is None
    meters = [event for event in normalized["events"] if event["kind"] == "usage"]
    assert len(meters) == 1 and meters[0]["output_tokens"] == expected


@pytest.mark.parametrize(
    ("creation", "expected_hour"),
    [
        (None, None),
        ({}, None),
        ({"ephemeral_5m_input_tokens": 25}, None),
        ({"ephemeral_1h_input_tokens": 5}, None),
        ({"ephemeral_5m_input_tokens": None, "ephemeral_1h_input_tokens": 5}, None),
        ({"ephemeral_5m_input_tokens": 25, "ephemeral_1h_input_tokens": None}, None),
        ({"ephemeral_5m_input_tokens": 25, "ephemeral_1h_input_tokens": 5}, 5),
        ({"ephemeral_5m_input_tokens": 30, "ephemeral_1h_input_tokens": 0}, 0),
    ],
)
def test_messages_ttl_breakdown_matches_native_observation(
    creation: JsonValue, expected_hour: int | None
) -> None:
    """Only a complete consistent split proves TTL allocation on either provider path."""
    usage: JsonObject = {
        "input_tokens": 60,
        "output_tokens": 20,
        "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 30,
        "cache_creation": creation,
    }
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
    assert response.economics.usage is not None
    assert response.economics.usage.cache_write_1h_input_tokens == expected_hour
    normalized = _native_messages_usage(usage)
    assert normalized["failure"] is None
    events = normalized["events"]
    assert isinstance(events, list)
    meters = [event for event in events if isinstance(event, dict) and event["kind"] == "usage"]
    assert meters[-1].get("cache_creation_1h_input_tokens") == expected_hour


@pytest.mark.parametrize(
    "creation",
    [
        7,
        {"ephemeral_5m_input_tokens": -1},
        {"ephemeral_1h_input_tokens": True},
        {"ephemeral_5m_input_tokens": "25"},
        {"ephemeral_5m_input_tokens": 31},
        {"ephemeral_1h_input_tokens": 31},
        {"ephemeral_5m_input_tokens": 25, "ephemeral_1h_input_tokens": 4},
        {"ephemeral_5m_input_tokens": 25, "ephemeral_1h_input_tokens": 6},
    ],
)
def test_messages_ttl_breakdown_rejects_malformed_or_contradictory_evidence(
    creation: JsonValue,
) -> None:
    """Malformed partial counts and contradictory complete totals never acquire a price."""
    usage: JsonObject = {
        "input_tokens": 60,
        "output_tokens": 20,
        "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 30,
        "cache_creation": creation,
    }
    with pytest.raises(ProviderResponseError, match="cache_creation"):
        anthropic_messages_response(
            {
                "model": "fixture",
                "content": [{"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
                "usage": usage,
            },
            configured_model=_snapshot("anthropic"),
            latency_seconds=1,
        )
    assert _native_messages_usage(usage)["failure"] is not None


def _native_messages_usage(usage: JsonObject) -> JsonObject:
    """Observe identical usage through the existing compiled Messages normalizer."""
    native = pytest.importorskip("exp_gateway_native")
    events: list[JsonObject] = [
        {"type": "message_start", "message": {"usage": usage}},
        {"type": "message_stop"},
    ]
    frames = ["data: " + json.dumps(event) + "\n\n" for event in events]
    result = json.loads(native.normalize_stream_fixture("anthropic_messages", json.dumps(frames)))
    assert isinstance(result, dict)
    return result
