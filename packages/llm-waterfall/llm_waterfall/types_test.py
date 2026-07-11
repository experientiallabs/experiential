"""Tests for public value types."""

from __future__ import annotations

import pytest

from llm_waterfall.types import (
    Backend,
    ChatRequest,
    ChatResponse,
    Message,
    RetryPolicy,
    normalize_messages,
)


def test_backend_positional_call_site() -> None:
    b = Backend("bedrock", "us.anthropic.claude-opus-4-6-v1", profile="endflow", region="us-west-1")
    assert b.provider == "bedrock"
    assert b.model == "us.anthropic.claude-opus-4-6-v1"
    assert b.profile == "endflow"
    assert b.read_timeout_s == 600.0


def test_backend_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        Backend("gcp", "gemini")


def test_backend_is_frozen_and_hashable() -> None:
    b = Backend("openai", "gpt-5.5")
    with pytest.raises(AttributeError):
        b.model = "other"  # ty: ignore[invalid-assignment]
    assert hash(b) == hash(Backend("openai", "gpt-5.5"))


def test_retry_policy_backoff_sequence() -> None:
    p = RetryPolicy(rounds=5, backoff_base_s=15.0, backoff_max_s=120.0)
    assert [p.backoff_before_round(n) for n in range(1, 6)] == [0.0, 15.0, 30.0, 60.0, 120.0]


def test_retry_policy_rejects_zero_rounds() -> None:
    with pytest.raises(ValueError, match="rounds"):
        RetryPolicy(rounds=0)


def test_normalize_messages_accepts_dicts_and_typed() -> None:
    typed = Message(role="user", content="hi")
    out = normalize_messages([typed, {"role": "assistant", "content": "yo"}])
    assert all(isinstance(m, Message) for m in out)
    assert [m.role for m in out] == ["user", "assistant"]
    assert out[0] is typed


def test_structured_chat_normalizes_provider_payload_without_losing_tools() -> None:
    request = ChatRequest.model_validate(
        {
            "model": "placeholder",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "bash", "parameters": {"type": "object"}},
                }
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": 2048,
            "store": False,
        }
    )

    payload = request.provider_payload("routed-model")

    assert payload["model"] == "routed-model"
    assert payload["stream"] is False
    assert payload["max_completion_tokens"] == 2048
    assert "max_tokens" not in payload
    assert "stream_options" not in payload
    assert payload["tools"]
    assert payload["store"] is False


def test_structured_chat_can_target_legacy_max_tokens_backends() -> None:
    request = ChatRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 2048,
        }
    )

    payload = request.provider_payload("compatible-model", max_tokens_field="max_tokens")

    assert payload["max_tokens"] == 2048
    assert "max_completion_tokens" not in payload


def test_structured_response_preserves_tool_calls_and_usage() -> None:
    response = ChatResponse.model_validate(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }
    )

    assert response.choices[0].message.tool_calls is not None
    assert response.choices[0].message.tool_calls[0].function.name == "bash"
    assert response.token_usage().input_tokens == 10
