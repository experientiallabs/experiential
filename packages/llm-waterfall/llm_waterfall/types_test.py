"""Tests for public value types."""

from __future__ import annotations

import pytest

from llm_waterfall.types import Backend, Message, RetryPolicy, normalize_messages


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
