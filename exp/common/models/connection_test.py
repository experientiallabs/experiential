"""Tests for secret-free provider connection metadata, including plan sign-in connections."""

from __future__ import annotations

import pytest

from exp.common.models.connection import SUBSCRIPTION_PROVIDERS, ConnectionConfig


def test_chatgpt_subscription_is_a_bare_openai_connection() -> None:
    """A plan connection names its provider and kind and nothing else."""
    connection = ConnectionConfig(provider="openai", subscription="chatgpt")

    assert connection.api_key_env is None
    assert connection.base_url is None
    assert SUBSCRIPTION_PROVIDERS[connection.subscription or "chatgpt"] == "openai"


def test_subscription_rejects_another_provider() -> None:
    """A chatgpt plan is a sign-in for openai only."""
    with pytest.raises(ValueError, match="sign-in for provider 'openai'"):
        ConnectionConfig(provider="anthropic", subscription="chatgpt")


def test_subscription_rejects_a_credential_name() -> None:
    """A plan sign-in cannot double as an API-key connection."""
    with pytest.raises(ValueError, match="omit api_key_env"):
        ConnectionConfig(provider="openai", subscription="chatgpt", api_key_env="OPENAI_API_KEY")


def test_subscription_rejects_endpoint_overrides() -> None:
    """A plan sign-in reaches its fixed backend and cannot point elsewhere."""
    with pytest.raises(ValueError, match="omit base_url"):
        ConnectionConfig(
            provider="openai",
            subscription="chatgpt",
            base_url="https://example.com/v1",
            trusted_custom_origin=True,
        )


def test_subscription_identity_differs_from_the_api_key_origin() -> None:
    """The plan backend is a different endpoint, so stored sign-ins never bind to API-key rows."""
    plan = ConnectionConfig(provider="openai", subscription="chatgpt")
    key = ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY")

    assert plan.identity_sha256() != key.identity_sha256()
    assert (
        plan.identity_sha256()
        == ConnectionConfig(provider="openai", subscription="chatgpt").identity_sha256()
    )


def test_serialization_omits_an_absent_subscription_and_keeps_a_present_one() -> None:
    """API-key connections keep their exact persisted bytes; plan connections carry the kind."""
    key = ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY").model_dump(mode="json")
    plan = ConnectionConfig(provider="openai", subscription="chatgpt").model_dump(mode="json")

    assert "subscription" not in key
    assert plan["subscription"] == "chatgpt"
    assert ConnectionConfig.model_validate(plan) == ConnectionConfig(
        provider="openai", subscription="chatgpt"
    )
