"""Tests for canonical and generated credential environment names."""

from __future__ import annotations

from exp.common.auth.env_names import CANONICAL_API_KEY_ENV, derived_api_key_env


def test_known_providers_keep_their_canonical_override_name() -> None:
    """Native providers always resolve to the documented environment variable."""
    assert derived_api_key_env("openai", "openai-work") == "OPENAI_API_KEY"
    assert derived_api_key_env("anthropic", "anthropic") == CANONICAL_API_KEY_ENV["anthropic"]
    assert derived_api_key_env("tinker", "tinker") == "TINKER_API_KEY"


def test_openai_compatible_generates_a_name_from_the_connection_id() -> None:
    """The first compatible connection keeps the canonical name; extras derive from the ID."""
    assert derived_api_key_env("openai-compatible", "openai-compatible") == (
        "OPENAI_COMPATIBLE_API_KEY"
    )
    assert derived_api_key_env("openai-compatible", "openai-compatible-2") == (
        "OPENAI_COMPATIBLE_2_API_KEY"
    )
    assert derived_api_key_env("openai-compatible", "acme-gateway") == "ACME_GATEWAY_API_KEY"


def test_bedrock_has_no_api_key_environment_variable() -> None:
    """Bedrock stays on the AWS credential chain."""
    assert derived_api_key_env("bedrock", "bedrock") is None
