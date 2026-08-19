"""Tests for environment-only provider secret resolution."""

from __future__ import annotations

import pytest

from wmo.runtime.gateway.secrets import EnvironmentSecretResolver, SecretResolutionError


def test_environment_secret_resolver_accepts_only_populated_env_references() -> None:
    """The resolver returns a current value without supporting inline secret forms."""
    resolver = EnvironmentSecretResolver({"PROVIDER_API_KEY": "canary-secret"})

    assert resolver.resolve("env://PROVIDER_API_KEY") == "canary-secret"
    with pytest.raises(SecretResolutionError, match="env://"):
        resolver.resolve("literal://canary-secret")
    with pytest.raises(SecretResolutionError, match="not populated"):
        resolver.resolve("env://MISSING_API_KEY")
