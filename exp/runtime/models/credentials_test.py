"""Tests for environment-then-store provider credential resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.auth import ProviderAuthStore
from exp.common.models import ConnectionConfig
from exp.runtime.models.credentials import (
    CredentialResolution,
    ModelCredentialError,
    lookup_connection_credential,
    read_connection_api_key,
    resolve_or_prompt_connection_api_key,
)

_SECRET = "sk-resolver-stored-secret"
_ENV_SECRET = "sk-resolver-env-secret"


def _openai(env_name: str = "OPENAI_API_KEY") -> ConnectionConfig:
    """Return one OpenAI connection that names a credential environment variable.

    Args:
        env_name: Environment-variable name configured on the connection.

    Returns:
        Secret-free OpenAI connection metadata.
    """
    return ConnectionConfig(provider="openai", api_key_env=env_name)


def test_environment_overrides_store_without_rewriting_it(tmp_path: Path) -> None:
    """A non-empty environment value wins and leaves the stored credential unchanged."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    store.put("openai", _SECRET)
    connection = _openai()

    resolved = lookup_connection_credential(
        connection,
        connection_id="openai",
        environment={"OPENAI_API_KEY": _ENV_SECRET},
        store=store,
    )

    assert resolved is not None
    assert resolved.value == _ENV_SECRET
    assert resolved.source == "environment"
    assert store.get("openai") == _SECRET


def test_empty_environment_falls_through_to_the_store(tmp_path: Path) -> None:
    """Whitespace-only environment values do not hide a stored credential."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    store.put("openai", _SECRET)

    resolved = lookup_connection_credential(
        _openai(),
        connection_id="openai",
        environment={"OPENAI_API_KEY": "   "},
        store=store,
    )

    assert resolved is not None
    assert resolved.value == _SECRET
    assert resolved.source == "stored"


def test_fresh_resolver_reads_the_store_written_by_another_instance(tmp_path: Path) -> None:
    """A new store and resolver pair still sees the persisted connection key."""
    path = tmp_path / "auth.json"
    ProviderAuthStore(path).put("openai", _SECRET)

    api_key = read_connection_api_key(
        _openai(),
        connection_id="openai",
        environment={},
        store=ProviderAuthStore(path),
    )

    assert api_key == _SECRET


def test_store_resolves_when_the_connection_omits_an_env_name(tmp_path: Path) -> None:
    """A stored key is enough when the catalog has no environment override name."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    store.put("acme", _SECRET)
    connection = ConnectionConfig(
        provider="openai-compatible",
        base_url="https://acme.example/v1",
    )

    api_key = read_connection_api_key(
        connection,
        connection_id="acme",
        environment={},
        store=store,
    )

    assert api_key == _SECRET


def test_same_provider_connections_resolve_independently(tmp_path: Path) -> None:
    """Two OpenAI connections with the same env name keep distinct stored keys."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    store.put("openai", _SECRET)
    store.put("openai-work", _ENV_SECRET)
    connection = _openai()

    first = read_connection_api_key(connection, connection_id="openai", environment={}, store=store)
    second = read_connection_api_key(
        connection, connection_id="openai-work", environment={}, store=store
    )

    assert first == _SECRET
    assert second == _ENV_SECRET


def test_missing_environment_and_store_fails_without_prompting() -> None:
    """Noninteractive resolution names the env var and login command, never a secret."""
    with pytest.raises(ModelCredentialError, match="OPENAI_API_KEY") as captured:
        read_connection_api_key(_openai(), connection_id="openai", environment={})

    message = str(captured.value)
    assert "exp auth login openai" in message
    assert _SECRET not in message


def test_prompt_persists_without_writing_the_environment(tmp_path: Path) -> None:
    """A pasted key is stored for the connection and not copied into the environment map."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    environment: dict[str, str] = {}

    api_key = resolve_or_prompt_connection_api_key(
        _openai(),
        connection_id="openai",
        environment=environment,
        store=store,
        prompt=lambda: _SECRET,
    )

    assert api_key == _SECRET
    assert environment == {}
    assert ProviderAuthStore(tmp_path / "auth.json").get("openai") == _SECRET


def test_prompt_is_skipped_when_environment_or_store_already_resolves(tmp_path: Path) -> None:
    """Interactive resolution does not prompt when a higher-precedence source exists."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    store.put("openai", _SECRET)
    prompted = {"called": False}

    def _forbidden() -> str:
        """Fail if the prompt is reached.

        Returns:
            A dummy secret that must never be used.
        """
        prompted["called"] = True
        return "should-not-prompt"

    api_key = resolve_or_prompt_connection_api_key(
        _openai(),
        connection_id="openai",
        environment={},
        store=store,
        prompt=_forbidden,
    )

    assert api_key == _SECRET
    assert prompted["called"] is False


def test_force_prompt_replaces_the_stored_credential(tmp_path: Path) -> None:
    """An explicit replacement prompt overwrites only that connection's stored key."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    store.put("openai", _SECRET)
    store.put("openai-work", _ENV_SECRET)

    api_key = resolve_or_prompt_connection_api_key(
        _openai(),
        connection_id="openai",
        environment={"OPENAI_API_KEY": "rejected-in-session"},
        store=store,
        prompt=lambda: "replacement-secret",
        force_prompt=True,
    )

    assert api_key == "replacement-secret"
    assert store.get("openai") == "replacement-secret"
    assert store.get("openai-work") == _ENV_SECRET


def test_empty_prompt_skips_the_connection(tmp_path: Path) -> None:
    """An empty paste skips the provider without writing a stored credential."""
    store = ProviderAuthStore(tmp_path / "auth.json")

    api_key = resolve_or_prompt_connection_api_key(
        _openai(),
        connection_id="openai",
        environment={},
        store=store,
        prompt=lambda: "",
    )

    assert api_key is None
    assert store.get("openai") is None


def test_bedrock_lookup_never_uses_the_store(tmp_path: Path) -> None:
    """Bedrock stays on the AWS credential chain and ignores stored API keys."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    store.put("bedrock", _SECRET)
    connection = ConnectionConfig(provider="bedrock", region="us-east-1")

    assert (
        lookup_connection_credential(
            connection, connection_id="bedrock", environment={}, store=store
        )
        is None
    )


def test_stored_key_is_rejected_when_the_endpoint_identity_changes(tmp_path: Path) -> None:
    """Reusing a connection ID for another OpenAI-compatible endpoint does not leak the key."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    original = ConnectionConfig(
        provider="openai-compatible",
        base_url="https://acme.example/v1",
    )
    resolve_or_prompt_connection_api_key(
        original,
        connection_id="acme",
        environment={},
        store=store,
        prompt=lambda: _SECRET,
    )
    replacement = ConnectionConfig(
        provider="openai-compatible",
        base_url="https://other.example/v1",
    )

    with pytest.raises(ModelCredentialError, match="does not match") as captured:
        read_connection_api_key(
            replacement,
            connection_id="acme",
            environment={},
            store=store,
        )

    assert _SECRET not in str(captured.value)
    assert store.get("acme") == _SECRET


def test_resolution_repr_never_includes_the_secret() -> None:
    """Public representations of a resolved credential stay redacted."""
    resolved = CredentialResolution(_SECRET, "stored")

    assert _SECRET not in repr(resolved)
    assert _SECRET not in str(resolved)
    assert resolved.value == _SECRET
    assert resolved.source == "stored"
