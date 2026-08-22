"""Resolve one provider API key from environment, then the user-local credential store.

Runtime and CI callers never prompt. Interactive ``exp config providers`` setup can
persist a pasted key for the exact connection ID.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Literal

from exp.common.auth import (
    ProviderAuthStore,
    ProviderAuthStoreError,
    StoredCredentialBinding,
    StoredCredentialEndpointMismatch,
    StoredCredentialStatus,
)
from exp.common.models import ConnectionConfig

CredentialSource = Literal["environment", "stored", "prompt"]
CredentialPrompt = Callable[[], str | None]


class ModelCredentialError(ValueError):
    """A configured model connection could not resolve its credential."""


class CredentialResolution:
    """One resolved API key plus the source that supplied it.

    The secret is omitted from ``repr`` and ``str``.
    """

    def __init__(self, value: str, source: CredentialSource) -> None:
        """Bind one non-empty secret and its source.

        Args:
            value: Resolved API key.
            source: Whether the key came from the environment, store, or a prompt.
        """
        self._value = value
        self.source = source

    @property
    def value(self) -> str:
        """Return the resolved API key."""
        return self._value

    def __repr__(self) -> str:
        """Describe the source without the secret."""
        return f"CredentialResolution(source={self.source!r}, value='[REDACTED]')"

    def __str__(self) -> str:
        """Describe the source without the secret."""
        return self.__repr__()


def lookup_connection_credential(
    connection: ConnectionConfig,
    *,
    connection_id: str,
    environment: Mapping[str, str] | None = None,
    store: ProviderAuthStore | None = None,
) -> CredentialResolution | None:
    """Resolve one credential from the environment, then the stored connection record.

    A non-empty environment value wins and does not rewrite the store. Bedrock connections
    do not participate; they authenticate through the AWS credential chain.

    Args:
        connection: Secret-free connection metadata.
        connection_id: Exact catalog or gateway connection name used as the store key.
        environment: Optional mapping used by deterministic tests instead of process environment.
        store: Optional credential store. When omitted, the platform user-data file is used.

    Returns:
        The resolved secret and source, or ``None`` when neither source has a value.

    Raises:
        ProviderAuthStoreError: The local credential file exists but cannot be used.
    """
    if connection.provider == "bedrock":
        return None
    values = os.environ if environment is None else environment
    if connection.api_key_env is not None:
        env_value = (values.get(connection.api_key_env) or "").strip()
        if env_value:
            return CredentialResolution(env_value, "environment")
    auth_store = store if store is not None else ProviderAuthStore()
    stored = auth_store.get(connection_id, binding=_credential_binding(connection))
    if stored:
        return CredentialResolution(stored, "stored")
    return None


def describe_connection_credential(
    connection: ConnectionConfig,
    *,
    connection_id: str,
    environment: Mapping[str, str] | None = None,
    store: ProviderAuthStore | None = None,
) -> StoredCredentialStatus:
    """Return public credential metadata for one connection without the secret.

    Args:
        connection: Secret-free connection metadata.
        connection_id: Exact catalog or gateway connection name.
        environment: Optional mapping used instead of the process environment.
        store: Optional credential store. When omitted, the platform user-data file is used.

    Returns:
        Connection identity, provider, source, and environment-variable name.
    """
    if connection.provider == "bedrock":
        return StoredCredentialStatus(
            connection_id=connection_id,
            provider=connection.provider,
            source="aws_chain",
        )
    try:
        resolved = lookup_connection_credential(
            connection,
            connection_id=connection_id,
            environment=environment,
            store=store,
        )
    except StoredCredentialEndpointMismatch:
        return StoredCredentialStatus(
            connection_id=connection_id,
            provider=connection.provider,
            source="mismatch",
            environment_variable=connection.api_key_env,
        )
    source: Literal["environment", "stored", "missing"]
    if resolved is None:
        source = "missing"
    elif resolved.source == "environment":
        source = "environment"
    else:
        source = "stored"
    return StoredCredentialStatus(
        connection_id=connection_id,
        provider=connection.provider,
        source=source,
        environment_variable=connection.api_key_env,
    )


class MissingModelCredentialError(ModelCredentialError):
    """A configured credential is absent from both the environment and local store."""

    def __init__(self, environment_variable: str, *, connection_id: str) -> None:
        """Record the missing references without reading or exposing a secret.

        Args:
            environment_variable: Name of the configured credential environment variable.
            connection_id: Exact catalog or gateway connection name checked in the store.
        """
        self.environment_variable = environment_variable
        self.connection_id = connection_id
        self.detail = (
            f"connection credential environment variable {environment_variable!r} is not set "
            f"and no stored credential exists for connection {connection_id!r}"
        )
        super().__init__(f"{self.detail}; run 'exp config providers' or export the variable")


def read_connection_api_key(
    connection: ConnectionConfig,
    *,
    connection_id: str,
    environment: Mapping[str, str] | None = None,
    store: ProviderAuthStore | None = None,
) -> str:
    """Read one configured key without exposing its value in an exception.

    Args:
        connection: Secret-free connection metadata, including an optional environment name.
        connection_id: Exact catalog or gateway connection name used as the store key.
        environment: Optional mapping used by deterministic tests instead of process environment.
        store: Optional credential store. When omitted, the platform user-data file is used.

    Returns:
        The non-empty credential value.

    Raises:
        ModelCredentialError: The connection does not name a key variable, or both the
            environment variable and stored credential are absent.
        ProviderAuthStoreError: The local credential file exists but cannot be used.
    """
    if connection.provider == "bedrock":
        raise ModelCredentialError(
            "bedrock authenticates through the AWS credential chain and has no stored API key"
        )
    try:
        resolved = lookup_connection_credential(
            connection,
            connection_id=connection_id,
            environment=environment,
            store=store,
        )
    except ProviderAuthStoreError as exc:
        raise ModelCredentialError(str(exc)) from exc
    if resolved is None:
        if connection.api_key_env is None:
            raise ModelCredentialError(
                f"no stored credential exists for connection {connection_id!r}; "
                "run 'exp config providers' to supply one"
            )
        raise MissingModelCredentialError(
            connection.api_key_env,
            connection_id=connection_id,
        )
    return resolved.value


def resolve_or_prompt_connection_api_key(
    connection: ConnectionConfig,
    *,
    connection_id: str,
    environment: Mapping[str, str] | None = None,
    store: ProviderAuthStore | None = None,
    prompt: CredentialPrompt,
    persist: bool = True,
    force_prompt: bool = False,
) -> str | None:
    """Resolve from environment or store, or accept a masked prompt and persist it.

    Args:
        connection: Secret-free connection metadata.
        connection_id: Exact catalog or gateway connection name used as the store key.
        environment: Optional mapping used by deterministic tests instead of process environment.
        store: Optional credential store. When omitted, the platform user-data file is used.
        prompt: Interactive callback that returns a pasted key or ``None`` to skip.
        persist: Whether a newly pasted key is written to the store.
        force_prompt: Whether to ignore current environment and stored values and ask again.

    Returns:
        The resolved or pasted key, or ``None`` when the operator skips the provider.

    Raises:
        ProviderAuthStoreError: The local credential file exists but cannot be used.
    """
    auth_store = store if store is not None else ProviderAuthStore()
    if not force_prompt:
        resolved = lookup_connection_credential(
            connection,
            connection_id=connection_id,
            environment=environment,
            store=auth_store,
        )
        if resolved is not None:
            return resolved.value
    pasted = prompt()
    if pasted is None:
        return None
    key = pasted.strip()
    if not key:
        return None
    if persist:
        auth_store.put(connection_id, key, binding=_credential_binding(connection))
    return key


def _credential_binding(connection: ConnectionConfig) -> StoredCredentialBinding:
    """Return the secret-free endpoint identity stored with one API key.

    Args:
        connection: Secret-free connection metadata.

    Returns:
        Provider name plus the catalog endpoint digest.
    """
    return StoredCredentialBinding(
        provider=connection.provider,
        endpoint_sha256=connection.identity_sha256(),
    )
