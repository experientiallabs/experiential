"""Translate storage-neutral provider commands into SQLite environment locators."""

from __future__ import annotations

from exp.common.models import ConnectionConfig
from exp.runtime.gateway.platform import (
    OpaqueSecretReference,
    OpaqueSecretScheme,
    UpsertProviderConnectionCommand,
)


def sqlite_connection_config(command: UpsertProviderConnectionCommand) -> ConnectionConfig:
    """Return the secret-free connection accepted by the local SQLite authority."""
    references = (command.secret_reference, command.access_key_id_reference)
    if any(
        reference is not None and reference.scheme is not OpaqueSecretScheme.ENVIRONMENT
        for reference in references
    ):
        raise ValueError(
            "SQLite provider connections currently require an environment secret reference"
        )
    return ConnectionConfig(
        provider=command.provider,
        base_url=command.base_url,
        api_key_env=_environment_name(command.secret_reference),
        api_version=command.api_version,
        azure_api_surface=command.azure_api_surface,
        region=command.region,
        aws_access_key_id_env=_environment_name(command.access_key_id_reference),
        bedrock_auth_mode=command.bedrock_auth_mode,
    )


def _environment_name(reference: OpaqueSecretReference | None) -> str | None:
    """Return one already-validated environment locator name."""
    return None if reference is None else reference.reference
