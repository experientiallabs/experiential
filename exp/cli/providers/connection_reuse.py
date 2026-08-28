"""Configured provider-connection reuse rules for interactive setup."""

from typing import Literal

from exp.common.models import ConnectionConfig
from exp.common.models import ProviderConnection


def reused_connection(
    existing_connections: tuple[ProviderConnection, ...],
    *,
    provider: str,
    api_key_env: str | None,
    base_url: str | None,
    api_version: str | None,
    azure_api_surface: Literal["openai_deployments", "model_inference"] | None,
    region: str | None,
    aws_access_key_id_env: str | None = None,
    bedrock_auth_mode: str | None = None,
    require_ambient_bedrock_auth: bool = False,
) -> ProviderConnection | None:
    """Return the sole configured connection that semantically matches these fields."""
    matches: list[ProviderConnection] = []
    unspecified_bedrock_auth = (
        provider == "bedrock"
        and api_key_env is None
        and aws_access_key_id_env is None
        and bedrock_auth_mode is None
    )
    candidate = ConnectionConfig(
        provider=provider,
        base_url=base_url,
        api_key_env=api_key_env,
        api_version=api_version,
        azure_api_surface=azure_api_surface,
        region=region,
        aws_access_key_id_env=aws_access_key_id_env,
        bedrock_auth_mode=bedrock_auth_mode,
    )
    for connection in existing_connections:
        if require_ambient_bedrock_auth and (
            connection.api_key_env is not None
            or connection.aws_access_key_id_env is not None
            or connection.bedrock_auth_mode is not None
        ):
            continue
        configured = connection.catalog_config()
        if unspecified_bedrock_auth:
            configured = configured.model_copy(
                update={
                    "api_key_env": None,
                    "aws_access_key_id_env": None,
                    "bedrock_auth_mode": None,
                }
            )
        if configured.identity_sha256() != candidate.identity_sha256():
            continue
        if (
            not unspecified_bedrock_auth
            and api_key_env is not None
            and connection.api_key_env != api_key_env
        ):
            continue
        matches.append(connection)
    return matches[0] if len(matches) == 1 else None
