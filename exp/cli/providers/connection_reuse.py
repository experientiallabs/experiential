"""Configured provider-connection reuse rules for interactive setup."""

from exp.common.models import ProviderConnection


def reused_connection(
    existing_connections: tuple[ProviderConnection, ...],
    *,
    provider: str,
    api_key_env: str | None,
    base_url: str | None,
    api_version: str | None,
    region: str | None,
    aws_access_key_id_env: str | None,
    bedrock_auth_mode: str | None,
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
    for connection in existing_connections:
        if require_ambient_bedrock_auth and (
            connection.api_key_env is not None
            or connection.aws_access_key_id_env is not None
            or connection.bedrock_auth_mode is not None
        ):
            continue
        if (
            connection.provider != provider
            or connection.base_url != base_url
            or connection.api_version != api_version
            or connection.region != region
            or (
                not unspecified_bedrock_auth
                and connection.aws_access_key_id_env != aws_access_key_id_env
            )
            or (not unspecified_bedrock_auth and connection.bedrock_auth_mode != bedrock_auth_mode)
        ):
            continue
        if (
            not unspecified_bedrock_auth
            and api_key_env is not None
            and connection.api_key_env != api_key_env
        ):
            continue
        matches.append(connection)
    return matches[0] if len(matches) == 1 else None
