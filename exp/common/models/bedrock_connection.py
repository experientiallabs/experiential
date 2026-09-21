"""Shared secret-free Bedrock connection validation for catalog and setup."""

from __future__ import annotations

from typing import Literal


def require_bedrock_connection_shape(
    *,
    bedrock_auth_mode: Literal["access_key_pair", "api_key"] | None,
    api_key_env: str | None,
    aws_access_key_id_env: str | None,
    base_url: str | None,
    api_version: str | None,
) -> None:
    """Reject a Bedrock connection whose credential and endpoint fields are inconsistent.

    Args:
        bedrock_auth_mode: Explicit auth mode, or ``None`` to infer it from the env names.
        api_key_env: Environment variable naming the API key or secret access key.
        aws_access_key_id_env: Environment variable naming the access key id.
        base_url: Custom endpoint, which Bedrock never accepts.
        api_version: Azure-only API version, which Bedrock never accepts.

    Raises:
        ValueError: The field combination cannot describe one Bedrock credential source.
    """
    if bedrock_auth_mode == "api_key":
        if api_key_env is None or aws_access_key_id_env is not None:
            raise ValueError(
                "bedrock api_key auth requires api_key_env and forbids aws_access_key_id_env"
            )
    elif bedrock_auth_mode == "access_key_pair":
        if api_key_env is None or aws_access_key_id_env is None:
            raise ValueError(
                "bedrock access_key_pair auth requires both credential environment names"
            )
    elif (api_key_env is None) != (aws_access_key_id_env is None):
        raise ValueError(
            "bedrock explicit access-key auth requires both api_key_env naming the "
            "secret access key and aws_access_key_id_env naming the access key id"
        )
    if base_url is not None:
        raise ValueError("bedrock does not accept base_url")
    if api_version is not None:
        raise ValueError("api_version is only accepted for provider='azure'")
