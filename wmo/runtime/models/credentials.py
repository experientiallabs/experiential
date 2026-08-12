"""Credential lookup that reads only a named environment variable at construction time."""

from __future__ import annotations

import os
from collections.abc import Mapping

from wmo.common.models import ConnectionConfig


class ModelCredentialError(ValueError):
    """A configured model connection could not resolve its named credential."""


def read_connection_api_key(
    connection: ConnectionConfig,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Read one configured key without exposing its value in an exception.

    Args:
        connection: Local connection metadata containing only an environment variable name.
        environment: Optional mapping used by deterministic tests instead of process environment.

    Returns:
        The non-empty credential value.

    Raises:
        ModelCredentialError: The connection does not name a key variable or it is unset.
    """
    if connection.api_key_env is None:
        raise ModelCredentialError(
            f"connection provider {connection.provider!r} needs api_key_env in .wmo/models.toml"
        )
    values = os.environ if environment is None else environment
    api_key = values.get(connection.api_key_env)
    if not api_key:
        raise ModelCredentialError(
            f"connection credential environment variable {connection.api_key_env!r} is not set"
        )
    return api_key
