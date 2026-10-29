"""Tests for shared Bedrock credential modes and endpoint restrictions."""

from __future__ import annotations

from typing import Literal

import pytest

from exp.common.models.bedrock_connection import require_bedrock_connection_shape


@pytest.mark.parametrize(
    ("mode", "secret_env", "access_env", "expected_error"),
    [
        (None, None, None, None),
        (None, "SECRET_ENV", "ACCESS_ENV", None),
        (None, "SECRET_ENV", None, "bedrock explicit access-key auth"),
        (None, None, "ACCESS_ENV", "bedrock explicit access-key auth"),
        ("api_key", "TOKEN_ENV", None, None),
        ("api_key", None, None, "bedrock api_key auth"),
        ("api_key", None, "ACCESS_ENV", "bedrock api_key auth"),
        ("api_key", "TOKEN_ENV", "ACCESS_ENV", "bedrock api_key auth"),
        ("access_key_pair", "SECRET_ENV", "ACCESS_ENV", None),
        ("access_key_pair", "SECRET_ENV", None, "bedrock access_key_pair auth"),
        ("access_key_pair", None, "ACCESS_ENV", "bedrock access_key_pair auth"),
        ("access_key_pair", None, None, "bedrock access_key_pair auth"),
    ],
)
def test_bedrock_credential_mode_requires_the_exact_environment_fields(
    mode: Literal["access_key_pair", "api_key"] | None,
    secret_env: str | None,
    access_env: str | None,
    expected_error: str | None,
) -> None:
    """Ambient, bearer, and access-pair credentials keep distinct field requirements."""
    if expected_error is None:
        require_bedrock_connection_shape(
            bedrock_auth_mode=mode,
            api_key_env=secret_env,
            aws_access_key_id_env=access_env,
            base_url=None,
            api_version=None,
        )
    else:
        with pytest.raises(ValueError, match=expected_error):
            require_bedrock_connection_shape(
                bedrock_auth_mode=mode,
                api_key_env=secret_env,
                aws_access_key_id_env=access_env,
                base_url=None,
                api_version=None,
            )


@pytest.mark.parametrize(
    ("base_url", "api_version", "expected_error"),
    [
        ("https://example.test/bedrock", None, "bedrock does not accept base_url"),
        (None, "v1", "api_version is only accepted for provider='azure'"),
        ("https://example.test/bedrock", "v1", "bedrock does not accept base_url"),
    ],
)
def test_bedrock_rejects_custom_endpoint_and_azure_version(
    base_url: str | None, api_version: str | None, expected_error: str
) -> None:
    """A valid credential mode cannot enable custom Bedrock HTTP endpoints."""
    with pytest.raises(ValueError, match=expected_error):
        require_bedrock_connection_shape(
            bedrock_auth_mode="api_key",
            api_key_env="TOKEN_ENV",
            aws_access_key_id_env=None,
            base_url=base_url,
            api_version=api_version,
        )
