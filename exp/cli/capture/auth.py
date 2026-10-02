"""Reuse the ordinary Experiential login for capture uploads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from rich.console import Console

from exp.cli.auth import run_login
from exp.cli.providers.experiential_cloud import (
    HOSTED_PLATFORM_DEFAULT_URL,
    hosted_connection,
    hosted_credential_binding,
    hosted_gateway_base_url,
    hosted_platform_url,
)
from exp.common.auth import ProviderAuthStore, StoredCredentialEndpointMismatch
from exp.runtime.models.credentials import lookup_connection_credential


@dataclass(frozen=True)
class CaptureCredentials:
    """Endpoint-bound login material, with the key excluded from representations."""

    api_url: str
    web_url: str
    api_key: str = field(repr=False)


def capture_credentials(
    *,
    console: Console,
    environment: Mapping[str, str],
    root: Path,
    store: ProviderAuthStore | None = None,
) -> CaptureCredentials:
    """Reuse a saved login or complete the standard browser login once.

    Args:
        console: Terminal receiving ordinary login progress.
        environment: Hosted endpoint overrides and optional EXPLABS_API_KEY.
        root: Catalog root used when login synchronizes account models.
        store: Optional credential store for deterministic tests.

    Returns:
        The existing login bound to its configured API origin.

    Raises:
        ValueError: The configured API origin is unsafe or login saved no key.
    """
    endpoint = urlsplit(hosted_gateway_base_url(environment))
    if (
        endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
        or not endpoint.hostname
        or (
            endpoint.scheme != "https"
            and not (
                endpoint.scheme == "http" and endpoint.hostname in {"localhost", "127.0.0.1", "::1"}
            )
        )
        or endpoint.path.rstrip("/") != "/v1"
    ):
        raise ValueError(
            "EXP_GATEWAY_URL must be an HTTPS /v1 endpoint (HTTP is allowed on loopback)."
        )
    connection = hosted_connection(environment)
    try:
        credential = lookup_connection_credential(
            connection.catalog_config(),
            connection_id=connection.name,
            environment=environment,
            store=store,
        )
    except StoredCredentialEndpointMismatch:
        console.print("Saved login does not match this endpoint. Sign in to update your CLI login.")
        credential = None
    if credential is None:
        if (
            hosted_credential_binding(environment) != hosted_credential_binding({})
            and urlsplit(hosted_platform_url(environment)).hostname
            == urlsplit(HOSTED_PLATFORM_DEFAULT_URL).hostname
        ):
            raise ValueError(
                "Set EXP_PLATFORM_URL to this API endpoint's Platform web origin before login. "
                "Capture cannot use production browser login with a different API endpoint."
            )
        run_login(console=console, environment=environment, root=root, store=store)
        credential = lookup_connection_credential(
            connection.catalog_config(),
            connection_id=connection.name,
            environment=environment,
            store=store,
        )
    if credential is None:
        raise ValueError("No Experiential login was saved. Run exp login and try again.")
    return CaptureCredentials(
        api_url=urlunsplit((endpoint.scheme, endpoint.netloc, "", "", "")),
        web_url=hosted_platform_url(environment),
        api_key=credential.value,
    )
