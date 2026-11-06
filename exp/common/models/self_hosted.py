"""Self-hosted native decision servers: the explicit, loopback-only catalog exception.

A native decision model with open weights (for example Clef, served by SGLang on the SystemOne
wire) can run on the operator's own GPU and be reached through a local tunnel. The catalog
refuses a custom origin for a native provider unless it is a trusted HTTPS origin, so such a
server needs its own explicit opt-in: ``self_hosted = true`` on the connection.

The exception is deliberately narrow:

- only the ``typesafe`` provider (the native decisions wire) may be self-hosted;
- the base URL must name a loopback host (``localhost``, ``127.0.0.0/8`` or ``::1``), so plain
  HTTP never leaves the machine; a remote server is reached through an SSH or similar tunnel;
- the flag cannot be combined with a trusted custom origin, a tenant origin or a subscription;
- every model on the connection pins its exact weights ``revision``, which enters the model
  snapshot and so every cache key and artifact that names the model.

Because a loopback address names no remote party and tunnels move between ports and boxes, the
connection identity of a self-hosted server is the provider and the self-hosted marker, not the
port: the pinned model revision is what identifies the served weights.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from exp.common.core.artifacts import JsonObject

SELF_HOSTED_PROVIDERS = frozenset({"typesafe"})
"""Providers whose wire may be served by a self-hosted loopback server."""


def is_loopback_url(url: str) -> bool:
    """Whether a URL names a loopback host.

    Args:
        url: An absolute HTTP(S) URL.

    Returns:
        True for ``localhost`` and loopback IPv4 or IPv6 literals.
    """
    host = (urlsplit(url).hostname or "").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def require_self_hosted_connection(
    *,
    provider: str,
    base_url: str | None,
    trusted_custom_origin: bool,
    tenant_origin: bool,
    subscription: str | None,
) -> None:
    """Refuse a self-hosted connection outside the loopback decisions exception.

    Args:
        provider: The connection's provider.
        base_url: The connection's base URL.
        trusted_custom_origin: Whether the connection also claims a trusted HTTPS origin.
        tenant_origin: Whether the connection also claims a tenant origin.
        subscription: The connection's plan sign-in, if any.

    Raises:
        ValueError: The provider is not a native decisions wire, the base URL is missing or not
            loopback, or another origin policy is combined with it.
    """
    if provider not in SELF_HOSTED_PROVIDERS:
        raise ValueError(
            f"self_hosted applies only to a native decisions provider "
            f"({', '.join(sorted(SELF_HOSTED_PROVIDERS))}), not {provider!r}"
        )
    if base_url is None:
        raise ValueError("self_hosted requires an explicit loopback base_url")
    if not is_loopback_url(base_url):
        raise ValueError(
            "self_hosted requires a loopback base_url such as http://127.0.0.1:30001/v1; "
            "reach a remote server through a local tunnel"
        )
    if trusted_custom_origin or tenant_origin or subscription is not None:
        raise ValueError(
            "self_hosted cannot be combined with trusted_custom_origin, tenant_origin or a "
            "subscription"
        )


def self_hosted_identity(provider: str) -> JsonObject:
    """Return the connection identity of a self-hosted server: provider and marker only.

    Args:
        provider: The connection's provider.

    Returns:
        The identity mapping hashed into the connection digest.
    """
    return {"provider": provider, "self_hosted": True}


def require_pinned_revision(alias: str, revision: str | None) -> None:
    """Refuse a self-hosted model without its exact weights revision.

    Args:
        alias: The catalog alias.
        revision: The model record's revision.

    Raises:
        ValueError: No revision is pinned.
    """
    if not revision:
        raise ValueError(
            f"model alias {alias!r} is on a self_hosted connection and must pin the exact weights "
            "revision it serves (for example the Hugging Face commit)"
        )
