"""Opaque per-organization scheduling tag for self-hosted SystemOne origins.

A self-hosted decision server (the house Clef box) shares one GPU queue across
every organization the gateway sends it. Without knowing which requests belong
together it can only serve them oldest-first, so one organization's burst of
long states delays every other caller. The gateway therefore attaches a stable,
opaque tag per organization so the server can fair-queue across organizations.

The tag never carries the organization id: it is an HMAC of the id under a key
derived from the rung's own released credential, truncated to 128 bits. The
origin already holds that credential, so the tag discloses nothing it could not
already authenticate, and anyone without the credential cannot link a tag to an
organization. A credential rotation rotates every tag, which only resets the
server's fairness history.

The tag is attached only on a host-funded rung whose connection is not a
customer-addressed origin, so a tenant's own endpoint never receives a stable
cross-request identifier for the caller.
"""

from __future__ import annotations

import hashlib
import hmac

ORGANIZATION_TAG_HEADER = "X-Experiential-Org-Tag"
_KEY_LABEL = b"exp-systemone-org-tag-v1"
_TAG_BYTES = 16


def organization_tag_key(api_key: str) -> bytes:
    """Derive the private tag key from a rung's released credential.

    Args:
        api_key: The bearer credential the rung authenticates with.

    Returns:
        A 32-byte key that never equals or reveals the credential itself.
    """
    return hmac.new(api_key.encode(), _KEY_LABEL, hashlib.sha256).digest()


def organization_tag(key: bytes, organization_id: str) -> str:
    """Return the opaque tag one organization carries to a tagged rung.

    Args:
        key: The rung's tag key from :func:`organization_tag_key`.
        organization_id: The authorized organization of the request.

    Returns:
        32 lowercase hex characters, stable per (credential, organization).
    """
    digest = hmac.new(key, organization_id.encode(), hashlib.sha256).digest()
    return digest[:_TAG_BYTES].hex()


def tagged_headers(
    headers: dict[str, str],
    key: bytes | None,
    organization_id: str,
    *,
    billing_customer_managed: bool,
    tenant_origin: bool,
) -> dict[str, str] | None:
    """Return the rung's headers plus the tag, or None when the rung gets no tag.

    Args:
        headers: The rung's static wire headers.
        key: The rung's tag key, None on connections that do not take a tag.
        organization_id: The authorized organization of the request.
        billing_customer_managed: The rung dispatches on a customer's credential.
        tenant_origin: The rung's connection is a customer-addressed origin.

    Returns:
        A new header mapping including the tag, or None to keep the static headers.
    """
    if key is None or billing_customer_managed or tenant_origin:
        return None
    return {**headers, ORGANIZATION_TAG_HEADER: organization_tag(key, organization_id)}
