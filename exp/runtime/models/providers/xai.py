# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""xAI first-party origin detection.

xAI's API (``https://api.x.ai/v1``) is served to the gateway as an
OpenAI-compatible connection: its Chat Completions and Images endpoints
follow the OpenAI wire. xAI refuses hosted (``url``) image output for Zero
Data Retention teams, so image requests to xAI's origin always ask for
inline base64 (``response_format: "b64_json"``). Its video generation
endpoint does not exist on that wire, so a connection only speaks xAI's own
asynchronous video job API (``POST /v1/videos/generations`` then
``GET /v1/videos/{request_id}``) when its root is xAI's origin; every other
OpenAI-compatible root has no video endpoint at all.
"""

from __future__ import annotations

from urllib.parse import urlsplit

_XAI_HOST = "api.x.ai"
_XAI_ROOT_PATH = "/v1"


def is_xai_base_url(base_url: str) -> bool:
    """Return whether one endpoint is xAI's own OpenAI-compatible ``/v1`` root.

    Matches ``https://api.x.ai/v1`` on the default port with no credentials,
    query, or fragment. Hosts that resell Grok models (OpenRouter) are not
    xAI's origin and never match.

    Args:
        base_url: The connection's normalized base URL.

    Returns:
        Whether the root is xAI's origin.
    """
    parsed = urlsplit(base_url)
    return (
        parsed.scheme == "https"
        and parsed.hostname == _XAI_HOST
        and parsed.username is None
        and parsed.password is None
        and parsed.port in {None, 443}
        and parsed.path.rstrip("/") == _XAI_ROOT_PATH
        and not parsed.query
        and not parsed.fragment
    )


def xai_videos_url(base_url: str) -> str | None:
    """Return xAI's video job creation endpoint for an xAI root, else None.

    Args:
        base_url: The connection's normalized base URL.

    Returns:
        ``{root}/videos/generations`` on xAI's origin, otherwise None.
    """
    if not is_xai_base_url(base_url):
        return None
    return f"{base_url.rstrip('/')}/videos/generations"


def is_xai_images_url(images_url: str) -> bool:
    """Return whether one image endpoint is xAI's own ``/images/generations``.

    Args:
        images_url: A rung's resolved image-generation endpoint.

    Returns:
        Whether the endpoint is ``{xAI root}/images/generations``.
    """
    suffix = "/images/generations"
    endpoint = images_url.rstrip("/")
    return endpoint.endswith(suffix) and is_xai_base_url(endpoint.removesuffix(suffix))
