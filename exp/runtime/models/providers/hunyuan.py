# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Tencent Hunyuan preserved-thinking route detection.

Tencent's Hunyuan models (hy3/hy4) serve an OpenAI-compatible Chat Completions
endpoint that returns the model's chain-of-thought in a native ``reasoning_content``
response field and accepts it back on an assistant turn. This module identifies
those endpoints so the gateway marks the rung as an exposable-plaintext
reasoning route whose replay is protected by a gateway-issued opaque carrier,
mirroring :mod:`exp.runtime.models.providers.fireworks`.

Two OpenAI-compatible origins serve these models:

- ``https://api.hunyuan.cloud.tencent.com/v1`` — the mainland Hunyuan root.
- ``https://tokenhub-intl.tencentcloudmaas.com/v1`` — Tencent's TokenHub
  international (Singapore) MaaS gateway, which is the origin the platform's
  Tencent lane actually dispatches through. TokenHub is multi-model, exactly
  like the Fireworks host: recognizing it here resolves a reasoning-carrier
  route for every rung served through it, but the plaintext reasoning is still
  captured only from models that emit ``reasoning_content`` and exposed only on
  rungs the catalog stamps ``reasoning_output_exposed``.
"""

from __future__ import annotations

from urllib.parse import urlsplit

# Exact OpenAI-compatible hosts that return native ``reasoning_content``. The
# TC3-HMAC-signed ``hunyuan.tencentcloudapi.com`` API is a different protocol
# the gateway never speaks and is deliberately excluded. The site-bound TokenHub
# siblings (``tokenhub-us`` and any mainland site) are omitted until a lane
# actually dispatches through them, per the "no speculative surface" rule.
_HUNYUAN_REASONING_HOSTS = frozenset(
    {
        "api.hunyuan.cloud.tencent.com",
        "tokenhub-intl.tencentcloudmaas.com",
    }
)


def is_hunyuan_base_url(base_url: str) -> bool:
    """Return whether one endpoint is an exact Hunyuan reasoning inference root.

    Matches the mainland Hunyuan root and the TokenHub international MaaS origin
    the platform's Tencent lane serves through, each on ``https``, the exact
    ``/v1`` path, the default port, and no credentials, query, or fragment.
    """
    parsed = urlsplit(base_url)
    return (
        parsed.scheme == "https"
        and parsed.hostname in _HUNYUAN_REASONING_HOSTS
        and parsed.username is None
        and parsed.password is None
        and parsed.port in {None, 443}
        and parsed.path.rstrip("/") == "/v1"
        and not parsed.query
        and not parsed.fragment
    )
