# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Tencent Hunyuan preserved-thinking route detection.

Tencent's Hunyuan models (hy3/hy4) serve an OpenAI-compatible Chat Completions
endpoint that returns the model's chain-of-thought in a native ``reasoning_content``
response field and accepts it back on an assistant turn. This module identifies
that exact endpoint so the gateway marks the rung as an exposable-plaintext
reasoning route whose replay is protected by a gateway-issued opaque carrier,
mirroring :mod:`exp.runtime.models.providers.fireworks`.
"""

from __future__ import annotations

from urllib.parse import urlsplit


def is_hunyuan_base_url(base_url: str) -> bool:
    """Return whether one endpoint is the exact public Hunyuan inference root.

    The OpenAI-compatible root is ``https://api.hunyuan.cloud.tencent.com/v1``;
    the TC3-HMAC-signed ``hunyuan.tencentcloudapi.com`` host is a different API
    the gateway never speaks and is deliberately not matched here.
    """
    parsed = urlsplit(base_url)
    return (
        parsed.scheme == "https"
        and parsed.hostname == "api.hunyuan.cloud.tencent.com"
        and parsed.username is None
        and parsed.password is None
        and parsed.port in {None, 443}
        and parsed.path.rstrip("/") == "/v1"
        and not parsed.query
        and not parsed.fragment
    )
