"""Google Gemini API (``generateContent``) protocol decoding for the gateway.

This package owns the request decoding of the native Gemini surface: one
``POST /v1beta/models/{model}:generateContent`` (or ``:streamGenerateContent``,
``:countTokens``) body becomes the same canonical
:class:`~exp.runtime.gateway.contracts.GatewayRequest` the OpenAI and
Anthropic surfaces produce, so every model the gateway serves is reachable
from Google's own SDK. Response encoding and the Google error envelope for the
surface live in the native data plane.
"""

from exp.runtime.google_protocol.requests import (
    decode_generate_content,
    decode_generate_content_count_tokens,
)

__all__ = [
    "decode_generate_content",
    "decode_generate_content_count_tokens",
]
