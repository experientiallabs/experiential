"""Anthropic Messages protocol decoding for the gateway.

This package owns the Anthropic Messages request decoding used by the
gateway: ``decode_messages`` converts one public ``POST /v1/messages`` body
into the same canonical :class:`~exp.runtime.gateway.contracts.GatewayRequest`
the OpenAI surfaces produce. Response encoding and error envelopes for the
Messages surface live in the native data plane.
"""

from exp.runtime.anthropic_protocol.requests import decode_messages

__all__ = [
    "decode_messages",
]
