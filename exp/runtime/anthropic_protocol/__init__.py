"""Anthropic Messages protocol decoding, encoding, and error envelopes.

This package is the only Anthropic Messages wire implementation in the
gateway. ``decode_messages`` converts one public ``POST /v1/messages`` body
into the same canonical :class:`~exp.runtime.gateway.contracts.GatewayRequest`
the OpenAI surfaces produce, and the encoders in
:mod:`exp.runtime.anthropic_protocol.encoding` render the canonical gateway
event stream back into Anthropic streaming SSE and the non-streaming message
object. Failures are Anthropic-enveloped through
:mod:`exp.runtime.anthropic_protocol.errors`.
"""

from exp.runtime.anthropic_protocol.encoding import (
    MessagesSseEncoder,
    completed_messages_body,
)
from exp.runtime.anthropic_protocol.errors import anthropic_error_body
from exp.runtime.anthropic_protocol.requests import decode_messages

__all__ = [
    "MessagesSseEncoder",
    "anthropic_error_body",
    "completed_messages_body",
    "decode_messages",
]
