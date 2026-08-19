"""Shared OpenAI protocol decoding, streaming, and replay state."""

from wmo.runtime.openai_protocol.errors import OpenAIProtocolError
from wmo.runtime.openai_protocol.model_adapter import model_request, model_response_events
from wmo.runtime.openai_protocol.requests import (
    DecodedGatewayRequest,
    decode_chat,
    decode_responses,
)
from wmo.runtime.openai_protocol.response import completed_body, stream_encoder

__all__ = [
    "DecodedGatewayRequest",
    "OpenAIProtocolError",
    "decode_chat",
    "decode_responses",
    "completed_body",
    "model_request",
    "model_response_events",
    "stream_encoder",
]
