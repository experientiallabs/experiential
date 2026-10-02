"""Sanitized request decoding callback shared by native control-plane mixins."""

from __future__ import annotations

from exp.runtime.gateway.native_accounting_errors import NativeBridgeError
from exp.runtime.gateway.native_decode import NativeDecodeError, decode_native_body
from exp.runtime.openai_protocol.requests import DecodedGatewayRequest


class NativeDecodeMixin:
    """Decode one raw request using the protocol owner's typed boundary."""

    def _decode_body(
        self,
        body: str,
        *,
        surface: str = "chat",
        idempotency_key: str | None = None,
        client_request_id: str | None = None,
        anthropic_beta: str | None = None,
    ) -> DecodedGatewayRequest:
        """Preserve protocol failures as sanitized native bridge errors."""
        try:
            return decode_native_body(
                body,
                surface=surface,
                idempotency_key=idempotency_key,
                client_request_id=client_request_id,
                anthropic_beta=anthropic_beta,
            )
        except NativeDecodeError as exc:
            raise NativeBridgeError(exc.error) from exc
