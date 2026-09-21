"""Boundary errors the native data plane receives from attempt accounting.

Split from :mod:`exp.runtime.gateway.native_accounting` for the module line
budget; that module re-exports every name here, so import paths are unchanged.
"""

from __future__ import annotations

import json

from exp.runtime.gateway.boundary import boundary_protocol_error
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


class NativeBridgeError(Exception):
    """One sanitized boundary failure delivered to the native data plane."""

    def __init__(self, error: OpenAIProtocolError) -> None:
        """Retain the public error as the JSON payload the data plane returns.

        Args:
            error: Sanitized protocol error carrying its HTTP representation.
        """
        super().__init__(error.detail.message)
        self.public_error_json = json.dumps(
            {
                "status_code": error.status_code,
                "code": error.detail.code,
                "message": error.detail.message,
                "error_type": error.detail.type,
                "param": error.detail.param,
                "retry_after_seconds": error.retry_after_seconds,
            },
            separators=(",", ":"),
        )


def authority_error(exception: Exception) -> NativeBridgeError:
    """Map boundary failures through the shared service-layer mapper.

    Args:
        exception: Store, grant, routing, or execution failure.

    Returns:
        A boundary error carrying the matching public OpenAI error.
    """
    return NativeBridgeError(boundary_protocol_error(exception))


def internal_protocol_error() -> OpenAIProtocolError:
    """Return the public internal error for a broken data-plane wire contract."""
    return OpenAIProtocolError(
        status_code=500,
        code="internal_error",
        message="The gateway request failed.",
        error_type="api_error",
    )
