"""Exception-to-public-error boundary for the native gateway data plane.

This is the single authority for mapping sanitized protocol, authority,
routing, and execution failures to their stable public OpenAI errors, so
every control-plane surface answers the same failure identically.
"""

from __future__ import annotations

import asyncio

from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.guardrails.contracts import GuardrailRejected
from exp.runtime.gateway.ledger import (
    AttemptRejectedError,
    IdempotencyConflictError,
    IdempotencyReplayUnavailableError,
)
from exp.runtime.gateway.routing import GatewayRoutingError
from exp.runtime.gateway.sqlite.store import (
    AliasNotGrantedError,
    GatewayStoreError,
    InvalidVirtualKeyError,
)
from exp.runtime.models.providers.async_transport import ProviderDeadlineExceeded
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, public_failure_error

_DRAINING_RETRY_AFTER_SECONDS = 10


class GatewayDrainingError(RuntimeError):
    """The gateway is draining and is not accepting new requests."""


def boundary_protocol_error(exception: BaseException) -> OpenAIProtocolError:
    """Map one sanitized boundary failure to its stable public protocol error.

    This is the single authority for exception-to-public-error mapping, so
    every control-plane surface answers the same failure identically.

    Args:
        exception: Protocol, authority, routing, or execution failure.

    Returns:
        OpenAI-shaped protocol error carrying its HTTP representation.
    """
    error: OpenAIProtocolError
    if isinstance(exception, OpenAIProtocolError):
        error = exception
    elif isinstance(exception, InvalidVirtualKeyError):
        error = OpenAIProtocolError(
            status_code=401,
            code="invalid_key",
            message=(
                "The gateway key is invalid, expired, or revoked. Ask the gateway operator "
                "to issue a new virtual key."
            ),
            error_type="authentication_error",
        )
    elif isinstance(exception, AliasNotGrantedError):
        error = OpenAIProtocolError(
            status_code=403,
            code="model_not_granted",
            message=(
                "The requested model alias is not granted to this identity. "
                "GET /v1/models lists the model aliases available to this key."
            ),
            error_type="permission_error",
            param="model",
        )
    elif isinstance(exception, IdempotencyConflictError):
        error = OpenAIProtocolError(
            status_code=409,
            code="idempotency_conflict",
            message=(
                "The caller operation was reused with different request content. "
                "Send a new Idempotency-Key for each distinct request."
            ),
            param="Idempotency-Key",
        )
    elif isinstance(exception, IdempotencyReplayUnavailableError):
        error = OpenAIProtocolError(
            status_code=409,
            code="idempotency_replay_unavailable",
            message=(
                "The completed keyed result is unavailable after restart. "
                "Resend the request with a new Idempotency-Key."
            ),
            error_type="api_error",
            param="Idempotency-Key",
        )
    elif isinstance(exception, GuardrailRejected):
        error = public_failure_error(exception.failure)
    elif isinstance(exception, AttemptRejectedError):
        # A typed pre-dispatch rejection without a more specific branch above
        # keeps the shape its ledger assigned (for example an injected ledger's
        # dispatch-time key revocation surfacing as authentication).
        error = public_failure_error(exception.failure)
    elif isinstance(exception, ProviderDeadlineExceeded):
        error = public_failure_error(
            GatewayFailure(
                failure_class=GatewayFailureClass.TIMEOUT,
                safe_message=(
                    "Gateway request deadline exceeded. Retry with a shorter prompt "
                    "or a smaller max_tokens value."
                ),
            )
        )
    elif isinstance(exception, (GatewayRoutingError, GatewayStoreError)):
        error = OpenAIProtocolError(
            status_code=503 if isinstance(exception, GatewayRoutingError) else 400,
            code=(
                "unavailable_route"
                if isinstance(exception, GatewayRoutingError)
                else "invalid_request"
            ),
            message=(
                "The authorized model route is unavailable. Retry after a short delay; "
                "if this persists, ask the gateway operator to check the alias deployments."
                if isinstance(exception, GatewayRoutingError)
                else (
                    "The gateway request is invalid. Verify the model alias and request "
                    "fields, then resend."
                )
            ),
            error_type=(
                "api_error"
                if isinstance(exception, GatewayRoutingError)
                else "invalid_request_error"
            ),
        )
    elif isinstance(exception, asyncio.CancelledError):
        error = OpenAIProtocolError(
            status_code=499,
            code="request_cancelled",
            message=(
                "The gateway request was cancelled. Resend the request if cancellation "
                "was not intended."
            ),
            error_type="api_error",
        )
    elif isinstance(exception, GatewayDrainingError):
        error = OpenAIProtocolError(
            status_code=503,
            code="gateway_draining",
            message=(
                "The gateway is draining and is not accepting new requests. "
                "Retry after the delay in the Retry-After header."
            ),
            error_type="api_error",
            retry_after_seconds=_DRAINING_RETRY_AFTER_SECONDS,
        )
    else:
        error = OpenAIProtocolError(
            status_code=500,
            code="internal_error",
            message=(
                "The gateway request failed. Retry the request; if this persists, "
                "ask the gateway operator to inspect the server logs."
            ),
            error_type="api_error",
        )
    return error
