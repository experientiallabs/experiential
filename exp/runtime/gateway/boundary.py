"""Exception-to-public-error boundary for the native gateway data plane.

This is the single authority for mapping sanitized protocol, authority,
routing, and execution failures to their stable public OpenAI errors, so
every control-plane surface answers the same failure identically.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.guardrails.contracts import GuardrailRejected
from exp.runtime.gateway.ledger import (
    AttemptRejectedError,
    IdempotencyConflictError,
    IdempotencyReplayUnavailableError,
)
from exp.runtime.gateway.model_chain_authority import ModelChainAuthorityError
from exp.runtime.gateway.routing import GatewayRoutingError
from exp.runtime.gateway.sqlite.store import (
    AliasNotGrantedError,
    GatewayStoreError,
    InvalidVirtualKeyError,
    ZdrRoutingUnavailableError,
)
from exp.runtime.models.providers.async_transport import ProviderDeadlineExceeded
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, public_failure_error

_DRAINING_RETRY_AFTER_SECONDS = 10
_AUTHORITY_RETRY_AFTER_SECONDS = 5
_LOGGER = logging.getLogger(__name__)


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
    elif isinstance(exception, ZdrRoutingUnavailableError):
        error = OpenAIProtocolError(
            status_code=403,
            code="model_not_granted",
            message=(
                "provider.zdr demands zero-data-retention routing, which this gateway cannot "
                "judge: it publishes no provider data-retention postures. Remove provider.zdr, "
                "or use a host that publishes postures and routes on them."
            ),
            error_type="permission_error",
            param="provider.zdr",
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
    elif isinstance(exception, ModelChainAuthorityError):
        error = OpenAIProtocolError(
            status_code=503,
            code="model_chain_authority_unavailable",
            message="Model-chain serving requires current enforcing host authority. "
            "Ask the operator to verify publication and rollback protections.",
            error_type="api_error",
        )
    elif isinstance(exception, GatewayRoutingError):
        error = OpenAIProtocolError(
            status_code=503,
            code="unavailable_route",
            message=(
                "The authorized model route is unavailable. Retry after a short delay; "
                "if this persists, ask the gateway operator to check the alias deployments."
            ),
            error_type="api_error",
        )
    elif isinstance(exception, GatewayStoreError):
        # Every typed caller fault (key, grant, ZDR, idempotency) has its own
        # branch above. A bare store refusal is the control store declining to
        # vouch for its own serving authority (a model-chain release crossing,
        # a malformed revision target, a missing registry row), never a field
        # the caller sent, so it is a retryable 503, and the store's internal
        # reason goes to the worker log rather than the caller (2026-10-17:
        # Claude Code read it as an unnamed 400 for hours).
        _LOGGER.warning("gateway store refused the request: %s", exception)
        error = OpenAIProtocolError(
            status_code=503,
            code="gateway_authority_unavailable",
            message=(
                "The gateway could not confirm serving authority for this request. "
                "Retry after the delay in the Retry-After header; if this persists, "
                "ask the gateway operator to check the gateway logs."
            ),
            error_type="api_error",
            retry_after_seconds=_AUTHORITY_RETRY_AFTER_SECONDS,
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
        # Nothing typed claimed it, so the caller gets an anonymous 500 and,
        # before admission, no request id: the log is the only witness
        # (2026-10-18: an OverflowError in a host's authorize answered about
        # 2,000 requests this way over 2h45m with no line naming it). The
        # message and traceback text may carry request content, so only the
        # type and the raising frame are logged.
        raised_at = _raising_frame(exception)
        _LOGGER.error(
            "gateway boundary answered an unclassified %s raised at %s as a 500 internal_error",
            type(exception).__name__,
            raised_at,
            extra={
                "operation": "boundary",
                "exception_type": type(exception).__name__,
                "raised_at": raised_at,
            },
        )
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


def _raising_frame(exception: BaseException) -> str:
    """Where an exception was raised, as ``module.py:line in function``; never its message."""
    trace = exception.__traceback__
    if trace is None:
        return "unknown"
    while trace.tb_next is not None:
        trace = trace.tb_next
    code = trace.tb_frame.f_code
    return f"{Path(code.co_filename).name}:{trace.tb_lineno} in {code.co_name}"
