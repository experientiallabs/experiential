"""Anthropic Messages HTTP routes over the shared gateway service.

``POST /v1/messages`` serves Anthropic SDK callers natively: the body is
decoded into the same canonical gateway request the OpenAI surfaces use,
authorization and execution run through the one shared service, and every
failure is rendered in the Anthropic error envelope. Anthropic callers
authenticate with ``x-api-key`` (their SDK default) or a standard Bearer
header; both carry the same virtual key. The Anthropic protocol defines no
idempotency header, so this surface never joins the keyed replay stores.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse

from exp.common.core.artifacts import JsonObject
from exp.runtime.anthropic_protocol.errors import anthropic_error_body
from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.gateway.boundary import boundary_protocol_error
from exp.runtime.openai_protocol.errors import OpenAIProtocolError

if TYPE_CHECKING:
    from exp.runtime.gateway.service import GatewayService

COUNT_TOKENS_MESSAGE = "count_tokens is not served by this gateway."


def register_messages_routes(app: FastAPI, service: GatewayService) -> None:
    """Register the Anthropic Messages routes on the gateway application.

    Args:
        app: The gateway FastAPI application.
        service: The shared authorization and execution service.
    """

    @app.post("/v1/messages")
    async def messages(
        request: Request,
        x_api_key: str | None = Header(default=None, alias="x-api-key"),
        authorization: str | None = Header(default=None),
        app_referer: str | None = Header(default=None, alias="HTTP-Referer"),
        app_title: str | None = Header(default=None, alias="X-Title"),
    ) -> Response:
        """Decode, authorize, and serve one Anthropic Messages request."""
        try:
            raw_key = presented_api_key(x_api_key, authorization)
            service.authenticate(raw_key=raw_key)
            try:
                payload = await request.json()
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise OpenAIProtocolError(
                    status_code=400,
                    code="invalid_json",
                    message=(
                        "Request body must contain valid JSON. Re-encode the payload and resend."
                    ),
                ) from exc
            if not isinstance(payload, dict):
                raise OpenAIProtocolError(
                    status_code=400,
                    code="invalid_request",
                    message=(
                        "Request body must be a JSON object. Re-encode the payload and resend."
                    ),
                )
            decoded = decode_messages(cast("JsonObject", payload))
            return await service.complete(
                raw_key=raw_key,
                decoded=decoded,
                app_referer=app_referer,
                app_title=app_title,
            )
        except Exception as exc:  # noqa: BLE001 - HTTP boundary sanitizes every failure.
            return anthropic_exception_response(exc)

    @app.post("/v1/messages/count_tokens")
    async def count_tokens() -> Response:
        """Refuse token counting in the caller's own envelope.

        Anthropic clients probe this endpoint; the gateway has no tokenizer
        authority to answer truthfully, so it refuses explicitly in Anthropic
        shape instead of an OpenAI-shaped 404 the client cannot parse.
        Clients fall back to their local estimate on failure.
        """
        error = OpenAIProtocolError(
            status_code=404,
            code="route_not_served",
            message=COUNT_TOKENS_MESSAGE,
            error_type="invalid_request_error",
        )
        return anthropic_error_response(error)


def presented_api_key(x_api_key: str | None, authorization: str | None) -> str:
    """Accept the Anthropic SDK's ``x-api-key`` or a standard Bearer credential.

    Args:
        x_api_key: Raw ``x-api-key`` header value, if any.
        authorization: Raw ``Authorization`` header value, if any.

    Returns:
        The presented virtual key.

    Raises:
        OpenAIProtocolError: Neither header carries a non-empty credential.
            The HTTP layer renders it in the Anthropic envelope.
    """
    if x_api_key is not None and x_api_key.strip():
        return x_api_key.strip()
    if authorization is not None and authorization.startswith("Bearer "):
        credential = authorization[len("Bearer ") :].strip()
        if credential:
            return credential
    raise OpenAIProtocolError(
        status_code=401,
        code="invalid_key",
        message="A valid API key is required: send x-api-key or Authorization: Bearer.",
        error_type="authentication_error",
    )


def anthropic_exception_response(exception: BaseException) -> Response:
    """Map one boundary failure to its Anthropic-enveloped HTTP response."""
    return anthropic_error_response(boundary_protocol_error(exception))


def anthropic_error_response(error: OpenAIProtocolError) -> Response:
    """Render one sanitized protocol error in the Anthropic envelope."""
    return JSONResponse(
        anthropic_error_body(error),
        status_code=error.status_code,
        headers=error.headers(),
    )
