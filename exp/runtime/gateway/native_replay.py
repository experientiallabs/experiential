"""Serialize authorized replay identity for the native data-plane store."""

from __future__ import annotations

import json

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayRequest
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.state import ProtocolNamespace, replay_key


def replay_scope_payload(authorization: AuthorizationSnapshot, request: GatewayRequest) -> str:
    """Return one replay-store scope after the caller and request have been authorized.

    Args:
        authorization: Frozen caller and canonical request identity.
        request: Decoded request carrying its standard idempotency key.

    Returns:
        JSON containing only scope identifiers and hashed operation identity.

    Raises:
        OpenAIProtocolError: The request has no standard idempotency key.
    """
    key = replay_key(
        namespace=ProtocolNamespace(
            organization_id=authorization.organization_id,
            identity_id=authorization.identity_id,
            alias_revision_id=authorization.alias_revision_id,
        ),
        surface=request.surface,
        caller_operation=request.idempotency_key,
        canonical_request_sha256=authorization.canonical_request_sha256,
    )
    if key is None:
        raise OpenAIProtocolError(
            status_code=400,
            code="invalid_request",
            message="A replay scope requires an Idempotency-Key header.",
            param="Idempotency-Key",
        )
    scope: JsonObject = {
        "organization_id": key.namespace.organization_id,
        "identity_id": key.namespace.identity_id,
        "alias_revision_id": key.namespace.alias_revision_id,
        "surface": key.surface.value,
        "caller_operation_sha256": key.caller_operation_sha256,
        "canonical_request_sha256": key.canonical_request_sha256,
    }
    return json.dumps(scope, separators=(",", ":"))
