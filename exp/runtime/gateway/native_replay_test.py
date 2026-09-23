"""Native replay scope stays tenant-bound and requires the standard operation key."""

import json

import pytest

from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.gateway.native_execution_test import _route
from exp.runtime.gateway.native_replay import replay_scope_payload
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


def test_scope_keeps_authorized_identity_and_hashes_operation() -> None:
    """The scope uses authorized identities, never a caller's raw secret or session hint."""
    auth = _route().snapshot.authorization
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
        idempotency_key="one-operation",
        client_request_id="a-session",
    )
    encoded = replay_scope_payload(auth, request)
    scope = json.loads(encoded)
    assert scope["organization_id"] == auth.organization_id
    assert scope["identity_id"] == auth.identity_id
    assert scope["canonical_request_sha256"] == auth.canonical_request_sha256
    assert "one-operation" not in encoded
    assert "a-session" not in encoded
    alternate = auth.model_copy(update={"organization_id": "other-org"})
    assert json.loads(replay_scope_payload(alternate, request))["organization_id"] == "other-org"


def test_session_hint_does_not_substitute_for_idempotency_key() -> None:
    """Unkeyed traffic cannot claim replay from a session correlation identifier."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
        client_request_id="a-session",
    )
    with pytest.raises(OpenAIProtocolError, match="Idempotency-Key"):
        replay_scope_payload(_route().snapshot.authorization, request)
