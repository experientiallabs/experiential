"""Tests for admission-time dispatch freezing and body signing."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.native_dispatch import dispatch_signature_headers, frozen_dispatch
from exp.runtime.gateway.routing import GatewayRoutingError
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


class _Signer:
    """Record the exact signed body and return deterministic headers."""

    def __init__(self, profile: GatewayWireProfile) -> None:
        """Retain the profile this signing client describes."""
        self.signed: list[tuple[str, str]] = []
        self._profile = profile

    def gateway_wire_profile(self) -> GatewayWireProfile:
        """Return the retained wire profile."""
        return self._profile

    def sign_gateway_dispatch(self, *, url: str, body: str) -> Mapping[str, str]:
        """Capture one signing call and return canned SigV4-shaped headers."""
        self.signed.append((url, body))
        return {"Authorization": "AWS4-HMAC-SHA256 test", "X-Amz-Date": "20260824T000000Z"}


def test_unsigned_dialects_freeze_nothing_and_retain_no_signer() -> None:
    """Dialects without body signing keep the data plane's own serialization."""
    profile = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://example.invalid/v1/chat/completions",
        headers={"authorization": "Bearer k"},
    )
    body, signer = frozen_dispatch(profile, None, {"model": "m"})
    assert body is None
    assert signer is None


def test_signing_dialect_freezes_the_exact_body_the_signer_later_covers() -> None:
    """The frozen body string is byte-identical to what signing covers."""
    profile = GatewayWireProfile(
        dialect="bedrock_converse_stream",
        url="https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse-stream",
        headers={},
        signs_request_body=True,
    )
    client = _Signer(profile)
    payload: JsonObject = {"messages": [{"role": "user", "content": [{"text": "Zürich"}]}]}
    body, signer = frozen_dispatch(profile, client, payload)
    assert body == '{"messages":[{"role":"user","content":[{"text":"Zürich"}]}]}'
    assert signer is client
    headers = dispatch_signature_headers(signer, url=profile.url, body=body)
    assert client.signed == [(profile.url, body)]
    assert headers["Authorization"] == "AWS4-HMAC-SHA256 test"
    assert headers["X-Amz-Date"] == "20260824T000000Z"


def test_signing_dialect_without_a_signer_fails_closed() -> None:
    """A body-signing profile on a non-signing client is a routing failure."""
    profile = GatewayWireProfile(
        dialect="bedrock_converse_stream",
        url="https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse-stream",
        signs_request_body=True,
    )
    with pytest.raises(GatewayRoutingError, match="cannot sign"):
        frozen_dispatch(profile, None, {"messages": []})


def test_dispatch_signing_without_a_retained_signer_is_sanitized() -> None:
    """Signing an unknown or unsigned attempt raises the public boundary error."""
    with pytest.raises(OpenAIProtocolError):
        dispatch_signature_headers(None, url="https://example.invalid", body="{}")
