"""Tests for admission-time dispatch freezing and body signing."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.native_dispatch import signed_dispatch
from exp.runtime.gateway.routing import GatewayRoutingError
from exp.runtime.models.providers.base import GatewayWireProfile


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


def test_unsigned_dialects_pass_profile_headers_and_no_frozen_body() -> None:
    """Dialects without body signing keep the data plane's own serialization."""
    profile = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://example.invalid/v1/chat/completions",
        headers={"authorization": "Bearer k"},
    )
    body, headers = signed_dispatch(profile, None, {"model": "m"})
    assert body is None
    assert headers == {"authorization": "Bearer k"}


def test_signing_dialect_freezes_the_exact_body_it_signs() -> None:
    """The returned body string is byte-identical to what the signer covered."""
    profile = GatewayWireProfile(
        dialect="bedrock_converse_stream",
        url="https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse-stream",
        headers={},
        signs_request_body=True,
    )
    signer = _Signer(profile)
    payload: JsonObject = {"messages": [{"role": "user", "content": [{"text": "Zürich"}]}]}
    body, headers = signed_dispatch(profile, signer, payload)
    assert body == '{"messages":[{"role":"user","content":[{"text":"Zürich"}]}]}'
    assert signer.signed == [(profile.url, body)]
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
        signed_dispatch(profile, None, {"messages": []})
