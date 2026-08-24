"""HTTP-boundary tests for the Anthropic Messages routes."""

from __future__ import annotations

from typing import cast

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from exp.runtime.gateway.messages import register_messages_routes
from exp.runtime.gateway.service import GatewayService
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import DecodedGatewayRequest


class _StubService:
    """Capture the decoded request and answer a fixed completion."""

    def __init__(self, complete_error: Exception | None = None) -> None:
        """Optionally fail completion with one injected boundary error."""
        self.raw_keys: list[str] = []
        self.decoded: list[DecodedGatewayRequest] = []
        self.app_identity: list[tuple[str | None, str | None]] = []
        self._complete_error = complete_error

    def authenticate(self, *, raw_key: str) -> None:
        """Reject the one known-bad key, accept everything else."""
        if raw_key == "revoked":
            raise OpenAIProtocolError(
                status_code=401,
                code="invalid_key",
                message="The gateway key is invalid, expired, or revoked.",
                error_type="authentication_error",
            )
        self.raw_keys.append(raw_key)

    async def complete(
        self,
        *,
        raw_key: str,
        decoded: DecodedGatewayRequest,
        app_referer: str | None = None,
        app_title: str | None = None,
    ) -> Response:
        """Record the canonical request and answer one canned body."""
        del raw_key
        if self._complete_error is not None:
            raise self._complete_error
        self.decoded.append(decoded)
        self.app_identity.append((app_referer, app_title))
        return JSONResponse({"ok": True})


def _client(service: _StubService) -> TestClient:
    """Return a test client over an app carrying only the Messages routes."""
    app = FastAPI()
    register_messages_routes(app, cast("GatewayService", service))
    return TestClient(app)


_BODY = {
    "model": "coding",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "hi"}],
}


def test_x_api_key_and_bearer_both_authenticate() -> None:
    """Anthropic callers may present x-api-key or a standard Bearer header."""
    service = _StubService()
    client = _client(service)
    with_api_key = client.post("/v1/messages", json=_BODY, headers={"x-api-key": "key-a"})
    assert with_api_key.status_code == 200
    assert (
        client.post(
            "/v1/messages", json=_BODY, headers={"Authorization": "Bearer key-b"}
        ).status_code
        == 200
    )
    assert service.raw_keys == ["key-a", "key-b"]
    assert service.decoded[0].alias == "coding"
    assert service.decoded[0].request.surface.value == "messages"


def test_missing_and_rejected_keys_answer_anthropic_shaped_401s() -> None:
    """Credential failures are Anthropic-enveloped on this surface."""
    client = _client(_StubService())
    missing = client.post("/v1/messages", json=_BODY)
    assert missing.status_code == 401
    assert missing.json() == {
        "type": "error",
        "error": {
            "type": "authentication_error",
            "message": "A valid API key is required: send x-api-key or Authorization: Bearer.",
        },
    }
    revoked = client.post("/v1/messages", json=_BODY, headers={"x-api-key": "revoked"})
    assert revoked.status_code == 401
    assert revoked.json()["error"]["type"] == "authentication_error"


def test_invalid_json_and_unknown_fields_answer_anthropic_shaped_400s() -> None:
    """Protocol decode failures carry the Anthropic envelope and field path."""
    client = _client(_StubService())
    invalid = client.post(
        "/v1/messages",
        content=b"{not json",
        headers={"x-api-key": "key", "content-type": "application/json"},
    )
    assert invalid.status_code == 400
    assert invalid.json()["error"]["type"] == "invalid_request_error"
    unknown = client.post("/v1/messages", json={**_BODY, "top_k": 4}, headers={"x-api-key": "key"})
    assert unknown.status_code == 400
    assert "top_k" in unknown.json()["error"]["message"]


def test_boundary_failures_translate_with_retry_after() -> None:
    """Quota exhaustion keeps its Retry-After wait in Anthropic shape."""
    service = _StubService(
        complete_error=OpenAIProtocolError(
            status_code=429,
            code="insufficient_quota",
            message="monthly gateway allocation is exhausted",
            error_type="insufficient_quota",
            retry_after_seconds=60,
        )
    )
    client = _client(service)
    response = client.post("/v1/messages", json=_BODY, headers={"x-api-key": "key"})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"
    assert response.json()["error"]["type"] == "rate_limit_error"


def test_count_tokens_refuses_in_anthropic_shape() -> None:
    """The count_tokens probe is refused explicitly in the caller's envelope."""
    client = _client(_StubService())
    response = client.post("/v1/messages/count_tokens", json={})
    assert response.status_code == 404
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "not_found_error",
            "message": "count_tokens is not served by this gateway.",
        },
    }
