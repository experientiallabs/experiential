"""Tests for the dedicated HTTP JSON classifier adapter."""

from __future__ import annotations

import asyncio
import json
import os

import httpx
import pytest

from exp.common.core.artifacts import JsonObject, canonical_json_bytes
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.gateway.guardrails.contracts import (
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailCompletion,
    request_content_bytes,
)
from exp.runtime.gateway.guardrails.http_json import (
    ClassifierProtocolError,
    HttpJsonClassifier,
    shared_http_json_client,
    validate_classifier_url,
)

_URL = "https://classifier.example.invalid/v1/inspect"


def _input_check() -> GuardrailCheck:
    """Return one input PII modify check."""
    return GuardrailCheck(
        check_id="standard-input-pii",
        capability=GuardrailCapabilityKind.PII,
        stage=GuardrailCheckStage.INPUT,
        action=GuardrailAction.MODIFY,
        timeout_ms=250,
        adapter_id="hosted-pii",
    )


def _block_check() -> GuardrailCheck:
    """Return one input content-safety block check."""
    return GuardrailCheck(
        check_id="standard-input-content-safety",
        capability=GuardrailCapabilityKind.CONTENT_SAFETY,
        stage=GuardrailCheckStage.INPUT,
        action=GuardrailAction.BLOCK,
        timeout_ms=250,
        adapter_id="hosted-safety",
    )


def _output_check() -> GuardrailCheck:
    """Return one output PII modify check."""
    return GuardrailCheck(
        check_id="standard-output-pii",
        capability=GuardrailCapabilityKind.PII,
        stage=GuardrailCheckStage.OUTPUT,
        action=GuardrailAction.MODIFY,
        timeout_ms=250,
        adapter_id="hosted-pii",
    )


def _request() -> GatewayRequest:
    """Return one chat request used as the inspect subject."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
    )


def _classifier(
    handler: httpx.MockTransport | None = None,
    *,
    bearer_env: str | None = None,
    max_response_bytes: int = 4096,
    client: httpx.AsyncClient | None = None,
) -> tuple[HttpJsonClassifier, httpx.AsyncClient]:
    """Bind one adapter to an injected MockTransport client."""
    resolved = client or httpx.AsyncClient(transport=handler or httpx.MockTransport(_allow_handler))
    adapter = HttpJsonClassifier(
        adapter_id="hosted-pii",
        url=_URL,
        bearer_env=bearer_env,
        max_response_bytes=max_response_bytes,
        client=resolved,
    )
    return adapter, resolved


def _allow_handler(_request: httpx.Request) -> httpx.Response:
    """Return an unflagged verdict."""
    return httpx.Response(200, json={"flagged": False})


def test_shared_client_is_reused_on_the_same_event_loop() -> None:
    """Two inspects on one loop share one pooled client identity."""

    async def scenario() -> None:
        """Compare client identity across two lookups."""
        first = shared_http_json_client()
        second = shared_http_json_client()
        assert first is second

    asyncio.run(scenario())


def test_injected_client_is_reused_across_inspects() -> None:
    """The adapter does not construct a new client per inspect."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the request URL and allow the inspect."""
        seen.append(str(request.url))
        return httpx.Response(200, json={"flagged": False})

    adapter, _injected = _classifier(httpx.MockTransport(handler))

    async def scenario() -> None:
        """Inspect twice through the same injected client."""
        check = _input_check()
        request = _request()
        first = await adapter.inspect_input(request=request, check=check)
        second = await adapter.inspect_input(request=request, check=check)
        assert first.flagged is False
        assert second.flagged is False

    asyncio.run(scenario())
    assert seen == [_URL, _URL]


def test_outbound_request_includes_capability_stage_action_and_one_subject() -> None:
    """The inspect envelope is the documented HTTP contract."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Capture the JSON body without logging it."""
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"flagged": False})

    adapter, _client = _classifier(httpx.MockTransport(handler))

    asyncio.run(adapter.inspect_input(request=_request(), check=_input_check()))

    assert captured["capability"] == "pii"
    assert captured["stage"] == "input"
    assert captured["action"] == "modify"
    assert captured["check_id"] == "standard-input-pii"
    assert "request" in captured
    assert "completion" not in captured

    captured.clear()
    asyncio.run(
        adapter.inspect_output(
            completion=GuardrailCompletion(text="ok"),
            check=_output_check(),
        )
    )
    assert captured["stage"] == "output"
    assert "completion" in captured
    assert "request" not in captured


def test_outbound_request_subject_matches_request_content_bytes() -> None:
    """The adapter sends the same compact JSON counted by the request bound."""
    captured: JsonObject = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Capture the request subject without logging it."""
        captured["request"] = json.loads(request.content)["request"]
        return httpx.Response(200, json={"flagged": False})

    adapter, _client = _classifier(httpx.MockTransport(handler))
    request = _request()
    asyncio.run(adapter.inspect_input(request=request, check=_input_check()))

    assert canonical_json_bytes(captured["request"]) == canonical_json_bytes(request)
    assert request_content_bytes(request) == len(canonical_json_bytes(request))


def test_bearer_env_sends_authorization_without_embedding_the_value() -> None:
    """Auth is present as Bearer plus the live environment value."""
    env_name = "CLASSIFIER_BEARER"
    os.environ[env_name] = "test-auth"
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Record whether authorization was supplied."""
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"flagged": False})

    try:
        adapter, _client = _classifier(httpx.MockTransport(handler), bearer_env=env_name)
        asyncio.run(adapter.inspect_input(request=_request(), check=_input_check()))
    finally:
        del os.environ[env_name]

    authorization = seen["authorization"]
    assert authorization is not None
    scheme, separator, value = authorization.partition(" ")
    assert scheme == "Bearer"
    assert separator == " "
    assert value == "test-auth"


def test_missing_bearer_env_is_classifier_uncertainty() -> None:
    """A configured name with no process value fails the inspect contract."""
    os.environ.pop("CLASSIFIER_BEARER", None)
    adapter, _client = _classifier(bearer_env="CLASSIFIER_BEARER")

    with pytest.raises(ClassifierProtocolError, match="unset"):
        asyncio.run(adapter.inspect_input(request=_request(), check=_input_check()))


def test_non_2xx_response_is_classifier_uncertainty() -> None:
    """Error statuses do not parse a body as a verdict."""

    def handler(_request: httpx.Request) -> httpx.Response:
        """Return a server error without a verdict."""
        return httpx.Response(503, text="unavailable")

    adapter, _client = _classifier(httpx.MockTransport(handler))

    with pytest.raises(ClassifierProtocolError, match="HTTP 503"):
        asyncio.run(adapter.inspect_input(request=_request(), check=_input_check()))


def test_malformed_json_is_classifier_uncertainty() -> None:
    """A 2xx body that is not JSON cannot become a verdict."""

    def handler(_request: httpx.Request) -> httpx.Response:
        """Return bytes that are not JSON."""
        return httpx.Response(200, content=b"not-json")

    adapter, _client = _classifier(httpx.MockTransport(handler))

    with pytest.raises(ClassifierProtocolError, match="not valid JSON"):
        asyncio.run(adapter.inspect_input(request=_request(), check=_input_check()))


def test_oversized_body_is_classifier_uncertainty() -> None:
    """Bodies larger than the configured bound are rejected before parse."""

    def handler(_request: httpx.Request) -> httpx.Response:
        """Return a large JSON object."""
        return httpx.Response(200, json={"flagged": False, "padding": "x" * 128})

    adapter, _client = _classifier(httpx.MockTransport(handler), max_response_bytes=32)

    with pytest.raises(ClassifierProtocolError, match="max_response_bytes"):
        asyncio.run(adapter.inspect_input(request=_request(), check=_input_check()))


def test_contract_drift_is_classifier_uncertainty() -> None:
    """Unknown verdict fields are rejected by ClassifierVerdict."""

    def handler(_request: httpx.Request) -> httpx.Response:
        """Return a verdict with an extra field."""
        return httpx.Response(200, json={"flagged": False, "vendor_score": 1})

    adapter, _client = _classifier(httpx.MockTransport(handler))

    with pytest.raises(ClassifierProtocolError, match="ClassifierVerdict"):
        asyncio.run(adapter.inspect_input(request=_request(), check=_input_check()))


def test_invalid_input_replacement_is_classifier_uncertainty() -> None:
    """Input inspects accept replacement messages, not completion text."""

    def handler(_request: httpx.Request) -> httpx.Response:
        """Return a text replacement on an input inspect."""
        return httpx.Response(200, json={"flagged": True, "replacement_text": "redacted"})

    adapter, _client = _classifier(httpx.MockTransport(handler))

    with pytest.raises(ClassifierProtocolError, match="replacement_text"):
        asyncio.run(adapter.inspect_input(request=_request(), check=_input_check()))


def test_non_modify_verdict_cannot_include_a_replacement() -> None:
    """Block and allow checks reject any replacement payload."""

    def handler(_request: httpx.Request) -> httpx.Response:
        """Return a replacement on a block check."""
        return httpx.Response(
            200,
            json={
                "flagged": True,
                "replacement_messages": [{"role": "user", "content": "redacted"}],
            },
        )

    adapter, _client = _classifier(httpx.MockTransport(handler))

    with pytest.raises(ClassifierProtocolError, match="non-modify"):
        asyncio.run(adapter.inspect_input(request=_request(), check=_block_check()))


def test_flagged_modify_requires_the_stage_replacement() -> None:
    """A flagged modify verdict without its replacement is contract drift."""

    def handler(_request: httpx.Request) -> httpx.Response:
        """Flag without supplying a replacement."""
        return httpx.Response(200, json={"flagged": True})

    adapter, _client = _classifier(httpx.MockTransport(handler))

    with pytest.raises(ClassifierProtocolError, match="replacement_messages"):
        asyncio.run(adapter.inspect_input(request=_request(), check=_input_check()))
    with pytest.raises(ClassifierProtocolError, match="replacement_text"):
        asyncio.run(
            adapter.inspect_output(
                completion=GuardrailCompletion(text="ok"),
                check=_output_check(),
            )
        )


def test_both_replacement_types_together_are_rejected() -> None:
    """A verdict may carry at most one replacement subject."""

    def handler(_request: httpx.Request) -> httpx.Response:
        """Return both replacement fields."""
        return httpx.Response(
            200,
            json={
                "flagged": True,
                "replacement_text": "redacted",
                "replacement_messages": [{"role": "user", "content": "redacted"}],
            },
        )

    adapter, _client = _classifier(httpx.MockTransport(handler))

    with pytest.raises(ClassifierProtocolError, match="both replacement types"):
        asyncio.run(adapter.inspect_input(request=_request(), check=_input_check()))


def test_unflagged_verdict_cannot_include_a_replacement() -> None:
    """Allowing a request still forbids leftover replacement fields."""

    def handler(_request: httpx.Request) -> httpx.Response:
        """Return an unflagged verdict with a replacement."""
        return httpx.Response(
            200,
            json={
                "flagged": False,
                "replacement_messages": [{"role": "user", "content": "redacted"}],
            },
        )

    adapter, _client = _classifier(httpx.MockTransport(handler))

    with pytest.raises(ClassifierProtocolError, match="unflagged"):
        asyncio.run(adapter.inspect_input(request=_request(), check=_input_check()))


def test_invalid_output_replacement_is_classifier_uncertainty() -> None:
    """Output inspects accept replacement text, not request messages."""

    def handler(_request: httpx.Request) -> httpx.Response:
        """Return request-message replacement on an output inspect."""
        return httpx.Response(
            200,
            json={
                "flagged": True,
                "replacement_messages": [{"role": "user", "content": "redacted"}],
            },
        )

    adapter, _client = _classifier(httpx.MockTransport(handler))

    with pytest.raises(ClassifierProtocolError, match="replacement_messages"):
        asyncio.run(
            adapter.inspect_output(
                completion=GuardrailCompletion(text="ok"),
                check=_output_check(),
            )
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://classifier.example.invalid/v1/chat/completions",
        "https://classifier.example.invalid/v1/responses",
        "https://classifier.example.invalid/v1/models",
        "https://classifier.example.invalid/v1/chat/completions/extra",
        "file:///tmp/classifier",
        "/v1/inspect",
        "https://user:pass@classifier.example.invalid/v1/inspect",
        "https://classifier.example.invalid/v1/inspect#next",
        "https://classifier.example.invalid/%76%31/chat/completions",
        "https://classifier.example.invalid/v1/%63hat/completions",
        "https://classifier.example.invalid/foo/../v1/chat/completions",
        "https://classifier.example.invalid/v1/./responses",
        "https://classifier.example.invalid/%2e%2e/v1/models",
        "https://classifier.example.invalid/%2fv1%2fchat%2fcompletions",
        "https://classifier.example.invalid/v1/chat/completions%2fextra",
        "https://classifier.example.invalid/%2576%2531/chat/completions",
    ],
)
def test_public_gateway_and_invalid_urls_are_rejected(url: str) -> None:
    """Dedicated classifiers cannot be pointed at public gateway paths."""
    with pytest.raises(ValueError):
        validate_classifier_url(url)


def test_dedicated_classifier_url_is_accepted() -> None:
    """A non-gateway inspect path is valid."""
    assert validate_classifier_url(_URL) == _URL


def test_shared_client_does_not_follow_redirects() -> None:
    """Redirects stay disabled so a classifier cannot bounce to a public route."""

    async def scenario() -> None:
        """Inspect the shared client's redirect setting."""
        assert shared_http_json_client().follow_redirects is False

    asyncio.run(scenario())
