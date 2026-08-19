"""Tests for the shared HTTP provider template that every concrete client inherits.

The per-provider suites already cover their own wire shapes, so these tests pin only what the
base owns: constructor validation, the authenticated header set including the per-class
`default_headers` merge, and the URL/timing plumbing of the `complete` template.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import pytest

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import (
    AssistantAction,
    BillingSource,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
)
from wmo.runtime.models.conftest import ScriptedAsyncJsonTransport
from wmo.runtime.models.providers.base import (
    DEFAULT_TIMEOUT_SECONDS,
    MAXIMUM_COMPLETION_TIMEOUT_SECONDS,
    ProviderHttpClient,
    completion_timeout_seconds,
)
from wmo.runtime.models.providers.transport import JsonHttpResponse


def _ok_transport() -> ScriptedAsyncJsonTransport:
    """Build a transport scripted with the single success body these tests expect."""
    return ScriptedAsyncJsonTransport([JsonHttpResponse(status_code=200, body={"answer": "ok"})])


class _EchoClient(ProviderHttpClient):
    """Minimal concrete client that exposes exactly what the base template supplies."""

    default_headers: ClassVar[Mapping[str, str]] = {"X-Fixture": "echo"}

    def _completion_path(self) -> str:
        """Return the fixture completion route."""
        return "echo/completions"

    def _build_request(self, request: ModelRequest) -> JsonObject:
        """Serialize just enough of the request to assert it reached the wire."""
        return {"first_content": request.messages[0].content}

    def _parse_response(self, payload: JsonObject, *, latency_seconds: float) -> ModelResponse:
        """Surface the payload and observed latency the template measured."""
        return ModelResponse.completed(
            output=AssistantAction(content=str(payload["answer"])),
            configured_model=self._model,
            served_model_id=None,
            usage=None,
            latency_seconds=latency_seconds,
        )


def _snapshot() -> ModelSnapshot:
    """Build an immutable identity fixture for the echo client."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="echo",
        model_id="echo-1",
        revision="fixture-revision",
        capabilities_sha256="a" * 64,
        connection_sha256="a" * 64,
    )


def _client(
    transport: ScriptedAsyncJsonTransport, *, base_url: str = "https://echo.test/v1/"
) -> _EchoClient:
    """Build the client under test against a deterministic transport."""
    return _EchoClient(
        model=_snapshot(),
        api_key="secret-key",
        base_url=base_url,
        transport=transport,
    )


def test_rejects_an_empty_api_key_naming_the_concrete_client() -> None:
    """Failing fast on a blank credential must name the client the operator configured."""
    with pytest.raises(ValueError, match="_EchoClient requires a non-empty API key"):
        _EchoClient(model=_snapshot(), api_key="", base_url="https://echo.test")


def test_rejects_a_nonpositive_timeout() -> None:
    """A zero timeout would turn every request into an immediate transport failure."""
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        _EchoClient(
            model=_snapshot(),
            api_key="secret-key",
            base_url="https://echo.test",
            timeout_seconds=0,
        )


def test_complete_posts_bearer_headers_and_class_defaults_to_the_completion_route() -> None:
    """The template owns the URL join, the auth headers, and the `default_headers` merge."""
    transport = _ok_transport()
    client = _client(transport)

    response = client.complete(ModelRequest(messages=(ModelMessage(role="user", content="hi"),)))

    (url, headers, payload) = transport.requests[0]
    assert url == "https://echo.test/v1/echo/completions"  # trailing slash stripped, path joined
    assert headers["Authorization"] == "Bearer secret-key"
    assert headers["Content-Type"] == "application/json"
    assert headers["X-Fixture"] == "echo"
    assert payload == {"first_content": "hi"}
    assert response.output.content == "ok"


def test_complete_reports_an_observed_nonnegative_latency() -> None:
    """`_parse_response` must receive the latency the template measured around the post."""
    client = _client(_ok_transport())

    response = client.complete(ModelRequest(messages=(ModelMessage(role="user", content="hi"),)))

    latency = response.economics.latency_seconds
    assert latency is not None
    assert latency.provenance == "observed"
    assert latency.value >= 0.0


def test_completion_timeout_keeps_the_configured_floor_for_small_or_absent_budgets() -> None:
    """Requests without a large output budget keep the configured per-attempt timeout."""
    assert completion_timeout_seconds(DEFAULT_TIMEOUT_SECONDS, None) == DEFAULT_TIMEOUT_SECONDS
    assert completion_timeout_seconds(DEFAULT_TIMEOUT_SECONDS, 1000) == DEFAULT_TIMEOUT_SECONDS


def test_completion_timeout_scales_with_the_requested_output_budget() -> None:
    """A 16000-token request gets a proportionally longer bounded attempt timeout."""
    assert completion_timeout_seconds(DEFAULT_TIMEOUT_SECONDS, 16_000) == pytest.approx(480.0)


def test_completion_timeout_caps_the_scaled_value_at_the_module_ceiling() -> None:
    """No output budget can push the derived attempt timeout past the finite ceiling."""
    derived = completion_timeout_seconds(DEFAULT_TIMEOUT_SECONDS, 10_000_000)
    assert derived == MAXIMUM_COMPLETION_TIMEOUT_SECONDS


def test_completion_timeout_never_reduces_an_explicitly_configured_timeout() -> None:
    """An operator floor above the scaled value and the ceiling is respected exactly."""
    assert completion_timeout_seconds(900.0, 16_000) == 900.0
    assert completion_timeout_seconds(120.0, 1000) == 120.0


def test_complete_passes_the_derived_timeout_to_the_transport() -> None:
    """The completion template posts each attempt under the token-scaled timeout."""
    transport = ScriptedAsyncJsonTransport(
        [
            JsonHttpResponse(status_code=200, body={"answer": "ok"}),
            JsonHttpResponse(status_code=200, body={"answer": "ok"}),
        ]
    )
    client = _client(transport)
    message = ModelMessage(role="user", content="hi")

    client.complete(ModelRequest(messages=(message,), maximum_output_tokens=16_000))
    client.complete(ModelRequest(messages=(message,)))

    assert transport.timeouts == [
        pytest.approx(480.0),
        pytest.approx(DEFAULT_TIMEOUT_SECONDS),
    ]
