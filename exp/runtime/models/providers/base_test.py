"""Tests for the shared HTTP provider template that every concrete client inherits.

The per-provider suites already cover their own wire shapes, so these tests pin only what the
base owns: constructor validation, the authenticated header set including the per-class
`default_headers` merge, and the URL/timing plumbing of the `complete` template.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    AssistantAction,
    BillingSource,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
)
from exp.runtime.models.providers.base import (
    DEFAULT_TIMEOUT_SECONDS,
    MAXIMUM_COMPLETION_TIMEOUT_SECONDS,
    GatewayWireProfile,
    ProviderHttpClient,
    completion_timeout_seconds,
)
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    JsonHttpTransport,
    ScriptedJsonTransport,
)


class _TimeoutRecordingTransport(ScriptedJsonTransport):
    """Scripted transport that also records the timeout given to every POST."""

    def __init__(self, responses: list[JsonHttpResponse]) -> None:
        """Store the scripted answers and start an empty timeout log.

        Args:
            responses: Responses to return, consumed in order.
        """
        super().__init__(responses)
        self.timeouts: list[float] = []

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Record the per-attempt timeout, then delegate to the scripted answer.

        Args:
            url: Absolute provider endpoint URL.
            headers: Request headers sent by the caller.
            payload: JSON request object sent by the caller.
            timeout_seconds: Bounded per-attempt timeout given by the client.

        Returns:
            The next scripted response.
        """
        self.timeouts.append(timeout_seconds)
        return super().post(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)


def _ok_transport() -> ScriptedJsonTransport:
    """Build a transport scripted with the single success body these tests expect."""
    return ScriptedJsonTransport([JsonHttpResponse(status_code=200, body={"answer": "ok"})])


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
    transport: JsonHttpTransport, *, base_url: str = "https://echo.test/v1/"
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


def test_gateway_wire_profile_rejects_malformed_generation_contracts() -> None:
    """Operator metadata cannot create an incoherent admission contract."""
    constructors = (
        lambda: GatewayWireProfile(
            dialect="openai_compatible",
            url="https://provider.test/v1/chat/completions",
            minimum_temperature=1.0,
            maximum_temperature=0.5,
        ),
        lambda: GatewayWireProfile(
            dialect="openai_compatible",
            url="https://provider.test/v1/chat/completions",
            minimum_top_p=0.8,
            maximum_top_p=0.2,
        ),
        lambda: GatewayWireProfile(
            dialect="openai_compatible",
            url="https://provider.test/v1/chat/completions",
            minimum_top_k=10,
            maximum_top_k=5,
        ),
        lambda: GatewayWireProfile(
            dialect="openai_compatible",
            url="https://provider.test/v1/chat/completions",
            maximum_temperature=float("nan"),
        ),
        lambda: GatewayWireProfile(
            dialect="openai_compatible",
            url="https://provider.test/v1/chat/completions",
            minimum_temperature=-0.1,
        ),
        lambda: GatewayWireProfile(
            dialect="openai_compatible",
            url="https://provider.test/v1/chat/completions",
            maximum_top_p=1.1,
        ),
        lambda: GatewayWireProfile(
            dialect="openai_compatible",
            url="https://provider.test/v1/chat/completions",
            maximum_output_tokens=0,
        ),
        lambda: GatewayWireProfile(
            dialect="openai_compatible",
            url="https://provider.test/v1/chat/completions",
            minimum_temperature=True,
        ),
        lambda: GatewayWireProfile(
            dialect="openai_compatible",
            url="https://provider.test/v1/chat/completions",
            maximum_output_tokens=True,
        ),
        lambda: GatewayWireProfile(
            dialect="not_implemented",
            url="https://provider.test/v1/chat/completions",
        ),
        lambda: GatewayWireProfile(
            dialect="openai_compatible",
            url="https://provider.test/v1/chat/completions",
            reasoning_effort="high",
        ),
        lambda: GatewayWireProfile(
            dialect="openai_compatible",
            url="https://provider.test/v1/chat/completions",
            sampling_requires_reasoning_none=True,
        ),
    )
    for constructor in constructors:
        with pytest.raises(ValueError):
            constructor()


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
    transport = _TimeoutRecordingTransport(
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
