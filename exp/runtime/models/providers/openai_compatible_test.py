"""Contract tests for shared OpenAI-compatible conversion and client behavior.

This module owns the fixtures shared by the provider suites: `_snapshot` and `_request`
are imported by `azure_test` and `native_test` so every adapter exercises one transcript.
"""

from __future__ import annotations

import math
import os
from typing import Literal

import pytest

from exp.common.models import (
    AssistantAction,
    BillingSource,
    ModelMessage,
    ModelRequest,
    ModelSnapshot,
    ToolCall,
    ToolChoice,
)
from exp.common.tasks import ToolSchema
from exp.runtime.models.providers.errors import (
    ProviderRefusalError,
    ProviderRefusalSignal,
    ProviderResponseError,
)
from exp.runtime.models.providers.openai_compatible import (
    OpenAICompatibleClient,
    OpenAICompatibleResponseError,
    openai_compatible_request,
    openai_compatible_response,
    openai_embedding_request,
    openai_embedding_response_raw,
)
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    RetryPolicy,
    ScriptedJsonTransport,
)


def _snapshot(provider: str = "openai-compatible", model_id: str = "fake-model") -> ModelSnapshot:
    """Build an immutable identity fixture for one adapter.

    Args:
        provider: Catalog provider name under test.
        model_id: Exact configured model or deployment identity.

    Returns:
        A frozen snapshot with fixture digests.
    """
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider=provider,
        model_id=model_id,
        revision="fixture-revision",
        capabilities_sha256="a" * 64,
        connection_sha256="a" * 64,
    )


def _request(
    *,
    tool_choice: ToolChoice | Literal["auto", "none", "required"] | None = None,
    top_p: float | None = None,
) -> ModelRequest:
    """Build a visible transcript containing an earlier tool call and result.

    Args:
        tool_choice: Optional tool-choice constraint forwarded to the request.
        top_p: Optional nucleus-sampling mass forwarded to the request.

    Returns:
        A typed request with system, user, assistant tool-call, and tool-result turns.
    """
    return ModelRequest(
        messages=(
            ModelMessage(role="system", content="You are precise."),
            ModelMessage(role="user", content="Create a ticket."),
            ModelMessage(
                role="assistant",
                assistant_action=AssistantAction(
                    tool_calls=(
                        ToolCall(
                            call_id="call-old",
                            name="create_ticket",
                            arguments={"priority": "normal"},
                        ),
                    )
                ),
            ),
            ModelMessage(role="tool", content="created", tool_call_id="call-old"),
        ),
        tools=(
            ToolSchema(
                name="create_ticket",
                description="Create one support ticket.",
                input_schema={"type": "object"},
            ),
        ),
        tool_choice=tool_choice,
        temperature=0.2,
        top_p=top_p,
        maximum_output_tokens=128,
    )


def test_openai_compatible_request_keeps_history_tools_and_non_streaming_cap() -> None:
    """Shared conversion keeps every tool turn and emits no streaming request."""
    payload = openai_compatible_request(
        "fake-model", _request(tool_choice=ToolChoice(name="create_ticket"), top_p=1.0)
    )

    assert payload["stream"] is False
    assert payload["max_tokens"] == 128
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 1.0
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "create_ticket"},
    }
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "create_ticket",
                "description": "Create one support ticket.",
                "parameters": {"type": "object"},
            },
        }
    ]
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert messages[2]["tool_calls"] == [
        {
            "id": "call-old",
            "type": "function",
            "function": {"name": "create_ticket", "arguments": '{"priority": "normal"}'},
        }
    ]


def test_openai_compatible_request_omits_absent_top_p() -> None:
    """Buffered Chat payloads do not invent a nucleus-sampling value."""
    payload = openai_compatible_request(
        "fake-model",
        ModelRequest(messages=(ModelMessage(role="user", content="hello"),)),
    )

    assert "top_p" not in payload
    assert "temperature" not in payload


def test_openai_compatible_client_converts_tool_usage_and_resolved_identity() -> None:
    """One frozen tool response produces typed output, normalized usage, and actual model ID."""
    transport = ScriptedJsonTransport(
        [
            JsonHttpResponse(
                status_code=200,
                body={
                    "model": "served-model-20260811",
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-new",
                                        "function": {
                                            "name": "create_ticket",
                                            "arguments": '{"priority":"urgent"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 7,
                        "prompt_tokens_details": {"cached_tokens": 5},
                    },
                },
            )
        ]
    )
    client = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://example.test/v1",
        api_key="fake-key",
        transport=transport,
    )

    response = client.complete(_request())

    assert response.model.model_id == "served-model-20260811"
    assert response.output.content is None
    assert response.output.tool_calls == (
        ToolCall(
            call_id="call-new",
            name="create_ticket",
            arguments={"priority": "urgent"},
            raw_arguments='{"priority":"urgent"}',
        ),
    )
    assert response.economics.usage is not None
    assert response.economics.usage.input_tokens == 12
    assert response.economics.usage.cached_input_tokens == 5
    assert response.economics.latency_seconds is not None
    assert transport.requests[0][0] == "https://example.test/v1/chat/completions"
    assert transport.requests[0][1]["Authorization"] == "Bearer fake-key"


def test_openai_compatible_client_retries_only_the_same_endpoint() -> None:
    """A retryable status retries the frozen request without a failover model path."""
    transport = ScriptedJsonTransport(
        [
            JsonHttpResponse(status_code=503, body={"error": {"message": "busy"}}),
            JsonHttpResponse(
                status_code=200,
                body={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            ),
        ]
    )
    client = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://example.test/v1",
        api_key="fake-key",
        transport=transport,
        retry_policy=RetryPolicy(maximum_attempts=2, initial_delay_seconds=0),
    )

    assert client.complete(_request()).output.content == "ok"
    assert [request[0] for request in transport.requests] == [
        "https://example.test/v1/chat/completions",
        "https://example.test/v1/chat/completions",
    ]
    idempotency_keys = [request[1]["Idempotency-Key"] for request in transport.requests]
    assert idempotency_keys[0].startswith("exp-")
    assert idempotency_keys[0] == idempotency_keys[1]


def test_response_without_choices_fails_closed_without_exposing_the_key() -> None:
    """A response with no choices raises a typed error that never includes the credential."""
    secret = "fake-secret-key-value"
    transport = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body={"choices": []})])
    client = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://example.test/v1",
        api_key=secret,
        transport=transport,
    )

    with pytest.raises(OpenAICompatibleResponseError, match="no choices") as captured:
        client.complete(_request())
    assert secret not in str(captured.value)
    assert secret not in repr(captured.value)


def test_openai_compatible_embedding_response_is_ordered_and_normalized() -> None:
    """Embedding conversion restores provider indexes and returns unit-length vectors."""
    transport = ScriptedJsonTransport(
        [
            JsonHttpResponse(
                status_code=200,
                body={
                    "data": [
                        {"index": 1, "embedding": [0.0, 3.0]},
                        {"index": 0, "embedding": [4.0, 0.0]},
                    ]
                },
            )
        ]
    )
    client = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://example.test/v1",
        api_key="fake-key",
        transport=transport,
    )

    embeddings = client.embed(("first", "second"))

    assert embeddings[0].values == (1.0, 0.0)
    assert embeddings[1].values == (0.0, 1.0)
    assert all(
        math.isclose(sum(value * value for value in item.values), 1.0) for item in embeddings
    )
    assert transport.requests[0][0] == "https://example.test/v1/embeddings"


def test_openai_embedding_request_carries_optional_dimensions_and_encoding() -> None:
    """Optional dimensions and encoding_format ride the wire only when supplied."""
    assert openai_embedding_request("m", ("a", "b")) == {"model": "m", "input": ["a", "b"]}
    assert openai_embedding_request("m", ("a",), dimensions=256, encoding_format="float") == {
        "model": "m",
        "input": ["a"],
        "dimensions": 256,
        "encoding_format": "float",
    }


def test_openai_embedding_response_raw_preserves_vectors_and_reads_usage() -> None:
    """The raw parser restores input order, keeps raw magnitude, and reads prompt tokens."""
    batch = openai_embedding_response_raw(
        {
            "model": "text-embedding-3-small",
            "data": [
                {"index": 1, "embedding": [0.0, 3.0]},
                {"index": 0, "embedding": [4.0, 0.0]},
            ],
            "usage": {"prompt_tokens": 7, "total_tokens": 7},
        },
        expected_count=2,
    )

    # Raw magnitudes are preserved, not renormalized to unit length.
    assert batch.embeddings[0].values == (4.0, 0.0)
    assert batch.embeddings[1].values == (0.0, 3.0)
    assert batch.prompt_tokens == 7
    assert batch.served_model_id == "text-embedding-3-small"


def test_openai_embedding_response_raw_requires_usage_for_billing() -> None:
    """The billed surface refuses a response missing the input-token count."""
    with pytest.raises(ProviderResponseError, match="usage"):
        openai_embedding_response_raw(
            {"data": [{"index": 0, "embedding": [1.0, 2.0]}]},
            expected_count=1,
        )
    # A present usage object with an omitted prompt_tokens must not bill as zero.
    with pytest.raises(ProviderResponseError, match="usage.prompt_tokens"):
        openai_embedding_response_raw(
            {
                "data": [{"index": 0, "embedding": [1.0, 2.0]}],
                "usage": {"total_tokens": 5},
            },
            expected_count=1,
        )


def test_embed_raw_rejects_empty_input() -> None:
    """Embedding no text is a caller error on the public surface, not an empty request."""
    client = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://example.test/v1",
        api_key="fake-key",
        transport=ScriptedJsonTransport([]),
    )
    with pytest.raises(ValueError, match="at least one input text"):
        client.embed_raw(())


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="live OpenAI embeddings test requires OPENAI_API_KEY",
)
def test_embed_raw_against_live_openai() -> None:
    """A real text-embedding-3-small call returns raw vectors and billed input tokens."""
    client = OpenAICompatibleClient(
        model=_snapshot(provider="openai", model_id="text-embedding-3-small"),
        base_url="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    batch = client.embed_raw(("hello world", "second input"))

    assert len(batch.embeddings) == 2
    assert len(batch.embeddings[0].values) == 1536
    # Distinct inputs yield distinct vectors: the raw parser preserved order and content.
    assert batch.embeddings[0].values != batch.embeddings[1].values
    # The surface bills the provider's reported input tokens, so they must be present.
    assert batch.prompt_tokens > 0
    # A reduced-dimension request rides the wire and returns the narrower vector.
    batch_dim = client.embed_raw(("hello world",), dimensions=256)
    assert len(batch_dim.embeddings[0].values) == 256


def test_openai_compatible_conversion_rejects_malformed_tool_arguments() -> None:
    """A provider cannot turn malformed tool JSON into an invented empty argument object."""
    transport = ScriptedJsonTransport(
        [
            JsonHttpResponse(
                status_code=200,
                body={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "bad",
                                        "function": {"name": "create_ticket", "arguments": "{"},
                                    }
                                ],
                            }
                        }
                    ]
                },
            )
        ]
    )
    client = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://example.test/v1",
        api_key="fake-key",
        transport=transport,
    )

    with pytest.raises(OpenAICompatibleResponseError, match="arguments is not JSON"):
        client.complete(_request())


def test_openai_compatible_refusal_is_typed_without_exposing_content() -> None:
    """Content-filter finish state must not be folded into visible assistant text."""
    canary = "compatible-refusal-canary"

    with pytest.raises(ProviderRefusalError) as error:
        openai_compatible_response(
            {
                "choices": [
                    {
                        "finish_reason": "content_filter",
                        "message": {"content": canary},
                    }
                ]
            },
            configured_model=_snapshot(),
            latency_seconds=0.1,
        )

    assert error.value.signal is ProviderRefusalSignal.CONTENT_POLICY
    assert canary not in str(error.value)


_HUNYUAN_BASE_URL = "https://api.hunyuan.cloud.tencent.com/v1"


def test_hunyuan_rung_exposes_reasoning_only_when_the_capability_is_declared() -> None:
    """Plaintext reasoning is exposed per rung, never inferred from the endpoint."""
    exposed = OpenAICompatibleClient(
        model=_snapshot(),
        base_url=_HUNYUAN_BASE_URL,
        api_key="fake-key",
        reasoning_output_exposed=True,
    ).gateway_wire_profile()
    assert exposed.hunyuan_reasoning_route_sha256 is not None
    assert exposed.reasoning_output_exposed is True


def test_hunyuan_rung_without_the_capability_stays_stripped_but_keeps_its_carrier_route() -> None:
    """An undeclared rung on the Hunyuan endpoint fails closed on exposure.

    The carrier route identity still resolves so tool-loop replay stays sealed,
    but the caller never sees plaintext ``reasoning_content`` — closing the hole
    where endpoint detection alone would expose every model on the endpoint.
    """
    profile = OpenAICompatibleClient(
        model=_snapshot(),
        base_url=_HUNYUAN_BASE_URL,
        api_key="fake-key",
    ).gateway_wire_profile()
    assert profile.hunyuan_reasoning_route_sha256 is not None
    assert profile.reasoning_output_exposed is False


def test_reasoning_exposure_requires_a_carrier_route_even_when_declared() -> None:
    """A declared capability without a carrier route exposes nothing (both gates)."""
    profile = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://example.test/v1",
        api_key="fake-key",
        reasoning_output_exposed=True,
    ).gateway_wire_profile()
    assert profile.hunyuan_reasoning_route_sha256 is None
    assert profile.reasoning_output_exposed is False


def test_tokenhub_intl_rung_resolves_a_carrier_route_and_exposes_when_declared() -> None:
    """The TokenHub-intl origin the platform serves through is a Hunyuan route.

    This is the endpoint the live Tencent lane dispatches through; recognizing
    it is what makes the carrier route resolve so plaintext reasoning returns and
    round-trips instead of being stripped.
    """
    profile = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://tokenhub-intl.tencentcloudmaas.com/v1",
        api_key="fake-key",
        reasoning_output_exposed=True,
    ).gateway_wire_profile()
    assert profile.hunyuan_reasoning_route_sha256 is not None
    assert profile.reasoning_output_exposed is True
