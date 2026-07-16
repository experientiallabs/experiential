"""Tests for the Bedrock provider's Nova request/response path (no network)."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, cast

from wmh.providers.base import ChatRequest, Message, ProviderConfig, ProviderKind
from wmh.providers.bedrock import BedrockProvider, _is_nova

if TYPE_CHECKING:
    from botocore.client import BaseClient


class _StubClient:
    """Captures invoke_model calls and returns a canned body."""

    def __init__(self, response: dict) -> None:  # noqa: ANN401 - boto3 responses are untyped dicts
        self._response = response
        self.model_id: str | None = None
        self.body: dict | None = None

    def invoke_model(self, *, modelId: str, body: str) -> dict:  # noqa: N803 - boto3 kwarg name
        self.model_id = modelId
        self.body = json.loads(body)
        return {"body": io.BytesIO(json.dumps(self._response).encode("utf-8"))}


class _StubConverseClient:
    """Captures structured Converse requests and returns one text response."""

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def converse(self, **kwargs: object) -> dict[str, object]:
        self.requests.append(kwargs)
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 2, "outputTokens": 1},
        }


def test_is_nova_matches_nova_model_ids_only() -> None:
    assert _is_nova("us.amazon.nova-lite-v1:0")
    assert _is_nova("amazon.nova-micro-v1:0")
    assert not _is_nova("us.anthropic.claude-opus-4-8")
    assert not _is_nova("amazon.titan-embed-text-v2:0")


def test_nova_complete_builds_nova_body_and_parses_response() -> None:
    provider = BedrockProvider(
        ProviderConfig(kind=ProviderKind.BEDROCK, model="us.amazon.nova-lite-v1:0")
    )
    stub = _StubClient(
        {
            "output": {"message": {"content": [{"text": "hello "}, {"text": "world"}]}},
            "usage": {"inputTokens": 12, "outputTokens": 5},
        }
    )
    provider._client = cast("BaseClient", stub)  # inject; _get_client returns it

    completion = provider.complete(
        "be brief",
        [Message(role="user", content="hi")],
        temperature=0.4,
        max_tokens=64,
    )

    assert completion.text == "hello world"
    assert completion.usage.input_tokens == 12
    assert completion.usage.output_tokens == 5
    assert stub.model_id == "us.amazon.nova-lite-v1:0"
    assert stub.body == {
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
        "inferenceConfig": {"maxTokens": 64, "temperature": 0.4},
        "system": [{"text": "be brief"}],
    }


def test_nova_complete_omits_empty_system() -> None:
    provider = BedrockProvider(
        ProviderConfig(kind=ProviderKind.BEDROCK, model="us.amazon.nova-lite-v1:0")
    )
    stub = _StubClient(
        {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
    )
    provider._client = cast("BaseClient", stub)

    provider.complete("", [Message(role="user", content="hi")])
    assert stub.body is not None
    assert "system" not in stub.body


def test_structured_chat_normalizes_temperature_for_unsupported_model() -> None:
    provider = BedrockProvider(
        ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model_type="claude-opus-4-8",
            model="us.anthropic.claude-opus-4-8",
        )
    )
    stub = _StubConverseClient()
    provider._client = cast("BaseClient", stub)
    request = ChatRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.3,
            "max_completion_tokens": 64,
        }
    )

    provider.complete_chat(request)

    assert request.temperature == 0.3  # normalization does not mutate the reusable request
    assert stub.requests[0]["inferenceConfig"] == {"maxTokens": 64}


def test_structured_chat_preserves_temperature_for_supported_model() -> None:
    provider = BedrockProvider(
        ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model_type="claude-sonnet-4-6",
            model="us.anthropic.claude-sonnet-4-6",
        )
    )
    stub = _StubConverseClient()
    provider._client = cast("BaseClient", stub)

    provider.complete_chat(
        ChatRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.3,
                "max_completion_tokens": 64,
            }
        )
    )

    assert stub.requests[0]["inferenceConfig"] == {"maxTokens": 64, "temperature": 0.3}
