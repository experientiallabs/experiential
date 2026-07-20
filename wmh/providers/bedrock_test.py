"""Tests for the Bedrock provider's Nova request/response path (no network)."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, cast

import boto3
import pytest
from botocore.stub import Stubber

import wmh.providers.bedrock as mod
from wmh.providers.base import ChatRequest, Message, ProviderConfig, ProviderKind
from wmh.providers.bedrock import BedrockProvider, _is_nova
from wmh.providers.failure_attribution import ProviderBoundaryError, ProviderFailureStage

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
            "ResponseMetadata": {"RequestId": f"request-{len(self.requests)}"},
        }


def _haiku_provider() -> BedrockProvider:
    return BedrockProvider(
        ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model_type="claude-haiku-4-5",
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            region="us-east-1",
        )
    )


def _chat_request() -> ChatRequest:
    return ChatRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "submit"}],
            "max_completion_tokens": 64,
        }
    )


def test_structured_chat_stages_client_initialization_without_raw_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _haiku_provider()
    secret = "client-init-secret-sentinel"

    def fail_client() -> object:
        raise ValueError(secret)

    monkeypatch.setattr(provider, "_get_client", fail_client)

    with pytest.raises(ProviderBoundaryError) as caught:
        provider.complete_chat(_chat_request())

    assert caught.value.stage is ProviderFailureStage.CLIENT_INIT
    assert secret not in str(caught.value)


def test_structured_chat_stages_dispatch_without_raw_text() -> None:
    class FailingClient:
        def converse(self, **_kwargs: object) -> dict[str, object]:
            raise ValueError("dispatch-secret-sentinel")

    provider = _haiku_provider()
    provider._client = cast("BaseClient", FailingClient())

    with pytest.raises(ProviderBoundaryError) as caught:
        provider.complete_chat(_chat_request())

    assert caught.value.stage is ProviderFailureStage.DISPATCH
    assert "dispatch-secret-sentinel" not in str(caught.value)


def test_structured_chat_stages_response_translation_without_raw_text() -> None:
    class InvalidResponseClient:
        def converse(self, **_kwargs: object) -> dict[str, object]:
            return {"ResponseMetadata": {"RequestId": "request-1"}}

    provider = _haiku_provider()
    provider._client = cast("BaseClient", InvalidResponseClient())

    with pytest.raises(ProviderBoundaryError) as caught:
        provider.complete_chat(_chat_request())

    assert caught.value.stage is ProviderFailureStage.RESPONSE_TRANSLATION
    assert "Bedrock Converse" not in str(caught.value)


def test_structured_chat_stages_receipt_construction_without_raw_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _haiku_provider()
    provider._client = cast("BaseClient", _StubConverseClient())

    def fail_receipt(**_kwargs: object) -> object:
        raise ValueError("receipt-secret-sentinel")

    monkeypatch.setattr(mod, "build_chat_provider_receipt", fail_receipt)

    with pytest.raises(ProviderBoundaryError) as caught:
        provider.complete_chat(_chat_request())

    assert caught.value.stage is ProviderFailureStage.RECEIPT
    assert "receipt-secret-sentinel" not in str(caught.value)


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

    response = provider.complete_chat(request)

    assert request.temperature == 0.3  # normalization does not mutate the reusable request
    assert stub.requests[0]["inferenceConfig"] == {"maxTokens": 64}
    assert response.provider_receipt is not None
    assert response.provider_receipt.provider == "bedrock"
    assert response.provider_receipt.provider_request_id == "request-1"
    assert response.provider_receipt.response_id is None
    assert response.provider_receipt.requested_model == "us.anthropic.claude-opus-4-8"
    assert response.provider_receipt.response_model is None
    assert response.provider_receipt.temperature is None
    inference_config = cast("dict[str, object]", stub.requests[0]["inferenceConfig"])
    assert response.provider_receipt.max_tokens == inference_config["maxTokens"]
    assert response.provider_receipt.max_tokens_field == "inferenceConfig.maxTokens"


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

    response = provider.complete_chat(
        ChatRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.3,
                "max_completion_tokens": 64,
            }
        )
    )

    assert stub.requests[0]["inferenceConfig"] == {"maxTokens": 64, "temperature": 0.3}
    assert response.provider_receipt is not None
    assert response.provider_receipt.temperature == 0.3
    inference_config = cast("dict[str, object]", stub.requests[0]["inferenceConfig"])
    assert response.provider_receipt.max_tokens == inference_config["maxTokens"]
    assert response.provider_receipt.seed_supplied is False
    assert response.provider_receipt.cache_config_supplied is False


def test_structured_chat_without_request_metadata_remains_usable_without_receipt() -> None:
    class MissingRequestIdClient(_StubConverseClient):
        def converse(self, **kwargs: object) -> dict[str, object]:
            response = super().converse(**kwargs)
            response.pop("ResponseMetadata")
            return response

    provider = BedrockProvider(
        ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model_type="claude-sonnet-4-6",
            model="us.anthropic.claude-sonnet-4-6",
        )
    )
    stub = MissingRequestIdClient()
    provider._client = cast("BaseClient", stub)

    response = provider.complete_chat(
        ChatRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 64,
            }
        )
    )

    assert response.choices[0].message.content == "ok"
    assert response.provider_receipt is None


def test_structured_chat_binds_adaptive_max_effort_and_omits_temperature() -> None:
    provider = BedrockProvider(
        ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model_type="claude-opus-4-6",
            model="us.anthropic.claude-opus-4-6-v1",
            reasoning_effort="max",
        )
    )
    stub = _StubConverseClient()
    provider._client = cast("BaseClient", stub)

    response = provider.complete_chat(
        ChatRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.3,
                "max_completion_tokens": 64,
            }
        )
    )

    assert stub.requests[0]["inferenceConfig"] == {"maxTokens": 64}
    assert stub.requests[0]["additionalModelRequestFields"] == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "max"},
    }
    assert response.provider_receipt is not None
    assert response.provider_receipt.temperature is None
    assert response.provider_receipt.max_tokens == 64


def test_structured_chat_rejects_forced_tool_choice_before_sdk_request() -> None:
    provider = BedrockProvider(
        ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model_type="claude-opus-4-6",
            model="us.anthropic.claude-opus-4-6-v1",
            reasoning_effort="max",
        )
    )
    stub = _StubConverseClient()
    provider._client = cast("BaseClient", stub)

    with pytest.raises(ProviderBoundaryError) as caught:
        provider.complete_chat(
            ChatRequest.model_validate(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [
                        {
                            "function": {
                                "name": "read_file",
                                "parameters": {"type": "object"},
                            }
                        }
                    ],
                    "tool_choice": "required",
                }
            )
        )

    assert caught.value.stage is ProviderFailureStage.REQUEST_TRANSLATION
    assert "only auto or none" not in str(caught.value)
    assert stub.requests == []


def test_text_completion_cannot_silently_ignore_configured_reasoning() -> None:
    provider = BedrockProvider(
        ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model_type="claude-opus-4-6",
            model="us.anthropic.claude-opus-4-6-v1",
            reasoning_effort="max",
        )
    )
    stub = _StubConverseClient()
    provider._client = cast("BaseClient", stub)

    completion = provider.complete(
        "be precise",
        [Message(role="user", content="hi")],
        temperature=0.2,
        max_tokens=128,
    )

    assert completion.text == "ok"
    assert stub.requests[0]["system"] == [{"text": "be precise"}]
    assert stub.requests[0]["additionalModelRequestFields"] == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "max"},
    }


def test_real_sdk_shape_accepts_adaptive_request_and_signed_multiturn_replay() -> None:
    model = "us.anthropic.claude-opus-4-6-v1"
    provider = BedrockProvider(
        ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model_type="claude-opus-4-6",
            model=model,
            reasoning_effort="max",
            region="us-east-1",
        )
    )
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    provider._client = cast("BaseClient", client)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "read a file",
                "parameters": {"type": "object"},
            },
        }
    ]
    first_request = ChatRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "inspect"}],
            "tools": tools,
            "max_completion_tokens": 64,
        }
    )
    signed_content = [
        {
            "reasoningContent": {
                "reasoningText": {"text": "inspect first", "signature": "signature"}
            }
        },
        {"reasoningContent": {"redactedContent": b"redacted"}},
        {
            "toolUse": {
                "toolUseId": "call-1",
                "name": "read_file",
                "input": {"path": "README.md"},
            }
        },
    ]
    common = {
        "modelId": model,
        "inferenceConfig": {"maxTokens": 64},
        "additionalModelRequestFields": {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "max"},
        },
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "name": "read_file",
                        "description": "read a file",
                        "inputSchema": {"json": {"type": "object"}},
                    }
                }
            ]
        },
    }
    stubber = Stubber(client)
    stubber.add_response(
        "converse",
        {
            "output": {"message": {"role": "assistant", "content": signed_content}},
            "stopReason": "tool_use",
            "usage": {"inputTokens": 3, "outputTokens": 4, "totalTokens": 7},
            "metrics": {"latencyMs": 1},
        },
        {**common, "messages": [{"role": "user", "content": [{"text": "inspect"}]}]},
    )

    with stubber:
        first = provider.complete_chat(first_request)
        assistant = first.choices[0].message.model_dump(mode="json", exclude_none=True)
        followup = ChatRequest.model_validate(
            {
                "messages": [
                    {"role": "user", "content": "inspect"},
                    assistant,
                    {"role": "tool", "tool_call_id": "call-1", "content": "contents"},
                ],
                "tools": tools,
                "max_completion_tokens": 64,
            }
        )
        stubber.add_response(
            "converse",
            {
                "output": {"message": {"role": "assistant", "content": [{"text": "done"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 8, "outputTokens": 1, "totalTokens": 9},
                "metrics": {"latencyMs": 1},
            },
            {
                **common,
                "messages": [
                    {"role": "user", "content": [{"text": "inspect"}]},
                    {"role": "assistant", "content": signed_content},
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": "call-1",
                                    "content": [{"text": "contents"}],
                                }
                            }
                        ],
                    },
                ],
            },
        )

        second = provider.complete_chat(followup)

    assert second.choices[0].message.content == "done"
