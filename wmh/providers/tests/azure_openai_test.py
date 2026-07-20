"""Unit tests for AzureOpenAIProvider. No network: the SDK client is faked via _get_client."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

import httpx
import pytest
from openai import AzureOpenAI, OpenAI

from wmh.providers.azure_openai import AzureOpenAIProvider
from wmh.providers.base import ChatRequest, Message, ProviderConfig, ProviderKind
from wmh.providers.receipt import validate_chat_provider_receipt


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _FakeChatResponse:
    def __init__(self, content: str, usage: _FakeUsage | None) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = usage
        self.id = "completion-azure-1"
        self.model = "gpt-5.5-2026-06-01"
        self.system_fingerprint = "fp-azure-1"

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {
            "id": self.id,
            "model": self.model,
            "system_fingerprint": self.system_fingerprint,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.choices[0].message.content},
                    "finish_reason": "stop",
                }
            ],
            "usage": (
                {
                    "prompt_tokens": self.usage.prompt_tokens,
                    "completion_tokens": self.usage.completion_tokens,
                }
                if self.usage is not None
                else None
            ),
        }


class _FakeChatCompletions:
    def __init__(
        self,
        response: _FakeChatResponse,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.response = response
        self.headers = (
            headers if headers is not None else {"apim-request-id": "provider-request-azure-1"}
        )
        self.last_kwargs: dict[str, object] = {}
        self.with_raw_response = _FakeWithRawResponse(self)

    def create(self, **kwargs: object) -> _FakeChatResponse:
        self.last_kwargs = kwargs
        return self.response


class _FakeRawAPIResponse:
    def __init__(self, response: _FakeChatResponse, headers: dict[str, str]) -> None:
        self.response = response
        self.headers = headers

    def parse(self) -> _FakeChatResponse:
        return self.response


class _FakeWithRawResponse:
    def __init__(self, completions: _FakeChatCompletions) -> None:
        self.completions = completions

    def create(self, **kwargs: object) -> _FakeRawAPIResponse:
        self.completions.last_kwargs = kwargs
        return _FakeRawAPIResponse(self.completions.response, self.completions.headers)


class _FakeChat:
    def __init__(self, completions: _FakeChatCompletions) -> None:
        self.completions = completions


class _FakeResponsesResponse:
    def __init__(self, tool_name: str = "bash") -> None:
        self.tool_name = tool_name

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {
            "id": "response-azure-1",
            "model": "gpt-5.5-2026-06-01",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "reasoning-1",
                    "summary": [],
                    "encrypted_content": "ciphertext-1",
                },
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": self.tool_name,
                    "arguments": '{"command":"pwd"}',
                },
            ],
            "usage": {"input_tokens": 11, "output_tokens": 7},
        }


class _FakeResponses:
    def __init__(
        self,
        *,
        tool_name: str = "bash",
        headers: dict[str, str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.headers = (
            {"apim-request-id": "provider-request-azure-responses-1"}
            if headers is None
            else headers
        )
        self.error = error
        self.last_kwargs: dict[str, object] = {}
        self.calls: list[dict[str, object]] = []
        self.with_raw_response = _FakeResponsesWithRawResponse(self)

    def create(self, **kwargs: object) -> _FakeResponsesResponse:
        self.last_kwargs = kwargs
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return _FakeResponsesResponse(self.tool_name)


class _FakeResponsesRawResponse:
    def __init__(self, response: _FakeResponsesResponse, headers: dict[str, str]) -> None:
        self._response = response
        self.headers = headers

    def parse(self) -> _FakeResponsesResponse:
        return self._response


class _FakeResponsesWithRawResponse:
    def __init__(self, responses: _FakeResponses) -> None:
        self._responses = responses

    def create(self, **kwargs: object) -> _FakeResponsesRawResponse:
        return _FakeResponsesRawResponse(
            self._responses.create(**kwargs),
            self._responses.headers,
        )


class _FakeResponsesClient:
    def __init__(self, responses: _FakeResponses) -> None:
        self.responses = responses


class _FakeEmbeddingItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [_FakeEmbeddingItem(v) for v in vectors]


class _FakeEmbeddings:
    def __init__(self, response: _FakeEmbeddingResponse) -> None:
        self.response = response
        self.last_kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> _FakeEmbeddingResponse:
        self.last_kwargs = kwargs
        return self.response


class _FakeClient:
    def __init__(
        self, chat: _FakeChatCompletions, embeddings: _FakeEmbeddings | None = None
    ) -> None:
        self.chat = _FakeChat(chat)
        self.embeddings = embeddings


def _config() -> ProviderConfig:
    return ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        endpoint="https://example.openai.azure.com",
        deployment="gpt55-deploy",
        api_version="2024-10-21",
    )


def _reasoning_config() -> ProviderConfig:
    return ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        endpoint="https://example.openai.azure.com",
        deployment="gpt55-deploy",
        api_version="2024-10-21",
        reasoning_effort="high",
        responses_api_version="v1",
    )


def test_reasoning_config_requires_explicit_responses_api_version() -> None:
    with pytest.raises(ValueError, match="responses_api_version"):
        ProviderConfig(
            kind=ProviderKind.AZURE_OPENAI,
            model="gpt-5.5",
            endpoint="https://example.openai.azure.com",
            deployment="gpt55-deploy",
            api_version="2024-10-21",
            reasoning_effort="high",
        )


def test_complete_sends_deployment_as_model(monkeypatch: pytest.MonkeyPatch) -> None:
    chat = _FakeChatCompletions(_FakeChatResponse("yo", _FakeUsage(3, 2)))
    provider = AzureOpenAIProvider(_config())
    fake = _FakeClient(chat)
    monkeypatch.setattr(provider, "_get_client", lambda: fake)  # inject fake; no network

    completion = provider.complete("sys", [Message(role="user", content="hi")], max_tokens=16)

    assert completion.text == "yo"
    assert completion.usage.input_tokens == 3
    # On Azure the `model` arg carries the deployment name, not the base model id.
    assert chat.last_kwargs["model"] == "gpt55-deploy"
    assert chat.last_kwargs["max_completion_tokens"] == 16


def test_complete_binds_configured_reasoning_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    chat = _FakeChatCompletions(_FakeChatResponse("yo", _FakeUsage(3, 2)))
    provider = AzureOpenAIProvider(_reasoning_config())
    monkeypatch.setattr(provider, "_get_client", lambda: _FakeClient(chat))

    provider.complete("sys", [Message(role="user", content="hi")], max_tokens=16)

    assert chat.last_kwargs["reasoning_effort"] == "high"
    assert "temperature" not in chat.last_kwargs


def test_complete_preserves_missing_usage_for_budget_forfeit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = _FakeChatCompletions(_FakeChatResponse("yo", None))
    provider = AzureOpenAIProvider(_config())
    monkeypatch.setattr(provider, "_get_client", lambda: _FakeClient(chat))

    completion = provider.complete("sys", [Message(role="user", content="hi")], max_tokens=16)

    assert completion.text == "yo"
    assert "usage" not in completion.model_fields_set


@pytest.mark.parametrize(
    ("model_type", "expects_temperature"),
    [("gpt-5.5", False), ("deepseek-v4-pro", True)],
)
def test_structured_chat_applies_model_temperature_capability(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    expects_temperature: bool,
) -> None:
    config = _config().model_copy(update={"model_type": model_type, "model": model_type})
    provider = AzureOpenAIProvider(config)
    chat = _FakeChatCompletions(_FakeChatResponse("ok", _FakeUsage(1, 1)))
    monkeypatch.setattr(provider, "_get_client", lambda: _FakeClient(chat))

    response = provider.complete_chat(
        ChatRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "go"}],
                "temperature": 0.3,
                "max_completion_tokens": 64,
            }
        )
    )

    assert ("temperature" in chat.last_kwargs) is expects_temperature
    assert response.provider_receipt is not None
    assert response.provider_receipt.provider == "azure"
    assert response.provider_receipt.provider_request_id == "provider-request-azure-1"
    assert response.provider_receipt.response_id == "completion-azure-1"
    assert response.provider_receipt.requested_model == "gpt55-deploy"
    assert response.provider_receipt.response_model == "gpt-5.5-2026-06-01"
    assert response.provider_receipt.system_fingerprint == "fp-azure-1"
    assert response.provider_receipt.temperature == (0.3 if expects_temperature else None)


def test_structured_gpt55_reasoning_uses_azure_v1_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = _FakeResponses()
    provider = AzureOpenAIProvider(_reasoning_config())
    monkeypatch.setattr(
        provider,
        "_get_responses_client",
        lambda: _FakeResponsesClient(responses),
    )
    monkeypatch.setattr(
        provider,
        "_get_client",
        lambda: (_ for _ in ()).throw(AssertionError("reasoning tools must not use chat")),
    )

    response = provider.complete_chat(
        ChatRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "inspect"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "description": "run a command",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "temperature": 0.3,
                "top_p": 0.7,
                "max_completion_tokens": 4096,
            }
        )
    )

    assert responses.last_kwargs == {
        "model": "gpt55-deploy",
        "input": [{"role": "user", "content": "inspect"}],
        "stream": False,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "max_output_tokens": 4096,
        "tools": [
            {
                "type": "function",
                "name": "bash",
                "description": "run a command",
                "parameters": {"type": "object"},
            }
        ],
        "reasoning": {"effort": "high"},
    }
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls is not None
    assert response.choices[0].message.tool_calls[0].function.name == "bash"
    assert response.token_usage().input_tokens == 11
    assert response.token_usage().output_tokens == 7
    assert response.provider_receipt is not None
    assert response.provider_receipt.provider == "azure"
    assert response.provider_receipt.provider_request_id == ("provider-request-azure-responses-1")
    assert response.provider_receipt.response_id == "response-azure-1"
    assert response.provider_receipt.requested_model == "gpt55-deploy"
    assert response.provider_receipt.response_model == "gpt-5.5-2026-06-01"
    assert response.provider_receipt.max_tokens_field == "max_output_tokens"
    validate_chat_provider_receipt(
        response.provider_receipt,
        provider_config=provider.config,
        requested_temperature=0.3,
        max_tokens=4096,
    )

    assistant = response.choices[0].message.model_dump(mode="json", exclude_none=True)
    reasoning_details = assistant["reasoning_details"]
    assert isinstance(reasoning_details, list)
    reasoning_data = reasoning_details[0]["data"]
    assert isinstance(reasoning_data, str)
    assert json.loads(reasoning_data)["provider"] == "azure"
    provider.complete_chat(
        ChatRequest.model_validate(
            {
                "messages": [
                    {"role": "user", "content": "inspect"},
                    assistant,
                    {"role": "tool", "tool_call_id": "call-1", "content": "/workspace"},
                ],
                "max_completion_tokens": 4096,
            }
        )
    )
    assert len(responses.calls) == 2
    replay_input = responses.last_kwargs["input"]
    assert isinstance(replay_input, list)
    assert replay_input[-1] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "/workspace",
    }


def test_responses_client_uses_trusted_azure_key_and_normalized_v1_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[dict[str, object]] = []

    class _FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            constructed.append(kwargs)

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-real-secret")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://Trusted.openai.azure.com")
    monkeypatch.setenv("WMH_ENDPOINT_API_KEY", "endpoint-token")
    provider = AzureOpenAIProvider(
        _endpoint_config("https://trusted.openai.azure.com/").model_copy(
            update={"reasoning_effort": "high", "responses_api_version": "v1"}
        )
    )

    first = provider._get_responses_client()
    second = provider._get_responses_client()

    assert first is second
    assert constructed == [
        {
            "api_key": "az-real-secret",
            "base_url": "https://trusted.openai.azure.com/openai/v1/",
            "default_query": {"api-version": "v1"},
            "max_retries": 0,
        }
    ]


def test_untrusted_responses_endpoint_never_receives_real_azure_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[dict[str, object]] = []

    class _FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            constructed.append(kwargs)

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-real-secret")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://trusted.openai.azure.com")
    monkeypatch.setenv("WMH_ENDPOINT_API_KEY", "endpoint-token")
    provider = AzureOpenAIProvider(
        _endpoint_config("https://untrusted.example").model_copy(
            update={"reasoning_effort": "high", "responses_api_version": "v1"}
        )
    )

    provider._get_responses_client()

    assert constructed == [
        {
            "api_key": "endpoint-token",
            "base_url": "https://untrusted.example/openai/v1/",
            "default_query": {"api-version": "v1"},
            "max_retries": 0,
        }
    ]


def test_reasoning_transport_sends_explicit_v1_query_and_receipt_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"apim-request-id": "azure-v1-request-1"},
            json={
                "id": "resp-azure-v1-1",
                "model": "gpt-5.5-2026-06-01",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "reasoning-1",
                        "summary": [],
                        "encrypted_content": "ciphertext-1",
                    },
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "bash",
                        "arguments": '{"command":"pwd"}',
                        "status": "completed",
                    },
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    real_openai = OpenAI

    def build_openai(**kwargs: object) -> OpenAI:
        api_key = kwargs.get("api_key")
        base_url = kwargs.get("base_url")
        default_query = kwargs.get("default_query")
        max_retries = kwargs.get("max_retries")
        assert isinstance(api_key, str)
        assert isinstance(base_url, str)
        assert isinstance(default_query, Mapping)
        assert isinstance(max_retries, int)
        return real_openai(
            api_key=api_key,
            base_url=base_url,
            default_query=cast("Mapping[str, object]", default_query),
            max_retries=max_retries,
            http_client=http_client,
        )

    monkeypatch.setattr("openai.OpenAI", build_openai)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    provider = AzureOpenAIProvider(_reasoning_config())
    try:
        response = provider.complete_chat(
            ChatRequest.model_validate(
                {
                    "messages": [{"role": "user", "content": "inspect"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                    "max_completion_tokens": 64,
                }
            )
        )
    finally:
        if provider._responses_client is not None:
            provider._responses_client.close()

    assert len(requests) == 1
    assert str(requests[0].url) == (
        "https://example.openai.azure.com/openai/v1/responses?api-version=v1"
    )
    assert response.provider_receipt is not None
    assert response.provider_receipt.provider_request_id == "azure-v1-request-1"


def test_structured_chat_uses_azure_apim_request_id_from_real_sdk_transport() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "apim-request-id": "azure-transport-request-1",
                "x-sensitive-test-header": "must-not-be-persisted",
            },
            json={
                "id": "completion-azure-sdk-1",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-5.5-2026-06-01",
                "system_fingerprint": "fp-azure-sdk-1",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = AzureOpenAI(
        api_key="dummy-test-key",
        api_version="2024-10-21",
        azure_endpoint="https://example.openai.azure.com",
        http_client=http_client,
        max_retries=0,
    )
    provider = AzureOpenAIProvider(_config())
    provider._client = client
    try:
        response = provider.complete_chat(
            ChatRequest.model_validate(
                {
                    "messages": [{"role": "user", "content": "go"}],
                    "max_completion_tokens": 64,
                }
            )
        )
    finally:
        client.close()

    assert response.provider_receipt is not None
    assert response.provider_receipt.provider_request_id == "azure-transport-request-1"
    assert response.provider_receipt.response_id == "completion-azure-sdk-1"
    receipt_json = response.provider_receipt.model_dump_json()
    assert "x-sensitive-test-header" not in receipt_json
    assert "must-not-be-persisted" not in receipt_json


def test_missing_deployment_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = AzureOpenAIProvider(
        ProviderConfig(kind=ProviderKind.AZURE_OPENAI, model="gpt-5.5", api_version="2024-10-21")
    )
    # Fake the client so the missing-deployment ValueError is the only thing that can raise
    # (complete() evaluates _get_client() before _deployment(), so a real client would try to
    # construct first).
    fake = _FakeClient(_FakeChatCompletions(_FakeChatResponse("", _FakeUsage(0, 0))))
    monkeypatch.setattr(provider, "_get_client", lambda: fake)
    with pytest.raises(ValueError, match="deployment"):
        provider.complete("", [Message(role="user", content="x")])


def test_embed_uses_embed_model_as_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    embeddings = _FakeEmbeddings(_FakeEmbeddingResponse([[0.5, 0.6]]))
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        endpoint="https://example.openai.azure.com",
        deployment="gpt55-deploy",
        api_version="2024-10-21",
        embed_model="embed-deploy",
    )
    provider = AzureOpenAIProvider(config)
    fake = _FakeClient(_FakeChatCompletions(_FakeChatResponse("", _FakeUsage(0, 0))), embeddings)
    monkeypatch.setattr(provider, "_get_client", lambda: fake)

    vectors = provider.embed(["a"])

    assert vectors == [[0.5, 0.6]]
    # embed_model is sent as the Azure deployment name (the `model` arg).
    assert embeddings.last_kwargs["model"] == "embed-deploy"


def test_embed_requires_embed_model() -> None:
    provider = AzureOpenAIProvider(_config())  # _config() sets no embed_model
    with pytest.raises(ValueError, match="embed_model"):
        provider.embed(["x"])


def test_get_client_requires_api_version(monkeypatch: pytest.MonkeyPatch) -> None:
    # api_version is config-supplied; without it we must fail clearly before constructing.
    provider = AzureOpenAIProvider(
        ProviderConfig(
            kind=ProviderKind.AZURE_OPENAI,
            model="gpt-5.5",
            endpoint="https://example.openai.azure.com",
            deployment="d",
        )
    )
    with pytest.raises(ValueError, match="api_version"):
        provider._get_client()


def test_verify_reports_failure_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        class completions:  # noqa: N801 - mimic the SDK attribute path
            @staticmethod
            def create(**kwargs: object) -> object:
                raise RuntimeError("bad endpoint")

    fake = type("C", (), {"chat": _Boom()})()
    provider = AzureOpenAIProvider(_config())
    monkeypatch.setattr(provider, "_get_client", lambda: fake)
    result = provider.verify()
    assert result.ok is False
    assert "bad endpoint" in result.detail
    assert result.kind is ProviderKind.AZURE_OPENAI


def test_reasoning_verify_probes_one_structured_responses_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = _FakeResponses(tool_name="health_check")
    provider = AzureOpenAIProvider(_reasoning_config())
    monkeypatch.setattr(
        provider,
        "_get_responses_client",
        lambda: _FakeResponsesClient(responses),
    )
    monkeypatch.setattr(
        provider,
        "_get_client",
        lambda: (_ for _ in ()).throw(AssertionError("verify must probe Responses")),
    )

    result = provider.verify()

    assert result.ok is True
    assert len(responses.calls) == 1
    assert responses.last_kwargs["reasoning"] == {"effort": "high"}
    assert responses.last_kwargs["tool_choice"] == "required"
    tools = responses.last_kwargs["tools"]
    assert isinstance(tools, list)
    first_tool = tools[0]
    assert isinstance(first_tool, dict)
    assert cast("dict[str, object]", first_tool)["name"] == "health_check"


@pytest.mark.parametrize(
    "responses",
    [
        _FakeResponses(tool_name="health_check", headers={}),
        _FakeResponses(error=RuntimeError("Unsupported parameter: max_output_tokens")),
    ],
)
def test_reasoning_verify_rejects_missing_receipt_and_output_parameter_error(
    monkeypatch: pytest.MonkeyPatch,
    responses: _FakeResponses,
) -> None:
    provider = AzureOpenAIProvider(_reasoning_config())
    monkeypatch.setattr(
        provider,
        "_get_responses_client",
        lambda: _FakeResponsesClient(responses),
    )

    result = provider.verify()

    assert result.ok is False
    assert len(responses.calls) == 1


def _endpoint_config(endpoint: str) -> ProviderConfig:
    return ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        endpoint=endpoint,
        deployment="gpt55-deploy",
        api_version="2024-10-21",
    )


def test_config_endpoint_never_receives_the_real_azure_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config-controlled endpoint (untrusted bundle) must not get AZURE_OPENAI_API_KEY."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-real-secret")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://trusted.openai.azure.com")
    monkeypatch.delenv("WMH_ENDPOINT_API_KEY", raising=False)
    client = AzureOpenAIProvider(_endpoint_config("https://attacker.host"))._get_client()
    assert client.api_key != "az-real-secret"


def test_config_endpoint_uses_dedicated_key_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auth for a config endpoint comes from WMH_ENDPOINT_API_KEY, mirroring OpenAIProvider."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-real-secret")
    monkeypatch.setenv("WMH_ENDPOINT_API_KEY", "endpoint-token")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    client = AzureOpenAIProvider(_endpoint_config("https://attacker.host"))._get_client()
    assert client.api_key == "endpoint-token"


def test_trusted_env_endpoint_uses_the_real_azure_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator-supplied AZURE_OPENAI_ENDPOINT keeps using the real AZURE_OPENAI_API_KEY."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-real-secret")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://trusted.openai.azure.com")
    monkeypatch.setenv("WMH_ENDPOINT_API_KEY", "endpoint-token")
    # config.endpoint matching the trusted env endpoint is still trusted.
    client = AzureOpenAIProvider(_endpoint_config("https://trusted.openai.azure.com"))._get_client()
    assert client.api_key == "az-real-secret"
    assert client.max_retries == 0


def test_config_endpoint_also_disables_hidden_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WMH_ENDPOINT_API_KEY", "endpoint-token")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)

    client = AzureOpenAIProvider(_endpoint_config("https://proxy.example.com"))._get_client()

    assert client.max_retries == 0


def test_trusted_endpoint_matches_despite_trailing_slash_and_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trailing slash or casing difference must not strip the real key from a trusted host."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-real-secret")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://Trusted.openai.azure.com")
    monkeypatch.setenv("WMH_ENDPOINT_API_KEY", "endpoint-token")
    client = AzureOpenAIProvider(
        _endpoint_config("https://trusted.openai.azure.com/")
    )._get_client()
    assert client.api_key == "az-real-secret"


def test_case_sensitive_path_difference_is_untrusted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A path that differs only by case is a different (untrusted) resource: no real key."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-real-secret")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://proxy.example.com/Azure")
    monkeypatch.delenv("WMH_ENDPOINT_API_KEY", raising=False)
    client = AzureOpenAIProvider(_endpoint_config("https://proxy.example.com/azure"))._get_client()
    assert client.api_key != "az-real-secret"


def test_query_string_difference_is_untrusted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config endpoint that differs only by query is a different resource: no real key."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-real-secret")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://proxy.example.com/azure?tenant=trusted")
    monkeypatch.delenv("WMH_ENDPOINT_API_KEY", raising=False)
    client = AzureOpenAIProvider(
        _endpoint_config("https://proxy.example.com/azure?tenant=attacker")
    )._get_client()
    assert client.api_key != "az-real-secret"


def test_no_config_endpoint_falls_back_to_env_and_real_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no config.endpoint, the SDK uses AZURE_OPENAI_ENDPOINT + the real key."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-real-secret")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://trusted.openai.azure.com")
    provider = AzureOpenAIProvider(
        ProviderConfig(
            kind=ProviderKind.AZURE_OPENAI,
            model="gpt-5.5",
            deployment="gpt55-deploy",
            api_version="2024-10-21",
        )
    )
    assert provider._get_client().api_key == "az-real-secret"


def test_missing_endpoint_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """No config.endpoint and no AZURE_OPENAI_ENDPOINT is a clear config error."""
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    provider = AzureOpenAIProvider(
        ProviderConfig(
            kind=ProviderKind.AZURE_OPENAI,
            model="gpt-5.5",
            deployment="gpt55-deploy",
            api_version="2024-10-21",
        )
    )
    with pytest.raises(ValueError, match="endpoint"):
        provider._get_client()


@pytest.mark.skipif(
    not {"AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"}.issubset(__import__("os").environ),
    reason="no Azure OpenAI creds; skipping live smoke test",
)
def test_live_verify() -> None:  # pragma: no cover - network
    import os

    provider = AzureOpenAIProvider(
        ProviderConfig(
            kind=ProviderKind.AZURE_OPENAI,
            model="gpt-5.5",
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.5"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
    )
    assert provider.verify().ok is True
