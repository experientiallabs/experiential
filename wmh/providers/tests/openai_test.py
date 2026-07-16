"""Unit tests for OpenAIProvider. No network: the SDK client is faked via _get_client."""

from __future__ import annotations

import pytest

from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    ChatRequest,
    Message,
    ProviderConfig,
    ProviderKind,
)
from wmh.providers.openai import OpenAIProvider


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
    def __init__(self, content: str, usage: _FakeUsage) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = usage

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.choices[0].message.content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
            },
        }


class _FakeChatCompletions:
    def __init__(self, response: _FakeChatResponse) -> None:
        self.response = response
        self.last_kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> _FakeChatResponse:
        self.last_kwargs = kwargs
        return self.response


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


class _FakeChat:
    def __init__(self, completions: _FakeChatCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, chat: _FakeChatCompletions, embeddings: _FakeEmbeddings) -> None:
        self.chat = _FakeChat(chat)
        self.embeddings = embeddings


def _config() -> ProviderConfig:
    return ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-5.5", embed_model="text-embed-3")


def test_complete_folds_system_and_uses_max_completion_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = _FakeChatCompletions(_FakeChatResponse("hi there", _FakeUsage(9, 4)))
    provider = OpenAIProvider(_config())
    fake = _FakeClient(chat, _FakeEmbeddings(_FakeEmbeddingResponse([])))
    monkeypatch.setattr(provider, "_get_client", lambda: fake)

    completion = provider.complete("be nice", [Message(role="user", content="yo")], max_tokens=128)

    assert completion.text == "hi there"
    assert completion.usage.input_tokens == 9
    assert completion.usage.output_tokens == 4
    sent = chat.last_kwargs
    assert sent["model"] == "gpt-5.5"
    assert sent["max_completion_tokens"] == 128
    assert "max_tokens" not in sent
    assert "temperature" not in sent
    assert sent["messages"] == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "yo"},
    ]


def test_complete_default_max_tokens_is_8k(monkeypatch: pytest.MonkeyPatch) -> None:
    chat = _FakeChatCompletions(_FakeChatResponse("hi there", _FakeUsage(9, 4)))
    provider = OpenAIProvider(_config())
    fake = _FakeClient(chat, _FakeEmbeddings(_FakeEmbeddingResponse([])))
    monkeypatch.setattr(provider, "_get_client", lambda: fake)

    provider.complete("be nice", [Message(role="user", content="yo")])

    assert chat.last_kwargs["max_completion_tokens"] == DEFAULT_MAX_TOKENS


def test_embed_uses_embed_model(monkeypatch: pytest.MonkeyPatch) -> None:
    embeddings = _FakeEmbeddings(_FakeEmbeddingResponse([[0.1, 0.2], [0.3, 0.4]]))
    provider = OpenAIProvider(_config())
    chat = _FakeChatCompletions(_FakeChatResponse("", _FakeUsage(0, 0)))
    fake = _FakeClient(chat, embeddings)
    monkeypatch.setattr(provider, "_get_client", lambda: fake)

    vectors = provider.embed(["a", "b"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert embeddings.last_kwargs["model"] == "text-embed-3"
    assert embeddings.last_kwargs["input"] == ["a", "b"]
    assert "dimensions" not in embeddings.last_kwargs  # omitted when embed_dim unset


def test_embed_threads_embed_dim_as_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    embeddings = _FakeEmbeddings(_FakeEmbeddingResponse([[0.1, 0.2, 0.3]]))
    config = ProviderConfig(
        kind=ProviderKind.OPENAI, model="gpt-5.5", embed_model="text-embed-3", embed_dim=3
    )
    provider = OpenAIProvider(config)
    chat = _FakeChatCompletions(_FakeChatResponse("", _FakeUsage(0, 0)))
    monkeypatch.setattr(provider, "_get_client", lambda: _FakeClient(chat, embeddings))

    provider.embed(["a"])

    assert embeddings.last_kwargs["dimensions"] == 3  # embed_dim -> dimensions param


def test_embed_requires_embed_model() -> None:
    provider = OpenAIProvider(ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-5.5"))
    with pytest.raises(ValueError, match="embed_model"):
        provider.embed(["x"])


def test_verify_reports_failure_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        class completions:  # noqa: N801 - mimic the SDK attribute path
            @staticmethod
            def create(**kwargs: object) -> object:
                raise RuntimeError("401")

    fake = type("C", (), {"chat": _Boom()})()
    provider = OpenAIProvider(_config())
    monkeypatch.setattr(provider, "_get_client", lambda: fake)
    result = provider.verify()
    assert result.ok is False
    assert "401" in result.detail


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in __import__("os").environ,
    reason="no OPENAI_API_KEY; skipping live smoke test",
)
def test_live_verify() -> None:  # pragma: no cover - network
    provider = OpenAIProvider(_config())
    assert provider.verify().ok is True


def _endpoint_provider() -> OpenAIProvider:
    return OpenAIProvider(
        ProviderConfig(
            kind=ProviderKind.OPENAI, model="qwen3.5-9b", endpoint="http://localhost:8001/v1"
        )
    )


def test_custom_endpoint_reaches_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """ProviderConfig.endpoint must become the client's base_url (vLLM / OpenAI-compatible)."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = _endpoint_provider()._get_client()
    assert str(client.base_url).rstrip("/") == "http://localhost:8001/v1"


def test_custom_endpoint_never_receives_the_real_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENAI_API_KEY must not be sent as a Bearer token to an arbitrary custom endpoint."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-secret")
    monkeypatch.delenv("WMH_ENDPOINT_API_KEY", raising=False)
    client = _endpoint_provider()._get_client()
    assert client.api_key != "sk-real-openai-secret"


def test_custom_endpoint_uses_dedicated_key_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """An authenticated OpenAI-compatible server takes its key from WMH_ENDPOINT_API_KEY."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-secret")
    monkeypatch.setenv("WMH_ENDPOINT_API_KEY", "endpoint-token")
    client = _endpoint_provider()._get_client()
    assert client.api_key == "endpoint-token"


def test_custom_endpoint_needs_no_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A self-hosted OpenAI-compatible server (vLLM) has no real key; loading must not raise."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("WMH_ENDPOINT_API_KEY", raising=False)
    client = _endpoint_provider()._get_client()
    assert client.api_key  # placeholder key, not an exception


def test_custom_endpoint_forwards_temperature_but_openai_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Self-hosted servers get the sampling param; real OpenAI (GPT-5.5) must not (rejects it)."""
    for endpoint, expects_temperature in [("http://localhost:8001/v1", True), (None, False)]:
        provider = OpenAIProvider(
            ProviderConfig(kind=ProviderKind.OPENAI, model="m", endpoint=endpoint)
        )
        chat = _FakeChatCompletions(_FakeChatResponse("ok", _FakeUsage(1, 1)))
        fake = _FakeClient(chat, _FakeEmbeddings(_FakeEmbeddingResponse([[0.0]])))
        monkeypatch.setattr(provider, "_get_client", lambda fake=fake: fake)
        provider.complete("sys", [Message(role="user", content="hi")], temperature=0.3)
        assert ("temperature" in chat.last_kwargs) is expects_temperature


@pytest.mark.parametrize(
    ("endpoint", "expects_temperature"),
    [(None, False), ("http://localhost:8001/v1", True)],
)
def test_structured_chat_applies_temperature_capability_before_wire(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str | None,
    expects_temperature: bool,
) -> None:
    provider = OpenAIProvider(
        ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-5.5", endpoint=endpoint)
    )
    chat = _FakeChatCompletions(_FakeChatResponse("ok", _FakeUsage(1, 1)))
    fake = _FakeClient(chat, _FakeEmbeddings(_FakeEmbeddingResponse([])))
    monkeypatch.setattr(provider, "_get_client", lambda: fake)

    provider.complete_chat(
        ChatRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "go"}],
                "temperature": 0.3,
                "max_completion_tokens": 64,
            }
        )
    )

    assert ("temperature" in chat.last_kwargs) is expects_temperature
