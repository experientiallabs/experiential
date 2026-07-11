"""Unit tests for AzureOpenAIProvider. No network: the SDK client is faked via _get_client."""

from __future__ import annotations

import pytest

from wmh.providers.azure_openai import AzureOpenAIProvider
from wmh.providers.base import Message, ProviderConfig, ProviderKind


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


class _FakeChatCompletions:
    def __init__(self, response: _FakeChatResponse) -> None:
        self.response = response
        self.last_kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> _FakeChatResponse:
        self.last_kwargs = kwargs
        return self.response


class _FakeChat:
    def __init__(self, completions: _FakeChatCompletions) -> None:
        self.completions = completions


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
