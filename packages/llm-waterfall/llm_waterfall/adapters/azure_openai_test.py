"""Tests for the Azure OpenAI adapter (SDK stubbed — no network)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from llm_waterfall.adapters.azure_openai import AzureOpenAIAdapter
from llm_waterfall.types import Backend, Message

if TYPE_CHECKING:
    from collections.abc import Iterator


class _FakeAzureOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.chat_calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat_create))
        self.embeddings = SimpleNamespace(create=self._embed_create)

    def _chat_create(self, **kwargs: Any) -> SimpleNamespace:
        self.chat_calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="azure says hi"))],
            usage=SimpleNamespace(prompt_tokens=9, completion_tokens=4),
        )

    def _embed_create(self, **kwargs: Any) -> SimpleNamespace:
        self.embed_calls.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.5])], usage=SimpleNamespace(prompt_tokens=2)
        )


@pytest.fixture
def fake_azure(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[_FakeAzureOpenAI]]:
    clients: list[_FakeAzureOpenAI] = []
    module = types.ModuleType("openai")

    def factory(**kwargs: Any) -> _FakeAzureOpenAI:
        c = _FakeAzureOpenAI(**kwargs)
        clients.append(c)
        return c

    setattr(module, "AzureOpenAI", factory)  # noqa: B010 - fake module attr
    monkeypatch.setitem(sys.modules, "openai", module)
    yield clients


def _backend() -> Backend:
    return Backend(
        "azure_openai",
        "gpt-5.4",
        endpoint="https://x.openai.azure.com",
        deployment="gpt-54-deploy",
        api_version="2024-12-01-preview",
    )


def test_client_config_and_deployment_as_model(fake_azure: list[_FakeAzureOpenAI]) -> None:
    adapter = AzureOpenAIAdapter(_backend())
    text, usage = adapter.complete(
        "be terse", [Message(role="user", content="hi")], temperature=None, max_tokens=64
    )
    (client,) = fake_azure
    assert client.kwargs["azure_endpoint"] == "https://x.openai.azure.com"
    assert client.kwargs["api_version"] == "2024-12-01-preview"
    assert client.kwargs["max_retries"] == 0
    assert client.kwargs["timeout"].connect == 15.0
    call = client.chat_calls[0]
    # Azure routes by DEPLOYMENT name, not the base model id.
    assert call["model"] == "gpt-54-deploy"
    assert call["max_completion_tokens"] == 64
    assert call["messages"][0] == {"role": "system", "content": "be terse"}
    assert text == "azure says hi"
    assert usage.input_tokens == 9 and usage.output_tokens == 4


def test_deployment_defaults_to_model(fake_azure: list[_FakeAzureOpenAI]) -> None:
    backend = Backend(
        "azure_openai", "gpt-5.4", endpoint="https://x.openai.azure.com", api_version="v"
    )
    adapter = AzureOpenAIAdapter(backend)
    adapter.complete("", [Message(role="user", content="x")], temperature=None, max_tokens=8)
    assert fake_azure[0].chat_calls[0]["model"] == "gpt-5.4"


def test_embed_uses_embedding_deployment(fake_azure: list[_FakeAzureOpenAI]) -> None:
    backend = Backend(
        "azure_openai",
        "gpt-5.4",
        endpoint="https://x.openai.azure.com",
        api_version="v",
        embed_model="embed-deploy",
    )
    adapter = AzureOpenAIAdapter(backend)
    vectors, usage = adapter.embed(["a"])
    assert fake_azure[0].embed_calls[0]["model"] == "embed-deploy"
    assert adapter.embed_model_id() == "embed-deploy"
    assert vectors == [[0.5]]


def test_construction_requires_endpoint_and_api_version() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        AzureOpenAIAdapter(Backend("azure_openai", "gpt-5.4", api_version="v"))
    with pytest.raises(ValueError, match="api_version"):
        AzureOpenAIAdapter(Backend("azure_openai", "gpt-5.4", endpoint="https://x"))


def test_embed_without_deployment_is_unsupported(fake_azure: list[_FakeAzureOpenAI]) -> None:
    from llm_waterfall.types import EmbeddingsUnsupported

    adapter = AzureOpenAIAdapter(_backend())
    with pytest.raises(EmbeddingsUnsupported):
        adapter.embed(["x"])
