"""Tests for the OpenAI adapter (SDK stubbed — no network)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from llm_waterfall.adapters.openai import OpenAIAdapter
from llm_waterfall.types import Backend, Message

if TYPE_CHECKING:
    from collections.abc import Iterator


class _FakeOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.chat_calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []
        self.choices: list[Any] = [SimpleNamespace(message=SimpleNamespace(content="hi there"))]
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat_create))
        self.embeddings = SimpleNamespace(create=self._embed_create)

    def _chat_create(self, **kwargs: Any) -> SimpleNamespace:
        self.chat_calls.append(kwargs)
        return SimpleNamespace(
            choices=self.choices,
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        )

    def _embed_create(self, **kwargs: Any) -> SimpleNamespace:
        self.embed_calls.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.5, 0.5])],
            usage=SimpleNamespace(prompt_tokens=4),
        )


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[_FakeOpenAI]]:
    clients: list[_FakeOpenAI] = []
    module = types.ModuleType("openai")

    def factory(**kwargs: Any) -> _FakeOpenAI:
        c = _FakeOpenAI(**kwargs)
        clients.append(c)
        return c

    setattr(module, "OpenAI", factory)  # noqa: B010 - fake module attr
    monkeypatch.setitem(sys.modules, "openai", module)
    yield clients


def test_request_shape_and_client_config(fake_openai: list[_FakeOpenAI]) -> None:
    adapter = OpenAIAdapter(Backend("openai", "gpt-5.5"))
    text, usage = adapter.complete(
        "be terse", [Message(role="user", content="hi")], temperature=None, max_tokens=64
    )
    (client,) = fake_openai
    # SDK-internal retries disabled; requests bounded by the backend's timeouts.
    assert client.kwargs["max_retries"] == 0
    # Granular timeout: connect bounded separately so a dead endpoint fails over in ~15s.
    timeout = client.kwargs["timeout"]
    assert timeout.connect == 15.0 and timeout.read == 600.0
    call = client.chat_calls[0]
    assert call["model"] == "gpt-5.5"
    assert call["max_completion_tokens"] == 64  # not the deprecated max_tokens
    assert "temperature" not in call
    assert call["messages"][0] == {"role": "system", "content": "be terse"}  # folded system
    assert text == "hi there"
    assert usage.input_tokens == 7 and usage.output_tokens == 3


def test_zero_choices_raises_value_error(fake_openai: list[_FakeOpenAI]) -> None:
    adapter = OpenAIAdapter(Backend("openai", "gpt-5.5"))
    fake = adapter  # trigger client creation via one call setup
    _ = fake
    # Prime a client whose next response has no choices.
    adapter.complete("", [Message(role="user", content="x")], temperature=None, max_tokens=8)
    fake_openai[0].choices = []
    with pytest.raises(ValueError, match="no choices"):
        adapter.complete("", [Message(role="user", content="x")], temperature=None, max_tokens=8)


def test_embed_default_model_and_dimensions(fake_openai: list[_FakeOpenAI]) -> None:
    adapter = OpenAIAdapter(Backend("openai", "gpt-5.5", embed_dim=256))
    vectors, usage = adapter.embed(["a"])
    call = fake_openai[0].embed_calls[0]
    assert call["model"] == "text-embedding-3-small"
    assert call["dimensions"] == 256
    assert vectors == [[0.5, 0.5]]
    assert usage.input_tokens == 4
