"""Tests for the Anthropic adapter (SDK stubbed — no network)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from llm_waterfall.adapters.anthropic import AnthropicAdapter
from llm_waterfall.types import Backend, EmbeddingsUnsupported, Message

if TYPE_CHECKING:
    from collections.abc import Iterator


class _FakeAnthropic:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="claude says hi")],
            usage=SimpleNamespace(input_tokens=12, output_tokens=6),
        )


@pytest.fixture
def fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[_FakeAnthropic]]:
    clients: list[_FakeAnthropic] = []
    module = types.ModuleType("anthropic")

    def factory(**kwargs: Any) -> _FakeAnthropic:
        c = _FakeAnthropic(**kwargs)
        clients.append(c)
        return c

    setattr(module, "Anthropic", factory)  # noqa: B010 - fake module attr
    monkeypatch.setitem(sys.modules, "anthropic", module)
    yield clients


def test_request_shape_and_client_config(fake_anthropic: list[_FakeAnthropic]) -> None:
    adapter = AnthropicAdapter(Backend("anthropic", "claude-opus-4-8"))
    text, usage = adapter.complete(
        "be terse", [Message(role="user", content="hi")], temperature=None, max_tokens=64
    )
    (client,) = fake_anthropic
    assert client.kwargs["max_retries"] == 0
    # Granular timeout: connect bounded separately so a dead endpoint fails over in ~15s.
    timeout = client.kwargs["timeout"]
    assert timeout.connect == 15.0 and timeout.read == 600.0
    call = client.calls[0]
    assert call["model"] == "claude-opus-4-8"
    assert call["system"] == "be terse"  # top-level, not folded into messages
    assert call["max_tokens"] == 64
    assert "temperature" not in call
    assert text == "claude says hi"
    assert usage.input_tokens == 12 and usage.output_tokens == 6


def test_embed_raises_unsupported(fake_anthropic: list[_FakeAnthropic]) -> None:
    adapter = AnthropicAdapter(Backend("anthropic", "claude-opus-4-8"))
    with pytest.raises(EmbeddingsUnsupported):
        adapter.embed(["x"])
