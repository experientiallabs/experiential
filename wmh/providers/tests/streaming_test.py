"""Native streaming across every backend: deltas in order, one final chunk with usage.

The SDK clients are faked via each provider's `_get_client` (same idiom as the sibling tests);
no network. The contract under test: `stream()` yields text deltas as they arrive and exactly
one terminal chunk (`done=True`) carrying the call's `TokenUsage`, so metering and request logs
can account for streamed traffic the same way they do `complete()`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from wmh.providers.anthropic import AnthropicProvider
from wmh.providers.azure_openai import AzureOpenAIProvider
from wmh.providers.base import (
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    StreamChunk,
    StreamingProvider,
    TokenUsage,
    VerifyResult,
)
from wmh.providers.bedrock import BedrockProvider
from wmh.providers.openai import OpenAIProvider
from wmh.providers.openai_responses import OpenAIResponsesProvider
from wmh.tracking.metered import MeteredProvider
from wmh.tracking.tracker import Phase, RunTracker

if TYPE_CHECKING:
    from collections.abc import Iterator

_MESSAGES = [Message(role="user", content="hi")]


def _collect(chunks: Iterator[StreamChunk]) -> tuple[str, StreamChunk]:
    parts: list[str] = []
    final: StreamChunk | None = None
    for chunk in chunks:
        if chunk.done:
            assert final is None, "stream yielded more than one terminal chunk"
            final = chunk
        else:
            parts.append(chunk.delta)
    assert final is not None, "stream never yielded a terminal chunk"
    return "".join(parts), final


# --- Anthropic -----------------------------------------------------------------------------


class _FakeAnthropicMessages:
    def __init__(self, events: list[object]) -> None:
        self._events = events
        self.last_kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> Iterator[object]:
        self.last_kwargs = kwargs
        return iter(self._events)


def test_anthropic_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[object] = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(input_tokens=9)),
        ),
        SimpleNamespace(
            type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="hel")
        ),
        SimpleNamespace(
            type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="lo")
        ),
        SimpleNamespace(type="message_delta", usage=SimpleNamespace(output_tokens=5)),
        SimpleNamespace(type="message_stop"),
    ]
    fake_messages = _FakeAnthropicMessages(events)
    provider = AnthropicProvider(
        ProviderConfig(kind=ProviderKind.ANTHROPIC, model="claude-fable-5")
    )
    monkeypatch.setattr(provider, "_get_client", lambda: SimpleNamespace(messages=fake_messages))

    text, final = _collect(provider.stream("be terse", _MESSAGES, max_tokens=64))

    assert text == "hello"
    assert final.usage == TokenUsage(input_tokens=9, output_tokens=5)
    assert fake_messages.last_kwargs["stream"] is True
    assert fake_messages.last_kwargs["max_tokens"] == 64
    assert "temperature" not in fake_messages.last_kwargs


# --- OpenAI-shaped (OpenAI + Azure chat completions) ----------------------------------------


def _openai_chunks() -> list[object]:
    return [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="hel"))], usage=None
        ),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="lo"))], usage=None),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))], usage=None),
        SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=9, completion_tokens=5)),
    ]


class _FakeChatCompletions:
    def __init__(self, chunks: list[object]) -> None:
        self._chunks = chunks
        self.last_kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> Iterator[object]:
        self.last_kwargs = kwargs
        return iter(self._chunks)


def test_openai_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeChatCompletions(_openai_chunks())
    provider = OpenAIProvider(ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-5.5"))
    monkeypatch.setattr(
        provider,
        "_get_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=fake)),
    )

    text, final = _collect(provider.stream("sys", _MESSAGES))

    assert text == "hello"
    assert final.usage == TokenUsage(input_tokens=9, output_tokens=5)
    assert fake.last_kwargs["stream"] is True
    assert fake.last_kwargs["stream_options"] == {"include_usage": True}


def test_azure_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeChatCompletions(_openai_chunks())
    provider = AzureOpenAIProvider(
        ProviderConfig(
            kind=ProviderKind.AZURE_OPENAI,
            model="gpt-5.5",
            deployment="gpt-5.5",
            endpoint="https://example.openai.azure.com",
            api_version="2024-10-21",
        ),
        api_key="sk-test",
    )
    monkeypatch.setattr(
        provider,
        "_get_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=fake)),
    )

    text, final = _collect(provider.stream("sys", _MESSAGES))

    assert text == "hello"
    assert final.usage == TokenUsage(input_tokens=9, output_tokens=5)
    # Azure routes stream() to the deployment name, not the base model id.
    assert fake.last_kwargs["model"] == "gpt-5.5"


def test_azure_stream_rejects_reasoning_configs() -> None:
    provider = AzureOpenAIProvider(
        ProviderConfig(
            kind=ProviderKind.AZURE_OPENAI,
            model="gpt-5.5",
            deployment="gpt-5.5",
            endpoint="https://example.openai.azure.com",
            api_version="2024-10-21",
            reasoning_effort="high",
        ),
        api_key="sk-test",
    )
    with pytest.raises(NotImplementedError, match="reasoning_effort"):
        next(iter(provider.stream("sys", _MESSAGES)))


# --- OpenAI Responses ------------------------------------------------------------------------


class _FakeResponses:
    def __init__(self, events: list[object]) -> None:
        self._events = events
        self.last_kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> Iterator[object]:
        self.last_kwargs = kwargs
        return iter(self._events)


def test_openai_responses_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[object] = [
        SimpleNamespace(type="response.output_text.delta", delta="hel"),
        SimpleNamespace(type="response.output_text.delta", delta="lo"),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(usage=SimpleNamespace(input_tokens=9, output_tokens=5)),
        ),
    ]
    fake = _FakeResponses(events)
    provider = OpenAIResponsesProvider(
        ProviderConfig(kind=ProviderKind.OPENAI_RESPONSES, model="gpt-5.5")
    )
    monkeypatch.setattr(provider, "_get_client", lambda: SimpleNamespace(responses=fake))

    text, final = _collect(provider.stream("sys", _MESSAGES))

    assert text == "hello"
    assert final.usage == TokenUsage(input_tokens=9, output_tokens=5)
    assert fake.last_kwargs["stream"] is True
    assert fake.last_kwargs["store"] is False


# --- Bedrock ---------------------------------------------------------------------------------


class _FakeBedrockClient:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self._events = events
        self.last_kwargs: dict[str, object] = {}

    def converse_stream(self, **kwargs: object) -> dict[str, object]:
        self.last_kwargs = kwargs
        return {"stream": iter(self._events)}


def test_bedrock_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[dict[str, object]] = [
        {"contentBlockDelta": {"delta": {"text": "hel"}}},
        {"contentBlockDelta": {"delta": {"text": "lo"}}},
        {"metadata": {"usage": {"inputTokens": 9, "outputTokens": 5}}},
    ]
    fake = _FakeBedrockClient(events)
    provider = BedrockProvider(
        ProviderConfig(kind=ProviderKind.BEDROCK, model="us.anthropic.claude-opus-4-8")
    )
    monkeypatch.setattr(provider, "_get_client", lambda: fake)

    text, final = _collect(provider.stream("sys", _MESSAGES, max_tokens=64))

    assert text == "hello"
    assert final.usage == TokenUsage(input_tokens=9, output_tokens=5)
    assert fake.last_kwargs["modelId"] == "us.anthropic.claude-opus-4-8"
    # Claude via Converse: max tokens forwarded, temperature withheld (catalog says no sampling).
    assert fake.last_kwargs["inferenceConfig"] == {"maxTokens": 64}


# --- Protocol + metering ----------------------------------------------------------------------


def test_all_backends_are_streaming_providers() -> None:
    providers: list[object] = [
        AnthropicProvider(ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")),
        OpenAIProvider(ProviderConfig(kind=ProviderKind.OPENAI, model="m")),
        AzureOpenAIProvider(ProviderConfig(kind=ProviderKind.AZURE_OPENAI, model="m")),
        OpenAIResponsesProvider(ProviderConfig(kind=ProviderKind.OPENAI_RESPONSES, model="m")),
        BedrockProvider(ProviderConfig(kind=ProviderKind.BEDROCK, model="m")),
    ]
    for provider in providers:
        assert isinstance(provider, StreamingProvider), type(provider).__name__


class _ProviderStub:
    """Minimal full-Provider surface so MeteredProvider accepts the fake."""

    def __init__(self, model: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model=model)

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def verify(self) -> VerifyResult:
        raise NotImplementedError


def test_metered_stream_records_usage_once() -> None:
    class _FakeStreamingProvider(_ProviderStub):
        def stream(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 8192,
        ) -> Iterator[StreamChunk]:
            yield StreamChunk(delta="hi")
            yield StreamChunk(done=True, usage=TokenUsage(input_tokens=9, output_tokens=5))

    tracker = RunTracker(run_id="r", kind="test")
    metered = MeteredProvider(
        _FakeStreamingProvider("claude-fable-5"), tracker, base_phase=Phase.SERVE
    )

    text, final = _collect(metered.stream("sys", _MESSAGES))

    assert text == "hi"
    assert final.usage == TokenUsage(input_tokens=9, output_tokens=5)
    assert len(tracker.events) == 1
    event = tracker.events[0]
    assert event.phase is Phase.SERVE
    assert event.model == "claude-fable-5"
    assert event.usage.input_tokens == 9


def test_metered_stream_rejects_non_streaming_provider() -> None:
    tracker = RunTracker(run_id="r", kind="test")
    metered = MeteredProvider(_ProviderStub("m"), tracker)
    with pytest.raises(TypeError, match="stream"):
        next(iter(metered.stream("sys", _MESSAGES)))
