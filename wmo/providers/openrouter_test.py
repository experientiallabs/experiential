"""Unit tests for OpenRouterProvider. No network: the SDK client is faked via _get_client."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wmo.config.config import PROVIDER_ENV_VARS
from wmo.providers.base import (
    ChatRequest,
    Message,
    ProviderConfig,
    ProviderKind,
)
from wmo.providers.openrouter import (
    DEFAULT_REFERER,
    DEFAULT_TITLE,
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_BASE_URL,
    OPENROUTER_REFERER_ENV,
    OPENROUTER_TITLE_ENV,
    OpenRouterProvider,
    attribution_headers,
)
from wmo.providers.registry import get_provider

if TYPE_CHECKING:
    from collections.abc import Iterator


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


class _FakeStreamChunk:
    def __init__(self, delta: str) -> None:
        self.choices = [_FakeStreamChoice(delta)]
        self.usage = None


class _FakeStreamDelta:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeStreamChoice:
    def __init__(self, content: str) -> None:
        self.delta = _FakeStreamDelta(content)


class _FakeStream:
    def __init__(self, deltas: list[str], usage: _FakeUsage) -> None:
        self._deltas = deltas
        self._usage = usage
        self.closed = False

    def __iter__(self) -> Iterator[object]:
        for delta in self._deltas:
            yield _FakeStreamChunk(delta)
        yield _FakeUsageChunk(self._usage)

    def close(self) -> None:
        self.closed = True


class _FakeUsageChunk:
    def __init__(self, usage: _FakeUsage) -> None:
        self.choices: list[object] = []
        self.usage = usage


class _FakeChatCompletions:
    def __init__(self, response: _FakeChatResponse | _FakeStream) -> None:
        self.response = response
        self.last_kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> _FakeChatResponse | _FakeStream:
        self.last_kwargs = kwargs
        return self.response


class _FakeChat:
    def __init__(self, completions: _FakeChatCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: _FakeChatCompletions) -> None:
        self.chat = _FakeChat(completions)


def _config(endpoint: str | None = None) -> ProviderConfig:
    return ProviderConfig(
        kind=ProviderKind.OPENROUTER, model="anthropic/claude-sonnet-4", endpoint=endpoint
    )


def _faked(
    monkeypatch: pytest.MonkeyPatch, response: _FakeChatResponse | _FakeStream
) -> tuple[OpenRouterProvider, _FakeChatCompletions]:
    completions = _FakeChatCompletions(response)
    provider = OpenRouterProvider(_config())
    monkeypatch.setattr(provider, "_get_client", lambda: _FakeClient(completions))
    return provider, completions


def test_registry_constructs_the_openrouter_backend() -> None:
    provider = get_provider(_config())
    assert isinstance(provider, OpenRouterProvider)


def test_provider_env_vars_pin_the_variable_the_provider_actually_reads() -> None:
    # PROVIDER_ENV_VARS keeps literals (importing providers from config would invert the
    # dependency), so only this pin catches the two drifting apart and silently degrading the
    # credential prompt, the picker annotation, and the `providers verify` hint.
    assert PROVIDER_ENV_VARS[ProviderKind.OPENROUTER] == [OPENROUTER_API_KEY_ENV]


def test_client_uses_openrouter_base_url_key_and_attribution_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "sk-or-env")
    monkeypatch.delenv(OPENROUTER_REFERER_ENV, raising=False)
    monkeypatch.delenv(OPENROUTER_TITLE_ENV, raising=False)

    client = OpenRouterProvider(_config())._get_client()  # noqa: SLF001 - asserting the wiring

    assert str(client.base_url).rstrip("/") == OPENROUTER_BASE_URL
    assert client.api_key == "sk-or-env"
    assert client.default_headers["HTTP-Referer"] == DEFAULT_REFERER
    assert client.default_headers["X-Title"] == DEFAULT_TITLE


def test_attribution_headers_are_configurable_and_never_personal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The defaults name the open-source project; an operator can point the leaderboard entry at
    # their own app, but nothing about the machine or the user leaks in by default.
    assert "http" in DEFAULT_REFERER
    monkeypatch.setenv(OPENROUTER_REFERER_ENV, "https://acme.example")
    monkeypatch.setenv(OPENROUTER_TITLE_ENV, "Acme Router")
    assert attribution_headers() == {
        "HTTP-Referer": "https://acme.example",
        "X-Title": "Acme Router",
    }


def test_explicit_api_key_wins_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # The pool's `api_key_env` channel: one entry per OpenRouter account.
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "sk-or-env")
    provider = OpenRouterProvider(_config(), api_key="sk-or-entry")
    assert provider._get_client().api_key == "sk-or-entry"  # noqa: SLF001 - asserting the wiring


def test_endpoint_overrides_the_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "sk-or-env")
    provider = OpenRouterProvider(_config(endpoint="https://proxy.example/api/v1"))
    assert str(provider._get_client().base_url).rstrip("/") == (  # noqa: SLF001 - asserting wiring
        "https://proxy.example/api/v1"
    )


def test_missing_key_names_the_variable_and_where_to_get_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(OPENROUTER_API_KEY_ENV, raising=False)
    with pytest.raises(ValueError, match=OPENROUTER_API_KEY_ENV):
        OpenRouterProvider(_config())._get_client()  # noqa: SLF001 - asserting the wiring


def test_never_falls_back_to_the_generic_endpoint_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # WMO_ENDPOINT_API_KEY exists for arbitrary self-hosted OpenAI-compatible servers. OpenRouter
    # is a named vendor with its own credential, so an unset OPENROUTER_API_KEY must fail loudly
    # rather than quietly authenticating with somebody else's placeholder key.
    monkeypatch.delenv(OPENROUTER_API_KEY_ENV, raising=False)
    monkeypatch.setenv("WMO_ENDPOINT_API_KEY", "sk-not-mine")
    with pytest.raises(ValueError, match=OPENROUTER_API_KEY_ENV):
        OpenRouterProvider(_config())._get_client()  # noqa: SLF001 - asserting the wiring


def test_verify_reports_a_missing_key_instead_of_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OPENROUTER_API_KEY_ENV, raising=False)
    result = OpenRouterProvider(_config()).verify()
    assert result.ok is False
    assert result.kind is ProviderKind.OPENROUTER
    assert OPENROUTER_API_KEY_ENV in result.detail


def test_complete_sends_max_tokens_not_max_completion_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `max_tokens` is the field in OpenRouter's request schema; sending OpenAI's newer name
    # would leave the call uncapped on any upstream that ignores unknown fields.
    provider, completions = _faked(monkeypatch, _FakeChatResponse("hi", _FakeUsage(9, 4)))

    completion = provider.complete("be nice", [Message(role="user", content="yo")], max_tokens=128)

    assert completion.text == "hi"
    assert completion.usage.input_tokens == 9
    assert completion.usage.output_tokens == 4
    sent = completions.last_kwargs
    assert sent["model"] == "anthropic/claude-sonnet-4"
    assert sent["max_tokens"] == 128
    assert "max_completion_tokens" not in sent
    assert sent["temperature"] == 0.7
    assert sent["messages"] == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "yo"},
    ]


def test_stream_yields_deltas_then_a_usage_bearing_terminal_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = _FakeStream(["he", "llo"], _FakeUsage(3, 2))
    provider, completions = _faked(monkeypatch, upstream)

    chunks = list(provider.stream("", [Message(role="user", content="yo")], max_tokens=64))

    assert "".join(c.delta for c in chunks) == "hello"
    assert chunks[-1].done is True
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.output_tokens == 2
    assert completions.last_kwargs["max_tokens"] == 64
    assert completions.last_kwargs["stream_options"] == {"include_usage": True}
    # The SDK stream holds an httpx response; an abandoned one must not wait for the collector.
    assert upstream.closed is True


def test_complete_chat_preserves_tools_and_uses_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, completions = _faked(monkeypatch, _FakeChatResponse("done", _FakeUsage(5, 1)))
    request = ChatRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "read it"}],
            "max_completion_tokens": 256,
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "read_file", "parameters": {"type": "object"}},
                }
            ],
        }
    )

    response = provider.complete_chat(request)

    assert response.choices[0].message.content == "done"
    sent = completions.last_kwargs
    assert sent["max_tokens"] == 256
    assert "max_completion_tokens" not in sent
    assert sent["tools"] is not None


def test_embed_explains_that_openrouter_has_no_embeddings_api() -> None:
    with pytest.raises(ValueError, match="no embeddings API"):
        OpenRouterProvider(_config()).embed(["hi"])
