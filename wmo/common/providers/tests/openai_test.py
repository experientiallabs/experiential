"""Unit tests for OpenAIProvider. No network: the SDK client is faked via _get_client."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wmo.common.providers.base import (
    DEFAULT_MAX_TOKENS,
    ChatRequest,
    Message,
    ProviderConfig,
    ProviderKind,
)
from wmo.common.providers.openai import OpenAIProvider
from wmo.common.vendor.waterfall import ChatMaxTokensField

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


class _FakeChatCompletions:
    def __init__(self, response: _FakeChatResponse) -> None:
        self.response = response
        self.last_kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> _FakeChatResponse:
        self.last_kwargs = kwargs
        return self.response


class _FakeStreamDelta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeStreamChoice:
    def __init__(self, content: str | None) -> None:
        self.delta = _FakeStreamDelta(content)


class _FakeStreamChunk:
    """One wire chunk: a content delta, or the terminal usage-bearing chunk with no choices."""

    def __init__(self, content: str | None, usage: _FakeUsage | None = None) -> None:
        self.choices = [_FakeStreamChoice(content)] if content is not None else []
        self.usage = usage


class _FakeStreamingCompletions:
    """A chat.completions whose `create` returns the chunk sequence a stream would yield."""

    def __init__(self, chunks: list[_FakeStreamChunk]) -> None:
        self.chunks = chunks
        self.last_kwargs: dict[str, object] = {}
        self.closed = False

    def create(self, **kwargs: object) -> _FakeStreamingCompletions:
        self.last_kwargs = kwargs
        return self

    def __iter__(self) -> Iterator[_FakeStreamChunk]:
        return iter(self.chunks)

    def close(self) -> None:
        self.closed = True


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
    def __init__(self, completions: _FakeChatCompletions | _FakeStreamingCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(
        self,
        chat: _FakeChatCompletions | _FakeStreamingCompletions,
        embeddings: _FakeEmbeddings,
    ) -> None:
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
    monkeypatch.delenv("WMO_ENDPOINT_API_KEY", raising=False)
    client = _endpoint_provider()._get_client()
    assert client.api_key != "sk-real-openai-secret"


def test_custom_endpoint_uses_dedicated_key_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """An authenticated OpenAI-compatible server takes its key from WMO_ENDPOINT_API_KEY."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-secret")
    monkeypatch.setenv("WMO_ENDPOINT_API_KEY", "endpoint-token")
    client = _endpoint_provider()._get_client()
    assert client.api_key == "endpoint-token"


def test_custom_endpoint_needs_no_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A self-hosted OpenAI-compatible server (vLLM) has no real key; loading must not raise."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("WMO_ENDPOINT_API_KEY", raising=False)
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


def _student_provider(field: ChatMaxTokensField) -> OpenAIProvider:
    """A student-shaped config: a weights path the catalog cannot resolve, plus explicit field."""
    return OpenAIProvider(
        ProviderConfig(
            kind=ProviderKind.OPENAI,
            model="tinker://weights/abc123",
            model_type="Qwen/Qwen3-30B-A3B",
            chat_max_tokens_field=field,
            endpoint="https://tinker.example/oai/api/v1",
        )
    )


def test_complete_honors_the_configured_output_budget_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that wants classic `max_tokens` must get it from `complete`, not just chat.

    An OpenAI-compatible endpoint outside the built-in catalog (Tinker's serving endpoint, a
    vLLM build) 400s on the name it does not accept, so a routed student would fail every
    non-structured call while `complete_chat` worked.
    """
    provider = _student_provider("max_tokens")
    chat = _FakeChatCompletions(_FakeChatResponse("ok", _FakeUsage(3, 2)))
    monkeypatch.setattr(
        provider,
        "_get_client",
        lambda: _FakeClient(chat, _FakeEmbeddings(_FakeEmbeddingResponse([]))),
    )

    provider.complete("sys", [Message(role="user", content="hi")], max_tokens=77)

    assert chat.last_kwargs["max_tokens"] == 77
    assert "max_completion_tokens" not in chat.last_kwargs


def test_stream_honors_the_configured_output_budget_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same contract on the streaming path, which the endpoint uses for `stream=True`."""
    provider = _student_provider("max_tokens")
    chat = _FakeStreamingCompletions(
        [_FakeStreamChunk("hi"), _FakeStreamChunk(None, _FakeUsage(3, 2))]
    )
    monkeypatch.setattr(
        provider,
        "_get_client",
        lambda: _FakeClient(chat, _FakeEmbeddings(_FakeEmbeddingResponse([]))),
    )

    chunks = list(provider.stream("sys", [Message(role="user", content="hi")], max_tokens=55))

    assert [c.delta for c in chunks if c.delta] == ["hi"]
    assert chunks[-1].done is True
    assert chat.last_kwargs["max_tokens"] == 55
    assert "max_completion_tokens" not in chat.last_kwargs


def test_built_in_models_still_send_max_completion_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """The catalog still wins for a known model, so this fix cannot regress GPT-5.x."""
    provider = OpenAIProvider(_config())
    chat = _FakeChatCompletions(_FakeChatResponse("ok", _FakeUsage(1, 1)))
    monkeypatch.setattr(
        provider,
        "_get_client",
        lambda: _FakeClient(chat, _FakeEmbeddings(_FakeEmbeddingResponse([]))),
    )

    provider.complete("sys", [Message(role="user", content="hi")], max_tokens=64)

    assert chat.last_kwargs["max_completion_tokens"] == 64
    assert "max_tokens" not in chat.last_kwargs


class _FakeResponsesResource:
    """Records the Responses-API payload an effort-dialed call sends."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> object:
        self.last_kwargs = kwargs
        raise AssertionError("payload captured")


class _FakeResponsesClient:
    def __init__(self, responses: _FakeResponsesResource) -> None:
        self.responses = responses


def _recording_responses(
    provider: OpenAIProvider, monkeypatch: pytest.MonkeyPatch
) -> _FakeResponsesResource:
    """Point the provider's Responses delegate at a recording client, as the chat tests do."""
    resource = _FakeResponsesResource()
    delegate = provider._responses_delegate()  # noqa: SLF001 - asserting the dispatch target
    monkeypatch.setattr(delegate, "_get_client", lambda: _FakeResponsesClient(resource))
    return resource


def test_reasoning_effort_on_real_openai_goes_out_as_responses_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dial must reach the wire: chat/completions silently dropped it, billing a
    default-effort run as an effort-dialed arm, and rejects the family's top `max` outright."""
    provider = OpenAIProvider(
        ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-5.6-sol", reasoning_effort="max")
    )
    resource = _recording_responses(provider, monkeypatch)

    assert provider._dispatch_to_responses() is True  # noqa: SLF001
    with pytest.raises(AssertionError, match="payload captured"):
        provider.complete("sys", [Message(role="user", content="hi")], max_tokens=64)

    assert resource.last_kwargs["reasoning"] == {"effort": "max"}


def test_reasoning_effort_on_a_custom_endpoint_stays_on_the_chat_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAIResponsesProvider builds its client with no base_url, so it cannot reach a
    self-hosted server; the dial rides chat/completions there under vLLM's spelling."""
    provider = OpenAIProvider(
        ProviderConfig(
            kind=ProviderKind.OPENAI,
            model="my-model",
            endpoint="http://localhost:8000/v1",
            reasoning_effort="xhigh",
        )
    )
    chat = _FakeChatCompletions(_FakeChatResponse("ok", _FakeUsage(1, 1)))
    monkeypatch.setattr(
        provider,
        "_get_client",
        lambda: _FakeClient(chat, _FakeEmbeddings(_FakeEmbeddingResponse([]))),
    )

    assert provider._dispatch_to_responses() is False  # noqa: SLF001
    provider.complete("sys", [Message(role="user", content="hi")], max_tokens=64)

    assert chat.last_kwargs["reasoning_effort"] == "xhigh"


def test_no_reasoning_effort_leaves_the_chat_payload_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configs with no dial must be byte-identical to before the dispatch existed."""
    provider = OpenAIProvider(_config())
    chat = _FakeChatCompletions(_FakeChatResponse("ok", _FakeUsage(1, 1)))
    monkeypatch.setattr(
        provider,
        "_get_client",
        lambda: _FakeClient(chat, _FakeEmbeddings(_FakeEmbeddingResponse([]))),
    )

    assert provider._dispatch_to_responses() is False  # noqa: SLF001
    provider.complete("sys", [Message(role="user", content="hi")], max_tokens=64)

    assert "reasoning_effort" not in chat.last_kwargs
    assert "reasoning" not in chat.last_kwargs


def test_complete_chat_routes_the_dial_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """Structured requests take the same route as text ones, or the two disagree per call."""
    provider = OpenAIProvider(
        ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-5.6-sol", reasoning_effort="high")
    )
    resource = _recording_responses(provider, monkeypatch)

    with pytest.raises(AssertionError, match="payload captured"):
        provider.complete_chat(
            ChatRequest.model_validate(
                {"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 32}
            )
        )

    assert resource.last_kwargs["reasoning"] == {"effort": "high"}


def test_prepare_prepares_the_route_the_request_will_take(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An effort-dialed config must never prepare (and so never key-check) the unused client."""
    provider = OpenAIProvider(
        ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-5.6-sol", reasoning_effort="max")
    )
    monkeypatch.setattr(
        provider, "_get_client", lambda: pytest.fail("prepared the chat client instead")
    )
    prepared: list[str] = []
    monkeypatch.setattr(
        provider._responses_delegate(),  # noqa: SLF001
        "prepare",
        lambda: prepared.append("responses"),
    )

    provider.prepare()

    assert prepared == ["responses"]


def test_effort_dialed_stream_raises_when_it_yields_nothing_at_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream that emits no text but burns the whole budget is the same starvation complete()
    guards; unguarded, the consumer records an empty assistant turn as a success."""
    provider = OpenAIProvider(
        ProviderConfig(
            kind=ProviderKind.OPENAI,
            model="my-model",
            endpoint="http://localhost:8000/v1",
            reasoning_effort="max",
        )
    )
    chat = _FakeStreamingCompletions([_FakeStreamChunk(None, _FakeUsage(10, 64))])
    monkeypatch.setattr(
        provider,
        "_get_client",
        lambda: _FakeClient(chat, _FakeEmbeddings(_FakeEmbeddingResponse([]))),
    )

    seen = []
    with pytest.raises(ValueError, match="reasoning consumed"):
        for chunk in provider.stream("sys", [Message(role="user", content="hi")], max_tokens=64):
            seen.append(chunk)

    # The terminal usage chunk must arrive BEFORE the raise: those tokens were billed either way,
    # so raising first would drop a real, paid-for call from the consumer's metering.
    assert [c.done for c in seen] == [True]
    assert seen[0].usage is not None
    assert seen[0].usage.output_tokens == 64


def test_effort_dialed_stream_that_emits_text_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deltas mean the call produced output; the guard must not touch a working stream."""
    provider = OpenAIProvider(
        ProviderConfig(
            kind=ProviderKind.OPENAI,
            model="my-model",
            endpoint="http://localhost:8000/v1",
            reasoning_effort="max",
        )
    )
    chat = _FakeStreamingCompletions(
        [_FakeStreamChunk("hi"), _FakeStreamChunk(None, _FakeUsage(10, 64))]
    )
    monkeypatch.setattr(
        provider,
        "_get_client",
        lambda: _FakeClient(chat, _FakeEmbeddings(_FakeEmbeddingResponse([]))),
    )

    chunks = list(provider.stream("sys", [Message(role="user", content="hi")], max_tokens=64))

    assert [c.delta for c in chunks if c.delta] == ["hi"]
    assert chunks[-1].done is True
