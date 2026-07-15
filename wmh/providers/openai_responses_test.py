"""Unit tests for OpenAIResponsesProvider. No network: the SDK client is faked."""

from __future__ import annotations

import pytest
from llm_waterfall import ChatRequest

from wmh.providers.base import DEFAULT_MAX_TOKENS, Message, ProviderConfig, ProviderKind
from wmh.providers.openai_responses import OpenAIResponsesProvider


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponsesResponse:
    def __init__(
        self,
        output_text: str,
        usage: _FakeUsage | None,
        output: list[object] | None = None,
    ) -> None:
        self.output_text = output_text
        self.usage = usage
        self.output = output or []

    def model_dump(self, *, mode: str) -> dict[str, object]:
        """Return the native Responses shape used by structured completion tests."""
        assert mode == "json"
        return {
            "model": "gpt-5.5-2026-05-01",
            "status": "completed",
            "service_tier": "priority",
            "output": self.output,
            "usage": (
                {
                    "input_tokens": self.usage.input_tokens,
                    "output_tokens": self.usage.output_tokens,
                }
                if self.usage is not None
                else None
            ),
        }


class _FakeResponses:
    def __init__(self, response: _FakeResponsesResponse) -> None:
        self.response = response
        self.last_kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> _FakeResponsesResponse:
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


class _FakeClient:
    def __init__(self, responses: _FakeResponses, embeddings: _FakeEmbeddings) -> None:
        self.responses = responses
        self.embeddings = embeddings


def _config() -> ProviderConfig:
    return ProviderConfig(
        kind=ProviderKind.OPENAI_RESPONSES,
        model="gpt-5.4-mini",
        embed_model="text-embedding-3-small",
        reasoning_effort="low",
    )


def test_complete_uses_responses_api_and_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = _FakeResponses(_FakeResponsesResponse("ok", _FakeUsage(11, 7)))
    provider = OpenAIResponsesProvider(_config())
    fake = _FakeClient(responses, _FakeEmbeddings(_FakeEmbeddingResponse([])))
    monkeypatch.setattr(provider, "_get_client", lambda: fake)

    completion = provider.complete(
        "be exact",
        [Message(role="user", content="ping")],
        max_tokens=64,
    )

    assert completion.text == "ok"
    assert completion.usage.input_tokens == 11
    assert completion.usage.output_tokens == 7
    sent = responses.last_kwargs
    assert sent["model"] == "gpt-5.4-mini"
    assert sent["max_output_tokens"] == 64
    assert sent["store"] is False
    assert sent["reasoning"] == {"effort": "low"}
    assert "temperature" not in sent
    assert sent["input"] == [
        {"role": "system", "content": "be exact"},
        {"role": "user", "content": "ping"},
    ]


def test_complete_default_max_tokens_is_8k(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = _FakeResponses(_FakeResponsesResponse("ok", _FakeUsage(1, 2)))
    provider = OpenAIResponsesProvider(_config())
    fake = _FakeClient(responses, _FakeEmbeddings(_FakeEmbeddingResponse([])))
    monkeypatch.setattr(provider, "_get_client", lambda: fake)

    provider.complete("", [Message(role="user", content="ping")])

    assert responses.last_kwargs["max_output_tokens"] == DEFAULT_MAX_TOKENS


def test_complete_parses_nested_output_when_output_text_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponsesResponse(
        "",
        None,
        output=[{"content": [{"type": "output_text", "text": "hello"}, {"text": " world"}]}],
    )
    responses = _FakeResponses(response)
    provider = OpenAIResponsesProvider(_config())
    fake = _FakeClient(responses, _FakeEmbeddings(_FakeEmbeddingResponse([])))
    monkeypatch.setattr(provider, "_get_client", lambda: fake)

    completion = provider.complete("", [Message(role="user", content="ping")])

    assert completion.text == "hello world"
    assert completion.usage.input_tokens == 0
    assert completion.usage.output_tokens == 0


def test_reasoning_omitted_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = _FakeResponses(_FakeResponsesResponse("ok", _FakeUsage(1, 2)))
    config = ProviderConfig(kind=ProviderKind.OPENAI_RESPONSES, model="gpt-5.5")
    provider = OpenAIResponsesProvider(config)
    fake = _FakeClient(responses, _FakeEmbeddings(_FakeEmbeddingResponse([])))
    monkeypatch.setattr(provider, "_get_client", lambda: fake)

    provider.complete("", [Message(role="user", content="ping")])

    assert "reasoning" not in responses.last_kwargs


def test_complete_chat_uses_native_responses_resource_without_legacy_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponsesResponse(
        "",
        _FakeUsage(17, 9),
        output=[
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "read_file",
                "arguments": '{"path":"README.md"}',
            }
        ],
    )
    responses = _FakeResponses(response)
    provider = OpenAIResponsesProvider(_config())
    fake = type("ResponsesOnlyClient", (), {"responses": responses})()
    monkeypatch.setattr(provider, "_get_client", lambda: fake)

    completion = provider.complete_chat(
        ChatRequest.model_validate(
            {
                "model": "worker",
                "messages": [{"role": "user", "content": "inspect"}],
                "max_completion_tokens": 2048,
                "store": False,
            }
        )
    )

    assert responses.last_kwargs == {
        "model": "gpt-5.4-mini",
        "input": [{"role": "user", "content": "inspect"}],
        "stream": False,
        "store": False,
        "max_output_tokens": 2048,
        "reasoning": {"effort": "low"},
        "include": ["reasoning.encrypted_content"],
    }
    assert completion.choices[0].finish_reason == "tool_calls"
    assert completion.choices[0].message.tool_calls is not None
    assert completion.choices[0].message.tool_calls[0].id == "call-1"
    assert completion.token_usage().input_tokens == 17
    assert completion.wire_payload()["service_tier"] == "priority"


def test_complete_chat_never_forwards_sampling_for_gpt5_without_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponsesResponse(
        "",
        _FakeUsage(3, 2),
        output=[
            {
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
    )
    responses = _FakeResponses(response)
    provider = OpenAIResponsesProvider(
        ProviderConfig(kind=ProviderKind.OPENAI_RESPONSES, model="gpt-5.5")
    )
    fake = type("ResponsesOnlyClient", (), {"responses": responses})()
    monkeypatch.setattr(provider, "_get_client", lambda: fake)

    provider.complete_chat(
        ChatRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "go"}],
                "temperature": 0.2,
                "top_p": 0.8,
            }
        )
    )

    assert "temperature" not in responses.last_kwargs
    assert "top_p" not in responses.last_kwargs
    assert responses.last_kwargs["include"] == ["reasoning.encrypted_content"]


def test_embed_uses_openai_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    embeddings = _FakeEmbeddings(_FakeEmbeddingResponse([[0.1, 0.2]]))
    config = _config().model_copy(update={"embed_dim": 2})
    provider = OpenAIResponsesProvider(config)
    responses = _FakeResponses(_FakeResponsesResponse("", _FakeUsage(0, 0)))
    monkeypatch.setattr(provider, "_get_client", lambda: _FakeClient(responses, embeddings))

    vectors = provider.embed(["a"])

    assert vectors == [[0.1, 0.2]]
    assert embeddings.last_kwargs["model"] == "text-embedding-3-small"
    assert embeddings.last_kwargs["input"] == ["a"]
    assert embeddings.last_kwargs["dimensions"] == 2


def test_embed_requires_embed_model() -> None:
    provider = OpenAIResponsesProvider(
        ProviderConfig(kind=ProviderKind.OPENAI_RESPONSES, model="gpt-5.5")
    )
    with pytest.raises(ValueError, match="embed_model"):
        provider.embed(["x"])


def test_verify_reports_failure_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        def create(self, **kwargs: object) -> object:
            raise RuntimeError("401")

    fake = type("C", (), {"responses": _Boom()})()
    provider = OpenAIResponsesProvider(_config())
    monkeypatch.setattr(provider, "_get_client", lambda: fake)

    result = provider.verify()

    assert result.ok is False
    assert result.kind is ProviderKind.OPENAI_RESPONSES
    assert "401" in result.detail


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in __import__("os").environ,
    reason="no OPENAI_API_KEY; skipping live smoke test",
)
def test_live_verify() -> None:  # pragma: no cover - network
    provider = OpenAIResponsesProvider(
        ProviderConfig(kind=ProviderKind.OPENAI_RESPONSES, model="gpt-5.4-mini")
    )
    assert provider.verify().ok is True
