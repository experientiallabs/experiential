"""Unit tests for AnthropicProvider. No network: the SDK client is faked via _get_client."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wmo.providers.anthropic import AnthropicProvider
from wmo.providers.base import ChatRequest, Message, ProviderConfig, ProviderKind

if TYPE_CHECKING:
    from collections.abc import Sequence


class _FakeTextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, content: Sequence[object], usage: _FakeUsage) -> None:
        self.content = content
        self.usage = usage


class _FakeMessages:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.last_kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> _FakeResponse:
        self.last_kwargs = kwargs
        return self.response


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.messages = _FakeMessages(response)


def _config() -> ProviderConfig:
    return ProviderConfig(kind=ProviderKind.ANTHROPIC, model="claude-opus-4-8")


def test_complete_maps_request_and_parses_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        _FakeResponse([_FakeTextBlock("hello "), _FakeTextBlock("world")], _FakeUsage(11, 7))
    )
    provider = AnthropicProvider(_config())
    monkeypatch.setattr(provider, "_get_client", lambda: fake)  # inject fake; no network

    completion = provider.complete("be terse", [Message(role="user", content="hi")], max_tokens=64)

    assert completion.text == "hello world"
    assert completion.usage.input_tokens == 11
    assert completion.usage.output_tokens == 7
    sent = fake.messages.last_kwargs
    assert sent["model"] == "claude-opus-4-8"
    assert sent["system"] == "be terse"
    assert sent["max_tokens"] == 64
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    assert "temperature" not in sent  # Opus 4.8 rejects sampling params


def test_embed_raises_pointing_at_embed_provider() -> None:
    provider = AnthropicProvider(_config())
    with pytest.raises(NotImplementedError, match="OpenAI or Bedrock embed provider"):
        provider.embed(["x"])


def test_complete_chat_uses_native_messages_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeClient(
        _FakeResponse(
            [
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "bash",
                    "input": {"command": "pwd"},
                }
            ],
            _FakeUsage(11, 7),
        )
    )
    provider = AnthropicProvider(
        ProviderConfig(
            kind=ProviderKind.ANTHROPIC,
            model="claude-sonnet-5",
            reasoning_effort="high",
        )
    )
    monkeypatch.setattr(provider, "_get_client", lambda: fake)

    response = provider.complete_chat(
        ChatRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Inspect."}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            }
        )
    )

    calls = response.choices[0].message.tool_calls
    assert calls is not None
    assert calls[0].function.name == "bash"
    assert fake.messages.last_kwargs["thinking"] == {"type": "adaptive"}
    assert fake.messages.last_kwargs["output_config"] == {"effort": "high"}


def test_verify_ok_on_successful_ping(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(_FakeResponse([_FakeTextBlock("p")], _FakeUsage(1, 1)))
    provider = AnthropicProvider(_config())
    monkeypatch.setattr(provider, "_get_client", lambda: fake)
    result = provider.verify()
    assert result.ok is True
    assert result.kind is ProviderKind.ANTHROPIC


def test_verify_reports_failure_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        def create(self, **kwargs: object) -> object:
            raise RuntimeError("bad key")

    fake = type("C", (), {"messages": _Boom()})()
    provider = AnthropicProvider(_config())
    monkeypatch.setattr(provider, "_get_client", lambda: fake)
    result = provider.verify()
    assert result.ok is False
    assert "bad key" in result.detail


@pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in __import__("os").environ,
    reason="no ANTHROPIC_API_KEY; skipping live smoke test",
)
def test_live_verify() -> None:  # pragma: no cover - network
    provider = AnthropicProvider(_config())
    assert provider.verify().ok is True


class _FakeChatMessages:
    """Records create() kwargs and returns a dict-shaped Messages API response."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> dict[str, object]:
        self.last_kwargs = kwargs
        return {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }


class _FakeChatClient:
    def __init__(self) -> None:
        self.messages = _FakeChatMessages()


def test_complete_chat_wires_the_configs_reasoning_effort_to_the_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one line that makes PoolEntry.reasoning_effort mean anything on this path.

    The translator is covered on its own; this pins the provider handing the
    CONFIG's effort through, so two pool arms differing only in effort stay
    two distinct arms instead of both running at the backend default.
    """
    fake = _FakeChatClient()
    provider = AnthropicProvider(
        ProviderConfig(kind=ProviderKind.ANTHROPIC, model="claude-fable-5", reasoning_effort="none")
    )
    monkeypatch.setattr(provider, "_get_client", lambda: fake)  # inject fake; no network

    response = provider.complete_chat(
        ChatRequest.model_validate({"messages": [{"role": "user", "content": "hi"}]})
    )

    assert fake.messages.last_kwargs["output_config"] == {"effort": "none"}
    assert fake.messages.last_kwargs["model"] == "claude-fable-5"
    assert response.choices[0].message.content == "ok"


def test_text_paths_refuse_a_config_whose_effort_they_would_drop() -> None:
    """Two arms differing only in effort must never silently collapse into one."""
    provider = AnthropicProvider(
        ProviderConfig(kind=ProviderKind.ANTHROPIC, model="claude-fable-5", reasoning_effort="max")
    )

    with pytest.raises(ValueError, match="does not forward reasoning_effort"):
        provider.complete("system", [Message(role="user", content="hi")])
    with pytest.raises(ValueError, match="does not forward reasoning_effort"):
        next(provider.stream("system", [Message(role="user", content="hi")]))
