"""Tests for the narrated retry wrapper."""

from __future__ import annotations

import pytest

from wmh.providers.base import (
    ChatRequest,
    ChatResponse,
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    ToolCallingProvider,
)
from wmh.providers.retry import (
    RetryingProvider,
    RetryingToolCallingProvider,
    wrap_provider_with_retries,
)


class _Throttle(Exception):
    def __init__(self) -> None:
        super().__init__("Bedrock is unable to process your request")
        self.response = {"Error": {"Code": "ServiceUnavailableException"}}


class FlakyProvider:
    """Raises `failures` capacity errors, then succeeds."""

    def __init__(self, failures: int) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._failures = failures
        self.calls = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.calls += 1
        if self.calls <= self._failures:
            raise _Throttle()
        return Completion(text="ok")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


class FlakyToolCallingProvider(FlakyProvider):
    """Raises capacity errors from structured calls, then succeeds."""

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        _ = request
        self.calls += 1
        if self.calls <= self._failures:
            raise _Throttle()
        return ChatResponse.model_validate(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            }
        )


def test_retries_capacity_errors_with_backoff_and_narration() -> None:
    events: list[tuple[int, int, float]] = []
    slept: list[float] = []
    provider = FlakyProvider(failures=2)
    retrying = RetryingProvider(
        provider,
        on_retry=lambda a, t, d, e: events.append((a, t, d)),
        sleep=slept.append,
    )
    result = retrying.complete("s", [Message(role="user", content="hi")])
    assert result.text == "ok"
    assert events == [(1, 3, 1.0), (2, 3, 3.0)]  # narrated attempt k/total with the delay
    assert slept == [1.0, 3.0]


def test_exhausted_retries_reraise_the_capacity_error() -> None:
    provider = FlakyProvider(failures=10)
    retrying = RetryingProvider(provider, sleep=lambda _s: None)
    with pytest.raises(_Throttle):
        retrying.complete("s", [Message(role="user", content="hi")])
    assert provider.calls == 4  # 3 backoff attempts + the final propagate attempt


def test_non_capacity_errors_propagate_immediately() -> None:
    class Auth(Exception):
        pass

    class BadProvider(FlakyProvider):
        def complete(self, system, messages, *, temperature=0.7, max_tokens=8192):  # noqa: ANN001, ANN202
            self.calls += 1
            raise Auth("invalid api key")

    provider = BadProvider(failures=0)
    retrying = RetryingProvider(provider, sleep=lambda _s: None)
    with pytest.raises(Auth):
        retrying.complete("s", [Message(role="user", content="hi")])
    assert provider.calls == 1


def test_tool_calling_retry_preserves_structured_surface() -> None:
    events: list[tuple[int, int, float]] = []
    slept: list[float] = []
    provider = FlakyToolCallingProvider(failures=1)
    retrying = RetryingToolCallingProvider(
        provider,
        on_retry=lambda attempt, total, delay, error: events.append((attempt, total, delay)),
        sleep=slept.append,
    )

    assert isinstance(retrying, ToolCallingProvider)
    response = retrying.complete_chat(ChatRequest())

    assert response.token_usage().input_tokens == 11
    assert response.token_usage().output_tokens == 7
    assert provider.calls == 2
    assert events == [(1, 3, 1.0)]
    assert slept == [1.0]


def test_retry_factory_preserves_only_the_underlying_provider_capabilities() -> None:
    structured = wrap_provider_with_retries(FlakyToolCallingProvider(failures=0))
    text_only = wrap_provider_with_retries(FlakyProvider(failures=0))

    assert isinstance(structured, RetryingToolCallingProvider)
    assert isinstance(structured, ToolCallingProvider)
    assert type(text_only) is RetryingProvider
    assert not isinstance(text_only, ToolCallingProvider)
