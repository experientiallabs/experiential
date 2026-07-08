"""Tests for the shared provider verify ping."""

from __future__ import annotations

from wmh.providers.base import (
    PING_MAX_TOKENS,
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    verify_via_ping,
)


class RecordingProvider:
    """Captures the ping's max_tokens so the budget is pinned by a test."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-5.5")
        self.seen_max_tokens: int | None = None

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.seen_max_tokens = max_tokens
        return Completion(text="pong")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        return verify_via_ping(self)


def test_ping_budget_covers_reasoning_models() -> None:
    # Reasoning models (GPT-5.x) burn output budget on reasoning before any visible token, so a
    # tiny ping budget makes OpenAI 400 with "max_tokens or model output limit was reached" even
    # though the credentials are fine. The ping must send a budget with headroom for that.
    provider = RecordingProvider()
    result = verify_via_ping(provider)
    assert result.ok is True
    assert provider.seen_max_tokens == PING_MAX_TOKENS
    assert PING_MAX_TOKENS >= 1024
