"""Tests for shared provider helpers (verify ping budget + reachability semantics)."""

from __future__ import annotations

from wmh.providers.base import (
    PING_MAX_TOKENS,
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    VerifyResult,
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


class _RaisingProvider:
    """A provider whose complete() raises a fixed exception (to drive verify_via_ping)."""

    def __init__(self, exc: Exception) -> None:
        self.config = ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-5.5")
        self._exc = exc

    def complete(
        self, system: str, messages: list[Message], *, temperature: float = 0.7, max_tokens: int = 1
    ) -> Completion:
        raise self._exc

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - unused
        return [[0.0] for _ in texts]

    def verify(self) -> VerifyResult:  # pragma: no cover - unused
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


def test_verify_treats_max_tokens_limit_as_reachable() -> None:
    # If a reasoning model spends even the larger budget on reasoning and 400s before any output,
    # that error PROVES auth + model id are valid, so verify must still report ok=True.
    exc = Exception(
        "Error code: 400 - {'error': {'message': 'Could not finish the message because "
        "max_tokens or model output limit was reached. Please try again with higher max_tokens.'}}"
    )
    result = verify_via_ping(_RaisingProvider(exc))
    assert result.ok
    assert result.model == "gpt-5.5"


def test_verify_reports_real_failures() -> None:
    # Auth / missing-model / network errors are genuine failures - not reachability confirmations.
    exc = Exception("Error code: 401 - invalid api key")
    result = verify_via_ping(_RaisingProvider(exc))
    assert not result.ok
    assert "401" in (result.detail or "")
