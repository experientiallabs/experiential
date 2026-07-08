"""Tests for FallbackProvider: fail over on capacity errors, propagate real errors."""

from __future__ import annotations

import pytest

from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.providers.fallback import (
    FallbackProvider,
    _is_capacity_error,
    anthropic_direct_id,
    with_anthropic_fallover,
)


class _StubProvider:
    """Returns `text`, or raises `raises` (an Exception) on complete()."""

    def __init__(self, name: str, *, text: str = "", raises: Exception | None = None) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model=name)
        self._text = text
        self._raises = raises
        self.calls = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return Completion(text=self._text)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN202
        raise NotImplementedError


def _msg() -> list[Message]:
    return [Message(role="user", content="hi")]


def test_uses_primary_when_healthy() -> None:
    primary = _StubProvider("opus-4-6", text="from-primary")
    backup = _StubProvider("opus-4-7", text="from-backup")
    fb = FallbackProvider([primary, backup])
    assert fb.complete("s", _msg()).text == "from-primary"
    assert backup.calls == 0  # never touched


def test_fails_over_on_capacity_error() -> None:
    primary = _StubProvider("opus-4-6", raises=RuntimeError("ThrottlingException: slow down"))
    backup = _StubProvider("opus-4-7", text="from-backup")
    fb = FallbackProvider([primary, backup])
    assert fb.complete("s", _msg()).text == "from-backup"
    assert primary.calls == 1 and backup.calls == 1


def test_propagates_non_capacity_error() -> None:
    primary = _StubProvider("opus-4-6", raises=ValueError("malformed request: bad field"))
    backup = _StubProvider("opus-4-7", text="from-backup")
    fb = FallbackProvider([primary, backup])
    with pytest.raises(ValueError, match="malformed"):
        fb.complete("s", _msg())
    assert backup.calls == 0  # a real error must NOT silently fall through to the backup


def test_raises_last_capacity_error_when_all_constrained() -> None:
    p1 = _StubProvider("opus-4-6", raises=RuntimeError("throttled"))
    p2 = _StubProvider("opus-4-7", raises=RuntimeError("503 service unavailable"))
    fb = FallbackProvider([p1, p2])
    with pytest.raises(RuntimeError, match="service unavailable"):
        fb.complete("s", _msg())


def test_config_reports_primary() -> None:
    fb = FallbackProvider([_StubProvider("opus-4-6"), _StubProvider("opus-4-7")])
    assert fb.config.model == "opus-4-6"


def test_empty_chain_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FallbackProvider([])


class _ClientError(Exception):
    """Mimics botocore ClientError: carries a `.response` dict with Error.Code."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(f"{code}: {message}")
        self.response = {"Error": {"Code": code, "Message": message}}


def test_capacity_classifier_prefers_error_code() -> None:
    # Structured botocore error codes are the reliable signal.
    assert _is_capacity_error(_ClientError("ThrottlingException", "slow down"))
    assert _is_capacity_error(_ClientError("ServiceUnavailableException"))
    assert _is_capacity_error(_ClientError("ModelNotReadyException"))
    # A real client error is NOT capacity, even if its MESSAGE mentions a scary token — the code
    # decides, so it propagates instead of being wrongly retried across the whole chain.
    assert not _is_capacity_error(_ClientError("ValidationException", "request timeout too large"))
    assert not _is_capacity_error(_ClientError("AccessDeniedException", "not available"))


def test_capacity_classifier_falls_back_to_markers_for_transport_errors() -> None:
    # Transport errors (read/connect timeouts) have no response code -> conservative string match.
    assert _is_capacity_error(RuntimeError("Read timeout on endpoint URL"))
    assert _is_capacity_error(RuntimeError("throttled"))
    # A plain bad-request with no code and no capacity phrasing must NOT be treated as capacity.
    assert not _is_capacity_error(ValueError("malformed request: bad field"))


def test_anthropic_direct_id_maps_bedrock_ids() -> None:
    assert anthropic_direct_id("us.anthropic.claude-opus-4-8") == "claude-opus-4-8"
    assert anthropic_direct_id("us.anthropic.claude-haiku-4-5-20251001-v1:0") == "claude-haiku-4-5"
    assert anthropic_direct_id("us.anthropic.claude-opus-4-6-v1") == "claude-opus-4-6"
    assert anthropic_direct_id("gpt-5.5") is None  # non-Anthropic -> no direct equivalent


def test_with_anthropic_fallover_wraps_bedrock_only() -> None:
    built: list[str] = []

    def factory(config: ProviderConfig) -> _StubProvider:
        built.append(f"{config.kind.value}:{config.model}")
        return _StubProvider(config.model)

    # Bedrock Anthropic model -> FallbackProvider [bedrock, direct-anthropic same model].
    p = with_anthropic_fallover(
        ProviderConfig(kind=ProviderKind.BEDROCK, model="us.anthropic.claude-opus-4-8"), factory
    )
    assert isinstance(p, FallbackProvider)
    assert built == ["bedrock:us.anthropic.claude-opus-4-8", "anthropic:claude-opus-4-8"]

    # Non-Bedrock (or non-Anthropic) -> a single provider, no fallover.
    built.clear()
    single = with_anthropic_fallover(
        ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-5.5"), factory
    )
    assert not isinstance(single, FallbackProvider)
