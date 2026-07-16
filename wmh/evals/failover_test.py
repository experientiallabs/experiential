"""Tests for the grid's same-model failover: fail over on capacity, propagate real errors."""

from __future__ import annotations

import pytest

from wmh.evals.failover import SameModelFailover, anthropic_direct_id, same_model_chain
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind


class _StubProvider:
    """Returns `text`, or raises `raises` on complete()."""

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
    primary = _StubProvider("opus-4-8", text="from-primary")
    backup = _StubProvider("opus-4-8-direct", text="from-backup")
    fo = SameModelFailover([primary, backup])
    assert fo.complete("s", _msg()).text == "from-primary"
    assert backup.calls == 0  # never touched


def test_fails_over_on_capacity_error() -> None:
    primary = _StubProvider("opus-4-8", raises=RuntimeError("ThrottlingException: slow down"))
    backup = _StubProvider("opus-4-8-direct", text="from-backup")
    fo = SameModelFailover([primary, backup])
    assert fo.complete("s", _msg()).text == "from-backup"
    assert primary.calls == 1 and backup.calls == 1


def test_propagates_non_capacity_error() -> None:
    primary = _StubProvider("opus-4-8", raises=ValueError("malformed request: bad field"))
    backup = _StubProvider("opus-4-8-direct", text="from-backup")
    fo = SameModelFailover([primary, backup])
    with pytest.raises(ValueError, match="malformed"):
        fo.complete("s", _msg())
    assert backup.calls == 0  # a real error must NOT silently fall through to the backup


def test_raises_last_capacity_error_when_all_constrained() -> None:
    p1 = _StubProvider("opus-4-8", raises=RuntimeError("throttled"))
    p2 = _StubProvider("opus-4-8-direct", raises=RuntimeError("503 service unavailable"))
    fo = SameModelFailover([p1, p2])
    with pytest.raises(RuntimeError, match="service unavailable"):
        fo.complete("s", _msg())


def test_config_reports_primary() -> None:
    fo = SameModelFailover([_StubProvider("opus-4-8"), _StubProvider("opus-4-8-direct")])
    assert fo.config.model == "opus-4-8"


def test_empty_chain_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SameModelFailover([])


def test_same_model_chain_single_config_is_plain_provider() -> None:
    def factory(config: ProviderConfig) -> _StubProvider:
        return _StubProvider(config.model)

    cfg = ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-5.5")
    # single rung -> unwrapped plain provider
    assert not isinstance(same_model_chain([cfg], factory), SameModelFailover)
    two = [cfg, ProviderConfig(kind=ProviderKind.ANTHROPIC, model="gpt-5.5")]
    assert isinstance(same_model_chain(two, factory), SameModelFailover)


def test_anthropic_direct_id_maps_bedrock_ids() -> None:
    assert anthropic_direct_id("us.anthropic.claude-opus-4-8") == "claude-opus-4-8"
    assert anthropic_direct_id("us.anthropic.claude-haiku-4-5-20251001-v1:0") == "claude-haiku-4-5"
    assert anthropic_direct_id("us.anthropic.claude-opus-4-6-v1") == "claude-opus-4-6"
    assert anthropic_direct_id("gpt-5.5") is None  # non-Anthropic -> no direct equivalent
