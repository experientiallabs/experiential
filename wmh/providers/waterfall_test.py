"""Tests for the llm-waterfall backed provider (fake waterfall — no SDKs, no network)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
from llm_waterfall import Backend, CompletionResult, EmbeddingResult
from llm_waterfall import Message as WfMessage
from llm_waterfall import TokenUsage as WfTokenUsage
from llm_waterfall import VerifyResult as WfVerifyResult

from wmh.providers.base import Message, ProviderConfig, ProviderKind
from wmh.providers.waterfall import WaterfallProvider, to_backend


class _FakeWaterfall:
    def __init__(self) -> None:
        self.complete_calls: list[dict[str, object]] = []
        self.embed_calls: list[list[str]] = []

    def complete(
        self,
        system: str = "",
        messages: Sequence[WfMessage | Mapping[str, str]] = (),
        *,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        self.complete_calls.append(
            {
                "system": system,
                "messages": list(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return CompletionResult(
            text="served",
            model_used="us.anthropic.claude-sonnet-4-6",
            provider_used="bedrock",
            usage=WfTokenUsage(input_tokens=5, output_tokens=2),
            cost_usd=0.001,
        )

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        self.embed_calls.append(list(texts))
        return EmbeddingResult(
            vectors=[[0.1] for _ in texts],
            model_used="amazon.titan-embed-text-v2:0",
            provider_used="bedrock",
        )

    def verify(self) -> list[WfVerifyResult]:
        return [
            WfVerifyResult(ok=True, provider="bedrock", model="opus"),
            WfVerifyResult(ok=False, provider="bedrock", model="sonnet", detail="expired creds"),
        ]


def _configs() -> list[ProviderConfig]:
    return [
        ProviderConfig(
            kind=ProviderKind.BEDROCK, model="us.anthropic.claude-opus-4-8", region="us-west-2"
        ),
        ProviderConfig(
            kind=ProviderKind.BEDROCK, model="us.anthropic.claude-sonnet-4-6", region="us-west-2"
        ),
    ]


def test_to_backend_maps_config_fields() -> None:
    config = ProviderConfig(
        kind=ProviderKind.OPENAI,
        model="gpt-5.5",
        endpoint="https://proxy.example.com/v1",
        embed_model="text-embedding-3-large",
        embed_dim=512,
    )
    backend = to_backend(config, profile=None)
    assert backend == Backend(
        "openai",
        "gpt-5.5",
        endpoint="https://proxy.example.com/v1",
        embed_model="text-embedding-3-large",
        embed_dim=512,
    )


def test_to_backend_rejects_kinds_without_real_adapters() -> None:
    # openai_responses has no package equivalent; azure_openai is a construction-time stub
    # upstream — advertising it here would turn a fallback rung into a mid-run landmine.
    for kind in (ProviderKind.OPENAI_RESPONSES, ProviderKind.AZURE_OPENAI):
        with pytest.raises(ValueError, match="no llm-waterfall backend"):
            to_backend(ProviderConfig(kind=kind, model="m"))


def test_complete_maps_to_wmh_completion() -> None:
    fake = _FakeWaterfall()
    provider = WaterfallProvider(_configs(), waterfall=fake)
    completion = provider.complete("sys", [Message(role="user", content="hi")], max_tokens=64)
    assert completion.text == "served"
    assert completion.usage.input_tokens == 5 and completion.usage.output_tokens == 2
    # The served model (sonnet fallback), not the configured primary (opus).
    assert completion.model == "us.anthropic.claude-sonnet-4-6"
    call = fake.complete_calls[0]
    assert call["system"] == "sys" and call["max_tokens"] == 64
    # Temperature is intentionally not forwarded (current models reject sampling params).
    assert call["temperature"] is None


def test_config_reports_primary_for_metering() -> None:
    provider = WaterfallProvider(_configs(), waterfall=_FakeWaterfall())
    assert provider.config.model == "us.anthropic.claude-opus-4-8"
    assert provider.config.kind is ProviderKind.BEDROCK


def test_embed_delegates_to_waterfall() -> None:
    fake = _FakeWaterfall()
    provider = WaterfallProvider(_configs(), waterfall=fake)
    assert provider.embed(["a", "b"]) == [[0.1], [0.1]]
    assert fake.embed_calls == [["a", "b"]]


def test_empty_chain_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        WaterfallProvider([], waterfall=_FakeWaterfall())


def test_profiles_pin_rungs_to_named_aws_accounts(monkeypatch: pytest.MonkeyPatch) -> None:
    # Multi-account chains are the headline use case: profiles zip 1:1 onto configs.
    captured: list[Backend] = []

    def capture(backends: list[Backend], retry: object) -> _FakeWaterfall:
        captured.extend(backends)
        return _FakeWaterfall()

    monkeypatch.setattr("wmh.providers.waterfall.Waterfall", capture)
    WaterfallProvider(_configs(), profiles=["endflow", "stackwise"])
    assert [b.profile for b in captured] == ["endflow", "stackwise"]


def test_profiles_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="one-to-one"):
        WaterfallProvider(_configs(), profiles=["endflow"], waterfall=_FakeWaterfall())


def test_verify_checks_every_rung_and_names_failures() -> None:
    # A ping through the chain would let a fallback answer for a dead primary; per-rung
    # verification must fail the chain and say which rung is broken.
    provider = WaterfallProvider(_configs(), waterfall=_FakeWaterfall())
    result = provider.verify()
    assert result.ok is False
    assert "bedrock/sonnet: expired creds" in result.detail
