"""Tests for the llm-waterfall backed provider (fake waterfall — no SDKs, no network)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from llm_waterfall import Backend, CompletionResult, EmbeddingResult
from llm_waterfall import TokenUsage as WfTokenUsage

from wmh.providers.base import Message, ProviderConfig, ProviderKind
from wmh.providers.waterfall import WaterfallProvider, to_backend


class _FakeWaterfall:
    def __init__(self) -> None:
        self.complete_calls: list[dict[str, object]] = []
        self.embed_calls: list[list[str]] = []

    def complete(
        self,
        system: str = "",
        messages: object = (),
        *,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        self.complete_calls.append(
            {
                "system": system,
                "messages": messages,
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
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        endpoint="https://x.openai.azure.com",
        deployment="gpt-55",
        api_version="2024-12-01-preview",
        embed_model="embed-dep",
        embed_dim=512,
    )
    backend = to_backend(config)
    assert backend == Backend(
        "azure_openai",
        "gpt-5.5",
        endpoint="https://x.openai.azure.com",
        deployment="gpt-55",
        api_version="2024-12-01-preview",
        embed_model="embed-dep",
        embed_dim=512,
    )


def test_to_backend_rejects_openai_responses() -> None:
    config = ProviderConfig(kind=ProviderKind.OPENAI_RESPONSES, model="gpt-5.5")
    with pytest.raises(ValueError, match="openai_responses"):
        to_backend(config)


def test_complete_maps_to_wmh_completion() -> None:
    fake = _FakeWaterfall()
    provider = WaterfallProvider(_configs(), waterfall=fake)
    completion = provider.complete("sys", [Message(role="user", content="hi")], max_tokens=64)
    assert completion.text == "served"
    assert completion.usage.input_tokens == 5 and completion.usage.output_tokens == 2
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
