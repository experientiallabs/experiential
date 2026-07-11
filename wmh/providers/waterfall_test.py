"""Tests for the llm-waterfall backed provider (fake waterfall — no SDKs, no network)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from llm_waterfall import (
    Backend,
    ChatRequest,
    ChatResponse,
    ChatResult,
    CompletionResult,
    EmbeddingResult,
    Waterfall,
)
from llm_waterfall import Message as WfMessage
from llm_waterfall import TokenUsage as WfTokenUsage
from llm_waterfall import VerifyResult as WfVerifyResult

from wmh.providers.base import Message, ProviderConfig, ProviderKind
from wmh.providers.waterfall import WaterfallProvider, provider_or_chain, to_backend


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

    def complete_chat(self, request: ChatRequest) -> ChatResult:
        return ChatResult(
            response=ChatResponse.model_validate(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "structured"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ),
            model_used="sonnet",
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
    # openai_responses has no package equivalent (the package speaks chat-completions).
    with pytest.raises(ValueError, match="no llm-waterfall backend"):
        to_backend(ProviderConfig(kind=ProviderKind.OPENAI_RESPONSES, model="m"))


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


def test_complete_chat_delegates_without_collapsing_tool_shape() -> None:
    provider = WaterfallProvider(_configs(), waterfall=_FakeWaterfall())
    response = provider.complete_chat(
        ChatRequest.model_validate({"messages": [{"role": "user", "content": "hi"}]})
    )
    assert response.choices[0].message.content == "structured"


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


_CHAIN_TOML = """
default = "main"

[[chain.main]]
kind = "bedrock"
model = "us.anthropic.claude-opus-4-6-v1"
profile = "endflow"
region = "us-west-2"

[[chain.main]]
kind = "bedrock"
model = "us.anthropic.claude-opus-4-8"
region = "us-west-2"

[[chain.main]]
kind = "openai"
model = "gpt-5.5"
api_key = "sk-test-not-real"

[[chain.opus-48]]
kind = "bedrock"
model = "us.anthropic.claude-opus-4-8"
region = "us-west-2"

[[chain.opus-48]]
kind = "anthropic"
model = "claude-opus-4-8"
"""


def test_fallback_config_parses_chain_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = tmp_path / "fallback.toml"
    path.write_text(_CHAIN_TOML)
    requested = ProviderConfig(
        kind=ProviderKind.BEDROCK, model="us.anthropic.claude-opus-4-6-v1", region="us-west-2"
    )
    provider = provider_or_chain(requested, path=path)
    assert isinstance(provider, WaterfallProvider)
    assert isinstance(provider._waterfall, Waterfall)
    backends = provider._waterfall.backends
    assert [b.model for b in backends] == [
        "us.anthropic.claude-opus-4-6-v1",
        "us.anthropic.claude-opus-4-8",
        "gpt-5.5",
    ]
    assert [b.profile for b in backends] == ["endflow", None, None]
    # The gitignored file seeds the OpenAI key so the chain is self-contained.
    import os

    assert os.environ["OPENAI_API_KEY"] == "sk-test-not-real"


def test_requested_model_leads_when_not_heading_chain(tmp_path: Path) -> None:
    path = tmp_path / "fallback.toml"
    path.write_text(_CHAIN_TOML)
    requested = ProviderConfig(kind=ProviderKind.BEDROCK, model="us.anthropic.claude-haiku-4-5")
    provider = provider_or_chain(requested, path=path)
    assert isinstance(provider, WaterfallProvider)
    assert isinstance(provider._waterfall, Waterfall)
    assert provider._waterfall.backends[0].model == "us.anthropic.claude-haiku-4-5"
    assert len(provider._waterfall.backends) == 4  # requested + main's 3 rungs
    assert provider.config is requested  # metering still labels the intended primary


def test_no_chain_file_falls_back_to_single_provider(tmp_path: Path) -> None:
    requested = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
    provider = provider_or_chain(requested, path=tmp_path / "absent.toml")
    assert not isinstance(provider, WaterfallProvider)
    assert provider.config.model == "m"


def test_fallback_config_rejects_unknown_keys_and_kinds(tmp_path: Path) -> None:
    path = tmp_path / "fallback.toml"
    path.write_text('[[chain.c]]\nkind = "bedrock"\nmodel = "m"\ntypo_key = 1\n')
    with pytest.raises(ValueError, match="unknown key"):
        provider_or_chain(ProviderConfig(kind=ProviderKind.BEDROCK, model="m"), path=path)
    path.write_text('[[chain.c]]\nkind = "not-a-kind"\nmodel = "m"\n')
    with pytest.raises(ValueError, match="is invalid"):
        provider_or_chain(ProviderConfig(kind=ProviderKind.BEDROCK, model="m"), path=path)
    path.write_text('[[chain.c]]\nkind = "openai_responses"\nmodel = "m"\n')
    with pytest.raises(ValueError, match="no llm-waterfall backend"):
        provider_or_chain(ProviderConfig(kind=ProviderKind.BEDROCK, model="m"), path=path)
    path.write_text('[[chain.c]]\nkind = "bedrock"\nmodel = "m"\napi_key = "sk-x"\n')
    with pytest.raises(ValueError, match="api_key only applies"):
        provider_or_chain(ProviderConfig(kind=ProviderKind.BEDROCK, model="m"), path=path)


def test_azure_rung_maps_endpoint_deployment_and_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    path = tmp_path / "fallback.toml"
    path.write_text(
        "[[chain.az]]\n"
        'kind = "azure"\n'
        'model = "gpt-5.4"\n'
        'endpoint = "https://x.openai.azure.com"\n'
        'deployment = "gpt-54-deploy"\n'
        'api_version = "2024-12-01-preview"\n'
        'api_key = "az-test-key"\n'
    )
    requested = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.4",
        endpoint="https://x.openai.azure.com",
        api_version="2024-12-01-preview",
    )
    provider = provider_or_chain(requested, chain="az", path=path)
    assert isinstance(provider, WaterfallProvider)
    assert isinstance(provider._waterfall, Waterfall)
    backend = provider._waterfall.backends[0]
    # wmh's kind value is "azure"; the package spells the provider "azure_openai".
    assert backend.provider == "azure_openai"
    assert backend.deployment == "gpt-54-deploy"
    assert backend.api_version == "2024-12-01-preview"
    assert os.environ["AZURE_OPENAI_API_KEY"] == "az-test-key"
