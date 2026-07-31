"""Tests for the provider registry / entry point."""

from __future__ import annotations

import subprocess
import sys

import pytest

from wmo.providers import ProviderConfig, ProviderKind, get_provider, verify_embedder
from wmo.providers.base import Provider
from wmo.providers.tinker import TinkerChatProvider


def test_all_provider_kinds_construct_and_satisfy_protocol() -> None:
    for kind in ProviderKind:
        provider = get_provider(ProviderConfig(kind=kind, model="m"))
        assert isinstance(provider, Provider)


def test_tinker_kind_constructs_tinker_provider() -> None:
    # Construction is SDK-free: the tinker extra is imported lazily on first use.
    config = ProviderConfig(
        kind=ProviderKind.TINKER, model_type="Qwen/Qwen3-8B", model="tinker://run/weights/0"
    )
    assert isinstance(get_provider(config), TinkerChatProvider)


def test_get_provider_loads_only_the_selected_backend() -> None:
    """OpenAI construction must not import Anthropic or Tinker backend modules."""
    code = """
import sys
from wmo.providers.base import ProviderConfig, ProviderKind
from wmo.providers.registry import get_provider

get_provider(ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-5.5"))
assert "wmo.providers.openai" in sys.modules
bad = [
    m for m in ("wmo.providers.anthropic", "wmo.providers.tinker", "wmo.providers.bedrock")
    if m in sys.modules
]
assert not bad, f"unrelated backends loaded: {bad}"
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


def test_verify_never_raises_and_reports_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # No creds must surface as ok=False, never an exception — verify_all relies on this so
    # startup never crashes. Drop the key so this is deterministic regardless of the dev env.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = get_provider(ProviderConfig(kind=ProviderKind.ANTHROPIC, model="claude-opus-4-8"))
    result = provider.verify()
    assert result.ok is False
    assert result.kind is ProviderKind.ANTHROPIC


class _FakeEmbedProvider:
    """Minimal Embedder: returns a fixed-width vector per text."""

    def __init__(self, config: ProviderConfig, vector: list[float]) -> None:
        self.config = config
        self._vector = vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


class _BoomEmbedProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("no creds")


def test_verify_embedder_ok_reports_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    # A working embed path reports ok=True with the produced dimension and the embed_model.
    config = ProviderConfig(
        kind=ProviderKind.BEDROCK, model="opus", embed_model="amazon.titan-embed-text-v2:0"
    )
    monkeypatch.setattr(
        "wmo.providers.registry.get_provider",
        lambda cfg: _FakeEmbedProvider(cfg, [0.0, 1.0, 2.0]),
    )
    result = verify_embedder(config)
    assert result.ok is True
    assert result.model == "amazon.titan-embed-text-v2:0"
    assert result.detail == "dim=3"


def test_verify_embedder_reports_failure_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="opus")
    monkeypatch.setattr("wmo.providers.registry.get_provider", lambda cfg: _BoomEmbedProvider(cfg))
    result = verify_embedder(config)
    assert result.ok is False
    assert "no creds" in result.detail


def test_verify_embedder_empty_vector_is_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # A call that returns a zero-width vector ([[]]) didn't actually produce usable phi.
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="opus", embed_model="titan")
    monkeypatch.setattr(
        "wmo.providers.registry.get_provider", lambda cfg: _FakeEmbedProvider(cfg, [])
    )
    result = verify_embedder(config)
    assert result.ok is False
    assert "no vector" in result.detail
