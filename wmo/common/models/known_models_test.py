"""Maintained model metadata tests."""

from __future__ import annotations

import pytest

from wmo.common.models.known_models import (
    canonical_model_id,
    known_model_metadata,
    recommended_model_rank,
)


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("openai", "GPT-5.6-Sol", "gpt-5.6-sol"),
        ("openai", "gpt-5.1-20260210", "gpt-5.1"),
        ("anthropic", "claude-haiku-4-5-20251001", "claude-haiku-4-5"),
        ("anthropic", "claude-sonnet-5-latest", "claude-sonnet-5"),
        ("gemini", "models/gemini-3.5-flash", "gemini-3.5-flash"),
    ],
)
def test_canonical_identity_absorbs_snapshots_pointers_and_resource_prefixes(
    provider: str,
    model: str,
    expected: str,
) -> None:
    """Dated snapshots, pointer aliases, and resource prefixes name one documented model.

    Args:
        provider: Setup provider kind publishing the model.
        model: Provider-published model ID.
        expected: Normalized identity the maintained table indexes.
    """
    assert canonical_model_id(provider, model) == expected


def test_documented_chat_model_carries_verified_capabilities_and_prices() -> None:
    """A verified chat entry states protocol capabilities, limits, and every published price."""
    known = known_model_metadata("openai", "gpt-5.6-terra-20260216")

    assert known is not None
    assert known.supports_completions
    assert known.supports_tools
    assert known.supports_structured_output
    assert not known.supports_embeddings
    assert known.context_window_tokens == 1_050_000
    assert known.maximum_output_tokens == 128_000
    assert known.input_cost_per_million_tokens_usd == 2.5
    assert known.output_cost_per_million_tokens_usd == 15.0
    assert known.cached_input_cost_per_million_tokens_usd == 0.25
    assert known.cache_write_cost_per_million_tokens_usd == 3.125
    assert not known.supports_temperature
    assert known.reasoning_effort == "xhigh"


def test_reasoning_models_pin_sampling_and_small_models_carry_token_limits() -> None:
    """OpenAI reasoning models reject explicit temperature and small tiers state both limits."""
    mini = known_model_metadata("openai", "gpt-5.4-mini")
    embedding = known_model_metadata("openai", "text-embedding-3-large")

    assert mini is not None
    assert not mini.supports_temperature
    assert mini.reasoning_effort is None
    assert mini.context_window_tokens == 400_000
    assert mini.maximum_output_tokens == 64_000
    assert embedding is not None
    assert embedding.supports_temperature
    assert embedding.reasoning_effort is None


def test_documented_embedding_model_serves_embeddings_only() -> None:
    """An embedding entry prices input tokens and never claims completion support."""
    known = known_model_metadata("openai", "text-embedding-3-small")

    assert known is not None
    assert known.supports_embeddings
    assert not known.supports_completions
    assert known.input_cost_per_million_tokens_usd == 0.02
    assert known.output_cost_per_million_tokens_usd is None


def test_documented_anthropic_model_prices_both_cache_operations() -> None:
    """Anthropic publishes explicit cache write and cache hit prices per model."""
    known = known_model_metadata("anthropic", "claude-sonnet-5")

    assert known is not None
    assert known.input_cost_per_million_tokens_usd == 2.0
    assert known.output_cost_per_million_tokens_usd == 10.0
    assert known.cached_input_cost_per_million_tokens_usd == 0.2
    assert known.cache_write_cost_per_million_tokens_usd == 2.5
    assert known.context_window_tokens == 1_000_000
    assert known.maximum_output_tokens == 128_000


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("openai", "gpt-private-preview"),
        ("openrouter", "vendor/model"),
        ("openai-compatible", "internal-model"),
        ("tinker", "base-model"),
    ],
)
def test_unverified_models_have_no_maintained_metadata(provider: str, model: str) -> None:
    """An unverified or provider-published-metadata model stays unknown to this table.

    Args:
        provider: Setup provider kind publishing the model.
        model: Provider-published model ID absent from the maintained table.
    """
    assert known_model_metadata(provider, model) is None


def test_recommendation_ranks_are_explicit_per_provider_and_role() -> None:
    """Maintained wizard guidance ranks exact models without relying on alias order."""
    assert recommended_model_rank("openai", "gpt-5.6-luna", "world_model") == 0
    assert recommended_model_rank("openai", "text-embedding-3-large", "embedder") == 0
    assert recommended_model_rank("anthropic", "claude-sonnet-5", "judge") == 0
    assert recommended_model_rank("gemini", "models/gemini-3.6-flash", "router_candidate") == 0
    assert recommended_model_rank("openai", "gpt-5.4-mini", "world_model") is None
