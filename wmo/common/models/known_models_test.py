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
        ("openai", "gpt-5.4-2026-03-05", "gpt-5.4"),
        ("openai", "gpt-4.1-2025-04-14", "gpt-4.1"),
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
    assert known.input_cost_per_million_tokens_usd == 2.0
    assert known.output_cost_per_million_tokens_usd == 12.0
    assert known.cached_input_cost_per_million_tokens_usd == 0.2
    assert known.cache_write_cost_per_million_tokens_usd == 2.5
    assert known.supports_temperature is False
    assert known.supports_reasoning_effort


def test_reasoning_models_pin_sampling_and_small_models_carry_token_limits() -> None:
    """OpenAI reasoning models reject explicit temperature and small tiers state both limits."""
    mini = known_model_metadata("openai", "gpt-5.4-mini")
    embedding = known_model_metadata("openai", "text-embedding-3-large")

    assert mini is not None
    assert mini.supports_temperature is False
    assert mini.supports_reasoning_effort
    assert mini.context_window_tokens == 400_000
    assert mini.maximum_output_tokens == 128_000
    assert embedding is not None
    assert embedding.supports_temperature is None
    assert not embedding.supports_reasoning_effort


@pytest.mark.parametrize(
    (
        "model",
        "context_window_tokens",
        "maximum_output_tokens",
        "input_usd",
        "output_usd",
        "cached_input_usd",
        "cache_write_usd",
        "supports_structured_output",
    ),
    [
        ("gpt-5.6-sol", 1_050_000, 128_000, 5.0, 30.0, 0.5, 6.25, True),
        ("gpt-5.6-terra", 1_050_000, 128_000, 2.0, 12.0, 0.2, 2.5, True),
        ("gpt-5.6-luna", 1_050_000, 128_000, 0.2, 1.2, 0.02, 0.25, True),
        ("gpt-5.5", 1_050_000, 128_000, 5.0, 30.0, 0.5, 0.0, True),
        ("gpt-5.5-pro", 1_050_000, 128_000, 30.0, 180.0, None, 0.0, True),
        ("gpt-5.4", 1_050_000, 128_000, 2.5, 15.0, 0.25, 0.0, True),
        ("gpt-5.4-mini", 400_000, 128_000, 0.75, 4.5, 0.075, 0.0, True),
        ("gpt-5.4-nano", 400_000, 128_000, 0.2, 1.25, 0.02, 0.0, True),
        ("gpt-5.4-pro", 1_050_000, 128_000, 30.0, 180.0, None, 0.0, False),
        ("gpt-5.2", 400_000, 128_000, 1.75, 14.0, 0.175, 0.0, True),
        ("gpt-5.2-pro", 400_000, 128_000, 21.0, 168.0, None, 0.0, False),
        ("gpt-5.1", 400_000, 128_000, 1.25, 10.0, 0.125, 0.0, True),
        ("gpt-5", 400_000, 128_000, 1.25, 10.0, 0.125, 0.0, True),
        ("gpt-5-mini", 400_000, 128_000, 0.25, 2.0, 0.025, 0.0, True),
        ("gpt-5-nano", 400_000, 128_000, 0.05, 0.4, 0.005, 0.0, True),
        ("gpt-5-pro", 400_000, 272_000, 15.0, 120.0, None, 0.0, True),
    ],
)
def test_every_openai_chat_entry_has_complete_verified_metadata(
    model: str,
    context_window_tokens: int,
    maximum_output_tokens: int,
    input_usd: float,
    output_usd: float,
    cached_input_usd: float | None,
    cache_write_usd: float,
    supports_structured_output: bool,
) -> None:
    """Every advertised OpenAI chat role has verified limits, prices, and request shaping.

    Args:
        model: Canonical OpenAI model identifier.
        context_window_tokens: Documented input context ceiling.
        maximum_output_tokens: Documented output ceiling.
        input_usd: Documented input price per million tokens.
        output_usd: Documented output price per million tokens.
        cached_input_usd: Documented cache-hit price, or ``None`` when no discount exists.
        cache_write_usd: Documented cache-write price.
        supports_structured_output: Whether the model supports structured outputs.
    """
    known = known_model_metadata("openai", model)

    assert known is not None
    assert known.supports_completions
    assert known.supports_tools
    assert known.supports_temperature is False
    assert known.context_window_tokens == context_window_tokens
    assert known.maximum_output_tokens == maximum_output_tokens
    assert known.input_cost_per_million_tokens_usd == input_usd
    assert known.output_cost_per_million_tokens_usd == output_usd
    assert known.cached_input_cost_per_million_tokens_usd == cached_input_usd
    assert known.cache_write_cost_per_million_tokens_usd == cache_write_usd
    assert known.supports_structured_output is supports_structured_output


@pytest.mark.parametrize(
    ("model", "input_usd"),
    [
        ("text-embedding-3-small", 0.02),
        ("text-embedding-3-large", 0.13),
        ("text-embedding-ada-002", 0.1),
    ],
)
def test_documented_embedding_models_serve_embeddings_only(model: str, input_usd: float) -> None:
    """Every maintained OpenAI embedding entry has its verified input price and limit.

    Args:
        model: Canonical OpenAI embedding model identifier.
        input_usd: Documented input price per million tokens.
    """
    known = known_model_metadata("openai", model)

    assert known is not None
    assert known.supports_embeddings
    assert not known.supports_completions
    assert known.context_window_tokens == 8_192
    assert known.input_cost_per_million_tokens_usd == input_usd
    assert known.output_cost_per_million_tokens_usd is None
    assert known.cached_input_cost_per_million_tokens_usd is None
    assert known.cache_write_cost_per_million_tokens_usd is None


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
    ("model", "input_usd", "output_usd", "cached_input_usd"),
    [
        ("gemini-3.6-flash", 1.5, 7.5, 0.15),
        ("gemini-3.5-flash", 1.5, 9.0, 0.15),
        ("gemini-3.5-flash-lite", 0.3, 2.5, 0.03),
    ],
)
def test_every_gemini_chat_entry_has_complete_verified_metadata(
    model: str,
    input_usd: float,
    output_usd: float,
    cached_input_usd: float,
) -> None:
    """Every maintained Gemini chat entry has verified limits and every published price.

    Args:
        model: Canonical Gemini model identifier.
        input_usd: Documented input price per million tokens.
        output_usd: Documented output price per million tokens.
        cached_input_usd: Documented cached-input price per million tokens.
    """
    known = known_model_metadata("gemini", f"models/{model}")

    assert known is not None
    assert known.supports_completions
    assert known.supports_tools
    assert known.supports_structured_output
    assert known.supports_temperature is True
    assert known.context_window_tokens == 1_048_576
    assert known.maximum_output_tokens == 65_536
    assert known.input_cost_per_million_tokens_usd == input_usd
    assert known.output_cost_per_million_tokens_usd == output_usd
    assert known.cached_input_cost_per_million_tokens_usd == cached_input_usd
    assert known.cache_write_cost_per_million_tokens_usd == 0.0


def test_documented_gemini_embedding_model_serves_embeddings_only() -> None:
    """The Gemini embedding entry has its verified input price and token limit."""
    known = known_model_metadata("gemini", "models/gemini-embedding-001")

    assert known is not None
    assert known.supports_embeddings
    assert not known.supports_completions
    assert known.context_window_tokens == 2_048
    assert known.input_cost_per_million_tokens_usd == 0.15
    assert known.output_cost_per_million_tokens_usd is None


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
