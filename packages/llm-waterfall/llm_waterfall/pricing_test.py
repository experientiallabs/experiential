"""Tests for model-id normalization and USD cost computation."""

from __future__ import annotations

from llm_waterfall.pricing import ModelPrice, cost_usd, price_for
from llm_waterfall.types import TokenUsage


def test_us_bedrock_id_uses_the_audited_geo_route_price() -> None:
    dated = price_for("us.anthropic.claude-opus-4-8-20260101-v1:0")
    bare = price_for("claude-opus-4-8")
    assert dated == ModelPrice(input_per_mtok=5.5, output_per_mtok=27.5)
    assert bare == ModelPrice(input_per_mtok=5.0, output_per_mtok=25.0)


def test_unaudited_geo_route_does_not_inherit_a_direct_provider_price() -> None:
    assert price_for("eu.anthropic.claude-sonnet-4-6") is None
    assert price_for("us.anthropic.claude-sonnet-4-6") is None
    assert price_for("anthropic.claude-haiku-4-5") == price_for("claude-haiku-4-5")


def test_titan_id_kept_intact() -> None:
    assert price_for("amazon.titan-embed-text-v2:0") is not None


def test_unknown_model_costs_zero_and_prices_none() -> None:
    assert price_for("mystery-model-9000") is None
    assert cost_usd("mystery-model-9000", TokenUsage(input_tokens=1000, output_tokens=1000)) == 0.0


def test_cost_math() -> None:
    # Opus 4.8: $5/Mtok in, $25/Mtok out.
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=100_000)
    assert cost_usd("claude-opus-4-8", usage) == 5.0 + 2.5


def test_overrides_win_and_do_not_mutate_table() -> None:
    override = {"claude-opus-4-8": ModelPrice(input_per_mtok=1.0, output_per_mtok=1.0)}
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0)
    assert cost_usd("claude-opus-4-8", usage, prices=override) == 1.0
    # The static table is untouched for callers without the override.
    assert cost_usd("claude-opus-4-8", usage) == 5.0


def test_override_adds_unknown_model() -> None:
    override = {"my-azure-deployment": ModelPrice(input_per_mtok=2.5, output_per_mtok=15.0)}
    usage = TokenUsage(input_tokens=2_000_000, output_tokens=0)
    assert (
        cost_usd(
            "my-azure-deployment",
            usage,
            prices=override,
            provider="azure_openai",
        )
        == 5.0
    )


def test_azure_cannot_inherit_a_direct_openai_price() -> None:
    direct = price_for("gpt-5.5", provider="openai")

    assert direct == ModelPrice(input_per_mtok=5.0, output_per_mtok=30.0)
    assert price_for("gpt-5.5", provider="azure_openai") is None
    # WMH spells the same provider kind "azure"; accepting that alias keeps its adapter exact.
    assert price_for("gpt-5.5", provider="azure") is None
    assert price_for("gpt-5.5", provider="anthropic") is None
    assert price_for("claude-opus-4-8", provider="openai") is None
    assert price_for("gpt-5.5", provider="unknown") is None


def test_azure_override_is_scoped_to_the_exact_deployment() -> None:
    override = {"azure-gpt-production": ModelPrice(input_per_mtok=1.25, output_per_mtok=9.75)}

    assert price_for(
        "azure-gpt-production",
        prices=override,
        provider="azure_openai",
    ) == ModelPrice(input_per_mtok=1.25, output_per_mtok=9.75)
    assert price_for("gpt-5.5", prices=override, provider="azure_openai") is None
    assert price_for("gpt-5.5", prices=override, provider="openai") == ModelPrice(
        input_per_mtok=5.0,
        output_per_mtok=30.0,
    )


def test_other_unaudited_inference_profile_prefixes_are_unpriced() -> None:
    assert price_for("global.anthropic.claude-sonnet-4-6") is None
    assert price_for("jp.anthropic.claude-haiku-4-5") is None


def test_us_geo_haiku_snapshot_uses_the_audited_route_price() -> None:
    assert price_for("us.anthropic.claude-haiku-4-5-20251001-v1:0") == ModelPrice(
        input_per_mtok=1.1,
        output_per_mtok=5.5,
    )


def test_unaudited_bedrock_routes_do_not_inherit_model_only_prices() -> None:
    assert (
        price_for(
            "anthropic.claude-haiku-4-5-20251001-v1:0",
            provider="bedrock",
        )
        is None
    )
    assert price_for("claude-opus-4-8", provider="bedrock") is None
    assert price_for("zai.glm-5", provider="bedrock") is None
    assert price_for("zai.glm-5") is None


def test_bedrock_override_cannot_cross_the_normalized_provider_boundary() -> None:
    direct_override = {"claude-haiku-4-5": ModelPrice(input_per_mtok=1.0, output_per_mtok=5.0)}
    assert (
        price_for(
            "anthropic.claude-haiku-4-5-20251001-v1:0",
            prices=direct_override,
            provider="bedrock",
        )
        is None
    )


def test_no_zero_price_placeholder_rows() -> None:
    # Regression: a $0 row defeats the price_for()->None "cost unavailable" contract.
    from llm_waterfall.pricing import _PRICES

    assert price_for("qwen3-coder") is None
    assert all(p.input_per_mtok > 0 for p in _PRICES.values())
