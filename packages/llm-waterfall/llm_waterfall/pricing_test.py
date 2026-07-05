"""Tests for model-id normalization and USD cost computation."""

from __future__ import annotations

from llm_waterfall.pricing import ModelPrice, cost_usd, price_for
from llm_waterfall.types import TokenUsage


def test_bedrock_id_normalizes_to_table_row() -> None:
    dated = price_for("us.anthropic.claude-opus-4-8-20260101-v1:0")
    bare = price_for("claude-opus-4-8")
    assert dated is not None and dated == bare


def test_region_and_vendor_prefixes_stripped() -> None:
    assert price_for("eu.anthropic.claude-sonnet-4-6") == price_for("claude-sonnet-4-6")
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
    assert cost_usd("my-azure-deployment", usage, prices=override) == 5.0


def test_global_inference_profile_prefix_stripped() -> None:
    # Regression: global./jp./au. cross-region profiles must price like their us. siblings.
    assert price_for("global.anthropic.claude-sonnet-4-6") == price_for("claude-sonnet-4-6")
    assert price_for("jp.anthropic.claude-haiku-4-5") == price_for("claude-haiku-4-5")


def test_no_zero_price_placeholder_rows() -> None:
    # Regression: a $0 row defeats the price_for()->None "cost unavailable" contract.
    from llm_waterfall.pricing import _PRICES

    assert price_for("qwen3-coder") is None
    assert all(p.input_per_mtok > 0 for p in _PRICES.values())
