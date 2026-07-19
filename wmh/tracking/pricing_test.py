"""Tests for the token→cost pricing table."""

from __future__ import annotations

import pytest

from wmh.providers.base import TokenUsage
from wmh.tracking.pricing import cost_usd, price_for


def test_opus_4_8_cost_is_5_in_25_out_per_mtok() -> None:
    # 1M input + 1M output on Opus 4.8 = $5 + $25. (approx: float division, not exact arithmetic)
    cost = cost_usd("claude-opus-4-8", TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000))
    assert cost == pytest.approx(30.0)


def test_bedrock_us_geo_profile_uses_the_audited_route_price() -> None:
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0)
    assert cost_usd("us.anthropic.claude-opus-4-8", usage) == pytest.approx(5.5)
    assert cost_usd("claude-opus-4-8", usage) == pytest.approx(5.0)


def test_bedrock_dated_inference_profile_id_normalizes() -> None:
    # Bedrock inference-profile ids carry a snapshot date + version, but the geo route remains
    # cost-driving and must not collapse into the direct-provider row.
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0)
    assert price_for("us.anthropic.claude-haiku-4-5-20251001-v1:0") is not None
    assert cost_usd("us.anthropic.claude-haiku-4-5-20251001-v1:0", usage) == pytest.approx(1.1)
    # The `-v1` version suffix alone is also stripped.
    assert cost_usd("anthropic.claude-opus-4-6-v1", usage) == pytest.approx(5.0)


def test_unaudited_bedrock_geo_route_has_no_descriptive_price() -> None:
    assert price_for("us.anthropic.claude-sonnet-4-6") is None
    assert price_for("eu.anthropic.claude-opus-4-8") is None


def test_bedrock_glm_5_uses_the_verified_nominal_price() -> None:
    price = price_for("zai.glm-5")

    assert price is not None
    assert (price.input_per_mtok, price.output_per_mtok) == (1.0, 3.2)


def test_gpt_5_5_output_is_30_per_mtok() -> None:
    # Verified 2026-06-25 against OpenAI's live pricing page: gpt-5.5 is $5 in / $30 out.
    price = price_for("gpt-5.5", provider="openai")
    assert price is not None
    assert (price.input_per_mtok, price.output_per_mtok) == (5.0, 30.0)


def test_azure_gpt_does_not_reuse_the_direct_openai_rate() -> None:
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)

    assert cost_usd("gpt-5.5", usage, provider="openai") == pytest.approx(35.0)
    assert price_for("gpt-5.5", provider="azure") is None
    assert cost_usd("gpt-5.5", usage, provider="azure") == 0.0


def test_fable_5_is_10_in_50_out() -> None:
    price = price_for("claude-fable-5")
    assert price is not None
    assert (price.input_per_mtok, price.output_per_mtok) == (10.0, 50.0)


def test_titan_embedding_keeps_amazon_prefix() -> None:
    # `amazon.` is part of the Titan model id, not a routing prefix — it must still resolve.
    assert price_for("amazon.titan-embed-text-v2:0") is not None


def test_unknown_model_is_zero_and_flagged() -> None:
    assert price_for("totally-made-up-model") is None
    assert cost_usd("totally-made-up-model", TokenUsage(input_tokens=999, output_tokens=999)) == 0.0


def test_partial_usage_is_prorated() -> None:
    # 500k input + 200k output on Opus 4.8 = 0.5*5 + 0.2*25 = 2.5 + 5.0 = 7.5.
    cost = cost_usd("claude-opus-4-8", TokenUsage(input_tokens=500_000, output_tokens=200_000))
    assert cost == pytest.approx(7.5)
