"""Per-model token pricing → USD cost.

Provider-agnostic: prices are keyed by a normalized model id (provider prefixes like Bedrock's
`us.anthropic.` are stripped before lookup), so the same Opus 4.8 row covers the direct API and
Bedrock. Prices are USD per 1M tokens; an unknown model costs 0.0 and is flagged so callers can
surface "cost unavailable" rather than silently under-reporting.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from wmh.providers.base import TokenUsage

# Bedrock appends a snapshot date and/or version to the model id, e.g.
# `claude-haiku-4-5-20251001-v1:0` or `claude-opus-4-6-v1`. Strip them so the lookup key matches the
# undated table rows (`claude-haiku-4-5`). Only applied to `claude-*` ids.
_BEDROCK_SUFFIX = re.compile(r"(-\d{8})?(-v\d+)?(:\d+)?$")


class ModelPrice(BaseModel):
    """USD per 1,000,000 tokens, split by input/output."""

    input_per_mtok: float
    output_per_mtok: float


# Keyed by normalized model id (see `_normalize`). USD per 1M tokens.
#
# Completion prices verified 2026-06-25 against the live vendor pricing pages:
#   - Claude: platform.claude.com/docs/en/about-claude/models/overview
#   - OpenAI GPT-5.x: developers.openai.com/api/docs/pricing (Standard tier, short context)
# Embedding prices are long-stable list prices NOT re-fetched in that pass (the OpenAI pricing
# page no longer surfaces them); treat as approximate and re-verify if embed cost matters.
_PRICES: dict[str, ModelPrice] = {
    # --- Anthropic / Bedrock (Claude) ---
    "claude-fable-5": ModelPrice(input_per_mtok=10.0, output_per_mtok=50.0),
    "claude-mythos-5": ModelPrice(input_per_mtok=10.0, output_per_mtok=50.0),
    "claude-opus-4-8": ModelPrice(input_per_mtok=5.0, output_per_mtok=25.0),
    "claude-opus-4-7": ModelPrice(input_per_mtok=5.0, output_per_mtok=25.0),
    "claude-opus-4-6": ModelPrice(input_per_mtok=5.0, output_per_mtok=25.0),
    "claude-opus-4-5": ModelPrice(input_per_mtok=5.0, output_per_mtok=25.0),
    "claude-opus-4-1": ModelPrice(input_per_mtok=15.0, output_per_mtok=75.0),
    "claude-sonnet-5": ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0),
    "claude-sonnet-4-6": ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0),
    "claude-haiku-4-5": ModelPrice(input_per_mtok=1.0, output_per_mtok=5.0),
    # --- OpenAI / Azure OpenAI (GPT-5.x; Azure deployments reuse the base model's price) ---
    "gpt-5.5": ModelPrice(input_per_mtok=5.0, output_per_mtok=30.0),
    "gpt-5.5-pro": ModelPrice(input_per_mtok=30.0, output_per_mtok=180.0),
    "gpt-5.4": ModelPrice(input_per_mtok=2.5, output_per_mtok=15.0),
    "gpt-5.4-mini": ModelPrice(input_per_mtok=0.75, output_per_mtok=4.5),
    "gpt-5.4-nano": ModelPrice(input_per_mtok=0.2, output_per_mtok=1.25),
    # --- Embeddings (output tokens are always 0 for embed calls) ---
    "text-embedding-3-small": ModelPrice(input_per_mtok=0.02, output_per_mtok=0.0),
    "text-embedding-3-large": ModelPrice(input_per_mtok=0.13, output_per_mtok=0.0),
    "amazon.titan-embed-text-v2:0": ModelPrice(input_per_mtok=0.02, output_per_mtok=0.0),
}


def _normalize(model: str) -> str:
    """Strip provider/region routing prefixes so one row covers a model across providers.

    Bedrock ids look like `us.anthropic.claude-opus-4-8`; the direct API uses `claude-opus-4-8`.
    We drop a leading region segment (`us.`/`eu.`/...) and an `anthropic.` vendor segment, but keep
    `amazon.titan-...` (its `amazon.` is part of the canonical model id, not a routing prefix).
    """
    normalized = model.strip()
    region_prefixes = ("us.", "eu.", "apac.", "us-gov.", "global.", "jp.", "au.", "ca.")
    for prefix in region_prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    if normalized.startswith("anthropic."):
        normalized = normalized[len("anthropic.") :]
    if normalized.startswith("claude-"):
        # Drop a trailing Bedrock snapshot date / version (`-20251001-v1:0`, `-v1`) so dated
        # inference-profile ids match the undated table rows.
        normalized = _BEDROCK_SUFFIX.sub("", normalized)
    return normalized


def price_for(model: str) -> ModelPrice | None:
    """Return the price row for `model` (after normalization), or None if unknown."""
    return _PRICES.get(_normalize(model))


def cost_usd(model: str, usage: TokenUsage) -> float:
    """USD cost of `usage` on `model`. Unknown models cost 0.0 (see `price_for` to detect that)."""
    price = price_for(model)
    if price is None:
        return 0.0
    return (
        usage.input_tokens * price.input_per_mtok + usage.output_tokens * price.output_per_mtok
    ) / 1_000_000
