"""Per-model token pricing → USD cost.

Static prices are scoped to the provider that publishes them and keyed by a normalized model id.
Cost-driving Bedrock geographic inference-profile prefixes remain part of the route and resolve
only through an audited exact row. Prices are USD per 1M tokens; an unknown model costs 0.0 and
`price_for` returns None so callers can surface "cost unavailable" rather than silently
under-reporting. Per-call overrides are passed explicitly. There is no global mutable registry.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from pydantic import BaseModel

from llm_waterfall.types import TokenUsage

# Bedrock appends a snapshot date and/or version to the model id, e.g.
# `claude-haiku-4-5-20251001-v1:0` or `claude-opus-4-6-v1`. Strip them so the lookup key matches
# the undated table rows (`claude-haiku-4-5`). Only applied to `claude-*` ids.
_BEDROCK_SUFFIX = re.compile(r"(-\d{8})?(-v\d+)?(:\d+)?$")
_BEDROCK_GEO_PREFIXES = ("us.", "eu.", "apac.", "us-gov.", "global.", "jp.", "au.", "ca.")


class ModelPrice(BaseModel):
    """USD per 1,000,000 tokens, split by input/output."""

    input_per_mtok: float
    output_per_mtok: float


# Keyed by normalized model id (see `_normalize`). USD per 1M tokens.
#
# Completion prices verified 2026-07-01 against the live vendor pricing pages (Claude via
# platform.claude.com models overview; OpenAI GPT-5.x Standard tier, short context). Embedding
# prices are long-stable list prices; treat as approximate.
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
    "zai.glm-5": ModelPrice(input_per_mtok=1.0, output_per_mtok=3.2),
    # --- OpenAI direct (GPT-5.x Standard tier) ---
    "gpt-5.5": ModelPrice(input_per_mtok=5.0, output_per_mtok=30.0),
    "gpt-5.5-pro": ModelPrice(input_per_mtok=30.0, output_per_mtok=180.0),
    "gpt-5.4": ModelPrice(input_per_mtok=2.5, output_per_mtok=15.0),
    "gpt-5.4-mini": ModelPrice(input_per_mtok=0.75, output_per_mtok=4.5),
    "gpt-5.4-nano": ModelPrice(input_per_mtok=0.2, output_per_mtok=1.25),
    # Azure-hosted OSS deployments (qwen3-coder, agentworld, ...) are deliberately absent:
    # a $0 placeholder row would defeat the price_for()->None "cost unavailable" contract.
    # Supply their negotiated rates per Waterfall via the `prices` override.
    # --- Embeddings (output tokens are always 0 for embed calls) ---
    "text-embedding-3-small": ModelPrice(input_per_mtok=0.02, output_per_mtok=0.0),
    "text-embedding-3-large": ModelPrice(input_per_mtok=0.13, output_per_mtok=0.0),
    "amazon.titan-embed-text-v2:0": ModelPrice(input_per_mtok=0.02, output_per_mtok=0.0),
}

# Exact provider routes whose geographic inference-profile price was audited separately from the
# direct-provider row. A geographic route absent from this table is intentionally unpriced.
_ROUTE_PRICES: dict[str, ModelPrice] = {
    "us.anthropic.claude-haiku-4-5": ModelPrice(input_per_mtok=1.1, output_per_mtok=5.5),
    "us.anthropic.claude-opus-4-8": ModelPrice(input_per_mtok=5.5, output_per_mtok=27.5),
}

_OPENAI_MODEL_PREFIXES = ("gpt-", "text-embedding-")
_ANTHROPIC_MODEL_PREFIXES = ("claude-",)
_BEDROCK_MODEL_PREFIXES = ("claude-", "amazon.", "zai.")


def _provider_publishes_static_price(provider: str, key: str) -> bool:
    """Whether `key` belongs to `provider`'s built-in price namespace.

    Azure prices are intentionally absent: a deployment can have a different meter, region, or
    negotiated rate from a same-named direct OpenAI model. Callers may still provide an exact
    deployment override. Unknown providers and aws_mantle also fail closed.
    """
    if provider in {"openai", "openai_responses"}:
        return key.startswith(_OPENAI_MODEL_PREFIXES)
    if provider == "anthropic":
        return key.startswith(_ANTHROPIC_MODEL_PREFIXES)
    if provider == "bedrock":
        return key.startswith(_BEDROCK_MODEL_PREFIXES)
    return False


def _normalize(model: str) -> str:
    """Normalize a direct or non-geographic provider model id.

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


def _route_key(model: str) -> str:
    """Retain a geographic route while removing only a Claude snapshot suffix."""
    route = model.strip()
    if route.startswith(_BEDROCK_GEO_PREFIXES) and ".anthropic.claude-" in route:
        return _BEDROCK_SUFFIX.sub("", route)
    return route


def price_for(
    model: str,
    prices: Mapping[str, ModelPrice] | None = None,
    *,
    provider: str | None = None,
) -> ModelPrice | None:
    """The price row for a provider/model route, or None if unknown.

    `prices` are per-caller overrides consulted before the static table; they are never merged
    into it, so one Waterfall's overrides can't leak into another's.

    Pass `provider` whenever it is known. Omitting it retains the legacy model-only lookup for
    callers that have no provider context; internal Waterfall calls always pass it. In particular,
    Azure never inherits a same-named direct OpenAI price.
    """
    route_key = _route_key(model)
    if route_key.startswith(_BEDROCK_GEO_PREFIXES):
        if prices is not None:
            override = prices.get(model) or prices.get(route_key)
            if override is not None:
                return override
        if provider is not None and provider != "bedrock":
            return None
        return _ROUTE_PRICES.get(route_key)

    key = _normalize(route_key)
    if prices is not None:
        override = prices.get(key) or prices.get(model)
        if override is not None:
            return override
    if provider is not None and not _provider_publishes_static_price(provider, key):
        return None
    return _PRICES.get(key)


def cost_usd(
    model: str,
    usage: TokenUsage,
    prices: Mapping[str, ModelPrice] | None = None,
    *,
    provider: str | None = None,
) -> float:
    """USD cost of `usage` on a provider/model route; unknown routes cost 0.0."""
    price = price_for(model, prices, provider=provider)
    if price is None:
        return 0.0
    return (
        usage.input_tokens * price.input_per_mtok + usage.output_tokens * price.output_per_mtok
    ) / 1_000_000
