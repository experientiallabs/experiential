"""WMH adapter for the llm-waterfall model and exact-route pricing authority."""

from __future__ import annotations

from llm_waterfall import TokenUsage as WaterfallTokenUsage
from llm_waterfall.pricing import ModelPrice, price_for
from llm_waterfall.pricing import cost_usd as waterfall_cost_usd

from wmh.providers.base import TokenUsage


def cost_usd(model: str, usage: TokenUsage) -> float:
    """Price WMH usage through the shared descriptive pricing authority."""
    return waterfall_cost_usd(
        model,
        WaterfallTokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        ),
    )


__all__ = ["ModelPrice", "cost_usd", "price_for"]
