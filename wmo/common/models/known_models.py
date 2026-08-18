"""Maintained WMO capability and price metadata for published provider models.

Provider listing endpoints publish which models an authenticated account may call, but most of
them publish neither prices nor protocol capabilities. This table carries the values WMO verified
against each provider's public model and pricing documentation so setup never asks a user to copy
capability booleans, token limits, or prices by hand. A model absent from this table keeps unknown
capabilities and unknown prices, which fails closed everywhere those values are required.

Prices are USD per million tokens on each provider's standard synchronous tier. A cache price of
zero means the provider documents no separate charge for that cache operation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SNAPSHOT_SUFFIX_PATTERN = re.compile(r"(?:[-@]\d{8}|-latest)$")
_GEMINI_PREFIX = "models/"


@dataclass(frozen=True)
class KnownModel:
    """Verified capabilities, token limits, and prices for one published provider model."""

    supports_completions: bool = False
    supports_embeddings: bool = False
    supports_tools: bool = False
    supports_structured_output: bool = False
    supports_temperature: bool | None = None
    supports_reasoning_effort: bool = False
    context_window_tokens: int | None = None
    maximum_output_tokens: int | None = None
    input_cost_per_million_tokens_usd: float | None = None
    output_cost_per_million_tokens_usd: float | None = None
    cached_input_cost_per_million_tokens_usd: float | None = None
    cache_write_cost_per_million_tokens_usd: float | None = None


def _chat(
    *,
    input_usd: float,
    output_usd: float,
    cached_input_usd: float | None = None,
    cache_write_usd: float | None = None,
    context_window_tokens: int | None = None,
    maximum_output_tokens: int | None = None,
    supports_temperature: bool | None = True,
    supports_structured_output: bool = True,
    supports_reasoning_effort: bool = False,
) -> KnownModel:
    """Describe one documented chat model with its verified protocol capabilities.

    Args:
        input_usd: Documented input price per million tokens.
        output_usd: Documented output price per million tokens.
        cached_input_usd: Documented cached-input price, when the provider publishes one.
        cache_write_usd: Documented cache-write price, when the provider publishes one.
        context_window_tokens: Documented context window, when the provider publishes one.
        maximum_output_tokens: Documented output ceiling, when the provider publishes one.
        supports_temperature: Whether the model accepts an explicit temperature parameter.
        supports_structured_output: Whether the model supports structured outputs.
        supports_reasoning_effort: Whether the model accepts an explicit reasoning-effort
            parameter on the OpenAI Responses API.

    Returns:
        The verified metadata record for the model.
    """
    return KnownModel(
        supports_completions=True,
        supports_tools=True,
        supports_structured_output=supports_structured_output,
        supports_temperature=supports_temperature,
        supports_reasoning_effort=supports_reasoning_effort,
        context_window_tokens=context_window_tokens,
        maximum_output_tokens=maximum_output_tokens,
        input_cost_per_million_tokens_usd=input_usd,
        output_cost_per_million_tokens_usd=output_usd,
        cached_input_cost_per_million_tokens_usd=cached_input_usd,
        cache_write_cost_per_million_tokens_usd=cache_write_usd,
    )


def _embedding(*, input_usd: float, context_window_tokens: int | None = None) -> KnownModel:
    """Describe one documented embedding model that cannot serve completions.

    Args:
        input_usd: Documented input price per million tokens.
        context_window_tokens: Documented input token ceiling, when published.

    Returns:
        The verified metadata record for the model.
    """
    return KnownModel(
        supports_completions=False,
        supports_embeddings=True,
        context_window_tokens=context_window_tokens,
        input_cost_per_million_tokens_usd=input_usd,
    )


_OPENAI_MODELS: dict[str, KnownModel] = {
    "gpt-5.6-sol": _chat(
        input_usd=5.0,
        cached_input_usd=0.5,
        cache_write_usd=6.25,
        output_usd=30.0,
        context_window_tokens=1_050_000,
        maximum_output_tokens=128_000,
        supports_temperature=False,
        supports_reasoning_effort=True,
    ),
    "gpt-5.6-terra": _chat(
        input_usd=2.0,
        cached_input_usd=0.2,
        cache_write_usd=2.5,
        output_usd=12.0,
        context_window_tokens=1_050_000,
        maximum_output_tokens=128_000,
        supports_temperature=False,
        supports_reasoning_effort=True,
    ),
    "gpt-5.6-luna": _chat(
        input_usd=0.2,
        cached_input_usd=0.02,
        cache_write_usd=0.25,
        output_usd=1.2,
        context_window_tokens=1_050_000,
        maximum_output_tokens=128_000,
        supports_temperature=False,
        supports_reasoning_effort=True,
    ),
    "gpt-5.5": _chat(
        input_usd=5.0,
        cached_input_usd=0.5,
        cache_write_usd=0.0,
        output_usd=30.0,
        context_window_tokens=1_050_000,
        maximum_output_tokens=128_000,
        supports_temperature=False,
        supports_reasoning_effort=True,
    ),
    "gpt-5.5-pro": _chat(
        input_usd=30.0,
        cache_write_usd=0.0,
        output_usd=180.0,
        context_window_tokens=1_050_000,
        maximum_output_tokens=128_000,
        supports_temperature=False,
        supports_reasoning_effort=True,
    ),
    "gpt-5.4": _chat(
        input_usd=2.5,
        cached_input_usd=0.25,
        cache_write_usd=0.0,
        output_usd=15.0,
        context_window_tokens=1_050_000,
        maximum_output_tokens=128_000,
        supports_temperature=False,
        supports_reasoning_effort=True,
    ),
    "gpt-5.4-mini": _chat(
        input_usd=0.75,
        cached_input_usd=0.075,
        cache_write_usd=0.0,
        output_usd=4.5,
        context_window_tokens=400_000,
        maximum_output_tokens=128_000,
        supports_temperature=False,
        supports_reasoning_effort=True,
    ),
    "gpt-5.4-nano": _chat(
        input_usd=0.2,
        cached_input_usd=0.02,
        cache_write_usd=0.0,
        output_usd=1.25,
        context_window_tokens=400_000,
        maximum_output_tokens=128_000,
        supports_temperature=False,
        supports_reasoning_effort=True,
    ),
    "gpt-5.4-pro": _chat(
        input_usd=30.0,
        cache_write_usd=0.0,
        output_usd=180.0,
        supports_structured_output=False,
        context_window_tokens=1_050_000,
        maximum_output_tokens=128_000,
        supports_temperature=False,
        supports_reasoning_effort=True,
    ),
    "gpt-5.2": _chat(
        input_usd=1.75,
        cached_input_usd=0.175,
        cache_write_usd=0.0,
        output_usd=14.0,
        context_window_tokens=400_000,
        maximum_output_tokens=128_000,
        supports_temperature=False,
        supports_reasoning_effort=True,
    ),
    "gpt-5.2-pro": _chat(
        input_usd=21.0,
        cache_write_usd=0.0,
        output_usd=168.0,
        supports_structured_output=False,
        context_window_tokens=400_000,
        maximum_output_tokens=128_000,
        supports_temperature=False,
        supports_reasoning_effort=True,
    ),
    "gpt-5.1": _chat(
        input_usd=1.25,
        cached_input_usd=0.125,
        cache_write_usd=0.0,
        output_usd=10.0,
        context_window_tokens=400_000,
        maximum_output_tokens=128_000,
        supports_temperature=False,
        supports_reasoning_effort=True,
    ),
    "gpt-5": _chat(
        input_usd=1.25,
        cached_input_usd=0.125,
        cache_write_usd=0.0,
        output_usd=10.0,
        context_window_tokens=400_000,
        maximum_output_tokens=128_000,
        supports_temperature=False,
        supports_reasoning_effort=True,
    ),
    "gpt-5-mini": _chat(
        input_usd=0.25,
        cached_input_usd=0.025,
        cache_write_usd=0.0,
        output_usd=2.0,
        context_window_tokens=400_000,
        maximum_output_tokens=128_000,
        supports_temperature=False,
        supports_reasoning_effort=True,
    ),
    "gpt-5-nano": _chat(
        input_usd=0.05,
        cached_input_usd=0.005,
        cache_write_usd=0.0,
        output_usd=0.4,
        context_window_tokens=400_000,
        maximum_output_tokens=128_000,
        supports_temperature=False,
        supports_reasoning_effort=True,
    ),
    "gpt-5-pro": _chat(
        input_usd=15.0,
        cache_write_usd=0.0,
        output_usd=120.0,
        context_window_tokens=400_000,
        maximum_output_tokens=272_000,
        supports_temperature=False,
        supports_reasoning_effort=True,
    ),
    "text-embedding-3-small": _embedding(input_usd=0.02, context_window_tokens=8_192),
    "text-embedding-3-large": _embedding(input_usd=0.13, context_window_tokens=8_192),
    "text-embedding-ada-002": _embedding(input_usd=0.1, context_window_tokens=8_192),
}

_ANTHROPIC_MODELS: dict[str, KnownModel] = {
    "claude-fable-5": _chat(
        input_usd=10.0,
        cached_input_usd=1.0,
        cache_write_usd=12.5,
        output_usd=50.0,
        context_window_tokens=1_000_000,
        maximum_output_tokens=128_000,
    ),
    "claude-mythos-5": _chat(
        input_usd=10.0,
        cached_input_usd=1.0,
        cache_write_usd=12.5,
        output_usd=50.0,
        context_window_tokens=1_000_000,
        maximum_output_tokens=128_000,
    ),
    "claude-opus-5": _chat(
        input_usd=5.0,
        cached_input_usd=0.5,
        cache_write_usd=6.25,
        output_usd=25.0,
        context_window_tokens=1_000_000,
        maximum_output_tokens=128_000,
    ),
    "claude-sonnet-5": _chat(
        input_usd=2.0,
        cached_input_usd=0.2,
        cache_write_usd=2.5,
        output_usd=10.0,
        context_window_tokens=1_000_000,
        maximum_output_tokens=128_000,
    ),
    "claude-haiku-4-5": _chat(
        input_usd=1.0,
        cached_input_usd=0.1,
        cache_write_usd=1.25,
        output_usd=5.0,
        context_window_tokens=200_000,
        maximum_output_tokens=64_000,
    ),
    "claude-opus-4-8": _chat(
        input_usd=5.0, cached_input_usd=0.5, cache_write_usd=6.25, output_usd=25.0
    ),
    "claude-opus-4-7": _chat(
        input_usd=5.0, cached_input_usd=0.5, cache_write_usd=6.25, output_usd=25.0
    ),
    "claude-opus-4-6": _chat(
        input_usd=5.0, cached_input_usd=0.5, cache_write_usd=6.25, output_usd=25.0
    ),
    "claude-opus-4-5": _chat(
        input_usd=5.0, cached_input_usd=0.5, cache_write_usd=6.25, output_usd=25.0
    ),
    "claude-sonnet-4-6": _chat(
        input_usd=3.0, cached_input_usd=0.3, cache_write_usd=3.75, output_usd=15.0
    ),
    "claude-sonnet-4-5": _chat(
        input_usd=3.0, cached_input_usd=0.3, cache_write_usd=3.75, output_usd=15.0
    ),
}

_GEMINI_MODELS: dict[str, KnownModel] = {
    "gemini-3.6-flash": _chat(input_usd=1.5, cached_input_usd=0.15, output_usd=7.5),
    "gemini-3.5-flash": _chat(input_usd=1.5, cached_input_usd=0.15, output_usd=9.0),
    "gemini-3.5-flash-lite": _chat(input_usd=0.3, cached_input_usd=0.03, output_usd=2.5),
    "gemini-embedding-001": _embedding(input_usd=0.15),
}

_KNOWN_MODELS: dict[str, dict[str, KnownModel]] = {
    "anthropic": _ANTHROPIC_MODELS,
    "gemini": _GEMINI_MODELS,
    "openai": _OPENAI_MODELS,
}


def canonical_model_id(provider: str, model: str) -> str:
    """Normalize one provider model ID to the identity this table indexes.

    Providers publish dated snapshot IDs and pointer aliases beside the base model ID. Both name
    the same documented model and price, so the dated or pointer suffix and Gemini's resource
    prefix are removed before lookup.

    Args:
        provider: Setup provider kind such as ``openai`` or ``anthropic``.
        model: Provider-published model ID.

    Returns:
        The normalized lookup identity.
    """
    identity = model.strip().casefold()
    if provider == "gemini" and identity.startswith(_GEMINI_PREFIX):
        identity = identity[len(_GEMINI_PREFIX) :]
    return _SNAPSHOT_SUFFIX_PATTERN.sub("", identity)


def known_model_metadata(provider: str, model: str) -> KnownModel | None:
    """Look up verified metadata for one provider model.

    Args:
        provider: Setup provider kind such as ``openai`` or ``anthropic``.
        model: Provider-published model ID.

    Returns:
        The verified record, or ``None`` when WMO has not verified this model.
    """
    models = _KNOWN_MODELS.get(provider)
    if models is None:
        return None
    return models.get(canonical_model_id(provider, model))
