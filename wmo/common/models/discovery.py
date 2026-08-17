"""Provider-neutral model discovery contracts used by first-run provider setup.

A provider listing endpoint reports which models an authenticated account may call, and some
providers also publish capabilities, token limits, and prices there. This module merges that
response with WMO's maintained metadata into one resolved record per model, decides which build
roles the record can serve, and derives the stable connection names and readable aliases that setup
writes to ``.wmo/models.toml``. Every value stays unknown unless a provider or the maintained table
proves it, so an unverified model is hidden instead of being claimed as capable.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import Field

from wmo.common.core.artifacts import ContractModel
from wmo.common.models.known_models import KnownModel, canonical_model_id, known_model_metadata
from wmo.common.models.model import ModelCapabilities

_ALIAS_SEPARATOR_PATTERN = re.compile(r"[^a-z0-9]+")
_MAXIMUM_ALIAS_LENGTH = 128


class SetupRole(StrEnum):
    """Build roles a discovered model can be assigned during provider setup."""

    WORLD_MODEL = "world_model"
    JUDGE = "judge"
    EMBEDDER = "embedder"
    ROUTER_CANDIDATE = "router_candidate"


class PricingSource(StrEnum):
    """Where the resolved prices for one model came from."""

    PROVIDER = "provider"
    WMO_CATALOG = "wmo-catalog"
    CONFIGURED = "configured"
    UNKNOWN = "unknown"


class DiscoveredModel(ContractModel):
    """One model an authenticated provider account may call.

    Every optional field carries provider-published metadata. An omitted field means the provider
    published nothing for it, never that the capability, limit, or price is absent.
    """

    provider: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=512)
    supports_completions: bool | None = None
    supports_embeddings: bool | None = None
    supports_tools: bool | None = None
    supports_structured_output: bool | None = None
    context_window_tokens: int | None = Field(default=None, gt=0)
    maximum_output_tokens: int | None = Field(default=None, gt=0)
    input_cost_per_million_tokens_usd: float | None = Field(default=None, ge=0)
    output_cost_per_million_tokens_usd: float | None = Field(default=None, ge=0)
    cached_input_cost_per_million_tokens_usd: float | None = Field(default=None, ge=0)
    cache_write_cost_per_million_tokens_usd: float | None = Field(default=None, ge=0)


class ResolvedDiscoveredModel(ContractModel):
    """One discovered model with merged capabilities and its price provenance."""

    provider: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=512)
    capabilities: ModelCapabilities
    pricing_source: PricingSource


def serves_role(capabilities: ModelCapabilities, role: SetupRole) -> bool:
    """Report whether one capability snapshot proves a model can serve a build role.

    Args:
        capabilities: Verified capability, limit, and price snapshot.
        role: Build role the model would be assigned.

    Returns:
        ``True`` only when every value the role needs is proven.
    """
    if role is SetupRole.EMBEDDER:
        return (
            capabilities.supports_embeddings
            and capabilities.input_cost_per_million_tokens_usd is not None
        )
    if not _priced_completion_model(capabilities):
        return False
    if role is SetupRole.WORLD_MODEL:
        return True
    if role is SetupRole.JUDGE:
        return capabilities.supports_structured_output
    return (
        capabilities.cached_input_cost_per_million_tokens_usd is not None
        and capabilities.cache_write_cost_per_million_tokens_usd is not None
        and capabilities.context_window_tokens is not None
        and capabilities.maximum_output_tokens is not None
    )


def served_roles(capabilities: ModelCapabilities) -> tuple[SetupRole, ...]:
    """List every build role one capability snapshot can serve.

    Args:
        capabilities: Verified capability, limit, and price snapshot.

    Returns:
        Declaration-ordered roles the snapshot proves.
    """
    return tuple(role for role in SetupRole if serves_role(capabilities, role))


def resolve_discovered_model(discovered: DiscoveredModel) -> ResolvedDiscoveredModel:
    """Merge provider-published metadata with WMO's maintained model metadata.

    Provider-published values win, because they describe the authenticated account's actual
    endpoint. WMO's maintained table fills the rest. Every capability, limit, and price stays
    unknown unless one of the two sources states it, so no value is ever estimated or inferred from
    a neighboring value.

    Args:
        discovered: One model reported by a provider listing endpoint.

    Returns:
        The merged record with resolved capabilities and price provenance.
    """
    known = known_model_metadata(discovered.provider, discovered.model)
    input_cost = _first_price(
        discovered.input_cost_per_million_tokens_usd,
        known.input_cost_per_million_tokens_usd if known else None,
    )
    output_cost = _first_price(
        discovered.output_cost_per_million_tokens_usd,
        known.output_cost_per_million_tokens_usd if known else None,
    )
    cached_input_cost = _first_price(
        discovered.cached_input_cost_per_million_tokens_usd,
        known.cached_input_cost_per_million_tokens_usd if known else None,
    )
    cache_write_cost = _first_price(
        discovered.cache_write_cost_per_million_tokens_usd,
        known.cache_write_cost_per_million_tokens_usd if known else None,
    )
    capabilities = ModelCapabilities(
        supports_tools=_proven(
            discovered.supports_tools,
            known.supports_tools if known else None,
        ),
        supports_embeddings=_proven(
            discovered.supports_embeddings,
            known.supports_embeddings if known else None,
        ),
        supports_structured_output=_proven(
            discovered.supports_structured_output,
            known.supports_structured_output if known else None,
        ),
        supports_completions=_proven(
            discovered.supports_completions,
            known.supports_completions if known else None,
        ),
        supports_temperature=(
            known.supports_temperature
            if known is not None and known.supports_temperature is not None
            else True
        ),
        reasoning_effort=known.reasoning_effort if known is not None else None,
        context_window_tokens=discovered.context_window_tokens
        or (known.context_window_tokens if known else None),
        maximum_output_tokens=discovered.maximum_output_tokens
        or (known.maximum_output_tokens if known else None),
        input_cost_per_million_tokens_usd=input_cost,
        output_cost_per_million_tokens_usd=output_cost,
        cached_input_cost_per_million_tokens_usd=cached_input_cost,
        cache_write_cost_per_million_tokens_usd=cache_write_cost,
    )
    return ResolvedDiscoveredModel(
        provider=discovered.provider,
        model=discovered.model,
        capabilities=capabilities,
        pricing_source=_pricing_source(discovered, known=known, input_cost=input_cost),
    )


def derive_connection_name(provider: str, taken: frozenset[str]) -> str:
    """Derive one stable connection name for a provider without asking the user.

    Args:
        provider: Setup provider kind such as ``openai`` or ``openai-compatible``.
        taken: Connection names already present in the catalog or this setup session.

    Returns:
        The provider-derived name, suffixed with the smallest free ordinal on collision.
    """
    return _unique_identity(_identity_text(provider), taken=taken)


def derive_model_alias(provider: str, model: str, taken: frozenset[str]) -> str:
    """Derive one readable alias for a provider model without asking the user.

    Args:
        provider: Setup provider kind that published the model.
        model: Provider-published model ID.
        taken: Aliases already present in the catalog or this setup session.

    Returns:
        The readable alias, suffixed with the smallest free ordinal on collision.
    """
    return _unique_identity(_identity_text(canonical_model_id(provider, model)), taken=taken)


def _identity_text(value: str) -> str:
    """Reduce one provider or model name to a catalog-safe identity stem."""
    identity = _ALIAS_SEPARATOR_PATTERN.sub("-", value.strip().casefold()).strip("-")
    if not identity or not identity[0].isalpha():
        identity = f"model-{identity}" if identity else "model"
    return identity[:_MAXIMUM_ALIAS_LENGTH].rstrip("-")


def _unique_identity(identity: str, *, taken: frozenset[str]) -> str:
    """Return the identity itself, or the first free numbered variant of it."""
    if identity not in taken:
        return identity
    ordinal = 2
    while f"{identity}-{ordinal}" in taken:
        ordinal += 1
    return f"{identity}-{ordinal}"


def _priced_completion_model(capabilities: ModelCapabilities) -> bool:
    """Report whether a model is a completion model with both base prices proven."""
    return bool(
        capabilities.supports_completions
        and capabilities.input_cost_per_million_tokens_usd is not None
        and capabilities.output_cost_per_million_tokens_usd is not None
    )


def _first_price(published: float | None, known: float | None) -> float | None:
    """Prefer the provider-published price over WMO's maintained price."""
    return published if published is not None else known


def _proven(published: bool | None, known: bool | None) -> bool:
    """Report a capability as supported only when a provider or WMO proves it."""
    if published is not None:
        return published
    return known is True


def _pricing_source(
    discovered: DiscoveredModel,
    *,
    known: KnownModel | None,
    input_cost: float | None,
) -> PricingSource:
    """Classify where the resolved prices for one model came from."""
    if input_cost is None:
        return PricingSource.UNKNOWN
    if discovered.input_cost_per_million_tokens_usd is not None:
        return PricingSource.PROVIDER
    return PricingSource.WMO_CATALOG if known is not None else PricingSource.UNKNOWN
