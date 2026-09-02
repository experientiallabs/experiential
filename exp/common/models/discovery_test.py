"""Model discovery merge, role, and identity tests."""

from __future__ import annotations

from exp.common.models.discovery import (
    DiscoveredModel,
    PricingSource,
    SetupRole,
    derive_connection_name,
    derive_model_alias,
    resolve_discovered_model,
    served_roles,
    serves_role,
)


def test_provider_published_metadata_wins_over_maintained_metadata() -> None:
    """A provider describes the authenticated endpoint, so its published values are kept."""
    resolved = resolve_discovered_model(
        DiscoveredModel(
            provider="openai",
            model="gpt-5.6-sol",
            context_window_tokens=64_000,
            input_cost_per_million_tokens_usd=4.0,
            output_cost_per_million_tokens_usd=20.0,
        )
    )

    assert resolved.capabilities.context_window_tokens == 64_000
    assert resolved.capabilities.input_cost_per_million_tokens_usd == 4.0
    assert resolved.capabilities.output_cost_per_million_tokens_usd == 20.0
    assert resolved.pricing_source is PricingSource.PROVIDER


def test_maintained_metadata_fills_values_the_provider_omits() -> None:
    """Providers that publish only identities still yield verified capabilities and prices."""
    resolved = resolve_discovered_model(
        DiscoveredModel(provider="anthropic", model="claude-sonnet-5-20260101")
    )

    capabilities = resolved.capabilities
    assert capabilities.supports_completions
    assert capabilities.supports_tools
    assert capabilities.supports_structured_output
    assert capabilities.input_cost_per_million_tokens_usd == 2.0
    assert capabilities.cache_write_cost_per_million_tokens_usd == 2.5
    assert resolved.pricing_source is PricingSource.EXP_CATALOG


def test_maintained_sampling_pins_flow_into_resolved_capabilities() -> None:
    """A reasoning model's pinned sampling and effort reach the persisted catalog snapshot."""
    pinned = resolve_discovered_model(DiscoveredModel(provider="openai", model="gpt-5.6-luna"))
    unpinned = resolve_discovered_model(
        DiscoveredModel(provider="anthropic", model="claude-haiku-4-5")
    )

    assert not pinned.capabilities.supports_temperature
    assert pinned.capabilities.supports_reasoning
    assert pinned.capabilities.reasoning_effort == "medium"
    assert unpinned.capabilities.supports_temperature
    assert unpinned.capabilities.reasoning_effort is None


def test_unknown_model_keeps_every_capability_and_price_unknown() -> None:
    """An unverified model claims nothing and can serve no build role."""
    resolved = resolve_discovered_model(
        DiscoveredModel(provider="openai", model="internal-preview-model")
    )

    capabilities = resolved.capabilities
    assert not capabilities.supports_completions
    assert not capabilities.supports_tools
    assert not capabilities.supports_embeddings
    assert not capabilities.supports_structured_output
    assert capabilities.context_window_tokens is None
    assert capabilities.input_cost_per_million_tokens_usd is None
    assert resolved.pricing_source is PricingSource.UNKNOWN
    assert served_roles(resolved.capabilities) == ()


def test_cache_prices_are_never_inferred_from_the_base_input_price() -> None:
    """Absent cache prices stay absent, so a model without them is no router candidate."""
    resolved = resolve_discovered_model(
        DiscoveredModel(
            provider="openrouter",
            model="vendor/model",
            supports_completions=True,
            supports_structured_output=True,
            context_window_tokens=128_000,
            maximum_output_tokens=8_192,
            input_cost_per_million_tokens_usd=1.0,
            output_cost_per_million_tokens_usd=3.0,
        )
    )

    capabilities = resolved.capabilities
    assert capabilities.cached_input_cost_per_million_tokens_usd is None
    assert capabilities.cache_write_cost_per_million_tokens_usd is None
    assert not serves_role(resolved.capabilities, SetupRole.ROUTER_CANDIDATE)
    assert serves_role(resolved.capabilities, SetupRole.WORLD_MODEL)
    assert serves_role(resolved.capabilities, SetupRole.JUDGE)


def test_provider_denial_overrides_maintained_capability() -> None:
    """An explicit provider ``false`` is verified metadata and is not overridden."""
    resolved = resolve_discovered_model(
        DiscoveredModel(
            provider="openai",
            model="gpt-5.6-sol",
            supports_structured_output=False,
        )
    )

    assert not resolved.capabilities.supports_structured_output
    assert not serves_role(resolved.capabilities, SetupRole.JUDGE)
    assert serves_role(resolved.capabilities, SetupRole.WORLD_MODEL)


def test_router_candidate_role_requires_prices_and_both_token_limits() -> None:
    """Router candidates need complete price and limit evidence before they are offered."""
    resolved = resolve_discovered_model(DiscoveredModel(provider="openai", model="gpt-5.6-luna"))

    assert served_roles(resolved.capabilities) == (
        SetupRole.WORLD_MODEL,
        SetupRole.JUDGE,
        SetupRole.ROUTER_CANDIDATE,
    )


def test_catalog_request_shaping_capability_reaches_openai_runtime_metadata() -> None:
    """OpenAI reasoning models resolve with temperature disabled for request shaping."""
    resolved = resolve_discovered_model(DiscoveredModel(provider="openai", model="gpt-5.4-mini"))

    assert not resolved.capabilities.supports_temperature


def test_reasoning_capable_models_resolve_with_the_default_medium_pin() -> None:
    """Models proven to accept reasoning effort default to medium; others stay unpinned."""
    reasoning = resolve_discovered_model(DiscoveredModel(provider="openai", model="gpt-5.6-luna"))
    plain = resolve_discovered_model(
        DiscoveredModel(provider="anthropic", model="claude-haiku-4-5")
    )
    embedding = resolve_discovered_model(
        DiscoveredModel(provider="openai", model="text-embedding-3-small")
    )

    assert reasoning.capabilities.reasoning_effort == "medium"
    assert reasoning.capabilities.supports_reasoning
    assert plain.capabilities.reasoning_effort is None
    assert not plain.capabilities.supports_reasoning
    assert embedding.capabilities.reasoning_effort is None


def test_discovery_preserves_complete_generation_parameter_contracts() -> None:
    """Maintained and live metadata reach the persisted standalone catalog losslessly."""
    gpt_51 = resolve_discovered_model(DiscoveredModel(provider="openai", model="gpt-5.1"))
    claude = resolve_discovered_model(DiscoveredModel(provider="anthropic", model="claude-opus-5"))
    gemini = resolve_discovered_model(
        DiscoveredModel(
            provider="gemini",
            model="gemini-3.6-flash",
            maximum_temperature=1.5,
        )
    )

    assert gpt_51.capabilities.supports_temperature
    assert gpt_51.capabilities.supports_top_p
    assert gpt_51.capabilities.sampling_requires_reasoning_none
    assert gpt_51.capabilities.reasoning_effort == "medium"
    assert claude.capabilities.supports_reasoning
    assert claude.capabilities.minimum_temperature == 1.0
    assert claude.capabilities.maximum_temperature == 1.0
    assert claude.capabilities.minimum_top_p == 0.99
    assert claude.capabilities.supports_top_k is False
    assert gemini.capabilities.supports_top_k
    assert gemini.capabilities.maximum_temperature == 1.5
    assert gemini.capabilities.maximum_output_tokens == 65_536

    pro = resolve_discovered_model(DiscoveredModel(provider="openai", model="gpt-5-pro"))
    assert pro.capabilities.reasoning_effort == "high"


def test_openai_pro_models_without_structured_output_are_not_offered_as_judges() -> None:
    """Unsupported structured-output roles stay hidden while world-model use remains available."""
    for model in ("gpt-5.4-pro", "gpt-5.2-pro"):
        resolved = resolve_discovered_model(DiscoveredModel(provider="openai", model=model))

        assert not serves_role(resolved.capabilities, SetupRole.JUDGE)
        assert served_roles(resolved.capabilities) == (SetupRole.WORLD_MODEL,)


def test_embedder_role_requires_priced_embedding_support() -> None:
    """Only a priced embedding model is offered for the embedder role."""
    priced = resolve_discovered_model(
        DiscoveredModel(provider="openai", model="text-embedding-3-large")
    )
    unpriced = resolve_discovered_model(
        DiscoveredModel(provider="gemini", model="experimental-embedding", supports_embeddings=True)
    )

    assert served_roles(priced.capabilities) == (SetupRole.EMBEDDER,)
    assert served_roles(unpriced.capabilities) == ()


def test_connection_names_and_aliases_are_readable_and_collision_safe() -> None:
    """Setup derives stable names, then the smallest free ordinal on collision."""
    assert derive_connection_name("openai", frozenset()) == "openai"
    assert derive_connection_name("openai", frozenset({"openai"})) == "openai-2"
    assert (
        derive_connection_name("openai-compatible", frozenset({"openai-compatible"}))
        == "openai-compatible-2"
    )
    assert derive_model_alias("openai", "gpt-5.6-sol-20260216", frozenset()) == "gpt-5-6-sol"
    assert (
        derive_model_alias("openrouter", "vendor/model", frozenset({"vendor-model"}))
        == "vendor-model-2"
    )
    assert (
        derive_model_alias("gemini", "models/gemini-3.5-flash", frozenset()) == "gemini-3-5-flash"
    )


def test_anthropic_point_release_resolves_a_reasoning_route_without_prices() -> None:
    """A freshly launched point release must not degrade to a non-reasoning route.

    claude-fable-5-1 reached production as a non-reasoning route (every
    effort-carrying Claude Code request got a named 400) because the bare
    listing entry resolved no metadata. With the explicit 5.1 record the
    route carries its full verified contract; a future unrecorded minor
    (sonnet-5-2 here) still inherits the generation's controls while its
    prices stay unknown so priced lanes fail closed.
    """
    recorded = resolve_discovered_model(
        DiscoveredModel(provider="anthropic", model="claude-fable-5-1")
    )
    assert recorded.capabilities.supports_reasoning is True
    assert recorded.capabilities.supports_top_k is False
    assert recorded.capabilities.input_cost_per_million_tokens_usd == 10.0
    assert recorded.capabilities.cached_input_cost_per_million_tokens_usd == 0.25

    inherited = resolve_discovered_model(
        DiscoveredModel(provider="anthropic", model="claude-sonnet-5-2")
    )
    assert inherited.capabilities.supports_reasoning is True
    assert inherited.capabilities.input_cost_per_million_tokens_usd is None
