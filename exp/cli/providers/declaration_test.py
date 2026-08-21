"""Operator declaration tests for identity-only OpenAI-compatible models."""

from __future__ import annotations

from exp.cli.providers.declaration import (
    can_declare_role,
    declare_role_metadata,
    eligible_for_role,
    merge_declared_models,
    role_row_detail,
)
from exp.cli.providers.provider_picker import UNKNOWN_METADATA_LABEL, AvailableModel
from exp.cli.shared.picker_test import ScriptedConsole
from exp.common.models import (
    DiscoveredModel,
    ModelCapabilities,
    PricingSource,
    SetupRole,
    serves_role,
)


def _identity(
    model: str = "hosted-chat",
    *,
    published: DiscoveredModel | None = None,
    capabilities: ModelCapabilities | None = None,
    pricing_source: PricingSource = PricingSource.UNKNOWN,
) -> AvailableModel:
    """Build one OpenAI-compatible identity offered for role assignment.

    Args:
        model: Provider-side model ID.
        published: Optional listing metadata to confirm.
        capabilities: Optional capabilities already attached to the row.
        pricing_source: Provenance of any attached prices.

    Returns:
        A selectable setup row.
    """
    return AvailableModel(
        alias=model,
        connection="hosted",
        provider="openai-compatible",
        model=model,
        capabilities=capabilities,
        pricing_source=pricing_source,
        configured=False,
        published=published,
    )


def test_identity_only_rows_are_labeled_unknown_and_declarable() -> None:
    """An identity without proven metadata stays selectable and unnamed as capable."""
    item = _identity()

    assert role_row_detail(item) == UNKNOWN_METADATA_LABEL
    assert can_declare_role(item, SetupRole.WORLD_MODEL)
    assert eligible_for_role(item, SetupRole.WORLD_MODEL)
    assert item.detail() == f"openai-compatible, {UNKNOWN_METADATA_LABEL}"


def test_world_model_declaration_asks_only_completion_prices() -> None:
    """World-model setup persists completion prices without inventing advanced fields."""
    console = ScriptedConsole("\n1.5\n2.5\n0\n0\n")

    declared = declare_role_metadata(_identity(), SetupRole.WORLD_MODEL, console=console)

    assert declared is not None
    assert declared.pricing_source is PricingSource.CONFIGURED
    capabilities = declared.capabilities
    assert capabilities is not None
    assert capabilities.supports_completions is True
    assert capabilities.input_cost_per_million_tokens_usd == 1.5
    assert capabilities.output_cost_per_million_tokens_usd == 2.5
    assert capabilities.cached_input_cost_per_million_tokens_usd == 0
    assert capabilities.cache_write_cost_per_million_tokens_usd == 0
    assert capabilities.supports_tools is None
    assert capabilities.supports_structured_output is False
    assert capabilities.context_window_tokens is None
    assert capabilities.maximum_output_tokens is None
    assert serves_role(capabilities, SetupRole.WORLD_MODEL)
    assert not serves_role(capabilities, SetupRole.JUDGE)
    assert not serves_role(capabilities, SetupRole.ROUTER_CANDIDATE)
    assert "Supports tools" not in console.output
    assert "Context window" not in console.output


def test_judge_declaration_requires_structured_output_and_keeps_tools_unknown() -> None:
    """Judge setup confirms structured output and does not infer tool support."""
    console = ScriptedConsole("\n\n0\n0\n0\n0\n")

    declared = declare_role_metadata(_identity(), SetupRole.JUDGE, console=console)

    assert declared is not None
    assert declared.capabilities is not None
    assert declared.capabilities.supports_structured_output is True
    assert declared.capabilities.supports_tools is None
    assert serves_role(declared.capabilities, SetupRole.JUDGE)


def test_embedder_declaration_asks_only_embedding_support_and_input_price() -> None:
    """Embedder setup does not invent completion support or cache prices."""
    console = ScriptedConsole("\n0.02\n")

    declared = declare_role_metadata(_identity("hosted-embed"), SetupRole.EMBEDDER, console=console)

    assert declared is not None
    assert declared.capabilities is not None
    assert declared.capabilities.supports_embeddings is True
    assert declared.capabilities.supports_completions is None
    assert declared.capabilities.input_cost_per_million_tokens_usd == 0.02
    assert declared.capabilities.output_cost_per_million_tokens_usd is None
    assert serves_role(declared.capabilities, SetupRole.EMBEDDER)


def test_router_declaration_requires_limits_and_does_not_invent_cache_from_neighbors() -> None:
    """Router setup asks for token limits and keeps unpublished advanced fields unknown."""
    console = ScriptedConsole("\n1\n2\n3\n4\n128000\n8192\n")

    declared = declare_role_metadata(_identity(), SetupRole.ROUTER_CANDIDATE, console=console)

    assert declared is not None
    assert declared.capabilities is not None
    assert declared.capabilities.context_window_tokens == 128000
    assert declared.capabilities.maximum_output_tokens == 8192
    assert declared.capabilities.supports_tools is None
    assert serves_role(declared.capabilities, SetupRole.ROUTER_CANDIDATE)


def test_published_prices_are_confirmed_instead_of_retyped() -> None:
    """Provider-published prices are confirmed and never replaced by an inferred value."""
    published = DiscoveredModel(
        provider="openai-compatible",
        model="hosted-chat",
        supports_completions=True,
        input_cost_per_million_tokens_usd=1.0,
        output_cost_per_million_tokens_usd=2.0,
        cached_input_cost_per_million_tokens_usd=0.1,
        cache_write_cost_per_million_tokens_usd=0.2,
    )
    console = ScriptedConsole("\n\n\n\n\n")

    declared = declare_role_metadata(
        _identity(published=published),
        SetupRole.WORLD_MODEL,
        console=console,
    )

    assert declared is not None
    assert declared.capabilities is not None
    assert declared.capabilities.input_cost_per_million_tokens_usd == 1.0
    assert declared.capabilities.output_cost_per_million_tokens_usd == 2.0
    assert declared.capabilities.cached_input_cost_per_million_tokens_usd == 0.1
    assert declared.capabilities.cache_write_cost_per_million_tokens_usd == 0.2
    assert "Use published input cost" in console.output


def test_declining_a_required_capability_keeps_the_model_unassigned() -> None:
    """Refusing a required protocol feature does not fabricate a passing declaration."""
    console = ScriptedConsole("n\n")

    declared = declare_role_metadata(_identity(), SetupRole.WORLD_MODEL, console=console)

    assert declared is None
    assert "requires supports chat completions" in console.output.casefold()


def test_official_models_are_not_declarable() -> None:
    """Maintained official listings stay on the verified path, not a questionnaire."""
    item = AvailableModel(
        alias="luna",
        connection="openai",
        provider="openai",
        model="gpt-5.6-luna",
        capabilities=None,
        pricing_source=PricingSource.UNKNOWN,
        configured=False,
    )

    assert not can_declare_role(item, SetupRole.WORLD_MODEL)
    assert not eligible_for_role(item, SetupRole.WORLD_MODEL)


def test_merge_declared_models_replaces_the_same_alias() -> None:
    """Later operator declarations replace the identity-only row for that alias."""
    original = _identity()
    declared = original.__class__(
        alias=original.alias,
        connection=original.connection,
        provider=original.provider,
        model=original.model,
        capabilities=ModelCapabilities(supports_completions=True),
        pricing_source=PricingSource.CONFIGURED,
        configured=False,
    )

    merged = merge_declared_models((original,), (declared,))

    assert merged == (declared,)
