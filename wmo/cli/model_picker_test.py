"""Model selection, role assignment, and summary screen tests."""

from __future__ import annotations

import pytest

from wmo.cli.model_picker import (
    assign_roles,
    build_result,
    configured_models,
    declare_model,
    render_summary,
    select_models,
)
from wmo.cli.picker_test import ScriptedConsole
from wmo.cli.provider_picker import (
    AvailableModel,
    PreparedEndpoint,
    SetupCancelled,
    SetupRoleInputs,
    SetupSession,
)
from wmo.common.models import (
    ConnectionConfig,
    DiscoveredModel,
    ModelCapabilities,
    ModelRecord,
    PricingSource,
    ProviderConnection,
    ProviderModelSelection,
    resolve_discovered_model,
)

_OPENAI = ProviderConnection(name="openai", provider="openai", api_key_env="OPENAI_API_KEY")


def _endpoint(connection: ProviderConnection = _OPENAI) -> PreparedEndpoint:
    """Build one prepared endpoint whose credential never leaves the session.

    Args:
        connection: Connection the endpoint describes.

    Returns:
        The prepared endpoint with a session-only credential.
    """
    return PreparedEndpoint(connection=connection, api_key="secret-key", configured=False)


def _model(model: str, *, alias: str | None = None, provider: str = "openai") -> AvailableModel:
    """Build one configurable model from maintained provider metadata.

    Args:
        model: Provider-published model ID.
        alias: Explicit alias, defaulting to the model ID.
        provider: Provider kind publishing the model.

    Returns:
        The configurable model as discovery would resolve it.
    """
    resolved = resolve_discovered_model(DiscoveredModel(provider=provider, model=model))
    return AvailableModel(
        alias=alias or model,
        connection="openai",
        provider=provider,
        model=model,
        capabilities=resolved.capabilities,
        pricing_source=resolved.pricing_source,
        configured=False,
    )


_CHAT = _model("gpt-5.6-luna", alias="luna")
_OTHER_CHAT = _model("gpt-5.6-terra", alias="terra")
_EMBEDDER = _model("text-embedding-3-small", alias="embedder")


def _session(*models: AvailableModel, advanced_models: bool = False) -> SetupSession:
    """Build one setup session whose providers are already prepared.

    Args:
        models: Configurable models discovery produced.
        advanced_models: Whether the manual declaration row is offered.

    Returns:
        The session the model screens operate on.
    """
    return SetupSession(
        providers=("openai",),
        advanced_models=advanced_models,
        endpoints=(_endpoint(),),
        available=models,
    )


def test_model_screen_spans_providers_and_reports_roles_and_pricing_source() -> None:
    """One searchable screen lists every provider's models with their verified roles."""
    console = ScriptedConsole("1,3\n\n")
    session = _session(_CHAT, _OTHER_CHAT, _EMBEDDER)

    selected = select_models(session, console=console)

    assert selected == ("luna", "embedder")
    assert "luna (openai/gpt-5.6-luna)" in console.output
    assert "pricing: wmo-catalog" in console.output


def test_model_screen_back_navigation_returns_to_provider_selection() -> None:
    """Going back from the model screen reselects providers without cancelling."""
    assert select_models(_session(_CHAT), console=ScriptedConsole("b\n")) is None


def test_model_screen_cancellation_ends_setup() -> None:
    """Cancelling the model screen cancels the whole session."""
    with pytest.raises(SetupCancelled):
        select_models(_session(_CHAT), console=ScriptedConsole("q\n"))


def test_roles_are_filtered_to_the_selected_models_that_can_serve_them() -> None:
    """Each role screen offers only selected models whose metadata proves the role."""
    console = ScriptedConsole("1\n1\n1\n\n")

    roles = assign_roles(
        (_CHAT, _EMBEDDER),
        role_inputs=SetupRoleInputs(),
        console=console,
    )

    assert roles is not None
    assert (roles.world_model, roles.judge, roles.embedder) == ("luna", "luna", "embedder")
    assert roles.candidates == ()
    assert "Router candidates need two priced models" in console.output


def test_prior_role_answers_are_kept_with_an_empty_line() -> None:
    """Rerunning setup accepts the roles already persisted without retyping them."""
    console = ScriptedConsole("\n\n\n\n")

    roles = assign_roles(
        (_CHAT, _EMBEDDER),
        role_inputs=SetupRoleInputs(world_model="luna", judge="luna", embedder="embedder"),
        console=console,
    )

    assert roles is not None
    assert (roles.world_model, roles.judge, roles.embedder) == ("luna", "luna", "embedder")


def test_a_role_with_no_compatible_model_sends_the_user_back_to_selection() -> None:
    """Selecting no embedder is explained on the role screen, not saved as unknown."""
    console = ScriptedConsole("1\n1\n")

    roles = assign_roles((_CHAT,), role_inputs=SetupRoleInputs(), console=console)

    assert roles is None
    assert "No selected model can serve the embedder role" in console.output


def test_router_candidates_and_their_incumbent_are_assigned_from_selected_models() -> None:
    """Two priced candidates with explicit limits are offered with an incumbent screen."""
    console = ScriptedConsole("1\n1\n1\n1,2\n\n2\n")

    roles = assign_roles(
        (_CHAT, _OTHER_CHAT, _EMBEDDER),
        role_inputs=SetupRoleInputs(),
        console=console,
    )

    assert roles is not None
    assert roles.candidates == ("luna", "terra")
    assert roles.incumbent == "terra"


def test_a_single_selected_candidate_skips_the_router_role() -> None:
    """Router candidates require at least two models, so one selection is skipped."""
    console = ScriptedConsole("1\n1\n1\n1\n\n")

    roles = assign_roles(
        (_CHAT, _OTHER_CHAT, _EMBEDDER),
        role_inputs=SetupRoleInputs(),
        console=console,
    )

    assert roles is not None
    assert roles.candidates == ()
    assert roles.incumbent is None
    assert "Router candidates need at least two models" in console.output


def test_the_setup_written_from_confirmed_answers_carries_verified_metadata() -> None:
    """The saved setup states exactly the discovered metadata and assigned roles."""
    console = ScriptedConsole("1\n1\n1\n\n")
    chosen = (_CHAT, _EMBEDDER)
    roles = assign_roles(chosen, role_inputs=SetupRoleInputs(), console=console)

    assert roles is not None
    result = build_result(
        chosen,
        roles=roles,
        endpoints=(_endpoint(),),
        existing_connections=(),
        existing_models=(),
    )

    setup = result.setup
    assert [connection.name for connection in setup.connections] == ["openai"]
    assert [model.alias for model in setup.models] == ["luna", "embedder"]
    chat = setup.models[0]
    assert chat.model == "gpt-5.6-luna"
    assert chat.supports_tools
    assert chat.supports_structured_output
    assert chat.context_window_tokens == 1_050_000
    assert chat.input_cost_per_million_tokens_usd == 1.0
    assert chat.cache_write_cost_per_million_tokens_usd == 1.25
    assert (setup.world_model, setup.judge, setup.embedder) == ("luna", "luna", "embedder")


def test_already_configured_models_are_not_written_again() -> None:
    """Rerunning setup reassigns roles without duplicating configured catalog entries."""
    configured = AvailableModel(
        alias="luna",
        connection="openai",
        provider="openai",
        model="gpt-5.6-luna",
        capabilities=_CHAT.capabilities,
        pricing_source=PricingSource.CONFIGURED,
        configured=True,
    )
    existing_models = (
        ProviderModelSelection(alias="luna", connection="openai", model="gpt-5.6-luna"),
    )

    result = build_result(
        (configured, _EMBEDDER),
        roles=assign_roles(
            (configured, _EMBEDDER),
            role_inputs=SetupRoleInputs(),
            console=ScriptedConsole("1\n1\n1\n\n"),
        )
        or pytest.fail("roles were not assigned"),
        endpoints=(_endpoint(),),
        existing_connections=(_OPENAI,),
        existing_models=existing_models,
    )

    assert [model.alias for model in result.setup.models] == ["embedder"]
    assert result.setup.known_existing_aliases == ("luna",)


def test_configured_catalog_entries_become_selectable_rows() -> None:
    """A configured alias with verified metadata is offered beside discovered models."""
    record = ModelRecord(
        connection="openai",
        model="gpt-5.6-luna",
        capabilities=ModelCapabilities(
            supports_completions=True,
            supports_structured_output=True,
            input_cost_per_million_tokens_usd=1.0,
            output_cost_per_million_tokens_usd=6.0,
        ),
    )
    unusable = ModelRecord(connection="openai", model="unknown-model")

    rows = configured_models(
        {"luna": record, "mystery": unusable},
        connection_providers={"openai": "openai"},
    )

    assert record.capabilities is not None
    assert unusable.capabilities is None
    assert [row.alias for row in rows] == ["luna"]
    assert rows[0].configured
    assert rows[0].provider == "openai"


def test_manual_declaration_stays_behind_the_advanced_row() -> None:
    """A hand-declared model is only reachable from the explicit advanced row."""
    session = _session(_CHAT, advanced_models=True)
    console = ScriptedConsole("2\n\n1\nprivate-model\ny\nn\ny\ny\nn\nn\n2\n6\n0.5\n0.75\n\n")

    selected = select_models(session, console=console)

    assert selected is not None
    assert selected == ("private-model",)
    declared = session.manual[0]
    assert declared.alias == "private-model"
    assert declared.pricing_source is PricingSource.CONFIGURED
    assert declared.capabilities.supports_tools
    assert declared.capabilities.input_cost_per_million_tokens_usd == 2.0


def test_manual_declaration_requires_a_prepared_connection() -> None:
    """Declaring a model by hand still needs a prepared provider connection."""
    console = ScriptedConsole("")

    assert declare_model(SetupSession(), console=console) is None
    assert "Prepare a provider connection first." in console.output


def test_summary_states_providers_models_roles_prices_and_credential_behavior() -> None:
    """One summary explains everything about to be written, including credential handling."""
    console = ScriptedConsole("")
    chosen = (_CHAT, _EMBEDDER)
    roles = assign_roles(
        chosen, role_inputs=SetupRoleInputs(), console=ScriptedConsole("1\n1\n1\n\n")
    )

    assert roles is not None
    result = build_result(
        chosen,
        roles=roles,
        endpoints=(_endpoint(),),
        existing_connections=(),
        existing_models=(),
    )
    render_summary(result, chosen=chosen, endpoints=(_endpoint(),), console=console)

    printed = console.output
    assert "Configuration summary" in printed
    assert "provider openai: connection openai, credential OPENAI_API_KEY" in printed
    assert "model luna: openai/gpt-5.6-luna" in printed
    assert "pricing=wmo-catalog" in printed
    assert "roles: world_model=luna, judge=luna, embedder=embedder" in printed
    assert "WMO stores only the credential environment-variable name" in printed
    assert "secret-key" not in printed


def test_summary_names_the_aws_credential_chain_for_bedrock() -> None:
    """A Bedrock connection has no credential variable, so the summary names its chain."""
    console = ScriptedConsole("")
    bedrock = ProviderConnection(name="bedrock", provider="bedrock", region="us-east-1")
    chosen = (_CHAT, _EMBEDDER)
    roles = assign_roles(
        chosen, role_inputs=SetupRoleInputs(), console=ScriptedConsole("1\n1\n1\n\n")
    )

    assert roles is not None
    result = build_result(
        chosen,
        roles=roles,
        endpoints=(_endpoint(), _endpoint(bedrock)),
        existing_connections=(),
        existing_models=(),
    )
    render_summary(
        result,
        chosen=chosen,
        endpoints=(_endpoint(), _endpoint(bedrock)),
        console=console,
    )

    assert "provider bedrock: connection bedrock, credential AWS credential chain" in console.output


def test_summary_names_confirmed_router_candidates() -> None:
    """A confirmed router role is shown in the same single summary."""
    console = ScriptedConsole("")
    chosen = (_CHAT, _OTHER_CHAT, _EMBEDDER)
    roles = assign_roles(
        chosen,
        role_inputs=SetupRoleInputs(),
        console=ScriptedConsole("1\n1\n1\n1,2\n\n1\n"),
    )

    assert roles is not None
    result = build_result(
        chosen,
        roles=roles,
        endpoints=(_endpoint(),),
        existing_connections=(),
        existing_models=(),
    )
    render_summary(result, chosen=chosen, endpoints=(_endpoint(),), console=console)

    assert "router candidates: luna, terra; incumbent luna" in console.output


def test_a_connection_only_used_by_unselected_models_is_not_written() -> None:
    """Only connections behind selected models reach the catalog."""
    other = ProviderConnection(
        name="anthropic", provider="anthropic", api_key_env="ANTHROPIC_API_KEY"
    )
    console = ScriptedConsole("1\n1\n1\n\n")
    chosen = (_CHAT, _EMBEDDER)
    roles = assign_roles(chosen, role_inputs=SetupRoleInputs(), console=console)

    assert roles is not None
    result = build_result(
        chosen,
        roles=roles,
        endpoints=(_endpoint(), _endpoint(other)),
        existing_connections=(),
        existing_models=(),
    )

    assert [connection.name for connection in result.setup.connections] == ["openai"]
    assert ConnectionConfig(provider="anthropic", api_key_env="ANTHROPIC_API_KEY").provider


def test_duplicate_provider_model_names_receive_distinct_aliases() -> None:
    """Two providers publishing the same model ID still yield distinct saved aliases."""
    first = _model("gpt-5.6-luna", alias="gpt-5-6-luna")
    second = _model("gpt-5.6-luna", alias="gpt-5-6-luna-2", provider="openrouter")
    console = ScriptedConsole("1,2\n\n")

    selected = select_models(_session(first, second), console=console)

    assert selected == ("gpt-5-6-luna", "gpt-5-6-luna-2")
