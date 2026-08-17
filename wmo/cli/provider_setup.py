"""Provider and model setup shared by configuration and first build.

Interactive setup runs the provider and model picker in ``provider_picker``: providers, missing
credentials, models, roles, and one confirmation. Repeatable ``--provider`` flags skip the opening
list and feed the same session. Automation supplies the same catalog update as repeatable JSON.
Both paths end in one conflict-checked atomic ``models.toml`` write.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.prompt import Confirm

from wmo.cli.model_picker import (
    RoleAssignment,
    assign_roles,
    available_models,
    build_result,
    configured_models,
    render_summary,
    select_models,
)
from wmo.cli.provider_picker import (
    CREDENTIAL_NOTE,
    AvailableModel,
    ProviderSetupResult,
    SetupCancelled,
    SetupRoleInputs,
    SetupSession,
    explicit_provider_selection,
    prepare_providers,
    resolve_setup_providers,
    select_providers,
)
from wmo.common.models import (
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ProviderConnection,
    ProviderModelSelection,
    ProviderSetup,
    RouterCandidateSelection,
    SetupRole,
    catalog_state_sha256,
    configure_provider_catalog,
    configure_router_candidates,
    load_model_catalog,
    serves_role,
)
from wmo.common.models.known_models import recommended_model_rank
from wmo.common.models.setup import SETUP_PROVIDERS
from wmo.runtime.models.providers import HttpProviderModelLister, ProviderModelLister


@dataclass(frozen=True)
class ProviderSetupOptions:
    """Structured automation values or optional role choices for setup."""

    providers: tuple[str, ...] = ()
    connection_json: tuple[str, ...] = ()
    model_json: tuple[str, ...] = ()
    world_model: str | None = None
    judge: str | None = None
    embedder: str | None = None


@dataclass(frozen=True)
class _RecommendationModel:
    """Verified model fields needed by deterministic wizard recommendation policy."""

    alias: str
    provider: str
    model: str
    capabilities: ModelCapabilities


def run_provider_setup(
    root: Path,
    options: ProviderSetupOptions,
    *,
    non_interactive: bool,
    replace: bool,
    console: Console,
    lister: ProviderModelLister | None = None,
    offer_recommended_defaults: bool = False,
) -> ModelCatalog:
    """Collect a complete catalog update before one conflict-checked atomic write.

    Args:
        root: Local ``.wmo`` root.
        options: Structured connection, model, and role inputs.
        non_interactive: Whether absent values must be reported instead of prompted.
        replace: Whether conflicting collected entries may replace unprotected catalog state.
        console: Rich console used for prompts, summaries, and guidance.
        lister: Provider listing seam, injected by tests so no live request is made.
        offer_recommended_defaults: Whether verified discovery may fill every safe role at once.

    Returns:
        The complete catalog committed after final confirmation.

    Raises:
        typer.BadParameter: Structured input is invalid or incomplete.
        typer.Abort: Interactive collection is cancelled or reaches EOF.
    """
    path = root / "models.toml"
    starting_digest = catalog_state_sha256(path)
    existing = load_model_catalog(path) if path.exists() else None
    try:
        explicit_providers = resolve_setup_providers(options.providers)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    if non_interactive:
        try:
            setup = _noninteractive_setup(options, existing=existing)
        except (EOFError, KeyboardInterrupt):
            raise typer.Abort() from None
        return configure_provider_catalog(
            path,
            setup,
            replace=replace,
            expected_state_sha256=starting_digest,
        )
    result = _interactive_setup(
        existing_connections=_existing_connections(existing),
        existing_connection_providers=_existing_connection_providers(existing),
        existing_catalog_models={} if existing is None else existing.models,
        retainable_roles=_retained_setup_roles(existing),
        role_inputs=_role_inputs(options, existing=existing),
        explicit_providers=explicit_providers,
        offer_recommended_defaults=offer_recommended_defaults,
        console=console,
        lister=lister if lister is not None else HttpProviderModelLister(),
        environment=os.environ,
    )
    if result is None:
        raise typer.Abort()
    return _commit(path, result, replace=replace, expected_state_sha256=starting_digest)


def _interactive_setup(
    *,
    existing_connections: tuple[ProviderConnection, ...],
    existing_connection_providers: Mapping[str, str],
    existing_catalog_models: Mapping[str, ModelRecord],
    retainable_roles: Mapping[str, frozenset[SetupRole]],
    role_inputs: SetupRoleInputs,
    explicit_providers: tuple[str, ...],
    offer_recommended_defaults: bool,
    console: Console,
    lister: ProviderModelLister,
    environment: MutableMapping[str, str],
) -> ProviderSetupResult | None:
    """Collect one complete catalog update from the provider and model picker.

    Args:
        existing_connections: Secret-free connections already configured in the catalog.
        existing_connection_providers: Exact provider kind for every persisted connection.
        existing_catalog_models: Exact record for every persisted model alias.
        retainable_roles: Exact prior roles each incomplete alias may retain.
        role_inputs: Role values supplied by flags or already persisted.
        explicit_providers: Validated ``--provider`` values that skip the opening list once.
        offer_recommended_defaults: Whether verified discovery proposes one default
            assignment first, with manual model selection as the fallback.
        console: Terminal used for every screen.
        lister: Provider listing seam, injected by tests so no live request is made.
        environment: Process environment consulted and updated for pasted credentials.

    Returns:
        The confirmed setup, or ``None`` when the user cancelled or declined to save.
    """
    console.print("[bold]Model setup[/bold]")
    console.print(f"[dim]{CREDENTIAL_NOTE}[/dim]")
    configured = configured_models(
        existing_catalog_models,
        connection_providers=existing_connection_providers,
        retainable_roles=retainable_roles,
    )
    session = SetupSession(selected=tuple(model.alias for model in configured))
    skip_opening_list = bool(explicit_providers)
    if explicit_providers:
        session.providers, session.advanced_models = explicit_provider_selection(explicit_providers)
    try:
        while True:
            if skip_opening_list:
                skip_opening_list = False
            else:
                selection = select_providers(
                    session,
                    console=console,
                    environment=environment,
                    configured=bool(configured),
                )
                if selection is None:
                    return None
                session.providers, session.advanced_models = selection
            discovered: tuple[AvailableModel, ...] = ()
            session.endpoints = ()
            if session.providers:
                prepared = prepare_providers(
                    session,
                    existing_connections=existing_connections,
                    existing_aliases=tuple(sorted(existing_catalog_models)),
                    console=console,
                    lister=lister,
                    environment=environment,
                )
                if prepared is None:
                    continue
                session.endpoints, discovered = prepared
            session.available = (*configured, *discovered)
            if offer_recommended_defaults:
                try:
                    recommended = _recommended_result(
                        session,
                        known_existing_connections=tuple(sorted(existing_connection_providers)),
                        known_existing_aliases=tuple(sorted(existing_catalog_models)),
                        console=console,
                    )
                except ValueError as exc:
                    console.print(f"[yellow]note[/yellow] {exc}")
                else:
                    if Confirm.ask(
                        "Use these recommended models?",
                        default=True,
                        console=console,
                    ):
                        return recommended
            result = _collect_models_and_roles(
                session,
                known_existing_connections=tuple(sorted(existing_connection_providers)),
                known_existing_aliases=tuple(sorted(existing_catalog_models)),
                role_inputs=role_inputs,
                console=console,
            )
            if result is not None:
                return result
    except SetupCancelled:
        console.print("Setup cancelled. Nothing was written.")
        return None


def _recommended_result(
    session: SetupSession,
    *,
    known_existing_connections: tuple[str, ...],
    known_existing_aliases: tuple[str, ...],
    console: Console,
) -> ProviderSetupResult:
    """Assign every wizard role from verified discovered model availability.

    Args:
        session: Provider endpoints and verified available models.
        known_existing_connections: Every persisted connection name.
        known_existing_aliases: Every persisted model alias.
        console: Terminal receiving the deterministic summary.

    Returns:
        Complete setup result ready for the normal atomic catalog commit.

    Raises:
        ValueError: Verified availability cannot satisfy every required role.
    """
    available = tuple(item for item in available_models(session) if item.capabilities is not None)
    recommendations = tuple(
        _RecommendationModel(
            alias=item.alias,
            provider=item.provider,
            model=item.model,
            capabilities=item.capabilities,
        )
        for item in available
        if item.capabilities is not None
    )

    def eligible(role: SetupRole) -> tuple[_RecommendationModel, ...]:
        """Return verified models serving one role in deterministic alias order."""
        return tuple(
            sorted(
                (item for item in recommendations if serves_role(item.capabilities, role)),
                key=lambda item: _recommendation_key(item, role),
            )
        )

    world = eligible(SetupRole.WORLD_MODEL)
    judges = eligible(SetupRole.JUDGE)
    embedders = eligible(SetupRole.EMBEDDER)
    if not world or not judges or not embedders:
        raise ValueError(
            "recommended defaults need verified world, judge, embedder, and two distinct priced "
            "router models; continue with manual model selection to fill the missing roles"
        )
    selection = _recommended_router_selection_from_models(
        recommendations,
        world_alias=world[0].alias,
    )
    aliases = {
        world[0].alias,
        judges[0].alias,
        embedders[0].alias,
        *selection.candidates,
    }
    chosen = tuple(item for item in available if item.alias in aliases)
    result = build_result(
        chosen,
        roles=RoleAssignment(
            world_model=world[0].alias,
            judge=judges[0].alias,
            embedder=embedders[0].alias,
            candidates=selection.candidates,
            incumbent=selection.incumbent,
        ),
        endpoints=session.endpoints,
        known_existing_connections=known_existing_connections,
        known_existing_aliases=known_existing_aliases,
    )
    render_summary(result, chosen=chosen, endpoints=session.endpoints, console=console)
    return result


def _recommendation_key(
    item: _RecommendationModel,
    role: SetupRole,
) -> tuple[int, int, int, float, str, str]:
    """Rank one verified model by maintained guidance, then capability and cost.

    Args:
        item: Verified provider-listed model.
        role: Wizard role being filled.

    Returns:
        Stable sort key preferring maintained provider guidance before a cost fallback.
    """
    rank = recommended_model_rank(item.provider, item.model, role.value)
    provider_rank = {"openai": 0, "anthropic": 1, "gemini": 2}.get(item.provider, 3)
    capabilities = item.capabilities
    assert capabilities is not None
    if role is SetupRole.EMBEDDER:
        cost = capabilities.input_cost_per_million_tokens_usd
    else:
        input_cost = capabilities.input_cost_per_million_tokens_usd
        output_cost = capabilities.output_cost_per_million_tokens_usd
        cost = None if input_cost is None or output_cost is None else input_cost + output_cost
    return (
        0 if rank is not None else 1,
        provider_rank,
        rank if rank is not None else 10_000,
        cost if cost is not None else math.inf,
        item.provider,
        item.model,
    )


def _recommended_router_selection(catalog: ModelCatalog) -> RouterCandidateSelection:
    """Choose existing-catalog router defaults through the shared recommendation policy.

    Args:
        catalog: Secret-free catalog with verified provider, model, and capability metadata.

    Returns:
        Two deterministic candidates and the eligible world model as incumbent when possible.

    Raises:
        ValueError: The catalog has fewer than two eligible completion candidates.
    """
    models = tuple(
        _RecommendationModel(
            alias=alias,
            provider=catalog.connections[record.connection].provider,
            model=record.model,
            capabilities=record.capabilities,
        )
        for alias, record in catalog.models.items()
        if record.capabilities is not None
    )
    return _recommended_router_selection_from_models(
        models,
        world_alias=catalog.roles.world_model,
    )


def _recommended_router_selection_from_models(
    models: tuple[_RecommendationModel, ...],
    *,
    world_alias: str | None,
) -> RouterCandidateSelection:
    """Select an incumbent and provider-diverse alternative from verified models.

    Args:
        models: Exact verified models available to the current setup.
        world_alias: Configured world alias preferred as the incumbent when eligible.

    Returns:
        Two candidates ordered incumbent first and one exact incumbent alias.

    Raises:
        ValueError: Fewer than two distinct eligible router models are available.
    """
    ranked = tuple(
        sorted(
            (item for item in models if serves_role(item.capabilities, SetupRole.ROUTER_CANDIDATE)),
            key=lambda item: _recommendation_key(item, SetupRole.ROUTER_CANDIDATE),
        )
    )
    unique = []
    identities: set[tuple[str, str]] = set()
    for item in ranked:
        identity = (item.provider, item.model)
        if identity in identities:
            continue
        identities.add(identity)
        unique.append(item)
    if len(unique) < 2:
        raise ValueError(
            "recommended defaults need two distinct priced router models with verified limits"
        )
    incumbent = next((item for item in unique if item.alias == world_alias), unique[0])
    alternative = next(
        (item for item in unique if item.provider != incumbent.provider),
        None,
    ) or next(item for item in unique if item.alias != incumbent.alias)
    return RouterCandidateSelection(
        candidates=(incumbent.alias, alternative.alias),
        incumbent=incumbent.alias,
    )


def _collect_models_and_roles(
    session: SetupSession,
    *,
    known_existing_connections: tuple[str, ...],
    known_existing_aliases: tuple[str, ...],
    role_inputs: SetupRoleInputs,
    console: Console,
) -> ProviderSetupResult | None:
    """Run the model, role, and confirmation screens for one prepared provider set.

    Args:
        session: Answers already collected in this setup session.
        known_existing_connections: Every connection name in the persisted catalog.
        known_existing_aliases: Every model alias in the persisted catalog.
        role_inputs: Role values supplied by flags or already persisted.
        console: Terminal used for every screen.

    Returns:
        The confirmed result, or ``None`` when the user asked to change providers.

    Raises:
        SetupCancelled: The user cancelled setup or declined to save.
    """
    while True:
        selected = select_models(session, console=console)
        if selected is None:
            return None
        session.selected = selected
        chosen = tuple(item for item in available_models(session) if item.alias in selected)
        roles = assign_roles(chosen, role_inputs=role_inputs, console=console)
        if roles is None:
            continue
        result = build_result(
            chosen,
            roles=roles,
            endpoints=session.endpoints,
            known_existing_connections=known_existing_connections,
            known_existing_aliases=known_existing_aliases,
        )
        render_summary(result, chosen=chosen, endpoints=session.endpoints, console=console)
        if not Confirm.ask("Save this configuration?", default=True, console=console):
            raise SetupCancelled
        return result


def _commit(
    path: Path,
    result: ProviderSetupResult,
    *,
    replace: bool,
    expected_state_sha256: str | None,
) -> ModelCatalog:
    """Write the confirmed setup, then assign any confirmed router candidate roles.

    Args:
        path: Shared ``models.toml`` path.
        result: Confirmed setup and optional router roles.
        replace: Whether conflicting collected entries may replace unprotected catalog state.
        expected_state_sha256: Catalog digest captured before collection began.

    Returns:
        The complete catalog after every confirmed role is committed.
    """
    catalog = configure_provider_catalog(
        path,
        result.setup,
        replace=replace,
        expected_state_sha256=expected_state_sha256,
    )
    if not result.candidates or result.incumbent is None:
        return catalog
    if (
        result.candidates == catalog.roles.candidates
        and result.incumbent == catalog.roles.incumbent
    ):
        return catalog
    return configure_router_candidates(
        path,
        RouterCandidateSelection(candidates=result.candidates, incumbent=result.incumbent),
        expected_state_sha256=catalog_state_sha256(path),
    )


def _role_inputs(
    options: ProviderSetupOptions,
    *,
    existing: ModelCatalog | None,
) -> SetupRoleInputs:
    """Combine explicit role flags with the roles already present in the catalog.

    Args:
        options: Structured setup arguments, including optional role flags.
        existing: Existing catalog, or ``None`` on first setup.

    Returns:
        Prior role values the picker offers as defaults.
    """
    return SetupRoleInputs(
        world_model=options.world_model or _existing_role(existing, "world_model"),
        judge=options.judge or _existing_role(existing, "judge"),
        embedder=options.embedder or _existing_role(existing, "embedder"),
        candidates=existing.roles.candidates if existing is not None else (),
        incumbent=existing.roles.incumbent if existing is not None else None,
    )


def _noninteractive_setup(
    options: ProviderSetupOptions,
    *,
    existing: ModelCatalog | None,
) -> ProviderSetup:
    """Parse repeatable JSON objects and list every missing automation input.

    Args:
        options: Structured connection, model, and role arguments.
        existing: Existing catalog whose compatible entries remain available.

    Returns:
        Complete validated setup ready for atomic catalog merge.

    Raises:
        typer.BadParameter: Any JSON record is invalid or required input is missing.
    """
    connections: list[ProviderConnection] = []
    models: list[ProviderModelSelection] = []
    errors: list[str] = []
    for position, payload in enumerate(options.connection_json, start=1):
        try:
            connections.append(ProviderConnection.model_validate_json(payload))
        except ValidationError as exc:
            errors.append(f"--connection-json #{position}: {exc.errors()[0]['msg']}")
    for position, payload in enumerate(options.model_json, start=1):
        try:
            models.append(ProviderModelSelection.model_validate_json(payload))
        except ValidationError as exc:
            errors.append(f"--model-json #{position}: {exc.errors()[0]['msg']}")
    roles = {
        "--world-model": options.world_model or _existing_role(existing, "world_model"),
        "--judge": options.judge or _existing_role(existing, "judge"),
        "--embedder": options.embedder or _existing_role(existing, "embedder"),
    }
    existing_connection_names = tuple(sorted(_existing_connection_providers(existing)))
    existing_model_aliases = _existing_model_aliases(existing)
    available_connections = (*existing_connection_names, *(item.name for item in connections))
    available_models = (*existing_model_aliases, *(item.alias for item in models))
    missing = []
    if not available_connections:
        missing.append("at least one --connection-json")
    if not available_models:
        missing.append("at least one --model-json")
    missing.extend(flag for flag, value in roles.items() if value is None)
    if errors or missing:
        details = (*errors, *(f"missing {item}" for item in missing))
        raise typer.BadParameter(
            "noninteractive provider setup is incomplete: " + "; ".join(details)
        )
    assert roles["--world-model"] is not None
    assert roles["--judge"] is not None
    assert roles["--embedder"] is not None
    return ProviderSetup(
        connections=_deduplicate_connections(connections),
        models=_deduplicate_models(models),
        known_existing_connections=existing_connection_names,
        known_existing_aliases=existing_model_aliases,
        world_model=roles["--world-model"],
        judge=roles["--judge"],
        embedder=roles["--embedder"],
    )


def _existing_connections(existing: ModelCatalog | None) -> tuple[ProviderConnection, ...]:
    """Convert supported existing catalog connections into setup input records.

    Args:
        existing: Existing catalog, or ``None`` on first setup.

    Returns:
        Compatible secret-free connection records in deterministic order.
    """
    if existing is None:
        return ()
    return tuple(
        ProviderConnection(
            name=name,
            provider=connection.provider,
            api_key_env=connection.api_key_env,
            base_url=connection.base_url,
            api_version=connection.api_version,
            region=connection.region,
        )
        for name, connection in sorted(existing.connections.items())
        if connection.provider in SETUP_PROVIDERS
        and (connection.provider == "bedrock" or connection.api_key_env is not None)
    )


def _existing_model_aliases(existing: ModelCatalog | None) -> tuple[str, ...]:
    """Return every persisted model alias in deterministic order.

    Args:
        existing: Existing catalog, or ``None`` on first setup.

    Returns:
        Every exact alias already owned by the catalog.
    """
    if existing is None:
        return ()
    return tuple(sorted(existing.models))


def _existing_connection_providers(existing: ModelCatalog | None) -> dict[str, str]:
    """Return every persisted connection name and its exact provider kind.

    Args:
        existing: Existing catalog, or ``None`` on first setup.

    Returns:
        Provider kinds keyed by connection name in deterministic order.
    """
    if existing is None:
        return {}
    return {name: connection.provider for name, connection in sorted(existing.connections.items())}


def _retained_setup_roles(existing: ModelCatalog | None) -> dict[str, frozenset[SetupRole]]:
    """Map persisted setup roles to the aliases allowed to retain them without metadata.

    Args:
        existing: Existing catalog, or ``None`` on first setup.

    Returns:
        Exact retain-only roles keyed by their currently assigned aliases.
    """
    if existing is None:
        return {}
    retained: dict[str, set[SetupRole]] = {}
    assignments = (
        (existing.roles.world_model, SetupRole.WORLD_MODEL),
        (existing.roles.judge, SetupRole.JUDGE),
        (existing.roles.embedder, SetupRole.EMBEDDER),
    )
    for alias, role in assignments:
        if alias is not None:
            retained.setdefault(alias, set()).add(role)
    for alias in existing.roles.candidates:
        retained.setdefault(alias, set()).add(SetupRole.ROUTER_CANDIDATE)
    return {alias: frozenset(roles) for alias, roles in sorted(retained.items())}


def _existing_role(existing: ModelCatalog | None, role: str) -> str | None:
    """Return one prior build role when it is present.

    Args:
        existing: Existing catalog, or ``None`` on first setup.
        role: Build-role field to inspect.

    Returns:
        Existing alias string, or ``None`` when the role is absent.
    """
    if existing is None:
        return None
    value = getattr(existing.roles, role)
    return value if isinstance(value, str) else None


def _deduplicate_connections(
    connections: list[ProviderConnection],
) -> tuple[ProviderConnection, ...]:
    """Keep one equal record per name and reject ambiguous collected duplicates.

    Args:
        connections: Collected provider connection records.

    Returns:
        Unique records sorted by connection name.

    Raises:
        typer.BadParameter: One name maps to unequal records.
    """
    by_name: dict[str, ProviderConnection] = {}
    for connection in connections:
        prior = by_name.get(connection.name)
        if prior is not None and prior != connection:
            raise typer.BadParameter(f"connection {connection.name!r} was supplied more than once")
        by_name[connection.name] = connection
    return tuple(by_name[name] for name in sorted(by_name))


def _deduplicate_models(
    models: list[ProviderModelSelection],
) -> tuple[ProviderModelSelection, ...]:
    """Keep one equal record per alias and reject ambiguous collected duplicates.

    Args:
        models: Collected explicit model selections.

    Returns:
        Unique records sorted by alias.

    Raises:
        typer.BadParameter: One alias maps to unequal records.
    """
    by_alias: dict[str, ProviderModelSelection] = {}
    for model in models:
        prior = by_alias.get(model.alias)
        if prior is not None and prior != model:
            raise typer.BadParameter(f"model alias {model.alias!r} was supplied more than once")
        by_alias[model.alias] = model
    return tuple(by_alias[alias] for alias in sorted(by_alias))


def provider_setup_json_examples() -> tuple[str, str]:
    """Return compact structured examples used in noninteractive remediation messages.

    Returns:
        Connection and model JSON examples suitable for quoting as CLI values.
    """
    connection = json.dumps(
        {"name": "openai", "provider": "openai", "api_key_env": "OPENAI_API_KEY"},
        separators=(",", ":"),
    )
    model = json.dumps(
        {
            "alias": "model",
            "connection": "openai",
            "model": "your-model-id",
            "capabilities": {
                "supports_embeddings": True,
                "supports_structured_output": True,
                "supports_completions": True,
                "input_cost_per_million_tokens_usd": 0,
                "output_cost_per_million_tokens_usd": 0,
                "cached_input_cost_per_million_tokens_usd": 0,
                "cache_write_cost_per_million_tokens_usd": 0,
            },
        },
        separators=(",", ":"),
    )
    return connection, model
