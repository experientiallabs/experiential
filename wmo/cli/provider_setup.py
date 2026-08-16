"""Provider and model setup shared by configuration and first build.

Interactive setup runs the provider and model picker in ``provider_picker``: providers, missing
credentials, models, roles, and one confirmation. Automation supplies the same catalog update as
repeatable JSON. Both paths end in one conflict-checked atomic ``models.toml`` write.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.prompt import Confirm

from wmo.cli.model_picker import (
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
    prepare_providers,
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
    catalog_state_sha256,
    configure_provider_catalog,
    configure_router_candidates,
    load_model_catalog,
)
from wmo.common.models.setup import SETUP_PROVIDERS
from wmo.runtime.models.providers import HttpProviderModelLister, ProviderModelLister


@dataclass(frozen=True)
class ProviderSetupOptions:
    """Structured automation values or optional role choices for setup."""

    connection_json: tuple[str, ...] = ()
    model_json: tuple[str, ...] = ()
    world_model: str | None = None
    judge: str | None = None
    embedder: str | None = None


def run_provider_setup(
    root: Path,
    options: ProviderSetupOptions,
    *,
    non_interactive: bool,
    replace: bool,
    console: Console,
    lister: ProviderModelLister | None = None,
) -> ModelCatalog:
    """Collect a complete catalog update before one conflict-checked atomic write.

    Args:
        root: Local ``.wmo`` root.
        options: Structured connection, model, and role inputs.
        non_interactive: Whether absent values must be reported instead of prompted.
        replace: Whether conflicting collected entries may replace unprotected catalog state.
        console: Rich console used for prompts, summaries, and guidance.
        lister: Provider listing seam, injected by tests so no live request is made.

    Returns:
        The complete catalog committed after final confirmation.

    Raises:
        typer.BadParameter: Structured input is invalid or incomplete.
        typer.Abort: Interactive collection is cancelled or reaches EOF.
    """
    path = root / "models.toml"
    starting_digest = catalog_state_sha256(path)
    existing = load_model_catalog(path) if path.exists() else None
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
        existing_models=_existing_models(existing),
        role_inputs=_role_inputs(options, existing=existing),
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
    existing_models: tuple[ProviderModelSelection, ...],
    role_inputs: SetupRoleInputs,
    console: Console,
    lister: ProviderModelLister,
    environment: MutableMapping[str, str],
) -> ProviderSetupResult | None:
    """Collect one complete catalog update from the provider and model picker.

    Args:
        existing_connections: Secret-free connections already configured in the catalog.
        existing_connection_providers: Exact provider kind for every persisted connection.
        existing_catalog_models: Exact record for every persisted model alias.
        existing_models: Model aliases already configured in the catalog.
        role_inputs: Role values supplied by flags or already persisted.
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
    )
    session = SetupSession(selected=tuple(model.alias for model in configured))
    try:
        while True:
            selection = select_providers(
                session,
                console=console,
                environment=environment,
                configured=bool(configured),
            )
            if selection is None:
                return None
            session.providers, session.advanced_credentials, session.advanced_models = selection
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
            result = _collect_models_and_roles(
                session,
                existing_connections=existing_connections,
                known_existing_connections=tuple(sorted(existing_connection_providers)),
                known_existing_aliases=tuple(sorted(existing_catalog_models)),
                existing_models=existing_models,
                role_inputs=role_inputs,
                console=console,
            )
            if result is not None:
                return result
    except SetupCancelled:
        console.print("Setup cancelled. Nothing was written.")
        return None


def _collect_models_and_roles(
    session: SetupSession,
    *,
    existing_connections: tuple[ProviderConnection, ...],
    known_existing_connections: tuple[str, ...],
    known_existing_aliases: tuple[str, ...],
    existing_models: tuple[ProviderModelSelection, ...],
    role_inputs: SetupRoleInputs,
    console: Console,
) -> ProviderSetupResult | None:
    """Run the model, role, and confirmation screens for one prepared provider set.

    Args:
        session: Answers already collected in this setup session.
        existing_connections: Connections already configured in the catalog.
        known_existing_connections: Every connection name in the persisted catalog.
        known_existing_aliases: Every model alias in the persisted catalog.
        existing_models: Model aliases already configured in the catalog.
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
            existing_connections=existing_connections,
            existing_models=existing_models,
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


def _existing_models(existing: ModelCatalog | None) -> tuple[ProviderModelSelection, ...]:
    """Convert setup-compatible existing model aliases into input records.

    Args:
        existing: Existing catalog, or ``None`` on first setup.

    Returns:
        Compatible explicit model selections in deterministic order.
    """
    if existing is None:
        return ()
    supported_connections = {item.name for item in _existing_connections(existing)}
    records = []
    for alias, model in sorted(existing.models.items()):
        if model.connection not in supported_connections:
            continue
        records.append(
            ProviderModelSelection(
                alias=alias,
                connection=model.connection,
                model=model.model,
                capabilities=model.capabilities or ModelCapabilities(),
            )
        )
    return tuple(records)


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
