"""Interactive and structured collection of router candidates with provider discovery."""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt

from wmo.cli.providers.model_picker import (
    available_models,
    configured_models,
    model_selection,
    select_router_candidates,
)
from wmo.cli.providers.provider_picker import (
    AvailableModel,
    SetupCancelled,
    SetupSession,
    prepare_providers,
    select_providers,
)
from wmo.cli.providers.setup import existing_setup_connections
from wmo.cli.shared.options import usage_error
from wmo.cli.shared.picker import PickerAction, PickerOption, choose_many, choose_one
from wmo.common.models import (
    ModelCapabilities,
    ModelCatalog,
    ProviderConnection,
    ProviderModelSelection,
    RouterCandidateSelection,
    SetupRole,
    catalog_state_sha256,
    completion_candidate_aliases,
    serves_role,
    validate_router_candidate_selection,
)
from wmo.optimize.router.composition import RouterCandidateSetupPlan
from wmo.runtime.models.providers import HttpProviderModelLister, ProviderModelLister


@dataclass(frozen=True)
class RouterCandidatePickerResult:
    """Candidate roles plus newly discovered catalog records selected by the picker."""

    selection: RouterCandidateSelection
    candidate_models: tuple[ProviderModelSelection, ...] = ()
    connections: tuple[ProviderConnection, ...] = ()


def run_router_candidate_picker(
    catalog: ModelCatalog,
    *,
    candidates: tuple[str, ...] = (),
    incumbent: str | None = None,
    console: Console,
    lister: ProviderModelLister | None = None,
    environment: MutableMapping[str, str] | None = None,
) -> RouterCandidatePickerResult | None:
    """Discover provider models and choose strict router candidates through existing setup UX.

    Args:
        catalog: Current secret-free catalog used for configured-only choices and conflict checks.
        candidates: Candidate aliases to preselect when they are discovered or configured.
        incumbent: Explicit incumbent alias, if supplied by the command.
        console: Terminal used for provider discovery and candidate screens.
        lister: Provider listing seam, injected by tests instead of making live requests.
        environment: Mutable environment consulted for provider credentials.

    Returns:
        The confirmed candidate selection and any new provider records, or ``None`` when the user
        cancels provider selection.

    Raises:
        SetupCancelled: The user cancels credential or recovery prompts.
        ValueError: An explicit candidate or incumbent is not eligible after discovery.
    """
    configured = configured_models(
        catalog.models,
        connection_providers={
            name: connection.provider for name, connection in catalog.connections.items()
        },
    )
    session = SetupSession(selected=tuple(item.alias for item in configured))
    provider_lister = lister if lister is not None else HttpProviderModelLister()
    provider_environment = os.environ if environment is None else environment
    existing_connections = existing_setup_connections(catalog)
    existing_aliases = tuple(sorted(catalog.models))
    preselected = candidates
    while True:
        provider_selection = select_providers(
            session,
            console=console,
            environment=provider_environment,
            configured=bool(configured),
        )
        if provider_selection is None:
            return None
        session.providers, session.advanced_models = provider_selection
        session.endpoints = ()
        discovered: tuple[AvailableModel, ...] = ()
        if session.providers:
            prepared = prepare_providers(
                session,
                existing_connections=existing_connections,
                existing_aliases=existing_aliases,
                configured=configured,
                console=console,
                lister=provider_lister,
                environment=provider_environment,
            )
            if prepared is None:
                continue
            session.endpoints, discovered = prepared
        session.available = (*configured, *discovered)
        available = available_models(session)
        eligible = tuple(
            item
            for item in available
            if item.capabilities is not None
            and serves_role(item.capabilities, SetupRole.ROUTER_CANDIDATE)
        )
        if len(eligible) < 2:
            console.print(
                "[yellow]The selected providers exposed fewer than two eligible completion "
                "models. Choose another provider or use the advanced --candidate-model "
                "path.[/yellow]"
            )
            continue
        selection = select_router_candidates(
            available,
            preselected=preselected,
            incumbent=incumbent,
            effort_defaults=catalog.roles.candidate_reasoning_efforts,
            console=console,
        )
        if selection is None:
            continue
        selected_models = tuple(item for item in available if item.alias in selection.candidates)
        used_connections = {item.connection for item in selected_models}
        candidate_models = tuple(
            model_selection(item) for item in selected_models if not item.configured
        )
        connections = tuple(
            endpoint.connection
            for endpoint in session.endpoints
            if not endpoint.configured and endpoint.connection.name in used_connections
        )
        return RouterCandidatePickerResult(
            selection=selection,
            candidate_models=candidate_models,
            connections=connections,
        )


def collect_router_candidates(
    path: Path,
    catalog: ModelCatalog,
    *,
    candidates: tuple[str, ...],
    candidate_models: tuple[ProviderModelSelection, ...] = (),
    incumbent: str | None,
    non_interactive: bool,
    console: Console,
    provider_lister: ProviderModelLister | None = None,
    environment: MutableMapping[str, str] | None = None,
    interactive_command: str | None = None,
) -> RouterCandidateSetupPlan:
    """Collect and validate candidate roles without mutating the shared catalog.

    Args:
        path: Shared ``models.toml`` path whose state is being confirmed.
        catalog: Already validated catalog shown during collection.
        candidates: Repeatable explicit candidate aliases supplied by the caller.
        candidate_models: Repeatable complete candidate definitions supplied by the caller.
        incumbent: Optional explicit quality incumbent.
        non_interactive: Whether absent values must be reported instead of prompted.
        console: Rich terminal used for prompts and the final summary.
        provider_lister: Provider listing seam used by the configured-provider picker.
        environment: Mutable environment used by provider credential discovery.
        interactive_command: One complete command printed for missing non-interactive input.

    Returns:
        Confirmed selection and exact catalog digest for a later atomic write.

    Raises:
        typer.BadParameter: The selection is missing, ambiguous, or ineligible.
        typer.Abort: The operator rejects the interactive summary.
    """
    state_sha256 = catalog_state_sha256(path)
    collected_models = candidate_models
    candidate_connections: tuple[ProviderConnection, ...] = ()
    selection: RouterCandidateSelection | None = None
    if (
        not non_interactive
        and not candidate_models
        and len(completion_candidate_aliases(catalog)) < 2
    ):
        try:
            picked = run_router_candidate_picker(
                catalog,
                candidates=candidates,
                incumbent=incumbent,
                console=console,
                lister=provider_lister,
                environment=environment,
            )
        except SetupCancelled:
            raise typer.Abort() from None
        if picked is None:
            raise typer.Abort()
        selection = picked.selection
        collected_models = picked.candidate_models
        candidate_connections = picked.connections
    elif not non_interactive:
        collected_models = _interactive_candidate_models(catalog, candidate_models, console=console)
    prospective = _catalog_with_candidate_models(
        catalog,
        collected_models,
        candidate_connections=candidate_connections,
    )
    if non_interactive:
        selected = candidates or tuple(item.alias for item in collected_models)
        selection = _noninteractive_selection(
            prospective,
            selected,
            incumbent,
            interactive_command=interactive_command,
        )
    elif selection is None:
        selection = _interactive_selection(prospective, candidates, incumbent, console=console)
    problems = validate_router_candidate_selection(prospective, selection)
    if problems:
        raise typer.BadParameter("router candidate setup is incomplete: " + "; ".join(problems))
    prospective = prospective.model_copy(
        update={
            "roles": prospective.roles.model_copy(
                update={
                    "candidates": selection.candidates,
                    "incumbent": selection.incumbent,
                }
            )
        }
    )
    console.print("[bold]Router candidate summary[/bold]")
    console.print("candidates: " + ", ".join(selection.candidates))
    console.print(f"incumbent: {selection.incumbent}")
    for model in collected_models:
        caps = model.capabilities
        console.print(
            f"{model.alias}: {model.connection}/{model.model}, "
            f"context {caps.context_window_tokens}, output {caps.maximum_output_tokens}"
        )
    if not non_interactive and not Confirm.ask(
        "Save these router candidates?", default=True, console=console
    ):
        raise typer.Abort()
    return RouterCandidateSetupPlan(
        selection=selection,
        candidate_models=collected_models,
        candidate_connections=candidate_connections,
        prospective_catalog=prospective,
        expected_catalog_sha256=state_sha256,
    )


def _catalog_with_candidate_models(
    catalog: ModelCatalog,
    candidate_models: tuple[ProviderModelSelection, ...],
    *,
    candidate_connections: tuple[ProviderConnection, ...] = (),
) -> ModelCatalog:
    """Merge explicit candidate definitions into an in-memory catalog.

    Args:
        catalog: Current catalog that remains unchanged on disk.
        candidate_models: Complete candidate definitions collected for this optimize run.
        candidate_connections: New provider connections selected for those definitions.

    Returns:
        Prospective catalog used for aggregate validation.

    Raises:
        typer.BadParameter: An alias repeats or names an unknown connection.
    """
    aliases = tuple(item.alias for item in candidate_models)
    if len(set(aliases)) != len(aliases):
        raise typer.BadParameter("candidate model definitions must use unique aliases")
    connections = dict(catalog.connections)
    for connection in candidate_connections:
        configured = connection.catalog_config()
        existing = connections.get(connection.name)
        if existing is not None and existing != configured:
            raise typer.BadParameter(
                f"provider connection {connection.name!r} already names different settings; "
                "choose another provider connection"
            )
        connections[connection.name] = configured
    unknown_connections = sorted(
        {item.connection for item in candidate_models}.difference(connections)
    )
    if unknown_connections:
        raise typer.BadParameter(
            "candidate models name unknown provider connections: " + ", ".join(unknown_connections)
        )
    models = dict(catalog.models)
    for item in candidate_models:
        candidate_record = item.catalog_record()
        existing = models.get(item.alias)
        if existing is not None and existing != candidate_record:
            raise typer.BadParameter(
                f"candidate alias {item.alias!r} already names different model metadata; "
                "use a new alias"
            )
        models[item.alias] = candidate_record
    return catalog.model_copy(update={"connections": connections, "models": models})


def _interactive_candidate_models(
    catalog: ModelCatalog,
    candidate_models: tuple[ProviderModelSelection, ...],
    *,
    console: Console,
) -> tuple[ProviderModelSelection, ...]:
    """Collect complete candidate definitions until at least two aliases are eligible.

    Args:
        catalog: Existing provider connections and model aliases.
        candidate_models: Complete definitions already supplied on the command line.
        console: Rich terminal used for explicit model metadata prompts.

    Returns:
        Supplied and interactively collected candidate definitions.

    Raises:
        typer.BadParameter: Entered model metadata is invalid.
    """
    collected = list(candidate_models)
    prospective = _catalog_with_candidate_models(catalog, tuple(collected))
    while len(completion_candidate_aliases(prospective)) < 2:
        console.print(
            "Router optimization needs at least two completion candidates. "
            "Add a model using one configured provider connection."
        )
        try:
            selection = ProviderModelSelection(
                alias=Prompt.ask("Candidate alias", console=console),
                connection=Prompt.ask(
                    "Provider connection",
                    choices=sorted(catalog.connections),
                    console=console,
                ),
                model=Prompt.ask("Provider model ID", console=console),
                capabilities=ModelCapabilities(
                    supports_tools=Confirm.ask("Supports tools?", default=False, console=console),
                    supports_completions=True,
                    context_window_tokens=int(Prompt.ask("Context window tokens", console=console)),
                    maximum_output_tokens=int(Prompt.ask("Maximum output tokens", console=console)),
                    input_cost_per_million_tokens_usd=float(
                        Prompt.ask("Input USD per million tokens", console=console)
                    ),
                    output_cost_per_million_tokens_usd=float(
                        Prompt.ask("Output USD per million tokens", console=console)
                    ),
                    cached_input_cost_per_million_tokens_usd=float(
                        Prompt.ask("Cached input USD per million tokens", console=console)
                    ),
                    cache_write_cost_per_million_tokens_usd=float(
                        Prompt.ask("Cache write USD per million tokens", console=console)
                    ),
                ),
            )
        except ValueError as exc:
            raise typer.BadParameter(f"candidate model definition is invalid: {exc}") from None
        collected.append(selection)
        prospective = _catalog_with_candidate_models(catalog, tuple(collected))
    return tuple(collected)


def _noninteractive_selection(
    catalog: ModelCatalog,
    candidates: tuple[str, ...],
    incumbent: str | None,
    *,
    interactive_command: str | None = None,
) -> RouterCandidateSelection:
    """Resolve repeatable structured input or one complete persisted selection.

    Args:
        catalog: Current local catalog.
        candidates: Explicit repeatable aliases, possibly empty.
        incumbent: Explicit incumbent, possibly absent.
        interactive_command: Complete interactive command for repairing missing input.

    Returns:
        Unambiguous selected roles.

    Raises:
        typer.BadParameter: Inputs are partial or no complete persisted selection exists.
    """
    selected = candidates or catalog.roles.candidates
    selected_incumbent = incumbent or (catalog.roles.incumbent if not candidates else None)
    missing = []
    if not selected:
        missing.append("at least two repeatable --candidate ALIAS values")
    elif len(selected) < 2:
        missing.append("a second distinct --candidate ALIAS value")
    if selected_incumbent is None:
        missing.append("--incumbent ALIAS")
    if missing:
        message = "noninteractive router setup is missing: " + "; ".join(missing)
        if interactive_command:
            message += f". Run `{interactive_command}` to choose candidates interactively"
        raise typer.BadParameter(message)
    if selected_incumbent is None:
        raise AssertionError("validated noninteractive selection has an incumbent")
    with usage_error(ValueError):
        return RouterCandidateSelection(candidates=selected, incumbent=selected_incumbent)


def _candidate_option(catalog: ModelCatalog, alias: str) -> PickerOption:
    """Describe one eligible candidate alias with its identity and pricing.

    Args:
        catalog: Catalog holding the alias metadata.
        alias: Eligible completion candidate alias.

    Returns:
        Picker option carrying the provider identity, token limits, and token prices.
    """
    record = catalog.models[alias]
    caps = record.capabilities
    details = [f"roles: router candidate; connection: {record.connection}"]
    if caps is not None:
        details.append(
            f"context {caps.context_window_tokens}, output {caps.maximum_output_tokens}, "
            f"pricing in/out USD per million {caps.input_cost_per_million_tokens_usd}/"
            f"{caps.output_cost_per_million_tokens_usd}"
        )
    return PickerOption(
        value=alias,
        label=f"{alias} ({record.connection}/{record.model})",
        detail="; ".join(details),
    )


def _interactive_selection(
    catalog: ModelCatalog,
    candidates: tuple[str, ...],
    incumbent: str | None,
    *,
    console: Console,
) -> RouterCandidateSelection:
    """Choose multiple eligible aliases and one incumbent with no inferred model choice.

    Args:
        catalog: Current local catalog.
        candidates: Optional explicit aliases that skip only the candidate screen.
        incumbent: Optional explicit incumbent that skips only the incumbent screen.
        console: Rich terminal used for choices.

    Returns:
        Explicit selection ready for summary confirmation.

    Raises:
        typer.Abort: The operator cancels a selection screen.
        typer.BadParameter: Fewer than two eligible aliases exist or explicit aliases are
            ineligible.
    """
    eligible = completion_candidate_aliases(catalog)
    if len(eligible) < 2:
        raise typer.BadParameter(
            "router optimization needs at least two configured aliases with "
            "supports_completions=true and complete input, output, cached input, and cache write "
            "pricing"
        )
    options = [_candidate_option(catalog, alias) for alias in eligible]
    preselected = (
        catalog.roles.candidates if set(catalog.roles.candidates).issubset(eligible) else ()
    )
    while True:
        selected = candidates
        if not selected:
            chosen = choose_many(
                console,
                title="Router candidates (select at least two)",
                options=options,
                preselected=preselected,
                minimum=2,
            )
            if chosen.action is PickerAction.CANCEL:
                raise typer.Abort()
            if chosen.action is PickerAction.BACK:
                continue
            selected = chosen.values
        unknown = tuple(alias for alias in selected if alias not in eligible)
        if unknown:
            raise typer.BadParameter(
                "candidate aliases are not eligible: " + ", ".join(sorted(set(unknown)))
            )
        if incumbent is not None:
            with usage_error(ValueError):
                return RouterCandidateSelection(candidates=selected, incumbent=incumbent)
        chosen_incumbent = choose_one(
            console,
            title="Router incumbent among the candidates",
            options=[_candidate_option(catalog, alias) for alias in selected],
            default=catalog.roles.incumbent if catalog.roles.incumbent in selected else None,
        )
        if chosen_incumbent.action is PickerAction.CANCEL:
            raise typer.Abort()
        if chosen_incumbent.action is PickerAction.BACK:
            if candidates:
                raise typer.Abort()
            preselected = selected
            continue
        with usage_error(ValueError):
            return RouterCandidateSelection(
                candidates=selected, incumbent=chosen_incumbent.values[0]
            )
