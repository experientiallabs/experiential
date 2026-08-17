"""Model selection, role assignment, and confirmation screens for provider setup.

These screens run after every selected provider has been prepared. They present the discovered and
already-configured models as one searchable list, filter new build-role assignments to models whose
verified metadata can serve them, preserve exact prior assignments as retain-only choices, and
render the single summary shown before setup saves anything.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from rich.console import Console
from rich.prompt import Confirm

from wmo.cli.picker import PickerAction, PickerOption, select_many, select_one
from wmo.cli.provider_picker import (
    CREDENTIAL_NOTE,
    AvailableModel,
    PreparedEndpoint,
    ProviderSetupResult,
    SetupCancelled,
    SetupRoleInputs,
    SetupSession,
    ask_positive_int,
    ask_price,
    ask_text,
)
from wmo.common.models import (
    ModelCapabilities,
    ModelRecord,
    PricingSource,
    ProviderConnection,
    ProviderModelSelection,
    ProviderSetup,
    SetupRole,
    derive_model_alias,
    served_roles,
    serves_role,
)

_MANUAL_MODEL_ROW = "declare-model-manually"


@dataclass(frozen=True)
class RoleAssignment:
    """Confirmed build role assignments from one setup session."""

    world_model: str
    judge: str
    embedder: str
    candidates: tuple[str, ...] = ()
    incumbent: str | None = None


def available_models(session: SetupSession) -> tuple[AvailableModel, ...]:
    """List every configurable model, including models declared by hand."""
    return (*session.available, *session.manual)


def select_models(session: SetupSession, *, console: Console) -> tuple[str, ...] | None:
    """Show the searchable multi-select model screen across every prepared provider.

    Args:
        session: Answers already collected in this setup session.
        console: Terminal used for the screen.

    Returns:
        Selected aliases, or ``None`` when the user asked to change providers.

    Raises:
        SetupCancelled: The user cancelled setup.
    """
    while True:
        available = available_models(session)
        options = [
            PickerOption(value=item.alias, label=item.label(), detail=item.detail())
            for item in available
        ]
        if session.advanced_models:
            options.append(
                PickerOption(
                    value=_MANUAL_MODEL_ROW,
                    label="Advanced: declare another model by hand",
                )
            )
        result = select_many(
            console,
            title="Select the models to configure",
            options=options,
            preselected=session.selected,
        )
        if result.action is PickerAction.CANCEL:
            raise SetupCancelled
        if result.action is PickerAction.BACK:
            return None
        if _MANUAL_MODEL_ROW in result.values:
            session.selected = tuple(v for v in result.values if v != _MANUAL_MODEL_ROW)
            declared = declare_model(session, console=console)
            if declared is not None:
                session.manual.append(declared)
                session.selected = (*session.selected, declared.alias)
            continue
        return result.values


def declare_model(session: SetupSession, *, console: Console) -> AvailableModel | None:
    """Declare one model and its capabilities by hand under the advanced path.

    Args:
        session: Prepared endpoints and aliases already used in this session.
        console: Terminal used for prompts.

    Returns:
        The declared model, or ``None`` when no prepared connection can host it.

    Raises:
        SetupCancelled: The user cancelled setup at a prompt.
    """
    if not session.endpoints:
        console.print("[yellow]Prepare a provider connection first.[/yellow]")
        return None
    connections = [
        PickerOption(
            value=endpoint.connection.name,
            label=endpoint.connection.name,
            detail=f"{endpoint.connection.provider}",
        )
        for endpoint in session.endpoints
    ]
    chosen = select_one(console, title="Connection for the declared model", options=connections)
    if chosen.action is PickerAction.CANCEL:
        raise SetupCancelled
    if chosen.action is PickerAction.BACK:
        return None
    connection_name = chosen.values[0]
    provider = next(
        endpoint.connection.provider
        for endpoint in session.endpoints
        if endpoint.connection.name == connection_name
    )
    model = ask_text("Provider model ID", console=console)
    if not model:
        return None
    supports_completions = Confirm.ask("Supports chat completions?", default=True, console=console)
    supports_embeddings = Confirm.ask("Supports embeddings?", default=False, console=console)
    capabilities = ModelCapabilities(
        supports_completions=supports_completions,
        supports_embeddings=supports_embeddings,
        supports_tools=Confirm.ask("Supports tools?", default=False, console=console),
        supports_structured_output=Confirm.ask(
            "Supports structured output?", default=False, console=console
        ),
        context_window_tokens=ask_positive_int("Context window tokens", console=console),
        maximum_output_tokens=ask_positive_int("Maximum output tokens", console=console),
        input_cost_per_million_tokens_usd=ask_price(
            "Input cost per million tokens in USD", console=console
        )
        if supports_completions or supports_embeddings
        else None,
        output_cost_per_million_tokens_usd=ask_price(
            "Output cost per million tokens in USD", console=console
        )
        if supports_completions
        else None,
        cached_input_cost_per_million_tokens_usd=ask_price(
            "Cached input cost per million tokens in USD", console=console
        )
        if supports_completions
        else None,
        cache_write_cost_per_million_tokens_usd=ask_price(
            "Cache write cost per million tokens in USD", console=console
        )
        if supports_completions
        else None,
    )
    taken = frozenset(item.alias for item in available_models(session))
    return AvailableModel(
        alias=derive_model_alias(provider, model, taken),
        connection=connection_name,
        provider=provider,
        model=model,
        capabilities=capabilities,
        pricing_source=PricingSource.CONFIGURED,
        configured=False,
    )


def assign_roles(
    chosen: tuple[AvailableModel, ...],
    *,
    role_inputs: SetupRoleInputs,
    console: Console,
) -> RoleAssignment | None:
    """Assign every build role from the selected models only.

    Args:
        chosen: Models the user selected.
        role_inputs: Role values supplied by flags or already persisted.
        console: Terminal used for the role screens.

    Returns:
        Confirmed roles, or ``None`` when the user asked to change the model selection.

    Raises:
        SetupCancelled: The user cancelled setup.
    """
    world_model = _assign_one_role(
        chosen,
        role=SetupRole.WORLD_MODEL,
        title="World model",
        role_name="world model",
        default=role_inputs.world_model,
        console=console,
    )
    if world_model is None:
        return None
    judge = _assign_one_role(
        chosen,
        role=SetupRole.JUDGE,
        title="Judge model",
        role_name="judge",
        default=role_inputs.judge,
        console=console,
    )
    if judge is None:
        return None
    embedder = _assign_one_role(
        chosen,
        role=SetupRole.EMBEDDER,
        title="Embedder model",
        role_name="embedder",
        default=role_inputs.embedder,
        console=console,
    )
    if embedder is None:
        return None
    candidates = _assign_candidates(chosen, role_inputs=role_inputs, console=console)
    if candidates is None:
        return None
    return RoleAssignment(
        world_model=world_model,
        judge=judge,
        embedder=embedder,
        candidates=candidates[0],
        incumbent=candidates[1],
    )


def _assign_one_role(
    chosen: tuple[AvailableModel, ...],
    *,
    role: SetupRole,
    title: str,
    role_name: str,
    default: str | None,
    console: Console,
) -> str | None:
    """Choose one model for a role from the compatible selected models.

    Args:
        chosen: Models the user selected.
        role: Build role being assigned.
        title: Screen heading for the role.
        role_name: Readable role name used when no selected model can serve it.
        default: Prior alias accepted with an empty line.
        console: Terminal used for the screen.

    Returns:
        The chosen alias, or ``None`` when the model selection must change.

    Raises:
        SetupCancelled: The user cancelled setup.
    """
    eligible = tuple(item for item in chosen if _serves_or_retains(item, role))
    if not eligible:
        console.print(
            f"[yellow]No selected model can serve the {role_name} role. "
            "Select more models.[/yellow]"
        )
        return None
    result = select_one(
        console,
        title=title,
        options=[
            PickerOption(value=item.alias, label=item.label(), detail=item.detail())
            for item in eligible
        ],
        default=default,
    )
    if result.action is PickerAction.CANCEL:
        raise SetupCancelled
    if result.action is PickerAction.BACK:
        return None
    return result.values[0]


def _assign_candidates(
    chosen: tuple[AvailableModel, ...],
    *,
    role_inputs: SetupRoleInputs,
    console: Console,
) -> tuple[tuple[str, ...], str | None] | None:
    """Choose optional router candidates and their incumbent from the selected models.

    Args:
        chosen: Models the user selected.
        role_inputs: Prior candidate roles accepted with an empty line.
        console: Terminal used for the screens.

    Returns:
        Candidate aliases with their incumbent, or ``None`` when the selection must change.

    Raises:
        SetupCancelled: The user cancelled setup.
    """
    eligible = tuple(
        item for item in chosen if _serves_or_retains(item, SetupRole.ROUTER_CANDIDATE)
    )
    if len(eligible) < 2:
        console.print(
            "[dim]Router candidates need two priced models with explicit token limits. "
            "Skipping that role for now.[/dim]"
        )
        return (), None
    result = select_many(
        console,
        title="Router candidates (optional, empty line skips)",
        options=[
            PickerOption(value=item.alias, label=item.label(), detail=item.detail())
            for item in eligible
        ],
        preselected=role_inputs.candidates,
        minimum=0,
    )
    if result.action is PickerAction.CANCEL:
        raise SetupCancelled
    if result.action is PickerAction.BACK:
        return None
    if len(result.values) < 2:
        console.print("[dim]Router candidates need at least two models. Skipping that role.[/dim]")
        return (), None
    incumbent = select_one(
        console,
        title="Router incumbent among the candidates",
        options=[PickerOption(value=alias, label=alias) for alias in result.values],
        default=role_inputs.incumbent,
    )
    if incumbent.action is PickerAction.CANCEL:
        raise SetupCancelled
    if incumbent.action is PickerAction.BACK:
        return None
    return result.values, incumbent.values[0]


def build_result(
    chosen: tuple[AvailableModel, ...],
    *,
    roles: RoleAssignment,
    endpoints: tuple[PreparedEndpoint, ...],
    existing_connections: tuple[ProviderConnection, ...],
    existing_models: tuple[ProviderModelSelection, ...],
    known_existing_connections: tuple[str, ...] | None = None,
    known_existing_aliases: tuple[str, ...] | None = None,
) -> ProviderSetupResult:
    """Build the catalog update from confirmed models and roles.

    Args:
        chosen: Models the user selected.
        roles: Confirmed role assignments.
        endpoints: Prepared provider endpoints.
        existing_connections: Connections already configured in the catalog.
        existing_models: Model aliases already configured in the catalog.
        known_existing_connections: Every connection name in the persisted catalog. When omitted,
            the setup-compatible connection records supply the names.
        known_existing_aliases: Every model alias in the persisted catalog. When omitted, the
            setup-compatible model selections supply the aliases.

    Returns:
        The setup to merge plus any router roles to assign after it.
    """
    configured_aliases = set(
        (model.alias for model in existing_models)
        if known_existing_aliases is None
        else known_existing_aliases
    )
    used_connections = {item.connection for item in chosen}
    setup = ProviderSetup(
        connections=tuple(
            endpoint.connection
            for endpoint in endpoints
            if not endpoint.configured and endpoint.connection.name in used_connections
        ),
        models=tuple(
            model_selection(item) for item in chosen if item.alias not in configured_aliases
        ),
        known_existing_connections=(
            tuple(item.name for item in existing_connections)
            if known_existing_connections is None
            else known_existing_connections
        ),
        known_existing_aliases=tuple(sorted(configured_aliases)),
        world_model=roles.world_model,
        judge=roles.judge,
        embedder=roles.embedder,
    )
    return ProviderSetupResult(
        setup=setup,
        candidates=roles.candidates,
        incumbent=roles.incumbent,
    )


def model_selection(item: AvailableModel) -> ProviderModelSelection:
    """Convert one configurable model into its persisted setup selection."""
    capabilities = item.capabilities
    if capabilities is None:
        raise ValueError(f"new model alias {item.alias!r} needs explicit capabilities")
    return ProviderModelSelection(
        alias=item.alias,
        connection=item.connection,
        model=item.model,
        capabilities=capabilities,
    )


def configured_models(
    existing_models: Mapping[str, ModelRecord],
    *,
    connection_providers: Mapping[str, str],
    retainable_roles: Mapping[str, frozenset[SetupRole]] | None = None,
) -> tuple[AvailableModel, ...]:
    """Present already-configured catalog aliases as selectable models.

    Args:
        existing_models: Exact model records keyed by every persisted alias.
        connection_providers: Exact provider kind keyed by every persisted connection name.
        retainable_roles: Exact existing role assignments that incomplete aliases may retain.

    Returns:
        Configurable records for aliases with usable metadata or an exact role to retain.
    """
    retained = {} if retainable_roles is None else retainable_roles
    records = []
    for alias, model in sorted(existing_models.items()):
        capabilities = model.capabilities
        roles = retained.get(alias, frozenset())
        if (capabilities is None or not served_roles(capabilities)) and not roles:
            continue
        records.append(
            AvailableModel(
                alias=alias,
                connection=model.connection,
                provider=connection_providers[model.connection],
                model=model.model,
                capabilities=capabilities,
                pricing_source=PricingSource.CONFIGURED,
                configured=True,
                retainable_roles=roles,
            )
        )
    return tuple(records)


def _serves_or_retains(item: AvailableModel, role: SetupRole) -> bool:
    """Report whether verified metadata serves a role or the exact prior binding retains it.

    Args:
        item: Selected model with optional verified capabilities and prior role bindings.
        role: Role being assigned.

    Returns:
        ``True`` for verified compatibility or an exact retain-only prior assignment.
    """
    return (
        item.capabilities is not None and serves_role(item.capabilities, role)
    ) or role in item.retainable_roles


def render_summary(
    result: ProviderSetupResult,
    *,
    chosen: tuple[AvailableModel, ...],
    endpoints: tuple[PreparedEndpoint, ...],
    console: Console,
) -> None:
    """Show providers, models, roles, capabilities, prices, and credential behavior once.

    Args:
        result: The setup about to be saved.
        chosen: Models the user selected.
        endpoints: Prepared provider endpoints.
        console: Terminal receiving the summary.
    """
    console.print("[bold]Configuration summary[/bold]")
    for endpoint in endpoints:
        connection = endpoint.connection
        endpoint_text = f", base_url={connection.base_url}" if connection.base_url else ""
        credential = connection.api_key_env or "AWS credential chain"
        console.print(
            f"provider {connection.provider}: connection {connection.name}, "
            f"credential {credential}{endpoint_text}"
        )
    for item in chosen:
        capabilities = item.capabilities
        if capabilities is None:
            retained = ", ".join(role.value for role in item.retainable_roles)
            console.print(
                f"model {item.alias}: {item.provider}/{item.model}, "
                f"capabilities=unverified, retain_only={retained or 'none'}, "
                f"pricing={item.pricing_source.value}"
            )
            continue
        verified = frozenset(served_roles(capabilities))
        retain_only = ", ".join(
            role.value for role in SetupRole if role in item.retainable_roles - verified
        )
        retained = f", retain_only={retain_only}" if retain_only else ""
        console.print(
            f"model {item.alias}: {item.provider}/{item.model}, "
            f"tools={capabilities.supports_tools}, "
            f"embeddings={capabilities.supports_embeddings}, "
            f"structured_output={capabilities.supports_structured_output}, "
            f"completions={capabilities.supports_completions}, "
            f"context={capabilities.context_window_tokens}, "
            f"max_output={capabilities.maximum_output_tokens}, "
            f"pricing={item.pricing_source.value}{retained}"
        )
    setup = result.setup
    console.print(
        f"roles: world_model={setup.world_model}, judge={setup.judge}, embedder={setup.embedder}"
    )
    if result.candidates:
        console.print(
            f"router candidates: {', '.join(result.candidates)}; incumbent {result.incumbent}"
        )
    console.print(f"credentials: {CREDENTIAL_NOTE}")
