"""Model selection, role assignment, and confirmation screens for provider setup.

These screens run after every selected provider has been prepared. Every one of them uses the
shared picker: the model screen and the router-candidate screen are multi-select, and a single build
role and the router incumbent are single-select. Each row keeps its provider identity, served roles,
and pricing provenance visible. The screens filter new build-role assignments to models whose
verified metadata can serve them, preserve exact prior assignments as retain-only choices, and
render the single summary shown before setup saves anything.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import get_args

from rich.console import Console
from rich.prompt import Confirm

from wmo.cli.picker import PickerAction, PickerOption, choose_many, choose_one
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
    ProviderModelSelection,
    ProviderSetup,
    ReasoningEffort,
    RouterCandidateSelection,
    SetupRole,
    derive_model_alias,
    served_roles,
    serves_role,
)

_MANUAL_MODEL_ROW = "declare-model-manually"
_NO_REASONING_EFFORT = "none"
_REASONING_EFFORTS: tuple[ReasoningEffort, ...] = get_args(ReasoningEffort)


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


def _option(item: AvailableModel) -> PickerOption:
    """Present one configurable model as a selectable picker row."""
    return PickerOption(value=item.alias, label=item.label(), detail=item.detail())


def select_models(session: SetupSession, *, console: Console) -> tuple[str, ...] | None:
    """Show the multi-select model screen across every prepared provider.

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
        options = [_option(item) for item in available]
        if session.advanced_models:
            options.append(
                PickerOption(
                    value=_MANUAL_MODEL_ROW,
                    label="Advanced: declare another model by hand",
                )
            )
        result = choose_many(
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
    chosen = choose_one(console, title="Connection for the declared model", options=connections)
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
        reasoning_effort=_ask_reasoning_effort(console=console) if supports_completions else None,
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


def _parse_reasoning_effort(value: str) -> ReasoningEffort | None:
    """Return the reasoning effort named by a picker value, or ``None`` for no pin."""
    return next((effort for effort in _REASONING_EFFORTS if effort == value), None)


def _ask_reasoning_effort(*, console: Console) -> ReasoningEffort | None:
    """Choose an optional reasoning-effort pin for one manually declared completion model.

    Args:
        console: Terminal used for the screen.

    Returns:
        The chosen effort, or ``None`` so the parameter is never sent.

    Raises:
        SetupCancelled: The user cancelled setup.
    """
    result = choose_one(
        console,
        title="Reasoning effort (pin only for models accepting the OpenAI reasoning parameter)",
        options=[
            PickerOption(
                value=_NO_REASONING_EFFORT,
                label="none",
                detail="never send the reasoning parameter",
            ),
            *(PickerOption(value=effort, label=effort) for effort in _REASONING_EFFORTS),
        ],
        default=_NO_REASONING_EFFORT,
    )
    if result.action is PickerAction.CANCEL:
        raise SetupCancelled
    if result.action is PickerAction.BACK:
        return None
    return _parse_reasoning_effort(result.values[0])


def select_reasoning_efforts(
    chosen: tuple[AvailableModel, ...],
    *,
    roles: RoleAssignment,
    console: Console,
) -> tuple[AvailableModel, ...] | None:
    """Pin one reasoning effort per new reasoning-capable model holding a completion role.

    Only newly selected models that were just assigned a world-model, judge, or router-candidate
    role and whose verified metadata proves reasoning-effort support show a screen; each defaults
    to the entry's current pin. Already-configured aliases keep their persisted pin, and models
    without proven support are never asked and never pinned.

    Args:
        chosen: Models the user selected.
        roles: Confirmed role assignments naming the models to ask about.
        console: Terminal used for the screens.

    Returns:
        The selection with confirmed pins, or ``None`` when the user asked to go back.

    Raises:
        SetupCancelled: The user cancelled setup.
    """
    role_aliases = frozenset((roles.world_model, roles.judge, *roles.candidates))
    updated = list(chosen)
    for position, item in enumerate(chosen):
        capabilities = item.capabilities
        if (
            item.alias not in role_aliases
            or item.configured
            or capabilities is None
            or capabilities.reasoning_effort is None
        ):
            continue
        result = choose_one(
            console,
            title=f"Reasoning effort for {item.alias}",
            options=[PickerOption(value=effort, label=effort) for effort in _REASONING_EFFORTS],
            default=capabilities.reasoning_effort,
        )
        if result.action is PickerAction.CANCEL:
            raise SetupCancelled
        if result.action is PickerAction.BACK:
            return None
        effort = _parse_reasoning_effort(result.values[0])
        if effort != capabilities.reasoning_effort:
            updated[position] = replace(
                item,
                capabilities=capabilities.model_copy(update={"reasoning_effort": effort}),
            )
    return tuple(updated)


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
    result = choose_one(
        console,
        title=title,
        options=[_option(item) for item in eligible],
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
    return _assign_router_candidates(
        chosen,
        preselected=role_inputs.candidates,
        incumbent=role_inputs.incumbent,
        console=console,
        required=False,
    )


def select_router_candidates(
    models: tuple[AvailableModel, ...],
    *,
    preselected: tuple[str, ...] = (),
    incumbent: str | None = None,
    console: Console,
) -> RouterCandidateSelection | None:
    """Choose at least two eligible completion models and one incumbent.

    Args:
        models: Configured and newly discovered models offered by provider setup.
        preselected: Candidate aliases to retain in the multi-select screen.
        incumbent: Explicit incumbent that skips the incumbent screen.
        console: Terminal used for the screens.

    Returns:
        A confirmed candidate selection, or ``None`` when the caller should return to its
        previous setup screen.

    Raises:
        SetupCancelled: The user cancelled interactive setup.
        ValueError: An explicit candidate or incumbent is not eligible.
    """
    picked = _assign_router_candidates(
        models,
        preselected=preselected,
        incumbent=incumbent,
        console=console,
        required=True,
    )
    if picked is None or picked[1] is None:
        return None
    return RouterCandidateSelection(candidates=picked[0], incumbent=picked[1])


def _assign_router_candidates(
    chosen: tuple[AvailableModel, ...],
    *,
    preselected: tuple[str, ...],
    incumbent: str | None,
    console: Console,
    required: bool,
) -> tuple[tuple[str, ...], str | None] | None:
    """Share router-candidate selection between optional setup and required router setup.

    Args:
        chosen: Models currently visible to the picker.
        preselected: Candidate aliases to preselect.
        incumbent: Explicit incumbent, or ``None`` to show the incumbent screen.
        console: Terminal used for the screens.
        required: Whether fewer than two selected candidates is an error instead of a skip.

    Returns:
        Candidate aliases and incumbent, or ``None`` when the caller should go back.

    Raises:
        SetupCancelled: The user cancelled interactive setup.
        ValueError: An explicit candidate or incumbent is not eligible.
    """
    eligible = tuple(
        item
        for item in chosen
        if (
            item.capabilities is not None
            and serves_role(item.capabilities, SetupRole.ROUTER_CANDIDATE)
        )
        or (not required and SetupRole.ROUTER_CANDIDATE in item.retainable_roles)
    )
    if len(eligible) < 2:
        if required:
            console.print(
                "[yellow]Router optimization needs at least two eligible completion models. "
                "Choose another configured provider.[/yellow]"
            )
            return None
        console.print(
            "[dim]Router candidates need two priced models with explicit token limits. "
            "Skipping that role for now.[/dim]"
        )
        return (), None
    eligible_aliases = {item.alias for item in eligible}
    unknown = tuple(alias for alias in preselected if alias not in eligible_aliases)
    if unknown:
        raise ValueError(
            "router candidate aliases are not eligible completion models: "
            + ", ".join(sorted(set(unknown)))
        )
    result = choose_many(
        console,
        title=(
            "Router candidates (select at least two)"
            if required
            else "Router candidates (optional, Complete with none skips)"
        ),
        options=[_candidate_option(item) if required else _option(item) for item in eligible],
        preselected=preselected,
        minimum=2 if required else 0,
    )
    if result.action is PickerAction.CANCEL:
        raise SetupCancelled
    if result.action is PickerAction.BACK:
        return None
    if len(result.values) < 2:
        if required:
            console.print("[yellow]Router candidates need at least two models.[/yellow]")
            return None
        console.print("[dim]Router candidates need at least two models. Skipping that role.[/dim]")
        return (), None
    if incumbent is not None:
        if incumbent not in result.values:
            raise ValueError("router incumbent must also be a selected candidate")
        return result.values, incumbent
    incumbent_result = choose_one(
        console,
        title="Router incumbent among the candidates",
        options=(
            [_candidate_option(item) for item in eligible if item.alias in result.values]
            if required
            else [PickerOption(value=alias, label=alias) for alias in result.values]
        ),
        default=incumbent,
    )
    if incumbent_result.action is PickerAction.CANCEL:
        raise SetupCancelled
    if incumbent_result.action is PickerAction.BACK:
        return None
    return result.values, incumbent_result.values[0]


def _candidate_option(item: AvailableModel) -> PickerOption:
    """Present one strict router-candidate option with provider identity and capabilities.

    Args:
        item: Eligible model discovered or retained by provider setup.

    Returns:
        Picker row carrying the alias, provider/model identity, and role metadata.
    """
    return PickerOption(value=item.alias, label=item.label(), detail=item.detail())


def build_result(
    chosen: tuple[AvailableModel, ...],
    *,
    roles: RoleAssignment,
    endpoints: tuple[PreparedEndpoint, ...],
    known_existing_connections: tuple[str, ...],
    known_existing_aliases: tuple[str, ...],
) -> ProviderSetupResult:
    """Build the catalog update from confirmed models and roles.

    Args:
        chosen: Models the user selected.
        roles: Confirmed role assignments.
        endpoints: Prepared provider endpoints.
        known_existing_connections: Every connection name in the persisted catalog.
        known_existing_aliases: Every model alias in the persisted catalog.

    Returns:
        The setup to merge plus any router roles to assign after it.
    """
    configured_aliases = set(known_existing_aliases)
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
        known_existing_connections=known_existing_connections,
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
            f"reasoning_effort={capabilities.reasoning_effort or 'none'}, "
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
