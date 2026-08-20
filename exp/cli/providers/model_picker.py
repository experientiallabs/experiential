"""Role-first model assignment and confirmation screens for provider setup.

These screens run after every selected provider has been prepared. Setup asks for each model
role in turn: world model, judge, embedder, then optional router candidates and their incumbent.
Every screen uses the shared picker with concise shorthand aliases, sorted so the recommended
model for each role comes first. The screens filter new build-role assignments to models whose
verified metadata can serve them, preserve exact prior assignments as retain-only choices, ask
for a role-specific reasoning effort directly after each completion role names a reasoning-capable
model, and render the single compact summary shown before setup saves anything.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import get_args

from rich.console import Console
from rich.prompt import Confirm

from exp.cli.providers.provider_picker import (
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
from exp.cli.shared.picker import PickerAction, PickerOption, choose_many, choose_one
from exp.common.models import (
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
from exp.common.models.known_models import recommended_model_rank

_MANUAL_MODEL_ROW = "declare-model-manually"
_NO_REASONING_EFFORT = "none"
_REASONING_EFFORTS: tuple[ReasoningEffort, ...] = get_args(ReasoningEffort)


@dataclass(frozen=True)
class RoleAssignment:
    """Confirmed build role assignments and role-specific reasoning efforts from one session."""

    world_model: str
    judge: str
    embedder: str
    candidates: tuple[str, ...] = ()
    incumbent: str | None = None
    world_model_reasoning_effort: ReasoningEffort | None = None
    judge_reasoning_effort: ReasoningEffort | None = None
    candidate_reasoning_efforts: dict[str, ReasoningEffort] = field(default_factory=dict)


def available_models(session: SetupSession) -> tuple[AvailableModel, ...]:
    """List every configurable model, including models declared by hand."""
    return (*session.available, *session.manual)


def _option(item: AvailableModel) -> PickerOption:
    """Present one configurable model as a selectable picker row."""
    return PickerOption(value=item.alias, label=item.label(), detail=item.detail())


def _role_options(items: tuple[AvailableModel, ...]) -> list[PickerOption]:
    """Present eligible models by shorthand alias, keeping only retain-only notes."""
    return [
        PickerOption(value=item.alias, label=item.label(), detail=_role_detail(item))
        for item in items
    ]


def _role_detail(item: AvailableModel) -> str:
    """Annotate one role row, keeping only retain-only notes."""
    if item.capabilities is None:
        roles = ", ".join(sorted(role.value for role in item.retainable_roles))
        return f"retain only: {roles}" if roles else "unverified"
    return ""


def recommendation_key(
    provider: str,
    model: str,
    capabilities: ModelCapabilities,
    role: SetupRole,
) -> tuple[int, int, int, float, str, str]:
    """Rank one verified model by maintained guidance, then capability and cost.

    Args:
        provider: Provider kind publishing the model.
        model: Exact provider-side model ID.
        capabilities: Verified capability and price metadata.
        role: Role being filled.

    Returns:
        Stable sort key preferring maintained provider guidance before a cost fallback.
    """
    rank = recommended_model_rank(provider, model, role.value)
    provider_rank = {"openai": 0, "anthropic": 1, "gemini": 2}.get(provider, 3)
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
        provider,
        model,
    )


def _role_ordered(
    items: tuple[AvailableModel, ...],
    role: SetupRole,
) -> tuple[AvailableModel, ...]:
    """Order eligible models so the recommended choice for one role comes first.

    Args:
        items: Eligible models for the role.
        role: Role being assigned.

    Returns:
        Verified models in recommendation order, then retain-only models by alias.
    """

    def key(item: AvailableModel) -> tuple[int, tuple[int, int, int, float, str, str]]:
        """Sort verified models by recommendation and retain-only models after them."""
        if item.capabilities is None:
            return (1, (0, 0, 0, 0.0, item.provider, item.model))
        return (0, recommendation_key(item.provider, item.model, item.capabilities, role))

    return tuple(sorted(items, key=key))


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
            title="Models to configure",
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


class _RoleEffortBack:
    """Sentinel marking a back navigation from one role effort screen."""


_ROLE_EFFORT_BACK = _RoleEffortBack()


def _ask_role_reasoning_effort(
    chosen: tuple[AvailableModel, ...],
    *,
    alias: str,
    role_name: str,
    default: ReasoningEffort | None,
    console: Console,
) -> ReasoningEffort | None | _RoleEffortBack:
    """Choose the reasoning effort one completion role uses for its just-selected model.

    The screen appears only when the selected model's verified capabilities carry a reasoning
    pin; unsupported and unverified models are never asked and keep no role effort.

    Args:
        chosen: Models the user selected.
        alias: Model just assigned to the role.
        role_name: Readable role name shown in the screen title.
        default: Prior role effort accepted with an empty line.
        console: Terminal used for the screen.

    Returns:
        The chosen effort, ``None`` when the model carries no reasoning pin, or the back
        sentinel when the user asked to go back.

    Raises:
        SetupCancelled: The user cancelled setup.
    """
    item = next((entry for entry in chosen if entry.alias == alias), None)
    if item is None or item.capabilities is None or item.capabilities.reasoning_effort is None:
        return None
    result = choose_one(
        console,
        title=f"{role_name.capitalize()} effort ({alias})",
        options=[PickerOption(value=effort, label=effort) for effort in _REASONING_EFFORTS],
        default=default or item.capabilities.reasoning_effort,
    )
    if result.action is PickerAction.CANCEL:
        raise SetupCancelled
    if result.action is PickerAction.BACK:
        return _ROLE_EFFORT_BACK
    return _parse_reasoning_effort(result.values[0])


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
    world_effort = _ask_role_reasoning_effort(
        chosen,
        alias=world_model,
        role_name="world model",
        default=role_inputs.world_model_reasoning_effort
        if world_model == role_inputs.world_model
        else None,
        console=console,
    )
    if isinstance(world_effort, _RoleEffortBack):
        return None
    judge = _assign_one_role(
        chosen,
        role=SetupRole.JUDGE,
        title="Judge",
        role_name="judge",
        default=role_inputs.judge,
        console=console,
    )
    if judge is None:
        return None
    judge_effort = _ask_role_reasoning_effort(
        chosen,
        alias=judge,
        role_name="judge",
        default=role_inputs.judge_reasoning_effort if judge == role_inputs.judge else None,
        console=console,
    )
    if isinstance(judge_effort, _RoleEffortBack):
        return None
    embedder = _assign_one_role(
        chosen,
        role=SetupRole.EMBEDDER,
        title="Embedder",
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
        world_model_reasoning_effort=world_effort,
        judge_reasoning_effort=judge_effort,
        candidate_reasoning_efforts=candidates[2],
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
    eligible = _role_ordered(
        tuple(item for item in chosen if _serves_or_retains(item, role)),
        role,
    )
    if not eligible:
        console.print(
            f"[yellow]No available model can serve the {role_name} role. "
            "Choose another provider.[/yellow]"
        )
        return None
    result = choose_one(
        console,
        title=title,
        options=_role_options(eligible),
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
) -> tuple[tuple[str, ...], str | None, dict[str, ReasoningEffort]] | None:
    """Choose optional router candidates, their efforts, and the incumbent.

    Args:
        chosen: Models the user selected.
        role_inputs: Prior candidate roles and efforts accepted with an empty line.
        console: Terminal used for the screens.

    Returns:
        Candidate aliases, incumbent, and per-candidate reasoning efforts, or ``None`` when
        the selection must change.

    Raises:
        SetupCancelled: The user cancelled setup.
    """
    return _assign_router_candidates(
        chosen,
        preselected=role_inputs.candidates,
        incumbent=role_inputs.incumbent,
        effort_defaults=role_inputs.candidate_reasoning_efforts,
        console=console,
        required=False,
    )


def select_router_candidates(
    models: tuple[AvailableModel, ...],
    *,
    preselected: tuple[str, ...] = (),
    incumbent: str | None = None,
    effort_defaults: Mapping[str, ReasoningEffort] | None = None,
    console: Console,
) -> RouterCandidateSelection | None:
    """Choose at least two eligible completion models, their efforts, and one incumbent.

    Args:
        models: Configured and newly discovered models offered by provider setup.
        preselected: Candidate aliases to retain in the multi-select screen.
        incumbent: Explicit incumbent that skips the incumbent screen.
        effort_defaults: Persisted per-candidate efforts accepted with an empty line.
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
        effort_defaults={} if effort_defaults is None else effort_defaults,
        console=console,
        required=True,
    )
    if picked is None or picked[1] is None:
        return None
    return RouterCandidateSelection(
        candidates=picked[0],
        incumbent=picked[1],
        candidate_reasoning_efforts=picked[2],
    )


def _assign_router_candidates(
    chosen: tuple[AvailableModel, ...],
    *,
    preselected: tuple[str, ...],
    incumbent: str | None,
    effort_defaults: Mapping[str, ReasoningEffort],
    console: Console,
    required: bool,
) -> tuple[tuple[str, ...], str | None, dict[str, ReasoningEffort]] | None:
    """Share router-candidate selection between optional setup and required router setup.

    Args:
        chosen: Models currently visible to the picker.
        preselected: Candidate aliases to preselect.
        incumbent: Explicit incumbent, or ``None`` to show the incumbent screen.
        effort_defaults: Persisted per-candidate efforts accepted with an empty line.
        console: Terminal used for the screens.
        required: Whether fewer than two selected candidates is an error instead of a skip.

    Returns:
        Candidate aliases, incumbent, and per-candidate reasoning efforts, or ``None`` when
        the caller should go back.

    Raises:
        SetupCancelled: The user cancelled interactive setup.
        ValueError: An explicit candidate or incumbent is not eligible.
    """
    eligible = _role_ordered(
        tuple(
            item
            for item in chosen
            if (
                item.capabilities is not None
                and serves_role(item.capabilities, SetupRole.ROUTER_CANDIDATE)
            )
            or (not required and SetupRole.ROUTER_CANDIDATE in item.retainable_roles)
        ),
        SetupRole.ROUTER_CANDIDATE,
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
        return (), None, {}
    eligible_aliases = {item.alias for item in eligible}
    unknown = tuple(alias for alias in preselected if alias not in eligible_aliases)
    if unknown:
        raise ValueError(
            "router candidate aliases are not eligible completion models: "
            + ", ".join(sorted(set(unknown)))
        )
    result = choose_many(
        console,
        title=("Router candidates (2+)" if required else "Router candidates (optional)"),
        options=_role_options(eligible),
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
        return (), None, {}
    efforts: dict[str, ReasoningEffort] = {}
    for alias in result.values:
        effort = _ask_role_reasoning_effort(
            chosen,
            alias=alias,
            role_name="router candidate",
            default=effort_defaults.get(alias),
            console=console,
        )
        if isinstance(effort, _RoleEffortBack):
            return None
        if effort is not None:
            efforts[alias] = effort
    if incumbent is not None:
        if incumbent not in result.values:
            raise ValueError("router incumbent must also be a selected candidate")
        return result.values, incumbent, efforts
    incumbent_result = choose_one(
        console,
        title="Router incumbent",
        options=(
            _role_options(tuple(item for item in eligible if item.alias in result.values))
            if required
            else [PickerOption(value=alias, label=alias) for alias in result.values]
        ),
        default=incumbent,
    )
    if incumbent_result.action is PickerAction.CANCEL:
        raise SetupCancelled
    if incumbent_result.action is PickerAction.BACK:
        return None
    return result.values, incumbent_result.values[0], efforts


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
        world_model_reasoning_effort=roles.world_model_reasoning_effort,
        judge_reasoning_effort=roles.judge_reasoning_effort,
    )
    return ProviderSetupResult(
        setup=setup,
        candidates=roles.candidates,
        incumbent=roles.incumbent,
        candidate_reasoning_efforts=dict(roles.candidate_reasoning_efforts),
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
    endpoints: tuple[PreparedEndpoint, ...],
    console: Console,
) -> None:
    """Show the providers, models, and roles about to be saved, one compact line each.

    Args:
        result: The setup about to be saved.
        endpoints: Prepared provider endpoints.
        console: Terminal receiving the summary.
    """
    console.print("[bold]Configuration[/bold]")
    for endpoint in endpoints:
        connection = endpoint.connection
        endpoint_text = f"  [dim]({connection.base_url})[/dim]" if connection.base_url else ""
        console.print(f"  [green]\u2713[/green] {connection.provider}{endpoint_text}")
    setup = result.setup

    def line(label: str, alias: str | None, effort: ReasoningEffort | None = None) -> None:
        """Print one aligned role line with the alias and its dim reasoning effort."""
        if alias is None:
            return
        note = f"  [dim](effort {effort})[/dim]" if effort is not None else ""
        console.print(f"  [dim]{label:<12}[/dim] {alias}{note}")

    line("world model", setup.world_model, setup.world_model_reasoning_effort)
    line("judge", setup.judge, setup.judge_reasoning_effort)
    line("embedder", setup.embedder)
    if result.candidates:
        described = ", ".join(
            f"{alias} [dim](effort {result.candidate_reasoning_efforts[alias]})[/dim]"
            if alias in result.candidate_reasoning_efforts
            else alias
            for alias in result.candidates
        )
        console.print(
            f"  [dim]{'router':<12}[/dim] {described} [dim](incumbent {result.incumbent})[/dim]"
        )
