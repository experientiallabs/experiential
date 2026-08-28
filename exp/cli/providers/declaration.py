"""Operator-declared capability and price metadata for discovered model identities.

Discovery may list an identity before any capability or price is proven, either from an
OpenAI-compatible endpoint or from a router whose catalog omits the parameter array. This
module collects only the minimum fields a selected build role needs, confirms published
values when they exist, and never infers tools, structured output, token limits, or prices.
"""

from __future__ import annotations

from typing import get_args

from rich.console import Console
from rich.prompt import Confirm, IntPrompt

from exp.cli.providers.provider_picker import (
    OPERATOR_DECLARED_PROVIDERS,
    UNKNOWN_METADATA_LABEL,
    AvailableModel,
    SetupCancelled,
    SetupSession,
    ask_price,
    ask_text,
)
from exp.cli.shared.picker import PickerAction, PickerOption, choose_one
from exp.common.models import (
    ModelCapabilities,
    PricingSource,
    ReasoningEffort,
    SetupRole,
    derive_model_alias,
    served_roles,
    serves_role,
)

_NO_REASONING_EFFORT = "__unset_reasoning_effort__"
_REASONING_EFFORTS: tuple[ReasoningEffort, ...] = get_args(ReasoningEffort)
_COMPLETION_PRICE_FIELDS = (
    ("input_cost_per_million_tokens_usd", "Input cost per million tokens in USD"),
    ("output_cost_per_million_tokens_usd", "Output cost per million tokens in USD"),
    ("cached_input_cost_per_million_tokens_usd", "Cached input cost per million tokens in USD"),
    ("cache_write_cost_per_million_tokens_usd", "Cache write cost per million tokens in USD"),
)


def role_row_detail(item: AvailableModel) -> str:
    """Annotate one role row without claiming unverified capabilities.

    Args:
        item: Model offered for a build role.

    Returns:
        A retain-only note, an unknown-metadata label, or an empty verified marker.
    """
    if item.capabilities is None or not served_roles(item.capabilities):
        roles = ", ".join(sorted(role.value for role in item.retainable_roles))
        return f"retain only: {roles}" if roles else UNKNOWN_METADATA_LABEL
    return ""


def can_declare_role(item: AvailableModel, role: SetupRole) -> bool:
    """Report whether the operator may declare the minimum metadata for one role.

    Args:
        item: Discovered or already configured model.
        role: Build role the operator selected.

    Returns:
        ``True`` only for a provider whose listing can leave a model unknown, and only when
        that model does not already prove the role.
    """
    if item.provider not in OPERATOR_DECLARED_PROVIDERS:
        return False
    return item.capabilities is None or not serves_role(item.capabilities, role)


def eligible_for_role(item: AvailableModel, role: SetupRole) -> bool:
    """Report whether a model can be assigned a role now or after operator declaration.

    Args:
        item: Model offered for assignment.
        role: Build role being filled.

    Returns:
        ``True`` when verified metadata serves the role, the exact prior binding retains it,
        or the operator can declare the missing fields.
    """
    return (
        (item.capabilities is not None and serves_role(item.capabilities, role))
        or role in item.retainable_roles
        or can_declare_role(item, role)
    )


def merge_declared_models(
    pool: tuple[AvailableModel, ...],
    declared: tuple[AvailableModel, ...],
) -> tuple[AvailableModel, ...]:
    """Replace pool rows with operator-declared metadata for the same alias.

    Args:
        pool: Models visible before declaration.
        declared: Models whose capabilities the operator just confirmed.

    Returns:
        The pool with declared aliases replaced in their original order, then any new aliases.
    """
    by_alias = {item.alias: item for item in pool}
    extra: list[AvailableModel] = []
    for item in declared:
        if item.alias in by_alias:
            by_alias[item.alias] = item
        else:
            extra.append(item)
    return tuple(by_alias[item.alias] for item in pool) + tuple(extra)


def declare_role_metadata(
    item: AvailableModel,
    role: SetupRole,
    *,
    console: Console,
) -> AvailableModel | None:
    """Collect the minimum operator-declared metadata one selected role requires.

    Published values are confirmed. Missing required fields are asked. Advanced capabilities
    and prices that the role does not need stay unknown.

    Args:
        item: Identity-only or incomplete model selected for ``role``.
        role: Build role that needs explicit metadata.
        console: Terminal used for confirmation and numeric prompts.

    Returns:
        The same identity with configured capabilities, or ``None`` when the operator goes back
        or declines a required field.

    Raises:
        SetupCancelled: The operator cancelled setup at a prompt.
    """
    console.print(
        f"[dim]{item.alias} has {UNKNOWN_METADATA_LABEL}. "
        f"Declare the minimum {role.value.replace('_', ' ')} metadata to use it.[/dim]"
    )
    try:
        capabilities = _capabilities_for_role(item, role, console=console)
    except _DeclarationRejected:
        return None
    if capabilities is None:
        return None
    return AvailableModel(
        alias=item.alias,
        connection=item.connection,
        provider=item.provider,
        model=item.model,
        capabilities=capabilities,
        pricing_source=PricingSource.CONFIGURED,
        configured=item.configured,
        retainable_roles=item.retainable_roles,
        published=item.published,
    )


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
        context_window_tokens=_optional_positive_int("Context window tokens", console=console),
        maximum_output_tokens=_optional_positive_int("Maximum output tokens", console=console),
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
        supports_reasoning=False,
        reasoning_effort=None,
    )
    reasoning_effort = _ask_reasoning_effort(console=console) if supports_completions else None
    if reasoning_effort is not None:
        capabilities = capabilities.model_copy(
            update={
                "supports_temperature": False,
                "supports_top_p": False,
                "supports_reasoning": True,
                "reasoning_effort": reasoning_effort,
            }
        )
    taken = frozenset(item.alias for item in (*session.available, *session.manual))
    return AvailableModel(
        alias=derive_model_alias(provider, model, taken),
        connection=connection_name,
        provider=provider,
        model=model,
        capabilities=capabilities,
        pricing_source=PricingSource.CONFIGURED,
        configured=False,
    )


class _DeclarationRejected(Exception):
    """The operator declined a field the selected role requires."""


def _capabilities_for_role(
    item: AvailableModel,
    role: SetupRole,
    *,
    console: Console,
) -> ModelCapabilities | None:
    """Build configured capabilities that satisfy one role and persist its protocol prices.

    Args:
        item: Selected model, including any published or earlier declared fields.
        role: Build role being filled.
        console: Terminal used for prompts.

    Returns:
        Explicit capabilities for the role.

    Raises:
        SetupCancelled: The operator cancelled setup.
        _DeclarationRejected: A required capability was declined.
    """
    base = item.capabilities or ModelCapabilities()
    updates: dict[str, bool | float | int | None] = {}
    if role is SetupRole.EMBEDDER:
        _require_flag(
            item,
            field="supports_embeddings",
            question="Supports embeddings?",
            console=console,
        )
        updates["supports_embeddings"] = True
        updates["input_cost_per_million_tokens_usd"] = _require_price(
            item,
            field="input_cost_per_million_tokens_usd",
            label="Input cost per million tokens in USD",
            console=console,
        )
        return base.model_copy(update=updates)
    _require_flag(
        item,
        field="supports_completions",
        question="Supports chat completions?",
        console=console,
    )
    updates["supports_completions"] = True
    if role is SetupRole.JUDGE:
        _require_flag(
            item,
            field="supports_structured_output",
            question="Supports structured output?",
            console=console,
        )
        updates["supports_structured_output"] = True
    for field, label in _COMPLETION_PRICE_FIELDS:
        updates[field] = _require_price(item, field=field, label=label, console=console)
    if role is SetupRole.ROUTER_CANDIDATE:
        updates["context_window_tokens"] = _require_positive_int(
            item,
            field="context_window_tokens",
            label="Context window tokens",
            console=console,
        )
        updates["maximum_output_tokens"] = _require_positive_int(
            item,
            field="maximum_output_tokens",
            label="Maximum output tokens",
            console=console,
        )
    return base.model_copy(update=updates)


def _require_flag(
    item: AvailableModel,
    *,
    field: str,
    question: str,
    console: Console,
) -> None:
    """Confirm a required boolean capability, using a published value as the default.

    Args:
        item: Selected model.
        field: Capability field name.
        question: Confirmation prompt.
        console: Terminal used for the prompt.

    Raises:
        SetupCancelled: The operator cancelled setup.
        _DeclarationRejected: The operator declined the required capability.
    """
    published = _known_bool(item, field)
    default = True if published is None else published
    try:
        accepted = Confirm.ask(question, default=default, console=console)
    except (EOFError, KeyboardInterrupt) as exc:
        raise SetupCancelled from exc
    if not accepted:
        console.print(f"[yellow]This role requires {question[:-1].casefold()}.[/yellow]")
        raise _DeclarationRejected


def _require_price(
    item: AvailableModel,
    *,
    field: str,
    label: str,
    console: Console,
) -> float:
    """Confirm a published price or read an explicit nonnegative USD-per-million price.

    Args:
        item: Selected model.
        field: Price field name.
        label: Prompt text.
        console: Terminal used for the prompt.

    Returns:
        The confirmed or newly declared price.

    Raises:
        SetupCancelled: The operator cancelled setup.
    """
    published = _known_price(item, field)
    if published is not None:
        try:
            if Confirm.ask(
                f"Use published {label.casefold()} {published:g}?",
                default=True,
                console=console,
            ):
                return published
        except (EOFError, KeyboardInterrupt) as exc:
            raise SetupCancelled from exc
    return ask_price(label, console=console)


def _require_positive_int(
    item: AvailableModel,
    *,
    field: str,
    label: str,
    console: Console,
) -> int:
    """Confirm a published positive limit or read a new one.

    Args:
        item: Selected model.
        field: Limit field name.
        label: Prompt text.
        console: Terminal used for the prompt.

    Returns:
        The confirmed or newly declared positive integer.

    Raises:
        SetupCancelled: The operator cancelled setup.
        _DeclarationRejected: The entered value is not a positive integer.
    """
    published = _known_positive_int(item, field)
    if published is not None:
        try:
            if Confirm.ask(
                f"Use published {label.casefold()} {published}?",
                default=True,
                console=console,
            ):
                return published
        except (EOFError, KeyboardInterrupt) as exc:
            raise SetupCancelled from exc
    try:
        value = IntPrompt.ask(label, console=console)
    except (EOFError, KeyboardInterrupt) as exc:
        raise SetupCancelled from exc
    if value <= 0:
        console.print(f"[yellow]{label} must be a positive integer.[/yellow]")
        raise _DeclarationRejected
    return value


def _optional_positive_int(label: str, *, console: Console) -> int | None:
    """Read one optional positive integer under the advanced declaration path.

    Args:
        label: Prompt text.
        console: Terminal used for the prompt.

    Returns:
        The positive value, or ``None`` when the field is left unknown.

    Raises:
        SetupCancelled: The prompt reached end of input.
    """
    try:
        if not Confirm.ask(f"Record {label.casefold()}?", default=False, console=console):
            return None
        value = IntPrompt.ask(label, console=console)
    except (EOFError, KeyboardInterrupt) as exc:
        raise SetupCancelled from exc
    return value if value > 0 else None


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
                label="unset",
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
    return next((effort for effort in _REASONING_EFFORTS if effort == result.values[0]), None)


def _known_bool(item: AvailableModel, field: str) -> bool | None:
    """Return a previously declared or provider-published boolean, if one exists.

    ``ModelCapabilities.supports_structured_output`` defaults to ``False``, so a configured
    snapshot created for another role does not count as an explicit denial of that field.
    """
    if item.published is not None:
        value = getattr(item.published, field)
        if isinstance(value, bool):
            return value
    if item.capabilities is None or item.pricing_source is not PricingSource.CONFIGURED:
        return None
    value = getattr(item.capabilities, field)
    if value is True:
        return True
    declared_false = {"supports_completions", "supports_embeddings", "supports_tools"}
    if field in declared_false and value is False:
        return False
    return None


def _known_price(item: AvailableModel, field: str) -> float | None:
    """Return a previously declared or provider-published price, if one exists."""
    if item.capabilities is not None:
        value = getattr(item.capabilities, field)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    if item.published is None:
        return None
    value = getattr(item.published, field)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _known_positive_int(item: AvailableModel, field: str) -> int | None:
    """Return a previously declared or provider-published positive limit, if one exists."""
    if item.capabilities is not None:
        value = getattr(item.capabilities, field)
        if isinstance(value, int) and value > 0:
            return value
    if item.published is None:
        return None
    value = getattr(item.published, field)
    return value if isinstance(value, int) and value > 0 else None
