"""Provider selection, credential resolution, and authenticated model discovery for setup.

Setup opens with one provider screen, resolves each selected provider's credential from its
canonical environment variable or a masked paste, then asks that provider which models the
authenticated account may call. Discovered metadata is merged with WMO's maintained capability and
price table, so a model whose metadata cannot satisfy any build role is hidden rather than turned
into a questionnaire. Providers without a safe listing API keep manual declaration on the model
screen.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass, field
from getpass import getpass

from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt

from wmo.cli.shared.picker import (
    PickerAction,
    PickerKeyReader,
    PickerOption,
    PickerResult,
    choose_many,
    choose_one,
)
from wmo.common.models import (
    ModelCapabilities,
    PricingSource,
    ProviderConnection,
    ProviderSetup,
    ReasoningEffort,
    SetupRole,
    derive_connection_name,
    derive_model_alias,
    resolve_discovered_model,
    served_roles,
)
from wmo.runtime.models.providers import (
    ProviderEndpoint,
    ProviderListingError,
    ProviderModelLister,
)

SETUP_PROVIDER_LABELS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Gemini",
    "openrouter": "OpenRouter",
    "openai-compatible": "OpenAI-compatible endpoint",
    "azure": "Azure OpenAI",
    "bedrock": "Amazon Bedrock",
}
CANONICAL_CREDENTIAL_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai-compatible": "OPENAI_COMPATIBLE_API_KEY",
}
_MANUAL_MODEL_PROVIDERS = frozenset({"azure", "bedrock"})
_CONFIGURED_ONLY = "configured-models-only"
_RECOVERY_RETRY = "retry"
_RECOVERY_SKIP = "skip"
_RECOVERY_BACK = "back"
CREDENTIAL_NOTE = (
    "WMO stores only the credential environment-variable name. A pasted value stays in this "
    "process; export it or add it to .env to reuse it later."
)


class SetupCancelled(Exception):
    """The user cancelled interactive setup."""


@dataclass(frozen=True)
class SetupRoleInputs:
    """Role values already chosen by command flags or present in the catalog."""

    world_model: str | None = None
    judge: str | None = None
    embedder: str | None = None
    candidates: tuple[str, ...] = ()
    incumbent: str | None = None
    world_model_reasoning_effort: ReasoningEffort | None = None
    judge_reasoning_effort: ReasoningEffort | None = None
    candidate_reasoning_efforts: dict[str, ReasoningEffort] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderSetupResult:
    """One confirmed catalog update plus the router roles it also assigned."""

    setup: ProviderSetup
    candidates: tuple[str, ...] = ()
    incumbent: str | None = None
    candidate_reasoning_efforts: dict[str, ReasoningEffort] = field(default_factory=dict)


@dataclass(frozen=True)
class AvailableModel:
    """One model the user can configure, either already in the catalog or newly discovered."""

    alias: str
    connection: str
    provider: str
    model: str
    capabilities: ModelCapabilities | None
    pricing_source: PricingSource
    configured: bool
    retainable_roles: frozenset[SetupRole] = frozenset()

    def label(self) -> str:
        """Describe this model as one picker row."""
        return f"{self.alias} ({self.provider}/{self.model})"

    def detail(self) -> str:
        """Summarize the roles and price provenance behind this model."""
        verified = (
            frozenset(served_roles(self.capabilities))
            if self.capabilities is not None
            else frozenset()
        )
        roles = ", ".join(role.value for role in SetupRole if role in verified)
        retain_only = ", ".join(
            role.value for role in SetupRole if role in self.retainable_roles - verified
        )
        retained = f"; retain only: {retain_only}" if retain_only else ""
        return f"roles: {roles or 'unverified'}{retained}; pricing: {self.pricing_source.value}"


@dataclass(frozen=True)
class PreparedEndpoint:
    """One provider connection paired with the credential resolved for this session."""

    connection: ProviderConnection
    api_key: str
    configured: bool


@dataclass
class SetupSession:
    """Answers already given, kept across back navigation and provider retries."""

    providers: tuple[str, ...] = ()
    advanced_models: bool = False
    endpoints: tuple[PreparedEndpoint, ...] = ()
    available: tuple[AvailableModel, ...] = ()
    selected: tuple[str, ...] = ()
    manual: list[AvailableModel] = field(default_factory=list)


def resolve_setup_providers(values: Sequence[str]) -> tuple[str, ...]:
    """Validate explicit provider names and return them in catalog order.

    Args:
        values: Provider names supplied on the command line, in flag order.

    Returns:
        Distinct supported providers in ``SETUP_PROVIDER_LABELS`` order.

    Raises:
        ValueError: A value is unsupported or repeated.
    """
    if not values:
        return ()
    supported = tuple(SETUP_PROVIDER_LABELS)
    supported_lookup = {name: name for name in supported}
    unknown: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    for raw in values:
        name = raw.strip().casefold()
        canonical = supported_lookup.get(name)
        if canonical is None:
            unknown.append(raw.strip() or raw)
            continue
        if canonical in seen:
            duplicates.append(canonical)
            continue
        seen.add(canonical)
    errors: list[str] = []
    if unknown:
        listed = ", ".join(repr(item) for item in unknown)
        noun = "values" if len(unknown) != 1 else "value"
        errors.append(
            f"unsupported --provider {noun} {listed}; choose from: {', '.join(supported)}"
        )
    if duplicates:
        listed = ", ".join(repr(item) for item in dict.fromkeys(duplicates))
        noun = "values" if len(duplicates) != 1 else "value"
        errors.append(f"duplicate --provider {noun} {listed}")
    if errors:
        raise ValueError("; ".join(errors))
    return tuple(name for name in supported if name in seen)


def explicit_provider_selection(
    providers: Sequence[str],
) -> tuple[tuple[str, ...], bool]:
    """Turn validated provider names into the same selection the picker returns.

    Args:
        providers: Already validated provider names in catalog order.

    Returns:
        Providers plus the manual-model flag. Azure and Bedrock still require manual model
        declaration.
    """
    selected = tuple(providers)
    return (
        selected,
        any(provider in _MANUAL_MODEL_PROVIDERS for provider in selected),
    )


def select_providers(
    session: SetupSession,
    *,
    console: Console,
    environment: MutableMapping[str, str],
    configured: bool = False,
    read_key: PickerKeyReader | None = None,
) -> tuple[tuple[str, ...], bool] | None:
    """Show the one provider screen that opens setup.

    Args:
        session: Answers already collected, used to preselect prior choices.
        console: Terminal used for the screen.
        environment: Process environment used to annotate credential availability.
        configured: Whether the catalog already holds usable models, which makes provider
            selection optional so roles can be edited offline.
        read_key: Optional keyboard source used by tests instead of the controlling terminal.

    Returns:
        Selected providers plus whether manual model declaration is needed, or ``None`` when the
        user cancelled.
    """
    options = [
        PickerOption(
            value=provider,
            label=label,
            detail=credential_hint(provider, environment=environment),
        )
        for provider, label in SETUP_PROVIDER_LABELS.items()
    ]
    if configured:
        options.insert(
            0,
            PickerOption(
                value=_CONFIGURED_ONLY,
                label="Keep the models already configured",
                detail="no provider request, roles only",
            ),
        )
    preselected = list(session.providers)
    while True:
        result = _select_provider_rows(
            console,
            options=options,
            preselected=preselected,
            read_key=read_key,
        )
        if result.action is PickerAction.CANCEL:
            return None
        if result.action is PickerAction.BACK:
            console.print("[yellow]This is the first screen.[/yellow]")
            continue
        providers = tuple(value for value in result.values if value in SETUP_PROVIDER_LABELS)
        if not providers and _CONFIGURED_ONLY not in result.values:
            console.print("[yellow]Select at least one provider.[/yellow]")
            preselected = list(result.values)
            continue
        return providers, any(provider in _MANUAL_MODEL_PROVIDERS for provider in providers)


def _select_provider_rows(
    console: Console,
    *,
    options: Sequence[PickerOption],
    preselected: Sequence[str],
    read_key: PickerKeyReader | None,
) -> PickerResult:
    """Show the provider multi-select screen for this console.

    Args:
        console: Terminal used for the screen.
        options: Provider rows in presentation order.
        preselected: Values already chosen, kept when the screen is shown again.
        read_key: Optional keyboard source used by tests instead of the controlling terminal.

    Returns:
        The chosen rows, or the requested back or cancel navigation.
    """
    return choose_many(
        console,
        title="Select the providers you want to use",
        options=options,
        preselected=preselected,
        read_key=read_key,
    )


def credential_hint(provider: str, *, environment: MutableMapping[str, str]) -> str:
    """Describe whether the canonical credential for one provider is already readable."""
    if provider == "bedrock":
        return "uses the AWS credential chain; model IDs are declared manually"
    name = CANONICAL_CREDENTIAL_ENV[provider]
    status = f"{name} is set" if environment.get(name, "").strip() else f"{name} needs a value"
    if provider == "azure":
        return f"{status}; deployment IDs are declared manually"
    return status


def prepare_providers(
    session: SetupSession,
    *,
    existing_connections: tuple[ProviderConnection, ...],
    existing_aliases: tuple[str, ...],
    console: Console,
    lister: ProviderModelLister,
    environment: MutableMapping[str, str],
) -> tuple[tuple[PreparedEndpoint, ...], tuple[AvailableModel, ...]] | None:
    """Resolve one credential per selected provider and list that account's models.

    Args:
        session: Answers already collected in this setup session.
        existing_connections: Connections already configured in the catalog.
        existing_aliases: Aliases already configured in the catalog.
        console: Terminal used for prompts and progress.
        lister: Provider listing seam.
        environment: Process environment consulted and updated for pasted credentials.

    Returns:
        Prepared endpoints with their configurable models, or ``None`` to change providers.

    Raises:
        SetupCancelled: The user cancelled setup.
    """
    endpoints: list[PreparedEndpoint] = []
    available: list[AvailableModel] = []
    taken_names = {connection.name for connection in existing_connections}
    taken_aliases = set(existing_aliases)
    for provider in session.providers:
        label = SETUP_PROVIDER_LABELS[provider]
        endpoint = _resolve_endpoint(
            provider,
            existing_connections=existing_connections,
            taken_names=frozenset(taken_names),
            console=console,
            environment=environment,
        )
        if endpoint is None:
            console.print(f"[yellow]Skipping {label}.[/yellow]")
            continue
        if provider in _MANUAL_MODEL_PROVIDERS:
            console.print(
                f"[dim]{label} model IDs are deployment-specific; "
                "declare them on the model screen.[/dim]"
            )
            discovered: tuple[AvailableModel, ...] | None = ()
        else:
            discovered = _discover_models(
                endpoint,
                provider=provider,
                console=console,
                lister=lister,
                taken_aliases=frozenset(taken_aliases),
            )
        if discovered is None:
            return None
        if (
            not discovered
            and provider not in _MANUAL_MODEL_PROVIDERS
            and not session.advanced_models
        ):
            console.print(f"[yellow]Skipping {label}.[/yellow]")
            continue
        endpoints.append(endpoint)
        taken_names.add(endpoint.connection.name)
        for item in discovered:
            taken_aliases.add(item.alias)
        available.extend(discovered)
    if not endpoints:
        console.print("[yellow]No provider was prepared. Choose providers again.[/yellow]")
        return None
    return tuple(endpoints), tuple(available)


def _resolve_endpoint(
    provider: str,
    *,
    existing_connections: tuple[ProviderConnection, ...],
    taken_names: frozenset[str],
    console: Console,
    environment: MutableMapping[str, str],
) -> PreparedEndpoint | None:
    """Derive one connection for a provider and resolve its credential for this session.

    Args:
        provider: Selected provider kind.
        existing_connections: Connections already configured in the catalog.
        taken_names: Connection names already used by the catalog or this session.
        console: Terminal used for prompts.
        environment: Process environment consulted and updated for pasted credentials.

    Returns:
        The prepared endpoint, or ``None`` when the provider is skipped.

    Raises:
        SetupCancelled: The user cancelled setup at a prompt.
    """
    label = SETUP_PROVIDER_LABELS[provider]
    base_url = None
    api_version = None
    region = None
    if provider in ("openai-compatible", "azure"):
        base_url = ask_text(f"{label} base URL", console=console)
        if not base_url:
            return None
    if provider == "azure":
        api_version = ask_text("Azure OpenAI API version", console=console, default="v1")
        if not api_version:
            return None
    if provider == "bedrock":
        region = (
            ask_text(
                "AWS region (empty uses the AWS credential or session configuration)",
                console=console,
            )
            or None
        )
    api_key_env = CANONICAL_CREDENTIAL_ENV.get(provider)
    connection = _reused_connection(
        existing_connections,
        provider=provider,
        api_key_env=api_key_env,
        base_url=base_url,
        api_version=api_version,
        region=region,
    )
    configured = connection is not None
    if connection is None:
        connection = ProviderConnection(
            name=derive_connection_name(provider, taken_names),
            provider=provider,
            api_key_env=api_key_env,
            base_url=base_url,
            api_version=api_version,
            region=region,
        )
    if provider == "bedrock":
        return PreparedEndpoint(connection=connection, api_key="", configured=configured)
    assert connection.api_key_env is not None
    api_key = _resolve_credential(
        label,
        api_key_env=connection.api_key_env,
        console=console,
        environment=environment,
    )
    if api_key is None:
        return None
    return PreparedEndpoint(connection=connection, api_key=api_key, configured=configured)


def _reused_connection(
    existing_connections: tuple[ProviderConnection, ...],
    *,
    provider: str,
    api_key_env: str | None,
    base_url: str | None,
    api_version: str | None,
    region: str | None,
) -> ProviderConnection | None:
    """Return the configured connection that already describes this exact endpoint."""
    for connection in existing_connections:
        if (
            connection.provider == provider
            and connection.api_key_env == api_key_env
            and connection.base_url == base_url
            and connection.api_version == api_version
            and connection.region == region
        ):
            return connection
    return None


def _resolve_credential(
    label: str,
    *,
    api_key_env: str,
    console: Console,
    environment: MutableMapping[str, str],
) -> str | None:
    """Read one provider credential from the environment, or accept a masked paste.

    Args:
        label: Readable provider name.
        api_key_env: Credential environment-variable name for this connection.
        console: Terminal used for the masked prompt.
        environment: Process environment consulted and updated for pasted credentials.

    Returns:
        The resolved credential, or ``None`` when the provider is skipped.

    Raises:
        SetupCancelled: The prompt reached end of input.
    """
    existing = environment.get(api_key_env, "").strip()
    if existing:
        console.print(f"[dim]{label} credential read from {api_key_env}.[/dim]")
        return existing
    console.print(f"{label} has no {api_key_env} value. Paste a key to use it for this run.")
    try:
        pasted = getpass(f"{label} API key (hidden, empty line skips this provider): ").strip()
    except (EOFError, KeyboardInterrupt) as exc:
        raise SetupCancelled from exc
    if not pasted:
        return None
    environment[api_key_env] = pasted
    console.print(f"[dim]{label} credential kept in this process only.[/dim]")
    return pasted


def _discover_models(
    endpoint: PreparedEndpoint,
    *,
    provider: str,
    console: Console,
    lister: ProviderModelLister,
    taken_aliases: frozenset[str],
) -> tuple[AvailableModel, ...] | None:
    """List one provider account's models and keep the ones with usable metadata.

    Args:
        endpoint: Prepared connection and resolved credential.
        provider: Selected provider kind.
        console: Terminal used for progress and recovery.
        lister: Provider listing seam.
        taken_aliases: Aliases already used by the catalog or this session.

    Returns:
        Configurable models, or ``None`` when the user asked to change providers.

    Raises:
        SetupCancelled: The user cancelled setup during recovery.
    """
    label = SETUP_PROVIDER_LABELS[provider]
    request = ProviderEndpoint(
        provider=provider,
        api_key=endpoint.api_key,
        base_url=endpoint.connection.base_url,
    )
    aliases = set(taken_aliases)
    while True:
        console.print(f"Reading models available to your {label} account...")
        try:
            discovered = lister.list_models(request)
        except ProviderListingError as exc:
            console.print(f"[red]{exc}[/red]")
            recovery = _recover(f"{label} model listing failed", console=console)
            if recovery == _RECOVERY_RETRY:
                continue
            return () if recovery == _RECOVERY_SKIP else None
        resolved = tuple(resolve_discovered_model(model) for model in discovered)
        usable = tuple(model for model in resolved if served_roles(model.capabilities))
        hidden = len(resolved) - len(usable)
        if not usable:
            console.print(f"[yellow]{label} published no model with verified metadata.[/yellow]")
            recovery = _recover(f"{label} has no configurable model", console=console)
            if recovery == _RECOVERY_RETRY:
                continue
            return () if recovery == _RECOVERY_SKIP else None
        console.print(
            f"[dim]{label}: {len(usable)} configurable models, "
            f"{hidden} hidden without verified capabilities or prices.[/dim]"
        )
        models = []
        for model in usable:
            alias = derive_model_alias(provider, model.model, frozenset(aliases))
            aliases.add(alias)
            models.append(
                AvailableModel(
                    alias=alias,
                    connection=endpoint.connection.name,
                    provider=provider,
                    model=model.model,
                    capabilities=model.capabilities,
                    pricing_source=model.pricing_source,
                    configured=False,
                )
            )
        return tuple(models)


def _recover(title: str, *, console: Console) -> str:
    """Offer retry, skip, or provider reselection after a provider problem.

    Args:
        title: Screen heading naming the problem.
        console: Terminal used for the screen.

    Returns:
        One of the recovery actions.

    Raises:
        SetupCancelled: The user cancelled setup.
    """
    result = choose_one(
        console,
        title=title,
        options=[
            PickerOption(value=_RECOVERY_RETRY, label="Try this provider again"),
            PickerOption(value=_RECOVERY_SKIP, label="Continue without this provider"),
            PickerOption(value=_RECOVERY_BACK, label="Choose providers again"),
        ],
        default=_RECOVERY_RETRY,
    )
    if result.action is PickerAction.CANCEL:
        raise SetupCancelled
    if result.action is PickerAction.BACK:
        return _RECOVERY_BACK
    return result.values[0]


def ask_text(label: str, *, console: Console, default: str | None = None) -> str:
    """Read one trimmed line, treating end of input as cancellation.

    Args:
        label: Prompt text.
        console: Terminal used for the prompt.
        default: Value accepted with an empty line.

    Returns:
        The trimmed answer, which may be empty when no default exists.

    Raises:
        SetupCancelled: The prompt reached end of input.
    """
    try:
        if default is not None:
            return Prompt.ask(label, default=default, console=console).strip()
        return Prompt.ask(label, default="", console=console).strip()
    except (EOFError, KeyboardInterrupt) as exc:
        raise SetupCancelled from exc


def ask_positive_int(label: str, *, console: Console) -> int | None:
    """Read one optional positive integer under the advanced path.

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


def ask_price(label: str, *, console: Console) -> float:
    """Read one nonnegative price under the advanced path.

    Args:
        label: Prompt text.
        console: Terminal used for the prompt.

    Returns:
        The nonnegative price per million tokens in USD.

    Raises:
        SetupCancelled: The prompt reached end of input.
    """
    while True:
        answer = ask_text(label, console=console, default="0")
        try:
            value = float(answer)
        except ValueError:
            console.print(f"[yellow]{label} must be a number.[/yellow]")
            continue
        if value < 0:
            console.print(f"[yellow]{label} cannot be negative.[/yellow]")
            continue
        return value
