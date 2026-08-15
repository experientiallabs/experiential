"""Concise provider and model setup shared by configuration and first build."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt

from wmo.common.models import (
    ModelCapabilities,
    ModelCatalog,
    ProviderConnection,
    ProviderModelSelection,
    ProviderSetup,
    catalog_state_sha256,
    configure_provider_catalog,
    load_model_catalog,
)

_NATIVE_PROVIDERS = ("openai", "openrouter", "anthropic", "gemini")
_PROVIDER_LABELS = {
    "openai": "OpenAI",
    "openrouter": "OpenRouter",
    "anthropic": "Anthropic",
    "gemini": "Gemini",
    "openai-compatible": "OpenAI-compatible",
}
_CREDENTIAL_ENV_SUGGESTIONS = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai-compatible": "OPENAI_COMPATIBLE_API_KEY",
}


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
) -> ModelCatalog:
    """Collect a complete catalog update before one conflict-checked atomic write.

    Args:
        root: Local ``.wmo`` root.
        options: Structured connection, model, and role inputs.
        non_interactive: Whether absent values must be reported instead of prompted.
        replace: Whether conflicting collected entries may replace unprotected catalog state.
        console: Rich console used for prompts, summaries, and guidance.

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
        setup = (
            _noninteractive_setup(options, existing=existing)
            if non_interactive
            else _interactive_setup(options, existing=existing, console=console)
        )
    except (EOFError, KeyboardInterrupt):
        raise typer.Abort() from None
    return configure_provider_catalog(
        path,
        setup,
        replace=replace,
        expected_state_sha256=starting_digest,
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
    existing_connections = _existing_connections(existing)
    existing_models = _existing_models(existing)
    available_connections = (*existing_connections, *connections)
    available_models = (*existing_models, *models)
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
        known_existing_connections=tuple(item.name for item in existing_connections),
        known_existing_aliases=tuple(model.alias for model in existing_models),
        world_model=roles["--world-model"],
        judge=roles["--judge"],
        embedder=roles["--embedder"],
    )


def _interactive_setup(
    options: ProviderSetupOptions,
    *,
    existing: ModelCatalog | None,
    console: Console,
) -> ProviderSetup:
    """Collect available connections and models before selecting independent roles.

    Args:
        options: Optional explicit role selections.
        existing: Existing catalog whose compatible entries remain available.
        console: Terminal used for collection, summary, and confirmation.

    Returns:
        Confirmed provider setup ready for atomic catalog merge.

    Raises:
        typer.BadParameter: No usable provider connection or model is collected.
        typer.Abort: The user rejects the completed configuration summary.
    """
    console.print("[bold]Model setup[/bold]")
    console.print("WMO stores credential environment-variable names, never credentials.")
    existing_connections = _existing_connections(existing)
    existing_models = _existing_models(existing)
    connections = list(existing_connections)
    models = list(existing_models)
    if connections:
        console.print("Existing connections: " + ", ".join(item.name for item in connections))
    for provider in _NATIVE_PROVIDERS:
        _collect_provider_connections(connections, provider=provider, console=console)
    _collect_provider_connections(connections, provider="openai-compatible", console=console)
    if not connections:
        raise typer.BadParameter("setup needs at least one available provider connection")

    if models:
        console.print("Existing model aliases: " + ", ".join(item.alias for item in models))
    add_model = not models
    while add_model or Confirm.ask("Add another available model?", default=False, console=console):
        models.append(_prompt_model(connections, console=console))
        add_model = False
    models = list(_deduplicate_models(models))
    if not models:
        raise typer.BadParameter("setup needs at least one available model alias")

    aliases = tuple(model.alias for model in models)
    world_model = options.world_model or _prompt_alias(
        "World model alias",
        aliases,
        default=_existing_role(existing, "world_model"),
        console=console,
    )
    judge = options.judge or _prompt_judge(
        models,
        world_model=world_model,
        current=_existing_role(existing, "judge"),
        console=console,
    )
    embedder = options.embedder or _prompt_alias(
        "Embedder alias",
        tuple(model.alias for model in models if model.capabilities.supports_embeddings),
        default=_existing_role(existing, "embedder"),
        console=console,
    )
    setup = ProviderSetup(
        connections=tuple(
            connection
            for connection in _deduplicate_connections(connections)
            if connection.name not in {item.name for item in existing_connections}
        ),
        models=tuple(
            model for model in models if model.alias not in {item.alias for item in existing_models}
        ),
        known_existing_connections=tuple(item.name for item in existing_connections),
        known_existing_aliases=tuple(item.alias for item in existing_models),
        world_model=world_model,
        judge=judge,
        embedder=embedder,
    )
    _render_summary(
        setup,
        connections=tuple(connections),
        models=tuple(models),
        console=console,
    )
    if not Confirm.ask("Save this configuration?", default=False, console=console):
        raise typer.Abort()
    return setup


def _collect_provider_connections(
    connections: list[ProviderConnection],
    *,
    provider: str,
    console: Console,
) -> None:
    """Collect zero or more explicitly available connections for one provider kind.

    Args:
        connections: Mutable collection receiving confirmed connection records.
        provider: Supported provider kind being collected.
        console: Terminal used for prompts.
    """
    label = _PROVIDER_LABELS[provider]
    another = Confirm.ask(f"Add a {label} connection?", default=False, console=console)
    while another:
        default_name = provider if not any(item.name == provider for item in connections) else None
        name = (
            Prompt.ask("Connection name", default=default_name, console=console)
            if default_name is not None
            else Prompt.ask("Connection name", console=console)
        ).strip()
        default_env = _CREDENTIAL_ENV_SUGGESTIONS[provider]
        api_key_env = Prompt.ask(
            "API key environment variable", default=default_env, console=console
        ).strip()
        base_url = None
        if provider == "openai-compatible":
            base_url = Prompt.ask("Base URL", console=console).strip()
        connections.append(
            ProviderConnection(
                name=name,
                provider=provider,
                api_key_env=api_key_env,
                base_url=base_url,
            )
        )
        another = Confirm.ask(f"Add another {label} connection?", default=False, console=console)


def _prompt_model(
    connections: list[ProviderConnection],
    *,
    console: Console,
) -> ProviderModelSelection:
    """Collect one explicit model alias and its complete known local capabilities.

    Args:
        connections: Available provider connections.
        console: Terminal used for prompts.

    Returns:
        Explicit model selection with declared capabilities and pricing.
    """
    connection_names = tuple(connection.name for connection in connections)
    connection = _prompt_alias("Connection", connection_names, default=None, console=console)
    alias = Prompt.ask("Model alias", console=console).strip()
    model = Prompt.ask("Provider model ID", console=console).strip()
    supports_tools = Confirm.ask("Supports tools?", default=False, console=console)
    supports_embeddings = Confirm.ask("Supports embeddings?", default=False, console=console)
    supports_structured_output = Confirm.ask(
        "Supports structured output?", default=False, console=console
    )
    supports_completions = Confirm.ask("Supports chat completions?", default=False, console=console)
    context_window = _prompt_optional_positive_int("Context window tokens", console=console)
    maximum_output = _prompt_optional_positive_int("Maximum output tokens", console=console)
    input_cost = None
    output_cost = None
    cached_input_cost = None
    cache_write_cost = None
    if supports_embeddings or supports_completions:
        input_cost = _prompt_nonnegative_float(
            "Input cost per million tokens in USD", console=console
        )
    if supports_completions:
        output_cost = _prompt_nonnegative_float(
            "Output cost per million tokens in USD", console=console
        )
        cached_input_cost = _prompt_nonnegative_float(
            "Cached input cost per million tokens in USD", console=console
        )
        cache_write_cost = _prompt_nonnegative_float(
            "Cache write cost per million tokens in USD", console=console
        )
    return ProviderModelSelection(
        alias=alias,
        connection=connection,
        model=model,
        capabilities=ModelCapabilities(
            supports_tools=supports_tools,
            supports_embeddings=supports_embeddings,
            supports_structured_output=supports_structured_output,
            supports_completions=supports_completions,
            context_window_tokens=context_window,
            maximum_output_tokens=maximum_output,
            input_cost_per_million_tokens_usd=input_cost,
            output_cost_per_million_tokens_usd=output_cost,
            cached_input_cost_per_million_tokens_usd=cached_input_cost,
            cache_write_cost_per_million_tokens_usd=cache_write_cost,
        ),
    )


def _prompt_optional_positive_int(label: str, *, console: Console) -> int | None:
    """Collect an optional positive integer without inventing provider metadata.

    Args:
        label: Human-readable capability field.
        console: Terminal used for prompts.

    Returns:
        Confirmed positive value, or ``None`` when the field is omitted.
    """
    if not Confirm.ask(f"Record {label.casefold()}?", default=False, console=console):
        return None
    return IntPrompt.ask(label, console=console)


def _prompt_nonnegative_float(label: str, *, console: Console) -> float:
    """Collect explicit local pricing without contacting a provider.

    Args:
        label: Human-readable pricing field.
        console: Terminal used for prompts.

    Returns:
        Confirmed finite nonnegative value.

    Raises:
        typer.BadParameter: The entered price is negative.
    """
    value = float(Prompt.ask(label, console=console))
    if value < 0:
        raise typer.BadParameter(f"{label} cannot be negative")
    return value


def _prompt_alias(
    label: str,
    aliases: tuple[str, ...],
    *,
    default: str | None,
    console: Console,
) -> str:
    """Prompt for one alias from an explicit available set.

    Args:
        label: Human-readable role or connection label.
        aliases: Exact available choices.
        default: Optional preselected alias.
        console: Terminal used for prompts.

    Returns:
        Confirmed alias from ``aliases``.

    Raises:
        typer.BadParameter: No compatible aliases exist or the response is outside the set.
    """
    if not aliases:
        raise typer.BadParameter(f"{label} has no compatible configured model aliases")
    selected = (
        Prompt.ask(label, choices=list(aliases), default=default, console=console)
        if default is not None
        else Prompt.ask(label, choices=list(aliases), console=console)
    ).strip()
    if selected not in aliases:
        raise typer.BadParameter(f"{label} must be one of: {', '.join(aliases)}")
    return selected


def _prompt_judge(
    models: list[ProviderModelSelection],
    *,
    world_model: str,
    current: str | None,
    console: Console,
) -> str:
    """Offer one explainable judge alias suggestion and require explicit acceptance.

    Args:
        models: Available declared model selections.
        world_model: Confirmed world-model alias used as a locality preference.
        current: Existing judge role when still available.
        console: Terminal used for explanation and confirmation.

    Returns:
        Explicitly confirmed judge alias.
    """
    aliases = tuple(model.alias for model in models)
    if current in aliases:
        return _prompt_alias("Judge alias", aliases, default=current, console=console)
    by_alias = {model.alias: model for model in models}
    world_connection = by_alias[world_model].connection
    ranked = sorted(
        models,
        key=lambda model: (
            not model.capabilities.supports_structured_output,
            model.capabilities.context_window_tokens is None,
            -1 * (model.capabilities.context_window_tokens or 0),
            model.connection != world_connection,
            model.alias,
        ),
    )
    suggestion = ranked[0].alias
    console.print(
        f"[dim]Suggested judge: {suggestion}. It is the strongest declared structured-output "
        "and context match among your configured aliases.[/dim]"
    )
    if Confirm.ask(f"Use {suggestion!r} as the judge?", default=True, console=console):
        return suggestion
    return _prompt_alias("Judge alias", aliases, default=None, console=console)


def _render_summary(
    setup: ProviderSetup,
    *,
    connections: tuple[ProviderConnection, ...],
    models: tuple[ProviderModelSelection, ...],
    console: Console,
) -> None:
    """Show every connection, model, role, and credential reference before commit.

    Args:
        setup: Collected role assignments.
        connections: All existing and newly collected provider connections.
        models: All existing and newly collected model aliases.
        console: Terminal receiving the summary.
    """
    console.print("[bold]Configuration summary[/bold]")
    for connection in connections:
        endpoint = f", base_url={connection.base_url}" if connection.base_url else ""
        console.print(
            f"connection {connection.name}: {connection.provider}, "
            f"api_key_env={connection.api_key_env}{endpoint}"
        )
    for model in models:
        caps = model.capabilities
        console.print(
            f"model {model.alias}: {model.connection}/{model.model}, "
            f"tools={caps.supports_tools}, embeddings={caps.supports_embeddings}, "
            f"structured_output={caps.supports_structured_output}, "
            f"completions={caps.supports_completions}"
        )
    console.print(
        f"roles: world_model={setup.world_model}, judge={setup.judge}, embedder={setup.embedder}"
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
        )
        for name, connection in sorted(existing.connections.items())
        if connection.provider in _PROVIDER_LABELS and connection.api_key_env is not None
    )


def _existing_models(existing: ModelCatalog | None) -> tuple[ProviderModelSelection, ...]:
    """Convert supported existing model aliases into setup input records.

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
