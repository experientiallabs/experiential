"""Simple Rich provider setup reused by CLI entrypoints."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt

from wmo.common.models import (
    ModelCatalog,
    ProviderConnection,
    ProviderModelSelection,
    ProviderSetup,
    configure_provider_catalog,
)

_PROVIDERS = ("openai", "openrouter", "anthropic", "gemini", "openai-compatible")
_EMBEDDING_PROVIDERS = ("openai", "openrouter", "gemini", "openai-compatible")
_CREDENTIAL_ENV_SUGGESTIONS = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai-compatible": "OPENAI_COMPATIBLE_API_KEY",
}
_JUDGE_GUIDANCE = {
    "openai": "an exact OpenAI model ID with enough reasoning quality for stable scoring",
    "openrouter": ("an exact provider/model ID on OpenRouter, preferably a stable, pinned judge"),
    "anthropic": "an exact Anthropic model ID with enough quality for stable scoring",
    "gemini": "an exact Gemini model ID with enough quality for stable scoring",
    "openai-compatible": (
        "an exact model ID that this endpoint serves and can use reliably for scoring"
    ),
}


@dataclass(frozen=True)
class ProviderSetupOptions:
    """Optional CLI values used directly or completed with Rich prompts."""

    provider: str | None = None
    connection: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    world_model: str | None = None
    judge: str | None = None
    embedder: str | None = None
    embedder_provider: str | None = None
    embedder_connection: str | None = None
    embedder_api_key_env: str | None = None
    embedder_base_url: str | None = None
    world_model_tools: bool = False
    judge_tools: bool = False


def run_provider_setup(
    root: Path,
    options: ProviderSetupOptions,
    *,
    non_interactive: bool,
    replace: bool,
    console: Console,
) -> ModelCatalog:
    """Collect a complete setup before atomically updating ``models.toml``.

    This is the CLI reuse seam for a later build command. All prompts finish before the service
    writes, so cancellation cannot leave a partial catalog.

    Args:
        root: Local ``.wmo`` root.
        options: Values supplied through command flags.
        non_interactive: Whether missing values must fail instead of prompting.
        replace: Whether existing conflicting build-time role aliases may be replaced.
        console: Rich console used for prompts and guidance.

    Returns:
        The complete validated catalog written by the setup service.

    Raises:
        typer.BadParameter: Required noninteractive values are absent.
    """
    setup = (
        _noninteractive_setup(options)
        if non_interactive
        else _interactive_setup(options, console=console)
    )
    return configure_provider_catalog(root / "models.toml", setup, replace=replace)


def _noninteractive_setup(options: ProviderSetupOptions) -> ProviderSetup:
    """Build exact setup input from flags or report every missing value."""
    required = {
        "--provider": options.provider,
        "--connection": options.connection,
        "--api-key-env": options.api_key_env,
        "--world-model": options.world_model,
        "--judge": options.judge,
        "--embedder": options.embedder,
    }
    missing = tuple(flag for flag, value in required.items() if value is None)
    if missing:
        raise typer.BadParameter(
            "--non-interactive requires " + ", ".join(missing) + "; add the missing flags"
        )
    assert options.provider is not None
    assert options.connection is not None
    assert options.api_key_env is not None
    assert options.world_model is not None
    assert options.judge is not None
    assert options.embedder is not None

    primary = ProviderConnection(
        name=options.connection,
        provider=options.provider,
        api_key_env=options.api_key_env,
        base_url=options.base_url,
    )
    embedder_connection = _noninteractive_embedder_connection(options, primary=primary)
    connections = (primary,) if embedder_connection == primary else (primary, embedder_connection)
    return _setup_from_values(
        options, connections=connections, primary=primary, embedder_connection=embedder_connection
    )


def _noninteractive_embedder_connection(
    options: ProviderSetupOptions,
    *,
    primary: ProviderConnection,
) -> ProviderConnection:
    """Resolve an optional separate embedder connection without provider inference."""
    separate_values = (
        options.embedder_provider,
        options.embedder_connection,
        options.embedder_api_key_env,
        options.embedder_base_url,
    )
    if not any(value is not None for value in separate_values):
        return primary
    required = {
        "--embedder-provider": options.embedder_provider,
        "--embedder-connection": options.embedder_connection,
        "--embedder-api-key-env": options.embedder_api_key_env,
    }
    missing = tuple(flag for flag, value in required.items() if value is None)
    if missing:
        raise typer.BadParameter("a separate embedder connection requires " + ", ".join(missing))
    assert options.embedder_provider is not None
    assert options.embedder_connection is not None
    assert options.embedder_api_key_env is not None
    return ProviderConnection(
        name=options.embedder_connection,
        provider=options.embedder_provider,
        api_key_env=options.embedder_api_key_env,
        base_url=options.embedder_base_url,
    )


def _interactive_setup(options: ProviderSetupOptions, *, console: Console) -> ProviderSetup:
    """Collect provider connections first, then exact build-time role model IDs."""
    console.print("[bold]Provider setup[/bold]")
    primary = _prompt_connection(
        console,
        provider=options.provider,
        name=options.connection,
        api_key_env=options.api_key_env,
        base_url=options.base_url,
        providers=_PROVIDERS,
        label="Primary",
    )
    use_primary_for_embeddings = primary.provider != "anthropic" and Confirm.ask(
        "Use this connection for embeddings?", default=True, console=console
    )
    if use_primary_for_embeddings:
        embedder_connection = primary
    else:
        if primary.provider == "anthropic":
            console.print("Anthropic needs a separate embedding provider in the current runtime.")
        embedder_connection = _prompt_connection(
            console,
            provider=options.embedder_provider,
            name=options.embedder_connection,
            api_key_env=options.embedder_api_key_env,
            base_url=options.embedder_base_url,
            providers=_EMBEDDING_PROVIDERS,
            label="Embedder",
        )
    connections = (primary,) if embedder_connection == primary else (primary, embedder_connection)

    world_model = options.world_model or Prompt.ask("Exact world model ID", console=console).strip()
    console.print(
        f"[dim]Judge suggestion: reuse {world_model!r} for consistency, or choose "
        f"{_JUDGE_GUIDANCE[primary.provider]}.[/dim]"
    )
    judge = options.judge or Prompt.ask("Exact judge model ID", console=console).strip()
    if options.judge is None and not Confirm.ask(
        f"Use {judge!r} as the judge?", default=True, console=console
    ):
        raise typer.Abort()
    embedder = options.embedder or Prompt.ask("Exact embedding model ID", console=console).strip()
    completed = replace(
        options,
        world_model=world_model,
        judge=judge,
        embedder=embedder,
    )
    return _setup_from_values(
        completed,
        connections=connections,
        primary=primary,
        embedder_connection=embedder_connection,
    )


def _prompt_connection(
    console: Console,
    *,
    provider: str | None,
    name: str | None,
    api_key_env: str | None,
    base_url: str | None,
    providers: tuple[str, ...],
    label: str,
) -> ProviderConnection:
    """Prompt for one connection without selecting a provider automatically."""
    selected_provider = provider or Prompt.ask(
        f"{label} provider", choices=list(providers), console=console
    )
    if selected_provider not in providers:
        raise typer.BadParameter(f"{label} provider must be one of: {', '.join(providers)}")
    selected_name = (
        name
        or Prompt.ask(
            f"{label} connection name", default=selected_provider, console=console
        ).strip()
    )
    selected_env = (
        api_key_env
        or Prompt.ask(
            "Credential environment variable",
            default=_CREDENTIAL_ENV_SUGGESTIONS[selected_provider],
            console=console,
        ).strip()
    )
    selected_base_url = base_url
    if selected_provider == "openai-compatible" and selected_base_url is None:
        selected_base_url = Prompt.ask("OpenAI-compatible base URL", console=console).strip()
    return ProviderConnection(
        name=selected_name,
        provider=selected_provider,
        api_key_env=selected_env,
        base_url=selected_base_url,
    )


def _setup_from_values(
    options: ProviderSetupOptions,
    *,
    connections: tuple[ProviderConnection, ...],
    primary: ProviderConnection,
    embedder_connection: ProviderConnection,
) -> ProviderSetup:
    """Create typed role selections after all connection and model input is complete."""
    assert options.world_model is not None
    assert options.judge is not None
    assert options.embedder is not None
    return ProviderSetup(
        connections=connections,
        world_model=ProviderModelSelection(
            connection=primary.name,
            model=options.world_model,
            supports_tools=options.world_model_tools,
        ),
        judge=ProviderModelSelection(
            connection=primary.name,
            model=options.judge,
            supports_tools=options.judge_tools,
        ),
        embedder=ProviderModelSelection(
            connection=embedder_connection.name,
            model=options.embedder,
        ),
    )
