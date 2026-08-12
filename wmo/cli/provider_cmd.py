"""Provider configuration and verification commands for the root CLI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import typer
from rich.console import Console
from rich.markup import escape

from wmo.cli.model_roles import configured_role_configs, load_settings_or_abort
from wmo.common.config import (
    ARTIFACT_DIR,
    HarnessConfig,
    ModelRole,
    WorldModelStore,
    load_config,
    save_settings,
    settings_path,
)

if TYPE_CHECKING:
    import wmo.cli.pool_registry as pool_registry
    from wmo.common.providers import ProviderConfig, ProviderKind, VerifyResult
    from wmo.common.providers.pool import Tier

from wmo.cli.command_common import (
    _credential_hint,
    _provider_kind,
    _worker_provider_config,
)

providers_app = typer.Typer(help="Manage and verify LLM providers.", no_args_is_help=True)
_console = Console()
_POOL_MODEL_OPTION = typer.Option(
    None,
    "--pool-model",
    help="Register this model id as a routing candidate, with no prompts; repeat for several.",
)


@providers_app.command(
    "set",
    help=(
        r"Choose the local worker provider, stored in `<root>/settings.toml` as "
        r"`\[models.worker]`, and optionally register routing candidates in the pool."
    ),
)
def providers_set(
    provider: str = typer.Option(None, "--provider", help="Provider for the local worker agent."),
    model: str = typer.Option(None, "--model", help="Canonical model type for the worker."),
    region: str = typer.Option(None, "--region", help="AWS region for Bedrock."),
    endpoint: str = typer.Option(None, "--endpoint", help="OpenAI-compatible API endpoint."),
    deployment: str = typer.Option(None, "--deployment", help="Azure deployment name."),
    api_version: str = typer.Option(None, "--api-version", help="Azure API version."),
    pool: str = typer.Option(
        None,
        "--pool",
        help="Candidate pool TOML the router picks from (default: <root>/pool.toml).",
    ),
    pool_model: list[str] | None = _POOL_MODEL_OPTION,
    api_key_env: str = typer.Option(
        None, "--api-key-env", help="Env var holding the pool entries' API key (multi-account)."
    ),
    tier: str = typer.Option(None, "--tier", help="Pool entry tier: frontier | open."),
    input_per_mtok: float = typer.Option(
        None,
        "--input-per-mtok",
        min=0.0,
        help="Prompt-token price of every --pool-model, USD per 1M tokens.",
    ),
    output_per_mtok: float = typer.Option(
        None,
        "--output-per-mtok",
        min=0.0,
        help="Completion-token price of every --pool-model, USD per 1M tokens.",
    ),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir holding local settings."),
) -> None:
    r"""Choose the local worker's provider, and register the models the router can choose from.

    Two things a project needs, in one command. The worker provider lands in
    `<root>/settings.toml` as `\[models.worker]`, exactly as before. Then, on a terminal, this
    offers to register models as ROUTING CANDIDATES in `<root>/pool.toml`, the roster
    `wmo optimize route` selects over: pick a backend, search its catalog, and answer only what
    that backend needs (an Azure deployment; a price for a model with no published one, never
    for OpenRouter, which self-prices). Re-run it to add another provider's models beside the
    ones already registered.

    Scripts keep the old contract: with `--provider` and `--model` both given, nothing is
    prompted. Registering non-interactively is `--pool-model <id>`, repeated per model, with
    `--input-per-mtok`/`--output-per-mtok` when the model has no published price.

    A locally hosted OpenAI-compatible server (Ollama, vLLM, llama.cpp) registers through the
    same command: `--provider openai --endpoint http://localhost:11434/v1` with `--model` and
    `--pool-model` naming what that server serves. Self-hosted candidates are priced explicitly
    at $0 per Mtok unless the price flags say otherwise, and the interactive openai pass asks
    for the endpoint URL and lists the server's own models.

    Args:
        provider: Worker provider kind to configure.
        model: Canonical worker model type.
        pool_model: Optional routing candidates to add without interaction.
        root: Project artifact directory containing settings and the candidate pool.

    Raises:
        typer.BadParameter: Provider, model, pricing, or pool configuration is invalid.
    """
    import wmo.cli.pool_registry as pool_registry
    from wmo.cli.ui import select_provider_and_model
    from wmo.common.providers import verify_all

    if tier is not None and tier not in pool_registry.TIERS:
        raise typer.BadParameter(f"--tier must be one of: {', '.join(pool_registry.TIERS)}")
    if (input_per_mtok is None) != (output_per_mtok is None):
        # A pool entry takes both prices or neither, so half a pair would be silently dropped
        # (interactively) or rejected as an invalid entry (with --pool-model).
        raise typer.BadParameter(
            "set both --input-per-mtok and --output-per-mtok, or neither; a pool entry prices "
            "prompt and completion tokens together"
        )
    if provider is not None:
        # Reject a bad --provider before reading the project's settings, so the argument the
        # caller typed is what the error is about even in a project whose settings.toml is broken.
        _provider_kind(provider)
    existing = load_settings_or_abort(root).models.worker
    used_picker = _console.is_terminal and (provider is None or model is None)
    if used_picker:
        provider, model, region = select_provider_and_model(
            _console,
            lambda text: _console.input(text),
            lambda text: _console.input(text, password=True),
            default_provider=provider or (existing.provider if existing else None),
            default_model=model or (existing.model if existing else None),
            default_region=region or (existing.region if existing else None),
            interactive=True,
            check=lambda cfg: verify_all(
                [
                    _worker_provider_config(
                        cfg.kind.value,
                        cfg.model_type or cfg.model,
                        cfg.region,
                        endpoint=endpoint,
                        deployment=deployment,
                        api_version=api_version,
                    )
                ]
            )[0],
        )
    if provider is None or model is None:
        raise typer.BadParameter(
            "provide --provider and --model, or run `wmo providers set` in a terminal"
        )
    config = _worker_provider_config(
        provider,
        model,
        region,
        endpoint=endpoint,
        deployment=deployment,
        api_version=api_version,
    )
    if not used_picker:
        result = verify_all([config])[0]
        if not result.ok:
            detail = escape(result.detail or "unknown error")
            _console.print(f"[red]provider verification failed[/red]: {detail}")
            raise typer.Exit(1)

    settings = load_settings_or_abort(root)
    settings.models.worker = ModelRole(
        provider=config.kind.value,
        model=config.model_type or model,
        region=config.region,
        endpoint=config.endpoint,
        deployment=config.deployment,
        api_version=config.api_version,
    )
    save_settings(settings, root)
    _console.print(
        f"[green]set[/green] local worker provider to {config.kind.value} "
        f"({config.model_type or model}) in {settings_path(root)}"
    )
    _register_pool_models(
        pool_path=pool_registry.pool_path_for(root, pool),
        kind=config.kind,
        pool_model=list(pool_model or []),
        options=pool_registry.EntryOptions(
            endpoint=config.endpoint,
            region=config.region,
            # The RAW flags, not the worker config's: `_worker_provider_config` fills an Azure
            # deployment in from the model id when none was given, and a pool entry must never
            # inherit a guessed deployment name (nothing can derive an operator's own).
            deployment=deployment,
            api_version=api_version,
            api_key_env=api_key_env,
            tier=cast("Tier", tier) if tier is not None else "frontier",
            input_per_mtok=input_per_mtok,
            output_per_mtok=output_per_mtok,
        ),
        interactive=used_picker,
    )


def _register_pool_models(
    *,
    pool_path: Path,
    kind: ProviderKind,
    pool_model: list[str],
    options: pool_registry.EntryOptions,
    interactive: bool,
) -> None:
    """Register routing candidates after the worker provider is saved.

    Explicit `--pool-model` ids win and register with no prompts, so a script gets the same
    roster a person would build by hand. Otherwise the registry is OFFERED only on the run that
    already prompted (a bare `wmo providers set` at a terminal): a scripted
    `--provider ... --model ...` invocation keeps its exact pre-existing behavior and asks
    nothing.
    """
    import wmo.cli.pool_registry as pool_registry

    if pool_model:
        pool_registry.register_model_ids(
            _console, pool_path=pool_path, kind=kind, model_ids=pool_model, options=options
        )
        return
    if not interactive:
        return
    pool_registry.run_pool_registry(
        _console,
        lambda text: _console.input(text),
        lambda text: _console.input(text, password=True),
        pool_path=pool_path,
        default_kind=kind,
        options=options,
    )


@providers_app.command(
    "verify",
    help=(
        r"Ping configured completion and embedding providers, including the "
        r"`\[models.<role>]` roles in `<root>/settings.toml` and built world models."
    ),
)
def providers_verify(
    name: str = typer.Option(None, "--name", help="Verify one model's providers (default: all)."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir."),
) -> None:
    r"""Ping every configured provider (completion + embed path) and report status.

    Two sources count as "configured", because checking that credentials work is what you do
    BEFORE spending anything on `wmo build`: the `\[models.<role>]` roles in
    `<root>/settings.toml` (what `wmo providers set` writes), and the providers persisted inside
    every built world model. Completion providers are deduped by kind+model across both, so a
    role naming the same backend as a built model costs one ping, not two. The phi embed path
    belongs to a BUILT model, so on a project with nothing built that half is skipped with a
    note instead of aborting the command; otherwise every distinct provider-backed embedder is
    checked (the offline hashing embedder needs no credentials and is skipped). `--name` scopes
    the whole report to that one world model.

    Args:
        name: Optional world model whose provider configuration is verified.
        root: Project artifact directory containing settings and built models.

    Raises:
        typer.BadParameter: The selected model or its configuration cannot be read.
    """
    from wmo.common.providers import verify_all, verify_embedder
    from wmo.common.providers.base import EmbedderKind

    store = WorldModelStore(root)
    names = [name] if name is not None else store.list_names()
    configs: list[HarnessConfig] = []
    for model_name in names:
        try:
            # Reading the artifact is inside the guard too: a model dir that exists but whose
            # config.toml is corrupt or unreadable is a bad artifact, not an internal error.
            configs.append(load_config(str(store.resolve(model_name))))
        except (FileNotFoundError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc

    # Dedup identical completion calls across all selected models, then across the settings
    # roles. World models come FIRST so a config present in both is pinged with the built
    # artifact's own copy, exactly as before roles were a source.
    labelled = [
        (model_name, pc)
        for model_name, cfg in zip(names, configs, strict=True)
        for pc in cfg.providers
    ]
    # `--name` asks about one world model; widening it to the project's roles would answer a
    # question the caller did not ask (and bill for it).
    if name is None:
        labelled += [(f"models.{role}", pc) for role, pc in configured_role_configs(root)]
    index: dict[str, int] = {}
    provider_configs: list[ProviderConfig] = []
    sources: list[list[str]] = []
    for label, pc in labelled:
        key = _completion_identity(pc)
        if key not in index:
            index[key] = len(provider_configs)
            provider_configs.append(pc)
            sources.append([])
        if label not in sources[index[key]]:
            sources[index[key]].append(label)

    if not provider_configs:
        _console.print(
            "[yellow]nothing configured[/yellow]; run `wmo providers set` to choose a provider, "
            "or `wmo build --name <name>` to build a world model"
        )
        raise typer.Exit(1)

    for result, origin in zip(verify_all(provider_configs), sources, strict=True):
        _print_verify_result(result, origin)

    if not configs:
        _console.print(
            "[dim]embed path: skipped, no world model built yet "
            "(the embedder is chosen by `wmo build`)[/dim]"
        )
        return

    # Verify each distinct provider-backed embed path (skip the in-process embedders: hashing
    # and local have no provider to verify).
    embed_seen: set[str] = set()
    for model_name, config in zip(names, configs, strict=True):
        if config.embed_provider in (EmbedderKind.HASHING, EmbedderKind.LOCAL):
            continue
        embed_config = config.embed_provider_config()
        key = embed_config.model_dump_json()
        if key in embed_seen:
            continue
        embed_seen.add(key)
        result = verify_embedder(embed_config)
        _print_verify_result(result, [model_name], prefix="embed:")


def _completion_identity(config: ProviderConfig) -> str:
    """What makes two provider configs the SAME completion call, for `providers verify` dedup.

    Everything the config carries except the embed-only fields, which do not reach a completion
    request. Keying on kind+model alone would collapse one model served from two Bedrock regions,
    or one Azure model behind two deployments or endpoints, into a single ping, and then report
    the config that was never called as verified on the other one's result. Deriving the key
    from the whole config rather than a hand-listed subset means a field added later splits the
    key by default instead of silently widening that collapse.
    """
    return config.model_copy(update={"embed_model": None, "embed_dim": None}).model_dump_json()


def _print_verify_result(result: VerifyResult, sources: list[str], *, prefix: str = "") -> None:
    """Print one `providers verify` line, plus the next step to take when the ping failed.

    `sources` names what asked for this provider (world model names, `models.<role>` settings
    roles) so a failure points at the thing to fix. The detail and the hint are escaped: they
    carry raw provider error text and pip extras (`...[distill]`), and an unescaped `[...]` in
    either would be read as rich markup and silently dropped from the report.
    """
    mark = "[green]ok[/green]" if result.ok else "[red]fail[/red]"
    origin = f" [dim]({', '.join(sources)})[/dim]" if sources else ""
    detail = f" {escape(result.detail)}" if result.detail else ""
    _console.print(f"{mark} {prefix}{result.kind.value} ({result.model}){origin}{detail}")
    if not result.ok:
        hint = escape(_credential_hint(result.kind, result.detail))
        _console.print(f"  [yellow]{hint}[/yellow]")
