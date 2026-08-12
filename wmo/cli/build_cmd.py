"""Build command and input validation for trace-backed world models."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.markup import escape

from wmo.cli.model_roles import load_settings_or_abort
from wmo.common.config import (
    ARTIFACT_DIR,
    DEFAULT_MODEL_NAME,
    FIDELITY_TIERS,
    ArtifactPaths,
    FidelityTier,
    HarnessConfig,
    WorldModelStore,
    normalize_name,
    validate_name,
)

if TYPE_CHECKING:
    from wmo.common.providers import ProviderConfig
    from wmo.common.providers.base import Provider

from wmo.cli.command_common import _credential_hint

_console = Console()
_CHECK = "[green]✓[/green]"

# The serve provider `wmo build` falls back to when neither `--provider` nor a configured worker
# role names one.
_BUILD_PROVIDER = "bedrock"
# Sources that read a live database rather than an export. `wmo build` has no transport flags for
# them (`VendorPull.dsn/table` are only reachable from `wmo ingest`), so it rejects them up front
# instead of failing inside the adapter with advice about flags it does not define.
_DB_SOURCES = ("postgres",)


def _check_build_source(source: str) -> None:
    """Reject a `--source` that `wmo build` cannot drive, naming the `wmo ingest` two-step."""
    from wmo.simulation.ingest import list_adapters

    if source not in list_adapters():
        raise typer.BadParameter(
            f"unknown --source {source!r}; choose one of: {', '.join(list_adapters())}"
        )
    if source in _DB_SOURCES:
        raise typer.BadParameter(
            f"--source {source} reads a live database, which `wmo build` has no connection flags "
            f"for: normalize first with `wmo ingest --source {source} --dsn <dsn> --table <table> "
            "--out traces.otel.jsonl`, then `wmo build --file traces.otel.jsonl`"
        )


def _check_build_file(file: str) -> None:
    """Reject a missing/unreadable `--file` before the provider ping, not inside the adapter."""
    path = Path(file)
    if not path.exists():
        raise typer.BadParameter(
            f"trace file not found: {file} (export one with `wmo ingest --file <export> --out "
            f"{file}`, or pass an existing path)"
        )
    if path.is_dir():
        raise typer.BadParameter(f"--file must be a trace export, not a directory: {file}")


def _empty_corpus_error(file: str | None, source: str) -> typer.BadParameter:
    """Explain an empty ingest: usually `--source` does not match the export's real format."""
    from wmo.simulation.ingest.base import load_payloads
    from wmo.simulation.ingest.detect import detect_format

    if file is None:
        return typer.BadParameter(
            f"the --source {source} pull returned no traces; widen --since/--limit or check the "
            "--project"
        )
    detected: str | None = None
    try:
        detected = detect_format(load_payloads(Path(file).read_text(encoding="utf-8")))
    except (ValueError, OSError):
        detected = None
    if detected is not None and detected != source:
        return typer.BadParameter(
            f"no traces ingested from {file} with --source {source}: it looks like {detected}. "
            f"Retry with `--source {detected}`, or let `wmo ingest --file {file} --out "
            "traces.otel.jsonl` auto-detect the format and build from its output."
        )
    return typer.BadParameter(
        f"no traces ingested from {file} with --source {source} (build never auto-detects): "
        f"check that the export carries agent steps and that --source names its format, or run "
        f"`wmo ingest --file {file} --out traces.otel.jsonl` to auto-detect and normalize it "
        "first"
    )


def _chain_bad_parameter(err: ValueError) -> typer.BadParameter:
    """Render a failover-chain resolution failure as a usage error naming the file to write."""
    from wmo.common.providers.waterfall import FALLBACK_CONFIG_PATH

    return typer.BadParameter(
        f"{err}. Write {FALLBACK_CONFIG_PATH} as [[chain.<name>]] rung tables (one table per "
        "fallback rung, keys kind/model/profile/region; format in docs/reference/failover.md), "
        "or drop --chain to serve from --provider alone"
    )


def _provider_or_chain_or_abort(config: ProviderConfig, chain: str | None) -> Provider:
    """`providers.provider_or_chain`, with chain-resolution errors as clean usage errors."""
    import wmo.common.providers as providers

    try:
        return providers.provider_or_chain(config, chain=chain)
    except ValueError as err:
        raise _chain_bad_parameter(err) from None


def build(
    name: str = typer.Option(None, "--name", help="Name for this world model."),
    source: str = typer.Option(
        "otel-genai",
        "--source",
        help="Trace source adapter, pinned (build never auto-detects): otel-genai, chat-json, "
        "braintrust, phoenix, langfuse, langsmith, posthog, mastra. To auto-detect an export's "
        "format, normalize it first with `wmo ingest --file <export>`.",
    ),
    file: str = typer.Option(None, "--file", help="Path to an exported traces file for --source."),
    pull: bool = typer.Option(
        False, "--pull", help="Pull traces live from the source's vendor API (instead of --file)."
    ),
    project: str = typer.Option(None, "--project", help="Vendor project/workspace id (--pull)."),
    api_key: str = typer.Option(None, "--api-key", help="Vendor API key (else env var)."),
    since: str = typer.Option(None, "--since", help="Only pull traces since this ISO timestamp."),
    limit: int = typer.Option(
        None,
        "--limit",
        help="Only use the first N traces (caps a --file export too, not just --pull); cost "
        "control for a first build over a large corpus. A --pull is capped at fetch time, so "
        "with --drop-degenerate it can yield fewer than N; a --file export yields N usable.",
    ),
    vendor: str = typer.Option(
        None, "--vendor", help="\\[deprecated] alias for --source <name> --pull."
    ),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir holding all world models."),
    provider: str = typer.Option(
        None, "--provider", help=f"Provider that serves the model (default: {_BUILD_PROVIDER})."
    ),
    model: str = typer.Option(
        None, help="Canonical serve model type (default: the provider's suggested model)."
    ),
    judge_model: str = typer.Option(
        None, "--judge-model", help="Canonical GEPA judge model type (default: cheap per provider)."
    ),
    region: str = typer.Option(None, help="AWS region (Bedrock)."),
    fidelity: str = typer.Option(
        "low",
        help="Build effort (all searching tiers are floored at low's estimate - more effort "
        "never ships worse than low): low (default; free - the estimated-best config, no "
        "search) | medium (+light GEPA + cheap-lever search) | high (+GEPA + config search) | "
        "max (deep GEPA + full config search). Searching tiers cost real money: one observed "
        "medium build spent 73% of its total on GEPA. The chosen config serves under "
        "`--max-fidelity`.",
    ),
    chain: str = typer.Option(
        None, "--chain", help="Named failover chain from .wmo/fallback.toml (default: its default)."
    ),
    train_split: float = typer.Option(
        0.8,
        help="Train/held-out ratio for GEPA's internal split (lower = bigger valset). Shares "
        "`wmo eval`'s default: both cut the same trace-id hash line, so a mismatch leaks train "
        "traces into the eval holdout.",
    ),
    embed_provider: str = typer.Option(
        "hashing", help="phi embedder: hashing (offline) | bedrock | openai | azure."
    ),
    embed_model: str = typer.Option(None, help="Embeddings model id / Azure embedding deployment."),
    embed_dim: int = typer.Option(512, help="phi dimensionality (index + query must agree)."),
    knowledge: bool = typer.Option(
        False,
        "--knowledge/--no-knowledge",
        help="Seed a knowledge base (rules/entities/schemas markdown) from the train traces.",
    ),
    reasoning: bool = typer.Option(
        False,
        "--reasoning/--no-reasoning",
        help="Serve with the deliberate-then-answer output contract.",
    ),
    grounder: str = typer.Option(
        "none",
        help="Web grounding for unknown entities: none | brave (needs BRAVE_SEARCH_API_KEY).",
    ),
    drop_degenerate: bool = typer.Option(
        False,
        "--drop-degenerate",
        help="Drop all-empty-observation traces (failed captures) before building "
        "(swe-bench is ~66% such junk).",
    ),
    interactive: bool = typer.Option(
        None,
        "--interactive/--no-interactive",
        help="Guided creation wizard. Default: on at a TTY when inputs are missing.",
    ),
) -> None:
    """Ingest traces (file upload or vendor SDK pull) and build a named world model.

    Stores the artifact under `<root>/models/<name>/`: ingest -> normalize -> split(train/test) ->
    embed/index -> GEPA optimize -> write. Re-running with the same `--name` rebuilds it.

    With no `--name`/`--file` on an interactive terminal, this launches a guided creation wizard;
    pass `--no-interactive` (or any of those flags) to stay fully scriptable.

    Args:
        name: Artifact name, or a prompt to choose one in the interactive wizard.
        source: Registered trace adapter used to normalize the corpus.
        file: Exported trace file for a local build.
        root: Project artifact directory where the model is written.

    Raises:
        typer.BadParameter: Required inputs or provider settings are invalid.
    """
    # `--vendor <name>` is the deprecated alias for `--source <name> --pull`: it names the source
    # adapter and implies a live pull.
    import wmo.common.providers as providers
    from wmo.cli.ui import (
        BuildParams,
        RichBuildReporter,
        build_summary_panel,
        judge_model_default,
        run_build_wizard,
        serve_model_default,
    )
    from wmo.common.config.card import make_build_card, save_card
    from wmo.common.observability import (
        MeteredProvider,
        Phase,
        RunTracker,
        classify_build_call,
        save_run,
    )
    from wmo.common.observability.telemetry import (
        BuildTelemetryStats,
        TelemetryBuildReporter,
        capture_build_completed,
    )
    from wmo.common.providers import ProviderKind
    from wmo.common.providers.base import EmbedderKind
    from wmo.simulation.ingest import VendorPull
    from wmo.simulation.model.build import EmptyCorpusError
    from wmo.simulation.model.build import build as run_build
    from wmo.simulation.model.grounding import GROUNDER_KINDS
    from wmo.simulation.retrieval import get_embedder

    if vendor:
        source = vendor
        pull = True

    # Decide whether to run the wizard: explicit flag wins; otherwise auto when at a TTY and the
    # essential inputs (a name and a trace source - a file or a live pull) were not supplied.
    needs_input = name is None or (file is None and not pull)
    use_wizard = interactive if interactive is not None else (_console.is_terminal and needs_input)

    configured_worker = load_settings_or_abort(root).models.resolve("worker")
    use_configured_worker = configured_worker is not None and (
        provider is None or provider == configured_worker.provider
    )
    # Settle the serve provider here so the model default can follow it. A single hard-coded
    # Anthropic default wrote artifacts configured to call e.g. OpenAI with `claude-opus-4-8`.
    resolved_provider = (
        provider or (configured_worker.provider if configured_worker else None) or _BUILD_PROVIDER
    )
    resolved_model = (
        model
        or (configured_worker.model if use_configured_worker and configured_worker else None)
        or serve_model_default(resolved_provider)
    )
    if resolved_model is None and not use_wizard:
        raise typer.BadParameter(
            f"--provider {resolved_provider} has no default serve model: pass `--model <id>` "
            f"(`wmo build --provider {resolved_provider} --model <id> --file <export>`)"
        )
    params = BuildParams(
        name=name or DEFAULT_MODEL_NAME,
        source=source,
        file=file,
        pull=pull,
        project=project,
        api_key=api_key,
        since=since,
        limit=limit,
        provider=provider or (configured_worker.provider if configured_worker else None),
        # A wizard run re-picks provider and model from its own per-provider lists, so an
        # unresolved default here is only an unused suggestion; the flag path errored above.
        model=resolved_model or "",
        region=(
            region
            or (configured_worker.region if use_configured_worker and configured_worker else None)
        ),
        fidelity=fidelity,
        train_split=train_split,
        judge_model=judge_model,
        embed_provider=embed_provider,
        embed_model=embed_model,
        embed_dim=embed_dim,
    )
    if use_wizard:
        params = run_build_wizard(_console, params)
    elif params.file is None and not params.pull:
        # This guard also required `name is None`, so `wmo build --name x` (verbatim the
        # empty-state hint `wmo list` prints) fell through to a raw ValueError from the ingest
        # seam instead. A name is not a corpus: what is missing is the trace source.
        raise typer.BadParameter(
            "provide --file <export> or --pull (with --source), or run `wmo build` interactively"
        )
    if params.file and params.pull:
        raise typer.BadParameter("pass either --file or --pull, not both")
    _check_build_source(params.source)
    if params.file is not None:
        _check_build_file(params.file)
    if params.limit is not None and params.limit < 1:
        raise typer.BadParameter(f"--limit must be at least 1, got {params.limit}")
    # The wizard always resolves a provider; the flag path keeps its historical default.
    params.provider = params.provider or _BUILD_PROVIDER
    # The wizard may replace the configured worker's provider. Re-evaluate the match before
    # carrying provider-specific connection fields into the build config.
    use_configured_worker = (
        configured_worker is not None and params.provider == configured_worker.provider
    )

    # Flag-supplied names get the same whitespace-to-dash normalization as the wizard.
    params.name = normalize_name(params.name)
    try:
        validate_name(params.name)
    except ValueError as err:
        raise typer.BadParameter(str(err)) from None
    if params.name == "harbor":
        raise typer.BadParameter(
            "world model name 'harbor' is reserved: `wmo optimize harness <agent> harbor` selects "
            "the harbor benchmark environment; choose another name"
        )
    try:
        tier = FidelityTier(params.fidelity)
    except ValueError:
        tiers = ", ".join(t.value for t in FidelityTier)
        raise typer.BadParameter(
            f"unknown fidelity {params.fidelity!r}; choose one of: {tiers}"
        ) from None
    spec = FIDELITY_TIERS[tier]
    try:
        serve_provider = ProviderKind(params.provider)
    except ValueError:
        kinds = ", ".join(k.value for k in ProviderKind)
        raise typer.BadParameter(
            f"unknown provider {params.provider!r}; choose one of: {kinds}"
        ) from None
    try:
        embed_kind = EmbedderKind(params.embed_provider)
    except ValueError:
        kinds = ", ".join(k.value for k in EmbedderKind)
        raise typer.BadParameter(
            f"unknown embed provider {params.embed_provider!r}; choose one of: {kinds}"
        ) from None
    # A provider-backed embedder needs an embeddings model; fail fast, not deep inside embed().
    # The in-process kinds need none: hashing has no model, local carries its own default.
    if embed_kind not in (EmbedderKind.HASHING, EmbedderKind.LOCAL) and not params.embed_model:
        raise typer.BadParameter(
            f"--embed-provider {embed_kind.value} requires --embed-model "
            "(the embeddings model id / Azure embedding deployment)"
        )

    store = WorldModelStore(root)
    model_dir = str(store.model_dir(params.name))
    # Provider wiring (reuse-vs-separate embed config) lives in HarnessConfig.for_build, not here.
    config = HarnessConfig.for_build(
        serve_provider=serve_provider,
        serve_model=params.model,
        region=params.region,
        embed_provider=embed_kind,
        embed_model=params.embed_model,
        embed_dim=params.embed_dim,
        gepa_budget=spec.gepa_budget,
        train_split=params.train_split,
        judge_model=params.judge_model or judge_model_default(params.provider, params.model),
        trace_adapter=params.source,
    )
    if use_configured_worker and configured_worker is not None:
        config.providers[0] = config.providers[0].model_copy(
            update={
                "endpoint": configured_worker.endpoint,
                "deployment": configured_worker.deployment,
                "api_version": configured_worker.api_version,
            }
        )
    if grounder not in GROUNDER_KINDS:
        raise typer.BadParameter(
            f"unknown grounder {grounder!r}; choose one of: {', '.join(GROUNDER_KINDS)}"
        )
    # Agentic-mode flags (CLI-only, not in the wizard): persisted to config.toml so serve/load
    # pick them up; knowledge additionally seeds knowledge/ during this build.
    config.knowledge = knowledge
    config.reasoning = reasoning
    config.grounder = grounder
    # Fail fast: ping the serve provider (and the embed path, if provider-backed) before spending
    # any rollouts. A missing SDK or bad creds otherwise surfaces only deep inside GEPA, which
    # silently swallows it and "succeeds" with a useless held-out-0.0 model.
    if not use_wizard:
        # The wizard already live-pinged the serve provider and embedder inline.
        _verify_or_abort(config, chain=chain)

    # Meter the build at the provider boundary; `classify_build_call` splits judge vs GEPA by
    # system prompt. Rollouts/reflection may ride the failover chain, but the judge (GEPA's
    # fitness metric) is PINNED to the single configured backend - a judge that silently switches
    # models mid-build scores candidates on different scales. Both wrappers share one tracker,
    # so cost/tokens still land in a single run record.
    tracker = RunTracker(run_id=uuid.uuid4().hex, kind="build")
    metered = MeteredProvider(
        _provider_or_chain_or_abort(config.serve_provider_config(), chain),
        tracker,
        classify=classify_build_call,
    )
    metered_judge = metered
    if config.judge_model and config.judge_model != config.serve_provider_config().model:
        judge_cfg = config.serve_provider_config().model_copy(update={"model": config.judge_model})
        metered_judge = MeteredProvider(
            providers.get_provider(judge_cfg), tracker, classify=classify_build_call
        )
    build_stats = BuildTelemetryStats()
    with tracker.timed(), RichBuildReporter(_console, params.name) as reporter:
        try:
            result = run_build(
                config,
                file=None if params.pull else params.file,
                vendor=(
                    VendorPull(
                        api_key=params.api_key,
                        project=params.project,
                        since=params.since,
                        limit=params.limit,
                    )
                    if params.pull
                    else None
                ),
                root=model_dir,
                serve_provider=metered,
                judge_provider=metered_judge,
                embedder=get_embedder(config),
                reporter=TelemetryBuildReporter(reporter, build_stats),
                max_fidelity=spec.config_search,
                fidelity_budget=spec.search_budget,
                full_search=spec.full_ladder,
                cheap_search=spec.cheap_frontier_only,
                estimate_only=spec.estimate_only,
                drop_degenerate=drop_degenerate,
                gepa_val_cap=spec.gepa_val_cap or None,
                # One cap per build, never both: a pull is already sliced to `--limit` inside
                # `from_vendor`, so repeating it post-filter cannot restore what
                # `--drop-degenerate` removed and would only read as a promise of N usable
                # traces that the pull transport cannot keep.
                limit=None if params.pull else params.limit,
            )
        except EmptyCorpusError:
            # The one ingest outcome a user causes and can fix: the export does not parse under
            # the pinned --source. Only this type is caught, so real build bugs still surface.
            raise _empty_corpus_error(None if params.pull else params.file, params.source) from None
    record = tracker.record_summary()
    save_run(record, ArtifactPaths(model_dir).runs)
    # The card is additive metadata; a write failure (disk full, permissions) must not make an
    # otherwise-complete build exit non-zero and then block retries with "already exists".
    try:
        save_card(
            make_build_card(
                name=params.name,
                provider=params.provider,
                model_id=params.model,
                traces=build_stats.input_trace_count,
                steps=build_stats.input_step_count,
                built_at=datetime.now(UTC).isoformat(),
                source=Path(params.file).name if params.file else params.vendor,
            ),
            model_dir,
        )
    except OSError as err:
        _console.print(f"[yellow]warning[/yellow]: could not write card.json: {err}")
    capture_build_completed(
        stats=build_stats,
        gepa_budget=spec.gepa_budget,
        rollouts_used=result.metrics.rollouts_used,
        frontier_size=len(result.frontier),
        record=record,
        root=root,
    )

    _console.print(build_summary_panel(store.info(params.name), model_dir))
    auto_report = Path(model_dir) / "auto_fidelity.json"
    if auto_report.exists():
        auto = json.loads(auto_report.read_text(encoding="utf-8"))
        # The low tier writes an estimate with no scores (no search ran); higher tiers write
        # the searched scores. Render each honestly rather than an empty "(; 0 traces)".
        if auto.get("scores"):
            scores = ", ".join(f"{k}={v:.3f}" for k, v in auto["scores"].items())
            provenance = f"searched: {scores}; {auto['val_traces']} held-out traces"
        else:
            provenance = "signature estimate - no search"
        _console.print(
            f"[bold]max-fidelity config[/bold]: [bold]{auto['winner_label']}[/bold] "
            f"({provenance}) - activate with `wmo serve --max-fidelity`"
        )
    _console.print(
        f"[bold]run[/bold] {record.run_id[:8]}: {record.duration_seconds:.1f}s, "
        f"{record.total.total_tokens} tokens, ${record.total.cost_usd:.4f} "
        f"({record.total.calls} calls)"
    )
    for phase in (Phase.GEPA, Phase.JUDGE):
        bucket = record.by_phase.get(phase)
        if bucket is not None:
            _console.print(
                f"  {phase.value}: {bucket.total_tokens} tokens, "
                f"${bucket.cost_usd:.4f} ({bucket.calls} calls)"
            )


def _verify_or_abort(config: HarnessConfig, chain: str | None = None) -> None:
    """Ping the serve provider (and any provider-backed embedder) and abort on failure.

    Runs before any rollouts so a missing SDK or bad creds fails loudly and immediately, instead of
    being swallowed inside GEPA and yielding a useless model. Raises `typer.Exit(1)` with an
    actionable hint (`uv sync` for a missing SDK; "check creds / model id" otherwise).
    """
    import wmo.common.providers as providers
    from wmo.common.providers import verify_all, verify_embedder
    from wmo.common.providers.base import EmbedderKind

    checks = [(config.serve_provider_config(), False)]
    if config.embed_provider not in (EmbedderKind.HASHING, EmbedderKind.LOCAL):
        checks.append((config.embed_provider_config(), True))

    failed = False
    for cfg, is_embed in checks:
        label = f"embed:{cfg.kind.value}" if is_embed else cfg.kind.value
        serve_provider = None if is_embed else _provider_or_chain_or_abort(cfg, chain)
        if is_embed:
            _console.print(f"verifying {label}…")
            result = verify_embedder(cfg)
        elif isinstance(serve_provider, providers.WaterfallProvider):
            # Verify the provider the build will actually use - with a chain active, that means
            # pinging every rung (a broken fallback must fail here, not hours into the build).
            label = f"{label} chain (.wmo/fallback.toml)"
            _console.print(f"verifying {label}…")
            result = serve_provider.verify()
        else:
            _console.print(f"verifying {label}…")
            result = verify_all([cfg])[0]
        if result.ok:
            _console.print(f"  {_CHECK} {label} ({result.model}) reachable")
            continue
        failed = True
        _console.print(f"  [red]✗ {label} ({result.model}) failed[/red]: {result.detail}")
        hint = escape(_credential_hint(cfg.kind, result.detail))
        _console.print(f"    [yellow]{hint}[/yellow]")
    if failed:
        raise typer.Exit(1)
