"""`wmh` CLI — ingestion UI and operator console for the harness.

Deliberately small. The lifecycle is:
    providers verify -> build -> list -> serve / demo / play
`build` creates the project artifact directory itself, so there is no separate init step. World
models are named (`--name`), stored under `<root>/models/<name>/`, and listed with `wmh list`.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

import wmh.providers as providers
from wmh.cli.ui import (
    BuildParams,
    RichBuildReporter,
    build_summary_panel,
    models_table,
    run_build_wizard,
    run_play_repl,
    select_model,
)
from wmh.config import (
    ARTIFACT_DIR,
    DEFAULT_MODEL_NAME,
    PROVIDER_ENV_VARS,
    ArtifactPaths,
    HarnessConfig,
    WorldModelStore,
    load_config,
    load_env_file,
    load_settings,
    normalize_name,
    set_telemetry_enabled,
    settings_path,
    validate_name,
)
from wmh.engine.build import build as run_build
from wmh.engine.demo import run_demo
from wmh.engine.eval import EvalReport, evaluate_files
from wmh.engine.eval_suites import (
    discover_eval_suites,
    list_eval_results,
    resolve_eval_suite,
    result_path,
)
from wmh.engine.loader import load_world_model
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.ingest import VendorPull
from wmh.optimize.judge import LLMJudge, RubricJudge
from wmh.providers import ProviderConfig, ProviderKind, verify_all, verify_embedder
from wmh.providers.base import EmbedderKind
from wmh.retrieval import HashingEmbedder, get_embedder
from wmh.serving.server import create_app
from wmh.telemetry import (
    BuildTelemetryStats,
    TelemetryBuildReporter,
    capture_build_completed,
    capture_eval_completed,
    settings_root_from_results_root,
)
from wmh.tracking import MeteredProvider, Phase, RunTracker, classify_build_call, save_run

app = typer.Typer(
    help="World Model Harness: a frontier LLM acts as your agent's environment.",
    no_args_is_help=True,
)


providers_app = typer.Typer(help="Manage and verify LLM providers.", no_args_is_help=True)
examples_app = typer.Typer(
    help="List and launch self-contained task examples.", no_args_is_help=True
)
config_app = typer.Typer(help="Manage local harness config.", no_args_is_help=True)
app.add_typer(providers_app, name="providers")
app.add_typer(examples_app, name="examples")
app.add_typer(config_app, name="config")
_console = Console()
_CHECK = "[green]✓[/green]"

# Module-level singleton: a typer.Argument call can't be a default inline (ruff B008).
_EVAL_TOKENS = typer.Argument(
    None,
    help="Trace files to score, or eval flow: list | run <suite> | results optional-suite.",
)


@dataclass(frozen=True)
class _EvalOptions:
    prompt_file: str | None
    train_split: float
    embed_dim: int
    use_rag: bool
    judge: str
    sample_turns: str
    seed: int
    top_k: int


@config_app.command("telemetry")
def config_telemetry(
    action: str = typer.Argument("status", help="status | enable | disable"),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir holding local settings."),
) -> None:
    """View or change project-local usage telemetry settings."""
    normalized = action.lower()
    if normalized == "status":
        settings = load_settings(root)
    elif normalized == "enable":
        settings = set_telemetry_enabled(True, root)
    elif normalized == "disable":
        settings = set_telemetry_enabled(False, root)
    else:
        raise typer.BadParameter("action must be one of: status, enable, disable")
    state = "enabled" if settings.telemetry.enabled else "disabled"
    _console.print(f"telemetry {state} ({settings_path(root)})")


@providers_app.command("verify")
def providers_verify(
    name: str = typer.Option(None, "--name", help="Verify one model's providers (default: all)."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir."),
) -> None:
    """Ping every configured provider (completion + embed path) and report status.

    Gathers provider configs from the built world models (one `--name`, or all of them, deduped by
    kind+model), so a brand-new project with nothing built yet has nothing to verify. The phi embed
    path of each model is checked too, unless it is the offline (creds-free) hashing embedder.
    """
    store = WorldModelStore(root)
    names = [name] if name is not None else store.list_names()
    if not names:
        _console.print("[yellow]no world models built yet[/yellow]; run `wmh build --name <name>`")
        return
    configs: list[HarnessConfig] = []
    for model_name in names:
        try:
            model_dir = str(store.resolve(model_name))
        except (FileNotFoundError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        configs.append(load_config(model_dir))

    # Dedup completion providers by kind+model across all selected models.
    seen: set[tuple[str, str]] = set()
    providers: list[ProviderConfig] = []
    for config in configs:
        for pc in config.providers:
            key = (pc.kind.value, pc.model)
            if key not in seen:
                seen.add(key)
                providers.append(pc)
    for result in verify_all(providers):
        mark = "[green]ok[/green]" if result.ok else "[red]fail[/red]"
        _console.print(f"{mark} {result.kind.value} ({result.model}) {result.detail}")

    # Verify each distinct provider-backed embed path (skip the offline hashing embedder).
    embed_seen: set[tuple[str, str]] = set()
    for config in configs:
        if config.embed_provider is EmbedderKind.HASHING:
            continue
        embed_config = config.embed_provider_config()
        key = (embed_config.kind.value, embed_config.model)
        if key in embed_seen:
            continue
        embed_seen.add(key)
        result = verify_embedder(embed_config)
        mark = "[green]ok[/green]" if result.ok else "[red]fail[/red]"
        _console.print(f"{mark} embed:{result.kind.value} ({result.model}) {result.detail}")


@examples_app.command("list")
def examples_list() -> None:
    """List self-contained example tasks."""
    examples = _discover_examples()
    if not examples:
        _console.print("[yellow]no examples found[/yellow]")
        return
    for example in examples:
        _console.print(example.name)


@examples_app.command(
    "run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def examples_run(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Example task name."),
) -> None:
    """Run an example's local launcher, forwarding any extra args after `--`."""
    example_dir = _resolve_example(name)
    runner = example_dir / "run.sh"
    if not runner.exists():
        raise typer.BadParameter(f"example {name!r} has no run.sh launcher")
    result = subprocess.run([str(runner), *ctx.args], cwd=example_dir, check=False)
    raise typer.Exit(result.returncode)


@app.command("build")
def build(
    name: str = typer.Option(None, "--name", help="Name for this world model."),
    file: str = typer.Option(None, "--file", help="Path to exported traces (OTLP-JSON / JSONL)."),
    vendor: str = typer.Option(None, "--vendor", help="Vendor name to pull traces via SDK."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir holding all world models."),
    provider: str = typer.Option(
        None, "--provider", help="Provider that serves the model (default: bedrock)."
    ),
    model: str = typer.Option("us.anthropic.claude-opus-4-8", help="Serve provider model id."),
    region: str = typer.Option(None, help="AWS region (Bedrock)."),
    gepa_budget: int = typer.Option(10, help="GEPA iterations (each ~one capped valset pass)."),
    train_split: float = typer.Option(
        0.8, help="Train/held-out ratio for GEPA's internal split (lower = bigger valset)."
    ),
    embed_provider: str = typer.Option(
        "hashing", help="phi embedder: hashing (offline) | bedrock | openai | azure."
    ),
    embed_model: str = typer.Option(None, help="Embeddings model id / Azure embedding deployment."),
    embed_dim: int = typer.Option(512, help="phi dimensionality (index + query must agree)."),
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
    """
    # Decide whether to run the wizard: explicit flag wins; otherwise auto when at a TTY and the
    # essential inputs (a name and a trace source) were not supplied.
    needs_input = name is None or (file is None and vendor is None)
    use_wizard = interactive if interactive is not None else (_console.is_terminal and needs_input)

    params = BuildParams(
        name=name or DEFAULT_MODEL_NAME,
        file=file,
        vendor=vendor,
        provider=provider,
        model=model,
        region=region,
        gepa_budget=gepa_budget,
        train_split=train_split,
        embed_provider=embed_provider,
        embed_model=embed_model,
        embed_dim=embed_dim,
    )
    if use_wizard:
        params = run_build_wizard(_console, params)
    elif name is None and file is None and vendor is None:
        raise typer.BadParameter("provide --file (or --vendor), or run `wmh build` interactively")
    # The wizard always resolves a provider; the flag path keeps its historical default.
    params.provider = params.provider or "bedrock"

    # Flag-supplied names get the same whitespace-to-dash normalization as the wizard.
    params.name = normalize_name(params.name)
    try:
        validate_name(params.name)
    except ValueError as err:
        raise typer.BadParameter(str(err)) from None
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
    if embed_kind is not EmbedderKind.HASHING and not params.embed_model:
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
        gepa_budget=params.gepa_budget,
        train_split=params.train_split,
    )
    # Fail fast: ping the serve provider (and the embed path, if provider-backed) before spending
    # any rollouts. A missing SDK or bad creds otherwise surfaces only deep inside GEPA, which
    # silently swallows it and "succeeds" with a useless held-out-0.0 model.
    if not use_wizard:
        # The wizard already live-pinged the serve provider and embedder inline.
        _verify_or_abort(config)

    # Meter the build at the provider boundary: the one serve provider drives GEPA rollouts,
    # reflection, and the judge, so wrapping it captures all build LLM cost/tokens without touching
    # the optimizer. `classify_build_call` splits judge vs GEPA by system prompt.
    tracker = RunTracker(run_id=uuid.uuid4().hex, kind="build")
    metered = MeteredProvider(
        providers.get_provider(config.serve_provider_config()),
        tracker,
        classify=classify_build_call,
    )
    build_stats = BuildTelemetryStats()
    with tracker.timed(), RichBuildReporter(_console, params.name) as reporter:
        result = run_build(
            config,
            file=params.file,
            vendor=VendorPull() if params.vendor else None,
            root=model_dir,
            serve_provider=metered,
            embedder=get_embedder(config),
            reporter=TelemetryBuildReporter(reporter, build_stats),
        )
    record = tracker.record_summary()
    save_run(record, ArtifactPaths(model_dir).runs)
    capture_build_completed(
        stats=build_stats,
        gepa_budget=params.gepa_budget,
        rollouts_used=result.metrics.rollouts_used,
        frontier_size=len(result.frontier),
        record=record,
        root=root,
    )

    _console.print(build_summary_panel(store.info(params.name), model_dir))
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


def _verify_or_abort(config: HarnessConfig) -> None:
    """Ping the serve provider (and any provider-backed embedder) and abort on failure.

    Runs before any rollouts so a missing SDK or bad creds fails loudly and immediately, instead of
    being swallowed inside GEPA and yielding a useless model. Raises `typer.Exit(1)` with an
    actionable hint (`uv sync` for a missing SDK; "check creds / model id" otherwise).
    """
    checks = [(config.serve_provider_config(), False)]
    if config.embed_provider is not EmbedderKind.HASHING:
        checks.append((config.embed_provider_config(), True))

    failed = False
    for cfg, is_embed in checks:
        label = f"embed:{cfg.kind.value}" if is_embed else cfg.kind.value
        _console.print(f"verifying {label}…")
        result = verify_embedder(cfg) if is_embed else verify_all([cfg])[0]
        if result.ok:
            _console.print(f"  {_CHECK} {label} ({result.model}) reachable")
            continue
        failed = True
        _console.print(f"  [red]✗ {label} ({result.model}) failed[/red]: {result.detail}")
        if "No module named" in result.detail:
            # SDKs are core deps; a missing module means the env is stale or hand-rolled.
            _console.print("    [yellow]run `uv sync` to install the provider SDKs[/yellow]")
        else:
            envs = ", ".join(PROVIDER_ENV_VARS.get(cfg.kind, []))
            hint = f" ({envs})" if envs else ""
            _console.print(
                f"    [yellow]check the model id and that your credentials are set{hint}[/yellow]"
            )
    if failed:
        raise typer.Exit(1)


@app.command("list")
def list_models(root: str = typer.Option(ARTIFACT_DIR, help="Project dir to list.")) -> None:
    """List every world model built under the project dir."""
    infos = WorldModelStore(root).list_info()
    if not infos:
        _console.print("[yellow]no world models built yet[/yellow]; run `wmh build --name <name>`")
        return
    _console.print(models_table(infos))


@app.command("serve")
def serve(
    name: list[str] = typer.Option(  # noqa: B008 - typer reads option defaults at definition time
        None, "--name", help="World model(s) to serve. Repeatable; default: all built ones."
    ),
    port: int = typer.Option(8000, help="Port for the local backend."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir to serve from."),
) -> None:
    """Run the local FastAPI backend so agents can step against world models over HTTP.

    Serves every built model by default, or just the `--name` ones. Routes are namespaced:
    `/world_models/{name}/sessions` and `.../step`.
    """
    names = list(name) if name else None
    # Bad --name input (unsafe segment, unknown model, nothing built) is a usage error,
    # not a traceback; load the models before uvicorn takes over the process.
    try:
        server_app = create_app(root, names=names)
    except (ValueError, FileNotFoundError) as err:
        raise typer.BadParameter(str(err)) from None
    uvicorn.run(server_app, host="127.0.0.1", port=port)


@app.command("eval")
def eval_(  # noqa: A001 - `eval` is the user-facing command name; the builtin isn't used here
    tokens: list[str] | None = _EVAL_TOKENS,
    prompt_file: str | None = typer.Option(
        None, "--prompt", help="Prompt file; default=BASE_ENV_PROMPT."
    ),
    provider: str = typer.Option("bedrock", "--provider", help="Provider running the model."),
    model: str = typer.Option("us.anthropic.claude-opus-4-8", help="Model id."),
    region: str | None = typer.Option(None, help="AWS region (Bedrock)."),
    train_split: float | None = typer.Option(
        None, help="Train/holdout ratio per file (default: 0.7, or suite config)."
    ),
    embed_dim: int | None = typer.Option(
        None, help="phi dimensionality for the offline embedder (default: 512, or suite config)."
    ),
    rag: bool | None = typer.Option(
        None, "--rag/--no-rag", help="Enable retrieval, or disable it for zero-shot replay."
    ),
    judge: str | None = typer.Option(
        None, help="Scorer: rubric (5-dim) | match (functional). Default: rubric, or suite config."
    ),
    sample_turns: str | None = typer.Option(
        None, help="Turns scored per trace: all | sampled (5). Default: all, or suite config."
    ),
    seed: int | None = typer.Option(None, help="Seed for reproducible turn sampling."),
    top_k: int | None = typer.Option(
        None, help="Retrieved demos per step (default: 5, or suite config)."
    ),
    out: str | None = typer.Option(None, help="Optional path to write the full JSON report."),
    examples_root: str | None = typer.Option(
        None, help="Directory containing example eval suites. Default: repo-local examples/."
    ),
    results_root: str = typer.Option(
        f"{ARTIFACT_DIR}/evals", help="Local directory for named eval result JSON."
    ),
    limit: int = typer.Option(20, help="Rows to show for `wmh eval results`."),
) -> None:
    """Score reconstruction fidelity, or run named example-local eval suites.

    Flows:
    - `wmh eval <trace files...>`: ad hoc replay scoring.
    - `wmh eval list`: list named suites under `examples/<task>/evals/`.
    - `wmh eval run <suite>`: run a suite and save a local JSON result.
    - `wmh eval results optional-suite`: summarize local suite results.
    """
    args = tokens or []
    suite_root = str(_examples_root()) if examples_root is None else examples_root
    if args and args[0] == "list":
        if len(args) != 1:
            raise typer.BadParameter("usage: wmh eval list")
        _eval_list(suite_root)
        return
    if args and args[0] == "results":
        if len(args) > 2:
            raise typer.BadParameter("usage: wmh eval results [suite]")
        suite_filter = args[1] if len(args) == 2 else None
        _eval_results(results_root, suite_root, suite_filter, limit=limit)
        return
    if args and args[0] == "run":
        if len(args) != 2:
            raise typer.BadParameter("usage: wmh eval run <suite>")
        _eval_run_suite(
            args[1],
            examples_root=suite_root,
            results_root=results_root,
            prompt_file=prompt_file,
            provider=provider,
            model=model,
            region=region,
            train_split=train_split,
            embed_dim=embed_dim,
            rag=rag,
            judge=judge,
            sample_turns=sample_turns,
            seed=seed,
            top_k=top_k,
            out=out,
        )
        return
    if not args:
        raise typer.BadParameter(
            "provide trace files, or use `wmh eval list`, `wmh eval run <suite>`, "
            "or `wmh eval results`"
        )

    options = _eval_options(
        prompt_file=prompt_file,
        train_split=train_split,
        embed_dim=embed_dim,
        rag=rag,
        judge=judge,
        sample_turns=sample_turns,
        seed=seed,
        top_k=top_k,
    )
    report = _run_eval_files(
        [Path(f) for f in args],
        options,
        provider=provider,
        model=model,
        region=region,
    )
    _print_eval_report(report)
    if out:
        _write_ad_hoc_eval_report(Path(out), report)
    capture_eval_completed(
        mode="ad_hoc",
        file_count=len(args),
        scored_step_count=report.total_steps,
        rag_enabled=options.use_rag,
        judge_mode=options.judge,
        sample_turns=options.sample_turns,
        train_split=options.train_split,
        top_k=options.top_k,
        root=ARTIFACT_DIR,
    )


def _eval_list(examples_root: str) -> None:
    suites = discover_eval_suites(examples_root)
    if not suites:
        _console.print("[yellow]no eval suites found[/yellow]")
        return
    table = Table(title="Eval suites")
    table.add_column("Suite", no_wrap=True)
    table.add_column("Files")
    table.add_column("Split")
    table.add_column("Scorer")
    table.add_column("Description")
    for suite in suites:
        table.add_row(
            suite.id,
            ", ".join(suite.config.files),
            f"{suite.config.train_split:.2f}",
            suite.config.judge,
            suite.config.description or "",
        )
    _console.print(table)


def _eval_results(
    results_root: str,
    examples_root: str,
    suite_filter: str | None,
    *,
    limit: int,
) -> None:
    resolved_suite = suite_filter
    if suite_filter is not None:
        try:
            resolved_suite = resolve_eval_suite(suite_filter, examples_root).id
        except ValueError:
            resolved_suite = suite_filter
    summaries = list_eval_results(results_root, resolved_suite, limit=limit)
    if not summaries:
        _console.print("[yellow]no eval results found[/yellow]")
        return
    table = Table(title="Eval results")
    table.add_column("Suite", no_wrap=True)
    table.add_column("Run")
    table.add_column("Started")
    table.add_column("Model")
    table.add_column("Fidelity", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("Path")
    for summary in summaries:
        table.add_row(
            summary.suite,
            summary.run_id[:8],
            summary.started_at,
            summary.model,
            f"{summary.overall_fidelity:.3f}±{summary.overall_std:.3f}",
            str(summary.total_steps),
            str(summary.path),
        )
    _console.print(table)


def _eval_run_suite(
    selector: str,
    *,
    examples_root: str,
    results_root: str,
    prompt_file: str | None,
    provider: str,
    model: str,
    region: str | None,
    train_split: float | None,
    embed_dim: int | None,
    rag: bool | None,
    judge: str | None,
    sample_turns: str | None,
    seed: int | None,
    top_k: int | None,
    out: str | None,
) -> None:
    suite = resolve_eval_suite(selector, examples_root)
    suite_prompt = suite.resolve_prompt()
    options = _eval_options(
        prompt_file=prompt_file or (str(suite_prompt) if suite_prompt is not None else None),
        train_split=train_split if train_split is not None else suite.config.train_split,
        embed_dim=embed_dim if embed_dim is not None else suite.config.embed_dim,
        rag=rag if rag is not None else not suite.config.no_rag,
        judge=judge or suite.config.judge,
        sample_turns=sample_turns or suite.config.sample_turns,
        seed=seed if seed is not None else suite.config.seed,
        top_k=top_k if top_k is not None else suite.config.top_k,
    )
    files = suite.resolve_files()
    report = _run_eval_files(files, options, provider=provider, model=model, region=region)
    _print_eval_report(report)

    run_id = uuid4().hex
    started_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    destination = Path(out) if out else result_path(results_root, suite, run_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "started_at": started_at,
        "suite": suite.id,
        "suite_path": str(suite.path),
        "suite_config": suite.config.model_dump(mode="json"),
        "config": {
            "provider": provider,
            "model": model,
            "region": region,
            "prompt": options.prompt_file,
            "files": [str(path) for path in files],
            "train_split": options.train_split,
            "top_k": options.top_k,
            "sample_turns": options.sample_turns,
            "seed": options.seed,
            "rag": options.use_rag,
            "judge": options.judge,
            "embed_dim": options.embed_dim,
        },
        "report": _eval_report_payload(report),
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _console.print(f"wrote eval result -> {destination}")
    capture_eval_completed(
        mode="suite",
        file_count=len(files),
        scored_step_count=report.total_steps,
        rag_enabled=options.use_rag,
        judge_mode=options.judge,
        sample_turns=options.sample_turns,
        train_split=options.train_split,
        top_k=options.top_k,
        root=settings_root_from_results_root(results_root),
    )


def _eval_options(
    *,
    prompt_file: str | None,
    train_split: float | None,
    embed_dim: int | None,
    rag: bool | None,
    judge: str | None,
    sample_turns: str | None,
    seed: int | None,
    top_k: int | None,
) -> _EvalOptions:
    split = 0.7 if train_split is None else train_split
    dim = 512 if embed_dim is None else embed_dim
    retrieval = True if rag is None else rag
    scorer = "rubric" if judge is None else judge
    turns = "all" if sample_turns is None else sample_turns
    rng_seed = 0 if seed is None else seed
    demos = 5 if top_k is None else top_k
    if not 0.0 < split < 1.0:
        raise typer.BadParameter("--train-split must be between 0 and 1")
    if dim <= 0:
        raise typer.BadParameter("--embed-dim must be positive")
    if demos < 0:
        raise typer.BadParameter("--top-k must be >= 0")
    if scorer not in {"rubric", "match"}:
        raise typer.BadParameter("--judge must be one of: rubric, match")
    if turns not in {"all", "sampled"}:
        raise typer.BadParameter("--sample-turns must be one of: all, sampled")
    return _EvalOptions(
        prompt_file=prompt_file,
        train_split=split,
        embed_dim=dim,
        use_rag=retrieval,
        judge=scorer,
        sample_turns=turns,
        seed=rng_seed,
        top_k=demos,
    )


def _run_eval_files(
    files: list[Path],
    options: _EvalOptions,
    *,
    provider: str,
    model: str,
    region: str | None,
) -> EvalReport:
    for path in files:
        if not path.exists():
            raise typer.BadParameter(f"trace file not found: {path}")
    try:
        serve_provider = ProviderKind(provider)
    except ValueError:
        kinds = ", ".join(k.value for k in ProviderKind)
        raise typer.BadParameter(f"unknown provider {provider!r}; choose one of: {kinds}") from None
    llm = providers.get_provider(ProviderConfig(kind=serve_provider, model=model, region=region))
    prompt = (
        Path(options.prompt_file).read_text(encoding="utf-8")
        if options.prompt_file
        else BASE_ENV_PROMPT
    )
    embedder = HashingEmbedder(dim=options.embed_dim) if options.use_rag else None
    scorer = RubricJudge(llm) if options.judge == "rubric" else LLMJudge(llm)
    return evaluate_files(
        files,
        prompt,
        llm,
        scorer,
        embedder=embedder,
        train_split=options.train_split,
        top_k=options.top_k,
        sample_turns=options.sample_turns,
        seed=options.seed,
    )


def _print_eval_report(report: EvalReport) -> None:
    for name, rep in report.per_file.items():
        _console.print(f"  {name:28} {rep.summary()}")
    _console.print(
        f"[bold]OVERALL[/bold] fidelity={report.overall_fidelity:.3f}±{report.overall_std:.3f} "
        f"over {report.total_steps} held-out steps"
    )


def _write_ad_hoc_eval_report(path: Path, report: EvalReport) -> None:
    path.write_text(
        json.dumps({n: r.model_dump(mode="json") for n, r in report.per_file.items()}, indent=2),
        encoding="utf-8",
    )
    _console.print(f"wrote full report -> {path}")


def _eval_report_payload(report: EvalReport) -> dict[str, object]:
    return {
        "overall_fidelity": report.overall_fidelity,
        "overall_std": report.overall_std,
        "total_steps": report.total_steps,
        "per_file": {name: rep.model_dump(mode="json") for name, rep in report.per_file.items()},
    }


@app.command("demo")
def demo(
    name: str = typer.Option(None, "--name", help="World model to demo (default: the only one)."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir."),
) -> None:
    """Demo the harness: an LLM agent makes a tool call vs the world model; show prompt+output."""
    wm, _resolved_name, provider = _load_model(name, root)
    # Seed the demo agent from whatever steps the index holds.
    examples = wm.sample_steps(3)
    result = run_demo(wm, provider, examples)
    _console.print(f"[bold]agent action[/bold]: {result.agent_action.model_dump()}")
    _console.print(f"[bold]env prompt[/bold]:\n{result.env_prompt}")
    _console.print(f"[bold]observation[/bold]: {result.observation.model_dump()}")


@app.command("play")
def play(
    name: str = typer.Option(None, "--name", help="World model to play (default: the only one)."),
    task: str = typer.Option(None, "--task", help="Task to seed the session with."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir."),
) -> None:
    """Step into the environment yourself: type actions, the world model returns observations."""
    wm, resolved_name, _provider = _load_model(name, root)
    run_play_repl(_console, wm, resolved_name, task)


def _resolve_name(store: WorldModelStore, name: str | None) -> str:
    """Resolve which model to run: explicit `--name`, an interactive picker, or the sole model.

    With `--name`, validate it exists. Otherwise, when several models are built on an interactive
    terminal, show a numbered picker; on a non-TTY (or a single model) defer to `store.resolve`,
    which returns the lone model or raises a helpful "pass --name" error. Store errors
    (unknown/ambiguous name) are turned into a clean `typer.BadParameter` rather than a traceback.
    """
    try:
        if name is not None:
            store.resolve(name)  # validates existence, raising a friendly error if missing
            return name
        # Only enumerate full model summaries when we actually need the picker (>1 model on a TTY).
        # `list_names` is cheap (a dir scan); `list_info` reads every config/metrics/frontier file.
        if _console.is_terminal and len(store.list_names()) > 1:
            return select_model(_console, store.list_info())
        return store.resolve(None).name
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _examples_root() -> Path:
    """Repo-local examples directory."""
    return Path(__file__).resolve().parents[2] / "examples"


def _discover_examples() -> list[Path]:
    root = _examples_root()
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and _is_safe_example_name(path.name)
        and ((path / "traces.otel.jsonl").exists() or (path / "run.sh").exists())
    )


def _is_safe_example_name(name: str) -> bool:
    """Whether `name` would resolve via `wmh examples run` — keeps list/hint/run in agreement."""
    try:
        validate_name(name)
    except ValueError:
        return False
    return True


def _resolve_example(name: str) -> Path:
    # An unsafe segment (spaces, path separators, ...) is simply not an example name; fall
    # through to the same "unknown example" usage error instead of a ValueError traceback.
    try:
        example_dir = _examples_root() / validate_name(name)
    except ValueError:
        example_dir = None
    if example_dir is not None and example_dir.is_dir():
        return example_dir
    available = ", ".join(path.name for path in _discover_examples())
    hint = f" (available: {available})" if available else ""
    raise typer.BadParameter(f"unknown example {name!r}{hint}")


def _load_model(name: str | None, root: str):  # noqa: ANN202 - (WorldModel, name, Provider)
    """Resolve + load a named world model (or the single built one) with its serve provider.

    Returns `(world_model, resolved_name, provider)` so callers can reuse the provider without
    re-reading config / reconstructing it.
    """
    store = WorldModelStore(root)
    resolved_name = _resolve_name(store, name)
    world_model, provider = load_world_model(
        store.resolve(resolved_name), telemetry_root=store.root
    )
    return world_model, resolved_name, provider


if __name__ == "__main__":
    app()


def main() -> None:
    """CLI entry point: load `.env` from the working directory (so wizard-saved provider keys
    persist across sessions), then dispatch. Kept out of import time so importing the module
    never mutates os.environ."""
    load_env_file()
    app()
