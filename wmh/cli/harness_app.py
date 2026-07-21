"""Agent-harness optimization and inspection under `.wmh/harnesses/<name>/`.

A harness is the scaffold an agent runs with: prompt surfaces, a tool policy, loop parameters, and
skills, stored as immutable numbered versions with movable aliases (`champion` is what runs by
default). `init` writes the baseline as v1; `list`/`show` inspect what exists; `wmh optimize`
searches for a better harness by **inverting the world model**. Delta variants are scored
closed-loop against it and gated on non-regression, so the environment model the traces built now
steers what the agent's scaffold should be. Run one closed-loop with
`wmh eval <tasks> --mode closed-loop --harness <name>[@ref]`.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path
from typing import Literal

import typer
from pydantic import BaseModel, ConfigDict, ValidationError
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from wmh.agents.default import default_agent
from wmh.agents.optimizer import optimizer_agent
from wmh.agents.project import AgentProject
from wmh.cli.model_roles import resolve_opt_in_model_provider, resolve_required_model_config
from wmh.config import ARTIFACT_DIR, WorldModelStore
from wmh.config.store import validate_name
from wmh.core.types import JsonObject
from wmh.engine import load_world_model
from wmh.engine.world_model import WorldModel
from wmh.evals.gold import GoldJudge
from wmh.evals.tasks import TaskSpec, load_tasks
from wmh.harness.create import ProposalRecord, create_harness
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_sandbox import E2B_TEMPLATE_ENV, resolve_e2b_template
from wmh.harness.population import (
    CandidateProposer,
    PopulationResult,
    SlotOutcome,
    write_json_atomic,
)
from wmh.harness.population import (
    optimize as optimize_population,
)
from wmh.harness.project_proposer import CandidateProject, ProjectCandidateProposer
from wmh.harness.proposer import ProviderDeltaProposer
from wmh.harness.runtime import DEFAULT_EVAL_EPISODE_TIMEOUT_S
from wmh.harness.scoring import RewardMode, Scorer, ScoreRequest
from wmh.harness.source_tree import HarnessSourceTree
from wmh.harness.store import CHAMPION_ALIAS, HarnessStore
from wmh.providers.base import Provider, ProviderConfig, ToolCallingProvider
from wmh.providers.registry import get_provider

# The default agent seed's literal CLI name: `wmh optimize pi harbor ...` starts from the
# built-in pi agent and publishes new versions under the store name "pi".
DEFAULT_SEED_AGENT = "pi"
_DEFAULT_HARBOR_ITERATIONS = 10
_HARBOR_ENVIRONMENT = "harbor"
_HARBOR_EXTRA_HINT = (
    "the harbor environment needs the harbor extra; run `uv sync --extra harbor` "
    "(or `pip install 'world-model-harness[harbor]'`)"
)

harness_app = typer.Typer(
    help="Inspect and initialize named, versioned agent harnesses.",
    no_args_is_help=True,
)
_console = Console()


@harness_app.command("list")
def list_harnesses(root: str = typer.Option(ARTIFACT_DIR, help="Project dir.")) -> None:
    """List every harness with its versions and aliases."""
    store = HarnessStore(root)
    names = store.list_names()
    if not names:
        _console.print(
            "[yellow]no harnesses yet[/yellow]; `wmh harness init <name>` creates the baseline"
        )
        return
    table = Table(title="Harnesses")
    table.add_column("Name", no_wrap=True)
    table.add_column("Versions", justify="right")
    table.add_column("Aliases")
    table.add_column("Doc hash (champion)")
    broken: list[tuple[str, str]] = []
    for name in names:
        try:
            doc = store.load(name)
            aliases = ", ".join(f"{a}=v{v}" for a, v in sorted(store.aliases(name).items()))
            table.add_row(
                name,
                f"{len(store.versions(name))}",
                aliases or "—",
                doc.doc_hash[:12],
            )
        except (ValueError, FileNotFoundError) as exc:  # one broken dir must not hide the rest
            broken.append((name, str(exc)))
    _console.print(table)
    for name, reason in broken:
        _console.print(f"[red]broken[/red] {name}: {reason}")


@harness_app.command("show")
def show_harness(
    name: str = typer.Argument(..., help="Harness name, optionally name@ref (version or alias)."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir."),
) -> None:
    """Print one harness version's surfaces."""
    base, _, ref = name.partition("@")
    try:
        doc = HarnessStore(root).load(base, ref or None)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _console.print(f"[bold]{doc.name}[/bold] v{doc.version}  doc_hash={doc.doc_hash[:12]}")
    for surface in doc.surfaces:
        budget = f"  budget={surface.budget}" if surface.budget is not None else ""
        _console.print(
            f"\n[bold]{surface.id}[/bold]  ({surface.kind.value}, "
            f"hash={surface.content_hash[:12]}{budget})"
        )
        _console.print(surface.content)


def optimize(
    ctx: typer.Context,
    name: str = typer.Argument(None, help="Agent name for the optimized harness."),
    model: str = typer.Argument(
        None,
        help="World model to optimize against (default: the only built one), or the literal "
        "'harbor' to optimize on real harbor benchmark tasks.",
    ),
    tasks_file: str = typer.Option(None, "--tasks", help="JSONL task file to optimize against."),
    holdout_file: str = typer.Option(
        None,
        "--holdout",
        help="Optional JSONL held-out task file: accepted deltas must also be no worse here.",
    ),
    backend: str = typer.Option(
        "local",
        "--backend",
        help="Where the harness PROCESS runs: local (in/from this process) or e2b (the real "
        "pi agent inside pooled E2B sandboxes). The environment is always the world model.",
    ),
    eval_concurrency: int | None = typer.Option(
        None,
        "--eval-concurrency",
        min=0,
        help="(task, attempt) cells run at once per eval. Default: 1 for local; "
        "0 (= all cells at once) for e2b.",
    ),
    e2b_template: str | None = typer.Option(
        None,
        "--e2b-template",
        envvar=E2B_TEMPLATE_ENV,
        help="Prebaked E2B sandbox template for --backend e2b (default: "
        "$WMH_E2B_TEMPLATE; without one, every sandbox bootstraps node + the pi runner deps).",
    ),
    seed: str = typer.Option(
        None,
        "--seed",
        help="Harness to start from, as a name or name@version (default: built-in baseline).",
    ),
    iterations: int = typer.Option(None, min=1, help="Propose-and-gate steps (the search budget)."),
    proposal_batch_size: int = typer.Option(
        1,
        "--proposal-batch-size",
        min=1,
        help="Sibling proposals generated against each selected parent.",
    ),
    k: int = typer.Option(3, min=1, help="Closed-loop passes per task per variant."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir."),
    archive_out: str = typer.Option(
        None, "--archive", help="Also write the full delta archive JSON here."
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip the cost confirmation prompt."),
    harbor_config: str = typer.Option(
        None,
        "--harbor-config",
        help="(harbor) Harbor JobConfig template, YAML or JSON: the task-environment config "
        "and harbor tuning, with exactly one dataset and no direct tasks.",
    ),
    task_ids_file: str = typer.Option(
        None,
        "--task-ids",
        help="(harbor) JSON file containing the exact task-id string list to optimize on.",
    ),
    attempts: int = typer.Option(
        None, "--attempts", min=1, help="(harbor) Attempts per task per candidate (default 1)."
    ),
    reward_key: str = typer.Option(
        None,
        "--reward-key",
        help="(harbor) Verifier reward key to optimize (default 'reward').",
    ),
    reward_mode: str = typer.Option(
        None,
        "--reward-mode",
        help="(harbor) raw | positive-binary (default positive-binary: reward > 0 passes).",
    ),
    harbor_retries: int = typer.Option(
        None,
        "--harbor-retries",
        min=0,
        help="(harbor) Harbor-level retries per failed trial (default 0).",
    ),
    episode_timeout: float = typer.Option(
        None,
        "--episode-timeout",
        min=0.001,
        help="(harbor) Wall seconds per evaluated episode (default 300; needs --backend e2b).",
    ),
    run_dir: str = typer.Option(
        None,
        "--run-dir",
        help="(harbor) Directory holding ALL durable run state (required for harbor).",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="(harbor) Continue the interrupted run recorded in --run-dir.",
    ),
    max_iterations_this_run: int = typer.Option(
        None,
        "--max-iterations-this-run",
        min=1,
        help="(harbor) Stop after this many new boundaries this invocation; continue later "
        "with --resume.",
    ),
) -> None:
    """Optimize an agent harness by searching against a world model or on harbor tasks.

    The optimizer proposes harness changes, scores them against the environment, and saves the
    best result as the new champion. Configure distinct agent and optimizer models with
    `models.agent` and `models.meta` in `.wmh/settings.toml`.

    With a world-model ENVIRONMENT, the backend controls where the worker process runs and the
    environment remains the world model. With the literal `harbor` ENVIRONMENT, complete-source
    candidates are scored on real benchmark tasks: --backend controls the task environment and
    worker placement (local = docker tasks + local pi; e2b = E2B tasks + sandboxed pi), while
    the PROPOSER project always runs in E2B in this version.
    """
    if model == _HARBOR_ENVIRONMENT:
        _optimize_harbor(
            ctx,
            name=name,
            backend=backend,
            e2b_template=e2b_template,
            iterations=iterations,
            harbor_config=harbor_config,
            task_ids_file=task_ids_file,
            attempts=attempts,
            reward_key=reward_key,
            reward_mode=reward_mode,
            harbor_retries=harbor_retries,
            episode_timeout=episode_timeout,
            run_dir_option=run_dir,
            resume=resume,
            max_iterations_this_run=max_iterations_this_run,
            root=root,
            yes=yes,
        )
        return
    harbor_only = [
        flag
        for flag, provided in (
            ("--harbor-config", harbor_config is not None),
            ("--task-ids", task_ids_file is not None),
            ("--attempts", attempts is not None),
            ("--reward-key", reward_key is not None),
            ("--reward-mode", reward_mode is not None),
            ("--harbor-retries", harbor_retries is not None),
            ("--episode-timeout", episode_timeout is not None),
            ("--run-dir", run_dir is not None),
            ("--resume", resume),
            ("--max-iterations-this-run", max_iterations_this_run is not None),
        )
        if provided
    ]
    if harbor_only:
        raise typer.BadParameter(
            f"{', '.join(harbor_only)} apply only to the harbor environment; "
            "use `wmh optimize <agent> harbor ...`"
        )
    interactive = _console.is_terminal
    if name is None:
        if not interactive:
            raise typer.BadParameter("provide a harness NAME (or run at a TTY for the wizard)")
        name = Prompt.ask("Name for the created harness", default="evolved")
    if tasks_file is None:
        if not interactive:
            raise typer.BadParameter("provide --tasks (or run at a TTY for the wizard)")
        tasks_file = Prompt.ask("Task file (JSONL of task_id/instruction/gold)")
    if iterations is None:
        iterations = (
            IntPrompt.ask("Search iterations (each = 1 delta + 1 gated eval)", default=5)
            if interactive
            else 5
        )

    if backend not in ("local", "e2b"):
        raise typer.BadParameter(f"unknown --backend {backend!r}; choose local or e2b")
    # Fail on a bad name NOW, not after the search has spent its eval budget on the save.
    try:
        validate_name(name)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    tasks = _load_task_file(tasks_file)
    holdout = _load_task_file(holdout_file) if holdout_file else None
    store = HarnessStore(root)
    seed_doc = _resolve_seed(store, seed)
    # The world model IS the environment on every backend, so it is always required.
    world_model, provider, model_name = _load_world_model(model, root)
    meta_provider, meta_model = resolve_opt_in_model_provider(root, "meta", provider)
    agent_provider, agent_model = resolve_opt_in_model_provider(root, "agent", provider)

    candidate_count = iterations * proposal_batch_size
    rollouts = (candidate_count + 1) * k * len(tasks)
    holdout_note = (
        f" (+ up to {(candidate_count + 1) * k * len(holdout)} held-out)" if holdout else ""
    )
    backend_note = (
        " (pi harness in pooled E2B sandboxes; env stays the world model)"
        if backend == "e2b"
        else ""
    )
    meta_note = (
        f" (proposer: {meta_model} from settings models.meta)" if meta_model is not None else ""
    )
    agent_note = (
        f" (agent-under-test: {agent_model} from settings models.agent)"
        if agent_model is not None
        else ""
    )
    _console.print(
        f"searching from [bold]{seed_doc.name}[/bold] against world model "
        f"[bold]{model_name}[/bold]: {iterations} iteration(s), "
        f"{proposal_batch_size} proposal(s)/iteration, k={k}, {len(tasks)} task(s) "
        f"-> up to ~{rollouts} rollouts{holdout_note} + {candidate_count} proposals"
        f"{meta_note}{agent_note}{backend_note}"
    )
    if interactive and not yes and not Confirm.ask("Proceed?", default=True):
        raise typer.Exit(0)

    def _progress(iteration: int, variant: str, score: float, changed: bool) -> None:
        tag = "seed" if iteration == 0 else f"iter {iteration}"
        state = (
            "seed"
            if iteration == 0
            else "[green]selected[/green]"
            if changed
            else "[yellow]unchanged[/yellow]"
        )
        _console.print(f"  [{tag}] {variant}: success_rate={score:.3f} {state}")

    def _note(message: str) -> None:
        # Dead proposals narrate here; scored proposals use the structured callback below.
        _console.print(f"  [dim]{message}[/dim]")

    def _proposal(record: ProposalRecord) -> None:
        if record.outcome != "scored":
            return
        assert record.candidate is not None and record.score is not None
        state = (
            "[green]selected[/green]"
            if record.selected
            else "[cyan]eligible, not selected[/cyan]"
            if record.gate_eligible
            else "[yellow]rejected by gate[/yellow]"
        )
        _console.print(
            f"  [iteration {record.iteration} proposal {record.proposal_index}] "
            f"{record.candidate}: success_rate={record.score:.3f} {state}"
        )

    result = create_harness(
        name,
        seed_doc,
        tasks,
        world_model,
        agent_provider,
        ProviderDeltaProposer(meta_provider),
        GoldJudge(provider),
        iterations=iterations,
        proposal_batch_size=proposal_batch_size,
        k=k,
        holdout=holdout,
        harness_backend="e2b" if backend == "e2b" else "local",
        eval_concurrency=eval_concurrency,
        e2b_template=e2b_template,
        on_progress=_progress,
        on_note=_note,
        on_proposal=_proposal,
    )
    saved = store.save_version(result.best, alias=CHAMPION_ALIAS)
    selected = len(result.archive.accepted())
    _console.print(
        f"[green]created[/green] [bold]{name}[/bold] v{saved.version} (champion) "
        f"success_rate={result.best_score:.3f}: {len(result.archive.deltas)} delta(s) audited, "
        f"{selected} selected, {result.skipped} skipped -> {store.dir_for(name)}"
    )
    run_argv = [
        "wmh",
        "eval",
        tasks_file,
        "--mode",
        "closed-loop",
        "--name",
        model_name,
        "--root",
        root,
        "--k",
        str(k),
        "--harness",
        f"{name}@{saved.version}",
    ]
    if backend == "e2b":
        run_argv.extend(("--harness-backend", "e2b"))
    if eval_concurrency is not None:
        run_argv.extend(("--eval-concurrency", str(eval_concurrency)))
    if backend == "e2b" and e2b_template is not None:
        run_argv.extend(("--e2b-template", e2b_template))
    _console.print(
        f"  run it: [bold]{escape(shlex.join(run_argv))}[/bold]",
        soft_wrap=True,
    )
    if archive_out:
        Path(archive_out).write_text(result.archive.model_dump_json(indent=2), encoding="utf-8")
        _console.print(f"  wrote archive -> {archive_out}")


def _load_task_file(path: str) -> list[TaskSpec]:
    try:
        return load_tasks(path)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"cannot load tasks from {path!r}: {exc}") from exc


def _resolve_seed(store: HarnessStore, seed: str | None) -> HarnessDoc:
    if seed is None:
        return HarnessDoc.baseline()
    base, _, ref = seed.partition("@")
    try:
        return store.load(base, ref or None)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _load_world_model(name: str | None, root: str) -> tuple[WorldModel, Provider, str]:
    """Resolve a world model by name (or the sole built one) and load it with its provider."""
    store = WorldModelStore(root)
    try:
        model_dir = store.resolve(name)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    world_model, provider = load_world_model(model_dir)
    return world_model, provider, model_dir.name


# -- the harbor environment: complete-source population optimization on real tasks ------------


class _HarborRunConfig(BaseModel):
    """The resolved harbor optimize configuration persisted as `run-config.json`."""

    model_config = ConfigDict(frozen=True)

    agent: str
    attempts: int
    backend: Literal["local", "e2b"]
    e2b_template: str | None
    episode_timeout_s: float
    harbor_job_template: JsonObject
    harbor_retries: int
    iterations: int
    reward_key: str
    reward_mode: RewardMode
    task_ids: tuple[str, ...]


def _optimize_harbor(
    ctx: typer.Context,
    *,
    name: str | None,
    backend: str,
    e2b_template: str | None,
    iterations: int | None,
    harbor_config: str | None,
    task_ids_file: str | None,
    attempts: int | None,
    reward_key: str | None,
    reward_mode: str | None,
    harbor_retries: int | None,
    episode_timeout: float | None,
    run_dir_option: str | None,
    resume: bool,
    max_iterations_this_run: int | None,
    root: str,
    yes: bool,
) -> None:
    """Run the harbor population optimizer: fixed seed, sequential complete-source proposals."""
    if name is None:
        raise typer.BadParameter(
            "provide the seed agent NAME (the literal 'pi' is the built-in default agent): "
            "`wmh optimize pi harbor --harbor-config ... --task-ids ... --run-dir ...`"
        )
    if backend not in ("local", "e2b"):
        raise typer.BadParameter(f"unknown --backend {backend!r}; choose local or e2b")
    if reward_mode is not None and reward_mode not in ("raw", "positive-binary"):
        raise typer.BadParameter(
            f"unknown --reward-mode {reward_mode!r}; choose raw or positive-binary"
        )
    if run_dir_option is None:
        raise typer.BadParameter(
            "--run-dir is required for the harbor environment: it holds all durable run state"
        )
    if WorldModelStore(root).exists(_HARBOR_ENVIRONMENT):
        raise typer.BadParameter(
            "a stored world model is literally named 'harbor', which now selects the harbor "
            "benchmark environment; rename that model directory under <root>/models/ and retry"
        )
    if backend == "local" and episode_timeout is not None:
        raise typer.BadParameter("--episode-timeout requires --backend e2b")

    run_dir = Path(run_dir_option)
    config_path = run_dir / "run-config.json"
    if resume:
        config = _resumed_harbor_config(
            ctx,
            config_path,
            name=name,
            backend=backend,
            e2b_template=e2b_template,
            iterations=iterations,
            harbor_config=harbor_config,
            task_ids_file=task_ids_file,
            attempts=attempts,
            reward_key=reward_key,
            reward_mode=reward_mode,
            harbor_retries=harbor_retries,
            episode_timeout=episode_timeout,
        )
    else:
        if config_path.exists():
            raise typer.BadParameter(
                f"{run_dir} already holds a run; pass --resume to continue it or choose a "
                "fresh --run-dir"
            )
        if harbor_config is None or task_ids_file is None:
            raise typer.BadParameter(
                "--harbor-config and --task-ids are required to start a harbor optimization"
            )
        config = _HarborRunConfig(
            agent=name,
            attempts=attempts if attempts is not None else 1,
            backend="e2b" if backend == "e2b" else "local",
            e2b_template=resolve_e2b_template(e2b_template),
            episode_timeout_s=(
                episode_timeout if episode_timeout is not None else DEFAULT_EVAL_EPISODE_TIMEOUT_S
            ),
            harbor_job_template=_load_harbor_job_template(Path(harbor_config)),
            harbor_retries=harbor_retries if harbor_retries is not None else 0,
            iterations=iterations if iterations is not None else _DEFAULT_HARBOR_ITERATIONS,
            reward_key=reward_key if reward_key is not None else "reward",
            reward_mode="raw" if reward_mode == "raw" else "positive-binary",
            task_ids=_load_harbor_task_ids(Path(task_ids_file)),
        )

    seed_name, seed_tree = _resolve_harbor_seed(root, config.agent)
    meta_config = resolve_required_model_config(root, "meta")
    agent_config = resolve_required_model_config(root, "agent")
    if not resume:
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(config_path, config.model_dump(mode="json"))
    scorer = _build_harbor_scorer(config, run_dir=run_dir, provider_config=agent_config)
    proposer = _build_harbor_proposer(
        run_dir=run_dir, meta_config=meta_config, e2b_template=config.e2b_template
    )

    _console.print(
        f"harbor population search from [bold]{config.agent}[/bold]: 1 seed + "
        f"{config.iterations} proposal slot(s), {len(config.task_ids)} task(s), "
        f"{config.attempts} attempt(s), reward mode {config.reward_mode}, "
        f"worker backend {config.backend} (proposer project: E2B) -> {run_dir}"
    )
    if _console.is_terminal and not yes and not Confirm.ask("Proceed?", default=True):
        raise typer.Exit(0)

    def _on_boundary(outcome: SlotOutcome) -> None:
        if outcome.evaluated is None:
            _console.print(
                f"  [slot {outcome.slot}] {outcome.candidate_id}: "
                f"[yellow]invalid[/yellow] ({escape(outcome.reason)})"
            )
        else:
            tag = "seed" if outcome.slot == 0 else f"slot {outcome.slot}"
            _console.print(f"  [{tag}] {outcome.candidate_id}: score={outcome.evaluated.score:.3f}")

    result = optimize_population(
        seed_tree,
        scorer,
        proposer,
        config.iterations,
        run_dir=run_dir,
        max_new_boundaries=max_iterations_this_run,
        on_boundary=_on_boundary,
    )
    if not result.completed:
        _console.print(
            f"[yellow]checkpointed[/yellow] {len(result.outcomes)}/{config.iterations + 1} "
            f"boundaries; continue with: [bold]wmh optimize {config.agent} harbor --resume "
            f"--run-dir {run_dir}[/bold]"
        )
        return
    _publish_harbor_winner(result, root=root, seed_name=seed_name, run_dir=run_dir)


def _publish_harbor_winner(
    result: PopulationResult, *, root: str, seed_name: str, run_dir: Path
) -> None:
    """Save the score winner as the seed agent's next version and move champion."""
    store = HarnessStore(root)
    winner = result.best.source.to_doc(seed_name)
    saved = store.save_version(winner, alias=CHAMPION_ALIAS)
    scored = sum(1 for outcome in result.outcomes if outcome.evaluated is not None)
    _console.print(
        f"[green]optimized[/green] [bold]{seed_name}[/bold] v{saved.version} (champion) "
        f"score={result.best_score:.3f} from {result.best.candidate_id} "
        f"({scored} scored, {len(result.outcomes) - scored} invalid slot(s)) "
        f"-> {store.dir_for(seed_name)}; evidence -> {run_dir}"
    )


def _resumed_harbor_config(
    ctx: typer.Context,
    config_path: Path,
    *,
    name: str,
    backend: str,
    e2b_template: str | None,
    iterations: int | None,
    harbor_config: str | None,
    task_ids_file: str | None,
    attempts: int | None,
    reward_key: str | None,
    reward_mode: str | None,
    harbor_retries: int | None,
    episode_timeout: float | None,
) -> _HarborRunConfig:
    """Load the recorded run config, rejecting explicit CLI flags that conflict with it."""
    if not config_path.is_file():
        raise typer.BadParameter(
            f"--resume found no run-config.json under {config_path.parent}; "
            "start the run once without --resume"
        )
    try:
        stored = _HarborRunConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise typer.BadParameter(f"cannot load {config_path}: {error}") from error

    def explicit(param: str) -> bool:
        # Compared by name: typer vendors click, so its ParameterSource enum is not
        # click.core's class and an identity check would silently never match.
        source = ctx.get_parameter_source(param)
        return source is not None and source.name == "COMMANDLINE"

    conflicts: list[str] = []
    if name != stored.agent:
        conflicts.append(f"NAME {name!r} != recorded {stored.agent!r}")
    checks: list[tuple[str, str, object, object]] = [
        ("backend", "--backend", backend, stored.backend),
        ("iterations", "--iterations", iterations, stored.iterations),
        ("attempts", "--attempts", attempts, stored.attempts),
        ("reward_key", "--reward-key", reward_key, stored.reward_key),
        ("reward_mode", "--reward-mode", reward_mode, stored.reward_mode),
        ("harbor_retries", "--harbor-retries", harbor_retries, stored.harbor_retries),
        ("episode_timeout", "--episode-timeout", episode_timeout, stored.episode_timeout_s),
        (
            "e2b_template",
            "--e2b-template",
            resolve_e2b_template(e2b_template),
            stored.e2b_template,
        ),
    ]
    conflicts.extend(
        f"{flag} {value!r} != recorded {recorded!r}"
        for param, flag, value, recorded in checks
        if explicit(param) and value != recorded
    )
    if explicit("harbor_config") and harbor_config is not None:
        template = _load_harbor_job_template(Path(harbor_config))
        if template != stored.harbor_job_template:
            conflicts.append("--harbor-config differs from the recorded job template")
    if explicit("task_ids_file") and task_ids_file is not None:
        task_ids = _load_harbor_task_ids(Path(task_ids_file))
        if task_ids != stored.task_ids:
            conflicts.append("--task-ids differs from the recorded task list")
    if conflicts:
        raise typer.BadParameter(
            "--resume uses the recorded run-config.json; conflicting flag(s): "
            + "; ".join(conflicts)
            + ". Drop them to continue this run, or start a fresh --run-dir"
        )
    return stored


def _resolve_harbor_seed(root: str, agent_ref: str) -> tuple[str, HarnessSourceTree]:
    """Resolve the seed AGENT positional: literal 'pi' is built-in, otherwise store name@ref."""
    base, _, ref = agent_ref.partition("@")
    try:
        validate_name(base)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if base == DEFAULT_SEED_AGENT and not ref:
        return base, HarnessSourceTree.from_doc(default_agent(base))
    try:
        doc = HarnessStore(root).load(base, ref or None)
    except (FileNotFoundError, ValueError) as error:
        raise typer.BadParameter(
            f"{error}; the built-in default agent seed is the literal {DEFAULT_SEED_AGENT!r}"
        ) from error
    return base, HarnessSourceTree.from_doc(doc)


def _load_harbor_job_template(path: Path) -> JsonObject:
    """Load one harbor JobConfig template (YAML or JSON) as canonical JSON.

    The harbor SDK is an optional extra imported lazily, exactly like the e2b extra: only the
    harbor environment needs it, and `import wmh` must succeed without it.
    """
    try:
        import yaml
        from harbor.models.job.config import JobConfig
    except ImportError as error:
        raise typer.BadParameter(_HARBOR_EXTRA_HINT) from error
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        template = JobConfig.model_validate(raw)
    except (OSError, yaml.YAMLError, ValueError, TypeError) as error:
        raise typer.BadParameter(f"cannot load the harbor config from {path}: {error}") from error
    return template.model_dump(mode="json")


def _load_harbor_task_ids(path: Path) -> tuple[str, ...]:
    """Load the exact ordered task-id list, validated by the canonical score request rules."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise typer.BadParameter(f"cannot load task ids from {path}: {error}") from error
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise typer.BadParameter(f"{path} must contain one JSON array of task-id strings")
    try:
        request = ScoreRequest(task_ids=tuple(raw), attempts=1)
    except ValidationError as error:
        raise typer.BadParameter(f"invalid task ids in {path}: {error}") from error
    return request.task_ids


def _build_harbor_scorer(
    config: _HarborRunConfig, *, run_dir: Path, provider_config: ProviderConfig
) -> Scorer:
    """Construct the harbor evaluator for the resolved run config (lazy harbor import)."""
    try:
        from harbor.models.job.config import JobConfig

        from wmh.evals.harbor.scorer import HarborScorer
    except ImportError as error:
        raise typer.BadParameter(_HARBOR_EXTRA_HINT) from error
    template = JobConfig.model_validate(
        {**config.harbor_job_template, "jobs_dir": str(run_dir / "harbor")}
    )
    try:
        return asyncio.run(
            HarborScorer.create(
                template,
                list(config.task_ids),
                provider_config=provider_config,
                reward_key=config.reward_key,
                reward_mode=config.reward_mode,
                attempts=config.attempts,
                task_environment="e2b" if config.backend == "e2b" else "docker",
                harness_backend=config.backend,
                # "" pins template absence so the scorer cannot re-read a changed environment.
                e2b_template=(config.e2b_template or "") if config.backend == "e2b" else None,
                episode_timeout_s=(
                    config.episode_timeout_s
                    if config.backend == "e2b"
                    else DEFAULT_EVAL_EPISODE_TIMEOUT_S
                ),
                harbor_retries=config.harbor_retries,
            )
        )
    except ValueError as error:
        raise typer.BadParameter(f"cannot build the harbor scorer: {error}") from error


def _build_harbor_proposer(
    *, run_dir: Path, meta_config: ProviderConfig, e2b_template: str | None
) -> CandidateProposer:
    """Wire the proposer: a fresh E2B project per slot driving the optimizer persona.

    The proposer project requires E2B even under `--backend local` in this version: the
    optimizer persona needs a contained filesystem plus node for interface validation, and the
    pi worker template already ships both.
    """
    provider = get_provider(meta_config)
    if not isinstance(provider, ToolCallingProvider):
        raise typer.BadParameter(
            "settings [models.meta] provider lacks structured tool calling, which the "
            "harbor proposer requires"
        )

    def project_factory() -> CandidateProject:
        return AgentProject.create(
            template=e2b_template,
            metadata={"wmh_component": "optimize-harbor-proposer"},
        )

    return ProjectCandidateProposer(
        optimizer_agent(),
        provider,
        project_factory=project_factory,
        run_dir=run_dir,
    )


@harness_app.command("init")
def init_harness(
    name: str = typer.Argument("baseline", help="Name for the new harness."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir."),
) -> None:
    """Write the baseline harness as v1 and point `champion` at it."""
    store = HarnessStore(root)
    try:
        if store.exists(name):
            raise typer.BadParameter(
                f"harness {name!r} already exists; new versions are appended by "
                "`wmh optimize`, and aliases move with `set_alias`"
            )
        doc = store.save_version(HarnessDoc.baseline(name), alias=CHAMPION_ALIAS)
    except ValueError as exc:  # invalid name -> usage error, not a traceback
        raise typer.BadParameter(str(exc)) from exc
    _console.print(
        f"[green]wrote[/green] {name} v{doc.version} (champion) -> {store.dir_for(name)}"
    )
    _console.print(
        f"run it: [bold]wmh eval <tasks.jsonl> --mode closed-loop --harness {name}[/bold]"
    )
