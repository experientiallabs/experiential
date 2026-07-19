"""`wmh harness` — named, versioned agent harnesses under `.wmh/harnesses/<name>/`.

A harness is the scaffold an agent runs with: prompt surfaces, a tool policy, loop parameters, and
skills, stored as immutable numbered versions with movable aliases (`champion` is what runs by
default). `init` writes the baseline as v1; `list`/`show` inspect what exists; `create` searches
for a better harness by **inverting the world model** — delta variants are scored closed-loop
against it and gated on non-regression, so the environment model the traces built now steers what
the agent's scaffold should be. Run one closed-loop with
`wmh eval <tasks> --mode closed-loop --harness <name>[@ref]`.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from wmh.cli.model_roles import resolve_opt_in_model_provider
from wmh.config import ARTIFACT_DIR, WorldModelStore
from wmh.config.store import validate_name
from wmh.engine import load_world_model
from wmh.engine.world_model import WorldModel
from wmh.evals.gold import GoldJudge
from wmh.evals.tasks import TaskSpec, load_tasks
from wmh.harness.create import ProposalRecord, create_harness
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_sandbox import E2B_TEMPLATE_ENV
from wmh.harness.proposer import ProviderDeltaProposer
from wmh.harness.store import CHAMPION_ALIAS, HarnessStore
from wmh.providers.base import Provider

harness_app = typer.Typer(
    help="Named, versioned agent harnesses (.wmh/harnesses): create, init, list, show.",
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


@harness_app.command("create")
def create(
    name: str = typer.Argument(None, help="Name for the created harness."),
    tasks_file: str = typer.Option(None, "--tasks", help="JSONL task file to optimize against."),
    holdout_file: str = typer.Option(
        None,
        "--holdout",
        help="Optional JSONL held-out task file: accepted deltas must also be no worse here.",
    ),
    model: str = typer.Option(
        None,
        "--model",
        help="World model to search against (default: the only built one).",
    ),
    harness_backend: str = typer.Option(
        "local",
        "--harness-backend",
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
        help="Prebaked E2B sandbox template for --harness-backend e2b (default: "
        "$WMH_E2B_TEMPLATE; without one, every sandbox bootstraps node + the pi runner deps).",
    ),
    seed: str = typer.Option(
        None,
        "--seed",
        help="Harness to start from, as name[@ref] (default: the built-in baseline).",
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
) -> None:
    """Create a harness by inverting the world model: search harness-space against it.

    An LLM meta-agent proposes typed deltas against the harness document (surface-keyed ops with
    preconditions), each applied child is scored closed-loop (k passes per task) and gated on
    non-regression (regression suite, then full split, then the optional held-out split). The
    agent-under-test resolves from `.wmh/settings.toml` `[models.agent]` when set and otherwise
    uses the world model's provider. The proposer's model resolves from `[models.meta]` when set;
    use a long-context, long-output model because a proposal carries whole replacement surfaces.
    Otherwise the proposer also uses the world model's provider. The environment is ALWAYS the
    world-model simulation; `--harness-backend` only picks where the harness PROCESS runs:
    `local` (the default) keeps it in/from this process, `e2b` runs the real pi agent inside
    pooled E2B sandboxes (pi-node seeds only), its tool calls still answered by the world model
    host-side, with all (task, attempt) cells in parallel unless
    --eval-concurrency caps them. The champion is saved as a new immutable version with the
    `champion` alias. Interactive at a TTY: missing inputs are prompted for (the backend stays
    flag-only).
    """
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

    if harness_backend not in ("local", "e2b"):
        raise typer.BadParameter(
            f"unknown --harness-backend {harness_backend!r}; choose local or e2b"
        )
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
        if harness_backend == "e2b"
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
        harness_backend="e2b" if harness_backend == "e2b" else "local",
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
    if harness_backend == "e2b":
        run_argv.extend(("--harness-backend", "e2b"))
    if eval_concurrency is not None:
        run_argv.extend(("--eval-concurrency", str(eval_concurrency)))
    if harness_backend == "e2b" and e2b_template is not None:
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
                "`wmh harness create`, and aliases move with `set_alias`"
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
