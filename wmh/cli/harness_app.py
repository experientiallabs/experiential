"""`wmh harness` — named, versioned agent harnesses under `.wmh/harnesses/<name>/`.

A harness is the scaffold an agent runs with: prompt surfaces, a tool policy, loop parameters, and
skills, stored as immutable numbered versions with movable aliases (`champion` is what runs by
default). `init` writes the baseline as v1; `list`/`show` inspect what exists; `create` searches
for a better harness by **inverting the world model** — delta variants are scored closed-loop
against it and gated on non-regression, so the environment model the traces built now steers what
the agent's scaffold should be. Run one closed-loop with
`wmh eval closed-loop <tasks> --harness <name>[@ref]`.
"""

from __future__ import annotations

import subprocess
import tempfile
import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from wmh.config import ARTIFACT_DIR, WorldModelStore, load_settings
from wmh.config.store import validate_name
from wmh.engine import load_world_model
from wmh.engine.world_model import WorldModel
from wmh.evals.gold import GoldJudge
from wmh.evals.tasks import TaskSpec, load_tasks
from wmh.harness.create import create_harness
from wmh.harness.doc import HarnessDoc
from wmh.harness.ingest import (
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_TOTAL_BYTES,
    body_map,
    build_ingest_doc,
    collect_repo_files,
)
from wmh.harness.store import CHAMPION_ALIAS, HarnessStore
from wmh.providers import get_provider
from wmh.providers.base import Provider, ProviderConfig, ProviderKind
from wmh.providers.retry import RetryingProvider
from wmh.tracking import MeteredProvider, RunTracker

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
        None, "--model", help="World model to search against (default: the only built one)."
    ),
    seed: str = typer.Option(
        None,
        "--seed",
        help="Harness to start from, as name[@ref] (default: the built-in baseline).",
    ),
    iterations: int = typer.Option(None, min=1, help="Propose-and-gate steps (the search budget)."),
    k: int = typer.Option(3, min=1, help="Closed-loop passes per task per variant."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir."),
    archive_out: str = typer.Option(
        None, "--archive", help="Also write the full delta archive JSON here."
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip the cost confirmation prompt."),
) -> None:
    """Create a harness by inverting the world model: search harness-space against it.

    An LLM meta-agent proposes typed deltas against the harness document (surface-keyed ops with
    preconditions), each applied child is scored closed-loop against the world model (k passes per
    task) and gated on non-regression (regression suite, then full split, then the optional
    held-out split). The champion is saved as a new immutable version with the `champion` alias.
    Interactive at a TTY: missing inputs are prompted for.
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

    # Fail on a bad name NOW, not after the search has spent its eval budget on the save.
    try:
        validate_name(name)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    tasks = _load_task_file(tasks_file)
    holdout = _load_task_file(holdout_file) if holdout_file else None
    store = HarnessStore(root)
    seed_doc = _resolve_seed(store, seed)
    world_model, provider, model_name = _load_world_model(model, root)

    rollouts = (iterations + 1) * k * len(tasks)
    holdout_note = f" (+ up to {(iterations + 1) * k * len(holdout)} held-out)" if holdout else ""
    _console.print(
        f"searching from [bold]{seed_doc.name}[/bold] against world model "
        f"[bold]{model_name}[/bold]: {iterations} iteration(s), k={k}, {len(tasks)} task(s) "
        f"-> up to ~{rollouts} rollouts{holdout_note} + {iterations} proposal calls"
    )
    if interactive and not yes and not Confirm.ask("Proceed?", default=True):
        raise typer.Exit(0)

    def _progress(iteration: int, variant: str, score: float, accepted: bool) -> None:
        tag = "seed" if iteration == 0 else f"iter {iteration}"
        gate = "[green]accepted[/green]" if accepted else "[yellow]rejected[/yellow]"
        _console.print(f"  [{tag}] {variant}: success_rate={score:.3f} {gate}")

    result = create_harness(
        name,
        seed_doc,
        tasks,
        world_model,
        provider,
        provider,
        GoldJudge(provider),
        iterations=iterations,
        k=k,
        holdout=holdout,
        on_progress=_progress,
    )
    saved = store.save_version(result.best, alias=CHAMPION_ALIAS)
    accepted = len(result.archive.accepted())
    _console.print(
        f"[green]created[/green] [bold]{name}[/bold] v{saved.version} (champion) "
        f"success_rate={result.best_score:.3f}: {len(result.archive.deltas)} delta(s) audited, "
        f"{accepted} accepted, {result.skipped} skipped -> {store.dir_for(name)}"
    )
    _console.print(f"  run it: [bold]wmh eval closed-loop {tasks_file} --harness {name}[/bold]")
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
    _console.print(f"run it: [bold]wmh eval closed-loop <tasks.jsonl> --harness {name}[/bold]")


@harness_app.command("ingest")
def ingest(
    source: str = typer.Argument(..., help="Local repo directory, or a git URL to clone."),
    name: str = typer.Option(None, "--name", help="Harness name (default: the repo name)."),
    ref: str = typer.Option(None, "--ref", help="Branch or tag to clone (URL sources only)."),
    exclude: list[str] = typer.Option(  # noqa: B008
        [], "--exclude", help="Extra glob(s) to skip (repeatable)."
    ),
    max_file_bytes: int = typer.Option(
        DEFAULT_MAX_FILE_BYTES, min=1, help="Per-file content cap (bytes)."
    ),
    max_total_bytes: int = typer.Option(
        DEFAULT_MAX_TOTAL_BYTES, min=1, help="Total content cap (bytes)."
    ),
    provider_name: str = typer.Option(
        None, "--provider", help="Mapping LLM provider (default: [models.worker] from settings)."
    ),
    model: str = typer.Option(None, "--model", help="Mapping LLM model id."),
    region: str = typer.Option(None, "--region", help="AWS region (bedrock providers)."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir."),
    yes: bool = typer.Option(False, "--yes", help="Skip the cost confirmation prompt."),
) -> None:
    """Body-map an existing agent repo into a new harness.

    Every relevant textual file becomes a pathful code surface (directory hierarchy preserved)
    with an LLM-written harnessdoc; the built document also carries an overview prompt and a
    BODYMAP.md index. Inclusion is zealous: when in doubt a file is mapped, never dropped
    silently — skips are listed in BODYMAP.md with reasons. The result is saved as a new harness
    with the `champion` alias; push it with `wmh push <name> --kind harness`.
    """
    store = HarnessStore(root)
    resolved_name = name or _default_ingest_name(source)
    try:
        validate_name(resolved_name)
    except ValueError as exc:
        raise typer.BadParameter(f"{exc}; pass --name") from exc
    if store.exists(resolved_name):
        raise typer.BadParameter(f"harness {resolved_name!r} already exists; pick another --name")
    provider = _ingest_provider(provider_name, model, region, root)

    with tempfile.TemporaryDirectory(prefix="wmh-ingest-") as scratch:
        if _looks_like_git_url(source):
            checkout = Path(scratch) / "repo"
            _clone(source, ref, checkout)
            source_label = source if ref is None else f"{source}@{ref}"
        else:
            if ref is not None:
                raise typer.BadParameter("--ref applies only to git URL sources")
            checkout = Path(source)
            source_label = f"local checkout {checkout.resolve().name}"
        try:
            collected = collect_repo_files(
                checkout,
                max_file_bytes=max_file_bytes,
                max_total_bytes=max_total_bytes,
                extra_excludes=tuple(exclude),
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if not collected.files:
            raise typer.BadParameter(f"nothing to ingest under {checkout} (all files excluded?)")

        _console.print(
            f"body-mapping [bold]{resolved_name}[/bold]: {len(collected.files)} file(s) to map "
            f"({len(collected.skipped)} skipped) -> ~{len(collected.files) + 1} LLM calls "
            f"on {provider.config.model}"
        )
        if _console.is_terminal and not yes and not Confirm.ask("Proceed?", default=True):
            raise typer.Exit(0)

        tracker = RunTracker(run_id=uuid.uuid4().hex, kind="ingest")
        metered = MeteredProvider(RetryingProvider(provider), tracker)
        with tracker.timed(), Progress(console=_console, transient=True) as progress:
            task = progress.add_task("mapping", total=len(collected.files))

            def _tick(index: int, total: int, path: str) -> None:
                progress.update(task, completed=index, description=f"mapping {path}")

            mapping = body_map(collected, metered, name=resolved_name, on_progress=_tick)
        doc = build_ingest_doc(resolved_name, collected, mapping, source=source_label)

    saved = store.save_version(doc, alias=CHAMPION_ALIAS)
    totals = tracker.totals()
    unmapped = f", {len(mapping.unmapped)} unmapped" if mapping.unmapped else ""
    _console.print(
        f"[green]ingested[/green] [bold]{resolved_name}[/bold] v{saved.version} (champion): "
        f"{len(collected.files)} file(s) mapped{unmapped}, {len(collected.skipped)} skipped "
        f"-> {store.dir_for(resolved_name)}"
    )
    _console.print(
        f"  mapping cost: {totals.total_tokens} tokens, ${totals.cost_usd:.4f} "
        f"({totals.calls} calls, {tracker.duration_seconds():.0f}s)"
    )
    _console.print(
        f"  inspect: [bold]wmh harness show {resolved_name}[/bold]   "
        f"push: [bold]wmh push {resolved_name} --kind harness[/bold]"
    )


def _default_ingest_name(source: str) -> str:
    tail = source.rstrip("/").rpartition("/")[2]
    return tail.removesuffix(".git") or "ingested"


def _looks_like_git_url(source: str) -> bool:
    return source.startswith(("http://", "https://", "git@", "ssh://")) or source.endswith(".git")


def _clone(url: str, ref: str | None, target: Path) -> None:
    """Shallow-clone `url` (optionally a branch/tag) for a one-shot read; fail as a usage error."""
    command = ["git", "clone", "--depth", "1", "--quiet"]
    if ref is not None:
        command += ["--branch", ref]
    command += [url, str(target)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise typer.BadParameter(
            f"git clone failed for {url!r}: {detail[-1] if detail else 'unknown error'}"
        )


def _ingest_provider(
    provider_name: str | None, model: str | None, region: str | None, root: str
) -> Provider:
    """The mapping LLM: explicit --provider/--model, else the settings worker role."""
    if provider_name is not None or model is not None:
        if provider_name is None or model is None:
            raise typer.BadParameter("--provider and --model must be given together")
        try:
            kind = ProviderKind(provider_name)
        except ValueError:
            kinds = ", ".join(k.value for k in ProviderKind)
            raise typer.BadParameter(
                f"unknown provider {provider_name!r}; choose one of: {kinds}"
            ) from None
        return get_provider(ProviderConfig(kind=kind, model=model, region=region))
    configured = load_settings(root).models.resolve("worker")
    if configured is None:
        raise typer.BadParameter(
            "no mapping model configured: set [models.worker] in .wmh/settings.toml "
            "(wmh config models) or pass --provider/--model"
        )
    return get_provider(
        ProviderConfig(
            kind=ProviderKind(configured.provider),
            model=configured.model,
            region=configured.region or region,
            endpoint=configured.endpoint,
            deployment=configured.deployment,
        )
    )
