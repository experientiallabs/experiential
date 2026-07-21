"""Local CLI composition for scorer-driven complete-harness optimization."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

import typer
from harbor.models.job.config import JobConfig
from rich.console import Console
from rich.prompt import Confirm

from wmh.agents.default import default_agent
from wmh.agents.optimizer import optimizer_agent
from wmh.agents.project import DEFAULT_PROJECT_TIMEOUT_S, AgentProject
from wmh.cli.harbor_inputs import load_harbor_config, load_task_ids, write_json_atomic
from wmh.cli.model_roles import resolve_required_model_config
from wmh.config import ARTIFACT_DIR
from wmh.config.store import validate_name
from wmh.core.types import JsonObject
from wmh.evals.harbor.agent import MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC
from wmh.evals.harbor.scorer import HarborScorer
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_sandbox import E2B_TEMPLATE_ENV, SandboxUsage, resolve_e2b_template
from wmh.harness.population import HarnessPopulationOptimizer, PopulationOptimizationResult
from wmh.harness.population_archive import write_population_archive
from wmh.harness.project_proposer import (
    DEFAULT_MAX_HISTORY_BYTES,
    DEFAULT_MAX_HISTORY_CANDIDATES,
    ProjectCandidateProposer,
)
from wmh.harness.source_tree import HarnessSourceTree
from wmh.harness.store import CHAMPION_ALIAS, HarnessStore
from wmh.providers.base import ProviderConfig, ToolCallingProvider
from wmh.providers.registry import get_provider

_console = Console()


@dataclass(frozen=True)
class HarnessOptimizeOutcome:
    """Completed local run artifacts and the published local harness version."""

    result: PopulationOptimizationResult
    saved: HarnessDoc
    run_dir: Path
    archive_manifest: Path
    project_usage: SandboxUsage


def register(app: typer.Typer) -> None:
    """Register scorer-driven optimization under ``wmh harness``."""
    app.command("optimize")(optimize_harness)


def optimize_harness(
    name: str = typer.Argument(..., help="Name used to save the selected harness version."),
    harbor_config: str = typer.Option(
        ...,
        "--harbor-config",
        help="Harbor JobConfig template as JSON or YAML.",
    ),
    task_ids_file: str = typer.Option(
        ...,
        "--task-ids",
        help="JSON file containing the exact ordered task-ID string list.",
    ),
    reward_key: str = typer.Option(
        ...,
        "--reward-key",
        help="Official Harbor verifier reward field to optimize.",
    ),
    iterations: int = typer.Option(..., min=1, help="Fixed number of singular proposal slots."),
    attempts: int = typer.Option(..., min=1, help="Attempts per exact task and candidate."),
    seed: str | None = typer.Option(
        None,
        "--seed",
        help="Stored harness name@ref; default is the complete built-in Pi agent.",
    ),
    harness_backend: str = typer.Option(
        "local",
        "--harness-backend",
        help="Where each evaluated Pi process runs: local or e2b. The CLI remains local.",
    ),
    e2b_template: str | None = typer.Option(
        None,
        "--e2b-template",
        envvar=E2B_TEMPLATE_ENV,
        help="Optional template for E2B project and evaluated Pi processes.",
    ),
    environment_command_timeout_sec: int = typer.Option(
        MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
        "--environment-command-timeout",
        min=1,
        max=MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
        help="Maximum seconds for one command in the Harbor-owned task environment.",
    ),
    project_timeout_sec: float = typer.Option(
        DEFAULT_PROJECT_TIMEOUT_S,
        "--project-timeout",
        min=1,
        help="Lifetime in seconds for the proposer project's E2B sandbox.",
    ),
    max_history_candidates: int = typer.Option(
        DEFAULT_MAX_HISTORY_CANDIDATES,
        "--max-history-candidates",
        min=1,
        help="Maximum evaluated candidates materialized into proposer history.",
    ),
    max_history_bytes: int = typer.Option(
        DEFAULT_MAX_HISTORY_BYTES,
        "--max-history-bytes",
        min=1,
        help="Maximum complete source, score, and artifact bytes in proposer history.",
    ),
    result_out: str | None = typer.Option(
        None,
        "--result-out",
        help="Local run directory (default: <root>/runs/<opaque-id>).",
    ),
    root: str = typer.Option(ARTIFACT_DIR, help="Project artifact root and harness store."),
    yes: bool = typer.Option(False, "--yes", help="Acknowledge the displayed execution matrix."),
) -> None:
    """Optimize complete harness source against one exact ground-truth scorer request.

    The command host is always the local WMH process. The proposer project runs in E2B; evaluated
    Pi processes use ``--harness-backend``; Harbor's own JobConfig independently owns task
    environment backend and concurrency. Both ``models.meta`` and ``models.agent`` must be
    configured in the selected project's ``settings.toml``. This command has no world-model
    fallback and performs no held-out selection.
    """
    try:
        validate_name(name)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if harness_backend not in ("local", "e2b"):
        raise typer.BadParameter(
            f"unknown --harness-backend {harness_backend!r}; choose local or e2b"
        )
    if not reward_key:
        raise typer.BadParameter("--reward-key must be nonempty")

    job_config = load_harbor_config(Path(harbor_config))
    task_ids = load_task_ids(Path(task_ids_file))
    meta_config = resolve_required_model_config(root, "meta")
    agent_config = resolve_required_model_config(root, "agent")
    seed_source = _resolve_seed_source(root, seed)
    effective_e2b_template = resolve_e2b_template(e2b_template)
    run_dir = Path(result_out) if result_out is not None else Path(root) / "runs" / uuid4().hex
    if run_dir.exists():
        raise typer.BadParameter(f"result directory already exists: {run_dir}")

    score_cells = (iterations + 1) * len(task_ids) * attempts
    _console.print(
        f"fixed search: 1 seed + {iterations} proposal slot(s), {len(task_ids)} task(s), "
        f"{attempts} attempt(s) -> up to {score_cells} requested score cells; "
        f"evaluated Pi backend={harness_backend}; proposer project=e2b"
    )
    if not yes:
        if not _console.is_terminal:
            raise typer.BadParameter("pass --yes to acknowledge the execution matrix")
        if not Confirm.ask("Proceed?", default=False):
            raise typer.Exit(0)

    outcome = _execute_optimization(
        name=name,
        root=root,
        run_dir=run_dir,
        job_config=job_config,
        task_ids=task_ids,
        reward_key=reward_key,
        iterations=iterations,
        attempts=attempts,
        seed=seed_source,
        meta_config=meta_config,
        agent_config=agent_config,
        harness_backend=cast("Literal['local', 'e2b']", harness_backend),
        # An explicit empty string freezes template absence so lower layers cannot re-read a
        # concurrently changed environment after the input snapshot is written.
        e2b_template=effective_e2b_template or "",
        environment_command_timeout_sec=environment_command_timeout_sec,
        project_timeout_sec=project_timeout_sec,
        max_history_candidates=max_history_candidates,
        max_history_bytes=max_history_bytes,
    )
    _console.print(
        f"[green]optimized[/green] [bold]{name}[/bold] v{outcome.saved.version} "
        f"score={outcome.result.best_score:.6f}; evidence -> {outcome.run_dir}"
    )


def _execute_optimization(
    *,
    name: str,
    root: str,
    run_dir: Path,
    job_config: JobConfig,
    task_ids: tuple[str, ...],
    reward_key: str,
    iterations: int,
    attempts: int,
    seed: HarnessSourceTree,
    meta_config: ProviderConfig,
    agent_config: ProviderConfig,
    harness_backend: Literal["local", "e2b"],
    e2b_template: str | None,
    environment_command_timeout_sec: int,
    project_timeout_sec: float,
    max_history_candidates: int,
    max_history_bytes: int,
) -> HarnessOptimizeOutcome:
    """Resolve one immutable scorer request, execute it, and publish evidence before winner."""
    effective_job_config = JobConfig.model_validate(
        job_config.model_copy(
            update={"jobs_dir": run_dir / "harbor"},
            deep=True,
        ).model_dump(mode="python")
    )
    scorer = asyncio.run(
        HarborScorer.create(
            job_config=effective_job_config,
            task_ids=task_ids,
            provider_config=agent_config,
            reward_key=reward_key,
            environment_command_timeout_sec=environment_command_timeout_sec,
            harness_backend=harness_backend,
            e2b_template=e2b_template if harness_backend == "e2b" else None,
        )
    )
    request = scorer.request(attempts=attempts)
    inputs: JsonObject = {
        "schema_version": 1,
        "seed_source_tree_hash": seed.tree_hash,
        "iterations": iterations,
        "score_request": request.model_dump(mode="json"),
        "harbor_job_template": effective_job_config.model_dump(mode="json"),
        "meta_provider": meta_config.model_dump(mode="json"),
        "agent_provider": agent_config.model_dump(mode="json"),
        "harness_backend": harness_backend,
        "e2b_template": e2b_template or None,
        "environment_command_timeout_sec": environment_command_timeout_sec,
        "project_timeout_sec": project_timeout_sec,
        "max_history_candidates": max_history_candidates,
        "max_history_bytes": max_history_bytes,
    }
    meta_provider = get_provider(meta_config)
    if not isinstance(meta_provider, ToolCallingProvider):
        raise typer.BadParameter("settings [models.meta] provider lacks structured tool calling")
    # Only claim the requested output path after all read-only setup validation succeeds. Once
    # execution can incur spend, an incomplete directory remains deliberate failure evidence.
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(run_dir / "inputs.json", inputs)
    with AgentProject.create(
        timeout=project_timeout_sec,
        template=e2b_template,
        metadata={"wmh_component": "harness_optimize"},
    ) as project:
        proposer = ProjectCandidateProposer(
            project,
            optimizer_agent(),
            meta_provider,
            max_history_candidates=max_history_candidates,
            max_history_bytes=max_history_bytes,
        )
        result = HarnessPopulationOptimizer(proposer, scorer).optimize(
            seed=seed,
            request=request,
            iterations=iterations,
        )
    project_usage = project.usage()

    archive_manifest = write_population_archive(run_dir / "population", result)
    selected = result.best.candidate.model_copy(update={"name": name, "version": 0})
    saved = HarnessStore(root).save_version(selected, alias=CHAMPION_ALIAS)
    outcome: JsonObject = {
        "schema_version": 1,
        "best_candidate_id": result.best.candidate_id,
        "best_score": result.best_score,
        "best_document_hash": result.best.candidate.doc_hash,
        "saved_harness": saved.name,
        "saved_version": saved.version,
        "archive_manifest": archive_manifest.relative_to(run_dir).as_posix(),
        "project_sandbox_usage": project_usage.model_dump(mode="json"),
    }
    write_json_atomic(run_dir / "outcome.json", outcome)
    return HarnessOptimizeOutcome(
        result=result,
        saved=saved,
        run_dir=run_dir,
        archive_manifest=archive_manifest,
        project_usage=project_usage,
    )


def _resolve_seed_source(root: str, seed: str | None) -> HarnessSourceTree:
    if seed is None:
        return HarnessSourceTree.from_doc(default_agent())
    name, _, ref = seed.partition("@")
    try:
        return HarnessSourceTree.from_doc(HarnessStore(root).load(name, ref or None))
    except (FileNotFoundError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
