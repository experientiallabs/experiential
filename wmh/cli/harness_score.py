"""Local CLI composition for evaluator-driven multi-harness scoring."""

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
from wmh.cli.harbor_inputs import load_harbor_config, load_task_ids, write_json_atomic
from wmh.cli.model_roles import resolve_required_model_config
from wmh.config import ARTIFACT_DIR
from wmh.core.types import JsonObject
from wmh.evals.harbor.agent import MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC
from wmh.evals.harbor.scorer import HarborScorer
from wmh.harness.e2b_sandbox import E2B_TEMPLATE_ENV, resolve_e2b_template
from wmh.harness.runtime import (
    DEFAULT_EVAL_EPISODE_TIMEOUT_S,
    validate_episode_timeout_s,
)
from wmh.harness.score_archive import write_score_archive
from wmh.harness.score_batch import (
    HarnessScoreBatch,
    HarnessScoreTarget,
    score_harnesses,
    validate_score_targets,
)
from wmh.harness.store import HarnessStore
from wmh.providers.base import ProviderConfig

_console = Console()
_HARNESSES_OPTION = typer.Option(
    None,
    "--harness",
    help="Stored harness name@ref to score; repeat for several, in order.",
)


@dataclass(frozen=True)
class HarnessScoreOutcome:
    """Completed local multi-harness score artifacts."""

    result: HarnessScoreBatch
    run_dir: Path
    archive_manifest: Path


def register(app: typer.Typer) -> None:
    """Register evaluator-driven scoring under ``wmh harness``."""
    app.command("score")(score_harness_command)


def score_harness_command(
    include_default: bool = typer.Option(
        False,
        "--include-default",
        help="Score WMH's built-in default harness first.",
    ),
    harnesses: list[str] | None = _HARNESSES_OPTION,
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
        help="Official Harbor verifier reward field to score.",
    ),
    attempts: int = typer.Option(..., min=1, help="Attempts per exact task and harness."),
    harness_backend: str = typer.Option(
        "local",
        "--harness-backend",
        help="Where each evaluated harness process runs: local or e2b. The CLI remains local.",
    ),
    e2b_template: str | None = typer.Option(
        None,
        "--e2b-template",
        envvar=E2B_TEMPLATE_ENV,
        help="Optional template for evaluated harness processes using E2B.",
    ),
    environment_command_timeout_sec: int = typer.Option(
        MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
        "--environment-command-timeout",
        min=1,
        max=MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
        help="Maximum seconds for one command in the Harbor-owned task environment.",
    ),
    episode_timeout_sec: float = typer.Option(
        DEFAULT_EVAL_EPISODE_TIMEOUT_S,
        "--episode-timeout",
        min=0.001,
        help="Maximum seconds for one evaluated E2B harness episode.",
    ),
    result_out: str | None = typer.Option(
        None,
        "--result-out",
        help="Local run directory (default: <root>/runs/<opaque-id>).",
    ),
    root: str = typer.Option(ARTIFACT_DIR, help="Project artifact root and harness store."),
    yes: bool = typer.Option(False, "--yes", help="Acknowledge the displayed execution matrix."),
) -> None:
    """Score complete harnesses against one exact ground-truth scorer request.

    The command host is always the local WMH process. Evaluated harness processes use
    ``--harness-backend``; Harbor's JobConfig independently owns task environment backend and
    concurrency. ``models.agent`` must be configured in the selected project's
    ``settings.toml``. Stored aliases are resolved to immutable versions before any scoring.
    The command reports every declared harness and does not select or publish one.
    """
    if harness_backend not in ("local", "e2b"):
        raise typer.BadParameter(
            f"unknown --harness-backend {harness_backend!r}; choose local or e2b"
        )
    if not reward_key:
        raise typer.BadParameter("--reward-key must be nonempty")
    try:
        episode_timeout_sec = validate_episode_timeout_s(episode_timeout_sec)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--episode-timeout") from error
    if harness_backend == "local" and episode_timeout_sec != DEFAULT_EVAL_EPISODE_TIMEOUT_S:
        raise typer.BadParameter(
            "--episode-timeout requires --harness-backend e2b",
            param_hint="--episode-timeout",
        )

    targets = _resolve_targets(
        root,
        include_default=include_default,
        harnesses=tuple(harnesses or ()),
    )
    validate_score_targets(targets)
    job_config = load_harbor_config(Path(harbor_config))
    task_ids = load_task_ids(Path(task_ids_file))
    agent_config = resolve_required_model_config(root, "agent")
    effective_e2b_template = resolve_e2b_template(e2b_template)
    run_dir = Path(result_out) if result_out is not None else Path(root) / "runs" / uuid4().hex
    if run_dir.exists():
        raise typer.BadParameter(f"result directory already exists: {run_dir}")

    score_cells = len(targets) * len(task_ids) * attempts
    _console.print(
        f"scoring {len(targets)} harness(es), {len(task_ids)} task(s), "
        f"{attempts} attempt(s) -> {score_cells} requested score cells; "
        f"evaluated harness backend={harness_backend}; "
        f"episode timeout={episode_timeout_sec:g}s"
    )
    if not yes:
        if not _console.is_terminal:
            raise typer.BadParameter("pass --yes to acknowledge the execution matrix")
        if not Confirm.ask("Proceed?", default=False):
            raise typer.Exit(0)

    outcome = _execute_scoring(
        run_dir=run_dir,
        job_config=job_config,
        task_ids=task_ids,
        reward_key=reward_key,
        attempts=attempts,
        targets=targets,
        agent_config=agent_config,
        harness_backend=cast("Literal['local', 'e2b']", harness_backend),
        e2b_template=(effective_e2b_template or "") if harness_backend == "e2b" else None,
        environment_command_timeout_sec=environment_command_timeout_sec,
        episode_timeout_sec=episode_timeout_sec,
    )
    for entry in outcome.result.entries:
        _console.print(
            f"  [bold]{entry.target.label}[/bold]: "
            f"score={entry.score.report.score:.6f}, "
            f"pass_rate={entry.score.report.pass_rate:.6f}"
        )
    _console.print(f"[green]scored[/green] evidence -> {outcome.run_dir}")


def _execute_scoring(
    *,
    run_dir: Path,
    job_config: JobConfig,
    task_ids: tuple[str, ...],
    reward_key: str,
    attempts: int,
    targets: tuple[HarnessScoreTarget, ...],
    agent_config: ProviderConfig,
    harness_backend: Literal["local", "e2b"],
    e2b_template: str | None,
    environment_command_timeout_sec: int,
    episode_timeout_sec: float = DEFAULT_EVAL_EPISODE_TIMEOUT_S,
) -> HarnessScoreOutcome:
    """Resolve one immutable scorer request, score every target, and archive evidence."""
    validate_score_targets(targets)
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
            episode_timeout_sec=episode_timeout_sec,
            harness_backend=harness_backend,
            e2b_template=e2b_template if harness_backend == "e2b" else None,
        )
    )
    request = scorer.request(attempts=attempts)
    run_dir.mkdir(parents=True, exist_ok=False)
    inputs: JsonObject = {
        "schema_version": 2,
        "score_request": request.model_dump(mode="json"),
        "harbor_job_template": effective_job_config.model_dump(mode="json"),
        "agent_provider": agent_config.model_dump(mode="json"),
        "harness_backend": harness_backend,
        "e2b_template": e2b_template or None,
        "environment_command_timeout_sec": environment_command_timeout_sec,
        "episode_timeout_sec": episode_timeout_sec,
        "targets": [
            {
                "label": target.label,
                "name": target.harness.name,
                "version": target.harness.version,
                "document_hash": target.harness.doc_hash,
                "source_tree_hash": target.source.tree_hash,
            }
            for target in targets
        ],
    }
    write_json_atomic(run_dir / "inputs.json", inputs)
    result = score_harnesses(scorer, targets, request=request)
    archive_manifest = write_score_archive(run_dir / "scores", result)
    return HarnessScoreOutcome(
        result=result,
        run_dir=run_dir,
        archive_manifest=archive_manifest,
    )


def _resolve_targets(
    root: str,
    *,
    include_default: bool,
    harnesses: tuple[str, ...],
) -> tuple[HarnessScoreTarget, ...]:
    if not include_default and not harnesses:
        raise typer.BadParameter("pass --include-default or at least one --harness")
    targets: list[HarnessScoreTarget] = []
    if include_default:
        harness = default_agent()
        targets.append(
            HarnessScoreTarget(
                label="default",
                harness=harness,
            )
        )
    store = HarnessStore(root)
    for reference in harnesses:
        name, _, ref = reference.partition("@")
        try:
            version = store.resolve_version(name, ref or None)
            harness = store.load(name, f"v{version}")
        except (FileNotFoundError, ValueError) as error:
            raise typer.BadParameter(str(error)) from error
        targets.append(
            HarnessScoreTarget(
                label=f"{name}@v{version}",
                harness=harness,
            )
        )
    return tuple(targets)
