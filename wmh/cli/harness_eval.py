"""Ground-truth benchmark evaluation for stored WMH harnesses."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import typer
from click import ClickException
from harbor.models.job.config import DatasetConfig
from rich.console import Console

from wmh.config import ARTIFACT_DIR
from wmh.evals.harbor.config import HarborEnvironmentBackend, HarborJobSpec
from wmh.evals.harbor.evaluator import HarborEvaluator, harbor_job_lease_path
from wmh.harness.doc import HarnessDoc
from wmh.harness.pi_local import PI_CONTAINER_IMAGE
from wmh.harness.store import HarnessStore
from wmh.providers.base import ProviderConfig, ProviderKind

_console = Console()


def register(app: typer.Typer) -> None:
    """Register ground-truth harness evaluation on the harness command group."""
    app.command("eval")(eval_harness)


def eval_harness(
    harness: str = typer.Argument(..., help="Stored harness as NAME or NAME@REF."),
    dataset_path: str | None = typer.Option(
        None,
        "--dataset-path",
        help="Local Harbor dataset directory. Mutually exclusive with --dataset.",
    ),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        help="Harbor registry or package dataset name. Mutually exclusive with --dataset-path.",
    ),
    dataset_ref: str | None = typer.Option(
        None,
        "--dataset-ref",
        help="Required requested version or ref for --dataset. Harbor records resolved digests.",
    ),
    task_names: list[str] | None = typer.Option(  # noqa: B008 - Typer option declaration
        None,
        "--task",
        help="Task name or glob to include. Repeatable and mutually exclusive with --exclude-task.",
    ),
    excluded_task_names: list[str] | None = typer.Option(  # noqa: B008 - Typer option declaration
        None,
        "--exclude-task",
        help="Task name or glob to exclude. Repeatable and mutually exclusive with --task.",
    ),
    provider: str = typer.Option(..., "--provider", help="Worker provider: azure or bedrock."),
    model: str = typer.Option(..., "--model", help="Exact worker model identity."),
    azure_endpoint: str | None = typer.Option(
        None,
        "--azure-endpoint",
        help="Exact Azure OpenAI endpoint. Required for --provider azure.",
    ),
    azure_deployment: str | None = typer.Option(
        None,
        "--azure-deployment",
        help="Azure OpenAI deployment name. Required for --provider azure.",
    ),
    azure_api_version: str | None = typer.Option(
        None,
        "--azure-api-version",
        help="Azure OpenAI API version. Required for --provider azure.",
    ),
    bedrock_region: str | None = typer.Option(
        None,
        "--bedrock-region",
        help="Exact AWS source region. Required for --provider bedrock.",
    ),
    task_backend: str = typer.Option(
        "local",
        "--task-backend",
        help="Ground-truth task environment: local or e2b.",
    ),
    attempts: int = typer.Option(1, "--attempts", min=1, help="Attempts per task."),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        min=1,
        help="Maximum concurrent Harbor trials.",
    ),
    job_name: str = typer.Option(
        "harness-eval",
        "--job-name",
        help="Stable Harbor job name used for resume and stale-run checks.",
    ),
    jobs_dir: str | None = typer.Option(
        None,
        "--jobs-dir",
        help="Harbor job root. Default: <root>/eval-jobs.",
    ),
    turn_timeout_s: float = typer.Option(
        300.0,
        "--turn-timeout",
        min=0.001,
        help="Maximum seconds for one WMH pi turn.",
    ),
    runner_image: str = typer.Option(
        PI_CONTAINER_IMAGE,
        "--runner-image",
        help="Digest-pinned image used for the isolated pi runner.",
    ),
    root: str = typer.Option(ARTIFACT_DIR, "--root", help="Project artifact directory."),
    out: str = typer.Option(..., "--out", help="Canonical benchmark result JSON output."),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Confirm model and task-environment costs without prompting.",
    ),
) -> None:
    """Evaluate a stored harness against Harbor ground-truth task data."""
    candidate = _load_harness(harness, root)
    dataset_config = _build_dataset_config(
        dataset_path,
        dataset,
        dataset_ref,
        task_names=task_names,
        excluded_task_names=excluded_task_names,
    )
    provider_config = _build_provider_config(
        provider,
        model,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        azure_api_version=azure_api_version,
        bedrock_region=bedrock_region,
    )
    backend = _parse_task_backend(task_backend)
    resolved_jobs_dir = (
        Path(jobs_dir).expanduser().resolve()
        if jobs_dir is not None
        else (Path(root).expanduser().resolve() / "eval-jobs")
    )
    spec = HarborJobSpec(
        job_name=job_name,
        jobs_dir=resolved_jobs_dir,
        datasets=[dataset_config],
        n_attempts=attempts,
        n_concurrent_trials=concurrency,
        environment_backend=backend,
    )
    try:
        evaluator = HarborEvaluator(
            spec,
            provider_config,
            runner_image=runner_image,
            turn_timeout_s=turn_timeout_s,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    output_path = _validate_output_path(
        out,
        active_job_dir=resolved_jobs_dir / job_name,
        active_lease_path=harbor_job_lease_path(resolved_jobs_dir, job_name),
    )

    _console.print(
        f"harness eval: [bold]{candidate.name}@v{candidate.version}[/bold], "
        f"provider [bold]{provider_config.kind.value}/{provider_config.model}[/bold], "
        f"{attempts} attempt(s) per task, concurrency {concurrency}, "
        f"task backend {backend.value}. This run can incur model and environment costs."
    )
    if not yes and not typer.confirm("Proceed with paid evaluation?", default=False):
        raise typer.Exit(1)

    try:
        loaded = asyncio.run(evaluator.evaluate(candidate))
        _atomic_write(output_path, loaded.result.model_dump_json(indent=2) + "\n")
    except (OSError, RuntimeError, ValueError) as exc:
        raise ClickException(str(exc)) from exc

    result = loaded.result
    all_scored = result.expected_trials > 0 and result.n_scored == result.expected_trials
    state = "[green]complete[/green]" if all_scored else "[red]not fully scored[/red]"
    _console.print(
        f"{state} scored={result.n_scored} "
        f"timeout={result.n_task_timeouts} "
        f"infra={result.n_infrastructure_errors} "
        f"cancelled={result.n_cancelled} "
        f"incomplete={result.n_incomplete} "
        f"unclassified={result.n_unclassified_errors}"
    )
    digest_count = len({trial.task_checksum for trial in result.trials})
    _console.print(
        f"resolved task digests={digest_count}; full digests are recorded per cell in "
        f"{output_path} and {loaded.job_dir / 'wmh-manifest.json'}"
    )
    if result.n_cancelled:
        _console.print(
            "cancelled cells are terminal evidence; preserve this job and use a new --job-name "
            "for an explicit rerun"
        )
    _console.print(f"wrote canonical result -> {output_path}")
    if not all_scored:
        raise typer.Exit(1)


def _load_harness(selector: str, root: str) -> HarnessDoc:
    name, separator, ref = selector.partition("@")
    if not name or (separator and not ref):
        raise typer.BadParameter("harness must be name or name@ref", param_hint="HARNESS")
    try:
        candidate = HarnessStore(root).load(name, ref or None)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="HARNESS") from exc
    if candidate.runtime_kind() != "pi-node":
        raise typer.BadParameter(
            f"harness {selector!r} uses runtime {candidate.runtime_kind()!r}; "
            "ground-truth Harbor evaluation requires a pi-node harness",
            param_hint="HARNESS",
        )
    return candidate


def _build_dataset_config(
    dataset_path: str | None,
    dataset: str | None,
    dataset_ref: str | None,
    *,
    task_names: list[str] | None,
    excluded_task_names: list[str] | None,
) -> DatasetConfig:
    included, excluded = _normalize_task_filters(task_names, excluded_task_names)
    if (dataset_path is None) == (dataset is None):
        raise typer.BadParameter(
            "provide exactly one of --dataset-path or --dataset",
            param_hint="--dataset-path/--dataset",
        )
    if dataset_path is not None:
        if dataset_ref is not None:
            raise typer.BadParameter(
                "--dataset-ref is only valid with --dataset", param_hint="--dataset-ref"
            )
        path = Path(dataset_path).expanduser().resolve()
        if not path.is_dir():
            raise typer.BadParameter(
                f"local dataset directory does not exist: {path}", param_hint="--dataset-path"
            )
        return DatasetConfig(
            path=path,
            task_names=included,
            exclude_task_names=excluded,
        )

    assert dataset is not None
    if not dataset_ref:
        raise typer.BadParameter(
            "--dataset requires an explicit --dataset-ref", param_hint="--dataset-ref"
        )
    try:
        if "/" in dataset:
            return DatasetConfig(
                name=dataset,
                ref=dataset_ref,
                task_names=included,
                exclude_task_names=excluded,
            )
        return DatasetConfig(
            name=dataset,
            version=dataset_ref,
            task_names=included,
            exclude_task_names=excluded,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--dataset") from exc


def _normalize_task_filters(
    task_names: list[str] | None,
    excluded_task_names: list[str] | None,
) -> tuple[list[str] | None, list[str] | None]:
    """Validate repeatable Harbor task filters and preserve their first-seen order."""
    included = _normalize_task_filter(task_names, option="--task")
    excluded = _normalize_task_filter(excluded_task_names, option="--exclude-task")
    if included and excluded:
        raise typer.BadParameter(
            "--task and --exclude-task are mutually exclusive",
            param_hint="--task/--exclude-task",
        )
    return included, excluded


def _normalize_task_filter(values: list[str] | None, *, option: str) -> list[str] | None:
    """Strip, reject blanks, and deduplicate one repeatable task-filter option."""
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        value = raw.strip()
        if not value:
            raise typer.BadParameter("task filter cannot be blank", param_hint=option)
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized or None


def _build_provider_config(
    provider: str,
    model: str,
    *,
    azure_endpoint: str | None,
    azure_deployment: str | None,
    azure_api_version: str | None,
    bedrock_region: str | None,
) -> ProviderConfig:
    if provider == ProviderKind.AZURE_OPENAI.value:
        if bedrock_region is not None:
            raise typer.BadParameter(
                "--bedrock-region is only valid with --provider bedrock",
                param_hint="--bedrock-region",
            )
        if not azure_deployment:
            raise typer.BadParameter(
                "--provider azure requires --azure-deployment",
                param_hint="--azure-deployment",
            )
        if not azure_endpoint:
            raise typer.BadParameter(
                "--provider azure requires --azure-endpoint",
                param_hint="--azure-endpoint",
            )
        if not azure_api_version:
            raise typer.BadParameter(
                "--provider azure requires --azure-api-version",
                param_hint="--azure-api-version",
            )
        return ProviderConfig(
            kind=ProviderKind.AZURE_OPENAI,
            model=model,
            endpoint=azure_endpoint,
            deployment=azure_deployment,
            api_version=azure_api_version,
        )
    if provider == ProviderKind.BEDROCK.value:
        azure_values = (azure_endpoint, azure_deployment, azure_api_version)
        if any(value is not None for value in azure_values):
            raise typer.BadParameter(
                "Azure options are only valid with --provider azure",
                param_hint="--azure-endpoint/--azure-deployment/--azure-api-version",
            )
        if not bedrock_region:
            raise typer.BadParameter(
                "--provider bedrock requires --bedrock-region",
                param_hint="--bedrock-region",
            )
        return ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model=model,
            region=bedrock_region,
        )
    raise typer.BadParameter(
        f"unsupported provider {provider!r}; choose azure or bedrock", param_hint="--provider"
    )


def _parse_task_backend(value: str) -> HarborEnvironmentBackend:
    try:
        return HarborEnvironmentBackend(value)
    except ValueError:
        raise typer.BadParameter(
            f"unknown task backend {value!r}; choose local or e2b",
            param_hint="--task-backend",
        ) from None


def _validate_output_path(
    path: str,
    *,
    active_job_dir: Path,
    active_lease_path: Path,
) -> Path:
    """Resolve a safe canonical-output path outside Harbor's raw job evidence."""
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise typer.BadParameter("output path cannot be a symlink", param_hint="--out")
    if requested.exists() and not requested.is_file():
        raise typer.BadParameter(
            "output path must be a regular file location",
            param_hint="--out",
        )
    target = requested.resolve()
    job_dir = active_job_dir.expanduser().resolve()
    if target == job_dir or target.is_relative_to(job_dir):
        raise typer.BadParameter(
            "output path cannot be inside the active Harbor job directory",
            param_hint="--out",
        )
    lease_path = active_lease_path.expanduser().resolve()
    if target == lease_path:
        raise typer.BadParameter(
            "output path cannot replace the active Harbor job lease",
            param_hint="--out",
        )
    return target


def _atomic_write(path: Path, payload: str) -> None:
    target = path.expanduser()
    if target.is_symlink():
        raise OSError(f"refusing to replace output through symlink: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
