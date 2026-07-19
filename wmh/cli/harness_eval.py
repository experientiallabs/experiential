"""Ground-truth benchmark evaluation for stored WMH harnesses."""

from __future__ import annotations

import asyncio
import os
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TypedDict

import typer
from click import ClickException
from harbor.models.job.config import DatasetConfig
from pydantic import TypeAdapter
from rich.console import Console

from wmh.config import ARTIFACT_DIR
from wmh.evals.harbor._file_lease import exclusive_posix_file_lease
from wmh.evals.harbor.config import HarborEnvironmentBackend, HarborJobSpec
from wmh.evals.harbor.e2b_environment import (
    E2BSpendLimitAttestation,
    E2BSpendLimitTrust,
    freeze_exact_e2b_build_spec,
    prepare_exact_e2b_build,
    register_exact_e2b_build_record,
)
from wmh.evals.harbor.evaluator import HarborEvaluator, harbor_job_lease_path
from wmh.harness.doc import HarnessDoc
from wmh.harness.pi_runner_backend import LocalPiRunnerSpec, PiRunnerBackendSpec
from wmh.harness.store import HarnessStore
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.tracking.budget import BudgetAccount, TimedResourceBudgetAccount
from wmh.tracking.rate_limit import (
    E2B_SANDBOX_CREATE_RATE_POLICY,
    ExternalDispatchRateAuthority,
    bind_external_dispatch_rate_authority,
)

_console = Console()
_OUTPUT_LEASE_SUFFIX = ".wmh-eval-output.lock"
_MAX_RUNNER_SPEC_BYTES = 64 * 1024


class ConcurrentHarnessOutputError(RuntimeError):
    """Another process already holds the exclusive lease for a canonical output."""


class _CreateRateKwargs(TypedDict, total=False):
    create_rate_authority: ExternalDispatchRateAuthority


def register(app: typer.Typer) -> None:
    """Register ground-truth harness evaluation on the harness command group."""
    app.command("eval")(eval_harness)
    app.command("register-e2b-build")(register_e2b_build)
    app.command("prepare-e2b-build")(prepare_e2b_build)


def register_e2b_build(
    jobs_dir: str = typer.Option(..., "--jobs-dir"),
    environment_dir: str = typer.Option(..., "--environment-dir"),
    docker_image: str | None = typer.Option(None, "--docker-image"),
    template_id: str = typer.Option(..., "--template-id"),
    build_id: str = typer.Option(..., "--build-id"),
    cpu_count: int = typer.Option(..., "--cpu-count", min=1),
    memory_mb: int = typer.Option(..., "--memory-mb", min=1),
    acknowledge_preexisting_outside_study: bool = typer.Option(
        False,
        "--acknowledge-preexisting-outside-study",
        help="Acknowledge that this build was paid outside the current study ledger.",
    ),
) -> None:
    """Register already-built exact E2B IDs; this command never calls E2B."""
    try:
        spec = freeze_exact_e2b_build_spec(
            environment_dir=Path(environment_dir),
            docker_image=docker_image,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
        )
        record = register_exact_e2b_build_record(
            jobs_dir=Path(jobs_dir),
            environment_id=spec.environment_id,
            build_context_digest=spec.build_context_digest,
            docker_image=spec.docker_image,
            template_id=template_id,
            build_id=build_id,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
            acknowledge_preexisting_outside_study=acknowledge_preexisting_outside_study,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise ClickException(str(exc)) from exc
    _console.print(
        f"registered exact E2B build {record.exact_template_ref} for {record.build_config_digest}"
    )


def prepare_e2b_build(
    jobs_dir: str = typer.Option(..., "--jobs-dir"),
    environment_dir: str = typer.Option(..., "--environment-dir"),
    docker_image: str | None = typer.Option(None, "--docker-image"),
    cpu_count: int = typer.Option(..., "--cpu-count", min=1),
    memory_mb: int = typer.Option(..., "--memory-mb", min=1),
    resource_budget_account_path: str = typer.Option(..., "--resource-budget-account"),
    spend_limit_attestation_path: str = typer.Option(
        ...,
        "--e2b-spend-limit-attestation",
        help="Fresh signed operator statement for the active E2B account and remaining cap.",
    ),
    spend_limit_trust_path: str = typer.Option(
        ...,
        "--e2b-spend-limit-trust",
        help="Pinned Ed25519 public key and E2B team/account identity for that statement.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Confirm the budgeted E2B template-build provider call.",
    ),
) -> None:
    """Prepare one exact E2B build through a timed study-budget reservation."""
    if not yes:
        raise ClickException("budgeted E2B build preparation requires explicit --yes approval")
    try:
        source = Path(environment_dir)
        spec = freeze_exact_e2b_build_spec(
            environment_dir=source,
            docker_image=docker_image,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
        )
        account = _load_timed_resource_budget_account(
            resource_budget_account_path,
            param_hint="--resource-budget-account",
        )
        spend_limit = _load_e2b_spend_limit_attestation(spend_limit_attestation_path)
        spend_limit_trust = _load_e2b_spend_limit_trust(spend_limit_trust_path)
        record = asyncio.run(
            prepare_exact_e2b_build(
                jobs_dir=Path(jobs_dir),
                environment_dir=source,
                spec=spec,
                budget_account=account,
                provider_spend_limit=spend_limit,
                provider_spend_limit_trust=spend_limit_trust,
            )
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise ClickException(str(exc)) from exc
    _console.print(
        "prepared budgeted exact E2B build "
        f"{record.exact_template_ref} for {record.build_config_digest}"
    )


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
        help=(
            "Harbor task environment: local Docker or E2B. This is independent of the Pi "
            "runner backend."
        ),
    ),
    allow_preexisting_e2b_builds: bool = typer.Option(
        False,
        "--allow-preexisting-e2b-builds",
        help="Explicitly admit E2B builds paid outside this scored study. Disabled by default.",
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
    runner_spec_path: str | None = typer.Option(
        None,
        "--runner-spec",
        help="Pi runner backend spec JSON. Local Docker is the default when omitted.",
    ),
    root: str = typer.Option(ARTIFACT_DIR, "--root", help="Project artifact directory."),
    out: str = typer.Option(..., "--out", help="Canonical benchmark result JSON output."),
    budget_account_path: str | None = typer.Option(
        None,
        "--budget-account",
        help="Frozen BudgetAccount JSON. Required unless development bypass is explicit.",
    ),
    task_resource_budget_account_paths: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--task-resource-budget-account",
        help="TimedResourceBudgetAccount JSON for an E2B task class. Repeat per class.",
    ),
    runner_resource_budget_account_path: str | None = typer.Option(
        None,
        "--runner-resource-budget-account",
        help="TimedResourceBudgetAccount JSON for the exact E2B Pi runner class.",
    ),
    allow_unbudgeted_development: bool = typer.Option(
        False,
        "--allow-unbudgeted-development",
        help="Explicitly permit an unmetered development run. Never use for paid experiments.",
    ),
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
    resource_account_paths = tuple(task_resource_budget_account_paths or ())
    if allow_unbudgeted_development and (
        budget_account_path is not None
        or resource_account_paths
        or runner_resource_budget_account_path is not None
    ):
        raise typer.BadParameter(
            "budget accounts and --allow-unbudgeted-development are mutually exclusive"
        )
    if budget_account_path is None and not allow_unbudgeted_development:
        raise typer.BadParameter(
            "paid harness evaluation requires --budget-account; use "
            "--allow-unbudgeted-development only for an explicitly unmetered local check"
        )
    budget_account = (
        _load_budget_account(budget_account_path) if budget_account_path is not None else None
    )
    backend = _parse_task_backend(task_backend)
    if allow_preexisting_e2b_builds and backend is not HarborEnvironmentBackend.E2B:
        raise typer.BadParameter(
            "--allow-preexisting-e2b-builds requires --task-backend e2b",
            param_hint="--allow-preexisting-e2b-builds",
        )
    try:
        runner_spec = _load_runner_spec(runner_spec_path)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--runner-spec") from exc
    if allow_unbudgeted_development and (
        backend is HarborEnvironmentBackend.E2B or runner_spec.backend == "e2b"
    ):
        raise typer.BadParameter(
            "--allow-unbudgeted-development is restricted to local task and runner backends"
        )
    task_resource_budget_accounts = tuple(
        _load_timed_resource_budget_account(path, param_hint="--task-resource-budget-account")
        for path in resource_account_paths
    )
    runner_resource_budget_account = (
        _load_timed_resource_budget_account(
            runner_resource_budget_account_path,
            param_hint="--runner-resource-budget-account",
        )
        if runner_resource_budget_account_path is not None
        else None
    )
    resolved_jobs_dir = (
        Path(jobs_dir).expanduser().resolve()
        if jobs_dir is not None
        else (Path(root).expanduser().resolve() / "eval-jobs")
    )
    requires_create_rate = backend is HarborEnvironmentBackend.E2B or runner_spec.backend == "e2b"
    create_rate_authority: ExternalDispatchRateAuthority | None
    if requires_create_rate:
        create_rate_authority = ExternalDispatchRateAuthority.bootstrap(
            (resolved_jobs_dir / ".wmh-e2b-create-rate.json").resolve(),
            E2B_SANDBOX_CREATE_RATE_POLICY,
        )
        bind_external_dispatch_rate_authority(create_rate_authority)
    else:
        create_rate_authority = None
    spec = HarborJobSpec(
        job_name=job_name,
        jobs_dir=resolved_jobs_dir,
        datasets=[dataset_config],
        n_attempts=attempts,
        n_concurrent_trials=concurrency,
        environment_backend=backend,
        create_rate_policy=(E2B_SANDBOX_CREATE_RATE_POLICY if requires_create_rate else None),
        allow_preexisting_e2b_builds=allow_preexisting_e2b_builds,
    )
    create_rate_kwargs: _CreateRateKwargs = {}
    if create_rate_authority is not None:
        create_rate_kwargs["create_rate_authority"] = create_rate_authority
    try:
        evaluator = HarborEvaluator(
            spec,
            provider_config,
            runner_spec=runner_spec,
            turn_timeout_s=turn_timeout_s,
            budget_account=budget_account,
            task_resource_budget_accounts=task_resource_budget_accounts,
            runner_resource_budget_account=runner_resource_budget_account,
            **create_rate_kwargs,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    output_path = _validate_output_path(
        out,
        jobs_dir=resolved_jobs_dir,
        active_job_dir=resolved_jobs_dir / job_name,
        active_lease_path=harbor_job_lease_path(resolved_jobs_dir, job_name),
    )

    _console.print(
        f"harness eval: [bold]{candidate.name}@v{candidate.version}[/bold], "
        f"provider [bold]{provider_config.kind.value}/{provider_config.model}[/bold], "
        f"{attempts} attempt(s) per task, concurrency {concurrency}, "
        f"task backend {backend.value}, runner backend {runner_spec.backend}. "
        "This run can incur model and environment costs."
    )
    if not yes and not typer.confirm("Proceed with paid evaluation?", default=False):
        raise typer.Exit(1)

    try:
        with _exclusive_output_lease(output_path):
            loaded = asyncio.run(evaluator.evaluate(candidate))
            _atomic_write(output_path, loaded.result.model_dump_json(indent=2) + "\n")
    except (OSError, RuntimeError, ValueError) as exc:
        raise ClickException(str(exc)) from exc

    result = loaded.result
    all_scored = result.expected_trials > 0 and result.n_scored == result.expected_trials
    state = "[green]complete[/green]" if all_scored else "[red]not fully scored[/red]"
    _console.print(
        f"{state} scored={result.n_scored} "
        f"task_timeout={result.n_task_timeouts} "
        f"candidate_failed={result.n_candidate_failures} "
        f"candidate_timeout={result.n_candidate_timeouts} "
        f"candidate_resource_limit={result.n_candidate_resource_limits} "
        f"candidate_runtime_error={result.n_candidate_runtime_errors} "
        f"candidate_unclassified={result.n_candidate_unclassified_failures} "
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


def _load_budget_account(path: str) -> BudgetAccount:
    account_path = Path(path).expanduser()
    if account_path.is_symlink():
        raise typer.BadParameter(
            "budget account file cannot be a symlink",
            param_hint="--budget-account",
        )
    try:
        if not account_path.is_file():
            raise ValueError("budget account path must be a regular file")
        return BudgetAccount.model_validate_json(account_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--budget-account") from exc


def _load_timed_resource_budget_account(
    path: str,
    *,
    param_hint: str,
) -> TimedResourceBudgetAccount:
    account_path = Path(path).expanduser()
    if account_path.is_symlink():
        raise typer.BadParameter(
            "resource budget account file cannot be a symlink",
            param_hint=param_hint,
        )
    try:
        if not account_path.is_file():
            raise ValueError("resource budget account path must be a regular file")
        return TimedResourceBudgetAccount.model_validate_json(
            account_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint=param_hint) from exc


def _load_e2b_spend_limit_attestation(path: str) -> E2BSpendLimitAttestation:
    attestation_path = Path(path).expanduser()
    if attestation_path.is_symlink():
        raise typer.BadParameter(
            "E2B spending-limit attestation cannot be a symlink",
            param_hint="--e2b-spend-limit-attestation",
        )
    try:
        if not attestation_path.is_file():
            raise ValueError("E2B spending-limit attestation must be a regular file")
        payload = attestation_path.read_bytes()
        if len(payload) > 64 * 1024:
            raise ValueError("E2B spending-limit attestation exceeds 64 KiB")
        return E2BSpendLimitAttestation.model_validate_json(payload)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(
            str(exc),
            param_hint="--e2b-spend-limit-attestation",
        ) from exc


def _load_e2b_spend_limit_trust(path: str) -> E2BSpendLimitTrust:
    trust_path = Path(path).expanduser()
    if trust_path.is_symlink():
        raise typer.BadParameter(
            "E2B spend-limit trust cannot be a symlink",
            param_hint="--e2b-spend-limit-trust",
        )
    try:
        if not trust_path.is_file():
            raise ValueError("E2B spend-limit trust must be a regular file")
        payload = trust_path.read_bytes()
        if len(payload) > 64 * 1024:
            raise ValueError("E2B spend-limit trust exceeds 64 KiB")
        return E2BSpendLimitTrust.model_validate_json(payload)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(
            str(exc),
            param_hint="--e2b-spend-limit-trust",
        ) from exc


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


def _load_runner_spec(path: str | None) -> PiRunnerBackendSpec:
    """Load one bounded secret-free runner spec, or return the local default."""
    if path is None:
        return LocalPiRunnerSpec()
    payload = Path(path).expanduser().resolve().read_bytes()
    if len(payload) > _MAX_RUNNER_SPEC_BYTES:
        raise ValueError("runner spec exceeds 64 KiB")
    return TypeAdapter(PiRunnerBackendSpec).validate_json(payload)


def _validate_output_path(
    path: str,
    *,
    jobs_dir: Path,
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
    if requested.name.endswith(_OUTPUT_LEASE_SUFFIX):
        raise typer.BadParameter(
            "output path uses a reserved WMH evaluation lease name",
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
    jobs_root = jobs_dir.expanduser().resolve()
    if target == jobs_root or target.is_relative_to(jobs_root):
        raise typer.BadParameter(
            "output path cannot be inside another Harbor job or the Harbor jobs directory",
            param_hint="--out",
        )
    return target


def _output_lease_path(output_path: Path) -> Path:
    target = output_path.expanduser().resolve()
    return target.parent / f".{target.name}{_OUTPUT_LEASE_SUFFIX}"


def _exclusive_output_lease(output_path: Path) -> AbstractContextManager[None]:
    """Prevent duplicate paid work and last-writer-wins publication for one output."""
    target = output_path.expanduser().resolve()
    lock_path = _output_lease_path(target)
    return exclusive_posix_file_lease(
        lock_path,
        unsupported_error=RuntimeError(
            "harness evaluation output leases require POSIX file locking"
        ),
        irregular_file_error=OSError(
            f"harness evaluation output lock is not a regular file: {lock_path}"
        ),
        contention_error=ConcurrentHarnessOutputError(
            f"another process is already publishing harness evaluation output {target}"
        ),
    )


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
