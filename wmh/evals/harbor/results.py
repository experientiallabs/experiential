"""Strict ingestion of Harbor 0.18 job artifacts into benchmark-neutral results."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Self
from uuid import UUID

from harbor.models.job.lock import TrialLock
from harbor.models.job.result import JobResult
from harbor.models.trial.config import AgentConfig, TrialConfig
from harbor.models.trial.result import ExceptionInfo, TrialResult
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from wmh.evals.benchmark import (
    BenchmarkCandidateFailureReason,
    BenchmarkCandidateOutcome,
    BenchmarkCandidateStage,
    BenchmarkCandidateStatus,
    BenchmarkCandidateTerminalReason,
    BenchmarkCell,
    BenchmarkError,
    BenchmarkFailureKind,
    BenchmarkRunIdentity,
    BenchmarkRunResult,
    BenchmarkTrialResult,
    BenchmarkTrialStatus,
    BenchmarkUsage,
    BenchmarkUsageStatus,
    aggregate_benchmark_usage,
)
from wmh.evals.harbor.config import _require_supported_harbor_version
from wmh.harness.doc import HarnessDoc
from wmh.providers.base import ProviderConfig

_TASK_TIMEOUT_EXCEPTIONS = frozenset({"AgentTimeoutError"})
_CANCELLED_EXCEPTIONS = frozenset({"CancelledError", "RuntimeCancelled"})
_ENVIRONMENT_EXCEPTIONS = frozenset(
    {
        "AgentSetupTimeoutError",
        "BuildException",
        "ConnectError",
        "ConnectTimeout",
        "EnvironmentStartTimeoutError",
        "HealthcheckError",
        "PoolTimeout",
        "RateLimitException",
        "SandboxBuildFailedError",
        "SandboxException",
        "TimeoutException",
        "WmhPiCleanupError",
        "WmhPiEnvironmentError",
        "WmhPiRunnerError",
    }
)
_PROVIDER_EXCEPTIONS = frozenset(
    {
        "ApiConnectionClosedError",
        "ApiInternalServerError",
        "ApiOverloadedError",
        "ApiRateLimitError",
        "ApiUsageLimitError",
        "ContextLengthExceededError",
        "NetworkConnectionError",
        "OutputLengthExceededError",
        "UnknownApiError",
        "WmhPiProviderError",
    }
)
_VERIFIER_EXCEPTIONS = frozenset(
    {
        "AddTestsDirError",
        "DownloadVerifierDirError",
        "RewardFileEmptyError",
        "RewardFileNotFoundError",
        "VerifierOutputParseError",
        "VerifierTimeoutError",
    }
)
_REDACTED_ERROR_MESSAGES = {
    BenchmarkFailureKind.CANCELLED: "benchmark trial was cancelled",
    BenchmarkFailureKind.TASK_TIMEOUT: "agent execution exceeded the task time limit",
    BenchmarkFailureKind.ENVIRONMENT: "task environment infrastructure failed",
    BenchmarkFailureKind.PROVIDER: "model provider infrastructure failed",
    BenchmarkFailureKind.VERIFIER: "ground-truth verifier infrastructure failed",
    BenchmarkFailureKind.MALFORMED_RESULT: "Harbor result metadata was malformed",
    BenchmarkFailureKind.UNCLASSIFIED: "unclassified benchmark execution failure",
}
_CANDIDATE_STAGE_MAP = {
    "materialization": BenchmarkCandidateStage.SETUP,
    "setup": BenchmarkCandidateStage.SETUP,
    "turn": BenchmarkCandidateStage.EXECUTION,
    "execution": BenchmarkCandidateStage.EXECUTION,
}
_CANDIDATE_FAILURE_REASON_MAP = {
    "timeout": BenchmarkCandidateFailureReason.TIMEOUT,
    "resource_limit": BenchmarkCandidateFailureReason.RESOURCE_LIMIT,
    "runtime_error": BenchmarkCandidateFailureReason.RUNTIME_ERROR,
}
_CANDIDATE_TERMINAL_REASON_MAP = {
    "completed": BenchmarkCandidateTerminalReason.COMPLETED,
    "turn_limit": BenchmarkCandidateTerminalReason.LIMIT_REACHED,
    "max_turns": BenchmarkCandidateTerminalReason.LIMIT_REACHED,
    "aborted": BenchmarkCandidateTerminalReason.ABORTED,
}
_CANDIDATE_OUTCOME_METADATA_KEYS = frozenset(
    {
        "candidate_failure",
        "candidate_failure_stage",
        "candidate_failure_reason",
        "terminal_reason",
    }
)


class HarborTrialManifestEntry(BaseModel):
    """One trusted mapping from a stable benchmark cell to Harbor's trial directory."""

    cell: BenchmarkCell
    trial_name: str = Field(min_length=1)
    task_identity: str = Field(min_length=1)
    task_checksum: str = Field(min_length=1)
    task_source: str | None
    trial_lock_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("trial_name")
    @classmethod
    def _require_safe_trial_name(cls, value: str) -> str:
        if value in {".", ".."} or Path(value).is_absolute() or "/" in value or "\\" in value:
            raise ValueError("trial_name must be a single safe path component")
        return value

    @model_validator(mode="after")
    def _bind_portable_cell_config(self) -> Self:
        if self.cell.config_digest != self.trial_lock_digest:
            raise ValueError("cell config digest must match the resolved Harbor trial lock")
        return self


class HarborTrialManifest(BaseModel):
    """Exact trusted set of benchmark cells Harbor planned for one job."""

    job_name: str = Field(min_length=1)
    identity: BenchmarkRunIdentity
    agent_config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    entries: list[HarborTrialManifestEntry] = Field(default_factory=list)

    @field_validator("entries")
    @classmethod
    def _canonical_order(
        cls, value: list[HarborTrialManifestEntry]
    ) -> list[HarborTrialManifestEntry]:
        return sorted(
            value,
            key=lambda entry: (entry.cell.task_key, entry.cell.attempt, entry.cell.task_name),
        )

    @model_validator(mode="after")
    def _reject_duplicates(self) -> Self:
        cells = [(entry.cell.task_key, entry.cell.attempt) for entry in self.entries]
        duplicate_cells = sorted(cell for cell, count in Counter(cells).items() if count > 1)
        if duplicate_cells:
            raise ValueError(f"duplicate manifest benchmark cell(s): {duplicate_cells}")
        trial_names = [entry.trial_name for entry in self.entries]
        duplicate_names = sorted(name for name, count in Counter(trial_names).items() if count > 1)
        if duplicate_names:
            raise ValueError(f"duplicate manifest trial_name(s): {duplicate_names}")
        return self


class HarborTrialLocator(BaseModel):
    """Job-root-relative Harbor paths for one canonical benchmark cell."""

    cell: BenchmarkCell
    trial_dir: Path
    result_path: Path | None = None
    artifacts_dir: Path

    @field_validator("trial_dir", "result_path", "artifacts_dir")
    @classmethod
    def _require_relative_contained_path(cls, value: Path | None) -> Path | None:
        if value is not None and (value.is_absolute() or ".." in value.parts):
            raise ValueError("Harbor result locator must be relative to the job directory")
        return value


@dataclass(frozen=True)
class LoadedHarborJobResult:
    """Canonical evidence plus separate, contained filesystem locators."""

    result: BenchmarkRunResult
    job_dir: Path
    locators: tuple[HarborTrialLocator, ...]

    def resolve_path(self, relative_path: Path) -> Path:
        """Resolve one locator beneath this job root, rejecting traversal and symlink escapes."""
        return _resolve_contained(self.job_dir, relative_path)


def load_harbor_job_result(
    job_dir: str | Path,
    manifest: HarborTrialManifest,
) -> LoadedHarborJobResult:
    """Load one Harbor job against its exact trusted cell manifest.

    Missing expected cells are materialized as incomplete. Extra directories, mismatched task or
    trial identities, duplicate manifest entries, and paths escaping the job root are rejected.
    Malformed result files remain explicit infrastructure evidence rather than disappearing.
    """
    _require_supported_harbor_version()
    requested_root = Path(job_dir).expanduser()
    if requested_root.is_symlink():
        raise ValueError(f"Harbor job directory cannot be a symlink: {job_dir}")
    root = requested_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Harbor job directory does not exist: {job_dir}")
    if root.name != manifest.job_name:
        raise ValueError(
            f"Harbor manifest names job {manifest.job_name!r}, but directory is {root.name!r}"
        )

    job_result_path = _resolve_contained(root, Path("result.json"))
    if not job_result_path.is_file():
        raise FileNotFoundError(f"Harbor job is missing result.json: {root}")
    job_result = JobResult.model_validate_json(job_result_path.read_text(encoding="utf-8"))
    if job_result.n_total_trials != len(manifest.entries):
        raise ValueError(
            f"Harbor job declares {job_result.n_total_trials} trials but manifest contains "
            f"{len(manifest.entries)}"
        )

    expected_trial_names = {entry.trial_name for entry in manifest.entries}
    for child in root.iterdir():
        if (child.is_dir() or child.is_symlink()) and child.name not in expected_trial_names:
            raise ValueError(f"unexpected trial directory outside manifest: {child.name!r}")

    trials: list[BenchmarkTrialResult] = []
    locators: list[HarborTrialLocator] = []
    for entry in manifest.entries:
        trial, locator = _load_manifest_entry(
            root,
            entry,
            manifest,
            job_id=job_result.id,
        )
        trials.append(trial)
        locators.append(locator)

    return LoadedHarborJobResult(
        result=BenchmarkRunResult(
            job_name=manifest.job_name,
            identity=manifest.identity,
            expected_cells=[entry.cell for entry in manifest.entries],
            trials=trials,
            usage=aggregate_benchmark_usage(trial.usage for trial in trials),
        ),
        job_dir=root,
        locators=tuple(locators),
    )


def _load_manifest_entry(
    root: Path,
    entry: HarborTrialManifestEntry,
    manifest: HarborTrialManifest,
    *,
    job_id: UUID,
) -> tuple[BenchmarkTrialResult, HarborTrialLocator]:
    trial_rel = Path(entry.trial_name)
    trial_dir = _resolve_contained(root, trial_rel)
    result_rel = trial_rel / "result.json"
    artifacts_rel = trial_rel / "artifacts"

    if not trial_dir.exists():
        return (
            BenchmarkTrialResult(
                cell=entry.cell,
                task_identity=entry.task_identity,
                task_checksum=entry.task_checksum,
                source=entry.task_source,
                status=BenchmarkTrialStatus.INCOMPLETE,
            ),
            HarborTrialLocator(
                cell=entry.cell,
                trial_dir=trial_rel,
                result_path=None,
                artifacts_dir=artifacts_rel,
            ),
        )
    if not trial_dir.is_dir():
        raise ValueError(f"manifest trial path is not a directory: {entry.trial_name!r}")
    _resolve_contained(root, artifacts_rel)

    result_path = _resolve_contained(root, result_rel)
    locator = HarborTrialLocator(
        cell=entry.cell,
        trial_dir=trial_rel,
        result_path=(result_rel if result_path.exists() else None),
        artifacts_dir=artifacts_rel,
    )
    if not result_path.exists():
        return _load_incomplete_trial(root, trial_rel, entry, job_id=job_id), locator
    if not result_path.is_file():
        return _malformed_trial(entry, "InvalidTrialResultError"), locator

    try:
        result = TrialResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError):
        return _malformed_trial(entry, "InvalidTrialResultError"), locator

    trial_lock = _load_and_validate_trial_lock(root, trial_rel, entry)
    _validate_locked_config(result.config, trial_lock)
    _validate_trial_job_provenance(result.config, root=root, job_id=job_id)
    _validate_trial_identity(result, entry)
    candidate_outcome = _validate_run_identity(
        result,
        manifest.identity,
        manifest.agent_config_digest,
    )
    return _convert_trial(result, entry, candidate_outcome), locator


def _load_incomplete_trial(
    root: Path,
    trial_rel: Path,
    entry: HarborTrialManifestEntry,
    *,
    job_id: UUID,
) -> BenchmarkTrialResult:
    config_path = _resolve_contained(root, trial_rel / "config.json")
    if not config_path.exists():
        return BenchmarkTrialResult(
            cell=entry.cell,
            task_identity=entry.task_identity,
            task_checksum=entry.task_checksum,
            source=entry.task_source,
            status=BenchmarkTrialStatus.INCOMPLETE,
        )
    if not config_path.is_file():
        return _malformed_trial(entry, "InvalidTrialConfigError")
    try:
        config = TrialConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError):
        return _malformed_trial(entry, "InvalidTrialConfigError")
    _validate_trial_job_provenance(config, root=root, job_id=job_id)
    task_name = config.task.get_task_id().get_name()
    if (
        config.trial_name != entry.trial_name
        or task_name != entry.task_identity
        or config.task.source != entry.task_source
    ):
        raise ValueError(
            f"manifest expected task identity/source {entry.task_identity!r}/"
            f"{entry.task_source!r} in trial {entry.trial_name!r}, found "
            f"{task_name!r}/{config.task.source!r} in {config.trial_name!r}"
        )
    return BenchmarkTrialResult(
        cell=entry.cell,
        task_identity=entry.task_identity,
        task_checksum=entry.task_checksum,
        source=entry.task_source,
        status=BenchmarkTrialStatus.INCOMPLETE,
    )


def _validate_trial_job_provenance(
    config: TrialConfig,
    *,
    root: Path,
    job_id: UUID,
) -> None:
    if config.job_id != job_id:
        raise ValueError(
            f"Harbor trial {config.trial_name!r} job_id does not match its root job result"
        )
    try:
        trials_root = config.trials_dir.expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            f"Harbor trial {config.trial_name!r} trials_dir cannot be resolved safely"
        ) from exc
    if trials_root != root:
        raise ValueError(
            f"Harbor trial {config.trial_name!r} trials_dir does not resolve to its root job"
        )


def _validate_trial_identity(
    result: TrialResult,
    entry: HarborTrialManifestEntry,
) -> None:
    if result.trial_name != entry.trial_name or result.config.trial_name != entry.trial_name:
        raise ValueError(
            f"manifest expected trial {entry.trial_name!r}, found result {result.trial_name!r} "
            f"with config {result.config.trial_name!r}"
        )
    if result.task_name != entry.cell.task_name:
        raise ValueError(
            f"manifest expected top-level task {entry.cell.task_name!r}, found {result.task_name!r}"
        )
    task_id_names = {result.task_id.get_name(), result.config.task.get_task_id().get_name()}
    if task_id_names != {entry.task_identity}:
        found = sorted(task_id_names)
        found_label = found[0] if len(found) == 1 else found
        raise ValueError(
            f"manifest expected task identity {entry.task_identity!r} in trial "
            f"{entry.trial_name!r}, "
            f"found {found_label!r}"
        )
    if result.task_checksum != entry.task_checksum:
        raise ValueError(
            f"manifest expected task checksum {entry.task_checksum!r}, "
            f"found {result.task_checksum!r}"
        )
    if result.source != entry.task_source or result.config.task.source != entry.task_source:
        raise ValueError(
            f"manifest expected task source {entry.task_source!r}, found top-level "
            f"{result.source!r} with config {result.config.task.source!r}"
        )


def _validate_run_identity(
    result: TrialResult,
    expected: BenchmarkRunIdentity,
    expected_agent_config_digest: str,
) -> BenchmarkCandidateOutcome:
    actual_agent = result.agent_info
    if actual_agent.name != expected.agent_name or actual_agent.version != expected.agent_version:
        raise ValueError(
            f"manifest expected agent {expected.agent_name!r}@{expected.agent_version!r}, "
            f"found {actual_agent.name!r}@{actual_agent.version!r}"
        )
    model = actual_agent.model_info
    if model is None or model.provider != expected.provider or model.name != expected.model_name:
        found = None if model is None else f"{model.provider}/{model.name}"
        raise ValueError(
            f"manifest expected provider/model {expected.provider}/{expected.model_name}, "
            f"found {found!r}"
        )
    environment = result.config.environment.type
    actual_environment = environment.value if environment is not None else None
    if actual_environment != expected.task_environment.value:
        raise ValueError(
            f"manifest expected task environment {expected.task_environment.value!r}, "
            f"found {actual_environment!r}"
        )
    actual_agent_config_digest = harbor_agent_config_digest(result.config.agent)
    if actual_agent_config_digest != expected_agent_config_digest:
        raise ValueError(
            f"manifest expected agent config digest {expected_agent_config_digest!r}, "
            f"found {actual_agent_config_digest!r}"
        )
    _validate_agent_config_identity(result.config.agent, expected)
    contexts = [result.agent_result] if result.agent_result is not None else []
    contexts.extend(
        step.agent_result for step in result.step_results or [] if step.agent_result is not None
    )
    outcomes: list[BenchmarkCandidateOutcome] = []
    for context in contexts:
        metadata = context.metadata or {}
        candidate_hash = metadata.get("harness_hash")
        runner_image = metadata.get("runner_image")
        if candidate_hash != expected.candidate_hash:
            raise ValueError(
                f"manifest expected candidate hash {expected.candidate_hash!r}, "
                f"found {candidate_hash!r}"
            )
        if runner_image != expected.runner_image:
            raise ValueError(
                f"manifest expected runner image {expected.runner_image!r}, found {runner_image!r}"
            )
        outcomes.append(_parse_candidate_outcome(metadata))
    if not outcomes:
        return BenchmarkCandidateOutcome()
    first = outcomes[0]
    if any(outcome != first for outcome in outcomes[1:]):
        raise ValueError("Harbor step contexts contain inconsistent candidate outcome metadata")
    return first


def _parse_candidate_outcome(metadata: dict[str, object]) -> BenchmarkCandidateOutcome:
    """Parse only the trusted adapter's bounded candidate fields from one Harbor context."""
    present = _CANDIDATE_OUTCOME_METADATA_KEYS.intersection(metadata)
    if not present:
        return BenchmarkCandidateOutcome()
    if "candidate_failure" not in metadata:
        raise ValueError("candidate outcome metadata omits candidate_failure")
    failed = metadata["candidate_failure"]
    if not isinstance(failed, bool):
        raise ValueError("candidate outcome metadata candidate_failure must be boolean")

    if failed:
        if "terminal_reason" in metadata:
            raise ValueError("failed candidate outcome metadata cannot carry terminal_reason")
        raw_stage = metadata.get("candidate_failure_stage")
        if raw_stage is None and "candidate_failure_stage" in metadata:
            raise ValueError("candidate outcome metadata failure stage cannot be null")
        if raw_stage is not None:
            if not isinstance(raw_stage, str) or raw_stage not in _CANDIDATE_STAGE_MAP:
                raise ValueError("candidate outcome metadata has an unknown failure stage")
            stage = _CANDIDATE_STAGE_MAP[raw_stage]
        else:
            stage = None
        raw_reason = metadata.get("candidate_failure_reason")
        if not isinstance(raw_reason, str) or raw_reason not in _CANDIDATE_FAILURE_REASON_MAP:
            raise ValueError("candidate outcome metadata has an unknown failure reason")
        return BenchmarkCandidateOutcome(
            status=BenchmarkCandidateStatus.FAILED,
            stage=stage,
            failure_reason=_CANDIDATE_FAILURE_REASON_MAP[raw_reason],
        )

    if "candidate_failure_stage" in metadata or "candidate_failure_reason" in metadata:
        raise ValueError("completed candidate outcome metadata cannot carry failure details")
    raw_reason = metadata.get("terminal_reason")
    if raw_reason is None and "terminal_reason" in metadata:
        raise ValueError("candidate outcome metadata terminal reason cannot be null")
    if raw_reason is not None:
        if not isinstance(raw_reason, str) or raw_reason not in _CANDIDATE_TERMINAL_REASON_MAP:
            raise ValueError("candidate outcome metadata has an unknown terminal reason")
        terminal_reason = _CANDIDATE_TERMINAL_REASON_MAP[raw_reason]
    else:
        terminal_reason = None
    return BenchmarkCandidateOutcome(
        status=BenchmarkCandidateStatus.COMPLETED,
        terminal_reason=terminal_reason,
    )


def _validate_agent_config_identity(
    config: AgentConfig,
    expected: BenchmarkRunIdentity,
) -> None:
    expected_harbor_model = f"{expected.provider}/{expected.model_name}"
    if config.model_name != expected_harbor_model:
        raise ValueError(
            f"manifest expected Harbor agent model {expected_harbor_model!r}, "
            f"found {config.model_name!r}"
        )
    try:
        harness = HarnessDoc.model_validate(config.kwargs.get("harness"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Harbor agent config contains an invalid harness document") from exc
    if harness.execution_hash != expected.candidate_hash:
        raise ValueError(
            f"manifest expected candidate hash {expected.candidate_hash!r}, "
            f"found {harness.execution_hash!r} in Harbor agent config"
        )
    runner_image = config.kwargs.get("runner_image")
    if runner_image != expected.runner_image:
        raise ValueError(
            f"manifest expected runner image {expected.runner_image!r}, "
            f"found {runner_image!r} in Harbor agent config"
        )
    try:
        provider = ProviderConfig.model_validate(config.kwargs.get("provider_config"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Harbor agent config contains an invalid provider config") from exc
    if provider.kind.value != expected.provider or provider.model != expected.model_name:
        raise ValueError(
            f"manifest expected configured provider/model "
            f"{expected.provider}/{expected.model_name}, found "
            f"{provider.kind.value}/{provider.model}"
        )


def _convert_trial(
    result: TrialResult,
    entry: HarborTrialManifestEntry,
    candidate_outcome: BenchmarkCandidateOutcome,
) -> BenchmarkTrialResult:
    error = _convert_exception(result.exception_info)
    has_rewards = result.verifier_result is not None and result.verifier_result.rewards is not None
    if has_rewards and (error is None or error.kind is BenchmarkFailureKind.TASK_TIMEOUT):
        status = BenchmarkTrialStatus.SCORED
    elif error is not None:
        status = _status_for_failure(error.kind)
    else:
        status = BenchmarkTrialStatus.INCOMPLETE

    n_input, n_cache, n_output, cost = result.compute_token_cost_totals()
    usage_may_be_incomplete = error is not None and error.kind in {
        BenchmarkFailureKind.CANCELLED,
        BenchmarkFailureKind.TASK_TIMEOUT,
        BenchmarkFailureKind.PROVIDER,
    }
    rewards = None
    if status is BenchmarkTrialStatus.SCORED:
        assert result.verifier_result is not None
        assert result.verifier_result.rewards is not None
        rewards = dict(result.verifier_result.rewards)
    return BenchmarkTrialResult(
        cell=entry.cell,
        task_identity=entry.task_identity,
        task_checksum=entry.task_checksum,
        source=entry.task_source,
        status=status,
        rewards=rewards,
        error=error,
        candidate_outcome=candidate_outcome,
        usage=BenchmarkUsage(
            input_tokens=n_input,
            input_tokens_status=_usage_status(
                n_input,
                incomplete=usage_may_be_incomplete,
            ),
            cache_tokens=n_cache,
            cache_tokens_status=_usage_status(
                n_cache,
                incomplete=usage_may_be_incomplete,
            ),
            output_tokens=n_output,
            output_tokens_status=_usage_status(
                n_output,
                incomplete=usage_may_be_incomplete,
            ),
            cost_usd=cost,
            cost_usd_status=_usage_status(
                cost,
                incomplete=usage_may_be_incomplete,
            ),
        ),
    )


def _usage_status(
    value: int | float | None,
    *,
    incomplete: bool,
) -> BenchmarkUsageStatus:
    if value is None:
        return BenchmarkUsageStatus.UNAVAILABLE
    if incomplete:
        return BenchmarkUsageStatus.LOWER_BOUND
    return BenchmarkUsageStatus.EXACT


def _convert_exception(exception: ExceptionInfo | None) -> BenchmarkError | None:
    if exception is None:
        return None
    kind = _failure_kind(exception.exception_type)
    return BenchmarkError(
        kind=kind,
        type=exception.exception_type,
        message=_REDACTED_ERROR_MESSAGES[kind],
    )


def _failure_kind(exception_type: str) -> BenchmarkFailureKind:
    if exception_type in _TASK_TIMEOUT_EXCEPTIONS:
        return BenchmarkFailureKind.TASK_TIMEOUT
    if exception_type in _CANCELLED_EXCEPTIONS:
        return BenchmarkFailureKind.CANCELLED
    if exception_type in _ENVIRONMENT_EXCEPTIONS:
        return BenchmarkFailureKind.ENVIRONMENT
    if exception_type in _PROVIDER_EXCEPTIONS:
        return BenchmarkFailureKind.PROVIDER
    if exception_type in _VERIFIER_EXCEPTIONS:
        return BenchmarkFailureKind.VERIFIER
    return BenchmarkFailureKind.UNCLASSIFIED


def _status_for_failure(kind: BenchmarkFailureKind) -> BenchmarkTrialStatus:
    if kind is BenchmarkFailureKind.CANCELLED:
        return BenchmarkTrialStatus.CANCELLED
    if kind is BenchmarkFailureKind.TASK_TIMEOUT:
        return BenchmarkTrialStatus.TASK_TIMEOUT
    if kind is BenchmarkFailureKind.UNCLASSIFIED:
        return BenchmarkTrialStatus.UNCLASSIFIED_ERROR
    return BenchmarkTrialStatus.INFRASTRUCTURE_ERROR


def _malformed_trial(
    entry: HarborTrialManifestEntry,
    error_type: str,
) -> BenchmarkTrialResult:
    return BenchmarkTrialResult(
        cell=entry.cell,
        task_identity=entry.task_identity,
        task_checksum=entry.task_checksum,
        source=entry.task_source,
        status=BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
        error=BenchmarkError(
            kind=BenchmarkFailureKind.MALFORMED_RESULT,
            type=error_type,
            message="Harbor trial metadata is unreadable or invalid",
        ),
    )


def harbor_agent_config_digest(config: AgentConfig) -> str:
    """Hash replay-critical WMH agent code, model, concurrency, and kwargs."""
    payload = config.model_dump(
        mode="json",
        include={"import_path", "model_name", "n_concurrent", "kwargs"},
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def harbor_trial_lock_digest(lock: TrialLock) -> str:
    """Hash Harbor's resolved replay lock while removing host-only source locations."""
    payload = lock.model_dump(mode="json", exclude_none=True)
    task = payload["task"]
    for field in ("path", "git_url", "git_commit_id"):
        task.pop(field, None)
    for skill in payload.get("skills", []):
        for field in ("source", "git_url", "git_commit_id"):
            skill.pop(field, None)
    for instruction in payload.get("extra_instructions", []):
        instruction.pop("path", None)
    for compose in payload.get("extra_docker_compose", []):
        compose.pop("path", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _load_and_validate_trial_lock(
    root: Path,
    trial_rel: Path,
    entry: HarborTrialManifestEntry,
) -> TrialLock:
    lock_path = _resolve_contained(root, trial_rel / "lock.json")
    if not lock_path.is_file():
        raise ValueError(f"Harbor trial {entry.trial_name!r} is missing a valid lock.json")
    try:
        lock = TrialLock.model_validate_json(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise ValueError(
            f"Harbor trial {entry.trial_name!r} contains an invalid lock.json"
        ) from exc
    actual = harbor_trial_lock_digest(lock)
    if actual != entry.trial_lock_digest:
        raise ValueError(
            f"manifest expected trial lock digest {entry.trial_lock_digest!r}, found {actual!r}"
        )
    return lock


def _validate_locked_config(config: TrialConfig, lock: TrialLock) -> None:
    locked_fields = (
        ("install_only", config.install_only, lock.install_only),
        ("timeout_multiplier", config.timeout_multiplier, lock.timeout_multiplier),
        (
            "agent_timeout_multiplier",
            config.agent_timeout_multiplier,
            lock.agent_timeout_multiplier,
        ),
        (
            "verifier_timeout_multiplier",
            config.verifier_timeout_multiplier,
            lock.verifier_timeout_multiplier,
        ),
        (
            "agent_setup_timeout_multiplier",
            config.agent_setup_timeout_multiplier,
            lock.agent_setup_timeout_multiplier,
        ),
        (
            "environment_build_timeout_multiplier",
            config.environment_build_timeout_multiplier,
            lock.environment_build_timeout_multiplier,
        ),
        ("agent", config.agent, lock.agent),
        ("environment", config.environment, lock.environment),
        ("verifier", config.verifier, lock.verifier),
    )
    mismatched = [name for name, actual, expected in locked_fields if actual != expected]
    if mismatched:
        raise ValueError(f"Harbor result config differs from trial lock: {sorted(mismatched)}")


def _resolve_contained(root: Path, relative_path: Path) -> Path:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"path escapes job directory: {relative_path}")
    unresolved = root
    for part in relative_path.parts:
        unresolved /= part
        if unresolved.is_symlink():
            raise ValueError(f"path inside job directory cannot be a symlink: {relative_path}")
    candidate = unresolved.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"path escapes job directory: {relative_path}")
    return candidate
