"""Tests for strict, lossless Harbor 0.18 result ingestion."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from harbor.models.agent.context import AgentContext
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.lock import TaskLock, TrialLock
from harbor.models.job.result import JobResult, JobStats
from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, ModelInfo, StepResult, TrialResult
from harbor.models.verifier.result import VerifierResult
from llm_waterfall import ResponseTranslationFailure
from pydantic import ValidationError

from wmh.evals.benchmark import (
    BenchmarkCandidateFailureReason,
    BenchmarkCandidateStage,
    BenchmarkCandidateStatus,
    BenchmarkCandidateTerminalReason,
    BenchmarkCell,
    BenchmarkFailureKind,
    BenchmarkRunHealth,
    BenchmarkRunIdentity,
    BenchmarkTaskEnvironment,
    BenchmarkTrialStatus,
    BenchmarkUsageStatus,
)
from wmh.evals.harbor.e2b_environment import TASK_E2B_LEASE_FILE
from wmh.evals.harbor.results import (
    HarborTrialManifest,
    HarborTrialManifestEntry,
    harbor_agent_config_digest,
    harbor_trial_lock_digest,
    load_harbor_job_result,
)
from wmh.harness.pi_runner import pi_node_baseline
from wmh.harness.pi_runner_backend import LocalPiRunnerSpec, runner_owner_id
from wmh.providers.failure_attribution import ProviderFailureReason, ProviderFailureStage

_NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
_TASK_CHECKSUM = "sha256:" + "b" * 64
_RUNTIME_TASK_CHECKSUM = "a" * 64
_TASK_SOURCE = "example-benchmark"
_HARNESS = pi_node_baseline("candidate")
_TASK_ENVIRONMENT_ATTESTATION = {
    "schema_version": 2,
    "backend": "docker",
    "daemon_platform": "linux/amd64",
    "requested_storage_mb": None,
    "storage_capacity_scope": "shared_task_filesystem_available",
    "storage_provider_enforced": False,
    "storage_requirement_satisfied": True,
    "services": [
        {
            "service": "main",
            "replica": 1,
            "image_id": "sha256:" + "c" * 64,
            "image_platform": "linux/amd64",
        }
    ],
}
_TASK_ENVIRONMENT_DIGEST = (
    "sha256:"
    + hashlib.sha256(
        json.dumps(
            _TASK_ENVIRONMENT_ATTESTATION,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
)
_E2B_LAUNCH_CONFIG_DIGEST = "sha256:" + "d" * 64
_E2B_TASK_ENVIRONMENT_ATTESTATION = {
    "schema_version": 3,
    "backend": "e2b",
    "environment_id": "environment-immutable",
    "build_config_digest": "sha256:" + "e" * 64,
    "launch_config_digest": _E2B_LAUNCH_CONFIG_DIGEST,
    "template_id": "template-immutable",
    "build_id": "build-immutable",
    "platform": "linux/x86_64",
    "cpu_count": 2,
    "memory_mb": 1024,
    "requested_storage_mb": 10_240,
    "observed_storage_mb": 20_480,
    "storage_capacity_scope": "provider_reported_total",
    "envd_version": "1.2.3",
    "network_mode": "no_network",
    "allowed_hosts": [],
    "internet_access": False,
    "network_allow_out": [],
    "network_deny_out": ["0.0.0.0/0"],
    "lease_timeout_s": 86_400,
    "timeout_action": "kill",
    "auto_resume": False,
    "volume_mounts": False,
}
_E2B_TASK_ENVIRONMENT_DIGEST = (
    "sha256:"
    + hashlib.sha256(
        json.dumps(
            _E2B_TASK_ENVIRONMENT_ATTESTATION,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
)
_RUNNER = LocalPiRunnerSpec()


def _runner_lease_receipt(trial_name: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "backend": "local",
        "lease_id": f"lease-{trial_name}",
        "owner_id": runner_owner_id(trial_name),
        "config_digest": _RUNNER.config_digest,
        "state": "retired",
        "resource_id": f"container-{trial_name}",
        "created_at": "2026-07-18T11:59:00Z",
        "expected_end_at": None,
        "retired_at": "2026-07-18T12:00:00Z",
    }


def _task_environment_lease_receipt(trial_name: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "backend": "e2b",
        "lease_id": f"task-environment-{trial_name}",
        "owner_id": runner_owner_id(trial_name),
        "config_digest": _E2B_LAUNCH_CONFIG_DIGEST,
        "state": "retired",
        "resource_id": f"sandbox-{trial_name}",
        "created_at": "2026-07-18T11:59:00Z",
        "expected_end_at": "2026-07-18T12:59:00Z",
        "retired_at": "2026-07-18T12:00:00Z",
    }


def _agent_config() -> AgentConfig:
    return AgentConfig(
        import_path="wmh.evals.harbor.agent:WmhPiAgent",
        model_name="bedrock/model",
        kwargs={
            "harness": _HARNESS.model_dump(mode="json"),
            "provider_config": {"kind": "bedrock", "model": "model"},
            "runner_spec": _RUNNER.model_dump(mode="json"),
        },
    )


_AGENT_CONFIG_DIGEST = harbor_agent_config_digest(_agent_config())
_RUN_IDENTITY = BenchmarkRunIdentity(
    candidate_hash=_HARNESS.execution_hash,
    agent_name="wmh-pi",
    agent_version="0.1.0",
    provider="bedrock",
    model_name="model",
    task_environment=BenchmarkTaskEnvironment.DOCKER,
    runner_config_digest=_RUNNER.config_digest,
    runner_environment_digest=_RUNNER.attestation.digest,
    run_config_digest=_AGENT_CONFIG_DIGEST,
)


def _task_key(task_name: str) -> str:
    return f"{_TASK_SOURCE}/{task_name}@{_TASK_CHECKSUM}"


def _trial_lock(task_name: str, config: TrialConfig) -> TrialLock:
    return TrialLock(
        task=TaskLock(
            name=task_name,
            type="local",
            digest=_TASK_CHECKSUM,
            source=config.task.source,
        ),
        install_only=config.install_only,
        timeout_multiplier=config.timeout_multiplier,
        agent_timeout_multiplier=config.agent_timeout_multiplier,
        verifier_timeout_multiplier=config.verifier_timeout_multiplier,
        agent_setup_timeout_multiplier=config.agent_setup_timeout_multiplier,
        environment_build_timeout_multiplier=config.environment_build_timeout_multiplier,
        agent=config.agent,
        environment=config.environment,
        verifier=config.verifier,
    )


def _trial(
    tmp_path: Path,
    name: str,
    *,
    trial_name: str | None = None,
    rewards: dict[str, float | int] | None = None,
    exception_type: str | None = None,
    candidate_metadata: dict[str, object] | None = None,
) -> TrialResult:
    task = TaskConfig(path=tmp_path / name, source=_TASK_SOURCE)
    config = TrialConfig(
        task=task,
        trial_name=trial_name or f"{name}__harbor",
        agent=_agent_config(),
    )
    exception = (
        ExceptionInfo(
            exception_type=exception_type,
            exception_message="failed",
            exception_traceback="traceback",
            occurred_at=_NOW,
        )
        if exception_type is not None
        else None
    )
    return TrialResult(
        task_name=name,
        trial_name=config.trial_name,
        trial_uri=f"file://{tmp_path}/{config.trial_name}",
        task_id=task.get_task_id(),
        source=task.source,
        task_checksum=_RUNTIME_TASK_CHECKSUM,
        config=config,
        agent_info=AgentInfo(
            name="wmh-pi",
            version="0.1.0",
            model_info=ModelInfo(name="model", provider="bedrock"),
        ),
        agent_result=AgentContext(
            n_input_tokens=10,
            n_cache_tokens=2,
            n_output_tokens=4,
            cost_usd=0.05,
            metadata={
                "harness_hash": _HARNESS.execution_hash,
                "runner_config_digest": _RUNNER.config_digest,
                "runner_environment_digest": _RUNNER.attestation.digest,
                "runner_environment_attestation": _RUNNER.attestation.evidence,
                "runner_lease_receipt": _runner_lease_receipt(config.trial_name),
                "model_calls": 1,
                "task_environment_digest": _TASK_ENVIRONMENT_DIGEST,
                "task_environment_attestation": _TASK_ENVIRONMENT_ATTESTATION,
                "run_health": "valid",
                **(candidate_metadata or {}),
            },
        ),
        verifier_result=(VerifierResult(rewards=rewards) if rewards is not None else None),
        exception_info=exception,
        started_at=_NOW,
        finished_at=_NOW,
    )


def _write_job(job_dir: Path, trials: list[TrialResult], *, expected: int) -> None:
    job_dir.mkdir(parents=True)
    job_id = uuid4()
    for trial in trials:
        trial.config.job_id = job_id
        trial.config.trials_dir = job_dir.resolve()
    result = JobResult(
        id=job_id,
        started_at=_NOW,
        finished_at=_NOW,
        n_total_trials=expected,
        stats=JobStats.from_trial_results(trials, n_total_trials=expected),
        trial_results=trials,
    )
    (job_dir / "result.json").write_text(
        result.model_dump_json(indent=2, exclude={"trial_results"}), encoding="utf-8"
    )
    for trial in trials:
        trial_dir = job_dir / trial.trial_name
        trial_dir.mkdir()
        (trial_dir / "config.json").write_text(trial.config.model_dump_json(), encoding="utf-8")
        (trial_dir / "lock.json").write_text(
            _trial_lock(trial.task_name, trial.config).model_dump_json(), encoding="utf-8"
        )
        (trial_dir / "result.json").write_text(trial.model_dump_json(indent=2), encoding="utf-8")


def _manifest(
    job_name: str,
    *entries: tuple[str, int, str],
    agent_config_digest: str = _AGENT_CONFIG_DIGEST,
    trial_lock_digest: str | None = None,
) -> HarborTrialManifest:
    return HarborTrialManifest(
        schema_version=2,
        job_name=job_name,
        identity=_RUN_IDENTITY.model_copy(update={"run_config_digest": agent_config_digest}),
        agent_config_digest=agent_config_digest,
        entries=[
            _manifest_entry(
                task_name,
                attempt,
                trial_name,
                trial_lock_digest=trial_lock_digest,
            )
            for task_name, attempt, trial_name in entries
        ],
    )


def _manifest_entry(
    task_name: str,
    attempt: int,
    trial_name: str,
    *,
    trial_lock_digest: str | None,
) -> HarborTrialManifestEntry:
    config = TrialConfig(
        task=TaskConfig(path=Path(task_name), source=_TASK_SOURCE),
        trial_name=trial_name,
        agent=_agent_config(),
    )
    resolved_trial_lock_digest = (
        trial_lock_digest
        if trial_lock_digest is not None
        else harbor_trial_lock_digest(_trial_lock(task_name, config))
    )
    return HarborTrialManifestEntry(
        cell=BenchmarkCell(
            task_key=_task_key(task_name),
            task_name=task_name,
            attempt=attempt,
            config_digest=resolved_trial_lock_digest,
        ),
        trial_name=trial_name,
        task_identity=task_name,
        runtime_task_checksum=_RUNTIME_TASK_CHECKSUM,
        task_checksum=_TASK_CHECKSUM,
        task_source=_TASK_SOURCE,
        task_instruction=f"Instruction for {task_name}.",
        trial_lock_digest=resolved_trial_lock_digest,
    )


def test_load_uses_exact_manifest_and_preserves_rewards_usage_and_missing_cells(
    tmp_path: Path,
) -> None:
    scored = _trial(
        tmp_path,
        "scored",
        rewards={"reward": 0, "partial_credit": 0.5, "tests_passed": 4},
    )
    failed = _trial(tmp_path, "failed", exception_type="EnvironmentStartTimeoutError")
    job_dir = tmp_path / "job"
    _write_job(job_dir, [scored, failed], expected=3)
    manifest = _manifest(
        "job",
        ("scored", 1, scored.trial_name),
        ("failed", 1, failed.trial_name),
        ("missing", 1, "missing__harbor"),
    )
    for entry in manifest.entries:
        if entry.cell.task_name == "scored":
            entry.task_instruction = "Score this task."

    loaded = load_harbor_job_result(job_dir, manifest)
    result = loaded.result

    assert result.expected_trials == 3
    assert result.identity.run_config_digest == manifest.agent_config_digest
    assert result.n_scored == 1
    assert result.n_infrastructure_errors == 1
    assert result.n_incomplete == 1
    trials = {trial.cell.task_name: trial for trial in result.trials}
    assert scored.task_checksum == _RUNTIME_TASK_CHECKSUM
    assert trials["scored"].task_checksum == _TASK_CHECKSUM
    assert trials["scored"].task_checksum != scored.task_checksum
    assert trials["scored"].task_instruction == "Score this task."
    assert trials["scored"].task_environment_digest == _TASK_ENVIRONMENT_DIGEST
    assert trials["scored"].runner_environment_attestation == _RUNNER.attestation.evidence
    assert trials["scored"].runner_lease_receipts == [_runner_lease_receipt(scored.trial_name)]
    assert trials["failed"].task_environment_digest == _TASK_ENVIRONMENT_DIGEST
    assert trials["missing"].task_environment_digest is None
    manifest_config_digests = {
        entry.cell.task_name: entry.trial_lock_digest for entry in manifest.entries
    }
    assert {
        name: trial.cell.config_digest for name, trial in trials.items()
    } == manifest_config_digests
    assert trials["scored"].rewards == {
        "reward": 0,
        "partial_credit": 0.5,
        "tests_passed": 4,
    }
    assert trials["failed"].error is not None
    assert trials["failed"].error.kind is BenchmarkFailureKind.ENVIRONMENT
    assert trials["missing"].cell == BenchmarkCell(
        task_key=_task_key("missing"),
        task_name="missing",
        attempt=1,
        config_digest=next(
            entry.trial_lock_digest
            for entry in manifest.entries
            if entry.cell.task_name == "missing"
        ),
    )
    assert trials["missing"].source == _TASK_SOURCE
    assert trials["missing"].status is BenchmarkTrialStatus.INCOMPLETE
    assert result.usage.calls == 2
    assert result.usage.calls_status is BenchmarkUsageStatus.LOWER_BOUND
    assert result.usage.input_tokens == 20
    assert result.usage.input_tokens_status is BenchmarkUsageStatus.LOWER_BOUND
    assert result.usage.cache_tokens == 4
    assert result.usage.cache_tokens_status is BenchmarkUsageStatus.LOWER_BOUND
    assert result.usage.output_tokens == 8
    assert result.usage.output_tokens_status is BenchmarkUsageStatus.LOWER_BOUND
    assert result.usage.cost_usd == 0.1
    assert result.usage.cost_usd_status is BenchmarkUsageStatus.LOWER_BOUND

    locators = {locator.cell.task_name: locator for locator in loaded.locators}
    scored_locator = locators["scored"]
    assert scored_locator.result_path == Path(scored.trial_name) / "result.json"
    assert scored_locator.result_path is not None
    assert scored_locator.artifacts_dir == Path(scored.trial_name) / "artifacts"
    assert (
        loaded.resolve_path(scored_locator.result_path)
        == (job_dir / scored.trial_name / "result.json").resolve()
    )
    assert locators["missing"].result_path is None
    assert str(tmp_path) not in result.model_dump_json()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("region", "us-west-2"),
        ("deployment", "deployment-a"),
        ("api_version", "2026-01-01-preview"),
        ("turn_timeout_s", 301.0),
        ("n_concurrent", 2),
    ],
)
def test_agent_config_digest_component_binds_backend_and_turn_settings(
    field: str,
    value: str | float,
) -> None:
    baseline = _agent_config()
    changed = baseline.model_copy(deep=True)
    if field == "n_concurrent":
        changed.n_concurrent = int(value)
    elif field == "turn_timeout_s":
        changed.kwargs[field] = value
    else:
        provider_config = dict(changed.kwargs["provider_config"])
        provider_config[field] = value
        changed.kwargs["provider_config"] = provider_config

    assert harbor_agent_config_digest(changed) != harbor_agent_config_digest(baseline)


def test_canonical_json_is_independent_of_job_root(tmp_path: Path) -> None:
    manifests: list[HarborTrialManifest] = []
    results: list[str] = []
    for backend in ("docker", "e2b"):
        root = tmp_path / backend
        trial = _trial(root, "scored", rewards={"reward": 1})
        job_dir = root / "job"
        _write_job(job_dir, [trial], expected=1)
        manifest = _manifest("job", ("scored", 1, trial.trial_name))
        manifests.append(manifest)
        results.append(load_harbor_job_result(job_dir, manifest).result.model_dump_json())

    assert manifests[0] == manifests[1]
    assert results[0] == results[1]


def test_usage_total_is_reported_only_when_every_cell_is_metered(tmp_path: Path) -> None:
    first = _trial(tmp_path, "first", rewards={"reward": 1})
    second = _trial(tmp_path, "second", rewards={"reward": 0})
    job_dir = tmp_path / "job"
    _write_job(job_dir, [first, second], expected=2)

    result = load_harbor_job_result(
        job_dir,
        _manifest(
            "job",
            ("first", 1, first.trial_name),
            ("second", 1, second.trial_name),
        ),
    ).result

    assert result.usage.calls == 2
    assert result.usage.calls_status is BenchmarkUsageStatus.EXACT
    assert result.usage.input_tokens == 20
    assert result.usage.input_tokens_status is BenchmarkUsageStatus.EXACT
    assert result.usage.cache_tokens == 4
    assert result.usage.cache_tokens_status is BenchmarkUsageStatus.EXACT
    assert result.usage.output_tokens == 8
    assert result.usage.output_tokens_status is BenchmarkUsageStatus.EXACT
    assert result.usage.cost_usd == 0.1
    assert result.usage.cost_usd_status is BenchmarkUsageStatus.EXACT


@pytest.mark.parametrize(
    "exception_type",
    [
        "WmhPiProviderDeadlineError",
        "WmhPiProviderError",
        "WmhPiProviderReceiptError",
        "AgentTimeoutError",
        "CancelledError",
    ],
)
def test_interrupted_usage_is_a_known_lower_bound_not_an_exact_total(
    tmp_path: Path,
    exception_type: str,
) -> None:
    trial = _trial(tmp_path, "task", exception_type=exception_type)
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    result = load_harbor_job_result(
        job_dir,
        _manifest("job", ("task", 1, trial.trial_name)),
    ).result

    assert result.trials[0].usage.input_tokens == 10
    assert result.trials[0].usage.calls == 1
    assert result.trials[0].usage.calls_status is BenchmarkUsageStatus.EXACT
    assert result.trials[0].usage.input_tokens_status is BenchmarkUsageStatus.LOWER_BOUND
    assert result.trials[0].usage.output_tokens == 4
    assert result.trials[0].usage.output_tokens_status is BenchmarkUsageStatus.LOWER_BOUND
    assert result.usage.input_tokens == 10
    assert result.usage.input_tokens_status is BenchmarkUsageStatus.LOWER_BOUND
    assert result.usage.output_tokens == 4
    assert result.usage.output_tokens_status is BenchmarkUsageStatus.LOWER_BOUND


def test_provider_failure_attribution_is_retained_without_raw_error_text(tmp_path: Path) -> None:
    secret = "provider-secret-sentinel"
    trial = _trial(
        tmp_path,
        "task",
        exception_type="WmhPiProviderError",
        candidate_metadata={
            "run_health": "infrastructure_failure",
            "provider_failure_stage": "dispatch",
            "provider_failure_reason": "auth",
        },
    )
    assert trial.exception_info is not None
    trial.exception_info.exception_message = secret
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    result = load_harbor_job_result(
        job_dir,
        _manifest("job", ("task", 1, trial.trial_name)),
    ).result.trials[0]

    assert result.provider_failure_stage is ProviderFailureStage.DISPATCH
    assert result.provider_failure_reason is ProviderFailureReason.AUTH
    assert secret not in result.model_dump_json()


def test_response_translation_failure_is_retained_as_fixed_evidence(tmp_path: Path) -> None:
    trial = _trial(
        tmp_path,
        "task",
        exception_type="WmhPiProviderError",
        candidate_metadata={
            "run_health": "infrastructure_failure",
            "provider_failure_stage": "response_translation",
            "provider_failure_reason": "unknown",
            "provider_response_translation_failure": "tool_use_shape",
        },
    )
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    result = load_harbor_job_result(
        job_dir,
        _manifest("job", ("task", 1, trial.trial_name)),
    ).result.trials[0]

    assert result.provider_failure_stage is ProviderFailureStage.RESPONSE_TRANSLATION
    assert result.provider_failure_reason is ProviderFailureReason.UNKNOWN
    assert result.provider_response_translation_failure is ResponseTranslationFailure.TOOL_USE_SHAPE


def test_unclassified_response_translation_failure_remains_valid_legacy_evidence(
    tmp_path: Path,
) -> None:
    trial = _trial(
        tmp_path,
        "task",
        exception_type="WmhPiProviderError",
        candidate_metadata={
            "run_health": "infrastructure_failure",
            "provider_failure_stage": "response_translation",
            "provider_failure_reason": "unknown",
        },
    )
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    result = load_harbor_job_result(
        job_dir,
        _manifest("job", ("task", 1, trial.trial_name)),
    ).result.trials[0]

    assert result.provider_failure_stage is ProviderFailureStage.RESPONSE_TRANSLATION
    assert result.provider_response_translation_failure is None


@pytest.mark.parametrize(
    "candidate_metadata",
    [
        {"provider_failure_stage": "dispatch"},
        {"provider_failure_reason": "auth"},
        {"provider_failure_stage": "private-stage", "provider_failure_reason": "auth"},
        {"provider_failure_stage": "dispatch", "provider_failure_reason": "private-reason"},
        {
            "provider_failure_stage": "dispatch",
            "provider_failure_reason": "auth",
            "provider_response_translation_failure": "tool_use_shape",
        },
        {
            "provider_failure_stage": "response_translation",
            "provider_failure_reason": "unknown",
            "provider_response_translation_failure": "private-shape",
        },
        {"provider_response_translation_failure": "tool_use_shape"},
    ],
)
def test_unbounded_provider_failure_metadata_is_rejected(
    tmp_path: Path,
    candidate_metadata: dict[str, object],
) -> None:
    trial = _trial(
        tmp_path,
        "task",
        exception_type="WmhPiProviderError",
        candidate_metadata=candidate_metadata,
    )
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    with pytest.raises(ValueError, match="provider failure attribution"):
        load_harbor_job_result(job_dir, _manifest("job", ("task", 1, trial.trial_name)))


@pytest.mark.parametrize(
    ("exception_type", "status", "kind"),
    [
        ("AgentTimeoutError", BenchmarkTrialStatus.TASK_TIMEOUT, BenchmarkFailureKind.TASK_TIMEOUT),
        (
            "EnvironmentStartTimeoutError",
            BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
            BenchmarkFailureKind.ENVIRONMENT,
        ),
        (
            "ApiRateLimitError",
            BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
            BenchmarkFailureKind.PROVIDER,
        ),
        (
            "VerifierTimeoutError",
            BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
            BenchmarkFailureKind.VERIFIER,
        ),
        (
            "WmhPiProviderError",
            BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
            BenchmarkFailureKind.PROVIDER,
        ),
        (
            "WmhPiProviderDeadlineError",
            BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
            BenchmarkFailureKind.PROVIDER,
        ),
        (
            "WmhPiProviderReceiptError",
            BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
            BenchmarkFailureKind.PROVIDER,
        ),
        (
            "WmhPiEnvironmentError",
            BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
            BenchmarkFailureKind.ENVIRONMENT,
        ),
        (
            "WmhPiRunnerError",
            BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
            BenchmarkFailureKind.ENVIRONMENT,
        ),
        (
            "WmhPiCleanupError",
            BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
            BenchmarkFailureKind.ENVIRONMENT,
        ),
        ("CancelledError", BenchmarkTrialStatus.CANCELLED, BenchmarkFailureKind.CANCELLED),
        (
            "RuntimeError",
            BenchmarkTrialStatus.UNCLASSIFIED_ERROR,
            BenchmarkFailureKind.UNCLASSIFIED,
        ),
        (
            "CustomAgentBoom",
            BenchmarkTrialStatus.UNCLASSIFIED_ERROR,
            BenchmarkFailureKind.UNCLASSIFIED,
        ),
    ],
)
def test_exception_classification_is_not_blanket_infrastructure(
    tmp_path: Path,
    exception_type: str,
    status: BenchmarkTrialStatus,
    kind: BenchmarkFailureKind,
) -> None:
    trial = _trial(tmp_path, "task", exception_type=exception_type)
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    result = load_harbor_job_result(
        job_dir, _manifest("job", ("task", 1, trial.trial_name))
    ).result.trials[0]

    assert result.status is status
    assert result.error is not None
    assert result.error.kind is kind


@pytest.mark.parametrize(
    ("exception_type", "status", "kind"),
    [
        ("AgentTimeoutError", BenchmarkTrialStatus.SCORED, BenchmarkFailureKind.TASK_TIMEOUT),
        (
            "WmhPiProviderError",
            BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
            BenchmarkFailureKind.PROVIDER,
        ),
        (
            "WmhPiProviderDeadlineError",
            BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
            BenchmarkFailureKind.PROVIDER,
        ),
        (
            "WmhPiProviderReceiptError",
            BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
            BenchmarkFailureKind.PROVIDER,
        ),
        (
            "WmhPiEnvironmentError",
            BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
            BenchmarkFailureKind.ENVIRONMENT,
        ),
        (
            "VerifierTimeoutError",
            BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
            BenchmarkFailureKind.VERIFIER,
        ),
    ],
)
def test_verifier_rewards_after_exception_preserve_harbor_mixed_result_semantics(
    tmp_path: Path,
    exception_type: str,
    status: BenchmarkTrialStatus,
    kind: BenchmarkFailureKind,
) -> None:
    trial = _trial(
        tmp_path,
        "task",
        rewards={"reward": 0, "tests_passed": 2},
        exception_type=exception_type,
    )
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    run_result = load_harbor_job_result(
        job_dir, _manifest("job", ("task", 1, trial.trial_name))
    ).result
    result = run_result.trials[0]

    assert result.status is status
    if status is BenchmarkTrialStatus.SCORED:
        assert result.rewards == {"reward": 0, "tests_passed": 2}
        assert result.run_health is BenchmarkRunHealth.VALID
        assert run_result.mean_reward("reward") == 0.0
    else:
        assert result.rewards is None
        assert result.run_health is BenchmarkRunHealth.RETRY_REQUIRED
    assert result.error is not None
    assert result.error.kind is kind


def test_pre_agent_environment_failure_does_not_require_agent_result_metadata(
    tmp_path: Path,
) -> None:
    trial = _trial(tmp_path, "task", exception_type="EnvironmentStartTimeoutError")
    trial.agent_result = None
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    result = load_harbor_job_result(
        job_dir, _manifest("job", ("task", 1, trial.trial_name))
    ).result.trials[0]

    assert result.status is BenchmarkTrialStatus.INFRASTRUCTURE_ERROR
    assert result.error is not None
    assert result.error.kind is BenchmarkFailureKind.ENVIRONMENT


@pytest.mark.parametrize("exception_type", ["EnvironmentStartTimeoutError", "VerifierTimeoutError"])
def test_canonical_error_redacts_backend_messages_and_tracebacks(
    tmp_path: Path,
    exception_type: str,
) -> None:
    secret = "backend-secret-sentinel"
    trial = _trial(tmp_path, "task", exception_type=exception_type)
    assert trial.exception_info is not None
    trial.exception_info.exception_message = f"signed URL contained {secret}"
    trial.exception_info.exception_traceback = f"header={secret}"
    job_dir = tmp_path / exception_type / "job"
    _write_job(job_dir, [trial], expected=1)

    result = load_harbor_job_result(
        job_dir, _manifest("job", ("task", 1, trial.trial_name))
    ).result.trials[0]

    assert result.error is not None
    assert secret not in result.model_dump_json()
    assert result.error.traceback is None


def test_unverified_result_remains_incomplete(tmp_path: Path) -> None:
    trial = _trial(tmp_path, "unverified")
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    result = load_harbor_job_result(
        job_dir, _manifest("job", ("unverified", 1, trial.trial_name))
    ).result

    assert result.trials[0].status is BenchmarkTrialStatus.INCOMPLETE


def test_scored_candidate_failure_is_retained_without_arbitrary_metadata(tmp_path: Path) -> None:
    secret = "arbitrary-metadata-sentinel"
    trial = _trial(
        tmp_path,
        "task",
        rewards={"reward": 0.5},
        candidate_metadata={
            "candidate_failure": True,
            "candidate_failure_stage": "turn",
            "candidate_failure_reason": "timeout",
            "untrusted_extra": secret,
        },
    )
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    result = load_harbor_job_result(
        job_dir, _manifest("job", ("task", 1, trial.trial_name))
    ).result.trials[0]

    assert result.status is BenchmarkTrialStatus.SCORED
    assert result.candidate_outcome.status is BenchmarkCandidateStatus.FAILED
    assert result.candidate_outcome.stage is BenchmarkCandidateStage.EXECUTION
    assert result.candidate_outcome.failure_reason is BenchmarkCandidateFailureReason.TIMEOUT
    assert result.candidate_outcome.terminal_reason is None
    assert secret not in result.model_dump_json()


def test_candidate_request_rejection_reason_is_preserved_as_a_valid_zero_signal(
    tmp_path: Path,
) -> None:
    trial = _trial(
        tmp_path,
        "task",
        rewards={"reward": 0},
        candidate_metadata={
            "candidate_failure": True,
            "candidate_failure_stage": "turn",
            "candidate_failure_reason": "invalid_request",
            "run_health": "valid",
        },
    )
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    result = load_harbor_job_result(
        job_dir, _manifest("job", ("task", 1, trial.trial_name))
    ).result.trials[0]

    assert result.status is BenchmarkTrialStatus.SCORED
    assert result.run_health is BenchmarkRunHealth.VALID
    assert (
        result.candidate_outcome.failure_reason is BenchmarkCandidateFailureReason.INVALID_REQUEST
    )


@pytest.mark.parametrize("exception_type", ["WmhPiEnvironmentError", "VerifierTimeoutError"])
def test_candidate_damaged_task_environment_is_a_scoreable_zero_without_rewards(
    tmp_path: Path,
    exception_type: str,
) -> None:
    trial = _trial(
        tmp_path,
        "task",
        exception_type=exception_type,
        candidate_metadata={
            "candidate_failure": True,
            "candidate_failure_stage": "turn",
            "candidate_failure_reason": "resource_limit",
            "run_health": "candidate_damaged",
        },
    )
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    result = load_harbor_job_result(job_dir, _manifest("job", ("task", 1, trial.trial_name))).result

    assert result.trials[0].status is BenchmarkTrialStatus.CANDIDATE_FAILURE
    assert result.trials[0].run_health is BenchmarkRunHealth.CANDIDATE_DAMAGED
    assert result.n_candidate_failure_zeroes == 1
    assert result.n_infrastructure_errors == 0
    assert result.mean_reward("reward") == 0.0


def test_candidate_damage_retains_later_verifier_reward_for_diagnostics(tmp_path: Path) -> None:
    trial = _trial(
        tmp_path,
        "task",
        rewards={"reward": 1},
        exception_type="WmhPiEnvironmentError",
        candidate_metadata={
            "candidate_failure": True,
            "candidate_failure_stage": "turn",
            "candidate_failure_reason": "resource_limit",
            "run_health": "candidate_damaged",
        },
    )
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    run_result = load_harbor_job_result(
        job_dir, _manifest("job", ("task", 1, trial.trial_name))
    ).result
    result = run_result.trials[0]

    assert result.status is BenchmarkTrialStatus.SCORED
    assert result.rewards == {"reward": 1}
    assert result.run_health is BenchmarkRunHealth.CANDIDATE_DAMAGED
    assert result.error is not None
    assert result.error.kind is BenchmarkFailureKind.ENVIRONMENT
    assert run_result.mean_reward("reward") == 0.0


def test_ambiguous_environment_loss_is_explicitly_retryable(tmp_path: Path) -> None:
    trial = _trial(
        tmp_path,
        "task",
        exception_type="WmhPiEnvironmentConfirmationRequiredError",
        candidate_metadata={
            "run_health": "ambiguous",
        },
    )
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    result = load_harbor_job_result(job_dir, _manifest("job", ("task", 1, trial.trial_name))).result

    assert result.trials[0].status is BenchmarkTrialStatus.INFRASTRUCTURE_ERROR
    assert result.trials[0].error is not None
    assert result.trials[0].error.kind is BenchmarkFailureKind.ENVIRONMENT_CONFIRMATION_REQUIRED
    assert result.trials[0].run_health is BenchmarkRunHealth.RETRY_REQUIRED
    with pytest.raises(ValueError, match="run health is not valid"):
        result.mean_reward("reward")


@pytest.mark.parametrize(
    "candidate_metadata",
    [
        {"candidate_failure": "yes"},
        {"candidate_failure": False, "candidate_failure_stage": "turn"},
        {"candidate_failure": True, "terminal_reason": "completed"},
        {"candidate_failure": True, "candidate_failure_stage": "unknown-stage"},
        {
            "candidate_failure": True,
            "candidate_failure_stage": "turn",
            "candidate_failure_reason": "unknown-reason",
        },
        {"candidate_failure": False, "terminal_reason": "unknown-reason"},
        {"candidate_failure_stage": "turn"},
        {"candidate_failure": False, "run_health": "unknown-value"},
    ],
)
def test_malformed_candidate_outcome_metadata_is_rejected(
    tmp_path: Path,
    candidate_metadata: dict[str, object],
) -> None:
    trial = _trial(
        tmp_path,
        "task",
        rewards={"reward": 1},
        candidate_metadata=candidate_metadata,
    )
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    with pytest.raises(ValueError, match="candidate outcome metadata"):
        load_harbor_job_result(job_dir, _manifest("job", ("task", 1, trial.trial_name)))


@pytest.mark.parametrize(
    ("candidate_metadata", "message"),
    [
        ({"task_environment_digest": None}, "valid task environment digest"),
        (
            {"task_environment_digest": "sha256:" + "f" * 64},
            "does not match its digest",
        ),
        (
            {
                "task_environment_attestation": {
                    **_TASK_ENVIRONMENT_ATTESTATION,
                    "backend": "e2b",
                }
            },
            "wrong backend or schema",
        ),
    ],
)
def test_task_environment_attestation_must_be_complete_bound_and_backend_correct(
    tmp_path: Path,
    candidate_metadata: dict[str, object],
    message: str,
) -> None:
    trial = _trial(
        tmp_path,
        "task",
        rewards={"reward": 1},
        candidate_metadata=candidate_metadata,
    )
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    with pytest.raises(ValueError, match=message):
        load_harbor_job_result(job_dir, _manifest("job", ("task", 1, trial.trial_name)))


def _write_e2b_job(
    tmp_path: Path,
    *,
    receipt: dict[str, object] | None,
    attestation: dict[str, object] | None = None,
) -> tuple[Path, HarborTrialManifest, TrialResult]:
    effective_attestation = attestation or _E2B_TASK_ENVIRONMENT_ATTESTATION
    effective_digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                effective_attestation,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    trial = _trial(tmp_path, "task", rewards={"reward": 1})
    trial.config.environment.type = EnvironmentType.E2B
    assert trial.agent_result is not None
    assert trial.agent_result.metadata is not None
    trial.agent_result.metadata.update(
        {
            "task_environment_digest": effective_digest,
            "task_environment_attestation": effective_attestation,
        }
    )
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)
    if receipt is not None:
        (job_dir / trial.trial_name / TASK_E2B_LEASE_FILE).write_text(
            json.dumps(receipt),
            encoding="utf-8",
        )
    lock_digest = harbor_trial_lock_digest(_trial_lock(trial.task_name, trial.config))
    manifest = _manifest(
        "job",
        ("task", 1, trial.trial_name),
        trial_lock_digest=lock_digest,
    )
    manifest = manifest.model_copy(
        update={
            "identity": manifest.identity.model_copy(
                update={"task_environment": BenchmarkTaskEnvironment.E2B}
            )
        }
    )
    return job_dir, manifest, trial


def test_e2b_task_environment_retains_full_attestation_and_cleanup_receipt(
    tmp_path: Path,
) -> None:
    receipt = _task_environment_lease_receipt("task__harbor")
    job_dir, manifest, _trial_result = _write_e2b_job(tmp_path, receipt=receipt)

    loaded = load_harbor_job_result(job_dir, manifest)

    trial = loaded.result.trials[0]
    assert trial.task_environment_digest == _E2B_TASK_ENVIRONMENT_DIGEST
    assert trial.task_environment_attestation == _E2B_TASK_ENVIRONMENT_ATTESTATION
    assert trial.task_environment_lease_receipt == receipt


def test_e2b_task_environment_rejects_ambiguous_storage_capacity_scope(
    tmp_path: Path,
) -> None:
    receipt = _task_environment_lease_receipt("task__harbor")
    attestation: dict[str, object] = dict(_E2B_TASK_ENVIRONMENT_ATTESTATION)
    attestation.pop("storage_capacity_scope")
    job_dir, manifest, _trial_result = _write_e2b_job(
        tmp_path,
        receipt=receipt,
        attestation=attestation,
    )

    with pytest.raises(ValueError, match="storage capacity scope"):
        load_harbor_job_result(job_dir, manifest)


@pytest.mark.parametrize(
    ("receipt", "message"),
    [
        (None, "omits the task environment cleanup receipt"),
        (
            {
                **_task_environment_lease_receipt("task__harbor"),
                "state": "active",
                "retired_at": None,
            },
            "does not prove terminal cleanup",
        ),
        (
            {
                **_task_environment_lease_receipt("task__harbor"),
                "resource_id": None,
            },
            "does not prove terminal cleanup",
        ),
        (
            {
                **_task_environment_lease_receipt("task__harbor"),
                "config_digest": "sha256:" + "f" * 64,
            },
            "does not match its launch configuration",
        ),
        (
            {
                **_task_environment_lease_receipt("task__harbor"),
                "owner_id": "sha256:" + "f" * 64,
            },
            "wrong trial owner",
        ),
        (
            {
                **_task_environment_lease_receipt("task__harbor"),
                "backend": "local",
            },
            "wrong backend",
        ),
    ],
)
def test_valid_e2b_task_requires_bound_terminal_cleanup_receipt(
    tmp_path: Path,
    receipt: dict[str, object] | None,
    message: str,
) -> None:
    job_dir, manifest, _trial_result = _write_e2b_job(tmp_path, receipt=receipt)

    with pytest.raises(ValueError, match=message):
        load_harbor_job_result(job_dir, manifest)


@pytest.mark.parametrize(
    ("candidate_metadata", "message"),
    [
        (
            {
                "runner_environment_digest": None,
                "runner_environment_attestation": None,
            },
            "valid runner environment digest",
        ),
        (
            {"runner_environment_digest": "sha256:" + "f" * 64},
            "does not match the frozen run",
        ),
        (
            {
                "runner_environment_attestation": {
                    **_RUNNER.attestation.evidence,
                    "backend": "e2b",
                }
            },
            "does not match its digest",
        ),
        (
            {"runner_config_digest": "sha256:" + "f" * 64},
            "runner configuration digest",
        ),
    ],
)
def test_runner_attestation_must_be_complete_and_bound_to_the_frozen_spec(
    tmp_path: Path,
    candidate_metadata: dict[str, object],
    message: str,
) -> None:
    trial = _trial(
        tmp_path,
        "task",
        rewards={"reward": 1},
        candidate_metadata=candidate_metadata,
    )
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    with pytest.raises(ValueError, match=message):
        load_harbor_job_result(job_dir, _manifest("job", ("task", 1, trial.trial_name)))


@pytest.mark.parametrize(
    ("receipt", "message"),
    [
        (None, "omits the runner lease cleanup receipt"),
        (
            {**_runner_lease_receipt("task__harbor"), "state": "active", "retired_at": None},
            "terminal",
        ),
        (
            {**_runner_lease_receipt("task__harbor"), "resource_id": None},
            "terminal",
        ),
        (
            {**_runner_lease_receipt("task__harbor"), "resource_id": ""},
            "invalid",
        ),
        (
            {
                **_runner_lease_receipt("task__harbor"),
                "config_digest": "sha256:" + "f" * 64,
            },
            "frozen configuration",
        ),
        ({**_runner_lease_receipt("task__harbor"), "backend": "e2b"}, "wrong backend"),
        (
            {**_runner_lease_receipt("task__harbor"), "owner_id": "sha256:" + "f" * 64},
            "trial owner",
        ),
    ],
)
def test_valid_runner_requires_a_bound_terminal_lease_receipt(
    tmp_path: Path,
    receipt: dict[str, object] | None,
    message: str,
) -> None:
    trial = _trial(
        tmp_path,
        "task",
        rewards={"reward": 1},
        candidate_metadata={"runner_lease_receipt": receipt},
    )
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    with pytest.raises(ValueError, match=message):
        load_harbor_job_result(job_dir, _manifest("job", ("task", 1, trial.trial_name)))


def test_inconsistent_step_candidate_outcomes_are_rejected(tmp_path: Path) -> None:
    trial = _trial(tmp_path, "task", rewards={"reward": 1})
    identity = {
        "harness_hash": _HARNESS.execution_hash,
        "runner_config_digest": _RUNNER.config_digest,
        "runner_environment_digest": _RUNNER.attestation.digest,
        "runner_environment_attestation": _RUNNER.attestation.evidence,
        "task_environment_digest": _TASK_ENVIRONMENT_DIGEST,
        "task_environment_attestation": _TASK_ENVIRONMENT_ATTESTATION,
    }
    trial.agent_result = None
    trial.step_results = [
        StepResult(
            step_name="first",
            agent_result=AgentContext(
                metadata={
                    **identity,
                    "candidate_failure": False,
                    "terminal_reason": "completed",
                }
            ),
        ),
        StepResult(
            step_name="second",
            agent_result=AgentContext(
                metadata={
                    **identity,
                    "candidate_failure": True,
                    "candidate_failure_stage": "materialization",
                    "candidate_failure_reason": "runtime_error",
                }
            ),
        ),
    ]
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    with pytest.raises(ValueError, match="inconsistent candidate outcome"):
        load_harbor_job_result(job_dir, _manifest("job", ("task", 1, trial.trial_name)))


def test_inconsistent_step_model_call_counts_are_rejected(tmp_path: Path) -> None:
    trial = _trial(tmp_path, "task", rewards={"reward": 1})
    identity = {
        "harness_hash": _HARNESS.execution_hash,
        "runner_config_digest": _RUNNER.config_digest,
        "runner_environment_digest": _RUNNER.attestation.digest,
        "runner_environment_attestation": _RUNNER.attestation.evidence,
        "runner_lease_receipt": _runner_lease_receipt(trial.trial_name),
        "task_environment_digest": _TASK_ENVIRONMENT_DIGEST,
        "task_environment_attestation": _TASK_ENVIRONMENT_ATTESTATION,
        "candidate_failure": False,
        "terminal_reason": "completed",
    }
    trial.agent_result = None
    trial.step_results = [
        StepResult(
            step_name="first",
            agent_result=AgentContext(metadata={**identity, "model_calls": 1}),
        ),
        StepResult(
            step_name="second",
            agent_result=AgentContext(metadata={**identity, "model_calls": 2}),
        ),
    ]
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    with pytest.raises(ValueError, match="inconsistent model call metadata"):
        load_harbor_job_result(job_dir, _manifest("job", ("task", 1, trial.trial_name)))


@pytest.mark.parametrize("model_calls", [True, -1, 1.5, "1"])
def test_invalid_model_call_count_is_rejected(tmp_path: Path, model_calls: object) -> None:
    trial = _trial(
        tmp_path,
        "task",
        rewards={"reward": 1},
        candidate_metadata={"model_calls": model_calls},
    )
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    with pytest.raises(ValueError, match="model_calls must be a non-negative integer"):
        load_harbor_job_result(job_dir, _manifest("job", ("task", 1, trial.trial_name)))


def test_completed_candidate_terminal_reason_is_normalized(tmp_path: Path) -> None:
    trial = _trial(
        tmp_path,
        "task",
        rewards={"reward": 1},
        candidate_metadata={
            "candidate_failure": False,
            "terminal_reason": "turn_limit",
        },
    )
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    result = load_harbor_job_result(
        job_dir, _manifest("job", ("task", 1, trial.trial_name))
    ).result.trials[0]

    assert result.candidate_outcome.status is BenchmarkCandidateStatus.COMPLETED
    assert result.candidate_outcome.terminal_reason is (
        BenchmarkCandidateTerminalReason.LIMIT_REACHED
    )


def test_invalid_trial_result_is_malformed_infrastructure(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    _write_job(job_dir, [], expected=1)
    trial_dir = job_dir / "broken__harbor"
    trial_dir.mkdir()
    (trial_dir / "result.json").write_text("{not json", encoding="utf-8")

    result = load_harbor_job_result(
        job_dir, _manifest("job", ("broken", 1, "broken__harbor"))
    ).result

    assert result.n_infrastructure_errors == 1
    assert result.trials[0].error is not None
    assert result.trials[0].error.kind is BenchmarkFailureKind.MALFORMED_RESULT


def test_extra_trial_directory_is_rejected(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    _write_job(job_dir, [], expected=0)
    (job_dir / "unexpected").mkdir()

    with pytest.raises(ValueError, match="unexpected trial director"):
        load_harbor_job_result(job_dir, _manifest("job"))


def test_wrong_task_for_manifest_cell_is_rejected(tmp_path: Path) -> None:
    trial = _trial(tmp_path, "actual", trial_name="planned__harbor", rewards={"reward": 1})
    trial.task_name = "planned"
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    with pytest.raises(ValueError, match="expected task identity 'planned'.*found 'actual'"):
        load_harbor_job_result(job_dir, _manifest("job", ("planned", 1, "planned__harbor")))


@pytest.mark.parametrize("tamper", ["name", "digest", "source"])
def test_completed_trial_lock_must_match_canonical_task_domain(
    tmp_path: Path,
    tamper: str,
) -> None:
    trial = _trial(tmp_path, "task", rewards={"reward": 1})
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)
    lock_path = job_dir / trial.trial_name / "lock.json"
    lock = TrialLock.model_validate_json(lock_path.read_text(encoding="utf-8"))
    if tamper == "name":
        lock.task.name = "different-task"
    elif tamper == "digest":
        lock.task.digest = "sha256:" + "c" * 64
    else:
        lock.task.source = "different-source"
    lock_path.write_text(lock.model_dump_json(indent=2), encoding="utf-8")
    manifest = _manifest("job", ("task", 1, trial.trial_name))
    manifest.entries[0].trial_lock_digest = harbor_trial_lock_digest(lock)
    manifest.entries[0].cell = manifest.entries[0].cell.model_copy(
        update={"config_digest": manifest.entries[0].trial_lock_digest}
    )

    expected = {
        "name": "expected task identity",
        "digest": "canonical task checksum",
        "source": "expected task source",
    }[tamper]
    with pytest.raises(ValueError, match=expected):
        load_harbor_job_result(job_dir, manifest)


@pytest.mark.parametrize("completed", [True, False], ids=["completed", "incomplete"])
@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("job_id", "job_id does not match its root job result"),
        ("trials_dir", "trials_dir does not resolve to its root job"),
    ],
)
def test_trial_config_cannot_be_grafted_from_another_job(
    tmp_path: Path,
    completed: bool,
    mismatch: str,
    message: str,
) -> None:
    trial = _trial(tmp_path, "task", rewards={"reward": 1})
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)
    trial_dir = job_dir / trial.trial_name

    if mismatch == "job_id":
        trial.config.job_id = uuid4()
    else:
        foreign_job_dir = tmp_path / "foreign-job"
        foreign_job_dir.mkdir()
        trial.config.trials_dir = foreign_job_dir

    if completed:
        (trial_dir / "result.json").write_text(trial.model_dump_json(indent=2), encoding="utf-8")
    else:
        (trial_dir / "result.json").unlink()
        (trial_dir / "config.json").write_text(
            trial.config.model_dump_json(indent=2), encoding="utf-8"
        )

    with pytest.raises(ValueError, match=message):
        load_harbor_job_result(job_dir, _manifest("job", ("task", 1, trial.trial_name)))


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("task_checksum", "expected Harbor runtime task checksum"),
        ("agent_name", "expected agent"),
        ("agent_version", "expected agent"),
        ("provider", "expected provider/model"),
        ("model_name", "expected provider/model"),
        ("task_environment", "expected trial lock digest"),
        ("agent_import_path", "expected trial lock digest"),
        ("agent_config_model", "expected trial lock digest"),
        ("agent_kwargs", "expected trial lock digest"),
        ("candidate_hash", "expected candidate hash"),
        ("runner_config", "expected runner configuration digest"),
    ],
)
def test_run_identity_mismatch_is_rejected(
    tmp_path: Path,
    mismatch: str,
    message: str,
) -> None:
    trial = _trial(tmp_path, "task", rewards={"reward": 1})
    if mismatch == "task_checksum":
        trial.task_checksum = "c" * 64
    elif mismatch == "agent_name":
        trial.agent_info.name = "wrong-agent"
    elif mismatch == "agent_version":
        trial.agent_info.version = "9.9.9"
    elif mismatch == "provider":
        assert trial.agent_info.model_info is not None
        trial.agent_info.model_info.provider = "azure"
    elif mismatch == "model_name":
        assert trial.agent_info.model_info is not None
        trial.agent_info.model_info.name = "wrong-model"
    elif mismatch == "task_environment":
        trial.config.environment.type = EnvironmentType.E2B
    elif mismatch == "agent_import_path":
        trial.config.agent.import_path = "example.agent:WrongAgent"
    elif mismatch == "agent_config_model":
        trial.config.agent.model_name = "bedrock/wrong-model"
    elif mismatch == "agent_kwargs":
        trial.config.agent.kwargs["turn_timeout_s"] = 999
    elif mismatch == "candidate_hash":
        assert trial.agent_result is not None
        assert trial.agent_result.metadata is not None
        trial.agent_result.metadata["harness_hash"] = "wrong-hash"
    elif mismatch == "runner_config":
        assert trial.agent_result is not None
        assert trial.agent_result.metadata is not None
        trial.agent_result.metadata["runner_config_digest"] = "sha256:" + "f" * 64
    else:  # pragma: no cover - keeps additions to the parameter table fail-closed
        raise AssertionError(f"unknown mismatch: {mismatch}")

    job_dir = tmp_path / mismatch / "job"
    _write_job(job_dir, [trial], expected=1)

    with pytest.raises(ValueError, match=message):
        load_harbor_job_result(job_dir, _manifest("job", ("task", 1, trial.trial_name)))


@pytest.mark.parametrize("location", ["top_level", "config"])
def test_task_source_mismatch_is_rejected(tmp_path: Path, location: str) -> None:
    trial = _trial(tmp_path, "task", rewards={"reward": 1})
    if location == "top_level":
        trial.source = "wrong-source"
    else:
        trial.config.task.source = "wrong-source"
    job_dir = tmp_path / location / "job"
    _write_job(job_dir, [trial], expected=1)

    lock_digest = harbor_trial_lock_digest(_trial_lock(trial.task_name, trial.config))
    with pytest.raises(ValueError, match="expected task source"):
        load_harbor_job_result(
            job_dir,
            _manifest(
                "job",
                ("task", 1, trial.trial_name),
                trial_lock_digest=lock_digest,
            ),
        )


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("candidate_hash", "expected candidate hash"),
        ("runner_spec", "invalid runner spec"),
        ("provider_config", "expected configured provider/model"),
        ("harbor_model", "expected Harbor agent model"),
    ],
)
def test_agent_config_identity_is_validated_beyond_its_digest(
    tmp_path: Path,
    mismatch: str,
    message: str,
) -> None:
    trial = _trial(tmp_path, "task", rewards={"reward": 1})
    if mismatch == "candidate_hash":
        other_harness = _HARNESS.model_copy(deep=True)
        other_harness.surfaces[0].content += "\nchanged"
        trial.config.agent.kwargs["harness"] = other_harness.model_dump(mode="json")
    elif mismatch == "runner_spec":
        trial.config.agent.kwargs["runner_spec"] = {
            "backend": "local",
            "image": "node:22@sha256:" + "d" * 64,
        }
    elif mismatch == "provider_config":
        trial.config.agent.kwargs["provider_config"] = {"kind": "azure", "model": "model"}
    elif mismatch == "harbor_model":
        trial.config.agent.model_name = "bedrock/wrong-model"
    else:  # pragma: no cover - keeps additions to the parameter table fail-closed
        raise AssertionError(f"unknown mismatch: {mismatch}")
    digest = harbor_agent_config_digest(trial.config.agent)
    lock_digest = harbor_trial_lock_digest(_trial_lock(trial.task_name, trial.config))
    job_dir = tmp_path / mismatch / "job"
    _write_job(job_dir, [trial], expected=1)

    with pytest.raises(ValueError, match=message):
        load_harbor_job_result(
            job_dir,
            _manifest(
                "job",
                ("task", 1, trial.trial_name),
                agent_config_digest=digest,
                trial_lock_digest=lock_digest,
            ),
        )


@pytest.mark.parametrize(
    "setting",
    [
        "agent_timeout",
        "agent_network",
        "environment_resource",
        "verifier_timeout",
        "trial_timeout_multiplier",
    ],
)
def test_replay_critical_setting_change_breaks_trial_lock_identity(
    tmp_path: Path,
    setting: str,
) -> None:
    trial = _trial(tmp_path, "task", rewards={"reward": 1})
    if setting == "agent_timeout":
        trial.config.agent.override_timeout_sec = 10
    elif setting == "agent_network":
        trial.config.agent.extra_allowed_hosts = ["example.com"]
    elif setting == "environment_resource":
        trial.config.environment.override_memory_mb = 2048
    elif setting == "verifier_timeout":
        trial.config.verifier.override_timeout_sec = 30
    elif setting == "trial_timeout_multiplier":
        trial.config.timeout_multiplier = 2
    else:  # pragma: no cover - keeps additions to the parameter table fail-closed
        raise AssertionError(f"unknown setting: {setting}")
    job_dir = tmp_path / setting / "job"
    _write_job(job_dir, [trial], expected=1)

    with pytest.raises(ValueError, match="expected trial lock digest"):
        load_harbor_job_result(job_dir, _manifest("job", ("task", 1, trial.trial_name)))


def test_result_config_must_match_the_protected_trial_lock(tmp_path: Path) -> None:
    trial = _trial(tmp_path, "task", rewards={"reward": 1})
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)
    trial.config.agent.override_timeout_sec = 10
    (job_dir / trial.trial_name / "result.json").write_text(
        trial.model_dump_json(indent=2), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="result config differs from trial lock"):
        load_harbor_job_result(job_dir, _manifest("job", ("task", 1, trial.trial_name)))


def test_manifest_rejects_duplicate_and_traversing_cells() -> None:
    trial_config_digest = "sha256:" + "d" * 64
    entry = HarborTrialManifestEntry(
        cell=BenchmarkCell(
            task_key=_task_key("task"),
            task_name="task",
            attempt=1,
            config_digest=trial_config_digest,
        ),
        trial_name="task__harbor",
        task_identity="task",
        runtime_task_checksum=_RUNTIME_TASK_CHECKSUM,
        task_checksum=_TASK_CHECKSUM,
        task_source=_TASK_SOURCE,
        task_instruction="Instruction for task.",
        trial_lock_digest=trial_config_digest,
    )
    with pytest.raises(ValidationError, match="duplicate manifest benchmark cell"):
        HarborTrialManifest(
            schema_version=2,
            job_name="job",
            identity=_RUN_IDENTITY,
            agent_config_digest=_AGENT_CONFIG_DIGEST,
            entries=[entry, entry],
        )

    with pytest.raises(ValidationError, match="single safe path component"):
        HarborTrialManifestEntry(
            cell=BenchmarkCell(
                task_key=_task_key("task"),
                task_name="task",
                attempt=1,
                config_digest=trial_config_digest,
            ),
            trial_name="../escape",
            task_identity="task",
            runtime_task_checksum=_RUNTIME_TASK_CHECKSUM,
            task_checksum=_TASK_CHECKSUM,
            task_source=_TASK_SOURCE,
            task_instruction="Instruction for task.",
            trial_lock_digest=trial_config_digest,
        )


@pytest.mark.parametrize(
    "tamper",
    ["missing-version", "wrong-version", "top-level-extra", "entry-extra"],
)
def test_manifest_schema_is_explicit_and_strict(tamper: str) -> None:
    payload = _manifest("job", ("task", 1, "task__harbor")).model_dump(mode="json")
    if tamper == "missing-version":
        payload.pop("schema_version")
    elif tamper == "wrong-version":
        payload["schema_version"] = 1
    elif tamper == "top-level-extra":
        payload["unexpected"] = True
    else:
        entries = payload["entries"]
        assert isinstance(entries, list)
        assert isinstance(entries[0], dict)
        entries[0]["unexpected"] = True

    with pytest.raises(ValidationError):
        HarborTrialManifest.model_validate(payload)


def test_symlinked_trial_directory_cannot_escape_job_root(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    _write_job(job_dir, [], expected=1)
    outside = tmp_path / "outside"
    outside.mkdir()
    (job_dir / "task__harbor").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="cannot be a symlink"):
        load_harbor_job_result(job_dir, _manifest("job", ("task", 1, "task__harbor")))


def test_symlinked_job_root_is_rejected_even_when_target_is_a_valid_job(tmp_path: Path) -> None:
    real_job_dir = tmp_path / "real-job"
    _write_job(real_job_dir, [], expected=0)
    linked_job_dir = tmp_path / "job"
    linked_job_dir.symlink_to(real_job_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="job directory cannot be a symlink"):
        load_harbor_job_result(linked_job_dir, _manifest("job"))


def test_symlinked_trial_result_is_rejected_even_when_target_stays_inside_job(
    tmp_path: Path,
) -> None:
    trial = _trial(tmp_path, "task", rewards={"reward": 1})
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)
    trial_dir = job_dir / trial.trial_name
    original_result = trial_dir / "real-result.json"
    (trial_dir / "result.json").replace(original_result)
    (trial_dir / "result.json").symlink_to(original_result.name)

    with pytest.raises(ValueError, match="cannot be a symlink"):
        load_harbor_job_result(job_dir, _manifest("job", ("task", 1, trial.trial_name)))


def test_job_total_must_match_exact_manifest(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    _write_job(job_dir, [], expected=2)

    with pytest.raises(ValueError, match="declares 2 trials but manifest contains 1"):
        load_harbor_job_result(job_dir, _manifest("job", ("task", 1, "task__harbor")))
