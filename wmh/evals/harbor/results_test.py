"""Tests for strict, lossless Harbor 0.18 result ingestion."""

from __future__ import annotations

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
from pydantic import ValidationError

from wmh.evals.benchmark import (
    BenchmarkCandidateStage,
    BenchmarkCandidateStatus,
    BenchmarkCandidateTerminalReason,
    BenchmarkCell,
    BenchmarkFailureKind,
    BenchmarkRunIdentity,
    BenchmarkTaskEnvironment,
    BenchmarkTrialStatus,
)
from wmh.evals.harbor.results import (
    HarborTrialManifest,
    HarborTrialManifestEntry,
    harbor_agent_config_digest,
    harbor_trial_lock_digest,
    load_harbor_job_result,
)
from wmh.harness.pi_runner import pi_node_baseline

_NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
_TASK_CHECKSUM = "sha256:" + "b" * 64
_TASK_SOURCE = "example-benchmark"
_HARNESS = pi_node_baseline("candidate")


def _agent_config() -> AgentConfig:
    return AgentConfig(
        import_path="wmh.evals.harbor.agent:WmhPiAgent",
        model_name="bedrock/model",
        kwargs={
            "harness": _HARNESS.model_dump(mode="json"),
            "provider_config": {"kind": "bedrock", "model": "model"},
            "runner_image": "runner-image",
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
    runner_image="runner-image",
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
        task_checksum=_TASK_CHECKSUM,
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
                "runner_image": "runner-image",
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
    result = JobResult(
        id=uuid4(),
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
        task_checksum=_TASK_CHECKSUM,
        task_source=_TASK_SOURCE,
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

    loaded = load_harbor_job_result(job_dir, manifest)
    result = loaded.result

    assert result.expected_trials == 3
    assert result.identity.run_config_digest == manifest.agent_config_digest
    assert result.n_scored == 1
    assert result.n_infrastructure_errors == 1
    assert result.n_incomplete == 1
    trials = {trial.cell.task_name: trial for trial in result.trials}
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
    assert result.usage.input_tokens is None
    assert result.usage.cache_tokens is None
    assert result.usage.output_tokens is None
    assert result.usage.cost_usd is None

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

    assert result.usage.input_tokens == 20
    assert result.usage.cache_tokens == 4
    assert result.usage.output_tokens == 8
    assert result.usage.cost_usd == 0.1


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

    result = load_harbor_job_result(
        job_dir, _manifest("job", ("task", 1, trial.trial_name))
    ).result.trials[0]

    assert result.status is status
    if status is BenchmarkTrialStatus.SCORED:
        assert result.rewards == {"reward": 0, "tests_passed": 2}
    else:
        assert result.rewards is None
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
    assert result.candidate_outcome.terminal_reason is None
    assert secret not in result.model_dump_json()


@pytest.mark.parametrize(
    "candidate_metadata",
    [
        {"candidate_failure": "yes"},
        {"candidate_failure": False, "candidate_failure_stage": "turn"},
        {"candidate_failure": True, "terminal_reason": "completed"},
        {"candidate_failure": True, "candidate_failure_stage": "unknown-stage"},
        {"candidate_failure": False, "terminal_reason": "unknown-reason"},
        {"candidate_failure_stage": "turn"},
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


def test_inconsistent_step_candidate_outcomes_are_rejected(tmp_path: Path) -> None:
    trial = _trial(tmp_path, "task", rewards={"reward": 1})
    identity = {
        "harness_hash": _HARNESS.execution_hash,
        "runner_image": "runner-image",
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
                }
            ),
        ),
    ]
    job_dir = tmp_path / "job"
    _write_job(job_dir, [trial], expected=1)

    with pytest.raises(ValueError, match="inconsistent candidate outcome"):
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


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("task_checksum", "expected task checksum"),
        ("agent_name", "expected agent"),
        ("agent_version", "expected agent"),
        ("provider", "expected provider/model"),
        ("model_name", "expected provider/model"),
        ("task_environment", "expected trial lock digest"),
        ("agent_import_path", "expected trial lock digest"),
        ("agent_config_model", "expected trial lock digest"),
        ("agent_kwargs", "expected trial lock digest"),
        ("candidate_hash", "expected candidate hash"),
        ("runner_image", "expected runner image"),
    ],
)
def test_run_identity_mismatch_is_rejected(
    tmp_path: Path,
    mismatch: str,
    message: str,
) -> None:
    trial = _trial(tmp_path, "task", rewards={"reward": 1})
    if mismatch == "task_checksum":
        trial.task_checksum = "sha256:" + "c" * 64
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
    elif mismatch == "runner_image":
        assert trial.agent_result is not None
        assert trial.agent_result.metadata is not None
        trial.agent_result.metadata["runner_image"] = "wrong-image"
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
        ("runner_image", "expected runner image"),
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
    elif mismatch == "runner_image":
        trial.config.agent.kwargs["runner_image"] = "wrong-image"
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
        task_checksum=_TASK_CHECKSUM,
        task_source=_TASK_SOURCE,
        trial_lock_digest=trial_config_digest,
    )
    with pytest.raises(ValidationError, match="duplicate manifest benchmark cell"):
        HarborTrialManifest(
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
            task_checksum=_TASK_CHECKSUM,
            task_source=_TASK_SOURCE,
            trial_lock_digest=trial_config_digest,
        )


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
