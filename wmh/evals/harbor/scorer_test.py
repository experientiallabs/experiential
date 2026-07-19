"""Adversarial tests for projecting Harbor runs into generic harness scores."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path

import pytest
from harbor.models.job.config import DatasetConfig

import wmh.evals.harbor.scorer as mod
from wmh.evals.benchmark import (
    BenchmarkCandidateFailureReason,
    BenchmarkCandidateOutcome,
    BenchmarkCandidateStage,
    BenchmarkCandidateStatus,
    BenchmarkCell,
    BenchmarkError,
    BenchmarkFailureKind,
    BenchmarkRunHealth,
    BenchmarkRunResult,
    BenchmarkTrialResult,
    BenchmarkTrialStatus,
    BenchmarkUsage,
)
from wmh.evals.harbor.config import HarborEnvironmentBackend, HarborJobSpec
from wmh.evals.harbor.e2b_environment import ExactE2BBuildSpec, ExactE2BEnvironment
from wmh.evals.harbor.qualification_types import (
    QualifiedE2BBuildIdentity,
    QualifiedHarborTask,
)
from wmh.evals.harbor.results import HarborTrialLocator, LoadedHarborJobResult
from wmh.harness.cost import (
    ProviderCostBinding,
    SearchComponentCostBinding,
    SearchComponentRole,
    SearchCostBinding,
    SearchCostRuntime,
    TimedResourceCostBinding,
)
from wmh.harness.doc import HarnessDoc, Surface, SurfaceKind
from wmh.harness.pi_runner import pi_node_baseline
from wmh.harness.pi_runner_backend import (
    E2BPiRunnerSpec,
    LocalPiRunnerSpec,
    PiRunnerBackendSpec,
    e2b_runner_resource_class,
)
from wmh.harness.scoring import ScoreRequest, ScoreRunHealth
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.tracking._testing import (
    synthetic_provider_cost_meter,
    synthetic_tariff_provenance,
)
from wmh.tracking.budget import (
    BudgetPolicy,
    BudgetScope,
    TimedResourceClass,
    TimedResourceCostMeter,
    bind_budget_account,
    bind_timed_resource_account,
    bootstrap_budget_ledger,
)
from wmh.tracking.rate_limit import (
    E2B_SANDBOX_CREATE_RATE_POLICY,
    ExternalDispatchRateAuthority,
)

_TASK_IDS = ("alpha", "beta")
_TASK_KEYS = ("sha256:" + "a" * 64, "sha256:" + "b" * 64)
_TASK_ENVIRONMENT_DIGESTS = ("sha256:" + "c" * 64, "sha256:" + "d" * 64)
_CONFIG_DIGEST = "sha256:" + "1" * 64
_RUNNER = LocalPiRunnerSpec()


def _qualified_tasks(
    *,
    task_ids: tuple[str, ...] = _TASK_IDS,
    task_keys: tuple[str, ...] = _TASK_KEYS,
    task_environment_digests: tuple[str, ...] = _TASK_ENVIRONMENT_DIGESTS,
) -> tuple[QualifiedHarborTask, ...]:
    if len(task_keys) != len(task_ids):
        raise ValueError("task_keys must contain exactly one key per task_id")
    if len(task_environment_digests) != len(task_ids):
        raise ValueError("task_environment_digests must contain exactly one digest per task_id")
    return tuple(
        QualifiedHarborTask(
            task_id=task_id,
            dataset_id="terminalbench2",
            content_digest="sha256:" + f"{index + 1:x}" * 64,
            task_key=task_key,
            task_environment_digest=environment_digest,
            environment_backend=HarborEnvironmentBackend.LOCAL,
        )
        for index, (task_id, task_key, environment_digest) in enumerate(
            zip(task_ids, task_keys, task_environment_digests, strict=True)
        )
    )


def _e2b_qualified_tasks(
    task_class: TimedResourceClass,
) -> tuple[QualifiedHarborTask, ...]:
    build_spec = ExactE2BBuildSpec(
        environment_id="qualified-environment",
        build_context_digest="sha256:" + "6" * 64,
        docker_image="registry.example/task@sha256:" + "9" * 64,
        cpu_count=task_class.cpu_count,
        memory_mb=task_class.memory_mb,
    )
    build_identity = QualifiedE2BBuildIdentity(
        build_config_digest=build_spec.digest,
        build_record_digest="sha256:" + "8" * 64,
        environment_id=build_spec.environment_id,
        build_context_digest=build_spec.build_context_digest,
        docker_image=build_spec.docker_image,
        cpu_count=build_spec.cpu_count,
        memory_mb=build_spec.memory_mb,
        template_id="task-template",
        build_id="task-build",
    )
    return tuple(
        QualifiedHarborTask(
            task_id=task_id,
            dataset_id="terminalbench2",
            content_digest="sha256:" + f"{index + 1:x}" * 64,
            task_key=task_key,
            task_environment_digest=environment_digest,
            environment_backend=HarborEnvironmentBackend.E2B,
            e2b_launch_config_digest="sha256:" + "7" * 64,
            e2b_build_config_digest=build_identity.build_config_digest,
            e2b_build_record_digest=build_identity.build_record_digest,
            task_resource_class_digest=task_class.digest,
            e2b_build_identity=build_identity,
            task_resource_class=task_class,
        )
        for index, (task_id, task_key, environment_digest) in enumerate(
            zip(_TASK_IDS, _TASK_KEYS, _TASK_ENVIRONMENT_DIGESTS, strict=True)
        )
    )


@dataclass(frozen=True)
class _ResultOptions:
    rewards: dict[tuple[str, int], float] | None = None
    diagnostic_reward: float | None = None
    trace_suffix: str = ""
    status: BenchmarkTrialStatus = BenchmarkTrialStatus.SCORED
    candidate_outcome: BenchmarkCandidateOutcome | None = None
    task_ids: tuple[str, ...] = _TASK_IDS
    task_environment_digests: tuple[str, ...] = _TASK_ENVIRONMENT_DIGESTS
    task_timeout: bool = False
    run_health: BenchmarkRunHealth = BenchmarkRunHealth.VALID
    candidate_damage_error: bool = False
    failure_kind: BenchmarkFailureKind = BenchmarkFailureKind.ENVIRONMENT
    runner_environment_digest: str | None = _RUNNER.attestation.digest
    receipt_response_model: str | None = None
    receipt_system_fingerprint: str | None = None


@dataclass(frozen=True)
class _BuildRecordStub:
    build_config_digest: str
    digest: str
    template_id: str
    build_id: str


def _provider() -> ProviderConfig:
    return ProviderConfig(
        kind=ProviderKind.BEDROCK,
        model="worker-model",
        region="us-west-2",
    )


def _azure_provider() -> ProviderConfig:
    return ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.4-mini",
        deployment="worker-deployment",
        endpoint="https://example.openai.azure.com",
        api_version="2026-01-01",
    )


def _spec(
    tmp_path: Path,
    *,
    backend: HarborEnvironmentBackend = HarborEnvironmentBackend.LOCAL,
) -> HarborJobSpec:
    return HarborJobSpec(
        job_name="search-train",
        jobs_dir=tmp_path / "jobs",
        datasets=[DatasetConfig(path=tmp_path / "dataset", task_names=list(_TASK_IDS))],
        n_attempts=2,
        n_concurrent_trials=2,
        agent_n_concurrent=1,
        environment_backend=backend,
        create_rate_policy=(
            E2B_SANDBOX_CREATE_RATE_POLICY if backend is HarborEnvironmentBackend.E2B else None
        ),
    )


def _cell(task_id: str, attempt: int) -> BenchmarkCell:
    task_key = dict(zip(_TASK_IDS, _TASK_KEYS, strict=True)).get(
        task_id,
        "sha256:" + "c" * 64,
    )
    return BenchmarkCell(
        task_key=task_key,
        task_name=f"display-{task_id}",
        attempt=attempt,
        config_digest=_CONFIG_DIGEST,
    )


def _candidate_failure(
    reason: BenchmarkCandidateFailureReason = BenchmarkCandidateFailureReason.TIMEOUT,
) -> BenchmarkCandidateOutcome:
    return BenchmarkCandidateOutcome(
        status=BenchmarkCandidateStatus.FAILED,
        stage=BenchmarkCandidateStage.EXECUTION,
        failure_reason=reason,
    )


def _provider_receipt_record(
    task_id: str,
    attempt: int,
    *,
    provider_config: ProviderConfig,
    response_model: str | None,
    system_fingerprint: str | None,
) -> dict[str, object]:
    is_bedrock = provider_config.kind is ProviderKind.BEDROCK
    return {
        "kind": "provider_receipt",
        "payload": {
            "provider": provider_config.kind.value,
            "provider_request_id": f"provider-request-{task_id}-{attempt}",
            "response_id": None if is_bedrock else f"response-{task_id}-{attempt}",
            "requested_model": (
                provider_config.model
                if provider_config.deployment is None
                else provider_config.deployment
            ),
            "response_model": response_model,
            "system_fingerprint": system_fingerprint,
            "request_digest": "sha256:" + "f" * 64,
            "temperature": (0.7 if provider_config.resolved_chat_forward_temperature() else None),
            "max_tokens": 4_096,
            "max_tokens_field": (
                "inferenceConfig.maxTokens"
                if is_bedrock
                else provider_config.resolved_chat_max_tokens_field()
            ),
            "seed_supplied": False,
            "cache_config_supplied": False,
            "started_at_unix_s": 10.0,
            "finished_at_unix_s": 11.0,
            "turn_call_index": 1,
        },
    }


def _loaded_result(
    tmp_path: Path,
    candidate: HarnessDoc,
    spec: HarborJobSpec,
    *,
    rewards: dict[tuple[str, int], float] | None = None,
    trace_suffix: str = "",
    status: BenchmarkTrialStatus = BenchmarkTrialStatus.SCORED,
    candidate_outcome: BenchmarkCandidateOutcome | None = None,
    task_ids: tuple[str, ...] = _TASK_IDS,
    task_keys: tuple[str, ...] = _TASK_KEYS,
    task_environment_digests: tuple[str, ...] = _TASK_ENVIRONMENT_DIGESTS,
    task_timeout: bool = False,
    diagnostic_reward: float | None = None,
    run_health: BenchmarkRunHealth = BenchmarkRunHealth.VALID,
    candidate_damage_error: bool = False,
    failure_kind: BenchmarkFailureKind = BenchmarkFailureKind.ENVIRONMENT,
    runner_environment_digest: str | None = _RUNNER.attestation.digest,
    provider_config: ProviderConfig | None = None,
    runner_spec: PiRunnerBackendSpec = _RUNNER,
    turn_timeout_s: float = 300.0,
    receipt_response_model: str | None = "served-model",
    receipt_system_fingerprint: str | None = None,
) -> LoadedHarborJobResult:
    resolved_provider = provider_config or _provider()
    environment_digest_by_task = dict(zip(_TASK_IDS, task_environment_digests, strict=True))
    selected_rewards = rewards or {
        ("alpha", 1): 1.0,
        ("alpha", 2): 0.0,
        ("beta", 1): 1.0,
        ("beta", 2): 1.0,
    }
    job_dir = spec.jobs_dir / spec.job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    trials: list[BenchmarkTrialResult] = []
    locators: list[HarborTrialLocator] = []
    expected: list[BenchmarkCell] = []
    for task_id in task_ids:
        for attempt in range(1, spec.n_attempts + 1):
            cell = _cell(task_id, attempt)
            expected.append(cell)
            error = None
            trial_rewards = {"reward": selected_rewards.get((task_id, attempt), 0.0)}
            if diagnostic_reward is not None:
                trial_rewards["diagnostic"] = diagnostic_reward
            if task_timeout:
                error = BenchmarkError(
                    kind=BenchmarkFailureKind.TASK_TIMEOUT,
                    type="AgentTimeoutError",
                    message="agent execution exceeded the task time limit",
                )
            if candidate_damage_error:
                error = BenchmarkError(
                    kind=BenchmarkFailureKind.ENVIRONMENT,
                    type="WmhPiEnvironmentError",
                    message="task environment infrastructure failed",
                )
            if status is not BenchmarkTrialStatus.SCORED:
                trial_rewards = None
                error = BenchmarkError(
                    kind=failure_kind,
                    type=(
                        "WmhPiEnvironmentConfirmationRequiredError"
                        if failure_kind is BenchmarkFailureKind.ENVIRONMENT_CONFIRMATION_REQUIRED
                        else "EnvironmentStartTimeoutError"
                    ),
                    message="benchmark trial requires operational handling",
                )
            trials.append(
                BenchmarkTrialResult(
                    cell=cell,
                    task_identity=task_id,
                    task_checksum="sha256:" + "a" * 64,
                    source="benchmark",
                    task_instruction=f"Instruction for {task_id}.",
                    task_environment_digest=environment_digest_by_task.get(
                        task_id,
                        "sha256:" + "e" * 64,
                    ),
                    runner_environment_digest=runner_environment_digest,
                    status=status,
                    rewards=trial_rewards,
                    error=error,
                    candidate_outcome=candidate_outcome
                    or BenchmarkCandidateOutcome(status=BenchmarkCandidateStatus.COMPLETED),
                    run_health=run_health,
                    usage=BenchmarkUsage(calls=1),
                )
            )
            trial_dir = Path(f"trial-{task_id}-{attempt}")
            absolute_trial_dir = job_dir / trial_dir
            absolute_trial_dir.mkdir(exist_ok=True)
            trace_path = absolute_trial_dir / "wmh-events.jsonl"
            trace_path.unlink(missing_ok=True)
            trace_records = [
                _provider_receipt_record(
                    task_id,
                    attempt,
                    provider_config=resolved_provider,
                    response_model=(
                        None
                        if resolved_provider.kind is ProviderKind.BEDROCK
                        else receipt_response_model
                    ),
                    system_fingerprint=receipt_system_fingerprint,
                ),
                {
                    "kind": "assistant_message",
                    "payload": {"text": f"answer-{task_id}-{attempt}{trace_suffix}"},
                },
            ]
            trace_path.write_text(
                "".join(
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    for record in trace_records
                ),
                encoding="utf-8",
            )
            locators.append(
                HarborTrialLocator(
                    cell=cell,
                    trial_dir=trial_dir,
                    result_path=trial_dir / "result.json",
                    artifacts_dir=trial_dir / "artifacts",
                )
            )
    return LoadedHarborJobResult(
        result=BenchmarkRunResult(
            job_name=spec.job_name,
            identity=mod.harbor_run_expectation(
                candidate=candidate,
                spec=spec,
                provider_config=resolved_provider,
                runner_spec=runner_spec,
                turn_timeout_s=turn_timeout_s,
            ).identity,
            expected_cells=expected,
            trials=trials,
        ),
        job_dir=job_dir,
        locators=tuple(locators),
    )


def _install_fake_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    result_options: _ResultOptions | None = None,
) -> list[
    tuple[
        HarborJobSpec,
        ProviderConfig,
        PiRunnerBackendSpec,
        float,
        tuple[QualifiedHarborTask, ...],
    ]
]:
    captured: list[
        tuple[
            HarborJobSpec,
            ProviderConfig,
            PiRunnerBackendSpec,
            float,
            tuple[QualifiedHarborTask, ...],
        ]
    ] = []
    options = result_options or _ResultOptions()

    class FakeEvaluator:
        def __init__(
            self,
            spec: HarborJobSpec,
            provider_config: ProviderConfig,
            *,
            runner_spec: PiRunnerBackendSpec,
            turn_timeout_s: float,
            qualified_tasks: tuple[QualifiedHarborTask, ...],
        ) -> None:
            captured.append((spec, provider_config, runner_spec, turn_timeout_s, qualified_tasks))
            self._spec = spec
            self._provider_config = provider_config
            self._runner_spec = runner_spec
            self._turn_timeout_s = turn_timeout_s

        async def evaluate(self, candidate: HarnessDoc) -> LoadedHarborJobResult:
            return _loaded_result(
                tmp_path,
                candidate,
                self._spec,
                rewards=options.rewards,
                trace_suffix=options.trace_suffix,
                status=options.status,
                candidate_outcome=options.candidate_outcome,
                task_ids=options.task_ids,
                task_environment_digests=options.task_environment_digests,
                task_timeout=options.task_timeout,
                diagnostic_reward=options.diagnostic_reward,
                run_health=options.run_health,
                candidate_damage_error=options.candidate_damage_error,
                failure_kind=options.failure_kind,
                runner_environment_digest=options.runner_environment_digest,
                provider_config=self._provider_config,
                runner_spec=self._runner_spec,
                turn_timeout_s=self._turn_timeout_s,
                receipt_response_model=options.receipt_response_model,
                receipt_system_fingerprint=options.receipt_system_fingerprint,
            )

    monkeypatch.setattr(mod, "HarborEvaluator", FakeEvaluator)
    return captured


def test_scorer_passes_exact_qualification_evidence_to_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_evaluator(monkeypatch, tmp_path)
    qualified_tasks = _qualified_tasks()

    with _scorer(tmp_path) as scorer:
        scorer.score(pi_node_baseline("candidate"), request=_full_request())

    assert len(captured) == 1
    assert captured[0][4] == qualified_tasks


class _UnbudgetedTestHarborHarnessScorer(mod.HarborHarnessScorer):
    """Exercise score projection with fake evaluators that cannot dispatch paid work."""

    requires_search_cost_binding = False


def _scorer(
    tmp_path: Path,
    *,
    job_spec: HarborJobSpec | None = None,
    provider_config: ProviderConfig | None = None,
    reference_harness: HarnessDoc | None = None,
    task_ids: tuple[str, ...] = _TASK_IDS,
    task_keys: tuple[str, ...] = _TASK_KEYS,
    task_environment_digests: tuple[str, ...] = _TASK_ENVIRONMENT_DIGESTS,
    reward_key: str = "reward",
    runner_spec: PiRunnerBackendSpec = _RUNNER,
    turn_timeout_s: float = 300.0,
    response_identity: mod.ProviderResponseIdentity | None = None,
) -> mod.HarborHarnessScorer:
    return _UnbudgetedTestHarborHarnessScorer(
        job_spec=job_spec or _spec(tmp_path),
        provider_config=provider_config or _provider(),
        reference_harness=reference_harness or pi_node_baseline("baseline"),
        qualified_tasks=_qualified_tasks(
            task_ids=task_ids,
            task_keys=task_keys,
            task_environment_digests=task_environment_digests,
        ),
        reward_key=reward_key,
        runner_spec=runner_spec,
        turn_timeout_s=turn_timeout_s,
        response_identity=response_identity,
    )


def _full_request() -> ScoreRequest:
    return ScoreRequest(purpose="full")


def test_production_scorer_rejects_missing_cost_runtime_before_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator_constructed = False

    class ForbiddenEvaluator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            nonlocal evaluator_constructed
            evaluator_constructed = True

    monkeypatch.setattr(mod, "HarborEvaluator", ForbiddenEvaluator)
    with pytest.raises(ValueError, match="complete search cost runtime"):
        mod.HarborHarnessScorer(
            job_spec=_spec(tmp_path),
            provider_config=_provider(),
            reference_harness=pi_node_baseline("baseline"),
            qualified_tasks=_qualified_tasks(),
            reward_key="reward",
        )
    assert evaluator_constructed is False


def test_projects_exact_binary_task_means_and_bounded_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_evaluator(monkeypatch, tmp_path)
    candidate = pi_node_baseline("candidate")

    with _scorer(tmp_path) as scorer:
        report = scorer.score(candidate, request=_full_request())

    assert scorer.capabilities.task_subsets is False
    assert scorer.capabilities.attempt_overrides is False
    assert scorer.default_attempts == 2
    assert scorer.task_ids == _TASK_IDS
    assert scorer.task_keys == _TASK_KEYS
    assert scorer.task_environment_digests == _TASK_ENVIRONMENT_DIGESTS
    assert report.attempts == 2
    assert report.run_health is ScoreRunHealth.VALID
    assert report.score == pytest.approx(0.75)
    assert report.secondary_score == report.score
    assert list(report.per_task) == list(_TASK_IDS)
    assert report.per_task["alpha"].score == 0.5
    assert report.per_task["alpha"].passed is False
    assert report.per_task["beta"].score == 1.0
    assert report.per_task["beta"].passed is True
    assert report.per_task["alpha"].secondary_score == 0.5
    assert report.per_task["alpha"].description == "Instruction for alpha."
    assert "answer-alpha-1" in report.per_task["alpha"].evidence
    assert "answer-alpha-2" in report.per_task["alpha"].evidence
    assert "ground-truth reward below one" in report.per_task["alpha"].mechanisms
    assert captured[0][0].environment_backend is HarborEnvironmentBackend.LOCAL


@pytest.mark.parametrize(
    ("response_model", "system_fingerprint"),
    [
        ("retargeted-model", "fp-exact"),
        ("served-model", "fp-drifted"),
    ],
)
def test_scorer_rejects_served_model_or_fingerprint_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response_model: str,
    system_fingerprint: str,
) -> None:
    _install_fake_evaluator(
        monkeypatch,
        tmp_path,
        result_options=_ResultOptions(
            receipt_response_model=response_model,
            receipt_system_fingerprint=system_fingerprint,
        ),
    )
    response_identity = mod.ProviderResponseIdentity(
        provider=ProviderKind.AZURE_OPENAI,
        response_model="served-model",
        system_fingerprint="fp-exact",
    )

    with (
        _scorer(
            tmp_path,
            provider_config=_azure_provider(),
            response_identity=response_identity,
        ) as scorer,
        pytest.raises(ValueError, match="invalid provider-call evidence"),
    ):
        scorer.score(pi_node_baseline("candidate"), request=_full_request())


def test_scorer_accepts_exact_served_model_and_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_evaluator(
        monkeypatch,
        tmp_path,
        result_options=_ResultOptions(
            receipt_response_model="served-model",
            receipt_system_fingerprint="fp-exact",
        ),
    )
    response_identity = mod.ProviderResponseIdentity(
        provider=ProviderKind.AZURE_OPENAI,
        response_model="served-model",
        system_fingerprint="fp-exact",
    )

    with _scorer(
        tmp_path,
        provider_config=_azure_provider(),
        response_identity=response_identity,
    ) as scorer:
        report = scorer.score(pi_node_baseline("candidate"), request=_full_request())

    assert report.run_health is ScoreRunHealth.VALID


def test_configuration_id_binds_provider_and_qualified_task_matrix(tmp_path: Path) -> None:
    baseline = _scorer(tmp_path).configuration_id

    assert _scorer(tmp_path).configuration_id == baseline
    assert _scorer(tmp_path, reward_key="other-reward").configuration_id != baseline
    assert (
        _scorer(
            tmp_path,
            provider_config=_provider().model_copy(update={"model": "other-worker"}),
        ).configuration_id
        != baseline
    )
    assert (
        _scorer(
            tmp_path,
            task_keys=("sha256:" + "e" * 64, _TASK_KEYS[1]),
        ).configuration_id
        != baseline
    )
    first_identity = mod.ProviderResponseIdentity(
        provider=ProviderKind.AZURE_OPENAI,
        response_model="served-model",
        system_fingerprint="fp-first",
    )
    second_identity = first_identity.model_copy(update={"system_fingerprint": "fp-second"})
    assert (
        _scorer(
            tmp_path,
            provider_config=_azure_provider(),
            response_identity=first_identity,
        ).configuration_id
        != _scorer(
            tmp_path,
            provider_config=_azure_provider(),
            response_identity=second_identity,
        ).configuration_id
    )


def test_openai_shaped_scorer_requires_precommitted_response_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact provider response identity"):
        _scorer(tmp_path, provider_config=_azure_provider())


def test_configuration_plan_is_path_free_across_host_relocation(tmp_path: Path) -> None:
    first = _scorer(tmp_path / "first-host")
    second = _scorer(tmp_path / "second-host")

    assert first.plan == second.plan
    assert first.configuration_id == second.configuration_id
    serialized = first.plan.model_dump_json()
    assert str(tmp_path) not in serialized
    assert "jobs_dir" not in serialized


def test_configuration_plan_rejects_host_absolute_artifact_paths(tmp_path: Path) -> None:
    spec = _spec(tmp_path).model_copy(
        update={"artifact_paths": [str((tmp_path / "host-artifact").resolve())]}
    )

    with pytest.raises(ValueError, match="artifact_paths must be project-relative"):
        mod.harbor_harness_score_plan(
            job_spec=spec,
            provider_config=_provider(),
            reference_harness=pi_node_baseline("baseline"),
            qualified_tasks=_qualified_tasks(),
            reward_key="reward",
        )


def test_scorer_rejects_e2b_runner_lease_without_turn_cleanup_margin(
    tmp_path: Path,
) -> None:
    runner_spec = E2BPiRunnerSpec(
        template_id="runner-template",
        build_id="runner-build",
        cpu_count=2,
        memory_mb=2048,
        platform="linux/x86_64",
        envd_version="0.2.1",
        lease_timeout_s=3_659,
    )

    with pytest.raises(ValueError, match="lease_timeout_s"):
        _scorer(
            tmp_path,
            runner_spec=runner_spec,
            turn_timeout_s=3_600,
        )


def test_run_identity_validation_rejects_invalid_turn_before_matching(
    tmp_path: Path,
) -> None:
    candidate = pi_node_baseline("candidate")
    spec = _spec(tmp_path)
    result = _loaded_result(tmp_path, candidate, spec).result

    with pytest.raises(ValueError, match="finite and positive"):
        mod.validate_harbor_run_identity(
            result,
            candidate=candidate,
            spec=spec,
            provider_config=_provider(),
            turn_timeout_s=float("nan"),
        )


def test_scorer_propagates_one_budget_policy_and_ledger_to_e2b_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_spec = E2BPiRunnerSpec(
        template_id="runner-template",
        build_id="runner-build",
        cpu_count=2,
        memory_mb=2048,
        platform="linux/x86_64",
        envd_version="0.2.1",
        lease_timeout_s=420,
    )
    task_class = ExactE2BEnvironment._task_resource_class(
        cpu_count=2,
        memory_mb=1024,
    )
    qualified_tasks = _e2b_qualified_tasks(task_class)
    build_identity = qualified_tasks[0].e2b_build_identity
    assert build_identity is not None
    runner_class = e2b_runner_resource_class(runner_spec)
    provider_config = _provider()
    e2b_spec = _spec(tmp_path, backend=HarborEnvironmentBackend.E2B)
    reference = pi_node_baseline("baseline")
    rate_authority = ExternalDispatchRateAuthority.bootstrap(
        (tmp_path / "rate.json").resolve(),
        E2B_SANDBOX_CREATE_RATE_POLICY,
    )
    plan = mod.harbor_harness_score_plan(
        job_spec=e2b_spec,
        provider_config=provider_config,
        reference_harness=reference,
        qualified_tasks=qualified_tasks,
        reward_key="reward",
        runner_spec=runner_spec,
        create_rate_binding=rate_authority.binding,
    )
    policy = BudgetPolicy(
        study_id="scorer-e2b",
        manifest_digest="sha256:" + hashlib.sha256(str(tmp_path).encode()).hexdigest(),
        hard_limit_nano_usd=100_000,
        phase_limits_nano_usd={"search": 100_000},
        meters={
            "proposer": synthetic_provider_cost_meter(
                provider_config=provider_config,
                provenance=synthetic_tariff_provenance(provider_config),
                input_nano_usd_per_token=1,
                output_nano_usd_per_token=1,
            ),
            "worker": synthetic_provider_cost_meter(
                provider_config=provider_config,
                provenance=synthetic_tariff_provenance(provider_config),
                input_nano_usd_per_token=1,
                output_nano_usd_per_token=1,
            ),
            "task": TimedResourceCostMeter(
                resource_type=task_class.role.value,
                resource_class_digest=task_class.digest,
                nano_usd_per_second=1,
                max_billing_seconds=task_class.max_host_observation_seconds,
            ),
            "runner": TimedResourceCostMeter(
                resource_type=runner_class.role.value,
                resource_class_digest=runner_class.digest,
                nano_usd_per_second=1,
                max_billing_seconds=runner_class.max_host_observation_seconds,
            ),
        },
    )
    proposer_scope = BudgetScope(phase="search", category="proposer", run_id="test-run")
    scorer_scope = BudgetScope(phase="search", category="scorer", run_id="test-run")
    ledger_path = (tmp_path / "budget.sqlite3").resolve()
    authority = bootstrap_budget_ledger(ledger_path, policy)
    proposer_account = authority.provider_account(
        scope=proposer_scope,
        meter_id="proposer",
    )
    provider_account = authority.provider_account(
        scope=scorer_scope,
        meter_id="worker",
    )
    task_account = authority.timed_resource_account(
        scope=scorer_scope,
        meter_id="task",
    )
    runner_account = authority.timed_resource_account(
        scope=scorer_scope,
        meter_id="runner",
    )
    binding = SearchCostBinding(
        declared_hard_limit_nano_usd=policy.hard_limit_nano_usd,
        policy=policy,
        ledger_identity=authority.ledger_identity,
        phase="search",
        run_id="test-run",
        proposer=SearchComponentCostBinding(
            role=SearchComponentRole.PROPOSER,
            configuration_id="unused-proposer",
            scope_category="proposer",
            providers=(
                ProviderCostBinding(
                    component_configuration_id="unused-proposer",
                    provider_config=provider_config,
                    account=bind_budget_account(proposer_account),
                ),
            ),
        ),
        scorer=SearchComponentCostBinding(
            role=SearchComponentRole.SCORER,
            configuration_id=plan.configuration_id,
            scope_category="scorer",
            providers=(
                ProviderCostBinding(
                    component_configuration_id=plan.configuration_id,
                    provider_config=provider_config,
                    account=bind_budget_account(provider_account),
                ),
            ),
            timed_resources=(
                TimedResourceCostBinding(
                    component_configuration_id=plan.configuration_id,
                    resource_type=task_class.role.value,
                    resource_class_digest=task_class.digest,
                    account=bind_timed_resource_account(task_account),
                ),
                TimedResourceCostBinding(
                    component_configuration_id=plan.configuration_id,
                    resource_type=runner_class.role.value,
                    resource_class_digest=runner_class.digest,
                    account=bind_timed_resource_account(runner_account),
                ),
            ),
        ),
    )
    runtime = SearchCostRuntime(authority=authority, binding=binding).for_component(
        SearchComponentRole.SCORER
    )
    captured: dict[str, object] = {}
    current_build_record = [
        _BuildRecordStub(
            build_config_digest=build_identity.build_config_digest,
            digest=build_identity.build_record_digest,
            template_id=build_identity.template_id,
            build_id=build_identity.build_id,
        )
    ]

    def require_build(**_kwargs: object) -> _BuildRecordStub:
        return current_build_record[0]

    class CapturingEvaluator:
        def __init__(
            self,
            _spec: HarborJobSpec,
            _provider: ProviderConfig,
            **kwargs: object,
        ) -> None:
            captured.update(kwargs)

        async def evaluate(self, _candidate: HarnessDoc) -> LoadedHarborJobResult:
            raise RuntimeError("captured")

    monkeypatch.setattr(mod, "HarborEvaluator", CapturingEvaluator)
    monkeypatch.setattr(mod, "require_exact_e2b_build_record", require_build)
    scorer = mod.HarborHarnessScorer(
        job_spec=e2b_spec,
        provider_config=provider_config,
        reference_harness=reference,
        qualified_tasks=qualified_tasks,
        reward_key="reward",
        runner_spec=runner_spec,
        cost_runtime=runtime,
        create_rate_authority=rate_authority,
    )

    with pytest.raises(RuntimeError, match="captured"):
        scorer.score(pi_node_baseline("candidate"), request=_full_request())

    assert captured["budget_account"] == provider_account
    assert captured["task_resource_budget_accounts"] == (task_account,)
    assert captured["runner_resource_budget_account"] == runner_account
    assert captured["create_rate_authority"] is rate_authority
    assert scorer.search_cost_binding == runtime.binding
    assert {
        provider_account.policy.policy_digest,
        task_account.policy.policy_digest,
        runner_account.policy.policy_digest,
    } == {policy.policy_digest}
    assert {
        provider_account.ledger_path,
        task_account.ledger_path,
        runner_account.ledger_path,
    } == {ledger_path}

    captured_before_corruption = dict(captured)
    current_build_record[0] = _BuildRecordStub(
        build_config_digest=build_identity.build_config_digest,
        digest=build_identity.build_record_digest,
        template_id="replacement-template",
        build_id=build_identity.build_id,
    )
    with pytest.raises(ValueError, match="differs from its qualification evidence"):
        scorer.score(pi_node_baseline("candidate-build-drift"), request=_full_request())
    assert captured == captured_before_corruption
    current_build_record[0] = _BuildRecordStub(
        build_config_digest=build_identity.build_config_digest,
        digest=build_identity.build_record_digest,
        template_id=build_identity.template_id,
        build_id=build_identity.build_id,
    )
    with sqlite3.connect(runtime.authority.ledger_path) as connection:
        connection.execute("DROP TRIGGER budget_metadata_no_update")
        connection.execute("UPDATE budget_metadata SET schema_version = 1 WHERE id = 1")
    with pytest.raises(RuntimeError, match="unsupported budget schema version"):
        scorer.score(pi_node_baseline("candidate-2"), request=_full_request())
    assert captured == captured_before_corruption
    scorer.close()


def test_score_job_identity_binds_full_candidate_route_and_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_evaluator(monkeypatch, tmp_path)
    candidate = pi_node_baseline("seed")
    renamed = candidate.model_copy(update={"name": "no-op-child", "version": 7})
    assert candidate.execution_hash == renamed.execution_hash

    with _scorer(tmp_path) as scorer:
        first_name = scorer._job_name(candidate)
        second_name = scorer._job_name(renamed)
        scorer.score(candidate, request=_full_request())
        scorer.score(renamed, request=_full_request())
    assert first_name != second_name
    assert captured[0][0].job_name == first_name
    assert captured[1][0].job_name == second_name

    azure = ProviderConfig(kind=ProviderKind.AZURE_OPENAI, model="worker-model")
    with (
        _scorer(
            tmp_path,
            provider_config=azure,
            response_identity=mod.ProviderResponseIdentity(
                provider=ProviderKind.AZURE_OPENAI,
                response_model="served-worker-model",
            ),
        ) as route_scorer,
        _scorer(
            tmp_path,
            task_environment_digests=(
                "sha256:" + "e" * 64,
                _TASK_ENVIRONMENT_DIGESTS[1],
            ),
        ) as environment_scorer,
        _scorer(
            tmp_path,
            task_keys=("sha256:" + "e" * 64, _TASK_KEYS[1]),
        ) as task_key_scorer,
        _scorer(tmp_path, reward_key="other-reward") as reward_scorer,
    ):
        assert route_scorer._job_name(candidate) != first_name
        assert environment_scorer._job_name(candidate) != first_name
        assert task_key_scorer._job_name(candidate) != first_name
        assert reward_scorer._job_name(candidate) != first_name
    assert captured[0][0].job_name.startswith("wmh-score-")
    assert len(captured[0][0].job_name) == len("wmh-score-") + 64
    assert captured[0][1] == _provider()
    assert captured[0][2] == _RUNNER
    assert captured[0][3] == 300.0


@pytest.mark.parametrize(
    "score_request",
    [
        ScoreRequest(purpose="screen", task_ids=("alpha",)),
        ScoreRequest(purpose="full", attempts=2),
    ],
)
def test_rejects_unsupported_score_request_without_evaluating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    score_request: ScoreRequest,
) -> None:
    captured = _install_fake_evaluator(monkeypatch, tmp_path)

    with _scorer(tmp_path) as scorer, pytest.raises(ValueError, match="does not support"):
        scorer.score(pi_node_baseline("candidate"), request=score_request)

    assert captured == []


@pytest.mark.parametrize("task_ids", [("alpha",), ("alpha", "gamma")])
def test_rejects_missing_or_extra_task_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_ids: tuple[str, ...],
) -> None:
    _install_fake_evaluator(
        monkeypatch,
        tmp_path,
        result_options=_ResultOptions(task_ids=task_ids),
    )

    with _scorer(tmp_path) as scorer, pytest.raises(ValueError, match="task identities"):
        scorer.score(pi_node_baseline("candidate"), request=_full_request())


@pytest.mark.parametrize(
    ("digest", "message"),
    [
        (None, "omits trusted runner environment identity"),
        ("sha256:" + "f" * 64, "runner identity differs from plan"),
    ],
)
def test_rejects_missing_or_drifted_runner_environment_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    digest: str | None,
    message: str,
) -> None:
    _install_fake_evaluator(
        monkeypatch,
        tmp_path,
        result_options=_ResultOptions(runner_environment_digest=digest),
    )

    with _scorer(tmp_path) as scorer, pytest.raises(ValueError, match=message):
        scorer.score(pi_node_baseline("candidate"), request=_full_request())


def test_rejects_duplicate_attempt_even_if_result_model_validation_was_bypassed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = pi_node_baseline("candidate")

    class FakeEvaluator:
        def __init__(self, spec: HarborJobSpec, *_args: object, **_kwargs: object) -> None:
            self._spec = spec

        async def evaluate(self, _candidate: HarnessDoc) -> LoadedHarborJobResult:
            loaded = _loaded_result(tmp_path, candidate, self._spec)
            duplicated = loaded.result.trials[0].model_copy(deep=True)
            trials = [duplicated, *loaded.result.trials]
            malformed = BenchmarkRunResult.model_construct(
                job_name=loaded.result.job_name,
                identity=loaded.result.identity,
                expected_cells=loaded.result.expected_cells,
                trials=trials,
                usage=loaded.result.usage,
            )
            return LoadedHarborJobResult(
                result=malformed,
                job_dir=loaded.job_dir,
                locators=loaded.locators,
            )

    monkeypatch.setattr(mod, "HarborEvaluator", FakeEvaluator)

    with _scorer(tmp_path) as scorer, pytest.raises(ValueError, match="duplicate observed"):
        scorer.score(candidate, request=_full_request())


def test_rejects_task_content_key_drift_from_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_evaluator(monkeypatch, tmp_path)
    changed_keys = ("sha256:" + "c" * 64, "sha256:" + "d" * 64)

    with (
        _scorer(tmp_path, task_keys=changed_keys) as scorer,
        pytest.raises(
            ValueError,
            match="frozen qualification manifest",
        ),
    ):
        scorer.score(pi_node_baseline("candidate"), request=_full_request())


def test_rejects_task_environment_drift_from_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = ("sha256:" + "e" * 64, "sha256:" + "f" * 64)
    _install_fake_evaluator(
        monkeypatch,
        tmp_path,
        result_options=_ResultOptions(task_environment_digests=changed),
    )

    with (
        _scorer(tmp_path) as scorer,
        pytest.raises(
            ValueError,
            match="frozen qualification run",
        ),
    ):
        scorer.score(pi_node_baseline("candidate"), request=_full_request())


def test_scored_candidate_failure_is_data_but_infrastructure_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_evaluator(
        monkeypatch,
        tmp_path,
        result_options=_ResultOptions(candidate_outcome=_candidate_failure()),
    )
    with _scorer(tmp_path) as scorer:
        report = scorer.score(pi_node_baseline("candidate"), request=_full_request())
    assert report.score == 0.0
    assert report.secondary_score == 0.0
    assert report.per_task["alpha"].secondary_score == 0.0
    assert report.per_task["beta"].secondary_score == 0.0
    assert "candidate timeout" in report.per_task["alpha"].mechanisms
    assert "candidate timeout" in report.per_task["beta"].mechanisms
    assert "verifier_reward=1.0" in report.per_task["beta"].evidence
    assert "score=0.0" in report.per_task["beta"].evidence

    _install_fake_evaluator(
        monkeypatch,
        tmp_path,
        result_options=_ResultOptions(task_timeout=True),
    )
    with _scorer(tmp_path) as scorer:
        timeout_report = scorer.score(pi_node_baseline("candidate"), request=_full_request())
    assert timeout_report.score == 0.0
    assert timeout_report.secondary_score == 0.0
    assert timeout_report.per_task["alpha"].secondary_score == 0.0
    assert timeout_report.per_task["beta"].secondary_score == 0.0
    assert "agent task timeout" in timeout_report.per_task["beta"].mechanisms
    assert "verifier_reward=1.0" in timeout_report.per_task["beta"].evidence

    _install_fake_evaluator(
        monkeypatch,
        tmp_path,
        result_options=_ResultOptions(status=BenchmarkTrialStatus.INFRASTRUCTURE_ERROR),
    )
    with _scorer(tmp_path) as scorer, pytest.raises(ValueError, match="not scored"):
        scorer.score(pi_node_baseline("candidate"), request=_full_request())


def test_candidate_damaged_trial_without_verifier_reward_is_a_valid_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_evaluator(
        monkeypatch,
        tmp_path,
        result_options=_ResultOptions(
            status=BenchmarkTrialStatus.CANDIDATE_FAILURE,
            candidate_outcome=_candidate_failure(),
            run_health=BenchmarkRunHealth.CANDIDATE_DAMAGED,
        ),
    )

    with _scorer(tmp_path) as scorer:
        report = scorer.score(pi_node_baseline("candidate"), request=_full_request())

    assert report.run_health is ScoreRunHealth.VALID
    assert report.score == 0.0
    assert report.secondary_score == 0.0
    assert "candidate timeout" in report.per_task["alpha"].mechanisms
    assert "verifier_reward=unavailable" in report.per_task["alpha"].evidence


def test_candidate_invalid_provider_request_is_a_valid_zero_mechanism(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_evaluator(
        monkeypatch,
        tmp_path,
        result_options=_ResultOptions(
            candidate_outcome=_candidate_failure(BenchmarkCandidateFailureReason.INVALID_REQUEST),
        ),
    )

    with _scorer(tmp_path) as scorer:
        report = scorer.score(pi_node_baseline("candidate"), request=_full_request())

    assert report.run_health is ScoreRunHealth.VALID
    assert report.score == 0.0
    assert "candidate invalid request" in report.per_task["alpha"].mechanisms


def test_candidate_damage_with_a_later_reward_keeps_diagnostic_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_evaluator(
        monkeypatch,
        tmp_path,
        result_options=_ResultOptions(
            rewards={
                ("alpha", 1): 1.0,
                ("alpha", 2): 1.0,
                ("beta", 1): 1.0,
                ("beta", 2): 1.0,
            },
            candidate_outcome=_candidate_failure(),
            run_health=BenchmarkRunHealth.CANDIDATE_DAMAGED,
            candidate_damage_error=True,
        ),
    )

    with _scorer(tmp_path) as scorer:
        report = scorer.score(pi_node_baseline("candidate"), request=_full_request())

    assert report.run_health is ScoreRunHealth.VALID
    assert report.score == 0.0
    assert "verifier_reward=1.0" in report.per_task["alpha"].evidence
    assert "trial_error=environment" in report.per_task["alpha"].evidence


def test_confirmation_required_trial_with_existing_trace_never_enters_a_score_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_evaluator(
        monkeypatch,
        tmp_path,
        result_options=_ResultOptions(
            status=BenchmarkTrialStatus.INFRASTRUCTURE_ERROR,
            run_health=BenchmarkRunHealth.RETRY_REQUIRED,
            failure_kind=BenchmarkFailureKind.ENVIRONMENT_CONFIRMATION_REQUIRED,
        ),
    )

    with _scorer(tmp_path) as scorer, pytest.raises(ValueError, match="retry or invalidate"):
        scorer.score(pi_node_baseline("candidate"), request=_full_request())


def test_scored_status_cannot_hide_infrastructure_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = pi_node_baseline("candidate")

    class ForgedEvaluator:
        def __init__(self, spec: HarborJobSpec, *_args: object, **_kwargs: object) -> None:
            self._spec = spec

        async def evaluate(self, _candidate: HarnessDoc) -> LoadedHarborJobResult:
            loaded = _loaded_result(tmp_path, candidate, self._spec)
            forged_error = BenchmarkError(
                kind=BenchmarkFailureKind.PROVIDER,
                type="ApiInternalServerError",
                message="model provider infrastructure failed",
            )
            forged_trial = loaded.result.trials[0].model_copy(
                update={"error": forged_error},
                deep=True,
            )
            forged_result = loaded.result.model_copy(
                update={"trials": [forged_trial, *loaded.result.trials[1:]]},
                deep=True,
            )
            return LoadedHarborJobResult(
                result=forged_result,
                job_dir=loaded.job_dir,
                locators=loaded.locators,
            )

    monkeypatch.setattr(mod, "HarborEvaluator", ForgedEvaluator)

    with _scorer(tmp_path) as scorer, pytest.raises(ValueError, match="infrastructure error"):
        scorer.score(candidate, request=_full_request())


def test_evaluation_id_changes_with_reward_or_trace_and_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = pi_node_baseline("candidate")
    _install_fake_evaluator(monkeypatch, tmp_path)
    with _scorer(tmp_path) as scorer:
        first = scorer.score(candidate, request=_full_request())
        repeated = scorer.score(candidate, request=_full_request())
    assert repeated.evaluation_id == first.evaluation_id

    changed_rewards = {
        ("alpha", 1): 0.0,
        ("alpha", 2): 0.0,
        ("beta", 1): 1.0,
        ("beta", 2): 1.0,
    }
    _install_fake_evaluator(
        monkeypatch,
        tmp_path,
        result_options=_ResultOptions(rewards=changed_rewards),
    )
    with _scorer(tmp_path) as scorer:
        reward_changed = scorer.score(candidate, request=_full_request())
    assert reward_changed.evaluation_id != first.evaluation_id

    _install_fake_evaluator(
        monkeypatch,
        tmp_path,
        result_options=_ResultOptions(trace_suffix="-changed"),
    )
    with _scorer(tmp_path) as scorer:
        trace_changed = scorer.score(candidate, request=_full_request())
    assert trace_changed.evaluation_id != first.evaluation_id

    _install_fake_evaluator(
        monkeypatch,
        tmp_path,
        result_options=_ResultOptions(diagnostic_reward=0.25),
    )
    with _scorer(tmp_path) as scorer:
        diagnostic_changed = scorer.score(candidate, request=_full_request())
    assert diagnostic_changed.score == first.score
    assert diagnostic_changed.evaluation_id != first.evaluation_id


def test_evaluation_id_binds_request_and_report_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = pi_node_baseline("candidate")
    _install_fake_evaluator(monkeypatch, tmp_path)
    with _scorer(tmp_path) as scorer:
        seed = scorer.score(candidate, request=ScoreRequest(purpose="seed"))
        full = scorer.score(candidate, request=_full_request())
    assert seed.evaluation_id != full.evaluation_id

    renamed_spec = _spec(tmp_path).model_copy(update={"job_name": "other-prefix"}, deep=True)
    with _scorer(tmp_path, job_spec=renamed_spec) as scorer:
        renamed = scorer.score(candidate, request=_full_request())
    assert renamed.label != full.label
    assert renamed.evaluation_id != full.evaluation_id


def test_trace_read_is_contained_and_rendering_is_capped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"kind":"error","payload":{"message":"escape"}}\n', encoding="utf-8")

    class SymlinkEvaluator:
        def __init__(self, spec: HarborJobSpec, *_args: object, **_kwargs: object) -> None:
            self._spec = spec

        async def evaluate(self, candidate: HarnessDoc) -> LoadedHarborJobResult:
            loaded = _loaded_result(tmp_path, candidate, self._spec)
            trace = loaded.job_dir / loaded.locators[0].trial_dir / "wmh-events.jsonl"
            trace.unlink()
            trace.symlink_to(outside)
            return loaded

    monkeypatch.setattr(mod, "HarborEvaluator", SymlinkEvaluator)
    with _scorer(tmp_path) as scorer, pytest.raises(ValueError, match="cannot be a symlink"):
        scorer.score(pi_node_baseline("candidate"), request=_full_request())

    huge_suffix = "x" * 40_000
    _install_fake_evaluator(
        monkeypatch,
        tmp_path,
        result_options=_ResultOptions(trace_suffix=huge_suffix),
    )
    with _scorer(tmp_path) as scorer:
        report = scorer.score(pi_node_baseline("candidate"), request=_full_request())
    assert len(report.per_task["alpha"].evidence) <= mod.MAX_HARBOR_TASK_EVIDENCE_CHARS
    assert "truncated" in report.per_task["alpha"].evidence


def test_trace_records_are_strictly_typed_before_reaching_proposer_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidTraceEvaluator:
        def __init__(self, spec: HarborJobSpec, *_args: object, **_kwargs: object) -> None:
            self._spec = spec

        async def evaluate(self, candidate: HarnessDoc) -> LoadedHarborJobResult:
            loaded = _loaded_result(tmp_path, candidate, self._spec)
            trace = loaded.job_dir / loaded.locators[0].trial_dir / "wmh-events.jsonl"
            trace.write_text('{"kind":"backend-secret","payload":{}}\n', encoding="utf-8")
            return loaded

    monkeypatch.setattr(mod, "HarborEvaluator", InvalidTraceEvaluator)

    with _scorer(tmp_path) as scorer, pytest.raises(ValueError, match="invalid event"):
        scorer.score(pi_node_baseline("candidate"), request=_full_request())


def test_empty_trace_is_rejected_when_successful_calls_were_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyTraceEvaluator:
        def __init__(self, spec: HarborJobSpec, *_args: object, **_kwargs: object) -> None:
            self._spec = spec

        async def evaluate(self, candidate: HarnessDoc) -> LoadedHarborJobResult:
            loaded = _loaded_result(tmp_path, candidate, self._spec)
            trace = loaded.job_dir / loaded.locators[0].trial_dir / "wmh-events.jsonl"
            trace.write_text("", encoding="utf-8")
            return loaded

    monkeypatch.setattr(mod, "HarborEvaluator", EmptyTraceEvaluator)

    with (
        _scorer(tmp_path) as scorer,
        pytest.raises(ValueError, match="invalid provider-call evidence"),
    ):
        scorer.score(pi_node_baseline("candidate"), request=_full_request())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 0.2),
        ("max_tokens", 2_048),
        ("max_tokens_field", "max_tokens"),
        ("seed_supplied", True),
        ("cache_config_supplied", True),
    ],
)
def test_trace_rejects_receipts_with_altered_frozen_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    class AlteredControlEvaluator:
        def __init__(self, spec: HarborJobSpec, *_args: object, **_kwargs: object) -> None:
            self._spec = spec

        async def evaluate(self, candidate: HarnessDoc) -> LoadedHarborJobResult:
            loaded = _loaded_result(tmp_path, candidate, self._spec)
            trace = loaded.job_dir / loaded.locators[0].trial_dir / "wmh-events.jsonl"
            records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            records[0]["payload"][field] = value
            trace.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            return loaded

    monkeypatch.setattr(mod, "HarborEvaluator", AlteredControlEvaluator)

    with (
        _scorer(tmp_path) as scorer,
        pytest.raises(ValueError, match="invalid provider-call evidence"),
    ):
        scorer.score(pi_node_baseline("candidate"), request=_full_request())


def test_trace_rejects_omitted_explicit_nullable_receipt_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OmittedFieldEvaluator:
        def __init__(self, spec: HarborJobSpec, *_args: object, **_kwargs: object) -> None:
            self._spec = spec

        async def evaluate(self, candidate: HarnessDoc) -> LoadedHarborJobResult:
            loaded = _loaded_result(tmp_path, candidate, self._spec)
            trace = loaded.job_dir / loaded.locators[0].trial_dir / "wmh-events.jsonl"
            records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            records[0]["payload"].pop("response_id")
            trace.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            return loaded

    monkeypatch.setattr(mod, "HarborEvaluator", OmittedFieldEvaluator)

    with (
        _scorer(tmp_path) as scorer,
        pytest.raises(ValueError, match="invalid provider-call evidence"),
    ):
        scorer.score(pi_node_baseline("candidate"), request=_full_request())


def test_trace_receipt_cardinality_must_match_persisted_successful_call_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CardinalityEvaluator:
        def __init__(self, spec: HarborJobSpec, *_args: object, **_kwargs: object) -> None:
            self._spec = spec

        async def evaluate(self, candidate: HarnessDoc) -> LoadedHarborJobResult:
            loaded = _loaded_result(tmp_path, candidate, self._spec)
            trials = list(loaded.result.trials)
            trials[0] = trials[0].model_copy(update={"usage": BenchmarkUsage(calls=2)})
            return LoadedHarborJobResult(
                result=loaded.result.model_copy(update={"trials": trials}),
                job_dir=loaded.job_dir,
                locators=loaded.locators,
            )

    monkeypatch.setattr(mod, "HarborEvaluator", CardinalityEvaluator)

    with (
        _scorer(tmp_path) as scorer,
        pytest.raises(ValueError, match="invalid provider-call evidence"),
    ):
        scorer.score(pi_node_baseline("candidate"), request=_full_request())


def test_provider_request_identity_must_be_unique_across_all_trial_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DuplicateReceiptEvaluator:
        def __init__(self, spec: HarborJobSpec, *_args: object, **_kwargs: object) -> None:
            self._spec = spec

        async def evaluate(self, candidate: HarnessDoc) -> LoadedHarborJobResult:
            loaded = _loaded_result(tmp_path, candidate, self._spec)
            traces = [
                loaded.job_dir / locator.trial_dir / "wmh-events.jsonl"
                for locator in loaded.locators[:2]
            ]
            first = json.loads(traces[0].read_text(encoding="utf-8").splitlines()[0])
            records = [
                json.loads(line) for line in traces[1].read_text(encoding="utf-8").splitlines()
            ]
            records[0]["payload"]["provider_request_id"] = first["payload"]["provider_request_id"]
            traces[1].write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            return loaded

    monkeypatch.setattr(mod, "HarborEvaluator", DuplicateReceiptEvaluator)

    with _scorer(tmp_path) as scorer, pytest.raises(ValueError, match="reused across trials"):
        scorer.score(pi_node_baseline("candidate"), request=_full_request())


def test_missing_trace_is_allowed_only_for_harbor_task_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingTraceEvaluator:
        task_timeout = False
        candidate_failure = False

        def __init__(self, spec: HarborJobSpec, *_args: object, **_kwargs: object) -> None:
            self._spec = spec

        async def evaluate(self, candidate: HarnessDoc) -> LoadedHarborJobResult:
            loaded = _loaded_result(
                tmp_path,
                candidate,
                self._spec,
                task_timeout=self.task_timeout,
                candidate_outcome=_candidate_failure() if self.candidate_failure else None,
            )
            for locator in loaded.locators:
                (loaded.job_dir / locator.trial_dir / "wmh-events.jsonl").unlink()
            if self.task_timeout:
                zero_call_trials = [
                    trial.model_copy(update={"usage": BenchmarkUsage(calls=0)})
                    for trial in loaded.result.trials
                ]
                loaded = LoadedHarborJobResult(
                    result=loaded.result.model_copy(
                        update={
                            "trials": zero_call_trials,
                            "usage": BenchmarkUsage(calls=0),
                        }
                    ),
                    job_dir=loaded.job_dir,
                    locators=loaded.locators,
                )
            return loaded

    monkeypatch.setattr(mod, "HarborEvaluator", MissingTraceEvaluator)
    with _scorer(tmp_path) as scorer, pytest.raises(ValueError, match="missing its WMH trace"):
        scorer.score(pi_node_baseline("candidate"), request=_full_request())

    MissingTraceEvaluator.candidate_failure = True
    with _scorer(tmp_path) as scorer, pytest.raises(ValueError, match="missing its WMH trace"):
        scorer.score(pi_node_baseline("candidate"), request=_full_request())

    MissingTraceEvaluator.candidate_failure = False
    MissingTraceEvaluator.task_timeout = True
    with _scorer(tmp_path) as scorer:
        report = scorer.score(pi_node_baseline("candidate"), request=_full_request())
    assert report.score == 0.0
    assert "trace unavailable" in report.per_task["alpha"].evidence


def test_candidate_compute_envelope_is_frozen_while_source_may_change(tmp_path: Path) -> None:
    baseline = pi_node_baseline("baseline")
    scorer = _scorer(tmp_path)
    code = baseline.code_files()[0]
    changed_code = code.model_copy(update={"content": f"{code.content}\n// candidate edit\n"})
    source_changed = HarnessDoc(
        name="source-changed",
        surfaces=[
            changed_code if surface.id == code.id else surface for surface in baseline.surfaces
        ],
    )
    assert scorer.validate_candidate(source_changed) is None

    max_turns = baseline.surface("param:max-turns")
    assert max_turns is not None
    compute_changed = HarnessDoc(
        name="compute-changed",
        surfaces=[
            Surface(
                id=max_turns.id,
                kind=SurfaceKind.PARAM,
                content=str(baseline.max_turns() + 1),
            )
            if surface.id == max_turns.id
            else surface
            for surface in baseline.surfaces
        ],
    )
    reason = scorer.validate_candidate(compute_changed)
    assert reason is not None
    assert "compute envelope" in reason
    scorer.close()


def test_e2b_score_plan_accepts_trusted_exact_build_environment_evidence(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path, backend=HarborEnvironmentBackend.E2B)
    authority = ExternalDispatchRateAuthority.bootstrap(
        (tmp_path / "rate.json").resolve(),
        E2B_SANDBOX_CREATE_RATE_POLICY,
    )
    task_class = ExactE2BEnvironment._task_resource_class(cpu_count=2, memory_mb=1024)

    plan = mod.harbor_harness_score_plan(
        job_spec=spec,
        provider_config=_provider(),
        reference_harness=pi_node_baseline("baseline"),
        qualified_tasks=_e2b_qualified_tasks(task_class),
        reward_key="reward",
        create_rate_binding=authority.binding,
    )

    assert plan.environment_backend is HarborEnvironmentBackend.E2B
    assert plan.task_environment_digests == _TASK_ENVIRONMENT_DIGESTS
    assert plan.task_resource_classes == (task_class, task_class)


def test_e2b_score_plan_rejects_local_task_qualification(tmp_path: Path) -> None:
    spec = _spec(tmp_path, backend=HarborEnvironmentBackend.E2B)
    authority = ExternalDispatchRateAuthority.bootstrap(
        (tmp_path / "rate.json").resolve(),
        E2B_SANDBOX_CREATE_RATE_POLICY,
    )

    with pytest.raises(ValueError, match="task backends differ"):
        mod.harbor_harness_score_plan(
            job_spec=spec,
            provider_config=_provider(),
            reference_harness=pi_node_baseline("baseline"),
            qualified_tasks=_qualified_tasks(),
            reward_key="reward",
            create_rate_binding=authority.binding,
        )


def test_e2b_score_plan_rejects_conflicting_identity_for_one_build_config(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path, backend=HarborEnvironmentBackend.E2B)
    authority = ExternalDispatchRateAuthority.bootstrap(
        (tmp_path / "rate.json").resolve(),
        E2B_SANDBOX_CREATE_RATE_POLICY,
    )
    task_class = ExactE2BEnvironment._task_resource_class(cpu_count=2, memory_mb=1024)
    tasks = list(_e2b_qualified_tasks(task_class))
    identity = tasks[1].e2b_build_identity
    assert identity is not None
    tasks[1] = tasks[1].model_copy(
        update={
            "e2b_build_identity": identity.model_copy(
                update={"template_id": "replacement-template"}
            )
        }
    )

    with pytest.raises(ValueError, match="conflicting immutable builds"):
        mod.harbor_harness_score_plan(
            job_spec=spec,
            provider_config=_provider(),
            reference_harness=pi_node_baseline("baseline"),
            qualified_tasks=tuple(tasks),
            reward_key="reward",
            create_rate_binding=authority.binding,
        )


def test_runner_cleanup_on_success_error_and_construction_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[FakeRunner] = []

    class FakeRunner:
        def __init__(self) -> None:
            self.closed = False
            created.append(self)

        def run(
            self,
            coroutine: Coroutine[object, object, LoadedHarborJobResult],
        ) -> LoadedHarborJobResult:
            return asyncio.run(coroutine)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(mod, "_AsyncRunner", FakeRunner)
    _install_fake_evaluator(monkeypatch, tmp_path)
    with _scorer(tmp_path) as scorer:
        scorer.score(pi_node_baseline("candidate"), request=_full_request())
        assert created[-1].closed is False
        scorer.before_proposal_batch()
        assert created[-1].closed is True
        scorer.score(pi_node_baseline("candidate"), request=_full_request())
    assert created[-1].closed is True

    class FailingEvaluator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def evaluate(self, _candidate: HarnessDoc) -> LoadedHarborJobResult:
            raise RuntimeError("evaluation failed")

    monkeypatch.setattr(mod, "HarborEvaluator", FailingEvaluator)
    with _scorer(tmp_path) as scorer, pytest.raises(RuntimeError, match="evaluation failed"):
        scorer.score(pi_node_baseline("candidate"), request=_full_request())
    assert created[-1].closed is True

    _install_fake_evaluator(
        monkeypatch,
        tmp_path,
        result_options=_ResultOptions(status=BenchmarkTrialStatus.INFRASTRUCTURE_ERROR),
    )
    scorer = _scorer(tmp_path)
    with pytest.raises(ValueError, match="not scored"):
        scorer.score(pi_node_baseline("candidate"), request=_full_request())
    assert created[-1].closed is True
    scorer.close()

    before = len(created)
    invalid_spec = _spec(tmp_path).model_copy(
        update={"datasets": [DatasetConfig(path=tmp_path / "dataset")]}
    )
    with pytest.raises(ValueError, match="explicit task_names"):
        _scorer(tmp_path, job_spec=invalid_spec)
    assert len(created) == before


def test_constructor_requires_exact_explicit_unique_task_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly match task_ids"):
        _scorer(tmp_path, task_ids=("beta", "alpha"))
    with pytest.raises(ValueError, match="unique"):
        _scorer(tmp_path, task_ids=("alpha", "alpha"))
    with pytest.raises(ValueError, match="one key per task_id"):
        _scorer(tmp_path, task_keys=(_TASK_KEYS[0],))
    with pytest.raises(ValueError, match="one digest per task_id"):
        _scorer(tmp_path, task_environment_digests=(_TASK_ENVIRONMENT_DIGESTS[0],))
    with pytest.raises(ValueError, match="pattern"):
        _scorer(tmp_path, task_keys=(_TASK_KEYS[0], "not-a-digest"))
    remote = _spec(tmp_path).model_copy(
        update={
            "datasets": [
                DatasetConfig(
                    repo="https://example.invalid/tasks.git",
                    task_names=list(_TASK_IDS),
                )
            ]
        }
    )
    with pytest.raises(ValueError, match="preflightable local dataset"):
        _scorer(tmp_path, job_spec=remote)
    wildcard = _spec(tmp_path).model_copy(
        update={"datasets": [DatasetConfig(path=tmp_path / "dataset", task_names=["alpha*"])]}
    )
    with pytest.raises(ValueError, match="glob"):
        _scorer(
            tmp_path,
            job_spec=wildcard,
            task_ids=("alpha*",),
            task_keys=(_TASK_KEYS[0],),
            task_environment_digests=(_TASK_ENVIRONMENT_DIGESTS[0],),
        )


def test_closed_scorer_rejects_further_use(tmp_path: Path) -> None:
    scorer = _scorer(tmp_path)
    scorer.close()
    scorer.close()
    with pytest.raises(RuntimeError, match="closed"):
        scorer.score(pi_node_baseline("candidate"), request=_full_request())


def test_scoring_from_running_event_loop_fails_without_leaking_coroutine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_evaluator(monkeypatch, tmp_path)
    scorer = _scorer(tmp_path)

    async def call_sync_api() -> None:
        with pytest.raises(RuntimeError, match="running event loop"):
            scorer.score(pi_node_baseline("candidate"), request=_full_request())

    asyncio.run(call_sync_api())
    scorer.close()
