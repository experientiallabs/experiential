"""Offline tests for the reusable Harbor evaluator."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import pytest
from harbor.job import Job
from harbor.metrics.uv_script import UvScript
from harbor.models.agent.context import AgentContext
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import DatasetConfig, JobConfig
from harbor.models.job.lock import JobLock
from harbor.models.job.result import JobResult, JobStats
from harbor.models.metric.config import MetricConfig
from harbor.models.metric.type import MetricType
from harbor.models.registry import DatasetMetadata
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, ModelInfo, TrialResult
from harbor.models.verifier.result import VerifierResult
from harbor.utils.logger import logger as harbor_logger

import wmh.evals.harbor.evaluator as mod
from wmh.evals.benchmark import BenchmarkCell, BenchmarkTaskEnvironment
from wmh.evals.harbor import _file_lease
from wmh.evals.harbor.config import (
    HarborEnvironmentBackend,
    HarborJobSpec,
    build_harbor_job_config,
)
from wmh.evals.harbor.e2b_environment import (
    ExactE2BEnvironment,
    freeze_exact_e2b_build_spec,
    register_exact_e2b_build_record,
)
from wmh.evals.harbor.results import (
    HarborTrialManifest,
    HarborTrialManifestEntry,
    LoadedHarborJobResult,
    harbor_agent_config_digest,
    harbor_trial_lock_digest,
)
from wmh.harness.e2b_sandbox import E2B_API_KEY_ENV
from wmh.harness.pi_runner import pi_node_baseline
from wmh.harness.pi_runner_backend import (
    E2BPiRunnerSpec,
    LocalPiRunnerSpec,
    runner_owner_id,
)
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.tracking._testing import (
    synthetic_provider_cost_meter,
    synthetic_tariff_provenance,
)
from wmh.tracking.budget import (
    BudgetAccount,
    BudgetPolicy,
    BudgetScope,
    ProviderCostMeter,
    TimedResourceBudgetAccount,
    TimedResourceCostMeter,
    bootstrap_budget_ledger,
)
from wmh.tracking.rate_limit import (
    ExternalDispatchRateAuthority,
    ExternalDispatchRatePolicy,
    bind_external_dispatch_rate_authority,
)

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
_LOCAL_RUNNER = LocalPiRunnerSpec()


class _E2BBudgetKwargs(TypedDict):
    budget_account: BudgetAccount
    task_resource_budget_accounts: NotRequired[tuple[TimedResourceBudgetAccount, ...]]
    runner_resource_budget_account: NotRequired[TimedResourceBudgetAccount]
    create_rate_authority: NotRequired[ExternalDispatchRateAuthority]


def _runner_lease_receipt(trial_name: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "backend": "local",
        "lease_id": f"lease-{trial_name}",
        "owner_id": runner_owner_id(trial_name),
        "config_digest": _LOCAL_RUNNER.config_digest,
        "state": "retired",
        "resource_id": f"container-{trial_name}",
        "created_at": "2026-07-18T11:59:00Z",
        "expected_end_at": None,
        "retired_at": "2026-07-18T12:00:00Z",
    }


def _e2b_runner(*, lease_timeout_s: int = 420) -> E2BPiRunnerSpec:
    return E2BPiRunnerSpec(
        template_id="template-immutable",
        build_id="build-immutable",
        cpu_count=2,
        memory_mb=2048,
        platform="linux/x86_64",
        envd_version="0.2.1",
        lease_timeout_s=lease_timeout_s,
    )


def _rate_policy() -> ExternalDispatchRatePolicy:
    return ExternalDispatchRatePolicy(
        provider="e2b",
        operation="sandbox_create",
        maximum_dispatches=4,
        period_milliseconds=1000,
    )


def _rate_spec(spec: HarborJobSpec) -> HarborJobSpec:
    return spec.model_copy(update={"create_rate_policy": _rate_policy()}, deep=True)


def _write_task(
    dataset_dir: Path,
    task_name: str = "shared-task",
    *,
    separate_verifier: bool = False,
    multi_step: bool = False,
) -> Path:
    task_dir = dataset_dir / task_name
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    task_config = f'[environment]\ndocker_image = "example.invalid/{task_name}:frozen"\n'
    if separate_verifier:
        task_config += '[verifier]\nenvironment_mode = "separate"\n'
    if multi_step:
        task_config += '[[steps]]\nname = "first"\n\n[[steps]]\nname = "second"\n'
        for step_name in ("first", "second"):
            step_dir = task_dir / "steps" / step_name
            step_dir.mkdir(parents=True)
            (step_dir / "instruction.md").write_text(
                f"Complete {step_name}.\n",
                encoding="utf-8",
            )
    (task_dir / "task.toml").write_text(task_config, encoding="utf-8")
    (task_dir / "instruction.md").write_text("Solve the task.\n", encoding="utf-8")
    (task_dir / "environment" / "Dockerfile").write_text("FROM alpine:3.20\n", encoding="utf-8")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    return task_dir


def _spec(tmp_path: Path, *datasets: Path, job_name: str = "evaluation") -> HarborJobSpec:
    return HarborJobSpec(
        job_name=job_name,
        jobs_dir=tmp_path / "jobs",
        datasets=[DatasetConfig(path=dataset) for dataset in datasets],
        n_attempts=2,
        n_concurrent_trials=2,
    )


def _provider() -> ProviderConfig:
    return ProviderConfig(
        kind=ProviderKind.BEDROCK,
        model="model",
        region="test-region",
    )


def _e2b_budget_kwargs(
    tmp_path: Path,
    *,
    task_environment: bool = False,
    runner_spec: E2BPiRunnerSpec | None = None,
) -> _E2BBudgetKwargs:
    from wmh.harness.pi_runner_backend import e2b_runner_resource_class

    provider = _provider()
    meters: dict[str, ProviderCostMeter | TimedResourceCostMeter] = {
        "worker": synthetic_provider_cost_meter(
            provider_config=provider,
            provenance=synthetic_tariff_provenance(provider),
            input_nano_usd_per_token=1,
            output_nano_usd_per_token=1,
        )
    }
    if task_environment:
        task_class = ExactE2BEnvironment._task_resource_class(
            cpu_count=2,
            memory_mb=1024,
        )
        meters["task"] = TimedResourceCostMeter(
            resource_type=task_class.role.value,
            resource_class_digest=task_class.digest,
            nano_usd_per_second=1,
            max_billing_seconds=task_class.max_host_observation_seconds,
        )
    if runner_spec is not None:
        runner_class = e2b_runner_resource_class(runner_spec)
        meters["runner"] = TimedResourceCostMeter(
            resource_type=runner_class.role.value,
            resource_class_digest=runner_class.digest,
            nano_usd_per_second=1,
            max_billing_seconds=runner_class.max_host_observation_seconds,
        )
    policy = BudgetPolicy(
        study_id="test-study",
        manifest_digest="sha256:" + hashlib.sha256(str(tmp_path).encode()).hexdigest(),
        hard_limit_nano_usd=10_000_000,
        phase_limits_nano_usd={"test": 10_000_000},
        meters=meters,
    )
    scope = BudgetScope(phase="test", category="test", run_id="test-run")
    ledger_path = (tmp_path / "budget.sqlite3").resolve()
    ledger_identity = bootstrap_budget_ledger(ledger_path, policy).ledger_identity
    kwargs: _E2BBudgetKwargs = {
        "budget_account": BudgetAccount(
            ledger_path=ledger_path,
            ledger_identity=ledger_identity,
            policy=policy,
            scope=scope,
            meter_id="worker",
        ),
        "create_rate_authority": ExternalDispatchRateAuthority.bootstrap(
            (tmp_path / "e2b-create-rate.json").resolve(),
            _rate_policy(),
        ),
    }
    if task_environment:
        kwargs["task_resource_budget_accounts"] = (
            TimedResourceBudgetAccount(
                ledger_path=ledger_path,
                ledger_identity=ledger_identity,
                policy=policy,
                scope=scope,
                meter_id="task",
            ),
        )
    if runner_spec is not None:
        kwargs["runner_resource_budget_account"] = TimedResourceBudgetAccount(
            ledger_path=ledger_path,
            ledger_identity=ledger_identity,
            policy=policy,
            scope=scope,
            meter_id="runner",
        )
    return kwargs


def _register_e2b_task_build(tmp_path: Path, task_dir: Path) -> None:
    spec = freeze_exact_e2b_build_spec(
        environment_dir=task_dir / "environment",
        docker_image=f"example.invalid/{task_dir.name}:frozen",
        cpu_count=2,
        memory_mb=1024,
    )
    register_exact_e2b_build_record(
        jobs_dir=tmp_path / "jobs",
        environment_id=spec.environment_id,
        build_context_digest=spec.build_context_digest,
        docker_image=spec.docker_image,
        template_id="task-template",
        build_id="task-build",
        cpu_count=2,
        memory_mb=1024,
        acknowledge_preexisting_outside_study=True,
    )


@pytest.fixture(autouse=True)
def _stub_runner_readiness_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "verify_container_pi_runner_ready", lambda **_kwargs: None)
    monkeypatch.setattr(mod, "find_spec", lambda _name: object())
    monkeypatch.setattr(mod.EnvironmentFactory, "run_preflight", lambda **_kwargs: None)


@pytest.mark.parametrize("runner_image", ["node:latest", ""])
def test_evaluator_rejects_mutable_runner_spec_before_job_creation(
    tmp_path: Path,
    runner_image: str,
) -> None:
    with pytest.raises(ValueError, match="digest-qualified"):
        mod.HarborEvaluator(
            _spec(tmp_path, tmp_path / "dataset"),
            _provider(),
            runner_spec={"backend": "local", "image": runner_image},
        )


@pytest.mark.parametrize("turn_timeout_s", [float("nan"), float("inf"), float("-inf")])
def test_evaluator_rejects_non_finite_turn_timeout(
    tmp_path: Path,
    turn_timeout_s: float,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        mod.HarborEvaluator(
            _spec(tmp_path, tmp_path / "dataset"),
            _provider(),
            turn_timeout_s=turn_timeout_s,
        )


def test_evaluator_requires_fixed_e2b_lease_to_cover_the_turn(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lease_timeout_s"):
        mod.HarborEvaluator(
            _spec(tmp_path, tmp_path / "dataset"),
            _provider(),
            runner_spec=_e2b_runner(lease_timeout_s=300),
            turn_timeout_s=300,
        )


def test_evaluator_admits_hour_turn_with_long_e2b_runner_lease(tmp_path: Path) -> None:
    runner_spec = _e2b_runner(lease_timeout_s=7_200)
    evaluator = mod.HarborEvaluator(
        _rate_spec(_spec(tmp_path, tmp_path / "dataset")),
        _provider(),
        runner_spec=runner_spec,
        turn_timeout_s=3_600,
        **_e2b_budget_kwargs(tmp_path, runner_spec=runner_spec),
    )

    assert isinstance(evaluator._runner_spec, E2BPiRunnerSpec)
    assert evaluator._runner_spec.lease_timeout_s == 7_200


def test_evaluator_rejects_hour_turn_without_cleanup_margin(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lease_timeout_s"):
        mod.HarborEvaluator(
            _spec(tmp_path, tmp_path / "dataset"),
            _provider(),
            runner_spec=_e2b_runner(lease_timeout_s=3_659),
            turn_timeout_s=3_600,
        )


def test_e2b_runner_readiness_checks_only_local_prerequisites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(E2B_API_KEY_ENV, "present-but-never-read-by-the-test")
    monkeypatch.setattr(mod, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        mod,
        "verify_container_pi_runner_ready",
        lambda **_kwargs: pytest.fail("E2B runner readiness must not probe Docker"),
    )
    evaluator = mod.HarborEvaluator(
        _rate_spec(_spec(tmp_path, tmp_path / "dataset")),
        _provider(),
        runner_spec=_e2b_runner(),
        **_e2b_budget_kwargs(tmp_path, runner_spec=_e2b_runner()),
    )

    asyncio.run(evaluator._ensure_runner_ready())
    assert evaluator._runner_ready is True


def test_e2b_runner_missing_host_credential_fails_before_job_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(E2B_API_KEY_ENV, raising=False)
    monkeypatch.setattr(mod, "find_spec", lambda _name: object())
    evaluator = mod.HarborEvaluator(
        _rate_spec(_spec(tmp_path, tmp_path / "dataset")),
        _provider(),
        runner_spec=_e2b_runner(),
        **_e2b_budget_kwargs(tmp_path, runner_spec=_e2b_runner()),
    )

    with pytest.raises(RuntimeError, match=E2B_API_KEY_ENV):
        asyncio.run(evaluator._ensure_runner_ready())


def test_e2b_task_and_runner_backends_reject_unbudgeted_programmatic_entry(
    tmp_path: Path,
) -> None:
    task_spec = _spec(tmp_path, tmp_path / "dataset").model_copy(
        update={
            "environment_backend": HarborEnvironmentBackend.E2B,
            "create_rate_policy": _rate_policy(),
        },
        deep=True,
    )
    with pytest.raises(ValueError, match="E2B task environments require"):
        mod.HarborEvaluator(task_spec, _provider())

    with pytest.raises(ValueError, match="E2B Pi runners require"):
        mod.HarborEvaluator(
            _spec(tmp_path, tmp_path / "dataset"),
            _provider(),
            runner_spec=_e2b_runner(),
        )


def test_exact_e2b_build_preflight_fails_before_atomic_job_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)
    spec = _spec(tmp_path, dataset).model_copy(
        update={
            "environment_backend": HarborEnvironmentBackend.E2B,
            "create_rate_policy": _rate_policy(),
        },
        deep=True,
    )
    creates = 0

    async def unexpected_create(_cls: type[Job], _config: JobConfig) -> Job:
        nonlocal creates
        creates += 1
        raise AssertionError("exact build admission must precede Harbor job creation")

    monkeypatch.setattr(mod._AtomicHarborJob, "create", classmethod(unexpected_create))
    evaluator = mod.HarborEvaluator(
        spec,
        _provider(),
        **_e2b_budget_kwargs(tmp_path, task_environment=True),
    )

    with pytest.raises(RuntimeError, match="prebuilt exact template record"):
        asyncio.run(evaluator.evaluate(pi_node_baseline("candidate")))

    assert creates == 0
    assert not (tmp_path / "jobs" / "evaluation").exists()


@pytest.mark.parametrize(
    ("resource_config", "resource_name"),
    [
        ("gpus = 1\n", "GPU"),
        ('gpu_types = ["H100"]\n', "GPU"),
        ('tpu = {type = "v4", topology = "2x2"}\n', "TPU"),
    ],
)
def test_exact_e2b_rejects_unsupported_accelerators_before_build_lookup_or_job_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource_config: str,
    resource_name: str,
) -> None:
    dataset = tmp_path / "dataset"
    task_dir = _write_task(dataset)
    config_path = task_dir / "task.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + resource_config,
        encoding="utf-8",
    )
    spec = _spec(tmp_path, dataset).model_copy(
        update={
            "environment_backend": HarborEnvironmentBackend.E2B,
            "create_rate_policy": _rate_policy(),
        },
        deep=True,
    )

    def unexpected_build_lookup(**_kwargs: object) -> None:
        raise AssertionError("accelerator rejection must precede exact-build lookup")

    async def unexpected_create(_cls: type[Job], _config: JobConfig) -> Job:
        raise AssertionError("accelerator rejection must precede Harbor job creation")

    monkeypatch.setattr(mod, "require_exact_e2b_build_record", unexpected_build_lookup)
    monkeypatch.setattr(mod._AtomicHarborJob, "create", classmethod(unexpected_create))
    evaluator = mod.HarborEvaluator(
        spec,
        _provider(),
        **_e2b_budget_kwargs(tmp_path, task_environment=True),
    )

    with pytest.raises(mod.UnsupportedHarborTaskError, match=resource_name):
        asyncio.run(evaluator.evaluate(pi_node_baseline("candidate")))

    assert not (tmp_path / "jobs" / "evaluation").exists()


def test_exact_e2b_preflight_checks_only_explicitly_selected_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset, "selected")
    _write_task(dataset, "unselected")
    config = JobConfig(
        job_name="selected-only",
        jobs_dir=tmp_path / "jobs",
        datasets=[DatasetConfig(path=dataset, task_names=["selected"])],
        environment=EnvironmentConfig(type=EnvironmentType.E2B),
    )
    account = _e2b_budget_kwargs(tmp_path, task_environment=True)["task_resource_budget_accounts"][
        0
    ]
    observed: list[str] = []

    def freeze_build(*, environment_dir: Path, **_kwargs: object) -> object:
        task_name = environment_dir.parent.name

        class BuildSpec:
            environment_id = task_name
            build_context_digest = "sha256:" + "a" * 64
            docker_image = f"example.invalid/{task_name}:frozen"

        return BuildSpec()

    def require_build(*, environment_id: str, **_kwargs: object) -> None:
        observed.append(environment_id)

    monkeypatch.setattr(mod, "freeze_exact_e2b_build_spec", freeze_build)
    monkeypatch.setattr(mod, "require_exact_e2b_build_record", require_build)

    mod._preflight_exact_e2b_builds(config, (account,))

    assert observed == ["selected"]


def test_direct_executable_metric_is_rejected_without_constructing_harbor_job(
    tmp_path: Path,
) -> None:
    config = JobConfig(
        job_name="evaluation",
        jobs_dir=tmp_path / "jobs",
        datasets=[DatasetConfig(path=tmp_path / "dataset")],
        metrics=[
            MetricConfig(
                type=MetricType.UV_SCRIPT,
                kwargs={"script_path": str(tmp_path / "malicious.py")},
            )
        ],
    )

    with pytest.raises(
        mod.UnsupportedHarborMetricError,
        match="credential-bearing host",
    ):
        asyncio.run(mod._reject_executable_harbor_metrics(config))

    assert not (tmp_path / "jobs" / "evaluation").exists()


@pytest.mark.parametrize(
    ("dataset", "expected_name"),
    [
        (DatasetConfig(name="benchmark", version="v1"), "benchmark@v1"),
        (DatasetConfig(repo="owner/repository", path=Path("datasets/benchmark")), ""),
        (
            DatasetConfig(
                repo="owner/repository",
                name="benchmark",
                version="v1",
                registry_path=Path("registry.json"),
            ),
            "benchmark@v1",
        ),
    ],
)
def test_remote_dataset_executable_metric_is_rejected_from_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: DatasetConfig,
    expected_name: str,
) -> None:
    names: list[str] = []
    metadata = DatasetMetadata(
        name="benchmark",
        task_ids=[],
        metrics=[MetricConfig(type=MetricType.UV_SCRIPT)],
    )

    class FakeRegistryClient:
        async def get_dataset_metadata(self, name: str) -> DatasetMetadata:
            names.append(name)
            return metadata

    monkeypatch.setattr(
        mod.RegistryClientFactory,
        "create",
        lambda **_kwargs: FakeRegistryClient(),
    )
    config = JobConfig(
        job_name="evaluation",
        jobs_dir=tmp_path / "jobs",
        datasets=[dataset],
    )

    with pytest.raises(mod.UnsupportedHarborMetricError, match="'uv-script'"):
        asyncio.run(mod._reject_executable_harbor_metrics(config))

    assert names == [expected_name]


def test_remote_package_dataset_cannot_reach_metadata_provider_or_e2b_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    leaked_values: list[str] = []

    class FakePackageClient:
        async def get_dataset_metadata(self, name: str) -> DatasetMetadata:
            events.append(f"metadata:{name}")
            raise AssertionError("remote dataset metadata must not be resolved")

    async def unexpected_create(_cls: type[Job], _config: JobConfig) -> Job:
        events.append("job-create")
        leaked_values.extend([os.environ["AZURE_OPENAI_API_KEY"], os.environ["E2B_API_KEY"]])
        raise AssertionError("remote datasets must fail before Harbor job creation")

    def unexpected_task_environment(_config: JobConfig) -> None:
        events.append("task-environment")
        raise AssertionError("remote dataset rejection must precede task-environment setup")

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-provider-secret")
    monkeypatch.setenv("E2B_API_KEY", "e2b-secret")
    monkeypatch.setattr(mod, "PackageDatasetClient", FakePackageClient)
    monkeypatch.setattr(mod._AtomicHarborJob, "create", classmethod(unexpected_create))
    monkeypatch.setattr(mod, "_preflight_task_environment", unexpected_task_environment)
    spec = _spec(tmp_path, tmp_path / "unused").model_copy(
        update={
            "datasets": [DatasetConfig(name="owner/benchmark", ref="sha256:dataset")],
            "environment_backend": HarborEnvironmentBackend.E2B,
            "create_rate_policy": _rate_policy(),
        },
        deep=True,
    )

    with pytest.raises(mod.UnsupportedHarborTaskError, match="local dataset paths"):
        asyncio.run(
            mod.HarborEvaluator(
                spec,
                _provider(),
                **_e2b_budget_kwargs(tmp_path, task_environment=True),
            ).evaluate(pi_node_baseline("candidate"))
        )

    assert events == []
    assert leaked_values == []
    assert not (tmp_path / "jobs" / "evaluation").exists()


def test_resolved_metric_recheck_closes_harbor_job_mutation_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    dataset = tmp_path / "dataset"
    _write_task(dataset)
    metric_path = tmp_path / "mutated-metric.py"
    metric_path.write_text("raise AssertionError('must never execute')\n", encoding="utf-8")

    class MutatedJob:
        _metrics = {"owner/benchmark": [UvScript(metric_path)]}

        async def run(self) -> None:
            events.append("run")
            raise AssertionError("resolved executable metric must fail before Harbor runs")

        def _close_logger_handlers(self) -> None:
            events.append("closed")

    async def create_mutated_job(_cls: type[Job], _config: JobConfig) -> Job:
        events.append("create:mutated")
        return cast("Job", MutatedJob())

    monkeypatch.setattr(mod._AtomicHarborJob, "create", classmethod(create_mutated_job))
    spec = _spec(tmp_path, dataset)

    with pytest.raises(mod.UnsupportedHarborMetricError, match="resolved executable"):
        asyncio.run(mod.HarborEvaluator(spec, _provider()).evaluate(pi_node_baseline("candidate")))

    assert events == ["create:mutated", "closed"]


def test_evaluate_pins_agent_persists_exact_lock_manifest_and_qualifies_task_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_a = tmp_path / "dataset-a"
    dataset_b = tmp_path / "dataset-b"
    _write_task(dataset_a)
    _write_task(dataset_b)
    candidate = pi_node_baseline("candidate")
    captured: dict[str, object] = {}
    sentinel = object()

    async def fake_run(job: Job) -> None:
        assert isinstance(job, mod._AtomicHarborJob)
        manifest_path = job.job_dir / mod._MANIFEST_FILENAME
        assert manifest_path.is_file(), "the trusted manifest must exist before Harbor runs"
        captured["manifest_at_run"] = HarborTrialManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        captured["trial_configs"] = [config.model_copy(deep=True) for config in job._trial_configs]
        captured["job_lock"] = mod._build_prepared_job_lock(job)
        captured["agent"] = job.config.agents[0].model_copy(deep=True)

    def fake_load(_job_dir: Path, manifest: HarborTrialManifest) -> LoadedHarborJobResult:
        captured["loaded_manifest"] = manifest
        return cast("LoadedHarborJobResult", sentinel)

    monkeypatch.setattr(Job, "run", fake_run)
    monkeypatch.setattr(mod, "load_harbor_job_result", fake_load)

    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset_a, dataset_b), _provider())
    result = asyncio.run(evaluator.evaluate(candidate))

    assert result is sentinel
    manifest = cast("HarborTrialManifest", captured["manifest_at_run"])
    assert captured["loaded_manifest"] == manifest
    assert candidate.doc_hash != candidate.execution_hash
    assert manifest.identity.candidate_hash == candidate.execution_hash
    assert manifest.identity.task_environment is BenchmarkTaskEnvironment.DOCKER
    assert len(manifest.entries) == 4
    assert {entry.cell.task_name for entry in manifest.entries} == {"shared-task"}
    assert {entry.task_source for entry in manifest.entries} == {"dataset-a", "dataset-b"}
    assert {entry.task_instruction for entry in manifest.entries} == {"Solve the task.\n"}
    assert len({entry.cell.task_key for entry in manifest.entries}) == 2
    for source in ("dataset-a", "dataset-b"):
        assert sorted(
            entry.cell.attempt for entry in manifest.entries if entry.task_source == source
        ) == [1, 2]

    trial_configs = cast("list[TrialConfig]", captured["trial_configs"])
    assert all(config.environment.type is EnvironmentType.DOCKER for config in trial_configs)
    job_lock = cast("JobLock", captured["job_lock"])
    expected_locks = {
        trial_config.trial_name: harbor_trial_lock_digest(trial_lock)
        for trial_config, trial_lock in zip(trial_configs, job_lock.trials, strict=True)
    }
    assert {
        entry.trial_name: entry.trial_lock_digest for entry in manifest.entries
    } == expected_locks
    assert {
        entry.trial_name: entry.cell.config_digest for entry in manifest.entries
    } == expected_locks

    agent = cast("AgentConfig", captured["agent"])
    assert manifest.identity.run_config_digest == mod.harbor_run_config_digest(
        evaluator._spec,
        harbor_agent_config_digest(agent),
    )
    assert agent.import_path == "wmh.evals.harbor.agent:WmhPiAgent"
    assert agent.model_name == "bedrock/model"
    assert agent.env == {}
    assert agent.kwargs == {
        "harness": candidate.model_dump(mode="json"),
        "provider_config": _provider().model_dump(mode="json"),
        "runner_spec": _LOCAL_RUNNER.model_dump(mode="json"),
        "turn_timeout_s": 300.0,
        "require_provider_receipts": True,
    }


def test_runner_readiness_failure_precedes_harbor_job_and_task_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    dataset = tmp_path / "dataset"
    _write_task(dataset)

    def fail_probe(*, image: str, platform: str) -> None:
        assert image == _LOCAL_RUNNER.image
        assert platform == _LOCAL_RUNNER.platform
        events.append("runner-probe")
        raise RuntimeError("runner unavailable")

    async def unexpected_create(_cls: type[Job], _config: object) -> Job:
        events.append("job-create")
        raise AssertionError("Harbor job creation must follow runner readiness")

    async def unexpected_run(_job: Job) -> None:
        events.append("job-run")
        raise AssertionError("Harbor task execution must follow runner readiness")

    def unexpected_task(*_args: object, **_kwargs: object) -> object:
        events.append("task-load")
        raise AssertionError("Harbor task loading must follow runner readiness")

    monkeypatch.setattr(mod, "verify_container_pi_runner_ready", fail_probe)
    monkeypatch.setattr(mod._AtomicHarborJob, "create", classmethod(unexpected_create))
    monkeypatch.setattr(Job, "run", unexpected_run)
    monkeypatch.setattr(mod, "Task", unexpected_task)
    evaluator = mod.HarborEvaluator(
        _spec(tmp_path, tmp_path / "dataset"),
        _provider(),
    )

    with pytest.raises(RuntimeError, match="runner unavailable"):
        asyncio.run(evaluator.evaluate(pi_node_baseline("candidate")))

    assert events == ["runner-probe"]
    assert not (tmp_path / "jobs" / "evaluation").exists()


@pytest.mark.parametrize("missing_module", ["e2b", "dockerfile_parse"])
def test_e2b_missing_extra_fails_before_runner_or_harbor_job_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_module: str,
) -> None:
    events: list[str] = []
    dataset = tmp_path / "dataset"
    _write_task(dataset)

    def unexpected_probe(**_kwargs: object) -> None:
        events.append("runner-probe")

    async def unexpected_create(_cls: type[Job], _config: object) -> Job:
        events.append("job-create")
        raise AssertionError("Harbor job creation must follow task-backend preflight")

    monkeypatch.setattr(
        mod,
        "find_spec",
        lambda name: None if name == missing_module else object(),
    )
    monkeypatch.setattr(mod, "verify_container_pi_runner_ready", unexpected_probe)
    monkeypatch.setattr(mod._AtomicHarborJob, "create", classmethod(unexpected_create))
    spec = _spec(tmp_path, dataset).model_copy(
        update={
            "environment_backend": HarborEnvironmentBackend.E2B,
            "create_rate_policy": _rate_policy(),
        },
        deep=True,
    )

    with pytest.raises(RuntimeError, match=f"require the WMH e2b extra.*{missing_module}"):
        asyncio.run(
            mod.HarborEvaluator(
                spec,
                _provider(),
                **_e2b_budget_kwargs(tmp_path, task_environment=True),
            ).evaluate(pi_node_baseline("candidate"))
        )

    assert events == []
    assert not (tmp_path / "jobs" / "evaluation").exists()


def test_e2b_missing_api_key_fails_before_runner_or_harbor_job_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    dataset = tmp_path / "dataset"
    _write_task(dataset)

    def fail_preflight(**_kwargs: object) -> None:
        events.append("task-preflight")
        raise SystemExit("E2B requires E2B_API_KEY to be set")

    def unexpected_probe(**_kwargs: object) -> None:
        events.append("runner-probe")

    async def unexpected_create(_cls: type[Job], _config: object) -> Job:
        events.append("job-create")
        raise AssertionError("Harbor job creation must follow task-backend preflight")

    monkeypatch.setattr(mod.EnvironmentFactory, "run_preflight", fail_preflight)
    monkeypatch.setattr(mod, "verify_container_pi_runner_ready", unexpected_probe)
    monkeypatch.setattr(mod._AtomicHarborJob, "create", classmethod(unexpected_create))
    spec = _spec(tmp_path, dataset).model_copy(
        update={
            "environment_backend": HarborEnvironmentBackend.E2B,
            "create_rate_policy": _rate_policy(),
        },
        deep=True,
    )

    with pytest.raises(RuntimeError, match="preflight failed: E2B requires E2B_API_KEY"):
        asyncio.run(
            mod.HarborEvaluator(
                spec,
                _provider(),
                **_e2b_budget_kwargs(tmp_path, task_environment=True),
            ).evaluate(pi_node_baseline("candidate"))
        )

    assert events == ["task-preflight"]
    assert not (tmp_path / "jobs" / "evaluation").exists()


def test_atomic_root_result_replace_preserves_the_last_valid_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "jobs" / "evaluation"
    job_dir.mkdir(parents=True)
    result_path = job_dir / "result.json"
    result_path.write_text('{"valid": true}\n', encoding="utf-8")
    temporary_paths: list[Path] = []

    def fail_replace(source: Path, _target: Path) -> None:
        temporary_paths.append(source)
        raise OSError("simulated rename failure")

    monkeypatch.setattr(mod.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated rename failure"):
        mod._atomic_replace_job_result(result_path, '{"replacement": true}\n')

    assert result_path.read_text(encoding="utf-8") == '{"valid": true}\n'
    assert len(temporary_paths) == 1
    assert temporary_paths[0].parent == job_dir.parent
    assert temporary_paths[0].name.startswith(".evaluation.result.json-")
    assert list(job_dir.glob(".result.json-*")) == []
    assert not temporary_paths[0].exists()


def test_atomic_job_persists_result_when_trial_results_are_excluded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Harbor's lightweight progress checkpoint must use the same atomic write path."""
    result_path = tmp_path / "result.json"
    writes: list[tuple[Path, str]] = []

    class Result:
        def model_dump_json(
            self,
            *,
            indent: int,
            exclude: set[str] | None = None,
        ) -> str:
            assert indent == 4
            assert exclude == {"trial_results"}
            return '{"progress": true}'

    class JobView:
        _job_result = Result()
        _job_result_path = result_path

    monkeypatch.setattr(
        mod,
        "_atomic_replace_job_result",
        lambda path, payload: writes.append((path, payload)),
    )

    mod.AtomicHarborJob._write_job_result(
        cast("mod.AtomicHarborJob", JobView()),
        exclude_trial_results=True,
    )

    assert writes == [(result_path, '{"progress": true}')]


def test_run_config_digest_binds_semantics_and_concurrency_but_not_paths(tmp_path: Path) -> None:
    spec = _spec(tmp_path, tmp_path / "dataset")
    agent_digest = "sha256:" + "a" * 64
    baseline = mod.harbor_run_config_digest(spec, agent_digest)

    semantic_variants = [
        spec.model_copy(update={"n_attempts": 3}, deep=True),
        spec.model_copy(
            update={
                "environment_backend": HarborEnvironmentBackend.E2B,
                "create_rate_policy": _rate_policy(),
            },
            deep=True,
        ),
        spec.model_copy(update={"n_concurrent_trials": 7}, deep=True),
        spec.model_copy(update={"agent_n_concurrent": 1}, deep=True),
    ]
    assert all(
        mod.harbor_run_config_digest(variant, agent_digest) != baseline
        for variant in semantic_variants
    )
    assert mod.harbor_run_config_digest(spec, "sha256:" + "b" * 64) != baseline

    storage_variant = spec.model_copy(
        update={
            "job_name": "renamed",
            "jobs_dir": tmp_path / "other-jobs",
        },
        deep=True,
    )
    assert mod.harbor_run_config_digest(storage_variant, agent_digest) == baseline

    artifact_variant = spec.model_copy(
        update={"artifact_paths": ["/logs/extra", "/workspace/report.json"]},
        deep=True,
    )
    assert mod.harbor_run_config_digest(artifact_variant, agent_digest) != baseline
    reversed_artifacts = artifact_variant.model_copy(
        update={"artifact_paths": list(reversed(artifact_variant.artifact_paths))},
        deep=True,
    )
    assert mod.harbor_run_config_digest(
        reversed_artifacts,
        agent_digest,
    ) != mod.harbor_run_config_digest(artifact_variant, agent_digest)

    rate_variant = spec.model_copy(update={"create_rate_policy": _rate_policy()}, deep=True)
    assert mod.harbor_run_config_digest(rate_variant, agent_digest) != baseline


def test_e2b_runner_agent_binding_is_path_free_and_not_logical_identity(tmp_path: Path) -> None:
    first = ExternalDispatchRateAuthority.bootstrap(tmp_path / "first-rate.json", _rate_policy())
    second = ExternalDispatchRateAuthority.bootstrap(tmp_path / "second-rate.json", _rate_policy())
    candidate = pi_node_baseline("candidate")
    runner = _e2b_runner()

    first_agent = mod._build_harbor_agent_config(
        candidate=candidate,
        provider_config=_provider(),
        runner_spec=runner,
        turn_timeout_s=300.0,
        agent_n_concurrent=1,
        require_provider_receipts=True,
        create_rate_binding=bind_external_dispatch_rate_authority(first),
    )
    second_agent = mod._build_harbor_agent_config(
        candidate=candidate,
        provider_config=_provider(),
        runner_spec=runner,
        turn_timeout_s=300.0,
        agent_n_concurrent=1,
        require_provider_receipts=True,
        create_rate_binding=bind_external_dispatch_rate_authority(second),
    )

    assert str(tmp_path) not in json.dumps(first_agent.kwargs)
    assert first_agent.kwargs["create_rate_binding"] != second_agent.kwargs["create_rate_binding"]
    assert harbor_agent_config_digest(first_agent) == harbor_agent_config_digest(second_agent)


def test_evaluator_shares_one_rate_authority_between_e2b_task_and_runner(
    tmp_path: Path,
) -> None:
    runner = _e2b_runner()
    rate_authority = ExternalDispatchRateAuthority.bootstrap(
        tmp_path / "e2b-rate.json",
        _rate_policy(),
    )
    spec = _spec(tmp_path, tmp_path / "dataset").model_copy(
        update={
            "environment_backend": HarborEnvironmentBackend.E2B,
            "create_rate_policy": _rate_policy(),
        },
        deep=True,
    )
    budget_kwargs = _e2b_budget_kwargs(
        tmp_path,
        task_environment=True,
        runner_spec=runner,
    )
    budget_kwargs["create_rate_authority"] = rate_authority
    evaluator = mod.HarborEvaluator(
        spec,
        _provider(),
        runner_spec=runner,
        **budget_kwargs,
    )

    agent = evaluator._build_agent(pi_node_baseline("candidate"))
    job_config = build_harbor_job_config(
        evaluator._spec,
        agent=agent,
        task_resource_budget_bindings=evaluator._task_resource_budget_bindings,
        create_rate_binding=evaluator._create_rate_binding,
    )

    expected = rate_authority.binding.model_dump(mode="json")
    assert agent.kwargs["create_rate_binding"] == expected
    assert job_config.environment.kwargs["create_rate_binding"] == expected


def test_evaluator_rejects_e2b_create_without_frozen_rate_authority(tmp_path: Path) -> None:
    spec = _spec(tmp_path, tmp_path / "dataset").model_copy(
        update={
            "environment_backend": HarborEnvironmentBackend.E2B,
            "create_rate_policy": _rate_policy(),
        },
        deep=True,
    )
    budget_kwargs = _e2b_budget_kwargs(tmp_path, task_environment=True)
    del budget_kwargs["create_rate_authority"]

    with pytest.raises(ValueError, match="create-rate authority"):
        mod.HarborEvaluator(
            spec,
            _provider(),
            **budget_kwargs,
        )


def test_reasoning_effort_is_explicit_in_agent_config_and_run_identity(tmp_path: Path) -> None:
    provider = ProviderConfig(
        kind=ProviderKind.BEDROCK,
        model_type="claude-opus-4-6",
        model="us.anthropic.claude-opus-4-6-v1",
        reasoning_effort="max",
    )
    spec = _spec(tmp_path, tmp_path / "dataset")
    evaluator = mod.HarborEvaluator(spec, provider)
    candidate = pi_node_baseline("candidate")
    agent = evaluator._build_agent(candidate)
    agent_digest = harbor_agent_config_digest(agent)
    expectation = mod.harbor_run_expectation(
        candidate=candidate,
        spec=spec,
        provider_config=provider,
        turn_timeout_s=300.0,
    )
    identity = evaluator._run_identity(
        candidate,
        run_config_digest=expectation.identity.run_config_digest,
    )

    serialized_provider = agent.kwargs["provider_config"]
    assert isinstance(serialized_provider, dict)
    assert serialized_provider["reasoning_effort"] == "max"
    assert identity.reasoning_effort == "max"
    assert expectation.identity == identity

    unconfigured = mod.HarborEvaluator(
        _spec(tmp_path, tmp_path / "dataset"),
        provider.model_copy(update={"reasoning_effort": None}),
    )
    unconfigured_agent = unconfigured._build_agent(candidate)
    unconfigured_identity = unconfigured._run_identity(
        candidate,
        run_config_digest="sha256:" + "b" * 64,
    )
    serialized_unconfigured = unconfigured_agent.kwargs["provider_config"]
    assert isinstance(serialized_unconfigured, dict)
    assert serialized_unconfigured["reasoning_effort"] is None
    assert unconfigured_identity.reasoning_effort is None
    assert harbor_agent_config_digest(unconfigured_agent) != agent_digest


def test_evaluator_builds_and_hashes_the_final_agent_concurrency(tmp_path: Path) -> None:
    spec = _spec(tmp_path, tmp_path / "dataset").model_copy(
        update={"n_concurrent_trials": 4, "agent_n_concurrent": 2},
        deep=True,
    )
    evaluator = mod.HarborEvaluator(spec, _provider())

    agent = evaluator._build_agent(pi_node_baseline("candidate"))
    job_config = mod.build_harbor_job_config(evaluator._spec, agent=agent)

    assert agent.n_concurrent == 2
    assert job_config.agents[0] == agent
    assert harbor_agent_config_digest(job_config.agents[0]) == harbor_agent_config_digest(agent)


def test_evaluator_binds_path_free_budget_policy_into_agent_identity(tmp_path: Path) -> None:
    from wmh.tracking.budget import (
        BudgetAccount,
        BudgetPolicy,
        BudgetScope,
        bootstrap_budget_ledger,
    )

    provider_config = _provider()
    policy = BudgetPolicy(
        study_id="study",
        manifest_digest="sha256:" + "b" * 64,
        hard_limit_nano_usd=1_000_000,
        phase_limits_nano_usd={"search": 1_000_000},
        meters={
            "worker": synthetic_provider_cost_meter(
                provider_config=provider_config,
                provenance=synthetic_tariff_provenance(provider_config),
                input_nano_usd_per_token=1,
                output_nano_usd_per_token=5,
            )
        },
    )
    ledger_path = (tmp_path / "budget.sqlite3").resolve()
    account = BudgetAccount(
        ledger_path=ledger_path,
        ledger_identity=bootstrap_budget_ledger(ledger_path, policy).ledger_identity,
        policy=policy,
        scope=BudgetScope(
            phase="search",
            category="worker",
            run_id="candidate-1",
        ),
        meter_id="worker",
    )
    evaluator = mod.HarborEvaluator(
        _spec(tmp_path, tmp_path / "dataset"),
        provider_config,
        budget_account=account,
    )

    agent = evaluator._build_agent(pi_node_baseline("candidate"))

    serialized_binding = agent.kwargs["budget_binding"]
    assert serialized_binding == {
        "policy_digest": account.policy.policy_digest,
        "ledger_identity": account.ledger_identity,
        "scope": account.scope.model_dump(mode="json"),
        "meter_id": account.meter_id,
    }
    assert "ledger_path" not in json.dumps(serialized_binding)
    assert str(account.ledger_path) not in json.dumps(serialized_binding)
    assert agent.kwargs["budget_policy_digest"] == account.policy.policy_digest
    without_budget = mod.HarborEvaluator(
        _spec(tmp_path, tmp_path / "dataset"),
        _provider(),
    )._build_agent(pi_node_baseline("candidate"))
    assert harbor_agent_config_digest(agent) != harbor_agent_config_digest(without_budget)


def test_evaluator_runs_with_agent_concurrency_below_trial_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)
    candidate = pi_node_baseline("candidate")
    spec = _spec(tmp_path, dataset).model_copy(
        update={"n_concurrent_trials": 4, "agent_n_concurrent": 2},
        deep=True,
    )
    evaluator = mod.HarborEvaluator(spec, _provider())
    observed_concurrency: list[int | None] = []

    async def fake_run(job: Job) -> None:
        observed_concurrency.append(job.config.agents[0].n_concurrent)
        _materialize_completed_job(job, candidate.execution_hash)

    monkeypatch.setattr(Job, "run", fake_run)
    result = asyncio.run(evaluator.evaluate(candidate))

    assert observed_concurrency == [2]
    assert result.result.n_scored == result.result.expected_trials == 2


def test_evaluator_revalidates_and_rejects_bypassed_retry_configuration(
    tmp_path: Path,
) -> None:
    unsafe_spec = _spec(tmp_path, tmp_path / "dataset").model_copy(
        update={
            "max_retries": 1,
            "retry_exceptions": {"ApiRateLimitError"},
        },
        deep=True,
    )

    with pytest.raises(ValueError, match="unsupported.*attempt ledger"):
        mod.HarborEvaluator(unsafe_spec, _provider())


def test_e2b_backend_is_bound_into_harbor_config_and_run_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    task_dir = _write_task(dataset)
    _register_e2b_task_build(tmp_path, task_dir)
    candidate = pi_node_baseline("candidate")
    captured: dict[str, object] = {}
    sentinel = cast("LoadedHarborJobResult", object())

    async def fake_run(job: Job) -> None:
        captured["environment"] = job.config.environment.type
        captured["dataset_path"] = job.config.datasets[0].path
        captured["manifest"] = HarborTrialManifest.model_validate_json(
            (job.job_dir / mod._MANIFEST_FILENAME).read_text(encoding="utf-8")
        )

    monkeypatch.setattr(Job, "run", fake_run)
    monkeypatch.setattr(mod, "load_harbor_job_result", lambda *_args: sentinel)
    spec = _spec(tmp_path, dataset).model_copy(
        update={
            "environment_backend": HarborEnvironmentBackend.E2B,
            "create_rate_policy": _rate_policy(),
            "allow_preexisting_e2b_builds": True,
        }
    )

    result = asyncio.run(
        mod.HarborEvaluator(
            spec,
            _provider(),
            **_e2b_budget_kwargs(tmp_path, task_environment=True),
        ).evaluate(candidate)
    )

    assert result is sentinel
    assert captured["environment"] is EnvironmentType.E2B
    dataset_path = cast("Path", captured["dataset_path"])
    assert dataset_path != dataset.resolve()
    assert dataset_path.is_relative_to((tmp_path / "jobs" / mod._TASK_SNAPSHOT_ROOT).resolve())
    manifest = cast("HarborTrialManifest", captured["manifest"])
    assert manifest.identity.task_environment is BenchmarkTaskEnvironment.E2B


def _materialize_job(
    job: Job,
    candidate_hash: str,
    *,
    complete_names: set[str],
    incomplete_names: set[str] | None = None,
) -> HarborTrialManifest:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    job_lock = mod._build_prepared_job_lock(job)
    manifest = HarborTrialManifest.model_validate_json(
        (job.job_dir / mod._MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    entries = {entry.trial_name: entry for entry in manifest.entries}
    incomplete_names = incomplete_names or set()
    for trial_config, trial_lock in zip(job._trial_configs, job_lock.trials, strict=True):
        entry = entries[trial_config.trial_name]
        trial_dir = job.job_dir / trial_config.trial_name
        if trial_config.trial_name not in complete_names | incomplete_names:
            continue
        trial_dir.mkdir(exist_ok=True)
        (trial_dir / "config.json").write_text(
            trial_config.model_dump_json(indent=2), encoding="utf-8"
        )
        (trial_dir / "lock.json").write_text(
            trial_lock.model_dump_json(indent=2, exclude_none=True), encoding="utf-8"
        )
        if trial_config.trial_name not in complete_names:
            continue
        assert not (trial_dir / "result.json").exists()
        result = TrialResult(
            task_name=entry.cell.task_name,
            trial_name=trial_config.trial_name,
            trial_uri=trial_dir.resolve().as_uri(),
            task_id=trial_config.task.get_task_id(),
            source=entry.task_source,
            task_checksum=entry.task_checksum,
            config=trial_config,
            agent_info=AgentInfo(
                name="wmh-pi",
                version=mod.WMH_PI_AGENT_VERSION,
                model_info=ModelInfo(name="model", provider="bedrock"),
            ),
            agent_result=AgentContext(
                n_input_tokens=1,
                n_output_tokens=1,
                metadata={
                    "harness_hash": candidate_hash,
                    "runner_config_digest": _LOCAL_RUNNER.config_digest,
                    "runner_environment_digest": _LOCAL_RUNNER.attestation.digest,
                    "runner_environment_attestation": _LOCAL_RUNNER.attestation.evidence,
                    "runner_lease_receipt": _runner_lease_receipt(trial_config.trial_name),
                    "task_environment_digest": _TASK_ENVIRONMENT_DIGEST,
                    "task_environment_attestation": _TASK_ENVIRONMENT_ATTESTATION,
                    "run_health": "valid",
                },
            ),
            verifier_result=VerifierResult(rewards={"score": 1}),
            started_at=now,
            finished_at=now,
        )
        (trial_dir / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")

    (job.job_dir / "config.json").write_text(
        job.config.model_dump_json(indent=2, exclude_defaults=True), encoding="utf-8"
    )
    (job.job_dir / "lock.json").write_text(
        job_lock.model_dump_json(indent=2, exclude_none=True), encoding="utf-8"
    )
    trial_results = [
        TrialResult.model_validate_json(
            (job.job_dir / config.trial_name / "result.json").read_text(encoding="utf-8")
        )
        for config in job._trial_configs
        if (job.job_dir / config.trial_name / "result.json").is_file()
    ]
    job_result = JobResult(
        id=job.id,
        started_at=now,
        n_total_trials=len(job._trial_configs),
        stats=JobStats.from_trial_results(
            trial_results,
            n_total_trials=len(job._trial_configs),
        ),
        trial_results=trial_results,
    )
    if len(trial_results) == len(job._trial_configs):
        job_result.finished_at = now
    (job.job_dir / "result.json").write_text(
        job_result.model_dump_json(indent=2, exclude={"trial_results"}), encoding="utf-8"
    )
    return manifest


def _materialize_completed_job(job: Job, candidate_hash: str) -> None:
    _materialize_job(
        job,
        candidate_hash,
        complete_names={config.trial_name for config in job._trial_configs},
    )


def test_complete_matching_job_is_reused_without_rerunning_completed_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)
    candidate = pi_node_baseline("candidate")
    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset), _provider())
    readiness_configs: list[tuple[str, str]] = []

    def record_readiness(*, image: str, platform: str) -> None:
        readiness_configs.append((image, platform))

    monkeypatch.setattr(mod, "verify_container_pi_runner_ready", record_readiness)

    async def first_run(job: Job) -> None:
        _materialize_completed_job(job, candidate.execution_hash)

    monkeypatch.setattr(Job, "run", first_run)
    first = asyncio.run(evaluator.evaluate(candidate))
    assert first.result.n_scored == first.result.expected_trials == 2

    remaining: list[int] = []

    async def resumed_run(job: Job) -> None:
        remaining.append(len(job._remaining_trial_configs))

    monkeypatch.setattr(Job, "run", resumed_run)
    resumed = asyncio.run(evaluator.evaluate(candidate))

    assert remaining == [0]
    assert resumed.result == first.result
    assert readiness_configs == [(_LOCAL_RUNNER.image, _LOCAL_RUNNER.platform)]


def test_evaluator_session_shares_one_concurrent_runner_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)
    session = mod.HarborEvaluatorSession(runner_image=_LOCAL_RUNNER.image)
    evaluators = [
        mod.HarborEvaluator(
            _spec(tmp_path, dataset, job_name=f"evaluation-{index}"),
            _provider(),
            session=session,
        )
        for index in range(4)
    ]
    readiness_configs: list[tuple[str, str]] = []

    def record_readiness(*, image: str, platform: str) -> None:
        readiness_configs.append((image, platform))

    monkeypatch.setattr(mod, "verify_container_pi_runner_ready", record_readiness)

    async def exercise() -> None:
        await asyncio.gather(*(evaluator._ensure_runner_ready() for evaluator in evaluators))

    asyncio.run(exercise())

    assert readiness_configs == [(_LOCAL_RUNNER.image, _LOCAL_RUNNER.platform)]


def test_agent_runtime_version_change_rejects_stale_completed_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)
    candidate = pi_node_baseline("candidate")
    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset), _provider())

    async def first_run(job: Job) -> None:
        _materialize_completed_job(job, candidate.execution_hash)

    monkeypatch.setattr(Job, "run", first_run)
    asyncio.run(evaluator.evaluate(candidate))

    monkeypatch.setattr(mod, "WMH_PI_AGENT_VERSION", "next-runtime-version")
    with pytest.raises(mod.StaleHarborJobError, match="manifest"):
        asyncio.run(evaluator.evaluate(candidate))


@pytest.mark.parametrize("tamper", ["malformed", "mismatched"])
def test_invalid_existing_job_config_is_rejected_without_leaking_harbor_handlers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)
    candidate = pi_node_baseline("candidate")
    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset), _provider())

    async def first_run(job: Job) -> None:
        _materialize_completed_job(job, candidate.execution_hash)

    monkeypatch.setattr(Job, "run", first_run)
    first = asyncio.run(evaluator.evaluate(candidate))
    config_path = first.job_dir / "config.json"
    if tamper == "malformed":
        config_path.write_text("{", encoding="utf-8")
    else:
        existing = JobConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
        mismatched = existing.model_copy(update={"debug": not existing.debug})
        config_path.write_text(
            mismatched.model_dump_json(indent=2, exclude_defaults=True),
            encoding="utf-8",
        )

    handlers_before = tuple(harbor_logger.handlers)
    try:
        with pytest.raises(mod.StaleHarborJobError, match="config"):
            asyncio.run(evaluator.evaluate(candidate))
        assert tuple(harbor_logger.handlers) == handlers_before
    finally:
        for handler in tuple(harbor_logger.handlers):
            if handler not in handlers_before:
                harbor_logger.removeHandler(handler)
                handler.close()


def test_sibling_atomic_result_orphan_does_not_block_safe_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)
    candidate = pi_node_baseline("candidate")
    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset), _provider())

    async def first_run(job: Job) -> None:
        _materialize_completed_job(job, candidate.execution_hash)

    monkeypatch.setattr(Job, "run", first_run)
    first = asyncio.run(evaluator.evaluate(candidate))
    orphan = first.job_dir.parent / ".evaluation.result.json-interrupted"
    orphan.write_text("partial", encoding="utf-8")

    remaining: list[int] = []

    async def resumed_run(job: Job) -> None:
        remaining.append(len(job._remaining_trial_configs))

    monkeypatch.setattr(Job, "run", resumed_run)
    resumed = asyncio.run(evaluator.evaluate(candidate))

    assert remaining == [0]
    assert resumed.result == first.result
    assert orphan.read_text(encoding="utf-8") == "partial"


def test_cancelled_trial_is_preserved_as_terminal_on_same_job_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)
    candidate = pi_node_baseline("candidate")
    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset), _provider())
    cancelled_name = ""

    async def first_run(job: Job) -> None:
        nonlocal cancelled_name
        _materialize_completed_job(job, candidate.execution_hash)
        cancelled_name = sorted(config.trial_name for config in job._trial_configs)[0]
        result_path = job.job_dir / cancelled_name / "result.json"
        trial = TrialResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        trial.verifier_result = None
        trial.exception_info = ExceptionInfo(
            exception_type="CancelledError",
            exception_message="cancelled",
            exception_traceback="",
            occurred_at=datetime(2026, 7, 18, 12, 0, tzinfo=UTC),
        )
        result_path.write_text(trial.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.setattr(Job, "run", first_run)
    first = asyncio.run(evaluator.evaluate(candidate))
    cancelled_path = first.job_dir / cancelled_name / "result.json"
    cancelled_bytes = cancelled_path.read_bytes()

    remaining: list[int] = []

    async def resumed_run(job: Job) -> None:
        remaining.append(len(job._remaining_trial_configs))

    monkeypatch.setattr(Job, "run", resumed_run)
    resumed = asyncio.run(evaluator.evaluate(candidate))

    assert first.result.n_cancelled == resumed.result.n_cancelled == 1
    assert resumed.result.n_incomplete == 0
    assert resumed.result.is_complete is True
    assert remaining == [0]
    assert cancelled_path.read_bytes() == cancelled_bytes


def test_interrupted_job_resumes_only_missing_cell_with_original_manifest_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)
    candidate = pi_node_baseline("candidate")
    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset), _provider())
    planned: dict[str, str] = {}

    async def first_run(job: Job) -> None:
        complete_name, incomplete_name = sorted(config.trial_name for config in job._trial_configs)
        planned.update(complete=complete_name, incomplete=incomplete_name)
        _materialize_job(
            job,
            candidate.execution_hash,
            complete_names={complete_name},
            incomplete_names={incomplete_name},
        )

    monkeypatch.setattr(Job, "run", first_run)
    first = asyncio.run(evaluator.evaluate(candidate))
    assert first.result.n_scored == 1
    assert first.result.n_incomplete == 1

    manifest_path = first.job_dir / mod._MANIFEST_FILENAME
    manifest_before = manifest_path.read_bytes()
    completed_result_path = first.job_dir / planned["complete"] / "result.json"
    completed_result_before = completed_result_path.read_bytes()
    resumed_names: list[str] = []

    async def resumed_run(job: Job) -> None:
        resumed_names.extend(config.trial_name for config in job._remaining_trial_configs)
        _materialize_job(
            job,
            candidate.execution_hash,
            complete_names=set(resumed_names),
        )

    monkeypatch.setattr(Job, "run", resumed_run)
    resumed = asyncio.run(evaluator.evaluate(candidate))

    assert resumed_names == [planned["incomplete"]]
    assert resumed.result.n_scored == resumed.result.expected_trials == 2
    assert manifest_path.read_bytes() == manifest_before
    assert completed_result_path.read_bytes() == completed_result_before


def test_interrupted_job_rejects_tampered_incomplete_trial_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)
    candidate = pi_node_baseline("candidate")
    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset), _provider())
    planned: dict[str, str] = {}

    async def first_run(job: Job) -> None:
        complete_name, incomplete_name = sorted(config.trial_name for config in job._trial_configs)
        planned.update(complete=complete_name, incomplete=incomplete_name)
        _materialize_job(
            job,
            candidate.execution_hash,
            complete_names={complete_name},
            incomplete_names={incomplete_name},
        )

    monkeypatch.setattr(Job, "run", first_run)
    first = asyncio.run(evaluator.evaluate(candidate))
    completed_result_path = first.job_dir / planned["complete"] / "result.json"
    completed_result_before = completed_result_path.read_bytes()
    (first.job_dir / planned["incomplete"] / "lock.json").write_text("{}", encoding="utf-8")

    with pytest.raises(mod.StaleHarborJobError, match="invalid lock"):
        asyncio.run(evaluator.evaluate(candidate))

    assert completed_result_path.read_bytes() == completed_result_before


def test_interrupted_job_rejects_malformed_completed_result_instead_of_overwriting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)
    candidate = pi_node_baseline("candidate")
    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset), _provider())
    completed_name = ""

    async def first_run(job: Job) -> None:
        nonlocal completed_name
        complete_name, incomplete_name = sorted(config.trial_name for config in job._trial_configs)
        completed_name = complete_name
        _materialize_job(
            job,
            candidate.execution_hash,
            complete_names={complete_name},
            incomplete_names={incomplete_name},
        )

    monkeypatch.setattr(Job, "run", first_run)
    first = asyncio.run(evaluator.evaluate(candidate))
    malformed_result = first.job_dir / completed_name / "result.json"
    malformed_result.write_text("{", encoding="utf-8")

    with pytest.raises(mod.StaleHarborJobError, match="result is unreadable or invalid"):
        asyncio.run(evaluator.evaluate(candidate))

    assert malformed_result.read_text(encoding="utf-8") == "{"


def test_interrupted_job_rejects_changed_local_dataset_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    task_dir = _write_task(dataset)
    candidate = pi_node_baseline("candidate")
    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset), _provider())
    completed_name = ""

    async def first_run(job: Job) -> None:
        nonlocal completed_name
        complete_name, incomplete_name = sorted(config.trial_name for config in job._trial_configs)
        completed_name = complete_name
        _materialize_job(
            job,
            candidate.execution_hash,
            complete_names={complete_name},
            incomplete_names={incomplete_name},
        )

    monkeypatch.setattr(Job, "run", first_run)
    first = asyncio.run(evaluator.evaluate(candidate))
    completed_result_path = first.job_dir / completed_name / "result.json"
    completed_result_before = completed_result_path.read_bytes()
    (task_dir / "instruction.md").write_text("Changed task.\n", encoding="utf-8")

    async def unexpected_run(_job: Job) -> None:
        raise AssertionError("changed task lock must be rejected before Harbor runs")

    monkeypatch.setattr(Job, "run", unexpected_run)
    with pytest.raises(mod.StaleHarborJobError, match="job config does not match"):
        asyncio.run(evaluator.evaluate(candidate))

    assert completed_result_path.read_bytes() == completed_result_before


def _local_snapshot_config(tmp_path: Path, dataset: Path) -> JobConfig:
    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset), _provider())
    agent = evaluator._build_agent(pi_node_baseline("candidate"))
    config = mod.build_harbor_job_config(evaluator._spec, agent=agent)
    return mod._snapshot_local_datasets(config)


def test_local_docker_runs_harbor_from_read_only_snapshot_not_live_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    task_dir = _write_task(dataset)
    captured: dict[str, Path] = {}
    sentinel = cast("LoadedHarborJobResult", object())

    async def inspect_run(job: Job) -> None:
        snapshot = job.config.datasets[0].path
        assert snapshot is not None
        captured["snapshot"] = snapshot
        assert snapshot != dataset.resolve()
        assert snapshot.name == dataset.name
        assert snapshot.is_relative_to((tmp_path / "jobs" / mod._TASK_SNAPSHOT_ROOT).resolve())
        assert all(
            item.stat(follow_symlinks=False).st_mode & 0o222 == 0
            for item in (snapshot, *snapshot.rglob("*"))
        )

        # This is the precise validation-to-execution mutation that the snapshot closes. Harbor's
        # already-resolved config and every trial path must remain rooted in the frozen copy.
        (task_dir / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    privileged: true\n",
            encoding="utf-8",
        )
        assert not (snapshot / task_dir.name / "environment" / "docker-compose.yaml").exists()
        assert all(
            config.task.path is not None and config.task.path.is_relative_to(snapshot)
            for config in job._trial_configs
        )

    monkeypatch.setattr(Job, "run", inspect_run)
    monkeypatch.setattr(mod, "load_harbor_job_result", lambda *_args: sentinel)

    result = asyncio.run(
        mod.HarborEvaluator(_spec(tmp_path, dataset), _provider()).evaluate(
            pi_node_baseline("candidate")
        )
    )

    assert result is sentinel
    assert captured["snapshot"].is_dir()


def test_identical_local_dataset_reuses_exact_snapshot_path(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)

    first = _local_snapshot_config(tmp_path, dataset)
    second = _local_snapshot_config(tmp_path, dataset)

    assert first.datasets[0].path == second.datasets[0].path


def test_read_only_snapshot_remains_traversable_by_non_root_container_users(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    task_dir = _write_task(dataset)
    executable = task_dir / "environment" / "setup.sh"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)

    frozen = _local_snapshot_config(tmp_path, dataset)
    snapshot = frozen.datasets[0].path
    assert snapshot is not None

    directories = [snapshot, *(path for path in snapshot.rglob("*") if path.is_dir())]
    files = [path for path in snapshot.rglob("*") if path.is_file()]
    assert all(
        stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) == 0o555 for path in directories
    )
    assert all(
        stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) & 0o444 == 0o444 for path in files
    )
    assert (
        stat.S_IMODE((snapshot / task_dir.name / "environment" / "setup.sh").stat().st_mode)
        == 0o555
    )


def test_local_dataset_content_change_selects_new_snapshot_without_mutating_old(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    task_dir = _write_task(dataset)
    first = _local_snapshot_config(tmp_path, dataset)
    first_path = first.datasets[0].path
    assert first_path is not None
    frozen_instruction = first_path / task_dir.name / "instruction.md"
    frozen_bytes = frozen_instruction.read_bytes()

    (task_dir / "instruction.md").write_text("Changed task.\n", encoding="utf-8")
    second = _local_snapshot_config(tmp_path, dataset)

    assert second.datasets[0].path != first_path
    assert frozen_instruction.read_bytes() == frozen_bytes


def test_snapshot_digest_frames_file_content_before_the_next_tree_record(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    task_dir = _write_task(dataset)
    first_file = task_dir / "zz-a"
    second_file = task_dir / "zz-b"

    def framed(value: bytes) -> bytes:
        return len(value).to_bytes(8, "big") + value

    # Under the legacy unframed stream, this suffix is byte-for-byte the record that follows
    # `zz-a` in the two-file tree below, so both distinct source trees had the same digest.
    first_file.write_bytes(
        b"payload-a"
        + framed(b"file")
        + framed(second_file.relative_to(dataset).as_posix().encode())
        + framed((0).to_bytes(2, "big"))
        + b"payload-b"
    )
    first = _local_snapshot_config(tmp_path, dataset)
    first_path = first.datasets[0].path
    assert first_path is not None

    first_file.write_bytes(b"payload-a")
    second_file.write_bytes(b"payload-b")
    second = _local_snapshot_config(tmp_path, dataset)

    assert second.datasets[0].path != first_path


def test_existing_local_dataset_snapshot_must_remain_read_only(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    task_dir = _write_task(dataset)
    first = _local_snapshot_config(tmp_path, dataset)
    snapshot = first.datasets[0].path
    assert snapshot is not None
    frozen_instruction = snapshot / task_dir.name / "instruction.md"
    frozen_instruction.chmod(0o600)

    with pytest.raises(mod.UnsupportedHarborTaskError, match="must remain read-only"):
        _local_snapshot_config(tmp_path, dataset)


def test_copied_local_dataset_is_revalidated_before_snapshot_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    task_dir = _write_task(dataset)
    real_preflight = mod._preflight_local_task_trees

    def inject_compose_before_copied_tree_preflight(config: JobConfig) -> None:
        copied_dataset = config.datasets[0].path
        if copied_dataset is not None and any(
            part.startswith(".pending-") for part in copied_dataset.parts
        ):
            (copied_dataset / task_dir.name / "environment" / "docker-compose.yaml").write_text(
                "services: {}\n",
                encoding="utf-8",
            )
        real_preflight(config)

    monkeypatch.setattr(
        mod,
        "_preflight_local_task_trees",
        inject_compose_before_copied_tree_preflight,
    )

    with pytest.raises(mod.UnsupportedHarborTaskError, match="Docker Compose"):
        _local_snapshot_config(tmp_path, dataset)


@pytest.mark.parametrize("replacement", ["symlink", "fifo"])
def test_snapshot_copy_rejects_file_swapped_at_nofollow_open_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    dataset = tmp_path / "dataset"
    task_dir = _write_task(dataset)
    instruction = task_dir / "instruction.md"
    outside = tmp_path / "outside-secret"
    secret = "outside-secret-must-not-enter-snapshot"
    outside.write_text(secret, encoding="utf-8")
    mod._require_secure_copy_primitives()
    monkeypatch.setattr(mod, "_require_secure_copy_primitives", lambda: None)
    real_open = mod.os.open
    swapped = False

    def racing_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if (
            not swapped
            and path == "instruction.md"
            and dir_fd is not None
            and flags & os.O_NOFOLLOW
            and not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)
        ):
            instruction.unlink()
            if replacement == "symlink":
                instruction.symlink_to(outside)
            else:
                os.mkfifo(instruction)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(mod.os, "open", racing_open)

    with pytest.raises(mod.UnsupportedHarborTaskError) as caught:
        _local_snapshot_config(tmp_path, dataset)

    assert swapped is True
    assert secret not in str(caught.value)
    published = tmp_path / "jobs" / mod._TASK_SNAPSHOT_ROOT
    assert not any(path.name == dataset.name for path in published.rglob(dataset.name))


def test_snapshot_binds_dataset_root_identity_before_path_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)
    moved = tmp_path / "original-dataset"
    requested = dataset.absolute()
    real_resolve = Path.resolve
    swapped = False

    def racing_resolve(path: Path, strict: bool = False) -> Path:
        nonlocal swapped
        if not swapped and path == requested:
            dataset.rename(moved)
            _write_task(dataset)
            swapped = True
        return real_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", racing_resolve)

    with pytest.raises(mod.UnsupportedHarborTaskError, match="changed identity"):
        _local_snapshot_config(tmp_path, dataset)

    assert swapped is True


def test_snapshot_source_rejects_hardlinked_regular_file(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    task_dir = _write_task(dataset)
    outside = tmp_path / "outside"
    outside.write_text("must not be copied\n", encoding="utf-8")
    instruction = task_dir / "instruction.md"
    instruction.unlink()
    os.link(outside, instruction)

    with pytest.raises(mod.UnsupportedHarborTaskError, match="single-link|changed identity"):
        _local_snapshot_config(tmp_path, dataset)


def test_snapshot_namespace_must_not_be_group_or_world_writable(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)
    jobs_dir = tmp_path / "jobs"
    snapshots_root = jobs_dir / mod._TASK_SNAPSHOT_ROOT
    snapshots_root.mkdir(parents=True)
    snapshots_root.chmod(0o777)

    try:
        with pytest.raises(mod.UnsupportedHarborTaskError, match="group/world writable"):
            _local_snapshot_config(tmp_path, dataset)
    finally:
        snapshots_root.chmod(0o700)


def test_local_docker_rejects_remote_dataset_before_harbor_job_creation(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path, tmp_path / "unused").model_copy(
        update={"datasets": [DatasetConfig(name="terminal-bench", version="v2")]},
        deep=True,
    )

    with pytest.raises(mod.UnsupportedHarborTaskError, match="local dataset paths"):
        asyncio.run(mod.HarborEvaluator(spec, _provider()).evaluate(pi_node_baseline("candidate")))


def test_local_dataset_and_jobs_directory_cannot_contain_one_another(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)
    spec = _spec(tmp_path, dataset).model_copy(update={"jobs_dir": dataset / "jobs"}, deep=True)

    with pytest.raises(mod.UnsupportedHarborTaskError, match="must not contain one another"):
        asyncio.run(mod.HarborEvaluator(spec, _provider()).evaluate(pi_node_baseline("candidate")))


def test_local_datasets_require_unambiguous_harbor_source_names(tmp_path: Path) -> None:
    dataset_a = tmp_path / "a" / "dataset"
    dataset_b = tmp_path / "b" / "dataset"
    _write_task(dataset_a, "task-a")
    _write_task(dataset_b, "task-b")

    with pytest.raises(mod.UnsupportedHarborTaskError, match="unique directory names"):
        asyncio.run(
            mod.HarborEvaluator(_spec(tmp_path, dataset_a, dataset_b), _provider()).evaluate(
                pi_node_baseline("candidate")
            )
        )


def test_zero_resolved_trials_are_rejected_before_manifest_or_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)

    class EmptyJob:
        _trial_configs: list[TrialConfig] = []
        _metrics: dict[str, list[UvScript]] = {}

        def _close_logger_handlers(self) -> None:
            pass

    async def fake_create(_cls: type[Job], _config: object) -> EmptyJob:
        return EmptyJob()

    monkeypatch.setattr(Job, "create", classmethod(fake_create))
    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset), _provider())

    with pytest.raises(ValueError, match="resolved zero trials"):
        asyncio.run(evaluator.evaluate(pi_node_baseline("candidate")))

    assert not (tmp_path / "jobs" / "evaluation" / mod._MANIFEST_FILENAME).exists()


def test_concurrent_process_cannot_start_the_same_paid_harbor_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)
    spec = _spec(tmp_path, dataset)
    evaluator = mod.HarborEvaluator(spec, _provider())

    async def unexpected_create(_cls: type[Job], _config: object) -> Job:
        raise AssertionError("contended evaluation must not create or resume a Harbor job")

    monkeypatch.setattr(Job, "create", classmethod(unexpected_create))
    lease_path = mod.harbor_job_lease_path(spec.jobs_dir, spec.job_name)
    with mod._exclusive_job_run_lock(spec.jobs_dir, spec.job_name):
        assert lease_path.is_file()
        with pytest.raises(mod.ConcurrentHarborJobError, match="already evaluating"):
            asyncio.run(evaluator.evaluate(pi_node_baseline("candidate")))

    assert lease_path.is_file()


def test_job_run_lease_rejects_platform_without_posix_locking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_file_lease, "fcntl", None)
    jobs_dir = tmp_path / "jobs"

    with pytest.raises(RuntimeError, match="require POSIX file locking"):
        with mod._exclusive_job_run_lock(jobs_dir, "evaluation"):
            raise AssertionError("unsupported platform must not acquire the job lease")

    assert not jobs_dir.exists()


def test_existing_job_directory_without_trusted_manifest_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)
    job_dir = tmp_path / "jobs" / "evaluation"
    job_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text("{}", encoding="utf-8")

    def unexpected_probe(**_kwargs: object) -> None:
        raise AssertionError("stale local evidence must be rejected before runner work")

    monkeypatch.setattr(mod, "verify_container_pi_runner_ready", unexpected_probe)

    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset), _provider())

    with pytest.raises(mod.StaleHarborJobError, match="without wmh-manifest.json"):
        asyncio.run(evaluator.evaluate(pi_node_baseline("candidate")))


@pytest.mark.parametrize("evidence_name", ["config.json", "result.json", "lock.json"])
def test_existing_trial_evidence_must_be_regular_files(
    tmp_path: Path,
    evidence_name: str,
) -> None:
    candidate = pi_node_baseline("candidate")
    evaluator = mod.HarborEvaluator(_spec(tmp_path, tmp_path / "dataset"), _provider())
    agent_digest = harbor_agent_config_digest(evaluator._build_agent(candidate))
    identity = evaluator._run_identity(candidate, run_config_digest=agent_digest)
    trial_config_digest = "sha256:" + "b" * 64
    entry = HarborTrialManifestEntry(
        cell=BenchmarkCell(
            task_key="sha256:" + "c" * 64,
            task_name="task",
            attempt=1,
            config_digest=trial_config_digest,
        ),
        trial_name="task__trial",
        task_identity="task",
        task_checksum="sha256:" + "a" * 64,
        task_source="dataset",
        task_instruction="Instruction for task.",
        trial_lock_digest=trial_config_digest,
    )
    manifest = HarborTrialManifest(
        job_name="evaluation",
        identity=identity,
        agent_config_digest=agent_digest,
        entries=[entry],
    )
    job_dir = tmp_path / "jobs" / "evaluation"
    trial_dir = job_dir / entry.trial_name
    trial_dir.mkdir(parents=True)
    (job_dir / mod._MANIFEST_FILENAME).write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    for name in ("config.json", "lock.json", "result.json"):
        (job_dir / name).write_text("{}", encoding="utf-8")
        (trial_dir / name).write_text("{}", encoding="utf-8")

    unsafe_target = tmp_path / f"outside-{evidence_name}"
    unsafe_target.write_text("{}", encoding="utf-8")
    unsafe_evidence = trial_dir / evidence_name
    unsafe_evidence.unlink()
    unsafe_evidence.symlink_to(unsafe_target)

    with pytest.raises(mod.StaleHarborJobError, match="unsafe trial"):
        mod._inspect_existing_job(
            job_dir,
            expected_identity=identity,
            expected_agent_digest=agent_digest,
            expected_job_config=mod.build_harbor_job_config(
                evaluator._spec,
                agent=evaluator._build_agent(candidate),
            ),
        )


def test_existing_manifest_for_different_candidate_is_rejected_before_second_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)
    run_count = 0
    sentinel = cast("LoadedHarborJobResult", object())

    async def fake_run(_job: Job) -> None:
        nonlocal run_count
        run_count += 1

    monkeypatch.setattr(Job, "run", fake_run)
    monkeypatch.setattr(mod, "load_harbor_job_result", lambda *_args: sentinel)
    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset), _provider())
    asyncio.run(evaluator.evaluate(pi_node_baseline("candidate")))

    changed = pi_node_baseline("changed-candidate")
    changed.surfaces[0].content += "\nmaterial change"
    with pytest.raises(mod.StaleHarborJobError, match="does not match this evaluation"):
        asyncio.run(evaluator.evaluate(changed))

    assert run_count == 1


def test_separate_verifier_task_is_rejected_before_manifest_or_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset, separate_verifier=True)

    async def unexpected_run(_job: Job) -> None:
        raise AssertionError("unsafe task must be rejected before Harbor runs")

    monkeypatch.setattr(Job, "run", unexpected_run)
    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset), _provider())

    with pytest.raises(mod.UnsupportedHarborTaskError, match="separate verifier environment"):
        asyncio.run(evaluator.evaluate(pi_node_baseline("candidate")))

    assert not (tmp_path / "jobs" / "evaluation" / mod._MANIFEST_FILENAME).exists()


def test_multi_step_task_is_rejected_before_manifest_or_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset, multi_step=True)

    async def unexpected_run(_job: Job) -> None:
        raise AssertionError("multi-step task must be rejected before Harbor runs")

    monkeypatch.setattr(Job, "run", unexpected_run)
    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset), _provider())

    with pytest.raises(mod.UnsupportedHarborTaskError, match="multi-step execution"):
        asyncio.run(evaluator.evaluate(pi_node_baseline("candidate")))

    assert not (tmp_path / "jobs" / "evaluation" / mod._MANIFEST_FILENAME).exists()


@pytest.mark.parametrize("linked_name", ["instruction.md", "task.toml"])
def test_local_task_host_read_symlink_is_rejected_before_harbor_job_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    linked_name: str,
) -> None:
    dataset = tmp_path / "dataset"
    task_dir = _write_task(dataset)
    sentinel = "host-secret-sentinel-must-not-surface"
    outside = tmp_path / f"outside-{linked_name}"
    outside.write_text(
        f"# {sentinel}\n" if linked_name == "task.toml" else sentinel,
        encoding="utf-8",
    )
    linked = task_dir / linked_name
    linked.unlink()
    linked.symlink_to(outside)

    async def unexpected_create(_cls: type[Job], _config: JobConfig) -> Job:
        raise AssertionError("unsafe task tree must fail before Harbor inspects it")

    monkeypatch.setattr(Job, "create", classmethod(unexpected_create))
    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset), _provider())

    with pytest.raises(mod.UnsupportedHarborTaskError, match="symbolic link") as caught:
        asyncio.run(evaluator.evaluate(pi_node_baseline("candidate")))

    assert sentinel not in str(caught.value)
    assert not (tmp_path / "jobs" / "evaluation" / mod._MANIFEST_FILENAME).exists()


@pytest.mark.parametrize("unsafe_entry", ["task-root", "instruction.md", "task.toml"])
def test_unselected_local_sibling_is_preflighted_before_harbor_scans_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_entry: str,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset, "selected")
    sibling = _write_task(dataset, "unselected")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "task.toml").write_text("", encoding="utf-8")
    (outside / "instruction.md").write_text("host-secret-sentinel", encoding="utf-8")
    if unsafe_entry == "task-root":
        for path in sorted(sibling.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink()
        sibling.rmdir()
        sibling.symlink_to(outside, target_is_directory=True)
    else:
        linked = sibling / unsafe_entry
        linked.unlink()
        linked.symlink_to(outside / unsafe_entry)

    async def unexpected_create(_cls: type[Job], _config: JobConfig) -> Job:
        raise AssertionError("unsafe sibling must fail before Harbor scans local tasks")

    monkeypatch.setattr(Job, "create", classmethod(unexpected_create))
    spec = _spec(tmp_path, dataset).model_copy(
        update={
            "datasets": [
                DatasetConfig(path=dataset, task_names=["selected"]),
            ]
        }
    )
    evaluator = mod.HarborEvaluator(spec, _provider())

    with pytest.raises(mod.UnsupportedHarborTaskError, match="symbolic link"):
        asyncio.run(evaluator.evaluate(pi_node_baseline("candidate")))


def test_local_task_requires_prebuilt_image_before_harbor_job_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    task_dir = _write_task(dataset)
    (task_dir / "task.toml").write_text("", encoding="utf-8")

    async def unexpected_create(_cls: type[Job], _config: JobConfig) -> Job:
        raise AssertionError("Dockerfile-only task must fail before Harbor reads it")

    monkeypatch.setattr(Job, "create", classmethod(unexpected_create))
    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset), _provider())

    with pytest.raises(mod.UnsupportedHarborTaskError, match="prebuilt-image policy") as caught:
        asyncio.run(evaluator.evaluate(pi_node_baseline("candidate")))

    assert "docker_image" in str(caught.value)


@pytest.mark.parametrize(
    ("relative_source", "contents", "reason"),
    [
        (
            "docker-compose.yaml",
            "services:\n"
            "  main:\n"
            "    build: {context: /}\n"
            "    privileged: true\n"
            "    network_mode: host\n"
            "    devices: [/dev/disk0:/dev/disk0]\n"
            "    volumes: [/:/host]\n"
            "secrets: {prod: {file: /host/secret}}\n"
            "configs: {prod: {file: /host/config}}\n",
            "Docker Compose",
        ),
        ("nested/benchmark-compose.yml", "services: {}\n", "Docker Compose"),
        (".env", "PROD_SECRET=must-not-surface\n", "dotenv"),
        ("nested/runtime.env", "TOKEN=must-not-surface\n", "dotenv"),
    ],
)
def test_local_task_rejects_host_capability_sources_before_harbor_job_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_source: str,
    contents: str,
    reason: str,
) -> None:
    dataset = tmp_path / "dataset"
    task_dir = _write_task(dataset)
    source = task_dir / "environment" / relative_source
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(contents, encoding="utf-8")

    async def unexpected_create(_cls: type[Job], _config: JobConfig) -> Job:
        raise AssertionError("host-capability task source must fail before Harbor reads it")

    monkeypatch.setattr(Job, "create", classmethod(unexpected_create))
    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset), _provider())

    with pytest.raises(mod.UnsupportedHarborTaskError, match=reason) as caught:
        asyncio.run(evaluator.evaluate(pi_node_baseline("candidate")))

    assert "must-not-surface" not in str(caught.value)


@pytest.mark.parametrize(
    "task_config",
    [
        "[environment]\n"
        'docker_image = "example.invalid/task:frozen"\n'
        "[environment.env]\n"
        'MODE = "${UNPROTECTED_HOST_VALUE}"\n',
        "[environment]\n"
        'docker_image = "example.invalid/task:frozen"\n'
        "[environment.env]\n"
        'PATH = "/task-controlled/bin"\n'
        'DOCKER_HOST = "tcp://task-controlled.invalid:2375"\n',
        "[environment]\n"
        'docker_image = "example.invalid/task:frozen"\n'
        "[verifier.env]\n"
        'DOCKER_CONTEXT = "task-controlled"\n',
        '[environment]\ndocker_image = "example.invalid/task:${IMAGE_TAG}"\n',
        "[environment]\n"
        'docker_image = "example.invalid/task:frozen"\n'
        'skills_dir = "../host-skills"\n',
        "[environment]\n"
        'docker_image = "example.invalid/task:frozen"\n'
        'mcp_servers = [{name = "host", command = "host-tool"}]\n',
    ],
)
def test_local_task_rejects_host_resolved_configuration_before_harbor_job_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_config: str,
) -> None:
    dataset = tmp_path / "dataset"
    task_dir = _write_task(dataset)
    (task_dir / "task.toml").write_text(task_config, encoding="utf-8")

    async def unexpected_create(_cls: type[Job], _config: JobConfig) -> Job:
        raise AssertionError("host-resolved task config must fail before Harbor reads it")

    monkeypatch.setattr(Job, "create", classmethod(unexpected_create))
    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset), _provider())

    with pytest.raises(mod.UnsupportedHarborTaskError, match="prebuilt-image policy"):
        asyncio.run(evaluator.evaluate(pi_node_baseline("candidate")))


@pytest.mark.parametrize("keep_dockerfile", [False, True])
@pytest.mark.parametrize("compose_filename", ["docker-compose.yaml", "docker-compose.yml"])
def test_e2b_rejects_every_task_authored_compose_source_before_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    keep_dockerfile: bool,
    compose_filename: str,
) -> None:
    dataset = tmp_path / "dataset"
    task_dir = _write_task(dataset)
    if not keep_dockerfile:
        (task_dir / "environment" / "Dockerfile").unlink()
    (task_dir / "environment" / compose_filename).write_text(
        "services:\n  main:\n    image: alpine:3.20\n    environment: {MODE: compose}\n",
        encoding="utf-8",
    )

    async def unexpected_run(_job: Job) -> None:
        raise AssertionError("unsupported E2B task must be rejected before Harbor runs")

    monkeypatch.setattr(Job, "run", unexpected_run)
    spec = _spec(tmp_path, dataset).model_copy(
        update={
            "environment_backend": HarborEnvironmentBackend.E2B,
            "create_rate_policy": _rate_policy(),
        },
        deep=True,
    )
    evaluator = mod.HarborEvaluator(
        spec,
        _provider(),
        **_e2b_budget_kwargs(tmp_path, task_environment=True),
    )

    with pytest.raises(mod.UnsupportedHarborTaskError, match="ignores Compose semantics"):
        asyncio.run(evaluator.evaluate(pi_node_baseline("candidate")))

    assert not (tmp_path / "jobs" / "evaluation" / mod._MANIFEST_FILENAME).exists()


def test_task_host_credential_reference_is_rejected_before_manifest_or_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    task_dir = _write_task(dataset)
    (task_dir / "task.toml").write_text(
        "[environment]\n"
        'docker_image = "example.invalid/shared-task:frozen"\n'
        "[environment.env]\n"
        'TASK_AUTH = "${AZURE_OPENAI_API_KEY}"\n',
        encoding="utf-8",
    )
    secret = "credential-value-must-not-appear"
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", secret)

    async def unexpected_run(_job: Job) -> None:
        raise AssertionError("credential-importing task must be rejected before Harbor runs")

    monkeypatch.setattr(Job, "run", unexpected_run)
    evaluator = mod.HarborEvaluator(_spec(tmp_path, dataset), _provider())

    with pytest.raises(mod.UnsupportedHarborTaskError) as caught:
        asyncio.run(evaluator.evaluate(pi_node_baseline("candidate")))

    message = str(caught.value)
    assert "prebuilt-image policy" in message
    assert "cannot define environment maps" in message
    assert secret not in message
    assert not (tmp_path / "jobs" / "evaluation" / mod._MANIFEST_FILENAME).exists()


@pytest.mark.parametrize(
    "job_name",
    ["../escape", "nested/job", "nested\\job", mod._TASK_SNAPSHOT_ROOT],
)
def test_job_name_must_be_one_safe_component(tmp_path: Path, job_name: str) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)

    with pytest.raises(ValueError, match="single safe path component"):
        mod.HarborEvaluator(_spec(tmp_path, dataset, job_name=job_name), _provider())


def test_job_name_must_not_contain_nul_byte(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset)

    with pytest.raises(ValueError, match="single safe path component"):
        mod.HarborEvaluator(_spec(tmp_path, dataset, job_name="job\0name"), _provider())
