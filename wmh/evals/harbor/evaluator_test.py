"""Offline tests for the reusable Harbor evaluator."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

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
from harbor.models.registry import DatasetFileInfo, DatasetMetadata
from harbor.models.trial.config import AgentConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, ModelInfo, TrialResult
from harbor.models.verifier.result import VerifierResult
from harbor.utils.logger import logger as harbor_logger

import wmh.evals.harbor.evaluator as mod
from wmh.evals.benchmark import BenchmarkCell, BenchmarkTaskEnvironment
from wmh.evals.harbor import _file_lease
from wmh.evals.harbor.config import HarborEnvironmentBackend, HarborJobSpec
from wmh.evals.harbor.results import (
    HarborTrialManifest,
    HarborTrialManifestEntry,
    LoadedHarborJobResult,
    harbor_agent_config_digest,
    harbor_trial_lock_digest,
)
from wmh.harness.pi_runner import pi_node_baseline
from wmh.providers.base import ProviderConfig, ProviderKind

_TASK_ENVIRONMENT_ATTESTATION = {
    "schema_version": 1,
    "backend": "docker",
    "daemon_platform": "linux/amd64",
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
    task_config = '[verifier]\nenvironment_mode = "separate"\n' if separate_verifier else ""
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
    return ProviderConfig(kind=ProviderKind.BEDROCK, model="model")


@pytest.fixture(autouse=True)
def _stub_runner_readiness_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "verify_container_pi_runner_ready", lambda **_kwargs: None)
    monkeypatch.setattr(mod, "find_spec", lambda _name: object())
    monkeypatch.setattr(mod.EnvironmentFactory, "run_preflight", lambda **_kwargs: None)


@pytest.mark.parametrize("runner_image", ["node:latest", ""])
def test_evaluator_rejects_mutable_runner_image_before_job_creation(
    tmp_path: Path,
    runner_image: str,
) -> None:
    with pytest.raises(ValueError, match="digest-qualified"):
        mod.HarborEvaluator(
            _spec(tmp_path, tmp_path / "dataset"),
            _provider(),
            runner_image=runner_image,
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


def test_package_metric_file_cannot_run_or_reach_provider_or_e2b_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    leaked_values: list[str] = []
    metadata = DatasetMetadata(
        name="owner/benchmark",
        version="sha256:dataset",
        task_ids=[],
        files=[
            DatasetFileInfo(
                path="metric.py",
                storage_path="datasets/metric.py",
                content_hash="sha256:metric",
            )
        ],
    )

    class FakePackageClient:
        async def get_dataset_metadata(self, name: str) -> DatasetMetadata:
            events.append(f"metadata:{name}")
            return metadata

    async def unexpected_create(_cls: type[Job], _config: JobConfig) -> Job:
        events.append("job-create")
        leaked_values.extend([os.environ["AZURE_OPENAI_API_KEY"], os.environ["E2B_API_KEY"]])
        raise AssertionError("executable dataset metrics must fail before Harbor job creation")

    def unexpected_task_environment(_config: JobConfig) -> None:
        events.append("task-environment")
        raise AssertionError("metric rejection must precede task-environment setup")

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-provider-secret")
    monkeypatch.setenv("E2B_API_KEY", "e2b-secret")
    monkeypatch.setattr(mod, "PackageDatasetClient", FakePackageClient)
    monkeypatch.setattr(mod._AtomicHarborJob, "create", classmethod(unexpected_create))
    monkeypatch.setattr(mod, "_preflight_task_environment", unexpected_task_environment)
    spec = _spec(tmp_path, tmp_path / "unused").model_copy(
        update={
            "datasets": [DatasetConfig(name="owner/benchmark", ref="sha256:dataset")],
            "environment_backend": HarborEnvironmentBackend.E2B,
        },
        deep=True,
    )

    with pytest.raises(mod.UnsupportedHarborMetricError, match="'metric.py'"):
        asyncio.run(mod.HarborEvaluator(spec, _provider()).evaluate(pi_node_baseline("candidate")))

    assert events == ["metadata:owner/benchmark@sha256:dataset"]
    assert leaked_values == []
    assert not (tmp_path / "jobs" / "evaluation").exists()


def test_resolved_metric_recheck_closes_remote_metadata_mutation_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    safe_metadata = DatasetMetadata(
        name="owner/benchmark",
        version="sha256:safe",
        task_ids=[],
    )
    metric_path = tmp_path / "mutated-metric.py"
    metric_path.write_text("raise AssertionError('must never execute')\n", encoding="utf-8")

    class FakePackageClient:
        async def get_dataset_metadata(self, name: str) -> DatasetMetadata:
            events.append(f"audit:{name}")
            return safe_metadata

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

    monkeypatch.setattr(mod, "PackageDatasetClient", FakePackageClient)
    monkeypatch.setattr(mod._AtomicHarborJob, "create", classmethod(create_mutated_job))
    spec = _spec(tmp_path, tmp_path / "unused").model_copy(
        update={
            "datasets": [DatasetConfig(name="owner/benchmark", ref="latest")],
        },
        deep=True,
    )

    with pytest.raises(mod.UnsupportedHarborMetricError, match="resolved executable"):
        asyncio.run(mod.HarborEvaluator(spec, _provider()).evaluate(pi_node_baseline("candidate")))

    assert events == ["audit:owner/benchmark@latest", "create:mutated", "closed"]


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
        "runner_image": mod.PI_CONTAINER_IMAGE,
        "turn_timeout_s": 300.0,
    }


def test_runner_readiness_failure_precedes_harbor_job_and_task_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def fail_probe(*, image: str) -> None:
        assert image == mod.PI_CONTAINER_IMAGE
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
    spec = _spec(tmp_path, tmp_path / "dataset").model_copy(
        update={"environment_backend": HarborEnvironmentBackend.E2B},
        deep=True,
    )

    with pytest.raises(RuntimeError, match=f"require the WMH e2b extra.*{missing_module}"):
        asyncio.run(mod.HarborEvaluator(spec, _provider()).evaluate(pi_node_baseline("candidate")))

    assert events == []
    assert not (tmp_path / "jobs" / "evaluation").exists()


def test_e2b_missing_api_key_fails_before_runner_or_harbor_job_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

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
    spec = _spec(tmp_path, tmp_path / "dataset").model_copy(
        update={"environment_backend": HarborEnvironmentBackend.E2B},
        deep=True,
    )

    with pytest.raises(RuntimeError, match="preflight failed: E2B requires E2B_API_KEY"):
        asyncio.run(mod.HarborEvaluator(spec, _provider()).evaluate(pi_node_baseline("candidate")))

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


def test_run_config_digest_binds_semantics_and_concurrency_but_not_paths(tmp_path: Path) -> None:
    spec = _spec(tmp_path, tmp_path / "dataset")
    agent_digest = "sha256:" + "a" * 64
    baseline = mod.harbor_run_config_digest(spec, agent_digest)

    semantic_variants = [
        spec.model_copy(update={"n_attempts": 3}, deep=True),
        spec.model_copy(update={"environment_backend": HarborEnvironmentBackend.E2B}, deep=True),
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
    _write_task(dataset)
    candidate = pi_node_baseline("candidate")
    captured: dict[str, object] = {}
    sentinel = cast("LoadedHarborJobResult", object())

    async def fake_run(job: Job) -> None:
        captured["environment"] = job.config.environment.type
        captured["manifest"] = HarborTrialManifest.model_validate_json(
            (job.job_dir / mod._MANIFEST_FILENAME).read_text(encoding="utf-8")
        )

    monkeypatch.setattr(Job, "run", fake_run)
    monkeypatch.setattr(mod, "load_harbor_job_result", lambda *_args: sentinel)
    spec = _spec(tmp_path, dataset).model_copy(
        update={"environment_backend": HarborEnvironmentBackend.E2B}
    )

    result = asyncio.run(mod.HarborEvaluator(spec, _provider()).evaluate(candidate))

    assert result is sentinel
    assert captured["environment"] is EnvironmentType.E2B
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
                    "runner_image": mod.PI_CONTAINER_IMAGE,
                    "task_environment_digest": _TASK_ENVIRONMENT_DIGEST,
                    "task_environment_attestation": _TASK_ENVIRONMENT_ATTESTATION,
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
    readiness_images: list[str] = []

    def record_readiness(*, image: str) -> None:
        readiness_images.append(image)

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
    assert readiness_images == [mod.PI_CONTAINER_IMAGE]


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


def test_interrupted_job_rejects_changed_resolved_task_lock(
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
    with pytest.raises(mod.StaleHarborJobError, match="newly resolved task inputs"):
        asyncio.run(evaluator.evaluate(candidate))

    assert completed_result_path.read_bytes() == completed_result_before


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
        update={"environment_backend": HarborEnvironmentBackend.E2B},
        deep=True,
    )
    evaluator = mod.HarborEvaluator(spec, _provider())

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
        '[environment.env]\nTASK_AUTH = "${AZURE_OPENAI_API_KEY}"\n',
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
    assert "host credential boundary" in message
    assert "AZURE_OPENAI_API_KEY" in message
    assert secret not in message
    assert not (tmp_path / "jobs" / "evaluation" / mod._MANIFEST_FILENAME).exists()


@pytest.mark.parametrize("job_name", ["../escape", "nested/job", "nested\\job"])
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
