"""Adversarial tests for pre-open full-roster Harbor qualification."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from harbor.agents.factory import AgentFactory
from harbor.environments.factory import EnvironmentFactory
from harbor.job import Job
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.paths import TrialPaths
from harbor.verifier.factory import VerifierFactory

import wmh.evals.harbor.qualification as mod
from wmh.evals.harbor.agent import HarborTaskEnvironmentAttestation, WmhPiAgent
from wmh.evals.harbor.config import HarborEnvironmentBackend
from wmh.evals.harbor.e2b_environment import (
    TASK_E2B_LEASE_FILE,
    BudgetedE2BBuildAttribution,
    E2BSpendLimitAttestation,
    E2BSpendLimitStatement,
    E2BSpendLimitTrust,
    ExactE2BBuildRecord,
    ExactE2BBuildSpec,
    ExactE2BEnvironment,
    _e2b_spend_limit_statement_bytes,
    exact_e2b_build_resource_class,
)
from wmh.evals.harbor.paired_runner import HarborExecutionPlan
from wmh.harness.pi_runner import pi_node_baseline
from wmh.harness.pi_runner_backend import RunnerLeaseRecord
from wmh.tracking.budget import (
    BudgetPolicy,
    BudgetScope,
    ExternalSpendAuthority,
    TimedResourceCostMeter,
    bootstrap_budget_ledger,
)


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_task(
    dataset: Path,
    task_id: str,
    *,
    docker_image: str = "example.invalid/shared:frozen",
    cpu_count: int = 2,
    memory_mb: int = 1024,
    storage_mb: int | None = None,
    extra_environment_config: str = "",
) -> Path:
    task = dataset / task_id
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        "[environment]\n"
        f'docker_image = "{docker_image}"\n'
        f"cpus = {cpu_count}\n"
        f"memory_mb = {memory_mb}\n"
        + (f"storage_mb = {storage_mb}\n" if storage_mb is not None else "")
        + extra_environment_config,
        encoding="utf-8",
    )
    (task / "instruction.md").write_text(f"Solve {task_id}.\n", encoding="utf-8")
    (task / "environment" / "Dockerfile").write_text(
        "FROM alpine:3.20\n",
        encoding="utf-8",
    )
    (task / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    return task


class _FakeEnvironment:
    def __init__(
        self,
        *,
        environment_name: str,
        trial_paths: TrialPaths,
        backend: HarborEnvironmentBackend,
        starts: list[str],
        stops: list[str],
        fail_start: set[str],
        fail_stop: set[str],
        requested_storage_mb: int | None,
    ) -> None:
        self.environment_name = environment_name
        self.trial_paths = trial_paths
        self.backend = backend
        self.starts = starts
        self.stops = stops
        self.fail_start = fail_start
        self.fail_stop = fail_stop
        self.requested_storage_mb = requested_storage_mb

    def type(self) -> EnvironmentType:
        if self.backend is HarborEnvironmentBackend.LOCAL:
            return EnvironmentType.DOCKER
        return EnvironmentType.E2B

    async def start(self, force_build: bool) -> None:
        assert force_build is False
        self.starts.append(self.environment_name)
        if self.environment_name in self.fail_start:
            raise RuntimeError("synthetic environment start failure")

    async def stop(self, delete: bool) -> None:
        assert delete is True
        self.stops.append(self.environment_name)
        if self.environment_name in self.fail_stop:
            raise RuntimeError("synthetic environment cleanup failure")
        if self.backend is HarborEnvironmentBackend.E2B:
            now = datetime.now(UTC)
            receipt = RunnerLeaseRecord(
                backend="e2b",
                lease_id=f"lease-{self.environment_name}",
                owner_id="sha256:" + "1" * 64,
                config_digest="sha256:" + "2" * 64,
                state="retired",
                resource_id=f"sandbox-{self.environment_name}",
                created_at=now,
                expected_end_at=now + timedelta(minutes=1),
                retired_at=now + timedelta(seconds=1),
            )
            path = self.trial_paths.trial_dir / TASK_E2B_LEASE_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(receipt.model_dump_json(), encoding="utf-8")


def _forbid_agent_provider_and_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("qualification touched an agent, provider, or verifier")

    async def forbidden_async(*_args: object, **_kwargs: object) -> None:
        forbidden()

    monkeypatch.setattr(AgentFactory, "create_agent_from_config", forbidden)
    monkeypatch.setattr(VerifierFactory, "create_verifier_from_config", forbidden)
    monkeypatch.setattr(WmhPiAgent, "setup", forbidden_async)
    monkeypatch.setattr(WmhPiAgent, "run", forbidden_async)
    monkeypatch.setattr(Job, "run", forbidden_async)


def _install_fake_environments(
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: HarborEnvironmentBackend,
    starts: list[str],
    stops: list[str],
    fail_start: set[str] | None = None,
    fail_stop: set[str] | None = None,
    attestation_revision: dict[str, int] | None = None,
    e2b_builds_by_task: dict[str, ExactE2BBuildRecord] | None = None,
) -> list[dict[str, object]]:
    constructor_kwargs: list[dict[str, object]] = []
    failures = fail_start or set()
    cleanup_failures = fail_stop or set()
    revisions = attestation_revision or {}

    def create_environment(
        _cls: type[EnvironmentFactory],
        *,
        environment_name: str,
        trial_paths: TrialPaths,
        **kwargs: object,
    ) -> _FakeEnvironment:
        constructor_kwargs.append({"environment_name": environment_name, **kwargs})
        task_env_config = cast("Any", kwargs["task_env_config"])
        return _FakeEnvironment(
            environment_name=environment_name,
            trial_paths=trial_paths,
            backend=backend,
            starts=starts,
            stops=stops,
            fail_start=failures,
            fail_stop=cleanup_failures,
            requested_storage_mb=task_env_config.storage_mb,
        )

    async def attest(environment: _FakeEnvironment) -> HarborTaskEnvironmentAttestation:
        task_id = environment.environment_name
        if backend is HarborEnvironmentBackend.LOCAL:
            revision = revisions.get(task_id, 1)
            evidence = {
                "schema_version": 2,
                "backend": "docker",
                "daemon_platform": "linux/x86_64",
                "requested_storage_mb": environment.requested_storage_mb,
                "storage_capacity_scope": "shared_task_filesystem_available",
                "storage_provider_enforced": False,
                "storage_requirement_satisfied": True,
                "services": [
                    {
                        "service": "main",
                        "replica": 1,
                        "image_id": "sha256:" + str(revision) * 64,
                        "image_platform": "linux/x86_64",
                    }
                ],
            }
        else:
            assert e2b_builds_by_task is not None
            build = e2b_builds_by_task[task_id]
            evidence = {
                "schema_version": 3,
                "backend": "e2b",
                "template_id": build.template_id,
                "build_id": build.build_id,
                "environment_id": build.environment_id,
                "build_config_digest": build.build_config_digest,
                "launch_config_digest": "sha256:" + "2" * 64,
                "platform": "linux/x86_64",
                "cpu_count": build.cpu_count,
                "memory_mb": build.memory_mb,
                "requested_storage_mb": environment.requested_storage_mb,
                "observed_storage_mb": (
                    None
                    if environment.requested_storage_mb is None
                    else max(environment.requested_storage_mb, 20_480)
                ),
                "envd_version": "0.2.1",
                "internet_access": True,
                "lease_timeout_s": 3600,
                "timeout_action": "kill",
                "auto_resume": False,
                "volume_mounts": False,
                "network_mode": "public",
                "allowed_hosts": [],
                "network_allow_out": [],
                "network_deny_out": [],
            }
        return HarborTaskEnvironmentAttestation.from_evidence(cast("dict[str, Any]", evidence))

    monkeypatch.setattr(
        EnvironmentFactory,
        "create_environment_from_config",
        classmethod(create_environment),
    )
    monkeypatch.setattr(mod, "attest_harbor_task_environment", attest)
    monkeypatch.setattr(mod, "preflight_harbor_task_environment", lambda _config: None)
    return constructor_kwargs


def _local_qualifier(
    tmp_path: Path,
    dataset: Path,
    *,
    operation_id: str = "full-roster",
) -> mod.HarborRosterQualifier:
    plan = HarborExecutionPlan.freeze(
        reference_harness=pi_node_baseline("reference"),
        reward_key="reward",
    )
    runtime = mod.HarborRosterQualificationRuntime(
        jobs_dir=(tmp_path / "jobs").resolve(),
        dataset_paths_by_id={"terminalbench": dataset.resolve()},
    )
    return mod.HarborRosterQualifier(
        execution_plan=plan,
        runtime=runtime,
        operation_id=operation_id,
    )


def test_local_qualification_walks_complete_roster_without_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset, "task-b")
    _write_task(dataset, "task-a")
    starts: list[str] = []
    stops: list[str] = []
    _forbid_agent_provider_and_verifier(monkeypatch)
    _install_fake_environments(
        monkeypatch,
        backend=HarborEnvironmentBackend.LOCAL,
        starts=starts,
        stops=stops,
    )

    qualifier = _local_qualifier(tmp_path, dataset)
    roster = asyncio.run(qualifier.qualify())

    assert tuple(task.task_id for task in roster.tasks) == ("task-a", "task-b")
    assert {task.dataset_id for task in roster.tasks} == {"terminalbench"}
    assert all(task.environment_backend is HarborEnvironmentBackend.LOCAL for task in roster.tasks)
    assert starts == ["task-a", "task-b"]
    assert stops == starts
    assert qualifier.roster_path.is_file()

    starts.clear()
    stops.clear()
    assert asyncio.run(qualifier.qualify()) == roster
    assert starts == []
    assert stops == []


def test_partial_failure_cleans_up_and_resume_revalidates_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset, "task-a")
    _write_task(dataset, "task-b")
    starts: list[str] = []
    stops: list[str] = []
    failures = {"task-b"}
    _forbid_agent_provider_and_verifier(monkeypatch)
    _install_fake_environments(
        monkeypatch,
        backend=HarborEnvironmentBackend.LOCAL,
        starts=starts,
        stops=stops,
        fail_start=failures,
    )
    qualifier = _local_qualifier(tmp_path, dataset)

    with pytest.raises(mod.HarborRosterQualificationError):
        asyncio.run(qualifier.qualify())

    assert starts == ["task-a", "task-b"]
    assert stops == ["task-a", "task-b"]
    assert not qualifier.roster_path.exists()

    failures.clear()
    roster = asyncio.run(qualifier.qualify())

    assert tuple(task.task_id for task in roster.tasks) == ("task-a", "task-b")
    assert starts == ["task-a", "task-b", "task-a", "task-b"]
    assert stops == starts
    assert qualifier.roster_path.is_file()


def test_resume_rejects_environment_attestation_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset, "task-a")
    _write_task(dataset, "task-b")
    starts: list[str] = []
    stops: list[str] = []
    failures = {"task-b"}
    revisions = {"task-a": 1}
    _forbid_agent_provider_and_verifier(monkeypatch)
    _install_fake_environments(
        monkeypatch,
        backend=HarborEnvironmentBackend.LOCAL,
        starts=starts,
        stops=stops,
        fail_start=failures,
        attestation_revision=revisions,
    )
    qualifier = _local_qualifier(tmp_path, dataset)
    with pytest.raises(mod.HarborRosterQualificationError):
        asyncio.run(qualifier.qualify())

    failures.clear()
    revisions["task-a"] = 2
    with pytest.raises(mod.HarborRosterQualificationDriftError):
        asyncio.run(qualifier.qualify())

    assert not qualifier.roster_path.exists()


def test_resume_rejects_prepared_task_source_drift_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    task_a = _write_task(dataset, "task-a")
    _write_task(dataset, "task-b")
    starts: list[str] = []
    stops: list[str] = []
    failures = {"task-b"}
    _forbid_agent_provider_and_verifier(monkeypatch)
    _install_fake_environments(
        monkeypatch,
        backend=HarborEnvironmentBackend.LOCAL,
        starts=starts,
        stops=stops,
        fail_start=failures,
    )
    qualifier = _local_qualifier(tmp_path, dataset)
    with pytest.raises(mod.HarborRosterQualificationError):
        asyncio.run(qualifier.qualify())

    failures.clear()
    (task_a / "instruction.md").write_text("Changed after preparation.\n", encoding="utf-8")
    with pytest.raises(mod.HarborRosterQualificationDriftError):
        asyncio.run(qualifier.qualify())

    assert starts == ["task-a", "task-b"]
    assert not qualifier.roster_path.exists()


def test_published_roster_reload_rebinds_each_task_to_prepared_commitment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset, "task-a")
    starts: list[str] = []
    stops: list[str] = []
    _forbid_agent_provider_and_verifier(monkeypatch)
    _install_fake_environments(
        monkeypatch,
        backend=HarborEnvironmentBackend.LOCAL,
        starts=starts,
        stops=stops,
    )
    qualifier = _local_qualifier(tmp_path, dataset)
    roster = asyncio.run(qualifier.qualify())
    prepared = mod._read_model(
        qualifier.roster_path.parent / "prepared.json",
        mod._PreparedRosterCommitment,
    )
    assert prepared is not None
    prepared_task = prepared.tasks[0]
    evidence_path = qualifier._evidence_path(prepared_task)
    evidence = mod._read_model(evidence_path, mod._QualifiedTaskEvidence)
    assert evidence is not None

    changed_task = roster.tasks[0].model_copy(update={"content_digest": "sha256:" + "f" * 64})
    changed_evidence = mod._QualifiedTaskEvidence.freeze(
        prepared_commitment_digest=prepared.commitment_digest,
        qualification=changed_task,
        attestation=mod.HarborTaskEnvironmentAttestation.from_evidence(
            evidence.task_environment_attestation
        ),
        cleanup_receipt=None,
    )
    mod._atomic_write_model(evidence_path, changed_evidence)
    mod._atomic_write_model(
        qualifier.roster_path,
        mod.PrequalifiedHarborRoster(
            execution_plan_digest=roster.execution_plan_digest,
            tasks=(changed_task,),
        ),
    )

    with pytest.raises(
        mod.HarborRosterQualificationDriftError,
        match="prepared task identity",
    ):
        asyncio.run(qualifier.qualify())

    assert starts == ["task-a"]
    assert stops == starts


def test_cleanup_failure_never_publishes_task_or_roster_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset, "task-a")
    starts: list[str] = []
    stops: list[str] = []
    _forbid_agent_provider_and_verifier(monkeypatch)
    qualifier = _local_qualifier(tmp_path, dataset)
    _install_fake_environments(
        monkeypatch,
        backend=HarborEnvironmentBackend.LOCAL,
        starts=starts,
        stops=stops,
        fail_stop={"task-a"},
    )

    with pytest.raises(mod.HarborRosterQualificationError):
        asyncio.run(qualifier.qualify())

    assert starts == ["task-a"]
    assert stops == ["task-a"]
    assert not qualifier.roster_path.exists()
    assert list((qualifier.roster_path.parent / "evidence").glob("*.json")) == []


def test_cleanup_finishes_before_cancellation_is_propagated() -> None:
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = False

    class Environment:
        async def stop(self, delete: bool) -> None:
            nonlocal cleanup_finished
            assert delete is True
            cleanup_started.set()
            await release_cleanup.wait()
            cleanup_finished = True

    async def scenario() -> BaseException | None:
        task = asyncio.create_task(mod._stop_environment(cast("Any", Environment())))
        await cleanup_started.wait()
        task.cancel()
        release_cleanup.set()
        result = await task
        assert cleanup_finished is True
        return result

    result = asyncio.run(scenario())
    assert isinstance(result, asyncio.CancelledError)


def test_duplicate_task_ids_across_declared_datasets_fail_whole_roster(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first-dataset"
    second = tmp_path / "second-dataset"
    _write_task(first, "same-task")
    _write_task(second, "same-task")
    starts: list[str] = []
    stops: list[str] = []
    _forbid_agent_provider_and_verifier(monkeypatch)
    _install_fake_environments(
        monkeypatch,
        backend=HarborEnvironmentBackend.LOCAL,
        starts=starts,
        stops=stops,
    )
    plan = HarborExecutionPlan.freeze(
        reference_harness=pi_node_baseline("reference"),
        reward_key="reward",
    )
    qualifier = mod.HarborRosterQualifier(
        execution_plan=plan,
        runtime=mod.HarborRosterQualificationRuntime(
            jobs_dir=(tmp_path / "jobs").resolve(),
            dataset_paths_by_id={
                "first": first.resolve(),
                "second": second.resolve(),
            },
        ),
        operation_id="duplicate-task-roster",
    )

    with pytest.raises(mod.HarborRosterQualificationError) as caught:
        asyncio.run(qualifier.qualify())

    assert "duplicate Harbor task IDs" in str(caught.value.__cause__)
    assert starts == []
    assert not qualifier.roster_path.exists()


_SIGNER = Ed25519PrivateKey.generate()


def _e2b_budget_runtime(
    tmp_path: Path,
    *,
    cpu_count: int = 2,
    memory_mb: int = 1024,
) -> mod.HarborRosterQualificationBudgetRuntime:
    public_key = _SIGNER.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    trust = E2BSpendLimitTrust(
        key_id="qualification-test-key",
        account_identity="test-team/e2b",
        public_key_base64=base64.b64encode(public_key).decode(),
    )
    build_class = exact_e2b_build_resource_class(
        cpu_count=cpu_count,
        memory_mb=memory_mb,
    )
    task_class = ExactE2BEnvironment._task_resource_class(
        cpu_count=cpu_count,
        memory_mb=memory_mb,
    )
    build_meter = TimedResourceCostMeter(
        resource_type=build_class.role.value,
        resource_class_digest=build_class.digest,
        nano_usd_per_second=1,
        max_billing_seconds=build_class.max_host_observation_seconds,
        external_spend_authority=ExternalSpendAuthority(
            provider="e2b",
            account_identity=trust.account_identity,
            verifier_digest=trust.digest,
        ),
    )
    task_meter = TimedResourceCostMeter(
        resource_type=task_class.role.value,
        resource_class_digest=task_class.digest,
        nano_usd_per_second=1,
        max_billing_seconds=task_class.max_host_observation_seconds,
    )
    hard_limit = 10 * (build_meter.maximum_charge_nano_usd() + task_meter.maximum_charge_nano_usd())
    policy = BudgetPolicy(
        study_id="qualification-test",
        manifest_digest=_digest({"root": str(tmp_path)}),
        hard_limit_nano_usd=hard_limit,
        phase_limits_nano_usd={"qualification": hard_limit},
        meters={"build": build_meter, "task": task_meter},
    )
    ledger_path = (tmp_path / "budget.sqlite3").resolve()
    authority = bootstrap_budget_ledger(ledger_path, policy)
    now = datetime.now(UTC)
    statement = E2BSpendLimitStatement(
        account_identity=trust.account_identity,
        credential_fingerprint="sha256:" + "4" * 64,
        policy_digest=policy.policy_digest,
        ledger_identity=authority.ledger_identity,
        account_spend_nano_usd=0,
        account_limit_nano_usd=hard_limit,
        observed_at=now,
        expires_at=now + timedelta(minutes=5),
        dashboard_evidence_digest="sha256:" + "5" * 64,
    )
    spend_limit = E2BSpendLimitAttestation(
        statement=statement,
        key_id=trust.key_id,
        signature_base64=base64.b64encode(
            _SIGNER.sign(_e2b_spend_limit_statement_bytes(statement))
        ).decode(),
    )
    return mod.HarborRosterQualificationBudgetRuntime(
        ledger_path=authority.ledger_path,
        ledger_identity=authority.ledger_identity,
        policy=policy,
        phase="qualification",
        build_meter_by_class_digest={build_class.digest: "build"},
        task_meter_by_class_digest={task_class.digest: "task"},
        provider_spend_limit=spend_limit,
        provider_spend_limit_trust=trust,
    )


def _budgeted_build_record(
    *,
    spec: ExactE2BBuildSpec,
    budget: mod.HarborRosterQualificationBudgetRuntime,
    template_id: str,
    build_id: str,
) -> ExactE2BBuildRecord:
    return ExactE2BBuildRecord(
        build_config_digest=spec.digest,
        environment_id=spec.environment_id,
        build_context_digest=spec.build_context_digest,
        template_id=template_id,
        build_id=build_id,
        cpu_count=spec.cpu_count,
        memory_mb=spec.memory_mb,
        cost_attribution=BudgetedE2BBuildAttribution(
            policy_digest=budget.policy.policy_digest,
            ledger_identity=budget.ledger_identity,
            meter_id=next(iter(budget.build_meter_by_class_digest.values())),
            reservation_id="qualification-build-reservation",
            scope=BudgetScope(
                phase=budget.phase,
                category="task-environment-build",
                run_id="e2b-full-roster",
            ),
            provider_spend_limit=budget.provider_spend_limit,
            provider_spend_limit_trust=budget.provider_spend_limit_trust,
            provider_spend_limit_trust_digest=budget.provider_spend_limit_trust.digest,
        ),
    )


def _fake_e2b_attestation_evidence(
    build: ExactE2BBuildRecord,
    *,
    requested_storage_mb: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "backend": "e2b",
        "template_id": build.template_id,
        "build_id": build.build_id,
        "environment_id": build.environment_id,
        "build_config_digest": build.build_config_digest,
        "launch_config_digest": "sha256:" + "2" * 64,
        "platform": "linux/x86_64",
        "cpu_count": build.cpu_count,
        "memory_mb": build.memory_mb,
        "requested_storage_mb": requested_storage_mb,
        "observed_storage_mb": (
            None if requested_storage_mb is None else max(requested_storage_mb, 20_480)
        ),
        "envd_version": "0.2.1",
        "internet_access": True,
        "lease_timeout_s": 3600,
        "timeout_action": "kill",
        "auto_resume": False,
        "volume_mounts": False,
        "network_mode": "public",
        "allowed_hosts": [],
        "network_allow_out": [],
        "network_deny_out": [],
    }


def _e2b_qualifier(
    tmp_path: Path,
    dataset: Path,
    *,
    budget: mod.HarborRosterQualificationBudgetRuntime,
    spend_limit_provider: mod.E2BSpendLimitProvider | None = None,
) -> mod.HarborRosterQualifier:
    plan = HarborExecutionPlan.freeze(
        reference_harness=pi_node_baseline("reference"),
        reward_key="reward",
        environment_backend=HarborEnvironmentBackend.E2B,
    )
    return mod.HarborRosterQualifier(
        execution_plan=plan,
        runtime=mod.HarborRosterQualificationRuntime(
            jobs_dir=(tmp_path / "jobs").resolve(),
            dataset_paths_by_id={"terminalbench": dataset.resolve()},
            budget=budget,
        ),
        operation_id="e2b-full-roster",
        e2b_spend_limit_provider=spend_limit_provider,
    )


def test_e2b_qualification_dedupes_builds_and_binds_launch_accounts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset, "task-a", storage_mb=10_240)
    _write_task(dataset, "task-b", storage_mb=15_360)
    budget = _e2b_budget_runtime(tmp_path)
    build_calls: list[ExactE2BBuildSpec] = []
    builds_by_task: dict[str, ExactE2BBuildRecord] = {}

    async def prepare_build(
        *,
        spec: ExactE2BBuildSpec,
        budget_account: object,
        **_kwargs: object,
    ) -> ExactE2BBuildRecord:
        build_calls.append(spec)
        assert cast("Any", budget_account).ledger_identity == budget.ledger_identity
        return _budgeted_build_record(
            spec=spec,
            budget=budget,
            template_id="template-shared",
            build_id="build-shared",
        )

    monkeypatch.setattr(mod, "prepare_exact_e2b_build", prepare_build)
    starts: list[str] = []
    stops: list[str] = []
    _forbid_agent_provider_and_verifier(monkeypatch)

    async def attest(environment: _FakeEnvironment) -> HarborTaskEnvironmentAttestation:
        build = next(iter(builds_by_task.values()))
        evidence = _fake_e2b_attestation_evidence(
            build,
            requested_storage_mb=environment.requested_storage_mb,
        )
        return HarborTaskEnvironmentAttestation.from_evidence(evidence)

    constructor_kwargs = _install_fake_environments(
        monkeypatch,
        backend=HarborEnvironmentBackend.E2B,
        starts=starts,
        stops=stops,
        e2b_builds_by_task=builds_by_task,
    )

    original_prepare = prepare_build

    async def capture_build(**kwargs: object) -> ExactE2BBuildRecord:
        build = await original_prepare(**cast("Any", kwargs))
        builds_by_task.update({"task-a": build, "task-b": build})
        return build

    monkeypatch.setattr(mod, "prepare_exact_e2b_build", capture_build)
    monkeypatch.setattr(mod, "attest_harbor_task_environment", attest)

    roster = asyncio.run(_e2b_qualifier(tmp_path, dataset, budget=budget).qualify())

    assert len(build_calls) == 1
    assert starts == ["task-a", "task-b"]
    assert stops == starts
    assert len({task.e2b_build_config_digest for task in roster.tasks}) == 1
    assert all(task.task_resource_class is not None for task in roster.tasks)
    assert tuple(task.requested_storage_mb for task in roster.tasks) == (10_240, 15_360)
    assert tuple(task.observed_storage_mb for task in roster.tasks) == (20_480, 20_480)
    assert all(task.e2b_launch_config_digest == "sha256:" + "2" * 64 for task in roster.tasks)
    assert all(
        cast("Any", kwargs["config"]).kwargs["resource_budget_bindings"]
        for kwargs in constructor_kwargs
    )


def test_e2b_qualification_refreshes_provider_limit_for_each_unique_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset, "task-a", docker_image="example.invalid/a:frozen")
    _write_task(dataset, "task-b", docker_image="example.invalid/b:frozen")
    budget = _e2b_budget_runtime(tmp_path)
    builds_by_task: dict[str, ExactE2BBuildRecord] = {}
    supplied_for: list[str] = []

    async def spend_limit_provider(
        *,
        build_spec: ExactE2BBuildSpec,
        budget_account: object,
    ) -> E2BSpendLimitAttestation:
        assert cast("Any", budget_account).ledger_identity == budget.ledger_identity
        supplied_for.append(build_spec.digest)
        return budget.provider_spend_limit

    async def prepare_build(
        *,
        spec: ExactE2BBuildSpec,
        provider_spend_limit: E2BSpendLimitAttestation,
        **_kwargs: object,
    ) -> ExactE2BBuildRecord:
        assert provider_spend_limit == budget.provider_spend_limit
        task_id = "task-a" if spec.docker_image == "example.invalid/a:frozen" else "task-b"
        record = _budgeted_build_record(
            spec=spec,
            budget=budget,
            template_id=f"template-{task_id}",
            build_id=f"build-{task_id}",
        )
        builds_by_task[task_id] = record
        return record

    monkeypatch.setattr(mod, "prepare_exact_e2b_build", prepare_build)
    starts: list[str] = []
    stops: list[str] = []
    _forbid_agent_provider_and_verifier(monkeypatch)
    _install_fake_environments(
        monkeypatch,
        backend=HarborEnvironmentBackend.E2B,
        starts=starts,
        stops=stops,
        e2b_builds_by_task=builds_by_task,
    )

    roster = asyncio.run(
        _e2b_qualifier(
            tmp_path,
            dataset,
            budget=budget,
            spend_limit_provider=spend_limit_provider,
        ).qualify()
    )

    assert len(set(supplied_for)) == 2
    assert len(supplied_for) == 2
    assert tuple(task.task_id for task in roster.tasks) == ("task-a", "task-b")


def test_e2b_multiple_builds_require_refresh_supplier_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset, "task-a", docker_image="example.invalid/a:frozen")
    _write_task(dataset, "task-b", docker_image="example.invalid/b:frozen")
    budget = _e2b_budget_runtime(tmp_path)
    build_calls = 0

    async def unexpected_build(**_kwargs: object) -> ExactE2BBuildRecord:
        nonlocal build_calls
        build_calls += 1
        raise AssertionError("paid build dispatched without a refresh supplier")

    monkeypatch.setattr(mod, "prepare_exact_e2b_build", unexpected_build)
    _forbid_agent_provider_and_verifier(monkeypatch)
    _install_fake_environments(
        monkeypatch,
        backend=HarborEnvironmentBackend.E2B,
        starts=[],
        stops=[],
        e2b_builds_by_task={},
    )

    with pytest.raises(mod.HarborRosterQualificationError) as caught:
        asyncio.run(_e2b_qualifier(tmp_path, dataset, budget=budget).qualify())

    assert "fresh provider spend-limit supplier" in str(caught.value.__cause__)
    assert build_calls == 0


def test_e2b_qualification_rejects_missing_exact_resource_class_account(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset, "task-a", cpu_count=2, memory_mb=1024)
    wrong_budget = _e2b_budget_runtime(tmp_path, cpu_count=4, memory_mb=2048)
    starts: list[str] = []
    stops: list[str] = []
    _forbid_agent_provider_and_verifier(monkeypatch)
    _install_fake_environments(
        monkeypatch,
        backend=HarborEnvironmentBackend.E2B,
        starts=starts,
        stops=stops,
        e2b_builds_by_task={},
    )

    with pytest.raises(mod.HarborRosterQualificationError) as caught:
        asyncio.run(_e2b_qualifier(tmp_path, dataset, budget=wrong_budget).qualify())

    assert "resource class" in str(caught.value.__cause__)
    assert starts == []


@pytest.mark.parametrize(
    "resource_config",
    [
        "gpus = 1\n",
        'gpu_types = ["H100"]\n',
        'tpu = {type = "v4", topology = "2x2"}\n',
    ],
)
def test_e2b_qualification_rejects_unsupported_accelerators_before_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource_config: str,
) -> None:
    dataset = tmp_path / "dataset"
    _write_task(dataset, "task-a", extra_environment_config=resource_config)
    budget = _e2b_budget_runtime(tmp_path)
    build_calls = 0

    async def unexpected_build(**_kwargs: object) -> ExactE2BBuildRecord:
        nonlocal build_calls
        build_calls += 1
        raise AssertionError("unsupported accelerator reached paid build preparation")

    monkeypatch.setattr(mod, "prepare_exact_e2b_build", unexpected_build)
    _forbid_agent_provider_and_verifier(monkeypatch)
    _install_fake_environments(
        monkeypatch,
        backend=HarborEnvironmentBackend.E2B,
        starts=[],
        stops=[],
        e2b_builds_by_task={},
    )

    with pytest.raises(mod.HarborRosterQualificationError) as caught:
        asyncio.run(_e2b_qualifier(tmp_path, dataset, budget=budget).qualify())

    assert "do not support" in str(caught.value.__cause__)
    assert build_calls == 0
