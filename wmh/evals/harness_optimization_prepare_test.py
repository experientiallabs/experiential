"""Tests for the paid E2B harness-optimization preparation boundary."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from wmh.evals.harbor.config import HarborEnvironmentBackend
from wmh.evals.harbor.e2b_environment import (
    E2BSpendLimitAttestation,
    E2BSpendLimitStatement,
    E2BSpendLimitTrust,
    ExactE2BBuildSpec,
    _e2b_spend_limit_statement_bytes,
    exact_e2b_build_resource_class,
)
from wmh.evals.harbor.paired_runner import (
    HarborExecutionPlan,
    PrequalifiedHarborRoster,
)
from wmh.evals.harbor.qualification import (
    E2BSpendLimitProvider,
    HarborRosterQualificationBudgetRuntime,
    HarborRosterQualificationRuntime,
)
from wmh.evals.harbor.qualification_types import (
    QualifiedE2BBuildIdentity,
    QualifiedHarborTask,
)
from wmh.evals.harness_optimization_prepare import (
    E2BPiRunnerArtifact,
    E2BPiRunnerPreflightArtifact,
    E2BPiRunnerPreflightReceipt,
    GitCheckoutProof,
    HarnessOptimizationCanaryManifest,
    PreparedHarnessOptimizationCanary,
    _publish_and_reopen,
    prepare_e2b_harness_optimization_canary,
)
from wmh.evals.study_provenance import HarnessOptimizationCodeProvenance
from wmh.harness.pi_runner_backend import e2b_runner_resource_class
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.receipt import ProviderResponseIdentity
from wmh.tracking._testing import (
    synthetic_provider_cost_meter,
    synthetic_tariff_provenance,
)
from wmh.tracking.budget import (
    BudgetPolicy,
    ExternalSpendAuthority,
    TimedResourceBudgetAccount,
    TimedResourceClass,
    TimedResourceCostMeter,
    TimedResourceRole,
    bootstrap_budget_ledger,
)

_BASELINE_COMMIT = "a" * 40
_LAUNCH_COMMIT = "b" * 40
_DATASET_COMMIT = "6" * 40
_DATASET_TREE = "0" * 40
_RUNNER_BYTES = b"""{
  "backend": "e2b",
  "template_id": "synthetic-runner-template",
  "build_id": "synthetic-runner-build",
  "cpu_count": 2,
  "memory_mb": 4096,
  "platform": "linux/x86_64",
  "envd_version": "0.6.10",
  "lease_timeout_s": 900
}
"""
_RUNNER_FILE_DIGEST = "sha256:" + hashlib.sha256(_RUNNER_BYTES).hexdigest()
_RUNNER_PREFLIGHT_BYTES = json.dumps(
    {
        "template_alias": "synthetic-runner-alias",
        "template_id": "synthetic-runner-template",
        "build_id": "synthetic-runner-build",
        "cpu_count": 2,
        "memory_mb": 4096,
        "platform": "linux/x86_64",
        "envd_version": "0.6.10",
        "lease_timeout_s": 900,
        "internet_access": False,
        "sandbox_cleanup": "killed",
        "node_version": "v22.23.1",
        "modal_app_run": "synthetic-modal-preflight-run",
    },
    sort_keys=True,
    separators=(",", ":"),
).encode()
_RUNNER_PREFLIGHT_FILE_DIGEST = "sha256:" + hashlib.sha256(_RUNNER_PREFLIGHT_BYTES).hexdigest()
_DISCOVERY_BYTES = json.dumps(
    {
        "schema_version": "wmh.private-task-split.v1",
        "source_commit": _DATASET_COMMIT,
        "source_tree": _DATASET_TREE,
        "roster_digest": "sha256:" + "1" * 64,
        "family_catalog_digest": "sha256:" + "2" * 64,
        "split_seed": "sha256:" + "3" * 64,
        "discovery_tasks": [
            "discovery-task-a",
            "discovery-task-b",
            "discovery-task-c",
        ],
        "confirmation_tasks": ["private-heldout-task"],
    },
    sort_keys=True,
    separators=(",", ":"),
).encode()
_DISCOVERY_FILE_DIGEST = "sha256:" + hashlib.sha256(_DISCOVERY_BYTES).hexdigest()


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _provider_config() -> ProviderConfig:
    return ProviderConfig(
        kind=ProviderKind.BEDROCK,
        model="synthetic-worker-model",
        region="us-east-1",
    )


def _input_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    runner_path = inputs / "runner.json"
    runner_preflight_path = inputs / "runner-preflight.json"
    discovery_path = inputs / "locked-discovery.json"
    runner_path.write_bytes(_RUNNER_BYTES)
    runner_preflight_path.write_bytes(_RUNNER_PREFLIGHT_BYTES)
    discovery_path.write_bytes(_DISCOVERY_BYTES)
    repository = tmp_path / "orchestration"
    repository.mkdir()
    return runner_path, runner_preflight_path, discovery_path, repository


def _manifest(runner: E2BPiRunnerArtifact) -> HarnessOptimizationCanaryManifest:
    provider = _provider_config()
    return HarnessOptimizationCanaryManifest(
        code_provenance=HarnessOptimizationCodeProvenance(
            baseline_source_commit=_BASELINE_COMMIT,
            launch_orchestration_commit=_LAUNCH_COMMIT,
        ),
        runner_artifact_digest=runner.artifact_digest,
        runner_preflight_artifact_digest=_RUNNER_PREFLIGHT_FILE_DIGEST,
        experiment_id="optimizer-plumbing-canary",
        protocol_id="optimizer-plumbing-canary-v1",
        dataset_id="synthetic-benchmark",
        dataset_git_commit=_DATASET_COMMIT,
        dataset_git_tree=_DATASET_TREE,
        discovery_source_artifact_digest=_DISCOVERY_FILE_DIGEST,
        selected_task_names=(
            "discovery-task-a",
            "discovery-task-b",
        ),
        max_study_budget_nano_usd=15_000_000_000_000,
        benchmark_adapter_version="0.18.0",
        benchmark_dataset="synthetic-benchmark",
        proposer_provider_config=provider,
        proposer_response_identity=ProviderResponseIdentity(provider=ProviderKind.BEDROCK),
        scorer_provider_config=provider,
        scorer_response_identity=ProviderResponseIdentity(provider=ProviderKind.BEDROCK),
        proposer_provider_meter_id="proposer-provider",
        scorer_provider_meter_id="scorer-provider",
        confirmation_provider_meter_id="scorer-provider",
        runner_resource_meter_id="runner",
        reward_key="reward",
        turn_timeout_s=300,
        iterations=1,
        discovery_attempts_per_task=1,
        confirmation_attempts_per_task=1,
        schedule_seed="plumbing-schedule-seed",
        analysis_seed="plumbing-analysis-seed",
    )


def _runtime(
    tmp_path: Path,
    runner: E2BPiRunnerArtifact,
    manifest: HarnessOptimizationCanaryManifest,
) -> tuple[HarborRosterQualificationRuntime, TimedResourceClass]:
    signer = Ed25519PrivateKey.generate()
    public_key = signer.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    trust = E2BSpendLimitTrust(
        key_id="canary-test-signer",
        account_identity="test-team/e2b",
        public_key_base64=base64.b64encode(public_key).decode(),
    )
    build_class = exact_e2b_build_resource_class(cpu_count=1, memory_mb=2048)
    task_class = TimedResourceClass(
        role=TimedResourceRole.TASK_ENVIRONMENT,
        cpu_count=1,
        memory_mb=2048,
        provider_ttl_seconds=900,
        create_request_timeout_seconds=30,
        cleanup_horizon_seconds=60,
    )
    runner_class = e2b_runner_resource_class(runner.spec)
    provider = _provider_config()
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
    policy = BudgetPolicy(
        study_id=manifest.experiment_id,
        manifest_digest=manifest.digest,
        hard_limit_nano_usd=15_000_000_000_000,
        phase_limits_nano_usd={
            "qualification": 1_000_000_000_000,
            "discovery": 7_000_000_000_000,
            "confirmation": 7_000_000_000_000,
        },
        meters={
            "proposer-provider": synthetic_provider_cost_meter(
                provider_config=provider,
                provenance=synthetic_tariff_provenance(provider),
                input_nano_usd_per_token=1,
                output_nano_usd_per_token=5,
            ),
            "scorer-provider": synthetic_provider_cost_meter(
                provider_config=provider,
                provenance=synthetic_tariff_provenance(provider),
                input_nano_usd_per_token=1,
                output_nano_usd_per_token=5,
            ),
            "build": build_meter,
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
    authority = bootstrap_budget_ledger((tmp_path / "budget.sqlite3").resolve(), policy)
    now = datetime.now(UTC)
    statement = E2BSpendLimitStatement(
        account_identity=trust.account_identity,
        credential_fingerprint=_digest("test-e2b-credential"),
        policy_digest=policy.policy_digest,
        ledger_identity=authority.ledger_identity,
        account_spend_nano_usd=0,
        account_limit_nano_usd=policy.hard_limit_nano_usd,
        observed_at=now,
        expires_at=now + timedelta(minutes=5),
        dashboard_evidence_digest=_digest("test-e2b-dashboard"),
    )
    spend_limit = E2BSpendLimitAttestation(
        statement=statement,
        key_id=trust.key_id,
        signature_base64=base64.b64encode(
            signer.sign(_e2b_spend_limit_statement_bytes(statement))
        ).decode(),
    )
    qualification_budget = HarborRosterQualificationBudgetRuntime(
        ledger_path=authority.ledger_path,
        ledger_identity=authority.ledger_identity,
        policy=policy,
        phase="qualification",
        build_meter_by_class_digest={build_class.digest: "build"},
        task_meter_by_class_digest={task_class.digest: "task"},
        provider_spend_limit=spend_limit,
        provider_spend_limit_trust=trust,
    )
    dataset = (tmp_path / "dataset").resolve()
    dataset.mkdir()
    return (
        HarborRosterQualificationRuntime(
            jobs_dir=(tmp_path / "qualification-jobs").resolve(),
            dataset_paths_by_id={"synthetic-benchmark": dataset},
            task_names_by_dataset_id={
                "synthetic-benchmark": manifest.selected_task_names,
            },
            budget=qualification_budget,
            create_rate_ledger_path=(tmp_path / "create-rate.json").resolve(),
        ),
        task_class,
    )


def _qualified_roster(
    plan: HarborExecutionPlan,
    task_class: TimedResourceClass,
) -> PrequalifiedHarborRoster:
    tasks: list[QualifiedHarborTask] = []
    for index, task_id in enumerate(("discovery-task-a", "discovery-task-b")):
        build_spec = ExactE2BBuildSpec(
            environment_id=f"canary-environment-{index}",
            build_context_digest=_digest(f"build-context:{task_id}"),
            docker_image=f"example.invalid/{task_id}:frozen",
            cpu_count=task_class.cpu_count,
            memory_mb=task_class.memory_mb,
        )
        build_record_digest = _digest(f"build-record:{task_id}")
        tasks.append(
            QualifiedHarborTask(
                task_id=task_id,
                dataset_id="synthetic-benchmark",
                content_digest=_digest(f"content:{task_id}"),
                task_key=_digest(f"task-key:{task_id}"),
                task_environment_digest=_digest(f"task-environment:{task_id}"),
                environment_backend=HarborEnvironmentBackend.E2B,
                e2b_launch_config_digest=_digest(f"launch:{task_id}"),
                e2b_build_config_digest=build_spec.digest,
                e2b_build_record_digest=build_record_digest,
                task_resource_class_digest=task_class.digest,
                e2b_build_identity=QualifiedE2BBuildIdentity(
                    build_config_digest=build_spec.digest,
                    build_record_digest=build_record_digest,
                    environment_id=build_spec.environment_id,
                    build_context_digest=build_spec.build_context_digest,
                    docker_image=build_spec.docker_image,
                    cpu_count=build_spec.cpu_count,
                    memory_mb=build_spec.memory_mb,
                    template_id=f"task-template-{index}",
                    build_id=f"task-build-{index}",
                ),
                task_resource_class=task_class,
            )
        )
    return PrequalifiedHarborRoster(execution_plan_digest=plan.digest, tasks=tuple(tasks))


def _checkout_verifier(
    repository_path: Path,
    expected_commit: str,
    expected_tree: str | None,
    baseline_commit: str | None,
) -> GitCheckoutProof:
    del repository_path
    return GitCheckoutProof(
        head_commit=expected_commit,
        head_tree=expected_tree or "1" * 40,
        baseline_commit=baseline_commit,
        baseline_is_ancestor=None if baseline_commit is None else True,
        baseline_pi_vendor_tree=None if baseline_commit is None else "2" * 40,
        launch_pi_vendor_tree=None if baseline_commit is None else "2" * 40,
        baseline_pi_vendor_seam=None if baseline_commit is None else "3" * 40,
        launch_pi_vendor_seam=None if baseline_commit is None else "3" * 40,
    )


async def _unused_spend_limit_provider(
    *,
    build_spec: ExactE2BBuildSpec,
    budget_account: TimedResourceBudgetAccount,
) -> E2BSpendLimitAttestation:
    raise AssertionError(
        f"fake qualifier must not request spend evidence: {build_spec.digest}, "
        f"{budget_account.meter_id}"
    )


class _NeverQualifier:
    async def qualify(self) -> PrequalifiedHarborRoster:
        raise AssertionError("qualification must not be reached")


def test_exact_live_runner_preflight_receipt_binds_all_spec_fields_and_raw_bytes() -> None:
    runner = E2BPiRunnerArtifact.from_json_bytes(_RUNNER_BYTES)
    preflight = E2BPiRunnerPreflightArtifact(
        artifact_digest=_RUNNER_PREFLIGHT_FILE_DIGEST,
        receipt=E2BPiRunnerPreflightReceipt.model_validate_json(_RUNNER_PREFLIGHT_BYTES),
    )

    assert runner.artifact_digest == _RUNNER_FILE_DIGEST
    assert runner.spec.exact_template_ref == ("synthetic-runner-template:synthetic-runner-build")
    preflight.receipt.validate_runner_spec(runner.spec)
    mismatched = preflight.receipt.model_copy(update={"build_id": "foreign-build"})
    with pytest.raises(ValueError, match="preflight receipt differs"):
        mismatched.validate_runner_spec(runner.spec)
    with pytest.raises(ValueError):
        E2BPiRunnerArtifact.from_json_bytes(b'{"backend":"local"}')


def test_atomic_publication_recovers_after_sigkill_before_no_replace_link(
    tmp_path: Path,
) -> None:
    runner = E2BPiRunnerArtifact.from_json_bytes(_RUNNER_BYTES)
    destination = tmp_path / "sealed.json"
    orphaned_staging = tmp_path / ".sealed.json.staging-killed-process"
    orphaned_staging.write_bytes(b'{"partial":')

    reopened = _publish_and_reopen(destination, runner, E2BPiRunnerArtifact)

    assert reopened == runner
    assert destination.is_file()
    assert not orphaned_staging.exists()


def test_authoritative_split_commitment_rejects_non_discovery_task_before_qualification(
    tmp_path: Path,
) -> None:
    runner_path, runner_preflight_path, discovery_path, repository = _input_artifacts(tmp_path)
    runner = E2BPiRunnerArtifact.from_json_bytes(_RUNNER_BYTES)
    payload = _manifest(runner).model_dump(mode="python")
    payload["selected_task_names"] = ("discovery-task-b", "heldout-task")
    manifest = HarnessOptimizationCanaryManifest.model_validate(payload)
    runtime, _task_class = _runtime(tmp_path, runner, manifest)

    def qualifier_factory(
        *,
        execution_plan: HarborExecutionPlan,
        runtime: HarborRosterQualificationRuntime,
        operation_id: str,
        e2b_spend_limit_provider: E2BSpendLimitProvider,
    ) -> _NeverQualifier:
        del execution_plan, runtime, operation_id, e2b_spend_limit_provider
        raise AssertionError("locked discovery validation must precede qualification")

    with pytest.raises(ValueError, match="locked discovery artifact"):
        asyncio.run(
            prepare_e2b_harness_optimization_canary(
                manifest=manifest,
                runner_artifact_path=runner_path,
                runner_preflight_artifact_path=runner_preflight_path,
                locked_discovery_artifact_path=discovery_path,
                qualification_runtime=runtime,
                repository_path=repository,
                dataset_repository_path=runtime.dataset_paths_by_id[manifest.dataset_id],
                work_dir=(tmp_path / "work").resolve(),
                e2b_spend_limit_provider=_unused_spend_limit_provider,
                qualifier_factory=qualifier_factory,
                checkout_verifier=_checkout_verifier,
            )
        )


def test_prequalification_budget_policy_ledger_and_15k_cap_bind_final_study_spec(
    tmp_path: Path,
) -> None:
    runner_path, runner_preflight_path, discovery_path, repository = _input_artifacts(tmp_path)
    runner = E2BPiRunnerArtifact.from_json_bytes(_RUNNER_BYTES)
    manifest = _manifest(runner)
    runtime, task_class = _runtime(tmp_path, runner, manifest)
    qualifier_calls: list[dict[str, object]] = []

    class Qualifier:
        def __init__(self, roster: PrequalifiedHarborRoster) -> None:
            self._roster = roster

        async def qualify(self) -> PrequalifiedHarborRoster:
            sealed = tmp_path / "work" / "sealed-canary"
            assert (sealed / "prequalification-commitment.json").is_file()
            assert not (sealed / "qualification-report.json").exists()
            return self._roster

    def qualifier_factory(
        *,
        execution_plan: HarborExecutionPlan,
        runtime: HarborRosterQualificationRuntime,
        operation_id: str,
        e2b_spend_limit_provider: E2BSpendLimitProvider,
    ) -> Qualifier:
        qualifier_calls.append(
            {
                "execution_plan": execution_plan,
                "runtime": runtime,
                "operation_id": operation_id,
                "e2b_spend_limit_provider": e2b_spend_limit_provider,
            }
        )
        return Qualifier(_qualified_roster(execution_plan, task_class))

    launch = asyncio.run(
        prepare_e2b_harness_optimization_canary(
            manifest=manifest,
            runner_artifact_path=runner_path,
            runner_preflight_artifact_path=runner_preflight_path,
            locked_discovery_artifact_path=discovery_path,
            qualification_runtime=runtime,
            repository_path=repository,
            dataset_repository_path=runtime.dataset_paths_by_id[manifest.dataset_id],
            work_dir=(tmp_path / "work").resolve(),
            e2b_spend_limit_provider=_unused_spend_limit_provider,
            qualifier_factory=qualifier_factory,
            checkout_verifier=_checkout_verifier,
        )
    )

    plan = launch.study_spec.prepared.protocol.execution_plan
    assert plan.environment_backend is HarborEnvironmentBackend.E2B
    assert plan.runner_spec == runner.spec
    assert launch.runner_artifact_digest == _RUNNER_FILE_DIGEST
    assert launch.evidence_use == "plumbing-only"
    assert launch.optimizer_feedback_allowed is False
    assert launch.final_evidence_allowed is False
    assert len(launch.study_spec.prepared.protocol.discovery.tasks) == 1
    assert len(launch.study_spec.prepared.partition.confirmation_task_ids) == 1
    confirmation_task = launch.study_spec.prepared.partition.confirmation_task_ids[0]
    discovery_json = launch.study_spec.prepared.discovery_contract().model_dump_json()
    assert confirmation_task not in discovery_json
    assert launch.study_spec.qualification_report.qualified_task_count == 2
    assert (
        launch.study_spec.qualification_report.environment_backend is HarborEnvironmentBackend.E2B
    )
    assert qualifier_calls[0]["e2b_spend_limit_provider"] is _unused_spend_limit_provider
    sealed = tmp_path / "work" / "sealed-canary"
    prepared_path = sealed / "prepared-canary-launch.json"
    assert prepared_path.is_file()
    assert "private-heldout-task" not in (sealed / "prequalification-commitment.json").read_text()
    assert "private-heldout-task" not in prepared_path.read_text()

    foreign_ledger = _digest("foreign-ledger")
    foreign_search = launch.study_spec.prepared.search_cost_binding.model_copy(
        update={"ledger_identity": foreign_ledger}
    )
    foreign_confirmation = launch.study_spec.prepared.confirmation_budget.model_copy(
        update={"ledger_identity": foreign_ledger}
    )
    foreign_prepared = launch.study_spec.prepared.model_copy(
        update={
            "search_cost_binding": foreign_search,
            "confirmation_budget": foreign_confirmation,
        }
    )
    foreign_runtime = launch.study_spec.confirmation_runtime.model_copy(
        update={"budget": foreign_confirmation}
    )
    foreign_study = launch.study_spec.model_copy(
        update={"prepared": foreign_prepared, "confirmation_runtime": foreign_runtime}
    )
    launch_fields = {
        name: getattr(launch, name)
        for name in PreparedHarnessOptimizationCanary.model_fields
        if name != "study_spec"
    }
    with pytest.raises(ValueError, match="budget or runtime differs"):
        PreparedHarnessOptimizationCanary(**launch_fields, study_spec=foreign_study)

    over_cap_policy = launch.preparation_commitment.qualification_runtime.budget
    assert over_cap_policy is not None
    spliced_policy = over_cap_policy.policy.model_copy(
        update={"hard_limit_nano_usd": manifest.max_study_budget_nano_usd + 1}
    )
    over_cap_search = launch.study_spec.prepared.search_cost_binding.model_copy(
        update={
            "policy": spliced_policy,
            "declared_hard_limit_nano_usd": spliced_policy.hard_limit_nano_usd,
        }
    )
    over_cap_prepared = launch.study_spec.prepared.model_copy(
        update={"search_cost_binding": over_cap_search}
    )
    over_cap_study = launch.study_spec.model_copy(update={"prepared": over_cap_prepared})
    with pytest.raises(ValueError, match="budget or runtime differs"):
        PreparedHarnessOptimizationCanary(**launch_fields, study_spec=over_cap_study)


def test_checkout_failure_precedes_qualification_or_e2b_dispatch(tmp_path: Path) -> None:
    runner_path, runner_preflight_path, discovery_path, repository = _input_artifacts(tmp_path)
    runner = E2BPiRunnerArtifact.from_json_bytes(_RUNNER_BYTES)
    manifest = _manifest(runner)
    runtime, _task_class = _runtime(tmp_path, runner, manifest)
    qualifier_called = False

    def failing_verifier(
        repository_path: Path,
        expected_commit: str,
        expected_tree: str | None,
        baseline_commit: str | None,
    ) -> GitCheckoutProof:
        del repository_path, expected_commit, expected_tree, baseline_commit
        raise ValueError("source checkout must have no staged, unstaged, or untracked paths")

    def qualifier_factory(
        *,
        execution_plan: HarborExecutionPlan,
        runtime: HarborRosterQualificationRuntime,
        operation_id: str,
        e2b_spend_limit_provider: E2BSpendLimitProvider,
    ) -> Qualifier:
        nonlocal qualifier_called
        del execution_plan, runtime, operation_id, e2b_spend_limit_provider
        qualifier_called = True
        raise AssertionError("checkout validation must precede qualification")

    class Qualifier:
        async def qualify(self) -> PrequalifiedHarborRoster:
            raise AssertionError("checkout failure must precede qualifier construction")

    with pytest.raises(ValueError, match="no staged"):
        asyncio.run(
            prepare_e2b_harness_optimization_canary(
                manifest=manifest,
                runner_artifact_path=runner_path,
                runner_preflight_artifact_path=runner_preflight_path,
                locked_discovery_artifact_path=discovery_path,
                qualification_runtime=runtime,
                repository_path=repository,
                dataset_repository_path=runtime.dataset_paths_by_id[manifest.dataset_id],
                work_dir=(tmp_path / "work").resolve(),
                e2b_spend_limit_provider=_unused_spend_limit_provider,
                qualifier_factory=qualifier_factory,
                checkout_verifier=failing_verifier,
            )
        )

    assert qualifier_called is False
    assert not (tmp_path / "work").exists()
