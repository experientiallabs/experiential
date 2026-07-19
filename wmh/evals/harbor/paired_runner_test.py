"""Tests for exact paired Harbor execution and evidence admission."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import multiprocessing
import os
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Protocol, cast

import pytest
from harbor.models.job.config import DatasetConfig

import wmh.evals.harbor.paired_runner as mod
from wmh.evals.benchmark import (
    BenchmarkCandidateFailureReason,
    BenchmarkCandidateOutcome,
    BenchmarkCandidateStage,
    BenchmarkCandidateStatus,
    BenchmarkCell,
    BenchmarkRunHealth,
    BenchmarkRunResult,
    BenchmarkTrialResult,
    BenchmarkTrialStatus,
    BenchmarkUsage,
)
from wmh.evals.harbor.config import HarborEnvironmentBackend, HarborJobSpec
from wmh.evals.harbor.e2b_environment import ExactE2BBuildSpec, ExactE2BEnvironment
from wmh.evals.harbor.receipt_trace import validate_provider_receipt_trace
from wmh.evals.harbor.results import HarborTrialLocator, LoadedHarborJobResult
from wmh.evals.paired import (
    BoundedMeanBet,
    PairedArm,
    PairedEvaluationDesign,
    PairedPanelPlan,
    PairedTaskPlan,
)
from wmh.evals.paired_commitment import PairedEvaluationDesignTemplate
from wmh.evals.partition import (
    BenchmarkPartitionManifest,
    ConfirmationPartition,
    DiscoveryPartition,
    DiscoveryTask,
    PartitionControlScope,
    PartitionControlStore,
    PartitionTask,
    StratumCount,
    initialize_partition_genesis,
)
from wmh.harness.doc import HarnessDoc, Surface
from wmh.harness.pi_runner import pi_node_baseline
from wmh.harness.pi_runner_backend import (
    E2BPiRunnerSpec,
    LocalPiRunnerSpec,
    e2b_runner_resource_class,
)
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.tracking.budget import (
    BudgetAccount,
    BudgetPolicy,
    ProviderCostMeter,
    SpendLedger,
    TimedResourceClass,
    TimedResourceCostMeter,
    TokenPriceCeiling,
    bootstrap_budget_ledger,
)

_TASK_IDS = ("task-a", "task-b")
_FULL_TASK_IDS = (*_TASK_IDS, "task-discovery")
_TASK_KEYS = {
    "task-a": "sha256:" + "a" * 64,
    "task-b": "sha256:" + "b" * 64,
    "task-discovery": "sha256:" + "1" * 64,
}
_CONTENT_DIGESTS = {
    "task-a": "sha256:" + "c" * 64,
    "task-b": "sha256:" + "d" * 64,
    "task-discovery": "sha256:" + "2" * 64,
}
_ENVIRONMENT_DIGESTS = {
    "task-a": "sha256:" + "e" * 64,
    "task-b": "sha256:" + "f" * 64,
    "task-discovery": "sha256:" + "1" * 64,
}
_CONFIG_DIGEST = "sha256:" + "1" * 64
_RETRY_POLICY_DIGEST = "sha256:" + "5" * 64
_BUDGET_POLICY_DIGEST = "sha256:" + "6" * 64


class _ProcessEvent(Protocol):
    def set(self) -> None: ...

    def is_set(self) -> bool: ...


def _candidate() -> HarnessDoc:
    baseline = pi_node_baseline("candidate")
    surfaces = list(baseline.surfaces)
    prompt_index = next(
        index for index, surface in enumerate(surfaces) if surface.id == "prompt:core"
    )
    prompt = surfaces[prompt_index]
    surfaces[prompt_index] = Surface.model_validate(
        {**prompt.model_dump(), "content": prompt.content + "\nKeep a concise task ledger."}
    )
    return HarnessDoc(name="candidate", surfaces=surfaces)


def _design() -> PairedEvaluationDesign:
    return PairedEvaluationDesign.create(
        tasks=tuple(PairedTaskPlan(task_id=task_id, group_id=task_id) for task_id in _TASK_IDS),
        panel=(PairedPanelPlan(panel_member="worker", attempts=2),),
        primary_e_value_bets=(BoundedMeanBet(fraction=1.0, weight=1.0),),
        schedule_seed="paired-schedule-v1",
        analysis_seed="paired-analysis-v1",
        randomization_samples=1_000,
        minimum_equal_task_member_delta=0.03,
        noninferiority_margin=0.02,
    )


def _confirmation(
    candidate: HarnessDoc,
    *,
    confirmation_protocol_digest: str = "sha256:" + "7" * 64,
) -> ConfirmationPartition:
    return ConfirmationPartition(
        partition_version="2",
        partition_manifest_digest="sha256:" + "3" * 64,
        candidate_execution_digest=candidate.execution_digest,
        confirmation_protocol_digest=confirmation_protocol_digest,
        tasks=tuple(
            PartitionTask(
                task_id=task_id,
                stratum="held-out",
                group_id=task_id,
                content_digest=_CONTENT_DIGESTS[task_id],
            )
            for task_id in _TASK_IDS
        ),
        confirmation_commitment="sha256:" + "4" * 64,
        candidate_freeze_digest="sha256:" + "8" * 64,
        opening_record_digest="sha256:" + "9" * 64,
    )


def _discovery() -> DiscoveryPartition:
    return DiscoveryPartition(
        partition_version="2",
        partition_manifest_digest="sha256:" + "3" * 64,
        tasks=(
            DiscoveryTask(
                task_id="task-discovery",
                content_digest=_CONTENT_DIGESTS["task-discovery"],
            ),
        ),
        confirmation_strata=(StratumCount(stratum="held-out", count=2),),
        confirmation_commitment="sha256:" + "4" * 64,
    )


def _provider() -> ProviderConfig:
    return ProviderConfig(
        kind=ProviderKind.BEDROCK,
        model="worker-model",
        region="us-west-2",
    )


def _receipt_event(
    *,
    request_id: str,
    call_index: int = 1,
    provider: str = "bedrock",
    requested_model: str = "worker-model",
) -> dict[str, object]:
    receipt = mod.ChatProviderReceipt(
        provider=provider,
        provider_request_id=request_id,
        response_id=None,
        requested_model=requested_model,
        response_model=None,
        system_fingerprint=None,
        request_digest="sha256:" + "2" * 64,
        temperature=pi_node_baseline("limits").temperature(),
        max_tokens=pi_node_baseline("limits").max_output_tokens(),
        max_tokens_field="inferenceConfig.maxTokens",
        seed_supplied=False,
        cache_config_supplied=False,
        started_at_unix_s=1.0,
        finished_at_unix_s=2.0,
    )
    return {
        "kind": "provider_receipt",
        "payload": {**receipt.model_dump(mode="json"), "turn_call_index": call_index},
    }


def _hold_local_block_lease(
    jobs_dir: str,
    ready: _ProcessEvent,
    release: _ProcessEvent,
) -> None:
    block = mod.PairedBlock(
        task_id="task-a",
        panel_member="worker",
        attempt=1,
        first_arm=PairedArm.BASELINE,
    )
    coordinator = mod._LocalPairedHarborLeaseCoordinator(Path(jobs_dir))

    async def hold() -> None:
        async with coordinator.block_lease(
            protocol_digest="sha256:" + "a" * 64,
            block=block,
            max_concurrent_blocks=1,
            max_concurrent_route_blocks=1,
        ):
            ready.set()
            while not release.is_set():
                await asyncio.sleep(0.01)

    asyncio.run(hold())


def _spec(tmp_path: Path) -> HarborJobSpec:
    return HarborJobSpec(
        job_name="paired-template",
        jobs_dir=tmp_path / "jobs",
        datasets=[
            DatasetConfig(
                path=tmp_path / "dataset",
                task_names=list(_TASK_IDS),
            )
        ],
        n_attempts=1,
        n_concurrent_trials=1,
        agent_n_concurrent=1,
    )


def _qualifications() -> tuple[mod.QualifiedHarborTask, ...]:
    return tuple(
        mod.QualifiedHarborTask(
            task_id=task_id,
            dataset_id="terminalbench2",
            content_digest=_CONTENT_DIGESTS[task_id],
            task_key=_TASK_KEYS[task_id],
            task_environment_digest=_ENVIRONMENT_DIGESTS[task_id],
            environment_backend=HarborEnvironmentBackend.LOCAL,
        )
        for task_id in _FULL_TASK_IDS
    )


def test_execution_plan_is_task_and_host_path_independent() -> None:
    """The frozen execution contract contains no selected task or host coordinate."""
    baseline = pi_node_baseline("baseline")

    plan = mod.HarborExecutionPlan.freeze(
        reference_harness=baseline,
        reward_key="reward",
        artifact_paths=("agent/log.txt",),
    )

    assert plan.environment_backend is HarborEnvironmentBackend.LOCAL
    assert isinstance(plan.runner_spec, LocalPiRunnerSpec)
    assert plan.reward_key == "reward"
    assert plan.artifact_paths == ("agent/log.txt",)
    assert "task" not in plan.model_dump(mode="json")
    assert "jobs_dir" not in plan.model_dump(mode="json")


def test_confirmation_selection_is_a_deterministic_projection_of_full_roster() -> None:
    """Opening can select qualified entries but cannot supply a free-form task spec."""
    candidate = _candidate()
    plan = mod.HarborExecutionPlan.freeze(
        reference_harness=pi_node_baseline("baseline"),
        reward_key="reward",
    )
    roster = mod.PrequalifiedHarborRoster(
        execution_plan_digest=plan.digest,
        tasks=_qualifications()
        + (
            mod.QualifiedHarborTask(
                task_id="task-unopened",
                dataset_id="terminalbench2",
                content_digest="sha256:" + "1" * 64,
                task_key="sha256:" + "2" * 64,
                task_environment_digest="sha256:" + "3" * 64,
                environment_backend=HarborEnvironmentBackend.LOCAL,
            ),
        ),
    )

    selected = mod.OpenedHarborExecutionSelection.project(
        execution_plan=plan,
        roster=roster,
        confirmation=_confirmation(candidate),
        design=_design(),
    )

    assert tuple(task.task_id for task in selected.tasks) == _TASK_IDS
    assert selected.roster_digest == roster.digest
    assert "task-unopened" not in selected.task_ids
    assert selected == mod.OpenedHarborExecutionSelection.project(
        execution_plan=plan,
        roster=roster,
        confirmation=_confirmation(candidate),
        design=_design(),
    )

    changed_confirmation = _confirmation(candidate).model_copy(
        update={
            "tasks": (
                _confirmation(candidate)
                .tasks[0]
                .model_copy(update={"content_digest": "sha256:" + "9" * 64}),
                _confirmation(candidate).tasks[1],
            )
        }
    )
    with pytest.raises(ValueError, match="qualification content"):
        mod.OpenedHarborExecutionSelection.project(
            execution_plan=plan,
            roster=roster,
            confirmation=changed_confirmation,
            design=_design(),
        )


def test_preopen_commitment_derives_exact_design_and_selection_after_open(
    tmp_path: Path,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    routes = (
        mod.PairedHarborPanelRoute(
            panel_member="worker",
            provider_config=_provider(),
            max_concurrent_blocks=2,
        ),
    )
    plan = mod.HarborExecutionPlan.freeze(reference_harness=baseline, reward_key="reward")
    roster = mod.PrequalifiedHarborRoster(
        execution_plan_digest=plan.digest,
        tasks=_qualifications(),
    )
    budget = _budget_runtime(tmp_path, routes)
    commitment = mod.HarborConfirmationExecutionCommitment.freeze(
        discovery=_discovery(),
        design_template=PairedEvaluationDesignTemplate.from_design(_design()),
        baseline=baseline,
        candidate=candidate,
        execution_plan=plan,
        panel_routes=routes,
        qualification_roster=roster,
        max_concurrent_blocks=4,
        retry_policy_digest=_RETRY_POLICY_DIGEST,
        budget_policy_digest=budget.policy.policy_digest,
        budget_ledger_identity=budget.ledger_identity,
        budget_binding_digest=budget.binding_digest,
    )
    confirmation = _confirmation(
        candidate,
        confirmation_protocol_digest=commitment.digest,
    )

    design = commitment.derive_design(confirmation)
    selection = commitment.derive_selection(confirmation)

    assert design == _design()
    assert selection.design_digest == design.digest
    assert selection.task_ids == _TASK_IDS
    assert commitment.partition_manifest_digest == confirmation.partition_manifest_digest
    assert commitment.qualification_roster_digest == roster.digest

    original_design = _design()
    drifted_design = PairedEvaluationDesign.create(
        tasks=original_design.tasks,
        panel=original_design.panel,
        primary_e_value_bets=original_design.primary_e_value_bets,
        schedule_seed=original_design.schedule_seed,
        analysis_seed="drifted-after-open",
        randomization_samples=original_design.randomization_samples,
        alpha=original_design.alpha,
        minimum_equal_task_member_delta=original_design.minimum_equal_task_member_delta,
        noninferiority_margin=original_design.noninferiority_margin,
    )
    with pytest.raises(ValueError, match="design drifted from the pre-open commitment"):
        mod.PairedHarborProtocol.freeze(
            preopen_commitment=commitment,
            design=drifted_design,
            confirmation=confirmation,
            baseline=baseline,
            candidate=candidate,
            execution_plan=plan,
            panel_routes=routes,
            qualification_roster=roster,
            opened_selection=selection,
            max_concurrent_blocks=4,
            retry_policy_digest=_RETRY_POLICY_DIGEST,
            budget_policy_digest=budget.policy.policy_digest,
            budget_ledger_identity=budget.ledger_identity,
            budget_binding_digest=budget.binding_digest,
        )
    with pytest.raises(ValueError, match="differs from the pre-open commitment"):
        commitment.derive_design(_confirmation(candidate))


def test_typed_commitment_digest_controls_candidate_freeze_and_one_shot_open(
    tmp_path: Path,
) -> None:
    control_dir = tmp_path / "partition-control"
    control_dir.mkdir(mode=0o700)
    control_dir.chmod(0o700)
    store = PartitionControlStore(control_dir)
    partition_tasks = tuple(
        PartitionTask(
            task_id=task_id,
            stratum="discovery-only" if task_id == "task-discovery" else "held-out",
            group_id=task_id,
            content_digest=_CONTENT_DIGESTS[task_id],
        )
        for task_id in _FULL_TASK_IDS
    )
    counts = {"discovery-only": 1, "held-out": 0}
    genesis = initialize_partition_genesis(
        store,
        scope=PartitionControlScope(
            experiment_id="paired-test",
            protocol_id="typed-preopen-v1",
        ),
        tasks=partition_tasks,
        discovery_counts=counts,
    )
    manifest = BenchmarkPartitionManifest.create(
        tasks=partition_tasks,
        discovery_counts=counts,
        genesis=genesis,
    )
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    routes = (
        mod.PairedHarborPanelRoute(
            panel_member="worker",
            provider_config=_provider(),
        ),
    )
    plan = mod.HarborExecutionPlan.freeze(reference_harness=baseline, reward_key="reward")
    roster = mod.PrequalifiedHarborRoster(
        execution_plan_digest=plan.digest,
        tasks=_qualifications(),
    )
    budget = _budget_runtime(tmp_path, routes)
    commitment = mod.HarborConfirmationExecutionCommitment.freeze(
        discovery=manifest.discovery_view(),
        design_template=PairedEvaluationDesignTemplate.from_design(_design()),
        baseline=baseline,
        candidate=candidate,
        execution_plan=plan,
        panel_routes=routes,
        qualification_roster=roster,
        max_concurrent_blocks=1,
        retry_policy_digest=_RETRY_POLICY_DIGEST,
        budget_policy_digest=budget.policy.policy_digest,
        budget_ledger_identity=budget.ledger_identity,
        budget_binding_digest=budget.binding_digest,
    )

    freeze = mod.freeze_harbor_confirmation_candidate(
        store,
        manifest=manifest,
        commitment=commitment,
    )
    opened = mod.open_harbor_confirmation_once(
        store,
        manifest=manifest,
        commitment=commitment,
    )

    assert freeze.confirmation_protocol_digest == commitment.digest
    assert opened.confirmation_protocol_digest == commitment.digest
    assert tuple(task.task_id for task in opened.tasks) == _TASK_IDS
    assert commitment.derive_design(opened) == _design()

    drifted = commitment.model_copy(update={"budget_binding_digest": "sha256:" + "f" * 64})
    with pytest.raises(ValueError, match="already frozen"):
        mod.freeze_harbor_confirmation_candidate(
            store,
            manifest=manifest,
            commitment=drifted,
        )


def test_projection_rejects_backend_drift_anywhere_in_full_roster() -> None:
    """Unopened roster entries must still be qualified under the pre-search plan."""
    candidate = _candidate()
    plan = mod.HarborExecutionPlan.freeze(
        reference_harness=pi_node_baseline("baseline"),
        reward_key="reward",
        environment_backend=HarborEnvironmentBackend.E2B,
    )
    task_class = ExactE2BEnvironment._task_resource_class(cpu_count=2, memory_mb=1024)
    build = _e2b_build_identity()
    roster = mod.PrequalifiedHarborRoster(
        execution_plan_digest=plan.digest,
        tasks=tuple(
            mod.QualifiedHarborTask(
                task_id=task_id,
                dataset_id="terminalbench2",
                content_digest=_CONTENT_DIGESTS[task_id],
                task_key=_TASK_KEYS[task_id],
                task_environment_digest=_ENVIRONMENT_DIGESTS[task_id],
                environment_backend=HarborEnvironmentBackend.E2B,
                e2b_launch_config_digest="sha256:" + "7" * 64,
                e2b_build_config_digest=build.build_config_digest,
                e2b_build_record_digest=build.build_record_digest,
                task_resource_class_digest=task_class.digest,
                e2b_build_identity=build,
                task_resource_class=task_class,
            )
            for task_id in _TASK_IDS
        )
        + (
            mod.QualifiedHarborTask(
                task_id="task-unopened",
                dataset_id="terminalbench2",
                content_digest="sha256:" + "4" * 64,
                task_key="sha256:" + "5" * 64,
                task_environment_digest="sha256:" + "6" * 64,
                environment_backend=HarborEnvironmentBackend.LOCAL,
            ),
        ),
    )

    with pytest.raises(ValueError, match="full roster.*backend"):
        mod.OpenedHarborExecutionSelection.project(
            execution_plan=plan,
            roster=roster,
            confirmation=_confirmation(candidate),
            design=_design(),
        )


def _budget_runtime(
    tmp_path: Path,
    routes: tuple[mod.PairedHarborPanelRoute, ...],
) -> mod.PairedHarborBudgetRuntime:
    meter_by_member = {route.panel_member: f"worker-{route.panel_member}" for route in routes}
    policy = BudgetPolicy(
        study_id="paired-test",
        manifest_digest="sha256:" + hashlib.sha256(str(tmp_path).encode()).hexdigest(),
        hard_limit_nano_usd=1_000_000_000,
        phase_limits_nano_usd={"confirmation": 1_000_000_000},
        meters={
            meter_by_member[route.panel_member]: ProviderCostMeter(
                provider_config=route.provider_config,
                price=TokenPriceCeiling(
                    input_nano_usd_per_token=1,
                    output_nano_usd_per_token=5,
                ),
            )
            for route in routes
        },
    )
    ledger_path = (tmp_path / "budget.sqlite3").resolve()
    ledger_identity = (
        SpendLedger(ledger_path, policy, allow_create=False).ledger_identity
        if ledger_path.exists()
        else bootstrap_budget_ledger(ledger_path, policy).ledger_identity
    )
    return mod.PairedHarborBudgetRuntime(
        ledger_path=ledger_path,
        ledger_identity=ledger_identity,
        policy=policy,
        phase="confirmation",
        provider_meter_by_panel_member=meter_by_member,
    )


def _preopen_commitment(
    *,
    baseline: HarnessDoc,
    candidate: HarnessDoc,
    design: PairedEvaluationDesign,
    plan: mod.HarborExecutionPlan,
    routes: tuple[mod.PairedHarborPanelRoute, ...],
    roster: mod.PrequalifiedHarborRoster,
    max_concurrent_blocks: int,
    budget: mod.PairedHarborBudgetRuntime,
) -> mod.HarborConfirmationExecutionCommitment:
    return mod.HarborConfirmationExecutionCommitment.freeze(
        discovery=_discovery(),
        design_template=PairedEvaluationDesignTemplate.from_design(design),
        baseline=baseline,
        candidate=candidate,
        execution_plan=plan,
        panel_routes=routes,
        qualification_roster=roster,
        max_concurrent_blocks=max_concurrent_blocks,
        retry_policy_digest=_RETRY_POLICY_DIGEST,
        budget_policy_digest=budget.policy.policy_digest,
        budget_ledger_identity=budget.ledger_identity,
        budget_binding_digest=budget.binding_digest,
    )


def _e2b_runner_spec() -> E2BPiRunnerSpec:
    return E2BPiRunnerSpec(
        template_id="runner-template",
        build_id="runner-build",
        cpu_count=2,
        memory_mb=1024,
        platform="linux/x86_64",
        envd_version="v1",
        lease_timeout_s=360,
    )


def _e2b_build_identity() -> mod.QualifiedE2BBuildIdentity:
    spec = ExactE2BBuildSpec(
        environment_id="qualified-environment",
        build_context_digest="sha256:" + "6" * 64,
        docker_image="registry.example/task@sha256:" + "9" * 64,
        cpu_count=2,
        memory_mb=1024,
    )
    return mod.QualifiedE2BBuildIdentity(
        build_config_digest=spec.digest,
        build_record_digest="sha256:" + "8" * 64,
        environment_id=spec.environment_id,
        build_context_digest=spec.build_context_digest,
        docker_image=spec.docker_image,
        cpu_count=spec.cpu_count,
        memory_mb=spec.memory_mb,
        template_id="task-template",
        build_id="task-build",
    )


def _e2b_qualifications() -> tuple[mod.QualifiedHarborTask, ...]:
    task_class = ExactE2BEnvironment._task_resource_class(cpu_count=2, memory_mb=1024)
    build = _e2b_build_identity()
    return tuple(
        mod.QualifiedHarborTask(
            task_id=task_id,
            dataset_id="terminalbench2",
            content_digest=_CONTENT_DIGESTS[task_id],
            task_key=_TASK_KEYS[task_id],
            task_environment_digest=_ENVIRONMENT_DIGESTS[task_id],
            environment_backend=HarborEnvironmentBackend.E2B,
            e2b_launch_config_digest="sha256:" + "7" * 64,
            e2b_build_config_digest=build.build_config_digest,
            e2b_build_record_digest=build.build_record_digest,
            task_resource_class_digest=task_class.digest,
            e2b_build_identity=build,
            task_resource_class=task_class,
        )
        for task_id in _FULL_TASK_IDS
    )


def _e2b_budget_runtime(
    tmp_path: Path,
    routes: tuple[mod.PairedHarborPanelRoute, ...],
    runner_spec: E2BPiRunnerSpec,
    *,
    task_classes: tuple[TimedResourceClass, ...] | None = None,
) -> mod.PairedHarborBudgetRuntime:
    task_classes = task_classes or (
        ExactE2BEnvironment._task_resource_class(cpu_count=2, memory_mb=1024),
    )
    runner_class = e2b_runner_resource_class(runner_spec)
    provider_meters = {route.panel_member: f"worker-{route.panel_member}" for route in routes}
    task_meters = {
        task_class.digest: f"task-e2b-{index}" for index, task_class in enumerate(task_classes)
    }
    policy = BudgetPolicy(
        study_id="paired-e2b-test",
        manifest_digest="sha256:" + hashlib.sha256(str(tmp_path).encode()).hexdigest(),
        hard_limit_nano_usd=1_000_000_000,
        phase_limits_nano_usd={"confirmation": 1_000_000_000},
        meters={
            **{
                provider_meters[route.panel_member]: ProviderCostMeter(
                    provider_config=route.provider_config,
                    price=TokenPriceCeiling(
                        input_nano_usd_per_token=1,
                        output_nano_usd_per_token=5,
                    ),
                )
                for route in routes
            },
            **{
                task_meters[task_class.digest]: TimedResourceCostMeter(
                    resource_type=task_class.role.value,
                    resource_class_digest=task_class.digest,
                    nano_usd_per_second=1,
                    max_billing_seconds=task_class.max_host_observation_seconds,
                )
                for task_class in task_classes
            },
            "runner-e2b": TimedResourceCostMeter(
                resource_type=runner_class.role.value,
                resource_class_digest=runner_class.digest,
                nano_usd_per_second=1,
                max_billing_seconds=runner_class.max_host_observation_seconds,
            ),
        },
    )
    ledger_path = (tmp_path / "budget-e2b.sqlite3").resolve()
    ledger_identity = bootstrap_budget_ledger(ledger_path, policy).ledger_identity
    return mod.PairedHarborBudgetRuntime(
        ledger_path=ledger_path,
        ledger_identity=ledger_identity,
        policy=policy,
        phase="confirmation",
        provider_meter_by_panel_member=provider_meters,
        task_resource_meter_by_class_digest=task_meters,
        runner_resource_meter_id="runner-e2b",
    )


def _e2b_runner(
    tmp_path: Path,
    candidate: HarnessDoc,
    *,
    qualifications: tuple[mod.QualifiedHarborTask, ...] | None = None,
) -> mod.PairedHarborRunner:
    baseline = pi_node_baseline("baseline")
    routes = (
        mod.PairedHarborPanelRoute(
            panel_member="worker",
            provider_config=_provider(),
            max_concurrent_blocks=1,
        ),
    )
    runner_spec = _e2b_runner_spec()
    plan = mod.HarborExecutionPlan.freeze(
        reference_harness=baseline,
        reward_key="reward",
        environment_backend=HarborEnvironmentBackend.E2B,
        runner_spec=runner_spec,
    )
    roster_tasks = qualifications or _e2b_qualifications()
    roster = mod.PrequalifiedHarborRoster(
        execution_plan_digest=plan.digest,
        tasks=roster_tasks,
    )
    task_classes = tuple(
        {
            task.task_resource_class.digest: task.task_resource_class
            for task in roster_tasks
            if task.task_resource_class is not None
        }.values()
    )
    budget = _e2b_budget_runtime(
        tmp_path,
        routes,
        runner_spec,
        task_classes=task_classes,
    )
    commitment = _preopen_commitment(
        baseline=baseline,
        candidate=candidate,
        design=_design(),
        plan=plan,
        routes=routes,
        roster=roster,
        max_concurrent_blocks=1,
        budget=budget,
    )
    confirmation = _confirmation(
        candidate,
        confirmation_protocol_digest=commitment.digest,
    )
    selection = commitment.derive_selection(confirmation)
    protocol = mod.PairedHarborProtocol.freeze(
        preopen_commitment=commitment,
        design=_design(),
        confirmation=confirmation,
        baseline=baseline,
        candidate=candidate,
        execution_plan=plan,
        panel_routes=routes,
        qualification_roster=roster,
        opened_selection=selection,
        max_concurrent_blocks=1,
        retry_policy_digest=_RETRY_POLICY_DIGEST,
        budget_policy_digest=budget.policy.policy_digest,
        budget_ledger_identity=budget.ledger_identity,
        budget_binding_digest=budget.binding_digest,
    )
    return mod.PairedHarborRunner(
        protocol=protocol,
        runtime=mod.HarborExecutionRuntime(
            jobs_dir=(tmp_path / "jobs").resolve(),
            dataset_paths_by_id={"terminalbench2": (tmp_path / "dataset").resolve()},
            budget=budget,
        ),
        operation_id="e2b-wiring",
        generation_id=1,
    )


def _runner(
    tmp_path: Path,
    candidate: HarnessDoc,
    *,
    baseline: HarnessDoc | None = None,
    operation_id: str = "offline-test-operation",
    generation_id: int = 1,
    multi_host: bool = False,
    durable_coordinator: mod.PairedHarborLeaseCoordinator | None = None,
    **updates: object,
) -> mod.PairedHarborRunner:
    baseline = baseline or pi_node_baseline("baseline")
    default_routes = (
        mod.PairedHarborPanelRoute(
            panel_member="worker",
            provider_config=_provider(),
            max_concurrent_blocks=2,
        ),
    )
    design = cast("PairedEvaluationDesign", updates.pop("design", _design()))
    provided_confirmation = cast(
        "ConfirmationPartition | None",
        updates.pop("confirmation", None),
    )
    job_spec = cast("HarborJobSpec", updates.pop("job_spec", _spec(tmp_path)))
    routes = cast(
        "tuple[mod.PairedHarborPanelRoute, ...]",
        updates.pop("panel_routes", default_routes),
    )
    qualifications = cast(
        "tuple[mod.QualifiedHarborTask, ...]",
        updates.pop("qualified_tasks", _qualifications()),
    )
    max_concurrent_blocks = cast("int", updates.pop("max_concurrent_blocks", 4))
    if updates:
        raise AssertionError(f"unsupported paired test helper updates: {sorted(updates)}")
    plan = mod.HarborExecutionPlan.freeze(
        reference_harness=baseline,
        reward_key="reward",
        artifact_paths=tuple(job_spec.artifact_paths),
        environment_backend=job_spec.environment_backend,
    )
    roster = mod.PrequalifiedHarborRoster(
        execution_plan_digest=plan.digest,
        tasks=tuple(sorted(qualifications, key=lambda item: item.task_id)),
    )
    budget_runtime = _budget_runtime(tmp_path, routes)
    commitment = _preopen_commitment(
        baseline=baseline,
        candidate=candidate,
        design=design,
        plan=plan,
        routes=routes,
        roster=roster,
        max_concurrent_blocks=max_concurrent_blocks,
        budget=budget_runtime,
    )
    confirmation = (
        _confirmation(candidate, confirmation_protocol_digest=commitment.digest)
        if provided_confirmation is None
        else provided_confirmation.model_copy(
            update={"confirmation_protocol_digest": commitment.digest}
        )
    )
    selection = mod.OpenedHarborExecutionSelection.project(
        execution_plan=plan,
        roster=roster,
        confirmation=confirmation,
        design=design,
    )
    protocol = mod.PairedHarborProtocol.freeze(
        preopen_commitment=commitment,
        design=design,
        confirmation=confirmation,
        baseline=baseline,
        candidate=candidate,
        execution_plan=plan,
        panel_routes=routes,
        qualification_roster=roster,
        opened_selection=selection,
        max_concurrent_blocks=max_concurrent_blocks,
        retry_policy_digest=_RETRY_POLICY_DIGEST,
        budget_policy_digest=budget_runtime.policy.policy_digest,
        budget_ledger_identity=budget_runtime.ledger_identity,
        budget_binding_digest=budget_runtime.binding_digest,
    )
    runtime = mod.HarborExecutionRuntime(
        jobs_dir=job_spec.jobs_dir.resolve(),
        dataset_paths_by_id={"terminalbench2": cast("Path", job_spec.datasets[0].path).resolve()},
        budget=budget_runtime,
    )
    return mod.PairedHarborRunner(
        protocol=protocol,
        runtime=runtime,
        operation_id=operation_id,
        generation_id=generation_id,
        multi_host=multi_host,
        durable_coordinator=durable_coordinator,
    )


def _loaded_result(
    spec: HarborJobSpec,
    provider: ProviderConfig,
    harness: HarnessDoc,
    *,
    reward: float,
    budget_policy_digest: str,
) -> LoadedHarborJobResult:
    task_names = spec.datasets[0].task_names
    assert task_names is not None
    task_id = task_names[0]
    cell = BenchmarkCell(
        task_key=_TASK_KEYS[task_id],
        task_name=task_id,
        attempt=1,
        config_digest=_CONFIG_DIGEST,
    )
    trial = BenchmarkTrialResult(
        cell=cell,
        task_identity=task_id,
        task_checksum=_CONTENT_DIGESTS[task_id],
        source="test-dataset",
        task_instruction=f"Solve {task_id}.",
        task_environment_digest=_ENVIRONMENT_DIGESTS[task_id],
        runner_environment_digest=LocalPiRunnerSpec().attestation.digest,
        status=BenchmarkTrialStatus.SCORED,
        rewards={"reward": reward},
        candidate_outcome=BenchmarkCandidateOutcome(
            status=BenchmarkCandidateStatus.COMPLETED,
        ),
        run_health=BenchmarkRunHealth.VALID,
        usage=BenchmarkUsage(calls=1),
    )
    job_dir = spec.jobs_dir / spec.job_name
    trial_dir = Path("trial")
    (job_dir / trial_dir).mkdir(parents=True, exist_ok=True)
    (job_dir / trial_dir / "result.json").write_text("{}\n", encoding="utf-8")
    receipt = mod.ChatProviderReceipt(
        provider=provider.kind.value,
        provider_request_id="provider-" + spec.job_name,
        response_id=None,
        requested_model=provider.model,
        response_model=None,
        system_fingerprint=None,
        request_digest="sha256:" + "2" * 64,
        temperature=pi_node_baseline("limits").temperature(),
        max_tokens=pi_node_baseline("limits").max_output_tokens(),
        max_tokens_field="inferenceConfig.maxTokens",
        seed_supplied=False,
        cache_config_supplied=False,
        started_at_unix_s=1.0,
        finished_at_unix_s=2.0,
    )
    (job_dir / trial_dir / "wmh-events.jsonl").write_text(
        '{"kind":"assistant_message","payload":{"text":"done"}}\n'
        + json.dumps(
            {
                "kind": "provider_receipt",
                "payload": {**receipt.model_dump(mode="json"), "turn_call_index": 1},
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    identity = mod.harbor_run_expectation(
        candidate=harness,
        spec=spec,
        provider_config=provider,
        runner_spec=LocalPiRunnerSpec(),
        turn_timeout_s=300.0,
        budget_policy_digest=budget_policy_digest,
    ).identity
    return LoadedHarborJobResult(
        result=BenchmarkRunResult(
            job_name=spec.job_name,
            identity=identity,
            expected_cells=[cell],
            trials=[trial],
        ),
        job_dir=job_dir,
        locators=(
            HarborTrialLocator(
                cell=cell,
                trial_dir=trial_dir,
                result_path=trial_dir / "result.json",
                artifacts_dir=trial_dir / "artifacts",
            ),
        ),
    )


def _install_fake_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate: HarnessDoc,
    fail_job: int | None = None,
) -> tuple[
    list[tuple[HarborJobSpec, ProviderConfig, HarnessDoc]],
    dict[str, int],
]:
    calls: list[tuple[HarborJobSpec, ProviderConfig, HarnessDoc]] = []
    active: defaultdict[str, int] = defaultdict(int)
    maximum: defaultdict[str, int] = defaultdict(int)

    class FakeEvaluator:
        def __init__(
            self,
            spec: HarborJobSpec,
            provider_config: ProviderConfig,
            *,
            runner_spec: object,
            turn_timeout_s: float,
            require_provider_receipts: bool,
            session: object,
            budget_account: BudgetAccount,
            task_resource_budget_accounts: tuple[object, ...],
            runner_resource_budget_account: object | None,
        ) -> None:
            assert isinstance(runner_spec, LocalPiRunnerSpec)
            assert turn_timeout_s == 300.0
            assert require_provider_receipts is True
            assert isinstance(session, mod.HarborEvaluatorSession)
            assert task_resource_budget_accounts == ()
            assert runner_resource_budget_account is None
            self._spec = spec
            self._provider = provider_config
            self._budget_policy_digest = budget_account.policy.policy_digest

        async def evaluate(self, harness: HarnessDoc) -> LoadedHarborJobResult:
            task_names = self._spec.datasets[0].task_names
            assert task_names is not None
            task_id = task_names[0]
            active[task_id] += 1
            maximum[task_id] = max(maximum[task_id], active[task_id])
            try:
                await asyncio.sleep(0)
                calls.append((self._spec, self._provider, harness))
                if fail_job is not None and len(calls) == fail_job:
                    raise RuntimeError("synthetic infrastructure failure")
                reward = float(harness.execution_digest == candidate.execution_digest)
                return _loaded_result(
                    self._spec,
                    self._provider,
                    harness,
                    reward=reward,
                    budget_policy_digest=self._budget_policy_digest,
                )
            finally:
                active[task_id] -= 1

    monkeypatch.setattr(mod, "HarborEvaluator", FakeEvaluator)
    return calls, maximum


def _refresh_arm_admission_digest(arm: dict[str, Any]) -> None:
    arm["admission_digest"] = mod._canonical_digest(
        cast("Any", {key: value for key, value in arm.items() if key != "admission_digest"})
    )


def test_runs_every_frozen_block_in_order_and_analyzes_exact_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    calls, maximum = _install_fake_evaluator(monkeypatch, candidate=candidate)
    runner = _runner(tmp_path, candidate, baseline=baseline)

    report = asyncio.run(runner.run(baseline=baseline, candidate=candidate))

    design = _design()
    assert len(calls) == 2 * len(design.blocks)
    assert len(report.evidence) == len(design.blocks)
    assert report.run_version == "8"
    assert report.protocol.protocol_version == "8"
    assert report.protocol.design_digest == design.digest
    assert report.protocol.baseline_execution_digest == baseline.execution_digest
    assert report.protocol.candidate_execution_digest == candidate.execution_digest
    assert report.analysis.equal_task_panel_delta == 1.0
    assert all(item.outcome.baseline_reward == 0.0 for item in report.evidence)
    assert all(item.outcome.candidate_reward == 1.0 for item in report.evidence)
    assert all(item.first.arm is item.block.first_arm for item in report.evidence)
    assert all(item.first.job_name != item.second.job_name for item in report.evidence)
    assert all(value == 1 for value in maximum.values())
    assert report.digest.startswith("sha256:")

    for spec, provider, _harness in calls:
        assert spec.n_attempts == 1
        assert spec.n_concurrent_trials == 1
        assert spec.agent_n_concurrent == 1
        assert len(spec.datasets) == 1
        assert len(spec.datasets[0].task_names or []) == 1
        assert provider.model_dump() == _provider().model_dump()


def test_job_names_are_deterministic_and_bind_each_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    first_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    first = asyncio.run(
        _runner(tmp_path, candidate, baseline=baseline).run(
            baseline=baseline,
            candidate=candidate,
        )
    )
    first_names = [
        item.job_name for evidence in first.evidence for item in (evidence.first, evidence.second)
    ]

    second_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    second = asyncio.run(
        _runner(tmp_path, candidate, baseline=baseline).run(
            baseline=baseline,
            candidate=candidate,
        )
    )
    second_names = [
        item.job_name for evidence in second.evidence for item in (evidence.first, evidence.second)
    ]

    assert first_names == second_names
    assert len(set(first_names)) == len(first_names)
    assert [call[0].job_name for call in first_calls] == [call[0].job_name for call in second_calls]


def test_rejects_candidate_opening_and_compute_envelope_mismatches(tmp_path: Path) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    wrong_opening = _confirmation(candidate).model_copy(
        update={"candidate_execution_digest": "sha256:" + "9" * 64}
    )

    with pytest.raises(ValueError, match="confirmation opening"):
        asyncio.run(
            _runner(tmp_path, candidate, confirmation=wrong_opening).run(
                baseline=baseline,
                candidate=candidate,
            )
        )

    changed_surfaces = [
        (
            Surface.model_validate({**surface.model_dump(), "content": "99"})
            if surface.id == "param:max-turns"
            else surface
        )
        for surface in candidate.surfaces
    ]
    changed_compute = HarnessDoc(name="candidate", surfaces=changed_surfaces)
    with pytest.raises(ValueError, match="compute envelope"):
        asyncio.run(
            _runner(tmp_path, changed_compute, baseline=baseline).run(
                baseline=baseline,
                candidate=changed_compute,
            )
        )


def test_constructor_rejects_qualification_or_route_drift(tmp_path: Path) -> None:
    candidate = _candidate()
    confirmation = _confirmation(candidate)
    with pytest.raises(ValueError, match="opened confirmation tasks"):
        _runner(
            tmp_path,
            candidate,
            confirmation=confirmation.model_copy(
                update={"tasks": tuple(reversed(confirmation.tasks))}
            ),
        )
    changed_group = confirmation.tasks[0].model_copy(update={"group_id": "other-family"})
    with pytest.raises(ValueError, match="task clusters"):
        _runner(
            tmp_path,
            candidate,
            confirmation=confirmation.model_copy(
                update={"tasks": (changed_group, *confirmation.tasks[1:])}
            ),
        )

    qualifications = list(_qualifications())
    qualifications[0] = qualifications[0].model_copy(
        update={"content_digest": "sha256:" + "8" * 64}
    )
    with pytest.raises(ValueError, match="qualification content"):
        _runner(tmp_path, candidate, qualified_tasks=tuple(qualifications))

    duplicate_routes = (
        mod.PairedHarborPanelRoute(
            panel_member="worker",
            provider_config=_provider(),
        ),
        mod.PairedHarborPanelRoute(
            panel_member="worker",
            provider_config=_provider(),
        ),
    )
    with pytest.raises(ValueError, match="duplicate routes"):
        _runner(tmp_path, candidate, panel_routes=duplicate_routes)


def test_first_block_failure_stops_scheduling_and_returns_no_partial_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate, fail_job=1)

    with pytest.raises(mod.PairedHarborMatrixError) as captured:
        asyncio.run(
            _runner(
                tmp_path,
                candidate,
                baseline=baseline,
                max_concurrent_blocks=1,
            ).run(
                baseline=baseline,
                candidate=candidate,
            )
        )

    assert captured.value.failures
    assert len(calls) == 1
    assert "synthetic infrastructure failure" not in str(captured.value)


def test_fatal_block_failure_cancels_other_reserved_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    both_started = asyncio.Event()
    calls: list[str] = []
    cancelled: list[str] = []

    class CancellingEvaluator:
        def __init__(
            self,
            spec: HarborJobSpec,
            provider_config: ProviderConfig,
            **_kwargs: object,
        ) -> None:
            self._spec = spec
            self._provider = provider_config
            self._budget_policy_digest = cast(
                "BudgetAccount", _kwargs["budget_account"]
            ).policy.policy_digest

        async def evaluate(self, harness: HarnessDoc) -> LoadedHarborJobResult:
            calls.append(self._spec.job_name)
            is_failure = len(calls) == 1
            if len(calls) == 2:
                both_started.set()
            await both_started.wait()
            if is_failure:
                raise RuntimeError("synthetic fatal block")
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(self._spec.job_name)
                raise
            return _loaded_result(
                self._spec,
                self._provider,
                harness,
                reward=0.0,
                budget_policy_digest=self._budget_policy_digest,
            )

    monkeypatch.setattr(mod, "HarborEvaluator", CancellingEvaluator)
    with pytest.raises(mod.PairedHarborMatrixError):
        asyncio.run(
            _runner(
                tmp_path,
                candidate,
                baseline=baseline,
                max_concurrent_blocks=2,
            ).run(baseline=baseline, candidate=candidate)
        )

    assert len(calls) == 2
    assert len(cancelled) == 1


def test_schedule_has_both_first_arm_directions() -> None:
    counts = Counter(block.first_arm for block in _design().blocks)
    assert counts == {PairedArm.BASELINE: 2, PairedArm.CANDIDATE: 2}


def test_protocol_digest_binds_budget_authority_and_nonsecret_execution_inputs(
    tmp_path: Path,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    first = _runner(tmp_path / "host-a", candidate, baseline=baseline)._protocol
    second = _runner(tmp_path / "host-b", candidate, baseline=baseline)._protocol

    assert first.execution_plan == second.execution_plan
    assert first.qualification_roster == second.qualification_roster
    assert first.budget_policy_digest != second.budget_policy_digest
    assert first.budget_ledger_identity != second.budget_ledger_identity
    assert first.digest != second.digest

    changed_route = first.panel_routes[0].model_copy(update={"max_concurrent_blocks": 1})
    changed_commitment = mod.HarborConfirmationExecutionCommitment.freeze(
        discovery=_discovery(),
        design_template=PairedEvaluationDesignTemplate.from_design(_design()),
        baseline=baseline,
        candidate=candidate,
        execution_plan=first.execution_plan,
        panel_routes=(changed_route,),
        qualification_roster=first.qualification_roster,
        max_concurrent_blocks=4,
        retry_policy_digest=_RETRY_POLICY_DIGEST,
        budget_policy_digest=first.budget_policy_digest,
        budget_ledger_identity=first.budget_ledger_identity,
        budget_binding_digest=first.budget_binding_digest,
    )
    changed_confirmation = _confirmation(
        candidate,
        confirmation_protocol_digest=changed_commitment.digest,
    )
    changed = mod.PairedHarborProtocol.freeze(
        preopen_commitment=changed_commitment,
        design=_design(),
        confirmation=changed_confirmation,
        baseline=baseline,
        candidate=candidate,
        execution_plan=first.execution_plan,
        panel_routes=(changed_route,),
        qualification_roster=first.qualification_roster,
        opened_selection=changed_commitment.derive_selection(changed_confirmation),
        max_concurrent_blocks=4,
        retry_policy_digest=_RETRY_POLICY_DIGEST,
        budget_policy_digest=first.budget_policy_digest,
        budget_ledger_identity=first.budget_ledger_identity,
        budget_binding_digest=first.budget_binding_digest,
    )
    assert changed.digest != first.digest

    changed_budget = first.model_copy(update={"budget_policy_digest": "sha256:" + "7" * 64})
    assert changed_budget.digest != first.digest
    changed_ledger = first.model_copy(update={"budget_ledger_identity": "sha256:" + "8" * 64})
    assert changed_ledger.digest != first.digest


def test_paired_budget_runtime_and_protocol_reject_unknown_control_fields(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    runner = _runner(tmp_path, candidate)

    with pytest.raises(ValueError, match="unexpected_budget_control"):
        mod.PairedHarborBudgetRuntime.model_validate(
            {
                **runner._budget_runtime.model_dump(mode="json"),
                "unexpected_budget_control": "ignore-the-cap",
            }
        )
    with pytest.raises(ValueError, match="unexpected_budget_control"):
        mod.PairedHarborProtocol.model_validate(
            {
                **runner._protocol.model_dump(mode="json"),
                "unexpected_budget_control": "ignore-the-cap",
            }
        )


def test_runner_rejects_protocol_ledger_fork_before_evaluator_construction(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    runner = _runner(tmp_path, candidate)
    forked = runner._protocol.model_copy(update={"budget_ledger_identity": "sha256:" + "f" * 64})

    with pytest.raises(ValueError, match="pre-open commitment|frozen ledger identity"):
        mod.PairedHarborRunner(
            protocol=forked,
            runtime=runner._runtime,
            operation_id="forked-ledger",
            generation_id=1,
        )


def test_e2b_execution_requires_prequalified_e2b_tasks(tmp_path: Path) -> None:
    candidate = _candidate()
    e2b = _spec(tmp_path).model_copy(update={"environment_backend": HarborEnvironmentBackend.E2B})
    with pytest.raises(ValueError, match="full roster backend differs"):
        _runner(tmp_path, candidate, job_spec=e2b)


def test_e2b_budget_bindings_cover_full_prequalified_roster(tmp_path: Path) -> None:
    selected = tuple(task for task in _e2b_qualifications() if task.task_id in _TASK_IDS)
    discovery_class = ExactE2BEnvironment._task_resource_class(
        cpu_count=4,
        memory_mb=2048,
    )
    discovery_spec = ExactE2BBuildSpec(
        environment_id="discovery-only-environment",
        build_context_digest="sha256:" + "4" * 64,
        docker_image="registry.example/discovery@sha256:" + "7" * 64,
        cpu_count=discovery_class.cpu_count,
        memory_mb=discovery_class.memory_mb,
    )
    discovery_build = mod.QualifiedE2BBuildIdentity(
        build_config_digest=discovery_spec.digest,
        build_record_digest="sha256:" + "3" * 64,
        environment_id=discovery_spec.environment_id,
        build_context_digest=discovery_spec.build_context_digest,
        docker_image=discovery_spec.docker_image,
        cpu_count=discovery_spec.cpu_count,
        memory_mb=discovery_spec.memory_mb,
        template_id="discovery-template",
        build_id="discovery-build",
    )
    full_roster = selected + (
        mod.QualifiedHarborTask(
            task_id="task-discovery",
            dataset_id="terminalbench2",
            content_digest=_CONTENT_DIGESTS["task-discovery"],
            task_key=_TASK_KEYS["task-discovery"],
            task_environment_digest=_ENVIRONMENT_DIGESTS["task-discovery"],
            environment_backend=HarborEnvironmentBackend.E2B,
            e2b_launch_config_digest="sha256:" + "6" * 64,
            e2b_build_config_digest=discovery_build.build_config_digest,
            e2b_build_record_digest=discovery_build.build_record_digest,
            task_resource_class_digest=discovery_class.digest,
            e2b_build_identity=discovery_build,
            task_resource_class=discovery_class,
        ),
    )

    runner = _e2b_runner(
        tmp_path,
        _candidate(),
        qualifications=full_roster,
    )

    assert set(runner._budget_runtime.task_resource_meter_by_class_digest) == {
        task.task_resource_class_digest for task in full_roster
    }
    assert set(runner._qualifications) == set(_TASK_IDS)


def test_runtime_host_paths_do_not_change_frozen_protocol(tmp_path: Path) -> None:
    candidate = _candidate()
    first = _runner(tmp_path, candidate)
    alternate_runtime = first._runtime.model_copy(
        update={
            "jobs_dir": (tmp_path / "alternate-jobs").resolve(),
            "dataset_paths_by_id": {"terminalbench2": (tmp_path / "alternate-dataset").resolve()},
        },
        deep=True,
    )

    second = mod.PairedHarborRunner(
        protocol=first._protocol,
        runtime=alternate_runtime,
        operation_id="alternate-host-paths",
        generation_id=1,
    )

    assert second._protocol == first._protocol
    assert second._protocol.digest == first._protocol.digest
    assert second._runtime.jobs_dir != first._runtime.jobs_dir


def test_e2b_scored_wiring_uses_exact_accounts_and_never_builds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    observed: list[dict[str, object]] = []
    build_calls = 0
    record_loads = 0

    async def unexpected_build(**_kwargs: object) -> None:
        nonlocal build_calls
        build_calls += 1
        raise AssertionError("scored paired execution dispatched an E2B build")

    def load_prequalified_record(**_kwargs: object) -> object:
        nonlocal record_loads
        record_loads += 1
        identity = _e2b_build_identity()

        class Record:
            build_config_digest = identity.build_config_digest
            template_id = identity.template_id
            build_id = identity.build_id
            digest = identity.build_record_digest

        return Record()

    class WiringEvaluator:
        def __init__(
            self,
            spec: HarborJobSpec,
            provider_config: ProviderConfig,
            **kwargs: object,
        ) -> None:
            observed.append({"spec": spec, "provider": provider_config, **kwargs})

        async def evaluate(self, _harness: HarnessDoc) -> LoadedHarborJobResult:
            raise RuntimeError("stop after wiring inspection")

    monkeypatch.setattr(
        "wmh.evals.harbor.e2b_environment.prepare_exact_e2b_build",
        unexpected_build,
    )
    monkeypatch.setattr(mod, "HarborEvaluator", WiringEvaluator)
    monkeypatch.setattr(mod, "require_exact_e2b_build_record", load_prequalified_record)
    runner = _e2b_runner(tmp_path, candidate)

    with pytest.raises(mod.PairedHarborMatrixError):
        asyncio.run(
            runner.run(
                baseline=pi_node_baseline("baseline"),
                candidate=candidate,
            )
        )

    assert build_calls == 0
    assert record_loads == 1
    assert len(observed) == 1
    wiring = observed[0]
    spec = cast("HarborJobSpec", wiring["spec"])
    provider_account = cast("BudgetAccount", wiring["budget_account"])
    task_accounts = cast(
        "tuple[object, ...]",
        wiring["task_resource_budget_accounts"],
    )
    runner_account = wiring["runner_resource_budget_account"]
    assert spec.environment_backend is HarborEnvironmentBackend.E2B
    assert spec.allow_preexisting_e2b_builds is False
    assert isinstance(wiring["runner_spec"], E2BPiRunnerSpec)
    assert len(task_accounts) == 1
    assert runner_account is not None
    accounts = (provider_account, task_accounts[0], runner_account)
    assert len({cast("Any", account).ledger_path for account in accounts}) == 1
    assert len({cast("Any", account).ledger_identity for account in accounts}) == 1
    assert len({cast("Any", account).policy.policy_digest for account in accounts}) == 1
    assert len({cast("Any", account).scope.run_id for account in accounts}) == 1
    assert len({cast("Any", account).scope.lane for account in accounts}) == 1
    assert len({cast("Any", account).scope.arm for account in accounts}) == 1


def test_protocol_rejects_opened_e2b_build_drift_on_reload(tmp_path: Path) -> None:
    protocol = _e2b_runner(tmp_path, _candidate())._protocol
    payload = cast("dict[str, Any]", protocol.model_dump(mode="json"))
    payload["opened_selection"]["tasks"][0]["e2b_build_record_digest"] = "sha256:" + "f" * 64

    with pytest.raises(ValueError, match="qualification identities|opened selection"):
        mod.PairedHarborProtocol.model_validate(payload)


def test_provider_receipt_contract_distinguishes_bedrock_and_openai_evidence() -> None:
    with pytest.raises(ValueError, match="Bedrock Converse does not return"):
        mod.PairedHarborPanelRoute(
            panel_member="bedrock",
            provider_config=_provider(),
            expected_response_model="fabricated-model",
        )

    azure = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model="glm-model-family",
        deployment="glm-deployment",
        endpoint="https://example.openai.azure.com",
        api_version="2026-01-01",
    )
    with pytest.raises(ValueError, match="require an expected response model"):
        mod.PairedHarborPanelRoute(panel_member="azure", provider_config=azure)
    route = mod.PairedHarborPanelRoute(
        panel_member="azure",
        provider_config=azure,
        expected_response_model="glm-served-model",
        expected_system_fingerprint="fp-123",
    )
    assert route.expected_response_model == "glm-served-model"

    invalid_receipt = mod.ChatProviderReceipt(
        provider=ProviderKind.AZURE_OPENAI.value,
        provider_request_id="request-id",
        response_id="response-id",
        requested_model="glm-deployment",
        response_model=None,
        system_fingerprint=None,
        request_digest="sha256:" + "2" * 64,
        temperature=pi_node_baseline("limits").temperature(),
        max_tokens=1_000,
        max_tokens_field="max_completion_tokens",
        seed_supplied=False,
        cache_config_supplied=False,
        started_at_unix_s=1.0,
        finished_at_unix_s=2.0,
    )
    with pytest.raises(ValueError, match="missing response identity"):
        mod._validate_provider_receipt_for_route(
            invalid_receipt,
            route=route,
            max_output_tokens=1_000,
            temperature=pi_node_baseline("limits").temperature(),
        )


def test_canonical_receipt_trace_preserves_exact_turn_call_indexes() -> None:
    events = (
        _receipt_event(request_id="request-1", call_index=1),
        _receipt_event(request_id="request-2", call_index=2),
    )
    trace = validate_provider_receipt_trace(
        (cast("dict[str, object]", event["payload"]) for event in events),
        expected_calls=2,
        provider_config=_provider(),
        requested_temperature=pi_node_baseline("limits").temperature(),
        max_tokens=pi_node_baseline("limits").max_output_tokens(),
    )

    assert all(isinstance(receipt, mod.ChatProviderReceipt) for receipt in trace.receipts)
    assert tuple(receipt.provider_request_id for receipt in trace.receipts) == (
        "request-1",
        "request-2",
    )
    assert trace.call_indexes == (1, 2)

    noncontiguous = (
        _receipt_event(request_id="request-1", call_index=1),
        _receipt_event(request_id="request-2", call_index=3),
    )
    with pytest.raises(ValueError, match="not exact and contiguous"):
        validate_provider_receipt_trace(
            (cast("dict[str, object]", event["payload"]) for event in noncontiguous),
            expected_calls=2,
            provider_config=_provider(),
            requested_temperature=pi_node_baseline("limits").temperature(),
            max_tokens=pi_node_baseline("limits").max_output_tokens(),
        )

    duplicate = (
        _receipt_event(request_id="request-1", call_index=1),
        _receipt_event(request_id="request-1", call_index=2),
    )
    with pytest.raises(ValueError, match="reused within one trial"):
        validate_provider_receipt_trace(
            (cast("dict[str, object]", event["payload"]) for event in duplicate),
            expected_calls=2,
            provider_config=_provider(),
            requested_temperature=pi_node_baseline("limits").temperature(),
            max_tokens=pi_node_baseline("limits").max_output_tokens(),
        )


def test_multi_host_execution_rejects_host_local_budget_authority(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not supported.*shared transactional"):
        _runner(tmp_path, _candidate(), multi_host=True)


def test_multi_host_execution_rejects_process_local_budget_authority(tmp_path: Path) -> None:
    coordinator = mod._LocalPairedHarborLeaseCoordinator(tmp_path / "jobs")
    with pytest.raises(ValueError, match="not supported.*shared transactional"):
        _runner(
            tmp_path,
            _candidate(),
            multi_host=True,
            durable_coordinator=coordinator,
        )


def test_job_and_pair_generation_identities_bind_operation_generation_and_block(
    tmp_path: Path,
) -> None:
    protocol = _runner(tmp_path, _candidate())._protocol
    first, second = protocol.design.blocks[:2]
    pair = mod.paired_harbor_pair_generation_id(
        protocol_digest=protocol.digest,
        operation_id="operation-a",
        generation_id=1,
        block=first,
    )
    assert pair != mod.paired_harbor_pair_generation_id(
        protocol_digest=protocol.digest,
        operation_id="operation-a",
        generation_id=2,
        block=first,
    )
    assert pair != mod.paired_harbor_pair_generation_id(
        protocol_digest=protocol.digest,
        operation_id="operation-a",
        generation_id=1,
        block=second,
    )
    names = {
        mod.paired_harbor_job_name(
            protocol_digest=protocol.digest,
            operation_id=operation,
            generation_id=generation,
            block=first,
            arm=arm,
        )
        for operation in ("operation-a", "operation-b")
        for generation in (1, 2)
        for arm in (PairedArm.BASELINE, PairedArm.CANDIDATE)
    }
    assert len(names) == 8


def test_pair_state_replacement_closes_descriptor_when_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(tmp_path, _candidate())
    block = runner._protocol.design.blocks[0]
    state = runner._pair_state(block, status="running")
    path = runner._pair_state_path(block)
    mod._create_pair_generation_state(path, state)
    real_mkstemp = mod.tempfile.mkstemp
    created_descriptor = -1
    temporary_path: Path | None = None

    def tracked_mkstemp(*, prefix: str, dir: str | Path) -> tuple[int, str]:
        nonlocal created_descriptor, temporary_path
        created_descriptor, temporary = real_mkstemp(prefix=prefix, dir=dir)
        temporary_path = Path(temporary)
        return created_descriptor, temporary

    def fail_fchmod(_descriptor: int, _mode: int) -> None:
        raise OSError("synthetic mode failure")

    monkeypatch.setattr(mod.tempfile, "mkstemp", tracked_mkstemp)
    monkeypatch.setattr(mod.os, "fchmod", fail_fchmod)
    try:
        with pytest.raises(OSError, match="synthetic mode failure"):
            mod._replace_pair_generation_state(path, state)
        with pytest.raises(OSError):
            os.fstat(created_descriptor)
    finally:
        try:
            os.close(created_descriptor)
        except OSError:
            pass
    assert temporary_path is not None
    assert not temporary_path.exists()


def test_completed_pair_generation_cannot_be_downgraded_to_failed(tmp_path: Path) -> None:
    runner = _runner(tmp_path, _candidate())
    block = runner._protocol.design.blocks[0]
    completed = runner._pair_state(
        block,
        status="complete",
        baseline_admission_digest="sha256:" + "a" * 64,
        candidate_admission_digest="sha256:" + "b" * 64,
    )
    path = runner._pair_state_path(block)
    mod._create_pair_generation_state(path, completed)

    runner._fail_pair_generation(completed)

    assert mod._read_pair_generation_state(path) == completed


def test_partial_existing_pair_is_rejected_before_any_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    runner = _runner(tmp_path, candidate, baseline=baseline)
    block = runner._protocol.design.blocks[0]
    one_arm = mod.paired_harbor_job_name(
        protocol_digest=runner._protocol.digest,
        operation_id="offline-test-operation",
        generation_id=1,
        block=block,
        arm=PairedArm.BASELINE,
    )
    (runner._runtime.jobs_dir / one_arm).mkdir(parents=True)

    with pytest.raises(
        mod.PartialPairedHarborReuseError,
        match="arm artifacts without durable pair state",
    ):
        asyncio.run(runner.run(baseline=baseline, candidate=candidate))
    assert calls == []


def test_two_arm_directories_without_pair_state_are_rejected_before_provider_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    runner = _runner(tmp_path, candidate, baseline=baseline)
    block = runner._protocol.design.blocks[0]
    for arm in (PairedArm.BASELINE, PairedArm.CANDIDATE):
        name = mod.paired_harbor_job_name(
            protocol_digest=runner._protocol.digest,
            operation_id="offline-test-operation",
            generation_id=1,
            block=block,
            arm=arm,
        )
        (runner._runtime.jobs_dir / name).mkdir(parents=True)

    with pytest.raises(
        mod.PartialPairedHarborReuseError,
        match="arm artifacts without durable pair state",
    ):
        asyncio.run(runner.run(baseline=baseline, candidate=candidate))
    assert calls == []


def test_failed_pair_with_both_arm_directories_requires_new_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    failed_calls: list[str] = []

    class IncompleteSecondArmEvaluator:
        def __init__(
            self,
            spec: HarborJobSpec,
            provider_config: ProviderConfig,
            **_kwargs: object,
        ) -> None:
            self._spec = spec
            self._provider = provider_config
            self._budget_policy_digest = cast(
                "BudgetAccount", _kwargs["budget_account"]
            ).policy.policy_digest

        async def evaluate(self, harness: HarnessDoc) -> LoadedHarborJobResult:
            failed_calls.append(self._spec.job_name)
            if len(failed_calls) == 2:
                (self._spec.jobs_dir / self._spec.job_name).mkdir(parents=True)
                raise RuntimeError("synthetic incomplete second arm")
            return _loaded_result(
                self._spec,
                self._provider,
                harness,
                reward=0.0,
                budget_policy_digest=self._budget_policy_digest,
            )

    monkeypatch.setattr(mod, "HarborEvaluator", IncompleteSecondArmEvaluator)
    first_generation = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    with pytest.raises(mod.PairedHarborMatrixError):
        asyncio.run(first_generation.run(baseline=baseline, candidate=candidate))
    assert len(failed_calls) == 2
    first_block = first_generation._protocol.design.blocks[0]
    failed_state = mod._read_pair_generation_state(first_generation._pair_state_path(first_block))
    assert failed_state.status == "failed"
    assert all(
        (first_generation._runtime.jobs_dir / name).exists()
        for name in (
            failed_state.baseline_job_name,
            failed_state.candidate_job_name,
        )
    )

    replay_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    with pytest.raises(mod.PartialPairedHarborReuseError, match="is 'failed'"):
        asyncio.run(first_generation.run(baseline=baseline, candidate=candidate))
    assert replay_calls == []

    next_generation = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        operation_id="offline-test-operation",
        generation_id=2,
        max_concurrent_blocks=1,
    )
    report = asyncio.run(next_generation.run(baseline=baseline, candidate=candidate))
    assert len(replay_calls) == 2 * len(_design().blocks)
    assert report.generation_id == 2


def test_complete_state_with_incomplete_arm_is_rejected_before_provider_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    _install_fake_evaluator(monkeypatch, candidate=candidate)
    runner = _runner(tmp_path, candidate, baseline=baseline)
    report = asyncio.run(runner.run(baseline=baseline, candidate=candidate))
    incomplete_job = runner._runtime.jobs_dir / report.evidence[0].second.job_name
    (incomplete_job / "trial" / "result.json").unlink()

    replay_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    with pytest.raises(
        mod.PartialPairedHarborReuseError,
        match="contains an incomplete arm trial",
    ):
        asyncio.run(runner.run(baseline=baseline, candidate=candidate))
    assert replay_calls == []


def test_realistic_trace_without_provider_authored_receipt_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()

    class ReceiptlessEvaluator:
        def __init__(
            self,
            spec: HarborJobSpec,
            provider_config: ProviderConfig,
            **_kwargs: object,
        ) -> None:
            self._spec = spec
            self._provider = provider_config
            self._budget_policy_digest = cast(
                "BudgetAccount", _kwargs["budget_account"]
            ).policy.policy_digest

        async def evaluate(self, harness: HarnessDoc) -> LoadedHarborJobResult:
            loaded = _loaded_result(
                self._spec,
                self._provider,
                harness,
                reward=1.0,
                budget_policy_digest=self._budget_policy_digest,
            )
            trace = loaded.job_dir / loaded.locators[0].trial_dir / "wmh-events.jsonl"
            trace.write_text(
                '{"kind":"assistant_message","payload":{"text":"done"}}\n',
                encoding="utf-8",
            )
            return loaded

    monkeypatch.setattr(mod, "HarborEvaluator", ReceiptlessEvaluator)
    with pytest.raises(mod.PairedHarborMatrixError) as captured:
        asyncio.run(
            _runner(tmp_path, candidate, baseline=baseline).run(
                baseline=baseline,
                candidate=candidate,
            )
        )
    assert any(
        "invalid provider-call evidence" in str(error) for _, error in captured.value.failures
    )


def test_report_json_reload_recomputes_every_binding_and_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    _install_fake_evaluator(monkeypatch, candidate=candidate)
    report = asyncio.run(
        _runner(tmp_path, candidate, baseline=baseline).run(
            baseline=baseline,
            candidate=candidate,
        )
    )
    canonical = cast("dict[str, Any]", report.model_dump(mode="json"))
    assert mod.PairedHarborRunReport.model_validate_json(json.dumps(canonical)) == report

    mutations = []

    def change_job(payload: dict[str, Any]) -> None:
        payload["evidence"][0]["first"]["job_name"] += "-tampered"

    mutations.append(change_job)

    def change_task(payload: dict[str, Any]) -> None:
        payload["evidence"][0]["first"]["trial"]["task_checksum"] = "sha256:" + "9" * 64

    mutations.append(change_task)

    def change_cell_config(payload: dict[str, Any]) -> None:
        payload["evidence"][0]["first"]["trial"]["cell"]["config_digest"] = "sha256:" + "7" * 64

    mutations.append(change_cell_config)

    def change_environment(payload: dict[str, Any]) -> None:
        payload["evidence"][0]["first"]["trial"]["task_environment_digest"] = "sha256:" + "8" * 64

    mutations.append(change_environment)

    def change_route(payload: dict[str, Any]) -> None:
        payload["evidence"][0]["first"]["run_identity"]["model_name"] = "other"

    mutations.append(change_route)

    def change_score(payload: dict[str, Any]) -> None:
        current = payload["evidence"][0]["first"]["analysis_score"]
        payload["evidence"][0]["first"]["analysis_score"] = 1.0 - current

    mutations.append(change_score)

    def change_analysis(payload: dict[str, Any]) -> None:
        payload["analysis"]["equal_task_panel_delta"] = 0.5

    mutations.append(change_analysis)

    def duplicate_request_id(payload: dict[str, Any]) -> None:
        first_id = payload["evidence"][0]["first"]["provider_receipts"][0]["provider_request_id"]
        payload["evidence"][0]["second"]["provider_receipts"][0]["provider_request_id"] = first_id

    mutations.append(duplicate_request_id)

    def change_receipt_index(payload: dict[str, Any]) -> None:
        payload["evidence"][0]["first"]["provider_receipt_call_indexes"][0] = 2

    mutations.append(change_receipt_index)

    def change_receipt_provider(payload: dict[str, Any]) -> None:
        payload["evidence"][0]["first"]["provider_receipts"][0]["provider"] = "azure"

    mutations.append(change_receipt_provider)

    def fabricate_bedrock_response_model(payload: dict[str, Any]) -> None:
        payload["evidence"][0]["first"]["provider_receipts"][0]["response_model"] = "unfrozen-model"

    mutations.append(fabricate_bedrock_response_model)

    def add_seed(payload: dict[str, Any]) -> None:
        payload["evidence"][0]["first"]["provider_receipts"][0]["seed_supplied"] = True

    mutations.append(add_seed)

    def change_temperature(payload: dict[str, Any]) -> None:
        payload["evidence"][0]["first"]["provider_receipts"][0]["temperature"] = 0.1

    mutations.append(change_temperature)

    def change_pair_generation(payload: dict[str, Any]) -> None:
        payload["evidence"][0]["pair_generation_id"] = "sha256:" + "8" * 64

    mutations.append(change_pair_generation)

    def reverse_blocks(payload: dict[str, Any]) -> None:
        payload["evidence"] = list(reversed(payload["evidence"]))

    mutations.append(reverse_blocks)

    for mutate in mutations:
        payload = copy.deepcopy(canonical)
        mutate(payload)
        with pytest.raises(ValueError):
            mod.PairedHarborRunReport.model_validate(payload)


def test_reload_enforces_exact_call_count_controls_and_report_wide_request_uniqueness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    _install_fake_evaluator(monkeypatch, candidate=candidate)
    report = asyncio.run(
        _runner(tmp_path, candidate, baseline=baseline).run(
            baseline=baseline,
            candidate=candidate,
        )
    )
    canonical = cast("dict[str, Any]", report.model_dump(mode="json"))

    wrong_count = copy.deepcopy(canonical["evidence"][0]["first"])
    wrong_count["trial"]["usage"]["calls"] = 2
    _refresh_arm_admission_digest(wrong_count)
    with pytest.raises(ValueError, match="receipt count differs"):
        mod.PairedHarborArmEvidence.model_validate(wrong_count)

    unavailable_count = copy.deepcopy(canonical["evidence"][0]["first"])
    unavailable_count["trial"]["usage"]["calls"] = None
    unavailable_count["trial"]["usage"]["calls_status"] = "unavailable"
    _refresh_arm_admission_digest(unavailable_count)
    with pytest.raises(ValueError, match="lacks an exact provider call count"):
        mod.PairedHarborArmEvidence.model_validate(unavailable_count)

    altered_controls = copy.deepcopy(canonical)
    altered_arm = altered_controls["evidence"][0]["first"]
    altered_arm["provider_receipts"][0]["temperature"] = 0.1
    _refresh_arm_admission_digest(altered_arm)
    with pytest.raises(ValueError, match="frozen temperature"):
        mod.PairedHarborRunReport.model_validate(altered_controls)

    reused_request = copy.deepcopy(canonical)
    first_id = reused_request["evidence"][0]["first"]["provider_receipts"][0]["provider_request_id"]
    second_arm = reused_request["evidence"][0]["second"]
    second_arm["provider_receipts"][0]["provider_request_id"] = first_id
    _refresh_arm_admission_digest(second_arm)
    with pytest.raises(ValueError, match="reuses a provider request ID"):
        mod.PairedHarborRunReport.model_validate(reused_request)


def test_zero_successful_calls_are_admissible_only_for_candidate_owned_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    _install_fake_evaluator(monkeypatch, candidate=candidate)
    report = asyncio.run(
        _runner(tmp_path, candidate, baseline=baseline).run(
            baseline=baseline,
            candidate=candidate,
        )
    )
    arm = cast("dict[str, Any]", report.evidence[0].first.model_dump(mode="json"))
    arm["trial"]["status"] = "candidate_failure"
    arm["trial"]["candidate_outcome"] = BenchmarkCandidateOutcome(
        status=BenchmarkCandidateStatus.FAILED,
        stage=BenchmarkCandidateStage.EXECUTION,
        failure_reason=BenchmarkCandidateFailureReason.INVALID_REQUEST,
    ).model_dump(mode="json")
    arm["trial"]["run_health"] = "candidate_damaged"
    arm["trial"]["rewards"] = None
    arm["trial"]["usage"]["calls"] = 0
    arm["verifier_reward"] = None
    arm["analysis_score"] = 0.0
    arm["provider_receipts"] = []
    arm["provider_receipt_call_indexes"] = []
    _refresh_arm_admission_digest(arm)

    admitted = mod.PairedHarborArmEvidence.model_validate(arm)
    assert admitted.trial.candidate_outcome.failure_reason is (
        BenchmarkCandidateFailureReason.INVALID_REQUEST
    )

    arm["trial"]["status"] = "scored"
    arm["trial"]["candidate_outcome"] = BenchmarkCandidateOutcome(
        status=BenchmarkCandidateStatus.COMPLETED,
    ).model_dump(mode="json")
    arm["trial"]["run_health"] = "valid"
    arm["trial"]["rewards"] = {"reward": 0.0}
    _refresh_arm_admission_digest(arm)
    with pytest.raises(ValueError, match="lacks provider-authored request receipts"):
        mod.PairedHarborArmEvidence.model_validate(arm)


def test_same_host_operations_share_global_route_and_task_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    route = mod.PairedHarborPanelRoute(
        panel_member="worker",
        provider_config=_provider(),
        max_concurrent_blocks=1,
    )
    active = 0
    maximum = 0

    class CapacityEvaluator:
        def __init__(
            self,
            spec: HarborJobSpec,
            provider_config: ProviderConfig,
            **_kwargs: object,
        ) -> None:
            self._spec = spec
            self._provider = provider_config
            self._budget_policy_digest = cast(
                "BudgetAccount", _kwargs["budget_account"]
            ).policy.policy_digest

        async def evaluate(self, harness: HarnessDoc) -> LoadedHarborJobResult:
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            try:
                await asyncio.sleep(0.005)
                reward = float(harness.execution_digest == candidate.execution_digest)
                return _loaded_result(
                    self._spec,
                    self._provider,
                    harness,
                    reward=reward,
                    budget_policy_digest=self._budget_policy_digest,
                )
            finally:
                active -= 1

    monkeypatch.setattr(mod, "HarborEvaluator", CapacityEvaluator)
    first = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        operation_id="operation-a",
        panel_routes=(route,),
        max_concurrent_blocks=1,
    )
    second = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        operation_id="operation-b",
        panel_routes=(route,),
        max_concurrent_blocks=1,
    )

    async def run_both() -> None:
        await asyncio.gather(
            first.run(baseline=baseline, candidate=candidate),
            second.run(baseline=baseline, candidate=candidate),
        )

    asyncio.run(run_both())
    assert maximum == 1


@pytest.mark.skipif(os.name != "posix", reason="local file leases require POSIX")
def test_local_block_capacity_is_enforced_across_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    first_ready = context.Event()
    first_release = context.Event()
    second_ready = context.Event()
    second_release = context.Event()
    first = context.Process(
        target=_hold_local_block_lease,
        args=(str(tmp_path / "jobs"), first_ready, first_release),
    )
    second = context.Process(
        target=_hold_local_block_lease,
        args=(str(tmp_path / "jobs"), second_ready, second_release),
    )
    try:
        first.start()
        assert first_ready.wait(timeout=5)
        second.start()
        assert not second_ready.wait(timeout=0.2)
        first_release.set()
        assert second_ready.wait(timeout=5)
        second_release.set()
        first.join(timeout=5)
        second.join(timeout=5)
        assert first.exitcode == 0
        assert second.exitcode == 0
    finally:
        first_release.set()
        second_release.set()
        for process in (first, second):
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def test_local_block_lease_releases_partial_acquisition_on_noncontention_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = mod._LocalPairedHarborLeaseCoordinator(tmp_path / "jobs")
    block = mod.PairedBlock(
        task_id="task-a",
        panel_member="worker",
        attempt=1,
        first_arm=PairedArm.BASELINE,
    )
    acquisition_calls = 0
    released = False

    @contextmanager
    def tracked_lease() -> Iterator[None]:
        nonlocal released
        try:
            yield
        finally:
            released = True

    def enter_first_available(stack: ExitStack, _paths: tuple[Path, ...]) -> None:
        nonlocal acquisition_calls
        acquisition_calls += 1
        if acquisition_calls == 1:
            stack.enter_context(tracked_lease())
            return
        raise OSError("irregular route lease")

    monkeypatch.setattr(mod, "_enter_first_available_lease", enter_first_available)

    async def acquire() -> None:
        async with coordinator.block_lease(
            protocol_digest="sha256:" + "a" * 64,
            block=block,
            max_concurrent_blocks=1,
            max_concurrent_route_blocks=1,
        ):
            pytest.fail("block lease unexpectedly succeeded")

    with pytest.raises(OSError, match="irregular route lease"):
        asyncio.run(acquire())
    assert released is True


def test_scheduler_is_bounded_route_fair_and_serializes_each_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    design = PairedEvaluationDesign.create(
        tasks=tuple(PairedTaskPlan(task_id=task_id, group_id=task_id) for task_id in _TASK_IDS),
        panel=(
            PairedPanelPlan(panel_member="route-a", attempts=2),
            PairedPanelPlan(panel_member="route-b", attempts=2),
        ),
        primary_e_value_bets=(BoundedMeanBet(fraction=1.0, weight=1.0),),
        schedule_seed="fair-schedule",
        analysis_seed="fair-analysis",
        randomization_samples=1_000,
        minimum_equal_task_member_delta=0.03,
        noninferiority_margin=0.02,
    )
    providers = {
        "worker-a": ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model="worker-a",
            region="us-west-2",
        ),
        "worker-b": ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model="worker-b",
            region="us-west-2",
        ),
    }
    routes = (
        mod.PairedHarborPanelRoute(
            panel_member="route-a",
            provider_config=providers["worker-a"],
            max_concurrent_blocks=1,
        ),
        mod.PairedHarborPanelRoute(
            panel_member="route-b",
            provider_config=providers["worker-b"],
            max_concurrent_blocks=1,
        ),
    )
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        design=design,
        panel_routes=routes,
        max_concurrent_blocks=2,
    )

    active_global = 0
    maximum_global = 0
    active_routes: Counter[str] = Counter()
    maximum_routes: Counter[str] = Counter()
    active_tasks: Counter[str] = Counter()
    maximum_tasks: Counter[str] = Counter()
    starts: list[tuple[str, str]] = []

    class FairEvaluator:
        def __init__(
            self,
            spec: HarborJobSpec,
            provider_config: ProviderConfig,
            **_kwargs: object,
        ) -> None:
            self._spec = spec
            self._provider = provider_config
            self._budget_policy_digest = cast(
                "BudgetAccount", _kwargs["budget_account"]
            ).policy.policy_digest

        async def evaluate(self, harness: HarnessDoc) -> LoadedHarborJobResult:
            nonlocal active_global, maximum_global
            task_names = self._spec.datasets[0].task_names
            assert task_names is not None
            task_id = task_names[0]
            model = self._provider.model
            starts.append((model, task_id))
            active_global += 1
            active_routes[model] += 1
            active_tasks[task_id] += 1
            maximum_global = max(maximum_global, active_global)
            maximum_routes[model] = max(maximum_routes[model], active_routes[model])
            maximum_tasks[task_id] = max(maximum_tasks[task_id], active_tasks[task_id])
            try:
                await asyncio.sleep(0.001)
                reward = float(harness.execution_digest == candidate.execution_digest)
                return _loaded_result(
                    self._spec,
                    self._provider,
                    harness,
                    reward=reward,
                    budget_policy_digest=self._budget_policy_digest,
                )
            finally:
                active_global -= 1
                active_routes[model] -= 1
                active_tasks[task_id] -= 1

    monkeypatch.setattr(mod, "HarborEvaluator", FairEvaluator)
    report = asyncio.run(runner.run(baseline=baseline, candidate=candidate))

    assert len(report.evidence) == len(design.blocks)
    assert maximum_global == 2
    assert maximum_routes == {"worker-a": 1, "worker-b": 1}
    assert all(value == 1 for value in maximum_tasks.values())
    assert {model for model, _task in starts[:2]} == {"worker-a", "worker-b"}
    assert starts[0][1] != starts[1][1]
