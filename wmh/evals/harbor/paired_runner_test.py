"""Tests for exact paired Harbor execution and evidence admission."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import multiprocessing
import os
from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import ExitStack, asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Protocol, cast

import pytest
from harbor.models.job.config import DatasetConfig

import wmh.evals.harbor.paired_runner as mod
from wmh.core.types import JsonObject
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
from wmh.evals.harbor.receipt_trace import validate_provider_receipt_trace
from wmh.evals.harbor.results import HarborTrialLocator, LoadedHarborJobResult
from wmh.evals.paired import (
    BoundedMeanBet,
    PairedArm,
    PairedBlock,
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
from wmh.tracking._testing import (
    synthetic_provider_cost_meter,
    synthetic_tariff_provenance,
)
from wmh.tracking.budget import (
    BudgetAccount,
    BudgetPolicy,
    SpendLedger,
    TimedResourceBudgetAccount,
    TimedResourceClass,
    TimedResourceCostMeter,
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


def _slice_policy(
    *,
    max_new_blocks: int = 4,
    max_waves_per_invocation: int = 4,
) -> mod.PairedHarborSlicePolicy:
    return mod.PairedHarborSlicePolicy(
        max_new_blocks=max_new_blocks,
        max_waves_per_invocation=max_waves_per_invocation,
        max_block_runtime_s=900,
        max_invocation_runtime_s=max(7_200, max_waves_per_invocation * 900 + 1),
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
    assert plan.create_rate_policy is None
    assert plan.reward_key == "reward"
    assert plan.artifact_paths == ("agent/log.txt",)
    assert "task" not in plan.model_dump(mode="json")
    assert "jobs_dir" not in plan.model_dump(mode="json")


def test_execution_plan_admits_long_official_task_timeout_with_e2b_runner_lease() -> None:
    """The sealed plan remains generic enough for long official benchmark tasks."""
    runner_spec = E2BPiRunnerSpec(
        template_id="runner-template",
        build_id="runner-build",
        cpu_count=2,
        memory_mb=1024,
        platform="linux/x86_64",
        envd_version="v1",
        lease_timeout_s=18_000,
    )

    plan = mod.HarborExecutionPlan.freeze(
        reference_harness=pi_node_baseline("baseline"),
        reward_key="reward",
        environment_backend=HarborEnvironmentBackend.E2B,
        runner_spec=runner_spec,
        turn_timeout_s=12_000,
    )

    assert plan.turn_timeout_s == 12_000
    assert plan.compute_envelope.turn_timeout_s == 12_000
    assert isinstance(plan.runner_spec, E2BPiRunnerSpec)
    assert plan.runner_spec.lease_timeout_s == 18_000
    assert plan.create_rate_policy is not None
    assert plan.create_rate_policy.maximum_dispatches == 4
    assert plan.create_rate_policy.period_milliseconds == 1000
    assert plan.create_rate_policy.minimum_spacing_ns == 250_000_000


def test_slice_policy_is_path_free_and_fails_closed_on_unsafe_runtime_bounds(
    tmp_path: Path,
) -> None:
    policy = mod.PairedHarborSlicePolicy(
        max_new_blocks=24,
        max_waves_per_invocation=1,
        max_block_runtime_s=28_800,
        max_invocation_runtime_s=82_800,
    )

    assert policy.digest.startswith("sha256:")
    assert "path" not in policy.model_dump(mode="json")
    with pytest.raises(ValueError, match="less than or equal to 82800"):
        mod.PairedHarborSlicePolicy(
            max_new_blocks=1,
            max_waves_per_invocation=1,
            max_block_runtime_s=1,
            max_invocation_runtime_s=86_400,
        )
    with pytest.raises(ValueError, match="leave headroom"):
        mod.PairedHarborSlicePolicy(
            max_new_blocks=2,
            max_waves_per_invocation=2,
            max_block_runtime_s=1_000,
            max_invocation_runtime_s=1_999,
        )

    with pytest.raises(ValueError, match="leave headroom"):
        mod.PairedHarborSlicePolicy(
            max_new_blocks=2,
            max_waves_per_invocation=2,
            max_block_runtime_s=1_000,
            max_invocation_runtime_s=2_000,
        )

    below_two_arms = mod.PairedHarborSlicePolicy(
        max_new_blocks=1,
        max_waves_per_invocation=1,
        max_block_runtime_s=599,
        max_invocation_runtime_s=1_000,
    )
    with pytest.raises(ValueError, match="below two frozen arm timeouts"):
        _runner(tmp_path / "arm-timeout", _candidate(), slice_policy=below_two_arms)

    above_capacity = mod.PairedHarborSlicePolicy(
        max_new_blocks=3,
        max_waves_per_invocation=2,
        max_block_runtime_s=900,
        max_invocation_runtime_s=2_000,
    )
    with pytest.raises(ValueError, match="exceeds its frozen wave capacity"):
        _runner(
            tmp_path / "capacity",
            _candidate(),
            max_concurrent_blocks=1,
            slice_policy=above_capacity,
        )


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
        slice_policy=_slice_policy(),
        retry_policy_digest=_RETRY_POLICY_DIGEST,
        budget_runtime=budget,
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
        slice_policy=_slice_policy(),
        retry_policy_digest=_RETRY_POLICY_DIGEST,
        budget_runtime=budget,
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


def test_candidate_freeze_rejects_wrong_full_qualification_roster_before_persisting(
    tmp_path: Path,
) -> None:
    """A same-sized replacement held-out roster must fail before candidate freeze."""
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
            experiment_id="paired-wrong-roster-test",
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
    budget = _budget_runtime(tmp_path, routes)
    wrong_roster = mod.PrequalifiedHarborRoster(
        execution_plan_digest=plan.digest,
        tasks=tuple(
            qualification
            for qualification in _qualifications()
            if qualification.task_id != "task-b"
        )
        + (
            mod.QualifiedHarborTask(
                task_id="task-replacement",
                dataset_id="terminalbench2",
                content_digest="sha256:" + "9" * 64,
                task_key="sha256:" + "8" * 64,
                task_environment_digest="sha256:" + "7" * 64,
                environment_backend=HarborEnvironmentBackend.LOCAL,
            ),
        ),
    )
    wrong_commitment = mod.HarborConfirmationExecutionCommitment.freeze(
        discovery=manifest.discovery_view(),
        design_template=PairedEvaluationDesignTemplate.from_design(_design()),
        baseline=baseline,
        candidate=candidate,
        execution_plan=plan,
        panel_routes=routes,
        qualification_roster=wrong_roster,
        max_concurrent_blocks=1,
        slice_policy=_slice_policy(),
        retry_policy_digest=_RETRY_POLICY_DIGEST,
        budget_runtime=budget,
    )

    with pytest.raises(ValueError, match="full qualification roster differs"):
        mod.freeze_harbor_confirmation_candidate(
            store,
            manifest=manifest,
            commitment=wrong_commitment,
        )

    correct_roster = mod.PrequalifiedHarborRoster(
        execution_plan_digest=plan.digest,
        tasks=_qualifications(),
    )
    correct_commitment = mod.HarborConfirmationExecutionCommitment.freeze(
        discovery=manifest.discovery_view(),
        design_template=PairedEvaluationDesignTemplate.from_design(_design()),
        baseline=baseline,
        candidate=candidate,
        execution_plan=plan,
        panel_routes=routes,
        qualification_roster=correct_roster,
        max_concurrent_blocks=1,
        slice_policy=_slice_policy(),
        retry_policy_digest=_RETRY_POLICY_DIGEST,
        budget_runtime=budget,
    )
    frozen = mod.freeze_harbor_confirmation_candidate(
        store,
        manifest=manifest,
        commitment=correct_commitment,
    )
    assert frozen.confirmation_protocol_digest == correct_commitment.digest


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
            meter_by_member[route.panel_member]: synthetic_provider_cost_meter(
                provider_config=route.provider_config,
                provenance=synthetic_tariff_provenance(route.provider_config),
                input_nano_usd_per_token=1,
                output_nano_usd_per_token=5,
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
    slice_policy: mod.PairedHarborSlicePolicy | None = None,
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
        slice_policy=slice_policy or _slice_policy(),
        retry_policy_digest=_RETRY_POLICY_DIGEST,
        budget_runtime=budget,
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
                provider_meters[route.panel_member]: synthetic_provider_cost_meter(
                    provider_config=route.provider_config,
                    provenance=synthetic_tariff_provenance(route.provider_config),
                    input_nano_usd_per_token=1,
                    output_nano_usd_per_token=5,
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


def test_preopen_commitment_rejects_budget_route_name_drift(tmp_path: Path) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    routes = (mod.PairedHarborPanelRoute(panel_member="worker", provider_config=_provider()),)
    plan = mod.HarborExecutionPlan.freeze(reference_harness=baseline, reward_key="reward")
    roster = mod.PrequalifiedHarborRoster(
        execution_plan_digest=plan.digest,
        tasks=_qualifications(),
    )
    budget = _budget_runtime(tmp_path, routes)
    drifted_budget = budget.model_copy(
        update={
            "provider_meter_by_panel_member": {
                "workre": budget.provider_meter_by_panel_member["worker"]
            }
        }
    )

    with pytest.raises(ValueError, match="routes differ from the frozen panel"):
        mod.HarborConfirmationExecutionCommitment.freeze(
            discovery=_discovery(),
            design_template=PairedEvaluationDesignTemplate.from_design(_design()),
            baseline=baseline,
            candidate=candidate,
            execution_plan=plan,
            panel_routes=routes,
            qualification_roster=roster,
            max_concurrent_blocks=1,
            slice_policy=_slice_policy(),
            retry_policy_digest=_RETRY_POLICY_DIGEST,
            budget_runtime=drifted_budget,
        )


def test_preopen_commitment_rejects_provider_meter_route_drift(tmp_path: Path) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    original_routes = (
        mod.PairedHarborPanelRoute(panel_member="worker", provider_config=_provider()),
    )
    drifted_routes = (
        mod.PairedHarborPanelRoute(
            panel_member="worker",
            provider_config=_provider().model_copy(update={"model": "different-worker-model"}),
        ),
    )
    plan = mod.HarborExecutionPlan.freeze(reference_harness=baseline, reward_key="reward")
    roster = mod.PrequalifiedHarborRoster(
        execution_plan_digest=plan.digest,
        tasks=_qualifications(),
    )
    budget = _budget_runtime(tmp_path, original_routes)

    with pytest.raises(ValueError, match="differs from its provider route"):
        mod.HarborConfirmationExecutionCommitment.freeze(
            discovery=_discovery(),
            design_template=PairedEvaluationDesignTemplate.from_design(_design()),
            baseline=baseline,
            candidate=candidate,
            execution_plan=plan,
            panel_routes=drifted_routes,
            qualification_roster=roster,
            max_concurrent_blocks=1,
            slice_policy=_slice_policy(),
            retry_policy_digest=_RETRY_POLICY_DIGEST,
            budget_runtime=budget,
        )


@pytest.mark.parametrize(
    ("budget_update", "error"),
    [
        (
            {"task_resource_meter_by_class_digest": {}},
            "task resource meters differ from full-roster E2B qualification classes",
        ),
        (
            {"runner_resource_meter_id": None},
            "no E2B runner meter",
        ),
    ],
)
def test_preopen_commitment_rejects_incomplete_e2b_resource_meters(
    tmp_path: Path,
    budget_update: dict[str, object],
    error: str,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    routes = (mod.PairedHarborPanelRoute(panel_member="worker", provider_config=_provider()),)
    runner_spec = _e2b_runner_spec()
    plan = mod.HarborExecutionPlan.freeze(
        reference_harness=baseline,
        reward_key="reward",
        environment_backend=HarborEnvironmentBackend.E2B,
        runner_spec=runner_spec,
    )
    roster = mod.PrequalifiedHarborRoster(
        execution_plan_digest=plan.digest,
        tasks=_e2b_qualifications(),
    )
    budget = _e2b_budget_runtime(tmp_path, routes, runner_spec).model_copy(update=budget_update)

    with pytest.raises(ValueError, match=error):
        mod.HarborConfirmationExecutionCommitment.freeze(
            discovery=_discovery(),
            design_template=PairedEvaluationDesignTemplate.from_design(_design()),
            baseline=baseline,
            candidate=candidate,
            execution_plan=plan,
            panel_routes=routes,
            qualification_roster=roster,
            max_concurrent_blocks=1,
            slice_policy=_slice_policy(),
            retry_policy_digest=_RETRY_POLICY_DIGEST,
            budget_runtime=budget,
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
    )
    return mod.PairedHarborRunner(
        protocol=protocol,
        runtime=mod.HarborExecutionRuntime(
            jobs_dir=(tmp_path / "jobs").resolve(),
            dataset_paths_by_id={"terminalbench2": (tmp_path / "dataset").resolve()},
            budget=budget,
            create_rate_ledger_path=(tmp_path / "paired-create-rate.json").resolve(),
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
    slice_policy = cast(
        "mod.PairedHarborSlicePolicy",
        updates.pop(
            "slice_policy",
            _slice_policy(
                max_new_blocks=len(design.blocks),
                max_waves_per_invocation=len(design.blocks),
            ),
        ),
    )
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
        slice_policy=slice_policy,
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
    response_identity: mod.ProviderResponseIdentity | None = None,
    failure_kind: BenchmarkFailureKind | None = None,
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
        status=(
            BenchmarkTrialStatus.SCORED
            if failure_kind is None
            else BenchmarkTrialStatus.INFRASTRUCTURE_ERROR
        ),
        rewards={"reward": reward} if failure_kind is None else None,
        error=(
            None
            if failure_kind is None
            else BenchmarkError(
                kind=failure_kind,
                type="SyntheticInfrastructureError",
                message="synthetic infrastructure failure",
            )
        ),
        candidate_outcome=BenchmarkCandidateOutcome(
            status=(
                BenchmarkCandidateStatus.COMPLETED
                if failure_kind is None
                else BenchmarkCandidateStatus.UNKNOWN
            ),
        ),
        run_health=(
            BenchmarkRunHealth.VALID if failure_kind is None else BenchmarkRunHealth.RETRY_REQUIRED
        ),
        usage=BenchmarkUsage(calls=1),
    )
    job_dir = spec.jobs_dir / spec.job_name
    trial_dir = Path("trial")
    (job_dir / trial_dir).mkdir(parents=True, exist_ok=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "started_at": "2026-07-19T00:00:00Z",
                "finished_at": "2026-07-19T00:00:01Z",
                "n_total_trials": 1,
                "stats": {
                    "n_completed_trials": 1,
                    "n_running_trials": 0,
                    "n_pending_trials": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
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
        response_identity=response_identity,
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
    result_transform: Callable[[LoadedHarborJobResult], LoadedHarborJobResult] | None = None,
) -> tuple[
    list[tuple[HarborJobSpec, ProviderConfig, HarnessDoc]],
    dict[str, int],
]:
    calls: list[tuple[HarborJobSpec, ProviderConfig, HarnessDoc]] = []
    active: defaultdict[str, int] = defaultdict(int)
    maximum: defaultdict[str, int] = defaultdict(int)
    terminal_results: dict[str, LoadedHarborJobResult] = {}

    class FakeEvaluator:
        def __init__(
            self,
            spec: HarborJobSpec,
            provider_config: ProviderConfig,
            *,
            runner_spec: object,
            turn_timeout_s: float,
            require_provider_receipts: bool,
            response_identity: mod.ProviderResponseIdentity,
            session: object,
            budget_account: BudgetAccount,
            task_resource_budget_accounts: tuple[object, ...],
            runner_resource_budget_account: object | None,
            qualified_tasks: tuple[mod.QualifiedHarborTask, ...],
        ) -> None:
            assert isinstance(runner_spec, LocalPiRunnerSpec)
            assert turn_timeout_s == 300.0
            assert require_provider_receipts is True
            assert response_identity.provider is provider_config.kind
            assert isinstance(session, mod.HarborEvaluatorSession)
            assert task_resource_budget_accounts == ()
            assert runner_resource_budget_account is None
            assert len(qualified_tasks) == 1
            assert qualified_tasks[0].task_id in {"task-a", "task-b"}
            self._spec = spec
            self._provider = provider_config
            self._budget_policy_digest = budget_account.policy.policy_digest
            self._response_identity = response_identity

        async def evaluate(self, harness: HarnessDoc) -> LoadedHarborJobResult:
            existing = terminal_results.get(self._spec.job_name)
            if existing is not None:
                maximum["resume-existing"] += 1
                root_result_path = existing.job_dir / "result.json"
                root_result = json.loads(root_result_path.read_text(encoding="utf-8"))
                root_result["finished_at"] = "2026-07-19T00:00:01Z"
                root_result["stats"].update(
                    {
                        "n_completed_trials": 1,
                        "n_running_trials": 0,
                        "n_pending_trials": 0,
                    }
                )
                root_result_path.write_text(
                    json.dumps(root_result) + "\n",
                    encoding="utf-8",
                )
                return existing
            task_names = self._spec.datasets[0].task_names
            assert task_names is not None
            task_id = task_names[0]
            active[task_id] += 1
            maximum[task_id] = max(maximum[task_id], active[task_id])
            try:
                await asyncio.sleep(0)
                calls.append((self._spec, self._provider, harness))
                if fail_job is not None and len(calls) == fail_job:
                    loaded = _loaded_result(
                        self._spec,
                        self._provider,
                        harness,
                        reward=0.0,
                        budget_policy_digest=self._budget_policy_digest,
                        response_identity=self._response_identity,
                        failure_kind=BenchmarkFailureKind.PROVIDER,
                    )
                    loaded = loaded if result_transform is None else result_transform(loaded)
                    terminal_results[self._spec.job_name] = loaded
                    return loaded
                reward = float(harness.execution_digest == candidate.execution_digest)
                loaded = _loaded_result(
                    self._spec,
                    self._provider,
                    harness,
                    reward=reward,
                    budget_policy_digest=self._budget_policy_digest,
                    response_identity=self._response_identity,
                )
                loaded = loaded if result_transform is None else result_transform(loaded)
                terminal_results[self._spec.job_name] = loaded
                return loaded
            except asyncio.CancelledError:
                maximum["cancelled"] += 1
                raise
            finally:
                active[task_id] -= 1

        async def load_existing(self, _harness: HarnessDoc) -> LoadedHarborJobResult:
            maximum["load-existing"] += 1
            return terminal_results[self._spec.job_name]

    monkeypatch.setattr(mod, "HarborEvaluator", FakeEvaluator)
    return calls, maximum


def _refresh_arm_admission_digest(arm: JsonObject) -> None:
    arm["admission_digest"] = mod._canonical_digest(
        {key: value for key, value in arm.items() if key != "admission_digest"}
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
    assert report.run_version == "11"
    assert report.protocol.protocol_version == "9"
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


def test_bounded_slices_complete_across_restarts_without_replaying_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    policy = _slice_policy(max_new_blocks=2, max_waves_per_invocation=1)
    first_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    first_runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        slice_policy=policy,
    )

    first = asyncio.run(first_runner.run_slice(baseline=baseline, candidate=candidate))

    assert first.report is None
    assert first.progress.selected_blocks == (_design().blocks[0], _design().blocks[2])
    assert first.progress.completed_block_count == 2
    assert first.progress.remaining_block_count == 2
    assert len(first_calls) == 4

    second_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    second_runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        slice_policy=policy,
    )
    second = asyncio.run(second_runner.run_slice(baseline=baseline, candidate=candidate))

    assert second.report is not None
    assert second.progress.selected_blocks == (_design().blocks[1], _design().blocks[3])
    assert second.progress.completed_block_count == len(_design().blocks)
    assert second.progress.remaining_block_count == 0
    assert second.progress.previous_progress_digest == first.progress.progress_digest
    assert len(second_calls) == 4
    assert tuple(item.block for item in second.report.evidence) == _design().blocks

    late_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    late = asyncio.run(
        _runner(
            tmp_path,
            candidate,
            baseline=baseline,
            slice_policy=policy,
        ).run_slice(baseline=baseline, candidate=candidate)
    )

    assert late.progress == second.progress
    assert late.report == second.report
    assert late_calls == []


def test_recover_persisted_slice_closes_genesis_evidence_ahead_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    policy = _slice_policy(max_new_blocks=2, max_waves_per_invocation=1)
    execution_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        slice_policy=policy,
    )

    class SyntheticCheckpointCrash(BaseException):
        pass

    def crash_before_progress(_progress: mod.PairedHarborSliceProgress) -> None:
        raise SyntheticCheckpointCrash

    monkeypatch.setattr(runner, "_persist_progress", crash_before_progress)
    with pytest.raises(SyntheticCheckpointCrash):
        asyncio.run(runner.run_slice(baseline=baseline, candidate=candidate))
    assert runner._load_progress_chain() == ()
    assert len(execution_calls) == 4

    recovery_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    reconstructed = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        slice_policy=policy,
    )
    recovered = asyncio.run(
        reconstructed.recover_persisted_slice(
            baseline=baseline,
            candidate=candidate,
        )
    )

    assert recovered is not None
    assert recovered.progress.selected_blocks == mod._select_paired_harbor_slice_blocks(
        reconstructed._protocol,
        completed_blocks=frozenset(),
        max_new_blocks=2,
    )
    assert recovered.progress.completed_block_count == 2
    assert reconstructed._load_progress_chain() == (recovered.progress,)
    assert recovery_calls == []


def test_recover_persisted_slice_reuses_matching_checkpoint_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    policy = _slice_policy(max_new_blocks=2, max_waves_per_invocation=1)
    _install_fake_evaluator(monkeypatch, candidate=candidate)
    first = asyncio.run(
        _runner(
            tmp_path,
            candidate,
            baseline=baseline,
            slice_policy=policy,
        ).run_slice(baseline=baseline, candidate=candidate)
    )

    recovery_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    recovered = asyncio.run(
        _runner(
            tmp_path,
            candidate,
            baseline=baseline,
            slice_policy=policy,
        ).recover_persisted_slice(baseline=baseline, candidate=candidate)
    )

    assert recovered == first
    assert recovery_calls == []
    with pytest.raises(mod.PairedHarborNoActiveSliceIntentError):
        asyncio.run(
            _runner(
                tmp_path,
                candidate,
                baseline=baseline,
                slice_policy=policy,
            ).resume_persisted_slice(baseline=baseline, candidate=candidate)
        )


def test_recover_persisted_slice_closes_later_evidence_ahead_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    policy = _slice_policy(max_new_blocks=2, max_waves_per_invocation=1)
    _install_fake_evaluator(monkeypatch, candidate=candidate)
    first_runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        slice_policy=policy,
    )
    first = asyncio.run(first_runner.run_slice(baseline=baseline, candidate=candidate))

    second_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    second_runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        slice_policy=policy,
    )

    class SyntheticCheckpointCrash(BaseException):
        pass

    def crash_before_progress(_progress: mod.PairedHarborSliceProgress) -> None:
        raise SyntheticCheckpointCrash

    monkeypatch.setattr(second_runner, "_persist_progress", crash_before_progress)
    with pytest.raises(SyntheticCheckpointCrash):
        asyncio.run(second_runner.run_slice(baseline=baseline, candidate=candidate))
    assert len(second_calls) == 4
    assert second_runner._load_progress_chain() == (first.progress,)

    recovery_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    reconstructed = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        slice_policy=policy,
    )
    recovered = asyncio.run(
        reconstructed.recover_persisted_slice(
            baseline=baseline,
            candidate=candidate,
        )
    )

    assert recovered is not None
    assert recovered.report is not None
    assert recovered.progress.previous_progress_digest == first.progress.progress_digest
    assert recovered.progress.completed_before == first.progress.completed_blocks
    assert reconstructed._load_progress_chain() == (first.progress, recovered.progress)
    assert recovery_calls == []


def test_recover_persisted_slice_uses_changed_current_bound_from_durable_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    policy = _slice_policy(max_new_blocks=2, max_waves_per_invocation=1)
    _install_fake_evaluator(monkeypatch, candidate=candidate)
    first = asyncio.run(
        _runner(
            tmp_path,
            candidate,
            baseline=baseline,
            slice_policy=policy,
        ).run_slice(
            baseline=baseline,
            candidate=candidate,
            max_new_blocks=2,
        )
    )

    second_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    second_runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        slice_policy=policy,
    )

    class SyntheticCheckpointCrash(BaseException):
        pass

    def crash_before_progress(_progress: mod.PairedHarborSliceProgress) -> None:
        raise SyntheticCheckpointCrash

    monkeypatch.setattr(second_runner, "_persist_progress", crash_before_progress)
    with pytest.raises(SyntheticCheckpointCrash):
        asyncio.run(
            second_runner.run_slice(
                baseline=baseline,
                candidate=candidate,
                max_new_blocks=1,
            )
        )
    assert len(second_calls) == 2

    recovery_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    reconstructed = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        slice_policy=policy,
    )
    recovered = asyncio.run(
        reconstructed.recover_persisted_slice(
            baseline=baseline,
            candidate=candidate,
        )
    )

    assert recovered is not None
    assert recovered.progress.previous_progress_digest == first.progress.progress_digest
    assert recovered.progress.requested_max_new_blocks == 1
    assert recovered.progress.selected_blocks == mod._select_paired_harbor_slice_blocks(
        reconstructed._protocol,
        completed_blocks=frozenset(item.block for item in first.progress.completed_blocks),
        max_new_blocks=1,
    )
    assert recovery_calls == []


def test_recover_persisted_slice_does_not_false_checkpoint_changed_bound_partial_wave(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    policy = _slice_policy(max_new_blocks=2, max_waves_per_invocation=2)
    _install_fake_evaluator(monkeypatch, candidate=candidate)
    first_runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
        slice_policy=policy,
    )
    first = asyncio.run(
        first_runner.run_slice(
            baseline=baseline,
            candidate=candidate,
            max_new_blocks=1,
        )
    )

    second_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    second_runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
        slice_policy=policy,
    )

    class SyntheticBetweenWaveCrash(BaseException):
        pass

    async def complete_one_then_crash(
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
        generation_by_block: dict[PairedBlock, int],
        blocks: tuple[PairedBlock, ...],
    ) -> tuple[mod.PairedHarborBlockEvidence, ...]:
        assert len(blocks) == 2
        await second_runner._run_block(
            blocks[0],
            baseline=baseline,
            candidate=candidate,
            evaluator_session=mod.HarborEvaluatorSession(
                runner_spec=second_runner._protocol.execution_plan.runner_spec
            ),
            generation_id=generation_by_block[blocks[0]],
        )
        raise SyntheticBetweenWaveCrash

    monkeypatch.setattr(second_runner, "_run_fair_matrix", complete_one_then_crash)
    with pytest.raises(SyntheticBetweenWaveCrash):
        asyncio.run(
            second_runner.run_slice(
                baseline=baseline,
                candidate=candidate,
                max_new_blocks=2,
            )
        )
    assert len(second_calls) == 2
    assert second_runner._load_progress_chain() == (first.progress,)
    assert len(second_runner._load_slice_intent_chain()) == 2
    active_selection = second_runner._load_slice_intent_chain()[-1].selected_blocks

    recovery_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    reconstructed = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
        slice_policy=policy,
    )
    recovered = asyncio.run(
        reconstructed.recover_persisted_slice(
            baseline=baseline,
            candidate=candidate,
        )
    )
    assert recovered == first
    assert reconstructed._load_progress_chain() == (first.progress,)
    assert recovery_calls == []

    mismatch_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    with pytest.raises(mod.PairedHarborProgressStateError, match="exact precommitted"):
        asyncio.run(
            reconstructed.run_slice(
                baseline=baseline,
                candidate=candidate,
                max_new_blocks=1,
            )
        )
    assert mismatch_calls == []

    resume_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    resumed = asyncio.run(
        reconstructed.resume_persisted_slice(
            baseline=baseline,
            candidate=candidate,
        )
    )
    assert resumed.progress.requested_max_new_blocks == 2
    assert resumed.progress.selected_blocks == active_selection
    assert resumed.progress.completed_block_count == 3
    assert len(resume_calls) == 2


def test_recover_persisted_slice_returns_none_for_intent_only_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    policy = _slice_policy(max_new_blocks=2, max_waves_per_invocation=1)
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        slice_policy=policy,
    )
    real_persist_intent = runner._persist_slice_intent

    class SyntheticIntentCrash(BaseException):
        pass

    def persist_then_crash(intent: mod.PairedHarborSliceIntent) -> None:
        real_persist_intent(intent)
        raise SyntheticIntentCrash

    monkeypatch.setattr(runner, "_persist_slice_intent", persist_then_crash)
    with pytest.raises(SyntheticIntentCrash):
        asyncio.run(runner.run_slice(baseline=baseline, candidate=candidate))
    assert runner._load_progress_chain() == ()
    assert len(runner._load_slice_intent_chain()) == 1
    active_selection = runner._load_slice_intent_chain()[-1].selected_blocks

    recovery_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    reconstructed = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        slice_policy=policy,
    )
    assert (
        asyncio.run(
            reconstructed.recover_persisted_slice(
                baseline=baseline,
                candidate=candidate,
            )
        )
        is None
    )
    assert recovery_calls == []

    resume_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    resumed = asyncio.run(
        reconstructed.resume_persisted_slice(baseline=baseline, candidate=candidate)
    )
    assert resumed.progress.selected_blocks == active_selection
    assert len(resume_calls) == 4


def test_recover_persisted_slice_rejects_partial_deterministic_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    policy = _slice_policy(max_new_blocks=2, max_waves_per_invocation=1)
    _install_fake_evaluator(monkeypatch, candidate=candidate)
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        slice_policy=policy,
    )
    selected = mod._select_paired_harbor_slice_blocks(
        runner._protocol,
        completed_blocks=frozenset(),
        max_new_blocks=2,
    )
    asyncio.run(
        runner._run_block(
            selected[0],
            baseline=baseline,
            candidate=candidate,
            evaluator_session=mod.HarborEvaluatorSession(
                runner_spec=runner._protocol.execution_plan.runner_spec
            ),
            generation_id=1,
        )
    )
    assert runner._load_progress_chain() == ()

    recovery_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    reconstructed = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        slice_policy=policy,
    )
    with pytest.raises(mod.PairedHarborProgressStateError, match="durable slice intent"):
        asyncio.run(
            reconstructed.recover_persisted_slice(
                baseline=baseline,
                candidate=candidate,
            )
        )
    assert reconstructed._load_progress_chain() == ()
    assert recovery_calls == []


def test_slice_enforces_frozen_invocation_and_wave_deadlines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    policy = _slice_policy(max_new_blocks=2, max_waves_per_invocation=1)
    calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    real_timeout = asyncio.timeout
    observed: list[float | None] = []

    def recording_timeout(delay: float | None) -> asyncio.Timeout:
        observed.append(delay)
        return real_timeout(delay)

    monkeypatch.setattr(mod.asyncio, "timeout", recording_timeout)
    result = asyncio.run(
        _runner(
            tmp_path,
            candidate,
            baseline=baseline,
            slice_policy=policy,
        ).run_slice(baseline=baseline, candidate=candidate)
    )

    assert result.report is None
    assert observed == [policy.max_invocation_runtime_s, policy.max_block_runtime_s]
    assert len(calls) == 4


def test_slice_timeout_fails_before_unbounded_wave_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    policy = _slice_policy(max_new_blocks=2, max_waves_per_invocation=1)
    calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    real_timeout = asyncio.timeout

    class ImmediateTimeout:
        async def __aenter__(self) -> None:
            raise TimeoutError("synthetic wave timeout")

        async def __aexit__(
            self,
            error_type: type[BaseException] | None,
            error: BaseException | None,
            traceback: object,
        ) -> bool:
            del error_type, error, traceback
            return False

    def timeout_for_scope(delay: float | None) -> asyncio.Timeout | ImmediateTimeout:
        if delay == policy.max_block_runtime_s:
            return ImmediateTimeout()
        return real_timeout(delay)

    monkeypatch.setattr(mod.asyncio, "timeout", timeout_for_scope)
    with pytest.raises(mod.PairedHarborSliceTimeoutError, match="wave.*900") as captured:
        asyncio.run(
            _runner(
                tmp_path,
                candidate,
                baseline=baseline,
                slice_policy=policy,
            ).run_slice(baseline=baseline, candidate=candidate)
        )

    assert captured.value.scope == "wave"
    assert calls == []


def test_each_slice_holds_one_operation_lease_through_all_block_leases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    policy = _slice_policy(max_new_blocks=2, max_waves_per_invocation=1)

    class RecordingCoordinator:
        def __init__(self) -> None:
            self.active = False
            self.operation_entries = 0
            self.operation_exits = 0
            self.block_entries = 0

        @asynccontextmanager
        async def operation_lease(
            self,
            *,
            protocol_digest: str,
            operation_id: str,
        ) -> AsyncIterator[None]:
            assert protocol_digest.startswith("sha256:")
            assert operation_id == "offline-test-operation"
            assert not self.active
            self.active = True
            self.operation_entries += 1
            try:
                yield
            finally:
                self.active = False
                self.operation_exits += 1

        @asynccontextmanager
        async def block_lease(
            self,
            *,
            protocol_digest: str,
            block: PairedBlock,
            max_concurrent_blocks: int,
            max_concurrent_route_blocks: int,
        ) -> AsyncIterator[None]:
            assert self.active
            assert protocol_digest.startswith("sha256:")
            assert block in _design().blocks
            assert max_concurrent_blocks == 4
            assert max_concurrent_route_blocks == 2
            self.block_entries += 1
            yield

    coordinator = RecordingCoordinator()
    for _ in range(2):
        _install_fake_evaluator(monkeypatch, candidate=candidate)
        asyncio.run(
            _runner(
                tmp_path,
                candidate,
                baseline=baseline,
                slice_policy=policy,
                durable_coordinator=coordinator,
            ).run_slice(baseline=baseline, candidate=candidate)
        )

    assert coordinator.operation_entries == 2
    assert coordinator.operation_exits == 2
    assert coordinator.block_entries == len(_design().blocks)
    assert not coordinator.active


def test_crash_before_progress_publish_recovers_pairs_without_provider_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    policy = _slice_policy(max_new_blocks=2, max_waves_per_invocation=2)
    crashed_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    crashed_runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        slice_policy=policy,
    )

    def crash_before_publish(_progress: mod.PairedHarborSliceProgress) -> None:
        raise RuntimeError("synthetic coordinator crash before progress publish")

    monkeypatch.setattr(crashed_runner, "_persist_progress", crash_before_publish)
    with pytest.raises(RuntimeError, match="before progress publish"):
        asyncio.run(crashed_runner.run_slice(baseline=baseline, candidate=candidate))
    assert len(crashed_calls) == 4
    assert not crashed_runner._progress_directory().exists()

    resumed_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    resumed = asyncio.run(
        _runner(
            tmp_path,
            candidate,
            baseline=baseline,
            slice_policy=policy,
        ).run_slice(baseline=baseline, candidate=candidate)
    )

    assert resumed.report is None
    assert resumed.progress.slice_index == 1
    assert resumed.progress.completed_before == ()
    assert resumed.progress.selected_blocks == _design().blocks[:2]
    assert resumed_calls == []

    final_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    final = asyncio.run(
        _runner(
            tmp_path,
            candidate,
            baseline=baseline,
            slice_policy=policy,
        ).run_slice(baseline=baseline, candidate=candidate)
    )
    assert final.report is not None
    assert final.progress.selected_blocks == _design().blocks[2:]
    assert len(final_calls) == 4


def test_partial_final_slice_is_smaller_and_analysis_waits_for_complete_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    policy = _slice_policy(max_new_blocks=3, max_waves_per_invocation=2)
    calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    real_analyze = mod.analyze_paired_outcomes

    def reject_partial_analysis(*_args: object, **_kwargs: object) -> None:
        pytest.fail("partial paired evidence reached final analysis")

    monkeypatch.setattr(mod, "analyze_paired_outcomes", reject_partial_analysis)
    first = asyncio.run(
        _runner(
            tmp_path,
            candidate,
            baseline=baseline,
            slice_policy=policy,
        ).run_slice(
            baseline=baseline,
            candidate=candidate,
            max_new_blocks=3,
        )
    )
    assert first.report is None
    assert len(first.progress.selected_blocks) == 3
    assert len(calls) == 6

    monkeypatch.setattr(mod, "analyze_paired_outcomes", real_analyze)
    final_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    final = asyncio.run(
        _runner(
            tmp_path,
            candidate,
            baseline=baseline,
            slice_policy=policy,
        ).run_slice(baseline=baseline, candidate=candidate)
    )

    assert final.report is not None
    assert final.progress.selected_blocks == (_design().blocks[-1],)
    assert len(final_calls) == 2


def test_slice_crash_requires_pair_retry_and_never_replays_completed_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    policy = _slice_policy(max_new_blocks=2, max_waves_per_invocation=2)
    failed_calls, _ = _install_fake_evaluator(
        monkeypatch,
        candidate=candidate,
        fail_job=4,
    )
    first_runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
        slice_policy=policy,
    )

    with pytest.raises(mod.PairedHarborMatrixError):
        asyncio.run(first_runner.run_slice(baseline=baseline, candidate=candidate))
    assert len(failed_calls) == 4
    first_block = first_runner._protocol.design.blocks[0]
    failed_block = first_runner._protocol.design.blocks[1]
    assert (
        mod._read_pair_generation_state(first_runner._pair_state_path(first_block)).status
        == "complete"
    )
    failed_state = mod._read_pair_generation_state(first_runner._pair_state_path(failed_block))
    assert failed_state.status == "failed"

    mod.authorize_paired_harbor_pair_retry(
        jobs_dir=first_runner._runtime.jobs_dir,
        protocol=first_runner._protocol,
        operation_id="offline-test-operation",
        failed_state=failed_state,
    )
    retry_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    resumed = asyncio.run(
        _runner(
            tmp_path,
            candidate,
            baseline=baseline,
            generation_id=2,
            max_concurrent_blocks=1,
            slice_policy=policy,
        ).run_slice(baseline=baseline, candidate=candidate)
    )

    assert resumed.report is None
    assert resumed.progress.completed_block_count == 2
    assert resumed.progress.selected_blocks == _design().blocks[:2]
    assert len(retry_calls) == 2
    completion_generations = {
        item.block: item.generation_id for item in resumed.progress.completed_blocks
    }
    assert completion_generations[first_block] == 1
    assert completion_generations[failed_block] == 2

    final_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    final = asyncio.run(
        _runner(
            tmp_path,
            candidate,
            baseline=baseline,
            generation_id=2,
            max_concurrent_blocks=1,
            slice_policy=policy,
        ).run_slice(baseline=baseline, candidate=candidate)
    )
    assert final.report is not None
    assert len(final_calls) == 4


def test_slice_rejects_policy_progress_and_generation_drift_before_provider_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    policy = _slice_policy(max_new_blocks=2, max_waves_per_invocation=1)
    _install_fake_evaluator(monkeypatch, candidate=candidate)
    first_runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        slice_policy=policy,
    )
    first = asyncio.run(first_runner.run_slice(baseline=baseline, candidate=candidate))

    with pytest.raises(ValueError, match="execution semantics|slice policy digest"):
        mod.PairedHarborProtocol.model_validate(
            first_runner._protocol.model_copy(
                update={"slice_policy_digest": "sha256:" + "f" * 64}
            ).model_dump()
        )
    with pytest.raises(ValueError, match="within the frozen slice policy"):
        asyncio.run(
            first_runner.run_slice(
                baseline=baseline,
                candidate=candidate,
                max_new_blocks=3,
            )
        )

    generation_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    with pytest.raises(mod.PartialPairedHarborReuseError, match="generation.*retry"):
        asyncio.run(
            _runner(
                tmp_path,
                candidate,
                baseline=baseline,
                generation_id=2,
                slice_policy=policy,
            ).run_slice(baseline=baseline, candidate=candidate)
        )
    assert generation_calls == []

    progress_path = first_runner._progress_record_path(first.progress)
    progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
    progress_payload["progress_digest"] = "sha256:" + "e" * 64
    progress_path.write_text(json.dumps(progress_payload), encoding="utf-8")
    progress_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    with pytest.raises(mod.PairedHarborProgressStateError, match="progress"):
        asyncio.run(
            _runner(
                tmp_path,
                candidate,
                baseline=baseline,
                slice_policy=policy,
            ).run_slice(baseline=baseline, candidate=candidate)
        )
    assert progress_calls == []


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
    assert {call[0].job_name for call in first_calls} == set(first_names)
    assert second_calls == []


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


def test_fatal_block_failure_allows_other_reserved_block_to_finish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    both_started = asyncio.Event()
    calls: list[str] = []

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
            await asyncio.sleep(0)
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

    assert len(calls) == 3


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
    first_budget = _budget_runtime(tmp_path / "host-a", (changed_route,))
    changed_commitment = mod.HarborConfirmationExecutionCommitment.freeze(
        discovery=_discovery(),
        design_template=PairedEvaluationDesignTemplate.from_design(_design()),
        baseline=baseline,
        candidate=candidate,
        execution_plan=first.execution_plan,
        panel_routes=(changed_route,),
        qualification_roster=first.qualification_roster,
        max_concurrent_blocks=4,
        slice_policy=_slice_policy(),
        retry_policy_digest=_RETRY_POLICY_DIGEST,
        budget_runtime=first_budget,
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


def test_e2b_rate_ledger_paths_are_host_private_and_not_protocol_identity(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    first = _e2b_runner(tmp_path, candidate)
    alternate_runtime = first._runtime.model_copy(
        update={
            "jobs_dir": (tmp_path / "alternate-jobs").resolve(),
            "dataset_paths_by_id": {"terminalbench2": (tmp_path / "alternate-dataset").resolve()},
            "create_rate_ledger_path": (tmp_path / "alternate-rate.json").resolve(),
        },
        deep=True,
    )

    second = mod.PairedHarborRunner(
        protocol=first._protocol,
        runtime=alternate_runtime,
        operation_id="alternate-e2b-host-paths",
        generation_id=1,
    )

    assert second._protocol == first._protocol
    assert second._protocol.digest == first._protocol.digest
    assert str(tmp_path) not in second._protocol.model_dump_json()
    assert second._create_rate_authority is not first._create_rate_authority
    assert second._create_rate_authority is not None
    assert first._create_rate_authority is not None
    assert second._create_rate_authority.binding != first._create_rate_authority.binding


def test_local_paired_execution_rejects_e2b_rate_ledger(tmp_path: Path) -> None:
    candidate = _candidate()
    runner = _runner(tmp_path, candidate)
    invalid_runtime = runner._runtime.model_copy(
        update={"create_rate_ledger_path": (tmp_path / "unexpected-rate.json").resolve()}
    )

    with pytest.raises(ValueError, match="local paired execution"):
        mod.PairedHarborRunner(
            protocol=runner._protocol,
            runtime=invalid_runtime,
            operation_id="local-rate-input",
            generation_id=1,
        )


def test_e2b_paired_execution_rejects_missing_rate_ledger(tmp_path: Path) -> None:
    runner = _e2b_runner(tmp_path, _candidate())
    invalid_runtime = runner._runtime.model_copy(update={"create_rate_ledger_path": None})

    with pytest.raises(ValueError, match="requires a create-rate ledger"):
        mod.PairedHarborRunner(
            protocol=runner._protocol,
            runtime=invalid_runtime,
            operation_id="missing-e2b-rate-ledger",
            generation_id=1,
        )


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
        "tuple[TimedResourceBudgetAccount, ...]",
        wiring["task_resource_budget_accounts"],
    )
    runner_account = cast(
        "TimedResourceBudgetAccount | None",
        wiring["runner_resource_budget_account"],
    )
    assert spec.environment_backend is HarborEnvironmentBackend.E2B
    assert spec.create_rate_policy == runner._protocol.execution_plan.create_rate_policy
    assert spec.allow_preexisting_e2b_builds is False
    assert wiring["create_rate_authority"] is runner._create_rate_authority
    route = runner._protocol.panel_routes[0]
    assert wiring["response_identity"] == route.response_identity
    assert isinstance(wiring["runner_spec"], E2BPiRunnerSpec)
    assert len(task_accounts) == 1
    assert runner_account is not None
    accounts = (provider_account, task_accounts[0], runner_account)
    assert len({account.ledger_path for account in accounts}) == 1
    assert len({account.ledger_identity for account in accounts}) == 1
    assert len({account.policy.policy_digest for account in accounts}) == 1
    assert len({account.scope.run_id for account in accounts}) == 1
    assert len({account.scope.lane for account in accounts}) == 1
    assert len({account.scope.arm for account in accounts}) == 1


def test_protocol_rejects_opened_e2b_build_drift_on_reload(tmp_path: Path) -> None:
    protocol = _e2b_runner(tmp_path, _candidate())._protocol
    payload = cast("JsonObject", protocol.model_dump(mode="json"))
    opened_selection = cast("JsonObject", payload["opened_selection"])
    tasks = cast("list[JsonObject]", opened_selection["tasks"])
    tasks[0]["e2b_build_record_digest"] = "sha256:" + "f" * 64

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
    with pytest.raises(ValueError, match="at most"):
        mod.PairedHarborPanelRoute(
            panel_member="azure",
            provider_config=azure,
            expected_response_model="m" * 2_049,
        )
    with pytest.raises(ValueError, match="at most"):
        mod.PairedHarborPanelRoute(
            panel_member="azure",
            provider_config=azure,
            expected_response_model="glm-served-model",
            expected_system_fingerprint="f" * 513,
        )

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


def test_pair_state_rejects_pre_evidence_schema_with_explicit_restart_path(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path, _candidate())
    block = runner._protocol.design.blocks[0]
    current = runner._pair_state(block, status="failed")
    assert current.state_version == "3"
    legacy = cast("JsonObject", current.model_dump(mode="json"))
    legacy["state_version"] = "2"
    legacy.pop("evidence_digest")
    legacy["state_digest"] = mod._canonical_digest(
        {key: value for key, value in legacy.items() if key != "state_digest"}
    )

    restart_error = "version 2 predates evidence binding.*new operation_id"
    with pytest.raises(ValueError, match=restart_error):
        mod.PairedHarborPairGenerationState.model_validate(legacy)

    state_path = tmp_path / "legacy-pair-state.json"
    state_path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(mod.PairedHarborPairStateError, match=restart_error):
        mod._read_pair_generation_state(state_path)


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
        evidence_digest="sha256:" + "c" * 64,
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


def test_restart_reuses_completed_pairs_and_reruns_only_authorized_failed_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    failed_calls, _ = _install_fake_evaluator(
        monkeypatch,
        candidate=candidate,
        fail_job=2 * len(_design().blocks),
    )
    first_generation = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    with pytest.raises(mod.PairedHarborMatrixError):
        asyncio.run(first_generation.run(baseline=baseline, candidate=candidate))
    assert len(failed_calls) == 2 * len(_design().blocks)
    failed_block = first_generation._protocol.design.blocks[-1]
    failed_state = mod._read_pair_generation_state(first_generation._pair_state_path(failed_block))
    assert failed_state.status == "failed"
    completed_blocks = first_generation._protocol.design.blocks[:-1]
    completed_states = tuple(
        mod._read_pair_generation_state(first_generation._pair_state_path(block))
        for block in completed_blocks
    )
    assert all(state.status == "complete" for state in completed_states)

    replay_calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    with pytest.raises(mod.PartialPairedHarborReuseError, match="is 'failed'"):
        asyncio.run(first_generation.run(baseline=baseline, candidate=candidate))
    assert replay_calls == []

    unauthorized_generation = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        operation_id="offline-test-operation",
        generation_id=2,
        max_concurrent_blocks=1,
    )
    with pytest.raises(mod.PartialPairedHarborReuseError, match="retry authorization"):
        asyncio.run(unauthorized_generation.run(baseline=baseline, candidate=candidate))
    assert replay_calls == []

    authorization = mod.authorize_paired_harbor_pair_retry(
        jobs_dir=first_generation._runtime.jobs_dir,
        protocol=first_generation._protocol,
        operation_id="offline-test-operation",
        failed_state=failed_state,
    )
    assert authorization.from_generation_id == 1
    assert authorization.to_generation_id == 2
    assert authorization.failed_state_digest == failed_state.state_digest
    assert authorization.failure_evidence.failed_state == failed_state
    assert authorization.failure_evidence.owner is mod.PairedHarborPairFailureOwner.INFRASTRUCTURE
    assert authorization.failure_evidence.failure_kind is BenchmarkFailureKind.PROVIDER
    assert (
        authorization.failure_evidence.retry_eligibility
        is mod.PairedHarborPairRetryEligibility.WHOLE_PAIR
    )
    assert authorization.failure_evidence_digest == authorization.failure_evidence.evidence_digest

    authorized_generation = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        operation_id="offline-test-operation",
        generation_id=2,
        max_concurrent_blocks=1,
    )
    report = asyncio.run(authorized_generation.run(baseline=baseline, candidate=candidate))

    assert len(replay_calls) == 2
    assert report.generation_id == 2
    assert report.retry_authorizations == (authorization,)
    assert tuple(item.generation_id for item in report.evidence) == (1, 1, 1, 2)
    assert tuple(item.pair_generation_id for item in report.evidence[:-1]) == tuple(
        state.pair_generation_id for state in completed_states
    )

    drifted_authorization = cast("JsonObject", authorization.model_dump(mode="json"))
    drifted_authorization["failed_state_digest"] = "sha256:" + "f" * 64
    drifted_authorization["authorization_digest"] = mod._canonical_digest(
        {
            key: value
            for key, value in drifted_authorization.items()
            if key != "authorization_digest"
        }
    )
    with pytest.raises(ValueError, match="differs from its failed state"):
        mod.PairedHarborPairRetryAuthorization.model_validate(drifted_authorization)


def test_concurrent_sibling_finishes_before_retryable_failure_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    first_calls, first_counters = _install_fake_evaluator(
        monkeypatch,
        candidate=candidate,
        fail_job=1,
    )
    first = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=2,
    )

    with pytest.raises(mod.PairedHarborMatrixError):
        asyncio.run(first.run(baseline=baseline, candidate=candidate))

    materialized = tuple(
        state
        for block in first._protocol.design.blocks
        if (
            state := first._inspect_pair_generation(
                block,
                generation_id=1,
                create=False,
                allow_incomplete=True,
            )
        )
        is not None
    )
    failed_states = tuple(state for state in materialized if state.status == "failed")
    complete_states = tuple(state for state in materialized if state.status == "complete")
    assert len(first_calls) == 3
    assert first_counters["cancelled"] == 0
    assert len(failed_states) == 1
    assert len(complete_states) == 1
    failed_state = failed_states[0]
    authorization = mod.authorize_paired_harbor_pair_retry(
        jobs_dir=first._runtime.jobs_dir,
        protocol=first._protocol,
        operation_id="offline-test-operation",
        failed_state=failed_state,
    )

    retry_calls, retry_counters = _install_fake_evaluator(monkeypatch, candidate=candidate)
    retry = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        generation_id=2,
        max_concurrent_blocks=2,
    )
    report = asyncio.run(retry.run(baseline=baseline, candidate=candidate))

    assert len(retry_calls) == 6
    assert retry_counters["cancelled"] == 0
    assert report.retry_authorizations == (authorization,)
    assert sum(item.generation_id == 1 for item in report.evidence) == 1
    assert sum(item.generation_id == 2 for item in report.evidence) == 3


def test_unclassified_evaluator_exception_is_durable_and_cannot_authorize_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )

    async def fail_without_typed_evidence(
        _self: mod.PairedHarborRunner,
        _block: PairedBlock,
        **_kwargs: object,
    ) -> mod.PairedHarborArmEvidence:
        raise RuntimeError("synthetic unclassified evaluator failure")

    monkeypatch.setattr(
        mod.PairedHarborRunner,
        "_evaluate_arm",
        fail_without_typed_evidence,
    )
    with pytest.raises(mod.PairedHarborMatrixError):
        asyncio.run(runner.run(baseline=baseline, candidate=candidate))

    block = runner._protocol.design.blocks[0]
    failed_state = mod._read_pair_generation_state(runner._pair_state_path(block))
    failure_evidence = mod.load_paired_harbor_pair_failure_evidence(
        jobs_dir=runner._runtime.jobs_dir,
        failed_state=failed_state,
    )
    assert failure_evidence.failed_state_digest == failed_state.state_digest
    assert failure_evidence.owner is mod.PairedHarborPairFailureOwner.UNCLASSIFIED
    assert failure_evidence.source is mod.PairedHarborPairFailureSource.EVALUATOR_EXCEPTION
    assert failure_evidence.retry_eligibility is mod.PairedHarborPairRetryEligibility.FORBIDDEN
    with pytest.raises(ValueError, match="not eligible for whole-pair retry"):
        mod.authorize_paired_harbor_pair_retry(
            jobs_dir=runner._runtime.jobs_dir,
            protocol=runner._protocol,
            operation_id="offline-test-operation",
            failed_state=failed_state,
        )


@pytest.mark.parametrize(
    "identity_field",
    ["task_checksum", "task_environment_digest", "runner_environment_digest"],
)
def test_foreign_infrastructure_result_cannot_become_retry_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_field: str,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()

    def replace_task_identity(loaded: LoadedHarborJobResult) -> LoadedHarborJobResult:
        trial = loaded.result.trials[0].model_copy(update={identity_field: "sha256:" + "0" * 64})
        return LoadedHarborJobResult(
            result=loaded.result.model_copy(update={"trials": [trial]}),
            job_dir=loaded.job_dir,
            locators=loaded.locators,
        )

    _install_fake_evaluator(
        monkeypatch,
        candidate=candidate,
        fail_job=1,
        result_transform=replace_task_identity,
    )
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    with pytest.raises(mod.PairedHarborMatrixError):
        asyncio.run(runner.run(baseline=baseline, candidate=candidate))

    block = runner._protocol.design.blocks[0]
    failed_state = mod._read_pair_generation_state(runner._pair_state_path(block))
    failure_evidence = mod.load_paired_harbor_pair_failure_evidence(
        jobs_dir=runner._runtime.jobs_dir,
        failed_state=failed_state,
    )
    assert failure_evidence.owner is mod.PairedHarborPairFailureOwner.SCORING
    assert failure_evidence.retry_eligibility is mod.PairedHarborPairRetryEligibility.FORBIDDEN
    with pytest.raises(ValueError, match="not eligible for whole-pair retry"):
        mod.authorize_paired_harbor_pair_retry(
            jobs_dir=runner._runtime.jobs_dir,
            protocol=runner._protocol,
            operation_id="offline-test-operation",
            failed_state=failed_state,
        )


def test_malformed_result_without_optional_environment_digests_can_authorize_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()

    def malformed_early_result(loaded: LoadedHarborJobResult) -> LoadedHarborJobResult:
        trial = loaded.result.trials[0].model_copy(
            update={
                "task_environment_digest": None,
                "runner_environment_digest": None,
                "error": BenchmarkError(
                    kind=BenchmarkFailureKind.MALFORMED_RESULT,
                    type="SyntheticMalformedResult",
                    message="result could not be decoded",
                ),
            }
        )
        return LoadedHarborJobResult(
            result=loaded.result.model_copy(update={"trials": [trial]}),
            job_dir=loaded.job_dir,
            locators=loaded.locators,
        )

    _install_fake_evaluator(
        monkeypatch,
        candidate=candidate,
        fail_job=1,
        result_transform=malformed_early_result,
    )
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    with pytest.raises(mod.PairedHarborMatrixError):
        asyncio.run(runner.run(baseline=baseline, candidate=candidate))

    block = runner._protocol.design.blocks[0]
    failed_state = mod._read_pair_generation_state(runner._pair_state_path(block))
    failure_evidence = mod.load_paired_harbor_pair_failure_evidence(
        jobs_dir=runner._runtime.jobs_dir,
        failed_state=failed_state,
    )
    assert failure_evidence.owner is mod.PairedHarborPairFailureOwner.INFRASTRUCTURE
    assert failure_evidence.failure_kind is BenchmarkFailureKind.MALFORMED_RESULT
    assert failure_evidence.retry_eligibility is mod.PairedHarborPairRetryEligibility.WHOLE_PAIR
    assert (
        mod.authorize_paired_harbor_pair_retry(
            jobs_dir=runner._runtime.jobs_dir,
            protocol=runner._protocol,
            operation_id="offline-test-operation",
            failed_state=failed_state,
        ).failure_evidence
        == failure_evidence
    )


def test_candidate_task_and_score_failures_remain_analysis_zero_without_retry() -> None:
    cell = BenchmarkCell(
        task_key=_TASK_KEYS["task-a"],
        task_name="task-a",
        attempt=1,
        config_digest=_CONFIG_DIGEST,
    )
    common = {
        "cell": cell.model_dump(mode="json"),
        "task_identity": "task-a",
        "task_checksum": _CONTENT_DIGESTS["task-a"],
        "source": "test-dataset",
        "task_instruction": "Solve task-a.",
        "task_environment_digest": _ENVIRONMENT_DIGESTS["task-a"],
        "runner_environment_digest": LocalPiRunnerSpec().attestation.digest,
        "usage": BenchmarkUsage(calls=1).model_dump(mode="json"),
    }
    low_score = BenchmarkTrialResult.model_validate(
        {
            **common,
            "status": BenchmarkTrialStatus.SCORED.value,
            "rewards": {"reward": 0.0},
            "candidate_outcome": {"status": BenchmarkCandidateStatus.COMPLETED.value},
            "run_health": BenchmarkRunHealth.VALID.value,
        }
    )
    candidate_failure = BenchmarkTrialResult.model_validate(
        {
            **common,
            "status": BenchmarkTrialStatus.CANDIDATE_FAILURE.value,
            "candidate_outcome": {
                "status": BenchmarkCandidateStatus.FAILED.value,
                "stage": BenchmarkCandidateStage.EXECUTION.value,
                "failure_reason": BenchmarkCandidateFailureReason.RUNTIME_ERROR.value,
            },
            "run_health": BenchmarkRunHealth.CANDIDATE_DAMAGED.value,
        }
    )
    task_timeout = BenchmarkTrialResult.model_validate(
        {
            **common,
            "status": BenchmarkTrialStatus.SCORED.value,
            "rewards": {"reward": 1.0},
            "error": {
                "kind": BenchmarkFailureKind.TASK_TIMEOUT.value,
                "type": "AgentTimeoutError",
                "message": "agent execution exceeded the task time limit",
            },
            "candidate_outcome": {"status": BenchmarkCandidateStatus.UNKNOWN.value},
            "run_health": BenchmarkRunHealth.VALID.value,
        }
    )

    for trial in (low_score, candidate_failure, task_timeout):
        assert (
            mod._classify_nonadmissible_benchmark_result(
                [trial],
                arm=PairedArm.CANDIDATE,
            )
            is None
        )
        assert mod.harbor_trial_analysis_values(trial, reward_key="reward")[1] == 0.0

    candidate_owned_provider_error = BenchmarkTrialResult.model_validate(
        {
            **common,
            "status": BenchmarkTrialStatus.INFRASTRUCTURE_ERROR.value,
            "error": {
                "kind": BenchmarkFailureKind.PROVIDER.value,
                "type": "SyntheticProviderError",
                "message": "synthetic provider failure",
            },
            "candidate_outcome": {
                "status": BenchmarkCandidateStatus.FAILED.value,
                "stage": BenchmarkCandidateStage.EXECUTION.value,
                "failure_reason": BenchmarkCandidateFailureReason.INVALID_REQUEST.value,
            },
            "run_health": BenchmarkRunHealth.CANDIDATE_DAMAGED.value,
        }
    )
    candidate_descriptor = mod._classify_nonadmissible_benchmark_result(
        [candidate_owned_provider_error],
        arm=PairedArm.CANDIDATE,
    )
    assert candidate_descriptor is not None
    assert candidate_descriptor.owner is mod.PairedHarborPairFailureOwner.CANDIDATE
    assert candidate_descriptor.retry_eligibility is mod.PairedHarborPairRetryEligibility.FORBIDDEN


def test_explicit_process_crash_classification_authorizes_fresh_whole_pair(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path, _candidate(), max_concurrent_blocks=1)
    block = runner._protocol.design.blocks[0]
    running_state = runner._begin_pair_generation(block, generation_id=1)

    with pytest.raises(ValueError, match="must be failed"):
        mod.authorize_paired_harbor_pair_retry(
            jobs_dir=runner._runtime.jobs_dir,
            protocol=runner._protocol,
            operation_id="offline-test-operation",
            failed_state=running_state,
        )

    failure_evidence = mod.classify_paired_harbor_process_crash(
        jobs_dir=runner._runtime.jobs_dir,
        protocol=runner._protocol,
        operation_id="offline-test-operation",
        interrupted_state=running_state,
    )
    assert failure_evidence.failed_state.status == "failed"
    assert failure_evidence.owner is mod.PairedHarborPairFailureOwner.PROCESS
    assert failure_evidence.source is mod.PairedHarborPairFailureSource.PROCESS_CRASH
    assert failure_evidence.retry_eligibility is mod.PairedHarborPairRetryEligibility.WHOLE_PAIR

    authorization = mod.authorize_paired_harbor_pair_retry(
        jobs_dir=runner._runtime.jobs_dir,
        protocol=runner._protocol,
        operation_id="offline-test-operation",
        failed_state=failure_evidence.failed_state,
    )
    assert authorization.failure_evidence == failure_evidence
    assert authorization.to_generation_id == 2


@pytest.mark.parametrize("crash_after_arm", [1, 2])
def test_restart_reconciles_terminal_arms_without_new_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_after_arm: int,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    calls, _ = _install_fake_evaluator(monkeypatch, candidate=candidate)
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    block = runner._protocol.design.blocks[0]
    real_publish = mod._create_or_compare_pair_arm_evidence
    published = 0

    class SyntheticProcessCrash(BaseException):
        pass

    def crash_after_publication(
        path: Path,
        evidence: mod.PairedHarborArmEvidence,
    ) -> None:
        nonlocal published
        real_publish(path, evidence)
        published += 1
        if published == crash_after_arm:
            raise SyntheticProcessCrash

    monkeypatch.setattr(
        mod,
        "_create_or_compare_pair_arm_evidence",
        crash_after_publication,
    )
    with pytest.raises(SyntheticProcessCrash):
        asyncio.run(
            runner._run_block(
                block,
                baseline=baseline,
                candidate=candidate,
                evaluator_session=mod.HarborEvaluatorSession(
                    runner_spec=runner._protocol.execution_plan.runner_spec
                ),
                generation_id=1,
            )
        )
    running_state = mod._read_pair_generation_state(runner._pair_state_path(block))
    assert running_state.status == "running"
    with pytest.raises(mod.PairedHarborPairStateError, match="resume the same generation"):
        mod.classify_paired_harbor_process_crash(
            jobs_dir=runner._runtime.jobs_dir,
            protocol=runner._protocol,
            operation_id="offline-test-operation",
            interrupted_state=running_state,
        )

    monkeypatch.setattr(mod, "_create_or_compare_pair_arm_evidence", real_publish)
    recovered = asyncio.run(
        runner._run_block(
            block,
            baseline=baseline,
            candidate=candidate,
            evaluator_session=mod.HarborEvaluatorSession(
                runner_spec=runner._protocol.execution_plan.runner_spec
            ),
            generation_id=1,
        )
    )

    assert recovered.generation_id == 1
    assert len(calls) == 2
    assert mod._read_pair_generation_state(runner._pair_state_path(block)).status == "complete"


def test_restart_reingests_terminal_result_before_arm_evidence_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    calls, counters = _install_fake_evaluator(monkeypatch, candidate=candidate)
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    block = runner._protocol.design.blocks[0]

    class SyntheticProcessCrash(BaseException):
        pass

    def crash_before_arm_evidence(**_kwargs: object) -> None:
        raise SyntheticProcessCrash

    monkeypatch.setattr(runner, "_validate_arm_result_identity", crash_before_arm_evidence)
    with pytest.raises(SyntheticProcessCrash):
        asyncio.run(
            runner._run_block(
                block,
                baseline=baseline,
                candidate=candidate,
                evaluator_session=mod.HarborEvaluatorSession(
                    runner_spec=runner._protocol.execution_plan.runner_spec
                ),
                generation_id=1,
            )
        )

    running_state = mod._read_pair_generation_state(runner._pair_state_path(block))
    first_arm = block.first_arm
    assert not os.path.lexists(
        mod._pair_arm_evidence_path(
            runner._runtime.jobs_dir,
            state=running_state,
            arm=first_arm,
        )
    )
    with pytest.raises(mod.PairedHarborPairStateError, match="unreconciled terminal arm"):
        mod.classify_paired_harbor_process_crash(
            jobs_dir=runner._runtime.jobs_dir,
            protocol=runner._protocol,
            operation_id="offline-test-operation",
            interrupted_state=running_state,
        )

    reconstructed = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    recovered = asyncio.run(
        reconstructed._run_block(
            block,
            baseline=baseline,
            candidate=candidate,
            evaluator_session=mod.HarborEvaluatorSession(
                runner_spec=reconstructed._protocol.execution_plan.runner_spec
            ),
            generation_id=1,
        )
    )

    assert recovered.generation_id == 1
    assert len(calls) == 2
    assert counters["load-existing"] == 1
    assert mod._read_pair_generation_state(reconstructed._pair_state_path(block)).status == (
        "complete"
    )


def test_unfinished_root_with_published_outcome_forces_same_generation_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    calls, counters = _install_fake_evaluator(monkeypatch, candidate=candidate)
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    block = runner._protocol.design.blocks[0]

    class SyntheticProcessCrash(BaseException):
        pass

    def crash_before_arm_evidence(**_kwargs: object) -> None:
        raise SyntheticProcessCrash

    monkeypatch.setattr(runner, "_validate_arm_result_identity", crash_before_arm_evidence)
    with pytest.raises(SyntheticProcessCrash):
        asyncio.run(
            runner._run_block(
                block,
                baseline=baseline,
                candidate=candidate,
                evaluator_session=mod.HarborEvaluatorSession(
                    runner_spec=runner._protocol.execution_plan.runner_spec
                ),
                generation_id=1,
            )
        )
    running_state = mod._read_pair_generation_state(runner._pair_state_path(block))
    first_job_name = {
        PairedArm.BASELINE: running_state.baseline_job_name,
        PairedArm.CANDIDATE: running_state.candidate_job_name,
    }[block.first_arm]
    root_result_path = runner._runtime.jobs_dir / first_job_name / "result.json"
    root_result = json.loads(root_result_path.read_text(encoding="utf-8"))
    root_result["finished_at"] = None
    root_result["stats"].update(
        {
            "n_completed_trials": 0,
            "n_running_trials": 1,
            "n_pending_trials": 0,
        }
    )
    root_result_path.write_text(json.dumps(root_result) + "\n", encoding="utf-8")

    assert (
        mod._arm_job_recovery_state(root_result_path.parent)
        is mod._HarborArmJobRecoveryState.OUTCOME_PUBLISHED
    )
    assert mod._pair_generation_can_resume_same_generation(
        runner._runtime.jobs_dir,
        running_state,
    )
    with pytest.raises(mod.PairedHarborPairStateError, match="published arm outcome"):
        mod.classify_paired_harbor_process_crash(
            jobs_dir=runner._runtime.jobs_dir,
            protocol=runner._protocol,
            operation_id="offline-test-operation",
            interrupted_state=running_state,
        )

    reconstructed = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    recovered = asyncio.run(
        reconstructed._run_block(
            block,
            baseline=baseline,
            candidate=candidate,
            evaluator_session=mod.HarborEvaluatorSession(
                runner_spec=reconstructed._protocol.execution_plan.runner_spec
            ),
            generation_id=1,
        )
    )
    assert recovered.generation_id == 1
    assert len(calls) == 2
    assert counters["resume-existing"] == 1


def test_unfinished_root_with_explicit_cancelled_child_can_authorize_process_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    _install_fake_evaluator(monkeypatch, candidate=candidate)
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    block = runner._protocol.design.blocks[0]

    class SyntheticProcessCrash(BaseException):
        pass

    def crash_before_arm_evidence(**_kwargs: object) -> None:
        raise SyntheticProcessCrash

    monkeypatch.setattr(runner, "_validate_arm_result_identity", crash_before_arm_evidence)
    with pytest.raises(SyntheticProcessCrash):
        asyncio.run(
            runner._run_block(
                block,
                baseline=baseline,
                candidate=candidate,
                evaluator_session=mod.HarborEvaluatorSession(
                    runner_spec=runner._protocol.execution_plan.runner_spec
                ),
                generation_id=1,
            )
        )
    running_state = mod._read_pair_generation_state(runner._pair_state_path(block))
    first_job_name = {
        PairedArm.BASELINE: running_state.baseline_job_name,
        PairedArm.CANDIDATE: running_state.candidate_job_name,
    }[block.first_arm]
    job_path = runner._runtime.jobs_dir / first_job_name
    root_result_path = job_path / "result.json"
    root_result = json.loads(root_result_path.read_text(encoding="utf-8"))
    root_result["finished_at"] = None
    root_result["stats"].update(
        {
            "n_completed_trials": 0,
            "n_running_trials": 0,
            "n_pending_trials": 1,
        }
    )
    root_result_path.write_text(json.dumps(root_result) + "\n", encoding="utf-8")
    (job_path / "trial" / "result.json").write_text(
        json.dumps(
            {
                "exception_info": {"exception_type": "CancelledError"},
                "verifier_result": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert mod._arm_job_recovery_state(job_path) is mod._HarborArmJobRecoveryState.CANCELLED
    failure_evidence = mod.classify_paired_harbor_process_crash(
        jobs_dir=runner._runtime.jobs_dir,
        protocol=runner._protocol,
        operation_id="offline-test-operation",
        interrupted_state=running_state,
    )
    assert failure_evidence.owner is mod.PairedHarborPairFailureOwner.PROCESS
    assert (
        mod.authorize_paired_harbor_pair_retry(
            jobs_dir=runner._runtime.jobs_dir,
            protocol=runner._protocol,
            operation_id="offline-test-operation",
            failed_state=failure_evidence.failed_state,
        ).failure_evidence
        == failure_evidence
    )


def test_cancelled_child_with_verifier_result_is_a_published_outcome(
    tmp_path: Path,
) -> None:
    job_path = tmp_path / "jobs" / "cancelled-with-score"
    trial_path = job_path / "trial"
    trial_path.mkdir(parents=True)
    (job_path / "result.json").write_text(
        json.dumps(
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "started_at": "2026-07-19T00:00:00Z",
                "finished_at": None,
                "n_total_trials": 1,
                "stats": {
                    "n_completed_trials": 0,
                    "n_running_trials": 0,
                    "n_pending_trials": 1,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (trial_path / "result.json").write_text(
        json.dumps(
            {
                "exception_info": {"exception_type": "CancelledError"},
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert mod._arm_job_recovery_state(job_path) is mod._HarborArmJobRecoveryState.OUTCOME_PUBLISHED


def test_restart_classifies_terminal_infrastructure_result_before_arm_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    calls, counters = _install_fake_evaluator(
        monkeypatch,
        candidate=candidate,
        fail_job=1,
    )
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    block = runner._protocol.design.blocks[0]

    class SyntheticProcessCrash(BaseException):
        pass

    def crash_before_failure_classification(**_kwargs: object) -> None:
        raise SyntheticProcessCrash

    monkeypatch.setattr(
        runner,
        "_validate_arm_result_identity",
        crash_before_failure_classification,
    )
    with pytest.raises(SyntheticProcessCrash):
        asyncio.run(
            runner._run_block(
                block,
                baseline=baseline,
                candidate=candidate,
                evaluator_session=mod.HarborEvaluatorSession(
                    runner_spec=runner._protocol.execution_plan.runner_spec
                ),
                generation_id=1,
            )
        )
    running_state = mod._read_pair_generation_state(runner._pair_state_path(block))
    with pytest.raises(mod.PairedHarborPairStateError, match="unreconciled terminal arm"):
        mod.classify_paired_harbor_process_crash(
            jobs_dir=runner._runtime.jobs_dir,
            protocol=runner._protocol,
            operation_id="offline-test-operation",
            interrupted_state=running_state,
        )

    reconstructed = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    with pytest.raises(mod._ClassifiedPairFailure):
        asyncio.run(
            reconstructed._run_block(
                block,
                baseline=baseline,
                candidate=candidate,
                evaluator_session=mod.HarborEvaluatorSession(
                    runner_spec=reconstructed._protocol.execution_plan.runner_spec
                ),
                generation_id=1,
            )
        )
    failed_state = mod._read_pair_generation_state(reconstructed._pair_state_path(block))
    failure_evidence = mod.load_paired_harbor_pair_failure_evidence(
        jobs_dir=reconstructed._runtime.jobs_dir,
        failed_state=failed_state,
    )

    assert len(calls) == 1
    assert counters["load-existing"] == 1
    assert failure_evidence.owner is mod.PairedHarborPairFailureOwner.INFRASTRUCTURE
    assert failure_evidence.failure_kind is BenchmarkFailureKind.PROVIDER
    assert failure_evidence.retry_eligibility is mod.PairedHarborPairRetryEligibility.WHOLE_PAIR
    assert (
        mod.authorize_paired_harbor_pair_retry(
            jobs_dir=reconstructed._runtime.jobs_dir,
            protocol=reconstructed._protocol,
            operation_id="offline-test-operation",
            failed_state=failed_state,
        ).failure_evidence
        == failure_evidence
    )


def test_restart_completes_evidence_first_infrastructure_failure_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    _install_fake_evaluator(monkeypatch, candidate=candidate, fail_job=1)
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    block = runner._protocol.design.blocks[0]
    real_replace = mod._replace_pair_generation_state

    def crash_before_failed_state(
        path: Path,
        state: mod.PairedHarborPairGenerationState,
    ) -> None:
        if state.status == "failed":
            raise OSError("synthetic process crash before failed-state publication")
        real_replace(path, state)

    monkeypatch.setattr(mod, "_replace_pair_generation_state", crash_before_failed_state)
    with pytest.raises(OSError, match="synthetic process crash"):
        asyncio.run(
            runner._run_block(
                block,
                baseline=baseline,
                candidate=candidate,
                evaluator_session=mod.HarborEvaluatorSession(
                    runner_spec=runner._protocol.execution_plan.runner_spec
                ),
                generation_id=1,
            )
        )
    interrupted = mod._read_pair_generation_state(runner._pair_state_path(block))
    assert interrupted.status == "running"
    expected_failed = mod._failed_pair_generation_state(interrupted)
    assert mod._pair_failure_evidence_path(
        runner._runtime.jobs_dir,
        failed_state=expected_failed,
    ).is_file()

    monkeypatch.setattr(mod, "_replace_pair_generation_state", real_replace)
    reconstructed = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    recovered = reconstructed._inspect_pair_generation(
        block,
        generation_id=1,
        create=False,
        allow_incomplete=True,
    )
    assert recovered == expected_failed
    failure_evidence = mod.load_paired_harbor_pair_failure_evidence(
        jobs_dir=reconstructed._runtime.jobs_dir,
        failed_state=expected_failed,
    )
    assert failure_evidence.owner is mod.PairedHarborPairFailureOwner.INFRASTRUCTURE
    assert (
        mod.authorize_paired_harbor_pair_retry(
            jobs_dir=reconstructed._runtime.jobs_dir,
            protocol=reconstructed._protocol,
            operation_id="offline-test-operation",
            failed_state=expected_failed,
        ).failure_evidence
        == failure_evidence
    )


def test_restart_completes_evidence_first_process_crash_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(tmp_path, _candidate(), max_concurrent_blocks=1)
    block = runner._protocol.design.blocks[0]
    running_state = runner._begin_pair_generation(block, generation_id=1)
    real_replace = mod._replace_pair_generation_state

    def crash_before_failed_state(
        path: Path,
        state: mod.PairedHarborPairGenerationState,
    ) -> None:
        if state.status == "failed":
            raise OSError("synthetic process crash before failed-state publication")
        real_replace(path, state)

    monkeypatch.setattr(mod, "_replace_pair_generation_state", crash_before_failed_state)
    with pytest.raises(OSError, match="synthetic process crash"):
        mod.classify_paired_harbor_process_crash(
            jobs_dir=runner._runtime.jobs_dir,
            protocol=runner._protocol,
            operation_id="offline-test-operation",
            interrupted_state=running_state,
        )
    assert mod._read_pair_generation_state(runner._pair_state_path(block)) == running_state

    monkeypatch.setattr(mod, "_replace_pair_generation_state", real_replace)
    failure_evidence = mod.classify_paired_harbor_process_crash(
        jobs_dir=runner._runtime.jobs_dir,
        protocol=runner._protocol,
        operation_id="offline-test-operation",
        interrupted_state=running_state,
    )
    assert failure_evidence.owner is mod.PairedHarborPairFailureOwner.PROCESS
    assert mod._read_pair_generation_state(runner._pair_state_path(block)) == (
        failure_evidence.failed_state
    )
    assert (
        mod.authorize_paired_harbor_pair_retry(
            jobs_dir=runner._runtime.jobs_dir,
            protocol=runner._protocol,
            operation_id="offline-test-operation",
            failed_state=failure_evidence.failed_state,
        ).failure_evidence
        == failure_evidence
    )


def test_interrupted_failure_evidence_publication_leaves_no_truncated_final_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    _install_fake_evaluator(monkeypatch, candidate=candidate, fail_job=1)
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    block = runner._protocol.design.blocks[0]
    running_state = runner._begin_pair_generation(block, generation_id=1)
    failed_state = mod._failed_pair_generation_state(running_state)
    failure_path = mod._pair_failure_evidence_path(
        runner._runtime.jobs_dir,
        failed_state=failed_state,
    )
    real_link = os.link

    class SyntheticProcessCrash(BaseException):
        pass

    def crash_before_final_link(
        source: Path,
        destination: Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        if Path(destination) == failure_path:
            raise SyntheticProcessCrash
        real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "link", crash_before_final_link)
    with pytest.raises(SyntheticProcessCrash):
        asyncio.run(
            runner._run_block(
                block,
                baseline=baseline,
                candidate=candidate,
                evaluator_session=mod.HarborEvaluatorSession(
                    runner_spec=runner._protocol.execution_plan.runner_spec
                ),
                generation_id=1,
            )
        )
    assert not os.path.lexists(failure_path)
    assert mod._read_pair_generation_state(runner._pair_state_path(block)) == running_state

    monkeypatch.setattr(os, "link", real_link)
    with pytest.raises(mod._ClassifiedPairFailure):
        asyncio.run(
            runner._run_block(
                block,
                baseline=baseline,
                candidate=candidate,
                evaluator_session=mod.HarborEvaluatorSession(
                    runner_spec=runner._protocol.execution_plan.runner_spec
                ),
                generation_id=1,
            )
        )
    assert failure_path.is_file()
    assert mod._read_pair_generation_state(runner._pair_state_path(block)).status == "failed"


def test_interrupted_completion_witness_publication_recovers_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    calls, counters = _install_fake_evaluator(monkeypatch, candidate=candidate)
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    block = runner._protocol.design.blocks[0]
    running_state = runner._begin_pair_generation(block, generation_id=1)
    witness_path = mod._pair_arm_completion_witness_path(
        runner._runtime.jobs_dir,
        state=running_state,
        arm=block.first_arm,
    )
    real_link = os.link

    class SyntheticProcessCrash(BaseException):
        pass

    def crash_before_final_link(
        source: Path,
        destination: Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        if Path(destination) == witness_path:
            raise SyntheticProcessCrash
        real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "link", crash_before_final_link)
    with pytest.raises(SyntheticProcessCrash):
        asyncio.run(
            runner._run_block(
                block,
                baseline=baseline,
                candidate=candidate,
                evaluator_session=mod.HarborEvaluatorSession(
                    runner_spec=runner._protocol.execution_plan.runner_spec
                ),
                generation_id=1,
            )
        )
    assert not os.path.lexists(witness_path)
    assert mod._read_pair_generation_state(runner._pair_state_path(block)) == running_state

    monkeypatch.setattr(os, "link", real_link)
    reconstructed = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    recovered = asyncio.run(
        reconstructed._run_block(
            block,
            baseline=baseline,
            candidate=candidate,
            evaluator_session=mod.HarborEvaluatorSession(
                runner_spec=reconstructed._protocol.execution_plan.runner_spec
            ),
            generation_id=1,
        )
    )
    assert recovered.generation_id == 1
    assert len(calls) == 2
    assert counters["load-existing"] == 1
    assert witness_path.is_file()


def test_process_crash_reconciliation_uses_only_score_blind_completion_witness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    _install_fake_evaluator(monkeypatch, candidate=candidate)
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    block = runner._protocol.design.blocks[0]
    running_state = runner._begin_pair_generation(block, generation_id=1)
    harnesses = {
        PairedArm.BASELINE: baseline,
        PairedArm.CANDIDATE: candidate,
    }
    asyncio.run(
        runner._evaluate_arm(
            block,
            arm=block.first_arm,
            harness=harnesses[block.first_arm],
            evaluator_session=mod.HarborEvaluatorSession(
                runner_spec=runner._protocol.execution_plan.runner_spec
            ),
            generation_id=1,
        )
    )
    second_arm = (
        PairedArm.CANDIDATE if block.first_arm is PairedArm.BASELINE else PairedArm.BASELINE
    )
    second_job_name = {
        PairedArm.BASELINE: running_state.baseline_job_name,
        PairedArm.CANDIDATE: running_state.candidate_job_name,
    }[second_arm]
    (runner._runtime.jobs_dir / second_job_name).mkdir()

    def reject_score_access(*_args: object, **_kwargs: object) -> tuple[float | None, float]:
        raise AssertionError("process-crash classification must not read arm scores")

    monkeypatch.setattr(mod, "harbor_trial_analysis_values", reject_score_access)
    failure_evidence = mod.classify_paired_harbor_process_crash(
        jobs_dir=runner._runtime.jobs_dir,
        protocol=runner._protocol,
        operation_id="offline-test-operation",
        interrupted_state=running_state,
    )
    authorization = mod.authorize_paired_harbor_pair_retry(
        jobs_dir=runner._runtime.jobs_dir,
        protocol=runner._protocol,
        operation_id="offline-test-operation",
        failed_state=failure_evidence.failed_state,
    )

    assert failure_evidence.owner is mod.PairedHarborPairFailureOwner.PROCESS
    assert authorization.failure_evidence == failure_evidence


def test_swapped_arm_cache_record_is_rejected_by_expected_path_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    _install_fake_evaluator(monkeypatch, candidate=candidate)
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    block = runner._protocol.design.blocks[0]
    running_state = runner._begin_pair_generation(block, generation_id=1)
    harnesses = {
        PairedArm.BASELINE: baseline,
        PairedArm.CANDIDATE: candidate,
    }
    for arm in (PairedArm.BASELINE, PairedArm.CANDIDATE):
        asyncio.run(
            runner._evaluate_arm(
                block,
                arm=arm,
                harness=harnesses[arm],
                evaluator_session=mod.HarborEvaluatorSession(
                    runner_spec=runner._protocol.execution_plan.runner_spec
                ),
                generation_id=1,
            )
        )
    baseline_path = mod._pair_arm_evidence_path(
        runner._runtime.jobs_dir,
        state=running_state,
        arm=PairedArm.BASELINE,
    )
    candidate_path = mod._pair_arm_evidence_path(
        runner._runtime.jobs_dir,
        state=running_state,
        arm=PairedArm.CANDIDATE,
    )
    baseline_path.write_bytes(candidate_path.read_bytes())

    with pytest.raises(mod.PairedHarborPairStateError, match="pair generation"):
        asyncio.run(
            runner._evaluate_arm(
                block,
                arm=PairedArm.BASELINE,
                harness=baseline,
                evaluator_session=mod.HarborEvaluatorSession(
                    runner_spec=runner._protocol.execution_plan.runner_spec
                ),
                generation_id=1,
            )
        )


def test_swapped_arm_completion_witness_is_rejected_by_expected_path_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    _install_fake_evaluator(monkeypatch, candidate=candidate)
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    block = runner._protocol.design.blocks[0]
    running_state = runner._begin_pair_generation(block, generation_id=1)
    harnesses = {
        PairedArm.BASELINE: baseline,
        PairedArm.CANDIDATE: candidate,
    }
    for arm in (PairedArm.BASELINE, PairedArm.CANDIDATE):
        asyncio.run(
            runner._evaluate_arm(
                block,
                arm=arm,
                harness=harnesses[arm],
                evaluator_session=mod.HarborEvaluatorSession(
                    runner_spec=runner._protocol.execution_plan.runner_spec
                ),
                generation_id=1,
            )
        )
    baseline_path = mod._pair_arm_completion_witness_path(
        runner._runtime.jobs_dir,
        state=running_state,
        arm=PairedArm.BASELINE,
    )
    candidate_path = mod._pair_arm_completion_witness_path(
        runner._runtime.jobs_dir,
        state=running_state,
        arm=PairedArm.CANDIDATE,
    )
    baseline_path.write_bytes(candidate_path.read_bytes())

    with pytest.raises(mod.PairedHarborPairStateError, match="pair generation"):
        mod.classify_paired_harbor_process_crash(
            jobs_dir=runner._runtime.jobs_dir,
            protocol=runner._protocol,
            operation_id="offline-test-operation",
            interrupted_state=running_state,
        )


def test_arm_cache_is_reingested_before_reuse_and_rejects_score_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    _install_fake_evaluator(monkeypatch, candidate=candidate)
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    block = runner._protocol.design.blocks[0]
    running_state = runner._begin_pair_generation(block, generation_id=1)
    arm = block.first_arm
    harness = baseline if arm is PairedArm.BASELINE else candidate
    asyncio.run(
        runner._evaluate_arm(
            block,
            arm=arm,
            harness=harness,
            evaluator_session=mod.HarborEvaluatorSession(
                runner_spec=runner._protocol.execution_plan.runner_spec
            ),
            generation_id=1,
        )
    )
    evidence_path = mod._pair_arm_evidence_path(
        runner._runtime.jobs_dir,
        state=running_state,
        arm=arm,
    )
    tampered = json.loads(evidence_path.read_text(encoding="utf-8"))
    score = 1.0 - tampered["analysis_score"]
    tampered["trial"]["rewards"]["reward"] = score
    tampered["verifier_reward"] = score
    tampered["analysis_score"] = score
    _refresh_arm_admission_digest(tampered)
    evidence_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(mod.PairedHarborPairStateError, match="different contents"):
        asyncio.run(
            runner._evaluate_arm(
                block,
                arm=arm,
                harness=harness,
                evaluator_session=mod.HarborEvaluatorSession(
                    runner_spec=runner._protocol.execution_plan.runner_spec
                ),
                generation_id=1,
            )
        )


def test_arm_completion_witness_rejects_raw_terminal_artifact_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    _install_fake_evaluator(monkeypatch, candidate=candidate)
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    block = runner._protocol.design.blocks[0]
    arm = block.first_arm
    harness = baseline if arm is PairedArm.BASELINE else candidate
    admitted = asyncio.run(
        runner._evaluate_arm(
            block,
            arm=arm,
            harness=harness,
            evaluator_session=mod.HarborEvaluatorSession(
                runner_spec=runner._protocol.execution_plan.runner_spec
            ),
            generation_id=1,
        )
    )
    raw_result = runner._runtime.jobs_dir / admitted.job_name / "trial" / "result.json"
    raw_result.write_text('{"tampered":true}\n', encoding="utf-8")

    with pytest.raises(mod.PairedHarborPairStateError, match="different contents"):
        asyncio.run(
            runner._evaluate_arm(
                block,
                arm=arm,
                harness=harness,
                evaluator_session=mod.HarborEvaluatorSession(
                    runner_spec=runner._protocol.execution_plan.runner_spec
                ),
                generation_id=1,
            )
        )


def test_failure_classification_tamper_blocks_restart_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = pi_node_baseline("baseline")
    candidate = _candidate()
    _install_fake_evaluator(monkeypatch, candidate=candidate, fail_job=1)
    runner = _runner(
        tmp_path,
        candidate,
        baseline=baseline,
        max_concurrent_blocks=1,
    )
    with pytest.raises(mod.PairedHarborMatrixError):
        asyncio.run(runner.run(baseline=baseline, candidate=candidate))

    block = runner._protocol.design.blocks[0]
    failed_state = mod._read_pair_generation_state(runner._pair_state_path(block))
    evidence_path = mod._pair_failure_evidence_path(
        runner._runtime.jobs_dir,
        failed_state=failed_state,
    )
    tampered = json.loads(evidence_path.read_text(encoding="utf-8"))
    tampered["owner"] = mod.PairedHarborPairFailureOwner.TASK.value
    evidence_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(mod.PairedHarborPairStateError, match="unreadable or invalid"):
        mod.authorize_paired_harbor_pair_retry(
            jobs_dir=runner._runtime.jobs_dir,
            protocol=runner._protocol,
            operation_id="offline-test-operation",
            failed_state=failed_state,
        )


def test_operation_lease_excludes_concurrent_cross_generation_restart(tmp_path: Path) -> None:
    coordinator = mod._LocalPairedHarborLeaseCoordinator(tmp_path / "jobs")

    async def contend() -> None:
        async with coordinator.operation_lease(
            protocol_digest="sha256:" + "a" * 64,
            operation_id="operation-a",
        ):
            with pytest.raises(mod.ConcurrentPairedHarborRunError):
                async with coordinator.operation_lease(
                    protocol_digest="sha256:" + "a" * 64,
                    operation_id="operation-a",
                ):
                    pytest.fail("cross-generation operation lease unexpectedly succeeded")

    asyncio.run(contend())


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
