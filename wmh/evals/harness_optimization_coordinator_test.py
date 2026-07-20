"""Tests for crash-durable harness optimization study composition."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from harbor.models.job.config import DatasetConfig
from llm_waterfall import ChatRequest, ChatResponse
from typer.testing import CliRunner

from wmh.agents import AgentProject, default_agent, meta_agent
from wmh.agents import project as project_module
from wmh.cli.app import app
from wmh.evals import harness_optimization_coordinator as coordinator_module
from wmh.evals.harbor.config import HarborEnvironmentBackend, HarborJobSpec
from wmh.evals.harbor.paired_runner import (
    HarborExecutionPlan,
    HarborExecutionRuntime,
    PairedHarborBudgetRuntime,
    PairedHarborCompletedBlock,
    PairedHarborNoActiveSliceIntentError,
    PairedHarborProtocol,
    PairedHarborSliceIntent,
    PairedHarborSlicePolicy,
    PairedHarborSliceProgress,
    PairedHarborSliceResult,
    _select_paired_harbor_slice_blocks,
    paired_harbor_pair_generation_id,
)
from wmh.evals.harness_optimization import (
    DiscoverySearchPlan,
    HarnessOptimizationProtocol,
    OpenedHarnessOptimizationConfirmation,
    freeze_harness_optimization_candidate,
    freeze_harness_optimization_harbor_protocol,
    open_harness_optimization_confirmation,
    prepare_harness_optimization_study,
    run_harness_optimization_search,
    run_harness_optimization_search_slice,
)
from wmh.evals.harness_optimization_coordinator import (
    DeterministicHarnessOptimizationRehearsalFactory,
    HarnessOptimizationAdvanceKind,
    HarnessOptimizationDiscoveryRuntime,
    HarnessOptimizationStudyCoordinator,
    HarnessOptimizationStudySpec,
    LocalHarnessOptimizationStateStore,
    LocalStudyEvidenceStore,
    PairedHarborSliceRunner,
    ProductionHarnessOptimizationRuntimeFactory,
    ProjectDiscoveryProposerRuntime,
    confirmation_initial_run_state_digest,
    reconcile_harness_optimization_confirmation_slice,
    run_harness_optimization_confirmation_slice,
)
from wmh.evals.harness_optimization_test import (
    _artifact_publication,
    _budget_authority,
    _canonical_digest,
    _CodeProposer,
    _digest,
    _discovery_lifecycle,
    _partition,
    _prepare,
    _Scorer,
)
from wmh.evals.study_journal import StudyJournalStore, StudyPhase
from wmh.evals.study_lifecycle import (
    ConfirmationFrozenPayload,
    ConfirmationOpenedPayload,
    ConfirmationRunningPayload,
    StudyArtifactPublication,
    StudyBudgetReport,
    StudyLifecycleController,
)
from wmh.harness.cost import (
    ProviderCostBinding,
    SearchComponentCostBinding,
    SearchComponentCostRuntime,
    SearchComponentRole,
    SearchCostBinding,
    TimedResourceCostBinding,
)
from wmh.harness.create import SearchCheckpoint, SearchProposalBatchWitness
from wmh.harness.doc import HarnessDoc
from wmh.harness.proposer import ProjectDeltaProposer
from wmh.providers.base import (
    Completion,
    Message,
    ProviderConfig,
    ToolCallingProvider,
    VerifyResult,
)
from wmh.tracking.budget import (
    BudgetPolicy,
    TimedResourceCostMeter,
    bind_budget_account,
    bind_timed_resource_account,
    bootstrap_budget_ledger,
)
from wmh.tracking.rate_limit import (
    E2B_SANDBOX_CREATE_RATE_POLICY,
    ExternalDispatchRateAuthority,
    bind_external_dispatch_rate_authority,
)

runner = CliRunner()


class _NoDispatchToolProvider:
    """Tool provider identity used to prove project construction stays pre-dispatch."""

    paid_request_attempts = 1

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.calls = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        del system, messages, temperature, max_tokens
        self.calls += 1
        raise AssertionError("project runtime construction cannot call the provider")

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        del request
        self.calls += 1
        raise AssertionError("project runtime construction cannot call the provider")

    def embed(self, texts: list[str]) -> list[list[float]]:
        del texts
        self.calls += 1
        raise AssertionError("project runtime construction cannot call the provider")

    def verify(self) -> VerifyResult:
        self.calls += 1
        raise AssertionError("project runtime construction cannot verify the provider")


class _NoDispatchHarborScorer:
    """Cost-bound scorer double that never opens Harbor or provider resources."""

    def __init__(self, **kwargs: object) -> None:
        runtime = kwargs.get("cost_runtime")
        if not isinstance(runtime, SearchComponentCostRuntime):
            raise AssertionError("production factory omitted the scorer cost runtime")
        self.configuration_id = runtime.binding.configuration_id
        self.search_cost_binding = runtime.binding
        self.create_rate_binding = None
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _UnusedRuntimeFactory:
    """Fail if an early phase accidentally constructs a paid runtime."""

    def build_discovery(
        self,
        spec: HarnessOptimizationStudySpec,
    ) -> HarnessOptimizationDiscoveryRuntime:
        del spec
        raise AssertionError("discovery runtime must not be constructed")

    def build_confirmation(
        self,
        spec: HarnessOptimizationStudySpec,
        protocol: PairedHarborProtocol,
    ) -> PairedHarborSliceRunner:
        del spec, protocol
        raise AssertionError("confirmation runtime must not be constructed")


def _spec(tmp_path: Path) -> HarnessOptimizationStudySpec:
    control_store, manifest = _partition(tmp_path)
    prepared, _protocol = _prepare(tmp_path, manifest, default_agent("baseline"))
    dataset = (tmp_path / "dataset").resolve()
    dataset.mkdir()
    jobs = (tmp_path / "jobs").resolve()
    confirmation_jobs = (tmp_path / "confirmation-jobs").resolve()
    discovery_ids = [task.task_id for task in prepared.protocol.discovery.tasks]
    return HarnessOptimizationStudySpec(
        prepared=prepared,
        partition_control_dir=control_store.directory,
        discovery_job_spec=HarborJobSpec(
            job_name="discovery",
            jobs_dir=jobs,
            datasets=[DatasetConfig(path=dataset, task_names=discovery_ids)],
            n_attempts=prepared.protocol.search.attempts_per_task,
            n_concurrent_trials=1,
            agent_n_concurrent=1,
            environment_backend=HarborEnvironmentBackend.LOCAL,
        ),
        confirmation_runtime=HarborExecutionRuntime(
            jobs_dir=confirmation_jobs,
            dataset_paths_by_id={"synthetic": dataset},
            budget=prepared.confirmation_budget,
        ),
        qualification_report_digest=_digest("qualified-roster-report"),
        confirmation_operation_id="confirmation-run",
    )


def _project_spec(
    tmp_path: Path,
) -> tuple[HarnessOptimizationStudySpec, _NoDispatchToolProvider]:
    """Re-freeze the synthetic study around one real deferred AgentProject contract."""
    base = _spec(tmp_path)
    prepared = base.prepared
    rate_path = (tmp_path / "project-create-rate.json").resolve()
    rate_authority = ExternalDispatchRateAuthority.bootstrap(
        rate_path,
        E2B_SANDBOX_CREATE_RATE_POLICY,
    )
    execution = AgentProject.execution_commitment_for(
        timeout=60,
        template="wmh-project-template:immutable-build",
        cpu_count=2,
        memory_mb=2048,
        create_rate_authority=rate_authority,
    )
    proposer_agent = meta_agent("optimizer")
    old_binding = prepared.search_cost_binding
    proposer_provider = _NoDispatchToolProvider(old_binding.proposer.providers[0].provider_config)
    proposer_configuration_id = ProjectDeltaProposer.configuration_id_for(
        project_type=AgentProject,
        project_workspace="/home/user/project",
        project_execution_configuration_id=execution.digest,
        agent=proposer_agent,
        provider=proposer_provider,
        project_create_rate_binding=execution.create_rate_binding,
        response_identity=old_binding.proposer.providers[0].response_identity,
    )
    project_meter_id = "proposer-project"
    project_meter = TimedResourceCostMeter(
        resource_type=execution.resource_class.role.value,
        resource_class_digest=execution.resource_class.digest,
        nano_usd_per_second=1,
        max_billing_seconds=execution.resource_class.max_host_observation_seconds,
    )
    policy = BudgetPolicy.model_validate(
        {
            **old_binding.policy.model_dump(mode="python"),
            "meters": {**old_binding.policy.meters, project_meter_id: project_meter},
        }
    )
    authority = bootstrap_budget_ledger((tmp_path / "project-budget.sqlite3").resolve(), policy)

    def provider_binding(
        original: ProviderCostBinding,
        *,
        configuration_id: str,
    ) -> ProviderCostBinding:
        account = authority.provider_account(
            scope=original.account.scope,
            meter_id=original.account.meter_id,
        )
        return ProviderCostBinding(
            component_configuration_id=configuration_id,
            provider_config=original.provider_config,
            response_identity=original.response_identity,
            account=bind_budget_account(account),
        )

    proposer_scope = old_binding.proposer.providers[0].account.scope
    project_account = authority.timed_resource_account(
        scope=proposer_scope,
        meter_id=project_meter_id,
    )
    search_cost_binding = SearchCostBinding(
        declared_hard_limit_nano_usd=policy.hard_limit_nano_usd,
        policy=policy,
        ledger_identity=authority.ledger_identity,
        phase=old_binding.phase,
        run_id=old_binding.run_id,
        external_dispatch_rate_binding=bind_external_dispatch_rate_authority(rate_authority),
        proposer=SearchComponentCostBinding(
            role=SearchComponentRole.PROPOSER,
            configuration_id=proposer_configuration_id,
            scope_category=old_binding.proposer.scope_category,
            providers=(
                provider_binding(
                    old_binding.proposer.providers[0],
                    configuration_id=proposer_configuration_id,
                ),
            ),
            timed_resources=(
                TimedResourceCostBinding(
                    component_configuration_id=proposer_configuration_id,
                    resource_type=execution.resource_class.role.value,
                    resource_class_digest=execution.resource_class.digest,
                    account=bind_timed_resource_account(project_account),
                ),
            ),
        ),
        scorer=SearchComponentCostBinding(
            role=SearchComponentRole.SCORER,
            configuration_id=old_binding.scorer.configuration_id,
            scope_category=old_binding.scorer.scope_category,
            providers=(
                provider_binding(
                    old_binding.scorer.providers[0],
                    configuration_id=old_binding.scorer.configuration_id,
                ),
            ),
        ),
    )
    old_budget = prepared.confirmation_budget
    confirmation_budget = PairedHarborBudgetRuntime(
        ledger_path=authority.ledger_path,
        ledger_identity=authority.ledger_identity,
        policy=policy,
        phase=old_budget.phase,
        provider_meter_by_panel_member=old_budget.provider_meter_by_panel_member,
        task_resource_meter_by_class_digest=old_budget.task_resource_meter_by_class_digest,
        runner_resource_meter_id=old_budget.runner_resource_meter_id,
    )
    old_protocol = prepared.protocol
    slice_policy = PairedHarborSlicePolicy(
        max_new_blocks=3,
        max_waves_per_invocation=1,
        max_block_runtime_s=900,
        max_invocation_runtime_s=7_200,
    )
    assert slice_policy.digest == old_protocol.confirmation_slice_policy_digest
    protocol = HarnessOptimizationProtocol.create(
        experiment_id=old_protocol.experiment_id,
        protocol_id=old_protocol.protocol_id,
        provenance=old_protocol.provenance,
        partition=prepared.partition,
        baseline=prepared.baseline,
        search=DiscoverySearchPlan.model_validate(
            {
                **old_protocol.search.model_dump(mode="python"),
                "proposer_configuration_id": proposer_configuration_id,
            }
        ),
        candidate_policy=old_protocol.candidate_policy,
        confirmation=old_protocol.confirmation,
        panel_routes=old_protocol.panel_routes,
        execution_plan=old_protocol.execution_plan,
        qualification_roster=prepared.qualification_roster,
        max_concurrent_blocks=old_protocol.max_concurrent_blocks,
        retry_policy_digest=old_protocol.retry_policy_digest,
        search_cost_binding=search_cost_binding,
        confirmation_budget=confirmation_budget,
        confirmation_slice_policy=slice_policy,
    )
    project_prepared = prepare_harness_optimization_study(
        protocol=protocol,
        partition=prepared.partition,
        baseline=prepared.baseline,
        qualification_roster=prepared.qualification_roster,
        confirmation_budget=confirmation_budget,
        search_cost_binding=search_cost_binding,
        confirmation_slice_policy=slice_policy,
    )
    spec = HarnessOptimizationStudySpec(
        prepared=project_prepared,
        partition_control_dir=base.partition_control_dir,
        discovery_job_spec=base.discovery_job_spec,
        confirmation_runtime=base.confirmation_runtime.model_copy(
            update={"budget": confirmation_budget},
            deep=True,
        ),
        discovery_proposer=ProjectDiscoveryProposerRuntime(
            agent=proposer_agent,
            execution=execution,
            lease_ledger_dir=(tmp_path / "project-leases").resolve(),
        ),
        discovery_create_rate_ledger_path=rate_path,
        qualification_report_digest=base.qualification_report_digest,
        confirmation_operation_id=base.confirmation_operation_id,
    )
    return spec, proposer_provider


def test_advance_publishes_one_phase_at_a_time_without_constructing_runtimes(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    coordinator = HarnessOptimizationStudyCoordinator.local(
        spec,
        state_dir=tmp_path / "state",
        resume=False,
        runtime_factory=_UnusedRuntimeFactory(),
        clock=lambda: datetime(2026, 7, 19, 12, 0, tzinfo=UTC),
    )

    first = coordinator.advance()
    second = coordinator.advance()
    third = coordinator.advance()
    fourth = coordinator.advance()

    assert first.kind is HarnessOptimizationAdvanceKind.PHASE
    assert first.previous_phase is None
    assert first.current_phase is StudyPhase.PREPARATION_PLANNED
    assert second.current_phase is StudyPhase.ROSTER_QUALIFIED
    assert third.current_phase is StudyPhase.PROTOCOL_PUBLISHED
    assert fourth.current_phase is StudyPhase.DISCOVERY_RUNNING
    assert coordinator.current_phase is StudyPhase.DISCOVERY_RUNNING


def test_local_state_requires_explicit_resume_and_continues_the_same_chain(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    state_dir = tmp_path / "state"
    first = HarnessOptimizationStudyCoordinator.local(
        spec,
        state_dir=state_dir,
        resume=False,
        runtime_factory=_UnusedRuntimeFactory(),
    )
    first.advance()

    with pytest.raises(ValueError, match="--resume"):
        HarnessOptimizationStudyCoordinator.local(
            spec,
            state_dir=state_dir,
            resume=False,
            runtime_factory=_UnusedRuntimeFactory(),
        )
    resumed = HarnessOptimizationStudyCoordinator.local(
        spec,
        state_dir=state_dir,
        resume=True,
        runtime_factory=_UnusedRuntimeFactory(),
    )

    result = resumed.advance()

    assert result.previous_phase is StudyPhase.PREPARATION_PLANNED
    assert result.current_phase is StudyPhase.ROSTER_QUALIFIED


def test_backend_identity_is_frozen_in_the_execution_plan() -> None:
    baseline = default_agent("baseline")
    local = HarborExecutionPlan.freeze(reference_harness=baseline, reward_key="reward")
    e2b = HarborExecutionPlan.freeze(
        reference_harness=baseline,
        reward_key="reward",
        environment_backend=HarborEnvironmentBackend.E2B,
    )

    assert local.environment_backend is HarborEnvironmentBackend.LOCAL
    assert local.create_rate_policy is None
    assert e2b.environment_backend is HarborEnvironmentBackend.E2B
    assert e2b.create_rate_policy is not None
    assert e2b.create_rate_policy_digest == e2b.create_rate_policy.digest
    assert local.create_rate_policy_digest != e2b.create_rate_policy_digest


def test_local_harbor_can_build_a_deferred_project_backed_proposer_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, provider = _project_spec(tmp_path)
    assert spec.prepared.protocol.execution_plan.environment_backend is (
        HarborEnvironmentBackend.LOCAL
    )
    assert isinstance(provider, ToolCallingProvider)
    monkeypatch.setattr(coordinator_module, "HarborHarnessScorer", _NoDispatchHarborScorer)

    def unexpected_sandbox_factory(**_kwargs: object) -> object:
        raise AssertionError("project construction cannot create an E2B sandbox")

    monkeypatch.setattr(project_module, "default_sandbox_factory", unexpected_sandbox_factory)
    factory = ProductionHarnessOptimizationRuntimeFactory(
        provider_factory=lambda config: (
            provider
            if config == provider.config
            else (_ for _ in ()).throw(AssertionError("unexpected provider config"))
        )
    )

    runtime = factory.build_discovery(spec)

    assert isinstance(runtime.proposer, ProjectDeltaProposer)
    assert provider.calls == 0
    project_runtime = spec.discovery_proposer
    assert isinstance(project_runtime, ProjectDiscoveryProposerRuntime)
    assert not project_runtime.lease_ledger_dir.exists()
    runtime.close()
    assert provider.calls == 0


def test_project_proposer_rate_authority_is_independent_of_local_harbor(
    tmp_path: Path,
) -> None:
    spec, _provider = _project_spec(tmp_path)
    payload = spec.model_dump(mode="python")
    payload["discovery_create_rate_ledger_path"] = None

    with pytest.raises(ValueError, match="E2B discovery requires a create-rate ledger"):
        HarnessOptimizationStudySpec.model_validate(payload)


def test_discovery_factory_closes_scorer_when_provider_construction_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, _provider = _project_spec(tmp_path)
    scorers: list[_NoDispatchHarborScorer] = []

    class RecordingScorer(_NoDispatchHarborScorer):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            scorers.append(self)

    monkeypatch.setattr(coordinator_module, "HarborHarnessScorer", RecordingScorer)

    def fail_provider(_config: ProviderConfig) -> _NoDispatchToolProvider:
        raise RuntimeError("synthetic provider construction failure")

    factory = ProductionHarnessOptimizationRuntimeFactory(provider_factory=fail_provider)

    with pytest.raises(RuntimeError, match="provider construction failure"):
        factory.build_discovery(spec)

    assert len(scorers) == 1
    assert scorers[0].closed is True


def test_rehearsal_is_pure_and_never_claims_a_complete_publication(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    unused_state_dir = tmp_path / "rehearsal-state"

    rehearsal = DeterministicHarnessOptimizationRehearsalFactory().build(spec)

    assert rehearsal.backend is HarborEnvironmentBackend.LOCAL
    assert rehearsal.phase_order[-1] is StudyPhase.CONFIRMATION_RUNNING
    assert rehearsal.would_publish_complete is False
    assert not unused_state_dir.exists()


def test_cli_rehearsal_does_not_create_state_or_execution_effects(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec_file = tmp_path / "study.json"
    spec_file.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    state_dir = tmp_path / "unused-state"

    result = runner.invoke(
        app,
        [
            "harness",
            "optimize",
            "--spec",
            str(spec_file),
            "--state-dir",
            str(state_dir),
            "--rehearse",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"backend": "local"' in result.output
    assert '"would_publish_complete": false' in result.output
    assert not state_dir.exists()


def test_cli_one_slice_defaults_to_local_and_requires_resume_for_existing_state(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    spec_file = tmp_path / "study.json"
    spec_file.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    state_dir = tmp_path / "state"
    command = [
        "harness",
        "optimize",
        "--spec",
        str(spec_file),
        "--state-dir",
        str(state_dir),
        "--one-slice",
    ]

    first = runner.invoke(app, command)
    without_resume = runner.invoke(app, command)
    resumed = runner.invoke(app, [*command, "--resume"])

    assert first.exit_code == 0, first.output
    assert "phase=preparation_planned" in first.output
    assert without_resume.exit_code == 2
    assert "--resume" in without_resume.output
    assert resumed.exit_code == 0, resumed.output
    assert "phase=roster_qualified" in resumed.output


class _OneSliceRunner:
    """Return one deterministic persisted paired slice."""

    def __init__(self, result: PairedHarborSliceResult) -> None:
        self.result = result
        self.calls = 0
        self.resume_calls = 0

    async def run_slice(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
        max_new_blocks: int | None = None,
    ) -> PairedHarborSliceResult:
        del baseline, candidate, max_new_blocks
        self.calls += 1
        return self.result

    async def resume_persisted_slice(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
    ) -> PairedHarborSliceResult:
        del baseline, candidate
        self.resume_calls += 1
        raise PairedHarborNoActiveSliceIntentError

    async def recover_persisted_slice(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
    ) -> PairedHarborSliceResult | None:
        del baseline, candidate
        return None


class _RecoveringRunner:
    """Expose runner-persisted progress while rejecting new confirmation work."""

    def __init__(self) -> None:
        self.result: PairedHarborSliceResult | None = None
        self.recovery_calls = 0
        self.run_calls = 0

    async def run_slice(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
        max_new_blocks: int | None = None,
    ) -> PairedHarborSliceResult:
        del baseline, candidate, max_new_blocks
        self.run_calls += 1
        raise AssertionError("recovery must not dispatch another confirmation slice")

    async def resume_persisted_slice(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
    ) -> PairedHarborSliceResult:
        del baseline, candidate
        raise AssertionError("recovery must reconcile before resuming confirmation work")

    async def recover_persisted_slice(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
    ) -> PairedHarborSliceResult | None:
        del baseline, candidate
        self.recovery_calls += 1
        return self.result


class _CrashBeforePairedIntentRunner(_OneSliceRunner):
    """Crash once after the outer intent but before any paired intent exists."""

    async def resume_persisted_slice(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
    ) -> PairedHarborSliceResult:
        del baseline, candidate
        self.resume_calls += 1
        if self.resume_calls == 1:
            raise RuntimeError("synthetic crash before paired intent")
        raise PairedHarborNoActiveSliceIntentError


class _CrashAfterPairedIntentRunner(_OneSliceRunner):
    """Crash one fresh paired slice, then resume its exact durable intent."""

    def __init__(self, result: PairedHarborSliceResult) -> None:
        super().__init__(result)
        self.has_active_intent = False

    async def run_slice(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
        max_new_blocks: int | None = None,
    ) -> PairedHarborSliceResult:
        del baseline, candidate, max_new_blocks
        self.calls += 1
        self.has_active_intent = True
        raise RuntimeError("synthetic crash after paired intent")

    async def resume_persisted_slice(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
    ) -> PairedHarborSliceResult:
        del baseline, candidate
        self.resume_calls += 1
        if not self.has_active_intent:
            raise PairedHarborNoActiveSliceIntentError
        return self.result


class _SyntheticRuntimeFactory:
    """Run discovery in memory and expose a controllable confirmation runner."""

    def __init__(self, spec: HarnessOptimizationStudySpec) -> None:
        task_ids = tuple(task.task_id for task in spec.prepared.protocol.discovery.tasks)
        self.scorer = _Scorer(task_ids)
        self.proposer = _CodeProposer()
        self.scorer.search_cost_binding = spec.prepared.search_cost_binding.scorer.model_copy(
            deep=True
        )
        self.proposer.search_cost_binding = spec.prepared.search_cost_binding.proposer.model_copy(
            deep=True
        )
        self.confirmation = _RecoveringRunner()

    def build_discovery(
        self,
        spec: HarnessOptimizationStudySpec,
    ) -> HarnessOptimizationDiscoveryRuntime:
        del spec
        return HarnessOptimizationDiscoveryRuntime(
            scorer=self.scorer,
            proposer=self.proposer,
            close=lambda: None,
        )

    def build_confirmation(
        self,
        spec: HarnessOptimizationStudySpec,
        protocol: PairedHarborProtocol,
    ) -> PairedHarborSliceRunner:
        del spec, protocol
        return self.confirmation


def _confirmation_context(
    tmp_path: Path,
) -> tuple[
    StudyLifecycleController,
    OpenedHarnessOptimizationConfirmation,
    PairedHarborProtocol,
    ConfirmationRunningPayload,
]:
    control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    scorer = _Scorer(manifest.discovery_task_ids)
    proposer = _CodeProposer()
    prepared, protocol = _prepare(
        tmp_path,
        manifest,
        baseline,
        scorer=scorer,
        proposer=proposer,
    )
    lifecycle, discovery = _discovery_lifecycle(tmp_path, protocol)
    checkpoints = []
    run_harness_optimization_search(
        prepared.discovery_contract(),
        scorer=scorer,
        proposer=proposer,
        lifecycle=lifecycle,
        authorization=discovery,
        on_checkpoint=checkpoints.append,
        on_proposal_batch_prepare=lambda _witness: None,
        on_proposal_batch_witness=lambda _witness: None,
    )
    checkpoint = checkpoints[-1]
    frozen = freeze_harness_optimization_candidate(
        control_store,
        prepared=prepared,
        checkpoint=checkpoint,
        lifecycle=lifecycle,
        authorization=discovery,
    )
    search_budget = StudyBudgetReport.capture(_budget_authority(prepared))
    lifecycle.publish_candidate_frozen(
        protocol_digest=protocol.digest,
        candidate=frozen.candidate,
        checkpoint=checkpoint,
        search_configuration_digest=discovery.search_configuration_digest,
        search_cost_binding_digest=protocol.search_cost_binding_digest,
        budget_authority=_budget_authority(prepared),
        search_cost_report=search_budget,
        search_cost_report_publication=_artifact_publication(search_budget.digest),
        freeze_record=frozen.freeze_record,
        completed_iterations=protocol.search.iterations,
    )
    candidate_publication = lifecycle.publish_candidate_source(
        protocol_digest=protocol.digest,
        candidate=frozen.candidate,
        publication=_artifact_publication(
            _canonical_digest(frozen.candidate.model_dump(mode="json"))
        ),
    )
    opened = open_harness_optimization_confirmation(
        control_store,
        prepared=prepared,
        frozen=frozen,
        lifecycle=lifecycle,
        authorization=candidate_publication,
    )
    opened_payload = ConfirmationOpenedPayload(
        protocol_digest=protocol.digest,
        candidate_execution_digest=frozen.candidate.execution_digest,
        candidate_freeze_record_digest=frozen.freeze_record.digest,
        confirmation_partition_digest=_canonical_digest(
            opened.confirmation.model_dump(mode="json")
        ),
        confirmation_opening_record_digest=opened.confirmation.opening_record_digest,
        paired_design_digest=opened.design.digest,
        confirmation_task_count=len(opened.confirmation.tasks),
    )
    lifecycle.publish(opened_payload)
    paired = freeze_harness_optimization_harbor_protocol(
        opened,
        lifecycle=lifecycle,
        authorization=opened_payload,
    )
    lifecycle.publish(
        ConfirmationFrozenPayload(
            protocol_digest=protocol.digest,
            paired_protocol_digest=paired.digest,
            budget_binding_digest=paired.budget_binding_digest,
            create_rate_policy_digest=protocol.execution_plan.create_rate_policy_digest,
            slice_policy_digest=paired.slice_policy_digest,
            planned_blocks=len(paired.design.blocks),
            planned_arms=len(paired.design.blocks) * 2,
        )
    )
    running = ConfirmationRunningPayload(
        paired_protocol_digest=paired.digest,
        initial_run_state_digest=confirmation_initial_run_state_digest(
            paired,
            operation_id="confirmation-run",
            generation_id=1,
        ),
        slice_policy_digest=paired.slice_policy_digest,
        confirmation_run_id="confirmation-run",
    )
    lifecycle.publish(running)
    return lifecycle, opened, paired, running


def _first_slice(protocol: PairedHarborProtocol) -> PairedHarborSliceResult:
    selected = _select_paired_harbor_slice_blocks(
        protocol,
        completed_blocks=frozenset(),
        max_new_blocks=protocol.slice_policy.max_new_blocks,
    )
    completed = tuple(
        PairedHarborCompletedBlock(
            block=block,
            generation_id=1,
            pair_generation_id=paired_harbor_pair_generation_id(
                protocol_digest=protocol.digest,
                operation_id="confirmation-run",
                generation_id=1,
                block=block,
            ),
            evidence_digest=_digest(f"evidence:{index}"),
        )
        for index, block in enumerate(selected)
    )
    intent_payload = {
        "intent_version": "1",
        "protocol_digest": protocol.digest,
        "slice_policy_digest": protocol.slice_policy_digest,
        "operation_id": "confirmation-run",
        "intent_index": 1,
        "intent_generation_id": 1,
        "requested_max_new_blocks": protocol.slice_policy.max_new_blocks,
        "previous_intent_digest": None,
        "previous_progress_digest": None,
        "completed_before": (),
        "selected_blocks": tuple(block.model_dump(mode="json") for block in selected),
        "expected_block_count": len(protocol.design.blocks),
    }
    intent = PairedHarborSliceIntent.model_validate(
        {**intent_payload, "intent_digest": _canonical_digest(intent_payload)}
    )
    intent.require_protocol(protocol)
    draft = PairedHarborSliceProgress.model_construct(
        progress_version="2",
        protocol_digest=protocol.digest,
        slice_policy_digest=protocol.slice_policy_digest,
        operation_id="confirmation-run",
        invocation_generation_id=1,
        slice_index=1,
        requested_max_new_blocks=protocol.slice_policy.max_new_blocks,
        slice_intent_digest=intent.intent_digest,
        previous_progress_digest=None,
        completed_before=(),
        selected_blocks=selected,
        completed_blocks=completed,
        expected_block_count=len(protocol.design.blocks),
        completed_block_count=len(completed),
        remaining_block_count=len(protocol.design.blocks) - len(completed),
        complete=False,
        progress_digest=_digest("placeholder"),
    )
    payload = draft.model_dump(mode="json", exclude={"progress_digest"})
    progress = PairedHarborSliceProgress.model_validate(
        {**payload, "progress_digest": _canonical_digest(payload)}
    )
    progress.require_protocol(protocol)
    return PairedHarborSliceResult(progress=progress)


def test_confirmation_adapter_checkpoints_and_reconciles_without_a_second_run(
    tmp_path: Path,
) -> None:
    lifecycle, opened, protocol, running = _confirmation_context(tmp_path)
    runner = _OneSliceRunner(_first_slice(protocol))
    persisted: list[PairedHarborSliceResult] = []

    first = run_harness_optimization_confirmation_slice(
        opened=opened,
        protocol=protocol,
        runner=runner,
        lifecycle=lifecycle,
        authorization=running,
        operation_id="confirmation-run",
        generation_id=1,
        resume_from=None,
        on_checkpoint=persisted.append,
    )
    reconciled = reconcile_harness_optimization_confirmation_slice(
        opened=opened,
        protocol=protocol,
        lifecycle=lifecycle,
        authorization=running,
        operation_id="confirmation-run",
        generation_id=1,
        persisted=first,
    )

    assert persisted == [first]
    assert reconciled == first
    assert runner.resume_calls == 1
    assert runner.calls == 1
    assert lifecycle.current_phase is StudyPhase.CONFIRMATION_RUNNING


def test_confirmation_adapter_reenters_outer_intent_before_creating_paired_intent(
    tmp_path: Path,
) -> None:
    lifecycle, opened, protocol, running = _confirmation_context(tmp_path)
    expected = _first_slice(protocol)
    runner = _CrashBeforePairedIntentRunner(expected)
    persisted: list[PairedHarborSliceResult] = []

    with pytest.raises(RuntimeError, match="before paired intent"):
        run_harness_optimization_confirmation_slice(
            opened=opened,
            protocol=protocol,
            runner=runner,
            lifecycle=lifecycle,
            authorization=running,
            operation_id="confirmation-run",
            generation_id=1,
            resume_from=None,
            on_checkpoint=persisted.append,
        )

    resumed = run_harness_optimization_confirmation_slice(
        opened=opened,
        protocol=protocol,
        runner=runner,
        lifecycle=lifecycle,
        authorization=running,
        operation_id="confirmation-run",
        generation_id=1,
        resume_from=None,
        on_checkpoint=persisted.append,
    )

    assert resumed == expected
    assert persisted == [expected]
    assert runner.resume_calls == 2
    assert runner.calls == 1


def test_confirmation_adapter_resumes_inner_intent_without_fresh_slice_dispatch(
    tmp_path: Path,
) -> None:
    lifecycle, opened, protocol, running = _confirmation_context(tmp_path)
    expected = _first_slice(protocol)
    runner = _CrashAfterPairedIntentRunner(expected)
    persisted: list[PairedHarborSliceResult] = []

    with pytest.raises(RuntimeError, match="after paired intent"):
        run_harness_optimization_confirmation_slice(
            opened=opened,
            protocol=protocol,
            runner=runner,
            lifecycle=lifecycle,
            authorization=running,
            operation_id="confirmation-run",
            generation_id=1,
            resume_from=None,
            on_checkpoint=persisted.append,
        )

    resumed = run_harness_optimization_confirmation_slice(
        opened=opened,
        protocol=protocol,
        runner=runner,
        lifecycle=lifecycle,
        authorization=running,
        operation_id="confirmation-run",
        generation_id=1,
        resume_from=None,
        on_checkpoint=persisted.append,
    )

    assert resumed == expected
    assert persisted == [expected]
    assert runner.resume_calls == 2
    assert runner.calls == 1


def test_confirmation_authorization_drift_fails_before_resume_or_dispatch(
    tmp_path: Path,
) -> None:
    lifecycle, opened, protocol, running = _confirmation_context(tmp_path)
    runner = _OneSliceRunner(_first_slice(protocol))
    drifted = running.model_copy(update={"confirmation_run_id": "different-run"})

    with pytest.raises(ValueError, match="authorization differs"):
        run_harness_optimization_confirmation_slice(
            opened=opened,
            protocol=protocol,
            runner=runner,
            lifecycle=lifecycle,
            authorization=drifted,
            operation_id="confirmation-run",
            generation_id=1,
            resume_from=None,
            on_checkpoint=lambda _result: None,
        )

    assert runner.resume_calls == 0
    assert runner.calls == 0


def test_coordinator_recovers_runner_progress_after_crash_before_state_checkpoint(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    state_dir = tmp_path / "state"
    factory = _SyntheticRuntimeFactory(spec)
    coordinator = HarnessOptimizationStudyCoordinator.local(
        spec,
        state_dir=state_dir,
        resume=False,
        runtime_factory=factory,
    )
    while coordinator.current_phase is not StudyPhase.CONFIRMATION_RUNNING:
        coordinator.advance()

    state_store = LocalHarnessOptimizationStateStore(state_dir / "coordinator")
    state = state_store.load()
    opened = state.opened_confirmation
    protocol = state.paired_protocol
    running = state.confirmation_running
    assert opened is not None
    assert protocol is not None
    assert running is not None
    persisted = _first_slice(protocol)
    factory.confirmation.result = persisted

    evidence = LocalStudyEvidenceStore(
        state_dir / "evidence",
        study_id=spec.prepared.protocol.experiment_id,
    )
    lifecycle = StudyLifecycleController(
        store=StudyJournalStore.create(
            state_dir / "journal",
            study_id=spec.prepared.protocol.experiment_id,
            publisher_configuration_digest=evidence.configuration_digest,
        ),
        publisher=evidence,
        artifact_verifier=evidence,
    )

    def crash_before_state_checkpoint(_result: PairedHarborSliceResult) -> None:
        raise RuntimeError("synthetic crash before coordinator checkpoint")

    with pytest.raises(RuntimeError, match="before coordinator checkpoint"):
        run_harness_optimization_confirmation_slice(
            opened=opened,
            protocol=protocol,
            runner=_OneSliceRunner(persisted),
            lifecycle=lifecycle,
            authorization=running,
            operation_id=spec.confirmation_operation_id,
            generation_id=spec.confirmation_generation_id,
            resume_from=None,
            on_checkpoint=crash_before_state_checkpoint,
        )

    resumed = coordinator.advance()
    recovered_state = state_store.load()

    assert resumed.kind is HarnessOptimizationAdvanceKind.CONFIRMATION_RECONCILED
    assert resumed.checkpoint_sequence == 0
    assert recovered_state.confirmation_slice == persisted
    assert recovered_state.confirmation_journaled_sequence == 0
    assert factory.confirmation.recovery_calls == 1
    assert factory.confirmation.run_calls == 0


def test_append_only_evidence_crash_before_link_leaves_no_poisoned_final_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalStudyEvidenceStore(
        tmp_path / "evidence",
        study_id="atomic-publication-study",
        clock=lambda: datetime(2026, 7, 19, 12, 0, tzinfo=UTC),
    )
    content = {"kind": "protocol", "value": 1}
    digest = _canonical_digest(content)
    real_link = os.link

    def crash_before_link(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("synthetic crash before immutable link")

    monkeypatch.setattr(os, "link", crash_before_link)
    with pytest.raises(OSError, match="before immutable link"):
        store.publish_artifact(kind="protocol", artifact_digest=digest, content=content)

    final = tmp_path / "evidence" / "artifacts" / f"artifact-{digest.removeprefix('sha256:')}.json"
    assert not final.exists()

    monkeypatch.setattr(os, "link", real_link)
    publication = store.publish_artifact(
        kind="protocol",
        artifact_digest=digest,
        content=content,
    )

    assert publication.artifact_digest == digest
    store.verify_artifact(publication)


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL recovery requires POSIX")
def test_append_only_evidence_recovers_staging_from_sigkill_before_link(
    tmp_path: Path,
) -> None:
    store = LocalStudyEvidenceStore(
        tmp_path / "evidence",
        study_id="pre-link-sigkill-study",
        clock=lambda: datetime(2026, 7, 19, 12, 0, tzinfo=UTC),
    )
    content = {"kind": "protocol", "value": 4}
    digest = _canonical_digest(content)
    child_code = """
import os
import signal
import sys
from pathlib import Path

from wmh.evals.harness_optimization_coordinator import LocalStudyEvidenceStore

store = LocalStudyEvidenceStore(Path(sys.argv[1]), study_id="pre-link-sigkill-study")
os.link = lambda *args, **kwargs: os.kill(os.getpid(), signal.SIGKILL)
store.publish_artifact(
    kind="protocol",
    artifact_digest=sys.argv[2],
    content={"kind": "protocol", "value": 4},
)
"""

    child = subprocess.run(  # noqa: S603 - exact test interpreter and fixed local script
        [sys.executable, "-c", child_code, str(tmp_path / "evidence"), digest],
        check=False,
    )

    assert child.returncode == -signal.SIGKILL
    final = tmp_path / "evidence" / "artifacts" / f"artifact-{digest.removeprefix('sha256:')}.json"
    staging = tuple(final.parent.glob(f".publish-{final.name}-*"))
    assert len(staging) == 1
    assert os.lstat(staging[0]).st_nlink == 1

    publication = store.publish_artifact(
        kind="protocol",
        artifact_digest=digest,
        content=content,
    )

    store.verify_artifact(publication)
    assert not tuple(final.parent.glob(f".publish-{final.name}-*"))


def test_append_only_evidence_preserves_a_live_pre_link_staging_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalStudyEvidenceStore(
        tmp_path / "evidence",
        study_id="live-pre-link-study",
        clock=lambda: datetime(2026, 7, 19, 12, 0, tzinfo=UTC),
    )
    content = {"kind": "protocol", "value": 5}
    digest = _canonical_digest(content)
    final = tmp_path / "evidence" / "artifacts" / f"artifact-{digest.removeprefix('sha256:')}.json"
    link_started = threading.Event()
    allow_link = threading.Event()
    real_link = os.link

    def blocked_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        if Path(os.fsdecode(destination)) == final:
            link_started.set()
            assert allow_link.wait(timeout=5)
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(os, "link", blocked_link)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            store.publish_artifact,
            kind="protocol",
            artifact_digest=digest,
            content=content,
        )
        assert link_started.wait(timeout=5)
        try:
            staging = tuple(final.parent.glob(f".publish-{final.name}-*"))
            assert len(staging) == 1
            coordinator_module._recover_atomic_publication_aliases(final)
            assert staging[0].exists()
        finally:
            allow_link.set()
        publication = future.result(timeout=5)

    store.verify_artifact(publication)
    assert not tuple(final.parent.glob(f".publish-{final.name}-*"))


def test_append_only_evidence_recovers_post_link_crash_alias(tmp_path: Path) -> None:
    store = LocalStudyEvidenceStore(
        tmp_path / "evidence",
        study_id="post-link-crash-study",
        clock=lambda: datetime(2026, 7, 19, 12, 0, tzinfo=UTC),
    )
    content = {"kind": "protocol", "value": 2}
    digest = _canonical_digest(content)
    publication = store.publish_artifact(
        kind="protocol",
        artifact_digest=digest,
        content=content,
    )
    final = tmp_path / "evidence" / "artifacts" / f"artifact-{digest.removeprefix('sha256:')}.json"
    alias = final.parent / f".publish-{final.name}-interrupted"
    os.link(final, alias)

    assert os.lstat(final).st_nlink == 2
    store.verify_artifact(publication)

    assert not alias.exists()
    assert os.lstat(final).st_nlink == 1


def test_concurrent_identical_artifact_publication_returns_one_persisted_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_lock = threading.Lock()
    clock_tick = 0

    def clock() -> datetime:
        nonlocal clock_tick
        with clock_lock:
            clock_tick += 1
            return datetime(2026, 7, 19, 12, 0, 0, clock_tick, tzinfo=UTC)

    store = LocalStudyEvidenceStore(
        tmp_path / "evidence",
        study_id="concurrent-publication-study",
        clock=clock,
    )
    content = {"kind": "protocol", "value": 3}
    digest = _canonical_digest(content)
    final = tmp_path / "evidence" / "artifacts" / f"artifact-{digest.removeprefix('sha256:')}.json"
    barrier = threading.Barrier(2)
    loser_reached_validation = threading.Event()
    loser_threads: set[int] = set()
    real_link = os.link
    real_require = coordinator_module._require_private_regular_file

    def racing_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        if Path(os.fsdecode(destination)) == final:
            barrier.wait(timeout=5)
        try:
            real_link(
                source,
                destination,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
                follow_symlinks=follow_symlinks,
            )
        except FileExistsError:
            loser_threads.add(threading.get_ident())
            raise
        if Path(os.fsdecode(destination)) == final:
            assert loser_reached_validation.wait(timeout=5)

    def racing_require(path: Path) -> None:
        if path == final and threading.get_ident() in loser_threads:
            try:
                real_require(path)
            finally:
                loser_reached_validation.set()
            return
        real_require(path)

    monkeypatch.setattr(os, "link", racing_link)
    monkeypatch.setattr(coordinator_module, "_require_private_regular_file", racing_require)

    def publish() -> StudyArtifactPublication:
        return store.publish_artifact(
            kind="protocol",
            artifact_digest=digest,
            content=content,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(publish)
        second_future = executor.submit(publish)
        first = first_future.result(timeout=10)
        second = second_future.result(timeout=10)

    assert first == second
    store.verify_artifact(first)


def test_coordinator_reconciles_terminal_discovery_checkpoint_before_freeze(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    state_dir = tmp_path / "state"
    factory = _SyntheticRuntimeFactory(spec)
    coordinator = HarnessOptimizationStudyCoordinator.local(
        spec,
        state_dir=state_dir,
        resume=False,
        runtime_factory=factory,
    )
    while coordinator.current_phase is not StudyPhase.DISCOVERY_RUNNING:
        coordinator.advance()
    coordinator.advance()

    state_store = LocalHarnessOptimizationStateStore(state_dir / "coordinator")
    working = state_store.load()
    assert working.search_checkpoint is not None
    assert working.search_checkpoint.completed_iteration == 0
    assert working.search_checkpoint_journaled_iteration == 0
    authorization = working.discovery_running
    assert authorization is not None

    evidence = LocalStudyEvidenceStore(
        state_dir / "evidence",
        study_id=spec.prepared.protocol.experiment_id,
    )
    lifecycle = StudyLifecycleController(
        store=StudyJournalStore.create(
            state_dir / "journal",
            study_id=spec.prepared.protocol.experiment_id,
            publisher_configuration_digest=evidence.configuration_digest,
        ),
        publisher=evidence,
        artifact_verifier=evidence,
    )

    def save_witness(witness: SearchProposalBatchWitness) -> None:
        nonlocal working
        working = working.model_copy(update={"proposal_batch_witness": witness}, deep=True)
        state_store.save(working)

    def crash_after_checkpoint(checkpoint: SearchCheckpoint) -> None:
        nonlocal working
        working = working.model_copy(
            update={"search_checkpoint": checkpoint, "proposal_batch_witness": None},
            deep=True,
        )
        state_store.save(working)
        raise RuntimeError("synthetic crash after terminal discovery checkpoint")

    with pytest.raises(RuntimeError, match="after terminal discovery checkpoint"):
        run_harness_optimization_search_slice(
            spec.prepared.discovery_contract(),
            scorer=factory.scorer,
            proposer=factory.proposer,
            lifecycle=lifecycle,
            authorization=authorization,
            resume_from=working.search_checkpoint,
            on_checkpoint=crash_after_checkpoint,
            on_proposal_batch_prepare=save_witness,
            on_proposal_batch_witness=save_witness,
        )
    persisted = state_store.load()
    assert persisted.search_checkpoint is not None
    assert (
        persisted.search_checkpoint.completed_iteration == spec.prepared.protocol.search.iterations
    )
    assert persisted.search_checkpoint_journaled_iteration == 0
    paid_score_calls = factory.scorer.score_calls

    reconciled = coordinator.advance()

    assert reconciled.kind is HarnessOptimizationAdvanceKind.DISCOVERY_RECONCILED
    assert coordinator.current_phase is StudyPhase.DISCOVERY_RUNNING
    assert (
        state_store.load().search_checkpoint_journaled_iteration
        == spec.prepared.protocol.search.iterations
    )
    assert factory.scorer.score_calls == paid_score_calls
    assert coordinator.advance().current_phase is StudyPhase.CANDIDATE_FROZEN
