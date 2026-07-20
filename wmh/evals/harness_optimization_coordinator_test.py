"""Tests for crash-durable harness optimization study composition."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from harbor.models.job.config import DatasetConfig
from typer.testing import CliRunner

from wmh.agents import default_agent
from wmh.cli.app import app
from wmh.evals.harbor.config import HarborEnvironmentBackend, HarborJobSpec
from wmh.evals.harbor.paired_runner import (
    HarborExecutionPlan,
    HarborExecutionRuntime,
    PairedHarborCompletedBlock,
    PairedHarborProtocol,
    PairedHarborSliceIntent,
    PairedHarborSliceProgress,
    PairedHarborSliceResult,
    _select_paired_harbor_slice_blocks,
    paired_harbor_pair_generation_id,
)
from wmh.evals.harness_optimization import (
    OpenedHarnessOptimizationConfirmation,
    freeze_harness_optimization_candidate,
    freeze_harness_optimization_harbor_protocol,
    open_harness_optimization_confirmation,
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
    StudyBudgetReport,
    StudyLifecycleController,
)
from wmh.harness.create import SearchCheckpoint, SearchProposalBatchWitness
from wmh.harness.doc import HarnessDoc

runner = CliRunner()


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

    async def recover_persisted_slice(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
    ) -> PairedHarborSliceResult | None:
        del baseline, candidate
        self.recovery_calls += 1
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
    assert runner.calls == 1
    assert lifecycle.current_phase is StudyPhase.CONFIRMATION_RUNNING


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
