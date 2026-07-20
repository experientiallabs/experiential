"""Tests for the sealed harness-optimization study lifecycle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from wmh.agents import default_agent
from wmh.evals.harbor.config import HarborEnvironmentBackend
from wmh.evals.harbor.paired_runner import (
    HarborExecutionPlan,
    PairedHarborBudgetRuntime,
    PairedHarborPanelRoute,
    PrequalifiedHarborRoster,
    QualifiedHarborTask,
)
from wmh.evals.harness_optimization import (
    BenchmarkProvenance,
    CandidateChangePolicy,
    ConfirmationDecisionRule,
    DiscoverySearchPlan,
    HarnessOptimizationMemberOutcome,
    HarnessOptimizationOutcome,
    HarnessOptimizationProtocol,
    PreparedHarnessOptimizationStudy,
    freeze_harness_optimization_candidate,
    open_harness_optimization_confirmation,
    prepare_harness_optimization_study,
    run_harness_optimization_search,
    run_harness_optimization_search_slice,
)
from wmh.evals.paired import BoundedMeanBet, PairedPanelPlan
from wmh.evals.partition import (
    BenchmarkPartitionManifest,
    PartitionControlScope,
    PartitionControlStore,
    PartitionTask,
    initialize_partition_genesis,
)
from wmh.evals.study_journal import (
    MAX_STUDY_RUN_CHECKPOINT_SEQUENCE,
    ExternalPublicationReceipt,
    StudyJournalGenesis,
    StudyJournalStore,
    StudyPhase,
    StudyPhaseCommitment,
    StudyPhaseRecord,
)
from wmh.evals.study_lifecycle import (
    DiscoveryRunningPayload,
    PreparationPlannedPayload,
    ProtocolPublishedPayload,
    RosterQualifiedPayload,
    StudyArtifactPublication,
    StudyBudgetReport,
    StudyLifecycleController,
)
from wmh.harness.create import SearchCheckpoint, SearchProposalBatchWitness
from wmh.harness.delta import (
    FailureSignature,
    HarnessDelta,
    SurfaceOp,
    compute_delta_id,
)
from wmh.harness.doc import HarnessDoc, SurfaceKind
from wmh.harness.proposer import ProposalFailure
from wmh.harness.runtime import HarnessSearchCancelled
from wmh.harness.scoring import (
    HarnessScoreReport,
    ScoreCapabilities,
    ScoreRequest,
    ScoreRunHealth,
    TaskScore,
)
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.tracking.budget import (
    BudgetLedgerAuthority,
    BudgetPolicy,
    ProviderCostMeter,
    TokenPriceCeiling,
    bootstrap_budget_ledger,
)


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


class _Publisher:
    configuration_digest = _digest("optimizer-lifecycle-publisher")

    def __init__(self) -> None:
        self.receipts: dict[str, ExternalPublicationReceipt] = {}

    def publish(self, commitment: StudyPhaseCommitment) -> ExternalPublicationReceipt:
        receipt = self.receipts.get(commitment.digest)
        if receipt is None:
            receipt = ExternalPublicationReceipt(
                commitment_digest=commitment.digest,
                publisher="test-log",
                publication_id=f"entry-{commitment.sequence}",
                immutable_locator=f"test://log/{commitment.digest}",
                published_at=datetime(2026, 7, 19, 12, 0, tzinfo=UTC),
                evidence={},
            )
            self.receipts[commitment.digest] = receipt
        return receipt

    def verify(
        self,
        commitment: StudyPhaseCommitment,
        receipt: ExternalPublicationReceipt,
    ) -> None:
        if self.receipts.get(commitment.digest) != receipt:
            raise ValueError("missing publication")

    def verify_chain_head(
        self,
        genesis: StudyJournalGenesis,
        records: tuple[StudyPhaseRecord, ...],
        pending: StudyPhaseCommitment | None,
    ) -> None:
        del genesis, records, pending

    def verify_artifact(self, publication: StudyArtifactPublication) -> None:
        expected = _artifact_publication(publication.artifact_digest)
        if publication != expected:
            raise ValueError("missing publication")


def _artifact_publication(artifact_digest: str) -> StudyArtifactPublication:
    return StudyArtifactPublication.create(
        artifact_digest=artifact_digest,
        publisher="test-artifacts",
        publication_id=f"artifact-{artifact_digest}",
        immutable_locator=f"test://artifacts/{artifact_digest}",
        published_at=datetime(2026, 7, 19, 11, 0, tzinfo=UTC),
        evidence={},
    )


def _roster_digest(manifest: BenchmarkPartitionManifest) -> str:
    payload = json.dumps(
        [task.model_dump(mode="json") for task in manifest.tasks],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _checkpoint_with_search_run_id(
    checkpoint: SearchCheckpoint,
    search_run_id: str,
) -> SearchCheckpoint:
    changed = checkpoint.model_copy(
        update={
            "configuration": checkpoint.configuration.model_copy(
                update={"search_run_id": search_run_id}
            ),
            "payload_sha256": "0" * 64,
        },
        deep=True,
    )
    payload = changed.model_dump(mode="json", exclude={"payload_sha256"})
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return SearchCheckpoint.model_validate(
        {**payload, "payload_sha256": hashlib.sha256(serialized).hexdigest()}
    )


def _partition(tmp_path: Path) -> tuple[PartitionControlStore, BenchmarkPartitionManifest]:
    control_dir = tmp_path / "partition-control"
    control_dir.mkdir(mode=0o700)
    store = PartitionControlStore(control_dir)
    tasks = tuple(
        PartitionTask(
            task_id=f"task-{index}",
            stratum="shell",
            group_id=f"family-{index}",
            content_digest=_digest(f"task-{index}"),
        )
        for index in range(4)
    )
    genesis = initialize_partition_genesis(
        store,
        scope=PartitionControlScope(
            experiment_id="optimizer-study",
            protocol_id="protocol-v1",
        ),
        tasks=tasks,
        discovery_counts={"shell": 2},
    )
    return store, BenchmarkPartitionManifest.create(
        tasks=tasks,
        discovery_counts={"shell": 2},
        genesis=genesis,
    )


class _Scorer:
    capabilities = ScoreCapabilities(task_subsets=False, attempt_overrides=False)
    default_attempts = 2
    configuration_id = "scorer-config"

    def __init__(self, task_ids: tuple[str, ...]) -> None:
        self.task_ids = task_ids
        self.score_calls = 0

    def validate_candidate(self, candidate: HarnessDoc) -> str | None:
        return None

    def before_proposal_batch(self) -> None:
        return None

    def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScoreReport:
        self.score_calls += 1
        improved = any("wmh-test-improvement" in item.content for item in candidate.code_files())
        score = float(improved)
        per_task = {
            task_id: TaskScore(
                task_id=task_id,
                score=score,
                secondary_score=score,
                passed=improved,
                mechanisms=() if improved else ("agent-control-flow",),
                evidence="synthetic score evidence",
            )
            for task_id in self.task_ids
        }
        identity = hashlib.sha256(
            f"{candidate.execution_hash}:{request.purpose}".encode()
        ).hexdigest()
        return HarnessScoreReport(
            evaluation_id=identity,
            score=score,
            secondary_score=score,
            attempts=self.default_attempts,
            run_health=ScoreRunHealth.VALID,
            per_task=per_task,
        )


class _FailOnceScorer(_Scorer):
    def __init__(self, task_ids: tuple[str, ...]) -> None:
        super().__init__(task_ids)
        self.score_attempts = 0

    def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScoreReport:
        self.score_attempts += 1
        if self.score_attempts == 1:
            raise RuntimeError("provider failed before checkpoint zero")
        return super().score(candidate, request=request)


class _AlternateScorer(_Scorer):
    """A distinct runtime implementation that deliberately reuses the same opaque ID."""


class _CodeProposer:
    configuration_id = "proposer-config"
    durable_state_required = False
    score_archive_required = False

    def __init__(self) -> None:
        self.proposal_calls = 0

    def propose_batch(
        self,
        parent: HarnessDoc,
        trigger: FailureSignature,
        evidence: str,
        *,
        history: list[HarnessDelta],
        count: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[HarnessDelta | ProposalFailure | None]:
        self.proposal_calls += 1
        del evidence, history, should_cancel
        target = parent.code_files()[0]
        op = SurfaceOp(
            op="replace",
            surface_id=target.id,
            content=target.content + "\n// wmh-test-improvement\n",
            rationale="Exercise a source-code candidate in the lifecycle test.",
        )
        delta = HarnessDelta(
            delta_id=compute_delta_id(parent.doc_hash, [op]),
            parent_doc_hash=parent.doc_hash,
            trigger=trigger,
            preconditions={target.id: target.content_hash},
            ops=[op],
            expected_effect="Every synthetic discovery task changes from failure to success.",
        )
        return [delta.model_copy(deep=True) for _ in range(count)]


class _PromptProposer(_CodeProposer):
    configuration_id = "prompt-proposer-config"

    def propose_batch(
        self,
        parent: HarnessDoc,
        trigger: FailureSignature,
        evidence: str,
        *,
        history: list[HarnessDelta],
        count: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[HarnessDelta | ProposalFailure | None]:
        del evidence, history, should_cancel
        target = next(item for item in parent.surfaces if item.kind is SurfaceKind.PROMPT)
        op = SurfaceOp(
            op="replace",
            surface_id=target.id,
            content=target.content + "\nBe careful.",
            rationale="Exercise the source-code policy rejection.",
        )
        delta = HarnessDelta(
            delta_id=compute_delta_id(parent.doc_hash, [op]),
            parent_doc_hash=parent.doc_hash,
            trigger=trigger,
            preconditions={target.id: target.content_hash},
            ops=[op],
            expected_effect="The prompt-only test candidate changes the synthetic score.",
        )
        return [delta.model_copy(deep=True) for _ in range(count)]


def _protocol(
    tmp_path: Path,
    manifest: BenchmarkPartitionManifest,
    baseline: HarnessDoc,
    *,
    proposer_configuration_id: str = "proposer-config",
    roster_digest: str | None = None,
) -> tuple[HarnessOptimizationProtocol, PrequalifiedHarborRoster, PairedHarborBudgetRuntime]:
    panel = tuple(
        PairedPanelPlan(panel_member=member, attempts=15) for member in ("glm", "haiku", "opus")
    )
    routes = tuple(
        PairedHarborPanelRoute(
            panel_member=member,
            provider_config=ProviderConfig(
                kind=ProviderKind.BEDROCK,
                model=f"model-{member}",
                region="us-west-2",
            ),
        )
        for member in ("glm", "haiku", "opus")
    )
    execution_plan = HarborExecutionPlan.freeze(
        reference_harness=baseline,
        reward_key="reward",
    )
    qualification_roster = PrequalifiedHarborRoster(
        execution_plan_digest=execution_plan.digest,
        tasks=tuple(
            QualifiedHarborTask(
                task_id=task.task_id,
                dataset_id="synthetic",
                content_digest=task.content_digest,
                task_key=_digest(f"task-key:{task.task_id}"),
                task_environment_digest=_digest(f"environment:{task.task_id}"),
                environment_backend=HarborEnvironmentBackend.LOCAL,
            )
            for task in manifest.tasks
        ),
    )
    meter_by_member: dict[str, str] = {
        member: f"worker-{member}" for member in ("glm", "haiku", "opus")
    }
    budget_policy = BudgetPolicy(
        study_id="optimizer-study",
        manifest_digest=manifest.digest,
        hard_limit_nano_usd=15_000_000_000_000,
        phase_limits_nano_usd={"confirmation": 15_000_000_000_000},
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
    ledger = bootstrap_budget_ledger(
        (tmp_path / f"budget-{proposer_configuration_id}.sqlite3").resolve(),
        budget_policy,
    )
    budget = PairedHarborBudgetRuntime(
        ledger_path=ledger.ledger_path,
        ledger_identity=ledger.ledger_identity,
        policy=budget_policy,
        phase="confirmation",
        provider_meter_by_panel_member=meter_by_member,
    )
    protocol = HarnessOptimizationProtocol.create(
        experiment_id="optimizer-study",
        protocol_id="protocol-v1",
        provenance=BenchmarkProvenance(
            adapter="harbor",
            adapter_version="0.18.0",
            dataset="terminal-benchmark",
            dataset_revision="revision-1",
            roster_digest=roster_digest or _roster_digest(manifest),
        ),
        partition=manifest,
        baseline=baseline,
        search=DiscoverySearchPlan(
            iterations=1,
            proposal_batch_size=1,
            attempts_per_task=2,
            scorer_configuration_id="scorer-config",
            proposer_configuration_id=proposer_configuration_id,
        ),
        candidate_policy=CandidateChangePolicy(minimum_changed_code_surfaces=1),
        confirmation=ConfirmationDecisionRule(
            panel=panel,
            primary_e_value_bets=(BoundedMeanBet(fraction=0.5, weight=1.0),),
            schedule_seed="schedule-seed",
            analysis_seed="analysis-seed",
            randomization_samples=999,
            alpha=0.05,
            minimum_equal_task_member_delta=0.03,
            noninferiority_margin=0.0,
        ),
        panel_routes=routes,
        execution_plan=execution_plan,
        qualification_roster=qualification_roster,
        max_concurrent_blocks=3,
        retry_policy_digest=_digest("no-retries"),
        search_cost_binding_digest=_digest("search-cost-binding"),
        confirmation_budget=budget,
        create_rate_policy_digest=_digest("create-rate-policy"),
        confirmation_slice_policy_digest=_digest("slice-policy"),
    )
    return protocol, qualification_roster, budget


def _prepare(
    tmp_path: Path,
    manifest: BenchmarkPartitionManifest,
    baseline: HarnessDoc,
    *,
    proposer_configuration_id: str = "proposer-config",
) -> tuple[PreparedHarnessOptimizationStudy, HarnessOptimizationProtocol]:
    protocol, roster, budget = _protocol(
        tmp_path,
        manifest,
        baseline,
        proposer_configuration_id=proposer_configuration_id,
    )
    prepared = prepare_harness_optimization_study(
        protocol=protocol,
        partition=manifest,
        baseline=baseline,
        qualification_roster=roster,
        confirmation_budget=budget,
    )
    return prepared, protocol


def _budget_authority(prepared: PreparedHarnessOptimizationStudy) -> BudgetLedgerAuthority:
    runtime = prepared.confirmation_budget
    return BudgetLedgerAuthority(
        ledger_path=runtime.ledger_path,
        ledger_identity=runtime.ledger_identity,
        policy=runtime.policy,
    )


def _discovery_lifecycle(
    tmp_path: Path,
    protocol: HarnessOptimizationProtocol,
    *,
    start_discovery: bool = True,
) -> tuple[StudyLifecycleController, DiscoveryRunningPayload]:
    publisher = _Publisher()
    store = StudyJournalStore.create(
        tmp_path / "study-journal",
        study_id=protocol.experiment_id,
        publisher_configuration_digest=publisher.configuration_digest,
    )
    controller = StudyLifecycleController(
        store=store,
        publisher=publisher,
        artifact_verifier=publisher,
    )
    controller.publish(
        PreparationPlannedPayload(
            study_plan_digest=_digest("study-plan"),
            budget_policy_digest=protocol.confirmation_budget_policy_digest,
            budget_binding_digest=protocol.confirmation_budget_binding_digest,
            budget_ledger_identity=protocol.confirmation_budget_ledger_identity,
            maximum_paid_cost_nano_usd=15_000_000_000_000,
        )
    )
    controller.publish(
        RosterQualifiedPayload(
            qualified_roster_digest=protocol.qualification_roster_digest,
            qualification_report_digest=_digest("qualification-report"),
            execution_plan_digest=protocol.execution_plan.digest,
            qualified_task_count=4,
        )
    )
    protocol_publication = _artifact_publication(protocol.digest)
    controller.publish_protocol(
        ProtocolPublishedPayload(
            protocol_digest=protocol.digest,
            protocol_artifact_publication_digest=protocol_publication.digest,
            partition_manifest_digest=protocol.partition_manifest_digest,
            qualified_roster_digest=protocol.qualification_roster_digest,
            search_cost_binding_digest=protocol.search_cost_binding_digest,
            confirmation_budget_binding_digest=protocol.confirmation_budget_binding_digest,
        ),
        publication=protocol_publication,
    )
    authorization = DiscoveryRunningPayload(
        protocol_digest=protocol.digest,
        search_configuration_digest=_canonical_digest(protocol.search.model_dump(mode="json")),
        search_run_id="discovery-run",
    )
    if start_discovery:
        controller.publish(authorization)
    return controller, authorization


def test_search_freeze_and_open_confirmation_without_exposing_heldout_ids(
    tmp_path: Path,
) -> None:
    control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    scorer = _Scorer(manifest.discovery_task_ids)
    proposer = _CodeProposer()
    prepared, protocol = _prepare(tmp_path, manifest, baseline)
    lifecycle, discovery_authorization = _discovery_lifecycle(tmp_path, protocol)

    public_json = prepared.discovery_contract().model_dump_json()
    assert all(task_id in public_json for task_id in manifest.discovery_task_ids)
    assert all(task_id not in public_json for task_id in manifest.confirmation_task_ids)
    assert manifest.seal_nonce not in public_json

    checkpoints: list[SearchCheckpoint] = []
    result = run_harness_optimization_search(
        prepared.discovery_contract(),
        scorer=scorer,
        proposer=proposer,
        lifecycle=lifecycle,
        authorization=discovery_authorization,
        on_checkpoint=checkpoints.append,
        on_proposal_batch_prepare=lambda _witness: None,
        on_proposal_batch_witness=lambda _witness: None,
    )
    assert result.best.execution_digest != baseline.execution_digest
    assert checkpoints[-1].completed_iteration == protocol.search.iterations
    assert checkpoints[-1].configuration.search_run_id == discovery_authorization.search_run_id

    with pytest.raises(ValueError, match="search run"):
        freeze_harness_optimization_candidate(
            control_store,
            prepared=prepared,
            checkpoint=_checkpoint_with_search_run_id(checkpoints[-1], "different-run"),
            lifecycle=lifecycle,
            authorization=discovery_authorization,
        )

    frozen = freeze_harness_optimization_candidate(
        control_store,
        prepared=prepared,
        checkpoint=checkpoints[-1],
        lifecycle=lifecycle,
        authorization=discovery_authorization,
    )
    assert frozen.candidate == result.best
    assert (
        frozen.freeze_record.confirmation_protocol_digest == frozen.confirmation_commitment.digest
    )
    assert frozen.freeze_record.selection_evidence_digest == frozen.checkpoint_payload_digest
    with pytest.raises(ValueError, match="selection checkpoint"):
        type(frozen).model_validate(
            {
                **frozen.model_dump(mode="json"),
                "checkpoint_payload_digest": _digest("different-checkpoint"),
            }
        )

    budget_authority = _budget_authority(prepared)
    search_cost_report = StudyBudgetReport.capture(budget_authority)
    cost_publication = _artifact_publication(search_cost_report.digest)
    lifecycle.publish_candidate_frozen(
        protocol_digest=protocol.digest,
        candidate=frozen.candidate,
        checkpoint=checkpoints[-1],
        search_configuration_digest=discovery_authorization.search_configuration_digest,
        search_cost_binding_digest=protocol.search_cost_binding_digest,
        budget_authority=budget_authority,
        search_cost_report=search_cost_report,
        search_cost_report_publication=cost_publication,
        freeze_record=frozen.freeze_record,
        completed_iterations=protocol.search.iterations,
    )
    candidate_source_digest = _canonical_digest(frozen.candidate.model_dump(mode="json"))
    candidate_publication = lifecycle.publish_candidate_source(
        protocol_digest=protocol.digest,
        candidate=frozen.candidate,
        publication=_artifact_publication(candidate_source_digest),
    )
    opened = open_harness_optimization_confirmation(
        control_store,
        prepared=prepared,
        frozen=frozen,
        lifecycle=lifecycle,
        authorization=candidate_publication,
    )
    assert tuple(task.task_id for task in opened.confirmation.tasks) == tuple(
        sorted(manifest.confirmation_task_ids)
    )
    assert opened.design.panel_members == ("glm", "haiku", "opus")
    assert opened.design.attempts_by_member == {"glm": 15, "haiku": 15, "opus": 15}
    assert opened.confirmation.confirmation_protocol_digest == frozen.confirmation_commitment.digest


def test_search_slice_returns_after_one_durable_checkpoint_and_resumes(
    tmp_path: Path,
) -> None:
    _control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    scorer = _Scorer(manifest.discovery_task_ids)
    proposer = _CodeProposer()
    prepared, protocol = _prepare(tmp_path, manifest, baseline)
    lifecycle, authorization = _discovery_lifecycle(tmp_path, protocol)
    persisted: list[SearchCheckpoint] = []

    first = run_harness_optimization_search_slice(
        prepared.discovery_contract(),
        scorer=scorer,
        proposer=proposer,
        lifecycle=lifecycle,
        authorization=authorization,
        on_checkpoint=persisted.append,
    )

    assert first.checkpoint == persisted[-1]
    assert first.checkpoint.completed_iteration == 0
    assert first.result is None
    assert scorer.score_calls == 1

    second = run_harness_optimization_search_slice(
        prepared.discovery_contract(),
        scorer=scorer,
        proposer=proposer,
        lifecycle=lifecycle,
        authorization=authorization,
        resume_from=first.checkpoint,
        on_checkpoint=persisted.append,
        on_proposal_batch_prepare=lambda _witness: None,
        on_proposal_batch_witness=lambda _witness: None,
    )

    assert [checkpoint.completed_iteration for checkpoint in persisted] == [0, 1]
    assert second.checkpoint == persisted[-1]
    assert second.result is not None
    assert second.result.best.execution_digest != baseline.execution_digest
    assert scorer.score_calls == 2


def test_search_slice_rejects_identity_drift_before_paid_resume_work(
    tmp_path: Path,
) -> None:
    _control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    prepared, protocol = _prepare(tmp_path, manifest, baseline)
    lifecycle, authorization = _discovery_lifecycle(tmp_path, protocol)
    first_scorer = _Scorer(manifest.discovery_task_ids)
    first = run_harness_optimization_search_slice(
        prepared.discovery_contract(),
        scorer=first_scorer,
        proposer=_CodeProposer(),
        lifecycle=lifecycle,
        authorization=authorization,
        on_checkpoint=lambda _checkpoint: None,
    )
    drifted_scorer = _Scorer(tuple(reversed(manifest.discovery_task_ids)))

    with pytest.raises(ValueError, match="task matrix"):
        run_harness_optimization_search_slice(
            prepared.discovery_contract(),
            scorer=drifted_scorer,
            proposer=_CodeProposer(),
            lifecycle=lifecycle,
            authorization=authorization,
            resume_from=first.checkpoint,
            on_checkpoint=lambda _checkpoint: None,
        )

    assert drifted_scorer.score_calls == 0


def test_search_slice_checkpoint_callback_crash_resumes_from_captured_state(
    tmp_path: Path,
) -> None:
    _control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    prepared, protocol = _prepare(tmp_path, manifest, baseline)
    lifecycle, authorization = _discovery_lifecycle(tmp_path, protocol)
    scorer = _Scorer(manifest.discovery_task_ids)
    persisted: list[SearchCheckpoint] = []

    def _persist_then_crash(checkpoint: SearchCheckpoint) -> None:
        persisted.append(checkpoint)
        raise RuntimeError("simulated host crash after checkpoint persistence")

    with pytest.raises(RuntimeError, match="simulated host crash"):
        run_harness_optimization_search_slice(
            prepared.discovery_contract(),
            scorer=scorer,
            proposer=_CodeProposer(),
            lifecycle=lifecycle,
            authorization=authorization,
            on_checkpoint=_persist_then_crash,
        )

    assert persisted[-1].completed_iteration == 0
    resumed = run_harness_optimization_search_slice(
        prepared.discovery_contract(),
        scorer=scorer,
        proposer=_CodeProposer(),
        lifecycle=lifecycle,
        authorization=authorization,
        resume_from=persisted[-1],
        on_checkpoint=persisted.append,
        on_proposal_batch_prepare=lambda _witness: None,
        on_proposal_batch_witness=lambda _witness: None,
    )

    assert resumed.result is not None
    assert resumed.checkpoint.completed_iteration == 1
    assert scorer.score_calls == 2


def test_search_slice_fails_closed_after_provider_failure_before_checkpoint_zero(
    tmp_path: Path,
) -> None:
    _control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    prepared, protocol = _prepare(tmp_path, manifest, baseline)
    lifecycle, authorization = _discovery_lifecycle(tmp_path, protocol)
    scorer = _FailOnceScorer(manifest.discovery_task_ids)
    proposer = _CodeProposer()
    persisted: list[SearchCheckpoint] = []

    with pytest.raises(RuntimeError, match="provider failed before checkpoint zero"):
        run_harness_optimization_search_slice(
            prepared.discovery_contract(),
            scorer=scorer,
            proposer=proposer,
            lifecycle=lifecycle,
            authorization=authorization,
            on_checkpoint=persisted.append,
        )

    with pytest.raises(ValueError, match="ambiguous durable slice intent"):
        run_harness_optimization_search_slice(
            prepared.discovery_contract(),
            scorer=scorer,
            proposer=proposer,
            lifecycle=lifecycle,
            authorization=authorization,
            on_checkpoint=persisted.append,
        )

    assert persisted == []
    assert scorer.score_attempts == 1
    assert scorer.score_calls == 0
    assert proposer.proposal_calls == 0


def test_full_search_uses_the_same_prepaid_slice_fence(tmp_path: Path) -> None:
    _control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    prepared, protocol = _prepare(tmp_path, manifest, baseline)
    lifecycle, authorization = _discovery_lifecycle(tmp_path, protocol)
    scorer = _FailOnceScorer(manifest.discovery_task_ids)

    with pytest.raises(RuntimeError, match="provider failed before checkpoint zero"):
        run_harness_optimization_search(
            prepared.discovery_contract(),
            scorer=scorer,
            proposer=_CodeProposer(),
            lifecycle=lifecycle,
            authorization=authorization,
            on_checkpoint=lambda _checkpoint: None,
            on_proposal_batch_prepare=lambda _witness: None,
            on_proposal_batch_witness=lambda _witness: None,
        )

    with pytest.raises(ValueError, match="ambiguous durable slice intent"):
        run_harness_optimization_search(
            prepared.discovery_contract(),
            scorer=scorer,
            proposer=_CodeProposer(),
            lifecycle=lifecycle,
            authorization=authorization,
            on_checkpoint=lambda _checkpoint: None,
            on_proposal_batch_prepare=lambda _witness: None,
            on_proposal_batch_witness=lambda _witness: None,
        )

    assert scorer.score_attempts == 1


def test_full_search_preflights_durable_proposal_callbacks_before_scoring(
    tmp_path: Path,
) -> None:
    _control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    prepared, protocol = _prepare(tmp_path, manifest, baseline)
    lifecycle, authorization = _discovery_lifecycle(tmp_path, protocol)
    scorer = _Scorer(manifest.discovery_task_ids)

    with pytest.raises(ValueError, match="durable proposal prepare and witness"):
        run_harness_optimization_search(
            prepared.discovery_contract(),
            scorer=scorer,
            proposer=_CodeProposer(),
            lifecycle=lifecycle,
            authorization=authorization,
            on_checkpoint=lambda _checkpoint: None,
        )

    assert scorer.score_calls == 0
    recovered = run_harness_optimization_search(
        prepared.discovery_contract(),
        scorer=scorer,
        proposer=_CodeProposer(),
        lifecycle=lifecycle,
        authorization=authorization,
        on_checkpoint=lambda _checkpoint: None,
        on_proposal_batch_prepare=lambda _witness: None,
        on_proposal_batch_witness=lambda _witness: None,
    )
    assert recovered.best.execution_digest != baseline.execution_digest
    assert scorer.score_calls == 2


def test_search_slice_rejects_witness_without_resume_before_creating_an_intent(
    tmp_path: Path,
) -> None:
    _control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    prepared, protocol = _prepare(tmp_path, manifest, baseline)
    source_lifecycle, authorization = _discovery_lifecycle(tmp_path, protocol)
    prepared_witnesses: list[SearchProposalBatchWitness] = []
    first = run_harness_optimization_search_slice(
        prepared.discovery_contract(),
        scorer=_Scorer(manifest.discovery_task_ids),
        proposer=_CodeProposer(),
        lifecycle=source_lifecycle,
        authorization=authorization,
        on_checkpoint=lambda _checkpoint: None,
    )
    run_harness_optimization_search_slice(
        prepared.discovery_contract(),
        scorer=_Scorer(manifest.discovery_task_ids),
        proposer=_CodeProposer(),
        lifecycle=source_lifecycle,
        authorization=authorization,
        resume_from=first.checkpoint,
        on_checkpoint=lambda _checkpoint: None,
        on_proposal_batch_prepare=prepared_witnesses.append,
        on_proposal_batch_witness=lambda _witness: None,
    )
    target_root = tmp_path / "target"
    target_root.mkdir()
    target_lifecycle, target_authorization = _discovery_lifecycle(target_root, protocol)
    target_scorer = _Scorer(manifest.discovery_task_ids)

    with pytest.raises(ValueError, match="requires resume_from"):
        run_harness_optimization_search_slice(
            prepared.discovery_contract(),
            scorer=target_scorer,
            proposer=_CodeProposer(),
            lifecycle=target_lifecycle,
            authorization=target_authorization,
            resume_proposal_batch_witness=prepared_witnesses[-1],
            on_checkpoint=lambda _checkpoint: None,
        )

    recovered = run_harness_optimization_search_slice(
        prepared.discovery_contract(),
        scorer=target_scorer,
        proposer=_CodeProposer(),
        lifecycle=target_lifecycle,
        authorization=target_authorization,
        on_checkpoint=lambda _checkpoint: None,
    )
    assert recovered.checkpoint.completed_iteration == 0
    assert target_scorer.score_calls == 1


def test_search_slice_binds_runtime_implementation_before_first_paid_call(
    tmp_path: Path,
) -> None:
    _control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    prepared, protocol = _prepare(tmp_path, manifest, baseline)
    lifecycle, authorization = _discovery_lifecycle(tmp_path, protocol)
    failed_scorer = _FailOnceScorer(manifest.discovery_task_ids)

    with pytest.raises(RuntimeError, match="provider failed before checkpoint zero"):
        run_harness_optimization_search_slice(
            prepared.discovery_contract(),
            scorer=failed_scorer,
            proposer=_CodeProposer(),
            lifecycle=lifecycle,
            authorization=authorization,
            on_checkpoint=lambda _checkpoint: None,
        )

    drifted_scorer = _AlternateScorer(manifest.discovery_task_ids)
    with pytest.raises(ValueError, match="different run identity or configuration"):
        run_harness_optimization_search_slice(
            prepared.discovery_contract(),
            scorer=drifted_scorer,
            proposer=_CodeProposer(),
            lifecycle=lifecycle,
            authorization=authorization,
            on_checkpoint=lambda _checkpoint: None,
        )

    assert failed_scorer.score_attempts == 1
    assert drifted_scorer.score_calls == 0


def test_search_slice_recovers_final_checkpoint_callback_crash_without_paid_work(
    tmp_path: Path,
) -> None:
    _control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    prepared, protocol = _prepare(tmp_path, manifest, baseline)
    lifecycle, authorization = _discovery_lifecycle(tmp_path, protocol)
    scorer = _Scorer(manifest.discovery_task_ids)
    proposer = _CodeProposer()
    persisted: list[SearchCheckpoint] = []
    prepared_witnesses: list[SearchProposalBatchWitness] = []
    completed_witnesses: list[SearchProposalBatchWitness] = []
    first = run_harness_optimization_search_slice(
        prepared.discovery_contract(),
        scorer=scorer,
        proposer=proposer,
        lifecycle=lifecycle,
        authorization=authorization,
        on_checkpoint=persisted.append,
    )

    def _persist_final_then_crash(checkpoint: SearchCheckpoint) -> None:
        persisted.append(checkpoint)
        raise RuntimeError("simulated crash after final checkpoint persistence")

    with pytest.raises(RuntimeError, match="after final checkpoint persistence"):
        run_harness_optimization_search_slice(
            prepared.discovery_contract(),
            scorer=scorer,
            proposer=proposer,
            lifecycle=lifecycle,
            authorization=authorization,
            resume_from=first.checkpoint,
            on_checkpoint=_persist_final_then_crash,
            on_proposal_batch_prepare=prepared_witnesses.append,
            on_proposal_batch_witness=completed_witnesses.append,
        )

    completed = persisted[-1]
    assert completed.completed_iteration == protocol.search.iterations
    score_calls = scorer.score_calls
    proposal_calls = proposer.proposal_calls
    persisted_count = len(persisted)

    with pytest.raises(ValueError, match="committed proposal batch witness must be completed"):
        run_harness_optimization_search_slice(
            prepared.discovery_contract(),
            scorer=scorer,
            proposer=proposer,
            lifecycle=lifecycle,
            authorization=authorization,
            resume_from=completed,
            resume_proposal_batch_witness=prepared_witnesses[-1],
            on_checkpoint=persisted.append,
        )

    recovered = run_harness_optimization_search_slice(
        prepared.discovery_contract(),
        scorer=scorer,
        proposer=proposer,
        lifecycle=lifecycle,
        authorization=authorization,
        resume_from=completed,
        resume_proposal_batch_witness=completed_witnesses[-1],
        on_checkpoint=persisted.append,
    )

    assert recovered.checkpoint == completed
    assert recovered.result is not None
    assert recovered.result.best.execution_digest != baseline.execution_digest
    assert scorer.score_calls == score_calls
    assert proposer.proposal_calls == proposal_calls
    assert len(persisted) == persisted_count


def test_discovery_iterations_fit_the_durable_journal_sequence_space() -> None:
    with pytest.raises(ValueError, match="less than or equal to 99999999"):
        DiscoverySearchPlan(
            iterations=MAX_STUDY_RUN_CHECKPOINT_SEQUENCE + 1,
            attempts_per_task=1,
            scorer_configuration_id="scorer",
            proposer_configuration_id="proposer",
        )


def test_search_slice_completion_wins_cancellation_after_final_checkpoint(
    tmp_path: Path,
) -> None:
    _control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    prepared, protocol = _prepare(tmp_path, manifest, baseline)
    lifecycle, authorization = _discovery_lifecycle(tmp_path, protocol)
    scorer = _Scorer(manifest.discovery_task_ids)
    proposer = _CodeProposer()
    persisted: list[SearchCheckpoint] = []
    cancelled = False

    def _persist_and_cancel_final(checkpoint: SearchCheckpoint) -> None:
        nonlocal cancelled
        persisted.append(checkpoint)
        if checkpoint.completed_iteration == protocol.search.iterations:
            cancelled = True

    first = run_harness_optimization_search_slice(
        prepared.discovery_contract(),
        scorer=scorer,
        proposer=proposer,
        lifecycle=lifecycle,
        authorization=authorization,
        on_checkpoint=_persist_and_cancel_final,
        should_cancel=lambda: cancelled,
    )
    completed = run_harness_optimization_search_slice(
        prepared.discovery_contract(),
        scorer=scorer,
        proposer=proposer,
        lifecycle=lifecycle,
        authorization=authorization,
        resume_from=first.checkpoint,
        on_checkpoint=_persist_and_cancel_final,
        on_proposal_batch_prepare=lambda _witness: None,
        on_proposal_batch_witness=lambda _witness: None,
        should_cancel=lambda: cancelled,
    )

    assert cancelled
    assert completed.checkpoint == persisted[-1]
    assert completed.result is not None
    assert scorer.score_calls == 2
    assert proposer.proposal_calls == 1


def test_search_slice_rejects_fresh_replay_without_new_paid_work(tmp_path: Path) -> None:
    _control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    prepared, protocol = _prepare(tmp_path, manifest, baseline)
    lifecycle, authorization = _discovery_lifecycle(tmp_path, protocol)
    scorer = _Scorer(manifest.discovery_task_ids)
    first = run_harness_optimization_search_slice(
        prepared.discovery_contract(),
        scorer=scorer,
        proposer=_CodeProposer(),
        lifecycle=lifecycle,
        authorization=authorization,
        on_checkpoint=lambda _checkpoint: None,
    )
    assert first.checkpoint.completed_iteration == 0
    paid_calls = scorer.score_calls

    with pytest.raises(ValueError, match="already started"):
        run_harness_optimization_search_slice(
            prepared.discovery_contract(),
            scorer=scorer,
            proposer=_CodeProposer(),
            lifecycle=lifecycle,
            authorization=authorization,
            on_checkpoint=lambda _checkpoint: None,
        )

    assert scorer.score_calls == paid_calls


def test_search_slice_requires_durable_proposal_witnesses_before_paid_work(
    tmp_path: Path,
) -> None:
    _control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    prepared, protocol = _prepare(tmp_path, manifest, baseline)
    lifecycle, authorization = _discovery_lifecycle(tmp_path, protocol)
    scorer = _Scorer(manifest.discovery_task_ids)
    first = run_harness_optimization_search_slice(
        prepared.discovery_contract(),
        scorer=scorer,
        proposer=_CodeProposer(),
        lifecycle=lifecycle,
        authorization=authorization,
        on_checkpoint=lambda _checkpoint: None,
    )
    paid_calls = scorer.score_calls

    with pytest.raises(ValueError, match="durable proposal prepare and witness"):
        run_harness_optimization_search_slice(
            prepared.discovery_contract(),
            scorer=scorer,
            proposer=_CodeProposer(),
            lifecycle=lifecycle,
            authorization=authorization,
            resume_from=first.checkpoint,
            on_checkpoint=lambda _checkpoint: None,
        )

    assert scorer.score_calls == paid_calls


def test_search_slice_rejects_stale_checkpoint_replay_without_paid_work(
    tmp_path: Path,
) -> None:
    _control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    prepared, protocol = _prepare(tmp_path, manifest, baseline)
    lifecycle, authorization = _discovery_lifecycle(tmp_path, protocol)
    scorer = _Scorer(manifest.discovery_task_ids)
    first = run_harness_optimization_search_slice(
        prepared.discovery_contract(),
        scorer=scorer,
        proposer=_CodeProposer(),
        lifecycle=lifecycle,
        authorization=authorization,
        on_checkpoint=lambda _checkpoint: None,
    )
    second = run_harness_optimization_search_slice(
        prepared.discovery_contract(),
        scorer=scorer,
        proposer=_CodeProposer(),
        lifecycle=lifecycle,
        authorization=authorization,
        resume_from=first.checkpoint,
        on_checkpoint=lambda _checkpoint: None,
        on_proposal_batch_prepare=lambda _witness: None,
        on_proposal_batch_witness=lambda _witness: None,
    )
    assert second.result is not None
    paid_calls = scorer.score_calls

    with pytest.raises(ValueError, match="latest durable checkpoint"):
        run_harness_optimization_search_slice(
            prepared.discovery_contract(),
            scorer=scorer,
            proposer=_CodeProposer(),
            lifecycle=lifecycle,
            authorization=authorization,
            resume_from=first.checkpoint,
            on_checkpoint=lambda _checkpoint: None,
            on_proposal_batch_prepare=lambda _witness: None,
            on_proposal_batch_witness=lambda _witness: None,
        )

    assert scorer.score_calls == paid_calls

    repeated = run_harness_optimization_search_slice(
        prepared.discovery_contract(),
        scorer=scorer,
        proposer=_CodeProposer(),
        lifecycle=lifecycle,
        authorization=authorization,
        resume_from=second.checkpoint,
        on_checkpoint=lambda _checkpoint: None,
    )

    assert repeated.result == second.result
    assert scorer.score_calls == paid_calls


def test_search_slice_never_returns_without_a_new_durable_checkpoint(
    tmp_path: Path,
) -> None:
    _control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    prepared, protocol = _prepare(tmp_path, manifest, baseline)
    lifecycle, authorization = _discovery_lifecycle(tmp_path, protocol)
    scorer = _Scorer(manifest.discovery_task_ids)
    persisted: list[SearchCheckpoint] = []

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        run_harness_optimization_search_slice(
            prepared.discovery_contract(),
            scorer=scorer,
            proposer=_CodeProposer(),
            lifecycle=lifecycle,
            authorization=authorization,
            on_checkpoint=persisted.append,
            should_cancel=lambda: True,
        )

    assert persisted == []
    assert scorer.score_calls == 0


def test_heldout_open_rejects_a_candidate_publication_for_different_source(
    tmp_path: Path,
) -> None:
    control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    prepared, protocol = _prepare(tmp_path, manifest, baseline)
    lifecycle, authorization = _discovery_lifecycle(tmp_path, protocol)
    checkpoints: list[SearchCheckpoint] = []
    run_harness_optimization_search(
        prepared.discovery_contract(),
        scorer=_Scorer(manifest.discovery_task_ids),
        proposer=_CodeProposer(),
        lifecycle=lifecycle,
        authorization=authorization,
        on_checkpoint=checkpoints.append,
        on_proposal_batch_prepare=lambda _witness: None,
        on_proposal_batch_witness=lambda _witness: None,
    )
    frozen = freeze_harness_optimization_candidate(
        control_store,
        prepared=prepared,
        checkpoint=checkpoints[-1],
        lifecycle=lifecycle,
        authorization=authorization,
    )
    budget_authority = _budget_authority(prepared)
    search_cost_report = StudyBudgetReport.capture(budget_authority)
    cost_publication = _artifact_publication(search_cost_report.digest)
    lifecycle.publish_candidate_frozen(
        protocol_digest=protocol.digest,
        candidate=frozen.candidate,
        checkpoint=checkpoints[-1],
        search_configuration_digest=authorization.search_configuration_digest,
        search_cost_binding_digest=protocol.search_cost_binding_digest,
        budget_authority=budget_authority,
        search_cost_report=search_cost_report,
        search_cost_report_publication=cost_publication,
        freeze_record=frozen.freeze_record,
        completed_iterations=protocol.search.iterations,
    )

    wrong_source = default_agent("different-source")
    wrong_digest = _canonical_digest(wrong_source.model_dump(mode="json"))
    with pytest.raises(ValueError, match="exact candidate source"):
        lifecycle.publish_candidate_source(
            protocol_digest=protocol.digest,
            candidate=frozen.candidate,
            publication=_artifact_publication(wrong_digest),
        )

    assert lifecycle.current_phase is StudyPhase.CANDIDATE_FROZEN
    assert set(manifest.confirmation_task_ids).isdisjoint(
        prepared.discovery_contract().protocol.discovery.tasks[index].task_id
        for index in range(len(prepared.discovery_contract().protocol.discovery.tasks))
    )


def test_freeze_rejects_a_prompt_only_champion_when_code_change_is_required(
    tmp_path: Path,
) -> None:
    control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    scorer = _Scorer(manifest.discovery_task_ids)
    proposer = _PromptProposer()
    prepared, protocol = _prepare(
        tmp_path,
        manifest,
        baseline,
        proposer_configuration_id=proposer.configuration_id,
    )
    lifecycle, discovery_authorization = _discovery_lifecycle(tmp_path, protocol)
    checkpoints: list[SearchCheckpoint] = []
    run_harness_optimization_search(
        prepared.discovery_contract(),
        scorer=scorer,
        proposer=proposer,
        lifecycle=lifecycle,
        authorization=discovery_authorization,
        on_checkpoint=checkpoints.append,
        on_proposal_batch_prepare=lambda _witness: None,
        on_proposal_batch_witness=lambda _witness: None,
    )

    with pytest.raises(ValueError, match="code surface"):
        freeze_harness_optimization_candidate(
            control_store,
            prepared=prepared,
            checkpoint=checkpoints[-1],
            lifecycle=lifecycle,
            authorization=discovery_authorization,
        )


def test_search_rejects_runtime_component_drift_before_scoring(tmp_path: Path) -> None:
    _control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    prepared, protocol = _prepare(tmp_path, manifest, baseline)
    lifecycle, discovery_authorization = _discovery_lifecycle(tmp_path, protocol)
    scorer = _Scorer(tuple(reversed(manifest.discovery_task_ids)))

    with pytest.raises(ValueError, match="task matrix"):
        run_harness_optimization_search(
            prepared.discovery_contract(),
            scorer=scorer,
            proposer=_CodeProposer(),
            lifecycle=lifecycle,
            authorization=discovery_authorization,
            on_checkpoint=lambda _checkpoint: None,
            on_proposal_batch_prepare=lambda _witness: None,
            on_proposal_batch_witness=lambda _witness: None,
        )


def test_search_phase_guard_rejects_before_the_scorer_or_proposer_can_run(
    tmp_path: Path,
) -> None:
    _control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    prepared, protocol = _prepare(tmp_path, manifest, baseline)
    lifecycle, discovery_authorization = _discovery_lifecycle(
        tmp_path,
        protocol,
        start_discovery=False,
    )
    scorer = _Scorer(manifest.discovery_task_ids)

    with pytest.raises(ValueError, match="current study phase"):
        run_harness_optimization_search(
            prepared.discovery_contract(),
            scorer=scorer,
            proposer=_CodeProposer(),
            lifecycle=lifecycle,
            authorization=discovery_authorization,
            on_checkpoint=lambda _checkpoint: None,
        )

    assert scorer.score_calls == 0


def test_discovery_authorization_cannot_start_a_second_fresh_search(
    tmp_path: Path,
) -> None:
    _control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    prepared, protocol = _prepare(tmp_path, manifest, baseline)
    lifecycle, discovery_authorization = _discovery_lifecycle(tmp_path, protocol)
    scorer = _Scorer(manifest.discovery_task_ids)
    checkpoints: list[SearchCheckpoint] = []

    run_harness_optimization_search(
        prepared.discovery_contract(),
        scorer=scorer,
        proposer=_CodeProposer(),
        lifecycle=lifecycle,
        authorization=discovery_authorization,
        on_checkpoint=checkpoints.append,
        on_proposal_batch_prepare=lambda _witness: None,
        on_proposal_batch_witness=lambda _witness: None,
    )
    first_call_count = scorer.score_calls

    with pytest.raises(ValueError, match="already started"):
        run_harness_optimization_search(
            prepared.discovery_contract(),
            scorer=scorer,
            proposer=_CodeProposer(),
            lifecycle=lifecycle,
            authorization=discovery_authorization,
            on_checkpoint=lambda _checkpoint: None,
            on_proposal_batch_prepare=lambda _witness: None,
            on_proposal_batch_witness=lambda _witness: None,
        )

    assert scorer.score_calls == first_call_count


def test_protocol_rejects_a_caller_asserted_roster_digest(tmp_path: Path) -> None:
    _control_store, manifest = _partition(tmp_path)
    baseline = default_agent("baseline")
    with pytest.raises(ValueError, match="roster_digest"):
        _protocol(
            tmp_path,
            manifest,
            baseline,
            roster_digest=_digest("caller-asserted-roster"),
        )


def test_compact_outcome_requires_every_predeclared_lane_to_pass() -> None:
    members = tuple(
        HarnessOptimizationMemberOutcome(
            panel_member=member,
            equal_task_delta=0.04,
            primary_lower_bound=0.001,
            minimum_required_delta=0.03,
            passed=True,
        )
        for member in ("glm", "haiku", "opus")
    )
    outcome = HarnessOptimizationOutcome(
        protocol_digest=_digest("protocol"),
        paired_protocol_digest=_digest("paired-protocol"),
        paired_report_digest=_digest("paired-report"),
        baseline_execution_digest=_digest("baseline"),
        candidate_execution_digest=_digest("candidate"),
        equal_task_panel_delta=0.04,
        members=members,
        passed=True,
    )
    assert outcome.passed

    with pytest.raises(ValueError, match="frozen decisions"):
        HarnessOptimizationOutcome.model_validate(
            {**outcome.model_dump(mode="json"), "passed": False}
        )

    member_failure = HarnessOptimizationOutcome.model_validate(
        {
            **outcome.model_dump(mode="json"),
            "members": [
                {
                    **members[0].model_dump(mode="json"),
                    "equal_task_delta": 0.02,
                    "passed": False,
                },
                *(item.model_dump(mode="json") for item in members[1:]),
            ],
            "passed": False,
        }
    )
    assert not member_failure.passed

    with pytest.raises(ValueError, match="frozen decisions"):
        HarnessOptimizationOutcome.model_validate(
            {**member_failure.model_dump(mode="json"), "passed": True}
        )
