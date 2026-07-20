"""Tests for typed, phase-guarded optimization study lifecycle control."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from wmh.evals.study_journal import (
    ExternalPublicationReceipt,
    StudyJournalGenesis,
    StudyJournalStore,
    StudyPhase,
    StudyPhaseCommitment,
    StudyPhaseRecord,
    StudyRunCheckpointIdentity,
)
from wmh.evals.study_lifecycle import (
    CandidateFrozenPayload,
    DiscoveryRunningPayload,
    PreparationPlannedPayload,
    ProtocolPublishedPayload,
    RosterQualifiedPayload,
    StoppedPayload,
    StudyArtifactPublication,
    StudyBudgetReport,
    StudyLifecycleController,
    StudySliceResult,
    StudyStopReason,
)
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.tracking._testing import (
    synthetic_provider_cost_meter,
    synthetic_tariff_provenance,
)
from wmh.tracking.budget import (
    BudgetLedgerAuthority,
    BudgetPolicy,
    BudgetScope,
    bootstrap_budget_ledger,
    open_shared_spend_ledger,
)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


class _Publisher:
    configuration_digest = _digest("lifecycle-publisher")

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
                evidence={"sequence": commitment.sequence},
            )
            self.receipts[commitment.digest] = receipt
        return receipt

    def verify(
        self,
        commitment: StudyPhaseCommitment,
        receipt: ExternalPublicationReceipt,
    ) -> None:
        if self.receipts.get(commitment.digest) != receipt:
            raise ValueError("missing external receipt")

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
            raise ValueError("missing external artifact")


def _artifact_publication(artifact_digest: str) -> StudyArtifactPublication:
    return StudyArtifactPublication.create(
        artifact_digest=artifact_digest,
        publisher="test-artifacts",
        publication_id=f"artifact-{artifact_digest}",
        immutable_locator=f"test://artifacts/{artifact_digest}",
        published_at=datetime(2026, 7, 19, 11, 0, tzinfo=UTC),
        evidence={},
    )


def _controller(tmp_path: Path) -> StudyLifecycleController:
    publisher = _Publisher()
    store = StudyJournalStore.create(
        tmp_path / "journal",
        study_id="study-1",
        publisher_configuration_digest=publisher.configuration_digest,
    )
    return StudyLifecycleController(
        store=store,
        publisher=publisher,
        artifact_verifier=publisher,
    )


def _preparation() -> PreparationPlannedPayload:
    return PreparationPlannedPayload(
        study_plan_digest=_digest("plan"),
        budget_policy_digest=_digest("budget-policy"),
        budget_binding_digest=_digest("budget-binding"),
        budget_ledger_identity=_digest("budget-ledger"),
        maximum_paid_cost_nano_usd=15_000_000_000_000,
    )


def _budget_authority(tmp_path: Path) -> BudgetLedgerAuthority:
    provider_config = ProviderConfig(
        kind=ProviderKind.BEDROCK,
        model="test-model",
        region="us-east-1",
    )
    policy = BudgetPolicy(
        study_id="study-1",
        manifest_digest=_digest("study-manifest"),
        hard_limit_nano_usd=15_000_000_000_000,
        phase_limits_nano_usd={"discovery": 5_000_000_000_000},
        meters={
            "worker": synthetic_provider_cost_meter(
                provider_config=provider_config,
                provenance=synthetic_tariff_provenance(provider_config),
                input_nano_usd_per_token=1,
                output_nano_usd_per_token=5,
            )
        },
    )
    return bootstrap_budget_ledger((tmp_path / "study-budget.sqlite3").resolve(), policy)


def test_typed_payload_digest_is_canonical_and_phase_bound() -> None:
    payload = _preparation()

    assert payload.phase is StudyPhase.PREPARATION_PLANNED
    assert payload.digest.startswith("sha256:")
    assert (
        payload.digest
        == PreparationPlannedPayload.model_validate(payload.model_dump(mode="json")).digest
    )
    with pytest.raises(ValueError, match="maximum_paid_cost_nano_usd"):
        PreparationPlannedPayload(
            study_plan_digest=_digest("plan"),
            budget_policy_digest=_digest("budget-policy"),
            budget_binding_digest=_digest("budget-binding"),
            budget_ledger_identity=_digest("budget-ledger"),
            maximum_paid_cost_nano_usd=True,
        )


def test_controller_rejects_wrong_phase_before_invoking_any_side_effect(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    calls: list[str] = []

    with pytest.raises(ValueError, match="current study phase"):
        controller.call_in_phase(
            StudyPhase.DISCOVERY_RUNNING,
            lambda: calls.append("provider-call"),
        )
    assert calls == []

    controller.publish(_preparation())
    with pytest.raises(ValueError, match="current study phase"):
        controller.call_in_phase(
            StudyPhase.DISCOVERY_RUNNING,
            lambda: calls.append("provider-call"),
        )
    assert calls == []


def test_controller_publishes_typed_chronology_and_guards_the_active_phase(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    preparation = _preparation()
    roster = RosterQualifiedPayload(
        qualified_roster_digest=_digest("roster"),
        qualification_report_digest=_digest("qualification"),
        execution_plan_digest=_digest("execution-plan"),
        qualified_task_count=89,
    )
    protocol = ProtocolPublishedPayload(
        protocol_digest=_digest("protocol"),
        protocol_artifact_publication_digest=_digest("protocol-publication"),
        partition_manifest_digest=_digest("partition"),
        qualified_roster_digest=roster.qualified_roster_digest,
        search_cost_binding_digest=_digest("search-cost-binding"),
        confirmation_budget_binding_digest=_digest("confirmation-budget-binding"),
    )
    discovery = DiscoveryRunningPayload(
        protocol_digest=protocol.protocol_digest,
        search_configuration_digest=_digest("search-configuration"),
        search_run_id="discovery-run-1",
    )

    for payload in (preparation, roster):
        record = controller.publish(payload)
        assert record.commitment.phase is payload.phase
        assert record.commitment.payload_digest == payload.digest
    protocol = protocol.model_copy(
        update={
            "protocol_artifact_publication_digest": _artifact_publication(
                protocol.protocol_digest
            ).digest
        }
    )
    record = controller.publish_protocol(
        protocol,
        publication=_artifact_publication(protocol.protocol_digest),
    )
    assert record.commitment.payload_digest == protocol.digest
    record = controller.publish(discovery)
    assert record.commitment.payload_digest == discovery.digest

    calls: list[str] = []
    result = controller.call_in_phase(
        StudyPhase.DISCOVERY_RUNNING,
        lambda: calls.append("provider-call") or "completed",
    )
    assert result == "completed"
    assert calls == ["provider-call"]


def test_run_claim_rejects_a_second_fresh_start_but_allows_exact_resume(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    preparation = _preparation()
    roster = RosterQualifiedPayload(
        qualified_roster_digest=_digest("roster"),
        qualification_report_digest=_digest("qualification"),
        execution_plan_digest=_digest("execution-plan"),
        qualified_task_count=89,
    )
    protocol = ProtocolPublishedPayload(
        protocol_digest=_digest("protocol"),
        protocol_artifact_publication_digest=_digest("protocol-publication"),
        partition_manifest_digest=_digest("partition"),
        qualified_roster_digest=roster.qualified_roster_digest,
        search_cost_binding_digest=_digest("search-cost-binding"),
        confirmation_budget_binding_digest=_digest("confirmation-budget-binding"),
    )
    discovery = DiscoveryRunningPayload(
        protocol_digest=protocol.protocol_digest,
        search_configuration_digest=_digest("search-configuration"),
        search_run_id="discovery-run-1",
    )
    controller.publish(preparation)
    controller.publish(roster)
    protocol = protocol.model_copy(
        update={
            "protocol_artifact_publication_digest": _artifact_publication(
                protocol.protocol_digest
            ).digest
        }
    )
    controller.publish_protocol(
        protocol,
        publication=_artifact_publication(protocol.protocol_digest),
    )
    controller.publish(discovery)

    claim = controller.claim_run(
        StudyPhase.DISCOVERY_RUNNING,
        discovery.search_run_id,
        payload_digest=discovery.digest,
        resume=False,
    )

    assert claim.run_id == discovery.search_run_id
    with pytest.raises(ValueError, match="already started"):
        controller.claim_run(
            StudyPhase.DISCOVERY_RUNNING,
            discovery.search_run_id,
            payload_digest=discovery.digest,
            resume=False,
        )
    assert (
        controller.claim_run(
            StudyPhase.DISCOVERY_RUNNING,
            discovery.search_run_id,
            payload_digest=discovery.digest,
            resume=True,
        )
        == claim
    )


def test_guard_holds_operation_lease_against_a_concurrent_terminal_transition(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    controller.publish(_preparation())
    authority = _budget_authority(tmp_path)
    report = StudyBudgetReport.capture(authority)
    transition_errors: list[RuntimeError] = []

    def operation() -> str:
        try:
            controller.stop(
                reason=StudyStopReason.OPERATOR_STOPPED,
                budget_authority=authority,
                budget_report=report,
                budget_report_publication=_artifact_publication(report.digest),
                detail="Operator requested a stop.",
            )
        except RuntimeError as error:
            transition_errors.append(error)
        return "completed"

    assert (
        controller.call_in_phase(
            StudyPhase.PREPARATION_PLANNED,
            operation,
            payload_digest=_preparation().digest,
        )
        == "completed"
    )
    assert len(transition_errors) == 1
    assert "locked" in str(transition_errors[0])
    assert controller.current_phase is StudyPhase.PREPARATION_PLANNED

    controller.stop(
        reason=StudyStopReason.OPERATOR_STOPPED,
        budget_authority=authority,
        budget_report=report,
        budget_report_publication=_artifact_publication(report.digest),
        detail="Operator requested a stop.",
    )
    assert controller.current_phase is StudyPhase.STOPPED


def test_study_slice_holds_the_lifecycle_lease_across_bounded_work(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    preparation = _preparation()
    controller.publish(preparation)
    nested_errors: list[RuntimeError] = []

    def _checkpoint(sequence: int) -> StudyRunCheckpointIdentity:
        return StudyRunCheckpointIdentity(
            sequence=sequence,
            checkpoint_digest=_digest(f"checkpoint-{sequence}"),
        )

    def _outer_operation() -> StudySliceResult[
        StudyRunCheckpointIdentity,
        StudyRunCheckpointIdentity,
    ]:
        try:
            controller.run_slice(
                StudyPhase.PREPARATION_PLANNED,
                "run-1",
                lambda: StudySliceResult[
                    StudyRunCheckpointIdentity,
                    StudyRunCheckpointIdentity,
                ](checkpoint=_checkpoint(0)),
                payload_digest=preparation.digest,
                configuration_digest=_digest("configuration"),
                resume_from=None,
                checkpoint_identity=lambda checkpoint: checkpoint,
            )
        except RuntimeError as error:
            nested_errors.append(error)
        return StudySliceResult[
            StudyRunCheckpointIdentity,
            StudyRunCheckpointIdentity,
        ](checkpoint=_checkpoint(0))

    result = controller.run_slice(
        StudyPhase.PREPARATION_PLANNED,
        "run-1",
        _outer_operation,
        payload_digest=preparation.digest,
        configuration_digest=_digest("configuration"),
        resume_from=None,
        checkpoint_identity=lambda checkpoint: checkpoint,
    )

    assert result.checkpoint == _checkpoint(0)
    assert len(nested_errors) == 1
    assert "locked" in str(nested_errors[0])


def test_completed_slice_reconstruction_must_name_the_reconciled_checkpoint(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    preparation = _preparation()
    controller.publish(preparation)
    configuration_digest = _digest("configuration")

    def _checkpoint(sequence: int) -> StudyRunCheckpointIdentity:
        return StudyRunCheckpointIdentity(
            sequence=sequence,
            checkpoint_digest=_digest(f"checkpoint-{sequence}"),
        )

    first = controller.run_slice(
        StudyPhase.PREPARATION_PLANNED,
        "run-1",
        lambda: StudySliceResult[
            StudyRunCheckpointIdentity,
            StudyRunCheckpointIdentity,
        ](checkpoint=_checkpoint(0)),
        payload_digest=preparation.digest,
        configuration_digest=configuration_digest,
        resume_from=None,
        checkpoint_identity=lambda checkpoint: checkpoint,
    )
    persisted = _checkpoint(1)

    with pytest.raises(RuntimeError, match="crash after persistence"):
        controller.run_slice(
            StudyPhase.PREPARATION_PLANNED,
            "run-1",
            lambda: (_ for _ in ()).throw(RuntimeError("crash after persistence")),
            payload_digest=preparation.digest,
            configuration_digest=configuration_digest,
            resume_from=first.checkpoint,
            checkpoint_identity=lambda checkpoint: checkpoint,
        )

    with pytest.raises(ValueError, match="reconstructed checkpoint identity"):
        controller.reconcile_slice(
            StudyPhase.PREPARATION_PLANNED,
            "run-1",
            lambda: StudySliceResult[
                StudyRunCheckpointIdentity,
                StudyRunCheckpointIdentity,
            ](checkpoint=_checkpoint(2)),
            payload_digest=preparation.digest,
            configuration_digest=configuration_digest,
            resume_from=persisted,
            checkpoint_identity=lambda checkpoint: checkpoint,
        )


def test_candidate_freeze_payload_binds_completed_search_and_cost_evidence() -> None:
    payload = CandidateFrozenPayload(
        protocol_digest=_digest("protocol"),
        candidate_execution_digest=_digest("candidate"),
        search_checkpoint_digest=_digest("checkpoint"),
        search_configuration_digest=_digest("search-configuration"),
        search_cost_binding_digest=_digest("search-cost-binding"),
        search_cost_report_digest=_digest("search-cost-report"),
        search_cost_report_publication_digest=_digest("search-cost-report-publication"),
        champion_reconstruction_digest=_digest("champion-reconstruction"),
        candidate_freeze_record_digest=_digest("freeze-record"),
        completed_iterations=10,
    )

    assert payload.phase is StudyPhase.CANDIDATE_FROZEN
    with pytest.raises(ValueError, match="completed_iterations"):
        CandidateFrozenPayload.model_validate(
            {**payload.model_dump(mode="json"), "completed_iterations": 0}
        )


def test_protected_evidence_phases_reject_unverified_generic_publication(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    controller.publish(_preparation())
    controller.publish(
        RosterQualifiedPayload(
            qualified_roster_digest=_digest("roster"),
            qualification_report_digest=_digest("qualification"),
            execution_plan_digest=_digest("execution-plan"),
            qualified_task_count=89,
        )
    )
    protocol = ProtocolPublishedPayload(
        protocol_digest=_digest("protocol"),
        protocol_artifact_publication_digest=_digest("invented-publication"),
        partition_manifest_digest=_digest("partition"),
        qualified_roster_digest=_digest("roster"),
        search_cost_binding_digest=_digest("search-cost-binding"),
        confirmation_budget_binding_digest=_digest("confirmation-budget-binding"),
    )

    with pytest.raises(ValueError, match="publish_protocol"):
        controller.publish(protocol)
    with pytest.raises(ValueError, match="exact protocol"):
        controller.publish_protocol(
            protocol,
            publication=_artifact_publication(_digest("different-protocol")),
        )


def test_stop_derives_exact_terminal_accounting_from_the_live_ledger(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.publish(_preparation())
    authority = _budget_authority(tmp_path)
    report = StudyBudgetReport.capture(authority)
    stopped = controller.stop(
        reason=StudyStopReason.BUDGET_EXHAUSTED,
        budget_authority=authority,
        budget_report=report,
        budget_report_publication=_artifact_publication(report.digest),
        blocked_request_nano_usd=15_000_000_000_001,
        detail="Hard paid budget ceiling reached.",
    )

    assert controller.current_phase is StudyPhase.STOPPED
    assert stopped.last_evidence_digest == controller.records[-2].digest
    assert stopped.budget_policy_digest == authority.policy.policy_digest
    assert stopped.budget_ledger_identity == authority.ledger_identity
    assert stopped.budget_report_digest == report.digest
    assert stopped.cumulative_paid_cost_nano_usd == 0
    assert stopped.blocked_request_nano_usd == 15_000_000_000_001
    with pytest.raises(ValueError, match="terminal"):
        controller.publish(
            RosterQualifiedPayload(
                qualified_roster_digest=_digest("roster"),
                qualification_report_digest=_digest("qualification"),
                execution_plan_digest=_digest("execution-plan"),
                qualified_task_count=89,
            )
        )


def test_stop_rejects_caller_asserted_accounting_and_false_budget_exhaustion(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    controller.publish(_preparation())
    authority = _budget_authority(tmp_path)
    report = StudyBudgetReport.capture(authority)
    fabricated = StoppedPayload(
        reason=StudyStopReason.BUDGET_EXHAUSTED,
        last_evidence_digest=_digest("fabricated-last-evidence"),
        budget_policy_digest=authority.policy.policy_digest,
        budget_ledger_identity=authority.ledger_identity,
        ledger_head_sequence=report.ledger_head_sequence,
        ledger_head_digest=report.ledger_head_digest,
        budget_report_digest=_digest("fabricated-ledger-report"),
        budget_report_publication_digest=_digest("fabricated-publication"),
        cumulative_paid_cost_nano_usd=99_000_000_000_000,
        outstanding_reserved_cost_nano_usd=0,
        budget_hard_limit_nano_usd=15_000_000_000_000,
        budget_remaining_nano_usd=0,
        budget_breached=True,
        blocked_request_nano_usd=1,
        detail="Fabricated accounting.",
    )

    with pytest.raises(ValueError, match="stop"):
        controller.publish(fabricated)
    with pytest.raises(ValueError, match="does not exhaust"):
        controller.stop(
            reason=StudyStopReason.BUDGET_EXHAUSTED,
            budget_authority=authority,
            budget_report=report,
            budget_report_publication=_artifact_publication(report.digest),
            blocked_request_nano_usd=1,
            detail="False exhaustion.",
        )
    assert controller.current_phase is StudyPhase.PREPARATION_PLANNED


def test_stop_rejects_a_report_from_an_older_live_ledger_head(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.publish(_preparation())
    authority = _budget_authority(tmp_path)
    stale = StudyBudgetReport.capture(authority)
    ledger = open_shared_spend_ledger(
        authority.ledger_path,
        authority.policy,
        expected_ledger_identity=authority.ledger_identity,
    )
    ledger.reserve(
        BudgetScope(phase="discovery", category="worker", run_id="run-1"),
        meter_id="worker",
        max_nano_usd=1,
        reservation_id="newer-reservation",
    )

    with pytest.raises(ValueError, match="exact current live-ledger state"):
        controller.stop(
            reason=StudyStopReason.OPERATOR_STOPPED,
            budget_authority=authority,
            budget_report=stale,
            budget_report_publication=_artifact_publication(stale.digest),
            detail="Operator requested a stop.",
        )

    assert controller.current_phase is StudyPhase.PREPARATION_PLANNED
