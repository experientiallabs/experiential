"""Typed phase payloads and side-effect guards for optimization studies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Generic, Literal, Protocol, TypedDict, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    TypeAdapter,
    field_validator,
    model_validator,
)

from wmh.core.text import validate_durable_text
from wmh.evals.partition import CandidateFreezeRecord
from wmh.evals.study_journal import (
    ExternalCommitmentPublisher,
    StudyJournalStore,
    StudyPhase,
    StudyPhaseRecord,
    StudyRunCheckpointIdentity,
    StudyRunClaim,
    append_study_phase,
    append_study_phase_derived,
    call_in_resumable_study_slice,
    call_in_study_phase,
    call_in_study_slice,
    claim_study_run,
    load_study_journal,
    reconcile_study_slice,
)
from wmh.harness.create import SearchCheckpoint
from wmh.harness.doc import HarnessDoc
from wmh.tracking.budget import (
    BudgetAuditState,
    BudgetLedgerAuthority,
    open_shared_spend_ledger,
)

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_ResultT = TypeVar("_ResultT")
_CheckpointT = TypeVar("_CheckpointT", bound=BaseModel)
_CompletedResultT = TypeVar("_CompletedResultT", bound=BaseModel)


class _TerminalBudgetFields(TypedDict):
    budget_policy_digest: str
    budget_ledger_identity: str
    ledger_head_sequence: int
    ledger_head_digest: str
    budget_report_digest: str
    budget_report_publication_digest: str
    cumulative_paid_cost_nano_usd: int
    outstanding_reserved_cost_nano_usd: int
    budget_hard_limit_nano_usd: int
    budget_remaining_nano_usd: int
    budget_breached: bool


class _StudyPayload(BaseModel):
    """Canonical content addressed evidence for one journal transition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def digest(self) -> str:
        """Return the canonical payload identity committed by the phase journal."""
        serialized = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        return "sha256:" + hashlib.sha256(serialized).hexdigest()


_StudyPayloadT = TypeVar("_StudyPayloadT", bound=_StudyPayload)


class StudySliceResult(BaseModel, Generic[_CheckpointT, _CompletedResultT]):
    """One newly durable checkpoint plus a result only when the run is complete."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    checkpoint: _CheckpointT
    result: _CompletedResultT | None = None


class StudyArtifactPublication(BaseModel):
    """Self-validating receipt for one immutable externally retrievable artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    publication_version: Literal["1"] = "1"
    artifact_digest: str = Field(pattern=_DIGEST_PATTERN)
    publisher: str = Field(min_length=1, max_length=256)
    publication_id: str = Field(min_length=1, max_length=2_048)
    immutable_locator: str = Field(min_length=1, max_length=4_096)
    published_at: datetime
    evidence: dict[str, JsonValue]
    receipt_digest: str = Field(pattern=_DIGEST_PATTERN)

    @classmethod
    def create(
        cls,
        *,
        artifact_digest: str,
        publisher: str,
        publication_id: str,
        immutable_locator: str,
        published_at: datetime,
        evidence: dict[str, JsonValue],
    ) -> StudyArtifactPublication:
        """Create a receipt whose digest covers every retrieval and evidence field."""
        draft = cls.model_construct(
            publication_version="1",
            artifact_digest=artifact_digest,
            publisher=publisher,
            publication_id=publication_id,
            immutable_locator=immutable_locator,
            published_at=published_at,
            evidence=evidence,
            receipt_digest="sha256:" + "0" * 64,
        )
        payload = draft.model_dump(mode="json", exclude={"receipt_digest"})
        return cls(**payload, receipt_digest=_canonical_digest(payload))

    @model_validator(mode="after")
    def _validate_receipt(self) -> StudyArtifactPublication:
        for field in ("publisher", "publication_id", "immutable_locator"):
            value = getattr(self, field)
            if value != value.strip():
                raise ValueError(f"artifact publication {field} cannot have surrounding whitespace")
            validate_durable_text(value, field=f"artifact publication {field}")
        if self.published_at.tzinfo is None or self.published_at.utcoffset() is None:
            raise ValueError("artifact publication timestamp must be timezone-aware")
        payload = self.model_dump(mode="json", exclude={"receipt_digest"})
        if self.receipt_digest != _canonical_digest(payload):
            raise ValueError("artifact publication receipt digest is inconsistent")
        return self

    @property
    def digest(self) -> str:
        """Return the canonical receipt identity committed by a study phase."""
        return self.receipt_digest


class ExternalArtifactVerifier(Protocol):
    """Verify an immutable artifact receipt against its external publication channel."""

    def verify_artifact(self, publication: StudyArtifactPublication) -> None:
        """Raise unless the receipt still resolves to its exact artifact digest."""
        ...


class StudyBudgetReport(BaseModel):
    """Canonical public report captured from one exact audited budget-ledger head."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report_version: Literal["1"] = "1"
    audit_state: BudgetAuditState

    @classmethod
    def capture(cls, authority: BudgetLedgerAuthority) -> StudyBudgetReport:
        """Audit and capture the exact ledger named by a pre-existing authority."""
        validated = BudgetLedgerAuthority.model_validate(authority.model_dump(mode="python"))
        state = open_shared_spend_ledger(
            validated.ledger_path,
            validated.policy,
            expected_ledger_identity=validated.ledger_identity,
        ).audit_state()
        if state.policy != validated.policy or state.ledger_identity != validated.ledger_identity:
            raise ValueError("budget report differs from its ledger authority")
        return cls(audit_state=state)

    @property
    def digest(self) -> str:
        """Return the exact identity of the audited report artifact."""
        return _canonical_digest(self.model_dump(mode="json"))

    @property
    def ledger_head_sequence(self) -> int:
        """Return the captured append-only ledger sequence."""
        return self.audit_state.ledger_head_sequence

    @property
    def ledger_head_digest(self) -> str:
        """Return the captured append-only ledger digest."""
        return self.audit_state.ledger_head_digest

    @property
    def cumulative_paid_cost_nano_usd(self) -> int:
        """Return exact settled exposure in nano-USD."""
        return self.audit_state.snapshot.charged_nano_usd

    @property
    def outstanding_reserved_cost_nano_usd(self) -> int:
        """Return exact unsettled conservative exposure in nano-USD."""
        return self.audit_state.snapshot.reserved_nano_usd


class PreparationPlannedPayload(_StudyPayload):
    """Budget and immutable plan identity accepted before any preparation side effect."""

    phase: Literal[StudyPhase.PREPARATION_PLANNED] = StudyPhase.PREPARATION_PLANNED
    study_plan_digest: str = Field(pattern=_DIGEST_PATTERN)
    budget_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    budget_binding_digest: str = Field(pattern=_DIGEST_PATTERN)
    budget_ledger_identity: str = Field(pattern=_DIGEST_PATTERN)
    maximum_paid_cost_nano_usd: StrictInt = Field(ge=0)


class RosterQualifiedPayload(_StudyPayload):
    """Full-roster qualification evidence produced before partition disclosure."""

    phase: Literal[StudyPhase.ROSTER_QUALIFIED] = StudyPhase.ROSTER_QUALIFIED
    qualified_roster_digest: str = Field(pattern=_DIGEST_PATTERN)
    qualification_report_digest: str = Field(pattern=_DIGEST_PATTERN)
    execution_plan_digest: str = Field(pattern=_DIGEST_PATTERN)
    qualified_task_count: StrictInt = Field(ge=1)


class ProtocolPublishedPayload(_StudyPayload):
    """Candidate-free preregistration and its externally retrievable artifact proof."""

    phase: Literal[StudyPhase.PROTOCOL_PUBLISHED] = StudyPhase.PROTOCOL_PUBLISHED
    protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    protocol_artifact_publication_digest: str = Field(pattern=_DIGEST_PATTERN)
    partition_manifest_digest: str = Field(pattern=_DIGEST_PATTERN)
    qualified_roster_digest: str = Field(pattern=_DIGEST_PATTERN)
    search_cost_binding_digest: str = Field(pattern=_DIGEST_PATTERN)
    confirmation_budget_binding_digest: str = Field(pattern=_DIGEST_PATTERN)


class DiscoveryRunningPayload(_StudyPayload):
    """Exact discovery configuration admitted to make proposer and worker calls."""

    phase: Literal[StudyPhase.DISCOVERY_RUNNING] = StudyPhase.DISCOVERY_RUNNING
    protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    search_configuration_digest: str = Field(pattern=_DIGEST_PATTERN)
    search_run_id: str = Field(min_length=1, max_length=512)

    @field_validator("search_run_id")
    @classmethod
    def _validate_search_run_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("search_run_id cannot have surrounding whitespace")
        validate_durable_text(value, field="search run id")
        return value


class CandidateFrozenPayload(_StudyPayload):
    """Selection proof tying one candidate to a completed and metered search."""

    phase: Literal[StudyPhase.CANDIDATE_FROZEN] = StudyPhase.CANDIDATE_FROZEN
    protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    candidate_execution_digest: str = Field(pattern=_DIGEST_PATTERN)
    search_checkpoint_digest: str = Field(pattern=_DIGEST_PATTERN)
    search_configuration_digest: str = Field(pattern=_DIGEST_PATTERN)
    search_cost_binding_digest: str = Field(pattern=_DIGEST_PATTERN)
    search_cost_report_digest: str = Field(pattern=_DIGEST_PATTERN)
    search_cost_report_publication_digest: str = Field(pattern=_DIGEST_PATTERN)
    champion_reconstruction_digest: str = Field(pattern=_DIGEST_PATTERN)
    candidate_freeze_record_digest: str = Field(pattern=_DIGEST_PATTERN)
    completed_iterations: StrictInt = Field(ge=1)


class CandidatePublishedPayload(_StudyPayload):
    """Public source artifact for the selected candidate before held-out opening."""

    phase: Literal[StudyPhase.CANDIDATE_PUBLISHED] = StudyPhase.CANDIDATE_PUBLISHED
    protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    candidate_execution_digest: str = Field(pattern=_DIGEST_PATTERN)
    candidate_source_artifact_digest: str = Field(pattern=_DIGEST_PATTERN)
    candidate_artifact_publication_digest: str = Field(pattern=_DIGEST_PATTERN)


class ConfirmationOpenedPayload(_StudyPayload):
    """One-shot held-out opening and the deterministic design derived from it."""

    phase: Literal[StudyPhase.CONFIRMATION_OPENED] = StudyPhase.CONFIRMATION_OPENED
    protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    candidate_execution_digest: str = Field(pattern=_DIGEST_PATTERN)
    candidate_freeze_record_digest: str = Field(pattern=_DIGEST_PATTERN)
    confirmation_partition_digest: str = Field(pattern=_DIGEST_PATTERN)
    confirmation_opening_record_digest: str = Field(pattern=_DIGEST_PATTERN)
    paired_design_digest: str = Field(pattern=_DIGEST_PATTERN)
    confirmation_task_count: StrictInt = Field(ge=1)


class ConfirmationFrozenPayload(_StudyPayload):
    """Exact paired execution protocol frozen before confirmation can launch."""

    phase: Literal[StudyPhase.CONFIRMATION_FROZEN] = StudyPhase.CONFIRMATION_FROZEN
    protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    paired_protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    budget_binding_digest: str = Field(pattern=_DIGEST_PATTERN)
    create_rate_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    slice_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    planned_blocks: StrictInt = Field(ge=1)
    planned_arms: StrictInt = Field(ge=2)


class ConfirmationRunningPayload(_StudyPayload):
    """Run identity and initial durable state admitted to launch paired blocks."""

    phase: Literal[StudyPhase.CONFIRMATION_RUNNING] = StudyPhase.CONFIRMATION_RUNNING
    paired_protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    initial_run_state_digest: str = Field(pattern=_DIGEST_PATTERN)
    slice_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    confirmation_run_id: str = Field(min_length=1, max_length=512)

    @field_validator("confirmation_run_id")
    @classmethod
    def _validate_confirmation_run_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("confirmation_run_id cannot have surrounding whitespace")
        validate_durable_text(value, field="confirmation run id")
        return value


class _BudgetTerminalPayload(_StudyPayload):
    """Exact live-ledger evidence shared by every terminal study record."""

    budget_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    budget_ledger_identity: str = Field(pattern=_DIGEST_PATTERN)
    ledger_head_sequence: StrictInt = Field(ge=1)
    ledger_head_digest: str = Field(pattern=_DIGEST_PATTERN)
    budget_report_digest: str = Field(pattern=_DIGEST_PATTERN)
    budget_report_publication_digest: str = Field(pattern=_DIGEST_PATTERN)
    cumulative_paid_cost_nano_usd: StrictInt = Field(ge=0)
    outstanding_reserved_cost_nano_usd: StrictInt = Field(ge=0)
    budget_hard_limit_nano_usd: StrictInt = Field(gt=0)
    budget_remaining_nano_usd: StrictInt = Field(ge=0)
    budget_breached: bool

    @model_validator(mode="after")
    def _validate_terminal_exposure(self) -> _BudgetTerminalPayload:
        exposure = self.cumulative_paid_cost_nano_usd + self.outstanding_reserved_cost_nano_usd
        expected_remaining = max(self.budget_hard_limit_nano_usd - exposure, 0)
        if self.budget_remaining_nano_usd != expected_remaining:
            raise ValueError("terminal budget remaining differs from exact exposure")
        return self


class StudyCompletePayload(_BudgetTerminalPayload):
    """Final outcome, complete evidence, and exact live-ledger accounting."""

    phase: Literal[StudyPhase.COMPLETE] = StudyPhase.COMPLETE
    paired_protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    paired_report_digest: str = Field(pattern=_DIGEST_PATTERN)
    outcome_digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _require_settled_budget(self) -> StudyCompletePayload:
        if self.outstanding_reserved_cost_nano_usd:
            raise ValueError("complete study cannot retain outstanding budget reservations")
        return self


class StudyStopReason(StrEnum):
    """Predeclared reasons an incomplete study can terminate without a result claim."""

    BUDGET_EXHAUSTED = "budget_exhausted"
    OPERATOR_STOPPED = "operator_stopped"
    INFRASTRUCTURE_INVALIDATED = "infrastructure_invalidated"
    PROTOCOL_INVALIDATED = "protocol_invalidated"


class StoppedPayload(_BudgetTerminalPayload):
    """Honest terminal record for a study that cannot complete its fixed protocol."""

    phase: Literal[StudyPhase.STOPPED] = StudyPhase.STOPPED
    reason: StudyStopReason
    last_evidence_digest: str = Field(pattern=_DIGEST_PATTERN)
    blocked_request_nano_usd: StrictInt | None = Field(default=None, ge=0)
    detail: str = Field(min_length=1, max_length=2_048)

    @field_validator("detail")
    @classmethod
    def _validate_detail(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("study stop detail cannot have surrounding whitespace")
        validate_durable_text(value, field="study stop detail")
        return value

    @model_validator(mode="after")
    def _validate_stop_reason(self) -> StoppedPayload:
        if self.reason is StudyStopReason.BUDGET_EXHAUSTED:
            if self.blocked_request_nano_usd is None:
                raise ValueError("budget-exhausted stop requires the blocked request cost")
            if self.blocked_request_nano_usd <= self.budget_remaining_nano_usd:
                raise ValueError("blocked request does not exhaust the remaining hard budget")
        elif self.blocked_request_nano_usd is not None:
            raise ValueError("only a budget-exhausted stop can name a blocked request cost")
        return self


StudyPhasePayload = Annotated[
    PreparationPlannedPayload
    | RosterQualifiedPayload
    | ProtocolPublishedPayload
    | DiscoveryRunningPayload
    | CandidateFrozenPayload
    | CandidatePublishedPayload
    | ConfirmationOpenedPayload
    | ConfirmationFrozenPayload
    | ConfirmationRunningPayload
    | StudyCompletePayload
    | StoppedPayload,
    Field(discriminator="phase"),
]
_PAYLOAD_ADAPTER = TypeAdapter(StudyPhasePayload)


class StudyLifecycleController:
    """Validate the external chain before any phase-scoped side effect is invoked."""

    def __init__(
        self,
        *,
        store: StudyJournalStore,
        publisher: ExternalCommitmentPublisher,
        artifact_verifier: ExternalArtifactVerifier | None = None,
    ) -> None:
        self._store = store
        self._publisher = publisher
        self._artifact_verifier = artifact_verifier

    @property
    def records(self) -> tuple[StudyPhaseRecord, ...]:
        """Return the externally reverified journal chain."""
        return load_study_journal(self._store, publisher=self._publisher)

    @property
    def current_phase(self) -> StudyPhase | None:
        """Return the last externally witnessed phase, or none before initialization."""
        records = self.records
        return records[-1].commitment.phase if records else None

    def publish(self, payload: StudyPhasePayload) -> StudyPhaseRecord:
        """Validate and externally witness one exact typed phase payload."""
        frozen = _PAYLOAD_ADAPTER.validate_python(payload.model_dump(mode="json"))
        protected = {
            StudyPhase.PROTOCOL_PUBLISHED: "publish_protocol",
            StudyPhase.CANDIDATE_FROZEN: "publish_candidate_frozen",
            StudyPhase.CANDIDATE_PUBLISHED: "publish_candidate_source",
            StudyPhase.COMPLETE: "publish_complete",
            StudyPhase.STOPPED: "stop",
        }
        required_method = protected.get(frozen.phase)
        if required_method is not None:
            raise ValueError(
                f"{frozen.phase.value} requires verified artifacts; use {required_method}"
            )
        return self._publish(frozen)

    def publish_protocol(
        self,
        payload: ProtocolPublishedPayload,
        *,
        publication: StudyArtifactPublication,
    ) -> StudyPhaseRecord:
        """Publish a preregistration only after verifying its exact public artifact."""
        frozen = ProtocolPublishedPayload.model_validate(payload.model_dump(mode="json"))
        receipt = self._verify_artifact(publication)
        if receipt.artifact_digest != frozen.protocol_digest:
            raise ValueError("protocol publication does not contain the exact protocol")
        if receipt.digest != frozen.protocol_artifact_publication_digest:
            raise ValueError("protocol payload differs from its publication receipt")
        return self._publish(frozen)

    def publish_candidate_frozen(
        self,
        *,
        protocol_digest: str,
        candidate: HarnessDoc,
        checkpoint: SearchCheckpoint,
        search_configuration_digest: str,
        search_cost_binding_digest: str,
        budget_authority: BudgetLedgerAuthority,
        search_cost_report: StudyBudgetReport,
        search_cost_report_publication: StudyArtifactPublication,
        freeze_record: CandidateFreezeRecord,
        completed_iterations: int,
    ) -> CandidateFrozenPayload:
        """Derive and publish candidate selection evidence from validated source artifacts."""
        selected = HarnessDoc.model_validate(candidate.model_dump(mode="json"))
        state = SearchCheckpoint.model_validate(checkpoint.model_dump(mode="json"))
        frozen_record = CandidateFreezeRecord.model_validate(freeze_record.model_dump(mode="json"))
        if state.completed_iteration != completed_iterations:
            raise ValueError("search checkpoint does not complete the declared iteration count")
        reconstructed = state.archive.reconstruct(state.champion_doc_hash)
        if reconstructed.execution_digest != selected.execution_digest:
            raise ValueError("candidate source differs from the reconstructed search champion")
        if (
            frozen_record.candidate_execution_digest != selected.execution_digest
            or frozen_record.selection_evidence_digest != "sha256:" + state.payload_sha256
        ):
            raise ValueError("candidate freeze record differs from the completed search")

        def _derive(_records: tuple[StudyPhaseRecord, ...]) -> CandidateFrozenPayload:
            report, cost_publication = self._verify_budget_report(
                budget_authority,
                search_cost_report,
                search_cost_report_publication,
            )
            return CandidateFrozenPayload(
                protocol_digest=protocol_digest,
                candidate_execution_digest=selected.execution_digest,
                search_checkpoint_digest="sha256:" + state.payload_sha256,
                search_configuration_digest=search_configuration_digest,
                search_cost_binding_digest=search_cost_binding_digest,
                search_cost_report_digest=report.digest,
                search_cost_report_publication_digest=cost_publication.digest,
                champion_reconstruction_digest=_canonical_digest(selected.model_dump(mode="json")),
                candidate_freeze_record_digest=frozen_record.digest,
                completed_iterations=completed_iterations,
            )

        return self._publish_derived(StudyPhase.CANDIDATE_FROZEN, _derive)

    def publish_complete(
        self,
        *,
        paired_protocol_digest: str,
        paired_report_digest: str,
        outcome_digest: str,
        budget_authority: BudgetLedgerAuthority,
        budget_report: StudyBudgetReport,
        budget_report_publication: StudyArtifactPublication,
    ) -> StudyCompletePayload:
        """Publish final evidence with accounting recaptured from the exact live ledger."""

        def _derive(_records: tuple[StudyPhaseRecord, ...]) -> StudyCompletePayload:
            report, publication = self._verify_budget_report(
                budget_authority,
                budget_report,
                budget_report_publication,
            )
            if report.outstanding_reserved_cost_nano_usd:
                raise ValueError("complete study cannot retain outstanding budget reservations")
            return StudyCompletePayload(
                paired_protocol_digest=paired_protocol_digest,
                paired_report_digest=paired_report_digest,
                outcome_digest=outcome_digest,
                **self._terminal_budget_fields(report, publication),
            )

        return self._publish_derived(StudyPhase.COMPLETE, _derive)

    def stop(
        self,
        *,
        reason: StudyStopReason,
        budget_authority: BudgetLedgerAuthority,
        budget_report: StudyBudgetReport,
        budget_report_publication: StudyArtifactPublication,
        blocked_request_nano_usd: int | None = None,
        detail: str,
    ) -> StoppedPayload:
        """Stop without a result claim using exact current evidence and ledger accounting."""

        def _derive(records: tuple[StudyPhaseRecord, ...]) -> StoppedPayload:
            if not records:
                raise ValueError("study cannot stop before preparation is committed")
            report, publication = self._verify_budget_report(
                budget_authority,
                budget_report,
                budget_report_publication,
            )
            return StoppedPayload(
                reason=reason,
                last_evidence_digest=records[-1].digest,
                blocked_request_nano_usd=blocked_request_nano_usd,
                detail=detail,
                **self._terminal_budget_fields(report, publication),
            )

        return self._publish_derived(StudyPhase.STOPPED, _derive)

    def publish_candidate_source(
        self,
        *,
        protocol_digest: str,
        candidate: HarnessDoc,
        publication: StudyArtifactPublication,
    ) -> CandidatePublishedPayload:
        """Publish candidate source only when the external artifact is byte-for-byte exact."""
        selected = HarnessDoc.model_validate(candidate.model_dump(mode="json"))
        source_digest = _canonical_digest(selected.model_dump(mode="json"))
        receipt = self._verify_artifact(publication)
        if receipt.artifact_digest != source_digest:
            raise ValueError("candidate publication does not contain the exact candidate source")
        payload = CandidatePublishedPayload(
            protocol_digest=protocol_digest,
            candidate_execution_digest=selected.execution_digest,
            candidate_source_artifact_digest=source_digest,
            candidate_artifact_publication_digest=receipt.digest,
        )
        self._publish(payload)
        return payload

    def _verify_artifact(
        self,
        publication: StudyArtifactPublication,
    ) -> StudyArtifactPublication:
        receipt = StudyArtifactPublication.model_validate(publication.model_dump(mode="json"))
        if self._artifact_verifier is None:
            raise ValueError("study lifecycle has no external artifact verifier")
        self._artifact_verifier.verify_artifact(receipt)
        return receipt

    def _verify_budget_report(
        self,
        authority: BudgetLedgerAuthority,
        report: StudyBudgetReport,
        publication: StudyArtifactPublication,
    ) -> tuple[StudyBudgetReport, StudyArtifactPublication]:
        validated_authority = BudgetLedgerAuthority.model_validate(
            authority.model_dump(mode="python")
        )
        if validated_authority.policy.study_id != self._store.genesis.study_id:
            raise ValueError("budget authority belongs to a different study")
        claimed = StudyBudgetReport.model_validate(report.model_dump(mode="json"))
        current = StudyBudgetReport.capture(validated_authority)
        if claimed != current:
            raise ValueError("budget report is not the exact current live-ledger state")
        receipt = self._verify_artifact(publication)
        if receipt.artifact_digest != current.digest:
            raise ValueError("budget publication does not contain the exact current report")
        return current, receipt

    @staticmethod
    def _terminal_budget_fields(
        report: StudyBudgetReport,
        publication: StudyArtifactPublication,
    ) -> _TerminalBudgetFields:
        state = report.audit_state
        snapshot = state.snapshot
        return {
            "budget_policy_digest": state.policy_digest,
            "budget_ledger_identity": state.ledger_identity,
            "ledger_head_sequence": state.ledger_head_sequence,
            "ledger_head_digest": state.ledger_head_digest,
            "budget_report_digest": report.digest,
            "budget_report_publication_digest": publication.digest,
            "cumulative_paid_cost_nano_usd": snapshot.charged_nano_usd,
            "outstanding_reserved_cost_nano_usd": snapshot.reserved_nano_usd,
            "budget_hard_limit_nano_usd": snapshot.hard_limit_nano_usd,
            "budget_remaining_nano_usd": snapshot.remaining_nano_usd,
            "budget_breached": snapshot.breached,
        }

    def _publish(self, payload: StudyPhasePayload) -> StudyPhaseRecord:
        frozen = _PAYLOAD_ADAPTER.validate_python(payload.model_dump(mode="json"))
        return append_study_phase(
            self._store,
            phase=frozen.phase,
            payload_digest=frozen.digest,
            publisher=self._publisher,
        )

    def _publish_derived(
        self,
        phase: StudyPhase,
        derive: Callable[[tuple[StudyPhaseRecord, ...]], _StudyPayloadT],
    ) -> _StudyPayloadT:
        requested_phase = StudyPhase(phase)
        selected: _StudyPayloadT | None = None

        def _derive_digest(records: tuple[StudyPhaseRecord, ...]) -> str:
            nonlocal selected
            selected = derive(records)
            frozen = _PAYLOAD_ADAPTER.validate_python(selected.model_dump(mode="json"))
            if frozen.phase is not requested_phase:
                raise ValueError("derived study payload has the wrong phase")
            return frozen.digest

        append_study_phase_derived(
            self._store,
            phase=requested_phase,
            derive_payload_digest=_derive_digest,
            publisher=self._publisher,
        )
        if selected is None:
            raise RuntimeError("study phase payload was not derived")
        return selected

    def require_current_phase(
        self,
        expected: StudyPhase,
        *,
        payload_digest: str | None = None,
    ) -> StudyPhaseRecord:
        """Reverify the chain and fail unless its current phase is exactly expected."""
        required = StudyPhase(expected)
        records = self.records
        actual = records[-1].commitment.phase if records else None
        if actual is not required:
            actual_label = actual.value if actual is not None else "unstarted"
            raise ValueError(f"current study phase is {actual_label}, required {required.value}")
        record = records[-1]
        if payload_digest is not None and record.commitment.payload_digest != payload_digest:
            raise ValueError("current study phase carries different authorization evidence")
        return record

    def claim_run(
        self,
        expected: StudyPhase,
        run_id: str,
        *,
        payload_digest: str,
        resume: bool,
    ) -> StudyRunClaim:
        """Durably admit one exact run or an exact checkpoint-backed resume."""
        return claim_study_run(
            self._store,
            phase=expected,
            authorization_payload_digest=payload_digest,
            run_id=run_id,
            publisher=self._publisher,
            resume=resume,
        )

    def run_slice(
        self,
        expected: StudyPhase,
        run_id: str,
        operation: Callable[[], StudySliceResult[_CheckpointT, _CompletedResultT]],
        *,
        payload_digest: str,
        configuration_digest: str,
        resume_from: StudyRunCheckpointIdentity | None,
        checkpoint_identity: Callable[[_CheckpointT], StudyRunCheckpointIdentity],
    ) -> StudySliceResult[_CheckpointT, _CompletedResultT]:
        """Run one serialized invocation and bind its newly persisted checkpoint.

        Args:
            expected: Active phase authorized to execute the slice.
            run_id: Stable identity shared by the fresh invocation and every resume.
            operation: Bounded work that returns one new durable checkpoint.
            payload_digest: Exact typed phase authorization digest.
            configuration_digest: Frozen path-free slice configuration identity.
            resume_from: Latest checkpoint identity, or none for a fresh run.
            checkpoint_identity: Extract the path-free identity from the returned checkpoint.

        Returns:
            The detached checkpoint and optional complete result returned by the operation.
        """

        def _run() -> tuple[
            StudySliceResult[_CheckpointT, _CompletedResultT],
            StudyRunCheckpointIdentity,
        ]:
            frozen = operation().model_copy(deep=True)
            identity = StudyRunCheckpointIdentity.model_validate(
                checkpoint_identity(frozen.checkpoint).model_dump(mode="json")
            )
            return frozen, identity

        return call_in_study_slice(
            self._store,
            phase=expected,
            authorization_payload_digest=payload_digest,
            run_id=run_id,
            configuration_digest=configuration_digest,
            resume_from=resume_from,
            publisher=self._publisher,
            operation=_run,
        )

    def reconcile_slice(
        self,
        expected: StudyPhase,
        run_id: str,
        operation: Callable[[], StudySliceResult[_CheckpointT, _CompletedResultT]],
        *,
        payload_digest: str,
        configuration_digest: str,
        resume_from: StudyRunCheckpointIdentity,
        checkpoint_identity: Callable[[_CheckpointT], StudyRunCheckpointIdentity],
    ) -> StudySliceResult[_CheckpointT, _CompletedResultT]:
        """Reconcile a completed checkpoint and reconstruct its result without another slice.

        Args:
            expected: Active phase authorized to reconcile the run.
            run_id: Stable identity shared by the fresh invocation and every resume.
            operation: Side-effect-free reconstruction of the completed result.
            payload_digest: Exact typed phase authorization digest.
            configuration_digest: Frozen path-free slice configuration identity.
            resume_from: Completed caller-persisted checkpoint identity.
            checkpoint_identity: Extract the path-free identity from the reconstructed checkpoint.

        Returns:
            The detached completed checkpoint and its reconstructed result.
        """

        def _reconstruct() -> StudySliceResult[_CheckpointT, _CompletedResultT]:
            frozen = operation().model_copy(deep=True)
            reconstructed_identity = StudyRunCheckpointIdentity.model_validate(
                checkpoint_identity(frozen.checkpoint).model_dump(mode="json")
            )
            if reconstructed_identity != resume_from:
                raise ValueError(
                    "reconstructed checkpoint identity differs from the reconciled checkpoint"
                )
            return frozen

        return reconcile_study_slice(
            self._store,
            phase=expected,
            authorization_payload_digest=payload_digest,
            run_id=run_id,
            configuration_digest=configuration_digest,
            resume_from=resume_from,
            publisher=self._publisher,
            operation=_reconstruct,
        )

    def run_resumable_slice(
        self,
        expected: StudyPhase,
        run_id: str,
        operation: Callable[[], StudySliceResult[_CheckpointT, _CompletedResultT]],
        *,
        payload_digest: str,
        configuration_digest: str,
        resume_from: StudyRunCheckpointIdentity | None,
        checkpoint_identity: Callable[[_CheckpointT], StudyRunCheckpointIdentity],
    ) -> StudySliceResult[_CheckpointT, _CompletedResultT]:
        """Run idempotent work under a fresh or exact uncheckpointed slice intent.

        The operation must durably reuse completed work and execute only the outstanding work
        already bound by its own nested intent. Ordinary callbacks must use :meth:`run_slice`.
        """

        def _run() -> tuple[
            StudySliceResult[_CheckpointT, _CompletedResultT],
            StudyRunCheckpointIdentity,
        ]:
            frozen = operation().model_copy(deep=True)
            identity = StudyRunCheckpointIdentity.model_validate(
                checkpoint_identity(frozen.checkpoint).model_dump(mode="json")
            )
            return frozen, identity

        return call_in_resumable_study_slice(
            self._store,
            phase=expected,
            authorization_payload_digest=payload_digest,
            run_id=run_id,
            configuration_digest=configuration_digest,
            resume_from=resume_from,
            publisher=self._publisher,
            operation=_run,
        )

    def call_in_phase(
        self,
        expected: StudyPhase,
        operation: Callable[[], _ResultT],
        *,
        payload_digest: str | None = None,
    ) -> _ResultT:
        """Invoke one side effect only after externally reverifying its required phase."""
        return call_in_study_phase(
            self._store,
            phase=expected,
            payload_digest=payload_digest,
            publisher=self._publisher,
            operation=operation,
        )


def _canonical_digest(value: object) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(serialized).hexdigest()
