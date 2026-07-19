"""Typed phase payloads and side-effect guards for optimization studies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from enum import StrEnum
from typing import Annotated, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    TypeAdapter,
    field_validator,
)

from wmh.core.text import validate_durable_text
from wmh.evals.study_journal import (
    ExternalCommitmentPublisher,
    StudyJournalStore,
    StudyPhase,
    StudyPhaseRecord,
    append_study_phase,
    load_study_journal,
)

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_ResultT = TypeVar("_ResultT")


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


class PreparationPlannedPayload(_StudyPayload):
    """Budget and immutable plan identity accepted before any preparation side effect."""

    phase: Literal[StudyPhase.PREPARATION_PLANNED] = StudyPhase.PREPARATION_PLANNED
    study_plan_digest: str = Field(pattern=_DIGEST_PATTERN)
    budget_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    budget_binding_digest: str = Field(pattern=_DIGEST_PATTERN)
    maximum_paid_cost_microusd: StrictInt = Field(ge=0)


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


class StudyCompletePayload(_StudyPayload):
    """Final outcome, complete evidence, and exact cumulative paid cost."""

    phase: Literal[StudyPhase.COMPLETE] = StudyPhase.COMPLETE
    paired_protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    paired_report_digest: str = Field(pattern=_DIGEST_PATTERN)
    outcome_digest: str = Field(pattern=_DIGEST_PATTERN)
    spend_ledger_digest: str = Field(pattern=_DIGEST_PATTERN)
    cumulative_paid_cost_microusd: StrictInt = Field(ge=0)


class StudyStopReason(StrEnum):
    """Predeclared reasons an incomplete study can terminate without a result claim."""

    BUDGET_EXHAUSTED = "budget_exhausted"
    OPERATOR_STOPPED = "operator_stopped"
    INFRASTRUCTURE_INVALIDATED = "infrastructure_invalidated"
    PROTOCOL_INVALIDATED = "protocol_invalidated"


class StoppedPayload(_StudyPayload):
    """Honest terminal record for a study that cannot complete its fixed protocol."""

    phase: Literal[StudyPhase.STOPPED] = StudyPhase.STOPPED
    reason: StudyStopReason
    last_evidence_digest: str = Field(pattern=_DIGEST_PATTERN)
    spend_ledger_digest: str = Field(pattern=_DIGEST_PATTERN)
    cumulative_paid_cost_microusd: StrictInt = Field(ge=0)
    detail: str = Field(min_length=1, max_length=2_048)

    @field_validator("detail")
    @classmethod
    def _validate_detail(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("study stop detail cannot have surrounding whitespace")
        validate_durable_text(value, field="study stop detail")
        return value


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
    ) -> None:
        self._store = store
        self._publisher = publisher

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
        return append_study_phase(
            self._store,
            phase=frozen.phase,
            payload_digest=frozen.digest,
            publisher=self._publisher,
        )

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

    def call_in_phase(
        self,
        expected: StudyPhase,
        operation: Callable[[], _ResultT],
        *,
        payload_digest: str | None = None,
    ) -> _ResultT:
        """Invoke one side effect only after externally reverifying its required phase."""
        self.require_current_phase(expected, payload_digest=payload_digest)
        return operation()
