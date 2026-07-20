"""Crash-durable composition for one sealed harness optimization study."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Protocol, Self, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    field_validator,
    model_validator,
)

from wmh.agents.project import (
    PROJECT_WORKSPACE,
    AgentProject,
    AgentProjectExecutionCommitment,
)
from wmh.core.file_lease import exclusive_posix_file_lease
from wmh.core.text import validate_durable_text
from wmh.evals.harbor.config import HarborEnvironmentBackend, HarborJobSpec
from wmh.evals.harbor.paired_runner import (
    HarborExecutionRuntime,
    PairedHarborProtocol,
    PairedHarborRunner,
    PairedHarborRunReport,
    PairedHarborSliceProgress,
    PairedHarborSliceResult,
)
from wmh.evals.harbor.scorer import HarborHarnessScorer
from wmh.evals.harness_optimization import (
    FrozenHarnessOptimizationCandidate,
    HarnessOptimizationOutcome,
    OpenedHarnessOptimizationConfirmation,
    PreparedHarnessOptimizationStudy,
    freeze_harness_optimization_candidate,
    freeze_harness_optimization_harbor_protocol,
    open_harness_optimization_confirmation,
    run_harness_optimization_search_slice,
    summarize_harness_optimization_outcome,
)
from wmh.evals.partition import PartitionControlStore
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
    CandidatePublishedPayload,
    ConfirmationFrozenPayload,
    ConfirmationOpenedPayload,
    ConfirmationRunningPayload,
    DiscoveryRunningPayload,
    PreparationPlannedPayload,
    ProtocolPublishedPayload,
    RosterQualifiedPayload,
    StoppedPayload,
    StudyArtifactPublication,
    StudyBudgetReport,
    StudyCompletePayload,
    StudyLifecycleController,
    StudyPhasePayload,
    StudySliceResult,
)
from wmh.harness.cost import SearchComponentRole, SearchCostRuntime
from wmh.harness.create import SearchCheckpoint, SearchProposalBatchWitness
from wmh.harness.doc import HarnessDoc
from wmh.harness.proposer import DeltaProposer, ProjectDeltaProposer, ProviderDeltaProposer
from wmh.harness.scoring import HarnessScorer
from wmh.providers import Provider, get_provider
from wmh.providers.base import ProviderConfig, ToolCallingProvider
from wmh.tracking.budget import BudgetLedgerAuthority
from wmh.tracking.rate_limit import (
    E2B_SANDBOX_CREATE_RATE_POLICY,
    ExternalDispatchRateAuthority,
    bind_external_dispatch_rate_authority,
)

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_STATE_FILE = "study-state.json"
_STATE_LOCK_FILE = "study-state.lock"
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600


class DirectDiscoveryProposerRuntime(BaseModel):
    """Use one provider completion directly for each proposal batch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["direct"] = "direct"


class ProjectDiscoveryProposerRuntime(BaseModel):
    """Run a tool-using proposer agent inside one durable project sandbox."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["project"] = "project"
    agent: HarnessDoc
    workspace: str = PROJECT_WORKSPACE
    execution: AgentProjectExecutionCommitment
    lease_ledger_dir: Path
    preserve_runtime_kind: bool = False

    @model_validator(mode="after")
    def _validate_host_coordinates(self) -> Self:
        if self.workspace != PROJECT_WORKSPACE:
            raise ValueError("project proposer workspace differs from the supported project root")
        if not self.lease_ledger_dir.is_absolute():
            raise ValueError("project proposer lease ledger directory must be absolute")
        return self


DiscoveryProposerRuntime = Annotated[
    DirectDiscoveryProposerRuntime | ProjectDiscoveryProposerRuntime,
    Field(discriminator="kind"),
]


class HarnessOptimizationStudySpec(BaseModel):
    """Complete nonsecret study inputs and host coordinates loaded by the coordinator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    spec_version: Literal["1"] = "1"
    prepared: PreparedHarnessOptimizationStudy
    partition_control_dir: Path
    discovery_job_spec: HarborJobSpec
    confirmation_runtime: HarborExecutionRuntime
    discovery_proposer: DiscoveryProposerRuntime = Field(
        default_factory=DirectDiscoveryProposerRuntime
    )
    discovery_create_rate_ledger_path: Path | None = None
    qualification_report_digest: str = Field(pattern=_DIGEST_PATTERN)
    confirmation_operation_id: str = Field(min_length=1, max_length=256)
    confirmation_generation_id: StrictInt = Field(default=1, ge=1)

    @field_validator("confirmation_generation_id", mode="before")
    @classmethod
    def _reject_boolean_generation(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("confirmation generation cannot be boolean")
        return value

    @field_validator("confirmation_operation_id")
    @classmethod
    def _validate_operation_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("confirmation operation id cannot have surrounding whitespace")
        validate_durable_text(value, field="confirmation operation id")
        return value

    @model_validator(mode="after")
    def _validate_composition(self) -> Self:
        for label, path in (
            ("partition control directory", self.partition_control_dir),
            ("discovery jobs directory", self.discovery_job_spec.jobs_dir),
            ("confirmation jobs directory", self.confirmation_runtime.jobs_dir),
        ):
            if not path.is_absolute():
                raise ValueError(f"{label} must be absolute")
        plan = self.prepared.protocol.execution_plan
        job = self.discovery_job_spec
        if job.environment_backend is not plan.environment_backend:
            raise ValueError("discovery backend differs from the frozen Harbor execution plan")
        if job.create_rate_policy != plan.create_rate_policy:
            raise ValueError(
                "discovery create-rate policy differs from the frozen Harbor execution plan"
            )
        if job.n_attempts != self.prepared.protocol.search.attempts_per_task:
            raise ValueError("discovery attempts differ from the frozen search plan")
        selected_tasks: list[str] = []
        for dataset in job.datasets:
            if dataset.task_names is None:
                raise ValueError("discovery datasets require explicit task_names")
            selected_tasks.extend(dataset.task_names)
        expected_tasks = tuple(task.task_id for task in self.prepared.protocol.discovery.tasks)
        if tuple(selected_tasks) != expected_tasks:
            raise ValueError("discovery dataset task_names differ from the frozen partition")
        if self.confirmation_runtime.budget != self.prepared.confirmation_budget:
            raise ValueError("confirmation runtime budget differs from the prepared study")
        if set(self.confirmation_runtime.dataset_paths_by_id) != {
            task.dataset_id for task in self.prepared.qualification_roster.tasks
        }:
            raise ValueError("confirmation runtime datasets differ from the qualified roster")
        project_proposer = isinstance(self.discovery_proposer, ProjectDiscoveryProposerRuntime)
        discovery_uses_e2b = plan.create_rate_policy is not None or project_proposer
        if not discovery_uses_e2b:
            if self.discovery_create_rate_ledger_path is not None:
                raise ValueError("discovery without E2B cannot carry a create-rate ledger")
            if self.prepared.search_cost_binding.external_dispatch_rate_binding is not None:
                raise ValueError("discovery without E2B cannot bind a create-rate authority")
        else:
            if self.discovery_create_rate_ledger_path is None:
                raise ValueError("E2B discovery requires a create-rate ledger")
            if not self.discovery_create_rate_ledger_path.is_absolute():
                raise ValueError("discovery create-rate ledger must be absolute")
            expected_rate = self.prepared.search_cost_binding.external_dispatch_rate_binding
            if expected_rate is None:
                raise ValueError("E2B discovery requires a frozen create-rate binding")
            if expected_rate.policy_digest != E2B_SANDBOX_CREATE_RATE_POLICY.digest:
                raise ValueError("discovery create-rate binding differs from the E2B policy")
            if project_proposer and self.discovery_proposer.execution.create_rate_binding != (
                expected_rate
            ):
                raise ValueError(
                    "project proposer create-rate binding differs from the search cost binding"
                )
        if plan.create_rate_policy is None:
            if self.confirmation_runtime.create_rate_ledger_path is not None:
                raise ValueError("local confirmation cannot carry a create-rate ledger")
        else:
            if self.confirmation_runtime.create_rate_ledger_path is None:
                raise ValueError("E2B confirmation requires a create-rate ledger")
        proposer_resources = self.prepared.search_cost_binding.proposer.timed_resources
        if project_proposer:
            if len(proposer_resources) != 1:
                raise ValueError("project proposer requires one timed project resource")
            resource = proposer_resources[0]
            execution_resource = self.discovery_proposer.execution.resource_class
            if (
                resource.resource_type != execution_resource.role.value
                or resource.resource_class_digest != execution_resource.digest
            ):
                raise ValueError(
                    "project proposer execution differs from its timed resource binding"
                )
        elif proposer_resources:
            raise ValueError("direct proposer cannot bind a timed project resource")
        return self

    @property
    def digest(self) -> str:
        """Return the exact identity of the coordinator inputs and host bindings."""
        return _canonical_digest(self.model_dump(mode="json"))


class HarnessOptimizationStudyState(BaseModel):
    """Host-private artifacts needed to resume every phase without resampling paid work."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state_version: Literal["1"] = "1"
    spec_digest: str = Field(pattern=_DIGEST_PATTERN)
    preparation: PreparationPlannedPayload | None = None
    roster_qualified: RosterQualifiedPayload | None = None
    protocol_publication: StudyArtifactPublication | None = None
    protocol_published: ProtocolPublishedPayload | None = None
    discovery_running: DiscoveryRunningPayload | None = None
    search_checkpoint: SearchCheckpoint | None = None
    search_checkpoint_journaled_iteration: StrictInt | None = Field(default=None, ge=0)
    proposal_batch_witness: SearchProposalBatchWitness | None = None
    frozen_candidate: FrozenHarnessOptimizationCandidate | None = None
    search_budget_report: StudyBudgetReport | None = None
    search_budget_publication: StudyArtifactPublication | None = None
    candidate_frozen: CandidateFrozenPayload | None = None
    candidate_publication: StudyArtifactPublication | None = None
    candidate_published: CandidatePublishedPayload | None = None
    opened_confirmation: OpenedHarnessOptimizationConfirmation | None = None
    confirmation_opened: ConfirmationOpenedPayload | None = None
    paired_protocol: PairedHarborProtocol | None = None
    confirmation_frozen: ConfirmationFrozenPayload | None = None
    confirmation_running: ConfirmationRunningPayload | None = None
    confirmation_slice: PairedHarborSliceResult | None = None
    confirmation_journaled_sequence: StrictInt | None = Field(default=None, ge=0)
    outcome: HarnessOptimizationOutcome | None = None
    final_budget_report: StudyBudgetReport | None = None
    final_budget_publication: StudyArtifactPublication | None = None
    complete: StudyCompletePayload | None = None
    stopped: StoppedPayload | None = None

    @model_validator(mode="after")
    def _validate_dependencies(self) -> Self:
        if self.protocol_published is not None and self.protocol_publication is None:
            raise ValueError("protocol payload requires its artifact publication")
        if self.search_checkpoint_journaled_iteration is not None:
            if (
                self.search_checkpoint is None
                or self.search_checkpoint.completed_iteration
                < self.search_checkpoint_journaled_iteration
            ):
                raise ValueError("journaled search iteration lacks its checkpoint")
        if self.frozen_candidate is not None and self.search_checkpoint is None:
            raise ValueError("frozen candidate requires a completed search checkpoint")
        if self.candidate_frozen is not None and (
            self.frozen_candidate is None
            or self.search_budget_report is None
            or self.search_budget_publication is None
        ):
            raise ValueError("candidate freeze payload lacks its staged evidence")
        if self.candidate_published is not None and (
            self.frozen_candidate is None or self.candidate_publication is None
        ):
            raise ValueError("candidate publication payload lacks its staged evidence")
        if self.opened_confirmation is not None and self.frozen_candidate is None:
            raise ValueError("opened confirmation requires a frozen candidate")
        if self.paired_protocol is not None and self.opened_confirmation is None:
            raise ValueError("paired protocol requires an opened confirmation")
        if self.confirmation_slice is not None and self.confirmation_running is None:
            raise ValueError("confirmation progress requires running authorization")
        if self.confirmation_journaled_sequence is not None:
            if self.confirmation_slice is None:
                raise ValueError("journaled confirmation sequence lacks its progress")
            expected = self.confirmation_slice.progress.slice_index - 1
            if self.confirmation_journaled_sequence > expected:
                raise ValueError("journaled confirmation sequence exceeds persisted progress")
        if self.complete is not None and (
            self.outcome is None
            or self.final_budget_report is None
            or self.final_budget_publication is None
        ):
            raise ValueError("complete payload lacks its staged evidence")
        return self

    def payload_for_phase(self, phase: StudyPhase) -> StudyPhasePayload | None:
        """Return the staged typed payload corresponding to one lifecycle phase."""
        return {
            StudyPhase.PREPARATION_PLANNED: self.preparation,
            StudyPhase.ROSTER_QUALIFIED: self.roster_qualified,
            StudyPhase.PROTOCOL_PUBLISHED: self.protocol_published,
            StudyPhase.DISCOVERY_RUNNING: self.discovery_running,
            StudyPhase.CANDIDATE_FROZEN: self.candidate_frozen,
            StudyPhase.CANDIDATE_PUBLISHED: self.candidate_published,
            StudyPhase.CONFIRMATION_OPENED: self.confirmation_opened,
            StudyPhase.CONFIRMATION_FROZEN: self.confirmation_frozen,
            StudyPhase.CONFIRMATION_RUNNING: self.confirmation_running,
            StudyPhase.COMPLETE: self.complete,
            StudyPhase.STOPPED: self.stopped,
        }[phase]


class HarnessOptimizationAdvanceKind(StrEnum):
    """One bounded unit of coordinator progress."""

    PHASE = "phase"
    DISCOVERY_SLICE = "discovery_slice"
    DISCOVERY_RECONCILED = "discovery_reconciled"
    CONFIRMATION_SLICE = "confirmation_slice"
    CONFIRMATION_RECONCILED = "confirmation_reconciled"
    TERMINAL = "terminal"


class HarnessOptimizationAdvance(BaseModel):
    """Compact result from exactly one phase transition or execution slice."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: HarnessOptimizationAdvanceKind
    previous_phase: StudyPhase | None
    current_phase: StudyPhase | None
    checkpoint_sequence: StrictInt | None = Field(default=None, ge=0)
    terminal: bool = False


class HarnessOptimizationStateStore(Protocol):
    """Durable host-state operations needed by the coordinator."""

    def load(self) -> HarnessOptimizationStudyState: ...

    def save(self, state: HarnessOptimizationStudyState) -> None: ...

    def locked(self) -> AbstractContextManager[None]: ...


class LocalHarnessOptimizationStateStore:
    """Canonical atomic JSON state with a process-crash-safe local lease."""

    def __init__(self, directory: Path) -> None:
        self._directory = _ensure_private_directory(directory)
        self._state_path = self._directory / _STATE_FILE
        self._lock_path = self._directory / _STATE_LOCK_FILE

    @property
    def directory(self) -> Path:
        """Return the private coordinator state directory."""
        return self._directory

    def initialize(self, *, spec_digest: str, resume: bool) -> HarnessOptimizationStudyState:
        """Create a fresh state once or require an exact existing state for resume."""
        existing = self._load_optional()
        if existing is None:
            if resume:
                raise ValueError("cannot resume before coordinator state exists")
            state = HarnessOptimizationStudyState(spec_digest=spec_digest)
            self.save(state)
            return state
        if not resume:
            raise ValueError("coordinator state already exists; pass --resume")
        if existing.spec_digest != spec_digest:
            raise ValueError("coordinator state belongs to a different study spec")
        return existing

    def load(self) -> HarnessOptimizationStudyState:
        """Load and revalidate the exact canonical state file."""
        state = self._load_optional()
        if state is None:
            raise ValueError("coordinator state has not been initialized")
        return state

    def save(self, state: HarnessOptimizationStudyState) -> None:
        """Atomically replace the current state with one canonical validated snapshot."""
        frozen = HarnessOptimizationStudyState.model_validate(state.model_dump(mode="json"))
        _atomic_write_private_json(self._state_path, frozen.model_dump(mode="json"))

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Serialize one coordinator advance across local processes."""
        with exclusive_posix_file_lease(
            self._lock_path,
            unsupported_error=RuntimeError("study coordination requires POSIX file locking"),
            irregular_file_error=OSError("study coordinator lease is not a regular file"),
            contention_error=RuntimeError("study coordinator is already advancing"),
        ):
            yield

    def _load_optional(self) -> HarnessOptimizationStudyState | None:
        payload = _read_optional_private_regular_file(self._state_path)
        if payload is None:
            return None
        state = HarnessOptimizationStudyState.model_validate_json(payload)
        if payload != _canonical_json_bytes(state.model_dump(mode="json")):
            raise ValueError("study coordinator state is not canonical")
        return state


class StudyArtifactPublisher(Protocol):
    """Publish canonical nonsecret JSON as an immutable artifact."""

    def publish_artifact(
        self,
        *,
        kind: str,
        artifact_digest: str,
        content: JsonValue,
    ) -> StudyArtifactPublication: ...


class _LocalCommitmentEntry(BaseModel):
    """Canonical local evidence for one externally shaped journal publication."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    commitment: StudyPhaseCommitment
    receipt: ExternalPublicationReceipt


class _LocalArtifactEntry(BaseModel):
    """Canonical local evidence for one immutable artifact publication."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str = Field(min_length=1, max_length=128)
    content: JsonValue
    publication: StudyArtifactPublication


class LocalStudyEvidenceStore:
    """Local append-only publication adapter used by the default single-host coordinator."""

    def __init__(
        self,
        directory: Path,
        *,
        study_id: str,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._directory = _ensure_private_directory(directory)
        self._study_id = study_id
        self._clock = clock
        self._commitments = _ensure_private_directory(self._directory / "commitments")
        self._artifacts = _ensure_private_directory(self._directory / "artifacts")
        self._configuration_digest = _canonical_digest(
            {"adapter": "wmh.local-study-evidence.v1", "study_id": study_id}
        )

    @property
    def configuration_digest(self) -> str:
        """Return the path-free identity bound into the phase journal genesis."""
        return self._configuration_digest

    def publish(self, commitment: StudyPhaseCommitment) -> ExternalPublicationReceipt:
        """Publish one journal sequence idempotently and reject any fork."""
        self._require_study(commitment.study_id)
        path = self._commitment_path(commitment.sequence)
        existing = self._read_commitment(path)
        if existing is not None:
            if existing.commitment != commitment:
                raise ValueError("local study publication sequence is already occupied")
            return existing.receipt
        receipt = ExternalPublicationReceipt(
            commitment_digest=commitment.digest,
            publisher="wmh-local-study-evidence",
            publication_id=f"phase-{commitment.sequence:03d}",
            immutable_locator=f"wmh-local-study://{self._study_id}/phase/{commitment.sequence}",
            published_at=self._timestamp(),
            evidence={"sequence": commitment.sequence, "phase": commitment.phase.value},
        )
        entry = _LocalCommitmentEntry(commitment=commitment, receipt=receipt)
        won_publication = _atomic_publish_private_json(path, entry.model_dump(mode="json"))
        persisted = self._read_commitment(path)
        if persisted is None:
            raise RuntimeError("local study publication did not persist evidence")
        if not won_publication and persisted.commitment == commitment:
            return persisted.receipt
        if persisted != entry:
            if persisted.commitment != commitment:
                raise ValueError("local study publication sequence is already occupied")
            raise RuntimeError("local study publication did not persist exact evidence")
        return receipt

    def verify(
        self,
        commitment: StudyPhaseCommitment,
        receipt: ExternalPublicationReceipt,
    ) -> None:
        """Verify one local publication against its immutable sequence slot."""
        entry = self._read_commitment(self._commitment_path(commitment.sequence))
        if entry is None or entry.commitment != commitment or entry.receipt != receipt:
            raise ValueError("local study publication evidence is missing or inconsistent")

    def verify_chain_head(
        self,
        genesis: StudyJournalGenesis,
        records: tuple[StudyPhaseRecord, ...],
        pending: StudyPhaseCommitment | None,
    ) -> None:
        """Reject local publication forks beyond the exact journal head."""
        self._require_study(genesis.study_id)
        entries = self._commitment_entries()
        committed = tuple(record.commitment for record in records)
        actual = tuple(entry.commitment for entry in entries)
        allowed = (committed, (*committed, pending)) if pending is not None else (committed,)
        if actual not in allowed:
            raise ValueError("local study publication chain differs from the journal head")

    def publish_artifact(
        self,
        *,
        kind: str,
        artifact_digest: str,
        content: JsonValue,
    ) -> StudyArtifactPublication:
        """Publish canonical JSON under its content digest, idempotently."""
        if kind != kind.strip() or not kind:
            raise ValueError("artifact kind must be canonical")
        if _canonical_digest(content) != artifact_digest:
            raise ValueError("artifact content differs from its declared digest")
        path = self._artifact_path(artifact_digest)
        existing = self._read_artifact(path)
        if existing is not None:
            if existing.kind != kind or existing.content != content:
                raise ValueError("artifact digest is already published with different content")
            return existing.publication
        publication = StudyArtifactPublication.create(
            artifact_digest=artifact_digest,
            publisher="wmh-local-study-evidence",
            publication_id=f"artifact-{artifact_digest.removeprefix('sha256:')}",
            immutable_locator=f"wmh-local-study://{self._study_id}/artifact/{artifact_digest}",
            published_at=self._timestamp(),
            evidence={"kind": kind},
        )
        entry = _LocalArtifactEntry(kind=kind, content=content, publication=publication)
        won_publication = _atomic_publish_private_json(path, entry.model_dump(mode="json"))
        persisted = self._read_artifact(path)
        if persisted is None:
            raise RuntimeError("local study artifact did not persist evidence")
        if not won_publication and persisted.kind == kind and persisted.content == content:
            return persisted.publication
        if persisted != entry:
            if persisted.kind != kind or persisted.content != content:
                raise ValueError("artifact digest is already published with different content")
            raise RuntimeError("local study artifact did not persist exact evidence")
        return publication

    def verify_artifact(self, publication: StudyArtifactPublication) -> None:
        """Verify an artifact receipt against the immutable local content slot."""
        entry = self._read_artifact(self._artifact_path(publication.artifact_digest))
        if entry is None or entry.publication != publication:
            raise ValueError("local study artifact evidence is missing or inconsistent")
        if _canonical_digest(entry.content) != publication.artifact_digest:
            raise ValueError("local study artifact content digest is inconsistent")

    def _timestamp(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("study evidence clock must return a timezone-aware timestamp")
        return value

    def _require_study(self, study_id: str) -> None:
        if study_id != self._study_id:
            raise ValueError("local study publication belongs to a different study")

    def _commitment_path(self, sequence: int) -> Path:
        return self._commitments / f"phase-{sequence:03d}.json"

    def _artifact_path(self, digest: str) -> Path:
        hexadecimal = digest.removeprefix("sha256:")
        if not _is_digest(digest):
            raise ValueError("artifact key must be a canonical SHA-256 digest")
        return self._artifacts / f"artifact-{hexadecimal}.json"

    def _commitment_entries(self) -> tuple[_LocalCommitmentEntry, ...]:
        paths = sorted(self._commitments.glob("phase-*.json"))
        entries: list[_LocalCommitmentEntry] = []
        for expected, path in enumerate(paths):
            if path != self._commitment_path(expected):
                raise ValueError("local study publication sequences are not contiguous")
            entry = self._read_commitment(path)
            if entry is None:
                raise RuntimeError("listed local study publication disappeared")
            entries.append(entry)
        return tuple(entries)

    @staticmethod
    def _read_commitment(path: Path) -> _LocalCommitmentEntry | None:
        _recover_atomic_publication_aliases(path)
        return _read_optional_canonical_model(path, _LocalCommitmentEntry)

    @staticmethod
    def _read_artifact(path: Path) -> _LocalArtifactEntry | None:
        _recover_atomic_publication_aliases(path)
        return _read_optional_canonical_model(path, _LocalArtifactEntry)


@dataclass(frozen=True)
class HarnessOptimizationDiscoveryRuntime:
    """One exact scorer and proposer pair plus deterministic cleanup."""

    scorer: HarnessScorer
    proposer: DeltaProposer
    close: Callable[[], None]


class PairedHarborSliceRunner(Protocol):
    """Minimal asynchronous paired runner surface used by the confirmation adapter."""

    async def run_slice(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
        max_new_blocks: int | None = None,
    ) -> PairedHarborSliceResult: ...

    async def recover_persisted_slice(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
    ) -> PairedHarborSliceResult | None: ...


class HarnessOptimizationRuntimeFactory(Protocol):
    """Build paid runtime components only after lifecycle authorization succeeds."""

    def build_discovery(
        self,
        spec: HarnessOptimizationStudySpec,
    ) -> HarnessOptimizationDiscoveryRuntime: ...

    def build_confirmation(
        self,
        spec: HarnessOptimizationStudySpec,
        protocol: PairedHarborProtocol,
    ) -> PairedHarborSliceRunner: ...


def _close_discovery_resources(
    project: AgentProject | None,
    scorer: HarborHarnessScorer,
) -> None:
    """Close every owned discovery resource even when an earlier cleanup fails."""
    failures: list[BaseException] = []
    if project is not None:
        try:
            project.close()
        except BaseException as error:  # noqa: BLE001 - all cleanup must still be attempted
            failures.append(error)
    try:
        scorer.close()
    except BaseException as error:  # noqa: BLE001 - preserve every cleanup failure
        failures.append(error)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup("discovery resource cleanup failed", failures)


class ProductionHarnessOptimizationRuntimeFactory:
    """Construct frozen direct or project-backed discovery and Harbor confirmation runtimes."""

    def __init__(
        self,
        provider_factory: Callable[[ProviderConfig], Provider] = get_provider,
    ) -> None:
        self._provider_factory = provider_factory

    def build_discovery(
        self,
        spec: HarnessOptimizationStudySpec,
    ) -> HarnessOptimizationDiscoveryRuntime:
        """Build cost-bound discovery components from the sealed spec."""
        prepared = spec.prepared
        binding = prepared.search_cost_binding
        proposer_binding = binding.proposer
        scorer_binding = binding.scorer
        if len(proposer_binding.providers) != 1 or len(scorer_binding.providers) != 1:
            raise ValueError("production direct discovery requires one provider per component")
        authority = _budget_authority(prepared)
        cost_runtime = SearchCostRuntime(authority=authority, binding=binding)
        create_rate_authority: ExternalDispatchRateAuthority | None = None
        if spec.discovery_create_rate_ledger_path is not None:
            create_rate_authority = ExternalDispatchRateAuthority.bootstrap(
                spec.discovery_create_rate_ledger_path,
                E2B_SANDBOX_CREATE_RATE_POLICY,
            )
            expected_binding = binding.external_dispatch_rate_binding
            if (
                expected_binding is None
                or bind_external_dispatch_rate_authority(create_rate_authority) != expected_binding
            ):
                raise ValueError("discovery create-rate authority differs from its cost binding")
        scorer_provider = scorer_binding.providers[0]
        qualifications = {task.task_id: task for task in prepared.qualification_roster.tasks}
        discovery_tasks = tuple(
            qualifications[task.task_id] for task in prepared.protocol.discovery.tasks
        )
        scorer_create_rate_authority = (
            create_rate_authority
            if prepared.protocol.execution_plan.create_rate_policy is not None
            else None
        )
        scorer = HarborHarnessScorer(
            job_spec=spec.discovery_job_spec,
            provider_config=scorer_provider.provider_config,
            response_identity=scorer_provider.response_identity,
            reference_harness=prepared.baseline,
            qualified_tasks=discovery_tasks,
            reward_key=prepared.protocol.execution_plan.reward_key,
            runner_spec=prepared.protocol.execution_plan.runner_spec,
            turn_timeout_s=prepared.protocol.execution_plan.turn_timeout_s,
            cost_runtime=cost_runtime.for_component(SearchComponentRole.SCORER),
            create_rate_authority=scorer_create_rate_authority,
        )
        proposer_provider = proposer_binding.providers[0]
        project: AgentProject | None = None
        try:
            raw_proposer = self._provider_factory(proposer_provider.provider_config)
            if isinstance(spec.discovery_proposer, DirectDiscoveryProposerRuntime):
                proposer: DeltaProposer = ProviderDeltaProposer(
                    raw_proposer,
                    cost_runtime=cost_runtime.for_component(SearchComponentRole.PROPOSER),
                    response_identity=proposer_provider.response_identity,
                )
            else:
                if create_rate_authority is None:
                    raise ValueError("project proposer requires a create-rate authority")
                if not isinstance(raw_proposer, ToolCallingProvider):
                    raise TypeError("project proposer requires a tool-calling provider")
                project_spec = spec.discovery_proposer
                execution = AgentProject.execution_commitment_for(
                    timeout=project_spec.execution.resource_class.provider_ttl_seconds,
                    template=project_spec.execution.template,
                    cpu_count=project_spec.execution.resource_class.cpu_count,
                    memory_mb=project_spec.execution.resource_class.memory_mb,
                    create_rate_authority=create_rate_authority,
                )
                if execution != project_spec.execution:
                    raise ValueError("project proposer runtime differs from its frozen execution")
                configuration_id = ProjectDeltaProposer.configuration_id_for(
                    project_type=AgentProject,
                    project_workspace=project_spec.workspace,
                    project_execution_configuration_id=execution.digest,
                    agent=project_spec.agent,
                    provider=raw_proposer,
                    preserve_runtime_kind=project_spec.preserve_runtime_kind,
                    project_create_rate_binding=execution.create_rate_binding,
                    response_identity=proposer_provider.response_identity,
                )
                if configuration_id != proposer_binding.configuration_id:
                    raise ValueError(
                        "project proposer configuration differs from its search cost binding"
                    )
                component_runtime = cost_runtime.for_component(SearchComponentRole.PROPOSER)
                project = AgentProject.create(
                    timeout=execution.resource_class.provider_ttl_seconds,
                    template=execution.template,
                    cpu_count=execution.resource_class.cpu_count,
                    memory_mb=execution.resource_class.memory_mb,
                    cost_runtime=component_runtime,
                    component_configuration_id=configuration_id,
                    lease_ledger_dir=project_spec.lease_ledger_dir,
                    create_rate_authority=create_rate_authority,
                )
                proposer = ProjectDeltaProposer(
                    project,
                    project_spec.agent,
                    raw_proposer,
                    preserve_runtime_kind=project_spec.preserve_runtime_kind,
                    cost_runtime=component_runtime,
                    response_identity=proposer_provider.response_identity,
                )
        except BaseException as construction_error:
            try:
                _close_discovery_resources(project, scorer)
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve both failures
                raise BaseExceptionGroup(
                    "discovery construction and cleanup failed",
                    [construction_error, cleanup_error],
                ) from None
            raise
        return HarnessOptimizationDiscoveryRuntime(
            scorer=scorer,
            proposer=proposer,
            close=lambda: _close_discovery_resources(project, scorer),
        )

    def build_confirmation(
        self,
        spec: HarnessOptimizationStudySpec,
        protocol: PairedHarborProtocol,
    ) -> PairedHarborRunner:
        """Build the frozen paired Harbor runner without altering backend semantics."""
        return PairedHarborRunner(
            protocol=protocol,
            runtime=spec.confirmation_runtime,
            operation_id=spec.confirmation_operation_id,
            generation_id=spec.confirmation_generation_id,
        )


def confirmation_initial_run_state_digest(
    protocol: PairedHarborProtocol,
    *,
    operation_id: str,
    generation_id: int,
) -> str:
    """Return the path-free empty progress identity admitted before paired execution."""
    return _canonical_digest(
        {
            "schema_version": "wmh.paired-confirmation-initial-state.v1",
            "protocol_digest": protocol.digest,
            "operation_id": operation_id,
            "generation_id": generation_id,
            "completed_blocks": [],
        }
    )


def run_harness_optimization_confirmation_slice(
    *,
    opened: OpenedHarnessOptimizationConfirmation,
    protocol: PairedHarborProtocol,
    runner: PairedHarborSliceRunner,
    lifecycle: StudyLifecycleController,
    authorization: ConfirmationRunningPayload,
    operation_id: str,
    generation_id: int,
    resume_from: PairedHarborSliceResult | None,
    on_checkpoint: Callable[[PairedHarborSliceResult], None],
) -> PairedHarborSliceResult:
    """Run one bounded paired slice through lifecycle checkpoint admission."""
    _validate_confirmation_runtime_authorization(
        opened=opened,
        protocol=protocol,
        authorization=authorization,
        operation_id=operation_id,
        generation_id=generation_id,
    )
    previous = _freeze_optional_slice(resume_from)

    def _checkpoint_identity(progress: PairedHarborSliceProgress) -> StudyRunCheckpointIdentity:
        return StudyRunCheckpointIdentity(
            sequence=progress.slice_index - 1,
            checkpoint_digest=progress.progress_digest,
        )

    def _run() -> StudySliceResult[PairedHarborSliceProgress, PairedHarborRunReport]:
        result = _run_async_confirmation_slice(
            runner,
            baseline=opened.baseline,
            candidate=opened.candidate,
        )
        result.progress.require_protocol(protocol)
        if previous is not None:
            expected_previous = previous.progress.progress_digest
            if result.progress.previous_progress_digest != expected_previous:
                raise ValueError("paired slice did not extend the persisted confirmation progress")
        on_checkpoint(result.model_copy(deep=True))
        return StudySliceResult[PairedHarborSliceProgress, PairedHarborRunReport](
            checkpoint=result.progress,
            result=result.report,
        )

    sliced = lifecycle.run_slice(
        StudyPhase.CONFIRMATION_RUNNING,
        authorization.confirmation_run_id,
        _run,
        payload_digest=authorization.digest,
        configuration_digest=protocol.digest,
        resume_from=(_checkpoint_identity(previous.progress) if previous is not None else None),
        checkpoint_identity=_checkpoint_identity,
    )
    return PairedHarborSliceResult(progress=sliced.checkpoint, report=sliced.result)


def reconcile_harness_optimization_confirmation_slice(
    *,
    opened: OpenedHarnessOptimizationConfirmation,
    protocol: PairedHarborProtocol,
    lifecycle: StudyLifecycleController,
    authorization: ConfirmationRunningPayload,
    operation_id: str,
    generation_id: int,
    persisted: PairedHarborSliceResult,
) -> PairedHarborSliceResult:
    """Journal one caller-persisted paired checkpoint without dispatching another block."""
    _validate_confirmation_runtime_authorization(
        opened=opened,
        protocol=protocol,
        authorization=authorization,
        operation_id=operation_id,
        generation_id=generation_id,
    )
    frozen = PairedHarborSliceResult.model_validate(persisted.model_dump(mode="json"))
    frozen.progress.require_protocol(protocol)
    identity = StudyRunCheckpointIdentity(
        sequence=frozen.progress.slice_index - 1,
        checkpoint_digest=frozen.progress.progress_digest,
    )
    sliced = lifecycle.reconcile_slice(
        StudyPhase.CONFIRMATION_RUNNING,
        authorization.confirmation_run_id,
        lambda: StudySliceResult[PairedHarborSliceProgress, PairedHarborRunReport](
            checkpoint=frozen.progress,
            result=frozen.report,
        ),
        payload_digest=authorization.digest,
        configuration_digest=protocol.digest,
        resume_from=identity,
        checkpoint_identity=lambda progress: StudyRunCheckpointIdentity(
            sequence=progress.slice_index - 1,
            checkpoint_digest=progress.progress_digest,
        ),
    )
    return PairedHarborSliceResult(progress=sliced.checkpoint, report=sliced.result)


class HarnessOptimizationStudyCoordinator:
    """Advance one sealed study by exactly one phase or bounded execution slice."""

    def __init__(
        self,
        *,
        spec: HarnessOptimizationStudySpec,
        state_store: HarnessOptimizationStateStore,
        lifecycle: StudyLifecycleController,
        control_store: PartitionControlStore,
        artifact_publisher: StudyArtifactPublisher,
        runtime_factory: HarnessOptimizationRuntimeFactory,
    ) -> None:
        self._spec = HarnessOptimizationStudySpec.model_validate(spec.model_dump(mode="json"))
        self._state_store = state_store
        self._lifecycle = lifecycle
        self._control_store = control_store
        self._artifact_publisher = artifact_publisher
        self._runtime_factory = runtime_factory

    @classmethod
    def local(
        cls,
        spec: HarnessOptimizationStudySpec,
        *,
        state_dir: Path,
        resume: bool,
        runtime_factory: HarnessOptimizationRuntimeFactory | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> HarnessOptimizationStudyCoordinator:
        """Open the default single-host coordinator and local append-only evidence stores."""
        frozen = HarnessOptimizationStudySpec.model_validate(spec.model_dump(mode="json"))
        root = _ensure_private_directory(state_dir)
        state_store = LocalHarnessOptimizationStateStore(root / "coordinator")
        state_store.initialize(spec_digest=frozen.digest, resume=resume)
        evidence = LocalStudyEvidenceStore(
            root / "evidence",
            study_id=frozen.prepared.protocol.experiment_id,
            clock=clock,
        )
        journal = StudyJournalStore.create(
            root / "journal",
            study_id=frozen.prepared.protocol.experiment_id,
            publisher_configuration_digest=evidence.configuration_digest,
        )
        lifecycle = StudyLifecycleController(
            store=journal,
            publisher=evidence,
            artifact_verifier=evidence,
        )
        return cls(
            spec=frozen,
            state_store=state_store,
            lifecycle=lifecycle,
            control_store=PartitionControlStore(frozen.partition_control_dir),
            artifact_publisher=evidence,
            runtime_factory=runtime_factory or ProductionHarnessOptimizationRuntimeFactory(),
        )

    @property
    def current_phase(self) -> StudyPhase | None:
        """Return the externally reverified current lifecycle phase."""
        return self._lifecycle.current_phase

    @property
    def terminal(self) -> bool:
        """Return whether the study has a complete or stopped terminal record."""
        return self.current_phase in {StudyPhase.COMPLETE, StudyPhase.STOPPED}

    def advance(self) -> HarnessOptimizationAdvance:
        """Durably perform exactly one legal phase transition or bounded execution slice."""
        with self._state_store.locked():
            state = self._state_store.load()
            if state.spec_digest != self._spec.digest:
                raise ValueError("coordinator state belongs to a different study spec")
            previous = self._lifecycle.current_phase
            self._validate_current_payload(state, previous)
            if previous is None:
                return self._advance_preparation(state)
            if previous is StudyPhase.PREPARATION_PLANNED:
                return self._advance_roster(state, previous)
            if previous is StudyPhase.ROSTER_QUALIFIED:
                return self._advance_protocol(state, previous)
            if previous is StudyPhase.PROTOCOL_PUBLISHED:
                return self._advance_discovery_admission(state, previous)
            if previous is StudyPhase.DISCOVERY_RUNNING:
                return self._advance_discovery(state, previous)
            if previous is StudyPhase.CANDIDATE_FROZEN:
                return self._advance_candidate_publication(state, previous)
            if previous is StudyPhase.CANDIDATE_PUBLISHED:
                return self._advance_confirmation_open(state, previous)
            if previous is StudyPhase.CONFIRMATION_OPENED:
                return self._advance_confirmation_freeze(state, previous)
            if previous is StudyPhase.CONFIRMATION_FROZEN:
                return self._advance_confirmation_admission(state, previous)
            if previous is StudyPhase.CONFIRMATION_RUNNING:
                return self._advance_confirmation(state, previous)
            return HarnessOptimizationAdvance(
                kind=HarnessOptimizationAdvanceKind.TERMINAL,
                previous_phase=previous,
                current_phase=previous,
                terminal=True,
            )

    def _advance_preparation(
        self,
        state: HarnessOptimizationStudyState,
    ) -> HarnessOptimizationAdvance:
        payload = state.preparation or PreparationPlannedPayload(
            study_plan_digest=self._spec.digest,
            budget_policy_digest=self._spec.prepared.protocol.confirmation_budget_policy_digest,
            budget_binding_digest=self._spec.prepared.protocol.confirmation_budget_binding_digest,
            budget_ledger_identity=self._spec.prepared.protocol.confirmation_budget_ledger_identity,
            maximum_paid_cost_nano_usd=(
                self._spec.prepared.confirmation_budget.policy.hard_limit_nano_usd
            ),
        )
        if state.preparation is None:
            self._save(state.model_copy(update={"preparation": payload}, deep=True))
        self._lifecycle.publish(payload)
        return self._phase_advance(None, StudyPhase.PREPARATION_PLANNED)

    def _advance_roster(
        self,
        state: HarnessOptimizationStudyState,
        previous: StudyPhase,
    ) -> HarnessOptimizationAdvance:
        prepared = self._spec.prepared
        payload = state.roster_qualified or RosterQualifiedPayload(
            qualified_roster_digest=prepared.protocol.qualification_roster_digest,
            qualification_report_digest=self._spec.qualification_report_digest,
            execution_plan_digest=prepared.protocol.execution_plan.digest,
            qualified_task_count=len(prepared.qualification_roster.tasks),
        )
        if state.roster_qualified is None:
            self._save(state.model_copy(update={"roster_qualified": payload}, deep=True))
        self._lifecycle.publish(payload)
        return self._phase_advance(previous, StudyPhase.ROSTER_QUALIFIED)

    def _advance_protocol(
        self,
        state: HarnessOptimizationStudyState,
        previous: StudyPhase,
    ) -> HarnessOptimizationAdvance:
        prepared = self._spec.prepared
        publication = state.protocol_publication or self._artifact_publisher.publish_artifact(
            kind="harness-optimization-protocol",
            artifact_digest=prepared.protocol.digest,
            content=prepared.protocol.model_dump(mode="json"),
        )
        payload = state.protocol_published or ProtocolPublishedPayload(
            protocol_digest=prepared.protocol.digest,
            protocol_artifact_publication_digest=publication.digest,
            partition_manifest_digest=prepared.protocol.partition_manifest_digest,
            qualified_roster_digest=prepared.protocol.qualification_roster_digest,
            search_cost_binding_digest=prepared.protocol.search_cost_binding_digest,
            confirmation_budget_binding_digest=(
                prepared.protocol.confirmation_budget_binding_digest
            ),
        )
        if state.protocol_published is None:
            state = state.model_copy(
                update={"protocol_publication": publication, "protocol_published": payload},
                deep=True,
            )
            self._save(state)
        self._lifecycle.publish_protocol(payload, publication=publication)
        return self._phase_advance(previous, StudyPhase.PROTOCOL_PUBLISHED)

    def _advance_discovery_admission(
        self,
        state: HarnessOptimizationStudyState,
        previous: StudyPhase,
    ) -> HarnessOptimizationAdvance:
        protocol = self._spec.prepared.protocol
        payload = state.discovery_running or DiscoveryRunningPayload(
            protocol_digest=protocol.digest,
            search_configuration_digest=_canonical_digest(protocol.search.model_dump(mode="json")),
            search_run_id=self._spec.prepared.search_cost_binding.run_id,
        )
        if state.discovery_running is None:
            self._save(state.model_copy(update={"discovery_running": payload}, deep=True))
        self._lifecycle.publish(payload)
        return self._phase_advance(previous, StudyPhase.DISCOVERY_RUNNING)

    def _advance_discovery(
        self,
        state: HarnessOptimizationStudyState,
        previous: StudyPhase,
    ) -> HarnessOptimizationAdvance:
        prepared = self._spec.prepared
        checkpoint = state.search_checkpoint
        if (
            checkpoint is not None
            and checkpoint.completed_iteration == prepared.protocol.search.iterations
            and state.search_checkpoint_journaled_iteration != checkpoint.completed_iteration
        ):
            return self._run_discovery_slice(state, previous)
        if (
            checkpoint is None
            or checkpoint.completed_iteration < prepared.protocol.search.iterations
        ):
            return self._run_discovery_slice(state, previous)
        authorization = _required(state.discovery_running, "discovery authorization")
        frozen = state.frozen_candidate
        if frozen is None:
            frozen = freeze_harness_optimization_candidate(
                self._control_store,
                prepared=prepared,
                checkpoint=checkpoint,
                lifecycle=self._lifecycle,
                authorization=authorization,
            )
            state = state.model_copy(update={"frozen_candidate": frozen}, deep=True)
            self._save(state)
        report = state.search_budget_report or StudyBudgetReport.capture(
            _budget_authority(prepared)
        )
        publication = state.search_budget_publication or self._artifact_publisher.publish_artifact(
            kind="harness-optimization-search-budget",
            artifact_digest=report.digest,
            content=report.model_dump(mode="json"),
        )
        payload = state.candidate_frozen or CandidateFrozenPayload(
            protocol_digest=prepared.protocol.digest,
            candidate_execution_digest=frozen.candidate.execution_digest,
            search_checkpoint_digest="sha256:" + checkpoint.payload_sha256,
            search_configuration_digest=authorization.search_configuration_digest,
            search_cost_binding_digest=prepared.protocol.search_cost_binding_digest,
            search_cost_report_digest=report.digest,
            search_cost_report_publication_digest=publication.digest,
            champion_reconstruction_digest=_canonical_digest(
                frozen.candidate.model_dump(mode="json")
            ),
            candidate_freeze_record_digest=frozen.freeze_record.digest,
            completed_iterations=prepared.protocol.search.iterations,
        )
        if state.candidate_frozen is None:
            state = state.model_copy(
                update={
                    "search_budget_report": report,
                    "search_budget_publication": publication,
                    "candidate_frozen": payload,
                },
                deep=True,
            )
            self._save(state)
        published = self._lifecycle.publish_candidate_frozen(
            protocol_digest=prepared.protocol.digest,
            candidate=frozen.candidate,
            checkpoint=checkpoint,
            search_configuration_digest=authorization.search_configuration_digest,
            search_cost_binding_digest=prepared.protocol.search_cost_binding_digest,
            budget_authority=_budget_authority(prepared),
            search_cost_report=report,
            search_cost_report_publication=publication,
            freeze_record=frozen.freeze_record,
            completed_iterations=prepared.protocol.search.iterations,
        )
        if published != payload:
            raise RuntimeError("published candidate freeze differs from staged state")
        return self._phase_advance(previous, StudyPhase.CANDIDATE_FROZEN)

    def _run_discovery_slice(
        self,
        state: HarnessOptimizationStudyState,
        previous: StudyPhase,
    ) -> HarnessOptimizationAdvance:
        authorization = _required(state.discovery_running, "discovery authorization")
        reconciling_terminal = (
            state.search_checkpoint is not None
            and state.search_checkpoint.completed_iteration
            == self._spec.prepared.protocol.search.iterations
            and state.search_checkpoint_journaled_iteration
            != state.search_checkpoint.completed_iteration
        )
        runtime = self._runtime_factory.build_discovery(self._spec)
        working = state

        def _save_checkpoint(checkpoint: SearchCheckpoint) -> None:
            nonlocal working
            working = working.model_copy(
                update={"search_checkpoint": checkpoint, "proposal_batch_witness": None},
                deep=True,
            )
            self._save(working)

        def _save_witness(witness: SearchProposalBatchWitness) -> None:
            nonlocal working
            working = working.model_copy(update={"proposal_batch_witness": witness}, deep=True)
            self._save(working)

        try:
            sliced = run_harness_optimization_search_slice(
                self._spec.prepared.discovery_contract(),
                scorer=runtime.scorer,
                proposer=runtime.proposer,
                lifecycle=self._lifecycle,
                authorization=authorization,
                resume_from=state.search_checkpoint,
                resume_proposal_batch_witness=state.proposal_batch_witness,
                on_checkpoint=_save_checkpoint,
                on_proposal_batch_prepare=_save_witness,
                on_proposal_batch_witness=_save_witness,
            )
        finally:
            runtime.close()
        working = working.model_copy(
            update={
                "search_checkpoint": sliced.checkpoint,
                "search_checkpoint_journaled_iteration": sliced.checkpoint.completed_iteration,
                "proposal_batch_witness": None,
            },
            deep=True,
        )
        self._save(working)
        return HarnessOptimizationAdvance(
            kind=(
                HarnessOptimizationAdvanceKind.DISCOVERY_RECONCILED
                if reconciling_terminal
                else HarnessOptimizationAdvanceKind.DISCOVERY_SLICE
            ),
            previous_phase=previous,
            current_phase=previous,
            checkpoint_sequence=sliced.checkpoint.completed_iteration,
        )

    def _advance_candidate_publication(
        self,
        state: HarnessOptimizationStudyState,
        previous: StudyPhase,
    ) -> HarnessOptimizationAdvance:
        frozen = _required(state.frozen_candidate, "frozen candidate")
        source_digest = _canonical_digest(frozen.candidate.model_dump(mode="json"))
        publication = state.candidate_publication or self._artifact_publisher.publish_artifact(
            kind="harness-optimization-candidate",
            artifact_digest=source_digest,
            content=frozen.candidate.model_dump(mode="json"),
        )
        payload = state.candidate_published or CandidatePublishedPayload(
            protocol_digest=self._spec.prepared.protocol.digest,
            candidate_execution_digest=frozen.candidate.execution_digest,
            candidate_source_artifact_digest=source_digest,
            candidate_artifact_publication_digest=publication.digest,
        )
        if state.candidate_published is None:
            self._save(
                state.model_copy(
                    update={
                        "candidate_publication": publication,
                        "candidate_published": payload,
                    },
                    deep=True,
                )
            )
        published = self._lifecycle.publish_candidate_source(
            protocol_digest=self._spec.prepared.protocol.digest,
            candidate=frozen.candidate,
            publication=publication,
        )
        if published != payload:
            raise RuntimeError("published candidate source differs from staged state")
        return self._phase_advance(previous, StudyPhase.CANDIDATE_PUBLISHED)

    def _advance_confirmation_open(
        self,
        state: HarnessOptimizationStudyState,
        previous: StudyPhase,
    ) -> HarnessOptimizationAdvance:
        frozen = _required(state.frozen_candidate, "frozen candidate")
        authorization = _required(state.candidate_published, "candidate publication")
        opened = state.opened_confirmation
        if opened is None:
            opened = open_harness_optimization_confirmation(
                self._control_store,
                prepared=self._spec.prepared,
                frozen=frozen,
                lifecycle=self._lifecycle,
                authorization=authorization,
            )
        payload = state.confirmation_opened or ConfirmationOpenedPayload(
            protocol_digest=opened.protocol.digest,
            candidate_execution_digest=opened.candidate.execution_digest,
            candidate_freeze_record_digest=opened.freeze_record.digest,
            confirmation_partition_digest=_canonical_digest(
                opened.confirmation.model_dump(mode="json")
            ),
            confirmation_opening_record_digest=opened.confirmation.opening_record_digest,
            paired_design_digest=opened.design.digest,
            confirmation_task_count=len(opened.confirmation.tasks),
        )
        if state.confirmation_opened is None:
            self._save(
                state.model_copy(
                    update={"opened_confirmation": opened, "confirmation_opened": payload},
                    deep=True,
                )
            )
        self._lifecycle.publish(payload)
        return self._phase_advance(previous, StudyPhase.CONFIRMATION_OPENED)

    def _advance_confirmation_freeze(
        self,
        state: HarnessOptimizationStudyState,
        previous: StudyPhase,
    ) -> HarnessOptimizationAdvance:
        opened = _required(state.opened_confirmation, "opened confirmation")
        authorization = _required(state.confirmation_opened, "confirmation opening")
        protocol = state.paired_protocol or freeze_harness_optimization_harbor_protocol(
            opened,
            lifecycle=self._lifecycle,
            authorization=authorization,
        )
        payload = state.confirmation_frozen or ConfirmationFrozenPayload(
            protocol_digest=opened.protocol.digest,
            paired_protocol_digest=protocol.digest,
            budget_binding_digest=protocol.budget_binding_digest,
            create_rate_policy_digest=(opened.protocol.execution_plan.create_rate_policy_digest),
            slice_policy_digest=protocol.slice_policy_digest,
            planned_blocks=len(protocol.design.blocks),
            planned_arms=len(protocol.design.blocks) * 2,
        )
        if state.confirmation_frozen is None:
            self._save(
                state.model_copy(
                    update={"paired_protocol": protocol, "confirmation_frozen": payload},
                    deep=True,
                )
            )
        self._lifecycle.publish(payload)
        return self._phase_advance(previous, StudyPhase.CONFIRMATION_FROZEN)

    def _advance_confirmation_admission(
        self,
        state: HarnessOptimizationStudyState,
        previous: StudyPhase,
    ) -> HarnessOptimizationAdvance:
        protocol = _required(state.paired_protocol, "paired protocol")
        payload = state.confirmation_running or ConfirmationRunningPayload(
            paired_protocol_digest=protocol.digest,
            initial_run_state_digest=confirmation_initial_run_state_digest(
                protocol,
                operation_id=self._spec.confirmation_operation_id,
                generation_id=self._spec.confirmation_generation_id,
            ),
            slice_policy_digest=protocol.slice_policy_digest,
            confirmation_run_id=self._spec.confirmation_operation_id,
        )
        if state.confirmation_running is None:
            self._save(state.model_copy(update={"confirmation_running": payload}, deep=True))
        self._lifecycle.publish(payload)
        return self._phase_advance(previous, StudyPhase.CONFIRMATION_RUNNING)

    def _advance_confirmation(
        self,
        state: HarnessOptimizationStudyState,
        previous: StudyPhase,
    ) -> HarnessOptimizationAdvance:
        persisted = state.confirmation_slice
        if persisted is not None:
            sequence = persisted.progress.slice_index - 1
            if state.confirmation_journaled_sequence != sequence:
                reconciled = reconcile_harness_optimization_confirmation_slice(
                    opened=_required(state.opened_confirmation, "opened confirmation"),
                    protocol=_required(state.paired_protocol, "paired protocol"),
                    lifecycle=self._lifecycle,
                    authorization=_required(
                        state.confirmation_running,
                        "confirmation authorization",
                    ),
                    operation_id=self._spec.confirmation_operation_id,
                    generation_id=self._spec.confirmation_generation_id,
                    persisted=persisted,
                )
                self._save(
                    state.model_copy(
                        update={
                            "confirmation_slice": reconciled,
                            "confirmation_journaled_sequence": sequence,
                        },
                        deep=True,
                    )
                )
                return HarnessOptimizationAdvance(
                    kind=HarnessOptimizationAdvanceKind.CONFIRMATION_RECONCILED,
                    previous_phase=previous,
                    current_phase=previous,
                    checkpoint_sequence=sequence,
                )
        runner = self._runtime_factory.build_confirmation(
            self._spec,
            _required(state.paired_protocol, "paired protocol"),
        )
        recovered = _recover_async_confirmation_slice(
            runner,
            baseline=_required(state.opened_confirmation, "opened confirmation").baseline,
            candidate=_required(state.opened_confirmation, "opened confirmation").candidate,
        )
        if recovered is not None:
            if persisted is None or recovered.progress.slice_index > persisted.progress.slice_index:
                recovered_state = state.model_copy(
                    update={"confirmation_slice": recovered},
                    deep=True,
                )
                self._save(recovered_state)
                return self._advance_confirmation(recovered_state, previous)
            if recovered != persisted:
                raise ValueError("paired runner progress differs from the coordinator checkpoint")
        elif persisted is not None:
            raise ValueError(
                "coordinator confirmation checkpoint is missing from the paired runner"
            )
        if persisted is not None and persisted.report is not None:
            return self._advance_complete(state, previous, persisted.report)
        working = state

        def _save_checkpoint(result: PairedHarborSliceResult) -> None:
            nonlocal working
            working = working.model_copy(update={"confirmation_slice": result}, deep=True)
            self._save(working)

        result = run_harness_optimization_confirmation_slice(
            opened=_required(state.opened_confirmation, "opened confirmation"),
            protocol=_required(state.paired_protocol, "paired protocol"),
            runner=runner,
            lifecycle=self._lifecycle,
            authorization=_required(state.confirmation_running, "confirmation authorization"),
            operation_id=self._spec.confirmation_operation_id,
            generation_id=self._spec.confirmation_generation_id,
            resume_from=state.confirmation_slice,
            on_checkpoint=_save_checkpoint,
        )
        sequence = result.progress.slice_index - 1
        self._save(
            working.model_copy(
                update={
                    "confirmation_slice": result,
                    "confirmation_journaled_sequence": sequence,
                },
                deep=True,
            )
        )
        return HarnessOptimizationAdvance(
            kind=HarnessOptimizationAdvanceKind.CONFIRMATION_SLICE,
            previous_phase=previous,
            current_phase=previous,
            checkpoint_sequence=sequence,
        )

    def _advance_complete(
        self,
        state: HarnessOptimizationStudyState,
        previous: StudyPhase,
        report: PairedHarborRunReport,
    ) -> HarnessOptimizationAdvance:
        opened = _required(state.opened_confirmation, "opened confirmation")
        outcome = state.outcome or summarize_harness_optimization_outcome(opened, report)
        budget_report = state.final_budget_report or StudyBudgetReport.capture(
            _budget_authority(self._spec.prepared)
        )
        publication = state.final_budget_publication or self._artifact_publisher.publish_artifact(
            kind="harness-optimization-final-budget",
            artifact_digest=budget_report.digest,
            content=budget_report.model_dump(mode="json"),
        )
        payload = state.complete or _complete_payload(
            report=report,
            outcome=outcome,
            budget_report=budget_report,
            publication=publication,
        )
        if state.complete is None:
            self._save(
                state.model_copy(
                    update={
                        "outcome": outcome,
                        "final_budget_report": budget_report,
                        "final_budget_publication": publication,
                        "complete": payload,
                    },
                    deep=True,
                )
            )
        published = self._lifecycle.publish_complete(
            paired_protocol_digest=report.protocol_digest,
            paired_report_digest=report.digest,
            outcome_digest=_canonical_digest(outcome.model_dump(mode="json")),
            budget_authority=_budget_authority(self._spec.prepared),
            budget_report=budget_report,
            budget_report_publication=publication,
        )
        if published != payload:
            raise RuntimeError("published study result differs from staged state")
        return self._phase_advance(previous, StudyPhase.COMPLETE, terminal=True)

    def _validate_current_payload(
        self,
        state: HarnessOptimizationStudyState,
        phase: StudyPhase | None,
    ) -> None:
        if phase is None:
            return
        payload = state.payload_for_phase(phase)
        if payload is None:
            raise ValueError(f"coordinator state lacks current {phase.value} payload")
        record = self._lifecycle.records[-1]
        if record.commitment.payload_digest != payload.digest:
            raise ValueError("coordinator state payload differs from the lifecycle journal")

    def _save(self, state: HarnessOptimizationStudyState) -> None:
        self._state_store.save(state)

    @staticmethod
    def _phase_advance(
        previous: StudyPhase | None,
        current: StudyPhase,
        *,
        terminal: bool = False,
    ) -> HarnessOptimizationAdvance:
        return HarnessOptimizationAdvance(
            kind=HarnessOptimizationAdvanceKind.PHASE,
            previous_phase=previous,
            current_phase=current,
            terminal=terminal,
        )


class HarnessOptimizationRehearsal(BaseModel):
    """Deterministic zero-effect validation result for a frozen study spec."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rehearsal_version: Literal["1"] = "1"
    spec_digest: str = Field(pattern=_DIGEST_PATTERN)
    protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    backend: HarborEnvironmentBackend
    phase_order: tuple[StudyPhase, ...]
    confirmation_block_count: StrictInt = Field(ge=1)
    would_publish_complete: Literal[False] = False


class HarnessOptimizationRehearsalFactory(Protocol):
    """Build a deterministic rehearsal without constructing execution runtimes."""

    def build(self, spec: HarnessOptimizationStudySpec) -> HarnessOptimizationRehearsal: ...


class DeterministicHarnessOptimizationRehearsalFactory:
    """Validate frozen identities without provider, Harbor, sandbox, or credential effects."""

    def build(self, spec: HarnessOptimizationStudySpec) -> HarnessOptimizationRehearsal:
        """Return a pure manifest that deliberately stops before a complete publication."""
        frozen = HarnessOptimizationStudySpec.model_validate(spec.model_dump(mode="json"))
        return HarnessOptimizationRehearsal(
            spec_digest=frozen.digest,
            protocol_digest=frozen.prepared.protocol.digest,
            backend=frozen.prepared.protocol.execution_plan.environment_backend,
            phase_order=(
                StudyPhase.PREPARATION_PLANNED,
                StudyPhase.ROSTER_QUALIFIED,
                StudyPhase.PROTOCOL_PUBLISHED,
                StudyPhase.DISCOVERY_RUNNING,
                StudyPhase.CANDIDATE_FROZEN,
                StudyPhase.CANDIDATE_PUBLISHED,
                StudyPhase.CONFIRMATION_OPENED,
                StudyPhase.CONFIRMATION_FROZEN,
                StudyPhase.CONFIRMATION_RUNNING,
            ),
            confirmation_block_count=(
                sum(item.count for item in frozen.prepared.protocol.discovery.confirmation_strata)
                * sum(item.attempts for item in frozen.prepared.protocol.confirmation.panel)
            ),
        )


def _validate_confirmation_runtime_authorization(
    *,
    opened: OpenedHarnessOptimizationConfirmation,
    protocol: PairedHarborProtocol,
    authorization: ConfirmationRunningPayload,
    operation_id: str,
    generation_id: int,
) -> None:
    if (
        protocol.preopen_commitment != opened.confirmation_commitment
        or protocol.confirmation != opened.confirmation
        or protocol.design != opened.design
        or protocol.baseline_execution_digest != opened.baseline.execution_digest
        or protocol.candidate_execution_digest != opened.candidate.execution_digest
    ):
        raise ValueError("paired protocol differs from the opened optimization study")
    if (
        authorization.paired_protocol_digest != protocol.digest
        or authorization.slice_policy_digest != protocol.slice_policy_digest
        or authorization.confirmation_run_id != operation_id
        or authorization.initial_run_state_digest
        != confirmation_initial_run_state_digest(
            protocol,
            operation_id=operation_id,
            generation_id=generation_id,
        )
    ):
        raise ValueError("confirmation authorization differs from the frozen paired run")


def _run_async_confirmation_slice(
    runner: PairedHarborSliceRunner,
    *,
    baseline: HarnessDoc,
    candidate: HarnessDoc,
) -> PairedHarborSliceResult:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(runner.run_slice(baseline=baseline, candidate=candidate))
    raise RuntimeError(
        "synchronous study coordination cannot run inside an active event loop; invoke it from "
        "a worker thread or process"
    )


def _recover_async_confirmation_slice(
    runner: PairedHarborSliceRunner,
    *,
    baseline: HarnessDoc,
    candidate: HarnessDoc,
) -> PairedHarborSliceResult | None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(runner.recover_persisted_slice(baseline=baseline, candidate=candidate))
    raise RuntimeError(
        "synchronous study coordination cannot run inside an active event loop; invoke it from "
        "a worker thread or process"
    )


def _freeze_optional_slice(
    value: PairedHarborSliceResult | None,
) -> PairedHarborSliceResult | None:
    if value is None:
        return None
    return PairedHarborSliceResult.model_validate(value.model_dump(mode="json"))


def _complete_payload(
    *,
    report: PairedHarborRunReport,
    outcome: HarnessOptimizationOutcome,
    budget_report: StudyBudgetReport,
    publication: StudyArtifactPublication,
) -> StudyCompletePayload:
    state = budget_report.audit_state
    snapshot = state.snapshot
    return StudyCompletePayload(
        paired_protocol_digest=report.protocol_digest,
        paired_report_digest=report.digest,
        outcome_digest=_canonical_digest(outcome.model_dump(mode="json")),
        budget_policy_digest=state.policy_digest,
        budget_ledger_identity=state.ledger_identity,
        ledger_head_sequence=state.ledger_head_sequence,
        ledger_head_digest=state.ledger_head_digest,
        budget_report_digest=budget_report.digest,
        budget_report_publication_digest=publication.digest,
        cumulative_paid_cost_nano_usd=snapshot.charged_nano_usd,
        outstanding_reserved_cost_nano_usd=snapshot.reserved_nano_usd,
        budget_hard_limit_nano_usd=snapshot.hard_limit_nano_usd,
        budget_remaining_nano_usd=snapshot.remaining_nano_usd,
        budget_breached=snapshot.breached,
    )


def _budget_authority(prepared: PreparedHarnessOptimizationStudy) -> BudgetLedgerAuthority:
    runtime = prepared.confirmation_budget
    return BudgetLedgerAuthority(
        ledger_path=runtime.ledger_path,
        ledger_identity=runtime.ledger_identity,
        policy=runtime.policy,
    )


_ValueT = TypeVar("_ValueT")


def _required(value: _ValueT | None, label: str) -> _ValueT:
    if value is None:
        raise ValueError(f"coordinator state lacks {label}")
    return value


def _canonical_digest(value: JsonValue) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: JsonValue) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _is_digest(value: str) -> bool:
    hexadecimal = value.removeprefix("sha256:")
    return (
        value.startswith("sha256:")
        and len(hexadecimal) == 64
        and all(character in "0123456789abcdef" for character in hexadecimal)
    )


def _ensure_private_directory(path: Path) -> Path:
    resolved = Path(os.path.abspath(path.expanduser()))
    resolved.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIRECTORY_MODE)
    metadata = os.lstat(resolved)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"study state directory must be a real directory: {resolved}")
    if metadata.st_uid != os.getuid():
        raise ValueError(f"study state directory must be owned by the current uid: {resolved}")
    if stat.S_IMODE(metadata.st_mode) != _PRIVATE_DIRECTORY_MODE:
        raise ValueError(f"study state directory must have mode 0700: {resolved}")
    return resolved


def _require_private_regular_file(path: Path) -> None:
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"study state file must be a regular file: {path}")
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != _PRIVATE_FILE_MODE:
        raise ValueError(f"study state file must be current-uid mode 0600: {path}")
    if metadata.st_nlink != 1:
        raise ValueError(f"study state file must have exactly one link: {path}")


def _read_optional_private_regular_file(path: Path) -> bytes | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"study state file must be a regular file: {path}")
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != _PRIVATE_FILE_MODE:
            raise ValueError(f"study state file must be current-uid mode 0600: {path}")
        if metadata.st_nlink != 1:
            raise ValueError(f"study state file must have exactly one link: {path}")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _atomic_write_private_json(path: Path, value: JsonValue) -> None:
    payload = _canonical_json_bytes(value)
    temporary: Path | None = path.parent / f".tmp-{path.name}-{uuid.uuid4().hex}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        _PRIVATE_FILE_MODE,
    )
    try:
        os.fchmod(descriptor, _PRIVATE_FILE_MODE)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        _require_private_regular_file(path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _atomic_publish_private_json(path: Path, value: JsonValue) -> bool:
    """Publish one complete file without replacement and report whether this call won."""
    _recover_atomic_publication_aliases(path)
    if path.exists():
        return False
    payload = _canonical_json_bytes(value)
    temporary: Path | None = path.parent / (
        f".publish-{path.name}-pid-{os.getpid()}-{uuid.uuid4().hex}"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        _PRIVATE_FILE_MODE,
    )
    try:
        os.fchmod(descriptor, _PRIVATE_FILE_MODE)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        won_publication = True
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            won_publication = False
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        temporary = None
        # Another publisher can complete its link while this call loses the no-replace race, then
        # remain paused before removing its staging alias. Recover aliases before enforcing the
        # single-link invariant so identical concurrent publication remains idempotent for every
        # valid interleaving. Removing an alias after its final link is safe for the paused winner.
        _recover_atomic_publication_aliases(path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        _require_private_regular_file(path)
        return won_publication
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _recover_atomic_publication_aliases(path: Path) -> None:
    """Remove recoverable staging files without disturbing a live pre-link publisher."""
    try:
        published = os.lstat(path)
    except FileNotFoundError:
        published = None
    if published is not None and (
        not stat.S_ISREG(published.st_mode) or stat.S_ISLNK(published.st_mode)
    ):
        raise ValueError(f"study evidence file must be a regular file: {path}")
    prefix = f".publish-{path.name}-"
    changed = False
    for candidate in path.parent.glob(prefix + "*"):
        try:
            staged = os.lstat(candidate)
        except FileNotFoundError:
            continue
        published_alias = published is not None and (staged.st_dev, staged.st_ino) == (
            published.st_dev,
            published.st_ino,
        )
        if not published_alias:
            owner_pid = _publication_staging_owner_pid(path, candidate)
            if owner_pid is None or _process_is_alive(owner_pid):
                continue
            # A dead pre-link publisher cannot run its finally block. Only remove the exact private
            # regular file shape created above; malformed or hard-linked staging evidence remains a
            # fail-closed integrity error rather than becoming a cleanup primitive.
            _require_private_regular_file(candidate)
        try:
            candidate.unlink()
        except FileNotFoundError:
            # Concurrent recovery of the same post-link alias is already the desired state.
            continue
        changed = True
    if changed:
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)


def _publication_staging_owner_pid(path: Path, candidate: Path) -> int | None:
    """Return the canonical owner PID embedded in a publication staging filename."""
    prefix = f".publish-{path.name}-pid-"
    if not candidate.name.startswith(prefix):
        return None
    suffix = candidate.name.removeprefix(prefix)
    pid_text, separator, nonce = suffix.partition("-")
    if (
        not separator
        or not pid_text.isdecimal()
        or pid_text.startswith("0")
        or len(nonce) != 32
        or any(character not in "0123456789abcdef" for character in nonce)
    ):
        return None
    pid = int(pid_text)
    return pid if 0 < pid <= 2_147_483_647 else None


def _process_is_alive(pid: int) -> bool:
    """Conservatively report whether a staging owner may still publish its file."""
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _read_optional_canonical_model(
    path: Path,
    model: type[_ModelT],
) -> _ModelT | None:
    payload = _read_optional_private_regular_file(path)
    if payload is None:
        return None
    value = model.model_validate_json(payload)
    if payload != _canonical_json_bytes(value.model_dump(mode="json")):
        raise ValueError(f"study evidence file is not canonical: {path.name}")
    return value
