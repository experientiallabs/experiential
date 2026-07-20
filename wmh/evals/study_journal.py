"""Externally witnessed, crash-safe phase journal for optimization studies."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictInt, model_validator

from wmh.core.file_lease import exclusive_posix_directory_lease
from wmh.core.text import validate_durable_text

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_JOURNAL_VERSION: Literal["1"] = "1"
_RECORD_VERSION: Literal["1"] = "1"
_GENESIS_FILE = "study-journal.json"
_PENDING_FILE = "pending-phase.json"
_MAX_RECORD_BYTES = 64 * 1024
_RECORD_PATTERN = re.compile(r"^(?P<sequence>[0-9]{3})-(?P<phase>[a-z_]+)\.json$")
_RUN_CLAIM_PATTERN = re.compile(r"^run-claim-(?P<phase>[a-z_]+)\.json$")
_RUN_CHECKPOINT_PATTERN = re.compile(
    r"^run-checkpoint-(?P<phase>[a-z_]+)-(?P<sequence>[0-9]{8})\.json$"
)
_RUN_SLICE_INTENT_PATTERN = re.compile(
    r"^run-slice-intent-(?P<phase>[a-z_]+)-(?P<sequence>[0-9]{8})\.json$"
)
_TEMPORARY_PATTERN = re.compile(r"^\.tmp-(?P<target>.+)-(?P<nonce>[0-9a-f]{32})$")
_ResultT = TypeVar("_ResultT")
MAX_STUDY_RUN_CHECKPOINT_SEQUENCE = 99_999_999


class StudyPhase(StrEnum):
    """The only legal chronology for a sealed optimization study."""

    PREPARATION_PLANNED = "preparation_planned"
    ROSTER_QUALIFIED = "roster_qualified"
    PROTOCOL_PUBLISHED = "protocol_published"
    DISCOVERY_RUNNING = "discovery_running"
    CANDIDATE_FROZEN = "candidate_frozen"
    CANDIDATE_PUBLISHED = "candidate_published"
    CONFIRMATION_OPENED = "confirmation_opened"
    CONFIRMATION_FROZEN = "confirmation_frozen"
    CONFIRMATION_RUNNING = "confirmation_running"
    COMPLETE = "complete"
    STOPPED = "stopped"


STUDY_PHASES: tuple[StudyPhase, ...] = tuple(StudyPhase)
SUCCESSFUL_STUDY_PHASES: tuple[StudyPhase, ...] = (
    StudyPhase.PREPARATION_PLANNED,
    StudyPhase.ROSTER_QUALIFIED,
    StudyPhase.PROTOCOL_PUBLISHED,
    StudyPhase.DISCOVERY_RUNNING,
    StudyPhase.CANDIDATE_FROZEN,
    StudyPhase.CANDIDATE_PUBLISHED,
    StudyPhase.CONFIRMATION_OPENED,
    StudyPhase.CONFIRMATION_FROZEN,
    StudyPhase.CONFIRMATION_RUNNING,
    StudyPhase.COMPLETE,
)
_TERMINAL_STUDY_PHASES = frozenset({StudyPhase.COMPLETE, StudyPhase.STOPPED})


class StudyJournalGenesis(BaseModel):
    """Immutable identity of one journal, independent of its host path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    journal_version: Literal["1"] = _JOURNAL_VERSION
    study_id: str = Field(min_length=1, max_length=512)
    publisher_configuration_digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _validate_study_id(self) -> Self:
        if self.study_id != self.study_id.strip():
            raise ValueError("study_id cannot have surrounding whitespace")
        validate_durable_text(self.study_id, field="study journal id")
        return self

    @property
    def digest(self) -> str:
        """Return the path-independent journal genesis identity."""
        return _canonical_digest(self.model_dump(mode="json"))


class StudyPhaseCommitment(BaseModel):
    """Hash-chain link published before its phase is accepted locally."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    journal_genesis_digest: str = Field(pattern=_DIGEST_PATTERN)
    study_id: str = Field(min_length=1, max_length=512)
    sequence: StrictInt = Field(ge=0)
    phase: StudyPhase
    previous_record_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    payload_digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _validate_position(self) -> Self:
        if self.study_id != self.study_id.strip():
            raise ValueError("study_id cannot have surrounding whitespace")
        validate_durable_text(self.study_id, field="study journal id")
        if self.sequence == 0 and self.previous_record_digest is not None:
            raise ValueError("first study phase cannot name a previous record")
        if self.sequence > 0 and self.previous_record_digest is None:
            raise ValueError("later study phase must name its previous record")
        return self

    @property
    def digest(self) -> str:
        """Return the exact content committed to the external publisher."""
        return _canonical_digest(self.model_dump(mode="json"))


class ExternalPublicationReceipt(BaseModel):
    """Nonsecret locator and proof returned by an append-only publication adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    commitment_digest: str = Field(pattern=_DIGEST_PATTERN)
    publisher: str = Field(min_length=1, max_length=256)
    publication_id: str = Field(min_length=1, max_length=2_048)
    immutable_locator: str = Field(min_length=1, max_length=4_096)
    published_at: datetime
    evidence: dict[str, JsonValue]

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        for field in ("publisher", "publication_id", "immutable_locator"):
            value = getattr(self, field)
            if value != value.strip():
                raise ValueError(f"publication {field} cannot have surrounding whitespace")
            validate_durable_text(value, field=f"publication {field}")
        if self.published_at.tzinfo is None or self.published_at.utcoffset() is None:
            raise ValueError("publication timestamp must be timezone-aware")
        return self

    @property
    def digest(self) -> str:
        """Return the canonical identity of the externally verifiable receipt."""
        return _canonical_digest(self.model_dump(mode="json"))


class StudyPhaseRecord(BaseModel):
    """One externally witnessed phase and its self-validating local record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_version: Literal["1"] = _RECORD_VERSION
    commitment: StudyPhaseCommitment
    commitment_digest: str = Field(pattern=_DIGEST_PATTERN)
    publication: ExternalPublicationReceipt
    record_digest: str = Field(pattern=_DIGEST_PATTERN)

    @classmethod
    def create(
        cls,
        *,
        commitment: StudyPhaseCommitment,
        publication: ExternalPublicationReceipt,
    ) -> StudyPhaseRecord:
        """Create a record only from a receipt for the exact commitment."""
        if publication.commitment_digest != commitment.digest:
            raise ValueError("publication receipt names a different commitment")
        payload = {
            "record_version": _RECORD_VERSION,
            "commitment": commitment.model_dump(mode="json"),
            "commitment_digest": commitment.digest,
            "publication": publication.model_dump(mode="json"),
        }
        return cls(**payload, record_digest=_canonical_digest(payload))

    @model_validator(mode="after")
    def _validate_record(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"record_digest"})
        if self.record_digest != _canonical_digest(payload):
            raise ValueError("study record digest is inconsistent")
        if self.commitment_digest != self.commitment.digest:
            raise ValueError("study record commitment digest is inconsistent")
        if self.publication.commitment_digest != self.commitment_digest:
            raise ValueError("publication receipt names a different commitment")
        return self

    @property
    def digest(self) -> str:
        """Return the chain identity used by the next phase."""
        return self.record_digest


class StudyRunClaim(BaseModel):
    """Durable one-run identity admitted under an exact phase authorization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_version: Literal["1"] = "1"
    journal_genesis_digest: str = Field(pattern=_DIGEST_PATTERN)
    study_id: str = Field(min_length=1, max_length=512)
    phase: StudyPhase
    authorization_payload_digest: str = Field(pattern=_DIGEST_PATTERN)
    run_id: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def _validate_claim(self) -> Self:
        for field in ("study_id", "run_id"):
            value = getattr(self, field)
            if value != value.strip():
                raise ValueError(f"study run {field} cannot have surrounding whitespace")
            validate_durable_text(value, field=f"study run {field}")
        return self


class StudyRunCheckpointIdentity(BaseModel):
    """Path-free identity of one caller-persisted resumable checkpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: StrictInt = Field(ge=0, le=MAX_STUDY_RUN_CHECKPOINT_SEQUENCE)
    checkpoint_digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _reject_boolean_sequence(self) -> Self:
        if isinstance(self.sequence, bool):
            raise ValueError("study run checkpoint sequence cannot be boolean")
        return self


class StudyRunCheckpointRecord(BaseModel):
    """Append-only binding from one admitted run to its latest durable checkpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_version: Literal["1"] = "1"
    journal_genesis_digest: str = Field(pattern=_DIGEST_PATTERN)
    study_id: str = Field(min_length=1, max_length=512)
    phase: StudyPhase
    authorization_payload_digest: str = Field(pattern=_DIGEST_PATTERN)
    run_id: str = Field(min_length=1, max_length=512)
    configuration_digest: str = Field(pattern=_DIGEST_PATTERN)
    checkpoint: StudyRunCheckpointIdentity
    previous_record_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    record_digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _validate_record(self) -> Self:
        for field in ("study_id", "run_id"):
            value = getattr(self, field)
            if value != value.strip():
                raise ValueError(f"study run checkpoint {field} cannot have surrounding whitespace")
            validate_durable_text(value, field=f"study run checkpoint {field}")
        payload = self.model_dump(mode="json", exclude={"record_digest"})
        if self.record_digest != _canonical_digest(payload):
            raise ValueError("study run checkpoint record digest is inconsistent")
        return self

    @property
    def digest(self) -> str:
        """Return the append-only record identity used by its successor."""
        return self.record_digest


class StudyRunSliceIntentRecord(BaseModel):
    """Immutable pre-dispatch authority for exactly one run checkpoint sequence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_version: Literal["1"] = "1"
    journal_genesis_digest: str = Field(pattern=_DIGEST_PATTERN)
    study_id: str = Field(min_length=1, max_length=512)
    phase: StudyPhase
    authorization_payload_digest: str = Field(pattern=_DIGEST_PATTERN)
    run_id: str = Field(min_length=1, max_length=512)
    configuration_digest: str = Field(pattern=_DIGEST_PATTERN)
    checkpoint_sequence: StrictInt = Field(
        ge=0,
        le=MAX_STUDY_RUN_CHECKPOINT_SEQUENCE,
    )
    previous_checkpoint_record_digest: str | None = Field(
        default=None,
        pattern=_DIGEST_PATTERN,
    )
    intent_digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _validate_intent(self) -> Self:
        for field in ("study_id", "run_id"):
            value = getattr(self, field)
            if value != value.strip():
                raise ValueError(
                    f"study run slice intent {field} cannot have surrounding whitespace"
                )
            validate_durable_text(value, field=f"study run slice intent {field}")
        if self.checkpoint_sequence == 0 and self.previous_checkpoint_record_digest is not None:
            raise ValueError("first study run slice intent cannot name a previous checkpoint")
        if self.checkpoint_sequence > 0 and self.previous_checkpoint_record_digest is None:
            raise ValueError("later study run slice intent must name its previous checkpoint")
        payload = self.model_dump(mode="json", exclude={"intent_digest"})
        if self.intent_digest != _canonical_digest(payload):
            raise ValueError("study run slice intent digest is inconsistent")
        return self

    @property
    def digest(self) -> str:
        """Return the canonical identity of this pre-dispatch authority."""
        return self.intent_digest


class ExternalCommitmentPublisher(Protocol):
    """Idempotent, fork-rejecting adapter to an external append-only channel."""

    @property
    def configuration_digest(self) -> str:
        """Return the nonsecret identity of the publisher and immutable channel."""
        ...

    def publish(self, commitment: StudyPhaseCommitment) -> ExternalPublicationReceipt:
        """Publish or recover this journal-sequence slot, rejecting another digest."""
        ...

    def verify(
        self,
        commitment: StudyPhaseCommitment,
        receipt: ExternalPublicationReceipt,
    ) -> None:
        """Raise unless the external channel still proves this exact publication."""
        ...

    def verify_chain_head(
        self,
        genesis: StudyJournalGenesis,
        records: tuple[StudyPhaseRecord, ...],
        pending: StudyPhaseCommitment | None,
    ) -> None:
        """Raise unless the channel has this exact chain and optional next commitment."""
        ...


class StudyJournalStore:
    """Path-bound durable storage for one externally witnessed phase chain."""

    def __init__(
        self,
        directory: str | Path,
        *,
        study_id: str,
        publisher_configuration_digest: str,
    ) -> None:
        self._directory = Path(os.path.abspath(Path(directory).expanduser()))
        directory_descriptor = _open_private_directory(self._directory)
        with _managed_descriptor(directory_descriptor):
            self._directory_identity = _private_directory_identity(os.fstat(directory_descriptor))
            genesis_payload = _read_regular_file_at(directory_descriptor, _GENESIS_FILE)
            self._genesis = StudyJournalGenesis.model_validate_json(genesis_payload)
            if genesis_payload != _canonical_json_bytes(self._genesis.model_dump(mode="json")):
                raise ValueError("study journal genesis is not canonical")
            _require_directory_binding(self._directory, self._directory_identity)
            _require_genesis_binding(self, directory_descriptor)
            _require_directory_binding(self._directory, self._directory_identity)
        if self._genesis.study_id != study_id:
            raise ValueError("study journal belongs to a different study")
        if self._genesis.publisher_configuration_digest != publisher_configuration_digest:
            raise ValueError("study journal belongs to a different publisher configuration")

    @classmethod
    def create(
        cls,
        directory: str | Path,
        *,
        study_id: str,
        publisher_configuration_digest: str,
    ) -> StudyJournalStore:
        """Create or reopen the sole journal genesis for one directory."""
        path = Path(os.path.abspath(Path(directory).expanduser()))
        genesis = StudyJournalGenesis(
            study_id=study_id,
            publisher_configuration_digest=publisher_configuration_digest,
        )
        _validate_leaf_name(path.name)
        parent_descriptor = _open_parent_directory(path)
        with _managed_descriptor(parent_descriptor):
            try:
                os.mkdir(path.name, mode=0o700, dir_fd=parent_descriptor)
            except FileExistsError:
                pass
            directory_descriptor = _open_private_directory_at(
                parent_descriptor,
                path.name,
            )
            with _managed_descriptor(directory_descriptor):
                directory_identity = _private_directory_identity(os.fstat(directory_descriptor))
                _require_directory_binding(path, directory_identity)
                os.fsync(parent_descriptor)
                _publish_regular_file_once_at(
                    directory_descriptor,
                    _GENESIS_FILE,
                    _canonical_json_bytes(genesis.model_dump(mode="json")),
                )
                _require_directory_binding(path, directory_identity)
                store = cls(
                    path,
                    study_id=study_id,
                    publisher_configuration_digest=publisher_configuration_digest,
                )
                if store._directory_identity != directory_identity:
                    raise OSError("study journal directory was replaced during create")
                return store

    @property
    def directory(self) -> Path:
        """Return the host-private journal directory."""
        return self._directory

    @property
    def genesis(self) -> StudyJournalGenesis:
        """Return the path-independent journal identity."""
        return self._genesis

    @contextmanager
    def locked(self) -> Iterator[int]:
        """Hold the store lease and pin every operation to the opened directory."""
        directory_descriptor = _open_private_directory(
            self._directory,
            expected_identity=self._directory_identity,
        )
        with _managed_descriptor(directory_descriptor):
            with exclusive_posix_directory_lease(
                directory_descriptor,
                unsupported_error=RuntimeError("study journal requires POSIX file locking"),
                irregular_directory_error=OSError(
                    "study journal lease must name its pinned directory"
                ),
                contention_error=RuntimeError("study journal is already locked"),
            ):
                _require_directory_binding(self._directory, self._directory_identity)
                _require_genesis_binding(self, directory_descriptor)
                try:
                    yield directory_descriptor
                except BaseException as primary_error:
                    try:
                        _require_store_binding(self, directory_descriptor)
                    except OSError as binding_error:
                        primary_error.add_note(
                            f"study journal binding also failed: {binding_error}"
                        )
                    raise
                else:
                    _require_store_binding(self, directory_descriptor)


def load_study_journal(
    store: StudyJournalStore,
    *,
    publisher: ExternalCommitmentPublisher,
) -> tuple[StudyPhaseRecord, ...]:
    """Load the complete local chain after revalidating every external proof."""
    with store.locked() as directory_descriptor:
        records = _load_records_locked(store, directory_descriptor)
        pending = _load_pending_locked(store, directory_descriptor, records)
        _validate_publisher(store, publisher)
        _verify_external_chain_locked(
            store,
            directory_descriptor,
            publisher,
            records,
            pending,
        )
        return records


def append_study_phase(
    store: StudyJournalStore,
    *,
    phase: StudyPhase,
    payload_digest: str,
    publisher: ExternalCommitmentPublisher,
) -> StudyPhaseRecord:
    """Publish then durably append exactly the next phase, idempotently on restart."""
    return append_study_phase_derived(
        store,
        phase=phase,
        derive_payload_digest=lambda _records: payload_digest,
        publisher=publisher,
    )


def append_study_phase_derived(
    store: StudyJournalStore,
    *,
    phase: StudyPhase,
    derive_payload_digest: Callable[[tuple[StudyPhaseRecord, ...]], str],
    publisher: ExternalCommitmentPublisher,
) -> StudyPhaseRecord:
    """Derive evidence and append its phase under one journal lease."""
    requested_phase = StudyPhase(phase)
    _validate_publisher(store, publisher)
    with store.locked() as directory_descriptor:
        records = _load_records_locked(store, directory_descriptor)
        pending, pending_is_complete = _load_pending_locked(
            store,
            directory_descriptor,
            records,
        )
        if pending_is_complete:
            _remove_file_durably_at(directory_descriptor, _PENDING_FILE)
            pending = None
            pending_is_complete = False
        existing = next(
            (record for record in records if record.commitment.phase is requested_phase),
            None,
        )
        allowed = _allowed_next_phases(records) if existing is None else ()
        if existing is None and requested_phase not in allowed:
            if not allowed:
                raise ValueError("study journal is terminal and cannot accept another phase")
            expected = ", ".join(phase.value for phase in allowed)
            raise ValueError(f"next study phase must be one of: {expected}")
        if existing is None and requested_phase is not StudyPhase.STOPPED and records:
            checkpoints_by_phase = _load_run_checkpoints_locked(store, directory_descriptor)
            intents_by_phase = _load_run_slice_intents_locked(
                store,
                directory_descriptor,
                checkpoints_by_phase=checkpoints_by_phase,
            )
            if any(
                len(intents) > len(checkpoints_by_phase.get(intent_phase, ()))
                for intent_phase, intents in intents_by_phase.items()
            ):
                raise ValueError(
                    "study phase cannot advance with an ambiguous durable slice intent"
                )

        _verify_external_chain_locked(
            store,
            directory_descriptor,
            publisher,
            records,
            (pending, pending_is_complete),
        )
        payload_digest = derive_payload_digest(records)
        if not _is_digest(payload_digest):
            raise ValueError("study phase payload_digest must be a canonical SHA-256 digest")
        if existing is not None:
            if existing.commitment.payload_digest != payload_digest:
                raise ValueError("study phase is already committed with a different payload")
            return existing

        commitment = StudyPhaseCommitment(
            journal_genesis_digest=store.genesis.digest,
            study_id=store.genesis.study_id,
            sequence=len(records),
            phase=requested_phase,
            previous_record_digest=records[-1].digest if records else None,
            payload_digest=payload_digest,
        )
        if pending is None:
            _publish_regular_file_once_at(
                directory_descriptor,
                _PENDING_FILE,
                _canonical_json_bytes(commitment.model_dump(mode="json")),
            )
        elif pending != commitment:
            raise ValueError("pending commitment fixes a different payload or chain position")
        publication = ExternalPublicationReceipt.model_validate(
            publisher.publish(commitment).model_dump(mode="json")
        )
        _validate_publisher(store, publisher)
        if publication.commitment_digest != commitment.digest:
            raise ValueError("publication receipt names a different commitment")
        publisher.verify(commitment, publication)
        _validate_publisher(store, publisher)
        if records and publication.published_at < records[-1].publication.published_at:
            raise ValueError("study journal publication timestamps move backwards")
        _require_store_binding(store, directory_descriptor)
        record = StudyPhaseRecord.create(commitment=commitment, publication=publication)
        publisher.verify_chain_head(store.genesis, (*records, record), None)
        _validate_publisher(store, publisher)
        _require_store_binding(store, directory_descriptor)
        _publish_regular_file_once_at(
            directory_descriptor,
            _record_name(len(records), requested_phase),
            _canonical_json_bytes(record.model_dump(mode="json")),
        )
        persisted = _load_records_locked(store, directory_descriptor)
        if len(persisted) != len(records) + 1 or persisted[-1] != record:
            raise RuntimeError("study journal append did not persist the exact phase record")
        persisted_pending, pending_is_complete = _load_pending_locked(
            store,
            directory_descriptor,
            persisted,
        )
        if persisted_pending != commitment or not pending_is_complete:
            raise RuntimeError("study journal pending commitment changed during append")
        _remove_file_durably_at(directory_descriptor, _PENDING_FILE)
        return record


def claim_study_run(
    store: StudyJournalStore,
    *,
    phase: StudyPhase,
    authorization_payload_digest: str,
    run_id: str,
    publisher: ExternalCommitmentPublisher,
    resume: bool,
) -> StudyRunClaim:
    """Claim one phase run once, admitting later calls only as exact resumes."""
    requested_phase = StudyPhase(phase)
    proposed = StudyRunClaim(
        journal_genesis_digest=store.genesis.digest,
        study_id=store.genesis.study_id,
        phase=requested_phase,
        authorization_payload_digest=authorization_payload_digest,
        run_id=run_id,
    )
    with store.locked() as directory_descriptor:
        records = _load_records_locked(store, directory_descriptor)
        pending = _load_pending_locked(store, directory_descriptor, records)
        _verify_external_chain_locked(
            store,
            directory_descriptor,
            publisher,
            records,
            pending,
        )
        _require_current_phase_locked(
            records,
            requested_phase,
            payload_digest=authorization_payload_digest,
        )
        return _claim_study_run_locked(
            store,
            directory_descriptor,
            proposed=proposed,
            resume=resume,
        )


def call_in_study_slice(
    store: StudyJournalStore,
    *,
    phase: StudyPhase,
    authorization_payload_digest: str,
    run_id: str,
    configuration_digest: str,
    resume_from: StudyRunCheckpointIdentity | None,
    publisher: ExternalCommitmentPublisher,
    operation: Callable[[], tuple[_ResultT, StudyRunCheckpointIdentity]],
) -> _ResultT:
    """Execute one serialized run slice and append exactly one checkpoint identity.

    The operation must durably persist its checkpoint before returning its identity. The journal
    lease stays held across verification, execution, and checkpoint append, so another local
    invocation cannot race the host-local budget authority. A checkpoint that survived a process
    failure before its journal append is reconciled from ``resume_from`` before new work starts.

    Args:
        store: Host-private journal that owns the phase and run claim.
        phase: Active phase authorized to execute the slice.
        authorization_payload_digest: Exact current phase payload identity.
        run_id: Caller-issued identity shared by every resume.
        configuration_digest: Frozen path-free slice configuration identity.
        resume_from: Latest caller-persisted checkpoint, or none for a fresh run.
        publisher: External phase-chain verifier.
        operation: One bounded invocation returning its result and new checkpoint identity.

    Returns:
        The operation result after its checkpoint identity is durably appended.

    Raises:
        ValueError: If phase, run, configuration, sequence, or checkpoint identity drifts.
    """
    return _call_in_study_slice(
        store,
        phase=phase,
        authorization_payload_digest=authorization_payload_digest,
        run_id=run_id,
        configuration_digest=configuration_digest,
        resume_from=resume_from,
        publisher=publisher,
        operation=operation,
        reuse_uncheckpointed_intent=False,
    )


def call_in_resumable_study_slice(
    store: StudyJournalStore,
    *,
    phase: StudyPhase,
    authorization_payload_digest: str,
    run_id: str,
    configuration_digest: str,
    resume_from: StudyRunCheckpointIdentity | None,
    publisher: ExternalCommitmentPublisher,
    operation: Callable[[], tuple[_ResultT, StudyRunCheckpointIdentity]],
) -> _ResultT:
    """Run idempotent work under a fresh or exact unconsumed durable slice intent.

    Unlike :func:`call_in_study_slice`, this explicit recovery surface may reenter the sole
    uncheckpointed intent for the same claim, configuration, and prior checkpoint. Callers must
    make ``operation`` resume only precommitted outstanding work and durably reuse completed work.
    """
    return _call_in_study_slice(
        store,
        phase=phase,
        authorization_payload_digest=authorization_payload_digest,
        run_id=run_id,
        configuration_digest=configuration_digest,
        resume_from=resume_from,
        publisher=publisher,
        operation=operation,
        reuse_uncheckpointed_intent=True,
    )


def _call_in_study_slice(
    store: StudyJournalStore,
    *,
    phase: StudyPhase,
    authorization_payload_digest: str,
    run_id: str,
    configuration_digest: str,
    resume_from: StudyRunCheckpointIdentity | None,
    publisher: ExternalCommitmentPublisher,
    operation: Callable[[], tuple[_ResultT, StudyRunCheckpointIdentity]],
    reuse_uncheckpointed_intent: bool,
) -> _ResultT:
    """Share checkpoint append mechanics across strict and explicitly resumable calls."""
    with _locked_study_slice(
        store,
        phase=phase,
        authorization_payload_digest=authorization_payload_digest,
        run_id=run_id,
        configuration_digest=configuration_digest,
        resume_from=resume_from,
        publisher=publisher,
        prepare_next=True,
        reuse_uncheckpointed_intent=reuse_uncheckpointed_intent,
    ) as (directory_descriptor, claim, checkpoints):
        result, checkpoint = operation()
        frozen_checkpoint = StudyRunCheckpointIdentity.model_validate(
            checkpoint.model_dump(mode="json")
        )
        expected_sequence = len(checkpoints)
        if frozen_checkpoint.sequence != expected_sequence:
            raise ValueError(
                "study slice did not return exactly the next durable checkpoint sequence"
            )
        _require_checkpoint_digest_advance(checkpoints, frozen_checkpoint)
        _append_run_checkpoint_locked(
            store,
            directory_descriptor,
            claim=claim,
            configuration_digest=configuration_digest,
            checkpoint=frozen_checkpoint,
            previous=checkpoints[-1] if checkpoints else None,
        )
        return result


def reconcile_study_slice(
    store: StudyJournalStore,
    *,
    phase: StudyPhase,
    authorization_payload_digest: str,
    run_id: str,
    configuration_digest: str,
    resume_from: StudyRunCheckpointIdentity,
    publisher: ExternalCommitmentPublisher,
    operation: Callable[[], _ResultT],
) -> _ResultT:
    """Reconcile one completed caller checkpoint and reconstruct its result.

    This path appends no checkpoint beyond ``resume_from``. It accepts either a checkpoint that
    survived a crash before its journal append or the exact latest journaled checkpoint, then runs
    only the caller's result reconstruction under the lifecycle lease.

    Args:
        store: Host-private journal that owns the phase and run claim.
        phase: Active phase authorized to reconcile the completed run.
        authorization_payload_digest: Exact current phase payload identity.
        run_id: Caller-issued identity shared by every resume.
        configuration_digest: Frozen path-free slice configuration identity.
        resume_from: Completed caller-persisted checkpoint to reconcile.
        publisher: External phase-chain verifier.
        operation: Side-effect-free reconstruction of the completed result.

    Returns:
        The reconstructed result after the completed checkpoint is journaled.

    Raises:
        ValueError: If phase, run, configuration, sequence, or checkpoint identity drifts.
    """
    with _locked_study_slice(
        store,
        phase=phase,
        authorization_payload_digest=authorization_payload_digest,
        run_id=run_id,
        configuration_digest=configuration_digest,
        resume_from=resume_from,
        publisher=publisher,
        prepare_next=False,
    ):
        return operation()


@contextmanager
def _locked_study_slice(
    store: StudyJournalStore,
    *,
    phase: StudyPhase,
    authorization_payload_digest: str,
    run_id: str,
    configuration_digest: str,
    resume_from: StudyRunCheckpointIdentity | None,
    publisher: ExternalCommitmentPublisher,
    prepare_next: bool,
    reuse_uncheckpointed_intent: bool = False,
) -> Iterator[tuple[int, StudyRunClaim, tuple[StudyRunCheckpointRecord, ...]]]:
    """Admit one exact fresh, resumed, or completed-reconciliation slice."""
    requested_phase = StudyPhase(phase)
    if not _is_digest(configuration_digest):
        raise ValueError("study slice configuration_digest must be a canonical SHA-256 digest")
    proposed = StudyRunClaim(
        journal_genesis_digest=store.genesis.digest,
        study_id=store.genesis.study_id,
        phase=requested_phase,
        authorization_payload_digest=authorization_payload_digest,
        run_id=run_id,
    )
    resumed = (
        StudyRunCheckpointIdentity.model_validate(resume_from.model_dump(mode="json"))
        if resume_from is not None
        else None
    )
    with store.locked() as directory_descriptor:
        records = _load_records_locked(store, directory_descriptor)
        pending = _load_pending_locked(store, directory_descriptor, records)
        _verify_external_chain_locked(
            store,
            directory_descriptor,
            publisher,
            records,
            pending,
        )
        _require_current_phase_locked(
            records,
            requested_phase,
            payload_digest=authorization_payload_digest,
        )
        checkpoints_by_phase = _load_run_checkpoints_locked(store, directory_descriptor)
        checkpoints = checkpoints_by_phase.get(requested_phase, ())
        claim = _claim_study_run_locked(
            store,
            directory_descriptor,
            proposed=proposed,
            resume=resumed is not None,
            allow_uncheckpointed_reentry=resumed is None and not checkpoints,
        )
        intents_by_phase = _load_run_slice_intents_locked(
            store,
            directory_descriptor,
            checkpoints_by_phase=checkpoints_by_phase,
        )
        intents = intents_by_phase.get(requested_phase, ())
        if intents and intents[0].configuration_digest != configuration_digest:
            raise ValueError(
                "study phase is already claimed by a different run identity or configuration"
            )
        reconciled = _reconcile_resume_checkpoint_locked(
            store,
            directory_descriptor,
            claim=claim,
            configuration_digest=configuration_digest,
            checkpoints=checkpoints,
            intents=intents,
            resume_from=resumed,
        )
        has_uncheckpointed_intent = len(intents) > len(reconciled)
        if has_uncheckpointed_intent and not reuse_uncheckpointed_intent:
            raise ValueError(
                "study run has an ambiguous durable slice intent without its exact persisted "
                "checkpoint"
            )
        if prepare_next and not has_uncheckpointed_intent:
            _append_run_slice_intent_locked(
                store,
                directory_descriptor,
                claim=claim,
                configuration_digest=configuration_digest,
                checkpoint_sequence=len(reconciled),
                previous=reconciled[-1] if reconciled else None,
            )
        yield directory_descriptor, claim, reconciled


def call_in_study_phase(
    store: StudyJournalStore,
    *,
    phase: StudyPhase,
    payload_digest: str | None,
    publisher: ExternalCommitmentPublisher,
    operation: Callable[[], _ResultT],
) -> _ResultT:
    """Hold the journal lease from exact phase verification through one side effect."""
    requested_phase = StudyPhase(phase)
    with store.locked() as directory_descriptor:
        records = _load_records_locked(store, directory_descriptor)
        pending = _load_pending_locked(store, directory_descriptor, records)
        _verify_external_chain_locked(
            store,
            directory_descriptor,
            publisher,
            records,
            pending,
        )
        _require_current_phase_locked(records, requested_phase, payload_digest=payload_digest)
        return operation()


def _load_records_locked(
    store: StudyJournalStore,
    directory_descriptor: int,
) -> tuple[StudyPhaseRecord, ...]:
    record_names: list[tuple[int, StudyPhase, str]] = []
    for name in os.listdir(directory_descriptor):
        if name in {_GENESIS_FILE, _PENDING_FILE}:
            continue
        if _RUN_CLAIM_PATTERN.fullmatch(name) is not None:
            _validate_run_claim_payload(
                store,
                name,
                _read_regular_file_at(directory_descriptor, name),
            )
            continue
        if _RUN_CHECKPOINT_PATTERN.fullmatch(name) is not None:
            continue
        if _RUN_SLICE_INTENT_PATTERN.fullmatch(name) is not None:
            continue
        match = _RECORD_PATTERN.fullmatch(name)
        if match is None:
            if _is_valid_temporary_name(name):
                _read_regular_file_at(directory_descriptor, name)
                continue
            raise ValueError(f"study journal contains unexpected entry {name!r}")
        try:
            phase = StudyPhase(match.group("phase"))
        except ValueError as exc:
            raise ValueError(f"study journal contains unknown phase file {name!r}") from exc
        record_names.append((int(match.group("sequence")), phase, name))
    record_names.sort(key=lambda item: (item[0], item[1].value))

    records: list[StudyPhaseRecord] = []
    for expected_sequence, (sequence, phase, name) in enumerate(record_names):
        if sequence != expected_sequence:
            raise ValueError("study journal phase files are missing, duplicated, or out of order")
        if name != _record_name(sequence, phase):
            raise ValueError("study journal phase filename is not canonical")
        record_payload = _read_regular_file_at(directory_descriptor, name)
        record = StudyPhaseRecord.model_validate_json(record_payload)
        if record_payload != _canonical_json_bytes(record.model_dump(mode="json")):
            raise ValueError("study journal record is not canonical")
        expected_previous = records[-1].digest if records else None
        if phase not in _allowed_next_phases(tuple(records)):
            raise ValueError("study journal contains an illegal phase transition")
        if (
            record.commitment.journal_genesis_digest != store.genesis.digest
            or record.commitment.study_id != store.genesis.study_id
            or record.commitment.sequence != sequence
            or record.commitment.phase is not phase
            or record.commitment.previous_record_digest != expected_previous
        ):
            raise ValueError("study journal record differs from its chain position")
        if records and record.publication.published_at < records[-1].publication.published_at:
            raise ValueError("study journal publication timestamps move backwards")
        records.append(record)
    checkpoints_by_phase = _load_run_checkpoints_locked(store, directory_descriptor)
    _load_run_slice_intents_locked(
        store,
        directory_descriptor,
        checkpoints_by_phase=checkpoints_by_phase,
    )
    return tuple(records)


def _load_pending_locked(
    store: StudyJournalStore,
    directory_descriptor: int,
    records: tuple[StudyPhaseRecord, ...],
) -> tuple[StudyPhaseCommitment | None, bool]:
    try:
        payload = _read_regular_file_at(directory_descriptor, _PENDING_FILE)
    except FileNotFoundError:
        return None, False
    try:
        pending = StudyPhaseCommitment.model_validate_json(payload)
    except ValueError as exc:
        raise ValueError("study journal pending commitment is invalid") from exc
    if payload != _canonical_json_bytes(pending.model_dump(mode="json")):
        raise ValueError("study journal pending commitment is not canonical")
    if (
        pending.journal_genesis_digest != store.genesis.digest
        or pending.study_id != store.genesis.study_id
    ):
        raise ValueError("study journal pending commitment belongs to another journal")
    if records and pending == records[-1].commitment:
        return pending, True
    expected_sequence = len(records)
    allowed = _allowed_next_phases(records)
    if not allowed:
        raise ValueError("completed study journal cannot contain a pending commitment")
    expected_previous = records[-1].digest if records else None
    if (
        pending.sequence != expected_sequence
        or pending.phase not in allowed
        or pending.previous_record_digest != expected_previous
    ):
        raise ValueError("study journal pending commitment differs from its chain position")
    return pending, False


def _verify_external_chain_locked(
    store: StudyJournalStore,
    directory_descriptor: int,
    publisher: ExternalCommitmentPublisher,
    records: tuple[StudyPhaseRecord, ...],
    pending: tuple[StudyPhaseCommitment | None, bool],
) -> None:
    _validate_publisher(store, publisher)
    for record in records:
        publisher.verify(record.commitment, record.publication)
    active_pending = pending[0] if not pending[1] else None
    publisher.verify_chain_head(store.genesis, records, active_pending)
    _validate_publisher(store, publisher)
    persisted = _load_records_locked(store, directory_descriptor)
    persisted_pending = _load_pending_locked(store, directory_descriptor, persisted)
    if persisted != records or persisted_pending != pending:
        raise RuntimeError("study journal changed during external verification")


def _record_name(sequence: int, phase: StudyPhase) -> str:
    return f"{sequence:03d}-{phase.value}.json"


def _run_claim_name(phase: StudyPhase) -> str:
    return f"run-claim-{phase.value}.json"


def _run_checkpoint_name(phase: StudyPhase, sequence: int) -> str:
    return f"run-checkpoint-{phase.value}-{sequence:08d}.json"


def _run_slice_intent_name(phase: StudyPhase, sequence: int) -> str:
    return f"run-slice-intent-{phase.value}-{sequence:08d}.json"


def _claim_study_run_locked(
    store: StudyJournalStore,
    directory_descriptor: int,
    *,
    proposed: StudyRunClaim,
    resume: bool,
    allow_uncheckpointed_reentry: bool = False,
) -> StudyRunClaim:
    """Create or verify the sole phase run claim while its journal lease is held."""
    claim_name = _run_claim_name(proposed.phase)
    try:
        existing_payload = _read_regular_file_at(directory_descriptor, claim_name)
    except FileNotFoundError:
        existing_payload = None
    if existing_payload is None:
        if resume:
            raise ValueError("study run cannot resume before its durable claim exists")
        _publish_regular_file_once_at(
            directory_descriptor,
            claim_name,
            _canonical_json_bytes(proposed.model_dump(mode="json")),
        )
        return proposed
    existing = _validate_run_claim_payload(store, claim_name, existing_payload)
    if existing != proposed:
        raise ValueError(
            "study phase is already claimed by a different run identity or configuration"
        )
    if not resume and not allow_uncheckpointed_reentry:
        raise ValueError("study run already started; supply its exact durable checkpoint")
    return existing


def _load_run_checkpoints_locked(
    store: StudyJournalStore,
    directory_descriptor: int,
) -> dict[StudyPhase, tuple[StudyRunCheckpointRecord, ...]]:
    """Load and validate every append-only run checkpoint chain in the journal."""
    names_by_phase: dict[StudyPhase, list[tuple[int, str]]] = {}
    for name in os.listdir(directory_descriptor):
        match = _RUN_CHECKPOINT_PATTERN.fullmatch(name)
        if match is None:
            continue
        try:
            phase = StudyPhase(match.group("phase"))
        except ValueError as exc:
            raise ValueError(f"study journal contains unknown run checkpoint {name!r}") from exc
        names_by_phase.setdefault(phase, []).append((int(match.group("sequence")), name))

    loaded: dict[StudyPhase, tuple[StudyRunCheckpointRecord, ...]] = {}
    for phase, names in names_by_phase.items():
        claim_name = _run_claim_name(phase)
        try:
            claim_payload = _read_regular_file_at(directory_descriptor, claim_name)
        except FileNotFoundError:
            raise ValueError("study run checkpoints exist without a durable run claim") from None
        claim = _validate_run_claim_payload(store, claim_name, claim_payload)
        records: list[StudyRunCheckpointRecord] = []
        checkpoint_digests: set[str] = set()
        configuration_digest: str | None = None
        for expected_sequence, (sequence, name) in enumerate(sorted(names)):
            if sequence != expected_sequence:
                raise ValueError("study run checkpoints are missing, duplicated, or out of order")
            record = _validate_run_checkpoint_payload(
                store,
                name,
                _read_regular_file_at(directory_descriptor, name),
            )
            previous_digest = records[-1].digest if records else None
            if (
                record.phase is not phase
                or record.checkpoint.sequence != sequence
                or record.previous_record_digest != previous_digest
                or record.authorization_payload_digest != claim.authorization_payload_digest
                or record.run_id != claim.run_id
            ):
                raise ValueError("study run checkpoint differs from its chain position")
            if record.checkpoint.checkpoint_digest in checkpoint_digests:
                raise ValueError("study run checkpoint chain repeats a checkpoint identity")
            checkpoint_digests.add(record.checkpoint.checkpoint_digest)
            if configuration_digest is None:
                configuration_digest = record.configuration_digest
            elif record.configuration_digest != configuration_digest:
                raise ValueError("study run checkpoint configuration changed across resume")
            records.append(record)
        loaded[phase] = tuple(records)
    return loaded


def _load_run_slice_intents_locked(
    store: StudyJournalStore,
    directory_descriptor: int,
    *,
    checkpoints_by_phase: dict[StudyPhase, tuple[StudyRunCheckpointRecord, ...]],
) -> dict[StudyPhase, tuple[StudyRunSliceIntentRecord, ...]]:
    """Load the append-only pre-dispatch authority chain for every claimed run."""
    names_by_phase: dict[StudyPhase, list[tuple[int, str]]] = {}
    for name in os.listdir(directory_descriptor):
        match = _RUN_SLICE_INTENT_PATTERN.fullmatch(name)
        if match is None:
            continue
        try:
            phase = StudyPhase(match.group("phase"))
        except ValueError as exc:
            raise ValueError(f"study journal contains unknown run slice intent {name!r}") from exc
        names_by_phase.setdefault(phase, []).append((int(match.group("sequence")), name))

    loaded: dict[StudyPhase, tuple[StudyRunSliceIntentRecord, ...]] = {}
    for phase, names in names_by_phase.items():
        claim_name = _run_claim_name(phase)
        try:
            claim_payload = _read_regular_file_at(directory_descriptor, claim_name)
        except FileNotFoundError:
            raise ValueError("study run slice intents exist without a durable run claim") from None
        claim = _validate_run_claim_payload(store, claim_name, claim_payload)
        checkpoints = checkpoints_by_phase.get(phase, ())
        intents: list[StudyRunSliceIntentRecord] = []
        configuration_digest: str | None = None
        for expected_sequence, (sequence, name) in enumerate(sorted(names)):
            if sequence != expected_sequence:
                raise ValueError("study run slice intents are missing, duplicated, or out of order")
            intent = _validate_run_slice_intent_payload(
                store,
                name,
                _read_regular_file_at(directory_descriptor, name),
            )
            previous_checkpoint_digest = (
                checkpoints[sequence - 1].digest
                if sequence > 0 and sequence <= len(checkpoints)
                else None
            )
            if (
                intent.phase is not phase
                or intent.checkpoint_sequence != sequence
                or intent.previous_checkpoint_record_digest != previous_checkpoint_digest
                or intent.authorization_payload_digest != claim.authorization_payload_digest
                or intent.run_id != claim.run_id
            ):
                raise ValueError("study run slice intent differs from its chain position")
            if sequence > len(checkpoints):
                raise ValueError("study run slice intent skips its durable checkpoint sequence")
            if configuration_digest is None:
                configuration_digest = intent.configuration_digest
            elif intent.configuration_digest != configuration_digest:
                raise ValueError("study run slice intent configuration changed across resume")
            if sequence < len(checkpoints):
                checkpoint = checkpoints[sequence]
                if checkpoint.configuration_digest != intent.configuration_digest:
                    raise ValueError("study run slice intent differs from its durable checkpoint")
            intents.append(intent)
        if len(intents) < len(checkpoints):
            raise ValueError("study run checkpoint exists without its durable slice intent")
        if len(intents) > len(checkpoints) + 1:
            raise ValueError("study run has more than one unconsumed slice intent")
        loaded[phase] = tuple(intents)
    for phase, checkpoints in checkpoints_by_phase.items():
        if checkpoints and phase not in loaded:
            raise ValueError("study run checkpoint exists without its durable slice intent")
    return loaded


def _reconcile_resume_checkpoint_locked(
    store: StudyJournalStore,
    directory_descriptor: int,
    *,
    claim: StudyRunClaim,
    configuration_digest: str,
    checkpoints: tuple[StudyRunCheckpointRecord, ...],
    intents: tuple[StudyRunSliceIntentRecord, ...],
    resume_from: StudyRunCheckpointIdentity | None,
) -> tuple[StudyRunCheckpointRecord, ...]:
    """Match the latest record or append one checkpoint recovered after a crash."""
    if resume_from is None:
        if checkpoints:
            raise ValueError("study run already has a durable checkpoint and must resume from it")
        return checkpoints
    if checkpoints and checkpoints[0].configuration_digest != configuration_digest:
        raise ValueError("study slice configuration differs from its durable checkpoint chain")
    if resume_from.sequence < len(checkpoints) - 1:
        raise ValueError("resume checkpoint is not the latest durable checkpoint")
    if resume_from.sequence == len(checkpoints) - 1:
        latest = checkpoints[-1]
        if resume_from != latest.checkpoint:
            raise ValueError("resume checkpoint identity differs from the durable checkpoint")
        return checkpoints
    if resume_from.sequence != len(checkpoints):
        raise ValueError("resume checkpoint skips a durable checkpoint sequence")
    if len(intents) != len(checkpoints) + 1:
        raise ValueError("resume checkpoint has no matching durable slice intent")
    recovered_intent = intents[-1]
    if (
        recovered_intent.checkpoint_sequence != resume_from.sequence
        or recovered_intent.configuration_digest != configuration_digest
    ):
        raise ValueError("resume checkpoint differs from its durable slice intent")
    _require_checkpoint_digest_advance(checkpoints, resume_from)
    recovered = _append_run_checkpoint_locked(
        store,
        directory_descriptor,
        claim=claim,
        configuration_digest=configuration_digest,
        checkpoint=resume_from,
        previous=checkpoints[-1] if checkpoints else None,
    )
    return (*checkpoints, recovered)


def _require_checkpoint_digest_advance(
    checkpoints: tuple[StudyRunCheckpointRecord, ...],
    checkpoint: StudyRunCheckpointIdentity,
) -> None:
    if any(
        checkpoint.checkpoint_digest == record.checkpoint.checkpoint_digest
        for record in checkpoints
    ):
        raise ValueError("study slice did not advance the durable checkpoint identity")


def _append_run_slice_intent_locked(
    store: StudyJournalStore,
    directory_descriptor: int,
    *,
    claim: StudyRunClaim,
    configuration_digest: str,
    checkpoint_sequence: int,
    previous: StudyRunCheckpointRecord | None,
) -> StudyRunSliceIntentRecord:
    """Persist one immutable slice authority before any caller work can begin."""
    payload = {
        "intent_version": "1",
        "journal_genesis_digest": store.genesis.digest,
        "study_id": store.genesis.study_id,
        "phase": claim.phase.value,
        "authorization_payload_digest": claim.authorization_payload_digest,
        "run_id": claim.run_id,
        "configuration_digest": configuration_digest,
        "checkpoint_sequence": checkpoint_sequence,
        "previous_checkpoint_record_digest": previous.digest if previous is not None else None,
    }
    intent = StudyRunSliceIntentRecord.model_validate(
        {**payload, "intent_digest": _canonical_digest(payload)}
    )
    name = _run_slice_intent_name(claim.phase, checkpoint_sequence)
    _publish_regular_file_once_at(
        directory_descriptor,
        name,
        _canonical_json_bytes(intent.model_dump(mode="json")),
    )
    persisted = _validate_run_slice_intent_payload(
        store,
        name,
        _read_regular_file_at(directory_descriptor, name),
    )
    if persisted != intent:
        raise RuntimeError("study run slice intent append did not persist exact evidence")
    return intent


def _append_run_checkpoint_locked(
    store: StudyJournalStore,
    directory_descriptor: int,
    *,
    claim: StudyRunClaim,
    configuration_digest: str,
    checkpoint: StudyRunCheckpointIdentity,
    previous: StudyRunCheckpointRecord | None,
) -> StudyRunCheckpointRecord:
    """Append one immutable checkpoint record at the exact next sequence."""
    payload = {
        "record_version": "1",
        "journal_genesis_digest": store.genesis.digest,
        "study_id": store.genesis.study_id,
        "phase": claim.phase.value,
        "authorization_payload_digest": claim.authorization_payload_digest,
        "run_id": claim.run_id,
        "configuration_digest": configuration_digest,
        "checkpoint": checkpoint.model_dump(mode="json"),
        "previous_record_digest": previous.digest if previous is not None else None,
    }
    record = StudyRunCheckpointRecord.model_validate(
        {**payload, "record_digest": _canonical_digest(payload)}
    )
    name = _run_checkpoint_name(claim.phase, checkpoint.sequence)
    _publish_regular_file_once_at(
        directory_descriptor,
        name,
        _canonical_json_bytes(record.model_dump(mode="json")),
    )
    persisted = _validate_run_checkpoint_payload(
        store,
        name,
        _read_regular_file_at(directory_descriptor, name),
    )
    if persisted != record:
        raise RuntimeError("study run checkpoint append did not persist exact evidence")
    return record


def _validate_run_claim_payload(
    store: StudyJournalStore,
    name: str,
    payload: bytes,
) -> StudyRunClaim:
    match = _RUN_CLAIM_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError("study run claim filename is not canonical")
    try:
        filename_phase = StudyPhase(match.group("phase"))
        claim = StudyRunClaim.model_validate_json(payload)
    except ValueError as exc:
        raise ValueError("study run claim is invalid") from exc
    if payload != _canonical_json_bytes(claim.model_dump(mode="json")):
        raise ValueError("study run claim is not canonical")
    if (
        claim.phase is not filename_phase
        or name != _run_claim_name(claim.phase)
        or claim.journal_genesis_digest != store.genesis.digest
        or claim.study_id != store.genesis.study_id
    ):
        raise ValueError("study run claim belongs to another journal or phase")
    return claim


def _validate_run_checkpoint_payload(
    store: StudyJournalStore,
    name: str,
    payload: bytes,
) -> StudyRunCheckpointRecord:
    """Parse one checkpoint record and bind it to its canonical journal filename."""
    match = _RUN_CHECKPOINT_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError("study run checkpoint filename is not canonical")
    try:
        filename_phase = StudyPhase(match.group("phase"))
        filename_sequence = int(match.group("sequence"))
        record = StudyRunCheckpointRecord.model_validate_json(payload)
    except ValueError as exc:
        raise ValueError("study run checkpoint is invalid") from exc
    if payload != _canonical_json_bytes(record.model_dump(mode="json")):
        raise ValueError("study run checkpoint is not canonical")
    if (
        record.phase is not filename_phase
        or record.checkpoint.sequence != filename_sequence
        or name != _run_checkpoint_name(record.phase, record.checkpoint.sequence)
        or record.journal_genesis_digest != store.genesis.digest
        or record.study_id != store.genesis.study_id
    ):
        raise ValueError("study run checkpoint belongs to another journal or sequence")
    return record


def _validate_run_slice_intent_payload(
    store: StudyJournalStore,
    name: str,
    payload: bytes,
) -> StudyRunSliceIntentRecord:
    """Parse one intent record and bind it to its canonical journal filename."""
    match = _RUN_SLICE_INTENT_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError("study run slice intent filename is not canonical")
    try:
        filename_phase = StudyPhase(match.group("phase"))
        filename_sequence = int(match.group("sequence"))
        intent = StudyRunSliceIntentRecord.model_validate_json(payload)
    except ValueError as exc:
        raise ValueError("study run slice intent is invalid") from exc
    if payload != _canonical_json_bytes(intent.model_dump(mode="json")):
        raise ValueError("study run slice intent is not canonical")
    if (
        intent.phase is not filename_phase
        or intent.checkpoint_sequence != filename_sequence
        or name != _run_slice_intent_name(intent.phase, intent.checkpoint_sequence)
        or intent.journal_genesis_digest != store.genesis.digest
        or intent.study_id != store.genesis.study_id
    ):
        raise ValueError("study run slice intent belongs to another journal or sequence")
    return intent


def _require_current_phase_locked(
    records: tuple[StudyPhaseRecord, ...],
    expected: StudyPhase,
    *,
    payload_digest: str | None,
) -> StudyPhaseRecord:
    actual = records[-1].commitment.phase if records else None
    if actual is not expected:
        actual_label = actual.value if actual is not None else "unstarted"
        raise ValueError(f"current study phase is {actual_label}, required {expected.value}")
    record = records[-1]
    if payload_digest is not None and record.commitment.payload_digest != payload_digest:
        raise ValueError("current study phase carries different authorization evidence")
    return record


def _allowed_next_phases(
    records: tuple[StudyPhaseRecord, ...],
) -> tuple[StudyPhase, ...]:
    """Return the exact legal successors for a validated journal prefix."""
    if not records:
        return (StudyPhase.PREPARATION_PLANNED,)
    current = records[-1].commitment.phase
    if current in _TERMINAL_STUDY_PHASES:
        return ()
    successful_index = SUCCESSFUL_STUDY_PHASES.index(current)
    return (SUCCESSFUL_STUDY_PHASES[successful_index + 1], StudyPhase.STOPPED)


def _validate_publisher(
    store: StudyJournalStore,
    publisher: ExternalCommitmentPublisher,
) -> None:
    if publisher.configuration_digest != store.genesis.publisher_configuration_digest:
        raise ValueError("runtime publisher differs from the study journal configuration")


def _validate_private_directory(path: Path) -> tuple[int, int]:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        raise OSError("study journal directory does not exist") from None
    return _private_directory_identity(metadata)


def _private_directory_identity(metadata: os.stat_result) -> tuple[int, int]:
    if not stat.S_ISDIR(metadata.st_mode):
        raise OSError("study journal path must be a directory, not a symlink or file")
    if metadata.st_mode & 0o077:
        raise OSError("study journal directory cannot be accessible by group or other users")
    if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise OSError("study journal directory must be owned by the current user")
    return metadata.st_dev, metadata.st_ino


def _open_private_directory(
    path: Path,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> int:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except FileNotFoundError:
        raise OSError("study journal directory does not exist") from None
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise OSError("study journal path must be a directory, not a symlink or file") from exc
        raise
    try:
        identity = _private_directory_identity(os.fstat(descriptor))
        if expected_identity is not None and identity != expected_identity:
            raise OSError("study journal directory was replaced or changed")
        _require_directory_binding(path, identity)
        return descriptor
    except BaseException as primary_error:
        _close_descriptor_preserving_error(descriptor, primary_error=primary_error)
        raise


def _open_parent_directory(path: Path) -> int:
    try:
        descriptor = os.open(
            path.parent,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except FileNotFoundError:
        raise OSError("study journal parent directory does not exist") from None
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise OSError("study journal parent path must be a directory") from exc
        raise
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("study journal parent path must be a directory")
        return descriptor
    except BaseException as primary_error:
        _close_descriptor_preserving_error(descriptor, primary_error=primary_error)
        raise


def _open_private_directory_at(parent_descriptor: int, name: str) -> int:
    _validate_leaf_name(name)
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
    except FileNotFoundError:
        raise OSError("study journal directory does not exist") from None
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise OSError("study journal path must be a directory, not a symlink or file") from exc
        raise
    try:
        _private_directory_identity(os.fstat(descriptor))
        return descriptor
    except BaseException as primary_error:
        _close_descriptor_preserving_error(descriptor, primary_error=primary_error)
        raise


def _require_directory_binding(path: Path, expected_identity: tuple[int, int]) -> None:
    try:
        actual_identity = _validate_private_directory(path)
    except OSError as exc:
        raise OSError("study journal directory was replaced or changed") from exc
    if actual_identity != expected_identity:
        raise OSError("study journal directory was replaced or changed")


def _require_genesis_binding(
    store: StudyJournalStore,
    directory_descriptor: int,
) -> None:
    expected = _canonical_json_bytes(store.genesis.model_dump(mode="json"))
    try:
        actual = _read_regular_file_at(directory_descriptor, _GENESIS_FILE)
    except OSError as exc:
        raise OSError("study journal genesis was replaced or changed") from exc
    if actual != expected:
        raise OSError("study journal genesis was replaced or changed")


def _require_store_binding(
    store: StudyJournalStore,
    directory_descriptor: int,
) -> None:
    _require_directory_binding(store.directory, store._directory_identity)
    _require_genesis_binding(store, directory_descriptor)


@contextmanager
def _managed_descriptor(descriptor: int) -> Iterator[int]:
    primary_error: BaseException | None = None
    try:
        yield descriptor
    except BaseException as error:
        primary_error = error
        raise
    finally:
        _close_descriptor_preserving_error(descriptor, primary_error=primary_error)


def _close_descriptor_preserving_error(
    descriptor: int,
    *,
    primary_error: BaseException | None,
) -> None:
    try:
        os.close(descriptor)
    except OSError as cleanup_error:
        if primary_error is None:
            raise
        primary_error.add_note(f"study journal descriptor close also failed: {cleanup_error}")


def _read_regular_file_at(directory_descriptor: int, name: str) -> bytes:
    _validate_leaf_name(name)
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_descriptor,
        )
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise OSError("study journal record must be a regular file") from exc
        raise
    with _managed_descriptor(descriptor):
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("study journal record must be a regular file")
        if metadata.st_mode & 0o077:
            raise OSError("study journal record cannot be accessible by group or other users")
        if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
            raise OSError("study journal record must be owned by the current user")
        if metadata.st_size > _MAX_RECORD_BYTES:
            raise OSError("study journal record exceeds its size limit")
        chunks: list[bytes] = []
        remaining = _MAX_RECORD_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > _MAX_RECORD_BYTES:
            raise OSError("study journal record exceeds its size limit")
        return payload


def _publish_regular_file_once_at(
    directory_descriptor: int,
    name: str,
    payload: bytes,
) -> None:
    _validate_leaf_name(name)
    if len(payload) > _MAX_RECORD_BYTES:
        raise OSError("study journal record exceeds its size limit")
    temporary = f".tmp-{name}-{uuid.uuid4().hex}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_descriptor,
    )
    primary_error: BaseException | None = None
    try:
        with _managed_descriptor(descriptor):
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            if _read_regular_file_at(directory_descriptor, name) != payload:
                raise ValueError(
                    "study journal file already exists with different content"
                ) from None
        os.fsync(directory_descriptor)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        _remove_temporary_file_at(
            directory_descriptor,
            temporary,
            primary_error=primary_error,
        )


def _remove_temporary_file_at(
    directory_descriptor: int,
    name: str,
    *,
    primary_error: BaseException | None,
) -> None:
    cleanup_error: OSError | None = None
    try:
        os.unlink(name, dir_fd=directory_descriptor)
    except FileNotFoundError:
        pass
    except OSError as error:
        cleanup_error = error
    try:
        os.fsync(directory_descriptor)
    except OSError as error:
        if cleanup_error is None:
            cleanup_error = error
        else:
            cleanup_error.add_note(f"temporary-file directory fsync also failed: {error}")
    if cleanup_error is None:
        return
    if primary_error is None:
        raise cleanup_error
    primary_error.add_note(f"study journal temporary-file cleanup also failed: {cleanup_error}")


def _remove_file_durably_at(directory_descriptor: int, name: str) -> None:
    _validate_leaf_name(name)
    os.unlink(name, dir_fd=directory_descriptor)
    os.fsync(directory_descriptor)


def _is_valid_temporary_name(name: str) -> bool:
    match = _TEMPORARY_PATTERN.fullmatch(name)
    if match is None:
        return False
    target = match.group("target")
    if target in {_GENESIS_FILE, _PENDING_FILE}:
        return True
    if _RUN_CLAIM_PATTERN.fullmatch(target) is not None:
        return True
    checkpoint_match = _RUN_CHECKPOINT_PATTERN.fullmatch(target)
    if checkpoint_match is not None:
        try:
            phase = StudyPhase(checkpoint_match.group("phase"))
        except ValueError:
            return False
        sequence = int(checkpoint_match.group("sequence"))
        return target == _run_checkpoint_name(phase, sequence)
    intent_match = _RUN_SLICE_INTENT_PATTERN.fullmatch(target)
    if intent_match is not None:
        try:
            phase = StudyPhase(intent_match.group("phase"))
        except ValueError:
            return False
        sequence = int(intent_match.group("sequence"))
        return target == _run_slice_intent_name(phase, sequence)
    record_match = _RECORD_PATTERN.fullmatch(target)
    if record_match is None:
        return False
    try:
        phase = StudyPhase(record_match.group("phase"))
    except ValueError:
        return False
    sequence = int(record_match.group("sequence"))
    if phase is StudyPhase.STOPPED:
        position_is_possible = 1 <= sequence < len(SUCCESSFUL_STUDY_PHASES)
    else:
        position_is_possible = sequence == SUCCESSFUL_STUDY_PHASES.index(phase)
    return position_is_possible and target == _record_name(sequence, phase)


def _validate_leaf_name(name: str) -> None:
    if (
        not name
        or name in {".", ".."}
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
    ):
        raise ValueError("study journal filename must be one path component")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _canonical_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_digest(value: str) -> bool:
    return bool(re.fullmatch(_DIGEST_PATTERN, value))
