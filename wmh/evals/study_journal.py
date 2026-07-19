"""Externally witnessed, crash-safe phase journal for optimization studies."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictInt, model_validator

from wmh.core.text import validate_durable_text
from wmh.evals.harbor._file_lease import exclusive_posix_file_lease

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_JOURNAL_VERSION: Literal["1"] = "1"
_RECORD_VERSION: Literal["1"] = "1"
_GENESIS_FILE = "study-journal.json"
_LOCK_FILE = ".study-journal.lock"
_MAX_RECORD_BYTES = 64 * 1024
_RECORD_PATTERN = re.compile(r"^(?P<sequence>[0-9]{3})-(?P<phase>[a-z_]+)\.json$")


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


STUDY_PHASES: tuple[StudyPhase, ...] = tuple(StudyPhase)


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
        expected_sequence = STUDY_PHASES.index(self.phase)
        if self.sequence != expected_sequence:
            raise ValueError("study phase sequence differs from the fixed chronology")
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


class ExternalCommitmentPublisher(Protocol):
    """Idempotent adapter to an externally verifiable append-only channel."""

    @property
    def configuration_digest(self) -> str:
        """Return the nonsecret identity of the publisher and immutable channel."""
        ...

    def publish(self, commitment: StudyPhaseCommitment) -> ExternalPublicationReceipt:
        """Publish or recover the sole receipt for ``commitment.digest``."""
        ...

    def verify(
        self,
        commitment: StudyPhaseCommitment,
        receipt: ExternalPublicationReceipt,
    ) -> None:
        """Raise unless the external channel still proves this exact publication."""
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
        self._directory_identity = _validate_private_directory(self._directory)
        self._genesis = StudyJournalGenesis.model_validate_json(
            _read_regular_file(self._directory / _GENESIS_FILE)
        )
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
        try:
            path.mkdir(parents=True, mode=0o700)
        except FileExistsError:
            pass
        _validate_private_directory(path)
        _publish_regular_file_once(
            path / _GENESIS_FILE,
            _canonical_json_bytes(genesis.model_dump(mode="json")),
        )
        return cls(
            path,
            study_id=study_id,
            publisher_configuration_digest=publisher_configuration_digest,
        )

    @property
    def directory(self) -> Path:
        """Return the host-private journal directory."""
        return self._directory

    @property
    def genesis(self) -> StudyJournalGenesis:
        """Return the path-independent journal identity."""
        return self._genesis

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the store lease and reject directory replacement."""
        with exclusive_posix_file_lease(
            self._directory / _LOCK_FILE,
            unsupported_error=RuntimeError("study journal requires POSIX file locking"),
            irregular_file_error=OSError("study journal lock must be a regular file"),
            contention_error=RuntimeError("study journal is already locked"),
        ):
            if _validate_private_directory(self._directory) != self._directory_identity:
                raise OSError("study journal directory was replaced")
            yield


def load_study_journal(
    store: StudyJournalStore,
    *,
    publisher: ExternalCommitmentPublisher | None = None,
) -> tuple[StudyPhaseRecord, ...]:
    """Load and revalidate the complete local chain and optional external proofs."""
    with store.locked():
        records = _load_records_locked(store)
        if publisher is not None:
            _validate_publisher(store, publisher)
            for record in records:
                publisher.verify(record.commitment, record.publication)
        return records


def append_study_phase(
    store: StudyJournalStore,
    *,
    phase: StudyPhase,
    payload_digest: str,
    publisher: ExternalCommitmentPublisher,
) -> StudyPhaseRecord:
    """Publish then durably append exactly the next phase, idempotently on restart."""
    requested_phase = StudyPhase(phase)
    if not _is_digest(payload_digest):
        raise ValueError("study phase payload_digest must be a canonical SHA-256 digest")
    _validate_publisher(store, publisher)
    requested_index = STUDY_PHASES.index(requested_phase)
    with store.locked():
        records = _load_records_locked(store)
        if requested_index < len(records):
            existing = records[requested_index]
            if existing.commitment.payload_digest != payload_digest:
                raise ValueError("study phase is already committed with a different payload")
            publisher.verify(existing.commitment, existing.publication)
            return existing
        if requested_index != len(records):
            expected = STUDY_PHASES[len(records)] if len(records) < len(STUDY_PHASES) else None
            raise ValueError(f"next study phase must be {expected.value if expected else 'none'}")

        commitment = StudyPhaseCommitment(
            journal_genesis_digest=store.genesis.digest,
            study_id=store.genesis.study_id,
            sequence=requested_index,
            phase=requested_phase,
            previous_record_digest=records[-1].digest if records else None,
            payload_digest=payload_digest,
        )
        publication = ExternalPublicationReceipt.model_validate(
            publisher.publish(commitment).model_dump(mode="json")
        )
        if publication.commitment_digest != commitment.digest:
            raise ValueError("publication receipt names a different commitment")
        publisher.verify(commitment, publication)
        if records and publication.published_at < records[-1].publication.published_at:
            raise ValueError("study journal publication timestamps move backwards")
        record = StudyPhaseRecord.create(commitment=commitment, publication=publication)
        _publish_regular_file_once(
            _record_path(store.directory, requested_index, requested_phase),
            _canonical_json_bytes(record.model_dump(mode="json")),
        )
        persisted = _load_records_locked(store)
        if len(persisted) != len(records) + 1 or persisted[-1] != record:
            raise RuntimeError("study journal append did not persist the exact phase record")
        return record


def _load_records_locked(store: StudyJournalStore) -> tuple[StudyPhaseRecord, ...]:
    record_names: list[tuple[int, StudyPhase, Path]] = []
    for entry in os.scandir(store.directory):
        match = _RECORD_PATTERN.fullmatch(entry.name)
        if match is None:
            continue
        try:
            phase = StudyPhase(match.group("phase"))
        except ValueError as exc:
            raise ValueError(f"study journal contains unknown phase file {entry.name!r}") from exc
        record_names.append((int(match.group("sequence")), phase, Path(entry.path)))
    record_names.sort(key=lambda item: (item[0], item[1].value))

    records: list[StudyPhaseRecord] = []
    for expected_sequence, (sequence, phase, path) in enumerate(record_names):
        if sequence != expected_sequence or phase is not STUDY_PHASES[expected_sequence]:
            raise ValueError("study journal phase files are missing, duplicated, or out of order")
        expected_path = _record_path(store.directory, sequence, phase)
        if path != expected_path:
            raise ValueError("study journal phase filename is not canonical")
        record = StudyPhaseRecord.model_validate_json(_read_regular_file(path))
        expected_previous = records[-1].digest if records else None
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
    return tuple(records)


def _record_path(directory: Path, sequence: int, phase: StudyPhase) -> Path:
    return directory / f"{sequence:03d}-{phase.value}.json"


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
    if not stat.S_ISDIR(metadata.st_mode):
        raise OSError("study journal path must be a directory, not a symlink or file")
    if metadata.st_mode & 0o077:
        raise OSError("study journal directory cannot be accessible by group or other users")
    if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise OSError("study journal directory must be owned by the current user")
    return metadata.st_dev, metadata.st_ino


def _read_regular_file(path: Path) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise OSError("study journal record must be a regular file") from exc
        raise
    try:
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
    finally:
        os.close(descriptor)


def _publish_regular_file_once(path: Path, payload: bytes) -> None:
    if len(payload) > _MAX_RECORD_BYTES:
        raise OSError("study journal record exceeds its size limit")
    temporary = path.parent / f".tmp-{path.name}-{uuid.uuid4().hex}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if _read_regular_file(path) != payload:
                raise ValueError(
                    "study journal file already exists with different content"
                ) from None
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


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
