"""Score-independent grouped benchmark partitions with a sealed confirmation view."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import secrets
import stat
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictInt, model_validator

if os.name == "posix":
    import fcntl
else:
    fcntl = None

PARTITION_MANIFEST_VERSION: Literal["1"] = "1"
_CANDIDATE_FREEZE_RECORD_VERSION: Literal["1"] = "1"
_CONFIRMATION_OPENING_RECORD_VERSION: Literal["1"] = "1"
_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_PRIVATE_TOKEN_PATTERN = r"^[0-9a-f]{64}$"
_MAX_CONTROL_RECORD_BYTES = 64 * 1024
_CONTROL_RECORD_MODE = 0o600
_CONTROL_STORE_MODE = 0o700


class PartitionTask(BaseModel):
    """Immutable identity and score-independent grouping metadata for one task."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(min_length=1)
    stratum: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    content_digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _reject_ambiguous_names(self) -> Self:
        for field in ("task_id", "stratum", "group_id"):
            value = getattr(self, field)
            if value != value.strip():
                raise ValueError(f"{field} cannot have leading or trailing whitespace")
        return self


class StratumCount(BaseModel):
    """Canonical count for one score-independent task stratum."""

    model_config = ConfigDict(frozen=True)

    stratum: str = Field(min_length=1)
    count: StrictInt = Field(ge=0)


class PartitionControlScope(BaseModel):
    """Stable experiment and protocol identity for partition genesis."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str = Field(min_length=1)
    protocol_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_ambiguous_names(self) -> Self:
        for field in ("experiment_id", "protocol_id"):
            value = getattr(self, field)
            if value != value.strip():
                raise ValueError(f"{field} cannot have leading or trailing whitespace")
        return self


class PartitionControlStore:
    """Canonical private store for one machine's partition control records.

    Local records establish race-safe and process-crash-safe idempotence for cooperating WMH
    processes inside this configured store. They do not establish chronology across machines or
    resist same-uid writers that ignore the advisory lock. Production experiments must also commit
    their record digests to an authoritative external append-only ledger.
    """

    def __init__(self, directory: str | Path) -> None:
        """Open an existing private directory and reject unsafe filesystem metadata."""
        if fcntl is None:
            raise RuntimeError("partition control store requires POSIX file locking")
        self._directory = Path(os.path.abspath(Path(directory).expanduser()))
        descriptor = _open_control_directory(self._directory)
        try:
            metadata = os.fstat(descriptor)
            self._directory_identity = (metadata.st_dev, metadata.st_ino)
        finally:
            os.close(descriptor)

    @property
    def directory(self) -> Path:
        """Return the configured private control directory."""
        return self._directory

    @contextmanager
    def _locked_directory(self) -> Iterator[int]:
        """Lock the original directory inode and yield its anchored descriptor."""
        with _locked_control_directory(
            self._directory,
            expected_identity=self._directory_identity,
        ) as descriptor:
            yield descriptor


class PartitionGenesisRecord(BaseModel):
    """One-shot private seed commitment created before partition selection."""

    model_config = ConfigDict(frozen=True)

    partition_version: Literal["1"] = PARTITION_MANIFEST_VERSION
    control_scope: PartitionControlScope
    tasks_digest: str = Field(pattern=_DIGEST_PATTERN)
    discovery_strata: tuple[StratumCount, ...]
    selection_seed: str = Field(pattern=_PRIVATE_TOKEN_PATTERN, repr=False)
    seal_nonce: str = Field(pattern=_PRIVATE_TOKEN_PATTERN, repr=False)

    @property
    def digest(self) -> str:
        """Return the canonical identity of this private genesis record."""
        return _canonical_digest(self.model_dump(mode="json"))


class DiscoveryTask(BaseModel):
    """Proposer-safe task identity with no task-level split metadata."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(min_length=1)
    content_digest: str = Field(pattern=_DIGEST_PATTERN)


class CandidateFreezeRecord(BaseModel):
    """Append-only evidence freezing one candidate and protocol before confirmation opens."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_version: Literal["1"] = _CANDIDATE_FREEZE_RECORD_VERSION
    partition_manifest_digest: str = Field(pattern=_DIGEST_PATTERN)
    confirmation_commitment: str = Field(pattern=_DIGEST_PATTERN)
    candidate_execution_digest: str = Field(pattern=_DIGEST_PATTERN)
    confirmation_protocol_digest: str = Field(pattern=_DIGEST_PATTERN)

    @property
    def digest(self) -> str:
        """Return the canonical identity of the candidate-freeze event."""
        return _canonical_digest(self.model_dump(mode="json"))


class ConfirmationOpeningRecord(BaseModel):
    """Append-only evidence binding a sealed partition to its frozen candidate and protocol."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_version: Literal["1"] = _CONFIRMATION_OPENING_RECORD_VERSION
    partition_manifest_digest: str = Field(pattern=_DIGEST_PATTERN)
    confirmation_commitment: str = Field(pattern=_DIGEST_PATTERN)
    candidate_execution_digest: str = Field(pattern=_DIGEST_PATTERN)
    confirmation_protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    candidate_freeze_digest: str = Field(pattern=_DIGEST_PATTERN)

    @property
    def digest(self) -> str:
        """Return the canonical identity of the one-shot opening event."""
        return _canonical_digest(self.model_dump(mode="json"))


class DiscoveryPartition(BaseModel):
    """Proposer-safe partition view containing discovery tasks but no held-out identities."""

    model_config = ConfigDict(frozen=True)

    partition_version: Literal["1"]
    partition_manifest_digest: str = Field(pattern=_DIGEST_PATTERN)
    tasks: tuple[DiscoveryTask, ...]
    confirmation_strata: tuple[StratumCount, ...]
    confirmation_commitment: str = Field(pattern=_DIGEST_PATTERN)

    @property
    def confirmation_counts(self) -> dict[str, int]:
        """Return held-out counts without revealing held-out task identities."""
        return {item.stratum: item.count for item in self.confirmation_strata}


class ConfirmationPartition(BaseModel):
    """Held-out task view opened only after binding an already-frozen candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    partition_version: Literal["1"]
    partition_manifest_digest: str = Field(pattern=_DIGEST_PATTERN)
    candidate_execution_digest: str = Field(pattern=_DIGEST_PATTERN)
    confirmation_protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    tasks: tuple[PartitionTask, ...]
    confirmation_commitment: str = Field(pattern=_DIGEST_PATTERN)
    candidate_freeze_digest: str = Field(pattern=_DIGEST_PATTERN)
    opening_record_digest: str = Field(pattern=_DIGEST_PATTERN)


class BenchmarkPartitionManifest(BaseModel):
    """Private control-plane manifest for one grouped discovery/confirmation split."""

    model_config = ConfigDict(frozen=True)

    partition_version: Literal["1"] = PARTITION_MANIFEST_VERSION
    genesis_digest: str = Field(pattern=_DIGEST_PATTERN)
    control_scope: PartitionControlScope
    tasks: tuple[PartitionTask, ...]
    discovery_strata: tuple[StratumCount, ...]
    selection_seed: str = Field(min_length=1)
    seal_nonce: str = Field(min_length=16, repr=False)
    discovery_task_ids: tuple[str, ...]
    confirmation_task_ids: tuple[str, ...]
    confirmation_commitment: str = Field(pattern=_DIGEST_PATTERN)

    @classmethod
    def create(
        cls,
        *,
        tasks: tuple[PartitionTask, ...],
        discovery_counts: dict[str, int],
        genesis: PartitionGenesisRecord,
    ) -> BenchmarkPartitionManifest:
        """Select whole groups from a previously persisted private genesis record."""
        canonical_tasks = _canonical_tasks(tasks)
        canonical_counts = _canonical_counts(
            discovery_counts,
            tasks=canonical_tasks,
        )
        _validate_genesis(
            genesis,
            tasks=canonical_tasks,
            discovery_counts=canonical_counts,
        )
        discovery_ids, confirmation_ids = _select_partition(
            canonical_tasks,
            canonical_counts,
            seed=genesis.selection_seed,
        )
        commitment = _confirmation_commitment(
            tasks=canonical_tasks,
            confirmation_ids=confirmation_ids,
            nonce=genesis.seal_nonce,
        )
        return cls(
            genesis_digest=genesis.digest,
            control_scope=genesis.control_scope,
            tasks=canonical_tasks,
            discovery_strata=tuple(
                StratumCount(stratum=stratum, count=count)
                for stratum, count in canonical_counts.items()
            ),
            selection_seed=genesis.selection_seed,
            seal_nonce=genesis.seal_nonce,
            discovery_task_ids=discovery_ids,
            confirmation_task_ids=confirmation_ids,
            confirmation_commitment=commitment,
        )

    @property
    def digest(self) -> str:
        """Return the salted identity of the complete private partition manifest."""
        return _canonical_digest(self.model_dump(mode="json"))

    @property
    def discovery_counts(self) -> dict[str, int]:
        """Return exact per-stratum discovery counts."""
        return {item.stratum: item.count for item in self.discovery_strata}

    @property
    def confirmation_counts(self) -> dict[str, int]:
        """Return exact per-stratum confirmation counts."""
        discovery = self.discovery_counts
        totals = Counter(task.stratum for task in self.tasks)
        return {stratum: totals[stratum] - discovery[stratum] for stratum in sorted(totals)}

    def discovery_view(self) -> DiscoveryPartition:
        """Return the only partition view that optimizer search may receive."""
        discovery = frozenset(self.discovery_task_ids)
        return DiscoveryPartition(
            partition_version=self.partition_version,
            partition_manifest_digest=self.digest,
            tasks=tuple(
                DiscoveryTask(task_id=task.task_id, content_digest=task.content_digest)
                for task in self.tasks
                if task.task_id in discovery
            ),
            confirmation_strata=tuple(
                StratumCount(stratum=stratum, count=count)
                for stratum, count in self.confirmation_counts.items()
            ),
            confirmation_commitment=self.confirmation_commitment,
        )

    @model_validator(mode="after")
    def _validate_derived_partition(self) -> Self:
        canonical_tasks = _canonical_tasks(self.tasks)
        if self.tasks != canonical_tasks:
            raise ValueError("partition tasks must be in canonical order")
        counts = _canonical_counts(self.discovery_counts, tasks=self.tasks)
        if self.discovery_strata != tuple(
            StratumCount(stratum=stratum, count=count) for stratum, count in counts.items()
        ):
            raise ValueError("discovery strata must be unique and in canonical order")
        genesis = PartitionGenesisRecord(
            control_scope=self.control_scope,
            tasks_digest=_tasks_digest(self.tasks),
            discovery_strata=self.discovery_strata,
            selection_seed=self.selection_seed,
            seal_nonce=self.seal_nonce,
        )
        if self.genesis_digest != genesis.digest:
            raise ValueError("partition genesis digest does not match its frozen inputs")
        expected_discovery, expected_confirmation = _select_partition(
            self.tasks,
            counts,
            seed=self.selection_seed,
        )
        if (
            self.discovery_task_ids != expected_discovery
            or self.confirmation_task_ids != expected_confirmation
        ):
            raise ValueError("partition membership does not match the frozen selection_seed")
        expected_commitment = _confirmation_commitment(
            tasks=self.tasks,
            confirmation_ids=self.confirmation_task_ids,
            nonce=self.seal_nonce,
        )
        if self.confirmation_commitment != expected_commitment:
            raise ValueError("confirmation commitment does not match the sealed partition")
        return self


def initialize_partition_genesis(
    control_store: PartitionControlStore,
    *,
    scope: PartitionControlScope,
    tasks: tuple[PartitionTask, ...],
    discovery_counts: dict[str, int],
) -> PartitionGenesisRecord:
    """Create or load one immutable CSPRNG-backed partition genesis record.

    The canonical record key binds the experiment, protocol, task roster, and discovery quotas.
    Callers cannot choose a record path, so repeated initialization cannot silently shop for a
    favorable split within the configured control store.
    """
    canonical_tasks = _canonical_tasks(tasks)
    canonical_counts = _canonical_counts(discovery_counts, tasks=canonical_tasks)
    strata = tuple(
        StratumCount(stratum=stratum, count=count) for stratum, count in canonical_counts.items()
    )
    record_name = _genesis_record_name(
        scope=scope,
        tasks_digest=_tasks_digest(canonical_tasks),
        discovery_strata=strata,
    )
    with control_store._locked_directory() as directory_descriptor:
        existing = _read_optional_control_record(directory_descriptor, record_name)
        if existing is not None:
            record = PartitionGenesisRecord.model_validate_json(existing)
            _validate_genesis(
                record,
                scope=scope,
                tasks=canonical_tasks,
                discovery_counts=canonical_counts,
            )
            return record

        proposed = PartitionGenesisRecord(
            control_scope=scope,
            tasks_digest=_tasks_digest(canonical_tasks),
            discovery_strata=strata,
            selection_seed=secrets.token_hex(32),
            seal_nonce=secrets.token_hex(32),
        )
        if _publish_control_record(
            directory_descriptor,
            record_name,
            proposed.model_dump(mode="json"),
        ):
            return proposed
        winner = PartitionGenesisRecord.model_validate_json(
            _read_control_record(directory_descriptor, record_name)
        )
        _validate_genesis(
            winner,
            scope=scope,
            tasks=canonical_tasks,
            discovery_counts=canonical_counts,
        )
        return winner


def freeze_confirmation_candidate(
    control_store: PartitionControlStore,
    *,
    manifest: BenchmarkPartitionManifest,
    candidate_execution_digest: str,
    confirmation_protocol_digest: str,
) -> CandidateFreezeRecord:
    """Persist the sole candidate and protocol allowed to open one confirmation split."""
    manifest = _revalidate_partition_manifest(manifest)
    proposed = CandidateFreezeRecord(
        partition_manifest_digest=manifest.digest,
        confirmation_commitment=manifest.confirmation_commitment,
        candidate_execution_digest=candidate_execution_digest,
        confirmation_protocol_digest=confirmation_protocol_digest,
    )
    record_name = _candidate_freeze_record_name(manifest.digest)
    with control_store._locked_directory() as directory_descriptor:
        _require_manifest_genesis(directory_descriptor, manifest)
        existing = _read_optional_control_record(directory_descriptor, record_name)
        if existing is None and _publish_control_record(
            directory_descriptor,
            record_name,
            proposed.model_dump(mode="json"),
        ):
            return proposed
        record = CandidateFreezeRecord.model_validate_json(
            existing
            if existing is not None
            else _read_control_record(directory_descriptor, record_name)
        )
        if record != proposed:
            raise ValueError(
                "confirmation candidate is already frozen to a different manifest, candidate, "
                "or protocol"
            )
        return record


def open_confirmation_once(
    control_store: PartitionControlStore,
    *,
    manifest: BenchmarkPartitionManifest,
    confirmation_protocol_digest: str,
) -> ConfirmationPartition:
    """Open held-out identities once for the frozen candidate and confirmation protocol."""
    manifest = _revalidate_partition_manifest(manifest)
    freeze_name = _candidate_freeze_record_name(manifest.digest)
    opening_name = _confirmation_opening_record_name(manifest.digest)
    with control_store._locked_directory() as directory_descriptor:
        _require_manifest_genesis(directory_descriptor, manifest)
        freeze = CandidateFreezeRecord.model_validate_json(
            _read_control_record(directory_descriptor, freeze_name)
        )
        proposed = ConfirmationOpeningRecord(
            partition_manifest_digest=manifest.digest,
            confirmation_commitment=manifest.confirmation_commitment,
            candidate_execution_digest=freeze.candidate_execution_digest,
            confirmation_protocol_digest=confirmation_protocol_digest,
            candidate_freeze_digest=freeze.digest,
        )
        if (
            freeze.partition_manifest_digest != manifest.digest
            or freeze.confirmation_commitment != manifest.confirmation_commitment
            or freeze.confirmation_protocol_digest != proposed.confirmation_protocol_digest
        ):
            raise ValueError(
                "candidate freeze does not match the sealed partition manifest and protocol"
            )
        existing = _read_optional_control_record(directory_descriptor, opening_name)
        if existing is None and not _publish_control_record(
            directory_descriptor,
            opening_name,
            proposed.model_dump(mode="json"),
        ):
            existing = _read_control_record(directory_descriptor, opening_name)
        if existing is not None:
            record = ConfirmationOpeningRecord.model_validate_json(existing)
            if record != proposed:
                raise ValueError(
                    "confirmation partition is already opened for a different freeze record"
                )
        else:
            record = proposed

    confirmation = frozenset(manifest.confirmation_task_ids)
    return ConfirmationPartition(
        partition_version=manifest.partition_version,
        partition_manifest_digest=manifest.digest,
        candidate_execution_digest=record.candidate_execution_digest,
        confirmation_protocol_digest=record.confirmation_protocol_digest,
        tasks=tuple(task for task in manifest.tasks if task.task_id in confirmation),
        confirmation_commitment=manifest.confirmation_commitment,
        candidate_freeze_digest=record.candidate_freeze_digest,
        opening_record_digest=record.digest,
    )


def _validate_genesis(
    genesis: PartitionGenesisRecord,
    *,
    scope: PartitionControlScope | None = None,
    tasks: tuple[PartitionTask, ...],
    discovery_counts: dict[str, int],
) -> None:
    expected_strata = tuple(
        StratumCount(stratum=stratum, count=count) for stratum, count in discovery_counts.items()
    )
    if scope is not None and genesis.control_scope != scope:
        raise ValueError("partition genesis control scope differs from the requested scope")
    if genesis.tasks_digest != _tasks_digest(tasks):
        raise ValueError("partition genesis task roster differs from the requested manifest")
    if genesis.discovery_strata != expected_strata:
        raise ValueError("partition genesis quotas differ from the requested manifest")


def _revalidate_partition_manifest(
    manifest: BenchmarkPartitionManifest,
) -> BenchmarkPartitionManifest:
    """Re-run all derived invariants for a model instance received at a trust boundary."""
    return BenchmarkPartitionManifest.model_validate_json(manifest.model_dump_json())


def _require_manifest_genesis(
    directory_descriptor: int,
    manifest: BenchmarkPartitionManifest,
) -> None:
    record_name = _genesis_record_name(
        scope=manifest.control_scope,
        tasks_digest=_tasks_digest(manifest.tasks),
        discovery_strata=manifest.discovery_strata,
    )
    try:
        record = PartitionGenesisRecord.model_validate_json(
            _read_control_record(directory_descriptor, record_name)
        )
    except FileNotFoundError as error:
        raise ValueError(
            "partition control store does not contain this manifest's genesis record"
        ) from error
    _validate_genesis(
        record,
        scope=manifest.control_scope,
        tasks=manifest.tasks,
        discovery_counts=manifest.discovery_counts,
    )
    if record.digest != manifest.genesis_digest:
        raise ValueError("partition control store genesis differs from the manifest")


def _tasks_digest(tasks: tuple[PartitionTask, ...]) -> str:
    return _canonical_digest([task.model_dump(mode="json") for task in tasks])


def _genesis_record_name(
    *,
    scope: PartitionControlScope,
    tasks_digest: str,
    discovery_strata: tuple[StratumCount, ...],
) -> str:
    inputs_digest = _canonical_digest(
        {
            "tasks_digest": tasks_digest,
            "discovery_strata": [item.model_dump(mode="json") for item in discovery_strata],
        }
    )
    key_digest = _canonical_digest(
        {
            "domain": "wmh-partition-genesis-key-v1",
            "control_scope": scope.model_dump(mode="json"),
            "partition_inputs_digest": inputs_digest,
        }
    )
    return f"partition-genesis-{_digest_hex(key_digest)}.json"


def _candidate_freeze_record_name(partition_digest: str) -> str:
    return f"candidate-freeze-{_digest_hex(partition_digest)}.json"


def _confirmation_opening_record_name(partition_digest: str) -> str:
    return f"confirmation-opening-{_digest_hex(partition_digest)}.json"


def _digest_hex(digest: str) -> str:
    hexadecimal = digest.removeprefix("sha256:")
    if (
        not digest.startswith("sha256:")
        or len(hexadecimal) != 64
        or any(character not in "0123456789abcdef" for character in hexadecimal)
    ):
        raise ValueError("partition control record key must be a canonical SHA-256 digest")
    return hexadecimal


@contextmanager
def _locked_control_directory(
    path: Path,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> Iterator[int]:
    if fcntl is None:
        raise RuntimeError("partition control store requires POSIX file locking")
    descriptor = _open_control_directory(path)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _validate_control_directory(
            descriptor,
            path,
            expected_identity=expected_identity,
        )
        yield descriptor
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _open_control_directory(path: Path) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as error:
        raise ValueError(
            f"partition control store must be an existing private directory: {path}"
        ) from error
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR} and _path_is_symlink(path):
            raise ValueError(
                f"partition control store cannot be a symbolic link: {path}"
            ) from error
        raise
    try:
        _validate_control_directory(descriptor, path)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _path_is_symlink(path: Path) -> bool:
    try:
        return stat.S_ISLNK(os.lstat(path).st_mode)
    except FileNotFoundError:
        return False


def _validate_control_directory(
    descriptor: int,
    path: Path,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"partition control store is not a directory: {path}")
    if metadata.st_uid != os.getuid():
        raise ValueError(f"partition control store must be owned by the current uid: {path}")
    if stat.S_IMODE(metadata.st_mode) != _CONTROL_STORE_MODE:
        raise ValueError(f"partition control store must have mode 0700: {path}")
    actual_identity = (metadata.st_dev, metadata.st_ino)
    if expected_identity is not None and actual_identity != expected_identity:
        raise ValueError(f"partition control store changed after it was opened: {path}")


def _publish_control_record(
    directory_descriptor: int,
    record_name: str,
    value: JsonValue,
) -> bool:
    """Atomically publish one complete private record under the held directory lock."""
    if _read_optional_control_record(directory_descriptor, record_name) is not None:
        return False
    payload = _canonical_json(value).encode() + b"\n"
    temporary_name = f".partition-record-{secrets.token_hex(32)}.tmp"
    descriptor = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CLOEXEC | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        _CONTROL_RECORD_MODE,
        dir_fd=directory_descriptor,
    )
    try:
        os.fchmod(descriptor, _CONTROL_RECORD_MODE)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # The private directory lock serializes writers. Rename makes the complete record visible
        # in one step, and a directory fsync makes the publication durable before returning.
        os.rename(
            temporary_name,
            record_name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        temporary_name = ""
        os.fsync(directory_descriptor)
        published = os.open(
            record_name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_descriptor,
        )
        try:
            _validate_control_record(published, record_name)
        finally:
            os.close(published)
        return True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass


def _read_optional_control_record(
    directory_descriptor: int,
    record_name: str,
) -> bytes | None:
    try:
        return _read_control_record(directory_descriptor, record_name)
    except FileNotFoundError:
        return None


def _read_control_record(directory_descriptor: int, record_name: str) -> bytes:
    try:
        descriptor = os.open(
            record_name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError(
                f"partition control record cannot be a symbolic link: {record_name}"
            ) from error
        raise
    try:
        _validate_control_record(descriptor, record_name)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            payload = handle.read(_MAX_CONTROL_RECORD_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) > _MAX_CONTROL_RECORD_BYTES:
        raise ValueError(f"partition control record exceeds the size limit: {record_name}")
    return payload


def _validate_control_record(descriptor: int, record_name: str) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"partition control record is not a regular file: {record_name}")
    if metadata.st_uid != os.getuid():
        raise ValueError(
            f"partition control record must be owned by the current uid: {record_name}"
        )
    if stat.S_IMODE(metadata.st_mode) != _CONTROL_RECORD_MODE:
        raise ValueError(f"partition control record must have mode 0600: {record_name}")
    if metadata.st_nlink != 1:
        raise ValueError(f"partition control record must have exactly one link: {record_name}")


def _canonical_tasks(tasks: tuple[PartitionTask, ...]) -> tuple[PartitionTask, ...]:
    if not tasks:
        raise ValueError("benchmark partition needs at least one task")
    duplicate_ids = sorted(
        task_id for task_id, count in Counter(task.task_id for task in tasks).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"benchmark partition has duplicate task_id values: {duplicate_ids}")
    return tuple(sorted(tasks, key=lambda task: task.task_id))


def _canonical_counts(
    counts: dict[str, int],
    *,
    tasks: tuple[PartitionTask, ...],
) -> dict[str, int]:
    totals = Counter(task.stratum for task in tasks)
    if set(counts) != set(totals):
        missing = sorted(set(totals) - set(counts))
        extra = sorted(set(counts) - set(totals))
        raise ValueError(
            f"discovery count strata differ from tasks: missing={missing}, extra={extra}"
        )
    canonical: dict[str, int] = {}
    for stratum in sorted(totals):
        value = counts[stratum]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("discovery counts must be integers")
        if value < 0 or value > totals[stratum]:
            raise ValueError(f"discovery count for {stratum!r} is outside the task roster")
        canonical[stratum] = value
    if sum(canonical.values()) == 0 or sum(canonical.values()) == len(tasks):
        raise ValueError("benchmark partition needs non-empty discovery and confirmation sets")
    return canonical


def _select_partition(
    tasks: tuple[PartitionTask, ...],
    discovery_counts: dict[str, int],
    *,
    seed: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not seed or seed != seed.strip():
        raise ValueError("selection_seed must be a non-empty canonical string")
    strata = tuple(discovery_counts)
    target = tuple(discovery_counts[stratum] for stratum in strata)
    grouped: defaultdict[str, list[PartitionTask]] = defaultdict(list)
    for task in tasks:
        grouped[task.group_id].append(task)
    ranked_groups = sorted(
        grouped,
        key=lambda group_id: (_digest_bytes(seed, "group-order", group_id), group_id),
    )
    zero = (0,) * len(strata)
    states: dict[tuple[int, ...], tuple[int, tuple[str, ...]]] = {zero: (0, ())}
    for group_id in ranked_groups:
        vector_counts = Counter(task.stratum for task in grouped[group_id])
        vector = tuple(vector_counts[stratum] for stratum in strata)
        weight = int.from_bytes(_digest_bytes(seed, "group-weight", group_id)[:8], "big")
        updated = dict(states)
        for current, (score, selected) in states.items():
            candidate_vector = tuple(
                left + right for left, right in zip(current, vector, strict=True)
            )
            if any(value > limit for value, limit in zip(candidate_vector, target, strict=True)):
                continue
            candidate = (score + weight, tuple(sorted((*selected, group_id))))
            incumbent = updated.get(candidate_vector)
            if incumbent is None or candidate < incumbent:
                updated[candidate_vector] = candidate
        states = updated
    solution = states.get(target)
    if solution is None:
        raise ValueError(
            "whole-group benchmark partition cannot satisfy the exact discovery counts"
        )
    selected_groups = frozenset(solution[1])
    discovery = tuple(sorted(task.task_id for task in tasks if task.group_id in selected_groups))
    confirmation = tuple(
        sorted(task.task_id for task in tasks if task.group_id not in selected_groups)
    )
    return discovery, confirmation


def _confirmation_commitment(
    *,
    tasks: tuple[PartitionTask, ...],
    confirmation_ids: tuple[str, ...],
    nonce: str,
) -> str:
    if len(nonce) < 16 or nonce != nonce.strip():
        raise ValueError("seal_nonce must be a canonical secret with at least 16 characters")
    return _canonical_digest(
        {
            "domain": "wmh-benchmark-confirmation-v1",
            "nonce": nonce,
            "tasks": [task.model_dump(mode="json") for task in tasks],
            "confirmation_task_ids": list(confirmation_ids),
        }
    )


def _digest_bytes(*parts: str) -> bytes:
    hasher = hashlib.sha256()
    for part in parts:
        encoded = part.encode()
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.digest()


def _canonical_digest(value: JsonValue) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _canonical_json(value: JsonValue) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
