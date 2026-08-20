"""Durable cross-process paid-cell claims for text world-model simulation."""

from __future__ import annotations

import logging
import math
import os
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Self
from uuid import uuid4

from pydantic import AwareDatetime, Field, ValidationError, field_validator, model_validator

from exp.common.core.artifacts import ArtifactId, ContractModel, Sha256, canonical_json_bytes
from exp.common.core.files import fsync_directory_best_effort
from exp.common.core.locks import FileLockTimeout, file_write_lock

logger = logging.getLogger(__name__)

_LEASE_DIRECTORY_NAME = "simulation-leases"
_LEASE_SUFFIX = ".json"
_DEFAULT_STALE_AFTER_SECONDS = 15 * 60
_DEFAULT_POLL_INTERVAL_SECONDS = 0.02
_DEFAULT_WAIT_TIMEOUT_SECONDS = 10.0


class TextCellLeaseError(RuntimeError):
    """An in-flight paid-cell claim cannot safely be acquired or released."""


class TextCellLeaseState(StrEnum):
    """Result of a durable cell-admission attempt."""

    OWNED = "owned"
    COMPLETED = "completed"
    BUDGET_BLOCKED = "budget_blocked"
    CONTENDED = "contended"
    STALE = "stale"


class TextCellLeaseStatus(StrEnum):
    """Durable status of a paid-cell claim or non-replayable tombstone."""

    ACTIVE = "active"
    STALE = "stale"


class TextCellLease(ContractModel):
    """One fsync-backed exclusive claim made before any provider call in a cell."""

    lease_id: ArtifactId
    resolution_id: ArtifactId
    simulation_id: ArtifactId
    rollout_id: ArtifactId
    binding_sha256: Sha256
    maximum_cost_usd: float | None = Field(default=None, gt=0)
    reserved_cost_usd: float | None = Field(default=None, gt=0)
    owner_id: str = Field(min_length=1, max_length=128)
    owner_pid: int = Field(gt=0)
    claimed_at: AwareDatetime
    expires_at: AwareDatetime
    status: TextCellLeaseStatus = TextCellLeaseStatus.ACTIVE
    unknown_spend_blocks_budget: bool = False
    dispatch_intent_recorded: bool = False

    @field_validator("reserved_cost_usd")
    @classmethod
    def _require_matching_reservation(cls, value: float | None) -> float | None:
        """Require finite-budget leases to retain a concrete positive reservation."""
        if value is not None and not value > 0:
            raise ValueError("text-cell lease reservation must be positive")
        return value

    @model_validator(mode="after")
    def _require_stale_budget_barrier(self) -> Self:
        """Keep ambiguous finite-spend tombstones blocking until rollout evidence is durable."""
        if self.status == TextCellLeaseStatus.ACTIVE:
            if self.unknown_spend_blocks_budget:
                raise ValueError("active text-cell leases cannot be unknown-spend tombstones")
            return self
        if self.maximum_cost_usd is None:
            if self.unknown_spend_blocks_budget or self.reserved_cost_usd is not None:
                raise ValueError("unbounded stale leases cannot retain a budget barrier")
            return self
        if not self.unknown_spend_blocks_budget or self.reserved_cost_usd != self.maximum_cost_usd:
            raise ValueError("finite stale leases must retain the whole-ceiling budget barrier")
        return self


@dataclass(frozen=True)
class TextCellLeaseClaim:
    """One lease admission decision with the known pre-admission spend total."""

    state: TextCellLeaseState
    lease: TextCellLease | None
    observed_spend_usd: float | None

    @property
    def retryable(self) -> bool:
        """Return whether the caller should retry after another live claim changes."""
        return self.state == TextCellLeaseState.CONTENDED


class TextCellLeaseStore:
    """Coordinate one local project's paid text-simulation cells across processes.

    A claim file is atomically created before a candidate or world-model provider call. A process
    that sees a live claim waits for its immutable rollout up to a bounded deadline. An expired
    claim with a dead owner becomes a durable non-reserving tombstone: the earlier process may have
    paid a provider just before crashing, so the cell is not replayed.
    """

    def __init__(
        self,
        project_directory: Path,
        *,
        clock: Callable[[], datetime],
        owner_alive: Callable[[int], bool] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        stale_after_seconds: float = _DEFAULT_STALE_AFTER_SECONDS,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        wait_timeout_seconds: float = _DEFAULT_WAIT_TIMEOUT_SECONDS,
    ) -> None:
        """Create a project-local claim store without making a claim yet.

        Args:
            project_directory: Canonical mutable directory for one EXP project.
            clock: Aware wall clock used for durable lease timestamps.
            owner_alive: Process-liveness probe, injectable for deterministic recovery tests.
            sleep: Short follower wait seam, injectable for deterministic tests.
            monotonic: Deadline clock, injectable with ``sleep`` for deterministic tests.
            stale_after_seconds: Minimum retained duration before a dead claim is stale.
            poll_interval_seconds: Follower interval while another process owns a live claim.
            wait_timeout_seconds: Maximum wait for active same-cell or budget contention.

        Raises:
            ValueError: A lease lifetime or follower interval is not positive.
        """
        if stale_after_seconds <= 0:
            raise ValueError("text-cell lease stale_after_seconds must be positive")
        if poll_interval_seconds <= 0:
            raise ValueError("text-cell lease poll_interval_seconds must be positive")
        if not math.isfinite(wait_timeout_seconds) or wait_timeout_seconds <= 0:
            raise ValueError("text-cell lease wait_timeout_seconds must be finite and positive")
        self._directory = project_directory / _LEASE_DIRECTORY_NAME
        self._clock = clock
        self._owner_alive = _owner_process_is_alive if owner_alive is None else owner_alive
        self._sleep = sleep
        self._monotonic = monotonic
        self._stale_after = timedelta(seconds=stale_after_seconds)
        self._poll_interval_seconds = poll_interval_seconds
        self._wait_timeout_seconds = wait_timeout_seconds

    def acquire(
        self,
        *,
        lease_id: ArtifactId,
        resolution_id: ArtifactId,
        simulation_id: ArtifactId,
        rollout_id: ArtifactId,
        binding_sha256: Sha256,
        maximum_cost_usd: float | None,
        rollout_completed: Callable[[ArtifactId], bool],
        observed_spend_usd: Callable[[], float | None],
        stop_on_overspend: bool = False,
        cancelled: Callable[[], bool] | None = None,
    ) -> TextCellLeaseClaim:
        """Atomically reserve one paid cell, or wait for its completed immutable artifact.

        Args:
            lease_id: Stable local filename for the exact resolution and cell binding.
            resolution_id: Immutable alias and task resolution artifact identity.
            simulation_id: Parent simulation identity for budget coordination.
            rollout_id: Expected immutable rollout artifact identity.
            binding_sha256: Full cell binding digest, checked against any existing claim.
            maximum_cost_usd: Optional run-wide ceiling shared by the selected cells.
            rollout_completed: Checks a lease's expected immutable rollout by artifact identity.
            observed_spend_usd: Returns known spend for completed cells or ``None`` when a
                dispatched cell's spend is unpriced.
            stop_on_overspend: When true, unknown or ceiling-reaching spend blocks admission;
                by default the authorized run continues with a logged warning.
            cancelled: Optional cooperative cancellation probe checked before and during waits.

        Returns:
            An owned claim, completed follower result, budget block, or stale recovery result.

        Raises:
            TextCellLeaseError: A lease file is malformed, unsafe, or mismatched.
        """
        if maximum_cost_usd is not None and maximum_cost_usd <= 0:
            raise ValueError("text-cell maximum_cost_usd must be positive")
        deadline = self._monotonic() + self._wait_timeout_seconds
        is_cancelled = (lambda: False) if cancelled is None else cancelled
        while True:
            if is_cancelled():
                return TextCellLeaseClaim(TextCellLeaseState.CONTENDED, None, None)
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return TextCellLeaseClaim(TextCellLeaseState.CONTENDED, None, None)
            try:
                decision = self._admit_once(
                    lease_id=lease_id,
                    resolution_id=resolution_id,
                    simulation_id=simulation_id,
                    rollout_id=rollout_id,
                    binding_sha256=binding_sha256,
                    maximum_cost_usd=maximum_cost_usd,
                    rollout_completed=rollout_completed,
                    observed_spend_usd=observed_spend_usd,
                    stop_on_overspend=stop_on_overspend,
                    lock_timeout_seconds=min(self._poll_interval_seconds, remaining),
                )
            except FileLockTimeout:
                decision = None
            if decision is not None:
                return decision
            remaining = deadline - self._monotonic()
            if remaining <= 0 or is_cancelled():
                return TextCellLeaseClaim(TextCellLeaseState.CONTENDED, None, None)
            self._sleep(min(self._poll_interval_seconds, remaining))

    def release(self, lease: TextCellLease) -> None:
        """Remove this owner's claim after its immutable rollout is safely persisted.

        Args:
            lease: Exact active claim obtained from ``acquire`` or its durable intent successor.

        Raises:
            TextCellLeaseError: The durable claim changed before its owner could release it.
        """
        self._ensure_directory()
        path = self._path(lease.lease_id)
        try:
            with file_write_lock(self._admission_path(), what="text simulation cell admission"):
                existing = self._read_optional(path)
                if existing is None:
                    return
                intended = lease.model_copy(update={"dispatch_intent_recorded": True})
                if existing != lease and existing != intended:
                    raise TextCellLeaseError(
                        f"text-cell lease {lease.lease_id!r} changed before its owner released it"
                    )
                self._reap(path, existing)
        except OSError as exc:
            logger.warning(
                "could not release text-cell lease %s after immutable rollout persistence: %s",
                lease.lease_id,
                exc,
            )

    def stale_recovery_pending(self, lease_id: ArtifactId) -> bool:
        """Return whether a dead prior claim awaits this exact cell's recovery rollout.

        A stale tombstone, or an expired claim whose owner process is gone, keeps a
        whole-ceiling budget barrier that only this cell's persisted recovery evidence
        clears, so callers should run such cells before admitting sibling cells.

        Args:
            lease_id: Stable local filename for the exact resolution and cell binding.

        Returns:
            True when a stale tombstone or a dead expired claim awaits this cell's recovery.
        """
        try:
            lease = self._read_optional(self._path(lease_id))
        except TextCellLeaseError:
            return False
        if lease is None:
            return False
        if lease.status == TextCellLeaseStatus.STALE:
            return True
        return self._is_stale(lease, _aware_now(self._clock))

    def record_dispatch_intent(self, lease: TextCellLease) -> TextCellLease:
        """Durably record intent before a candidate or environment dispatch can begin.

        Args:
            lease: Exact active claim that owns the imminent external dispatch.

        Returns:
            The exact lease record with durable dispatch intent set.

        Raises:
            TextCellLeaseError: The claim is absent, changed, or not active before intent records.
        """
        if lease.status != TextCellLeaseStatus.ACTIVE:
            raise TextCellLeaseError("only active text-cell leases can record dispatch intent")
        self._ensure_directory()
        path = self._path(lease.lease_id)
        intended = lease.model_copy(update={"dispatch_intent_recorded": True})
        with file_write_lock(self._admission_path(), what="text simulation cell admission"):
            existing = self._read_optional(path)
            if existing is None:
                raise TextCellLeaseError(
                    f"text-cell lease {lease.lease_id!r} disappeared before dispatch intent"
                )
            if existing == intended:
                return existing
            if existing == lease:
                self._replace_exact(path, expected=existing, replacement=intended)
                return intended
            raise TextCellLeaseError(
                f"text-cell lease {lease.lease_id!r} changed before dispatch intent"
            )

    def _admit_once(
        self,
        *,
        lease_id: ArtifactId,
        resolution_id: ArtifactId,
        simulation_id: ArtifactId,
        rollout_id: ArtifactId,
        binding_sha256: Sha256,
        maximum_cost_usd: float | None,
        rollout_completed: Callable[[ArtifactId], bool],
        observed_spend_usd: Callable[[], float | None],
        stop_on_overspend: bool,
        lock_timeout_seconds: float,
    ) -> TextCellLeaseClaim | None:
        """Make one lock-protected admission attempt, returning ``None`` for a live follower."""
        self._ensure_directory()
        with file_write_lock(
            self._admission_path(),
            what="text simulation cell admission",
            timeout_s=lock_timeout_seconds,
        ):
            path = self._path(lease_id)
            existing = self._read_optional(path)
            now = _aware_now(self._clock)
            if existing is not None:
                self._require_same_claim(
                    existing,
                    resolution_id=resolution_id,
                    simulation_id=simulation_id,
                    rollout_id=rollout_id,
                    binding_sha256=binding_sha256,
                    maximum_cost_usd=maximum_cost_usd,
                )
                if rollout_completed(existing.rollout_id):
                    self._reap(path, existing)
                    return TextCellLeaseClaim(TextCellLeaseState.COMPLETED, None, None)
                if existing.status == TextCellLeaseStatus.STALE:
                    return TextCellLeaseClaim(TextCellLeaseState.STALE, existing, None)
                if self._is_stale(existing, now):
                    stale = self._tombstone(path, existing)
                    return TextCellLeaseClaim(TextCellLeaseState.STALE, stale, None)
                return None
            if rollout_completed(rollout_id):
                return TextCellLeaseClaim(TextCellLeaseState.COMPLETED, None, None)
            active_leases = self._active_leases_for(
                resolution_id=resolution_id,
                simulation_id=simulation_id,
                now=now,
                rollout_completed=rollout_completed,
            )
            spend = observed_spend_usd()
            reservation, contended = self._reserve_budget(
                maximum_cost_usd=maximum_cost_usd,
                observed_spend_usd=spend,
                active_leases=active_leases,
                stop_on_overspend=stop_on_overspend,
            )
            if contended:
                return None
            if maximum_cost_usd is not None and reservation is None:
                return TextCellLeaseClaim(TextCellLeaseState.BUDGET_BLOCKED, None, spend)
            lease = TextCellLease(
                lease_id=lease_id,
                resolution_id=resolution_id,
                simulation_id=simulation_id,
                rollout_id=rollout_id,
                binding_sha256=binding_sha256,
                maximum_cost_usd=maximum_cost_usd,
                reserved_cost_usd=reservation,
                owner_id=uuid4().hex,
                owner_pid=os.getpid(),
                claimed_at=now,
                expires_at=now + self._stale_after,
            )
            self._write_exclusive(path, lease)
            return TextCellLeaseClaim(TextCellLeaseState.OWNED, lease, spend)

    def _reserve_budget(
        self,
        *,
        maximum_cost_usd: float | None,
        observed_spend_usd: float | None,
        active_leases: tuple[TextCellLease, ...],
        stop_on_overspend: bool,
    ) -> tuple[float | None, bool]:
        """Reserve budget for one paid cell under the selected overspend policy.

        By default the finite budget authorizes the run upfront, so admission never fails on
        spend: once reconciled spend reaches the authorized amount, or a prior dispatch left
        spend unknown, the cell is admitted with a logged warning and a conservative
        whole-budget reservation. In stop mode unknown or ceiling-reaching spend yields no
        reservation, so the caller blocks the cell instead of dispatching it. Finite-budget
        cells serialize on live reservations in both modes so spend reconciliation stays
        exact.
        """
        if maximum_cost_usd is None:
            return None, False
        for lease in active_leases:
            if lease.maximum_cost_usd != maximum_cost_usd:
                raise TextCellLeaseError(
                    "text simulation has incompatible finite spend ceilings for one resolution"
                )
            if lease.reserved_cost_usd is None:
                raise TextCellLeaseError(
                    "finite-budget text simulation has an unreserved active paid-cell claim"
                )
        if stop_on_overspend:
            if observed_spend_usd is None:
                return None, False
            if active_leases:
                return None, True
            remaining_ceiling = maximum_cost_usd - observed_spend_usd
            if remaining_ceiling <= 0:
                return None, False
            return remaining_ceiling, False
        if active_leases:
            return None, True
        if observed_spend_usd is None:
            logger.warning(
                "prior simulation spend is unknown; admitting the next paid cell because the "
                "run is already authorized"
            )
            return maximum_cost_usd, False
        remaining_ceiling = maximum_cost_usd - observed_spend_usd
        if remaining_ceiling <= 0:
            logger.warning(
                "reconciled simulation spend $%.4f reached the authorized $%.4f; continuing "
                "because the run is already authorized",
                observed_spend_usd,
                maximum_cost_usd,
            )
            return maximum_cost_usd, False
        return remaining_ceiling, False

    def _active_leases_for(
        self,
        *,
        resolution_id: ArtifactId,
        simulation_id: ArtifactId,
        now: datetime,
        rollout_completed: Callable[[ArtifactId], bool],
    ) -> tuple[TextCellLease, ...]:
        """Reap completed claims, tombstone dead claims, and return valid reservations."""
        leases = []
        for path in sorted(self._directory.glob(f"*{_LEASE_SUFFIX}")):
            lease = self._read_optional(path)
            if lease is None:
                continue
            if lease.resolution_id == resolution_id or lease.simulation_id == simulation_id:
                if lease.resolution_id != resolution_id or lease.simulation_id != simulation_id:
                    raise TextCellLeaseError(
                        "text-cell lease mixes incompatible simulation and resolution identities"
                    )
                if rollout_completed(lease.rollout_id):
                    self._reap(path, lease)
                elif lease.status == TextCellLeaseStatus.STALE:
                    if lease.unknown_spend_blocks_budget:
                        leases.append(lease)
                elif self._is_stale(lease, now):
                    leases.append(self._tombstone(path, lease))
                else:
                    leases.append(lease)
        return tuple(leases)

    def _is_stale(self, lease: TextCellLease, now: datetime) -> bool:
        """Recognize only expired claims whose local owner process is no longer alive."""
        return now >= lease.expires_at and not self._owner_alive(lease.owner_pid)

    def _tombstone(self, path: Path, lease: TextCellLease) -> TextCellLease:
        """Atomically retain non-replay evidence without retaining its budget reservation."""
        stale = lease.model_copy(
            update={
                "status": TextCellLeaseStatus.STALE,
                "reserved_cost_usd": lease.maximum_cost_usd,
                "unknown_spend_blocks_budget": lease.maximum_cost_usd is not None,
            }
        )
        self._replace_exact(path, expected=lease, replacement=stale)
        return stale

    def _reap(self, path: Path, expected: TextCellLease) -> None:
        """Remove a claim whose immutable rollout now makes replay impossible."""
        descriptor, metadata = self._open_exact(path, expected)
        try:
            current = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise TextCellLeaseError(f"text-cell lease {path} changed before reap")
            os.unlink(path.name, dir_fd=descriptor)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _replace_exact(
        self,
        path: Path,
        *,
        expected: TextCellLease,
        replacement: TextCellLease,
    ) -> None:
        """Replace one unchanged regular lease by directory-relative no-follow mutation."""
        directory_descriptor, metadata = self._open_exact(path, expected)
        staging_name = f".{path.name}.{uuid4().hex}.partial"
        staging_descriptor: int | None = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            staging_descriptor = os.open(staging_name, flags, 0o600, dir_fd=directory_descriptor)
            payload = canonical_json_bytes(replacement)
            with os.fdopen(staging_descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.close(staging_descriptor)
            staging_descriptor = None
            current = os.stat(path.name, dir_fd=directory_descriptor, follow_symlinks=False)
            if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (
                metadata.st_dev,
                metadata.st_ino,
            ):
                raise TextCellLeaseError(f"text-cell lease {path} changed before tombstone")
            os.replace(
                staging_name,
                path.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
            )
            os.fsync(directory_descriptor)
        finally:
            if staging_descriptor is not None:
                os.close(staging_descriptor)
            try:
                os.unlink(staging_name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
            os.close(directory_descriptor)

    def _open_exact(self, path: Path, expected: TextCellLease) -> tuple[int, os.stat_result]:
        """Open the lease directory and prove its current name still denotes expected content."""
        directory_descriptor = self._open_directory(path.parent)
        lease_descriptor: int | None = None
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            lease_descriptor = os.open(path.name, flags, dir_fd=directory_descriptor)
            metadata = os.fstat(lease_descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise TextCellLeaseError(f"text-cell lease {path} is not a regular file")
            payload = b""
            while chunk := os.read(lease_descriptor, 64 * 1024):
                payload += chunk
            current = TextCellLease.model_validate_json(payload)
            if current != expected:
                raise TextCellLeaseError(f"text-cell lease {path} changed before mutation")
            return directory_descriptor, metadata
        except (OSError, ValidationError, ValueError, TextCellLeaseError) as exc:
            os.close(directory_descriptor)
            if isinstance(exc, TextCellLeaseError):
                raise
            raise TextCellLeaseError(f"text-cell lease {path} cannot be mutated safely") from exc
        finally:
            if lease_descriptor is not None:
                os.close(lease_descriptor)

    def _require_same_claim(
        self,
        lease: TextCellLease,
        *,
        resolution_id: ArtifactId,
        simulation_id: ArtifactId,
        rollout_id: ArtifactId,
        binding_sha256: Sha256,
        maximum_cost_usd: float | None,
    ) -> None:
        """Fail closed if a stable lease filename was rebound to different immutable work."""
        if (
            lease.resolution_id != resolution_id
            or lease.simulation_id != simulation_id
            or lease.rollout_id != rollout_id
            or lease.binding_sha256 != binding_sha256
            or lease.maximum_cost_usd != maximum_cost_usd
        ):
            raise TextCellLeaseError(
                f"text-cell lease {lease.lease_id!r} is bound to different immutable work"
            )

    def _ensure_directory(self) -> None:
        """Create the project-local lease directory without accepting a symlinked target."""
        self._directory.mkdir(parents=True, exist_ok=True)
        if self._directory.is_symlink() or not self._directory.is_dir():
            raise TextCellLeaseError(
                "text simulation lease directory must be an ordinary directory"
            )

    def _path(self, lease_id: ArtifactId) -> Path:
        """Return the one checked claim pathname for a validated stable lease identifier."""
        return self._directory / f"{lease_id}{_LEASE_SUFFIX}"

    def _admission_path(self) -> Path:
        """Return the stable local lock target that serializes claim and budget mutations."""
        return self._directory / "admission"

    def _read_optional(self, path: Path) -> TextCellLease | None:
        """Load one regular, complete, typed lease record without following a symlink."""
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise TextCellLeaseError(f"text-cell lease path {path} is not a safe regular file")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise TextCellLeaseError(f"text-cell lease {path} cannot be opened safely") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise TextCellLeaseError(f"text-cell lease {path} is not a regular file")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                payload = handle.read()
        finally:
            os.close(descriptor)
        try:
            lease = TextCellLease.model_validate_json(payload)
        except (ValidationError, ValueError) as exc:
            raise TextCellLeaseError(f"text-cell lease {path} is malformed") from exc
        expected_name = f"{lease.lease_id}{_LEASE_SUFFIX}"
        if path.name != expected_name:
            raise TextCellLeaseError(f"text-cell lease {path} does not match its record identity")
        return lease

    def _write_exclusive(self, path: Path, lease: TextCellLease) -> None:
        """Create and fsync one claim via ``O_EXCL`` before any provider call may begin."""
        directory_descriptor = self._open_directory(path.parent)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path.name, flags, 0o600, dir_fd=directory_descriptor)
        except FileExistsError as exc:  # pragma: no cover - admission lock serializes this race
            os.close(directory_descriptor)
            raise TextCellLeaseError(f"text-cell lease {lease.lease_id!r} already exists") from exc
        except OSError as exc:
            os.close(directory_descriptor)
            raise TextCellLeaseError(
                f"text-cell lease {lease.lease_id!r} cannot be created safely"
            ) from exc
        try:
            payload = canonical_json_bytes(lease)
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                os.unlink(path.name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
            raise
        finally:
            os.close(descriptor)
            os.close(directory_descriptor)
        fsync_directory_best_effort(path.parent)

    @staticmethod
    def _open_directory(directory: Path) -> int:
        """Open one real lease directory without following a swapped symlink."""
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            return os.open(directory, flags)
        except OSError as exc:
            raise TextCellLeaseError(
                f"text simulation lease directory {directory} cannot be opened safely"
            ) from exc


def _aware_now(clock: Callable[[], datetime]) -> datetime:
    """Read one timezone-aware durable lease timestamp."""
    now = clock()
    if now.tzinfo is None or now.utcoffset() is None:
        raise TextCellLeaseError("text-cell lease clock must return timezone-aware datetimes")
    return now.astimezone(UTC)


def _owner_process_is_alive(pid: int) -> bool:
    """Return whether a local owner PID still exists without signaling it."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
