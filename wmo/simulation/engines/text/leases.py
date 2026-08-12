"""Durable cross-process paid-cell claims for text world-model simulation."""

from __future__ import annotations

import logging
import os
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from pydantic import Field, ValidationError, field_validator

from wmo.common.core.artifacts import ArtifactId, ContractModel, Sha256, canonical_json_bytes
from wmo.common.core.locks import file_write_lock

logger = logging.getLogger(__name__)

_LEASE_DIRECTORY_NAME = "simulation-leases"
_LEASE_SUFFIX = ".json"
_DEFAULT_STALE_AFTER_SECONDS = 15 * 60
_DEFAULT_POLL_INTERVAL_SECONDS = 0.02


class TextCellLeaseError(RuntimeError):
    """An in-flight paid-cell claim cannot safely be acquired or released."""


class TextCellLeaseState(StrEnum):
    """Result of a durable cell-admission attempt."""

    OWNED = "owned"
    COMPLETED = "completed"
    BUDGET_BLOCKED = "budget_blocked"
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
    claimed_at: datetime
    expires_at: datetime

    @field_validator("claimed_at", "expires_at")
    @classmethod
    def _require_aware_timestamp(cls, value: datetime) -> datetime:
        """Reject a lease timestamp that cannot be compared across process boundaries."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("text-cell lease timestamps must include a timezone")
        return value

    @field_validator("reserved_cost_usd")
    @classmethod
    def _require_matching_reservation(cls, value: float | None) -> float | None:
        """Require finite-budget leases to retain a concrete positive reservation."""
        if value is not None and not value > 0:
            raise ValueError("text-cell lease reservation must be positive")
        return value


@dataclass(frozen=True)
class TextCellLeaseClaim:
    """One lease admission decision with the known pre-admission spend total."""

    state: TextCellLeaseState
    lease: TextCellLease | None
    observed_spend_usd: float | None


class TextCellLeaseStore:
    """Coordinate one local project's paid text-simulation cells across processes.

    A claim file is atomically created before a candidate or world-model provider call. A process
    that sees a live claim waits for its immutable rollout. An expired claim with a dead owner is
    intentionally not replayed: the earlier process may have paid a provider just before crashing,
    so the caller writes a recovery failure rather than risking duplicate spend.
    """

    def __init__(
        self,
        project_directory: Path,
        *,
        clock: Callable[[], datetime],
        owner_alive: Callable[[int], bool] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        stale_after_seconds: float = _DEFAULT_STALE_AFTER_SECONDS,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        """Create a project-local claim store without making a claim yet.

        Args:
            project_directory: Canonical mutable directory for one WMO project.
            clock: Aware wall clock used for durable lease timestamps.
            owner_alive: Process-liveness probe, injectable for deterministic recovery tests.
            sleep: Short follower wait seam, injectable for deterministic tests.
            stale_after_seconds: Minimum retained duration before a dead claim is stale.
            poll_interval_seconds: Follower interval while another process owns a live claim.

        Raises:
            ValueError: A lease lifetime or follower interval is not positive.
        """
        if stale_after_seconds <= 0:
            raise ValueError("text-cell lease stale_after_seconds must be positive")
        if poll_interval_seconds <= 0:
            raise ValueError("text-cell lease poll_interval_seconds must be positive")
        self._directory = project_directory / _LEASE_DIRECTORY_NAME
        self._clock = clock
        self._owner_alive = _owner_process_is_alive if owner_alive is None else owner_alive
        self._sleep = sleep
        self._stale_after = timedelta(seconds=stale_after_seconds)
        self._poll_interval_seconds = poll_interval_seconds

    def acquire(
        self,
        *,
        lease_id: ArtifactId,
        resolution_id: ArtifactId,
        simulation_id: ArtifactId,
        rollout_id: ArtifactId,
        binding_sha256: Sha256,
        maximum_cost_usd: float | None,
        rollout_completed: Callable[[], bool],
        observed_spend_usd: Callable[[], float | None],
    ) -> TextCellLeaseClaim:
        """Atomically reserve one paid cell, or wait for its completed immutable artifact.

        Args:
            lease_id: Stable local filename for the exact resolution and cell binding.
            resolution_id: Immutable alias and task resolution artifact identity.
            simulation_id: Parent simulation identity for budget coordination.
            rollout_id: Expected immutable rollout artifact identity.
            binding_sha256: Full cell binding digest, checked against any existing claim.
            maximum_cost_usd: Optional run-wide ceiling. A finite run reserves all currently
                uncommitted budget for this one cell, preventing parallel over-admission.
            rollout_completed: Checks whether another owner has already persisted the rollout.
            observed_spend_usd: Returns known spend for completed cells or ``None`` when it is
                unpriced and later paid work must fail closed.

        Returns:
            An owned claim, completed follower result, budget block, or stale recovery result.

        Raises:
            TextCellLeaseError: A lease file is malformed, unsafe, or mismatched.
        """
        if maximum_cost_usd is not None and maximum_cost_usd <= 0:
            raise ValueError("text-cell maximum_cost_usd must be positive")
        while True:
            decision = self._admit_once(
                lease_id=lease_id,
                resolution_id=resolution_id,
                simulation_id=simulation_id,
                rollout_id=rollout_id,
                binding_sha256=binding_sha256,
                maximum_cost_usd=maximum_cost_usd,
                rollout_completed=rollout_completed,
                observed_spend_usd=observed_spend_usd,
            )
            if decision is not None:
                return decision
            self._sleep(self._poll_interval_seconds)

    def release(self, lease: TextCellLease) -> None:
        """Remove this owner's claim after its immutable rollout is safely persisted.

        Args:
            lease: Exact claim obtained from ``acquire``.
        """
        self._ensure_directory()
        path = self._path(lease.lease_id)
        try:
            with file_write_lock(self._admission_path(), what="text simulation cell admission"):
                existing = self._read_optional(path)
                if existing is None:
                    return
                if existing != lease:
                    raise TextCellLeaseError(
                        f"text-cell lease {lease.lease_id!r} changed before its owner released it"
                    )
                path.unlink()
                _fsync_directory(path.parent)
        except OSError as exc:
            logger.warning(
                "could not release text-cell lease %s after immutable rollout persistence: %s",
                lease.lease_id,
                exc,
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
        rollout_completed: Callable[[], bool],
        observed_spend_usd: Callable[[], float | None],
    ) -> TextCellLeaseClaim | None:
        """Make one lock-protected admission attempt, returning ``None`` for a live follower."""
        self._ensure_directory()
        with file_write_lock(self._admission_path(), what="text simulation cell admission"):
            if rollout_completed():
                return TextCellLeaseClaim(TextCellLeaseState.COMPLETED, None, None)
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
                if self._is_stale(existing, now):
                    return TextCellLeaseClaim(TextCellLeaseState.STALE, existing, None)
                return None
            spend = observed_spend_usd()
            reservation = self._reserve_budget(
                resolution_id=resolution_id,
                simulation_id=simulation_id,
                maximum_cost_usd=maximum_cost_usd,
                observed_spend_usd=spend,
            )
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
        resolution_id: ArtifactId,
        simulation_id: ArtifactId,
        maximum_cost_usd: float | None,
        observed_spend_usd: float | None,
    ) -> float | None:
        """Reserve all uncommitted finite budget so parallel cells cannot over-admit."""
        if maximum_cost_usd is None:
            return None
        if observed_spend_usd is None:
            return None
        active_reservations = 0.0
        for lease in self._leases_for(resolution_id=resolution_id, simulation_id=simulation_id):
            if lease.maximum_cost_usd != maximum_cost_usd:
                raise TextCellLeaseError(
                    "text simulation has incompatible finite spend ceilings for one resolution"
                )
            if lease.reserved_cost_usd is None:
                raise TextCellLeaseError(
                    "finite-budget text simulation has an unreserved active paid-cell claim"
                )
            active_reservations += lease.reserved_cost_usd
        remaining = maximum_cost_usd - observed_spend_usd - active_reservations
        if remaining <= 0:
            return None
        return remaining

    def _leases_for(
        self,
        *,
        resolution_id: ArtifactId,
        simulation_id: ArtifactId,
    ) -> tuple[TextCellLease, ...]:
        """Load every durable active reservation for one immutable simulation resolution."""
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
                leases.append(lease)
        return tuple(leases)

    def _is_stale(self, lease: TextCellLease, now: datetime) -> bool:
        """Recognize only expired claims whose local owner process is no longer alive."""
        return now >= lease.expires_at and not self._owner_alive(lease.owner_pid)

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
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError as exc:  # pragma: no cover - admission lock serializes this race
            raise TextCellLeaseError(f"text-cell lease {lease.lease_id!r} already exists") from exc
        try:
            payload = canonical_json_bytes(lease)
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        finally:
            os.close(descriptor)
        _fsync_directory(path.parent)


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


def _fsync_directory(directory: Path) -> None:
    """Best-effort persistence for a claim creation or release directory mutation."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError as exc:
        logger.warning("could not open text-cell lease directory %s for fsync: %s", directory, exc)
        return
    try:
        os.fsync(descriptor)
    except OSError as exc:
        logger.warning("could not fsync text-cell lease directory %s: %s", directory, exc)
    finally:
        try:
            os.close(descriptor)
        except OSError as exc:
            logger.warning("could not close text-cell lease directory %s: %s", directory, exc)
