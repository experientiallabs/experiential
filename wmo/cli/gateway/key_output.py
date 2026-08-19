"""Crash-recoverable one-time virtual-key output ownership."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import ValidationError, field_validator

from wmo.common.core.artifacts import ContractModel, Sha256, canonical_json_bytes, sha256_json
from wmo.runtime.gateway.auth import utc_text
from wmo.runtime.gateway.sqlite import key_delivery
from wmo.runtime.gateway.sqlite.store import SQLiteGatewayStore

_KEY_OUTPUT_LENGTH: Literal[64] = 64


class KeyOutputRecoveryError(ValueError):
    """Durable key-output state cannot be safely reconciled automatically."""


class KeyOutputOutcomeUnknownError(KeyOutputRecoveryError):
    """SQLite cannot prove whether one durably delivered key committed."""

    def __init__(self, *, key_id: str, prefix: str) -> None:
        """Create one content-free recovery error.

        Args:
            key_id: Stable non-secret key identifier.
            prefix: Public virtual-key prefix.
        """
        self.key_id = key_id
        self.prefix = prefix
        super().__init__("operation_outcome_unknown: preserve the key output and recovery marker")


class _KeyOutputMarker(ContractModel):
    """Private content-free ownership record written before secret bytes."""

    schema_version: Literal[1] = 1
    organization_id: str
    identity_id: str
    key_id: str
    operation_id: str | None
    request_sha256: Sha256
    prefix: str
    fingerprint_version: int
    fingerprint_sha256: Sha256
    expires_at: str | None
    created_at: str
    target_path_sha256: Sha256
    output_sha256: Sha256
    output_length: Literal[64]
    reservation_name: str
    target_device: int
    target_inode: int

    @field_validator("reservation_name")
    @classmethod
    def _require_reservation_basename(cls, value: str) -> str:
        """Reject reservation paths that could escape the output directory."""
        if (
            not value
            or value in {".", ".."}
            or Path(value).name != value
            or not value.startswith(".")
            or not value.endswith(".reserve")
        ):
            raise ValueError("key output reservation must be one basename")
        return value

    def evidence(self) -> key_delivery.KeyDeliveryEvidence:
        """Return the exact database reconciliation identity."""
        return key_delivery.KeyDeliveryEvidence(
            organization_id=self.organization_id,
            identity_id=self.identity_id,
            key_id=self.key_id,
            operation_id=self.operation_id,
            request_sha256=self.request_sha256,
            prefix=self.prefix,
            fingerprint_version=self.fingerprint_version,
            fingerprint_sha256=self.fingerprint_sha256,
            expires_at=self.expires_at,
            created_at=self.created_at,
        )


@dataclass(frozen=True, slots=True)
class RecoveredKeyOutput:
    """A committed key whose exact one-time output survived a process crash."""

    key_id: str
    prefix: str


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    """Device and inode identity read from one no-follow descriptor."""

    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


def deliver_key_output(
    path: Path,
    raw_key: str,
    evidence: key_delivery.KeyDeliveryEvidence,
) -> key_delivery.KeyDeliveryHooks:
    """Publish one private key behind a durable content-free ownership marker.

    Args:
        path: New mode-0600 output path.
        raw_key: One-time virtual-key secret.
        evidence: Content-free authority identity used after a crash.

    Returns:
        Hooks that remove exact output after rollback or settle its marker after commit.
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = f"{raw_key}\n".encode()
    if len(payload) != _KEY_OUTPUT_LENGTH:
        raise ValueError("virtual key output has an unexpected length")
    reservation = path.with_name(f".{path.name}.{uuid.uuid4().hex}.reserve")
    marker_path = key_output_marker_path(path)
    descriptor = -1
    reservation_identity: _FileIdentity | None = None
    marker: _KeyOutputMarker | None = None
    marker_identity: _FileIdentity | None = None
    target_linked = False
    try:
        descriptor = os.open(
            reservation,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        created = os.fstat(descriptor)
        reservation_identity = _identity_from_stat(created)
        marker = _KeyOutputMarker(
            **asdict(evidence),
            target_path_sha256=_target_path_sha256(path),
            output_sha256=hashlib.sha256(payload).hexdigest(),
            output_length=_KEY_OUTPUT_LENGTH,
            reservation_name=reservation.name,
            target_device=created.st_dev,
            target_inode=created.st_ino,
        )
        marker_identity = _write_marker_once(marker_path, marker)
        os.link(reservation, path, follow_symlinks=False)
        target_linked = True
        _fsync_directory(path.parent)
        reservation_identity = _recorded_inode_identity(reservation, marker)
        if reservation_identity is None or reservation_identity.size != 0:
            raise KeyOutputRecoveryError("key output reservation changed before publication")
        _unlink_exact_inode(reservation, reservation_identity)
        _fsync_directory(reservation.parent)
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count == 0:
                raise OSError("secret output write made no progress")
            written += count
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _fsync_directory(path.parent)
    except BaseException as publication_error:
        if descriptor >= 0:
            os.close(descriptor)
        if marker is not None and target_linked and _lexists(path):
            target_identity = _recorded_inode_identity(path, marker)
            if target_identity is None:
                raise KeyOutputRecoveryError(
                    "key output changed during publication; preserving recovery evidence"
                ) from publication_error
            _unlink_exact_inode(path, target_identity)
            _fsync_directory(path.parent)
        if _lexists(reservation):
            current_reservation = (
                _recorded_inode_identity(reservation, marker)
                if marker is not None
                else reservation_identity
            )
            if current_reservation is None:
                raise KeyOutputRecoveryError(
                    "key output reservation changed; preserving recovery evidence"
                ) from publication_error
            _unlink_exact_inode(reservation, current_reservation)
            _fsync_directory(reservation.parent)
        if marker_identity is not None:
            _unlink_exact_inode(marker_path, marker_identity)
            _fsync_directory(marker_path.parent)
        raise

    assert marker is not None

    def rollback() -> None:
        """Remove only exact-owned output after SQLite proves rollback."""
        if _lexists(_reservation_path(path, marker)):
            raise KeyOutputRecoveryError(
                "key output reservation still exists; preserving recovery evidence"
            )
        identity = _require_exact_output(path, marker)
        _unlink_exact_inode(path, identity)
        _fsync_directory(path.parent)
        marker_identity = _read_marker(marker_path)[1]
        _unlink_exact_inode(marker_path, marker_identity)
        _fsync_directory(marker_path.parent)

    return key_delivery.KeyDeliveryHooks(rollback=rollback)


def recover_key_output(
    path: Path,
    *,
    store: SQLiteGatewayStore,
    organization_id: str,
    identity_id: str,
    key_id: str,
    operation_id: str | None,
    expires_at: datetime | None,
) -> RecoveredKeyOutput | None:
    """Reconcile one interrupted key-output publication before a retry.

    Args:
        path: Requested one-time output path.
        store: Initialized gateway authority store.
        organization_id: Owning tenant from the retry command.
        identity_id: Credential owner from the retry command.
        key_id: Stable credential identifier from the retry command.
        operation_id: Optional retry-safe mutation identifier.
        expires_at: Optional requested expiry.

    Returns:
        A committed recovered output, or ``None`` when issuance may proceed.

    Raises:
        KeyOutputRecoveryError: Existing evidence or output cannot be safely reconciled.
        KeyOutputOutcomeUnknownError: SQLite cannot prove commit or rollback.
    """
    marker_path = key_output_marker_path(path)
    if not _lexists(marker_path):
        return None
    marker, marker_identity = _read_marker(marker_path)
    reservation = _reservation_path(path, marker)
    expected_request_sha256 = sha256_json(
        {
            "organization_id": organization_id,
            "identity_id": identity_id,
            "key_id": key_id,
            "expires_at": None if expires_at is None else utc_text(expires_at),
        }
    )
    expected = (
        marker.organization_id == organization_id
        and marker.identity_id == identity_id
        and marker.key_id == key_id
        and marker.operation_id == operation_id
        and hmac.compare_digest(marker.request_sha256, expected_request_sha256)
        and hmac.compare_digest(marker.target_path_sha256, _target_path_sha256(path))
    )
    if not expected:
        raise KeyOutputRecoveryError(
            "existing key output recovery evidence belongs to a different operation"
        )
    outcome = key_delivery.reconcile_delivery_evidence(
        store.database_path,
        busy_timeout_ms=store.busy_timeout_ms,
        evidence=marker.evidence(),
    )
    output_exists = _lexists(path)
    reservation_exists = _lexists(reservation)
    output_identity = _exact_output_identity(path, marker) if output_exists else None
    if outcome is False:
        rollback_identity = _recorded_inode_identity(path, marker) if output_exists else None
        reservation_identity = (
            _recorded_inode_identity(reservation, marker) if reservation_exists else None
        )
        if output_exists and rollback_identity is None:
            raise KeyOutputRecoveryError(
                "interrupted key output differs from its durable ownership evidence"
            )
        if reservation_exists and (reservation_identity is None or reservation_identity.size != 0):
            raise KeyOutputRecoveryError(
                "interrupted key output reservation differs from its ownership evidence"
            )
        if rollback_identity is not None:
            _unlink_exact_inode(path, rollback_identity)
            _fsync_directory(path.parent)
        if reservation_identity is not None:
            current_reservation = _recorded_inode_identity(reservation, marker)
            if current_reservation is None or current_reservation.size != 0:
                raise KeyOutputRecoveryError(
                    "key output reservation changed before recovery cleanup"
                )
            _unlink_exact_inode(reservation, current_reservation)
            _fsync_directory(reservation.parent)
        _unlink_exact_inode(marker_path, marker_identity)
        _fsync_directory(marker_path.parent)
        return None
    if outcome is True:
        if output_identity is None or reservation_exists:
            raise KeyOutputRecoveryError(
                "committed key output or reservation differs from its ownership evidence"
            )
        return RecoveredKeyOutput(key_id=marker.key_id, prefix=marker.prefix)
    raise KeyOutputOutcomeUnknownError(key_id=marker.key_id, prefix=marker.prefix)


def settle_key_output(path: Path) -> None:
    """Remove a committed output marker only after visible CLI success.

    Args:
        path: Exact private output whose secret bytes must remain untouched.

    Raises:
        KeyOutputRecoveryError: Marker or output identity changed before settlement.
    """
    marker_path = key_output_marker_path(path)
    marker, marker_identity = _read_marker(marker_path)
    reservation = _reservation_path(path, marker)
    if _lexists(reservation):
        raise KeyOutputRecoveryError(
            "key output reservation still exists; preserving recovery evidence"
        )
    _require_exact_output(path, marker)
    _unlink_exact_inode(marker_path, marker_identity)
    _fsync_directory(marker_path.parent)


def key_output_marker_path(path: Path) -> Path:
    """Return the deterministic private recovery marker beside one output path.

    Args:
        path: Requested one-time secret output.

    Returns:
        Sidecar path that never contains raw secret material.
    """
    return path.with_name(f".{path.name}.wmo-key-delivery.json")


def _write_marker_once(path: Path, marker: _KeyOutputMarker) -> _FileIdentity:
    """Atomically persist one private marker without replacing prior evidence."""
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        payload = canonical_json_bytes(marker) + b"\n"
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count == 0:
                raise OSError("key output marker write made no progress")
            written += count
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        identity = _identity_from_stat(os.fstat(descriptor))
        os.close(descriptor)
        descriptor = -1
        os.link(staging, path, follow_symlinks=False)
        staging.unlink()
        _fsync_directory(path.parent)
        return identity
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        staging.unlink(missing_ok=True)
        raise


def _read_marker(path: Path) -> tuple[_KeyOutputMarker, _FileIdentity]:
    """Load one regular private marker without following a symlink."""
    try:
        payload, identity = _read_private_regular(path, maximum_length=16 * 1024)
        return _KeyOutputMarker.model_validate_json(payload), identity
    except (OSError, ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise KeyOutputRecoveryError("key output recovery evidence is invalid") from exc


def _require_exact_output(path: Path, marker: _KeyOutputMarker) -> _FileIdentity:
    """Reject cleanup when a secret output no longer matches durable ownership."""
    identity = _exact_output_identity(path, marker)
    if identity is None:
        raise KeyOutputRecoveryError(
            "key output differs from its durable ownership evidence; preserving both files"
        )
    return identity


def _exact_output_identity(path: Path, marker: _KeyOutputMarker) -> _FileIdentity | None:
    """Return exact no-follow output identity when bytes match durable evidence."""
    try:
        payload, identity = _read_private_regular(path, maximum_length=marker.output_length)
        if len(payload) != marker.output_length:
            return None
    except (KeyOutputRecoveryError, OSError):
        return None
    if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), marker.output_sha256):
        return None
    return identity


def _recorded_inode_identity(path: Path, marker: _KeyOutputMarker) -> _FileIdentity | None:
    """Return one bounded private file matching the recorded reservation inode."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            metadata = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        return None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o077
        or metadata.st_size > marker.output_length
        or (metadata.st_dev, metadata.st_ino) != (marker.target_device, marker.target_inode)
    ):
        return None
    return _identity_from_stat(metadata)


def _reservation_path(path: Path, marker: _KeyOutputMarker) -> Path:
    """Resolve and validate the marker-bound reservation beside one output."""
    prefix = f".{path.name}."
    suffix = ".reserve"
    name = marker.reservation_name
    token = name[len(prefix) : -len(suffix)] if name.startswith(prefix) else ""
    if (
        not name.endswith(suffix)
        or len(token) != 32
        or any(character not in "0123456789abcdef" for character in token)
    ):
        raise KeyOutputRecoveryError("key output reservation identity is invalid")
    return path.with_name(name)


def _read_private_regular(
    path: Path,
    *,
    maximum_length: int,
) -> tuple[bytes, _FileIdentity]:
    """Read bounded bytes and identity from the same private no-follow descriptor."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise KeyOutputRecoveryError("key output state is not a private regular file")
        if metadata.st_size > maximum_length:
            raise KeyOutputRecoveryError("key output state exceeds its expected size")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 4096))
            if not chunk:
                raise KeyOutputRecoveryError("key output state changed while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks), _identity_from_stat(metadata)
    finally:
        os.close(descriptor)


def _unlink_exact_inode(path: Path, identity: _FileIdentity) -> None:
    """Unlink only when a final no-follow stat still names the verified inode."""
    _require_exact_inode(path, identity)
    path.unlink()


def _require_exact_inode(path: Path, identity: _FileIdentity) -> None:
    """Require a final no-follow stat to retain every verified file attribute."""
    metadata = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    ) != (
        identity.device,
        identity.inode,
        identity.size,
        identity.modified_ns,
        identity.changed_ns,
    ):
        raise KeyOutputRecoveryError("key output state changed before cleanup")


def _identity_from_stat(metadata: os.stat_result) -> _FileIdentity:
    """Capture the cleanup-relevant identity from one descriptor stat."""
    return _FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def _lexists(path: Path) -> bool:
    """Return whether a path entry exists without following a symlink."""
    return os.path.lexists(path)


def _target_path_sha256(path: Path) -> Sha256:
    """Hash one canonical absolute target spelling without following symlinks."""
    return hashlib.sha256(os.fsencode(os.path.abspath(path))).hexdigest()


def _fsync_directory(path: Path) -> None:
    """Durably persist one directory mutation or raise."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
