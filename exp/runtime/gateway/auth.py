"""Virtual-key issuance and versioned local fingerprint protection."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import stat
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

from pydantic import AwareDatetime

from exp.common.core.artifacts import ContractModel, Sha256
from exp.runtime.gateway.contracts import IdentityId, OrganizationId, VirtualKeyId


class GatewayAuthError(ValueError):
    """A virtual key or fingerprint-pepper operation is invalid."""


class IssuedVirtualKey(ContractModel):
    """One newly issued virtual key whose raw value is revealed exactly once."""

    key_id: VirtualKeyId
    organization_id: OrganizationId
    identity_id: IdentityId
    prefix: str
    raw_key: str
    expires_at: AwareDatetime | None
    created_at: AwareDatetime


class PepperKey(ContractModel):
    """One decoded HMAC key selected by its stored fingerprint version."""

    version: int
    value: bytes


class FingerprintPepperFile:
    """Atomically created, user-only file containing versioned HMAC keys."""

    def __init__(self, path: Path) -> None:
        """Initialize the pepper file owner.

        Args:
            path: User-only file path outside the gateway database.
        """
        self._path = path
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        """Return the configured pepper file path."""
        return self._path

    def current(self) -> PepperKey:
        """Load or atomically create the current fingerprint key.

        Returns:
            The current version and decoded 256-bit key.
        """
        with self._lock:
            if not self._path.exists():
                self._create_initial()
            current_version, keys = self._read()
            return PepperKey(version=current_version, value=keys[current_version])

    def key(self, version: int) -> PepperKey:
        """Load one retained fingerprint key version.

        Args:
            version: Stored fingerprint version.

        Returns:
            The matching decoded HMAC key.

        Raises:
            GatewayAuthError: The version is not retained.
        """
        with self._lock:
            _, keys = self._read()
            try:
                value = keys[version]
            except KeyError as exc:
                raise GatewayAuthError("virtual-key fingerprint version is unavailable") from exc
            return PepperKey(version=version, value=value)

    def rotate(self) -> int:
        """Append and select a fresh key while retaining older versions.

        Returns:
            The new current version.
        """
        with self._lock:
            current_version, keys = self._read()
            next_version = current_version + 1
            keys[next_version] = secrets.token_bytes(32)
            self._replace(next_version, keys)
            return next_version

    def _create_initial(self) -> None:
        """Create version one without following links or replacing existing state."""
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        payload = self._encode(1, {1: secrets.token_bytes(32)})
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self._path, flags, 0o600)
        except FileExistsError:
            return
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            self._path.unlink(missing_ok=True)
            raise

    def _read(self) -> tuple[int, dict[int, bytes]]:
        """Validate permissions and decode all retained key versions."""
        try:
            metadata = self._path.lstat()
        except FileNotFoundError as exc:
            raise GatewayAuthError("virtual-key fingerprint pepper is missing") from exc
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise GatewayAuthError(
                "virtual-key fingerprint pepper must be a regular mode-0600 file"
            )
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            current_version = int(payload["current_version"])
            encoded_keys = payload["keys"]
            if not isinstance(encoded_keys, dict):
                raise TypeError("keys is not an object")
            keys = {
                int(version): base64.b64decode(encoded, validate=True)
                for version, encoded in encoded_keys.items()
                if isinstance(version, str) and isinstance(encoded, str)
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise GatewayAuthError("virtual-key fingerprint pepper is invalid") from exc
        if current_version not in keys or any(len(value) != 32 for value in keys.values()):
            raise GatewayAuthError("virtual-key fingerprint pepper is invalid")
        return current_version, keys

    def _replace(self, current_version: int, keys: dict[int, bytes]) -> None:
        """Atomically replace the pepper file with a validated version set."""
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self._path.name}.", dir=self._path.parent
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(self._encode(current_version, keys))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._path)
            os.chmod(self._path, 0o600)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _encode(current_version: int, keys: dict[int, bytes]) -> bytes:
        """Encode a deterministic pepper document without logging key bytes."""
        payload = {
            "current_version": current_version,
            "keys": {
                str(version): base64.b64encode(keys[version]).decode("ascii")
                for version in sorted(keys)
            },
        }
        return (json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n").encode()


def issue_key_material() -> tuple[str, str]:
    """Create a display-safe prefix and a raw key with at least 256 random bits.

    Returns:
        The non-secret prefix and one-time raw key.
    """
    prefix = secrets.token_hex(6)
    secret = secrets.token_urlsafe(32)
    return prefix, f"exp_vk_{prefix}_{secret}"


def key_prefix(raw_key: str) -> str:
    """Extract the lookup prefix without exposing the secret component.

    Args:
        raw_key: Caller-supplied virtual key.

    Returns:
        The non-secret prefix.

    Raises:
        GatewayAuthError: The key has no valid gateway shape.
    """
    parts = raw_key.split("_", 3)
    if len(parts) != 4 or parts[:2] != ["exp", "vk"] or len(parts[2]) != 12 or not parts[3]:
        raise GatewayAuthError("virtual key is invalid")
    return parts[2]


def fingerprint_virtual_key(raw_key: str, pepper: PepperKey) -> Sha256:
    """Return the version-selected HMAC-SHA256 fingerprint for one raw key.

    Args:
        raw_key: Caller-supplied or newly issued key.
        pepper: Retained HMAC key version.

    Returns:
        Lowercase hexadecimal HMAC digest.
    """
    return hmac.new(pepper.value, raw_key.encode("utf-8"), hashlib.sha256).hexdigest()


def utc_text(value: datetime) -> str:
    """Serialize a timezone-aware timestamp for lexical SQLite comparison.

    Args:
        value: Timezone-aware timestamp.

    Returns:
        UTC ISO-8601 text with microsecond precision.

    Raises:
        ValueError: The value is timezone-naive.
    """
    if value.tzinfo is None:
        raise ValueError("gateway timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds")
