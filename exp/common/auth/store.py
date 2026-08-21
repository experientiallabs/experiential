"""Atomic, user-only ``auth.json`` store keyed by provider connection ID.

The document follows the OpenCode ``auth.json`` shape: one object whose keys are connection
IDs and whose values are ``{"type": "api", "key": "..."}`` records. Secret values never
appear in ``repr``, ``str``, or raised messages.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from exp.common.auth.paths import default_auth_path
from exp.common.core.artifacts import SECRET_REDACTION_PLACEHOLDER, ContractModel
from exp.common.core.locks import file_write_lock

_CONNECTION_ID_MAX = 128
_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600


class ProviderAuthStoreError(ValueError):
    """The local credential file cannot be read or written safely."""

    def __init__(self, message: str) -> None:
        """Record a recovery message that never includes credential values.

        Args:
            message: Operator-facing explanation without secret material.
        """
        super().__init__(message)


class StoredCredentialStatus(ContractModel):
    """Public metadata for one stored or configured provider connection."""

    connection_id: str
    provider: str
    source: Literal["environment", "stored", "missing", "aws_chain"]
    environment_variable: str | None = None


class ProviderAuthStore:
    """Read and replace one user-data credential file without logging secrets."""

    def __init__(self, path: Path | None = None) -> None:
        """Bind one credential file path.

        Args:
            path: Explicit ``auth.json`` path. When omitted, the platform user-data path is used.
        """
        self._path = path if path is not None else default_auth_path()

    @property
    def path(self) -> Path:
        """Return the configured credential file path."""
        return self._path

    def __repr__(self) -> str:
        """Describe the store by path only."""
        return f"ProviderAuthStore(path={self._path!r})"

    def __str__(self) -> str:
        """Describe the store by path only."""
        return self.__repr__()

    def get(self, connection_id: str) -> str | None:
        """Return the stored API key for one connection, or ``None`` when absent.

        Args:
            connection_id: Exact catalog or gateway connection name.

        Returns:
            The non-empty stored key, or ``None`` when that connection has no record.

        Raises:
            ProviderAuthStoreError: The file exists but cannot be used.
        """
        records = self._load()
        return records.get(connection_id)

    def put(self, connection_id: str, secret: str) -> None:
        """Create or replace the stored API key for one connection.

        Args:
            connection_id: Exact catalog or gateway connection name.
            secret: Non-empty API key to persist.

        Raises:
            ProviderAuthStoreError: The identity or secret is invalid, or the write failed.
        """
        _validate_connection_id(connection_id)
        key = secret.strip()
        if not key:
            raise ProviderAuthStoreError("stored credential values must be non-empty")
        with file_write_lock(self._path, what="provider credential file"):
            records = self._load()
            records[connection_id] = key
            self._replace(records)

    def remove(self, connection_id: str) -> bool:
        """Delete only the stored credential for one connection.

        Args:
            connection_id: Exact catalog or gateway connection name.

        Returns:
            ``True`` when a stored record was removed.

        Raises:
            ProviderAuthStoreError: The file exists but cannot be used.
        """
        _validate_connection_id(connection_id)
        with file_write_lock(self._path, what="provider credential file"):
            records = self._load()
            if connection_id not in records:
                return False
            del records[connection_id]
            self._replace(records)
            return True

    def connection_ids(self) -> tuple[str, ...]:
        """Return stored connection IDs in sorted order without secret values.

        Returns:
            Connection IDs present in the file.

        Raises:
            ProviderAuthStoreError: The file exists but cannot be used.
        """
        return tuple(sorted(self._load()))

    def _load(self) -> dict[str, str]:
        """Read and validate the credential document.

        Returns:
            Connection ID to API key mapping.

        Raises:
            ProviderAuthStoreError: The path is unsafe or the document is malformed.
        """
        try:
            metadata = self._path.lstat()
        except FileNotFoundError:
            return {}
        if stat.S_ISLNK(metadata.st_mode):
            raise ProviderAuthStoreError(_malformed_message(self._path))
        if not stat.S_ISREG(metadata.st_mode):
            raise ProviderAuthStoreError(_malformed_message(self._path))
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProviderAuthStoreError(_malformed_message(self._path)) from exc
        return _parse_document(payload, path=self._path)

    def _replace(self, records: Mapping[str, str]) -> None:
        """Atomically replace the credential file with user-only permissions.

        Args:
            records: Complete connection ID to API key mapping to persist.

        Raises:
            ProviderAuthStoreError: The destination is unsafe or the write failed.
        """
        self._prepare_directory()
        if self._path.exists() or self._path.is_symlink():
            metadata = self._path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ProviderAuthStoreError(_malformed_message(self._path))
        payload = json.dumps(
            {name: {"type": "api", "key": records[name]} for name in sorted(records)},
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self._path.name}.",
            dir=self._path.parent,
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, _FILE_MODE)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._path)
            os.chmod(self._path, _FILE_MODE)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _prepare_directory(self) -> None:
        """Create the credential directory and restrict it to the current user."""
        directory = self._path.parent
        directory.mkdir(mode=_DIRECTORY_MODE, parents=True, exist_ok=True)
        if directory.is_symlink():
            raise ProviderAuthStoreError(
                f"provider credential directory {directory} must be a regular directory"
            )
        if not directory.is_dir():
            raise ProviderAuthStoreError(
                f"provider credential directory {directory} must be a regular directory"
            )
        os.chmod(directory, _DIRECTORY_MODE)


def _validate_connection_id(connection_id: str) -> None:
    """Reject empty or oversized connection identities.

    Args:
        connection_id: Candidate catalog or gateway connection name.

    Raises:
        ProviderAuthStoreError: The identity cannot be used as a store key.
    """
    if not connection_id or connection_id.strip() != connection_id:
        raise ProviderAuthStoreError("connection IDs must be non-empty and unpadded")
    if len(connection_id) > _CONNECTION_ID_MAX:
        raise ProviderAuthStoreError("connection IDs must be at most 128 characters")
    if "/" in connection_id or "\\" in connection_id or "\x00" in connection_id:
        raise ProviderAuthStoreError("connection IDs must not contain path separators")


def _parse_document(payload: object, *, path: Path) -> dict[str, str]:
    """Validate one OpenCode-shaped credential document.

    Args:
        payload: Decoded JSON value.
        path: File path used in recovery messages.

    Returns:
        Connection ID to API key mapping.

    Raises:
        ProviderAuthStoreError: The document is not a usable credential object.
    """
    if not isinstance(payload, dict):
        raise ProviderAuthStoreError(_malformed_message(path))
    records: dict[str, str] = {}
    for raw_name, raw_record in payload.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ProviderAuthStoreError(_malformed_message(path))
        if not isinstance(raw_record, dict):
            raise ProviderAuthStoreError(_malformed_message(path))
        record_type = raw_record.get("type")
        key = raw_record.get("key")
        extra = set(raw_record) - {"type", "key"}
        if extra or record_type != "api" or not isinstance(key, str) or not key.strip():
            raise ProviderAuthStoreError(_malformed_message(path))
        records[raw_name] = key
    return records


def _malformed_message(path: Path) -> str:
    """Return a recoverable malformed-file error that never quotes file contents.

    Args:
        path: Credential file the operator can move or replace.

    Returns:
        Error text naming the path and the login recovery command.
    """
    return (
        f"provider credential file {path} is malformed; move or delete it, then run "
        "'exp auth login'"
    )


def redacted_secret() -> str:
    """Return the shared secret placeholder used in public summaries."""
    return SECRET_REDACTION_PLACEHOLDER
