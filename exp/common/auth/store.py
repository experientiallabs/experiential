"""Atomic, user-only ``auth.json`` store keyed by provider connection ID.

The document follows the OpenCode ``auth.json`` shape: one object whose keys are connection
IDs and whose values are either ``{"type": "api", "key": "..."}`` API-key records or
``{"type": "oauth", "access": "...", "refresh": "...", "expires": <unix ms>}`` sign-in
records for subscription connections. Optional provider, endpoint, and credential-locator
fields bind a record to one secret-free connection identity. Secret values never appear in
``repr``, ``str``, or raised messages.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from exp.common.auth.paths import default_auth_path
from exp.common.core.artifacts import ContractModel, Sha256
from exp.common.core.files import fsync_directory_best_effort
from exp.common.core.locks import file_write_lock

logger = logging.getLogger(__name__)
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


class StoredCredentialBinding(ContractModel):
    """Secret-free provider endpoint identity bound to one stored API key."""

    provider: str
    endpoint_sha256: Sha256
    credential_locator_sha256: Sha256 | None = None


class StoredCredentialEndpointMismatch(ProviderAuthStoreError):
    """A stored key exists but was saved for a different provider endpoint."""


class StoredCredentialKindMismatch(ProviderAuthStoreError):
    """A stored record exists but is an API key where a sign-in was expected, or vice versa."""


class StoredCredentialStatus(ContractModel):
    """Public metadata for one stored or configured provider connection."""

    connection_id: str
    provider: str
    source: Literal["environment", "stored", "signed_in", "missing", "aws_chain", "mismatch"]
    environment_variable: str | None = None


@dataclass(frozen=True)
class StoredOAuthTokens:
    """One connection's subscription sign-in: bearer tokens plus the access-token expiry.

    Secret values are omitted from ``repr`` and ``str``.
    """

    access_token: str
    refresh_token: str
    expires_at_ms: int
    """Unix time in milliseconds at which ``access_token`` stops being accepted."""
    account_id: str | None = None
    """Provider account identifier the tokens belong to, when the provider issues one."""

    def __repr__(self) -> str:
        """Describe the record without token values."""
        return (
            f"StoredOAuthTokens(expires_at_ms={self.expires_at_ms!r}, "
            f"account_id={self.account_id!r}, access_token='[REDACTED]', "
            "refresh_token='[REDACTED]')"
        )

    def __str__(self) -> str:
        """Describe the record without token values."""
        return self.__repr__()

    def expires_within(self, seconds: float, *, now_ms: int) -> bool:
        """Whether the access token expires within ``seconds`` of ``now_ms``.

        Args:
            seconds: Look-ahead window.
            now_ms: Current unix time in milliseconds.

        Returns:
            ``True`` when the token is already expired or expires inside the window.
        """
        return self.expires_at_ms - now_ms <= seconds * 1_000


@dataclass(frozen=True)
class _StoredApiRecord:
    """One connection's API key plus optional endpoint binding."""

    key: str
    binding: StoredCredentialBinding | None = None


@dataclass(frozen=True)
class _StoredOAuthRecord:
    """One connection's subscription sign-in plus optional endpoint binding."""

    tokens: StoredOAuthTokens
    binding: StoredCredentialBinding | None = None


_StoredRecord = _StoredApiRecord | _StoredOAuthRecord


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

    def get(
        self,
        connection_id: str,
        *,
        binding: StoredCredentialBinding | None = None,
    ) -> str | None:
        """Return the stored API key for one connection, or ``None`` when absent.

        Args:
            connection_id: Exact catalog or gateway connection name.
            binding: Optional current endpoint identity. When supplied, an unbound
                record or a record bound to a different endpoint is refused.

        Returns:
            The non-empty stored key, or ``None`` when that connection has no record.

        Raises:
            StoredCredentialEndpointMismatch: The stored key belongs to another endpoint.
            ProviderAuthStoreError: The file exists but cannot be used.
        """
        record = self._record(connection_id, binding=binding)
        if record is None:
            return None
        if isinstance(record, _StoredOAuthRecord):
            raise StoredCredentialKindMismatch(
                f"stored credential for connection {connection_id!r} is a subscription sign-in, "
                "not an API key; remove the connection and add it again"
            )
        return record.key

    def get_oauth(
        self,
        connection_id: str,
        *,
        binding: StoredCredentialBinding | None = None,
    ) -> StoredOAuthTokens | None:
        """Return the stored subscription sign-in for one connection, or ``None`` when absent.

        Args:
            connection_id: Exact catalog or gateway connection name.
            binding: Optional current endpoint identity. When supplied, an unbound
                record or a record bound to a different endpoint is refused.

        Returns:
            The stored tokens, or ``None`` when that connection has no record.

        Raises:
            StoredCredentialKindMismatch: The stored record is an API key.
            StoredCredentialEndpointMismatch: The stored sign-in belongs to another endpoint.
            ProviderAuthStoreError: The file exists but cannot be used.
        """
        record = self._record(connection_id, binding=binding)
        if record is None:
            return None
        if isinstance(record, _StoredApiRecord):
            raise StoredCredentialKindMismatch(
                f"stored credential for connection {connection_id!r} is an API key, not a "
                "subscription sign-in; remove the connection and add it again"
            )
        return record.tokens

    def _record(
        self,
        connection_id: str,
        *,
        binding: StoredCredentialBinding | None,
    ) -> _StoredRecord | None:
        """Load one record and enforce its endpoint binding.

        Args:
            connection_id: Exact catalog or gateway connection name.
            binding: Optional current endpoint identity to enforce.

        Returns:
            The stored record of either kind, or ``None`` when absent.

        Raises:
            StoredCredentialEndpointMismatch: The record belongs to another endpoint.
            ProviderAuthStoreError: The file exists but cannot be used.
        """
        records = self._load()
        record = records.get(connection_id)
        if record is None:
            return None
        if binding is not None and record.binding != binding:
            raise StoredCredentialEndpointMismatch(
                f"stored credential for connection {connection_id!r} does not match the "
                f"configured {binding.provider} endpoint; run 'exp config providers'"
            )
        return record

    def put(
        self,
        connection_id: str,
        secret: str,
        *,
        binding: StoredCredentialBinding | None = None,
    ) -> None:
        """Create or replace the stored API key for one connection.

        Args:
            connection_id: Exact catalog or gateway connection name.
            secret: Non-empty API key to persist.
            binding: Optional endpoint identity to store with the key. When omitted, an
                existing binding on this connection is preserved.

        Raises:
            ProviderAuthStoreError: The identity or secret is invalid, or the write failed.
        """
        _validate_connection_id(connection_id)
        key = secret.strip()
        if not key:
            raise ProviderAuthStoreError("stored credential values must be non-empty")
        with file_write_lock(self._path, what="provider credential file"):
            records = self._load()
            existing = records.get(connection_id)
            preserved = existing.binding if existing is not None else None
            records[connection_id] = _StoredApiRecord(
                key=key,
                binding=binding if binding is not None else preserved,
            )
            self._replace(records)

    def put_oauth(
        self,
        connection_id: str,
        tokens: StoredOAuthTokens,
        *,
        binding: StoredCredentialBinding | None = None,
    ) -> None:
        """Create or replace the stored subscription sign-in for one connection.

        A refreshed token pair replaces the previous one in place, so the record always
        holds the newest refresh token the provider issued.

        Args:
            connection_id: Exact catalog or gateway connection name.
            tokens: Non-empty access and refresh tokens plus the access expiry.
            binding: Optional endpoint identity to store with the sign-in. When omitted, an
                existing binding on this connection is preserved.

        Raises:
            ProviderAuthStoreError: The identity or tokens are invalid, or the write failed.
        """
        _validate_connection_id(connection_id)
        if not tokens.access_token.strip() or not tokens.refresh_token.strip():
            raise ProviderAuthStoreError("stored sign-in tokens must be non-empty")
        if tokens.expires_at_ms < 0:
            raise ProviderAuthStoreError("stored sign-in expiry must be a unix millisecond time")
        with file_write_lock(self._path, what="provider credential file"):
            records = self._load()
            existing = records.get(connection_id)
            preserved = existing.binding if existing is not None else None
            records[connection_id] = _StoredOAuthRecord(
                tokens=tokens,
                binding=binding if binding is not None else preserved,
            )
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

    def _load(self) -> dict[str, _StoredRecord]:
        """Read and validate the credential document.

        Returns:
            Connection ID to stored record mapping.

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

    def _replace(self, records: Mapping[str, _StoredRecord]) -> None:
        """Atomically replace the credential file with user-only permissions.

        The staging file is restricted before the rename. After ``os.replace`` succeeds the
        requested document is installed, so a later destination ``chmod`` or directory fsync
        failure cannot report the write as lost.

        Args:
            records: Complete connection ID to stored record mapping to persist.

        Raises:
            ProviderAuthStoreError: The destination is unsafe or the write failed before replace.
        """
        self._prepare_directory()
        if self._path.exists() or self._path.is_symlink():
            metadata = self._path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ProviderAuthStoreError(_malformed_message(self._path))
        payload = json.dumps(
            {name: _record_payload(records[name]) for name in sorted(records)},
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
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        try:
            os.chmod(self._path, _FILE_MODE)
        except OSError as exc:
            logger.warning(
                "could not restrict provider credential file mode at %s: %s",
                self._path,
                exc,
            )
        fsync_directory_best_effort(self._path.parent)

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


def _record_payload(record: _StoredRecord) -> dict[str, str | int]:
    """Serialize one stored record without extra identity when unbound.

    Args:
        record: API key or sign-in tokens plus optional endpoint binding.

    Returns:
        OpenCode-shaped object, plus binding fields when present.
    """
    payload: dict[str, str | int]
    if isinstance(record, _StoredApiRecord):
        payload = {"type": "api", "key": record.key}
    else:
        payload = {
            "type": "oauth",
            "access": record.tokens.access_token,
            "refresh": record.tokens.refresh_token,
            "expires": record.tokens.expires_at_ms,
        }
        if record.tokens.account_id is not None:
            payload["account_id"] = record.tokens.account_id
    if record.binding is not None:
        payload["provider"] = record.binding.provider
        payload["endpoint_sha256"] = record.binding.endpoint_sha256
        if record.binding.credential_locator_sha256 is not None:
            payload["credential_locator_sha256"] = record.binding.credential_locator_sha256
    return payload


def _parse_document(payload: object, *, path: Path) -> dict[str, _StoredRecord]:
    """Validate one OpenCode-shaped credential document.

    Args:
        payload: Decoded JSON value.
        path: File path used in recovery messages.

    Returns:
        Connection ID to stored record mapping.

    Raises:
        ProviderAuthStoreError: The document is not a usable credential object.
    """
    if not isinstance(payload, dict):
        raise ProviderAuthStoreError(_malformed_message(path))
    records: dict[str, _StoredRecord] = {}
    for raw_name, raw_record in payload.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ProviderAuthStoreError(_malformed_message(path))
        if not isinstance(raw_record, dict):
            raise ProviderAuthStoreError(_malformed_message(path))
        fields: dict[str, object] = {}
        for field, value in raw_record.items():
            if not isinstance(field, str):
                raise ProviderAuthStoreError(_malformed_message(path))
            fields[field] = value
        records[raw_name] = _parse_record(fields, path=path)
    return records


_BINDING_FIELDS = frozenset({"provider", "endpoint_sha256", "credential_locator_sha256"})


def _parse_record(fields: Mapping[str, object], *, path: Path) -> _StoredRecord:
    """Validate one stored record of either kind.

    Args:
        fields: Decoded record object with string keys.
        path: File path used in recovery messages.

    Returns:
        The typed API-key or sign-in record.

    Raises:
        ProviderAuthStoreError: The record has an unknown type, unknown fields, or a
            missing or empty required value.
    """
    record_type = fields.get("type")
    binding = _parse_binding(fields, path=path)
    if record_type == "api":
        key = fields.get("key")
        extra = set(fields) - {"type", "key"} - _BINDING_FIELDS
        if extra or not isinstance(key, str) or not key.strip():
            raise ProviderAuthStoreError(_malformed_message(path))
        return _StoredApiRecord(key=key, binding=binding)
    if record_type == "oauth":
        access = fields.get("access")
        refresh = fields.get("refresh")
        expires = fields.get("expires")
        account_id = fields.get("account_id")
        extra = set(fields) - {"type", "access", "refresh", "expires", "account_id"}
        extra -= _BINDING_FIELDS
        if (
            extra
            or not isinstance(access, str)
            or not access.strip()
            or not isinstance(refresh, str)
            or not refresh.strip()
            or isinstance(expires, bool)
            or not isinstance(expires, int)
            or expires < 0
            or (account_id is not None and (not isinstance(account_id, str) or not account_id))
        ):
            raise ProviderAuthStoreError(_malformed_message(path))
        return _StoredOAuthRecord(
            tokens=StoredOAuthTokens(
                access_token=access,
                refresh_token=refresh,
                expires_at_ms=expires,
                account_id=account_id,
            ),
            binding=binding,
        )
    raise ProviderAuthStoreError(_malformed_message(path))


def _parse_binding(
    raw_record: Mapping[str, object],
    *,
    path: Path,
) -> StoredCredentialBinding | None:
    """Read optional endpoint binding fields from one stored record.

    Args:
        raw_record: Decoded record object.
        path: File path used in recovery messages.

    Returns:
        Binding when both identity fields are present, otherwise ``None``.

    Raises:
        ProviderAuthStoreError: Only one binding field is present or a field is invalid.
    """
    provider = raw_record.get("provider")
    endpoint_sha256 = raw_record.get("endpoint_sha256")
    credential_locator_sha256 = raw_record.get("credential_locator_sha256")
    if provider is None and endpoint_sha256 is None and credential_locator_sha256 is None:
        return None
    if not isinstance(provider, str) or not provider:
        raise ProviderAuthStoreError(_malformed_message(path))
    if not isinstance(endpoint_sha256, str) or not endpoint_sha256:
        raise ProviderAuthStoreError(_malformed_message(path))
    if credential_locator_sha256 is not None and (
        not isinstance(credential_locator_sha256, str) or not credential_locator_sha256
    ):
        raise ProviderAuthStoreError(_malformed_message(path))
    try:
        return StoredCredentialBinding(
            provider=provider,
            endpoint_sha256=endpoint_sha256,
            credential_locator_sha256=credential_locator_sha256,
        )
    except ValueError as exc:
        raise ProviderAuthStoreError(_malformed_message(path)) from exc


def _malformed_message(path: Path) -> str:
    """Return a recoverable malformed-file error that never quotes file contents.

    Args:
        path: Credential file the operator can move or replace.

    Returns:
        Error text naming the path and the providers recovery command.
    """
    return (
        f"provider credential file {path} is malformed; move or delete it, then run "
        "'exp config providers'"
    )
