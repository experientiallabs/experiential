"""Canonical provenance, identity, hashing, and failure contracts.

The models in this module are the small pieces shared by every new immutable EXP artifact.
They deliberately contain no provider connection details or credential references.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, field_validator

JsonObject = dict[str, JsonValue]

ArtifactId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$",
    ),
]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

_SECRET_FIELD_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "api_key_env",
        "authorization",
        "credential",
        "credential_env",
        "credential_ref",
        "password",
        "refresh_token",
        "secret",
        "secret_env",
        "secret_ref",
        "token_env",
    }
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b", re.IGNORECASE),
)
_SECRET_ENVIRONMENT_NAME_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Z0-9]*_)*(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN|REFRESH_TOKEN|SECRET|"
    r"CREDENTIAL)(?:_[A-Z0-9]+)*\b"
)
_SECRET_REFERENCE_PATTERN = re.compile(
    r"\b(?:"
    r"access[_ -]?token|"
    r"api[_ -]?key(?:[_ -]?env)?|"
    r"authorization|"
    r"credential(?:[_ -]?(?:env|ref))?|"
    r"password|"
    r"refresh[_ -]?token|"
    r"secret(?:[_ -]?(?:env|ref))?|"
    r"token[_ -]?env"
    r")\b",
    re.IGNORECASE,
)
SECRET_REDACTION_PLACEHOLDER = "[REDACTED]"
_SECRET_REDACTION_PATTERNS = (*_SECRET_VALUE_PATTERNS, _SECRET_ENVIRONMENT_NAME_PATTERN)


class ContractModel(BaseModel):
    """Base class for immutable, persisted contract boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class FailureCode(StrEnum):
    """Stable categories for a partial artifact or operation failure."""

    VALIDATION = "validation"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    CONTEXT_OVERFLOW = "context_overflow"
    PROVIDER = "provider"
    BUDGET = "budget"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


class FailureAttribution(StrEnum):
    """The runtime boundary that owns an operation failure."""

    AGENT = "agent"
    MODEL = "model"
    TOOL = "tool"
    ENVIRONMENT = "environment"
    RESET = "reset"
    CLEANUP = "cleanup"


class StructuredFailure(ContractModel):
    """A non-secret, machine-readable description of one failed operation."""

    code: FailureCode
    message: str = Field(min_length=1)
    retryable: bool = False
    exception_type: str | None = None
    attribution: FailureAttribution | None = None
    details: JsonObject = Field(default_factory=dict)


class SourceIdentity(ContractModel):
    """Identifies immutable source evidence without storing its raw payload."""

    kind: Literal["file", "otlp", "production", "simulation", "manual", "generated"]
    source_id: str = Field(min_length=1, max_length=512)
    sha256: Sha256 | None = None


class ArtifactInput(ContractModel):
    """An immutable artifact consumed to produce another artifact."""

    artifact_id: ArtifactId
    sha256: Sha256


class ArtifactEnvelope(ContractModel):
    """Provenance shared by every completed immutable artifact.

    Args:
        schema_version: Version of the stored contract.
        created_at: Time the completed artifact was materialized, with a timezone.
        inputs: Sorted immutable artifacts used to produce this artifact.
        code_revision: Exact code revision that wrote the artifact.
        source: Direct external source identity, when the artifact has one.
    """

    schema_version: int = Field(ge=1)
    created_at: AwareDatetime
    inputs: tuple[ArtifactInput, ...] = ()
    code_revision: str = Field(min_length=1, max_length=256)
    source: SourceIdentity | None = None

    @field_validator("inputs")
    @classmethod
    def _require_unique_sorted_inputs(
        cls, value: tuple[ArtifactInput, ...]
    ) -> tuple[ArtifactInput, ...]:
        input_ids = tuple(item.artifact_id for item in value)
        if len(set(input_ids)) != len(input_ids):
            raise ValueError("artifact inputs must not repeat an artifact_id")
        if input_ids != tuple(sorted(input_ids)):
            raise ValueError("artifact inputs must be sorted by artifact_id")
        return value


def envelope_matches_manifest(envelope: ArtifactEnvelope, manifest: ArtifactEnvelope) -> bool:
    """Report whether two artifact envelopes share identical provenance identity.

    Args:
        envelope: Domain artifact envelope loaded from a data file.
        manifest: Store-level manifest (or second envelope) to compare against.

    Returns:
        True when schema_version, created_at, inputs, code_revision, and source all match.
    """
    return (
        envelope.schema_version,
        envelope.created_at,
        envelope.inputs,
        envelope.code_revision,
        envelope.source,
    ) == (
        manifest.schema_version,
        manifest.created_at,
        manifest.inputs,
        manifest.code_revision,
        manifest.source,
    )


def sorted_unique_inputs(
    *inputs: ArtifactInput, error_type: type[Exception] = ValueError
) -> tuple[ArtifactInput, ...]:
    """Canonicalize artifact inputs into the exact order and uniqueness an envelope requires.

    Args:
        *inputs: Manifest-derived references, in any order and with repeated equal references.
        error_type: Domain error raised when one artifact ID carries conflicting digests.

    Returns:
        One input per artifact ID, ordered by artifact ID.

    Raises:
        Exception: `error_type` when an artifact ID appears with two different digests.
    """
    by_id: dict[str, ArtifactInput] = {}
    for item in inputs:
        previous = by_id.get(item.artifact_id)
        if previous is not None and previous != item:
            raise error_type(f"artifact input {item.artifact_id} has conflicting manifest digests")
        by_id[item.artifact_id] = item
    return tuple(by_id[artifact_id] for artifact_id in sorted(by_id))


class SecretBoundaryError(ValueError):
    """Raised when data intended for an immutable artifact names or contains a secret."""


def canonical_json_bytes(value: BaseModel | JsonValue) -> bytes:
    """Render a Pydantic model or JSON value as deterministic UTF-8 JSON.

    Args:
        value: The structured value to serialize.

    Returns:
        JSON with sorted keys, no insignificant whitespace, and no non-finite numbers.
    """
    payload: JsonValue
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", by_alias=True, exclude_none=False)
    else:
        payload = value
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_jsonl_bytes(records: Sequence[BaseModel | JsonValue]) -> bytes:
    """Serialize ordered records as deterministic newline-terminated JSONL.

    Args:
        records: Ordered structured records to serialize, one per line.

    Returns:
        Canonical UTF-8 JSONL with a trailing newline, or empty bytes for no records.
    """
    payload = b"\n".join(canonical_json_bytes(record) for record in records)
    return payload + b"\n" if payload else b""


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 hex digest of an immutable artifact payload.

    Args:
        payload: Exact persisted bytes whose content-addressed identity is needed.

    Returns:
        The lowercase hexadecimal SHA-256 digest of the payload.
    """
    # Content-addressed artifact identity, not password or credential storage.
    return hashlib.sha256(payload, usedforsecurity=False).hexdigest()


def sha256_json(value: BaseModel | JsonValue) -> str:
    """Return the SHA-256 digest of `value`'s deterministic JSON serialization."""
    return sha256_bytes(canonical_json_bytes(value))


def stable_id(prefix: str, value: BaseModel | JsonValue) -> str:
    """Create a readable, content-addressed stable ID.

    Args:
        prefix: Lowercase artifact-kind prefix, such as ``task-set``.
        value: The structured content that fixes the identifier.

    Returns:
        A valid artifact ID containing the first 20 hexadecimal digest characters.

    Raises:
        ValueError: If ``prefix`` cannot begin an artifact ID.
    """
    candidate = f"{prefix}-{sha256_json(value)[:20]}"
    try:
        return validate_artifact_id(candidate)
    except ValueError as exc:
        raise ValueError("stable ID prefixes must be lowercase artifact ID components") from exc


def validate_artifact_id(value: str) -> str:
    """Validate a stable identifier used as one local artifact path component.

    Args:
        value: Proposed stable identifier.

    Returns:
        The validated identifier.

    Raises:
        ValueError: The identifier is not a canonical lower-case artifact ID.
    """
    if not re.fullmatch(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$", value):
        raise ValueError("stable IDs must be lowercase components separated by '.', '_', or '-'")
    return value


def validate_artifact_file_path(value: str) -> PurePosixPath:
    """Validate one portable relative data-file path inside an artifact directory.

    Args:
        value: Proposed POSIX relative path stored in an immutable manifest.

    Returns:
        A normalized relative POSIX path safe to join below an artifact directory.

    Raises:
        ValueError: The path is empty, absolute, or contains a non-portable component.
    """
    path = PurePosixPath(value)
    components = value.split("/")
    if (
        not value
        or path.is_absolute()
        or "\x00" in value
        or "\\" in value
        or any(component in {"", ".", ".."} or ":" in component for component in components)
    ):
        raise ValueError(
            "artifact file paths must be non-empty relative POSIX paths with ordinary components"
        )
    return path


def assert_secret_free(value: BaseModel | JsonValue) -> None:
    """Reject secret values and credential references at immutable artifact boundaries.

    Args:
        value: The structured content about to enter an immutable artifact.

    Raises:
        SecretBoundaryError: A credential-like key or well-known secret value was found.
    """
    payload: JsonValue
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", by_alias=True, exclude_none=False)
    else:
        payload = value
    _assert_json_value_secret_free(payload, path="$")


def assert_text_secret_free(value: str) -> None:
    """Reject known credential references and secret-like values in persisted text.

    Args:
        value: UTF-8 text about to enter an immutable artifact.

    Raises:
        SecretBoundaryError: The text includes a credential reference or secret-like value.
    """
    if _SECRET_REFERENCE_PATTERN.search(value) or _SECRET_ENVIRONMENT_NAME_PATTERN.search(value):
        raise SecretBoundaryError("immutable artifacts cannot contain credential references")
    if any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS):
        raise SecretBoundaryError("immutable artifacts cannot contain secret-like values")


def redact_secret_text(value: str) -> tuple[str, int]:
    """Replace secret-like values and credential environment names with a fixed placeholder.

    Args:
        value: Text that may contain generated credential-shaped substrings.

    Returns:
        The redacted text and the number of substrings replaced. The deterministic
        placeholder never trips the immutable-artifact secret boundary itself.
    """
    redacted = value
    total = 0
    for pattern in _SECRET_REDACTION_PATTERNS:
        redacted, count = pattern.subn(SECRET_REDACTION_PLACEHOLDER, redacted)
        total += count
    return redacted, total


def redact_secret_json(value: JsonValue) -> tuple[JsonValue, int]:
    """Recursively replace secret-like substrings in JSON-safe content.

    Args:
        value: JSON-safe content about to enter an immutable artifact.

    Returns:
        A structurally equivalent value whose string leaves are redacted with
        :func:`redact_secret_text`, and the total number of replaced substrings.
    """
    if isinstance(value, dict):
        redacted_object: JsonObject = {}
        object_total = 0
        for key, item in value.items():
            redacted_item, count = redact_secret_json(item)
            redacted_object[key] = redacted_item
            object_total += count
        return redacted_object, object_total
    if isinstance(value, list):
        redacted_items: list[JsonValue] = []
        list_total = 0
        for item in value:
            redacted_item, count = redact_secret_json(item)
            redacted_items.append(redacted_item)
            list_total += count
        return redacted_items, list_total
    if isinstance(value, str):
        return redact_secret_text(value)
    return value, 0


def _assert_json_value_secret_free(value: JsonValue, *, path: str) -> None:
    """Recursively apply the immutable-artifact secret boundary."""
    if isinstance(value, dict):
        for key, nested_value in value.items():
            normalized_key = key.lower().replace("-", "_")
            nested_path = f"{path}.{key}"
            if normalized_key in _SECRET_FIELD_NAMES:
                raise SecretBoundaryError(f"immutable artifacts cannot contain {nested_path}")
            _assert_json_value_secret_free(nested_value, path=nested_path)
        return
    if isinstance(value, list):
        for index, nested_value in enumerate(value):
            _assert_json_value_secret_free(nested_value, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        if any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS):
            raise SecretBoundaryError(
                f"immutable artifacts cannot contain a secret-like value at {path}"
            )
        if _SECRET_ENVIRONMENT_NAME_PATTERN.search(value):
            raise SecretBoundaryError(
                f"immutable artifacts cannot contain a credential environment name at {path}"
            )
