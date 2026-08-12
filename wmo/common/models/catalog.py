"""Typed local `.wmo/models.toml` catalog loading without credential values."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import tomli_w
from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import (
    ContractModel,
    JsonObject,
    SecretBoundaryError,
    Sha256,
    assert_secret_free,
    sha256_json,
    validate_artifact_id,
)
from wmo.common.core.files import write_text_atomic
from wmo.common.models.model import ModelCapabilities

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FIXED_ORIGIN_PROVIDERS = frozenset({"anthropic", "gemini", "openai", "openrouter", "tinker"})


def _normalize_base_url(value: str) -> str:
    """Return the stable endpoint spelling used for connection identity."""
    parsed = urlsplit(value)
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError("base_url must include a hostname")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("base_url must use a valid port") from exc
    scheme = parsed.scheme.lower()
    host = hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    default_port = 443 if scheme == "https" else 80
    netloc = host if port in {None, default_port} else f"{host}:{port}"
    return urlunsplit((scheme, netloc, parsed.path.rstrip("/"), "", ""))


class ModelCatalogError(ValueError):
    """A local model catalog was malformed or named a credential value."""


class ConnectionConfig(ContractModel):
    """Local provider connection metadata, with an optional credential environment name only."""

    provider: str = Field(min_length=1, max_length=128)
    base_url: str | None = Field(default=None, max_length=2_048)
    api_key_env: str | None = Field(default=None, max_length=256)

    @field_validator("api_key_env")
    @classmethod
    def _require_environment_variable_name(cls, value: str | None) -> str | None:
        if value is not None and not _ENVIRONMENT_NAME.fullmatch(value):
            raise ValueError("api_key_env must name one environment variable")
        return value

    @field_validator("base_url")
    @classmethod
    def _reject_embedded_credentials(cls, value: str | None) -> str | None:
        if value is None:
            return value
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not embed credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not include query parameters or fragments")
        _normalize_base_url(value)
        return value

    @model_validator(mode="after")
    def _require_secret_free_connection_metadata(self) -> ConnectionConfig:
        if self.provider in _FIXED_ORIGIN_PROVIDERS and self.base_url is not None:
            raise ValueError(
                f"native provider {self.provider!r} uses its built-in official endpoint; "
                "use provider='openai-compatible' for a trusted custom endpoint"
            )
        try:
            assert_secret_free({"provider": self.provider, "base_url": self.base_url})
        except SecretBoundaryError as exc:
            raise ValueError("connection metadata must not contain credential values") from exc
        return self

    def identity_sha256(self) -> Sha256:
        """Return a deterministic digest of the secret-free provider endpoint identity.

        Returns:
            A SHA-256 digest over the provider and normalized base URL. Credential values and
            credential-environment metadata are deliberately excluded.
        """
        identity: JsonObject = {
            "provider": self.provider,
            "base_url": None if self.base_url is None else _normalize_base_url(self.base_url),
        }
        return sha256_json(identity)


class ModelRecord(ContractModel):
    """A stable local alias, exact capability snapshot, and provider-side model name.

    An omitted capability declaration means the catalog cannot prove any optional protocol
    feature or token limit. Callers may still resolve the alias for an unconstrained completion,
    but capability preflight fails closed instead of inferring support from a provider name.
    """

    connection: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=512)
    revision: str | None = Field(default=None, max_length=256)
    capabilities: ModelCapabilities | None = None

    @model_validator(mode="after")
    def _require_secret_free_model_identity(self) -> ModelRecord:
        try:
            assert_secret_free(
                {
                    "connection": self.connection,
                    "model": self.model,
                    "revision": self.revision,
                    "capabilities": (
                        self.capabilities.model_dump(mode="json")
                        if self.capabilities is not None
                        else None
                    ),
                }
            )
        except SecretBoundaryError as exc:
            raise ValueError("model identity must not contain credential values") from exc
        return self


class ModelRoles(ContractModel):
    """Project roles that select stable aliases without revealing credentials."""

    candidates: tuple[str, ...] = ()
    incumbent: str | None = None
    world_model: str | None = None
    judge: str | None = None
    rubric_proposer: str | None = None
    embedder: str | None = None
    teacher: str | None = None

    @field_validator("candidates")
    @classmethod
    def _require_unique_candidates(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("candidate aliases must not repeat")
        return value


class ModelCatalog(ContractModel):
    """The local model aliases, connection metadata, and project role assignments."""

    schema_version: int = Field(default=1, ge=1)
    connections: dict[str, ConnectionConfig]
    models: dict[str, ModelRecord]
    roles: ModelRoles = Field(default_factory=ModelRoles)

    @field_validator("connections")
    @classmethod
    def _require_valid_connection_names(
        cls, value: dict[str, ConnectionConfig]
    ) -> dict[str, ConnectionConfig]:
        if not value:
            raise ValueError("models.toml needs at least one connection")
        for connection_name in value:
            validate_artifact_id(connection_name)
        return value

    @field_validator("models")
    @classmethod
    def _require_valid_model_aliases(cls, value: dict[str, ModelRecord]) -> dict[str, ModelRecord]:
        if not value:
            raise ValueError("models.toml needs at least one model alias")
        for alias in value:
            validate_artifact_id(alias)
        return value

    @model_validator(mode="after")
    def _require_referenced_connections_and_roles(self) -> ModelCatalog:
        for alias, record in self.models.items():
            if record.connection not in self.connections:
                raise ValueError(
                    f"model alias {alias!r} names unknown connection {record.connection!r}"
                )
            connection = self.connections[record.connection]
            if connection.provider == "openai-compatible" and record.capabilities is None:
                raise ValueError(
                    f"OpenAI-compatible model alias {alias!r} needs an explicit capabilities "
                    "declaration because its endpoint cannot be discovered safely"
                )
        assigned_aliases = self.roles.candidates + tuple(
            alias
            for alias in (
                self.roles.incumbent,
                self.roles.world_model,
                self.roles.judge,
                self.roles.rubric_proposer,
                self.roles.embedder,
                self.roles.teacher,
            )
            if alias is not None
        )
        unknown_aliases = sorted(set(assigned_aliases).difference(self.models))
        if unknown_aliases:
            raise ValueError(f"roles name unknown model aliases: {', '.join(unknown_aliases)}")
        if self.roles.incumbent is not None and self.roles.incumbent not in self.roles.candidates:
            raise ValueError("incumbent must also appear in roles.candidates")
        return self


def load_model_catalog(path: Path) -> ModelCatalog:
    """Load and validate `.wmo/models.toml` without reading its environment variables.

    Args:
        path: Path to the local model catalog.

    Returns:
        Typed aliases, connection metadata, and role assignments.

    Raises:
        ModelCatalogError: The catalog is missing, malformed, or violates the no-secret contract.
    """
    try:
        with path.open("rb") as handle:
            raw_catalog = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ModelCatalogError(f"model catalog does not exist: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ModelCatalogError(f"model catalog is invalid TOML: {path}") from exc
    try:
        return ModelCatalog.model_validate(raw_catalog)
    except ValueError as exc:
        raise ModelCatalogError(f"model catalog is invalid: {exc}") from exc


def write_model_catalog(path: Path, catalog: ModelCatalog) -> None:
    """Atomically write validated model metadata and environment-variable names only.

    Args:
        path: Destination `.wmo/models.toml` path.
        catalog: Typed catalog containing no credential values.
    """
    payload = tomli_w.dumps(catalog.model_dump(mode="json", exclude_none=True))
    write_text_atomic(path, payload)
