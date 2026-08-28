"""Typed local `.exp/models.toml` catalog loading without credential values."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlsplit, urlunsplit

import tomli_w
from pydantic import AwareDatetime, Field, field_validator, model_validator

from exp.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    ContractModel,
    JsonObject,
    SecretBoundaryError,
    Sha256,
    assert_secret_free,
    sha256_json,
    validate_artifact_id,
)
from exp.common.core.files import write_text_atomic
from exp.common.models.model import (
    BillingSource,
    ModelCapabilities,
    ModelSnapshot,
    ReasoningEffort,
)

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_AZURE_API_VERSION = re.compile(r"^(?:v1|\d{4}-\d{2}-\d{2}(?:-preview)?)$")
_AWS_REGION_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_VERTEX_HOST = re.compile(r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?-)?aiplatform\.googleapis\.com")
_FIXED_ORIGIN_PROVIDERS = frozenset({"anthropic", "gemini", "openai", "openrouter", "tinker"})
_EXPLICIT_CAPABILITY_PROVIDERS = frozenset({"azure", "bedrock", "openai-compatible", "vertex"})


def _exclude_absent(value: object) -> bool:
    """Keep new optional connection metadata out of legacy canonical payloads."""
    return value is None


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


def _normalize_connection_base_url(connection: ConnectionConfig) -> str | None:
    """Normalize one endpoint while preserving provider-surface equivalence."""
    if connection.base_url is None:
        return None
    normalized = _normalize_base_url(connection.base_url)
    if (
        connection.provider == "azure"
        and connection.azure_api_surface == "model_inference"
        and normalized.lower().endswith("/models")
    ):
        return normalized[:-7].rstrip("/")
    return normalized


class ModelCatalogError(ValueError):
    """A local model catalog was malformed or named a credential value."""


class ConnectionConfig(ContractModel):
    """Local provider connection metadata, with an optional credential environment name only."""

    provider: str = Field(min_length=1, max_length=128)
    base_url: str | None = Field(default=None, max_length=2_048)
    api_key_env: str | None = Field(default=None, max_length=256)
    api_version: str | None = Field(default=None, max_length=64)
    azure_api_surface: Literal["openai_deployments", "model_inference"] | None = None
    region: str | None = Field(default=None, max_length=64)
    aws_access_key_id_env: str | None = Field(
        default=None,
        max_length=256,
        exclude_if=_exclude_absent,
    )
    bedrock_auth_mode: Literal["access_key_pair", "api_key"] | None = Field(
        default=None,
        exclude_if=_exclude_absent,
    )

    @field_validator("api_key_env", "aws_access_key_id_env")
    @classmethod
    def _require_environment_variable_name(cls, value: str | None) -> str | None:
        if value is not None and not _ENVIRONMENT_NAME.fullmatch(value):
            raise ValueError("credential environment fields must name environment variables")
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
        if self.provider != "azure" and self.azure_api_surface is not None:
            raise ValueError("azure_api_surface is only accepted for provider='azure'")
        if self.provider != "bedrock" and (
            self.aws_access_key_id_env is not None or self.bedrock_auth_mode is not None
        ):
            raise ValueError(
                "aws_access_key_id_env and bedrock_auth_mode are only accepted for "
                "provider='bedrock'"
            )
        if self.provider in _FIXED_ORIGIN_PROVIDERS and self.base_url is not None:
            raise ValueError(
                f"native provider {self.provider!r} uses its built-in official endpoint; "
                "use provider='openai-compatible' for a trusted custom endpoint"
            )
        if self.provider == "azure":
            if self.base_url is None:
                raise ValueError("azure requires an explicit resource endpoint in base_url")
            if self.api_key_env is None:
                raise ValueError("azure requires api_key_env")
            if self.api_version is None:
                raise ValueError(
                    "azure requires an explicit api_version such as 'v1' or a dated Azure "
                    "OpenAI version"
                )
            if not _AZURE_API_VERSION.fullmatch(self.api_version):
                raise ValueError(
                    "azure api_version must be 'v1' or a dated Azure OpenAI version such as "
                    "2024-10-21"
                )
            if self.azure_api_surface == "model_inference" and self.api_version == "v1":
                raise ValueError(
                    "azure model_inference requires a dated api_version for the mandatory "
                    "api-version query parameter"
                )
            if self.region is not None:
                raise ValueError("region is only accepted for provider='bedrock'")
        elif self.provider == "bedrock":
            if self.bedrock_auth_mode == "api_key":
                if self.api_key_env is None or self.aws_access_key_id_env is not None:
                    raise ValueError(
                        "bedrock api_key auth requires api_key_env and forbids "
                        "aws_access_key_id_env"
                    )
            elif self.bedrock_auth_mode == "access_key_pair":
                if self.api_key_env is None or self.aws_access_key_id_env is None:
                    raise ValueError(
                        "bedrock access_key_pair auth requires both credential environment names"
                    )
            elif (self.api_key_env is None) != (self.aws_access_key_id_env is None):
                raise ValueError(
                    "bedrock explicit access-key auth requires both api_key_env naming the "
                    "secret access key and aws_access_key_id_env naming the access key id"
                )
            if self.base_url is not None:
                raise ValueError("bedrock does not accept base_url")
            if self.api_version is not None:
                raise ValueError("api_version is only accepted for provider='azure'")
            if self.region is not None and not _AWS_REGION_NAME.fullmatch(self.region):
                raise ValueError("bedrock region must be an AWS region name")
        elif self.provider == "vertex":
            if self.base_url is None:
                raise ValueError(
                    "vertex requires base_url naming the project-and-location root, such as "
                    "https://us-central1-aiplatform.googleapis.com/v1/projects/PROJECT/"
                    "locations/us-central1"
                )
            # The runtime attaches a cloud-platform OAuth token to every request, so the
            # endpoint host is pinned to Vertex AI service hosts and never operator-chosen.
            vertex_parts = urlsplit(self.base_url)
            vertex_host = (vertex_parts.hostname or "").lower()
            if vertex_parts.scheme != "https" or not _VERTEX_HOST.fullmatch(vertex_host):
                raise ValueError(
                    "vertex base_url must use an HTTPS Vertex AI host such as "
                    "https://us-central1-aiplatform.googleapis.com; OAuth tokens are never "
                    "sent to other hosts"
                )
            if self.api_key_env is None:
                raise ValueError(
                    "vertex requires api_key_env naming the environment variable that holds "
                    "the service-account JSON credential"
                )
            if self.api_version is not None:
                raise ValueError("api_version is only accepted for provider='azure'")
            if self.region is not None:
                raise ValueError(
                    "region is only accepted for provider='bedrock'; the Vertex location "
                    "lives inside base_url"
                )
        else:
            if self.api_version is not None:
                raise ValueError("api_version is only accepted for provider='azure'")
            if self.region is not None:
                raise ValueError("region is only accepted for provider='bedrock'")
        try:
            assert_secret_free(
                {
                    "provider": self.provider,
                    "base_url": self.base_url,
                    "api_version": self.api_version,
                    "azure_api_surface": self.azure_api_surface,
                    "region": self.region,
                    "bedrock_auth_mode": self.bedrock_auth_mode,
                }
            )
        except SecretBoundaryError as exc:
            raise ValueError("connection metadata must not contain credential values") from exc
        return self

    def identity_sha256(self) -> Sha256:
        """Return a deterministic digest of the secret-free provider endpoint identity.

        Returns:
            A SHA-256 digest over the provider, normalized endpoint, and any Azure API version or
            Bedrock region. Credential values and credential-environment metadata are excluded.
        """
        identity: JsonObject = {
            "provider": self.provider,
            "base_url": _normalize_connection_base_url(self),
        }
        if self.api_version is not None:
            identity["api_version"] = self.api_version
        if self.provider == "azure" and self.azure_api_surface == "model_inference":
            # Keep classic Azure revisions byte-compatible with the identity
            # contract that predates this discriminator. Only the genuinely
            # different Foundry surface needs a new credential binding.
            identity["azure_api_surface"] = "model_inference"
        if self.region is not None:
            identity["region"] = self.region
        effective_bedrock_auth_mode = self.bedrock_auth_mode
        if (
            self.provider == "bedrock"
            and effective_bedrock_auth_mode is None
            and self.api_key_env is not None
            and self.aws_access_key_id_env is not None
        ):
            effective_bedrock_auth_mode = "access_key_pair"
        if effective_bedrock_auth_mode is not None:
            identity["bedrock_auth_mode"] = effective_bedrock_auth_mode
        return sha256_json(identity)

    def canonicalized(self) -> ConnectionConfig:
        """Return the canonical persisted shape for Bedrock access-key pairs."""
        if (
            self.provider == "bedrock"
            and self.bedrock_auth_mode is None
            and self.api_key_env is not None
            and self.aws_access_key_id_env is not None
        ):
            return self.model_copy(update={"bedrock_auth_mode": "access_key_pair"})
        return self


class SFTModelProvenance(ContractModel):
    """Immutable W12, W13, and base-model bindings for one registered SFT sampling handle."""

    source_dataset: ArtifactInput
    optimization_config: ArtifactInput
    training_spec_sha256: Sha256
    run_id: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=128)
    model_sha256: Sha256
    result_id: str = Field(min_length=1, max_length=128)
    result_sha256: Sha256
    base_model: ModelSnapshot
    connection_config_sha256: Sha256
    sampling_handle_sha256: Sha256


class GatewayDeploymentCapabilities(ContractModel):
    """Gateway protocol capabilities declared for one provider deployment.

    These fields are intentionally separate from ``ModelCapabilities``. The latter participates
    in frozen optimizer and runtime identities, while this declaration can evolve with the
    gateway protocol without invalidating existing router artifacts.
    """

    supports_developer_messages: bool = False
    supports_streaming: bool = False
    supports_streaming_tool_arguments: bool = False
    supports_strict_tools: bool = False
    supports_parallel_tool_calls: bool = False
    supports_structured_text: bool = False
    supports_stop_sequences: bool = False
    maximum_stop_sequences: int | None = Field(default=None, ge=1)
    """Largest stop-sequence count this route accepts, when the provider caps it.

    ``None`` leaves the count unbounded (only ``supports_stop_sequences`` gates the
    field). A concrete value lets admission reject an over-limit list locally with a
    named parameter error instead of forwarding it and surfacing the provider's
    opaque 4xx (e.g. Gemini caps ``stopSequences`` at 5)."""
    supported_reasoning_efforts: tuple[ReasoningEffort, ...] = ()
    """Exact caller values this deployment can preserve without normalization.

    An empty tuple means the gateway should use its maintained provider-family
    contract. OpenRouter and other catalog-driven providers declare the exact
    ordered set here because their supported values vary by model.
    """
    reasoning_default_effort: ReasoningEffort | None = None
    """Explicit provider default used only when the wire requires this field."""
    reasoning_effort_required: bool = False
    """Whether this deployment requires an explicit reasoning effort on its wire."""
    reports_refusals: bool = False
    reports_cached_input_tokens: bool = False
    reports_reasoning_tokens: bool = False

    @property
    def declares_reasoning_contract(self) -> bool:
        """Whether this metadata overrides provider-family reasoning behavior."""
        return bool(
            self.supported_reasoning_efforts
            or self.reasoning_default_effort is not None
            or self.reasoning_effort_required
        )

    @model_validator(mode="after")
    def _require_valid_reasoning_contract(self) -> GatewayDeploymentCapabilities:
        """Reject ambiguous or non-canonical reasoning declarations."""
        order = ("none", "minimal", "low", "medium", "high", "xhigh", "ultra", "max")
        indexes = tuple(order.index(effort) for effort in self.supported_reasoning_efforts)
        if len(set(self.supported_reasoning_efforts)) != len(self.supported_reasoning_efforts):
            raise ValueError("supported_reasoning_efforts cannot repeat values")
        if indexes != tuple(sorted(indexes)):
            raise ValueError("supported_reasoning_efforts must use canonical order")
        if (
            self.reasoning_default_effort is not None
            and self.reasoning_default_effort not in self.supported_reasoning_efforts
        ):
            raise ValueError(
                "reasoning_default_effort must be one of the supported reasoning efforts"
            )
        if self.reasoning_effort_required and not self.supported_reasoning_efforts:
            raise ValueError(
                "reasoning_effort_required needs at least one supported reasoning effort"
            )
        if self.reasoning_effort_required and self.reasoning_default_effort is None:
            raise ValueError("reasoning_effort_required needs reasoning_default_effort")
        return self


class GatewayTokenPrices(ContractModel):
    """Integer gateway attribution rates for one provider deployment.

    Values are micro-USD per million provider-reported tokens. ``None`` means the rate is unknown;
    it must never be interpreted as zero. Existing optimizer float pricing remains unchanged.
    """

    input_micro_usd_per_million_tokens: int | None = Field(default=None, ge=0)
    cached_input_micro_usd_per_million_tokens: int | None = Field(default=None, ge=0)
    output_micro_usd_per_million_tokens: int | None = Field(default=None, ge=0)
    reasoning_micro_usd_per_million_tokens: int | None = Field(default=None, ge=0)


class GatewayDeploymentMetadata(ContractModel):
    """Optional gateway-only metadata authored beside one existing model record."""

    exact_model_id: ArtifactId | None = None
    capabilities: GatewayDeploymentCapabilities = Field(
        default_factory=GatewayDeploymentCapabilities
    )
    prices: GatewayTokenPrices = Field(default_factory=GatewayTokenPrices)
    pricing_source: str | None = Field(default=None, min_length=1, max_length=512)
    pricing_effective_at: AwareDatetime | None = None


class GatewayEquivalenceCertification(ContractModel):
    """Operator-authored evidence that deployments serve one exact model revision."""

    authority: Literal["operator"] = "operator"
    certification_id: ArtifactId
    provenance: str = Field(min_length=1, max_length=2_048)
    evidence_sha256: Sha256
    certified_at: AwareDatetime

    @model_validator(mode="after")
    def _require_safe_provenance(self) -> GatewayEquivalenceCertification:
        """Reject credential-like or control-bearing equivalence provenance."""
        try:
            assert_secret_free(self.model_dump(mode="json"))
        except SecretBoundaryError as exc:
            raise ValueError("equivalence provenance must be secret-free") from exc
        if any(ord(character) < 32 for character in self.provenance):
            raise ValueError("equivalence provenance must be display-safe")
        return self


class GatewayPoolRecord(ContractModel):
    """Authored ordered deployments explicitly certified as one exact model."""

    exact_model_id: ArtifactId
    deployment_aliases: tuple[ArtifactId, ...] = Field(min_length=2)
    equivalence: GatewayEquivalenceCertification

    @model_validator(mode="after")
    def _require_unique_deployments(self) -> GatewayPoolRecord:
        """Reject repeated deployment aliases inside one equivalence pool."""
        if len(set(self.deployment_aliases)) != len(self.deployment_aliases):
            raise ValueError("gateway pool deployment aliases must not repeat")
        return self


class ModelRecord(ContractModel):
    """A stable local alias, exact capability snapshot, and provider-side model name.

    An omitted capability declaration means the catalog cannot prove any optional protocol
    feature or token limit. Unknown declarations stay permissive: capability preflight blocks
    only an explicit declaration that rules a requirement out, so an undeclared model remains
    usable and the provider reports any real protocol gap.

    ``served_model_id`` accepts an alternate identifier the provider echoes in responses when it
    differs from the requested ``model``, for example a vLLM endpoint that publishes an alias in
    ``/models`` but reports its canonical served name in every completion.
    """

    connection: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=2_048)
    revision: str | None = Field(default=None, max_length=256)
    served_model_id: str | None = Field(default=None, min_length=1, max_length=2_048)
    billing_source: BillingSource
    capabilities: ModelCapabilities | None = None
    gateway: GatewayDeploymentMetadata | None = None
    sft_provenance: SFTModelProvenance | None = None

    @model_validator(mode="after")
    def _require_secret_free_model_identity(self) -> ModelRecord:
        """Reject contradictory reasoning metadata and credential-bearing identity fields."""
        if (
            self.gateway is not None
            and self.gateway.capabilities.declares_reasoning_contract
            and (self.capabilities is None or not self.capabilities.supports_reasoning)
        ):
            raise ValueError(
                "gateway reasoning metadata requires model capabilities.supports_reasoning=true"
            )
        if (
            self.sft_provenance is not None
            and self.sft_provenance.sampling_handle_sha256
            != sha256_json({"sampling_handle": self.model})
        ):
            raise ValueError("SFT provenance does not bind this model sampling handle")
        try:
            assert_secret_free(
                {
                    "connection": self.connection,
                    "model": self.model,
                    "revision": self.revision,
                    "served_model_id": self.served_model_id,
                    "billing_source": self.billing_source.value,
                    "capabilities": (
                        self.capabilities.model_dump(mode="json")
                        if self.capabilities is not None
                        else None
                    ),
                    "gateway": (
                        self.gateway.model_dump(mode="json") if self.gateway is not None else None
                    ),
                    "sft_provenance": (
                        self.sft_provenance.model_dump(mode="json")
                        if self.sft_provenance is not None
                        else None
                    ),
                }
            )
        except SecretBoundaryError as exc:
            raise ValueError("model identity must not contain credential values") from exc
        return self


class ModelRoles(ContractModel):
    """Project roles that select stable aliases without revealing credentials.

    Each completion role may carry its own reasoning-effort choice, so one alias can use
    different efforts as world model, judge, or router candidate. An absent role effort means
    the alias's catalog capability pin applies unchanged.
    """

    candidates: tuple[str, ...] = ()
    incumbent: str | None = None
    world_model: str | None = None
    judge: str | None = None
    rubric_proposer: str | None = None
    embedder: str | None = None
    teacher: str | None = None
    world_model_reasoning_effort: ReasoningEffort | None = None
    judge_reasoning_effort: ReasoningEffort | None = None
    candidate_reasoning_efforts: dict[str, ReasoningEffort] = Field(default_factory=dict)

    @field_validator("candidates")
    @classmethod
    def _require_unique_candidates(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("candidate aliases must not repeat")
        return value

    @model_validator(mode="after")
    def _require_role_bound_reasoning_efforts(self) -> ModelRoles:
        """Require every role-specific effort to name a currently assigned role alias.

        Returns:
            The validated roles.

        Raises:
            ValueError: An effort is declared for an unassigned role or unknown candidate.
        """
        if self.world_model_reasoning_effort is not None and self.world_model is None:
            raise ValueError("world_model_reasoning_effort requires an assigned world_model")
        if self.judge_reasoning_effort is not None and self.judge is None:
            raise ValueError("judge_reasoning_effort requires an assigned judge")
        unknown = sorted(set(self.candidate_reasoning_efforts).difference(self.candidates))
        if unknown:
            raise ValueError(
                "candidate_reasoning_efforts name unassigned candidates: " + ", ".join(unknown)
            )
        return self


class ModelCatalog(ContractModel):
    """The local model aliases, connection metadata, and project role assignments."""

    schema_version: Literal[2] = 2
    connections: dict[str, ConnectionConfig]
    models: dict[str, ModelRecord]
    gateway_pools: dict[str, GatewayPoolRecord] = Field(default_factory=dict)
    roles: ModelRoles = Field(default_factory=ModelRoles)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _require_integer_schema_version(cls, value: object) -> object:
        """Reject boolean and floating-point lookalikes at the version boundary."""
        if type(value) is not int:
            raise ValueError("model catalog schema_version must be an integer")
        return value

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
        for alias in value:
            validate_artifact_id(alias)
        return value

    @field_validator("gateway_pools")
    @classmethod
    def _require_valid_gateway_pool_names(
        cls, value: dict[str, GatewayPoolRecord]
    ) -> dict[str, GatewayPoolRecord]:
        """Validate authored pool identifiers before cross-reference checks."""
        for pool_id in value:
            validate_artifact_id(pool_id)
        return value

    @model_validator(mode="after")
    def _require_referenced_connections_and_roles(self) -> ModelCatalog:
        for alias, record in self.models.items():
            if record.connection not in self.connections:
                raise ValueError(
                    f"model alias {alias!r} names unknown connection {record.connection!r}"
                )
            connection = self.connections[record.connection]
            if (
                connection.provider in _EXPLICIT_CAPABILITY_PROVIDERS
                and record.capabilities is None
            ):
                raise ValueError(
                    f"{connection.provider} model alias {alias!r} needs an explicit capabilities "
                    "declaration because provider names do not imply protocol support or prices"
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
        pooled_aliases: set[str] = set()
        for pool_id, pool in self.gateway_pools.items():
            for alias in pool.deployment_aliases:
                if alias in pooled_aliases:
                    raise ValueError(
                        f"gateway deployment alias {alias!r} appears in more than one pool"
                    )
                pooled_aliases.add(alias)
                record = self.models.get(alias)
                if record is None:
                    raise ValueError(
                        f"gateway pool {pool_id!r} names unknown model alias {alias!r}"
                    )
                connection = self.connections[record.connection]
                if connection.provider == "tinker" or record.sft_provenance is not None:
                    raise ValueError(
                        f"gateway pool {pool_id!r} cannot contain training handle {alias!r}"
                    )
                if record.gateway is None or record.gateway.exact_model_id != pool.exact_model_id:
                    raise ValueError(
                        f"gateway pool {pool_id!r} requires alias {alias!r} to declare exact "
                        "model identity"
                    )
        return self


def load_model_catalog(path: Path) -> ModelCatalog:
    """Load and validate `.exp/models.toml` without reading its environment variables.

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
        return ModelCatalog.model_validate(_migrate_legacy_model_catalog(raw_catalog))
    except ValueError as exc:
        raise ModelCatalogError(f"model catalog is invalid: {exc}") from exc


def _migrate_legacy_model_catalog(raw_catalog: JsonObject) -> JsonObject:
    """Upgrade only schema-v1 local catalogs with conservative customer-owned billing.

    Args:
        raw_catalog: Parsed secret-free TOML payload.

    Returns:
        A schema-v2 payload. Current schema records are returned unchanged so a missing
        ``billing_source`` remains a validation error.
    """
    raw_version = raw_catalog.get("schema_version", 1)
    if type(raw_version) is not int or raw_version != 1:
        return raw_catalog
    payload = cast(JsonObject, dict(raw_catalog))
    models = raw_catalog.get("models")
    if isinstance(models, dict):
        migrated_models: JsonObject = {}
        for alias, value in models.items():
            if isinstance(value, dict):
                record = cast(JsonObject, dict(value))
                if "billing_source" in record:
                    raise ValueError(
                        "schema-v1 model record must not declare current billing_source"
                    )
                record["billing_source"] = BillingSource.CUSTOMER_MANAGED.value
                provenance = record.get("sft_provenance")
                if isinstance(provenance, dict):
                    migrated_provenance = cast(JsonObject, dict(provenance))
                    base_model = provenance.get("base_model")
                    if isinstance(base_model, dict):
                        migrated_base = cast(JsonObject, dict(base_model))
                        if "billing_source" in migrated_base:
                            raise ValueError(
                                "schema-v1 SFT base model must not declare current billing_source"
                            )
                        migrated_base["billing_source"] = BillingSource.CUSTOMER_MANAGED.value
                        migrated_provenance["base_model"] = migrated_base
                    record["sft_provenance"] = migrated_provenance
                migrated_models[str(alias)] = record
            else:
                migrated_models[str(alias)] = value
        payload["models"] = migrated_models
    payload["schema_version"] = 2
    return payload


def write_model_catalog(path: Path, catalog: ModelCatalog) -> None:
    """Atomically write validated model metadata and environment-variable names only.

    Args:
        path: Destination `.exp/models.toml` path.
        catalog: Typed catalog containing no credential values.
    """
    payload = tomli_w.dumps(catalog.model_dump(mode="json", exclude_none=True))
    write_text_atomic(path, payload)
