"""Crash-safe hard-budget admission and provider-boundary cost ceilings.

The budget database is an append-only, hash-chained audit log plus a transactionally maintained
reservation index. An independent append-only high-water database makes a stale ledger-only
restore fail closed. A paid call must reserve its conservative maximum cost before dispatch. Exact
usage settles the reservation; missing usage or an exception forfeits the full ceiling. Open
reservations continue to consume budget after a process crash, so an orphaned request can never
silently release money for a second call. Both files remain one single-host authority; a storage
snapshot that rolls back both trusted files together is outside this local authority's guarantees.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import stat
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, date, datetime
from decimal import ROUND_CEILING, Decimal
from enum import StrEnum
from pathlib import Path
from time import monotonic as _system_monotonic
from typing import Annotated, Literal, Self, cast
from urllib.parse import urlsplit
from uuid import uuid4

from llm_waterfall import ChatRequest, ChatResponse
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    Message,
    Provider,
    ProviderConfig,
    ProviderKind,
    SingleDispatchProvider,
    ToolCallingProvider,
    VerifyResult,
    verify_via_ping,
)
from wmh.providers.models import resolve_provider_model

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_SOURCE_ID_PATTERN = r"^[a-z0-9][a-z0-9.-]{0,127}$"
_PUBLIC_QUERY_NAME_PATTERN = r"^[A-Za-z$][A-Za-z0-9$_.-]{0,127}$"
_PACKAGE_RESOURCE_PATTERN = r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z0-9][A-Za-z0-9._/-]*$"
_ZERO_DIGEST = "sha256:" + "0" * 64
_SCHEMA_VERSION = 5
_AUTHORITY_SCHEMA_VERSION = 1
_DEFAULT_BUSY_TIMEOUT_MS = 30_000
_NANO_USD_PER_USD = 1_000_000_000
_TOKENS_PER_MILLION = 1_000_000
_SQLITE_INTEGER_MAX = (1 << 63) - 1
_FAILURE_CODE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_SENSITIVE_QUERY_NAME = re.compile(
    r"(?:^|[._-])(?:api[-_]?key|access[-_]?key|subscription[-_]?key|token|secret|"
    r"signature|sig|credential|"
    r"password|authorization|auth|code)(?:$|[._-])",
    re.IGNORECASE,
)
_SENSITIVE_QUERY_VALUE = re.compile(
    r"(?:api[-_ ]?key|access[-_ ]?key|subscription[-_ ]?key|client[-_ ]?secret|"
    r"password|authorization|signature|(?:^|[ (])sig(?:[ )=]|$)|shared[-_ ]?access[-_ ]?"
    r"signature|bearer\s+[A-Za-z0-9])",
    re.IGNORECASE,
)
_KNOWN_NULL_USAGE_DETAIL_PLACEHOLDERS = frozenset(
    {
        "completion_tokens_details",
        "input_tokens_details",
        "output_tokens_details",
        "prompt_tokens_details",
    }
)
_PROVIDER_TARIFF_EVIDENCE_CONTRACTS = {
    "aws_bedrock_public_catalog_v1": (
        "sha256+gzip-bounds;exact-route-source-set;aws-publication-metadata;"
        "bedrock-route-source-paths;product-to-meter-sku-joins;meter-record-fields;"
        "geo-standard-table-cells-0-1-2"
    ),
    "azure_retail_arm_v1": (
        "sha256+gzip-bounds;exact-arm-account-deployment-route-records;"
        "azure-retail-single-page;azure-openai-usd-primary-consumption-tier-zero;"
        "exact-two-token-meters;sku-meter-region-unit-price-effective-date;"
        "deployment-etag-model-version-no-auto-upgrade-and-billing-mode-join;"
        "registered-exact-retail-label-profile"
    ),
}


def provider_tariff_evidence_verifier_digest(profile: str) -> str:
    """Return the local registry digest for one supported tariff evidence contract."""
    try:
        contract = _PROVIDER_TARIFF_EVIDENCE_CONTRACTS[profile]
    except KeyError as exc:
        raise ValueError(f"unknown provider tariff evidence verifier profile: {profile}") from exc
    payload = f"{profile}\0{contract}".encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class BudgetExceededError(RuntimeError):
    """A call cannot start inside the frozen hard or phase budget."""


class BudgetBreachError(RuntimeError):
    """Observed usage exceeded a pre-dispatch ceiling or a prior breach exists."""


class BudgetIntegrityError(RuntimeError):
    """The persisted policy, event chain, or derived reservation state is invalid."""


class UnpricedProviderUsageError(RuntimeError):
    """A provider response exposed usage dimensions absent from the frozen tariff."""


class ReservationStatus(StrEnum):
    """Terminal or open state of one pre-dispatch cost reservation."""

    RESERVED = "reserved"
    SETTLED = "settled"
    FORFEITED = "forfeited"
    BREACHED = "breached"


class BudgetBreachKind(StrEnum):
    """Bounded reason a charged call violated its reservation contract."""

    RESERVATION = "reservation"
    INPUT_TOKEN_CEILING = "input_token_ceiling"
    OUTPUT_TOKEN_CEILING = "output_token_ceiling"


class BudgetScope(BaseModel):
    """Typed attribution attached to every reservation without carrying prompt data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: str = Field(min_length=1)
    category: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    lane: str | None = None
    arm: str | None = None

    @field_validator("phase", "category", "run_id", "lane", "arm")
    @classmethod
    def _strip_nonempty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("budget scope strings cannot be blank")
        return stripped


class TokenPriceCeiling(BaseModel):
    """Frozen upper-bound price per token, represented exactly in nano-USD."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_nano_usd_per_token: int = Field(ge=0, le=_SQLITE_INTEGER_MAX)
    output_nano_usd_per_token: int = Field(ge=0, le=_SQLITE_INTEGER_MAX)

    @model_validator(mode="after")
    def _require_priced_calls(self) -> Self:
        if self.input_nano_usd_per_token == self.output_nano_usd_per_token == 0:
            raise ValueError("at least one token price must be positive")
        return self

    def charge(self, *, input_tokens: int, output_tokens: int) -> int:
        """Price nonnegative token counts without floating-point rounding."""
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts cannot be negative")
        if input_tokens > _SQLITE_INTEGER_MAX or output_tokens > _SQLITE_INTEGER_MAX:
            raise OverflowError("token counts must fit a SQLite integer")
        charged = (
            input_tokens * self.input_nano_usd_per_token
            + output_tokens * self.output_nano_usd_per_token
        )
        if charged > _SQLITE_INTEGER_MAX:
            raise OverflowError("priced token ceiling does not fit a SQLite integer")
        return charged

    @classmethod
    def from_usd_per_million(
        cls,
        *,
        input_usd: Decimal | str | int,
        output_usd: Decimal | str | int,
    ) -> TokenPriceCeiling:
        """Convert public USD-per-million prices into conservative integer ceilings."""
        return cls(
            input_nano_usd_per_token=_nano_usd_per_token(input_usd),
            output_nano_usd_per_token=_nano_usd_per_token(output_usd),
        )


class ProviderTariffPublicQueryParameter(BaseModel):
    """One canonical nonsecret parameter used to fetch a pricing or route snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal[1] = 1
    name: str = Field(pattern=_PUBLIC_QUERY_NAME_PATTERN)
    value: str = Field(min_length=1, max_length=4_096)

    @field_validator("name")
    @classmethod
    def _reject_credential_parameter_names(cls, value: str) -> str:
        if _SENSITIVE_QUERY_NAME.search(value):
            raise ValueError("public tariff query cannot use a credential-bearing parameter")
        return value

    @field_validator("value")
    @classmethod
    def _reject_credential_syntax(cls, value: str) -> str:
        if value != value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError("public tariff query values must be exact printable text")
        if _SENSITIVE_QUERY_VALUE.search(value):
            raise ValueError("public tariff query cannot contain credential-bearing syntax")
        return value


class ProviderTariffRetainedArtifact(BaseModel):
    """Content-addressed retained bytes from which a source snapshot can be recovered."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal[1] = 1
    storage_kind: Literal["package_resource", "https"]
    locator: str = Field(min_length=1, max_length=2_048)
    artifact_digest: str = Field(pattern=_DIGEST_PATTERN)
    content_encoding: Literal["identity", "gzip"]

    @model_validator(mode="after")
    def _require_safe_locator(self) -> Self:
        if self.locator != self.locator.strip():
            raise ValueError("retained artifact locator cannot have surrounding whitespace")
        if self.storage_kind == "package_resource":
            if re.fullmatch(_PACKAGE_RESOURCE_PATTERN, self.locator) is None:
                raise ValueError("retained package artifact requires a canonical resource locator")
            _, resource_path = self.locator.split(":", 1)
            if ".." in Path(resource_path).parts:
                raise ValueError("retained package artifact cannot traverse parent directories")
        else:
            parsed = urlsplit(self.locator)
            if (
                parsed.scheme != "https"
                or parsed.hostname is None
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "retained HTTPS artifact must not contain credentials, query, or fragment"
                )
        if self.artifact_digest == _ZERO_DIGEST:
            raise ValueError("retained artifact digest cannot be the zero digest")
        return self


class ProviderTariffSourceSnapshot(BaseModel):
    """One content-addressed provider source used to prove a provider tariff."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal[2] = 2
    source_id: str = Field(pattern=_SOURCE_ID_PATTERN)
    role: Literal["rate_catalog", "route_definition", "semantic_mapping"]
    source_locator: str = Field(min_length=1, max_length=2_048)
    source_snapshot_digest: str = Field(pattern=_DIGEST_PATTERN)
    digest_scope: Literal["decoded_http_entity"] = "decoded_http_entity"
    media_type: Literal["application/json", "text/html"] = Field(
        description="Semantic media type of the decoded source entity"
    )
    content_encoding: Literal["identity", "gzip"] = Field(
        description="Content-Encoding of the fetched HTTP entity"
    )
    public_request_query: tuple[ProviderTariffPublicQueryParameter, ...] = Field(
        default=(), max_length=16
    )
    retained_artifact: ProviderTariffRetainedArtifact
    publication_id: str | None = Field(default=None, min_length=1, max_length=256)
    published_on: date | None = None

    @field_validator("source_locator")
    @classmethod
    def _require_canonical_source_locator(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("tariff source locator cannot have surrounding whitespace")
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "tariff source locator must be HTTPS without credentials, query, or fragment"
            )
        return value

    @field_validator("source_snapshot_digest")
    @classmethod
    def _reject_placeholder_snapshot_digest(cls, value: str) -> str:
        if value == _ZERO_DIGEST:
            raise ValueError("tariff source snapshot digest cannot be the zero digest")
        return value

    @field_validator("publication_id")
    @classmethod
    def _require_exact_publication_id(cls, value: str | None) -> str | None:
        if value is not None and (
            value != value.strip() or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("tariff publication identifier must be an exact printable value")
        return value

    @field_validator("public_request_query")
    @classmethod
    def _require_canonical_public_query(
        cls,
        value: tuple[ProviderTariffPublicQueryParameter, ...],
    ) -> tuple[ProviderTariffPublicQueryParameter, ...]:
        coordinates = tuple((parameter.name, parameter.value) for parameter in value)
        if coordinates != tuple(sorted(coordinates)) or len(coordinates) != len(set(coordinates)):
            raise ValueError("public tariff query parameters require canonical ascending order")
        return value


class ProviderTariffVerifiedSource(BaseModel):
    """One retained artifact whose encoded and decoded bytes passed digest verification."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal[1] = 1
    source_id: str = Field(pattern=_SOURCE_ID_PATTERN)
    artifact_digest: str = Field(pattern=_DIGEST_PATTERN)
    source_snapshot_digest: str = Field(pattern=_DIGEST_PATTERN)
    artifact_size_bytes: int = Field(gt=0, le=_SQLITE_INTEGER_MAX)
    decoded_size_bytes: int = Field(gt=0, le=_SQLITE_INTEGER_MAX)


class ProviderTariffValidatedRecord(BaseModel):
    """One exact source coordinate interpreted by a registered semantic verifier."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal[1] = 1
    source_id: str = Field(pattern=_SOURCE_ID_PATTERN)
    source_record_path: str = Field(min_length=1, max_length=2_048)
    purpose: Literal[
        "route_identity",
        "usage_dimension",
        "input_tokens_billing_meter",
        "output_tokens_billing_meter",
    ]

    @field_validator("source_record_path")
    @classmethod
    def _require_absolute_record_path(cls, value: str) -> str:
        if (
            not value.startswith("/")
            or value != value.strip()
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("validated tariff record requires an exact absolute path")
        return value


class ProviderTariffEvidenceReceipt(BaseModel):
    """Deterministic audit result from one locally registered evidence verifier.

    This receipt guards trusted paid-run construction against stale or mismatched evidence. It is
    not a provider signature and does not defend against hostile Python code fabricating policy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal[1] = 1
    verifier_profile: Literal[
        "aws_bedrock_public_catalog_v1",
        "azure_retail_arm_v1",
    ]
    verifier_digest: str = Field(pattern=_DIGEST_PATTERN)
    tariff_claim_digest: str = Field(pattern=_DIGEST_PATTERN)
    verified_sources: tuple[ProviderTariffVerifiedSource, ...] = Field(min_length=1, max_length=8)
    validated_records: tuple[ProviderTariffValidatedRecord, ...] = Field(
        min_length=3,
        max_length=40,
    )

    @model_validator(mode="after")
    def _require_registered_canonical_verification(self) -> Self:
        expected_verifier = provider_tariff_evidence_verifier_digest(self.verifier_profile)
        if self.verifier_digest != expected_verifier:
            raise ValueError("tariff evidence verifier digest differs from the local registry")
        source_ids = tuple(source.source_id for source in self.verified_sources)
        if source_ids != tuple(sorted(source_ids)) or len(source_ids) != len(set(source_ids)):
            raise ValueError("verified tariff sources require unique ascending source IDs")
        record_coordinates = tuple(
            (record.source_id, record.source_record_path, record.purpose)
            for record in self.validated_records
        )
        if record_coordinates != tuple(sorted(record_coordinates)) or len(
            record_coordinates
        ) != len(set(record_coordinates)):
            raise ValueError("validated tariff records require unique canonical coordinates")
        return self

    @property
    def digest(self) -> str:
        """Return a stable digest of this complete verification result."""
        return _digest_json(self.model_dump(mode="json"))


class ProviderTariffSourceBinding(BaseModel):
    """One explicit translation from a retained source record into a tariff claim."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal[1] = 1
    claim: Literal["route_identity", "usage_dimension"]
    source_id: str = Field(pattern=_SOURCE_ID_PATTERN)
    source_record_path: str = Field(min_length=1, max_length=2_048)
    source_value: str = Field(min_length=1, max_length=2_048)
    canonical_value: str = Field(min_length=1, max_length=2_048)
    target_meter_source_id: str | None = Field(default=None, pattern=_SOURCE_ID_PATTERN)
    target_meter_record_path: str | None = Field(default=None, min_length=1, max_length=2_048)

    @field_validator("source_record_path", "target_meter_record_path")
    @classmethod
    def _require_absolute_source_record_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if (
            not value.startswith("/")
            or value != value.strip()
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("tariff source binding requires an absolute record path")
        return value

    @field_validator("source_value", "canonical_value")
    @classmethod
    def _require_exact_printable_value(cls, value: str) -> str:
        if value != value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError("tariff source binding values must be exact printable text")
        return value

    @model_validator(mode="after")
    def _require_complete_meter_target(self) -> Self:
        if (self.target_meter_source_id is None) != (self.target_meter_record_path is None):
            raise ValueError("tariff source binding meter target must be complete")
        if self.claim == "usage_dimension" and self.target_meter_source_id is None:
            raise ValueError("usage-dimension binding requires an exact billing meter target")
        return self


class ProviderTariffBillingMeter(BaseModel):
    """One exact provider billing identifier for a priced usage dimension."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal[3] = 3
    usage_dimension: Literal["input_tokens", "output_tokens"]
    source_id: str = Field(pattern=_SOURCE_ID_PATTERN)
    source_record_path: str = Field(min_length=1, max_length=2_048)
    sku_id: str = Field(min_length=1, max_length=256)
    rate_id: str = Field(min_length=1, max_length=256)
    billing_region: str = Field(min_length=1, max_length=128)
    billing_mode: str = Field(min_length=1, max_length=128)
    effective_on: date | None
    source_price_usd: Decimal = Field(gt=0)
    source_price_unit: Literal["per_1k_tokens", "per_1m_tokens"]

    @field_validator("source_record_path")
    @classmethod
    def _require_absolute_source_record_path(cls, value: str) -> str:
        if (
            not value.startswith("/")
            or value != value.strip()
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("tariff source record path must be an exact absolute printable path")
        return value

    @field_validator("sku_id", "rate_id", "billing_region", "billing_mode")
    @classmethod
    def _require_exact_nonblank_billing_id(cls, value: str) -> str:
        if value != value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError("tariff billing identifier must be an exact printable value")
        return value

    def nano_usd_per_token_ceiling(self) -> int:
        """Normalize the retained source rate into a conservative nano-USD token floor."""
        token_unit = 1_000 if self.source_price_unit == "per_1k_tokens" else 1_000_000
        nano_usd = self.source_price_usd * _NANO_USD_PER_USD / token_unit
        return int(nano_usd.to_integral_value(rounding=ROUND_CEILING))


class ProviderTariffRoute(BaseModel):
    """Exact nonsecret execution and billing coordinates for one price record."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal[3] = 3
    provider_config: ProviderConfig
    billing_region: str = Field(min_length=1, max_length=128)
    billing_mode: str = Field(min_length=1, max_length=128)
    billing_meters: tuple[ProviderTariffBillingMeter, ...] = Field(min_length=2, max_length=2)

    @field_validator("provider_config", mode="after")
    @classmethod
    def _freeze_provider_config(cls, value: ProviderConfig) -> ProviderConfig:
        return ProviderConfig.model_validate(value.model_dump())

    @field_validator("billing_region", "billing_mode")
    @classmethod
    def _require_exact_nonblank_coordinate(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("tariff billing coordinates cannot have surrounding whitespace")
        return value

    @model_validator(mode="after")
    def _require_complete_execution_route(self) -> Self:
        if tuple(meter.usage_dimension for meter in self.billing_meters) != (
            "input_tokens",
            "output_tokens",
        ):
            raise ValueError("tariff billing meters must identify input_tokens then output_tokens")
        if any(meter.billing_region != self.billing_region for meter in self.billing_meters):
            raise ValueError("tariff billing meter region must match the route billing region")
        if any(meter.billing_mode != self.billing_mode for meter in self.billing_meters):
            raise ValueError("tariff billing meter mode must match the route billing mode")
        config = self.provider_config
        if config.kind not in {ProviderKind.AZURE_OPENAI, ProviderKind.BEDROCK}:
            raise ValueError("provider tariff route has no registered paid evidence verifier")
        if config.kind is ProviderKind.BEDROCK:
            if config.region is None:
                raise ValueError("tariff route requires an explicit Bedrock region")
            if self.billing_region != config.region:
                raise ValueError("Bedrock billing region must match the provider region")
        if config.kind is ProviderKind.AZURE_OPENAI:
            if config.endpoint is None:
                raise ValueError("tariff route requires an explicit Azure endpoint")
            if config.deployment is None or config.api_version is None:
                raise ValueError(
                    "tariff route requires an explicit Azure deployment and API version"
                )
            if config.model_type is None:
                raise ValueError("tariff route requires an explicit Azure canonical model_type")
            parsed_endpoint = urlsplit(config.endpoint)
            if (
                parsed_endpoint.scheme != "https"
                or parsed_endpoint.hostname is None
                or parsed_endpoint.username is not None
                or parsed_endpoint.password is not None
                or parsed_endpoint.query
                or parsed_endpoint.fragment
            ):
                raise ValueError(
                    "tariff route Azure endpoint must be an HTTPS locator without credentials, "
                    "query, or fragment"
                )
        if config.model_type is not None:
            resolved = resolve_provider_model(config.kind, config.model)
            if resolved.model_type != config.model_type:
                raise ValueError("tariff route model_type differs from its runtime model")
        return self


class ProviderTariffProvenance(BaseModel):
    """Auditable source snapshot, exact route, and usage scope for one frozen tariff."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal[5] = 5
    source_snapshots: tuple[ProviderTariffSourceSnapshot, ...] = Field(min_length=1, max_length=8)
    source_bindings: tuple[ProviderTariffSourceBinding, ...] = Field(min_length=3, max_length=32)
    verified_on: date
    effective_on: date | None
    currency: Literal["USD"]
    price_unit: Literal["per_1m_tokens"]
    route: ProviderTariffRoute
    priced_usage_dimensions: tuple[str, ...] = ("input_tokens", "output_tokens")

    @field_validator("priced_usage_dimensions")
    @classmethod
    def _require_supported_usage_dimensions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != ("input_tokens", "output_tokens"):
            raise ValueError(
                "priced_usage_dimensions must be exactly input_tokens and output_tokens"
            )
        return value

    @model_validator(mode="after")
    def _require_canonical_source_chain(self) -> Self:
        source_ids = tuple(snapshot.source_id for snapshot in self.source_snapshots)
        if source_ids != tuple(sorted(source_ids)) or len(source_ids) != len(set(source_ids)):
            raise ValueError("tariff source snapshots require unique ascending source IDs")
        sources_by_id = {snapshot.source_id: snapshot for snapshot in self.source_snapshots}
        if not any(snapshot.role == "rate_catalog" for snapshot in self.source_snapshots):
            raise ValueError("tariff source snapshots require a rate catalog")
        unknown_sources = {
            meter.source_id
            for meter in self.route.billing_meters
            if meter.source_id not in source_ids
        }
        if unknown_sources:
            raise ValueError("tariff billing meters reference an unknown source snapshot")
        if any(
            sources_by_id[meter.source_id].role != "rate_catalog"
            for meter in self.route.billing_meters
        ):
            raise ValueError("tariff billing meters require rate_catalog sources")
        if any(meter.effective_on != self.effective_on for meter in self.route.billing_meters):
            raise ValueError("tariff billing meter effective date must match provenance")
        if self.effective_on is not None and self.effective_on > self.verified_on:
            raise ValueError("tariff rate cannot become effective after its verification date")
        if any(
            snapshot.published_on is not None and snapshot.published_on > self.verified_on
            for snapshot in self.source_snapshots
        ):
            raise ValueError("tariff source cannot be published after its verification date")

        binding_coordinates = tuple(
            (
                binding.claim,
                binding.canonical_value,
                binding.source_id,
                binding.source_record_path,
                binding.source_value,
                binding.target_meter_source_id or "",
                binding.target_meter_record_path or "",
            )
            for binding in self.source_bindings
        )
        if binding_coordinates != tuple(sorted(binding_coordinates)) or len(
            binding_coordinates
        ) != len(set(binding_coordinates)):
            raise ValueError("tariff source bindings require unique canonical ascending order")
        unknown_binding_sources = {
            binding.source_id
            for binding in self.source_bindings
            if binding.source_id not in sources_by_id
        }
        if unknown_binding_sources:
            raise ValueError("tariff source bindings reference an unknown source snapshot")
        for binding in self.source_bindings:
            role = sources_by_id[binding.source_id].role
            allowed_roles = (
                {"rate_catalog", "route_definition", "semantic_mapping"}
                if binding.claim == "route_identity"
                else {"rate_catalog", "semantic_mapping"}
            )
            if role not in allowed_roles:
                raise ValueError("tariff source binding claim is incompatible with its source role")
        meter_coordinates = {
            (meter.source_id, meter.source_record_path): meter
            for meter in self.route.billing_meters
        }
        for binding in self.source_bindings:
            if binding.target_meter_source_id is None:
                continue
            target_coordinate = (
                binding.target_meter_source_id,
                binding.target_meter_record_path,
            )
            target_meter = meter_coordinates.get(target_coordinate)
            if target_meter is None:
                raise ValueError("tariff source binding targets an unknown billing meter")
            if (
                binding.claim == "usage_dimension"
                and binding.canonical_value != target_meter.usage_dimension
            ):
                raise ValueError("usage-dimension binding targets the wrong billing meter")
        bound_source_ids = {binding.source_id for binding in self.source_bindings}
        unbound_non_rate_sources = {
            snapshot.source_id
            for snapshot in self.source_snapshots
            if snapshot.role != "rate_catalog" and snapshot.source_id not in bound_source_ids
        }
        if unbound_non_rate_sources:
            raise ValueError("tariff provenance contains an unbound non-rate source")

        route_values = {
            binding.canonical_value
            for binding in self.source_bindings
            if binding.claim == "route_identity"
        }
        if route_values != {self.route.provider_config.model}:
            raise ValueError("tariff route identity is not bound to its runtime model")
        usage_dimensions = {
            binding.canonical_value
            for binding in self.source_bindings
            if binding.claim == "usage_dimension"
        }
        if usage_dimensions != {"input_tokens", "output_tokens"}:
            raise ValueError("tariff source bindings require both priced usage dimensions")
        usage_targets = {
            (
                binding.target_meter_source_id,
                binding.target_meter_record_path,
            )
            for binding in self.source_bindings
            if binding.claim == "usage_dimension"
        }
        if usage_targets != set(meter_coordinates):
            raise ValueError("tariff usage bindings must target every billing meter exactly")
        if not any(snapshot.role == "route_definition" for snapshot in self.source_snapshots):
            route_targets = {
                (
                    binding.target_meter_source_id,
                    binding.target_meter_record_path,
                )
                for binding in self.source_bindings
                if binding.claim == "route_identity" and binding.target_meter_source_id is not None
            }
            if route_targets != set(meter_coordinates):
                raise ValueError("rate-catalog route identity must bind every billing meter")
        return self

    def token_price_floor(self) -> TokenPriceCeiling:
        """Return the exact conservative token floor normalized from retained meter records."""
        input_meter, output_meter = self.route.billing_meters
        return TokenPriceCeiling(
            input_nano_usd_per_token=input_meter.nano_usd_per_token_ceiling(),
            output_nano_usd_per_token=output_meter.nano_usd_per_token_ceiling(),
        )


def provider_tariff_claim_digest(
    *,
    provider_config: ProviderConfig,
    price: TokenPriceCeiling,
    provenance: ProviderTariffProvenance,
) -> str:
    """Digest the pre-receipt tariff claim without creating a circular receipt dependency."""
    return _digest_json(
        {
            "schema_version": 1,
            "provider_config": provider_config.model_dump(mode="json"),
            "price": price.model_dump(mode="json"),
            "provenance": provenance.model_dump(mode="json"),
        }
    )


def provider_tariff_validated_records(
    provenance: ProviderTariffProvenance,
) -> tuple[ProviderTariffValidatedRecord, ...]:
    """Return the canonical semantic coordinates a verifier must attest."""
    records = [
        ProviderTariffValidatedRecord(
            source_id=binding.source_id,
            source_record_path=binding.source_record_path,
            purpose=binding.claim,
        )
        for binding in provenance.source_bindings
    ]
    records.extend(
        ProviderTariffValidatedRecord(
            source_id=meter.source_id,
            source_record_path=meter.source_record_path,
            purpose=(
                "input_tokens_billing_meter"
                if meter.usage_dimension == "input_tokens"
                else "output_tokens_billing_meter"
            ),
        )
        for meter in provenance.route.billing_meters
    )
    return tuple(
        sorted(
            records,
            key=lambda record: (record.source_id, record.source_record_path, record.purpose),
        )
    )


class ProviderCostMeter(BaseModel):
    """One immutable provider route, tariff ceiling, and input estimator."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    kind: Literal["provider_tokens"] = "provider_tokens"
    provider_config: ProviderConfig
    price: TokenPriceCeiling
    tariff_provenance: ProviderTariffProvenance
    tariff_evidence_receipt: ProviderTariffEvidenceReceipt
    input_estimator: Literal["canonical-json-utf8-v1"] = "canonical-json-utf8-v1"
    input_overhead_tokens: int = Field(default=8192, ge=1, le=_SQLITE_INTEGER_MAX)

    @field_validator("provider_config", mode="after")
    @classmethod
    def _freeze_provider_config(cls, value: ProviderConfig) -> ProviderConfig:
        return ProviderConfig.model_validate(value.model_dump())

    @model_validator(mode="after")
    def _require_tariff_route_match(self) -> Self:
        if self.tariff_provenance.route.provider_config != self.provider_config:
            raise ValueError("tariff route does not match provider config")
        floor = self.tariff_provenance.token_price_floor()
        if (
            self.price.input_nano_usd_per_token < floor.input_nano_usd_per_token
            or self.price.output_nano_usd_per_token < floor.output_nano_usd_per_token
        ):
            raise ValueError(
                "provider token price ceiling understates retained billing meter rates"
            )
        receipt = self.tariff_evidence_receipt
        expected_profile = (
            "aws_bedrock_public_catalog_v1"
            if self.provider_config.kind is ProviderKind.BEDROCK
            else "azure_retail_arm_v1"
        )
        if receipt.verifier_profile != expected_profile:
            raise ValueError("tariff evidence verifier profile differs from the provider route")
        claim_digest = provider_tariff_claim_digest(
            provider_config=self.provider_config,
            price=self.price,
            provenance=self.tariff_provenance,
        )
        if receipt.tariff_claim_digest != claim_digest:
            raise ValueError("tariff evidence receipt belongs to a different tariff claim")
        expected_sources = tuple(
            (
                snapshot.source_id,
                snapshot.retained_artifact.artifact_digest,
                snapshot.source_snapshot_digest,
            )
            for snapshot in self.tariff_provenance.source_snapshots
        )
        receipt_sources = tuple(
            (
                source.source_id,
                source.artifact_digest,
                source.source_snapshot_digest,
            )
            for source in receipt.verified_sources
        )
        if receipt_sources != expected_sources:
            raise ValueError("tariff evidence receipt source set differs from provenance")
        if receipt.validated_records != provider_tariff_validated_records(self.tariff_provenance):
            raise ValueError("tariff evidence receipt record set differs from provenance")
        return self


class ExternalSpendAuthority(BaseModel):
    """Policy-pinned verifier and account for an independent provider spending cap."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_.-]*$")
    account_identity: str = Field(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9_.:@/-]+$",
    )
    verifier_digest: str = Field(pattern=_DIGEST_PATTERN)


class TimedResourceCostMeter(BaseModel):
    """Frozen upper-bound tariff for one class of externally billed timed resource."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["timed_resource"] = "timed_resource"
    resource_type: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_.-]*$")
    resource_class_digest: str = Field(pattern=_DIGEST_PATTERN)
    fixed_nano_usd: int = Field(default=0, ge=0, le=_SQLITE_INTEGER_MAX)
    nano_usd_per_second: int = Field(default=0, ge=0, le=_SQLITE_INTEGER_MAX)
    billing_quantum_seconds: int = Field(default=1, ge=1, le=_SQLITE_INTEGER_MAX)
    max_billing_seconds: int = Field(gt=0, le=_SQLITE_INTEGER_MAX)
    external_spend_authority: ExternalSpendAuthority | None = None

    @model_validator(mode="after")
    def _require_priced_resource(self) -> Self:
        if self.fixed_nano_usd == self.nano_usd_per_second == 0:
            raise ValueError("timed resource must have a positive fixed or duration price")
        self.maximum_charge_nano_usd()
        return self

    def billed_seconds(self, elapsed_seconds: Decimal | str | int | float) -> int:
        """Round one created resource's nonnegative duration up to at least one quantum."""
        duration = _decimal_duration(elapsed_seconds)
        if duration < 0:
            raise ValueError("resource duration cannot be negative")
        quanta = int(
            (duration / self.billing_quantum_seconds).to_integral_value(rounding=ROUND_CEILING)
        )
        billed = max(quanta, 1) * self.billing_quantum_seconds
        if billed > _SQLITE_INTEGER_MAX:
            raise OverflowError("resource billing duration does not fit a SQLite integer")
        return billed

    def charge_nano_usd(self, *, billed_seconds: int) -> int:
        if billed_seconds < 0 or billed_seconds > _SQLITE_INTEGER_MAX:
            raise ValueError("billed resource seconds must fit a nonnegative SQLite integer")
        charged = self.fixed_nano_usd + billed_seconds * self.nano_usd_per_second
        if charged > _SQLITE_INTEGER_MAX:
            raise OverflowError("timed resource charge does not fit a SQLite integer")
        return charged

    def maximum_charge_nano_usd(self) -> int:
        billed = self.billed_seconds(self.max_billing_seconds)
        return self.charge_nano_usd(billed_seconds=billed)


class TimedResourceRole(StrEnum):
    """Stable backend roles whose independently billed leases share one hard cap."""

    TASK_ENVIRONMENT = "task_environment"
    TASK_ENVIRONMENT_BUILD = "task_environment_build"
    AGENT_RUNNER = "agent_runner"
    PROPOSER_PROJECT = "proposer_project"


class TimedResourceClass(BaseModel):
    """Canonical cost-driving identity for one immutable external resource class.

    The host-observation horizon deliberately includes bounded create and cleanup request time
    around the provider-side TTL. This lets settlement use a conservative host monotonic clock
    without treating expected control-plane latency as an apparent reservation breach.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    backend: Literal["e2b"] = "e2b"
    role: TimedResourceRole
    cpu_count: int = Field(ge=1, le=_SQLITE_INTEGER_MAX)
    memory_mb: int = Field(ge=1, le=_SQLITE_INTEGER_MAX)
    provider_ttl_seconds: int = Field(gt=0, le=_SQLITE_INTEGER_MAX)
    create_request_timeout_seconds: int = Field(gt=0, le=_SQLITE_INTEGER_MAX)
    cleanup_horizon_seconds: int = Field(gt=0, le=_SQLITE_INTEGER_MAX)

    @property
    def digest(self) -> str:
        """Bind every cost-driving field without exposing credentials or host paths."""
        return _digest_json(self.model_dump(mode="json"))

    @property
    def max_host_observation_seconds(self) -> int:
        """Return the longest admitted reserve-to-cleanup-proof observation."""
        horizon = (
            self.create_request_timeout_seconds
            + self.provider_ttl_seconds
            + self.cleanup_horizon_seconds
        )
        if horizon > _SQLITE_INTEGER_MAX:
            raise OverflowError("timed resource observation horizon does not fit SQLite")
        return horizon


_CostMeter = Annotated[
    ProviderCostMeter | TimedResourceCostMeter,
    Field(discriminator="kind"),
]


class BudgetPolicy(BaseModel):
    """Immutable experiment caps and route-specific conservative tariffs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    study_id: str = Field(min_length=1)
    manifest_digest: str = Field(pattern=_DIGEST_PATTERN)
    hard_limit_nano_usd: int = Field(gt=0, le=_SQLITE_INTEGER_MAX)
    phase_limits_nano_usd: dict[str, int] = Field(min_length=1)
    meters: dict[str, _CostMeter] = Field(min_length=1)

    @field_validator("phase_limits_nano_usd")
    @classmethod
    def _validate_phases(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not phase.strip() for phase in value):
            raise ValueError("budget phase names cannot be blank")
        if any(limit < 0 or limit > _SQLITE_INTEGER_MAX for limit in value.values()):
            raise ValueError("budget phase limits must fit a nonnegative SQLite integer")
        return dict(sorted(value.items()))

    @field_validator("meters")
    @classmethod
    def _validate_meters(
        cls,
        value: dict[str, _CostMeter],
    ) -> dict[str, _CostMeter]:
        if any(not meter_id.strip() for meter_id in value):
            raise ValueError("budget meter ids cannot be blank")
        return dict(sorted(value.items()))

    @model_validator(mode="after")
    def _validate_total(self) -> Self:
        if sum(self.phase_limits_nano_usd.values()) > self.hard_limit_nano_usd:
            raise ValueError("budget phase limits cannot sum above the hard limit")
        return self

    @property
    def policy_digest(self) -> str:
        """Return the canonical digest that binds every ledger open."""
        return _digest_json(_budget_policy_dump(self))


class BudgetAccount(BaseModel):
    """Serializable provider-call budget binding used by local and worker processes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ledger_path: Path
    ledger_identity: str = Field(pattern=_DIGEST_PATTERN)
    policy: BudgetPolicy
    scope: BudgetScope
    meter_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _bind_scope_to_policy(self) -> Self:
        if not self.ledger_path.is_absolute():
            raise ValueError("budget ledger_path must be absolute for cross-process workers")
        if self.scope.phase not in self.policy.phase_limits_nano_usd:
            raise ValueError(f"budget scope phase {self.scope.phase!r} is absent from policy")
        if self.meter_id not in self.policy.meters:
            raise ValueError(f"budget meter {self.meter_id!r} is absent from policy")
        if not isinstance(self.policy.meters[self.meter_id], ProviderCostMeter):
            raise ValueError("provider budget account requires a provider token meter")
        return self


class TimedResourceBudgetAccount(BaseModel):
    """Serializable hard-budget account for one external timed resource lease."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ledger_path: Path
    ledger_identity: str = Field(pattern=_DIGEST_PATTERN)
    policy: BudgetPolicy
    scope: BudgetScope
    meter_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _bind_scope_to_policy(self) -> Self:
        if not self.ledger_path.is_absolute():
            raise ValueError("budget ledger_path must be absolute for resource workers")
        if self.scope.phase not in self.policy.phase_limits_nano_usd:
            raise ValueError(f"budget scope phase {self.scope.phase!r} is absent from policy")
        meter = self.policy.meters.get(self.meter_id)
        if not isinstance(meter, TimedResourceCostMeter):
            raise ValueError("timed resource account requires a timed resource meter")
        return self


class BudgetAccountBinding(BaseModel):
    """Path-free account reference safe to retain in durable evaluator configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    ledger_identity: str = Field(pattern=_DIGEST_PATTERN)
    scope: BudgetScope
    meter_id: str = Field(min_length=1)


class BudgetLedgerAuthority(BaseModel):
    """Explicitly bootstrapped durable ledger authority used to mint bound accounts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ledger_path: Path
    ledger_identity: str = Field(pattern=_DIGEST_PATTERN)
    policy: BudgetPolicy

    @model_validator(mode="after")
    def _require_absolute_path(self) -> Self:
        if not self.ledger_path.is_absolute():
            raise ValueError("budget authority ledger_path must be absolute")
        return self

    def provider_account(
        self,
        *,
        scope: BudgetScope,
        meter_id: str,
    ) -> BudgetAccount:
        """Mint one provider account bound to this exact pre-existing ledger."""
        return BudgetAccount(
            ledger_path=self.ledger_path,
            ledger_identity=self.ledger_identity,
            policy=self.policy,
            scope=scope,
            meter_id=meter_id,
        )

    def timed_resource_account(
        self,
        *,
        scope: BudgetScope,
        meter_id: str,
    ) -> TimedResourceBudgetAccount:
        """Mint one timed-resource account bound to this exact pre-existing ledger."""
        return TimedResourceBudgetAccount(
            ledger_path=self.ledger_path,
            ledger_identity=self.ledger_identity,
            policy=self.policy,
            scope=scope,
            meter_id=meter_id,
        )


class BudgetReservation(BaseModel):
    """Current state reconstructed from one reservation's append-only events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reservation_id: str = Field(min_length=1)
    scope: BudgetScope
    meter_id: str = Field(min_length=1)
    max_nano_usd: int = Field(ge=0, le=_SQLITE_INTEGER_MAX)
    charged_nano_usd: int = Field(ge=0, le=_SQLITE_INTEGER_MAX)
    status: ReservationStatus
    input_tokens: int | None = Field(default=None, ge=0, le=_SQLITE_INTEGER_MAX)
    output_tokens: int | None = Field(default=None, ge=0, le=_SQLITE_INTEGER_MAX)
    usage_quantity: int | None = Field(default=None, ge=0, le=_SQLITE_INTEGER_MAX)
    usage_unit: str | None = Field(default=None, min_length=1)
    failure_type: str | None = None
    breach_kind: BudgetBreachKind | None = None


class BudgetTerminalProvenance(BaseModel):
    """Domain-separated digest bound into one append-only terminal settlement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    namespace: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z][a-z0-9_.-]*$",
    )
    digest: str = Field(pattern=_DIGEST_PATTERN)


class BudgetSnapshot(BaseModel):
    """Atomic exposure summary used by operator gates and call admission."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hard_limit_nano_usd: int
    charged_nano_usd: int
    reserved_nano_usd: int
    remaining_nano_usd: int
    by_phase_nano_usd: dict[str, int]
    breached: bool


class _BudgetOpened(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["opened"] = "opened"
    policy: BudgetPolicy


class _BudgetReserved(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["reserved"] = "reserved"
    reservation_id: str = Field(min_length=1)
    scope: BudgetScope
    meter_id: str = Field(min_length=1)
    max_nano_usd: int = Field(ge=0, le=_SQLITE_INTEGER_MAX)


class _BudgetSettled(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["settled"] = "settled"
    reservation_id: str = Field(min_length=1)
    charged_nano_usd: int = Field(ge=0, le=_SQLITE_INTEGER_MAX)
    input_tokens: int | None = Field(default=None, ge=0, le=_SQLITE_INTEGER_MAX)
    output_tokens: int | None = Field(default=None, ge=0, le=_SQLITE_INTEGER_MAX)
    usage_quantity: int | None = Field(default=None, ge=0, le=_SQLITE_INTEGER_MAX)
    usage_unit: str | None = Field(default=None, min_length=1)
    status: Literal[ReservationStatus.SETTLED, ReservationStatus.BREACHED]
    breach_kind: BudgetBreachKind | None = None
    terminal_provenance: BudgetTerminalProvenance | None = None

    @model_validator(mode="after")
    def _validate_breach(self) -> Self:
        if self.status is ReservationStatus.BREACHED and self.breach_kind is None:
            raise ValueError("breached settlement requires a breach kind")
        if self.status is ReservationStatus.SETTLED and self.breach_kind is not None:
            raise ValueError("ordinary settlement cannot carry a breach kind")
        tokens = self.input_tokens is not None or self.output_tokens is not None
        resource = self.usage_quantity is not None or self.usage_unit is not None
        if tokens == resource:
            raise ValueError("settlement must carry exactly one typed usage shape")
        if tokens and (self.input_tokens is None or self.output_tokens is None):
            raise ValueError("token settlement requires both input and output counts")
        if resource and (self.usage_quantity is None or self.usage_unit is None):
            raise ValueError("resource settlement requires quantity and unit")
        return self


class _BudgetForfeited(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["forfeited"] = "forfeited"
    reservation_id: str = Field(min_length=1)
    charged_nano_usd: int = Field(ge=0, le=_SQLITE_INTEGER_MAX)
    failure_type: str = Field(pattern=_FAILURE_CODE_PATTERN.pattern)


_BudgetAction = Annotated[
    _BudgetOpened | _BudgetReserved | _BudgetSettled | _BudgetForfeited,
    Field(discriminator="kind"),
]


class BudgetEvent(BaseModel):
    """One immutable hash-linked ledger event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: int = Field(ge=1)
    recorded_at: datetime
    previous_digest: str = Field(pattern=_DIGEST_PATTERN)
    action: _BudgetAction
    digest: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator("recorded_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("budget event timestamps must include a timezone")
        return value


class SpendLedger:
    """SQLite-backed atomic reservation ledger with an append-only audit chain."""

    def __init__(
        self,
        path: str | Path,
        policy: BudgetPolicy,
        *,
        now: Callable[[], datetime] | None = None,
        allow_create: bool = True,
        allow_authority_create: bool | None = None,
    ) -> None:
        requested = Path(path).expanduser()
        if requested.is_symlink():
            raise BudgetIntegrityError("budget ledger path cannot be a symlink")
        if allow_create:
            requested.parent.mkdir(parents=True, exist_ok=True)
        elif not requested.parent.is_dir():
            raise BudgetIntegrityError("budget ledger parent must exist after bootstrap")
        self.path = requested.parent.resolve() / requested.name
        self._authority_path = _authority_path_for_ledger(self.path)
        self._allow_create = allow_create
        self._allow_authority_create = (
            allow_create if allow_authority_create is None else allow_authority_create
        )
        if not allow_create and not self.path.exists():
            raise BudgetIntegrityError("budget ledger must be explicitly bootstrapped before use")
        if self.path.exists() and not stat.S_ISREG(self.path.stat(follow_symlinks=False).st_mode):
            raise BudgetIntegrityError("budget ledger path must be a regular file")
        self._policy = BudgetPolicy.model_validate(policy.model_dump())
        self._now = now or (lambda: datetime.now(UTC))
        self._state_lock = threading.RLock()
        self._verified_sequence = 0
        self._verified_digest = _ZERO_DIGEST
        self._verified_reservations: dict[str, BudgetReservation] = {}
        self._verified_authority_sequence = 0
        self._verified_authority_digest = _ZERO_DIGEST
        self._initialize()
        ledger_stat = self.path.stat(follow_symlinks=False)
        self._file_identity = (ledger_stat.st_dev, ledger_stat.st_ino)
        self.audit()
        self._initialize_authority()
        authority_stat = self._authority_path.stat(follow_symlinks=False)
        self._authority_file_identity = (authority_stat.st_dev, authority_stat.st_ino)
        with self._authority_transaction() as connection:
            # Another process may have advanced the ledger between the first audit and our
            # acquisition of the cross-process authority lock.
            with self._state_lock:
                self._audit_unlocked()
            verified_authority = self._synchronize_authority(connection, full_audit=True)
        self._verified_authority_sequence, self._verified_authority_digest = verified_authority

    @property
    def policy(self) -> BudgetPolicy:
        """Return a defensive policy copy; callers cannot mutate admission state in memory."""
        return self._policy.model_copy(deep=True)

    @property
    def authority_path(self) -> Path:
        """Return the host-local high-water database paired with this ledger."""
        return self._authority_path

    @property
    def ledger_identity(self) -> str:
        """Return an opaque identity unique to this initialized ledger and canonical path."""
        return _digest_json(
            {
                "identity_version": 2,
                "ledger_nonce": self._ledger_nonce,
                "authority_nonce": self._authority_nonce,
                "canonical_path": str(self.path),
                "policy_digest": self._policy.policy_digest,
            }
        )

    def reserve(
        self,
        scope: BudgetScope,
        *,
        meter_id: str,
        max_nano_usd: int,
        reservation_id: str,
    ) -> BudgetReservation:
        """Atomically reserve a call ceiling before any external side effect."""
        if max_nano_usd < 0:
            raise ValueError("reservation ceiling cannot be negative")
        if not reservation_id.strip():
            raise ValueError("reservation_id cannot be blank")
        scope = BudgetScope.model_validate(scope.model_dump())
        if scope.phase not in self._policy.phase_limits_nano_usd:
            raise ValueError(f"unknown budget phase {scope.phase!r}")
        if meter_id not in self._policy.meters:
            raise ValueError(f"unknown budget meter {meter_id!r}")
        meter = self._policy.meters[meter_id]
        if (
            isinstance(meter, TimedResourceCostMeter)
            and max_nano_usd != meter.maximum_charge_nano_usd()
        ):
            raise ValueError("timed resource reservation must use its exact maximum charge")
        with self._verified_transaction() as connection:
            snapshot = _snapshot_from_reservations(
                self._policy,
                list(self._verified_reservations.values()),
            )
            if snapshot.breached:
                raise BudgetBreachError("budget ledger is already breached; no call may start")
            hard_remaining = snapshot.remaining_nano_usd
            if max_nano_usd > hard_remaining:
                raise BudgetExceededError(
                    f"hard budget cannot reserve {max_nano_usd} nano-USD; "
                    f"remaining={hard_remaining}"
                )
            phase_limit = self._policy.phase_limits_nano_usd[scope.phase]
            phase_used = snapshot.by_phase_nano_usd[scope.phase]
            phase_remaining = phase_limit - phase_used
            if max_nano_usd > phase_remaining:
                raise BudgetExceededError(
                    f"phase {scope.phase!r} cannot reserve {max_nano_usd} nano-USD; "
                    f"remaining={phase_remaining}"
                )
            action = _BudgetReserved(
                reservation_id=reservation_id,
                scope=scope,
                meter_id=meter_id,
                max_nano_usd=max_nano_usd,
            )
            self._append_event(connection, action)
            try:
                connection.execute(
                    """
                    INSERT INTO budget_reservations (
                        reservation_id, scope_json, meter_id, max_nano_usd, charged_nano_usd,
                        status, input_tokens, output_tokens, usage_quantity, usage_unit,
                        failure_type, breach_kind
                    ) VALUES (?, ?, ?, ?, 0, ?, NULL, NULL, NULL, NULL, NULL, NULL)
                    """,
                    (
                        reservation_id,
                        _canonical_json(scope.model_dump(mode="json")),
                        meter_id,
                        max_nano_usd,
                        ReservationStatus.RESERVED.value,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise BudgetIntegrityError(
                    f"duplicate budget reservation_id {reservation_id!r}"
                ) from exc
        return self._reservation(reservation_id)

    def settle(
        self,
        reservation_id: str,
        *,
        charged_nano_usd: int,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        usage_quantity: int | None = None,
        usage_unit: str | None = None,
        breach_kind: BudgetBreachKind | None = None,
        terminal_provenance: BudgetTerminalProvenance | None = None,
    ) -> BudgetReservation:
        """Settle exact usage, recording and raising after any ceiling breach."""
        if charged_nano_usd < 0:
            raise ValueError("settlement cost cannot be negative")
        settlement = _BudgetSettled(
            reservation_id=reservation_id,
            charged_nano_usd=charged_nano_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usage_quantity=usage_quantity,
            usage_unit=usage_unit,
            status=(
                ReservationStatus.BREACHED if breach_kind is not None else ReservationStatus.SETTLED
            ),
            breach_kind=breach_kind,
            terminal_provenance=terminal_provenance,
        )
        breached = False
        resolved_breach = breach_kind
        with self._verified_transaction() as connection:
            reservation = self._verified_reservation(connection, reservation_id)
            if reservation.status is not ReservationStatus.RESERVED:
                raise BudgetIntegrityError(
                    f"reservation {reservation_id!r} is already {reservation.status.value}"
                )
            if charged_nano_usd > reservation.max_nano_usd:
                resolved_breach = BudgetBreachKind.RESERVATION
            breached = resolved_breach is not None
            status = ReservationStatus.BREACHED if breached else ReservationStatus.SETTLED
            final_settlement = settlement.model_copy(
                update={"status": status, "breach_kind": resolved_breach}
            )
            _validate_settlement_event(final_settlement, reservation, self._policy)
            self._append_event(connection, final_settlement)
            connection.execute(
                """
                UPDATE budget_reservations
                SET charged_nano_usd = ?, status = ?, input_tokens = ?, output_tokens = ?,
                    usage_quantity = ?, usage_unit = ?, breach_kind = ?
                WHERE reservation_id = ?
                """,
                (
                    charged_nano_usd,
                    status.value,
                    input_tokens,
                    output_tokens,
                    usage_quantity,
                    usage_unit,
                    resolved_breach.value if resolved_breach is not None else None,
                    reservation_id,
                ),
            )
        settled = self._reservation(reservation_id)
        if breached:
            assert resolved_breach is not None
            detail = {
                BudgetBreachKind.RESERVATION: (
                    f"exceeded its {settled.max_nano_usd} nano-USD reservation"
                ),
                BudgetBreachKind.INPUT_TOKEN_CEILING: "exceeded its input-token ceiling",
                BudgetBreachKind.OUTPUT_TOKEN_CEILING: "exceeded its output-token ceiling",
            }[resolved_breach]
            raise BudgetBreachError(f"reservation {reservation_id!r} {detail}")
        return settled

    def forfeit(self, reservation_id: str, *, failure_type: str) -> BudgetReservation:
        """Charge a full ceiling when exact provider usage cannot be proved."""
        if _FAILURE_CODE_PATTERN.fullmatch(failure_type) is None:
            raise ValueError("failure_type must be a bounded nonsecret failure code")
        with self._verified_transaction() as connection:
            reservation = self._verified_reservation(connection, reservation_id)
            if reservation.status is not ReservationStatus.RESERVED:
                raise BudgetIntegrityError(
                    f"reservation {reservation_id!r} is already {reservation.status.value}"
                )
            self._append_event(
                connection,
                _BudgetForfeited(
                    reservation_id=reservation_id,
                    charged_nano_usd=reservation.max_nano_usd,
                    failure_type=failure_type,
                ),
            )
            connection.execute(
                """
                UPDATE budget_reservations
                SET charged_nano_usd = max_nano_usd, status = ?, failure_type = ?
                WHERE reservation_id = ?
                """,
                (ReservationStatus.FORFEITED.value, failure_type, reservation_id),
            )
        return self._reservation(reservation_id)

    def snapshot(self) -> BudgetSnapshot:
        """Return one transactionally consistent exposure snapshot."""
        with self._verified_transaction():
            return _snapshot_from_reservations(
                self._policy,
                list(self._verified_reservations.values()),
            )

    def reservations(self) -> list[BudgetReservation]:
        """Return every current reservation state in stable identity order."""
        with self._verified_transaction():
            return [
                self._verified_reservations[reservation_id].model_copy(deep=True)
                for reservation_id in sorted(self._verified_reservations)
            ]

    def settlement_provenance(
        self,
        reservation_id: str,
    ) -> BudgetTerminalProvenance | None:
        """Return the append-only provenance attached to one settled reservation, if any."""
        matches = [
            event.action.terminal_provenance
            for event in self.events()
            if isinstance(event.action, _BudgetSettled)
            and event.action.reservation_id == reservation_id
        ]
        if len(matches) > 1:
            raise BudgetIntegrityError("budget reservation has duplicate settlement provenance")
        if not matches or matches[0] is None:
            return None
        return matches[0].model_copy(deep=True)

    def events(self) -> list[BudgetEvent]:
        """Return the append-only audit events after verifying each serialized entry."""
        self.audit()
        with self._state_lock:
            verified_sequence = self._verified_sequence
            with self._connection() as connection:
                rows = connection.execute(
                    "SELECT sequence, entry_json FROM budget_events "
                    "WHERE sequence <= ? ORDER BY sequence",
                    (verified_sequence,),
                ).fetchall()
            events = [
                self._parse_event(row["entry_json"], sequence=row["sequence"]) for row in rows
            ]
            previous = _ZERO_DIGEST
            for expected_sequence, event in enumerate(events, start=1):
                if (
                    event.sequence != expected_sequence
                    or event.previous_digest != previous
                    or _event_digest(event) != event.digest
                ):
                    raise BudgetIntegrityError(
                        f"budget event {expected_sequence} changed after full audit"
                    )
                previous = event.digest
            return events

    def audit(self) -> None:
        """Rebuild state from the full hash chain and compare the derived index exactly."""
        with self._state_lock:
            if hasattr(self, "_authority_nonce"):
                with self._authority_transaction() as authority:
                    self._audit_unlocked()
                    verified_authority = self._synchronize_authority(
                        authority,
                        full_audit=True,
                    )
                (
                    self._verified_authority_sequence,
                    self._verified_authority_digest,
                ) = verified_authority
            else:
                self._audit_unlocked()

    def _audit_unlocked(self) -> None:
        """Full audit implementation; caller owns the process-local state lock."""
        # Read the log and its derived index from one SQLite snapshot. Paid workers may append
        # concurrently while orchestration audits; two independent connections would otherwise
        # report a false integrity failure merely because a reservation landed between reads.
        with self._connection() as connection:
            connection.execute("BEGIN")
            event_rows = connection.execute(
                "SELECT sequence, entry_json FROM budget_events ORDER BY sequence"
            ).fetchall()
            reservation_rows = connection.execute(
                "SELECT * FROM budget_reservations ORDER BY reservation_id"
            ).fetchall()
        events = [
            self._parse_event(row["entry_json"], sequence=row["sequence"]) for row in event_rows
        ]
        if not events:
            raise BudgetIntegrityError("budget ledger has no genesis event")
        previous = _ZERO_DIGEST
        reconstructed: dict[str, BudgetReservation] = {}
        for expected_sequence, event in enumerate(events, start=1):
            if event.sequence != expected_sequence:
                raise BudgetIntegrityError(
                    f"budget event sequence gap at {expected_sequence}, found {event.sequence}"
                )
            if event.previous_digest != previous:
                raise BudgetIntegrityError(
                    f"budget event {event.sequence} previous digest does not match"
                )
            if _event_digest(event) != event.digest:
                raise BudgetIntegrityError(f"budget event {event.sequence} digest does not match")
            _apply_budget_event(self._policy, event, reconstructed)
            previous = event.digest
        indexed = {
            item.reservation_id: item
            for item in (self._reservation_from_row(row) for row in reservation_rows)
        }
        if reconstructed != indexed:
            raise BudgetIntegrityError("budget reservation index differs from append-only events")
        reconstructed_reservations = list(reconstructed.values())
        _validate_final_snapshot(
            _snapshot_from_reservations(self._policy, reconstructed_reservations),
            reconstructed_reservations,
            self._policy,
        )
        prior_sequence = self._verified_sequence
        prior_digest = self._verified_digest
        if prior_sequence:
            if len(events) < prior_sequence or events[prior_sequence - 1].digest != prior_digest:
                raise BudgetIntegrityError("budget ledger was rolled back after audit")
        self._verified_sequence = events[-1].sequence
        self._verified_digest = events[-1].digest
        self._verified_reservations = {
            reservation_id: reservation.model_copy(deep=True)
            for reservation_id, reservation in reconstructed.items()
        }

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(_SCHEMA)
        if self.path.exists():
            os.chmod(self.path, 0o600)
        with self._transaction() as connection:
            version_row = connection.execute(
                "SELECT schema_version FROM budget_metadata WHERE id = 1"
            ).fetchone()
            if version_row is None:
                ledger_nonce = secrets.token_hex(32)
                connection.execute(
                    """
                    INSERT INTO budget_metadata (
                        id, schema_version, ledger_nonce, policy_digest, policy_json
                    )
                    VALUES (1, ?, ?, ?, ?)
                    """,
                    (
                        _SCHEMA_VERSION,
                        ledger_nonce,
                        self._policy.policy_digest,
                        _canonical_json(_budget_policy_dump(self._policy)),
                    ),
                )
                self._ledger_nonce = ledger_nonce
                self._append_event(connection, _BudgetOpened(policy=self._policy))
            else:
                if version_row["schema_version"] != _SCHEMA_VERSION:
                    raise BudgetIntegrityError(
                        f"unsupported budget schema version {version_row['schema_version']}"
                    )
                row = connection.execute(
                    "SELECT ledger_nonce, policy_digest, policy_json "
                    "FROM budget_metadata WHERE id = 1"
                ).fetchone()
                if row is None:
                    raise BudgetIntegrityError("budget metadata disappeared during initialization")
                ledger_nonce = row["ledger_nonce"]
                if (
                    not isinstance(ledger_nonce, str)
                    or re.fullmatch(r"[0-9a-f]{64}", ledger_nonce) is None
                ):
                    raise BudgetIntegrityError("budget ledger identity is malformed")
                self._ledger_nonce = ledger_nonce
                try:
                    persisted = BudgetPolicy.model_validate_json(row["policy_json"])
                except ValueError as exc:
                    raise BudgetIntegrityError("budget policy metadata is malformed") from exc
                if row["policy_digest"] != self._policy.policy_digest or persisted != self._policy:
                    raise BudgetIntegrityError("budget policy differs from the existing ledger")

    def _initialize_authority(self) -> None:
        """Create or validate the independent monotonic ledger-head authority."""
        path = self._authority_path
        if path.is_symlink():
            raise BudgetIntegrityError("budget authority path cannot be a symlink")
        if not path.exists():
            if not self._allow_authority_create:
                raise BudgetIntegrityError(
                    "budget authority must be explicitly bootstrapped before paid use"
                )
            flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(path, flags, 0o600)
            except FileExistsError:
                pass
            except OSError as exc:
                raise BudgetIntegrityError("budget authority could not be created safely") from exc
            else:
                os.close(descriptor)
        if not path.exists() or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
            raise BudgetIntegrityError("budget authority path must be a regular file")
        try:
            connection = sqlite3.connect(
                path.as_uri() + "?mode=rw",
                timeout=_DEFAULT_BUSY_TIMEOUT_MS / 1000,
                uri=True,
            )
        except sqlite3.OperationalError as exc:
            raise BudgetIntegrityError("budget authority must remain readable") from exc
        connection.row_factory = sqlite3.Row
        try:
            connection.execute(f"PRAGMA busy_timeout = {_DEFAULT_BUSY_TIMEOUT_MS}")
            connection.executescript(_AUTHORITY_SCHEMA)
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM budget_authority_metadata WHERE id = 1"
            ).fetchone()
            if row is None:
                if not self._allow_authority_create:
                    raise BudgetIntegrityError("budget authority metadata is missing")
                if self._verified_sequence != 1:
                    raise BudgetIntegrityError(
                        "missing budget authority cannot be recreated after ledger use"
                    )
                authority_nonce = secrets.token_hex(32)
                connection.execute(
                    """
                    INSERT INTO budget_authority_metadata (
                        id, schema_version, authority_nonce, ledger_nonce,
                        canonical_ledger_path, policy_digest
                    ) VALUES (1, ?, ?, ?, ?, ?)
                    """,
                    (
                        _AUTHORITY_SCHEMA_VERSION,
                        authority_nonce,
                        self._ledger_nonce,
                        str(self.path),
                        self._policy.policy_digest,
                    ),
                )
                connection.execute(
                    "INSERT INTO budget_authority_heads (sequence, digest) VALUES (?, ?)",
                    (self._verified_sequence, self._verified_digest),
                )
                self._authority_nonce = authority_nonce
            else:
                self._authority_nonce = self._validate_authority_metadata(row)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        os.chmod(path, 0o600)

    def _validate_authority_metadata(self, row: sqlite3.Row) -> str:
        if row["schema_version"] != _AUTHORITY_SCHEMA_VERSION:
            raise BudgetIntegrityError("unsupported budget authority schema version")
        authority_nonce = row["authority_nonce"]
        if (
            not isinstance(authority_nonce, str)
            or re.fullmatch(r"[0-9a-f]{64}", authority_nonce) is None
        ):
            raise BudgetIntegrityError("budget authority identity is malformed")
        if (
            row["ledger_nonce"] != self._ledger_nonce
            or row["canonical_ledger_path"] != str(self.path)
            or row["policy_digest"] != self._policy.policy_digest
        ):
            raise BudgetIntegrityError("budget authority differs from its ledger")
        return authority_nonce

    def _synchronize_authority(
        self,
        authority: sqlite3.Connection,
        *,
        ledger: sqlite3.Connection | None = None,
        full_audit: bool = False,
    ) -> tuple[int, str]:
        """Verify monotonic chain progress and append every newly durable ledger head."""
        metadata = authority.execute(
            "SELECT * FROM budget_authority_metadata WHERE id = 1"
        ).fetchone()
        if metadata is None or self._validate_authority_metadata(metadata) != self._authority_nonce:
            raise BudgetIntegrityError("budget authority identity changed after audit")
        latest = authority.execute(
            "SELECT sequence, digest FROM budget_authority_heads ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        if latest is None:
            raise BudgetIntegrityError("budget authority has no ledger head")
        latest_sequence = cast("int", latest["sequence"])
        latest_digest = cast("str", latest["digest"])
        if latest_sequence < self._verified_authority_sequence:
            raise BudgetIntegrityError("budget authority was rolled back after audit")
        if (
            latest_sequence == self._verified_authority_sequence
            and latest_digest != self._verified_authority_digest
        ):
            raise BudgetIntegrityError("budget authority head changed after audit")
        if latest_sequence > self._verified_sequence:
            raise BudgetIntegrityError("budget ledger was rolled back behind its authority")

        owns_ledger = ledger is None
        if ledger is None:
            ledger_context = self._connection()
            ledger = ledger_context.__enter__()
        try:
            start = 1 if full_audit else self._verified_authority_sequence + 1
            authority_rows = authority.execute(
                "SELECT sequence, digest FROM budget_authority_heads "
                "WHERE sequence >= ? ORDER BY sequence",
                (start,),
            ).fetchall()
            expected_sequence = start
            for row in authority_rows:
                sequence = cast("int", row["sequence"])
                digest = cast("str", row["digest"])
                if sequence != expected_sequence:
                    raise BudgetIntegrityError("budget authority head sequence is not contiguous")
                ledger_row = ledger.execute(
                    "SELECT digest FROM budget_events WHERE sequence = ?",
                    (sequence,),
                ).fetchone()
                if ledger_row is None or ledger_row["digest"] != digest:
                    raise BudgetIntegrityError("budget ledger was rolled back behind its authority")
                expected_sequence += 1
            if expected_sequence - 1 != latest_sequence:
                raise BudgetIntegrityError("budget authority head sequence is not contiguous")
            missing = ledger.execute(
                "SELECT sequence, digest FROM budget_events WHERE sequence > ? ORDER BY sequence",
                (latest_sequence,),
            ).fetchall()
            expected_sequence = latest_sequence + 1
            for row in missing:
                sequence = cast("int", row["sequence"])
                digest = cast("str", row["digest"])
                if sequence != expected_sequence:
                    raise BudgetIntegrityError("budget ledger head sequence is not contiguous")
                authority.execute(
                    "INSERT INTO budget_authority_heads (sequence, digest) VALUES (?, ?)",
                    (sequence, digest),
                )
                expected_sequence += 1
            if expected_sequence - 1 != self._verified_sequence:
                raise BudgetIntegrityError("budget ledger verified head changed unexpectedly")
            return self._verified_sequence, self._verified_digest
        finally:
            if owns_ledger:
                ledger_context.__exit__(None, None, None)

    def _append_event(
        self,
        connection: sqlite3.Connection,
        action: _BudgetAction,
    ) -> BudgetEvent:
        row = connection.execute(
            "SELECT sequence, digest FROM budget_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        sequence = 1 if row is None else cast("int", row["sequence"]) + 1
        previous_digest = _ZERO_DIGEST if row is None else cast("str", row["digest"])
        recorded_at = self._now()
        if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
            raise ValueError("budget ledger clock must return a timezone-aware datetime")
        unsigned = {
            "sequence": sequence,
            "recorded_at": recorded_at,
            "previous_digest": previous_digest,
            "action": action,
        }
        digest = _digest_json(_event_dump(unsigned))
        event = BudgetEvent.model_validate({**unsigned, "digest": digest})
        connection.execute(
            "INSERT INTO budget_events (sequence, digest, entry_json) VALUES (?, ?, ?)",
            (sequence, digest, _canonical_json(_budget_event_dump(event, include_digest=True))),
        )
        return event

    def _sync_verified_state(self, connection: sqlite3.Connection) -> None:
        """Apply newly committed hash-linked events to authoritative process-local state."""
        head = connection.execute(
            "SELECT sequence, digest FROM budget_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        if head is None:
            raise BudgetIntegrityError("budget ledger has no genesis event")
        head_sequence = cast("int", head["sequence"])
        head_digest = cast("str", head["digest"])
        if head_sequence < self._verified_sequence:
            raise BudgetIntegrityError("budget ledger was rolled back after audit")
        if head_sequence == self._verified_sequence and head_digest != self._verified_digest:
            raise BudgetIntegrityError("budget ledger verified head changed after audit")
        rows = connection.execute(
            "SELECT sequence, entry_json FROM budget_events WHERE sequence > ? ORDER BY sequence",
            (self._verified_sequence,),
        ).fetchall()
        if not rows:
            return
        reconstructed = dict(self._verified_reservations)
        previous = self._verified_digest
        expected_sequence = self._verified_sequence + 1
        touched: set[str] = set()
        for row in rows:
            event = self._parse_event(row["entry_json"], sequence=row["sequence"])
            if event.sequence != expected_sequence:
                raise BudgetIntegrityError(
                    f"budget event sequence gap at {expected_sequence}, found {event.sequence}"
                )
            if event.previous_digest != previous or _event_digest(event) != event.digest:
                raise BudgetIntegrityError(
                    f"budget event {event.sequence} is not linked to verified state"
                )
            _apply_budget_event(self._policy, event, reconstructed)
            action = event.action
            if not isinstance(action, _BudgetOpened):
                touched.add(action.reservation_id)
            expected_sequence += 1
            previous = event.digest
        for reservation_id in touched:
            indexed = self._reservation_from_connection(connection, reservation_id)
            if indexed != reconstructed[reservation_id]:
                raise BudgetIntegrityError(
                    f"budget reservation {reservation_id!r} index differs from verified events"
                )
        _validate_final_snapshot(
            _snapshot_from_reservations(self._policy, list(reconstructed.values())),
            list(reconstructed.values()),
            self._policy,
        )
        self._verified_sequence = expected_sequence - 1
        self._verified_digest = previous
        self._verified_reservations = reconstructed

    def _verified_reservation(
        self,
        connection: sqlite3.Connection,
        reservation_id: str,
    ) -> BudgetReservation:
        reservation = self._verified_reservations.get(reservation_id)
        if reservation is None:
            raise BudgetIntegrityError(f"unknown budget reservation {reservation_id!r}")
        indexed = self._reservation_from_connection(connection, reservation_id)
        if indexed != reservation:
            raise BudgetIntegrityError(
                f"budget reservation {reservation_id!r} index differs from verified events"
            )
        return reservation

    def _reservation(self, reservation_id: str) -> BudgetReservation:
        with self._connection() as connection:
            return self._reservation_from_connection(connection, reservation_id)

    def _reservation_from_connection(
        self,
        connection: sqlite3.Connection,
        reservation_id: str,
    ) -> BudgetReservation:
        row = connection.execute(
            "SELECT * FROM budget_reservations WHERE reservation_id = ?",
            (reservation_id,),
        ).fetchone()
        if row is None:
            raise BudgetIntegrityError(f"unknown budget reservation {reservation_id!r}")
        return self._reservation_from_row(row)

    def _reservation_from_row(self, row: sqlite3.Row) -> BudgetReservation:
        try:
            scope = BudgetScope.model_validate_json(row["scope_json"])
            return BudgetReservation(
                reservation_id=row["reservation_id"],
                scope=scope,
                meter_id=row["meter_id"],
                max_nano_usd=row["max_nano_usd"],
                charged_nano_usd=row["charged_nano_usd"],
                status=row["status"],
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                usage_quantity=row["usage_quantity"],
                usage_unit=row["usage_unit"],
                failure_type=row["failure_type"],
                breach_kind=row["breach_kind"],
            )
        except ValueError as exc:
            raise BudgetIntegrityError("budget reservation index contains malformed data") from exc

    def _parse_event(self, raw: str, *, sequence: int) -> BudgetEvent:
        try:
            event = BudgetEvent.model_validate_json(raw)
        except ValueError as exc:
            raise BudgetIntegrityError(f"budget event {sequence} is malformed") from exc
        if event.sequence != sequence:
            raise BudgetIntegrityError(
                f"budget event {sequence} serialized sequence is {event.sequence}"
            )
        return event

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self._validate_file_identity()
        try:
            if self._allow_create:
                connection = sqlite3.connect(
                    self.path,
                    timeout=_DEFAULT_BUSY_TIMEOUT_MS / 1000,
                )
            else:
                connection = sqlite3.connect(
                    self.path.as_uri() + "?mode=rw",
                    timeout=_DEFAULT_BUSY_TIMEOUT_MS / 1000,
                    uri=True,
                )
        except sqlite3.OperationalError as exc:
            if not self._allow_create:
                raise BudgetIntegrityError(
                    "budget ledger must exist and remain readable after bootstrap"
                ) from exc
            raise
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {_DEFAULT_BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            self._validate_file_identity()
            yield connection
        finally:
            connection.close()

    def _validate_file_identity(self) -> None:
        expected = getattr(self, "_file_identity", None)
        if expected is None:
            return
        try:
            current = self.path.stat(follow_symlinks=False)
        except OSError as exc:
            raise BudgetIntegrityError("budget ledger path disappeared after audit") from exc
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != expected:
            raise BudgetIntegrityError("budget ledger file changed after audit")

    @contextmanager
    def _authority_connection(self) -> Iterator[sqlite3.Connection]:
        expected = getattr(self, "_authority_file_identity", None)
        try:
            current = self._authority_path.stat(follow_symlinks=False)
        except OSError as exc:
            raise BudgetIntegrityError("budget authority path disappeared after audit") from exc
        if not stat.S_ISREG(current.st_mode) or (
            expected is not None and (current.st_dev, current.st_ino) != expected
        ):
            raise BudgetIntegrityError("budget authority file changed after audit")
        try:
            connection = sqlite3.connect(
                self._authority_path.as_uri() + "?mode=rw",
                timeout=_DEFAULT_BUSY_TIMEOUT_MS / 1000,
                uri=True,
            )
        except sqlite3.OperationalError as exc:
            raise BudgetIntegrityError("budget authority must remain readable after audit") from exc
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {_DEFAULT_BUSY_TIMEOUT_MS}")
        try:
            reopened = self._authority_path.stat(follow_symlinks=False)
            if not stat.S_ISREG(reopened.st_mode) or (
                expected is not None and (reopened.st_dev, reopened.st_ino) != expected
            ):
                raise BudgetIntegrityError("budget authority file changed while opening")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _authority_transaction(self) -> Iterator[sqlite3.Connection]:
        with self._authority_connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    @contextmanager
    def _verified_transaction(self) -> Iterator[sqlite3.Connection]:
        with self._state_lock:
            prior_state = (
                self._verified_sequence,
                self._verified_digest,
                {
                    reservation_id: reservation.model_copy(deep=True)
                    for reservation_id, reservation in self._verified_reservations.items()
                },
                self._verified_authority_sequence,
                self._verified_authority_digest,
            )
            try:
                with self._authority_transaction() as authority:
                    with self._transaction() as connection:
                        self._sync_verified_state(connection)
                        verified_authority = self._synchronize_authority(
                            authority,
                            ledger=connection,
                        )
                        yield connection
                        self._sync_verified_state(connection)
                        verified_authority = self._synchronize_authority(
                            authority,
                            ledger=connection,
                        )
            except BaseException:
                (
                    self._verified_sequence,
                    self._verified_digest,
                    self._verified_reservations,
                    self._verified_authority_sequence,
                    self._verified_authority_digest,
                ) = prior_state
                raise
            (
                self._verified_authority_sequence,
                self._verified_authority_digest,
            ) = verified_authority


_SHARED_LEDGER_LOCK = threading.Lock()
_SHARED_LEDGERS: dict[Path, SpendLedger] = {}
_REGISTERED_BUDGET_LOCK = threading.Lock()
_REGISTERED_BUDGETS: dict[str, tuple[Path, BudgetPolicy, str]] = {}


def bootstrap_budget_ledger(
    path: str | Path,
    policy: BudgetPolicy,
) -> BudgetLedgerAuthority:
    """Create one new authoritative ledger and return identity-bound account authority.

    Paid paths never call this function. Reusing an existing path is rejected so initialization
    cannot silently bless a replacement file after a prior cap has been spent.
    """
    frozen_policy = BudgetPolicy.model_validate(policy.model_dump())
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise BudgetIntegrityError("budget ledger path cannot be a symlink")
    requested.parent.mkdir(parents=True, exist_ok=True)
    canonical_path = requested.parent.resolve() / requested.name
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(canonical_path, flags, 0o600)
    except FileExistsError:
        raise BudgetIntegrityError(
            "budget ledger path already exists; load its recorded authority instead"
        ) from None
    except OSError as exc:
        raise BudgetIntegrityError("budget ledger could not be bootstrapped safely") from exc
    else:
        os.close(descriptor)
    ledger = SpendLedger(
        canonical_path,
        frozen_policy,
        allow_create=False,
        allow_authority_create=True,
    )
    with _SHARED_LEDGER_LOCK:
        existing = _SHARED_LEDGERS.get(canonical_path)
        if existing is not None:
            raise BudgetIntegrityError("budget ledger path was already opened in this process")
        _SHARED_LEDGERS[canonical_path] = ledger
    return BudgetLedgerAuthority(
        ledger_path=canonical_path,
        ledger_identity=ledger.ledger_identity,
        policy=frozen_policy,
    )


def open_shared_spend_ledger(
    path: str | Path,
    policy: BudgetPolicy,
    *,
    expected_ledger_identity: str,
) -> SpendLedger:
    """Fully audit one ledger once per process, then reuse its connectionless handle.

    Harbor creates an agent object per trial, but every object runs in the trusted evaluator
    process and the ledger opens a fresh SQLite connection per operation. Process-local reuse
    avoids quadratic full-log replay while every independently started process still performs a
    complete audit before its first paid call.
    """
    if re.fullmatch(_DIGEST_PATTERN, expected_ledger_identity) is None:
        raise BudgetIntegrityError("expected budget ledger identity is malformed")
    frozen_policy = BudgetPolicy.model_validate(policy.model_dump())
    requested = Path(path).expanduser()
    if requested.is_symlink() or not requested.parent.is_dir():
        raise BudgetIntegrityError("budget ledger path is unavailable after bootstrap")
    canonical_path = requested.parent.resolve() / requested.name
    with _SHARED_LEDGER_LOCK:
        existing = _SHARED_LEDGERS.get(canonical_path)
        if existing is not None:
            if existing.policy != frozen_policy:
                raise BudgetIntegrityError("shared budget ledger policy differs from requested")
            existing.audit()
            if existing.ledger_identity != expected_ledger_identity:
                raise BudgetIntegrityError("budget account names a different ledger identity")
            return existing
        ledger = SpendLedger(canonical_path, frozen_policy, allow_create=False)
        if ledger.ledger_identity != expected_ledger_identity:
            raise BudgetIntegrityError("budget account names a different ledger identity")
        _SHARED_LEDGERS[canonical_path] = ledger
        return ledger


def bind_budget_account(account: BudgetAccount) -> BudgetAccountBinding:
    """Register one trusted ledger path and return its path-free durable binding.

    A policy digest may map to only one ledger path in a process. Allowing two files for the same
    frozen cap would let concurrent evaluators fork the budget and overspend it independently.
    """
    validated = BudgetAccount.model_validate(account.model_dump())
    requested = validated.ledger_path.expanduser()
    canonical_candidate = requested.parent.resolve() / requested.name
    policy_digest = validated.policy.policy_digest
    with _REGISTERED_BUDGET_LOCK:
        existing = _REGISTERED_BUDGETS.get(policy_digest)
        if existing is not None and existing[0] != canonical_candidate:
            raise BudgetIntegrityError(
                "budget policy digest is already registered to a different ledger path"
            )
        if existing is not None and existing[2] != validated.ledger_identity:
            raise BudgetIntegrityError(
                "budget policy digest is already registered to a different ledger identity"
            )
    ledger = open_shared_spend_ledger(
        validated.ledger_path,
        validated.policy,
        expected_ledger_identity=validated.ledger_identity,
    )
    canonical_path = ledger.path
    with _REGISTERED_BUDGET_LOCK:
        existing = _REGISTERED_BUDGETS.get(policy_digest)
        if existing is not None and existing[0] != canonical_path:
            raise BudgetIntegrityError(
                "budget policy digest is already registered to a different ledger path"
            )
        _REGISTERED_BUDGETS[policy_digest] = (
            canonical_path,
            validated.policy.model_copy(deep=True),
            ledger.ledger_identity,
        )
    return BudgetAccountBinding(
        policy_digest=policy_digest,
        ledger_identity=ledger.ledger_identity,
        scope=validated.scope,
        meter_id=validated.meter_id,
    )


def bind_timed_resource_account(
    account: TimedResourceBudgetAccount,
) -> BudgetAccountBinding:
    """Register one timed-resource ledger and return its path-free durable binding."""
    validated = TimedResourceBudgetAccount.model_validate(account.model_dump())
    requested = validated.ledger_path.expanduser()
    canonical_candidate = requested.parent.resolve() / requested.name
    policy_digest = validated.policy.policy_digest
    with _REGISTERED_BUDGET_LOCK:
        existing = _REGISTERED_BUDGETS.get(policy_digest)
        if existing is not None and existing[0] != canonical_candidate:
            raise BudgetIntegrityError(
                "budget policy digest is already registered to a different ledger path"
            )
        if existing is not None and existing[2] != validated.ledger_identity:
            raise BudgetIntegrityError(
                "budget policy digest is already registered to a different ledger identity"
            )
    ledger = open_shared_spend_ledger(
        validated.ledger_path,
        validated.policy,
        expected_ledger_identity=validated.ledger_identity,
    )
    canonical_path = ledger.path
    with _REGISTERED_BUDGET_LOCK:
        existing = _REGISTERED_BUDGETS.get(policy_digest)
        if existing is not None and existing[0] != canonical_path:
            raise BudgetIntegrityError(
                "budget policy digest is already registered to a different ledger path"
            )
        _REGISTERED_BUDGETS[policy_digest] = (
            canonical_path,
            validated.policy.model_copy(deep=True),
            ledger.ledger_identity,
        )
    return BudgetAccountBinding(
        policy_digest=policy_digest,
        ledger_identity=ledger.ledger_identity,
        scope=validated.scope,
        meter_id=validated.meter_id,
    )


def resolve_budget_account(binding: BudgetAccountBinding) -> BudgetAccount:
    """Resolve a path-free binding only inside the trusted registered evaluator process."""
    validated = BudgetAccountBinding.model_validate(binding.model_dump())
    with _REGISTERED_BUDGET_LOCK:
        registered = _REGISTERED_BUDGETS.get(validated.policy_digest)
        if registered is None:
            raise BudgetIntegrityError("budget policy digest is not registered in this process")
        ledger_path, policy, ledger_identity = registered
        resolved_policy = policy.model_copy(deep=True)
    if validated.ledger_identity != ledger_identity:
        raise BudgetIntegrityError("budget binding differs from the registered ledger identity")
    if validated.scope.phase not in resolved_policy.phase_limits_nano_usd:
        raise BudgetIntegrityError("budget binding phase is absent from its registered policy")
    if validated.meter_id not in resolved_policy.meters:
        raise BudgetIntegrityError("budget binding meter is absent from its registered policy")
    if not isinstance(resolved_policy.meters[validated.meter_id], ProviderCostMeter):
        raise BudgetIntegrityError("budget binding does not name a provider token meter")
    return BudgetAccount(
        ledger_path=ledger_path,
        ledger_identity=validated.ledger_identity,
        policy=resolved_policy,
        scope=validated.scope,
        meter_id=validated.meter_id,
    )


def resolve_timed_resource_account(
    binding: BudgetAccountBinding,
) -> TimedResourceBudgetAccount:
    """Resolve a timed-resource binding inside the trusted registered process."""
    validated = BudgetAccountBinding.model_validate(binding.model_dump())
    with _REGISTERED_BUDGET_LOCK:
        registered = _REGISTERED_BUDGETS.get(validated.policy_digest)
        if registered is None:
            raise BudgetIntegrityError("budget policy digest is not registered in this process")
        ledger_path, policy, ledger_identity = registered
        resolved_policy = policy.model_copy(deep=True)
    if validated.ledger_identity != ledger_identity:
        raise BudgetIntegrityError("budget binding differs from the registered ledger identity")
    if validated.scope.phase not in resolved_policy.phase_limits_nano_usd:
        raise BudgetIntegrityError("budget binding phase is absent from its registered policy")
    if not isinstance(resolved_policy.meters.get(validated.meter_id), TimedResourceCostMeter):
        raise BudgetIntegrityError("budget binding does not name a timed resource meter")
    return TimedResourceBudgetAccount(
        ledger_path=ledger_path,
        ledger_identity=validated.ledger_identity,
        policy=resolved_policy,
        scope=validated.scope,
        meter_id=validated.meter_id,
    )


def validate_timed_resource_class(
    account: TimedResourceBudgetAccount,
    resource_class: TimedResourceClass,
) -> TimedResourceCostMeter:
    """Require one timed meter to name the exact class and cover its host horizon."""
    validated = TimedResourceBudgetAccount.model_validate(account.model_dump())
    frozen_class = TimedResourceClass.model_validate(resource_class.model_dump())
    meter = validated.policy.meters[validated.meter_id]
    if not isinstance(meter, TimedResourceCostMeter):  # account validation is defense in depth
        raise BudgetIntegrityError("timed resource account does not name a timed meter")
    if meter.resource_type != frozen_class.role.value:
        raise BudgetIntegrityError("timed resource meter role differs from the external resource")
    if meter.resource_class_digest != frozen_class.digest:
        raise BudgetIntegrityError("timed resource meter class differs from the external resource")
    if meter.max_billing_seconds < frozen_class.max_host_observation_seconds:
        raise BudgetIntegrityError(
            "timed resource meter horizon does not cover provider TTL and request cleanup"
        )
    return meter


def reconcile_orphaned_timed_resource(
    account: TimedResourceBudgetAccount,
    *,
    reservation_id: str,
) -> BudgetReservation | None:
    """Conservatively terminalize one prior lease reservation after absence is proved.

    A process can die after cleanup proof or after a full-ceiling forfeit but before its resource
    ledger is retired. Those terminal states are safe to replay. An open reservation has lost its
    monotonic start time, so reconciliation forfeits its full ceiling rather than inventing usage.
    """
    validated = TimedResourceBudgetAccount.model_validate(account.model_dump())
    ledger = open_shared_spend_ledger(
        validated.ledger_path,
        validated.policy,
        expected_ledger_identity=validated.ledger_identity,
    )
    matches = [item for item in ledger.reservations() if item.reservation_id == reservation_id]
    if not matches:
        # Resource ownership is durably claimed before budget admission. A crash in that narrow
        # pre-reservation window cannot have dispatched create, and provider absence was proved by
        # the caller before reaching this join.
        return None
    if len(matches) != 1:
        raise BudgetIntegrityError(
            "resource lease does not have exactly one matching budget reservation"
        )
    reservation = matches[0]
    if reservation.scope != validated.scope or reservation.meter_id != validated.meter_id:
        raise BudgetIntegrityError("resource lease budget reservation has the wrong account")
    if reservation.status is ReservationStatus.RESERVED:
        return ledger.forfeit(reservation_id, failure_type="OrphanedLease")
    return reservation


def orphaned_timed_resource_requires_reap(
    account: TimedResourceBudgetAccount,
    *,
    reservation_id: str,
) -> bool:
    """Join an orphan reservation and report whether provider cleanup remains unproved."""
    reservation = reconcile_orphaned_timed_resource(
        account,
        reservation_id=reservation_id,
    )
    if reservation is None:
        return False
    # Settlement and breach are written only after known-resource cleanup proof. A forfeit is
    # intentionally ambiguous and must still be reaped or reach its immutable provider TTL.
    return reservation.status is ReservationStatus.FORFEITED


class BudgetedProvider:
    """Reserve a conservative ceiling around every provider completion call."""

    def __init__(
        self,
        provider: Provider,
        account: BudgetAccount,
        *,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        if (
            not isinstance(provider, ToolCallingProvider)
            or getattr(provider, "paid_request_attempts", None) != 1
        ):
            raise TypeError(
                "BudgetedProvider requires a single-dispatch tool-calling provider with "
                "SDK retries and fallback disabled"
            )
        validated_account = BudgetAccount.model_validate(account.model_dump())
        meter = validated_account.policy.meters[validated_account.meter_id]
        if not isinstance(meter, ProviderCostMeter):
            raise TypeError("BudgetedProvider requires a provider token meter")
        if provider.config != meter.provider_config:
            raise ValueError("budget account provider config differs from the wrapped provider")
        self._provider = cast("SingleDispatchProvider", provider)
        self._account = validated_account
        self._meter = meter
        self._ledger = open_shared_spend_ledger(
            self._account.ledger_path,
            self._account.policy,
            expected_ledger_identity=self._account.ledger_identity,
        )
        self._id_factory = id_factory or (lambda: str(uuid4()))

    @property
    def config(self) -> ProviderConfig:
        return self._provider.config

    @property
    def budget_ledger_identity(self) -> str:
        """Return the durable authority shared by every paid surface in one experiment."""
        return self._ledger.ledger_identity

    @property
    def budget_policy_digest(self) -> str:
        """Return the nonsecret policy identity shared with resource budgets."""
        return self._account.policy.policy_digest

    @property
    def budget_ledger_path(self) -> Path:
        """Return the host-private canonical ledger path for trusted in-process joins."""
        return self._ledger.path

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        if max_tokens < 1:
            raise ValueError("provider output-token limit must be positive")
        payload = {
            "system": system,
            "messages": [message.model_dump(mode="json") for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        reservation_id, input_ceiling = self._reserve(
            cast("JsonValue", payload),
            max_output_tokens=max_tokens,
        )
        try:
            completion = self._provider.complete(
                system,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as error:
            self._forfeit_after_error(reservation_id, error)
            raise
        if "usage" not in completion.model_fields_set or not {
            "input_tokens",
            "output_tokens",
        }.issubset(completion.usage.model_fields_set):
            self._ledger.forfeit(reservation_id, failure_type="UsageUnavailable")
            return completion
        self._reject_unpriced_usage(
            reservation_id,
            completion.usage,
            input_field="input_tokens",
            output_field="output_tokens",
        )
        self._settle_usage(
            reservation_id,
            input_tokens=completion.usage.input_tokens,
            output_tokens=completion.usage.output_tokens,
            input_ceiling=input_ceiling,
            output_ceiling=max_tokens,
        )
        return completion

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Budget one structured call using its provider-routed output limit."""
        if request.model_extra:
            extras = ", ".join(sorted(request.model_extra))
            raise ValueError("budgeted chat requests reject unquoted provider fields: " + extras)
        _require_text_only_chat(request)
        output_ceiling = _chat_output_limit(request)
        reservation_id, input_ceiling = self._reserve(
            request.model_dump(mode="json", exclude_none=True),
            max_output_tokens=output_ceiling,
        )
        try:
            response = self._provider.complete_chat(request)
        except Exception as error:
            self._forfeit_after_error(reservation_id, error)
            raise
        if response.usage is None or not {
            "prompt_tokens",
            "completion_tokens",
        }.issubset(response.usage.model_fields_set):
            self._ledger.forfeit(reservation_id, failure_type="UsageUnavailable")
            return response
        self._reject_unpriced_usage(
            reservation_id,
            response.usage,
            input_field="prompt_tokens",
            output_field="completion_tokens",
        )
        self._settle_usage(
            reservation_id,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            input_ceiling=input_ceiling,
            output_ceiling=output_ceiling,
        )
        return response

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Reject unpriced embeddings instead of allowing an unreserved paid call."""
        del texts
        raise RuntimeError("embeddings are disabled for this budgeted provider account")

    def verify(self) -> VerifyResult:
        """Run the ordinary reachability ping through this same budget boundary."""
        return verify_via_ping(self)

    def _reserve(
        self,
        payload: JsonValue,
        *,
        max_output_tokens: int,
    ) -> tuple[str, int]:
        canonical = _canonical_json(payload)
        # Every model token consumes at least one encoded byte. The fixed overhead covers chat
        # framing and provider-added special tokens, so bytes plus overhead is a conservative
        # upper bound without relying on a provider-specific tokenizer.
        input_ceiling = len(canonical.encode("utf-8")) + self._meter.input_overhead_tokens
        max_nano_usd = self._meter.price.charge(
            input_tokens=input_ceiling,
            output_tokens=max_output_tokens,
        )
        reservation_id = self._id_factory()
        self._ledger.reserve(
            self._account.scope,
            meter_id=self._account.meter_id,
            max_nano_usd=max_nano_usd,
            reservation_id=reservation_id,
        )
        return reservation_id, input_ceiling

    def _settle_usage(
        self,
        reservation_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        input_ceiling: int,
        output_ceiling: int,
    ) -> None:
        breach_kind = None
        if input_tokens > input_ceiling:
            breach_kind = BudgetBreachKind.INPUT_TOKEN_CEILING
        elif output_tokens > output_ceiling:
            breach_kind = BudgetBreachKind.OUTPUT_TOKEN_CEILING
        charged = self._meter.price.charge(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self._ledger.settle(
            reservation_id,
            charged_nano_usd=charged,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            breach_kind=breach_kind,
        )

    def _reject_unpriced_usage(
        self,
        reservation_id: str,
        usage: BaseModel,
        *,
        input_field: str,
        output_field: str,
    ) -> None:
        """Forfeit and stop when a response exposes dimensions absent from the tariff."""
        # OpenAI's typed SDK models predeclare these detail objects and serialize an omitted
        # object as ``None``. A null container reports no usage dimension, so it is safe to drop.
        # Keep every populated detail object and every unknown field, including unknown nulls,
        # so a new or semantically present billing dimension still fails closed.
        # SDK upgrades that add null placeholder fields require explicit review before allowlisting.
        extras = {
            name: value
            for name, value in (usage.model_extra or {}).items()
            if not (name in _KNOWN_NULL_USAGE_DETAIL_PLACEHOLDERS and value is None)
        }
        total_present = "total_tokens" in extras
        total = extras.pop("total_tokens", None)
        input_tokens = getattr(usage, input_field)
        output_tokens = getattr(usage, output_field)
        if total_present and (
            isinstance(total, bool)
            or not isinstance(total, int)
            or total != input_tokens + output_tokens
        ):
            extras["total_tokens"] = total
        if not extras:
            return
        dimensions = ", ".join(sorted(extras))
        self._ledger.forfeit(reservation_id, failure_type="UnpricedUsage")
        raise UnpricedProviderUsageError(
            f"provider usage exposes unpriced dimension(s): {dimensions}"
        )

    def _forfeit_after_error(self, reservation_id: str, error: Exception) -> None:
        failure_type = type(error).__name__
        if _FAILURE_CODE_PATTERN.fullmatch(failure_type) is None:
            failure_type = "UnclassifiedError"
        try:
            self._ledger.forfeit(reservation_id, failure_type=failure_type)
        except Exception as ledger_error:
            raise ledger_error from error


class TimedResourceBudget:
    """Reserve a worst-case ceiling before one externally billed resource is created."""

    def __init__(
        self,
        account: TimedResourceBudgetAccount,
        *,
        resource_class: TimedResourceClass | None = None,
        id_factory: Callable[[], str] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        validated = TimedResourceBudgetAccount.model_validate(account.model_dump())
        meter = validated.policy.meters[validated.meter_id]
        if not isinstance(meter, TimedResourceCostMeter):
            raise TypeError("TimedResourceBudget requires a timed resource meter")
        if resource_class is not None:
            validate_timed_resource_class(validated, resource_class)
        self._account = validated
        self._meter = meter
        self._ledger = open_shared_spend_ledger(
            validated.ledger_path,
            validated.policy,
            expected_ledger_identity=validated.ledger_identity,
        )
        self._id_factory = id_factory or (lambda: str(uuid4()))
        self._monotonic = monotonic or _system_monotonic

    def reserve(self) -> TimedResourceReservation:
        """Atomically admit one resource lease before its create call."""
        reservation_id = self._id_factory()
        self._ledger.reserve(
            self._account.scope,
            meter_id=self._account.meter_id,
            max_nano_usd=self._meter.maximum_charge_nano_usd(),
            reservation_id=reservation_id,
        )
        started_at_s = self._monotonic()
        return TimedResourceReservation(
            ledger=self._ledger,
            meter=self._meter,
            reservation_id=reservation_id,
            started_at_s=started_at_s,
            monotonic=self._monotonic,
        )


class TimedResourceReservation:
    """Exactly-once terminal handle for a previously admitted timed resource."""

    def __init__(
        self,
        *,
        ledger: SpendLedger,
        meter: TimedResourceCostMeter,
        reservation_id: str,
        started_at_s: float,
        monotonic: Callable[[], float],
    ) -> None:
        self._ledger = ledger
        self._meter = meter
        self._reservation_id = reservation_id
        self._started_at_s = started_at_s
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._terminal = False
        self._settlement_usage: tuple[int, int] | None = None

    @property
    def reservation_id(self) -> str:
        return self._reservation_id

    def settle(self) -> BudgetReservation:
        """Settle a proved-cleaned lease using host-observed conservative duration."""
        with self._lock:
            if self._terminal:
                raise BudgetIntegrityError("timed resource reservation is already terminal")
            if self._settlement_usage is None:
                billed_seconds = self._meter.billed_seconds(self._monotonic() - self._started_at_s)
                charged = self._meter.charge_nano_usd(billed_seconds=billed_seconds)
                self._settlement_usage = (billed_seconds, charged)
            else:
                billed_seconds, charged = self._settlement_usage
            try:
                settled = self._ledger.settle(
                    self._reservation_id,
                    charged_nano_usd=charged,
                    usage_quantity=billed_seconds,
                    usage_unit="billing_second",
                )
            except Exception:
                self._reconcile_terminal_state()
                raise
            self._terminal = True
            return settled

    def forfeit(self, *, failure_type: str) -> BudgetReservation:
        """Charge the full ceiling when create or cleanup cost cannot be proved."""
        with self._lock:
            if self._terminal:
                raise BudgetIntegrityError("timed resource reservation is already terminal")
            try:
                forfeited = self._ledger.forfeit(
                    self._reservation_id,
                    failure_type=failure_type,
                )
            except Exception:
                self._reconcile_terminal_state()
                raise
            self._terminal = True
            return forfeited

    def _reconcile_terminal_state(self) -> None:
        """Permit retry only when the failed write is still durably open."""
        try:
            reservation = next(
                item
                for item in self._ledger.reservations()
                if item.reservation_id == self._reservation_id
            )
        except (BudgetIntegrityError, OSError, sqlite3.Error, StopIteration):
            return
        self._terminal = reservation.status is not ReservationStatus.RESERVED


def _chat_output_limit(request: ChatRequest) -> int:
    # Provider translators do not all prioritize the two compatibility fields identically. The
    # hard ceiling must dominate either dispatch path, including requests that set both fields.
    limits = [
        limit for limit in (request.max_tokens, request.max_completion_tokens) if limit is not None
    ]
    if not limits or min(limits) < 1:
        raise ValueError("budgeted chat request requires a positive output-token limit")
    return max(limits)


def _require_text_only_chat(request: ChatRequest) -> None:
    """Reject content whose provider charge is not bounded by its serialized text bytes."""
    for index, message in enumerate(request.messages):
        content = message.content
        if content is None or isinstance(content, str):
            continue
        if isinstance(content, list) and all(_is_text_content_part(item) for item in content):
            continue
        raise ValueError(
            f"budgeted chat requests reject unpriced non-text content in message {index}"
        )


def _is_text_content_part(value: JsonValue) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"type", "text"}
        and value.get("type") == "text"
        and isinstance(value.get("text"), str)
    )


def nano_usd_from_usd(value: Decimal | str | int) -> int:
    """Convert a nonnegative USD amount to nano-USD, rounding upward."""
    amount = _decimal_currency(value)
    if amount < 0:
        raise ValueError("USD amount cannot be negative")
    result = int((amount * _NANO_USD_PER_USD).to_integral_value(rounding=ROUND_CEILING))
    if result > _SQLITE_INTEGER_MAX:
        raise OverflowError("USD amount does not fit a SQLite integer in nano-USD")
    return result


def _nano_usd_per_token(value: Decimal | str | int) -> int:
    amount = _decimal_currency(value)
    if amount < 0:
        raise ValueError("token price cannot be negative")
    nano_per_token = amount * _NANO_USD_PER_USD / _TOKENS_PER_MILLION
    result = int(nano_per_token.to_integral_value(rounding=ROUND_CEILING))
    if result > _SQLITE_INTEGER_MAX:
        raise OverflowError("token price does not fit a SQLite integer in nano-USD")
    return result


def _decimal_currency(value: Decimal | str | int) -> Decimal:
    if isinstance(value, float):
        raise TypeError("currency inputs cannot use floating-point values")
    try:
        amount = Decimal(value)
    except ArithmeticError as exc:
        raise ValueError(f"invalid currency value {value!r}") from exc
    if not amount.is_finite():
        raise ValueError("currency values must be finite")
    return amount


def _decimal_duration(value: Decimal | str | int | float) -> Decimal:
    try:
        duration = Decimal(str(value)) if isinstance(value, float) else Decimal(value)
    except ArithmeticError as exc:
        raise ValueError(f"invalid resource duration {value!r}") from exc
    if not duration.is_finite():
        raise ValueError("resource duration must be finite")
    return duration


def _snapshot_from_reservations(
    policy: BudgetPolicy,
    reservations: list[BudgetReservation],
) -> BudgetSnapshot:
    charged = 0
    reserved = 0
    by_phase = {phase: 0 for phase in policy.phase_limits_nano_usd}
    breached = False
    for reservation in reservations:
        if reservation.scope.phase not in by_phase:
            raise BudgetIntegrityError(
                f"reservation uses unknown budget phase {reservation.scope.phase!r}"
            )
        if reservation.status is ReservationStatus.RESERVED:
            amount = reservation.max_nano_usd
            reserved += amount
        else:
            amount = reservation.charged_nano_usd
            charged += amount
        by_phase[reservation.scope.phase] += amount
        breached = breached or reservation.status is ReservationStatus.BREACHED
    exposure = charged + reserved
    if exposure > policy.hard_limit_nano_usd:
        breached = True
    if any(by_phase[phase] > limit for phase, limit in policy.phase_limits_nano_usd.items()):
        breached = True
    return BudgetSnapshot(
        hard_limit_nano_usd=policy.hard_limit_nano_usd,
        charged_nano_usd=charged,
        reserved_nano_usd=reserved,
        remaining_nano_usd=max(policy.hard_limit_nano_usd - exposure, 0),
        by_phase_nano_usd=by_phase,
        breached=breached,
    )


def _validate_settlement_event(
    action: _BudgetSettled,
    reservation: BudgetReservation,
    policy: BudgetPolicy,
) -> None:
    meter = policy.meters[reservation.meter_id]
    try:
        if isinstance(meter, ProviderCostMeter):
            if action.input_tokens is None or action.output_tokens is None:
                raise BudgetIntegrityError("provider meter settlement lacks token usage")
            expected_charge = meter.price.charge(
                input_tokens=action.input_tokens,
                output_tokens=action.output_tokens,
            )
        else:
            if action.usage_unit != "billing_second" or action.usage_quantity is None:
                raise BudgetIntegrityError("timed resource settlement lacks billed-second usage")
            if action.breach_kind in {
                BudgetBreachKind.INPUT_TOKEN_CEILING,
                BudgetBreachKind.OUTPUT_TOKEN_CEILING,
            }:
                raise BudgetIntegrityError("timed resource settlement uses a token breach kind")
            if action.usage_quantity < meter.billing_quantum_seconds:
                raise BudgetIntegrityError(
                    "timed resource settlement must include at least one billing quantum"
                )
            if action.usage_quantity % meter.billing_quantum_seconds:
                raise BudgetIntegrityError(
                    "timed resource settlement is not aligned to its billing quantum"
                )
            expected_charge = meter.charge_nano_usd(billed_seconds=action.usage_quantity)
    except (OverflowError, ValueError) as exc:
        raise BudgetIntegrityError(
            f"settlement {action.reservation_id!r} cannot be priced by its frozen tariff"
        ) from exc
    if action.charged_nano_usd != expected_charge:
        raise BudgetIntegrityError(
            f"settlement {action.reservation_id!r} differs from its frozen tariff"
        )
    exceeds_reservation = action.charged_nano_usd > reservation.max_nano_usd
    if exceeds_reservation:
        if (
            action.status is not ReservationStatus.BREACHED
            or action.breach_kind is not BudgetBreachKind.RESERVATION
        ):
            raise BudgetIntegrityError(
                f"settlement {action.reservation_id!r} exceeded its reservation without "
                "a reservation breach"
            )
    elif action.breach_kind is BudgetBreachKind.RESERVATION:
        raise BudgetIntegrityError(
            f"settlement {action.reservation_id!r} records a false reservation breach"
        )


def _validate_final_snapshot(
    snapshot: BudgetSnapshot,
    reservations: list[BudgetReservation],
    policy: BudgetPolicy,
) -> None:
    explicit_breach = any(
        reservation.status is ReservationStatus.BREACHED for reservation in reservations
    )
    hard_overrun = (
        snapshot.charged_nano_usd + snapshot.reserved_nano_usd > snapshot.hard_limit_nano_usd
    )
    phase_overrun = any(
        amount > policy.phase_limits_nano_usd[phase]
        for phase, amount in snapshot.by_phase_nano_usd.items()
    )
    if (hard_overrun or phase_overrun) and not explicit_breach:
        raise BudgetIntegrityError("budget caps were exceeded without an explicit breach event")
    if snapshot.breached != (explicit_breach or hard_overrun or phase_overrun):
        raise BudgetIntegrityError("budget breach snapshot is inconsistent with ledger events")


def _apply_budget_event(
    policy: BudgetPolicy,
    event: BudgetEvent,
    reconstructed: dict[str, BudgetReservation],
) -> None:
    action = event.action
    if event.sequence == 1:
        if not isinstance(action, _BudgetOpened):
            raise BudgetIntegrityError("first budget event is not the policy genesis")
        if action.policy != policy:
            raise BudgetIntegrityError("budget genesis policy differs from requested policy")
        return
    if isinstance(action, _BudgetOpened):
        raise BudgetIntegrityError("budget policy genesis appears more than once")
    if isinstance(action, _BudgetReserved):
        if action.reservation_id in reconstructed:
            raise BudgetIntegrityError(f"duplicate reservation event {action.reservation_id!r}")
        if action.scope.phase not in policy.phase_limits_nano_usd:
            raise BudgetIntegrityError(
                f"reservation {action.reservation_id!r} uses an unknown budget phase"
            )
        if action.meter_id not in policy.meters:
            raise BudgetIntegrityError(
                f"reservation {action.reservation_id!r} uses an unknown budget meter"
            )
        meter = policy.meters[action.meter_id]
        if (
            isinstance(meter, TimedResourceCostMeter)
            and action.max_nano_usd != meter.maximum_charge_nano_usd()
        ):
            raise BudgetIntegrityError(
                f"timed resource reservation {action.reservation_id!r} did not use its exact "
                "maximum charge"
            )
        before = _snapshot_from_reservations(policy, list(reconstructed.values()))
        if before.breached:
            raise BudgetIntegrityError("reservation was admitted after the ledger breached")
        if action.max_nano_usd > before.remaining_nano_usd:
            raise BudgetIntegrityError(
                f"reservation {action.reservation_id!r} exceeded the remaining hard budget"
            )
        phase_remaining = (
            policy.phase_limits_nano_usd[action.scope.phase]
            - before.by_phase_nano_usd[action.scope.phase]
        )
        if action.max_nano_usd > phase_remaining:
            raise BudgetIntegrityError(
                f"reservation {action.reservation_id!r} exceeded its phase budget"
            )
        reconstructed[action.reservation_id] = BudgetReservation(
            reservation_id=action.reservation_id,
            scope=action.scope,
            meter_id=action.meter_id,
            max_nano_usd=action.max_nano_usd,
            charged_nano_usd=0,
            status=ReservationStatus.RESERVED,
        )
        return
    if isinstance(action, _BudgetSettled):
        prior = _open_reconstructed(reconstructed, action.reservation_id)
        _validate_settlement_event(action, prior, policy)
        reconstructed[action.reservation_id] = prior.model_copy(
            update={
                "charged_nano_usd": action.charged_nano_usd,
                "status": action.status,
                "input_tokens": action.input_tokens,
                "output_tokens": action.output_tokens,
                "usage_quantity": action.usage_quantity,
                "usage_unit": action.usage_unit,
                "breach_kind": action.breach_kind,
            }
        )
        return
    assert isinstance(action, _BudgetForfeited)
    prior = _open_reconstructed(reconstructed, action.reservation_id)
    if action.charged_nano_usd != prior.max_nano_usd:
        raise BudgetIntegrityError(
            f"forfeit {action.reservation_id!r} did not charge its full ceiling"
        )
    reconstructed[action.reservation_id] = prior.model_copy(
        update={
            "charged_nano_usd": action.charged_nano_usd,
            "status": ReservationStatus.FORFEITED,
            "failure_type": action.failure_type,
        }
    )


def _open_reconstructed(
    reservations: dict[str, BudgetReservation],
    reservation_id: str,
) -> BudgetReservation:
    reservation = reservations.get(reservation_id)
    if reservation is None:
        raise BudgetIntegrityError(f"terminal event references unknown {reservation_id!r}")
    if reservation.status is not ReservationStatus.RESERVED:
        raise BudgetIntegrityError(f"reservation {reservation_id!r} has two terminal events")
    return reservation


def _event_digest(event: BudgetEvent) -> str:
    return _digest_json(_budget_event_dump(event, include_digest=False))


def _event_dump(value: Mapping[str, object]) -> dict[str, object]:
    event = BudgetEvent.model_validate({**value, "digest": _ZERO_DIGEST})
    return cast("dict[str, object]", _budget_event_dump(event, include_digest=False))


def _budget_event_dump(
    event: BudgetEvent,
    *,
    include_digest: bool,
) -> dict[str, JsonValue]:
    """Serialize an event while preserving legacy unbound timed-meter bytes."""
    payload = cast("dict[str, JsonValue]", event.model_dump(mode="json"))
    if not include_digest:
        payload.pop("digest")
    if isinstance(event.action, _BudgetOpened):
        raw_action = payload.get("action")
        if not isinstance(raw_action, dict):
            raise BudgetIntegrityError("budget opened event serialization is malformed")
        action = dict(raw_action)
        action["policy"] = _budget_policy_dump(event.action.policy)
        payload["action"] = action
    elif isinstance(event.action, _BudgetSettled) and event.action.terminal_provenance is None:
        raw_action = payload.get("action")
        if not isinstance(raw_action, dict):
            raise BudgetIntegrityError("budget settled event serialization is malformed")
        action = dict(raw_action)
        action.pop("terminal_provenance", None)
        payload["action"] = action
    return payload


def _budget_policy_dump(policy: BudgetPolicy) -> dict[str, JsonValue]:
    """Omit only an unbound new authority field from otherwise exact policy JSON."""
    payload = cast("dict[str, JsonValue]", policy.model_dump(mode="json"))
    raw_meters = payload.get("meters")
    if not isinstance(raw_meters, dict):
        raise BudgetIntegrityError("budget policy meter serialization is malformed")
    meters: dict[str, JsonValue] = {}
    for meter_id, raw_meter in raw_meters.items():
        if not isinstance(raw_meter, dict):
            raise BudgetIntegrityError("budget policy meter serialization is malformed")
        meter = dict(raw_meter)
        if meter.get("kind") == "timed_resource" and meter.get("external_spend_authority") is None:
            meter.pop("external_spend_authority", None)
        meters[meter_id] = meter
    payload["meters"] = meters
    return payload


def _digest_json(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _authority_path_for_ledger(path: Path) -> Path:
    return path.with_name(f".{path.name}.authority.sqlite3")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS budget_metadata (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    schema_version INTEGER NOT NULL,
    ledger_nonce TEXT NOT NULL CHECK (length(ledger_nonce) = 64),
    policy_digest TEXT NOT NULL,
    policy_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS budget_events (
    sequence INTEGER PRIMARY KEY,
    digest TEXT NOT NULL UNIQUE,
    entry_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS budget_reservations (
    reservation_id TEXT PRIMARY KEY,
    scope_json TEXT NOT NULL,
    meter_id TEXT NOT NULL,
    max_nano_usd INTEGER NOT NULL CHECK (max_nano_usd >= 0),
    charged_nano_usd INTEGER NOT NULL CHECK (charged_nano_usd >= 0),
    status TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    usage_quantity INTEGER,
    usage_unit TEXT,
    failure_type TEXT,
    breach_kind TEXT
);

CREATE TRIGGER IF NOT EXISTS budget_events_no_update
BEFORE UPDATE ON budget_events
BEGIN
    SELECT RAISE(ABORT, 'budget events are append-only');
END;

CREATE TRIGGER IF NOT EXISTS budget_events_no_delete
BEFORE DELETE ON budget_events
BEGIN
    SELECT RAISE(ABORT, 'budget events are append-only');
END;

CREATE TRIGGER IF NOT EXISTS budget_metadata_no_update
BEFORE UPDATE ON budget_metadata
BEGIN
    SELECT RAISE(ABORT, 'budget metadata is immutable');
END;

CREATE TRIGGER IF NOT EXISTS budget_metadata_no_delete
BEFORE DELETE ON budget_metadata
BEGIN
    SELECT RAISE(ABORT, 'budget metadata is immutable');
END;
"""


_AUTHORITY_SCHEMA = """
CREATE TABLE IF NOT EXISTS budget_authority_metadata (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    schema_version INTEGER NOT NULL,
    authority_nonce TEXT NOT NULL CHECK (length(authority_nonce) = 64),
    ledger_nonce TEXT NOT NULL CHECK (length(ledger_nonce) = 64),
    canonical_ledger_path TEXT NOT NULL,
    policy_digest TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS budget_authority_heads (
    sequence INTEGER PRIMARY KEY CHECK (sequence >= 1),
    digest TEXT NOT NULL UNIQUE
);

CREATE TRIGGER IF NOT EXISTS budget_authority_metadata_no_update
BEFORE UPDATE ON budget_authority_metadata
BEGIN
    SELECT RAISE(ABORT, 'budget authority metadata is immutable');
END;

CREATE TRIGGER IF NOT EXISTS budget_authority_metadata_no_delete
BEFORE DELETE ON budget_authority_metadata
BEGIN
    SELECT RAISE(ABORT, 'budget authority metadata is immutable');
END;

CREATE TRIGGER IF NOT EXISTS budget_authority_heads_no_update
BEFORE UPDATE ON budget_authority_heads
BEGIN
    SELECT RAISE(ABORT, 'budget authority heads are append-only');
END;

CREATE TRIGGER IF NOT EXISTS budget_authority_heads_no_delete
BEFORE DELETE ON budget_authority_heads
BEGIN
    SELECT RAISE(ABORT, 'budget authority heads are append-only');
END;
"""
