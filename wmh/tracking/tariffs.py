"""Audited provider tariff snapshots and exact hard-budget meter construction."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from decimal import Decimal
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.models import resolve_provider_model
from wmh.tracking.budget import (
    ProviderCostMeter,
    ProviderTariffProvenance,
    ProviderTariffRoute,
    TokenPriceCeiling,
)

_AWS_BEDROCK_PRICING = "https://aws.amazon.com/bedrock/pricing/"
_CATALOG_VERIFIED_ON = date(2026, 7, 19)
_CATALOG_EFFECTIVE_ON = date(2026, 7, 19)
_CATALOG_CURRENCY = "USD"
_CATALOG_PRICE_UNIT = "per_1m_tokens"


class _BedrockCatalogRecord(BaseModel):
    """One normalized row retained in the immutable built-in source snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_type: str
    model: str
    billing_sku: str
    input_usd: str
    output_usd: str


_BEDROCK_CATALOG_RECORDS = (
    _BedrockCatalogRecord(
        model_type="claude-haiku-4-5",
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        billing_sku="geo-cross-region",
        input_usd="1.1",
        output_usd="5.5",
    ),
    _BedrockCatalogRecord(
        model_type="glm-5",
        model="zai.glm-5",
        billing_sku="on-demand",
        input_usd="1",
        output_usd="3.2",
    ),
    _BedrockCatalogRecord(
        model_type="claude-opus-4-8",
        model="us.anthropic.claude-opus-4-8",
        billing_sku="geo-cross-region",
        input_usd="5.5",
        output_usd="27.5",
    ),
)

_AWS_BEDROCK_SOURCE_SNAPSHOT_DIGEST = (
    "sha256:1936d89d798e83cbee0d3d95a886a720c7e2de2bb6fe6e86cdfa3e249b5b8649"
)


def _normalized_bedrock_source_snapshot_digest() -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                {
                    "currency": _CATALOG_CURRENCY,
                    "effective_on": _CATALOG_EFFECTIVE_ON.isoformat(),
                    "price_unit": _CATALOG_PRICE_UNIT,
                    "records": [
                        record.model_dump(mode="json") for record in _BEDROCK_CATALOG_RECORDS
                    ],
                    "source_locator": _AWS_BEDROCK_PRICING,
                    "verified_on": _CATALOG_VERIFIED_ON.isoformat(),
                },
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
    )


# A new route, SKU, region, or effective price requires an explicitly reviewed source snapshot
# digest update. This fail-closed check prevents the evidence digest from following edited rows.
if _normalized_bedrock_source_snapshot_digest() != _AWS_BEDROCK_SOURCE_SNAPSHOT_DIGEST:
    raise RuntimeError("built-in Bedrock tariff rows differ from their source snapshot digest")


class ProviderTokenTariff(BaseModel):
    """One immutable tariff tied to the full nonsecret provider execution route."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    provider_config: ProviderConfig
    price: TokenPriceCeiling
    provenance: ProviderTariffProvenance

    @field_validator("provider_config", mode="after")
    @classmethod
    def _freeze_provider_config(cls, value: ProviderConfig) -> ProviderConfig:
        return ProviderConfig.model_validate(value.model_dump())

    @model_validator(mode="after")
    def _require_explicit_route_coordinates(self) -> Self:
        config = self.provider_config
        if self.provenance.route.provider_config != config:
            raise ValueError("tariff route does not match provider config")
        if config.kind is ProviderKind.BEDROCK and config.region is None:
            raise ValueError("audited provider tariff requires an explicit Bedrock region")
        if config.kind is ProviderKind.AZURE_OPENAI:
            if config.endpoint is None:
                raise ValueError("audited provider tariff requires an explicit Azure endpoint")
            if config.deployment is None or config.api_version is None:
                raise ValueError(
                    "audited provider tariff requires an explicit Azure deployment and API version"
                )
            if config.model_type is None:
                raise ValueError("audited Azure tariff requires an explicit canonical model_type")
        if config.model_type is not None:
            resolved = resolve_provider_model(config.kind, config.model)
            if resolved.model_type != config.model_type:
                raise ValueError("provider tariff model_type differs from its runtime model")
        return self

    @property
    def digest(self) -> str:
        """Return a stable digest covering route, price, and audit provenance."""
        payload = json.dumps(
            self.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_usd_per_million(
        cls,
        *,
        provider_config: ProviderConfig,
        input_usd: Decimal | str | int,
        output_usd: Decimal | str | int,
        source_locator: str,
        source_snapshot_digest: str,
        verified_on: date,
        effective_on: date,
        currency: Literal["USD"],
        price_unit: Literal["per_1m_tokens"],
        billing_region: str,
        billing_sku: str,
    ) -> ProviderTokenTariff:
        """Freeze a live-verified route price using exact nano-USD ceilings.

        Azure has no inferred catalog price. Its caller must supply the exact deployment, SKU,
        billing region, effective date, and content digest of the retained source snapshot.
        """
        return cls(
            provider_config=provider_config,
            price=TokenPriceCeiling.from_usd_per_million(
                input_usd=input_usd,
                output_usd=output_usd,
            ),
            provenance=ProviderTariffProvenance(
                source_locator=source_locator,
                source_snapshot_digest=source_snapshot_digest,
                verified_on=verified_on,
                effective_on=effective_on,
                currency=currency,
                price_unit=price_unit,
                route=ProviderTariffRoute(
                    provider_config=provider_config,
                    billing_region=billing_region,
                    billing_sku=billing_sku,
                ),
            ),
        )


def provider_cost_meter(
    tariff: ProviderTokenTariff,
    *,
    input_overhead_tokens: int = 8192,
) -> ProviderCostMeter:
    """Build one hard-budget meter without dropping tariff audit provenance."""
    snapshot = ProviderTokenTariff.model_validate(tariff.model_dump())
    return ProviderCostMeter(
        provider_config=snapshot.provider_config,
        price=snapshot.price,
        tariff_provenance=snapshot.provenance,
        input_overhead_tokens=input_overhead_tokens,
    )


def catalog_provider_token_tariffs() -> tuple[ProviderTokenTariff, ...]:
    """Return defensive copies of the currently audited built-in tariff snapshots."""
    return tuple(ProviderTokenTariff.model_validate(item.model_dump()) for item in _CATALOG)


def catalog_provider_token_tariff(provider_config: ProviderConfig) -> ProviderTokenTariff:
    """Resolve an exact built-in route or reject an unaudited route."""
    requested = ProviderConfig.model_validate(provider_config.model_dump())
    for tariff in _CATALOG:
        if tariff.provider_config == requested:
            return ProviderTokenTariff.model_validate(tariff.model_dump())
    raise ValueError("no audited provider tariff matches the exact provider route")


def _bedrock_tariff(
    *,
    record: _BedrockCatalogRecord,
) -> ProviderTokenTariff:
    return ProviderTokenTariff.from_usd_per_million(
        provider_config=ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model_type=record.model_type,
            model=record.model,
            region="us-east-1",
        ),
        input_usd=record.input_usd,
        output_usd=record.output_usd,
        source_locator=_AWS_BEDROCK_PRICING,
        source_snapshot_digest=_AWS_BEDROCK_SOURCE_SNAPSHOT_DIGEST,
        verified_on=_CATALOG_VERIFIED_ON,
        effective_on=_CATALOG_EFFECTIVE_ON,
        currency="USD",
        price_unit="per_1m_tokens",
        billing_region="us-east-1",
        billing_sku=record.billing_sku,
    )


# The ``us.`` inference profiles use Bedrock's Geo Cross-Region rates, not the lower ``global.``
# profile rates. All catalog objects are derived from the exact source snapshot rows above.
_CATALOG = tuple(_bedrock_tariff(record=record) for record in _BEDROCK_CATALOG_RECORDS)
