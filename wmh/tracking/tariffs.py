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
    TokenPriceCeiling,
)

_AWS_BEDROCK_PRICING = "https://aws.amazon.com/bedrock/pricing/"
_CATALOG_VERIFIED_ON = date(2026, 7, 19)


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
        verified_on: date,
        effective_on: date | None = None,
    ) -> ProviderTokenTariff:
        """Freeze a caller-audited nominal tariff using exact nano-USD ceilings."""
        return cls(
            provider_config=provider_config,
            price=TokenPriceCeiling.from_usd_per_million(
                input_usd=input_usd,
                output_usd=output_usd,
            ),
            provenance=ProviderTariffProvenance(
                source_locator=source_locator,
                verified_on=verified_on,
                effective_on=effective_on,
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
    model_type: str,
    model: str,
    input_usd: str,
    output_usd: str,
) -> ProviderTokenTariff:
    return ProviderTokenTariff.from_usd_per_million(
        provider_config=ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model_type=model_type,
            model=model,
            region="us-east-1",
        ),
        input_usd=input_usd,
        output_usd=output_usd,
        source_locator=_AWS_BEDROCK_PRICING,
        verified_on=_CATALOG_VERIFIED_ON,
    )


_CATALOG = (
    _bedrock_tariff(
        model_type="claude-haiku-4-5",
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        input_usd="1",
        output_usd="5",
    ),
    _bedrock_tariff(
        model_type="glm-5",
        model="zai.glm-5",
        input_usd="1",
        output_usd="3.2",
    ),
    _bedrock_tariff(
        model_type="claude-opus-4-8",
        model="us.anthropic.claude-opus-4-8",
        input_usd="5",
        output_usd="25",
    ),
)
