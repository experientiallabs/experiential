"""Audited provider tariff snapshots and exact hard-budget meter construction."""

from __future__ import annotations

import gzip
import hashlib
import html as html_lib
import io
import json
import re
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from importlib import resources
from typing import Literal, Self, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.models import resolve_provider_model
from wmh.tracking.budget import (
    ProviderCostMeter,
    ProviderTariffBillingMeter,
    ProviderTariffEvidenceReceipt,
    ProviderTariffProvenance,
    ProviderTariffRetainedArtifact,
    ProviderTariffRoute,
    ProviderTariffSourceBinding,
    ProviderTariffSourceSnapshot,
    ProviderTariffVerifiedSource,
    TokenPriceCeiling,
    provider_tariff_claim_digest,
    provider_tariff_evidence_verifier_digest,
    provider_tariff_validated_records,
)

_AWS_BEDROCK_METER_MAP = (
    "https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/"
    "bedrockfoundationmodels/USD/current/bedrockfoundationmodels.json"
)
_AWS_BEDROCK_METER_MAP_DIGEST = (
    "sha256:70ac2fe2f4153bf763492345b2029f06fefdb683023c420319a5b25679f02a11"
)
_AWS_BEDROCK_PRICE_LIST = (
    "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/"
    "AmazonBedrock/20260707080509/index.json"
)
_AWS_BEDROCK_PRICE_LIST_DIGEST = (
    "sha256:2df66ba105f0d831725f825d73bbb8c01d3d4fae2bc64ee5527619c62aedced2"
)
_CATALOG_VERIFIED_ON = date(2026, 7, 19)
_CATALOG_CURRENCY = "USD"
_CATALOG_PRICE_UNIT = "per_1m_tokens"
_EVIDENCE_PACKAGE = "wmh.tracking"
_EVIDENCE_ROOT = "evidence/aws-bedrock"
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_DECODED_SOURCE_BYTES = 256 * 1024 * 1024
_AZURE_API_VERSION_PATTERN = re.compile(r"^(?P<released_on>\d{4}-\d{2}-\d{2})(?:-preview)?$")
_AZURE_EFFECTIVE_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.0{1,9})?Z$"
)
_AZURE_RETAIL_EXACT_LABELS: dict[
    tuple[str, str],
    tuple[str, str, str, str],
] = {
    (
        "gpt-5.5",
        "global-standard",
    ): (
        "Azure OpenAI gpt-5.5",
        "gpt-5.5 GlobalStandard",
        "gpt-5.5 Input Tokens",
        "gpt-5.5 Output Tokens",
    ),
}


class ProviderTariffEvidenceIntegrityError(RuntimeError):
    """A retained tariff source is absent or differs from its frozen digests."""


def _packaged_evidence(
    *,
    source_digest: str,
    suffix: Literal["json", "html"],
    artifact_digest: str,
) -> ProviderTariffRetainedArtifact:
    return ProviderTariffRetainedArtifact(
        storage_kind="package_resource",
        locator=(
            f"{_EVIDENCE_PACKAGE}:{_EVIDENCE_ROOT}/"
            f"sha256-{source_digest.removeprefix('sha256:')}.{suffix}.gz"
        ),
        artifact_digest=artifact_digest,
        content_encoding="gzip",
    )


_AWS_BEDROCK_METER_MAP_SNAPSHOT = ProviderTariffSourceSnapshot(
    source_id="aws-bedrock-meter-map-20260703",
    role="rate_catalog",
    source_locator=_AWS_BEDROCK_METER_MAP,
    source_snapshot_digest=_AWS_BEDROCK_METER_MAP_DIGEST,
    media_type="application/json",
    content_encoding="gzip",
    publication_id="plc-bedrockfoundationmodels-usd-20260703085857",
    published_on=date(2026, 7, 3),
    retained_artifact=_packaged_evidence(
        source_digest=_AWS_BEDROCK_METER_MAP_DIGEST,
        suffix="json",
        artifact_digest=("sha256:36acc6cbfc79c073f96bae7262d3bc33a07d2809ed6b4c20922e1c9f2749f613"),
    ),
)
_AWS_BEDROCK_PRICING_PAGE_SNAPSHOT = ProviderTariffSourceSnapshot(
    source_id="aws-bedrock-pricing-page-20260719",
    role="semantic_mapping",
    source_locator="https://aws.amazon.com/bedrock/pricing/",
    source_snapshot_digest=(
        "sha256:fe5438f7f28cef02f6a3f439cbe323c3bdf109db34f0a713b1beacbc1c7f6c92"
    ),
    media_type="text/html",
    content_encoding="identity",
    retained_artifact=_packaged_evidence(
        source_digest=("sha256:fe5438f7f28cef02f6a3f439cbe323c3bdf109db34f0a713b1beacbc1c7f6c92"),
        suffix="html",
        artifact_digest=("sha256:06e4fd19f95e962d35effbb0211f914321f1158fbf4250df4871fa4cc9365fc1"),
    ),
)
_AWS_BEDROCK_HAIKU_MODEL_CARD_SNAPSHOT = ProviderTariffSourceSnapshot(
    source_id="aws-bedrock-haiku-4-5-model-card-20260719",
    role="route_definition",
    source_locator=(
        "https://docs.aws.amazon.com/bedrock/latest/userguide/"
        "model-card-anthropic-claude-haiku-4-5.html"
    ),
    source_snapshot_digest=(
        "sha256:4c0e99775e1e9fd9c0fc45ce01d92a250ffd56dab7eb24d8ddb188042e55e0f2"
    ),
    media_type="text/html",
    content_encoding="identity",
    retained_artifact=_packaged_evidence(
        source_digest=("sha256:4c0e99775e1e9fd9c0fc45ce01d92a250ffd56dab7eb24d8ddb188042e55e0f2"),
        suffix="html",
        artifact_digest=("sha256:bba714dfd93743060a9c20b534715c3fc289033102510fa7a28932758ac1a1d0"),
    ),
)
_AWS_BEDROCK_OPUS_MODEL_CARD_SNAPSHOT = ProviderTariffSourceSnapshot(
    source_id="aws-bedrock-opus-4-8-model-card-20260719",
    role="route_definition",
    source_locator=(
        "https://docs.aws.amazon.com/bedrock/latest/userguide/"
        "model-card-anthropic-claude-opus-4-8.html"
    ),
    source_snapshot_digest=(
        "sha256:682563efdd76f9b9099f320a532491123154d81a859222434f345ed93dee2f05"
    ),
    media_type="text/html",
    content_encoding="identity",
    retained_artifact=_packaged_evidence(
        source_digest=("sha256:682563efdd76f9b9099f320a532491123154d81a859222434f345ed93dee2f05"),
        suffix="html",
        artifact_digest=("sha256:e34b9a58873a31d114ac28570ecbeac27ef737efe25920a69552b38dce53283a"),
    ),
)
_AWS_BEDROCK_PRICE_LIST_SNAPSHOT = ProviderTariffSourceSnapshot(
    source_id="aws-bedrock-price-list-20260707",
    role="rate_catalog",
    source_locator=_AWS_BEDROCK_PRICE_LIST,
    source_snapshot_digest=_AWS_BEDROCK_PRICE_LIST_DIGEST,
    media_type="application/json",
    content_encoding="identity",
    publication_id="AmazonBedrock:20260707080509",
    published_on=date(2026, 7, 7),
    retained_artifact=_packaged_evidence(
        source_digest=_AWS_BEDROCK_PRICE_LIST_DIGEST,
        suffix="json",
        artifact_digest=("sha256:ae257969b20e179bd3570c3666a164cfaca1007e334e0f7ce7c00538fbb71af2"),
    ),
)

_BUILTIN_SOURCE_SNAPSHOTS = (
    _AWS_BEDROCK_HAIKU_MODEL_CARD_SNAPSHOT,
    _AWS_BEDROCK_METER_MAP_SNAPSHOT,
    _AWS_BEDROCK_OPUS_MODEL_CARD_SNAPSHOT,
    _AWS_BEDROCK_PRICE_LIST_SNAPSHOT,
    _AWS_BEDROCK_PRICING_PAGE_SNAPSHOT,
)


class _BedrockCatalogRecord(BaseModel):
    """One normalized row retained in the immutable built-in source snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_type: str
    model: str
    billing_mode: str
    source_snapshots: tuple[ProviderTariffSourceSnapshot, ...]
    source_bindings: tuple[ProviderTariffSourceBinding, ...]
    effective_on: date | None
    billing_meters: tuple[ProviderTariffBillingMeter, ProviderTariffBillingMeter]
    input_usd: str
    output_usd: str


_BEDROCK_CATALOG_RECORDS = (
    _BedrockCatalogRecord(
        model_type="claude-haiku-4-5",
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        billing_mode="geo-cross-region-standard",
        source_snapshots=(
            _AWS_BEDROCK_HAIKU_MODEL_CARD_SNAPSHOT,
            _AWS_BEDROCK_METER_MAP_SNAPSHOT,
            _AWS_BEDROCK_PRICING_PAGE_SNAPSHOT,
        ),
        source_bindings=(
            ProviderTariffSourceBinding(
                claim="route_identity",
                source_id="aws-bedrock-haiku-4-5-model-card-20260719",
                source_record_path="/geo-inference-id/us",
                source_value="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                canonical_value="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            ),
            ProviderTariffSourceBinding(
                claim="route_identity",
                source_id="aws-bedrock-pricing-page-20260719",
                source_record_path="/geo-cross-region/us/claude-4.5-haiku/model",
                source_value="Claude 4.5 Haiku",
                canonical_value="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            ),
            ProviderTariffSourceBinding(
                claim="usage_dimension",
                source_id="aws-bedrock-pricing-page-20260719",
                source_record_path=(
                    "/geo-cross-region/us/claude-4.5-haiku/input-token-price-token"
                ),
                source_value="APY9uM3JGxEtL5H9EDm5AwcPdLxfrWr-RFMq0UellFs",
                canonical_value="input_tokens",
                target_meter_source_id="aws-bedrock-meter-map-20260703",
                target_meter_record_path=(
                    "/regions/US East (N. Virginia)/APY9uM3JGxEtL5H9EDm5AwcPdLxfrWr-RFMq0UellFs"
                ),
            ),
            ProviderTariffSourceBinding(
                claim="usage_dimension",
                source_id="aws-bedrock-pricing-page-20260719",
                source_record_path=(
                    "/geo-cross-region/us/claude-4.5-haiku/output-token-price-token"
                ),
                source_value="O6zZHzgP5kfAsKpsBRIEEOBkSjbbqAoskrV4twdrBE4",
                canonical_value="output_tokens",
                target_meter_source_id="aws-bedrock-meter-map-20260703",
                target_meter_record_path=(
                    "/regions/US East (N. Virginia)/O6zZHzgP5kfAsKpsBRIEEOBkSjbbqAoskrV4twdrBE4"
                ),
            ),
        ),
        effective_on=None,
        billing_meters=(
            ProviderTariffBillingMeter(
                usage_dimension="input_tokens",
                source_id="aws-bedrock-meter-map-20260703",
                source_record_path=(
                    "/regions/US East (N. Virginia)/APY9uM3JGxEtL5H9EDm5AwcPdLxfrWr-RFMq0UellFs"
                ),
                sku_id="JQDUC8Q4K8C6GSGH",
                rate_id="JQDUC8Q4K8C6GSGH.4799GE89SK.6YS6EN2CT7",
                billing_region="us-east-1",
                billing_mode="geo-cross-region-standard",
                effective_on=None,
                source_price_usd=Decimal("1.1"),
                source_price_unit="per_1m_tokens",
            ),
            ProviderTariffBillingMeter(
                usage_dimension="output_tokens",
                source_id="aws-bedrock-meter-map-20260703",
                source_record_path=(
                    "/regions/US East (N. Virginia)/O6zZHzgP5kfAsKpsBRIEEOBkSjbbqAoskrV4twdrBE4"
                ),
                sku_id="X629GDA2GXAP6R54",
                rate_id="X629GDA2GXAP6R54.4799GE89SK.6YS6EN2CT7",
                billing_region="us-east-1",
                billing_mode="geo-cross-region-standard",
                effective_on=None,
                source_price_usd=Decimal("5.5"),
                source_price_unit="per_1m_tokens",
            ),
        ),
        input_usd="1.1",
        output_usd="5.5",
    ),
    _BedrockCatalogRecord(
        model_type="glm-5",
        model="zai.glm-5",
        billing_mode="in-region-on-demand-standard",
        source_snapshots=(_AWS_BEDROCK_PRICE_LIST_SNAPSHOT,),
        source_bindings=(
            ProviderTariffSourceBinding(
                claim="route_identity",
                source_id="aws-bedrock-price-list-20260707",
                source_record_path="/products/8RQBEKEP5KG2MZY7/attributes/model",
                source_value="GLM 5",
                canonical_value="zai.glm-5",
                target_meter_source_id="aws-bedrock-price-list-20260707",
                target_meter_record_path=(
                    "/terms/OnDemand/8RQBEKEP5KG2MZY7/"
                    "8RQBEKEP5KG2MZY7.JRTCKXETXF/priceDimensions/"
                    "8RQBEKEP5KG2MZY7.JRTCKXETXF.6YS6EN2CT7"
                ),
            ),
            ProviderTariffSourceBinding(
                claim="route_identity",
                source_id="aws-bedrock-price-list-20260707",
                source_record_path="/products/YTB2BH9W4UZVKTEG/attributes/model",
                source_value="GLM 5",
                canonical_value="zai.glm-5",
                target_meter_source_id="aws-bedrock-price-list-20260707",
                target_meter_record_path=(
                    "/terms/OnDemand/YTB2BH9W4UZVKTEG/"
                    "YTB2BH9W4UZVKTEG.JRTCKXETXF/priceDimensions/"
                    "YTB2BH9W4UZVKTEG.JRTCKXETXF.6YS6EN2CT7"
                ),
            ),
            ProviderTariffSourceBinding(
                claim="usage_dimension",
                source_id="aws-bedrock-price-list-20260707",
                source_record_path="/products/YTB2BH9W4UZVKTEG/attributes/inferenceType",
                source_value="Input tokens",
                canonical_value="input_tokens",
                target_meter_source_id="aws-bedrock-price-list-20260707",
                target_meter_record_path=(
                    "/terms/OnDemand/YTB2BH9W4UZVKTEG/"
                    "YTB2BH9W4UZVKTEG.JRTCKXETXF/priceDimensions/"
                    "YTB2BH9W4UZVKTEG.JRTCKXETXF.6YS6EN2CT7"
                ),
            ),
            ProviderTariffSourceBinding(
                claim="usage_dimension",
                source_id="aws-bedrock-price-list-20260707",
                source_record_path="/products/8RQBEKEP5KG2MZY7/attributes/inferenceType",
                source_value="Output tokens",
                canonical_value="output_tokens",
                target_meter_source_id="aws-bedrock-price-list-20260707",
                target_meter_record_path=(
                    "/terms/OnDemand/8RQBEKEP5KG2MZY7/"
                    "8RQBEKEP5KG2MZY7.JRTCKXETXF/priceDimensions/"
                    "8RQBEKEP5KG2MZY7.JRTCKXETXF.6YS6EN2CT7"
                ),
            ),
        ),
        effective_on=date(2026, 7, 1),
        billing_meters=(
            ProviderTariffBillingMeter(
                usage_dimension="input_tokens",
                source_id="aws-bedrock-price-list-20260707",
                source_record_path=(
                    "/terms/OnDemand/YTB2BH9W4UZVKTEG/"
                    "YTB2BH9W4UZVKTEG.JRTCKXETXF/priceDimensions/"
                    "YTB2BH9W4UZVKTEG.JRTCKXETXF.6YS6EN2CT7"
                ),
                sku_id="YTB2BH9W4UZVKTEG",
                rate_id="YTB2BH9W4UZVKTEG.JRTCKXETXF.6YS6EN2CT7",
                billing_region="us-east-1",
                billing_mode="in-region-on-demand-standard",
                effective_on=date(2026, 7, 1),
                source_price_usd=Decimal("0.001"),
                source_price_unit="per_1k_tokens",
            ),
            ProviderTariffBillingMeter(
                usage_dimension="output_tokens",
                source_id="aws-bedrock-price-list-20260707",
                source_record_path=(
                    "/terms/OnDemand/8RQBEKEP5KG2MZY7/"
                    "8RQBEKEP5KG2MZY7.JRTCKXETXF/priceDimensions/"
                    "8RQBEKEP5KG2MZY7.JRTCKXETXF.6YS6EN2CT7"
                ),
                sku_id="8RQBEKEP5KG2MZY7",
                rate_id="8RQBEKEP5KG2MZY7.JRTCKXETXF.6YS6EN2CT7",
                billing_region="us-east-1",
                billing_mode="in-region-on-demand-standard",
                effective_on=date(2026, 7, 1),
                source_price_usd=Decimal("0.0032"),
                source_price_unit="per_1k_tokens",
            ),
        ),
        input_usd="1",
        output_usd="3.2",
    ),
    _BedrockCatalogRecord(
        model_type="claude-opus-4-8",
        model="us.anthropic.claude-opus-4-8",
        billing_mode="geo-cross-region-standard",
        source_snapshots=(
            _AWS_BEDROCK_METER_MAP_SNAPSHOT,
            _AWS_BEDROCK_OPUS_MODEL_CARD_SNAPSHOT,
            _AWS_BEDROCK_PRICING_PAGE_SNAPSHOT,
        ),
        source_bindings=(
            ProviderTariffSourceBinding(
                claim="route_identity",
                source_id="aws-bedrock-opus-4-8-model-card-20260719",
                source_record_path="/geo-inference-id/us",
                source_value="us.anthropic.claude-opus-4-8",
                canonical_value="us.anthropic.claude-opus-4-8",
            ),
            ProviderTariffSourceBinding(
                claim="route_identity",
                source_id="aws-bedrock-pricing-page-20260719",
                source_record_path="/geo-cross-region/us/claude-opus-4.8/model",
                source_value="Claude Opus 4.8",
                canonical_value="us.anthropic.claude-opus-4-8",
            ),
            ProviderTariffSourceBinding(
                claim="usage_dimension",
                source_id="aws-bedrock-pricing-page-20260719",
                source_record_path="/geo-cross-region/us/claude-opus-4.8/input-token-price-token",
                source_value="f9TflP4QPpfmpoCUae8qAT5_ecgxUrtuuJXamyHHDLE",
                canonical_value="input_tokens",
                target_meter_source_id="aws-bedrock-meter-map-20260703",
                target_meter_record_path=(
                    "/regions/US East (N. Virginia)/f9TflP4QPpfmpoCUae8qAT5_ecgxUrtuuJXamyHHDLE"
                ),
            ),
            ProviderTariffSourceBinding(
                claim="usage_dimension",
                source_id="aws-bedrock-pricing-page-20260719",
                source_record_path="/geo-cross-region/us/claude-opus-4.8/output-token-price-token",
                source_value="jKk45pcJTXkhozzY9QxZVfUN3thwMVsHMmiZmLfi5lA",
                canonical_value="output_tokens",
                target_meter_source_id="aws-bedrock-meter-map-20260703",
                target_meter_record_path=(
                    "/regions/US East (N. Virginia)/jKk45pcJTXkhozzY9QxZVfUN3thwMVsHMmiZmLfi5lA"
                ),
            ),
        ),
        effective_on=None,
        billing_meters=(
            ProviderTariffBillingMeter(
                usage_dimension="input_tokens",
                source_id="aws-bedrock-meter-map-20260703",
                source_record_path=(
                    "/regions/US East (N. Virginia)/f9TflP4QPpfmpoCUae8qAT5_ecgxUrtuuJXamyHHDLE"
                ),
                sku_id="4AVHTD2NXFKSU6HU",
                rate_id="4AVHTD2NXFKSU6HU.4799GE89SK.6YS6EN2CT7",
                billing_region="us-east-1",
                billing_mode="geo-cross-region-standard",
                effective_on=None,
                source_price_usd=Decimal("5.5"),
                source_price_unit="per_1m_tokens",
            ),
            ProviderTariffBillingMeter(
                usage_dimension="output_tokens",
                source_id="aws-bedrock-meter-map-20260703",
                source_record_path=(
                    "/regions/US East (N. Virginia)/jKk45pcJTXkhozzY9QxZVfUN3thwMVsHMmiZmLfi5lA"
                ),
                sku_id="YKJ5FPMCZAQF5BHF",
                rate_id="YKJ5FPMCZAQF5BHF.4799GE89SK.6YS6EN2CT7",
                billing_region="us-east-1",
                billing_mode="geo-cross-region-standard",
                effective_on=None,
                source_price_usd=Decimal("27.5"),
                source_price_unit="per_1m_tokens",
            ),
        ),
        input_usd="5.5",
        output_usd="27.5",
    ),
)

_BEDROCK_CATALOG_RECORD_DIGESTS = {
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": (
        "sha256:d9460ca39c3d183914ddec29d1e035c340072f9c11289aee3deca9bb849e8517"
    ),
    "zai.glm-5": "sha256:95f751b7efce9e3ba68ff1a926e4d42a502b6caab279affca2df08f00d2e77ca",
    "us.anthropic.claude-opus-4-8": (
        "sha256:1435fdfb3bca497904994658076ff33c734472c806acff722e3dec51f4104bdd"
    ),
}


def _bedrock_catalog_record_digest(record: _BedrockCatalogRecord) -> str:
    payload = json.dumps(
        record.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _verify_bedrock_catalog_integrity(
    records: tuple[_BedrockCatalogRecord, ...],
) -> None:
    """Fail closed if embedded prices drift without an independently pinned digest edit."""
    coordinates = tuple(record.model for record in records)
    if coordinates != tuple(_BEDROCK_CATALOG_RECORD_DIGESTS):
        raise RuntimeError("embedded Bedrock tariff record set changed")
    for record in records:
        if _bedrock_catalog_record_digest(record) != _BEDROCK_CATALOG_RECORD_DIGESTS[record.model]:
            raise RuntimeError(f"embedded Bedrock tariff record changed: {record.model}")


_verify_bedrock_catalog_integrity(_BEDROCK_CATALOG_RECORDS)


class ProviderTokenTariff(BaseModel):
    """One immutable tariff tied to the full nonsecret provider execution route."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

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
        floor = self.provenance.token_price_floor()
        if (
            self.price.input_nano_usd_per_token < floor.input_nano_usd_per_token
            or self.price.output_nano_usd_per_token < floor.output_nano_usd_per_token
        ):
            raise ValueError(
                "provider token price ceiling understates retained billing meter rates"
            )
        return self

    @property
    def digest(self) -> str:
        """Return a stable digest covering route, price, and audit provenance."""
        return provider_tariff_claim_digest(
            provider_config=self.provider_config,
            price=self.price,
            provenance=self.provenance,
        )

    @classmethod
    def from_usd_per_million(
        cls,
        *,
        provider_config: ProviderConfig,
        input_usd: Decimal | str | int,
        output_usd: Decimal | str | int,
        source_snapshots: tuple[ProviderTariffSourceSnapshot, ...],
        source_bindings: tuple[ProviderTariffSourceBinding, ...],
        verified_on: date,
        effective_on: date | None,
        currency: Literal["USD"],
        price_unit: Literal["per_1m_tokens"],
        billing_region: str,
        billing_mode: str,
        billing_meters: tuple[ProviderTariffBillingMeter, ProviderTariffBillingMeter],
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
                source_snapshots=source_snapshots,
                source_bindings=source_bindings,
                verified_on=verified_on,
                effective_on=effective_on,
                currency=currency,
                price_unit=price_unit,
                route=ProviderTariffRoute(
                    provider_config=provider_config,
                    billing_region=billing_region,
                    billing_mode=billing_mode,
                    billing_meters=billing_meters,
                ),
            ),
        )


def provider_cost_meter(
    tariff: ProviderTokenTariff,
    *,
    input_overhead_tokens: int = 8192,
    evidence_artifacts: Mapping[str, bytes] | None = None,
) -> ProviderCostMeter:
    """Verify retained evidence and build one hard-budget provider meter."""
    snapshot = ProviderTokenTariff.model_validate(tariff.model_dump())
    evidence_receipt = verify_provider_tariff_evidence(
        snapshot,
        evidence_artifacts=evidence_artifacts,
    )
    return ProviderCostMeter(
        provider_config=snapshot.provider_config,
        price=snapshot.price,
        tariff_provenance=snapshot.provenance,
        tariff_evidence_receipt=evidence_receipt,
        input_overhead_tokens=input_overhead_tokens,
    )


def _decode_verified_artifact(
    *,
    locator: str,
    artifact_bytes: bytes,
    artifact_digest: str,
    artifact_encoding: str,
    source_digest: str,
) -> bytes:
    if not artifact_bytes or len(artifact_bytes) > _MAX_ARTIFACT_BYTES:
        raise ProviderTariffEvidenceIntegrityError(
            f"retained tariff artifact exceeds its safe encoded size: {locator}"
        )
    observed_artifact_digest = "sha256:" + hashlib.sha256(artifact_bytes).hexdigest()
    if observed_artifact_digest != artifact_digest:
        raise ProviderTariffEvidenceIntegrityError(
            f"retained tariff artifact digest mismatch: {locator}"
        )
    if artifact_encoding == "gzip":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(artifact_bytes)) as compressed:
                source_bytes = compressed.read(_MAX_DECODED_SOURCE_BYTES + 1)
        except (EOFError, OSError) as exc:
            raise ProviderTariffEvidenceIntegrityError(
                f"retained tariff artifact is not valid gzip: {locator}"
            ) from exc
    else:
        source_bytes = artifact_bytes
    if not source_bytes or len(source_bytes) > _MAX_DECODED_SOURCE_BYTES:
        raise ProviderTariffEvidenceIntegrityError(
            f"retained tariff source exceeds its safe decoded size: {locator}"
        )
    observed_source_digest = "sha256:" + hashlib.sha256(source_bytes).hexdigest()
    if observed_source_digest != source_digest:
        raise ProviderTariffEvidenceIntegrityError(
            f"retained tariff source digest mismatch: {locator}"
        )
    return source_bytes


@lru_cache(maxsize=16)
def _verify_packaged_source_bytes(
    *,
    locator: str,
    artifact_digest: str,
    artifact_encoding: str,
    source_digest: str,
) -> bytes:
    package, relative_path = locator.split(":", 1)
    try:
        artifact_bytes = resources.files(package).joinpath(relative_path).read_bytes()
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise ProviderTariffEvidenceIntegrityError(
            f"retained tariff artifact is unavailable: {locator}"
        ) from exc
    return _decode_verified_artifact(
        locator=locator,
        artifact_bytes=artifact_bytes,
        artifact_digest=artifact_digest,
        artifact_encoding=artifact_encoding,
        source_digest=source_digest,
    )


@lru_cache(maxsize=8)
def _parse_json_source(source_digest: str, source_bytes: bytes) -> object:
    del source_digest

    def reject_nonfinite(value: str) -> object:
        raise ValueError(f"nonfinite JSON number: {value}")

    try:
        return cast(
            "object",
            json.loads(
                source_bytes,
                parse_float=Decimal,
                parse_int=Decimal,
                parse_constant=reject_nonfinite,
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProviderTariffEvidenceIntegrityError(
            "retained tariff JSON source cannot be decoded"
        ) from exc


def _json_pointer_segments(path: str) -> tuple[str, ...]:
    if not path.startswith("/"):
        raise ProviderTariffEvidenceIntegrityError("tariff JSON record path is not absolute")
    return tuple(segment.replace("~1", "/").replace("~0", "~") for segment in path[1:].split("/"))


def _json_pointer(document: object, path: str) -> object:
    current = document
    for segment in _json_pointer_segments(path):
        if isinstance(current, dict):
            mapping = cast("dict[str, object]", current)
            if segment not in mapping:
                raise ProviderTariffEvidenceIntegrityError(
                    f"tariff source record path does not exist: {path}"
                )
            current = mapping[segment]
            continue
        if isinstance(current, list) and segment.isdigit():
            sequence = cast("list[object]", current)
            index = int(segment)
            if index >= len(sequence):
                raise ProviderTariffEvidenceIntegrityError(
                    f"tariff source record path does not exist: {path}"
                )
            current = sequence[index]
            continue
        raise ProviderTariffEvidenceIntegrityError(
            f"tariff source record path does not exist: {path}"
        )
    return current


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ProviderTariffEvidenceIntegrityError(f"{label} is not a JSON object")
    return cast("dict[str, object]", value)


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ProviderTariffEvidenceIntegrityError(f"{label} is not exact text")
    return value


def _decimal(value: object, *, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Decimal, int, str)):
        raise ProviderTariffEvidenceIntegrityError(f"{label} is not an exact decimal")
    try:
        return Decimal(value)
    except ArithmeticError as exc:
        raise ProviderTariffEvidenceIntegrityError(f"{label} is not an exact decimal") from exc


def _verify_source_binding_records(
    tariff: ProviderTokenTariff,
    snapshots: dict[str, ProviderTariffSourceSnapshot],
    source_bytes: dict[str, bytes],
) -> None:
    for binding in tariff.provenance.source_bindings:
        snapshot = snapshots[binding.source_id]
        retained = source_bytes[binding.source_id]
        if snapshot.media_type == "application/json":
            document = _parse_json_source(snapshot.source_snapshot_digest, retained)
            observed = _json_pointer(document, binding.source_record_path)
            if not isinstance(observed, str) or observed != binding.source_value:
                raise ProviderTariffEvidenceIntegrityError(
                    f"tariff source binding differs from retained JSON: {binding.source_id}"
                )
        else:
            try:
                decoded = html_lib.unescape(retained.decode())
            except UnicodeDecodeError as exc:
                raise ProviderTariffEvidenceIntegrityError(
                    f"retained tariff HTML is not UTF-8: {binding.source_id}"
                ) from exc
            if binding.source_value not in decoded:
                raise ProviderTariffEvidenceIntegrityError(
                    f"tariff source binding differs from retained HTML: {binding.source_id}"
                )


def _verify_source_publication_metadata(
    snapshot: ProviderTariffSourceSnapshot,
    retained: bytes,
) -> None:
    if snapshot.media_type != "application/json":
        if snapshot.publication_id is not None or snapshot.published_on is not None:
            raise ProviderTariffEvidenceIntegrityError(
                "HTML tariff source has unverifiable publication metadata"
            )
        return
    document = _mapping(
        _parse_json_source(snapshot.source_snapshot_digest, retained),
        label="tariff publication source",
    )
    if "manifest" in document:
        manifest = _mapping(document.get("manifest"), label="AWS meter-map manifest")
        publication_id = _text(manifest.get("esIndex"), label="AWS meter-map publication ID")
        publication_timestamp = _text(
            manifest.get("hawkFilePublicationDate"),
            label="AWS meter-map publication date",
        )
    elif "offerCode" in document and "version" in document:
        publication_id = (
            f"{_text(document.get('offerCode'), label='AWS offer code')}:"
            f"{_text(document.get('version'), label='AWS offer version')}"
        )
        publication_timestamp = _text(
            document.get("publicationDate"),
            label="AWS offer publication date",
        )
    else:
        if snapshot.publication_id is not None or snapshot.published_on is not None:
            raise ProviderTariffEvidenceIntegrityError(
                "tariff source publication metadata has no supported retained record"
            )
        return
    try:
        published_on = date.fromisoformat(publication_timestamp.split("T", 1)[0])
    except ValueError as exc:
        raise ProviderTariffEvidenceIntegrityError(
            "tariff source publication date is not an ISO date"
        ) from exc
    if snapshot.publication_id != publication_id or snapshot.published_on != published_on:
        raise ProviderTariffEvidenceIntegrityError(
            "tariff source publication metadata differs from retained evidence"
        )


def _verify_bedrock_binding_coordinates(tariff: ProviderTokenTariff) -> None:
    runtime_model = tariff.provider_config.model
    pricing_coordinates = {
        "us.anthropic.claude-haiku-4-5-20251001-v1:0": (
            "/geo-cross-region/us/claude-4.5-haiku",
            "Claude 4.5 Haiku",
        ),
        "us.anthropic.claude-opus-4-8": (
            "/geo-cross-region/us/claude-opus-4.8",
            "Claude Opus 4.8",
        ),
    }
    semantic_bindings = [
        binding
        for binding in tariff.provenance.source_bindings
        if next(
            snapshot
            for snapshot in tariff.provenance.source_snapshots
            if snapshot.source_id == binding.source_id
        ).role
        == "semantic_mapping"
    ]
    if semantic_bindings:
        try:
            prefix, model_label = pricing_coordinates[runtime_model]
        except KeyError as exc:
            raise ProviderTariffEvidenceIntegrityError(
                "Bedrock semantic pricing coordinates do not support the runtime model"
            ) from exc
        expected = {
            ("route_identity", f"{prefix}/model", runtime_model),
            (
                "usage_dimension",
                f"{prefix}/input-token-price-token",
                "input_tokens",
            ),
            (
                "usage_dimension",
                f"{prefix}/output-token-price-token",
                "output_tokens",
            ),
        }
        observed = {
            (binding.claim, binding.source_record_path, binding.canonical_value)
            for binding in semantic_bindings
        }
        if observed != expected:
            raise ProviderTariffEvidenceIntegrityError(
                "Bedrock semantic pricing bindings use unexpected record coordinates"
            )
        route_binding = next(
            binding for binding in semantic_bindings if binding.claim == "route_identity"
        )
        if route_binding.source_value != model_label:
            raise ProviderTariffEvidenceIntegrityError(
                "Bedrock semantic pricing row names a different runtime model"
            )

    for binding in tariff.provenance.source_bindings:
        snapshot = next(
            snapshot
            for snapshot in tariff.provenance.source_snapshots
            if snapshot.source_id == binding.source_id
        )
        if snapshot.role == "route_definition":
            expected_path = f"/geo-inference-id/{runtime_model.split('.', 1)[0]}"
            if (
                binding.claim != "route_identity"
                or binding.source_record_path != expected_path
                or binding.source_value != runtime_model
            ):
                raise ProviderTariffEvidenceIntegrityError(
                    "Bedrock model-card binding does not prove the exact inference profile"
                )
        if snapshot.role != "rate_catalog":
            continue
        segments = _json_pointer_segments(binding.source_record_path)
        if len(segments) < 4 or segments[0] != "products":
            continue
        target = next(
            (
                meter
                for meter in tariff.provenance.route.billing_meters
                if meter.source_id == binding.target_meter_source_id
                and meter.source_record_path == binding.target_meter_record_path
            ),
            None,
        )
        if target is None or segments[1] != target.sku_id:
            raise ProviderTariffEvidenceIntegrityError(
                "Bedrock product claim is not joined to its target billing SKU"
            )


def _verify_meter_map_record(
    *,
    document: object,
    meter: ProviderTariffBillingMeter,
    record: dict[str, object],
) -> None:
    del document
    segments = _json_pointer_segments(meter.source_record_path)
    if len(segments) != 3 or segments[0] != "regions":
        raise ProviderTariffEvidenceIntegrityError("AWS meter-map path has an unexpected shape")
    region_name, regionless_rate = segments[1:]
    region_codes = {"US East (N. Virginia)": "us-east-1"}
    if region_codes.get(region_name) != meter.billing_region:
        raise ProviderTariffEvidenceIntegrityError(
            "billing meter region differs from AWS meter map"
        )
    observed_rate = _text(record.get("rateCode"), label="AWS meter-map rateCode")
    if observed_rate != meter.rate_id or observed_rate.split(".", 1)[0] != meter.sku_id:
        raise ProviderTariffEvidenceIntegrityError("billing meter IDs differ from AWS meter map")
    if (
        _text(record.get("RegionlessRateCode"), label="AWS meter-map RegionlessRateCode")
        != regionless_rate
    ):
        raise ProviderTariffEvidenceIntegrityError(
            "billing meter path differs from AWS RegionlessRateCode"
        )
    if _decimal(record.get("price"), label="AWS meter-map price") != meter.source_price_usd:
        raise ProviderTariffEvidenceIntegrityError("billing meter price differs from AWS meter map")
    if meter.source_price_unit != "per_1m_tokens":
        raise ProviderTariffEvidenceIntegrityError(
            "AWS Bedrock semantic table requires per-1M rates"
        )
    if meter.billing_mode != "geo-cross-region-standard" or meter.effective_on is not None:
        raise ProviderTariffEvidenceIntegrityError(
            "billing meter mode or effectivity differs from retained Bedrock evidence"
        )


def _verify_offer_record(
    *,
    document: object,
    meter: ProviderTariffBillingMeter,
    record: dict[str, object],
) -> None:
    segments = _json_pointer_segments(meter.source_record_path)
    if (
        len(segments) != 6
        or segments[0:2] != ("terms", "OnDemand")
        or segments[4] != "priceDimensions"
    ):
        raise ProviderTariffEvidenceIntegrityError("AWS offer record path has an unexpected shape")
    sku_id, offer_id, rate_id = segments[2], segments[3], segments[5]
    if sku_id != meter.sku_id or rate_id != meter.rate_id:
        raise ProviderTariffEvidenceIntegrityError("billing meter IDs differ from AWS offer path")
    if _text(record.get("rateCode"), label="AWS offer rateCode") != meter.rate_id:
        raise ProviderTariffEvidenceIntegrityError("billing meter rate differs from AWS offer")
    price_per_unit = _mapping(record.get("pricePerUnit"), label="AWS offer pricePerUnit")
    if _decimal(price_per_unit.get("USD"), label="AWS offer USD price") != meter.source_price_usd:
        raise ProviderTariffEvidenceIntegrityError("billing meter price differs from AWS offer")
    source_unit = _text(record.get("unit"), label="AWS offer unit")
    expected_unit = "1K tokens" if meter.source_price_unit == "per_1k_tokens" else "1M tokens"
    if source_unit != expected_unit:
        raise ProviderTariffEvidenceIntegrityError("billing meter unit differs from AWS offer")

    root = _mapping(document, label="AWS offer root")
    products = _mapping(root.get("products"), label="AWS offer products")
    product = _mapping(products.get(sku_id), label="AWS offer product")
    if _text(product.get("sku"), label="AWS offer product SKU") != meter.sku_id:
        raise ProviderTariffEvidenceIntegrityError("billing meter SKU differs from AWS product")
    attributes = _mapping(product.get("attributes"), label="AWS offer product attributes")
    if _text(attributes.get("regionCode"), label="AWS offer region") != meter.billing_region:
        raise ProviderTariffEvidenceIntegrityError("billing meter region differs from AWS offer")
    if _text(attributes.get("feature"), label="AWS offer feature") != "On-demand Inference":
        raise ProviderTariffEvidenceIntegrityError("AWS offer is not on-demand inference")
    if meter.billing_mode != "in-region-on-demand-standard":
        raise ProviderTariffEvidenceIntegrityError("billing meter mode differs from AWS offer")
    expected_dimension = (
        "Input tokens" if meter.usage_dimension == "input_tokens" else "Output tokens"
    )
    if (
        _text(attributes.get("inferenceType"), label="AWS offer inference type")
        != expected_dimension
    ):
        raise ProviderTariffEvidenceIntegrityError(
            "billing meter usage dimension differs from AWS offer"
        )

    terms = _mapping(root.get("terms"), label="AWS offer terms")
    on_demand = _mapping(terms.get("OnDemand"), label="AWS on-demand terms")
    sku_terms = _mapping(on_demand.get(sku_id), label="AWS SKU terms")
    offer = _mapping(sku_terms.get(offer_id), label="AWS offer term")
    effective = _text(offer.get("effectiveDate"), label="AWS offer effectiveDate")
    try:
        effective_on = date.fromisoformat(effective.split("T", 1)[0])
    except ValueError as exc:
        raise ProviderTariffEvidenceIntegrityError(
            "AWS offer effectiveDate is not an ISO date"
        ) from exc
    if effective_on != meter.effective_on:
        raise ProviderTariffEvidenceIntegrityError(
            "billing meter effective date differs from AWS offer"
        )


def _verify_billing_meter_records(
    tariff: ProviderTokenTariff,
    snapshots: dict[str, ProviderTariffSourceSnapshot],
    source_bytes: dict[str, bytes],
) -> None:
    for meter in tariff.provenance.route.billing_meters:
        snapshot = snapshots[meter.source_id]
        if snapshot.media_type != "application/json":
            raise ProviderTariffEvidenceIntegrityError("billing meter source is not retained JSON")
        document = _parse_json_source(
            snapshot.source_snapshot_digest,
            source_bytes[meter.source_id],
        )
        record = _mapping(
            _json_pointer(document, meter.source_record_path),
            label="billing meter source record",
        )
        if "RegionlessRateCode" in record:
            _verify_meter_map_record(document=document, meter=meter, record=record)
        elif "pricePerUnit" in record:
            _verify_offer_record(document=document, meter=meter, record=record)
        else:
            raise ProviderTariffEvidenceIntegrityError(
                "billing meter source record has an unsupported shape"
            )


def _verify_html_semantic_rows(
    tariff: ProviderTokenTariff,
    snapshots: dict[str, ProviderTariffSourceSnapshot],
    source_bytes: dict[str, bytes],
) -> None:
    semantic_source_ids = {
        snapshot.source_id
        for snapshot in tariff.provenance.source_snapshots
        if snapshot.role == "semantic_mapping"
    }
    meter_coordinates = {
        (meter.source_id, meter.source_record_path): meter
        for meter in tariff.provenance.route.billing_meters
    }
    for source_id in semantic_source_ids:
        route_bindings = [
            binding
            for binding in tariff.provenance.source_bindings
            if binding.source_id == source_id and binding.claim == "route_identity"
        ]
        usage_bindings = {
            binding.canonical_value: binding
            for binding in tariff.provenance.source_bindings
            if binding.source_id == source_id and binding.claim == "usage_dimension"
        }
        if len(route_bindings) != 1 or set(usage_bindings) != {
            "input_tokens",
            "output_tokens",
        }:
            raise ProviderTariffEvidenceIntegrityError(
                "semantic pricing source lacks one route row and two usage bindings"
            )
        try:
            decoded = html_lib.unescape(source_bytes[source_id].decode())
        except UnicodeDecodeError as exc:
            raise ProviderTariffEvidenceIntegrityError(
                f"retained tariff HTML is not UTF-8: {source_id}"
            ) from exc
        route_value = route_bindings[0].source_value
        input_binding = usage_bindings["input_tokens"]
        output_binding = usage_bindings["output_tokens"]
        row_marker = f"<td>{route_value}</td>"
        price_expression_prefix = "{priceOf!bedrockfoundationmodels/bedrockfoundationmodels!"
        matching_tables: list[tuple[str, str]] = []
        search_from = 0
        while (row_position := decoded.find(row_marker, search_from)) >= 0:
            search_from = row_position + len(row_marker)
            row_start = decoded.rfind("<tr>", 0, row_position)
            row_end = decoded.find("</tr>", row_position)
            if row_start < 0 or row_end < 0:
                continue
            row = decoded[row_start : row_end + len("</tr>")]
            cells = re.findall(r"<td>(.*?)</td>", row, flags=re.DOTALL)
            table_start = decoded.rfind("<table>", 0, row_position)
            header_end = decoded.find("</thead>", table_start, row_position)
            section_start = decoded.rfind("<h2>", 0, table_start)
            if table_start < 0 or header_end < 0 or section_start < 0:
                continue
            header = decoded[table_start:header_end]
            section = decoded[section_start:table_start]
            if (
                len(cells) >= 3
                and cells[1] == f"{price_expression_prefix}{input_binding.source_value}}}"
                and cells[2] == f"{price_expression_prefix}{output_binding.source_value}}}"
                and "Price per 1M input tokens" in header
                and "Price per 1M output tokens" in header
                and "Geo and In-region Cross-region Inference" in section
            ):
                matching_tables.append((header, section))
        if len(matching_tables) != 1:
            raise ProviderTariffEvidenceIntegrityError(
                "semantic pricing row does not use the standard input and output columns"
            )
        for binding in (input_binding, output_binding):
            coordinate = (
                binding.target_meter_source_id,
                binding.target_meter_record_path,
            )
            meter = meter_coordinates.get(coordinate)
            if (
                meter is None
                or binding.source_value != meter.source_record_path.rsplit("/", 1)[-1]
                or meter.source_price_unit != "per_1m_tokens"
                or meter.billing_mode != "geo-cross-region-standard"
            ):
                raise ProviderTariffEvidenceIntegrityError(
                    "semantic pricing binding differs from its billing meter"
                )


def _verify_catalog_tariff_records(
    tariff: ProviderTokenTariff,
    snapshots: dict[str, ProviderTariffSourceSnapshot],
    source_bytes: dict[str, bytes],
) -> None:
    try:
        validated = ProviderTokenTariff.model_validate(tariff.model_dump())
    except ValueError as exc:
        raise ProviderTariffEvidenceIntegrityError("built-in tariff schema is invalid") from exc
    _verify_bedrock_binding_coordinates(validated)
    _verify_source_binding_records(validated, snapshots, source_bytes)
    _verify_billing_meter_records(validated, snapshots, source_bytes)
    _verify_html_semantic_rows(validated, snapshots, source_bytes)


def _azure_billing_mode(deployment_sku: str) -> str:
    modes = {
        "DataZoneBatch": "data-zone-batch",
        "DataZoneStandard": "data-zone-standard",
        "GlobalBatch": "global-batch",
        "GlobalStandard": "global-standard",
        "Standard": "standard",
    }
    try:
        return modes[deployment_sku]
    except KeyError as exc:
        raise ProviderTariffEvidenceIntegrityError(
            "Azure deployment uses an unsupported billing SKU"
        ) from exc


def _azure_source_by_shape(
    *,
    source_snapshots: tuple[ProviderTariffSourceSnapshot, ...],
    source_bytes: dict[str, bytes],
) -> tuple[
    tuple[ProviderTariffSourceSnapshot, dict[str, object]],
    tuple[ProviderTariffSourceSnapshot, dict[str, object]],
    tuple[ProviderTariffSourceSnapshot, dict[str, object]],
]:
    account_sources: list[tuple[ProviderTariffSourceSnapshot, dict[str, object]]] = []
    deployment_sources: list[tuple[ProviderTariffSourceSnapshot, dict[str, object]]] = []
    retail_sources: list[tuple[ProviderTariffSourceSnapshot, dict[str, object]]] = []
    for snapshot in source_snapshots:
        if snapshot.media_type != "application/json":
            raise ProviderTariffEvidenceIntegrityError(
                "Azure tariff evidence must be retained as exact JSON"
            )
        document = _mapping(
            _parse_json_source(snapshot.source_snapshot_digest, source_bytes[snapshot.source_id]),
            label="Azure tariff source",
        )
        parsed_locator = urlsplit(snapshot.source_locator)
        authority = parsed_locator.netloc
        locator_path = parsed_locator.path.casefold()
        if "%" in locator_path or "\\" in locator_path:
            raise ProviderTariffEvidenceIntegrityError(
                "Azure tariff evidence resource path must not use encoded or alternate separators"
            )
        if snapshot.role == "rate_catalog" and "Items" in document:
            if authority != "prices.azure.com" or locator_path != "/api/retail/prices":
                raise ProviderTariffEvidenceIntegrityError(
                    "Azure retail evidence must use the exact canonical authority and "
                    "public retail-prices resource"
                )
            retail_sources.append((snapshot, document))
        elif snapshot.role == "route_definition" and "/deployments/" in locator_path:
            if authority != "management.azure.com" or not re.fullmatch(
                r"/subscriptions/[^/]+/resourcegroups/[^/]+/providers/"
                r"microsoft\.cognitiveservices/accounts/[^/]+/deployments/[^/]+",
                locator_path,
            ):
                raise ProviderTariffEvidenceIntegrityError(
                    "Azure deployment evidence must use the exact canonical authority and "
                    "an exact ARM deployment resource"
                )
            deployment_sources.append((snapshot, document))
        elif snapshot.role == "route_definition":
            if authority != "management.azure.com" or not re.fullmatch(
                r"/subscriptions/[^/]+/resourcegroups/[^/]+/providers/"
                r"microsoft\.cognitiveservices/accounts/[^/]+",
                locator_path,
            ):
                raise ProviderTariffEvidenceIntegrityError(
                    "Azure account evidence must use the exact canonical authority and "
                    "an exact ARM account resource"
                )
            account_sources.append((snapshot, document))
        else:
            raise ProviderTariffEvidenceIntegrityError("Azure tariff source has no known role")
    if len(account_sources) != 1 or len(deployment_sources) != 1 or len(retail_sources) != 1:
        raise ProviderTariffEvidenceIntegrityError(
            "Azure tariff requires one account, one deployment, and one retail source"
        )
    return account_sources[0], deployment_sources[0], retail_sources[0]


def _azure_exact_public_api_query(
    snapshot: ProviderTariffSourceSnapshot,
    *,
    tariff: ProviderTokenTariff,
) -> None:
    query = {parameter.name: parameter.value for parameter in snapshot.public_request_query}
    query_names = set(query)
    api_version = query.get("api-version")
    if api_version is not None:
        match = _AZURE_API_VERSION_PATTERN.fullmatch(api_version)
        try:
            if match is None:
                raise ValueError
            date.fromisoformat(match.group("released_on"))
        except ValueError as exc:
            raise ProviderTariffEvidenceIntegrityError(
                "Azure public API version is not an exact ISO date version"
            ) from exc
    if snapshot.role == "route_definition":
        if query_names != {"api-version"}:
            raise ProviderTariffEvidenceIntegrityError(
                "Azure ARM evidence requires only an explicit public API version"
            )
        return
    if "$filter" not in query_names or query_names - {"$filter", "api-version", "currencyCode"}:
        raise ProviderTariffEvidenceIntegrityError(
            "Azure retail evidence requires a bounded public price query"
        )
    escaped_region = tariff.provenance.route.billing_region.replace("'", "''")
    input_meter, output_meter = tariff.provenance.route.billing_meters
    escaped_input_meter = input_meter.rate_id.replace("'", "''")
    escaped_output_meter = output_meter.rate_id.replace("'", "''")
    expected_filter = (
        "serviceName eq 'Azure OpenAI' and "
        f"armRegionName eq '{escaped_region}' and "
        f"(meterId eq '{escaped_input_meter}' or meterId eq '{escaped_output_meter}')"
    )
    if query["$filter"] != expected_filter or query.get("currencyCode") != "USD":
        raise ProviderTariffEvidenceIntegrityError(
            "Azure retail evidence query differs from its exact service, region, or currency"
        )


def _azure_normalized_sku(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _verify_azure_account_and_deployment(
    *,
    tariff: ProviderTokenTariff,
    account_snapshot: ProviderTariffSourceSnapshot,
    account: dict[str, object],
    deployment_snapshot: ProviderTariffSourceSnapshot,
    deployment: dict[str, object],
) -> str:
    config = tariff.provider_config
    if config.endpoint is None or config.deployment is None or config.model_type is None:
        raise ProviderTariffEvidenceIntegrityError("Azure tariff route is incomplete")
    account_path = urlsplit(account_snapshot.source_locator).path.rstrip("/")
    deployment_path = urlsplit(deployment_snapshot.source_locator).path.rstrip("/")
    expected_deployment_path = f"{account_path}/deployments/{config.deployment}"
    if deployment_path.casefold() != expected_deployment_path.casefold():
        raise ProviderTariffEvidenceIntegrityError(
            "Azure deployment evidence belongs to a different ARM account resource"
        )
    account_name = _text(account.get("name"), label="Azure account name")
    if account_name.casefold() != account_path.rsplit("/", 1)[-1].casefold():
        raise ProviderTariffEvidenceIntegrityError(
            "Azure account evidence name differs from its ARM resource path"
        )
    account_properties = _mapping(account.get("properties"), label="Azure account properties")
    observed_endpoint = _text(
        account_properties.get("endpoint"),
        label="Azure account endpoint",
    )
    if observed_endpoint.rstrip("/") != config.endpoint.rstrip("/"):
        raise ProviderTariffEvidenceIntegrityError(
            "Azure account evidence names a different provider endpoint"
        )
    if _text(account.get("location"), label="Azure account location") != (
        tariff.provenance.route.billing_region
    ):
        raise ProviderTariffEvidenceIntegrityError(
            "Azure account location differs from the billing region"
        )
    if _text(deployment.get("name"), label="Azure deployment name") != config.deployment:
        raise ProviderTariffEvidenceIntegrityError(
            "Azure deployment evidence names a different deployment"
        )
    etag = _text(deployment.get("etag"), label="Azure deployment ETag")
    if not etag:
        raise ProviderTariffEvidenceIntegrityError("Azure deployment ETag is absent")
    deployment_properties = _mapping(
        deployment.get("properties"),
        label="Azure deployment properties",
    )
    model = _mapping(deployment_properties.get("model"), label="Azure deployment model")
    if _text(model.get("format"), label="Azure deployment model format") != "OpenAI":
        raise ProviderTariffEvidenceIntegrityError("Azure deployment is not an OpenAI model")
    model_name = _text(model.get("name"), label="Azure deployment model name")
    if model_name not in {config.model, config.model_type}:
        raise ProviderTariffEvidenceIntegrityError(
            "Azure deployment model differs from the canonical provider model"
        )
    _text(model.get("version"), label="Azure deployment model version")
    upgrade_option = _text(
        deployment_properties.get("versionUpgradeOption"),
        label="Azure deployment version upgrade option",
    )
    if upgrade_option != "NoAutoUpgrade":
        raise ProviderTariffEvidenceIntegrityError(
            "Azure deployment must disable automatic model upgrades"
        )
    deployment_sku = _mapping(deployment.get("sku"), label="Azure deployment SKU")
    sku_name = _text(deployment_sku.get("name"), label="Azure deployment SKU name")
    if _azure_billing_mode(sku_name) != tariff.provenance.route.billing_mode:
        raise ProviderTariffEvidenceIntegrityError(
            "Azure deployment SKU differs from the tariff billing mode"
        )
    return sku_name


def _verify_azure_route_binding_coordinates(
    *,
    tariff: ProviderTokenTariff,
    account_source_id: str,
    deployment_source_id: str,
) -> None:
    required_coordinates = {
        (account_source_id, "/location"),
        (account_source_id, "/name"),
        (account_source_id, "/properties/endpoint"),
        (deployment_source_id, "/etag"),
        (deployment_source_id, "/name"),
        (deployment_source_id, "/properties/model/format"),
        (deployment_source_id, "/properties/model/name"),
        (deployment_source_id, "/properties/model/version"),
        (deployment_source_id, "/properties/versionUpgradeOption"),
        (deployment_source_id, "/sku/name"),
    }
    observed_coordinates = {
        (binding.source_id, binding.source_record_path)
        for binding in tariff.provenance.source_bindings
        if binding.claim == "route_identity"
    }
    if not required_coordinates.issubset(observed_coordinates):
        raise ProviderTariffEvidenceIntegrityError(
            "Azure tariff omits a required account or deployment route binding"
        )


def _verify_azure_retail_meter(
    *,
    tariff: ProviderTokenTariff,
    document: dict[str, object],
    meter: ProviderTariffBillingMeter,
    deployment_sku: str,
) -> None:
    record = _mapping(
        _json_pointer(document, meter.source_record_path),
        label="Azure retail meter",
    )
    if _text(record.get("currencyCode"), label="Azure retail currency") != "USD":
        raise ProviderTariffEvidenceIntegrityError("Azure retail meter is not priced in USD")
    if _text(record.get("serviceName"), label="Azure retail service") != "Azure OpenAI":
        raise ProviderTariffEvidenceIntegrityError(
            "Azure retail meter belongs to a different service"
        )
    if _decimal(record.get("tierMinimumUnits"), label="Azure retail tier minimum") != 0:
        raise ProviderTariffEvidenceIntegrityError("Azure retail meter is not the base tier")
    if _text(record.get("type"), label="Azure retail meter type") != "Consumption":
        raise ProviderTariffEvidenceIntegrityError("Azure retail meter is not standard consumption")
    if record.get("isPrimaryMeterRegion") is not True:
        raise ProviderTariffEvidenceIntegrityError(
            "Azure retail meter is not the primary billing record"
        )
    if record.get("savingsPlan") not in (None, []):
        raise ProviderTariffEvidenceIntegrityError(
            "Azure retail meter unexpectedly contains savings-plan pricing"
        )
    if _text(record.get("armRegionName"), label="Azure retail region") != (meter.billing_region):
        raise ProviderTariffEvidenceIntegrityError(
            "Azure retail meter region differs from the tariff"
        )
    if _text(record.get("meterId"), label="Azure retail meter ID") != meter.rate_id:
        raise ProviderTariffEvidenceIntegrityError("Azure retail meter ID differs from tariff")
    if _text(record.get("skuId"), label="Azure retail SKU ID") != meter.sku_id:
        raise ProviderTariffEvidenceIntegrityError("Azure retail SKU ID differs from tariff")
    retail_sku = _text(record.get("armSkuName"), label="Azure retail ARM SKU")
    if _azure_normalized_sku(retail_sku) != _azure_normalized_sku(deployment_sku):
        raise ProviderTariffEvidenceIntegrityError(
            "Azure deployment SKU is not joined to the retail meter"
        )
    retail_price = _decimal(record.get("retailPrice"), label="Azure retail price")
    unit_price = _decimal(record.get("unitPrice"), label="Azure retail unit price")
    if retail_price != meter.source_price_usd or unit_price != meter.source_price_usd:
        raise ProviderTariffEvidenceIntegrityError("Azure retail price differs from tariff")
    unit = _text(record.get("unitOfMeasure"), label="Azure retail unit")
    accepted_units = {
        "per_1k_tokens": {"1K", "1K Tokens", "1K tokens"},
        "per_1m_tokens": {"1M", "1M Tokens", "1M tokens"},
    }
    if unit not in accepted_units[meter.source_price_unit]:
        raise ProviderTariffEvidenceIntegrityError("Azure retail unit differs from tariff")
    effective_on = _azure_effective_date(record)
    if effective_on != meter.effective_on:
        raise ProviderTariffEvidenceIntegrityError(
            "Azure retail effective date differs from tariff"
        )
    expected_usage = "input" if meter.usage_dimension == "input_tokens" else "output"
    meter_name = _text(record.get("meterName"), label="Azure retail meter name")
    meter_words = set(re.findall(r"[a-z0-9]+", meter_name.casefold()))
    opposite_usage = "output" if expected_usage == "input" else "input"
    discounted_or_nonstandard_words = {
        "batch",
        "cache",
        "cached",
        "caching",
        "provisioned",
        "training",
    }
    if (
        expected_usage not in meter_words
        or opposite_usage in meter_words
        or meter_words & discounted_or_nonstandard_words
        or {"fine", "tuning"}.issubset(meter_words)
    ):
        raise ProviderTariffEvidenceIntegrityError(
            "Azure retail meter is not the exact standard token usage dimension"
        )
    profile_key = (tariff.provider_config.model, tariff.provenance.route.billing_mode)
    try:
        product_name, sku_name, input_meter_name, output_meter_name = _AZURE_RETAIL_EXACT_LABELS[
            profile_key
        ]
    except KeyError as exc:
        raise ProviderTariffEvidenceIntegrityError(
            "Azure route has no registered exact retail label profile"
        ) from exc
    expected_meter_name = (
        input_meter_name if meter.usage_dimension == "input_tokens" else output_meter_name
    )
    observed_labels = (
        _text(record.get("productName"), label="Azure retail productName"),
        _text(record.get("skuName"), label="Azure retail skuName"),
        meter_name,
    )
    if observed_labels != (product_name, sku_name, expected_meter_name):
        raise ProviderTariffEvidenceIntegrityError(
            "Azure retail meter differs from the registered exact deployment model labels"
        )


def _verify_azure_tariff_records(
    tariff: ProviderTokenTariff,
    snapshots: dict[str, ProviderTariffSourceSnapshot],
    source_bytes: dict[str, bytes],
) -> None:
    account_source, deployment_source, retail_source = _azure_source_by_shape(
        source_snapshots=tariff.provenance.source_snapshots,
        source_bytes=source_bytes,
    )
    for snapshot, _ in (account_source, deployment_source, retail_source):
        _azure_exact_public_api_query(
            snapshot,
            tariff=tariff,
        )
    deployment_sku = _verify_azure_account_and_deployment(
        tariff=tariff,
        account_snapshot=account_source[0],
        account=account_source[1],
        deployment_snapshot=deployment_source[0],
        deployment=deployment_source[1],
    )
    _verify_azure_route_binding_coordinates(
        tariff=tariff,
        account_source_id=account_source[0].source_id,
        deployment_source_id=deployment_source[0].source_id,
    )
    retail_document = retail_source[1]
    if _text(retail_document.get("BillingCurrency"), label="Azure billing currency") != "USD":
        raise ProviderTariffEvidenceIntegrityError(
            "Azure retail response uses a different billing currency"
        )
    items = retail_document.get("Items")
    if not isinstance(items, list) or len(items) != 2:
        raise ProviderTariffEvidenceIntegrityError(
            "Azure retail evidence must contain exactly two token meters"
        )
    if retail_document.get("NextPageLink") not in (None, ""):
        raise ProviderTariffEvidenceIntegrityError(
            "Azure retail evidence has an unverified pagination continuation"
        )
    if "Count" in retail_document and retail_document["Count"] != len(items):
        raise ProviderTariffEvidenceIntegrityError("Azure retail item count is inconsistent")
    retail_source_id = retail_source[0].source_id
    if any(meter.source_id != retail_source_id for meter in tariff.provenance.route.billing_meters):
        raise ProviderTariffEvidenceIntegrityError(
            "Azure billing meters must come from the exact retail response"
        )
    for meter in tariff.provenance.route.billing_meters:
        _verify_azure_retail_meter(
            tariff=tariff,
            document=retail_document,
            meter=meter,
            deployment_sku=deployment_sku,
        )
    _verify_source_binding_records(tariff, snapshots, source_bytes)


def _artifact_bytes_for_snapshot(
    snapshot: ProviderTariffSourceSnapshot,
    evidence_artifacts: Mapping[str, bytes] | None,
) -> bytes:
    artifact = snapshot.retained_artifact
    if artifact.storage_kind == "package_resource":
        package, relative_path = artifact.locator.split(":", 1)
        try:
            return resources.files(package).joinpath(relative_path).read_bytes()
        except (FileNotFoundError, ModuleNotFoundError) as exc:
            raise ProviderTariffEvidenceIntegrityError(
                f"retained tariff artifact is unavailable: {artifact.locator}"
            ) from exc
    if evidence_artifacts is None:
        raise ProviderTariffEvidenceIntegrityError(
            "HTTPS tariff evidence requires explicit retained artifact bytes"
        )
    try:
        artifact_bytes = evidence_artifacts[snapshot.source_id]
    except KeyError as exc:
        raise ProviderTariffEvidenceIntegrityError(
            f"retained tariff artifact bytes are absent: {snapshot.source_id}"
        ) from exc
    if not isinstance(artifact_bytes, bytes):
        raise ProviderTariffEvidenceIntegrityError(
            "tariff artifact fetcher must return exact bytes"
        )
    return artifact_bytes


def _verified_tariff_source_bytes(
    source_snapshots: tuple[ProviderTariffSourceSnapshot, ...],
    *,
    evidence_artifacts: Mapping[str, bytes] | None,
) -> tuple[dict[str, bytes], tuple[ProviderTariffVerifiedSource, ...]]:
    external_source_ids = {
        source.source_id
        for source in source_snapshots
        if source.retained_artifact.storage_kind == "https"
    }
    provided_source_ids = set(evidence_artifacts or {})
    if provided_source_ids != external_source_ids:
        raise ProviderTariffEvidenceIntegrityError(
            "explicit tariff artifact source set differs from provenance"
        )
    source_bytes: dict[str, bytes] = {}
    verified_sources: list[ProviderTariffVerifiedSource] = []
    for source in source_snapshots:
        artifact_bytes = _artifact_bytes_for_snapshot(source, evidence_artifacts)
        decoded = _decode_verified_artifact(
            locator=source.retained_artifact.locator,
            artifact_bytes=artifact_bytes,
            artifact_digest=source.retained_artifact.artifact_digest,
            artifact_encoding=source.retained_artifact.content_encoding,
            source_digest=source.source_snapshot_digest,
        )
        source_bytes[source.source_id] = decoded
        verified_sources.append(
            ProviderTariffVerifiedSource(
                source_id=source.source_id,
                artifact_digest=source.retained_artifact.artifact_digest,
                source_snapshot_digest=source.source_snapshot_digest,
                artifact_size_bytes=len(artifact_bytes),
                decoded_size_bytes=len(decoded),
            )
        )
    return source_bytes, tuple(verified_sources)


def _azure_price_unit(value: object) -> Literal["per_1k_tokens", "per_1m_tokens"]:
    unit = _text(value, label="Azure retail unit")
    if unit in {"1K", "1K Tokens", "1K tokens"}:
        return "per_1k_tokens"
    if unit in {"1M", "1M Tokens", "1M tokens"}:
        return "per_1m_tokens"
    raise ProviderTariffEvidenceIntegrityError("Azure retail unit is not token based")


def _azure_effective_date(record: dict[str, object]) -> date:
    effective = _text(
        record.get("effectiveStartDate"),
        label="Azure retail effective date",
    )
    if _AZURE_EFFECTIVE_TIMESTAMP_PATTERN.fullmatch(effective) is None:
        raise ProviderTariffEvidenceIntegrityError(
            "Azure retail effective date must be a complete UTC timestamp"
        )
    try:
        parsed = datetime.fromisoformat(effective.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ProviderTariffEvidenceIntegrityError(
            "Azure retail effective date must be a complete UTC timestamp"
        ) from exc
    if any((parsed.hour, parsed.minute, parsed.second, parsed.microsecond)):
        raise ProviderTariffEvidenceIntegrityError(
            "Azure retail effective date must be a complete UTC timestamp at midnight"
        )
    return parsed.date()


def _azure_route_bindings_from_document(
    *,
    source: ProviderTariffSourceSnapshot,
    document: dict[str, object],
    paths: tuple[str, ...],
    canonical_model: str,
) -> list[ProviderTariffSourceBinding]:
    return [
        ProviderTariffSourceBinding(
            claim="route_identity",
            source_id=source.source_id,
            source_record_path=path,
            source_value=_text(
                _json_pointer(document, path),
                label=f"Azure route value at {path}",
            ),
            canonical_value=canonical_model,
        )
        for path in paths
    ]


def azure_provider_cost_meter_from_evidence(
    *,
    provider_config: ProviderConfig,
    source_snapshots: tuple[ProviderTariffSourceSnapshot, ...],
    evidence_artifacts: Mapping[str, bytes],
    verified_on: date,
    input_overhead_tokens: int = 8192,
    price_ceiling: TokenPriceCeiling | None = None,
) -> ProviderCostMeter:
    """Derive and verify one Azure provider meter from exact retained responses.

    This boundary is pure and offline. It derives billing coordinates, meters, prices, and source
    bindings from one account response, one deployment response, and one bounded Retail Prices
    response. The resulting claim still passes through the registered evidence verifier before it
    can become a paid provider meter.
    """
    config = ProviderConfig.model_validate(provider_config.model_dump())
    if config.kind is not ProviderKind.AZURE_OPENAI:
        raise ValueError("Azure tariff evidence requires an Azure OpenAI provider route")

    frozen_sources = tuple(
        sorted(
            (
                ProviderTariffSourceSnapshot.model_validate(source.model_dump())
                for source in source_snapshots
            ),
            key=lambda source: source.source_id,
        )
    )
    source_ids = tuple(source.source_id for source in frozen_sources)
    if len(frozen_sources) != 3 or len(set(source_ids)) != 3:
        raise ProviderTariffEvidenceIntegrityError(
            "Azure tariff requires exactly three unique retained sources"
        )
    if any(source.retained_artifact.storage_kind != "https" for source in frozen_sources):
        raise ProviderTariffEvidenceIntegrityError(
            "Azure tariff factory requires explicitly supplied retained artifacts"
        )
    source_bytes, _ = _verified_tariff_source_bytes(
        frozen_sources,
        evidence_artifacts=evidence_artifacts,
    )
    account_source, deployment_source, retail_source = _azure_source_by_shape(
        source_snapshots=frozen_sources,
        source_bytes=source_bytes,
    )
    account_snapshot, account = account_source
    deployment_snapshot, deployment = deployment_source
    retail_snapshot, retail = retail_source

    billing_region = _text(account.get("location"), label="Azure account location")
    deployment_sku = _text(
        _mapping(deployment.get("sku"), label="Azure deployment SKU").get("name"),
        label="Azure deployment SKU name",
    )
    billing_mode = _azure_billing_mode(deployment_sku)
    try:
        _, _, input_meter_name, output_meter_name = _AZURE_RETAIL_EXACT_LABELS[
            (config.model, billing_mode)
        ]
    except KeyError as exc:
        raise ProviderTariffEvidenceIntegrityError(
            "Azure route has no registered exact retail label profile"
        ) from exc

    raw_items = retail.get("Items")
    if not isinstance(raw_items, list) or len(raw_items) != 2:
        raise ProviderTariffEvidenceIntegrityError(
            "Azure retail evidence must contain exactly two token meters"
        )
    expected_names = {
        "input_tokens": input_meter_name,
        "output_tokens": output_meter_name,
    }
    indexed_records: dict[str, tuple[int, dict[str, object]]] = {}
    for index, raw_item in enumerate(cast("list[object]", raw_items)):
        record = _mapping(raw_item, label="Azure retail meter")
        meter_name = _text(record.get("meterName"), label="Azure retail meter name")
        matching_dimensions = [
            dimension
            for dimension, expected_name in expected_names.items()
            if meter_name == expected_name
        ]
        if len(matching_dimensions) != 1 or matching_dimensions[0] in indexed_records:
            raise ProviderTariffEvidenceIntegrityError(
                "Azure retail evidence lacks one exact input and one exact output meter"
            )
        indexed_records[matching_dimensions[0]] = (index, record)
    if set(indexed_records) != set(expected_names):
        raise ProviderTariffEvidenceIntegrityError(
            "Azure retail evidence lacks one exact input and one exact output meter"
        )

    meters: list[ProviderTariffBillingMeter] = []
    bindings = _azure_route_bindings_from_document(
        source=account_snapshot,
        document=account,
        paths=("/location", "/name", "/properties/endpoint"),
        canonical_model=config.model,
    )
    bindings.extend(
        _azure_route_bindings_from_document(
            source=deployment_snapshot,
            document=deployment,
            paths=(
                "/etag",
                "/name",
                "/properties/model/format",
                "/properties/model/name",
                "/properties/model/version",
                "/properties/versionUpgradeOption",
                "/sku/name",
            ),
            canonical_model=config.model,
        )
    )
    for dimension in ("input_tokens", "output_tokens"):
        index, record = indexed_records[dimension]
        record_path = f"/Items/{index}"
        meters.append(
            ProviderTariffBillingMeter(
                usage_dimension=dimension,
                source_id=retail_snapshot.source_id,
                source_record_path=record_path,
                sku_id=_text(record.get("skuId"), label="Azure retail SKU ID"),
                rate_id=_text(record.get("meterId"), label="Azure retail meter ID"),
                billing_region=billing_region,
                billing_mode=billing_mode,
                effective_on=_azure_effective_date(record),
                source_price_usd=_decimal(
                    record.get("retailPrice"),
                    label="Azure retail price",
                ),
                source_price_unit=_azure_price_unit(record.get("unitOfMeasure")),
            )
        )
        bindings.extend(
            (
                ProviderTariffSourceBinding(
                    claim="route_identity",
                    source_id=retail_snapshot.source_id,
                    source_record_path=f"{record_path}/productName",
                    source_value=_text(
                        record.get("productName"),
                        label="Azure retail productName",
                    ),
                    canonical_value=config.model,
                    target_meter_source_id=retail_snapshot.source_id,
                    target_meter_record_path=record_path,
                ),
                ProviderTariffSourceBinding(
                    claim="usage_dimension",
                    source_id=retail_snapshot.source_id,
                    source_record_path=f"{record_path}/meterName",
                    source_value=_text(
                        record.get("meterName"),
                        label="Azure retail meterName",
                    ),
                    canonical_value=dimension,
                    target_meter_source_id=retail_snapshot.source_id,
                    target_meter_record_path=record_path,
                ),
            )
        )

    effective_dates = {meter.effective_on for meter in meters}
    if len(effective_dates) != 1:
        raise ProviderTariffEvidenceIntegrityError(
            "Azure input and output meters have different effective dates"
        )
    sorted_bindings = tuple(
        sorted(
            bindings,
            key=lambda binding: (
                binding.claim,
                binding.canonical_value,
                binding.source_id,
                binding.source_record_path,
                binding.source_value,
                binding.target_meter_source_id or "",
                binding.target_meter_record_path or "",
            ),
        )
    )
    route = ProviderTariffRoute(
        provider_config=config,
        billing_region=billing_region,
        billing_mode=billing_mode,
        billing_meters=(meters[0], meters[1]),
    )
    provenance = ProviderTariffProvenance(
        source_snapshots=frozen_sources,
        source_bindings=sorted_bindings,
        verified_on=verified_on,
        effective_on=meters[0].effective_on,
        currency="USD",
        price_unit="per_1m_tokens",
        route=route,
    )
    resolved_price = (
        provenance.token_price_floor()
        if price_ceiling is None
        else TokenPriceCeiling.model_validate(price_ceiling.model_dump())
    )
    tariff = ProviderTokenTariff(
        provider_config=config,
        price=resolved_price,
        provenance=provenance,
    )
    return provider_cost_meter(
        tariff,
        input_overhead_tokens=input_overhead_tokens,
        evidence_artifacts=evidence_artifacts,
    )


def verify_provider_tariff_evidence(
    tariff: ProviderTokenTariff,
    *,
    evidence_artifacts: Mapping[str, bytes] | None = None,
) -> ProviderTariffEvidenceReceipt:
    """Offline-verify injected retained bytes and return a deterministic policy receipt."""
    snapshot = ProviderTokenTariff.model_validate(tariff.model_dump())
    snapshots = {source.source_id: source for source in snapshot.provenance.source_snapshots}
    source_bytes, verified_sources = _verified_tariff_source_bytes(
        snapshot.provenance.source_snapshots,
        evidence_artifacts=evidence_artifacts,
    )
    if snapshot.provider_config.kind is ProviderKind.BEDROCK:
        route_matches = [
            catalog_tariff
            for catalog_tariff in _CATALOG
            if catalog_tariff.provider_config == snapshot.provider_config
        ]
        if len(route_matches) != 1:
            raise ProviderTariffEvidenceIntegrityError(
                "Bedrock tariff route has no unique built-in evidence profile"
            )
        expected_snapshots = {
            source.source_id: source for source in route_matches[0].provenance.source_snapshots
        }
        if snapshots != expected_snapshots:
            raise ProviderTariffEvidenceIntegrityError(
                "Bedrock tariff source set differs from its built-in route evidence"
            )
        for source_id, source in snapshots.items():
            _verify_source_publication_metadata(source, source_bytes[source_id])
        _verify_catalog_tariff_records(snapshot, snapshots, source_bytes)
        profile = "aws_bedrock_public_catalog_v1"
    elif snapshot.provider_config.kind is ProviderKind.AZURE_OPENAI:
        _verify_azure_tariff_records(snapshot, snapshots, source_bytes)
        profile = "azure_retail_arm_v1"
    else:
        raise ProviderTariffEvidenceIntegrityError(
            "provider route has no registered tariff evidence verifier"
        )
    return ProviderTariffEvidenceReceipt(
        verifier_profile=profile,
        verifier_digest=provider_tariff_evidence_verifier_digest(profile),
        tariff_claim_digest=snapshot.digest,
        verified_sources=verified_sources,
        validated_records=provider_tariff_validated_records(snapshot.provenance),
    )


def verify_catalog_provider_tariff_sources() -> tuple[str, ...]:
    """Verify and return every unique packaged source ID in the built-in catalog."""
    snapshots: dict[str, ProviderTariffSourceSnapshot] = {}
    for tariff in _CATALOG:
        for snapshot in tariff.provenance.source_snapshots:
            prior = snapshots.get(snapshot.source_id)
            if prior is not None and prior != snapshot:
                raise ProviderTariffEvidenceIntegrityError(
                    f"conflicting built-in tariff source metadata: {snapshot.source_id}"
                )
            snapshots[snapshot.source_id] = snapshot
    expected_snapshots = {snapshot.source_id: snapshot for snapshot in _BUILTIN_SOURCE_SNAPSHOTS}
    if snapshots != expected_snapshots:
        raise ProviderTariffEvidenceIntegrityError(
            "built-in tariff source metadata differs from the retained source registry"
        )
    decoded_sources: dict[str, bytes] = {}
    for source_id in sorted(snapshots):
        snapshot = snapshots[source_id]
        artifact = snapshot.retained_artifact
        if artifact.storage_kind != "package_resource":
            raise ProviderTariffEvidenceIntegrityError(
                f"built-in tariff source is not retained in the package: {source_id}"
            )
        decoded_sources[source_id] = _verify_packaged_source_bytes(
            locator=artifact.locator,
            artifact_digest=artifact.artifact_digest,
            artifact_encoding=artifact.content_encoding,
            source_digest=snapshot.source_snapshot_digest,
        )
        _verify_source_publication_metadata(snapshot, decoded_sources[source_id])
    for tariff in _CATALOG:
        _verify_catalog_tariff_records(tariff, snapshots, decoded_sources)
    return tuple(sorted(snapshots))


def catalog_provider_token_tariffs() -> tuple[ProviderTokenTariff, ...]:
    """Return defensive copies of the currently audited built-in tariff snapshots."""
    verify_catalog_provider_tariff_sources()
    return tuple(ProviderTokenTariff.model_validate(item.model_dump()) for item in _CATALOG)


def catalog_provider_token_tariff(provider_config: ProviderConfig) -> ProviderTokenTariff:
    """Resolve an exact built-in route or reject an unaudited route."""
    verify_catalog_provider_tariff_sources()
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
        source_snapshots=record.source_snapshots,
        source_bindings=record.source_bindings,
        verified_on=_CATALOG_VERIFIED_ON,
        effective_on=record.effective_on,
        currency="USD",
        price_unit="per_1m_tokens",
        billing_region="us-east-1",
        billing_mode=record.billing_mode,
        billing_meters=record.billing_meters,
    )


# The ``us.`` inference profiles use Bedrock's Geo Cross-Region rates, not the lower ``global.``
# profile rates. All catalog objects are derived from the exact source snapshot rows above.
_CATALOG = tuple(_bedrock_tariff(record=record) for record in _BEDROCK_CATALOG_RECORDS)
