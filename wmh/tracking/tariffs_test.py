"""Tests for immutable, route-bound provider tariff snapshots."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from decimal import Decimal
from typing import Literal, cast

import pytest
from pydantic import ValidationError

import wmh.tracking.tariffs as tariffs_module
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.tracking.budget import (
    ProviderCostMeter,
    ProviderTariffBillingMeter,
    ProviderTariffEvidenceReceipt,
    ProviderTariffProvenance,
    ProviderTariffPublicQueryParameter,
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
from wmh.tracking.pricing import price_for
from wmh.tracking.tariffs import (
    ProviderTokenTariff,
    azure_provider_cost_meter_from_evidence,
    catalog_provider_token_tariff,
    catalog_provider_token_tariffs,
    provider_cost_meter,
    verify_catalog_provider_tariff_sources,
    verify_provider_tariff_evidence,
)


def test_bedrock_catalog_integrity_guard_rejects_embedded_price_drift() -> None:
    records = tariffs_module._BEDROCK_CATALOG_RECORDS
    tariffs_module._verify_bedrock_catalog_integrity(records)
    mutated = (
        records[0].model_copy(update={"input_usd": "0.000001"}),
        *records[1:],
    )

    with pytest.raises(RuntimeError, match="embedded Bedrock tariff record changed"):
        tariffs_module._verify_bedrock_catalog_integrity(mutated)


def test_bedrock_catalog_integrity_guard_rejects_record_set_drift() -> None:
    with pytest.raises(RuntimeError, match="record set changed"):
        tariffs_module._verify_bedrock_catalog_integrity(
            tariffs_module._BEDROCK_CATALOG_RECORDS[:-1]
        )


def _test_source_snapshot() -> ProviderTariffSourceSnapshot:
    return ProviderTariffSourceSnapshot(
        source_id="synthetic-rate-catalog",
        role="rate_catalog",
        source_locator="https://example.test/immutable-price-record",
        source_snapshot_digest="sha256:" + "a" * 64,
        media_type="application/json",
        content_encoding="identity",
        retained_artifact=ProviderTariffRetainedArtifact(
            storage_kind="https",
            locator="https://example.test/evidence/sha256-a.json.gz",
            artifact_digest="sha256:" + "b" * 64,
            content_encoding="gzip",
        ),
    )


def _test_source_bindings(config: ProviderConfig) -> tuple[ProviderTariffSourceBinding, ...]:
    return (
        ProviderTariffSourceBinding(
            claim="route_identity",
            source_id="synthetic-rate-catalog",
            source_record_path="/input/model",
            source_value=config.model,
            canonical_value=config.model,
            target_meter_source_id="synthetic-rate-catalog",
            target_meter_record_path="/input",
        ),
        ProviderTariffSourceBinding(
            claim="route_identity",
            source_id="synthetic-rate-catalog",
            source_record_path="/output/model",
            source_value=config.model,
            canonical_value=config.model,
            target_meter_source_id="synthetic-rate-catalog",
            target_meter_record_path="/output",
        ),
        ProviderTariffSourceBinding(
            claim="usage_dimension",
            source_id="synthetic-rate-catalog",
            source_record_path="/input",
            source_value="Input tokens",
            canonical_value="input_tokens",
            target_meter_source_id="synthetic-rate-catalog",
            target_meter_record_path="/input",
        ),
        ProviderTariffSourceBinding(
            claim="usage_dimension",
            source_id="synthetic-rate-catalog",
            source_record_path="/output",
            source_value="Output tokens",
            canonical_value="output_tokens",
            target_meter_source_id="synthetic-rate-catalog",
            target_meter_record_path="/output",
        ),
    )


def _test_billing_meter(
    usage_dimension: str,
    label: str,
    *,
    billing_region: str = "eastus2",
    billing_mode: str = "global-standard",
) -> ProviderTariffBillingMeter:
    return ProviderTariffBillingMeter.model_validate(
        {
            "usage_dimension": usage_dimension,
            "source_id": "synthetic-rate-catalog",
            "source_record_path": f"/{label}",
            "sku_id": f"{label}-sku",
            "rate_id": f"{label}-rate",
            "billing_region": billing_region,
            "billing_mode": billing_mode,
            "effective_on": date(2026, 7, 1),
            "source_price_usd": "0.001",
            "source_price_unit": "per_1m_tokens",
        }
    )


def _test_billing_meters(
    *,
    billing_region: str = "eastus2",
    billing_mode: str = "global-standard",
) -> tuple[ProviderTariffBillingMeter, ProviderTariffBillingMeter]:
    return (
        _test_billing_meter(
            "input_tokens",
            "input",
            billing_region=billing_region,
            billing_mode=billing_mode,
        ),
        _test_billing_meter(
            "output_tokens",
            "output",
            billing_region=billing_region,
            billing_mode=billing_mode,
        ),
    )


def _test_receipt(
    *,
    config: ProviderConfig,
    price: TokenPriceCeiling,
    provenance: ProviderTariffProvenance,
) -> ProviderTariffEvidenceReceipt:
    profile = (
        "aws_bedrock_public_catalog_v1"
        if config.kind is ProviderKind.BEDROCK
        else "azure_retail_arm_v1"
    )
    return ProviderTariffEvidenceReceipt(
        verifier_profile=profile,
        verifier_digest=provider_tariff_evidence_verifier_digest(profile),
        tariff_claim_digest=provider_tariff_claim_digest(
            provider_config=config,
            price=price,
            provenance=provenance,
        ),
        verified_sources=tuple(
            ProviderTariffVerifiedSource(
                source_id=source.source_id,
                artifact_digest=source.retained_artifact.artifact_digest,
                source_snapshot_digest=source.source_snapshot_digest,
                artifact_size_bytes=1,
                decoded_size_bytes=1,
            )
            for source in provenance.source_snapshots
        ),
        validated_records=provider_tariff_validated_records(provenance),
    )


def _azure_snapshot(
    *,
    source_id: str,
    role: Literal["rate_catalog", "route_definition", "semantic_mapping"],
    source_locator: str,
    document: dict[str, object],
    query: tuple[ProviderTariffPublicQueryParameter, ...],
) -> tuple[ProviderTariffSourceSnapshot, bytes]:
    retained = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    digest = "sha256:" + hashlib.sha256(retained).hexdigest()
    return (
        ProviderTariffSourceSnapshot(
            source_id=source_id,
            role=role,
            source_locator=source_locator,
            source_snapshot_digest=digest,
            media_type="application/json",
            content_encoding="identity",
            public_request_query=query,
            retained_artifact=ProviderTariffRetainedArtifact(
                storage_kind="https",
                locator=f"https://example.test/evidence/{digest.removeprefix('sha256:')}.json",
                artifact_digest=digest,
                content_encoding="identity",
            ),
        ),
        retained,
    )


def _azure_tariff_fixture(
    config: ProviderConfig,
) -> tuple[ProviderTokenTariff, dict[str, bytes]]:
    account_document: dict[str, object] = {
        "location": "eastus2",
        "name": "example-resource",
        "properties": {"endpoint": "https://example-resource.openai.azure.com/"},
        "sku": {"name": "S0"},
    }
    deployment_document: dict[str, object] = {
        "etag": "opaque-route-version",
        "name": config.deployment,
        "properties": {
            "model": {
                "format": "OpenAI",
                "name": config.model,
                "version": "2026-07-01",
            },
            "versionUpgradeOption": "NoAutoUpgrade",
        },
        "sku": {"name": "GlobalStandard"},
    }
    retail_items: list[dict[str, object]] = []
    for usage, price, sku_id, meter_id in (
        ("Input", "1", "input-sku", "input-rate"),
        ("Output", "2", "output-sku", "output-rate"),
    ):
        retail_items.append(
            {
                "armRegionName": "eastus2",
                "armSkuName": "GlobalStandard",
                "currencyCode": "USD",
                "effectiveStartDate": "2026-07-01T00:00:00Z",
                "isPrimaryMeterRegion": True,
                "meterId": meter_id,
                "meterName": f"gpt-5.5 {usage} Tokens",
                "productName": "Azure OpenAI gpt-5.5",
                "retailPrice": price,
                "serviceName": "Azure OpenAI",
                "skuId": sku_id,
                "skuName": "gpt-5.5 GlobalStandard",
                "tierMinimumUnits": 0,
                "type": "Consumption",
                "unitOfMeasure": "1M Tokens",
                "unitPrice": price,
            }
        )
    retail_document: dict[str, object] = {
        "BillingCurrency": "USD",
        "Count": 2,
        "Items": retail_items,
        "NextPageLink": None,
    }
    arm_query = (ProviderTariffPublicQueryParameter(name="api-version", value="2024-10-01"),)
    retail_query = (
        ProviderTariffPublicQueryParameter(
            name="$filter",
            value=(
                "serviceName eq 'Azure OpenAI' and armRegionName eq 'eastus2' and "
                "(meterId eq 'input-rate' or meterId eq 'output-rate')"
            ),
        ),
        ProviderTariffPublicQueryParameter(name="currencyCode", value="USD"),
    )
    account, account_bytes = _azure_snapshot(
        source_id="azure-account-route",
        role="route_definition",
        source_locator=(
            "https://management.azure.com/subscriptions/example/resourceGroups/example/"
            "providers/Microsoft.CognitiveServices/accounts/example-resource"
        ),
        document=account_document,
        query=arm_query,
    )
    deployment, deployment_bytes = _azure_snapshot(
        source_id="azure-deployment-route",
        role="route_definition",
        source_locator=(
            "https://management.azure.com/subscriptions/example/resourceGroups/example/"
            "providers/Microsoft.CognitiveServices/accounts/example-resource/deployments/"
            f"{config.deployment}"
        ),
        document=deployment_document,
        query=arm_query,
    )
    retail, retail_bytes = _azure_snapshot(
        source_id="azure-retail-price",
        role="rate_catalog",
        source_locator="https://prices.azure.com/api/retail/prices",
        document=retail_document,
        query=retail_query,
    )
    meters = (
        ProviderTariffBillingMeter(
            usage_dimension="input_tokens",
            source_id=retail.source_id,
            source_record_path="/Items/0",
            sku_id="input-sku",
            rate_id="input-rate",
            billing_region="eastus2",
            billing_mode="global-standard",
            effective_on=date(2026, 7, 1),
            source_price_usd=Decimal(1),
            source_price_unit="per_1m_tokens",
        ),
        ProviderTariffBillingMeter(
            usage_dimension="output_tokens",
            source_id=retail.source_id,
            source_record_path="/Items/1",
            sku_id="output-sku",
            rate_id="output-rate",
            billing_region="eastus2",
            billing_mode="global-standard",
            effective_on=date(2026, 7, 1),
            source_price_usd=Decimal(2),
            source_price_unit="per_1m_tokens",
        ),
    )
    bindings = [
        ProviderTariffSourceBinding(
            claim="route_identity",
            source_id=account.source_id,
            source_record_path="/location",
            source_value="eastus2",
            canonical_value=config.model,
        ),
        ProviderTariffSourceBinding(
            claim="route_identity",
            source_id=account.source_id,
            source_record_path="/name",
            source_value="example-resource",
            canonical_value=config.model,
        ),
        ProviderTariffSourceBinding(
            claim="route_identity",
            source_id=account.source_id,
            source_record_path="/properties/endpoint",
            source_value="https://example-resource.openai.azure.com/",
            canonical_value=config.model,
        ),
        ProviderTariffSourceBinding(
            claim="route_identity",
            source_id=deployment.source_id,
            source_record_path="/etag",
            source_value="opaque-route-version",
            canonical_value=config.model,
        ),
        ProviderTariffSourceBinding(
            claim="route_identity",
            source_id=deployment.source_id,
            source_record_path="/name",
            source_value=config.deployment or "",
            canonical_value=config.model,
        ),
        ProviderTariffSourceBinding(
            claim="route_identity",
            source_id=deployment.source_id,
            source_record_path="/properties/model/format",
            source_value="OpenAI",
            canonical_value=config.model,
        ),
        ProviderTariffSourceBinding(
            claim="route_identity",
            source_id=deployment.source_id,
            source_record_path="/properties/model/name",
            source_value=config.model,
            canonical_value=config.model,
        ),
        ProviderTariffSourceBinding(
            claim="route_identity",
            source_id=deployment.source_id,
            source_record_path="/properties/model/version",
            source_value="2026-07-01",
            canonical_value=config.model,
        ),
        ProviderTariffSourceBinding(
            claim="route_identity",
            source_id=deployment.source_id,
            source_record_path="/properties/versionUpgradeOption",
            source_value="NoAutoUpgrade",
            canonical_value=config.model,
        ),
        ProviderTariffSourceBinding(
            claim="route_identity",
            source_id=deployment.source_id,
            source_record_path="/sku/name",
            source_value="GlobalStandard",
            canonical_value=config.model,
        ),
    ]
    for index, (usage, canonical) in enumerate(
        (("Input", "input_tokens"), ("Output", "output_tokens"))
    ):
        bindings.extend(
            (
                ProviderTariffSourceBinding(
                    claim="route_identity",
                    source_id=retail.source_id,
                    source_record_path=f"/Items/{index}/productName",
                    source_value="Azure OpenAI gpt-5.5",
                    canonical_value=config.model,
                    target_meter_source_id=retail.source_id,
                    target_meter_record_path=f"/Items/{index}",
                ),
                ProviderTariffSourceBinding(
                    claim="usage_dimension",
                    source_id=retail.source_id,
                    source_record_path=f"/Items/{index}/meterName",
                    source_value=f"gpt-5.5 {usage} Tokens",
                    canonical_value=canonical,
                    target_meter_source_id=retail.source_id,
                    target_meter_record_path=f"/Items/{index}",
                ),
            )
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
    tariff = ProviderTokenTariff.from_usd_per_million(
        provider_config=config,
        input_usd=1,
        output_usd=2,
        source_snapshots=(account, deployment, retail),
        source_bindings=sorted_bindings,
        verified_on=date(2026, 7, 19),
        effective_on=date(2026, 7, 1),
        currency="USD",
        price_unit="per_1m_tokens",
        billing_region="eastus2",
        billing_mode="global-standard",
        billing_meters=meters,
    )
    return tariff, {
        account.source_id: account_bytes,
        deployment.source_id: deployment_bytes,
        retail.source_id: retail_bytes,
    }


def _replace_tariff_source_bytes(
    tariff: ProviderTokenTariff,
    *,
    source_id: str,
    retained: bytes,
) -> ProviderTokenTariff:
    digest = "sha256:" + hashlib.sha256(retained).hexdigest()
    sources = [
        source.model_copy(
            update={
                "source_snapshot_digest": digest,
                "retained_artifact": source.retained_artifact.model_copy(
                    update={
                        "locator": (
                            f"https://example.test/evidence/{digest.removeprefix('sha256:')}.json"
                        ),
                        "artifact_digest": digest,
                    }
                ),
            }
        )
        if source.source_id == source_id
        else source
        for source in tariff.provenance.source_snapshots
    ]
    provenance = ProviderTariffProvenance.model_validate(
        {
            **tariff.provenance.model_dump(mode="json"),
            "source_snapshots": [source.model_dump(mode="json") for source in sources],
        }
    )
    return ProviderTokenTariff(
        provider_config=tariff.provider_config,
        price=tariff.price,
        provenance=provenance,
    )


def _sorted_source_bindings(
    bindings: list[ProviderTariffSourceBinding],
) -> tuple[ProviderTariffSourceBinding, ...]:
    return tuple(
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


def _replace_azure_retail_document(
    tariff: ProviderTokenTariff,
    document: dict[str, object],
) -> tuple[ProviderTokenTariff, bytes]:
    retained = json.dumps(document, separators=(",", ":"), sort_keys=True).encode()
    source_replaced = _replace_tariff_source_bytes(
        tariff,
        source_id="azure-retail-price",
        retained=retained,
    )
    items = document["Items"]
    assert isinstance(items, list)
    rebound: list[ProviderTariffSourceBinding] = []
    for binding in source_replaced.provenance.source_bindings:
        if binding.source_id != "azure-retail-price":
            rebound.append(binding)
            continue
        _, collection, index_text, field = binding.source_record_path.split("/", 3)
        assert collection == "Items"
        item = items[int(index_text)]
        assert isinstance(item, dict)
        value = cast("dict[str, object]", item)[field]
        assert isinstance(value, str)
        rebound.append(binding.model_copy(update={"source_value": value}))
    provenance = ProviderTariffProvenance.model_validate(
        {
            **source_replaced.provenance.model_dump(mode="json"),
            "source_bindings": [
                binding.model_dump(mode="json") for binding in _sorted_source_bindings(rebound)
            ],
        }
    )
    return (
        ProviderTokenTariff(
            provider_config=source_replaced.provider_config,
            price=source_replaced.price,
            provenance=provenance,
        ),
        retained,
    )


@pytest.mark.parametrize(
    (
        "model_type",
        "model_id",
        "input_nano_usd",
        "output_nano_usd",
        "source_snapshots",
        "effective_on",
        "billing_mode",
        "input_meter",
        "output_meter",
    ),
    [
        (
            "claude-haiku-4-5",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            1_100,
            5_500,
            (
                (
                    "aws-bedrock-haiku-4-5-model-card-20260719",
                    "route_definition",
                    "https://docs.aws.amazon.com/bedrock/latest/userguide/"
                    "model-card-anthropic-claude-haiku-4-5.html",
                    "sha256:4c0e99775e1e9fd9c0fc45ce01d92a250ffd56dab7eb24d8ddb188042e55e0f2",
                    "text/html",
                    "identity",
                    None,
                    None,
                ),
                (
                    "aws-bedrock-meter-map-20260703",
                    "rate_catalog",
                    "https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/"
                    "bedrockfoundationmodels/USD/current/bedrockfoundationmodels.json",
                    "sha256:70ac2fe2f4153bf763492345b2029f06fefdb683023c420319a5b25679f02a11",
                    "application/json",
                    "gzip",
                    "plc-bedrockfoundationmodels-usd-20260703085857",
                    date(2026, 7, 3),
                ),
                (
                    "aws-bedrock-pricing-page-20260719",
                    "semantic_mapping",
                    "https://aws.amazon.com/bedrock/pricing/",
                    "sha256:fe5438f7f28cef02f6a3f439cbe323c3bdf109db34f0a713b1beacbc1c7f6c92",
                    "text/html",
                    "identity",
                    None,
                    None,
                ),
            ),
            None,
            "geo-cross-region-standard",
            (
                "aws-bedrock-meter-map-20260703",
                "/regions/US East (N. Virginia)/APY9uM3JGxEtL5H9EDm5AwcPdLxfrWr-RFMq0UellFs",
                "JQDUC8Q4K8C6GSGH",
                "JQDUC8Q4K8C6GSGH.4799GE89SK.6YS6EN2CT7",
            ),
            (
                "aws-bedrock-meter-map-20260703",
                "/regions/US East (N. Virginia)/O6zZHzgP5kfAsKpsBRIEEOBkSjbbqAoskrV4twdrBE4",
                "X629GDA2GXAP6R54",
                "X629GDA2GXAP6R54.4799GE89SK.6YS6EN2CT7",
            ),
        ),
        (
            "glm-5",
            "zai.glm-5",
            1_000,
            3_200,
            (
                (
                    "aws-bedrock-price-list-20260707",
                    "rate_catalog",
                    "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/"
                    "AmazonBedrock/20260707080509/index.json",
                    "sha256:2df66ba105f0d831725f825d73bbb8c01d3d4fae2bc64ee5527619c62aedced2",
                    "application/json",
                    "identity",
                    "AmazonBedrock:20260707080509",
                    date(2026, 7, 7),
                ),
            ),
            date(2026, 7, 1),
            "in-region-on-demand-standard",
            (
                "aws-bedrock-price-list-20260707",
                "/terms/OnDemand/YTB2BH9W4UZVKTEG/"
                "YTB2BH9W4UZVKTEG.JRTCKXETXF/priceDimensions/"
                "YTB2BH9W4UZVKTEG.JRTCKXETXF.6YS6EN2CT7",
                "YTB2BH9W4UZVKTEG",
                "YTB2BH9W4UZVKTEG.JRTCKXETXF.6YS6EN2CT7",
            ),
            (
                "aws-bedrock-price-list-20260707",
                "/terms/OnDemand/8RQBEKEP5KG2MZY7/"
                "8RQBEKEP5KG2MZY7.JRTCKXETXF/priceDimensions/"
                "8RQBEKEP5KG2MZY7.JRTCKXETXF.6YS6EN2CT7",
                "8RQBEKEP5KG2MZY7",
                "8RQBEKEP5KG2MZY7.JRTCKXETXF.6YS6EN2CT7",
            ),
        ),
        (
            "claude-opus-4-8",
            "us.anthropic.claude-opus-4-8",
            5_500,
            27_500,
            (
                (
                    "aws-bedrock-meter-map-20260703",
                    "rate_catalog",
                    "https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/"
                    "bedrockfoundationmodels/USD/current/bedrockfoundationmodels.json",
                    "sha256:70ac2fe2f4153bf763492345b2029f06fefdb683023c420319a5b25679f02a11",
                    "application/json",
                    "gzip",
                    "plc-bedrockfoundationmodels-usd-20260703085857",
                    date(2026, 7, 3),
                ),
                (
                    "aws-bedrock-opus-4-8-model-card-20260719",
                    "route_definition",
                    "https://docs.aws.amazon.com/bedrock/latest/userguide/"
                    "model-card-anthropic-claude-opus-4-8.html",
                    "sha256:682563efdd76f9b9099f320a532491123154d81a859222434f345ed93dee2f05",
                    "text/html",
                    "identity",
                    None,
                    None,
                ),
                (
                    "aws-bedrock-pricing-page-20260719",
                    "semantic_mapping",
                    "https://aws.amazon.com/bedrock/pricing/",
                    "sha256:fe5438f7f28cef02f6a3f439cbe323c3bdf109db34f0a713b1beacbc1c7f6c92",
                    "text/html",
                    "identity",
                    None,
                    None,
                ),
            ),
            None,
            "geo-cross-region-standard",
            (
                "aws-bedrock-meter-map-20260703",
                "/regions/US East (N. Virginia)/f9TflP4QPpfmpoCUae8qAT5_ecgxUrtuuJXamyHHDLE",
                "4AVHTD2NXFKSU6HU",
                "4AVHTD2NXFKSU6HU.4799GE89SK.6YS6EN2CT7",
            ),
            (
                "aws-bedrock-meter-map-20260703",
                "/regions/US East (N. Virginia)/jKk45pcJTXkhozzY9QxZVfUN3thwMVsHMmiZmLfi5lA",
                "YKJ5FPMCZAQF5BHF",
                "YKJ5FPMCZAQF5BHF.4799GE89SK.6YS6EN2CT7",
            ),
        ),
    ],
)
def test_catalog_freezes_verified_bedrock_routes_and_exact_nominal_prices(
    model_type: str,
    model_id: str,
    input_nano_usd: int,
    output_nano_usd: int,
    source_snapshots: tuple[tuple[str, str, str, str, str, str, str | None, date | None], ...],
    effective_on: date | None,
    billing_mode: str,
    input_meter: tuple[str, str, str, str],
    output_meter: tuple[str, str, str, str],
) -> None:
    config = ProviderConfig(
        kind=ProviderKind.BEDROCK,
        model_type=model_type,
        model=model_id,
        region="us-east-1",
    )

    tariff = catalog_provider_token_tariff(config)
    meter = provider_cost_meter(tariff)

    assert tariff.provider_config == config
    assert tariff.provenance.verified_on == date(2026, 7, 19)
    assert tariff.provenance.effective_on == effective_on
    assert (
        tuple(
            (
                snapshot.source_id,
                snapshot.role,
                snapshot.source_locator,
                snapshot.source_snapshot_digest,
                snapshot.media_type,
                snapshot.content_encoding,
                snapshot.publication_id,
                snapshot.published_on,
            )
            for snapshot in tariff.provenance.source_snapshots
        )
        == source_snapshots
    )
    assert tariff.provenance.currency == "USD"
    assert tariff.provenance.price_unit == "per_1m_tokens"
    assert tariff.provenance.route.provider_config == config
    assert tariff.provenance.route.billing_region == "us-east-1"
    assert tariff.provenance.route.billing_mode == billing_mode
    assert tuple(
        (
            meter.source_id,
            meter.source_record_path,
            meter.sku_id,
            meter.rate_id,
        )
        for meter in tariff.provenance.route.billing_meters
    ) == (input_meter, output_meter)
    assert tariff.price == TokenPriceCeiling(
        input_nano_usd_per_token=input_nano_usd,
        output_nano_usd_per_token=output_nano_usd,
    )
    assert meter == ProviderCostMeter(
        provider_config=config,
        price=tariff.price,
        tariff_provenance=tariff.provenance,
        tariff_evidence_receipt=meter.tariff_evidence_receipt,
    )


def test_catalog_lookup_requires_the_exact_frozen_route() -> None:
    catalog = catalog_provider_token_tariffs()

    assert isinstance(catalog, tuple)
    assert len(catalog) == 3
    with pytest.raises(ValueError, match="no audited provider tariff"):
        catalog_provider_token_tariff(
            ProviderConfig(
                kind=ProviderKind.BEDROCK,
                model_type="glm-5",
                model="zai.glm-5",
                region="us-west-2",
            )
        )
    with pytest.raises(ValueError, match="no audited provider tariff"):
        catalog_provider_token_tariff(
            ProviderConfig(
                kind=ProviderKind.BEDROCK,
                model_type="glm-5",
                model="zai.glm-5-preview",
                region="us-east-1",
            )
        )


def test_catalog_retains_and_verifies_every_content_addressed_source() -> None:
    assert verify_catalog_provider_tariff_sources() == (
        "aws-bedrock-haiku-4-5-model-card-20260719",
        "aws-bedrock-meter-map-20260703",
        "aws-bedrock-opus-4-8-model-card-20260719",
        "aws-bedrock-price-list-20260707",
        "aws-bedrock-pricing-page-20260719",
    )
    snapshots = {
        snapshot.source_id: snapshot
        for tariff in catalog_provider_token_tariffs()
        for snapshot in tariff.provenance.source_snapshots
    }
    for snapshot in snapshots.values():
        assert snapshot.source_snapshot_digest.removeprefix("sha256:") in (
            snapshot.retained_artifact.locator
        )


def test_catalog_rejects_conflicting_metadata_for_one_source_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tariff = tariffs_module._CATALOG[0]
    snapshots = list(tariff.provenance.source_snapshots)
    snapshots[0] = snapshots[0].model_copy(update={"source_snapshot_digest": "sha256:" + "c" * 64})
    conflicting = tariff.model_copy(
        update={
            "provenance": tariff.provenance.model_copy(
                update={"source_snapshots": tuple(snapshots)}
            )
        }
    )
    monkeypatch.setattr(tariffs_module, "_CATALOG", (tariff, conflicting))

    with pytest.raises(
        tariffs_module.ProviderTariffEvidenceIntegrityError,
        match="conflicting built-in tariff source metadata",
    ):
        verify_catalog_provider_tariff_sources()


def test_catalog_rejects_coordinated_batch_meter_underpricing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tariff = tariffs_module._CATALOG[0]
    batch_meters = (
        ProviderTariffBillingMeter(
            usage_dimension="input_tokens",
            source_id="aws-bedrock-meter-map-20260703",
            source_record_path=(
                "/regions/US East (N. Virginia)/03OdAcs5iDRbkrNByVOR2ywGY16dSqcQ_E-KH6M1LMI"
            ),
            sku_id="8GAKPXRDT9CSGWEC",
            rate_id="8GAKPXRDT9CSGWEC.4799GE89SK.6YS6EN2CT7",
            billing_region="us-east-1",
            billing_mode="geo-cross-region-standard",
            effective_on=None,
            source_price_usd=Decimal("0.55"),
            source_price_unit="per_1m_tokens",
        ),
        ProviderTariffBillingMeter(
            usage_dimension="output_tokens",
            source_id="aws-bedrock-meter-map-20260703",
            source_record_path=(
                "/regions/US East (N. Virginia)/3HGzHYMPOLfiR_Cd5EwGZ3oa1jeMs-NNpoBjWYRP0WQ"
            ),
            sku_id="E56Y3URFEMX4ZGDC",
            rate_id="E56Y3URFEMX4ZGDC.4799GE89SK.6YS6EN2CT7",
            billing_region="us-east-1",
            billing_mode="geo-cross-region-standard",
            effective_on=None,
            source_price_usd=Decimal("2.75"),
            source_price_unit="per_1m_tokens",
        ),
    )
    bindings = []
    for binding in tariff.provenance.source_bindings:
        if binding.source_id != "aws-bedrock-pricing-page-20260719" or (
            binding.claim != "usage_dimension"
        ):
            bindings.append(binding)
            continue
        meter = batch_meters[0 if binding.canonical_value == "input_tokens" else 1]
        bindings.append(
            binding.model_copy(
                update={
                    "source_value": meter.source_record_path.rsplit("/", 1)[-1],
                    "target_meter_record_path": meter.source_record_path,
                }
            )
        )
    provenance = ProviderTariffProvenance.model_validate(
        {
            **tariff.provenance.model_dump(mode="json"),
            "source_bindings": [
                binding.model_dump(mode="json") for binding in _sorted_source_bindings(bindings)
            ],
            "route": {
                **tariff.provenance.route.model_dump(mode="json"),
                "billing_meters": [meter.model_dump(mode="json") for meter in batch_meters],
            },
        }
    )
    underpriced = ProviderTokenTariff(
        provider_config=tariff.provider_config,
        price=TokenPriceCeiling.from_usd_per_million(
            input_usd="0.55",
            output_usd="2.75",
        ),
        provenance=provenance,
    )
    monkeypatch.setattr(
        tariffs_module,
        "_CATALOG",
        (underpriced, *tariffs_module._CATALOG[1:]),
    )

    with pytest.raises(
        tariffs_module.ProviderTariffEvidenceIntegrityError,
        match="standard input and output columns",
    ):
        verify_catalog_provider_tariff_sources()


def test_catalog_rejects_model_card_path_or_geo_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tariff = tariffs_module._CATALOG[0]
    bindings = [
        binding.model_copy(
            update={
                "source_record_path": "/not/a/real/record",
                "source_value": "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
            }
        )
        if binding.source_id == "aws-bedrock-haiku-4-5-model-card-20260719"
        else binding
        for binding in tariff.provenance.source_bindings
    ]
    provenance = ProviderTariffProvenance.model_validate(
        {
            **tariff.provenance.model_dump(mode="json"),
            "source_bindings": [
                binding.model_dump(mode="json") for binding in _sorted_source_bindings(bindings)
            ],
        }
    )
    mutated = ProviderTokenTariff(
        provider_config=tariff.provider_config,
        price=tariff.price,
        provenance=provenance,
    )
    monkeypatch.setattr(
        tariffs_module,
        "_CATALOG",
        (mutated, *tariffs_module._CATALOG[1:]),
    )

    with pytest.raises(
        tariffs_module.ProviderTariffEvidenceIntegrityError,
        match="exact inference profile",
    ):
        verify_catalog_provider_tariff_sources()


def test_catalog_rejects_cross_sku_rate_catalog_route_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tariff = tariffs_module._CATALOG[1]
    bindings = list(tariff.provenance.source_bindings)
    route_index = next(
        index for index, binding in enumerate(bindings) if binding.claim == "route_identity"
    )
    bindings[route_index] = bindings[route_index].model_copy(
        update={"source_record_path": "/products/DIFFERENT/attributes/model"}
    )
    provenance = ProviderTariffProvenance.model_validate(
        {
            **tariff.provenance.model_dump(mode="json"),
            "source_bindings": [
                binding.model_dump(mode="json") for binding in _sorted_source_bindings(bindings)
            ],
        }
    )
    mutated = ProviderTokenTariff(
        provider_config=tariff.provider_config,
        price=tariff.price,
        provenance=provenance,
    )
    monkeypatch.setattr(
        tariffs_module,
        "_CATALOG",
        (tariffs_module._CATALOG[0], mutated, tariffs_module._CATALOG[2]),
    )

    with pytest.raises(
        tariffs_module.ProviderTariffEvidenceIntegrityError,
        match="target billing SKU",
    ):
        verify_catalog_provider_tariff_sources()


def test_catalog_rejects_publication_metadata_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tariff = tariffs_module._CATALOG[1]
    sources = list(tariff.provenance.source_snapshots)
    sources[0] = sources[0].model_copy(update={"published_on": date(2026, 7, 8)})
    mutated = tariff.model_copy(
        update={
            "provenance": tariff.provenance.model_copy(update={"source_snapshots": tuple(sources)})
        }
    )
    monkeypatch.setattr(
        tariffs_module,
        "_CATALOG",
        (tariffs_module._CATALOG[0], mutated, tariffs_module._CATALOG[2]),
    )

    with pytest.raises(
        tariffs_module.ProviderTariffEvidenceIntegrityError,
        match="retained source registry",
    ):
        verify_catalog_provider_tariff_sources()


def test_bedrock_receipt_rejects_an_extra_registered_rate_source() -> None:
    haiku, glm, _ = catalog_provider_token_tariffs()
    payload = haiku.model_dump(mode="json")
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    sources = provenance["source_snapshots"]
    assert isinstance(sources, list)
    sources.extend(source.model_dump(mode="json") for source in glm.provenance.source_snapshots)
    sources.sort(key=lambda source: source["source_id"])
    mutated = ProviderTokenTariff.model_validate(payload)

    with pytest.raises(
        tariffs_module.ProviderTariffEvidenceIntegrityError,
        match="source set differs",
    ):
        verify_provider_tariff_evidence(mutated)


def test_public_source_query_is_canonical_and_rejects_credential_channels() -> None:
    public_query = (
        ProviderTariffPublicQueryParameter(
            name="$filter",
            value=(
                "serviceName eq 'Azure OpenAI' and armRegionName eq 'eastus2' "
                "and meterName eq 'Input Tokens'"
            ),
        ),
        ProviderTariffPublicQueryParameter(name="api-version", value="2023-01-01-preview"),
    )
    source = _test_source_snapshot().model_copy(update={"public_request_query": public_query})

    assert source.public_request_query == public_query
    with pytest.raises(ValidationError, match="canonical ascending"):
        ProviderTariffSourceSnapshot.model_validate(
            {**source.model_dump(mode="json"), "public_request_query": list(reversed(public_query))}
        )
    for duplicate_name in ("$filter", "api-version"):
        duplicated = tuple(
            sorted(
                (
                    *public_query,
                    ProviderTariffPublicQueryParameter(
                        name=duplicate_name,
                        value="additional-public-coordinate",
                    ),
                ),
                key=lambda parameter: (parameter.name, parameter.value),
            )
        )
        with pytest.raises(ValidationError, match="unique parameter names"):
            ProviderTariffSourceSnapshot.model_validate(
                {
                    **source.model_dump(mode="json"),
                    "public_request_query": [
                        parameter.model_dump(mode="json") for parameter in duplicated
                    ],
                }
            )
    with pytest.raises(ValidationError, match="credential-bearing"):
        ProviderTariffPublicQueryParameter(name="sig", value="opaque")
    with pytest.raises(ValidationError, match="credential-bearing"):
        ProviderTariffPublicQueryParameter(name="$filter", value="apiKey eq 'opaque'")
    with pytest.raises(ValidationError, match="credential-bearing"):
        ProviderTariffPublicQueryParameter(name="subscription-key", value="opaque")
    with pytest.raises(ValidationError, match="credential-bearing"):
        ProviderTariffPublicQueryParameter(name="$filter", value="sig eq 'opaque'")


@pytest.mark.parametrize("control", ["\n", "\r", "\t", "\x7f", "\x85", "\u200b"])
def test_public_source_query_rejects_nonprintable_values(control: str) -> None:
    with pytest.raises(ValidationError, match="printable"):
        ProviderTariffPublicQueryParameter(name="$filter", value=f"safe{control}value")


@pytest.mark.parametrize("control", ["\n", "\r", "\t", "\x7f", "\x85", "\u200b"])
def test_public_source_locator_rejects_nonprintable_text(control: str) -> None:
    payload = _test_source_snapshot().model_dump(mode="json")
    payload["source_locator"] = f"https://prices.azure.com{control}/api/retail/prices"

    with pytest.raises(ValidationError, match="printable"):
        ProviderTariffSourceSnapshot.model_validate(payload)


@pytest.mark.parametrize("control", ["\n", "\r", "\t", "\x7f", "\x85", "\u200b"])
def test_retained_artifact_locator_rejects_nonprintable_text(control: str) -> None:
    payload = _test_source_snapshot().retained_artifact.model_dump(mode="json")
    payload["locator"] = f"https://example.test{control}/evidence/rates.json"

    with pytest.raises(ValidationError, match="printable"):
        ProviderTariffRetainedArtifact.model_validate(payload)


@pytest.mark.parametrize("segment", [".", ".."])
def test_retained_https_artifact_locator_rejects_dot_path_segments(segment: str) -> None:
    payload = _test_source_snapshot().retained_artifact.model_dump(mode="json")
    payload["locator"] = f"https://example.test/evidence/{segment}/rates.json"

    with pytest.raises(ValidationError, match="dot path segments"):
        ProviderTariffRetainedArtifact.model_validate(payload)


@pytest.mark.parametrize("segment", [".", ".."])
@pytest.mark.parametrize("resource", ["account", "deployment"])
def test_public_source_locator_rejects_dot_resource_segments(
    segment: str,
    resource: str,
) -> None:
    suffix = (
        "providers/Microsoft.CognitiveServices/accounts/example-resource"
        if resource == "account"
        else (
            "providers/Microsoft.CognitiveServices/accounts/example-resource/"
            "deployments/example-deployment"
        )
    )
    payload = _test_source_snapshot().model_dump(mode="json")
    payload["source_locator"] = (
        f"https://management.azure.com/subscriptions/{segment}/resourceGroups/example/{suffix}"
    )

    with pytest.raises(ValidationError, match="dot path segments"):
        ProviderTariffSourceSnapshot.model_validate(payload)


@pytest.mark.parametrize(
    "billing_meters",
    [
        (
            _test_billing_meter(
                "input_tokens",
                "input",
                billing_region="us-east-1",
                billing_mode="in-region-on-demand-standard",
            ),
        ),
        (
            _test_billing_meter(
                "output_tokens",
                "output",
                billing_region="us-east-1",
                billing_mode="in-region-on-demand-standard",
            ),
            _test_billing_meter(
                "input_tokens",
                "input",
                billing_region="us-east-1",
                billing_mode="in-region-on-demand-standard",
            ),
        ),
        (
            _test_billing_meter(
                "input_tokens",
                "input-one",
                billing_region="us-east-1",
                billing_mode="in-region-on-demand-standard",
            ),
            _test_billing_meter(
                "input_tokens",
                "input-two",
                billing_region="us-east-1",
                billing_mode="in-region-on-demand-standard",
            ),
        ),
    ],
)
def test_tariff_route_requires_exact_input_and_output_billing_meters(
    billing_meters: tuple[ProviderTariffBillingMeter, ...],
) -> None:
    with pytest.raises(ValidationError, match="billing_meters|input_tokens then output_tokens"):
        ProviderTariffRoute(
            provider_config=ProviderConfig(
                kind=ProviderKind.BEDROCK,
                model="zai.glm-5",
                region="us-east-1",
            ),
            billing_region="us-east-1",
            billing_mode="in-region-on-demand-standard",
            billing_meters=cast(
                "tuple[ProviderTariffBillingMeter, ProviderTariffBillingMeter]",
                billing_meters,
            ),
        )


def test_tariff_route_rejects_provider_kinds_without_an_evidence_verifier() -> None:
    config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="claude-haiku-4-5")

    with pytest.raises(ValidationError, match="no registered paid evidence verifier"):
        ProviderTariffRoute(
            provider_config=config,
            billing_region="global",
            billing_mode="standard",
            billing_meters=_test_billing_meters(
                billing_region="global",
                billing_mode="standard",
            ),
        )


def test_catalog_does_not_infer_an_azure_deployment_price() -> None:
    azure_config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="deployment-requiring-live-price-verification",
        api_version="2026-06-01",
    )

    with pytest.raises(ValueError, match="no audited provider tariff"):
        catalog_provider_token_tariff(azure_config)


def test_descriptive_prices_are_exact_or_fail_closed_without_full_route_context() -> None:
    for tariff in catalog_provider_token_tariffs():
        descriptive_price = price_for(
            tariff.provider_config.model,
            provider=tariff.provider_config.kind.value,
        )

        if tariff.provider_config.model == "zai.glm-5":
            assert descriptive_price is None
            continue
        assert descriptive_price is not None
        assert descriptive_price.input_per_mtok * 1_000 == pytest.approx(
            tariff.price.input_nano_usd_per_token
        )
        assert descriptive_price.output_per_mtok * 1_000 == pytest.approx(
            tariff.price.output_nano_usd_per_token
        )


def test_caller_supplied_tariff_freezes_an_exact_azure_responses_route() -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-gpt-55-deployment",
        api_version="2026-06-01",
        reasoning_effort="high",
        responses_api_version="v1",
    )
    tariff, artifacts = _azure_tariff_fixture(config)
    with pytest.raises(
        tariffs_module.ProviderTariffEvidenceIntegrityError,
        match="artifact source set",
    ):
        provider_cost_meter(tariff)

    receipt = verify_provider_tariff_evidence(tariff, evidence_artifacts=artifacts)
    meter = provider_cost_meter(
        tariff,
        input_overhead_tokens=16_384,
        evidence_artifacts=artifacts,
    )

    assert meter.provider_config == config
    assert meter.price == TokenPriceCeiling(
        input_nano_usd_per_token=1_000,
        output_nano_usd_per_token=2_000,
    )
    assert meter.input_overhead_tokens == 16_384
    assert meter.tariff_provenance == tariff.provenance
    assert meter.tariff_evidence_receipt == receipt
    assert tuple(source.source_id for source in receipt.verified_sources) == (
        "azure-account-route",
        "azure-deployment-route",
        "azure-retail-price",
    )
    assert meter.model_dump(mode="json")["provider_config"] == config.model_dump(mode="json")


def test_azure_evidence_factory_derives_the_complete_paid_meter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-gpt-55-deployment",
        api_version="2026-06-01",
        reasoning_effort="high",
        responses_api_version="v1",
    )
    declared, artifacts = _azure_tariff_fixture(config)
    decoded_artifacts: list[str] = []
    original_decode = tariffs_module._decode_verified_artifact

    def record_decode(
        *,
        locator: str,
        artifact_bytes: bytes,
        artifact_digest: str,
        artifact_encoding: str,
        source_digest: str,
    ) -> bytes:
        decoded_artifacts.append(locator)
        return original_decode(
            locator=locator,
            artifact_bytes=artifact_bytes,
            artifact_digest=artifact_digest,
            artifact_encoding=artifact_encoding,
            source_digest=source_digest,
        )

    monkeypatch.setattr(tariffs_module, "_decode_verified_artifact", record_decode)

    meter = azure_provider_cost_meter_from_evidence(
        provider_config=config,
        source_snapshots=tuple(reversed(declared.provenance.source_snapshots)),
        evidence_artifacts=artifacts,
        verified_on=date(2026, 7, 19),
        input_overhead_tokens=16_384,
    )

    assert meter.provider_config == config
    assert meter.price == declared.price
    assert meter.tariff_provenance == declared.provenance
    assert meter.input_overhead_tokens == 16_384
    assert len(decoded_artifacts) == 3
    assert meter.tariff_evidence_receipt == verify_provider_tariff_evidence(
        declared,
        evidence_artifacts=artifacts,
    )


def test_azure_evidence_factory_derives_usage_from_reordered_retail_rows() -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-gpt-55-deployment",
        api_version="2026-06-01",
    )
    declared, artifacts = _azure_tariff_fixture(config)
    retail = json.loads(artifacts["azure-retail-price"])
    retail["Items"].reverse()
    mutated, retained = _replace_azure_retail_document(declared, retail)

    meter = azure_provider_cost_meter_from_evidence(
        provider_config=config,
        source_snapshots=mutated.provenance.source_snapshots,
        evidence_artifacts={**artifacts, "azure-retail-price": retained},
        verified_on=date(2026, 7, 19),
    )

    input_meter, output_meter = meter.tariff_provenance.route.billing_meters
    assert input_meter.source_record_path == "/Items/1"
    assert output_meter.source_record_path == "/Items/0"
    assert meter.price == declared.price


def test_azure_evidence_factory_accepts_only_a_conservative_explicit_ceiling() -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-gpt-55-deployment",
        api_version="2026-06-01",
    )
    declared, artifacts = _azure_tariff_fixture(config)
    ceiling = TokenPriceCeiling(
        input_nano_usd_per_token=1_100,
        output_nano_usd_per_token=2_200,
    )

    meter = azure_provider_cost_meter_from_evidence(
        provider_config=config,
        source_snapshots=declared.provenance.source_snapshots,
        evidence_artifacts=artifacts,
        verified_on=date(2026, 7, 19),
        price_ceiling=ceiling,
    )
    assert meter.price == ceiling

    with pytest.raises(ValueError, match="understates retained billing meter rates"):
        azure_provider_cost_meter_from_evidence(
            provider_config=config,
            source_snapshots=declared.provenance.source_snapshots,
            evidence_artifacts=artifacts,
            verified_on=date(2026, 7, 19),
            price_ceiling=TokenPriceCeiling(
                input_nano_usd_per_token=999,
                output_nano_usd_per_token=2_000,
            ),
        )


@pytest.mark.parametrize(
    "effective_start",
    [
        "2026-07-01",
        "2026-07-01Tnot-an-iso-time",
        "2026-07-01T00:30:00+14:00",
        "2026-07-01T12:00:00Z",
        "2026-07-01T00:00:00.0000001Z",
        "2026-02-30T00:00:00Z",
    ],
)
def test_azure_evidence_factory_requires_complete_utc_effective_timestamps(
    effective_start: str,
) -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-gpt-55-deployment",
        api_version="2026-06-01",
    )
    declared, artifacts = _azure_tariff_fixture(config)
    retail = json.loads(artifacts["azure-retail-price"])
    for record in retail["Items"]:
        record["effectiveStartDate"] = effective_start
    mutated, retained = _replace_azure_retail_document(declared, retail)

    with pytest.raises(
        tariffs_module.ProviderTariffEvidenceIntegrityError,
        match="complete UTC timestamp",
    ):
        azure_provider_cost_meter_from_evidence(
            provider_config=config,
            source_snapshots=mutated.provenance.source_snapshots,
            evidence_artifacts={**artifacts, "azure-retail-price": retained},
            verified_on=date(2026, 7, 19),
        )


def test_azure_evidence_factory_accepts_seven_digit_zero_fraction_timestamp() -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-gpt-55-deployment",
        api_version="2026-06-01",
    )
    declared, artifacts = _azure_tariff_fixture(config)
    retail = json.loads(artifacts["azure-retail-price"])
    for record in retail["Items"]:
        record["effectiveStartDate"] = "2026-07-01T00:00:00.0000000Z"
    mutated, retained = _replace_azure_retail_document(declared, retail)

    meter = azure_provider_cost_meter_from_evidence(
        provider_config=config,
        source_snapshots=mutated.provenance.source_snapshots,
        evidence_artifacts={**artifacts, "azure-retail-price": retained},
        verified_on=date(2026, 7, 19),
    )

    assert meter.tariff_provenance.effective_on == date(2026, 7, 1)


@pytest.mark.parametrize(
    "source_id",
    ["azure-account-route", "azure-deployment-route", "azure-retail-price"],
)
@pytest.mark.parametrize("api_version", ["not-a-version", "2026-02-30", "2026-01-01-beta"])
def test_azure_evidence_factory_rejects_invalid_public_api_versions(
    source_id: str,
    api_version: str,
) -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-gpt-55-deployment",
        api_version="2026-06-01",
    )
    declared, artifacts = _azure_tariff_fixture(config)
    sources: list[ProviderTariffSourceSnapshot] = []
    for source in declared.provenance.source_snapshots:
        query = list(source.public_request_query)
        if source.source_id == source_id:
            query = [parameter for parameter in query if parameter.name != "api-version"]
            query.append(
                ProviderTariffPublicQueryParameter(
                    name="api-version",
                    value=api_version,
                )
            )
        sources.append(
            source.model_copy(
                update={
                    "public_request_query": tuple(
                        sorted(query, key=lambda parameter: (parameter.name, parameter.value))
                    )
                }
            )
        )

    with pytest.raises(
        tariffs_module.ProviderTariffEvidenceIntegrityError,
        match="API version",
    ):
        azure_provider_cost_meter_from_evidence(
            provider_config=config,
            source_snapshots=tuple(sources),
            evidence_artifacts=artifacts,
            verified_on=date(2026, 7, 19),
        )


def test_azure_evidence_factory_rejects_a_non_azure_route_before_evidence() -> None:
    with pytest.raises(ValueError, match="requires an Azure OpenAI provider route"):
        azure_provider_cost_meter_from_evidence(
            provider_config=ProviderConfig(
                kind=ProviderKind.BEDROCK,
                model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                region="us-east-1",
            ),
            source_snapshots=(),
            evidence_artifacts={},
            verified_on=date(2026, 7, 19),
        )


def test_external_tariff_evidence_requires_the_exact_complete_artifact_set() -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-gpt-55-deployment",
        api_version="2026-06-01",
        reasoning_effort="high",
        responses_api_version="v1",
    )
    tariff, artifacts = _azure_tariff_fixture(config)
    missing = dict(artifacts)
    del missing["azure-retail-price"]
    extra = {**artifacts, "unclaimed-source": b"not part of the tariff"}

    for invalid in (missing, extra):
        with pytest.raises(
            tariffs_module.ProviderTariffEvidenceIntegrityError,
            match="source set differs",
        ):
            verify_provider_tariff_evidence(tariff, evidence_artifacts=invalid)


def test_digest_correct_fake_azure_bytes_fail_semantic_verification() -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-gpt-55-deployment",
        api_version="2026-06-01",
        reasoning_effort="high",
        responses_api_version="v1",
    )
    tariff, artifacts = _azure_tariff_fixture(config)
    retail = json.loads(artifacts["azure-retail-price"])
    retail["Items"][0]["currencyCode"] = "EUR"
    retained = json.dumps(retail, separators=(",", ":"), sort_keys=True).encode()
    mutated = _replace_tariff_source_bytes(
        tariff,
        source_id="azure-retail-price",
        retained=retained,
    )
    mutated_artifacts = {**artifacts, "azure-retail-price": retained}

    with pytest.raises(
        tariffs_module.ProviderTariffEvidenceIntegrityError,
        match="not priced in USD",
    ):
        verify_provider_tariff_evidence(
            mutated,
            evidence_artifacts=mutated_artifacts,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("armRegionName", "westus", "region differs"),
        ("effectiveStartDate", "2026-07-02T00:00:00Z", "effective date differs"),
        (
            "effectiveStartDate",
            "2026-07-01Tnot-an-iso-time",
            "complete UTC timestamp",
        ),
        (
            "effectiveStartDate",
            "2026-07-01T00:30:00+14:00",
            "complete UTC timestamp",
        ),
        ("retailPrice", "0.01", "price differs"),
        ("unitOfMeasure", "1 Hour", "unit differs"),
        ("type", "Reservation", "not standard consumption"),
        ("serviceName", "Azure AI Services", "different service"),
        ("tierMinimumUnits", 1, "not the base tier"),
        ("isPrimaryMeterRegion", False, "not the primary billing record"),
        ("armSkuName", "Standard", "not joined"),
    ],
)
def test_azure_retail_semantics_fail_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-gpt-55-deployment",
        api_version="2026-06-01",
        reasoning_effort="high",
        responses_api_version="v1",
    )
    tariff, artifacts = _azure_tariff_fixture(config)
    retail = json.loads(artifacts["azure-retail-price"])
    retail["Items"][0][field] = value
    retained = json.dumps(retail, separators=(",", ":"), sort_keys=True).encode()
    mutated = _replace_tariff_source_bytes(
        tariff,
        source_id="azure-retail-price",
        retained=retained,
    )

    with pytest.raises(
        tariffs_module.ProviderTariffEvidenceIntegrityError,
        match=message,
    ):
        verify_provider_tariff_evidence(
            mutated,
            evidence_artifacts={**artifacts, "azure-retail-price": retained},
        )


@pytest.mark.parametrize(
    "retail_model",
    ["gpt-5.5-mini", "gpt-5.5 mini", "gpt-5.5 Preview"],
)
def test_azure_retail_rejects_an_overlapping_cheaper_model_identity(
    retail_model: str,
) -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-gpt-55-deployment",
        api_version="2026-06-01",
        reasoning_effort="high",
        responses_api_version="v1",
    )
    tariff, artifacts = _azure_tariff_fixture(config)
    retail = json.loads(artifacts["azure-retail-price"])
    for item in retail["Items"]:
        for field in ("meterName", "productName", "skuName"):
            item[field] = item[field].replace("gpt-5.5", retail_model)
    mutated, retained = _replace_azure_retail_document(tariff, retail)

    with pytest.raises(
        tariffs_module.ProviderTariffEvidenceIntegrityError,
        match="exact deployment model",
    ):
        verify_provider_tariff_evidence(
            mutated,
            evidence_artifacts={**artifacts, "azure-retail-price": retained},
        )


def test_azure_retail_rejects_a_coherently_bound_cached_input_meter() -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-gpt-55-deployment",
        api_version="2026-06-01",
        reasoning_effort="high",
        responses_api_version="v1",
    )
    tariff, artifacts = _azure_tariff_fixture(config)
    retail = json.loads(artifacts["azure-retail-price"])
    retail["Items"][0]["meterName"] = "gpt-5.5 Cached Input Tokens"
    mutated, retained = _replace_azure_retail_document(tariff, retail)

    with pytest.raises(
        tariffs_module.ProviderTariffEvidenceIntegrityError,
        match="exact standard token usage dimension",
    ):
        verify_provider_tariff_evidence(
            mutated,
            evidence_artifacts={**artifacts, "azure-retail-price": retained},
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"NextPageLink": "https://prices.azure.com/next"}, "pagination continuation"),
        ({"Count": 3}, "item count is inconsistent"),
        ({"BillingCurrency": "EUR"}, "different billing currency"),
    ],
)
def test_azure_retail_response_completeness_is_verified(
    mutation: dict[str, object],
    message: str,
) -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-gpt-55-deployment",
        api_version="2026-06-01",
        reasoning_effort="high",
        responses_api_version="v1",
    )
    tariff, artifacts = _azure_tariff_fixture(config)
    retail = json.loads(artifacts["azure-retail-price"])
    retail.update(mutation)
    retained = json.dumps(retail, separators=(",", ":"), sort_keys=True).encode()
    mutated = _replace_tariff_source_bytes(
        tariff,
        source_id="azure-retail-price",
        retained=retained,
    )

    with pytest.raises(
        tariffs_module.ProviderTariffEvidenceIntegrityError,
        match=message,
    ):
        verify_provider_tariff_evidence(
            mutated,
            evidence_artifacts={**artifacts, "azure-retail-price": retained},
        )


def test_azure_retail_query_is_bound_to_service_region_and_currency() -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-gpt-55-deployment",
        api_version="2026-06-01",
        reasoning_effort="high",
        responses_api_version="v1",
    )
    tariff, artifacts = _azure_tariff_fixture(config)
    payload = tariff.model_dump(mode="json")
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    sources = provenance["source_snapshots"]
    assert isinstance(sources, list)
    retail = next(source for source in sources if source["source_id"] == "azure-retail-price")
    retail["public_request_query"] = [
        {
            "name": "$filter",
            "value": "serviceName eq 'Azure OpenAI' and armRegionName eq 'westus'",
        },
        {"name": "currencyCode", "value": "USD"},
    ]
    mutated = ProviderTokenTariff.model_validate(payload)

    with pytest.raises(
        tariffs_module.ProviderTariffEvidenceIntegrityError,
        match="service, region, or currency",
    ):
        verify_provider_tariff_evidence(mutated, evidence_artifacts=artifacts)


def test_azure_deployment_resource_must_belong_to_the_bound_account() -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-gpt-55-deployment",
        api_version="2026-06-01",
        reasoning_effort="high",
        responses_api_version="v1",
    )
    tariff, artifacts = _azure_tariff_fixture(config)
    payload = tariff.model_dump(mode="json")
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    sources = provenance["source_snapshots"]
    assert isinstance(sources, list)
    deployment = next(
        source for source in sources if source["source_id"] == "azure-deployment-route"
    )
    deployment["source_locator"] = (
        "https://management.azure.com/subscriptions/example/resourceGroups/example/"
        "providers/Microsoft.CognitiveServices/accounts/different-resource/deployments/"
        "audited-gpt-55-deployment"
    )
    mutated = ProviderTokenTariff.model_validate(payload)

    with pytest.raises(
        tariffs_module.ProviderTariffEvidenceIntegrityError,
        match="different ARM account resource",
    ):
        verify_provider_tariff_evidence(mutated, evidence_artifacts=artifacts)


@pytest.mark.parametrize(
    "source_id",
    ["azure-account-route", "azure-deployment-route", "azure-retail-price"],
)
def test_azure_sources_require_the_exact_canonical_authority(source_id: str) -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-gpt-55-deployment",
        api_version="2026-06-01",
    )
    tariff, artifacts = _azure_tariff_fixture(config)
    payload = tariff.model_dump(mode="json")
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    sources = provenance["source_snapshots"]
    assert isinstance(sources, list)
    source = next(source for source in sources if source["source_id"] == source_id)
    locator = source["source_locator"]
    assert isinstance(locator, str)
    scheme, authority_and_path = locator.split("://", 1)
    authority, path = authority_and_path.split("/", 1)
    source["source_locator"] = f"{scheme}://{authority}:444/{path}"
    mutated = ProviderTokenTariff.model_validate(payload)

    with pytest.raises(
        tariffs_module.ProviderTariffEvidenceIntegrityError,
        match="canonical authority",
    ):
        verify_provider_tariff_evidence(mutated, evidence_artifacts=artifacts)


def test_azure_deployment_route_and_upgrade_policy_are_verified() -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-gpt-55-deployment",
        api_version="2026-06-01",
        reasoning_effort="high",
        responses_api_version="v1",
    )
    tariff, artifacts = _azure_tariff_fixture(config)
    deployment = json.loads(artifacts["azure-deployment-route"])
    deployment["properties"]["versionUpgradeOption"] = "OnceNewDefaultVersionAvailable"
    retained = json.dumps(deployment, separators=(",", ":"), sort_keys=True).encode()
    mutated = _replace_tariff_source_bytes(
        tariff,
        source_id="azure-deployment-route",
        retained=retained,
    )

    with pytest.raises(
        tariffs_module.ProviderTariffEvidenceIntegrityError,
        match="disable automatic model upgrades",
    ):
        verify_provider_tariff_evidence(
            mutated,
            evidence_artifacts={**artifacts, "azure-deployment-route": retained},
        )


def test_azure_receipt_requires_deployment_revision_route_bindings() -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-gpt-55-deployment",
        api_version="2026-06-01",
        reasoning_effort="high",
        responses_api_version="v1",
    )
    tariff, artifacts = _azure_tariff_fixture(config)
    provenance_payload = tariff.provenance.model_dump(mode="json")
    bindings = provenance_payload["source_bindings"]
    assert isinstance(bindings, list)
    provenance_payload["source_bindings"] = [
        binding
        for binding in bindings
        if binding["source_record_path"] != "/properties/model/version"
    ]
    mutated = ProviderTokenTariff(
        provider_config=tariff.provider_config,
        price=tariff.price,
        provenance=ProviderTariffProvenance.model_validate(provenance_payload),
    )

    with pytest.raises(
        tariffs_module.ProviderTariffEvidenceIntegrityError,
        match="omits a required.*route binding",
    ):
        verify_provider_tariff_evidence(mutated, evidence_artifacts=artifacts)


def test_tariff_receipt_replay_and_nested_copy_bypasses_are_rejected() -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-gpt-55-deployment",
        api_version="2026-06-01",
        reasoning_effort="high",
        responses_api_version="v1",
    )
    tariff, artifacts = _azure_tariff_fixture(config)
    receipt = verify_provider_tariff_evidence(tariff, evidence_artifacts=artifacts)
    higher_price = TokenPriceCeiling.from_usd_per_million(
        input_usd=3,
        output_usd=4,
    )

    with pytest.raises(ValidationError, match="different tariff claim"):
        ProviderCostMeter(
            provider_config=config,
            price=higher_price,
            tariff_provenance=tariff.provenance,
            tariff_evidence_receipt=receipt,
        )
    with pytest.raises(ValidationError, match="different tariff claim"):
        ProviderCostMeter(
            provider_config=config,
            price=tariff.price,
            tariff_provenance=tariff.provenance,
            tariff_evidence_receipt=receipt.model_copy(
                update={"tariff_claim_digest": "sha256:" + "c" * 64}
            ),
        )
    with pytest.raises(ValidationError, match="local registry"):
        ProviderTariffEvidenceReceipt.model_validate(
            {
                **receipt.model_dump(mode="json"),
                "verifier_digest": "sha256:" + "d" * 64,
            }
        )

    meter = provider_cost_meter(tariff, evidence_artifacts=artifacts)
    assert ProviderCostMeter.model_validate(meter.model_dump(mode="json")) == meter


def test_azure_tariff_cannot_be_rebound_to_a_different_deployment() -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-deployment",
        api_version="2026-06-01",
        reasoning_effort="high",
        responses_api_version="v1",
    )
    tariff = ProviderTokenTariff.from_usd_per_million(
        provider_config=config,
        input_usd="1",
        output_usd="2",
        source_snapshots=(_test_source_snapshot(),),
        source_bindings=_test_source_bindings(config),
        verified_on=date(2026, 7, 19),
        effective_on=date(2026, 7, 1),
        currency="USD",
        price_unit="per_1m_tokens",
        billing_region="eastus2",
        billing_mode="global-standard",
        billing_meters=_test_billing_meters(),
    )
    drifted_config = config.model_copy(update={"deployment": "different-deployment"})

    with pytest.raises(ValidationError, match="tariff route.*provider config"):
        ProviderTokenTariff.model_validate(
            {
                **tariff.model_dump(mode="json"),
                "provider_config": drifted_config.model_dump(mode="json"),
            }
        )
    with pytest.raises(ValidationError, match="tariff route.*provider config"):
        ProviderCostMeter(
            provider_config=drifted_config,
            price=tariff.price,
            tariff_provenance=tariff.provenance,
            tariff_evidence_receipt=_test_receipt(
                config=config,
                price=tariff.price,
                provenance=tariff.provenance,
            ),
        )


@pytest.mark.parametrize(
    "config",
    [
        ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model="zai.glm-5",
        ),
        ProviderConfig(
            kind=ProviderKind.AZURE_OPENAI,
            model_type="gpt-5.5",
            model="gpt-5.5",
            deployment="deployment",
            api_version="2026-06-01",
            reasoning_effort="high",
            responses_api_version="v1",
        ),
    ],
)
def test_tariff_rejects_routes_with_environment_resolved_coordinates(
    config: ProviderConfig,
) -> None:
    with pytest.raises(ValidationError, match="explicit (Bedrock region|Azure endpoint)"):
        ProviderTokenTariff.from_usd_per_million(
            provider_config=config,
            input_usd="1",
            output_usd="1",
            source_snapshots=(_test_source_snapshot(),),
            source_bindings=_test_source_bindings(config),
            verified_on=date(2026, 7, 19),
            effective_on=date(2026, 7, 1),
            currency="USD",
            price_unit="per_1m_tokens",
            billing_region="us-east-1",
            billing_mode="test-mode",
            billing_meters=_test_billing_meters(
                billing_region="us-east-1",
                billing_mode="test-mode",
            ),
        )


def test_generic_mutable_url_without_snapshot_and_route_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError, match="source_snapshots"):
        ProviderTariffProvenance.model_validate(
            {
                "verified_on": date(2026, 7, 19),
            }
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_snapshot_digest", "sha256:" + "0" * 64, "zero digest"),
        (
            "source_locator",
            "https://user@example.test/pricing",
            "without credentials",
        ),
        ("source_locator", "https://example.test/pricing?x=y", "query"),
    ],
)
def test_tariff_provenance_rejects_placeholder_or_secret_bearing_source_evidence(
    field: str,
    value: str,
    message: str,
) -> None:
    payload = catalog_provider_token_tariffs()[0].provenance.model_dump(mode="json")
    source = payload["source_snapshots"][0]
    assert isinstance(source, dict)
    source[field] = value

    with pytest.raises(ValidationError, match=message):
        ProviderTariffProvenance.model_validate(payload)


@pytest.mark.parametrize(
    "omitted_field",
    ["source_snapshots", "effective_on", "currency", "price_unit", "route"],
)
def test_tariff_provenance_requires_every_authority_field(omitted_field: str) -> None:
    payload = catalog_provider_token_tariffs()[0].provenance.model_dump(mode="json")
    del payload[omitted_field]

    with pytest.raises(ValidationError, match=omitted_field):
        ProviderTariffProvenance.model_validate(payload)


def test_tariff_provenance_requires_canonical_sources_and_meter_references() -> None:
    payload = catalog_provider_token_tariffs()[0].provenance.model_dump(mode="json")
    payload["source_snapshots"] = list(reversed(payload["source_snapshots"]))
    with pytest.raises(ValidationError, match="unique ascending source IDs"):
        ProviderTariffProvenance.model_validate(payload)

    payload = catalog_provider_token_tariffs()[0].provenance.model_dump(mode="json")
    route = payload["route"]
    assert isinstance(route, dict)
    meters = route["billing_meters"]
    assert isinstance(meters, list)
    meter = meters[0]
    assert isinstance(meter, dict)
    meter["source_id"] = "missing-source"
    with pytest.raises(ValidationError, match="unknown source snapshot"):
        ProviderTariffProvenance.model_validate(payload)

    payload = catalog_provider_token_tariffs()[0].provenance.model_dump(mode="json")
    sources = payload["source_snapshots"]
    assert isinstance(sources, list)
    meter_source = next(
        source for source in sources if source["source_id"] == "aws-bedrock-meter-map-20260703"
    )
    semantic_source = next(
        source for source in sources if source["source_id"] == "aws-bedrock-pricing-page-20260719"
    )
    meter_source["role"] = "route_definition"
    semantic_source["role"] = "rate_catalog"
    with pytest.raises(ValidationError, match="billing meters require rate_catalog sources"):
        ProviderTariffProvenance.model_validate(payload)


def test_tariff_provenance_requires_source_claim_bindings() -> None:
    payload = catalog_provider_token_tariffs()[0].provenance.model_dump(mode="json")
    bindings = payload["source_bindings"]
    assert isinstance(bindings, list)
    route_binding = next(binding for binding in bindings if binding["claim"] == "route_identity")
    route_binding["source_id"] = "aws-bedrock-meter-map-20260703"
    with pytest.raises(ValidationError, match="unbound non-rate source"):
        ProviderTariffProvenance.model_validate(payload)


def test_tariff_provenance_rejects_future_effective_or_publication_dates() -> None:
    payload = catalog_provider_token_tariffs()[1].provenance.model_dump(mode="json")
    payload["effective_on"] = "2026-07-20"
    route = payload["route"]
    assert isinstance(route, dict)
    meters = route["billing_meters"]
    assert isinstance(meters, list)
    for meter in meters:
        meter["effective_on"] = "2026-07-20"
    with pytest.raises(ValidationError, match="effective after its verification date"):
        ProviderTariffProvenance.model_validate(payload)

    payload = catalog_provider_token_tariffs()[1].provenance.model_dump(mode="json")
    sources = payload["source_snapshots"]
    assert isinstance(sources, list)
    sources[0]["published_on"] = "2026-07-20"
    with pytest.raises(ValidationError, match="published after its verification date"):
        ProviderTariffProvenance.model_validate(payload)


def test_every_route_binding_must_canonicalize_to_the_runtime_model() -> None:
    tariff = catalog_provider_token_tariffs()[0]
    payload = tariff.provenance.model_dump(mode="json")
    bindings = [
        ProviderTariffSourceBinding.model_validate(binding)
        for binding in payload["source_bindings"]
    ]
    route_index = next(
        index for index, binding in enumerate(bindings) if binding.claim == "route_identity"
    )
    bindings[route_index] = bindings[route_index].model_copy(
        update={"canonical_value": "different-runtime-model"}
    )
    payload["source_bindings"] = [
        binding.model_dump(mode="json") for binding in _sorted_source_bindings(bindings)
    ]

    with pytest.raises(ValidationError, match="runtime model"):
        ProviderTariffProvenance.model_validate(payload)

    payload = catalog_provider_token_tariffs()[0].provenance.model_dump(mode="json")
    bindings = payload["source_bindings"]
    assert isinstance(bindings, list)
    for route_binding in (binding for binding in bindings if binding["claim"] == "route_identity"):
        route_binding["canonical_value"] = "different-route"
    with pytest.raises(ValidationError, match="route identity"):
        ProviderTariffProvenance.model_validate(payload)

    payload = catalog_provider_token_tariffs()[0].provenance.model_dump(mode="json")
    route = payload["route"]
    assert isinstance(route, dict)
    meters = route["billing_meters"]
    assert isinstance(meters, list)
    meters[0]["source_record_path"] = "/different-rate-record"
    with pytest.raises(ValidationError, match="targets an unknown billing meter"):
        ProviderTariffProvenance.model_validate(payload)


def test_rate_catalog_route_identity_binds_both_glm_meters() -> None:
    glm = catalog_provider_token_tariff(
        ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model_type="glm-5",
            model="zai.glm-5",
            region="us-east-1",
        )
    )
    route_targets = {
        (binding.target_meter_source_id, binding.target_meter_record_path)
        for binding in glm.provenance.source_bindings
        if binding.claim == "route_identity"
    }
    meter_coordinates = {
        (meter.source_id, meter.source_record_path) for meter in glm.provenance.route.billing_meters
    }
    assert route_targets == meter_coordinates


def test_tariff_price_ceiling_cannot_understate_bound_rate_records() -> None:
    tariff = catalog_provider_token_tariffs()[0]
    understated = TokenPriceCeiling(
        input_nano_usd_per_token=1,
        output_nano_usd_per_token=1,
    )

    with pytest.raises(ValidationError, match="understates retained billing meter rates"):
        ProviderTokenTariff(
            provider_config=tariff.provider_config,
            price=understated,
            provenance=tariff.provenance,
        )
    with pytest.raises(ValidationError, match="understates retained billing meter rates"):
        ProviderCostMeter(
            provider_config=tariff.provider_config,
            price=understated,
            tariff_provenance=tariff.provenance,
            tariff_evidence_receipt=provider_cost_meter(tariff).tariff_evidence_receipt,
        )


def test_nested_tariff_models_are_defensively_revalidated() -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example-resource.openai.azure.com",
        deployment="audited-deployment",
        api_version="2026-06-01",
    )
    invalid_source = _test_source_snapshot().model_copy(
        update={"source_locator": "https://user:secret@example.test/pricing"}
    )

    with pytest.raises(ValidationError, match="without credentials"):
        ProviderTokenTariff.from_usd_per_million(
            provider_config=config,
            input_usd="1",
            output_usd="2",
            source_snapshots=(invalid_source,),
            source_bindings=_test_source_bindings(config),
            verified_on=date(2026, 7, 19),
            effective_on=date(2026, 7, 1),
            currency="USD",
            price_unit="per_1m_tokens",
            billing_region="eastus2",
            billing_mode="global-standard",
            billing_meters=_test_billing_meters(),
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://user@example-resource.openai.azure.com",
        "https://example-resource.openai.azure.com?x=y",
    ],
)
def test_azure_tariff_route_rejects_endpoint_credential_channels(endpoint: str) -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint=endpoint,
        deployment="audited-deployment",
        api_version="2026-06-01",
    )

    with pytest.raises(ValidationError, match="without credentials, query, or fragment"):
        ProviderTokenTariff.from_usd_per_million(
            provider_config=config,
            input_usd="1",
            output_usd="2",
            source_snapshots=(_test_source_snapshot(),),
            source_bindings=_test_source_bindings(config),
            verified_on=date(2026, 7, 19),
            effective_on=date(2026, 7, 1),
            currency="USD",
            price_unit="per_1m_tokens",
            billing_region="eastus2",
            billing_mode="global-standard",
            billing_meters=_test_billing_meters(),
        )


def test_tariff_provenance_is_immutable_and_rejects_unpriced_dimension_claims() -> None:
    tariff = catalog_provider_token_tariffs()[0]

    with pytest.raises(ValidationError, match="frozen"):
        tariff.provenance.verified_on = date(2026, 7, 20)
    with pytest.raises(ValidationError, match="priced_usage_dimensions"):
        type(tariff.provenance).model_validate(
            {
                **tariff.provenance.model_dump(mode="json"),
                "priced_usage_dimensions": ("input_tokens", "cached_input_tokens"),
            }
        )
