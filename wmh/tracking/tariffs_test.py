"""Tests for immutable, route-bound provider tariff snapshots."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

import wmh.tracking.tariffs as tariffs_module
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.tracking.budget import (
    ProviderCostMeter,
    ProviderTariffBillingMeter,
    ProviderTariffProvenance,
    ProviderTariffRoute,
    TokenPriceCeiling,
)
from wmh.tracking.pricing import price_for
from wmh.tracking.tariffs import (
    ProviderTokenTariff,
    catalog_provider_token_tariff,
    catalog_provider_token_tariffs,
    provider_cost_meter,
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


@pytest.mark.parametrize(
    (
        "model_type",
        "model_id",
        "input_nano_usd",
        "output_nano_usd",
        "source_locator",
        "source_snapshot_digest",
        "effective_on",
        "billing_sku",
        "input_rate_id",
        "output_rate_id",
    ),
    [
        (
            "claude-haiku-4-5",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            1_100,
            5_500,
            "https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/"
            "bedrockfoundationmodels/USD/current/bedrockfoundationmodels.json",
            "sha256:70ac2fe2f4153bf763492345b2029f06fefdb683023c420319a5b25679f02a11",
            date(2026, 7, 19),
            "geo-cross-region",
            "JQDUC8Q4K8C6GSGH.4799GE89SK.6YS6EN2CT7",
            "X629GDA2GXAP6R54.4799GE89SK.6YS6EN2CT7",
        ),
        (
            "glm-5",
            "zai.glm-5",
            1_000,
            3_200,
            "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/"
            "AmazonBedrock/current/index.json",
            "sha256:2df66ba105f0d831725f825d73bbb8c01d3d4fae2bc64ee5527619c62aedced2",
            date(2026, 7, 1),
            "on-demand",
            "YTB2BH9W4UZVKTEG.JRTCKXETXF.6YS6EN2CT7",
            "8RQBEKEP5KG2MZY7.JRTCKXETXF.6YS6EN2CT7",
        ),
        (
            "claude-opus-4-8",
            "us.anthropic.claude-opus-4-8",
            5_500,
            27_500,
            "https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/"
            "bedrockfoundationmodels/USD/current/bedrockfoundationmodels.json",
            "sha256:70ac2fe2f4153bf763492345b2029f06fefdb683023c420319a5b25679f02a11",
            date(2026, 7, 19),
            "geo-cross-region",
            "4AVHTD2NXFKSU6HU.4799GE89SK.6YS6EN2CT7",
            "YKJ5FPMCZAQF5BHF.4799GE89SK.6YS6EN2CT7",
        ),
    ],
)
def test_catalog_freezes_verified_bedrock_routes_and_exact_nominal_prices(
    model_type: str,
    model_id: str,
    input_nano_usd: int,
    output_nano_usd: int,
    source_locator: str,
    source_snapshot_digest: str,
    effective_on: date,
    billing_sku: str,
    input_rate_id: str,
    output_rate_id: str,
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
    assert tariff.provenance.source_locator == source_locator
    assert tariff.provenance.source_snapshot_digest == source_snapshot_digest
    assert tariff.provenance.currency == "USD"
    assert tariff.provenance.price_unit == "per_1m_tokens"
    assert tariff.provenance.route.provider_config == config
    assert tariff.provenance.route.billing_region == "us-east-1"
    assert tariff.provenance.route.billing_sku == billing_sku
    assert tuple(
        (meter.usage_dimension, meter.rate_id) for meter in tariff.provenance.route.billing_meters
    ) == (
        ("input_tokens", input_rate_id),
        ("output_tokens", output_rate_id),
    )
    assert tariff.price == TokenPriceCeiling(
        input_nano_usd_per_token=input_nano_usd,
        output_nano_usd_per_token=output_nano_usd,
    )
    assert meter == ProviderCostMeter(
        provider_config=config,
        price=tariff.price,
        tariff_provenance=tariff.provenance,
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


@pytest.mark.parametrize(
    "billing_meters",
    [
        (ProviderTariffBillingMeter(usage_dimension="input_tokens", rate_id="input-meter"),),
        (
            ProviderTariffBillingMeter(usage_dimension="output_tokens", rate_id="output-meter"),
            ProviderTariffBillingMeter(usage_dimension="input_tokens", rate_id="input-meter"),
        ),
        (
            ProviderTariffBillingMeter(usage_dimension="input_tokens", rate_id="input-meter-one"),
            ProviderTariffBillingMeter(usage_dimension="input_tokens", rate_id="input-meter-two"),
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
            billing_sku="on-demand",
            billing_meters=billing_meters,
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
    tariff = ProviderTokenTariff.from_usd_per_million(
        provider_config=config,
        input_usd="1",
        output_usd="2",
        source_locator="https://example.test/immutable-azure-price-record",
        source_snapshot_digest="sha256:" + "a" * 64,
        verified_on=date(2026, 7, 19),
        effective_on=date(2026, 7, 1),
        currency="USD",
        price_unit="per_1m_tokens",
        billing_region="eastus2",
        billing_sku="global-standard",
        input_rate_id="synthetic-input-meter",
        output_rate_id="synthetic-output-meter",
    )

    meter = provider_cost_meter(tariff, input_overhead_tokens=16_384)

    assert meter.provider_config == config
    assert meter.price == TokenPriceCeiling(
        input_nano_usd_per_token=1_000,
        output_nano_usd_per_token=2_000,
    )
    assert meter.input_overhead_tokens == 16_384
    assert meter.tariff_provenance == tariff.provenance
    assert meter.model_dump(mode="json")["provider_config"] == config.model_dump(mode="json")


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
        source_locator="https://example.test/immutable-azure-price-record",
        source_snapshot_digest="sha256:" + "a" * 64,
        verified_on=date(2026, 7, 19),
        effective_on=date(2026, 7, 1),
        currency="USD",
        price_unit="per_1m_tokens",
        billing_region="eastus2",
        billing_sku="global-standard",
        input_rate_id="synthetic-input-meter",
        output_rate_id="synthetic-output-meter",
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
            source_locator="https://example.test/pricing",
            source_snapshot_digest="sha256:" + "a" * 64,
            verified_on=date(2026, 7, 19),
            effective_on=date(2026, 7, 1),
            currency="USD",
            price_unit="per_1m_tokens",
            billing_region="us-east-1",
            billing_sku="test-sku",
            input_rate_id="test-input-meter",
            output_rate_id="test-output-meter",
        )


def test_generic_mutable_url_without_snapshot_and_route_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError, match="source_snapshot_digest"):
        ProviderTariffProvenance.model_validate(
            {
                "source_locator": "https://example.test/current-pricing",
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
            "cannot contain credentials",
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
    payload[field] = value

    with pytest.raises(ValidationError, match=message):
        ProviderTariffProvenance.model_validate(payload)


@pytest.mark.parametrize(
    "omitted_field",
    ["source_snapshot_digest", "effective_on", "currency", "price_unit", "route"],
)
def test_tariff_provenance_requires_every_authority_field(omitted_field: str) -> None:
    payload = catalog_provider_token_tariffs()[0].provenance.model_dump(mode="json")
    del payload[omitted_field]

    with pytest.raises(ValidationError, match=omitted_field):
        ProviderTariffProvenance.model_validate(payload)


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
            source_locator="https://example.test/immutable-azure-price-record",
            source_snapshot_digest="sha256:" + "a" * 64,
            verified_on=date(2026, 7, 19),
            effective_on=date(2026, 7, 1),
            currency="USD",
            price_unit="per_1m_tokens",
            billing_region="eastus2",
            billing_sku="global-standard",
            input_rate_id="synthetic-input-meter",
            output_rate_id="synthetic-output-meter",
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
