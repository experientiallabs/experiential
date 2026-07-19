"""Tests for immutable, route-bound provider tariff snapshots."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.tracking.budget import (
    ProviderCostMeter,
    ProviderTariffProvenance,
    TokenPriceCeiling,
)
from wmh.tracking.pricing import price_for
from wmh.tracking.tariffs import (
    ProviderTokenTariff,
    catalog_provider_token_tariff,
    catalog_provider_token_tariffs,
    provider_cost_meter,
)


@pytest.mark.parametrize(
    ("model_type", "model_id", "input_nano_usd", "output_nano_usd"),
    [
        (
            "claude-haiku-4-5",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            1_100,
            5_500,
        ),
        ("glm-5", "zai.glm-5", 1_000, 3_200),
        ("claude-opus-4-8", "us.anthropic.claude-opus-4-8", 5_500, 27_500),
    ],
)
def test_catalog_freezes_verified_bedrock_routes_and_exact_nominal_prices(
    model_type: str,
    model_id: str,
    input_nano_usd: int,
    output_nano_usd: int,
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
    assert tariff.provenance.effective_on == date(2026, 7, 19)
    assert tariff.provenance.source_locator == "https://aws.amazon.com/bedrock/pricing/"
    assert tariff.provenance.source_snapshot_digest == (
        "sha256:1936d89d798e83cbee0d3d95a886a720c7e2de2bb6fe6e86cdfa3e249b5b8649"
    )
    assert tariff.provenance.currency == "USD"
    assert tariff.provenance.price_unit == "per_1m_tokens"
    assert tariff.provenance.route.provider_config == config
    assert tariff.provenance.route.billing_region == "us-east-1"
    assert tariff.provenance.route.billing_sku in {"geo-cross-region", "on-demand"}
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


def test_descriptive_prices_match_every_hard_budget_tariff_route() -> None:
    for tariff in catalog_provider_token_tariffs():
        descriptive_price = price_for(tariff.provider_config.model)

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
        input_usd="5",
        output_usd="30",
        source_locator="https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/",
        source_snapshot_digest="sha256:" + "a" * 64,
        verified_on=date(2026, 7, 19),
        effective_on=date(2026, 7, 1),
        currency="USD",
        price_unit="per_1m_tokens",
        billing_region="eastus2",
        billing_sku="global-standard",
    )

    meter = provider_cost_meter(tariff, input_overhead_tokens=16_384)

    assert meter.provider_config == config
    assert meter.price == TokenPriceCeiling(
        input_nano_usd_per_token=5_000,
        output_nano_usd_per_token=30_000,
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
        input_usd="5",
        output_usd="30",
        source_locator="https://example.test/immutable-azure-price-record",
        source_snapshot_digest="sha256:" + "a" * 64,
        verified_on=date(2026, 7, 19),
        effective_on=date(2026, 7, 1),
        currency="USD",
        price_unit="per_1m_tokens",
        billing_region="eastus2",
        billing_sku="global-standard",
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
            input_usd="5",
            output_usd="30",
            source_locator="https://example.test/immutable-azure-price-record",
            source_snapshot_digest="sha256:" + "a" * 64,
            verified_on=date(2026, 7, 19),
            effective_on=date(2026, 7, 1),
            currency="USD",
            price_unit="per_1m_tokens",
            billing_region="eastus2",
            billing_sku="global-standard",
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
