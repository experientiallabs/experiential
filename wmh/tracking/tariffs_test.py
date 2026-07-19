"""Tests for immutable, route-bound provider tariff snapshots."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.tracking.budget import ProviderCostMeter, TokenPriceCeiling
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
            1_000,
            5_000,
        ),
        ("glm-5", "zai.glm-5", 1_000, 3_200),
        ("claude-opus-4-8", "us.anthropic.claude-opus-4-8", 5_000, 25_000),
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
    assert tariff.provenance.source_locator == "https://aws.amazon.com/bedrock/pricing/"
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
        verified_on=date(2026, 7, 19),
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
            verified_on=date(2026, 7, 19),
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
