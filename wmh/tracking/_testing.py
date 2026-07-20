"""Private builders for synthetic tariff evidence used only by unit tests."""

from datetime import date
from decimal import Decimal

from wmh.providers.base import ProviderConfig, ProviderKind
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


def synthetic_tariff_provenance(
    provider_config: ProviderConfig,
    *,
    default_billing_region: str = "test-region",
) -> ProviderTariffProvenance:
    """Build fully retained synthetic provenance for one no-dispatch unit test."""
    billing_region = provider_config.region or default_billing_region
    return ProviderTariffProvenance(
        source_snapshots=(
            ProviderTariffSourceSnapshot(
                source_id="test-rate-catalog",
                role="rate_catalog",
                source_locator="https://example.test/provider-pricing",
                source_snapshot_digest="sha256:" + "f" * 64,
                media_type="application/json",
                content_encoding="identity",
                retained_artifact=ProviderTariffRetainedArtifact(
                    storage_kind="https",
                    locator="https://example.test/evidence/test-rate-catalog.json.gz",
                    artifact_digest="sha256:" + "e" * 64,
                    content_encoding="gzip",
                ),
            ),
        ),
        source_bindings=_synthetic_tariff_source_bindings(provider_config),
        verified_on=date(2026, 7, 19),
        effective_on=date(2026, 7, 1),
        currency="USD",
        price_unit="per_1m_tokens",
        route=ProviderTariffRoute(
            provider_config=provider_config,
            billing_region=billing_region,
            billing_mode="test-mode",
            billing_meters=(
                ProviderTariffBillingMeter(
                    usage_dimension="input_tokens",
                    source_id="test-rate-catalog",
                    source_record_path="/input",
                    sku_id="test-input-sku",
                    rate_id="test-input-rate",
                    billing_region=billing_region,
                    billing_mode="test-mode",
                    effective_on=date(2026, 7, 1),
                    source_price_usd=Decimal("0.001"),
                    source_price_unit="per_1m_tokens",
                ),
                ProviderTariffBillingMeter(
                    usage_dimension="output_tokens",
                    source_id="test-rate-catalog",
                    source_record_path="/output",
                    sku_id="test-output-sku",
                    rate_id="test-output-rate",
                    billing_region=billing_region,
                    billing_mode="test-mode",
                    effective_on=date(2026, 7, 1),
                    source_price_usd=Decimal("0.001"),
                    source_price_unit="per_1m_tokens",
                ),
            ),
        ),
    )


def _synthetic_tariff_source_bindings(
    provider_config: ProviderConfig,
) -> tuple[ProviderTariffSourceBinding, ...]:
    return (
        ProviderTariffSourceBinding(
            claim="route_identity",
            source_id="test-rate-catalog",
            source_record_path="/input/model",
            source_value=provider_config.model,
            canonical_value=provider_config.model,
            target_meter_source_id="test-rate-catalog",
            target_meter_record_path="/input",
        ),
        ProviderTariffSourceBinding(
            claim="route_identity",
            source_id="test-rate-catalog",
            source_record_path="/output/model",
            source_value=provider_config.model,
            canonical_value=provider_config.model,
            target_meter_source_id="test-rate-catalog",
            target_meter_record_path="/output",
        ),
        ProviderTariffSourceBinding(
            claim="usage_dimension",
            source_id="test-rate-catalog",
            source_record_path="/input/dimension",
            source_value="Input tokens",
            canonical_value="input_tokens",
            target_meter_source_id="test-rate-catalog",
            target_meter_record_path="/input",
        ),
        ProviderTariffSourceBinding(
            claim="usage_dimension",
            source_id="test-rate-catalog",
            source_record_path="/output/dimension",
            source_value="Output tokens",
            canonical_value="output_tokens",
            target_meter_source_id="test-rate-catalog",
            target_meter_record_path="/output",
        ),
    )


def synthetic_provider_cost_meter(
    *,
    provider_config: ProviderConfig,
    provenance: ProviderTariffProvenance,
    input_nano_usd_per_token: int,
    output_nano_usd_per_token: int,
    input_overhead_tokens: int = 8192,
) -> ProviderCostMeter:
    """Build a fully bound provider meter for an isolated no-dispatch unit test."""
    price = TokenPriceCeiling(
        input_nano_usd_per_token=input_nano_usd_per_token,
        output_nano_usd_per_token=output_nano_usd_per_token,
    )
    return ProviderCostMeter(
        provider_config=provider_config,
        price=price,
        tariff_provenance=provenance,
        tariff_evidence_receipt=synthetic_tariff_evidence_receipt(
            provider_config=provider_config,
            price=price,
            provenance=provenance,
        ),
        input_overhead_tokens=input_overhead_tokens,
    )


def synthetic_tariff_evidence_receipt(
    *,
    provider_config: ProviderConfig,
    price: TokenPriceCeiling,
    provenance: ProviderTariffProvenance,
) -> ProviderTariffEvidenceReceipt:
    """Build audit-shaped evidence for tests that never issue a paid provider call."""
    profile = (
        "aws_bedrock_public_catalog_v1"
        if provider_config.kind is ProviderKind.BEDROCK
        else "azure_retail_arm_v1"
    )
    return ProviderTariffEvidenceReceipt(
        verifier_profile=profile,
        verifier_digest=provider_tariff_evidence_verifier_digest(profile),
        tariff_claim_digest=provider_tariff_claim_digest(
            provider_config=provider_config,
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
