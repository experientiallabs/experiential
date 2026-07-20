"""Private builders for synthetic tariff evidence used only by unit tests."""

from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.tracking.budget import (
    ProviderCostMeter,
    ProviderTariffEvidenceReceipt,
    ProviderTariffProvenance,
    ProviderTariffVerifiedSource,
    TokenPriceCeiling,
    provider_tariff_claim_digest,
    provider_tariff_evidence_verifier_digest,
    provider_tariff_validated_records,
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
