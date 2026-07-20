"""Private builders for synthetic tariff evidence used only by unit tests."""

from datetime import date

from wmh.providers.base import ProviderConfig
from wmh.tracking.budget import ProviderTariffProvenance, ProviderTariffRoute


def synthetic_tariff_provenance(
    provider_config: ProviderConfig,
    *,
    default_billing_region: str = "test-region",
) -> ProviderTariffProvenance:
    """Build minimal synthetic provenance for an isolated no-dispatch unit test."""
    return ProviderTariffProvenance(
        source_locator="https://example.test/provider-pricing",
        source_snapshot_digest="sha256:" + "f" * 64,
        verified_on=date(2026, 7, 19),
        effective_on=date(2026, 7, 1),
        currency="USD",
        price_unit="per_1m_tokens",
        route=ProviderTariffRoute(
            provider_config=provider_config,
            billing_region=provider_config.region or default_billing_region,
            billing_sku="test-sku",
        ),
    )
