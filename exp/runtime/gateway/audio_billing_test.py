"""Tests for an audio rung's billing-mode choice."""

from datetime import UTC, datetime

from exp.common.models import (
    BilledUnitKind,
    BillingSource,
    ExactModelDeployment,
    GatewayDeploymentMetadata,
    GatewayLongContextTier,
    GatewayTokenPrices,
    GatewayUnitPrices,
    ModelCapabilities,
)
from exp.runtime.gateway.audio_billing import billing_mode


def _deployment(
    prices: GatewayTokenPrices, *, maximum_output_tokens: int | None = None
) -> ExactModelDeployment:
    """Build one host-managed deployment carrying the given price card."""
    return ExactModelDeployment(
        deployment_id="deployment-one",
        source_alias="voice",
        exact_model_id="exact-one",
        connection="connection-one",
        provider="openai",
        provider_model="tts-1",
        billing_source=BillingSource.HOST_MANAGED,
        capabilities=ModelCapabilities(maximum_output_tokens=maximum_output_tokens),
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        gateway=GatewayDeploymentMetadata(
            prices=prices,
            pricing_source="provider-docs",
            pricing_effective_at=datetime(2026, 10, 6, tzinfo=UTC),
        ),
    )


def test_a_unit_card_of_the_surface_kind_bills_units_else_tokens() -> None:
    """The lane's own price card decides: its unit card first, then token rates."""
    characters = GatewayTokenPrices(
        units=GatewayUnitPrices(kind=BilledUnitKind.CHARACTER, rates={"": 15_000})
    )
    tokens = GatewayTokenPrices(
        input_nano_usd_per_million_tokens=1, output_nano_usd_per_million_tokens=1
    )
    assert billing_mode(_deployment(characters), "speech") == "units"
    # A token lane of either surface bills only with a declared output ceiling.
    assert billing_mode(_deployment(tokens, maximum_output_tokens=2_000), "speech") == "tokens"
    assert billing_mode(_deployment(tokens), "speech") is None
    # A per-character card does not price transcription seconds.
    mixed = characters.model_copy(update=tokens.model_dump(exclude={"units"}))
    assert billing_mode(_deployment(mixed, maximum_output_tokens=2_000), "transcription") == (
        "tokens"
    )
    assert billing_mode(_deployment(mixed), "transcription") is None
    # A lane priced both ways could never settle both legs: it serves nothing.
    hybrid = characters.model_copy(update=tokens.model_dump(exclude={"units"}))
    assert billing_mode(_deployment(hybrid), "speech") is None
    # Only one of the two token rates cannot price the usage either.
    half = GatewayTokenPrices(input_nano_usd_per_million_tokens=1)
    assert billing_mode(_deployment(half), "speech") is None
    # A lane priced neither way gives no basis to pick the meter: it serves nothing.
    assert billing_mode(_deployment(GatewayTokenPrices()), "transcription") is None
    # A long-context schedule that leaves a rate unknown could not price the hold.
    tier = GatewayLongContextTier(
        input_threshold_tokens=200_000, input_nano_usd_per_million_tokens=2
    )
    tiered = tokens.model_copy(update={"long_context": tier})
    assert billing_mode(_deployment(tiered, maximum_output_tokens=2_000), "speech") is None
    # Token rates only on the long-context schedule are still a second meter.
    tier_only = characters.model_copy(
        update={
            "long_context": GatewayLongContextTier(
                input_threshold_tokens=200_000,
                input_nano_usd_per_million_tokens=2,
                output_nano_usd_per_million_tokens=2,
            )
        }
    )
    assert billing_mode(_deployment(tier_only), "speech") is None
