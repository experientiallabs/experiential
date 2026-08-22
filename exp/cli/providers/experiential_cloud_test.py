"""Hosted Experiential Cloud setup constants and origin resolution."""

from __future__ import annotations

from exp.cli.providers.experiential_cloud import (
    CATALOG_PROVIDER,
    HOSTED_GATEWAY_API_KEY_ENV,
    HOSTED_GATEWAY_DEFAULT_BASE_URL,
    HOSTED_GATEWAY_URL_ENV,
    SETUP_PICKER_LABEL,
    SETUP_PICKER_NAME,
    hosted_gateway_base_url,
)


def test_picker_persists_the_hosted_openai_compatible_lane() -> None:
    """The picker name is product copy; the catalog provider stays frozen."""
    assert SETUP_PICKER_NAME == "experiential-cloud"
    assert SETUP_PICKER_LABEL == "Experiential Cloud"
    assert CATALOG_PROVIDER == "openai-compatible"
    assert HOSTED_GATEWAY_API_KEY_ENV == "EXPLABS_API_KEY"
    assert HOSTED_GATEWAY_DEFAULT_BASE_URL == "https://api.experientiallabs.ai/v1"


def test_hosted_origin_defaults_to_production_and_honors_override() -> None:
    """Empty override keeps production; preview or staging may replace it."""
    assert hosted_gateway_base_url({}) == HOSTED_GATEWAY_DEFAULT_BASE_URL
    assert hosted_gateway_base_url({HOSTED_GATEWAY_URL_ENV: "  "}) == (
        HOSTED_GATEWAY_DEFAULT_BASE_URL
    )
    assert (
        hosted_gateway_base_url(
            {HOSTED_GATEWAY_URL_ENV: "https://api.staging.experientiallabs.ai/v1"}
        )
        == "https://api.staging.experientiallabs.ai/v1"
    )
