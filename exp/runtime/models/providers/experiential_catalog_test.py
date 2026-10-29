"""Exact-route and origin tests for Experiential catalog projection."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.models.providers.experiential_catalog import catalog_url, model_metadata


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("https://api.experientiallabs.ai/v1", "https://api.experientiallabs.ai/api/models"),
        (
            "https://preview.example.test/prefix/v1/",
            "https://preview.example.test/prefix/api/models",
        ),
    ],
)
def test_catalog_stays_on_the_selected_gateway_origin(base: str, expected: str) -> None:
    """Preview and prefixed deployments keep credentials on the configured origin."""
    assert catalog_url(base) == expected


def test_catalog_rejects_an_ambiguous_gateway_url() -> None:
    """A malformed base cannot silently redirect metadata discovery elsewhere."""
    with pytest.raises(ValueError, match="must end in /v1"):
        catalog_url("https://gateway.example.test/v1?origin=elsewhere")


def test_metadata_uses_the_default_route_not_provider_display_order() -> None:
    """The published primary route owns pricing and can deny a model-level feature."""
    entry: JsonObject = {
        "model": {
            "slug": "chat",
            "output_modalities": ["text"],
            "context_window": 128000,
            "max_output_tokens": 32000,
            "supported_params": {"structured_outputs": True},
        },
        "providers": [
            {"id": "other", "status": "active", "input_nano_usd_per_million": 1},
            {
                "id": "primary",
                "status": "active",
                "routable": True,
                "input_nano_usd_per_million": 200000000,
                "output_nano_usd_per_million": 1200000000,
                "capabilities": {
                    "maximum_output_tokens": 16000,
                    "supports_structured_output": False,
                    "supports_reasoning": True,
                    "reasoning_default_effort": "high",
                },
            },
        ],
        "default_provider_ids": ["primary", "other"],
    }

    result = model_metadata(entry)

    assert result is not None
    slug, metadata = result
    assert slug == "chat"
    assert metadata["supports_completions"] is True
    assert metadata["supports_structured_output"] is False
    assert metadata["context_window_tokens"] == 128000
    assert metadata["maximum_output_tokens"] == 16000
    assert metadata["reasoning_effort"] == "high"
    assert metadata["pricing"] == {
        "input_nano_usd_per_million_tokens": 200000000,
        "output_nano_usd_per_million_tokens": 1200000000,
        "cached_input_nano_usd_per_million_tokens": 0,
        "cache_write_nano_usd_per_million_tokens": 0,
    }
    entry["default_provider_ids"] = ["missing"]
    assert model_metadata(entry) is None


@pytest.mark.parametrize("reporting", [True, False, None, "absent"])
def test_cache_prices_follow_the_gateway_billing_lanes(reporting: bool | str | None) -> None:
    """Inactive cache lanes are free; reported but unpriced usage remains unknown."""
    capabilities: JsonObject = {}
    if reporting != "absent":
        capabilities = {
            "reports_cached_input_tokens": reporting,
            "reports_cache_creation_input_tokens": reporting,
        }
    result = model_metadata(
        {
            "model": {"slug": "chat", "output_modalities": ["text"]},
            "providers": [{"id": "primary", "status": "active", "capabilities": capabilities}],
            "default_provider_ids": ["primary"],
        }
    )
    assert result is not None
    expected = 0 if reporting is False or reporting == "absent" else None
    assert result[1]["pricing"] == {
        "input_nano_usd_per_million_tokens": None,
        "output_nano_usd_per_million_tokens": None,
        "cached_input_nano_usd_per_million_tokens": expected,
        "cache_write_nano_usd_per_million_tokens": expected,
    }


def test_absent_capability_contract_does_not_invent_cache_prices() -> None:
    """Without any published capability object even cache billing applicability is unknown."""
    result = model_metadata(
        {
            "model": {"slug": "chat"},
            "providers": [{"id": "primary", "status": "active"}],
            "default_provider_ids": ["primary"],
        }
    )
    assert result is not None
    assert result[1]["pricing"] == {
        "input_nano_usd_per_million_tokens": None,
        "output_nano_usd_per_million_tokens": None,
        "cached_input_nano_usd_per_million_tokens": None,
        "cache_write_nano_usd_per_million_tokens": None,
    }


@pytest.mark.parametrize(
    "capabilities", [{"supports_completions": False}, {"supports_embeddings": True}]
)
def test_text_modalities_cannot_override_an_explicit_protocol_declaration(
    capabilities: JsonObject,
) -> None:
    """An embedding route can advertise text without becoming a chat model."""
    result = model_metadata(
        {
            "model": {"slug": "embed", "output_modalities": ["text"]},
            "providers": [{"id": "primary", "status": "active", "capabilities": capabilities}],
            "default_provider_ids": ["primary"],
        }
    )

    assert result is not None
    assert result[1]["supports_completions"] is False
