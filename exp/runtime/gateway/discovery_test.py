"""Tests for caller-facing model discovery objects."""

from __future__ import annotations

import pytest

from exp.common.models.catalog import GatewayDeploymentMetadata, GatewayTokenPrices
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.common.models.model import ModelCapabilities
from exp.runtime.gateway.discovery import (
    PublishedAliasMetadata,
    listing_metadata_by_alias,
    public_model_list,
    public_model_object,
    published_alias_metadata,
    require_granted_authority,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError

_AUTHORITY = ("coding", "revision-one", "a" * 64)
_WMO_LISTING_CONTRACT = {
    "id": "coding",
    "object": "model",
    "created": 0,
    "owned_by": "wmo",
    "wmo": {"alias_revision_id": "revision-one", "catalog_sha256": "a" * 64},
    "supports_completions": True,
    "supports_tools": True,
    "supports_structured_output": True,
    "maximum_output_tokens": 16_000,
    "pricing": {
        "input_micro_usd_per_million_tokens": 1_250_000,
        "output_micro_usd_per_million_tokens": 10_000_000,
        "cached_input_micro_usd_per_million_tokens": 125_000,
    },
}


def _deployment(
    *,
    capabilities: ModelCapabilities | None,
    prices: GatewayTokenPrices | None = None,
) -> ExactModelDeployment:
    """Build one catalog deployment used only for public listing projection.

    Args:
        capabilities: Authored capability snapshot, or ``None`` when undeclared.
        prices: Configured gateway prices, or defaults when omitted.

    Returns:
        A secret-free deployment whose public alias is ``coding``.
    """
    return ExactModelDeployment(
        deployment_id="coding",
        source_alias="coding",
        exact_model_id="exact-coding",
        connection="hosted",
        provider="openai-compatible",
        provider_model="hosted-coding",
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        capabilities=capabilities,
        gateway=GatewayDeploymentMetadata(prices=prices or GatewayTokenPrices()),
    )


def test_public_model_object_keeps_the_openai_keys_and_adds_authority() -> None:
    """The enriched object stays a valid OpenAI model for official clients."""
    assert public_model_object(_AUTHORITY) == {
        "id": "coding",
        "object": "model",
        "created": 0,
        "owned_by": "wmo",
        "wmo": {"alias_revision_id": "revision-one", "catalog_sha256": "a" * 64},
    }


def test_public_model_list_marks_even_an_empty_authority_envelope() -> None:
    """The caller can identify an authenticated gateway with no granted aliases."""
    assert public_model_list(()) == {
        "object": "list",
        "data": [],
        "wmo": {"authority_schema_version": 1},
    }


def test_require_granted_authority_returns_the_exact_granted_triple() -> None:
    """A granted alias resolves to its own frozen authority."""
    assert require_granted_authority((_AUTHORITY,), "coding") == _AUTHORITY


def test_unknown_and_ungranted_aliases_raise_the_identical_404() -> None:
    """The 404 never distinguishes an unknown alias from an ungranted one."""
    with pytest.raises(OpenAIProtocolError) as ungranted:
        require_granted_authority((_AUTHORITY,), "other-model")
    with pytest.raises(OpenAIProtocolError) as unknown:
        require_granted_authority((), "coding")

    assert ungranted.value.status_code == 404
    assert ungranted.value.detail == unknown.value.detail
    assert ungranted.value.detail.code == "model_not_found"
    assert "other-model" not in ungranted.value.detail.message
    assert (
        "GET /v1/models lists the model aliases available to this key."
        in ungranted.value.detail.message
    )


def test_public_model_object_copies_catalog_capabilities_limits_and_prices() -> None:
    """A unique catalog deployment publishes its declared fields and no invented ones."""
    metadata = published_alias_metadata(
        _deployment(
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_structured_output=True,
                maximum_output_tokens=16_000,
            ),
            prices=GatewayTokenPrices(
                input_micro_usd_per_million_tokens=1_250_000,
                output_micro_usd_per_million_tokens=10_000_000,
                cached_input_micro_usd_per_million_tokens=125_000,
            ),
        )
    )

    payload = public_model_object(_AUTHORITY, metadata=metadata)

    assert payload == _WMO_LISTING_CONTRACT
    assert "context_window_tokens" not in payload
    assert "cache_write" not in str(payload)


def test_public_model_object_omits_undeclared_catalog_fields() -> None:
    """An undeclared snapshot still marks completion support and invents nothing else."""
    payload = public_model_object(
        _AUTHORITY, metadata=published_alias_metadata(_deployment(capabilities=None))
    )

    assert payload["supports_completions"] is True
    assert "supports_tools" not in payload
    assert "supports_structured_output" not in payload
    assert "maximum_output_tokens" not in payload
    assert "pricing" not in payload
    assert "context_window_tokens" not in payload


def test_explicit_completion_denial_is_published() -> None:
    """A catalog ``supports_completions=False`` is copied instead of being overridden."""
    metadata = published_alias_metadata(
        _deployment(capabilities=ModelCapabilities(supports_completions=False))
    )

    assert metadata is not None
    assert metadata.supports_completions is False
    assert public_model_object(_AUTHORITY, metadata=metadata)["supports_completions"] is False


def test_missing_deployment_publishes_no_extension_fields() -> None:
    """A granted alias without a unique catalog deployment stays identity-only."""
    assert published_alias_metadata(None) is None
    assert public_model_object(_AUTHORITY) == {
        "id": "coding",
        "object": "model",
        "created": 0,
        "owned_by": "wmo",
        "wmo": {"alias_revision_id": "revision-one", "catalog_sha256": "a" * 64},
    }


def test_public_model_list_attaches_per_alias_catalog_metadata() -> None:
    """List entries receive only the metadata keyed to that granted alias."""
    metadata = published_alias_metadata(
        _deployment(
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_structured_output=True,
                maximum_output_tokens=16_000,
            ),
            prices=GatewayTokenPrices(
                input_micro_usd_per_million_tokens=1_250_000,
                output_micro_usd_per_million_tokens=10_000_000,
                cached_input_micro_usd_per_million_tokens=125_000,
            ),
        )
    )
    assert metadata is not None

    assert public_model_list((_AUTHORITY,), metadata_by_alias={"coding": metadata}) == {
        "object": "list",
        "data": [_WMO_LISTING_CONTRACT],
        "wmo": {"authority_schema_version": 1},
    }


def test_listing_metadata_by_alias_keeps_only_successful_lookups() -> None:
    """Failed catalog lookups are omitted so unmatched aliases stay identity-only."""

    def lookup(
        *, alias: str, revision_id: str, catalog_sha256: str
    ) -> PublishedAliasMetadata | None:
        """Return metadata only for the coding alias.

        Args:
            alias: Granted public alias.
            revision_id: Active revision.
            catalog_sha256: Frozen catalog digest.

        Returns:
            Catalog metadata for ``coding``, otherwise ``None``.
        """
        del revision_id, catalog_sha256
        if alias != "coding":
            return None
        return published_alias_metadata(
            _deployment(capabilities=ModelCapabilities(supports_tools=True))
        )

    published = listing_metadata_by_alias(
        (_AUTHORITY, ("other", "revision-two", "b" * 64)),
        lookup,
    )

    assert set(published) == {"coding"}
    assert published["coding"].supports_tools is True
