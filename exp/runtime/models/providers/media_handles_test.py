"""Tests for provider media handle admission and shared encoding."""

from __future__ import annotations

import pytest

from exp.common.models.content import MediaHandle
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.media_handles import (
    ANTHROPIC_HANDLE_PROVIDERS,
    BEDROCK_HANDLE_PROVIDERS,
    GEMINI_HANDLE_PROVIDERS,
    OPENAI_HANDLE_PROVIDERS,
    bedrock_s3_location,
    handle_provider_mismatch,
    preflight_media_handles,
    require_handle_provider,
)

_OPENAI = MediaHandle(provider="openai", reference="file-abc")
_GEMINI = MediaHandle(
    provider="gemini", reference="https://generativelanguage.googleapis.com/v1beta/files/abc"
)
_S3 = MediaHandle(provider="bedrock", reference="s3://bkt/key", bucket_owner="123456789012")


def test_preflight_admits_nothing_to_do() -> None:
    """A request without handles passes every route, declared or not."""
    preflight_media_handles((), supports_media_handle_input=False, route_provider=None)


def test_preflight_requires_the_route_declaration() -> None:
    """An undeclared route refuses any handle, even on the matching provider."""
    with pytest.raises(ProviderCapabilityError, match="media_handle_input") as error:
        preflight_media_handles(
            (_OPENAI,), supports_media_handle_input=False, route_provider="openai"
        )
    assert error.value.detail is None


def test_preflight_requires_the_routes_provider_to_match() -> None:
    """A declared route on another provider refuses the handle by provider."""
    with pytest.raises(ProviderCapabilityError, match="media_handle_provider") as error:
        preflight_media_handles(
            (_OPENAI,), supports_media_handle_input=True, route_provider="gemini"
        )
    detail = error.value.detail
    assert detail == handle_provider_mismatch(_OPENAI, "gemini")
    assert detail is not None
    assert "uploaded to openai" in detail
    assert "routes to gemini" in detail
    with pytest.raises(ProviderCapabilityError, match="media_handle_provider"):
        preflight_media_handles((_OPENAI,), supports_media_handle_input=True, route_provider=None)
    preflight_media_handles((_OPENAI,), supports_media_handle_input=True, route_provider="openai")


def test_preflight_checks_every_handle_in_caller_order() -> None:
    """The first handle from another provider names the refusal."""
    with pytest.raises(ProviderCapabilityError, match="media_handle_provider") as error:
        preflight_media_handles(
            (_GEMINI, _OPENAI), supports_media_handle_input=True, route_provider="gemini"
        )
    assert error.value.detail is not None
    assert "uploaded to openai" in error.value.detail


def test_mismatch_message_without_a_route_names_the_missing_provider() -> None:
    """An encoder-level refusal cannot name a route, so it names the gap."""
    assert "has no openai route" in handle_provider_mismatch(_OPENAI, None)


@pytest.mark.parametrize(
    ("providers", "handle"),
    [
        (OPENAI_HANDLE_PROVIDERS, _GEMINI),
        (ANTHROPIC_HANDLE_PROVIDERS, _OPENAI),
        (GEMINI_HANDLE_PROVIDERS, _S3),
        (BEDROCK_HANDLE_PROVIDERS, _GEMINI),
    ],
)
def test_require_handle_provider_refuses_a_foreign_handle(
    providers: frozenset[str], handle: MediaHandle
) -> None:
    """Each dialect's provider set refuses a handle it cannot resolve."""
    with pytest.raises(ProviderCapabilityError, match="media_handle_provider"):
        require_handle_provider(handle, providers)


def test_gemini_dialect_resolves_both_files_and_gcs_handles() -> None:
    """``generateContent`` serves Gemini Files URIs and ``gs://`` objects alike."""
    require_handle_provider(_GEMINI, GEMINI_HANDLE_PROVIDERS)
    require_handle_provider(
        MediaHandle(provider="vertex", reference="gs://bkt/key"), GEMINI_HANDLE_PROVIDERS
    )


def test_s3_location_carries_the_owner_only_when_set() -> None:
    """``bucketOwner`` appears exactly when the caller named a cross-account owner."""
    assert bedrock_s3_location(_S3) == {"uri": "s3://bkt/key", "bucketOwner": "123456789012"}
    assert bedrock_s3_location(MediaHandle(provider="bedrock", reference="s3://bkt/key")) == {
        "uri": "s3://bkt/key"
    }
