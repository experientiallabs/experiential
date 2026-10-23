"""Tests for the audio-only ElevenLabs connection client."""

from typing import cast

import pytest

from exp.common.models import BillingSource, ModelRequest, ModelSnapshot
from exp.runtime.models.providers.elevenlabs import ElevenLabsClient
from exp.runtime.models.providers.elevenlabs_audio import ELEVENLABS_AUDIO_DIALECT
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.registry import SUPPORTED_PROVIDERS


def _client() -> ElevenLabsClient:
    """Build one client against a loopback base URL."""
    return ElevenLabsClient(
        model=ModelSnapshot(
            billing_source=BillingSource.HOST_MANAGED,
            provider="elevenlabs",
            model_id="eleven_flash_v2_5",
            capabilities_sha256="a" * 64,
            connection_sha256="b" * 64,
        ),
        api_key="key-canary",
        base_url="http://127.0.0.1:9/v1/",
    )


def test_wire_profile_names_the_audio_roots_and_xi_api_key() -> None:
    """The profile carries ElevenLabs' routes and header, and never a bearer token."""
    profile = _client().gateway_wire_profile()
    assert profile.dialect == ELEVENLABS_AUDIO_DIALECT
    assert profile.speech_url == "http://127.0.0.1:9/v1/text-to-speech"
    assert profile.transcriptions_url == "http://127.0.0.1:9/v1/speech-to-text"
    assert dict(profile.headers) == {"xi-api-key": "key-canary"}
    assert profile.model_id == "eleven_flash_v2_5"
    assert "key-canary" not in repr(profile)


def test_completion_is_refused_and_the_provider_is_registered() -> None:
    """ElevenLabs serves no chat; the runtime registry can construct it."""
    with pytest.raises(ProviderCapabilityError):
        _client().complete(cast(ModelRequest, None))
    assert "elevenlabs" in SUPPORTED_PROVIDERS
