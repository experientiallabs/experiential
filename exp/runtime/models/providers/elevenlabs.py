"""ElevenLabs connection: speech and transcription on ElevenLabs' own wire.

ElevenLabs serves no chat or embeddings; a connection exists only so the
gateway's audio surfaces can dispatch to it. The client therefore describes
its wire for the native data plane (the ``xi-api-key`` header and the
``/text-to-speech`` and ``/speech-to-text`` roots) and refuses completion.
Request translation lives in :mod:`exp.runtime.models.providers.elevenlabs_audio`.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelRequest, ModelResponse
from exp.runtime.models.providers.base import GatewayWireProfile, ProviderHttpClient
from exp.runtime.models.providers.elevenlabs_audio import ELEVENLABS_AUDIO_DIALECT
from exp.runtime.models.providers.errors import ProviderCapabilityError


class ElevenLabsClient(ProviderHttpClient):
    """One ElevenLabs connection serving only the gateway's audio surfaces."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Refuse completion: ElevenLabs is an audio-only provider.

        Args:
            request: The completion request, never served.

        Raises:
            ProviderCapabilityError: Always; chat has no ElevenLabs wire.
        """
        del request
        raise ProviderCapabilityError(capability="chat_completion")

    def gateway_wire_profile(self) -> GatewayWireProfile:
        """Return the ElevenLabs audio wire for the native data plane.

        ``url`` names the speech root only because the profile requires one;
        no chat dispatch can reach it (admission routes only audio surfaces to
        this dialect). Admission renders the per-request speech endpoint (voice
        and output format) from ``speech_url``.
        """
        speech_url = f"{self._base_url}/text-to-speech"
        return GatewayWireProfile(
            dialect=ELEVENLABS_AUDIO_DIALECT,
            url=speech_url,
            speech_url=speech_url,
            transcriptions_url=f"{self._base_url}/speech-to-text",
            headers=self._headers(),
            model_id=self._model.model_id,
            timeout_seconds=self._timeout_seconds,
        )

    def _headers(self) -> dict[str, str]:
        """ElevenLabs authenticates with ``xi-api-key``, not a bearer token."""
        return {"xi-api-key": self._api_key}

    def _completion_path(self) -> str:
        """No completion route exists; see :meth:`complete`."""
        raise ProviderCapabilityError(capability="chat_completion")

    def _build_request(self, request: ModelRequest) -> JsonObject:
        """No completion body exists; see :meth:`complete`."""
        del request
        raise ProviderCapabilityError(capability="chat_completion")

    def _parse_response(self, payload: JsonObject, *, latency_seconds: float) -> ModelResponse:
        """No completion answer exists; see :meth:`complete`."""
        del payload, latency_seconds
        raise ProviderCapabilityError(capability="chat_completion")
