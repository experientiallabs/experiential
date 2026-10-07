"""Speech and transcription admission for the native data plane.

Both audio surfaces reuse the chat surface's authority, ledger, route
resolution, deployment-health, and reservation seams unchanged (the shape of
:mod:`exp.runtime.gateway.native_images`) and differ only in what they admit
and how each rung bills:

- Speech (``/audio/speech``): text in, one audio file out. A lane with a
  per-character unit card bills the input characters the gateway counted; a
  token-priced lane is asked for the provider's SSE format, whose final event
  reports the token usage, and the data plane reassembles the audio.
- Transcription (``/audio/transcriptions``): one audio upload in, a transcript
  out. The data plane parses the upload and keeps the audio bytes; admission
  sees only the upload's facts. A lane with a per-second unit card bills the
  provider's metered seconds; a token-priced lane bills its token usage.

A rung serves an audio surface only on a positive capability claim and an
OpenAI-wire endpoint for it (fail-closed). Keyed replay, streaming, voice
cloning, and non-OpenAI wires are not admitted.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

from exp.common.core.artifacts import JsonObject
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.attempt_tokens import worst_case_input_tokens, worst_case_output_tokens
from exp.runtime.gateway.audio_billing import AudioSurface, BillingMode, billing_mode
from exp.runtime.gateway.audio_contracts import SpeechRequest, TranscriptionRequest
from exp.runtime.gateway.client_apps import with_client_identity
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayFailure,
    GatewayFailureClass,
)
from exp.runtime.gateway.guardrails.client import assert_not_internal_classification
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.native import require_unguarded_surface
from exp.runtime.gateway.model_chain_authority import authorize_serving_model_chains
from exp.runtime.gateway.native_accounting import (
    NativeAttemptAccounting,
    NativeBridgeError,
    authority_error,
)
from exp.runtime.gateway.native_admission import record_dead_admission_rungs
from exp.runtime.gateway.native_components import NativeGatewayComponents, SyncWriteLedger
from exp.runtime.gateway.native_decode import (
    NativeDecodeError,
    decode_native_speech_body,
    decode_native_transcription_upload,
)
from exp.runtime.gateway.native_execution import (
    MAXIMUM_SAME_DEPLOYMENT_ATTEMPTS,
    MAXIMUM_TOTAL_ATTEMPTS,
    InflightRequest,
    NativeDialectUnavailableError,
    deployment_wire_entry,
    dispatchable_route_profiles,
    select_route_deployments,
)
from exp.runtime.gateway.native_settlement import gateway_updating_failure
from exp.runtime.gateway.routing import GatewayRoutingError
from exp.runtime.models.providers import require_gateway_provider
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.openai_audio_payloads import (
    openai_speech_request,
    openai_transcription_fields,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError

_SURFACE_NOUNS: dict[AudioSurface, tuple[str, str]] = {
    "speech": ("synthesize speech", "/v1/audio/speech"),
    "transcription": ("transcribe audio", "/v1/audio/transcriptions"),
}


class _AudioPlane(Protocol):
    """The control-plane state the audio admissions read and write."""

    _components: NativeGatewayComponents
    _accounting: NativeAttemptAccounting
    _write_ledger: SyncWriteLedger
    _request_timeout_seconds: float
    _guardrails: GuardrailEngine | None

    def _escalate_accepted(self, authorization: AuthorizationSnapshot, reason: str) -> str:
        """Hand an accepted request no rung can serve to the escalation path."""
        ...


def not_an_audio_model_error(alias: str, surface: AudioSurface) -> OpenAIProtocolError:
    """Build the public 400 for an alias none of whose rungs serve this audio surface.

    Args:
        alias: The public model alias the caller named.
        surface: The audio surface the caller called.

    Returns:
        A field-specific ``unsupported_capability`` error on ``model``.
    """
    verb, path = _SURFACE_NOUNS[surface]
    return OpenAIProtocolError(
        status_code=400,
        code="unsupported_capability",
        message=(
            f"The model {alias!r} does not {verb}. Resend the request to {path} "
            "naming a model alias your gateway operator has configured for it."
        ),
        param="model",
    )


def audio_deadline_error() -> OpenAIProtocolError:
    """Build the public 408 for an audio request whose budget ran out before acceptance."""
    return OpenAIProtocolError(
        status_code=408,
        code="request_timeout",
        message="The audio request used its whole deadline before it could be accepted.",
    )


def _speech_rung(profile: GatewayWireProfile, deployment: ExactModelDeployment) -> str | None:
    """The rung's speech endpoint when it positively claims speech (fail-closed)."""
    return profile.speech_url if deployment.gateway.capabilities.supports_speech else None


def _claims(deployment: ExactModelDeployment, surface: AudioSurface) -> bool:
    """Whether the deployment's gateway capabilities claim the audio surface."""
    capabilities = deployment.gateway.capabilities
    if surface == "speech":
        return capabilities.supports_speech
    return capabilities.supports_transcription


def _transcription_rung(
    profile: GatewayWireProfile, deployment: ExactModelDeployment
) -> str | None:
    """The rung's transcription endpoint when it positively claims transcription."""
    claimed = deployment.gateway.capabilities.supports_transcription
    return profile.transcriptions_url if claimed else None


class NativeAudioMixin:
    """The ``admit_speech`` and ``admit_transcription`` boundary methods."""

    def admit_speech(self: _AudioPlane, argument: str) -> str:
        """Decode, authorize, route, and durably accept one speech request.

        Args:
            argument: JSON object with ``raw_key`` and ``body`` (raw request body text).

        Returns:
            JSON wire configuration carrying the ordered ``route``, each rung's
            ``billing_modes`` entry, ``input_characters`` (the billed count of a
            per-character rung), and the frozen retry-policy facts, or an
            ``{"escalate": reason}`` disposition.

        Raises:
            NativeBridgeError: Decoding, authorization, or routing failed, or no
                rung of the alias synthesizes speech.
        """
        data = _begin(self, argument)
        try:
            decoded = decode_native_speech_body(str(data["body"]))
        except NativeDecodeError as exc:
            raise NativeBridgeError(exc.error) from exc
        request = decoded.request
        authorization, deadline = _accept(self, data, decoded.alias, request, "speech")

        def payload(profile: GatewayWireProfile, deployment: ExactModelDeployment) -> JsonObject:
            """Build one rung's speech body, asking for SSE on a token-priced rung.

            Args:
                profile: The rung's resolved wire profile (its served model id).
                deployment: The rung's deployment, whose price card picks the mode.

            Returns:
                The OpenAI ``/audio/speech`` JSON body for this rung.
            """
            metered = billing_mode(deployment, "speech") == "tokens"
            return openai_speech_request(profile.model_id, request, metered_stream=metered)

        return _route(
            self,
            authorization,
            request,
            deadline,
            surface="speech",
            rung_url=_speech_rung,
            payload=payload,
            extra={"input_characters": request.input_characters},
        )

    def admit_transcription(self: _AudioPlane, argument: str) -> str:
        """Decode, authorize, route, and durably accept one transcription request.

        Args:
            argument: JSON object with ``raw_key`` and ``upload`` (the data plane's
                parsed upload: text fields plus the measured audio facts).

        Returns:
            JSON wire configuration carrying the ordered ``route`` (each entry's
            payload is the upload's text fields), ``billing_modes``, the
            caller's ``response_format``, and the frozen retry-policy facts, or
            an ``{"escalate": reason}`` disposition.

        Raises:
            NativeBridgeError: Decoding, authorization, or routing failed, or no
                rung of the alias transcribes audio.
        """
        data = _begin(self, argument)
        try:
            decoded = decode_native_transcription_upload(str(data["upload"]))
        except NativeDecodeError as exc:
            raise NativeBridgeError(exc.error) from exc
        request = decoded.request
        authorization, deadline = _accept(self, data, decoded.alias, request, "transcription")

        def payload(profile: GatewayWireProfile, deployment: ExactModelDeployment) -> JsonObject:
            """Build one rung's transcription text fields (the audio is attached natively).

            Args:
                profile: The rung's resolved wire profile (its served model id).
                deployment: Unused: the fields do not depend on the billing mode.

            Returns:
                The ordered multipart text fields for this rung.
            """
            del deployment
            return openai_transcription_fields(profile.model_id, request)

        return _route(
            self,
            authorization,
            request,
            deadline,
            surface="transcription",
            rung_url=_transcription_rung,
            payload=payload,
            extra={
                "response_format": request.response_format or "json",
                "maximum_audio_milli": request.maximum_audio_seconds * 1_000,
            },
        )


def _begin(plane: _AudioPlane, argument: str) -> JsonObject:
    """Run the shared admission preamble and parse the bridge argument."""
    assert_not_internal_classification()
    plane._accounting.sweep_expired()  # noqa: SLF001
    plane._accounting.request_settlements.require_clear()  # noqa: SLF001
    return json.loads(argument)


def _accept(
    plane: _AudioPlane,
    data: JsonObject,
    alias: str,
    request: SpeechRequest | TranscriptionRequest,
    surface: AudioSurface,
) -> tuple[AuthorizationSnapshot, float]:
    """Authorize and durably accept one decoded audio request.

    The deadline is the data plane's original one: the upload and its duration
    probe already spent part of the budget, which ``remaining_milli`` carries.
    A budget spent during authorization refuses before anything is accepted.
    """
    budget = plane._request_timeout_seconds  # noqa: SLF001
    remaining_milli = data.get("remaining_milli")
    if isinstance(remaining_milli, int) and not isinstance(remaining_milli, bool):
        budget = min(budget, max(remaining_milli, 0) / 1000)
    deadline = time.monotonic() + budget
    try:
        authorization = plane._components.store.authorize_request(  # noqa: SLF001
            raw_key=str(data["raw_key"]),
            alias=alias,
            request=request,
            deadline_monotonic=deadline,
        )
        authorization = authorize_serving_model_chains(
            plane._components,  # noqa: SLF001
            authorization,
        )
        authorization = with_client_identity(authorization, data)  # reporting-only app facts
    except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
        raise authority_error(exc) from exc
    require_unguarded_surface(plane._guardrails, authorization, surface)  # noqa: SLF001
    if time.monotonic() >= deadline:
        raise NativeBridgeError(audio_deadline_error())
    try:
        plane._write_ledger.accept_request(authorization=authorization)  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
        raise authority_error(exc) from exc
    return authorization, deadline


def _route(  # noqa: PLR0913 - one admission seam shared by both audio surfaces.
    plane: _AudioPlane,
    authorization: AuthorizationSnapshot,
    request: SpeechRequest | TranscriptionRequest,
    deadline: float,
    *,
    surface: AudioSurface,
    rung_url: Callable[[GatewayWireProfile, ExactModelDeployment], str | None],
    payload: Callable[[GatewayWireProfile, ExactModelDeployment], JsonObject],
    extra: JsonObject,
) -> str:
    """Route and package one durably accepted audio request."""
    if not isinstance(authorization.target, DirectTarget):
        _finish_unsupported(plane, authorization, surface)
        raise NativeBridgeError(not_an_audio_model_error(authorization.alias, surface))
    try:
        route = plane._components.routes.resolve_direct(authorization)  # noqa: SLF001
        dispatchable = dispatchable_route_profiles(
            plane._components.runtime_catalogs,  # noqa: SLF001
            route,
        )
    except NativeDialectUnavailableError as exc:
        return plane._escalate_accepted(authorization, str(exc))  # noqa: SLF001
    except GatewayRoutingError as exc:
        plane._accounting.finish_request_quietly(  # noqa: SLF001
            authorization, gateway_updating_failure()
        )
        raise authority_error(exc) from exc
    record_dead_admission_rungs(
        plane._accounting,  # noqa: SLF001
        authorization,
        dispatchable.dead,
        fallback_available=bool(dispatchable.indexes),
    )
    # Surface compatibility is judged across the whole route, dead rungs
    # included: an alias whose only audio rung is down is unavailable (escalate),
    # not unsupported (a 400 the caller would never retry).
    if not any(
        _claims(deployment, surface) and billing_mode(deployment, surface) is not None
        for deployment in route.deployments
    ):
        _finish_unsupported(plane, authorization, surface)
        raise NativeBridgeError(not_an_audio_model_error(authorization.alias, surface))
    if not dispatchable.indexes:
        return plane._escalate_accepted(  # noqa: SLF001
            authorization, "every certified deployment was unavailable at admission"
        )
    serving: list[tuple[int, GatewayWireProfile, str]] = []
    for index, (profile, _client) in zip(
        dispatchable.indexes, dispatchable.resolved_wires, strict=True
    ):
        deployment = route.deployments[index]
        url = rung_url(profile, deployment)
        if url is not None and billing_mode(deployment, surface) is not None:
            serving.append((index, profile, url))
    if not serving:
        return plane._escalate_accepted(  # noqa: SLF001
            authorization, "every audio-capable deployment was unavailable at admission"
        )
    route = select_route_deployments(route, tuple(index for index, _, _ in serving))
    wire_route: list[JsonObject] = []
    modes: list[BillingMode] = []
    ceilings: list[list[int] | None] = []
    input_ceiling = worst_case_input_tokens(request)
    try:
        for deployment, (_index, profile, url) in zip(route.deployments, serving, strict=True):
            require_gateway_provider(deployment.provider)
            wire_route.append(
                deployment_wire_entry(
                    route,
                    deployment,
                    replace(profile, url=url),
                    payload(profile, deployment),
                )
            )
            mode = billing_mode(deployment, surface)
            if mode is None:  # pragma: no cover - hybrid rungs are filtered above.
                raise GatewayRoutingError("audio rung lost its billing mode")
            modes.append(mode)
            # A token rung's settle must stay inside its hold: the data plane
            # refuses usage above the input and output the reservation covers.
            ceilings.append(
                [input_ceiling, worst_case_output_tokens(request, deployment)]
                if mode == "tokens"
                else None
            )
    except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
        error = authority_error(exc)
        plane._accounting.finish_request_quietly(  # noqa: SLF001
            authorization,
            GatewayFailure(
                failure_class=GatewayFailureClass.INTERNAL,
                safe_message="gateway admission failed before provider dispatch",
            ),
        )
        raise error from exc
    depth = len(route.deployments)
    plane._accounting.register(  # noqa: SLF001
        InflightRequest(
            authorization=authorization,
            route=route,
            request=request,
            deadline_monotonic=deadline,
            no_paid_prework=True,
            signers=(None,) * depth,
            dispatch_bindings=(None,) * depth,
            reasoning_carrier_authorities=(None,) * depth,
        )
    )
    response: JsonObject = {
        "request_id": authorization.request_id,
        "alias": authorization.alias,
        "alias_revision_id": authorization.alias_revision_id,
        "exact_model_id": route.snapshot.exact_model_id,
        "route_reason": route.route_reason,
        "route": wire_route,
        "billing_modes": list(modes),
        "token_ceilings": ceilings,
        **extra,
        "maximum_total_attempts": MAXIMUM_TOTAL_ATTEMPTS,
        "maximum_same_deployment_attempts": MAXIMUM_SAME_DEPLOYMENT_ATTEMPTS,
    }
    return json.dumps(response, separators=(",", ":"))


def _finish_unsupported(
    plane: _AudioPlane, authorization: AuthorizationSnapshot, surface: AudioSurface
) -> None:
    """Finish one accepted request that named an alias without this audio surface."""
    verb, _ = _SURFACE_NOUNS[surface]
    plane._accounting.finish_request_quietly(  # noqa: SLF001
        authorization,
        GatewayFailure(
            failure_class=GatewayFailureClass.UNSUPPORTED_CAPABILITY,
            safe_message=f"the model alias does not {verb}",
        ),
    )
