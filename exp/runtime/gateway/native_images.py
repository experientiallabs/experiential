"""Image-generation admission for the native data plane.

The images surface reuses the chat surface's authority, ledger, route
resolution, deployment-health, and reservation seams unchanged (the same
shape as :mod:`exp.runtime.gateway.native_embeddings`) and differs only in
what it admits: a prompt-in, images-out request that dispatches one buffered
OpenAI-wire ``/images/generations`` POST per attempt and bills the provider's
reported prompt and image tokens.

Deliberately absent on day one: image edits and variations (multipart image
input), streaming partial images, per-image priced models (dall-e answers
without token usage and is refused as unbilled until the typed billed-units
ledger lands), keyed replay, and non-OpenAI wires.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Protocol

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayFailure,
    GatewayFailureClass,
)
from exp.runtime.gateway.guardrails.client import assert_not_internal_classification
from exp.runtime.gateway.images_contracts import ImagesRequest
from exp.runtime.gateway.native_accounting import (
    NativeAttemptAccounting,
    NativeBridgeError,
    authority_error,
    gateway_updating_failure,
    record_dead_admission_rungs,
)
from exp.runtime.gateway.native_components import NativeGatewayComponents, SyncWriteLedger
from exp.runtime.gateway.native_decode import NativeDecodeError, decode_native_images_body
from exp.runtime.gateway.native_execution import (
    MAXIMUM_SAME_DEPLOYMENT_ATTEMPTS,
    MAXIMUM_TOTAL_ATTEMPTS,
    InflightRequest,
    NativeDialectUnavailableError,
    deployment_wire_entry,
    dispatchable_route_profiles,
    select_route_deployments,
)
from exp.runtime.gateway.routing import GatewayRoutingError
from exp.runtime.models.providers import require_gateway_provider
from exp.runtime.models.providers.openai_compatible import openai_images_request
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


class _ImagesPlane(Protocol):
    """The control-plane state the images admission reads and writes."""

    _components: NativeGatewayComponents
    _accounting: NativeAttemptAccounting
    _write_ledger: SyncWriteLedger
    _request_timeout_seconds: float

    def _escalate_accepted(self, authorization: AuthorizationSnapshot, reason: str) -> str: ...


def not_an_image_model_error(alias: str) -> OpenAIProtocolError:
    """Build the public 400 for an alias none of whose rungs generate images."""
    return OpenAIProtocolError(
        status_code=400,
        code="unsupported_capability",
        message=(
            f"The model {alias!r} does not generate images. Choose an image model from "
            "GET /v1/models and resend the request to /v1/images/generations."
        ),
        param="model",
    )


def _images_rung(profile_url: str | None, supports_image_generation: bool | None) -> bool:
    """Whether one resolved rung may serve an image-generation request (fail-closed)."""
    return profile_url is not None and supports_image_generation is True


class NativeImagesMixin:
    """The ``admit_images`` boundary method for the native control plane."""

    def admit_images(self: _ImagesPlane, argument: str) -> str:
        """Decode, authorize, route, and durably accept one image-generation request.

        Args:
            argument: JSON object with ``raw_key`` and ``body`` (raw request body
                text). Any caller ``Idempotency-Key`` is ignored by the data plane.

        Returns:
            JSON wire configuration carrying the ordered ``route`` (one OpenAI-wire
            entry per deployment), ``image_count`` (the requested ``n``), and the
            frozen retry-policy facts, or an ``{"escalate": reason}`` disposition.

        Raises:
            NativeBridgeError: Decoding, authorization, or routing failed, or no
                rung of the alias generates images.
        """
        assert_not_internal_classification()
        self._accounting.sweep_expired()
        data = json.loads(argument)
        try:
            decoded = decode_native_images_body(str(data["body"]))
        except NativeDecodeError as exc:
            raise NativeBridgeError(exc.error) from exc
        request = decoded.request
        deadline = time.monotonic() + self._request_timeout_seconds
        try:
            authorization = self._components.store.authorize_request(
                raw_key=str(data["raw_key"]),
                alias=decoded.alias,
                request=request,
                deadline_monotonic=deadline,
            )
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise authority_error(exc) from exc
        try:
            self._write_ledger.accept_request(authorization=authorization)
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise authority_error(exc) from exc
        return _admit_accepted(self, authorization, request, deadline)


def _admit_accepted(
    plane: _ImagesPlane,
    authorization: AuthorizationSnapshot,
    request: ImagesRequest,
    deadline: float,
) -> str:
    """Route and package one durably accepted image-generation request."""
    if not isinstance(authorization.target, DirectTarget):
        _finish_unsupported(plane, authorization)
        raise NativeBridgeError(not_an_image_model_error(authorization.alias))
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
    if not dispatchable.indexes:
        return plane._escalate_accepted(  # noqa: SLF001
            authorization, "every certified deployment was unavailable at admission"
        )
    serving: list[int] = []
    for index, (profile, _client) in zip(
        dispatchable.indexes, dispatchable.resolved_wires, strict=True
    ):
        capabilities = route.deployments[index].capabilities
        supports = None if capabilities is None else capabilities.supports_image_generation
        if _images_rung(profile.images_url, supports):
            serving.append(index)
    if not serving:
        _finish_unsupported(plane, authorization)
        raise NativeBridgeError(not_an_image_model_error(authorization.alias))
    served_wires = [
        wire
        for index, wire in zip(dispatchable.indexes, dispatchable.resolved_wires, strict=True)
        if index in serving
    ]
    route = select_route_deployments(route, tuple(serving))
    wire_route: list[JsonObject] = []
    try:
        for deployment, (profile, _client) in zip(route.deployments, served_wires, strict=True):
            require_gateway_provider(deployment.provider)
            images_url = profile.images_url
            if images_url is None:  # pragma: no cover - filtered above.
                raise GatewayRoutingError("images rung lost its wire endpoint")
            wire_route.append(
                deployment_wire_entry(
                    route,
                    deployment,
                    replace(profile, url=images_url),
                    openai_images_request(profile.model_id, request),
                )
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
        "image_count": request.n,
        "maximum_total_attempts": MAXIMUM_TOTAL_ATTEMPTS,
        "maximum_same_deployment_attempts": MAXIMUM_SAME_DEPLOYMENT_ATTEMPTS,
    }
    return json.dumps(response, separators=(",", ":"))


def _finish_unsupported(plane: _ImagesPlane, authorization: AuthorizationSnapshot) -> None:
    """Finish one accepted request that named a non-image alias."""
    plane._accounting.finish_request_quietly(  # noqa: SLF001
        authorization,
        GatewayFailure(
            failure_class=GatewayFailureClass.UNSUPPORTED_CAPABILITY,
            safe_message="the model alias does not generate images",
        ),
    )
