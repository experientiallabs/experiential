"""Image-generation and image-edit admission for the native data plane.

The images surface reuses the chat surface's authority, ledger, route
resolution, deployment-health, and reservation seams unchanged (the same
shape as :mod:`exp.runtime.gateway.native_embeddings`) and differs only in
what it admits: a prompt-in (plus input images for an edit), images-out
request that dispatches one buffered POST per attempt. Two image wires exist:

- the OpenAI Images API (``/images/generations``) on OpenAI-compatible
  connections; a rung whose lane carries an ``image`` unit card (dall-e, xAI
  Grok Imagine) reserves ``n`` images at the request's ``quality/size``
  variant and settles one unit per image returned, and every other rung bills
  the provider's reported prompt and image tokens. An Azure rung dispatches on
  the wire its model family speaks
  (:mod:`exp.runtime.models.providers.azure_images`): the OpenAI Image API, the
  MAI image API, or the BFL FLUX.2 provider API. The MAI and BFL wires report
  no usage, so their rungs are admitted only with a per-image card, and a
  request they cannot serve faithfully (``n > 1``, an unrenderable size, an
  OpenAI-only control) excludes them rather than being silently changed;
- Stability AI's Bedrock InvokeModel body (``bedrock_stability``): one image
  per call, a frozen body signed at dispatch like every Bedrock body, always
  billed per image. Generation models serve ``/v1/images/generations``; every
  model that takes an input image serves ``/v1/images/edits``.

Deliberately absent: OpenAI-wire image edits and variations, streaming
partial images, and keyed replay.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from typing import Protocol

from exp.common.core.artifacts import JsonObject, sha256_bytes
from exp.common.models.catalog_prices import BilledUnitKind, GatewayTokenPrices
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
from exp.runtime.gateway.images_contracts import ImagesRequest, image_variant, per_image_rate
from exp.runtime.gateway.model_chain_authority import authorize_serving_model_chains
from exp.runtime.gateway.native_accounting import (
    NativeAttemptAccounting,
    NativeBridgeError,
    authority_error,
)
from exp.runtime.gateway.native_admission import (
    record_dead_admission_rungs,
)
from exp.runtime.gateway.native_components import NativeGatewayComponents, SyncWriteLedger
from exp.runtime.gateway.native_dead_rungs import UNAVAILABLE_AT_ADMISSION
from exp.runtime.gateway.native_decode import (
    NativeDecodeError,
    decode_native_image_edits_upload,
    decode_native_images_body,
)
from exp.runtime.gateway.native_dispatch import frozen_dispatch
from exp.runtime.gateway.native_execution import (
    MAXIMUM_SAME_DEPLOYMENT_ATTEMPTS,
    MAXIMUM_TOTAL_ATTEMPTS,
    FrozenDispatchBinding,
    InflightRequest,
    NativeDialectUnavailableError,
    deployment_wire_entry,
    dispatchable_route_profiles,
    select_route_deployments,
)
from exp.runtime.gateway.native_settlement import gateway_updating_failure
from exp.runtime.gateway.routing import GatewayRoutingError
from exp.runtime.models.providers import require_gateway_provider
from exp.runtime.models.providers.azure_images import (
    AzureImagesUnsupportedError,
    azure_images_dispatch,
    azure_images_wire,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.bedrock_stability import (
    StabilityParameterError,
    stability_body,
    stability_model,
)
from exp.runtime.models.providers.openai_compatible import openai_images_request
from exp.runtime.models.providers.protocol import GatewayDispatchSigner, NativeWireClient
from exp.runtime.models.providers.xai import is_xai_images_url
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


class _ImagesPlane(Protocol):
    """The control-plane state the images admission reads and writes."""

    _components: NativeGatewayComponents
    _accounting: NativeAttemptAccounting
    _write_ledger: SyncWriteLedger
    _request_timeout_seconds: float
    _guardrails: GuardrailEngine | None

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


def not_an_edit_model_error(alias: str) -> OpenAIProtocolError:
    """Build the public 400 for an edit naming an alias no rung edits images on."""
    return OpenAIProtocolError(
        status_code=400,
        code="unsupported_capability",
        message=(
            f"The model {alias!r} does not edit images. Choose an image-editing model "
            "from GET /v1/models, or send a text prompt to /v1/images/generations."
        ),
        param="model",
    )


def unsupported_image_parameter_error(param: str, message: str) -> OpenAIProtocolError:
    """Build the public 400 for a control no remaining image rung can carry."""
    return OpenAIProtocolError(
        status_code=400,
        code="unsupported_parameter",
        message=f"{message} Remove {param} or choose another image model.",
        param=param,
    )


def unpriced_variant_error(variant: str) -> OpenAIProtocolError:
    """Build the public 400 for a quality and size no per-image rung prices."""
    return OpenAIProtocolError(
        status_code=400,
        code="unsupported_value",
        message=(
            f"This model has no price for quality/size {variant or 'default'!r}. "
            "Choose a size and quality listed for the model."
        ),
        param="size",
    )


def unsupported_request_error(param: str, message: str) -> OpenAIProtocolError:
    """Build the public 400 for a request no image rung can serve faithfully."""
    return OpenAIProtocolError(
        status_code=400, code="unsupported_value", message=message, param=param
    )


def unbillable_wire_error() -> OpenAIProtocolError:
    """Build the public 400 for an image rung whose wire reports no usage and has no card."""
    return OpenAIProtocolError(
        status_code=400,
        code="unsupported_capability",
        message=(
            "This model's provider reports no token usage, and its lane has no per-image "
            "price, so it cannot be billed. Ask the organization admin to set a per-image price "
            "on the lane, or choose another image model."
        ),
        param="model",
    )


def _azure_usage_free_wire(exact_model_id: str, deployment: str) -> bool:
    """Whether one Azure rung's wire answers without token usage (MAI, BFL FLUX.2)."""
    return azure_images_wire(exact_model_id, deployment) != "openai_images"


def _per_image(prices: GatewayTokenPrices) -> bool:
    """Whether one lane bills images per image (an ``image`` unit card)."""
    return prices.units is not None and prices.units.kind == BilledUnitKind.IMAGE


def _images_rung(profile_url: str | None, supports_image_generation: bool | None) -> bool:
    """Whether one resolved rung may serve an image request (fail-closed)."""
    return profile_url is not None and supports_image_generation is True


# OpenAI image controls the Stability wire has no field for. Each is refused
# by name on a Stability rung rather than silently dropped.
_STABILITY_UNSUPPORTED_CONTROLS = (
    "quality",
    "style",
    "background",
    "moderation",
    "output_compression",
)


@dataclass(frozen=True)
class _RungWire:
    """One serving rung's provider body and answer shape.

    Attributes:
        payload: The structured provider body.
        stability: Whether the rung speaks Stability's Bedrock InvokeModel wire.
    """

    payload: JsonObject
    stability: bool


def _rung_wire(profile: GatewayWireProfile, request: ImagesRequest) -> _RungWire:
    """Build one rung's provider body for the request.

    Args:
        profile: The rung's resolved wire profile (``images_url`` present).
        request: The canonical image request.

    Returns:
        The rung's body and wire shape.

    Raises:
        StabilityParameterError: The rung's wire cannot carry the request (an
            OpenAI-wire rung asked to edit, or a Stability control mismatch).
    """
    if profile.images_wire == "openai":
        if request.edit is not None:
            raise StabilityParameterError(
                "model", "Image edits are served by Bedrock Stability image models only."
            )
        images_url = profile.images_url or ""
        return _RungWire(
            payload=openai_images_request(
                profile.model_id, request, inline_images=is_xai_images_url(images_url)
            ),
            stability=False,
        )
    model = stability_model(profile.model_id)
    if model is None:
        raise StabilityParameterError("model", "This Bedrock model is not a Stability image model.")
    for control in _STABILITY_UNSUPPORTED_CONTROLS:
        if getattr(request, control) is not None:
            raise StabilityParameterError(control, f"Stability image models take no {control}.")
    if request.n != 1:
        raise StabilityParameterError("n", "Stability image models return one image per request.")
    edit = request.edit
    if edit is not None and request.size is not None:
        raise StabilityParameterError(
            "size", "A Stability edit keeps the input image's size; omit size."
        )
    return _RungWire(
        payload=stability_body(
            model,
            operation="generate" if edit is None else "edit",
            prompt=request.prompt,
            size=request.size,
            output_format=request.output_format,
            image=None if edit is None else edit.image,
            mask=None if edit is None else edit.mask,
            style_image=None if edit is None else edit.style_image,
            options=request.edit_options,
        ),
        stability=True,
    )


class NativeImagesMixin:
    """The ``admit_images`` and ``admit_image_edits`` boundary methods."""

    def admit_images(self: _ImagesPlane, argument: str) -> str:
        """Decode, authorize, route, and durably accept one image-generation request.

        Args:
            argument: JSON object with ``raw_key`` and ``body`` (raw request body
                text). Any caller ``Idempotency-Key`` is ignored by the data plane.

        Returns:
            JSON wire configuration carrying the ordered ``route`` (one entry per
            deployment), ``image_count`` (the requested ``n``), the per-rung
            ``unit_billing`` and ``wire_shapes``, and the frozen retry-policy
            facts, or an ``{"escalate": reason}`` disposition.

        Raises:
            NativeBridgeError: Decoding, authorization, or routing failed, or no
                rung of the alias generates images.
        """
        assert_not_internal_classification()
        data = json.loads(argument)
        try:
            decoded = decode_native_images_body(str(data["body"]))
        except NativeDecodeError as exc:
            raise NativeBridgeError(exc.error) from exc
        return _admit_decoded(self, data, decoded.alias, decoded.request)

    def admit_image_edits(self: _ImagesPlane, argument: str) -> str:
        """Decode, authorize, route, and durably accept one image-edit request.

        Args:
            argument: JSON object with ``raw_key`` and ``upload``: the data
                plane's parsed multipart body, ``{"fields": {...}, "files":
                {...}}`` with base64 image parts.

        Returns:
            The same wire configuration as :meth:`admit_images`.

        Raises:
            NativeBridgeError: Decoding, authorization, or routing failed, or no
                rung of the alias edits images.
        """
        assert_not_internal_classification()
        data = json.loads(argument)
        try:
            decoded = decode_native_image_edits_upload(json.dumps(data["upload"]))
        except NativeDecodeError as exc:
            raise NativeBridgeError(exc.error) from exc
        return _admit_decoded(self, data, decoded.alias, decoded.request)


def _admit_decoded(
    plane: _ImagesPlane, data: JsonObject, alias: str, request: ImagesRequest
) -> str:
    """Authorize and durably accept one decoded image request, then route it."""
    deadline = time.monotonic() + plane._request_timeout_seconds  # noqa: SLF001
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
        plane._accounting.require_clear(authorization.organization_id)  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
        raise authority_error(exc) from exc
    require_unguarded_surface(plane._guardrails, authorization, "images")  # noqa: SLF001
    try:
        plane._write_ledger.accept_request(authorization=authorization)  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
        raise authority_error(exc) from exc
    return _admit_accepted(plane, authorization, request, deadline)


def _no_serving_rung_error(
    alias: str,
    request: ImagesRequest,
    *,
    unpriced: bool,
    refused: StabilityParameterError | None,
) -> OpenAIProtocolError:
    """The public 400 when no rung of the alias can serve this image request.

    Args:
        alias: The public model alias.
        request: The canonical image request.
        unpriced: Whether a rung was excluded for having no rate to bill.
        refused: The first rung refusal, naming the parameter at fault.

    Returns:
        The protocol error to answer.
    """
    if refused is not None and refused.param != "model":
        return unsupported_image_parameter_error(refused.param, str(refused))
    if unpriced:
        return unpriced_variant_error(image_variant(request))
    if request.edit is not None:
        return not_an_edit_model_error(alias)
    if refused is not None:
        return OpenAIProtocolError(
            status_code=400,
            code="unsupported_capability",
            message=f"The model {alias!r} cannot serve this request. {refused}",
            param="model",
        )
    return not_an_image_model_error(alias)


def _admit_accepted(
    plane: _ImagesPlane,
    authorization: AuthorizationSnapshot,
    request: ImagesRequest,
    deadline: float,
) -> str:
    """Route and package one durably accepted image request."""
    if not isinstance(authorization.target, DirectTarget):
        _finish_unsupported(plane, authorization)
        raise NativeBridgeError(not_an_image_model_error(authorization.alias))
    try:
        route = plane._components.routes.resolve_direct(authorization)  # noqa: SLF001
        dispatchable = dispatchable_route_profiles(
            plane._components.runtime_catalogs,  # noqa: SLF001
            route,
            reread=lambda: plane._components.runtime_catalogs,  # noqa: SLF001
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
            authorization, UNAVAILABLE_AT_ADMISSION
        )
    serving: list[int] = []
    rung_wires: dict[int, tuple[GatewayWireProfile, NativeWireClient | None, _RungWire]] = {}
    unpriced = False
    unbillable = False
    unsupported: AzureImagesUnsupportedError | None = None
    refused: StabilityParameterError | None = None
    for index, (profile, client) in zip(
        dispatchable.indexes, dispatchable.resolved_wires, strict=True
    ):
        deployment = route.deployments[index]
        capabilities = deployment.capabilities
        supports = None if capabilities is None else capabilities.supports_image_generation
        if not _images_rung(profile.images_url, supports):
            continue
        try:
            wire = _rung_wire(profile, request)
        except StabilityParameterError as exc:
            # The first refusal names the parameter if no other rung serves.
            refused = refused or exc
            continue
        prices = deployment.gateway.prices
        if (wire.stability or _per_image(prices)) and (
            not _per_image(prices) or per_image_rate(prices, request) is None
        ):
            # A Stability rung bills per image only; a per-image card with no
            # rate for this quality/size could only settle unpriced. Either
            # way the rung is excluded (fail-closed).
            unpriced = True
            continue
        if deployment.provider == "azure" and profile.images_url is not None:
            if _azure_usage_free_wire(deployment.exact_model_id, profile.model_id) and not (
                _per_image(prices)
            ):
                # MAI and BFL answers carry no usage; without a per-image card
                # the rung could only settle unbilled (fail-closed).
                unbillable = True
                continue
            try:
                azure_images_dispatch(
                    images_url=profile.images_url,
                    headers=dict(profile.headers),
                    deployment=profile.model_id,
                    exact_model_id=deployment.exact_model_id,
                    request=request,
                    openai_body={},
                )
            except AzureImagesUnsupportedError as exc:
                unsupported = unsupported or exc
                continue
        serving.append(index)
        rung_wires[index] = (profile, client, wire)
    if not serving:
        _finish_unsupported(plane, authorization)
        if unsupported is not None:
            raise NativeBridgeError(
                unsupported_request_error(unsupported.param, unsupported.message)
            )
        if unbillable and not unpriced and refused is None:
            raise NativeBridgeError(unbillable_wire_error())
        raise NativeBridgeError(
            _no_serving_rung_error(authorization.alias, request, unpriced=unpriced, refused=refused)
        )
    route = select_route_deployments(route, tuple(serving))
    wire_route: list[JsonObject] = []
    signers: list[GatewayDispatchSigner | None] = []
    bindings: list[FrozenDispatchBinding | None] = []
    shapes: list[JsonObject] = []
    try:
        for deployment, index in zip(route.deployments, serving, strict=True):
            require_gateway_provider(deployment.provider)
            profile, client, wire = rung_wires[index]
            images_url = profile.images_url
            if images_url is None:  # pragma: no cover - filtered above.
                raise GatewayRoutingError("images rung lost its wire endpoint")
            if deployment.provider == "azure":
                dispatch = azure_images_dispatch(
                    images_url=images_url,
                    headers=dict(profile.headers),
                    deployment=profile.model_id,
                    exact_model_id=deployment.exact_model_id,
                    request=request,
                    openai_body=wire.payload,
                )
                dispatch_profile = replace(profile, url=dispatch.url, headers=dispatch.headers)
                payload = dispatch.body
            else:
                dispatch_profile = replace(profile, url=images_url)
                payload = wire.payload
            upstream_body, signer = frozen_dispatch(dispatch_profile, client, payload)
            wire_route.append(
                deployment_wire_entry(route, deployment, dispatch_profile, payload, upstream_body)
            )
            signers.append(signer)
            bindings.append(
                None
                if signer is None or upstream_body is None
                else FrozenDispatchBinding(
                    url=images_url, body_sha256=sha256_bytes(upstream_body.encode("utf-8"))
                )
            )
            shapes.append(
                {"shape": "stability", "output_format": request.output_format or "png"}
                if wire.stability
                else {"shape": "openai"}
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
            signers=tuple(signers),
            dispatch_bindings=tuple(bindings),
            reasoning_carrier_authorities=(None,) * depth,
        )
    )
    response: JsonObject = {
        "request_id": authorization.request_id,
        "caller_scope": f"{authorization.organization_id}:{authorization.identity_id}",
        "alias": authorization.alias,
        "alias_revision_id": authorization.alias_revision_id,
        "exact_model_id": route.snapshot.exact_model_id,
        "route_reason": route.route_reason,
        "route": wire_route,
        "image_count": request.n,
        # Aligned with ``route``: a per-image rung settles image units at this variant.
        "unit_billing": [
            {"variant": image_variant(request)} if _per_image(deployment.gateway.prices) else None
            for deployment in route.deployments
        ],
        # Aligned with ``route``: the answer shape each rung's provider returns.
        "wire_shapes": shapes,
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
