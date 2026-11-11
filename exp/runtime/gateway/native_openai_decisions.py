"""Admission for OpenAI-shaped ``POST /v1/decisions`` on the native data plane.

The OpenAI Decisions API is a second wire of the decisions surface: the same
authority, ledger surface (``decisions``), direct-route resolution, explicit
input-only pricing rule and one-dispatch-per-deployment policy as the TypeSafe
SystemOne wire in :mod:`exp.runtime.gateway.native_decisions`. Only the body
shape and the qualifying rungs differ: a rung serves this surface when the
catalog positively declares decisions support and the connection exposes
either OpenAI's own ``/decisions`` endpoint (direct OpenAI, the body forwards
verbatim) or Google's Decisions API endpoint (a Gemini connection, the body is
translated by :mod:`exp.runtime.gateway.google_decisions_contracts` and the
answer back by the data plane). One request dispatches on one upstream wire:
the first capable rung in route order picks it and rungs of the other wire are
left out of the ladder.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Literal, Protocol

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.client_apps import with_client_identity
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayFailure,
    GatewayFailureClass,
)
from exp.runtime.gateway.google_decisions_contracts import (
    GoogleDecisionTranslationError,
    google_decision_body,
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
from exp.runtime.gateway.native_admission import (
    record_dead_admission_rungs,
)
from exp.runtime.gateway.native_components import NativeGatewayComponents, SyncWriteLedger
from exp.runtime.gateway.native_dead_rungs import UNAVAILABLE_AT_ADMISSION
from exp.runtime.gateway.native_decisions import (
    NativeDecisionsMixin,
    _decisions_rung,
    _priced_decision_rung,
    no_image_decisions_error,
)
from exp.runtime.gateway.native_execution import (
    MAXIMUM_TOTAL_ATTEMPTS,
    InflightRequest,
    NativeDialectUnavailableError,
    deployment_wire_entry,
    dispatchable_route_profiles,
    select_route_deployments,
)
from exp.runtime.gateway.native_settlement import gateway_updating_failure, optional_text
from exp.runtime.gateway.openai_decisions_contracts import (
    OpenAIDecisionRequest,
    decode_openai_decision_request,
)
from exp.runtime.gateway.routing import GatewayRoutingError
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, invalid_field


class _OpenAIDecisionsPlane(Protocol):
    """Control-plane authority, accounting, and direct-route seams."""

    _components: NativeGatewayComponents
    _accounting: NativeAttemptAccounting
    _write_ledger: SyncWriteLedger
    _request_timeout_seconds: float
    _guardrails: GuardrailEngine | None

    def _escalate_accepted(self, authorization: AuthorizationSnapshot, reason: str) -> str:
        """Finish an accepted request that cannot use the native data plane."""
        ...


def not_an_openai_decision_model_error(
    alias: str, *, serves_systemone: bool = False
) -> OpenAIProtocolError:
    """Return a field-specific refusal for an alias with no OpenAI decisions rung.

    Args:
        alias: The model id the caller requested.
        serves_systemone: Whether a dispatchable rung of the alias serves native
            decisions on ``/v1/systemone``, judged by the SystemOne lane's own
            rung predicate. The refusal then points the caller there with the
            same model id instead of telling them to choose another model.

    Returns:
        The typed 400 the boundary raises before any provider dispatch.
    """
    if serves_systemone:
        message = (
            f"The model {alias!r} supports native decisions, not the OpenAI Decisions API. "
            "POST /v1/systemone on this same API host with your same API key "
            f"(Authorization: Bearer) and the same model id {alias!r}, using model, state "
            "and questions. Key questions by id; each needs type (noul, choice or score) "
            "and instructions (choice also needs a criteria object; score a criteria "
            "array). Read answers[id]; do not send input or a questions list."
        )
    else:
        message = (
            f"The model {alias!r} does not serve the OpenAI Decisions API. Choose a "
            "decision model served by OpenAI from GET /v1/models and resend the request "
            "to /v1/decisions."
        )
    return OpenAIProtocolError(
        status_code=400,
        code="unsupported_capability",
        message=message,
        param="model",
    )


UpstreamDecisionWire = Literal["openai", "google"]
"""The provider wire an OpenAI-shaped decision dispatches on."""


def _upstream_wire(
    profile: GatewayWireProfile, supports_decisions: bool
) -> UpstreamDecisionWire | None:
    """The upstream decisions wire a rung serves this surface on, if any.

    Positive decisions evidence and an unsigned endpoint are required on both:
    OpenAI's own ``/decisions`` (direct OpenAI) or Google's Decisions API
    (a Gemini connection).
    """
    if supports_decisions is not True or profile.signs_request_body:
        return None
    if profile.openai_decisions_url is not None:
        return "openai"
    if profile.google_decisions_url is not None:
        return "google"
    return None


def _openai_decisions_rung(profile: GatewayWireProfile, supports_decisions: bool) -> bool:
    """Whether a rung serves the OpenAI-shaped decisions surface on any upstream wire."""
    return _upstream_wire(profile, supports_decisions) is not None


def _upstream_url(profile: GatewayWireProfile, wire: UpstreamDecisionWire) -> str | None:
    """The rung's endpoint on the chosen upstream wire."""
    return profile.openai_decisions_url if wire == "openai" else profile.google_decisions_url


class NativeOpenAIDecisionsMixin:
    """The ``admit_openai_decisions`` boundary of the native control plane."""

    def admit_openai_decisions(self: _OpenAIDecisionsPlane, argument: str) -> str:
        """Authenticate, decode, authorize, accept, and route one OpenAI decision.

        Args:
            argument: JSON object with raw_key and body text. Caller idempotency
                keys are ignored; no keyed replay contract is claimed.

        Returns:
            Frozen route entries, the admitted question list, the upstream wire
            marker (``openai`` or ``google``), and one-attempt-per-deployment
            limits, or a terminal content-free escalation.

        Raises:
            NativeBridgeError: Authentication, request validation, authorization,
                routing, or known-price admission failed before dispatch.
        """
        assert_not_internal_classification()
        data = json.loads(argument)
        try:
            self._components.store.authenticate_key(raw_key=str(data["raw_key"]))
        except Exception as exc:  # noqa: BLE001 - sanitize the authority boundary.
            raise authority_error(exc) from exc
        try:
            decoded = decode_openai_decision_request(str(data["body"]))
        except (ValueError, TypeError, RecursionError) as exc:
            # Validation text may quote input, instructions, or names; never copy
            # it into a public error or a ledger row.
            raise NativeBridgeError(
                invalid_field(
                    "body",
                    "Invalid decision request. Send model, input (text or user messages with "
                    "input_text and inline base64 input_image parts), questions of type "
                    "predicate, choice (with choices), or score (with levels), each with "
                    "instructions and an optional unique name, and optionally "
                    "safety_identifier. Streaming and chat controls are not supported.",
                )
            ) from exc
        deadline = time.monotonic() + self._request_timeout_seconds
        try:
            authorization = self._components.store.authorize_request(
                raw_key=str(data["raw_key"]),
                alias=decoded.alias,
                request=decoded.request,
                deadline_monotonic=deadline,
                app_referer=optional_text(data.get("app_referer")),
                app_title=optional_text(data.get("app_title")),
                client_ip=optional_text(data.get("client_ip")),
            )
            authorization = authorize_serving_model_chains(self._components, authorization)
            authorization = with_client_identity(authorization, data)  # reporting-only app facts
            self._accounting.require_clear(authorization.organization_id)
        except Exception as exc:  # noqa: BLE001 - sanitize the authority boundary.
            raise authority_error(exc) from exc
        require_unguarded_surface(self._guardrails, authorization, "decisions")
        try:
            self._write_ledger.accept_request(authorization=authorization)
        except Exception as exc:  # noqa: BLE001 - sanitize the authority boundary.
            raise authority_error(exc) from exc
        try:
            return _admit_accepted(self, authorization, decoded.request, deadline)
        except NativeBridgeError:
            raise
        except Exception as exc:  # noqa: BLE001 - close every accepted failure.
            self._accounting.finish_request_quietly(
                authorization,
                GatewayFailure(
                    failure_class=GatewayFailureClass.INTERNAL,
                    safe_message="gateway admission failed before provider dispatch",
                ),
            )
            raise authority_error(exc) from exc


def _admit_accepted(
    plane: _OpenAIDecisionsPlane,
    authorization: AuthorizationSnapshot,
    request: OpenAIDecisionRequest,
    deadline: float,
) -> str:
    """Package only certified, available, explicitly priced OpenAI decision rungs."""
    if not isinstance(authorization.target, DirectTarget):
        _finish_unsupported(plane, authorization)
        raise NativeBridgeError(not_an_openai_decision_model_error(authorization.alias))
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
    wires = {
        index: wire
        for index, (profile, _client) in zip(
            dispatchable.indexes, dispatchable.resolved_wires, strict=True
        )
        if (
            wire := _upstream_wire(
                profile, route.deployments[index].gateway.capabilities.supports_decisions
            )
        )
        is not None
    }
    capable = tuple(wires)
    if capable and request.has_images:
        # Google's image block (and OpenAI's input_image) reach only a rung
        # that declares image input; a declining rung's 400 is terminal, so
        # an image-capable fallback must be chosen here, never after.
        capable = tuple(
            index
            for index in capable
            if route.deployments[index].gateway.capabilities.supports_image_input
        )
        if not capable:
            _finish_unsupported(plane, authorization, no_images=True)
            raise NativeBridgeError(no_image_decisions_error(authorization.alias, "input"))
    if not capable:
        # The SystemOne lane's own rung predicate decides whether the same model
        # id is servable on /v1/systemone; there is no second rule here.
        serves_systemone = any(
            _decisions_rung(
                profile, route.deployments[index].gateway.capabilities.supports_decisions
            )
            for index, (profile, _client) in zip(
                dispatchable.indexes, dispatchable.resolved_wires, strict=True
            )
        )
        _finish_unsupported(plane, authorization, serves_systemone=serves_systemone)
        raise NativeBridgeError(
            not_an_openai_decision_model_error(
                authorization.alias, serves_systemone=serves_systemone
            )
        )
    priced = tuple(
        index for index in capable if _priced_decision_rung(route.deployments[index].gateway.prices)
    )
    # One upstream wire per request: the first PRICED capable rung in route
    # order picks it (an unpriced rung never suppresses a servable one), so
    # every answer in the ladder is validated the same way.
    upstream: UpstreamDecisionWire = wires[priced[0]] if priced else "openai"
    serving = tuple(index for index in priced if wires[index] == upstream)[:MAXIMUM_TOTAL_ATTEMPTS]
    if not serving:
        plane._accounting.finish_request_quietly(  # noqa: SLF001
            authorization, gateway_updating_failure()
        )
        raise NativeBridgeError(
            OpenAIProtocolError(
                status_code=503,
                code="model_unavailable",
                message="This decision model has no supported price configuration. "
                "Choose another model or ask the operator to configure its decision token rates.",
                param="model",
                error_type="api_error",
            )
        )
    served_wires = [
        wire
        for index, wire in zip(dispatchable.indexes, dispatchable.resolved_wires, strict=True)
        if index in serving
    ]
    route = select_route_deployments(route, serving)
    wire_route: list[JsonObject] = []
    ignored_parameters: tuple[str, ...] = ()
    for deployment, (profile, _client) in zip(route.deployments, served_wires, strict=True):
        decisions_url = _upstream_url(profile, upstream)
        if decisions_url is None:  # pragma: no cover - filtered above.
            raise GatewayRoutingError("OpenAI decision rung lost its wire endpoint")
        if upstream == "google":
            try:
                upstream_payload, ignored_parameters = google_decision_body(
                    request, profile.model_id
                )
            except GoogleDecisionTranslationError as exc:
                _finish_unsupported(plane, authorization, untranslatable=True)
                raise NativeBridgeError(invalid_field(exc.param, exc.message)) from exc
        else:
            upstream_payload = request.provider_body(profile.model_id)
        wire_route.append(
            deployment_wire_entry(
                route, deployment, replace(profile, url=decisions_url), upstream_payload
            )
        )
    depth = len(route.deployments)
    response: JsonObject = {
        "request_id": authorization.request_id,
        "caller_scope": f"{authorization.organization_id}:{authorization.identity_id}",
        "alias": authorization.alias,
        "alias_revision_id": authorization.alias_revision_id,
        "exact_model_id": route.snapshot.exact_model_id,
        "route_reason": route.route_reason,
        "route": wire_route,
        "wire": upstream,
        "openai_questions": request.question_definitions(),
        "maximum_total_attempts": depth,
        "maximum_same_deployment_attempts": 1,
        "ignored_parameters": list(ignored_parameters),
    }
    serialized = json.dumps(response, separators=(",", ":"))
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
            throttle_redial_budgets=(0,) * depth,
        )
    )
    return serialized


def _finish_unsupported(
    plane: _OpenAIDecisionsPlane,
    authorization: AuthorizationSnapshot,
    *,
    serves_systemone: bool = False,
    untranslatable: bool = False,
    no_images: bool = False,
) -> None:
    """Finish an accepted request naming an alias that cannot serve OpenAI decisions."""
    if no_images:
        safe_message = "no decision deployment accepts images"
    elif untranslatable:
        safe_message = "the request cannot be expressed on the model's Google Decisions API wire"
    elif serves_systemone:
        safe_message = (
            "the model alias serves native decisions on /v1/systemone, not the OpenAI Decisions API"
        )
    else:
        safe_message = "the model alias does not serve the OpenAI Decisions API"
    plane._accounting.finish_request_quietly(  # noqa: SLF001
        authorization,
        GatewayFailure(
            failure_class=GatewayFailureClass.UNSUPPORTED_CAPABILITY,
            safe_message=safe_message,
        ),
    )


class NativeDecisionSurfacesMixin(NativeDecisionsMixin, NativeOpenAIDecisionsMixin):
    """Both decisions wires' admission boundaries: SystemOne and OpenAI."""
