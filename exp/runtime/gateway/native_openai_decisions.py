"""Admission for OpenAI-shaped ``POST /v1/decisions`` on the native data plane.

The OpenAI Decisions API is a second wire of the decisions surface: the same
authority, ledger surface (``decisions``), direct-route resolution, explicit
input-only pricing rule and one-dispatch-per-deployment policy as the TypeSafe
SystemOne wire in :mod:`exp.runtime.gateway.native_decisions`. Only the body
shape and the qualifying rungs differ: a rung serves this wire when the catalog
positively declares decisions support and the connection exposes OpenAI's own
``/decisions`` endpoint (direct OpenAI only).
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Protocol

from exp.common.core.artifacts import JsonObject
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
from exp.runtime.gateway.native_decisions import NativeDecisionsMixin, _priced_decision_rung
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


def not_an_openai_decision_model_error(alias: str) -> OpenAIProtocolError:
    """Return a field-specific refusal for an alias with no OpenAI decisions rung."""
    return OpenAIProtocolError(
        status_code=400,
        code="unsupported_capability",
        message=(
            f"The model {alias!r} does not serve the OpenAI Decisions API. Choose a "
            "decision model served by OpenAI from GET /v1/models and resend the request "
            "to /v1/decisions."
        ),
        param="model",
    )


def _openai_decisions_rung(profile: GatewayWireProfile, supports_decisions: bool) -> bool:
    """Require positive decisions evidence and OpenAI's own unsigned endpoint."""
    return (
        supports_decisions is True
        and profile.openai_decisions_url is not None
        and not profile.signs_request_body
    )


class NativeOpenAIDecisionsMixin:
    """The ``admit_openai_decisions`` boundary of the native control plane."""

    def admit_openai_decisions(self: _OpenAIDecisionsPlane, argument: str) -> str:
        """Authenticate, decode, authorize, accept, and route one OpenAI decision.

        Args:
            argument: JSON object with raw_key and body text. Caller idempotency
                keys are ignored; no keyed replay contract is claimed.

        Returns:
            Frozen route entries, the admitted question list, the ``openai``
            wire marker, and one-attempt-per-deployment limits, or a terminal
            content-free escalation.

        Raises:
            NativeBridgeError: Authentication, request validation, authorization,
                routing, or known-price admission failed before dispatch.
        """
        assert_not_internal_classification()
        self._accounting.sweep_expired()
        self._accounting.request_settlements.require_clear()
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
    capable = tuple(
        index
        for index, (profile, _client) in zip(
            dispatchable.indexes, dispatchable.resolved_wires, strict=True
        )
        if _openai_decisions_rung(
            profile, route.deployments[index].gateway.capabilities.supports_decisions
        )
    )
    if not capable:
        _finish_unsupported(plane, authorization)
        raise NativeBridgeError(not_an_openai_decision_model_error(authorization.alias))
    serving = tuple(
        index for index in capable if _priced_decision_rung(route.deployments[index].gateway.prices)
    )[:MAXIMUM_TOTAL_ATTEMPTS]
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
    for deployment, (profile, _client) in zip(route.deployments, served_wires, strict=True):
        decisions_url = profile.openai_decisions_url
        if decisions_url is None:  # pragma: no cover - filtered above.
            raise GatewayRoutingError("OpenAI decision rung lost its wire endpoint")
        wire_route.append(
            deployment_wire_entry(
                route,
                deployment,
                replace(profile, url=decisions_url),
                request.provider_body(profile.model_id),
            )
        )
    depth = len(route.deployments)
    response: JsonObject = {
        "request_id": authorization.request_id,
        "alias": authorization.alias,
        "alias_revision_id": authorization.alias_revision_id,
        "exact_model_id": route.snapshot.exact_model_id,
        "route_reason": route.route_reason,
        "route": wire_route,
        "wire": "openai",
        "openai_questions": request.question_definitions(),
        "maximum_total_attempts": depth,
        "maximum_same_deployment_attempts": 1,
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


def _finish_unsupported(plane: _OpenAIDecisionsPlane, authorization: AuthorizationSnapshot) -> None:
    """Finish an accepted request naming an alias that cannot serve OpenAI decisions."""
    plane._accounting.finish_request_quietly(  # noqa: SLF001
        authorization,
        GatewayFailure(
            failure_class=GatewayFailureClass.UNSUPPORTED_CAPABILITY,
            safe_message="the model alias does not serve the OpenAI Decisions API",
        ),
    )


class NativeDecisionSurfacesMixin(NativeDecisionsMixin, NativeOpenAIDecisionsMixin):
    """Both decisions wires' admission boundaries: SystemOne and OpenAI."""
