"""Authenticated, direct-route admission for native TypeSafe decisions.

Decisions are not chat: they bypass prompt-based selection and chat guardrails,
carry no continuation or replay identity, and permit at most one dispatch per
certified deployment. Only provider-reported usage settles a paid attempt.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Protocol

from exp.common.core.artifacts import JsonObject
from exp.common.models.catalog import GatewayTokenPrices
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayFailure,
    GatewayFailureClass,
)
from exp.runtime.gateway.decisions_contracts import DecisionRequest, decode_decision_request
from exp.runtime.gateway.guardrails.client import assert_not_internal_classification
from exp.runtime.gateway.model_chain_authority import authorize_serving_model_chains
from exp.runtime.gateway.native_accounting import (
    NativeAttemptAccounting,
    NativeBridgeError,
    authority_error,
)
from exp.runtime.gateway.native_admission import record_dead_admission_rungs
from exp.runtime.gateway.native_components import NativeGatewayComponents, SyncWriteLedger
from exp.runtime.gateway.native_execution import (
    MAXIMUM_TOTAL_ATTEMPTS,
    InflightRequest,
    NativeDialectUnavailableError,
    deployment_wire_entry,
    dispatchable_route_profiles,
    select_route_deployments,
)
from exp.runtime.gateway.native_settlement import gateway_updating_failure, optional_text
from exp.runtime.gateway.routing import GatewayRoutingError
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, invalid_field


class _DecisionsPlane(Protocol):
    """Control-plane authority, accounting, and direct-route seams."""

    _components: NativeGatewayComponents
    _accounting: NativeAttemptAccounting
    _write_ledger: SyncWriteLedger
    _request_timeout_seconds: float

    def _escalate_accepted(self, authorization: AuthorizationSnapshot, reason: str) -> str:
        """Finish an accepted request that cannot use the native data plane."""
        ...


def not_a_decision_model_error(alias: str) -> OpenAIProtocolError:
    """Return a field-specific refusal for an alias with no native decision rung."""
    return OpenAIProtocolError(
        status_code=400,
        code="unsupported_capability",
        message=(
            f"The model {alias!r} does not serve native decisions. Choose a decision model "
            "from GET /v1/models and resend the request to /v1/systemone."
        ),
        param="model",
    )


def _decisions_rung(profile: GatewayWireProfile, supports_decisions: bool) -> bool:
    """Require positive deployment evidence and the exact unsigned SystemOne wire."""
    return (
        supports_decisions is True
        and profile.dialect == "typesafe_systemone"
        and profile.decisions_url is not None
        and not profile.signs_request_body
    )


def _priced_decision_rung(prices: GatewayTokenPrices) -> bool:
    """Require a known nonnegative input rate and explicitly free output tokens."""
    return (
        prices.input_nano_usd_per_million_tokens is not None
        and prices.input_nano_usd_per_million_tokens >= 0
        and prices.output_nano_usd_per_million_tokens == 0
    )


class NativeDecisionsMixin:
    """The ``admit_decisions`` boundary of the native control plane."""

    def admit_decisions(self: _DecisionsPlane, argument: str) -> str:
        """Authenticate, decode, authorize, accept, and route one decision request.

        Args:
            argument: JSON object with raw_key and body text. Caller idempotency
                keys are ignored; no keyed replay contract is claimed.

        Returns:
            Frozen route entries, original question definitions, and one-attempt
            per-deployment limits, or a terminal content-free escalation.

        Raises:
            NativeBridgeError: Authentication, request validation, authorization,
                routing, or known-price admission failed before dispatch.
        """
        assert_not_internal_classification()
        self._accounting.sweep_expired()
        data = json.loads(argument)
        try:
            self._components.store.authenticate_key(raw_key=str(data["raw_key"]))
        except Exception as exc:  # noqa: BLE001 - sanitize the authority boundary.
            raise authority_error(exc) from exc
        try:
            decoded = decode_decision_request(str(data["body"]))
        except (ValueError, TypeError, RecursionError) as exc:
            # Pydantic and JSON errors may include state, instructions, or key
            # names. Never copy their text into a public error or ledger row.
            raise NativeBridgeError(
                invalid_field(
                    "body",
                    "Invalid decision request. Send only model, state, and bounded typed "
                    "questions; chat, streaming, tools, and generation controls are not supported.",
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
    plane: _DecisionsPlane,
    authorization: AuthorizationSnapshot,
    request: DecisionRequest,
    deadline: float,
) -> str:
    """Package only certified, available, explicitly priced decision deployments."""
    if not isinstance(authorization.target, DirectTarget):
        _finish_unsupported(plane, authorization)
        raise NativeBridgeError(not_a_decision_model_error(authorization.alias))
    try:
        route = plane._components.routes.resolve_direct(authorization)  # noqa: SLF001
        dispatchable = dispatchable_route_profiles(
            plane._components.runtime_catalogs,
            route,  # noqa: SLF001
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
        if _decisions_rung(
            profile, route.deployments[index].gateway.capabilities.supports_decisions
        )
    )
    if not capable:
        _finish_unsupported(plane, authorization)
        raise NativeBridgeError(not_a_decision_model_error(authorization.alias))
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
        decisions_url = profile.decisions_url
        if decisions_url is None:  # pragma: no cover - filtered above.
            raise GatewayRoutingError("decision rung lost its wire endpoint")
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
        "questions": {
            name: question.model_dump(mode="json", exclude_none=True)
            for name, question in request.questions.items()
        },
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
            signers=(None,) * depth,
            dispatch_bindings=(None,) * depth,
            reasoning_carrier_authorities=(None,) * depth,
            throttle_redial_budgets=(0,) * depth,
        )
    )
    return serialized


def _finish_unsupported(plane: _DecisionsPlane, authorization: AuthorizationSnapshot) -> None:
    """Finish an accepted request naming an alias that cannot serve decisions."""
    plane._accounting.finish_request_quietly(  # noqa: SLF001
        authorization,
        GatewayFailure(
            failure_class=GatewayFailureClass.UNSUPPORTED_CAPABILITY,
            safe_message="the model alias does not serve native decisions",
        ),
    )
