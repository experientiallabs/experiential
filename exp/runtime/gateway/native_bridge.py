"""Python control plane for the Rust gateway data plane.

The Rust engine (`exp_gateway_native`) owns the HTTP socket, protocol decode,
upstream streaming, and SSE encoding. It calls back into this module for the
authority and accounting work the Python engine performs on the same request
path: key authentication, request authorization, ledger acceptance, attempt
start with budget reservation, and terminal settlement. Every method takes and
returns one JSON string so the boundary stays narrow and typed on both sides.

Boundary errors raise :class:`NativeBridgeError`, whose ``public_error_json``
attribute carries the sanitized OpenAI-shaped error the data plane returns to
the caller, mirroring ``GatewayService`` error mapping.
"""

from __future__ import annotations

import json
import threading
import time

from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject, stable_id
from exp.runtime.gateway.budgets import BudgetReservationRejected, maximum_attempt_cost_micro_usd
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.gateway.discovery import (
    public_model_list,
    public_model_object,
    require_granted_authority,
)
from exp.runtime.gateway.lifecycle import LocalGatewayComponents
from exp.runtime.gateway.routing import GatewayRoute, GatewayRoutingError
from exp.runtime.gateway.sqlite.store import (
    AliasNotGrantedError,
    GatewayStoreError,
    InvalidVirtualKeyError,
)
from exp.runtime.gateway.usage import read_usage_report
from exp.runtime.models.providers import (
    preflight_gateway_request,
    require_gateway_provider,
)
from exp.runtime.models.providers.base import ProviderHttpClient
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.openai_protocol.errors import (
    OpenAIProtocolError,
    invalid_field,
    public_failure_error,
)

_REQUEST_TIMEOUT_SECONDS = 120.0

_TERMINAL_KINDS = {
    "completed": GatewayEventKind.COMPLETED,
    "incomplete": GatewayEventKind.INCOMPLETE,
    "failed": GatewayEventKind.FAILED,
}


class _NativeDialectUnavailableError(RuntimeError):
    """The resolved provider has no native dialect; python must serve it."""


class NativeBridgeError(Exception):
    """One sanitized boundary failure delivered to the Rust data plane."""

    def __init__(self, error: OpenAIProtocolError) -> None:
        """Retain the public error as the JSON payload the data plane returns.

        Args:
            error: Sanitized protocol error carrying its HTTP representation.
        """
        super().__init__(error.detail.message)
        self.public_error_json = json.dumps(
            {
                "status_code": error.status_code,
                "code": error.detail.code,
                "message": error.detail.message,
                "error_type": error.detail.type,
                "param": error.detail.param,
                "retry_after_seconds": error.retry_after_seconds,
            },
            separators=(",", ":"),
        )


def _authority_error(exception: Exception) -> NativeBridgeError:
    """Map authority failures exactly like ``GatewayService`` boundary mapping.

    Args:
        exception: Store, grant, or routing failure raised during admission.

    Returns:
        A boundary error carrying the matching public OpenAI error.
    """
    if isinstance(exception, OpenAIProtocolError):
        return NativeBridgeError(exception)
    if isinstance(exception, InvalidVirtualKeyError):
        return NativeBridgeError(
            OpenAIProtocolError(
                status_code=401,
                code="invalid_key",
                message=(
                    "The gateway key is invalid, expired, or revoked. Ask the gateway operator "
                    "to issue a new virtual key."
                ),
                error_type="authentication_error",
            )
        )
    if isinstance(exception, AliasNotGrantedError):
        return NativeBridgeError(
            OpenAIProtocolError(
                status_code=403,
                code="model_not_granted",
                message=(
                    "The requested model alias is not granted to this identity. "
                    "GET /v1/models lists the model aliases available to this key."
                ),
                error_type="permission_error",
                param="model",
            )
        )
    if isinstance(exception, GatewayRoutingError):
        return NativeBridgeError(
            OpenAIProtocolError(
                status_code=503,
                code="unavailable_route",
                message=(
                    "The authorized model route is unavailable. Retry after a short delay; "
                    "if this persists, ask the gateway operator to check the alias deployments."
                ),
                error_type="api_error",
            )
        )
    if isinstance(exception, GatewayStoreError):
        return NativeBridgeError(
            OpenAIProtocolError(
                status_code=400,
                code="invalid_request",
                message=(
                    "The gateway request is invalid. Verify the model alias and request "
                    "fields, then resend."
                ),
                error_type="invalid_request_error",
            )
        )
    return NativeBridgeError(
        OpenAIProtocolError(
            status_code=500,
            code="internal_error",
            message=(
                "The gateway request failed. Retry the request; if this persists, "
                "ask the gateway operator to inspect the server logs."
            ),
            error_type="api_error",
        )
    )


def _native_unsupported_error(reason: str) -> NativeBridgeError:
    """Signal that the python engine must serve this request.

    The Rust data plane recognizes the ``native_unsupported`` code and
    replays the original HTTP request against the embedded python engine,
    which performs its own full authorization and accounting; no ledger
    row exists yet when this is raised.

    Args:
        reason: Display-safe reason the native path cannot serve the request.

    Returns:
        The boundary error carrying the sentinel code.
    """
    return NativeBridgeError(
        OpenAIProtocolError(
            status_code=501,
            code="native_unsupported",
            message=reason,
            error_type="api_error",
        )
    )


def _budget_quota_error() -> NativeBridgeError:
    """Return the public quota error for an exhausted monthly allocation."""
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.QUOTA_EXCEEDED,
        safe_message="monthly gateway allocation is exhausted",
    )
    return NativeBridgeError(public_failure_error(failure))


class NativeControlPlane:
    """Authority and accounting callbacks for the Rust data plane.

    Methods are called from multiple Rust worker threads. SQLite access is safe
    through the per-thread connection cache in the gateway store and ledger;
    the in-flight request registry is guarded by one lock.
    """

    def __init__(
        self,
        components: LocalGatewayComponents,
        *,
        request_timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        """Bind loaded gateway components for serving.

        Args:
            components: Authority, ledger, routes, and runtime catalogs.
            request_timeout_seconds: Total per-request budget from admission.
        """
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self._components = components
        self._request_timeout_seconds = request_timeout_seconds
        self._inflight: dict[str, tuple[AuthorizationSnapshot, str]] = {}
        self._lock = threading.Lock()
        self._accounting_healthy = True

    @property
    def reconciled_expired_requests(self) -> int:
        """Return crashed requests reconciled at startup."""
        return self._components.reconciled_expired_requests

    @property
    def reconciled_unknown_attempts(self) -> int:
        """Return crashed attempts reconciled at startup."""
        return self._components.reconciled_unknown_attempts

    def authenticate(self, argument: str) -> str:
        """Authenticate one virtual key before the data plane decodes content.

        Args:
            argument: JSON object with ``raw_key``.

        Returns:
            An empty JSON object on success.

        Raises:
            NativeBridgeError: The key is invalid, expired, or revoked.
        """
        data = json.loads(argument)
        try:
            self._components.store.authenticate_key(raw_key=data["raw_key"])
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise _authority_error(exc) from exc
        return "{}"

    def admit(self, argument: str) -> str:
        """Authorize, accept, route, and durably start one provider attempt.

        Args:
            argument: JSON object with ``raw_key``, ``alias``, ``request``
                (canonical ``GatewayRequest`` shape), and ``stream``.

        Returns:
            JSON wire configuration for the single resolved deployment.

        Raises:
            NativeBridgeError: Authorization, routing, capability, or budget
                admission failed; the ledger is finalized before raising when
                the request was already accepted.
        """
        data = json.loads(argument)
        try:
            request = GatewayRequest.model_validate(data["request"])
        except ValidationError as exc:
            raise NativeBridgeError(invalid_field("body")) from exc
        deadline = time.monotonic() + self._request_timeout_seconds
        try:
            authorization = self._components.store.authorize_request(
                raw_key=data["raw_key"],
                alias=data["alias"],
                request=request,
                deadline_monotonic=deadline,
            )
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise _authority_error(exc) from exc

        if not isinstance(authorization.target, DirectTarget):
            raise _native_unsupported_error(
                "project-backed aliases use learned selection on the python engine"
            )
        try:
            route = self._components.routes.resolve_direct(authorization)
            deployment = route.deployment
            provider_request = request.model_copy(update={"stream": True, "include_usage": True})
            require_gateway_provider(deployment.provider)
            preflight_gateway_request(provider_request, deployment.gateway.capabilities)
            wire = self._wire_configuration(route)
        except _NativeDialectUnavailableError as exc:
            raise _native_unsupported_error(str(exc)) from exc
        except ProviderCapabilityError as exc:
            failure = GatewayFailure(
                failure_class=GatewayFailureClass.UNSUPPORTED_CAPABILITY,
                safe_message="the resolved deployment does not support the requested capability",
            )
            raise NativeBridgeError(public_failure_error(failure)) from exc
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise _authority_error(exc) from exc

        accepted = False
        try:
            self._components.ledger.accept_request(authorization=authorization)
            accepted = True
            attempt_id = self._components.ledger.start_attempt(
                snapshot=route.snapshot,
                deployment=deployment,
                attempt_ordinal=0,
                route_depth=0,
                maximum_cost_micro_usd=maximum_attempt_cost_micro_usd(request, deployment),
            )
            self._components.ledger.record_route_context(
                attempt_id=attempt_id,
                route_reason=route.route_reason,
                fallback_reason=route.fallback_reason,
            )
        except BudgetReservationRejected as exc:
            error = _budget_quota_error()
            self._finish_request_quietly(
                authorization,
                GatewayFailure(
                    failure_class=GatewayFailureClass.QUOTA_EXCEEDED,
                    safe_message="monthly gateway allocation is exhausted",
                ),
            )
            raise error from exc
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            error = _authority_error(exc)
            if accepted:
                self._finish_request_quietly(
                    authorization,
                    GatewayFailure(
                        failure_class=GatewayFailureClass.INTERNAL,
                        safe_message="gateway admission failed before provider dispatch",
                    ),
                )
            raise error from exc

        with self._lock:
            self._inflight[authorization.request_id] = (authorization, attempt_id)
        response: JsonObject = {
            "request_id": authorization.request_id,
            "attempt_id": attempt_id,
            "alias": authorization.alias,
            "alias_revision_id": authorization.alias_revision_id,
            "exact_model_id": route.snapshot.exact_model_id,
            "provider": route.deployment.provider,
            "deployment_id": route.deployment.deployment_id,
            "route_reason": route.route_reason,
            **wire,
        }
        return json.dumps(response, separators=(",", ":"))

    def settle(self, argument: str) -> str:
        """Durably settle one previously admitted attempt exactly once.

        Args:
            argument: JSON object with ``request_id``, ``attempt_id``,
                ``outcome``, optional ``usage``, ``tool_names``, ``failure``.

        Returns:
            An empty JSON object; repeated settlement is a no-op.

        Raises:
            NativeBridgeError: The durable terminal write failed; readiness is
                latched unhealthy first, mirroring executor accounting.
        """
        data = json.loads(argument)
        request_id = str(data["request_id"])
        with self._lock:
            context = self._inflight.get(request_id)
        if context is None:
            return "{}"
        _authorization, attempt_id = context
        usage = _usage_from_payload(data.get("usage"), data.get("tool_names") or [])
        failure_payload = data.get("failure")
        failure = None
        if failure_payload is not None:
            failure = GatewayFailure(
                failure_class=GatewayFailureClass(failure_payload["failure_class"]),
                safe_message=failure_payload["safe_message"],
            )
        kind = _TERMINAL_KINDS[data["outcome"]]
        terminal = GatewayEvent(
            kind=kind,
            sequence_number=0,
            usage=usage,
            failure=failure if kind == GatewayEventKind.FAILED else None,
        )
        try:
            self._components.ledger.finish_attempt(
                attempt_id=attempt_id,
                terminal_event=terminal,
                failure=failure,
                finalize_request=True,
            )
        except Exception as exc:  # noqa: BLE001 - latch any durable ledger failure.
            # The in-flight entry is kept so a retried settlement can still
            # reach the ledger; finish_attempt is idempotent on success.
            self._accounting_healthy = False
            raise _authority_error(exc) from exc
        with self._lock:
            self._inflight.pop(request_id, None)
        return "{}"

    def models(self, argument: str) -> str:
        """Return the granted model list body for one authenticated key."""
        data = json.loads(argument)
        try:
            authorities = self._components.store.granted_alias_authorities(raw_key=data["raw_key"])
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise _authority_error(exc) from exc
        return json.dumps(public_model_list(authorities), separators=(",", ":"))

    def model_detail(self, argument: str) -> str:
        """Return one granted model object or the shared no-oracle 404."""
        data = json.loads(argument)
        try:
            authorities = self._components.store.granted_alias_authorities(raw_key=data["raw_key"])
            authority = require_granted_authority(authorities, data["model_id"])
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise _authority_error(exc) from exc
        return json.dumps(public_model_object(authority), separators=(",", ":"))

    def usage_json(self, argument: str) -> str:
        """Return the content-free usage report body."""
        del argument
        report = read_usage_report(
            self._components.ledger,
            organization_id=self._components.organization_id,
        )
        return json.dumps(report.model_dump(mode="json"), separators=(",", ":"))

    def readiness(self, argument: str) -> str:
        """Return whether startup proof holds and accounting stays healthy."""
        del argument
        ready = bool(self._components.readiness) and self._accounting_healthy
        return "true" if ready else "false"

    def _wire_configuration(self, route: GatewayRoute) -> JsonObject:
        """Resolve one deployment's public wire profile for the data plane.

        Args:
            route: Resolved single-deployment route.

        Returns:
            Dialect, URL, authenticated headers, and timing hints.

        Raises:
            _NativeDialectUnavailableError: The provider has no native-dialect
                implementation; the python engine serves the request.
            GatewayRoutingError: The resolved client cannot stream or the
                authorized catalog is not loaded.
        """
        authorization = route.snapshot.authorization
        catalog = self._components.runtime_catalogs.get(
            (authorization.alias_revision_id, authorization.catalog_sha256)
        )
        if catalog is None:
            raise GatewayRoutingError("runtime catalog is not loaded for the authorized revision")
        deployment = route.deployment
        resolved = catalog.resolve(deployment.source_alias)
        client = resolved.client
        if getattr(client, "stream", None) is None:
            raise GatewayRoutingError("resolved gateway deployment has no streaming capability")
        if not isinstance(client, ProviderHttpClient):
            raise _NativeDialectUnavailableError(
                f"provider {deployment.provider!r} has no native wire profile"
            )
        try:
            profile = client.gateway_wire_profile()
        except ProviderCapabilityError as exc:
            if exc.capability != "native_data_plane":
                raise
            raise _NativeDialectUnavailableError(
                f"provider {deployment.provider!r} has no native dialect implementation"
            ) from exc
        return {
            "dialect": profile.dialect,
            "url": profile.url,
            "headers": dict(profile.headers),
            "model_id": profile.model_id or resolved.snapshot.model_id,
            "timeout_seconds": profile.timeout_seconds,
            "supports_temperature": profile.supports_temperature,
            "reasoning_effort": profile.reasoning_effort,
            "token_limit_key": profile.token_limit_key,
            "idempotency_key": _deployment_operation_key(route),
        }

    def _finish_request_quietly(
        self,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
    ) -> None:
        """Finalize accepted pre-dispatch work without masking the primary failure."""
        try:
            self._components.ledger.finish_request(
                authorization=authorization,
                failure=failure,
            )
        except Exception:  # noqa: BLE001 - primary admission failure stays authoritative.
            self._accounting_healthy = False


def _deployment_operation_key(route: GatewayRoute) -> str:
    """Derive the stable per-deployment idempotency key used by dispatch.

    Mirrors the executor's provider-operation identity so retried physical
    dispatches of the same deployment reuse one caller operation.

    Args:
        route: Resolved single-deployment route.

    Returns:
        Stable content-addressed operation identity.
    """
    authorization = route.snapshot.authorization
    return stable_id(
        "gateway-provider-operation",
        {
            "request_id": authorization.request_id,
            "catalog_sha256": authorization.catalog_sha256,
            "deployment_id": route.deployment.deployment_id,
            "connection_sha256": route.deployment.connection_sha256,
        },
    )


def _usage_from_payload(
    payload: JsonObject | None,
    tool_names: list[str],
) -> GatewayUsage | None:
    """Build normalized usage from settlement scalars.

    Args:
        payload: Optional token totals observed by the data plane.
        tool_names: Invoked tool names in first-use order.

    Returns:
        Normalized usage, or ``None`` when nothing was observed.
    """
    names = tuple(str(name) for name in tool_names)
    if payload is None or payload.get("input_tokens") is None:
        if not names:
            return None
        return GatewayUsage(tool_names=names)
    return GatewayUsage(
        input_tokens=_optional_count(payload.get("input_tokens")),
        output_tokens=_optional_count(payload.get("output_tokens")),
        cached_input_tokens=_optional_count(payload.get("cached_input_tokens")),
        reasoning_tokens=_optional_count(payload.get("reasoning_tokens")),
        tool_names=names,
    )


def _optional_count(value: object) -> int | None:
    """Return one non-negative settlement token count or ``None``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
