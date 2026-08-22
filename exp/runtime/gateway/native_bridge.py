"""Python control plane for the native (Rust) gateway data plane.

The native engine (`exp_gateway_native`) owns the HTTP socket, upstream
streaming, provider-event normalization, and public SSE encoding. Everything
protocol- and authority-shaped happens here, in the same code the python
engine runs: request decoding through ``decode_chat``, authorization through
the shared control store, upstream payload construction through the shared
``streaming_requests`` builders, and the same durable SQLite ledger
transactions. Every boundary call takes and returns one JSON string so the
boundary stays narrow and typed on both sides.

Boundary errors raise :class:`NativeBridgeError`, whose ``public_error_json``
attribute carries the sanitized OpenAI-shaped error the data plane returns to
the caller, mirroring ``GatewayService`` error mapping. Requests the native
path cannot serve (project aliases, multi-deployment pools, providers without
a native dialect) are answered with an ``{"escalate": reason}`` admission
disposition before any ledger write; the data plane replays those against the
embedded python engine, which performs its own full authorization and
accounting, so nothing is double-counted.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace

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
    listing_metadata_by_alias,
    public_model_list,
    public_model_object,
    require_granted_authority,
)

# The executor's identity check is the authoritative pre-dispatch invariant;
# the native path must enforce the same one, so the private helper is shared.
from exp.runtime.gateway.execution import (
    GatewayExecutionError,
    _require_deployment_identity,  # noqa: PLC2701
)
from exp.runtime.gateway.lifecycle import LocalGatewayComponents
from exp.runtime.gateway.routing import GatewayRoute, GatewayRoutingError
from exp.runtime.gateway.service import boundary_protocol_error
from exp.runtime.gateway.usage import read_usage_report
from exp.runtime.models.providers import (
    preflight_gateway_request,
    require_gateway_provider,
)
from exp.runtime.models.providers.base import GatewayWireProfile, ProviderHttpClient
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.streaming_requests import (
    anthropic_messages_stream_payload,
    openai_compatible_stream_payload,
    openai_responses_stream_payload,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, public_failure_error
from exp.runtime.openai_protocol.requests import DecodedGatewayRequest, decode_chat

_REQUEST_TIMEOUT_SECONDS = 120.0
_SWEEP_GRACE_SECONDS = 5.0
_SWEEP_BATCH = 16

_TERMINAL_KINDS = {
    "completed": GatewayEventKind.COMPLETED,
    "incomplete": GatewayEventKind.INCOMPLETE,
    "failed": GatewayEventKind.FAILED,
}


class _NativeDialectUnavailableError(RuntimeError):
    """The resolved provider has no native dialect; python must serve it."""


class NativeBridgeError(Exception):
    """One sanitized boundary failure delivered to the native data plane."""

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
    """Map boundary failures through the shared service-layer mapper.

    Args:
        exception: Store, grant, routing, or execution failure.

    Returns:
        A boundary error carrying the matching public OpenAI error.
    """
    return NativeBridgeError(boundary_protocol_error(exception))


def _escalation(reason: str) -> str:
    """Return the admission disposition that hands this request to python.

    No ledger row exists when this is returned; the embedded python engine
    performs complete authorization and accounting on the replayed request.

    Args:
        reason: Display-safe reason the native path cannot serve the request.

    Returns:
        The JSON admission body carrying the escalation disposition.
    """
    return json.dumps({"escalate": reason}, separators=(",", ":"))


def _budget_quota_error() -> NativeBridgeError:
    """Return the public quota error for an exhausted monthly allocation."""
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.QUOTA_EXCEEDED,
        safe_message="monthly gateway allocation is exhausted",
    )
    return NativeBridgeError(public_failure_error(failure))


class NativeControlPlane:
    """Authority and accounting callbacks for the native data plane.

    Methods are called from multiple Rust worker threads. SQLite access is
    safe through the per-thread connection cache in the gateway store and
    ledger; the in-flight request registry is guarded by one lock and swept
    opportunistically so an abandoned reservation cannot outlive its request
    deadline by more than the sweep grace.
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
        self._inflight: dict[str, tuple[AuthorizationSnapshot, str, float]] = {}
        self._lock = threading.Lock()
        self._accounting_healthy = True

    @property
    def request_timeout_seconds(self) -> float:
        """Return the per-request budget shared with the data plane."""
        return self._request_timeout_seconds

    @property
    def reconciled_expired_requests(self) -> int:
        """Return crashed requests reconciled at startup."""
        return self._components.reconciled_expired_requests

    @property
    def reconciled_unknown_attempts(self) -> int:
        """Return crashed attempts reconciled at startup."""
        return self._components.reconciled_unknown_attempts

    def authenticate(self, argument: str) -> str:
        """Authenticate one virtual key before the data plane reads the body.

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
        """Decode, authorize, route, and durably start one provider attempt.

        The raw body is decoded with the same ``decode_chat`` the python
        engine uses, and the upstream payload is built with the same shared
        payload builders, so the two engines cannot drift at the protocol or
        provider boundary.

        Args:
            argument: JSON object with ``raw_key`` and ``body`` (raw request
                body text).

        Returns:
            JSON wire configuration for the single resolved deployment,
            including the fully built upstream payload, or an
            ``{"escalate": reason}`` disposition (returned only before any
            ledger write) handing the request to the python engine.

        Raises:
            NativeBridgeError: Decoding, authorization, routing, capability,
                or budget admission failed.
        """
        self._sweep_expired()
        data = json.loads(argument)
        decoded = self._decode_body(data["body"])
        request = decoded.request
        deadline = time.monotonic() + self._request_timeout_seconds
        try:
            authorization = self._components.store.authorize_request(
                raw_key=data["raw_key"],
                alias=decoded.alias,
                request=request,
                deadline_monotonic=deadline,
            )
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise _authority_error(exc) from exc

        # Escalation runs before any ledger write: the python engine performs
        # full accounting for every request it serves. Routing failures found
        # by the probe are recorded against the accepted request below, the
        # same order the python engine writes them.
        if not isinstance(authorization.target, DirectTarget):
            return _escalation("project-backed aliases use learned selection on the python engine")
        probe_failure: Exception | None = None
        route: GatewayRoute | None = None
        profile: GatewayWireProfile | None = None
        try:
            route = self._components.routes.resolve_direct(authorization)
            profile = self._wire_profile(route)
        except _NativeDialectUnavailableError as exc:
            return _escalation(str(exc))
        except Exception as exc:  # noqa: BLE001 - recorded after acceptance below.
            probe_failure = exc
        if route is not None and route.fallback_deployments:
            return _escalation("multi-deployment pools use the python engine's certified waterfall")

        provider_request = request.model_copy(update={"stream": True, "include_usage": True})
        accepted = False
        attempt_id: str | None = None
        try:
            self._components.ledger.accept_request(authorization=authorization)
            accepted = True
            if probe_failure is not None or route is None or profile is None:
                raise probe_failure or GatewayRoutingError(
                    "authorized direct route did not resolve"
                )
            deployment = route.deployment
            require_gateway_provider(deployment.provider)
            preflight_gateway_request(provider_request, deployment.gateway.capabilities)
            upstream_payload = _build_upstream_payload(profile, provider_request)
            attempt_id = self._components.ledger.start_attempt(
                snapshot=route.snapshot,
                deployment=deployment,
                attempt_ordinal=0,
                route_depth=0,
                maximum_cost_micro_usd=maximum_attempt_cost_micro_usd(request, deployment),
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
        except ProviderCapabilityError as exc:
            failure = GatewayFailure(
                failure_class=GatewayFailureClass.UNSUPPORTED_CAPABILITY,
                safe_message="the resolved deployment does not support the requested capability",
            )
            self._finish_request_quietly(authorization, failure)
            raise NativeBridgeError(public_failure_error(failure)) from exc
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            error = _authority_error(exc)
            failure = GatewayFailure(
                failure_class=GatewayFailureClass.INTERNAL,
                safe_message="gateway admission failed before provider dispatch",
            )
            if attempt_id is not None:
                self._finish_attempt_quietly(attempt_id, failure)
            elif accepted:
                self._finish_request_quietly(authorization, failure)
            raise error from exc

        with self._lock:
            self._inflight[authorization.request_id] = (authorization, attempt_id, deadline)
        response: JsonObject = {
            "request_id": authorization.request_id,
            "attempt_id": attempt_id,
            "alias": authorization.alias,
            "alias_revision_id": authorization.alias_revision_id,
            "stream": request.stream,
            "include_usage": request.include_usage,
            "exact_model_id": route.snapshot.exact_model_id,
            "provider": route.deployment.provider,
            "deployment_id": route.deployment.deployment_id,
            "route_reason": route.route_reason,
            "dialect": profile.dialect,
            "url": profile.url,
            "headers": dict(profile.headers),
            "model_id": profile.model_id,
            "timeout_seconds": profile.timeout_seconds,
            "upstream_payload": upstream_payload,
            "idempotency_key": _deployment_operation_key(route),
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
            NativeBridgeError: The durable terminal write failed; the
                in-flight entry is kept so a retried settlement (from the
                data plane or the deadline sweep) can still reach the ledger.
        """
        data = json.loads(argument)
        request_id = str(data["request_id"])
        with self._lock:
            context = self._inflight.get(request_id)
        if context is None:
            return "{}"
        _authorization, attempt_id, _deadline = context
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
        except Exception as exc:  # noqa: BLE001 - the data plane retries.
            # The in-flight entry is kept so a retried settlement can still
            # reach the ledger; a durable loss is latched by the sweep, which
            # keeps retrying the entry after its deadline.
            raise _authority_error(exc) from exc
        with self._lock:
            self._inflight.pop(request_id, None)
        return "{}"

    def models(self, argument: str) -> str:
        """Return the granted model list body for one authenticated key."""
        data = json.loads(argument)
        try:
            authorities = self._components.store.granted_alias_authorities(raw_key=data["raw_key"])
            body = public_model_list(
                authorities,
                metadata_by_alias=listing_metadata_by_alias(
                    authorities, self._components.routes.published_metadata
                ),
            )
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise _authority_error(exc) from exc
        return json.dumps(body, separators=(",", ":"))

    def model_detail(self, argument: str) -> str:
        """Return one granted model object or the shared no-oracle 404."""
        data = json.loads(argument)
        try:
            authorities = self._components.store.granted_alias_authorities(raw_key=data["raw_key"])
            authority = require_granted_authority(authorities, data["model_id"])
            body = public_model_object(
                authority,
                metadata=listing_metadata_by_alias(
                    (authority,), self._components.routes.published_metadata
                ).get(authority[0]),
            )
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise _authority_error(exc) from exc
        return json.dumps(body, separators=(",", ":"))

    def usage_json(self, argument: str) -> str:
        """Return the content-free usage report body."""
        del argument
        report = read_usage_report(
            self._components.ledger,
            organization_id=self._components.organization_id,
        )
        return json.dumps(report.model_dump(mode="json"), separators=(",", ":"))

    def readiness(self, argument: str) -> str:
        """Return whether shared executor and bridge accounting stay healthy."""
        del argument
        if not self._accounting_healthy:
            return "false"
        try:
            self._components.executor.require_healthy()
        except GatewayExecutionError:
            return "false"
        return "true"

    def _decode_body(self, body: str) -> DecodedGatewayRequest:
        """Decode one raw Chat Completions body with the shared decoder.

        Args:
            body: Raw request body text.

        Returns:
            The public alias and canonical request.

        Raises:
            NativeBridgeError: The body is not JSON, not an object, or fails
                the shared protocol validation.
        """
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise NativeBridgeError(
                OpenAIProtocolError(
                    status_code=400,
                    code="invalid_json",
                    message=(
                        "Request body must contain valid JSON. Re-encode the payload and resend."
                    ),
                )
            ) from exc
        if not isinstance(payload, dict):
            raise NativeBridgeError(
                OpenAIProtocolError(
                    status_code=400,
                    code="invalid_request",
                    message="Request body must be a JSON object. Re-encode the payload and resend.",
                )
            )
        try:
            return decode_chat(payload)
        except OpenAIProtocolError as exc:
            raise NativeBridgeError(exc) from exc

    def _wire_profile(self, route: GatewayRoute) -> GatewayWireProfile:
        """Resolve one deployment's public wire profile for the data plane.

        Args:
            route: Resolved single-deployment route.

        Returns:
            The dialect, endpoint, headers, and timing facts for dispatch,
            with the model identity filled from the resolved snapshot when
            the profile leaves it empty.

        Raises:
            _NativeDialectUnavailableError: The provider has no native-dialect
                implementation; the python engine serves the request.
            GatewayRoutingError: The resolved client cannot stream or the
                authorized catalog is not loaded.
            ValueError: The resolved client drifts from the frozen deployment.
        """
        authorization = route.snapshot.authorization
        catalog = self._components.runtime_catalogs.get(
            (authorization.alias_revision_id, authorization.catalog_sha256)
        )
        if catalog is None:
            raise GatewayRoutingError("runtime catalog is not loaded for the authorized revision")
        deployment = route.deployment
        resolved = catalog.resolve(deployment.source_alias)
        _require_deployment_identity(deployment, resolved)
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
        if not profile.model_id:
            profile = replace(profile, model_id=resolved.snapshot.model_id)
        return profile

    def _sweep_expired(self) -> None:
        """Settle abandoned in-flight attempts whose deadline has passed.

        The data plane settles every request it serves, including on client
        disconnect; this sweep is the backstop for wire-contract failures and
        data-plane crashes short of process death, so a budget reservation
        can never outlive its request deadline by more than the grace.
        """
        now = time.monotonic()
        with self._lock:
            expired = [
                (request_id, context)
                for request_id, context in self._inflight.items()
                if context[2] + _SWEEP_GRACE_SECONDS < now
            ][:_SWEEP_BATCH]
        if not expired:
            return
        failure = GatewayFailure(
            failure_class=GatewayFailureClass.CANCELLED,
            safe_message="gateway request was abandoned before settlement",
        )
        terminal = GatewayEvent(kind=GatewayEventKind.FAILED, sequence_number=0, failure=failure)
        for request_id, (_authorization, attempt_id, _deadline) in expired:
            try:
                self._components.ledger.finish_attempt(
                    attempt_id=attempt_id,
                    terminal_event=terminal,
                    failure=failure,
                    finalize_request=True,
                )
            except Exception:  # noqa: BLE001 - latch and keep the entry for retry.
                self._accounting_healthy = False
                continue
            with self._lock:
                self._inflight.pop(request_id, None)

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

    def _finish_attempt_quietly(self, attempt_id: str, failure: GatewayFailure) -> None:
        """Settle one started attempt without masking the primary failure."""
        terminal = GatewayEvent(kind=GatewayEventKind.FAILED, sequence_number=0, failure=failure)
        try:
            self._components.ledger.finish_attempt(
                attempt_id=attempt_id,
                terminal_event=terminal,
                failure=failure,
                finalize_request=True,
            )
        except Exception:  # noqa: BLE001 - primary admission failure stays authoritative.
            self._accounting_healthy = False


def _build_upstream_payload(
    profile: GatewayWireProfile,
    provider_request: GatewayRequest,
) -> JsonObject:
    """Build the provider wire payload with the shared dialect builders.

    Args:
        profile: The resolved connection's wire profile.
        provider_request: Canonical request forced into streaming mode.

    Returns:
        The exact JSON payload the python engine would send upstream.

    Raises:
        ProviderCapabilityError: The request uses a capability this dialect
            cannot preserve, mirroring the python engine's dispatch behavior.
    """
    if profile.dialect == "openai_responses":
        return openai_responses_stream_payload(
            profile.model_id,
            provider_request,
            supports_temperature=profile.supports_temperature,
            reasoning_effort=profile.reasoning_effort,
        )
    if profile.dialect == "anthropic_messages":
        return anthropic_messages_stream_payload(profile.model_id, provider_request)
    return openai_compatible_stream_payload(
        profile.model_id,
        provider_request,
        token_limit_key=profile.token_limit_key,
    )


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
