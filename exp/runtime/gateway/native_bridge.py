"""Python control plane for the native (Rust) gateway data plane.

The native engine (`exp_gateway_native`) owns the HTTP socket, upstream
streaming, provider-event normalization, and public SSE encoding. Everything
protocol- and authority-shaped happens here, in the same code the python
engine runs: request decoding through ``decode_chat`` and
``decode_responses``, authorization through the shared control store,
upstream payload construction through the shared ``streaming_requests``
builders, Responses continuation state through the shared bounded store, and
the same durable SQLite ledger transactions. Ledger writes go through the
blocking facade over the shared group-commit writer, so both engines' writes
interleave in the same batched fsyncs while every caller still blocks until
its own write is durable. Every boundary call takes and returns one JSON
string so the boundary stays narrow and typed on both sides.

Boundary errors raise :class:`NativeBridgeError`, whose ``public_error_json``
attribute carries the sanitized OpenAI-shaped error the data plane returns to
the caller, mirroring ``GatewayService`` error mapping. Requests the native
path cannot serve (multi-deployment pools, providers without a native
dialect) are answered with an ``{"escalate": reason}`` admission
disposition before any ledger write; the data plane replays those against the
embedded python engine, which performs its own full authorization and
accounting, so nothing is double-counted.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol, cast

from exp.common.core.artifacts import JsonObject, stable_id
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.boundary import boundary_protocol_error
from exp.runtime.gateway.budgets import BudgetReservationRejected, maximum_attempt_cost_micro_usd
from exp.runtime.gateway.contracts import (
    AttemptId,
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
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
    GatewayExecutor,
    _require_deployment_identity,  # noqa: PLC2701
)
from exp.runtime.gateway.group_commit import SyncGroupCommitLedger
from exp.runtime.gateway.interfaces import GatewayControlStore
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.native_responses import (
    ContinuationContext,
    continued_request,
    remember_turn,
    responses_envelope,
)
from exp.runtime.gateway.routing import CatalogRouteResolver, GatewayRoute, GatewayRoutingError
from exp.runtime.gateway.usage import GatewayUsageReport, read_usage_report, usage_html
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers import (
    preflight_gateway_request,
    require_gateway_provider,
)
from exp.runtime.models.providers.base import GatewayWireProfile, ProviderHttpClient
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.streaming_requests import dialect_stream_payload
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, public_failure_error
from exp.runtime.openai_protocol.requests import (
    DecodedGatewayRequest,
    decode_chat,
    decode_responses,
)
from exp.runtime.openai_protocol.state import (
    BoundedContinuationStore,
    ProtocolNamespace,
    episode_namespace,
    replay_key,
)

_REQUEST_TIMEOUT_SECONDS = 120.0
_SWEEP_GRACE_SECONDS = 5.0
_SWEEP_INTERVAL_SECONDS = 5.0
_SWEEP_BATCH = 16

_TERMINAL_KINDS = {
    "completed": GatewayEventKind.COMPLETED,
    "incomplete": GatewayEventKind.INCOMPLETE,
    "failed": GatewayEventKind.FAILED,
}


class NativeAttemptLedger(Protocol):
    """Synchronous durable ledger used from native callback threads."""

    def accept_request(self, *, authorization: AuthorizationSnapshot) -> None:
        """Persist one accepted request."""
        ...

    def start_attempt(
        self,
        *,
        snapshot: ExecutionSnapshot,
        deployment: ExactModelDeployment,
        attempt_ordinal: int,
        route_depth: int,
        maximum_cost_micro_usd: int | None = None,
        route_reason: str | None = None,
        fallback_reason: str | None = None,
    ) -> AttemptId:
        """Reserve and persist one provider attempt."""
        ...

    def finish_attempt(
        self,
        *,
        attempt_id: AttemptId,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
    ) -> None:
        """Settle one provider attempt."""
        ...

    def finish_request(
        self,
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
    ) -> None:
        """Finalize accepted work that failed before dispatch."""
        ...


class NativeGatewayComponents(Protocol):
    """Engine-neutral components required by the native control plane."""

    @property
    def store(self) -> GatewayControlStore:
        """Return the authority store."""
        ...

    @property
    def ledger(self) -> NativeAttemptLedger:
        """Return the synchronous durable accounting ledger."""
        ...

    @property
    def routes(self) -> CatalogRouteResolver:
        """Return the direct-route resolver."""
        ...

    @property
    def executor(self) -> GatewayExecutor:
        """Return the shared accounting-health latch."""
        ...

    @property
    def reconciled_expired_requests(self) -> int:
        """Return startup-reconciled request count."""
        ...

    @property
    def reconciled_unknown_attempts(self) -> int:
        """Return startup-reconciled attempt count."""
        ...

    @property
    def runtime_catalogs(self) -> Mapping[tuple[str, str], RuntimeModelCatalog]:
        """Return runtime catalogs keyed by alias revision and digest."""
        ...

    @property
    def organization_id(self) -> str:
        """Return the organization used by the local usage endpoint."""
        ...


class _NativeDialectUnavailableError(RuntimeError):
    """The resolved provider has no native dialect; python must serve it."""


@dataclass
class _InflightAttempt:
    """One admitted attempt awaiting its durable terminal settlement."""

    authorization: AuthorizationSnapshot
    attempt_id: str
    deadline_monotonic: float
    # The exact settlement the data plane could not land; the sweep replays it
    # verbatim so a completed outcome and its usage are never downgraded.
    pending_settlement: JsonObject | None = field(default=None)
    # Responses-only retention facts consumed by ``remember`` after a
    # successful terminal; chat attempts carry ``None``.
    continuation: ContinuationContext | None = field(default=None)


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

    Methods are called from multiple Rust worker threads. Ledger writes block
    on the shared group-commit writer's blocking facade, so concurrent
    settlements from both engines amortize one fsync per batch while each
    caller still observes only its own durable commit; reads use the raw
    ledger's per-thread connection cache. The in-flight request registry is
    guarded by one lock and swept opportunistically so an abandoned
    reservation cannot outlive its request deadline by more than the sweep
    grace.
    """

    def __init__(
        self,
        components: NativeGatewayComponents,
        *,
        request_timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS,
        continuation_store: BoundedContinuationStore | None = None,
        readiness_probe: Callable[[], bool] | None = None,
        usage_reporter: Callable[[], JsonObject] | None = None,
        budget_error_factory: Callable[[str], NativeBridgeError] | None = None,
        route_context_recorder: Callable[[AttemptId, str | None, str | None], None] | None = None,
    ) -> None:
        """Bind loaded gateway components for serving.

        Args:
            components: Authority, ledger, routes, and runtime catalogs.
            request_timeout_seconds: Total per-request budget from admission.
            continuation_store: Optional Responses continuation state, shared
                with the embedded python engine so both engines resolve and
                retain the same bounded namespaced history.
            readiness_probe: Optional hosted lifecycle readiness callback. The
                local engine defaults to the shared executor health latch.
            usage_reporter: Optional hosted usage report callback. The local
                engine defaults to its single-organization SQLite report.
            budget_error_factory: Optional hosted mapping for a rejected
                reservation, keyed by the presented virtual key.
            route_context_recorder: Optional hosted sink for display-safe
                route and fallback reason codes after durable reservation.
        """
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self._components = components
        self._write_ledger = SyncGroupCommitLedger(components.write_ledger)
        self._request_timeout_seconds = request_timeout_seconds
        self._continuations = (
            continuation_store if continuation_store is not None else BoundedContinuationStore()
        )
        self._readiness_probe = readiness_probe
        self._usage_reporter = usage_reporter
        self._budget_error_factory = budget_error_factory
        self._route_context_recorder = route_context_recorder
        self._inflight: dict[str, _InflightAttempt] = {}
        self._lock = threading.Lock()
        self._accounting_healthy = True
        # The sweep also runs on a timer so retained settlements and abandoned
        # attempts are recovered even when no further requests arrive.
        self._sweeper = threading.Thread(
            target=self._sweep_loop,
            name="exp-native-settlement-sweep",
            daemon=True,
        )
        self._sweeper.start()

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
            argument: JSON object with ``raw_key``, ``body`` (raw request
                body text), and optional ``surface`` (``"chat"`` or
                ``"responses"``, defaulting to chat).

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
        surface = str(data.get("surface", "chat"))
        decoded = self._decode_body(
            data["body"],
            surface=surface,
            idempotency_key=_optional_text(data.get("idempotency_key")),
            client_request_id=_optional_text(data.get("client_request_id")),
        )
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

        # Responses continuation resolves after authorization and before any
        # ledger write, the same order the python engine uses; unavailable,
        # expired, evicted, or cross-namespace state fails closed here.
        continuation_context: ContinuationContext | None = None
        if request.surface == GatewayApiSurface.RESPONSES:
            try:
                request, continuation_context = continued_request(
                    self._continuations,
                    authorization=authorization,
                    request=request,
                )
            except OpenAIProtocolError as exc:
                raise NativeBridgeError(exc) from exc

        # Escalation runs before any ledger write: the python engine performs
        # full accounting for every request it serves. Routing failures found
        # by the probe are recorded against the accepted request below, the
        # same order the python engine writes them.
        probe_failure: Exception | None = None
        route: GatewayRoute | None = None
        profile: GatewayWireProfile | None = None
        try:
            route = self._resolve_route(authorization, request)
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
            self._write_ledger.accept_request(authorization=authorization)
            accepted = True
            if probe_failure is not None or route is None or profile is None:
                raise probe_failure or GatewayRoutingError("authorized route did not resolve")
            deployment = route.deployment
            require_gateway_provider(deployment.provider)
            preflight_gateway_request(provider_request, deployment.gateway.capabilities)
            upstream_payload = dialect_stream_payload(profile, provider_request)
            maximum_cost = maximum_attempt_cost_micro_usd(request, deployment)
            if self._route_context_recorder is None:
                attempt_id = self._write_ledger.start_attempt(
                    snapshot=route.snapshot,
                    deployment=deployment,
                    attempt_ordinal=0,
                    route_depth=0,
                    maximum_cost_micro_usd=maximum_cost,
                    route_reason=route.route_reason,
                    fallback_reason=route.fallback_reason,
                )
            else:
                attempt_id = self._write_ledger.start_attempt(
                    snapshot=route.snapshot,
                    deployment=deployment,
                    attempt_ordinal=0,
                    route_depth=0,
                    maximum_cost_micro_usd=maximum_cost,
                )
                self._route_context_recorder(
                    attempt_id,
                    route.route_reason,
                    route.fallback_reason,
                )
        except BudgetReservationRejected as exc:
            error = (
                _budget_quota_error()
                if self._budget_error_factory is None
                else self._budget_error_factory(data["raw_key"])
            )
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
            self._inflight[authorization.request_id] = _InflightAttempt(
                authorization=authorization,
                attempt_id=attempt_id,
                deadline_monotonic=deadline,
                continuation=continuation_context,
            )
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
        if request.surface == GatewayApiSurface.RESPONSES:
            response["surface"] = "responses"
            response["envelope"] = responses_envelope(request)
        return json.dumps(response, separators=(",", ":"))

    def claim_scope(self, argument: str) -> str:
        """Resolve the replay-store scope for one keyed Chat Completions request.

        The data plane owns the bounded in-process replay store; this call
        performs the same decode and authorization the python engine runs so
        the store key (tenant namespace, hashed caller operation, canonical
        request digest) is computed by exactly one implementation. Requests
        the native path cannot serve are escalated before any replay claim,
        so one caller operation never spans both engines' replay stores.

        Args:
            argument: JSON object with ``raw_key``, ``body``, and optional
                ``idempotency_key`` and ``client_request_id`` header values.

        Returns:
            JSON replay scope with ``organization_id``, ``identity_id``,
            ``alias_revision_id``, ``surface``, ``caller_operation_sha256``,
            and ``canonical_request_sha256``, or an ``{"escalate": reason}``
            disposition handing the request to the python engine.

        Raises:
            NativeBridgeError: Decoding or authorization failed.
        """
        data = json.loads(argument)
        decoded = self._decode_body(
            data["body"],
            idempotency_key=_optional_text(data.get("idempotency_key")),
            client_request_id=_optional_text(data.get("client_request_id")),
        )
        request = decoded.request
        caller_operation = request.idempotency_key or request.client_request_id
        if caller_operation is None:
            raise NativeBridgeError(
                OpenAIProtocolError(
                    status_code=400,
                    code="invalid_request",
                    message="A replay scope requires an Idempotency-Key header.",
                    param="Idempotency-Key",
                )
            )
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
        if not isinstance(authorization.target, DirectTarget):
            return _escalation("project-backed aliases use learned selection on the python engine")
        try:
            route = self._components.routes.resolve_direct(authorization)
            self._wire_profile(route)
        except _NativeDialectUnavailableError as exc:
            return _escalation(str(exc))
        except Exception:  # noqa: BLE001 - the owner's admission records this failure.
            route = None
        if route is not None and route.fallback_deployments:
            return _escalation("multi-deployment pools use the python engine's certified waterfall")
        key = replay_key(
            namespace=ProtocolNamespace(
                organization_id=authorization.organization_id,
                identity_id=authorization.identity_id,
                alias_revision_id=authorization.alias_revision_id,
            ),
            surface=request.surface,
            caller_operation=caller_operation,
            canonical_request_sha256=authorization.canonical_request_sha256,
        )
        if key is None:  # pragma: no cover - caller_operation is checked above.
            raise NativeBridgeError(
                OpenAIProtocolError(
                    status_code=500,
                    code="internal_error",
                    message="The gateway request failed.",
                    error_type="api_error",
                )
            )
        scope: JsonObject = {
            "organization_id": key.namespace.organization_id,
            "identity_id": key.namespace.identity_id,
            "alias_revision_id": key.namespace.alias_revision_id,
            "surface": key.surface.value,
            "caller_operation_sha256": key.caller_operation_sha256,
            "canonical_request_sha256": key.canonical_request_sha256,
        }
        return json.dumps(scope, separators=(",", ":"))

    def remember(self, argument: str) -> str:
        """Retain one completed Responses continuation within strict bounds.

        Args:
            argument: JSON object with ``request_id``, aggregated ``text``,
                ``refusal`` presence, and completed ``tool_calls``.

        Returns:
            An empty JSON object; retention that does not apply is a no-op.

        Raises:
            NativeBridgeError: The continuation exceeds the bounded store or
                a completed tool call carried malformed fields.
        """
        data = json.loads(argument)
        request_id = str(data["request_id"])
        with self._lock:
            entry = self._inflight.get(request_id)
        context = entry.continuation if entry is not None else None
        if context is None:
            return "{}"
        try:
            remember_turn(self._continuations, context=context, data=data)
        except OpenAIProtocolError as exc:
            raise NativeBridgeError(exc) from exc
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise _authority_error(exc) from exc
        return "{}"

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
            entry = self._inflight.get(request_id)
        if entry is None:
            return "{}"
        terminal, failure = _terminal_from_settlement(data)
        try:
            self._write_ledger.finish_attempt(
                attempt_id=entry.attempt_id,
                terminal_event=terminal,
                failure=failure,
                finalize_request=True,
            )
        except Exception as exc:  # noqa: BLE001 - the data plane retries.
            # The exact settlement is retained so a retry (from the data
            # plane or the timer sweep) lands the ORIGINAL outcome and usage,
            # never a downgraded cancellation.
            with self._lock:
                entry.pending_settlement = data
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
        """Return the content-free usage report body.

        Args:
            argument: JSON object with an optional ``raw_key``. A presented
                key scopes the report to its owning identity; an absent key
                returns the organization-wide report.

        Returns:
            The schema-versioned usage report as one JSON object.

        Raises:
            NativeBridgeError: The presented key is invalid, expired, or revoked.
        """
        if self._usage_reporter is not None:
            return json.dumps(self._usage_reporter(), separators=(",", ":"))
        report = self._usage_report(argument)
        return json.dumps(report.model_dump(mode="json"), separators=(",", ":"))

    def usage_page(self, argument: str) -> str:
        """Return the content-free usage page rendering.

        Args:
            argument: JSON object with an optional ``raw_key``, scoped exactly
                like :meth:`usage_json`.

        Returns:
            JSON object with one ``html`` field holding the rendered page.

        Raises:
            NativeBridgeError: The presented key is invalid, expired, or revoked.
        """
        report = self._usage_report(argument)
        return json.dumps({"html": usage_html(report)}, separators=(",", ":"))

    def _usage_report(self, argument: str) -> GatewayUsageReport:
        """Read the usage report for one optionally key-scoped callback.

        Args:
            argument: JSON object with an optional ``raw_key``.

        Returns:
            The organization-wide report, or the report scoped to the
            presented key's identity.

        Raises:
            NativeBridgeError: The presented key is invalid, expired, or revoked.
        """
        data = json.loads(argument)
        raw_key = data.get("raw_key")
        identity_id: str | None = None
        if raw_key is not None:
            try:
                _, identity_id = self._components.store.authenticated_identity(raw_key=str(raw_key))
            except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
                raise _authority_error(exc) from exc
        return read_usage_report(
            cast("SQLiteAttemptLedger", self._components.ledger),
            organization_id=self._components.organization_id,
            identity_id=identity_id,
        )

    def readiness(self, argument: str) -> str:
        """Return whether shared executor and bridge accounting stay healthy."""
        del argument
        if not self._accounting_healthy:
            return "false"
        if self._readiness_probe is not None:
            try:
                return "true" if self._readiness_probe() else "false"
            except Exception:  # noqa: BLE001 - readiness fails closed at the boundary.
                return "false"
        try:
            self._components.executor.require_healthy()
        except GatewayExecutionError:
            return "false"
        return "true"

    def _decode_body(
        self,
        body: str,
        *,
        surface: str = "chat",
        idempotency_key: str | None = None,
        client_request_id: str | None = None,
    ) -> DecodedGatewayRequest:
        """Decode one raw request body with the shared surface decoder.

        Args:
            body: Raw request body text.
            surface: Public surface, ``"chat"`` or ``"responses"``.
            idempotency_key: Optional raw ``Idempotency-Key`` header value.
            client_request_id: Optional raw ``X-Client-Request-Id`` header value.

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
        decoder = decode_responses if surface == "responses" else decode_chat
        try:
            return decoder(
                payload,
                idempotency_key=idempotency_key,
                client_request_id=client_request_id,
            )
        except OpenAIProtocolError as exc:
            raise NativeBridgeError(exc) from exc

    def _resolve_route(
        self,
        authorization: AuthorizationSnapshot,
        request: GatewayRequest,
    ) -> GatewayRoute:
        """Resolve one direct or project route without an event loop.

        Direct pools resolve entirely inside frozen in-memory catalogs.
        Project targets run frozen learned selection synchronously on this
        worker thread through the same selection seam and the same episode
        identity derivation the python engine uses, so the two engines share
        one policy execution path. Request-time embedding failure falls back
        to the frozen conservative baseline inside the shared runtime, and
        neither path mutates policy or evidence.
        """
        if isinstance(authorization.target, DirectTarget):
            return self._components.routes.resolve_direct(authorization)
        episode = episode_namespace(
            namespace=ProtocolNamespace(
                organization_id=authorization.organization_id,
                identity_id=authorization.identity_id,
                alias_revision_id=authorization.alias_revision_id,
            ),
            caller_episode_key=request.idempotency_key or request.client_request_id,
            request_id=authorization.request_id,
        )
        return self._components.routes.resolve_project_blocking(
            authorization=authorization,
            request=request,
            episode_namespace=episode,
        )

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

    def _sweep_loop(self) -> None:
        """Run the settlement sweep on a timer for the process lifetime."""
        while True:
            time.sleep(_SWEEP_INTERVAL_SECONDS)
            self._sweep_expired()

    def _sweep_expired(self) -> None:
        """Recover retained settlements and close abandoned attempts.

        A retained settlement (the data plane's terminal write failed) is
        replayed verbatim so the original outcome and usage land. An attempt
        with no settlement at all past its deadline plus grace is closed as
        cancelled; that is the backstop for wire-contract failures and
        data-plane crashes short of process death. A retained settlement that
        fails again here latches accounting-unhealthy as a durable loss.
        """
        now = time.monotonic()
        with self._lock:
            retained = [
                (request_id, entry)
                for request_id, entry in self._inflight.items()
                if entry.pending_settlement is not None
            ][:_SWEEP_BATCH]
            abandoned = [
                (request_id, entry)
                for request_id, entry in self._inflight.items()
                if entry.pending_settlement is None
                and entry.deadline_monotonic + _SWEEP_GRACE_SECONDS < now
            ][:_SWEEP_BATCH]
        for request_id, entry in retained:
            settlement = entry.pending_settlement
            if settlement is None:
                continue
            terminal, failure = _terminal_from_settlement(settlement)
            self._settle_swept(request_id, entry, terminal, failure, latch_on_failure=True)
        if not abandoned:
            return
        cancelled = GatewayFailure(
            failure_class=GatewayFailureClass.CANCELLED,
            safe_message="gateway request was abandoned before settlement",
        )
        terminal = GatewayEvent(kind=GatewayEventKind.FAILED, sequence_number=0, failure=cancelled)
        for request_id, entry in abandoned:
            self._settle_swept(request_id, entry, terminal, cancelled, latch_on_failure=True)

    def _settle_swept(
        self,
        request_id: str,
        entry: _InflightAttempt,
        terminal: GatewayEvent,
        failure: GatewayFailure | None,
        *,
        latch_on_failure: bool,
    ) -> None:
        """Land one swept settlement; keep the entry for retry on failure."""
        try:
            self._write_ledger.finish_attempt(
                attempt_id=entry.attempt_id,
                terminal_event=terminal,
                failure=failure,
                finalize_request=True,
            )
        except Exception:  # noqa: BLE001 - keep the entry; the sweep retries.
            if latch_on_failure:
                self._accounting_healthy = False
            return
        with self._lock:
            self._inflight.pop(request_id, None)

    def _finish_request_quietly(
        self,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
    ) -> None:
        """Finalize accepted pre-dispatch work without masking the primary failure."""
        try:
            self._write_ledger.finish_request(
                authorization=authorization,
                failure=failure,
            )
        except Exception:  # noqa: BLE001 - primary admission failure stays authoritative.
            self._accounting_healthy = False

    def _finish_attempt_quietly(self, attempt_id: str, failure: GatewayFailure) -> None:
        """Settle one started attempt without masking the primary failure."""
        terminal = GatewayEvent(kind=GatewayEventKind.FAILED, sequence_number=0, failure=failure)
        try:
            self._write_ledger.finish_attempt(
                attempt_id=attempt_id,
                terminal_event=terminal,
                failure=failure,
                finalize_request=True,
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


def _optional_text(value: object) -> str | None:
    """Return one optional boundary string value or ``None``."""
    return value if isinstance(value, str) else None


def _terminal_from_settlement(
    data: JsonObject,
) -> tuple[GatewayEvent, GatewayFailure | None]:
    """Build the durable terminal event from one settlement payload.

    Args:
        data: Parsed settlement with ``outcome``, optional ``usage``,
            ``tool_names``, and ``failure``.

    Returns:
        The terminal event and the optional normalized failure.
    """
    raw_usage = data.get("usage")
    raw_tool_names = data.get("tool_names")
    usage = _usage_from_payload(
        raw_usage if isinstance(raw_usage, dict) else None,
        [str(name) for name in raw_tool_names] if isinstance(raw_tool_names, list) else [],
    )
    failure_payload = data.get("failure")
    failure = None
    if isinstance(failure_payload, dict):
        failure = GatewayFailure(
            failure_class=GatewayFailureClass(str(failure_payload["failure_class"])),
            safe_message=str(failure_payload["safe_message"]),
        )
    kind = _TERMINAL_KINDS[str(data["outcome"])]
    terminal = GatewayEvent(
        kind=kind,
        sequence_number=0,
        usage=usage,
        failure=failure if kind == GatewayEventKind.FAILED else None,
    )
    return terminal, failure


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
