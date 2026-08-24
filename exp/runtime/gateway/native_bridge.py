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
path cannot serve (multi-deployment pools, resolved clients exposing no
native wire profile) are answered with an ``{"escalate": reason}`` admission
disposition before any ledger write; the data plane replays those against the
embedded python engine, which performs its own full authorization and
accounting, so nothing is double-counted.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.boundary import boundary_protocol_error
from exp.runtime.gateway.budgets import BudgetReservationRejected, maximum_attempt_cost_micro_usd
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
)
from exp.runtime.gateway.discovery import (
    listing_metadata_by_alias,
    public_model_list,
    public_model_object,
    require_granted_authority,
)
from exp.runtime.gateway.execution import GatewayExecutionError
from exp.runtime.gateway.group_commit import SyncGroupCommitLedger
from exp.runtime.gateway.guardrails.client import assert_not_internal_classification
from exp.runtime.gateway.guardrails.contracts import GuardrailPolicy, GuardrailRejected
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.native import enforce_native_input, enforce_native_output
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.native_components import NativeGatewayComponents, SyncWriteLedger
from exp.runtime.gateway.native_decode import NativeDecodeError, decode_native_body
from exp.runtime.gateway.native_dispatch import (
    NativeDialectUnavailableError,
    dispatch_signature_headers,
    frozen_dispatch,
    resolve_wire_profile,
)
from exp.runtime.gateway.native_metrics_text import render_metrics_text
from exp.runtime.gateway.native_responses import (
    ContinuationContext,
    continued_request,
    remember_turn,
    responses_envelope,
)
from exp.runtime.gateway.native_settlement import (
    budget_quota_protocol_error,
    deployment_operation_key,
    first_token_at_from_settlement,
    optional_text,
    terminal_from_settlement,
)
from exp.runtime.gateway.routing import GatewayRoute, GatewayRoutingError
from exp.runtime.gateway.usage import GatewayUsageReport, read_usage_report, usage_html
from exp.runtime.models.providers import (
    preflight_gateway_request,
    require_gateway_provider,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.protocol import GatewayDispatchSigner, NativeWireClient
from exp.runtime.models.providers.streaming_requests import dialect_stream_payload
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, public_failure_error
from exp.runtime.openai_protocol.requests import DecodedGatewayRequest
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
    policy: GuardrailPolicy | None = field(default=None)
    # Body-signing dialects only: the resolved client that signs the frozen
    # body at dispatch time through the ``sign_dispatch`` callback.
    signer: GatewayDispatchSigner | None = field(default=None)


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
        data_plane_metrics: Callable[[], str] | None = None,
        continuation_store: BoundedContinuationStore | None = None,
        readiness_probe: Callable[[], bool] | None = None,
        usage_reporter: Callable[[], JsonObject] | None = None,
        budget_error_factory: Callable[[str], NativeBridgeError] | None = None,
        native_route_eligible: Callable[[GatewayRoute, GatewayRequest], bool] | None = None,
        guardrails: GuardrailEngine | None = None,
    ) -> None:
        """Bind loaded gateway components for serving.

        Args:
            components: Authority, ledger, routes, and runtime catalogs.
            request_timeout_seconds: Total per-request budget from admission.
            data_plane_metrics: Optional provider of the native engine's
                content-free metrics snapshot as one JSON string; the local
                launch injects ``exp_gateway_native.metrics_snapshot_json``.
                Without it the snapshot reports ``data_plane`` as ``None``.
            continuation_store: Optional Responses continuation state, shared
                with the embedded python engine so both engines resolve and
                retain the same bounded namespaced history.
            readiness_probe: Optional hosted lifecycle readiness callback.
            usage_reporter: Optional hosted usage report callback.
            budget_error_factory: Optional hosted mapping for a rejected reservation.
            native_route_eligible: Optional hosted policy for complete native semantics.
            guardrails: Optional identity-scoped engine. ``None`` leaves traffic unguarded.
        """
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self._components = components
        # Hosted compositions have no local group-commit writer; they settle
        # directly through their own synchronous ledger.
        group_writer = getattr(components, "write_ledger", None)
        self._write_ledger: SyncWriteLedger = (
            SyncGroupCommitLedger(group_writer) if group_writer is not None else components.ledger
        )
        self._request_timeout_seconds = request_timeout_seconds
        self._data_plane_metrics = data_plane_metrics
        self._continuations = (
            continuation_store if continuation_store is not None else BoundedContinuationStore()
        )
        self._readiness_probe = readiness_probe
        self._usage_reporter = usage_reporter
        self._budget_error_factory = budget_error_factory
        self._native_route_eligible = native_route_eligible
        self._guardrails = guardrails
        self._inflight: dict[str, _InflightAttempt] = {}
        self._lock = threading.Lock()
        self._accounting_healthy = True
        self._sweep_retained_replayed = 0
        self._sweep_abandoned_cancelled = 0
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
        """Decode, authorize, inspect, route, and durably start one attempt.

        The raw body is decoded with the same ``decode_chat`` the python
        engine uses, and the upstream payload is built with the same shared
        payload builders, so the two engines cannot drift at the protocol or
        provider boundary.

        Args:
            argument: JSON object with ``raw_key``, ``body`` (raw request
                body text), optional ``surface`` (``"chat"`` or
                ``"responses"``, defaulting to chat), and optional
                ``app_referer``/``app_title`` caller app identity.

        Returns:
            JSON wire configuration for the single resolved deployment,
            including the fully built upstream payload, or an
            ``{"escalate": reason}`` disposition (returned only before any
            ledger write) handing the request to the python engine.

        Raises:
            NativeBridgeError: Decoding, authorization, routing, capability,
                or budget admission failed.
        """
        assert_not_internal_classification()
        self._sweep_expired()
        data = json.loads(argument)
        surface = str(data.get("surface", "chat"))
        decoded = self._decode_body(
            data["body"],
            surface=surface,
            idempotency_key=optional_text(data.get("idempotency_key")),
            client_request_id=optional_text(data.get("client_request_id")),
        )
        request = decoded.request
        deadline = time.monotonic() + self._request_timeout_seconds
        try:
            # ``app_referer``/``app_title`` are forwarded when the native engine includes the
            # caller HTTP-Referer/X-Title in its admit payload; absent them app attribution
            # stays null on the default path until the Rust engine populates them.
            authorization = self._components.store.authorize_request(
                raw_key=data["raw_key"],
                alias=decoded.alias,
                request=request,
                deadline_monotonic=deadline,
                app_referer=optional_text(data.get("app_referer")),
                app_title=optional_text(data.get("app_title")),
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

        policy = None
        try:
            request, policy = enforce_native_input(
                self._guardrails,
                authorization=authorization,
                request=request,
                deadline_monotonic=deadline,
            )
        except GuardrailRejected as exc:
            raise NativeBridgeError(public_failure_error(exc.failure)) from exc

        # Escalation runs after input enforcement and before any ledger write.
        # The python engine re-inspects the original public body when it
        # serves the request. Routing failures found by the probe are recorded
        # against the accepted request below.
        probe_failure: Exception | None = None
        route: GatewayRoute | None = None
        profile: GatewayWireProfile | None = None
        wire_client: NativeWireClient | None = None
        try:
            route = self._resolve_route(authorization, request)
            profile, wire_client = resolve_wire_profile(self._components.runtime_catalogs, route)
        except NativeDialectUnavailableError as exc:
            return _escalation(str(exc))
        except Exception as exc:  # noqa: BLE001 - recorded after acceptance below.
            probe_failure = exc
        if route is not None and route.fallback_deployments:
            return _escalation("multi-deployment pools use the python engine's certified waterfall")
        if route is not None and self._native_route_eligible is not None:
            try:
                native_route_eligible = self._native_route_eligible(route, request)
            except Exception:  # noqa: BLE001 - hosted policy fails closed to Python.
                native_route_eligible = False
            if not native_route_eligible:
                return _escalation("host policy requires the python execution engine")

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
            upstream_body, dispatch_signer = frozen_dispatch(profile, wire_client, upstream_payload)
            maximum_cost = maximum_attempt_cost_micro_usd(request, deployment)
            attempt_id = self._write_ledger.start_attempt(
                snapshot=route.snapshot,
                deployment=deployment,
                attempt_ordinal=0,
                route_depth=0,
                maximum_cost_micro_usd=maximum_cost,
                route_reason=route.route_reason,
                fallback_reason=route.fallback_reason,
            )
        except BudgetReservationRejected as exc:
            error = (
                NativeBridgeError(budget_quota_protocol_error())
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
                policy=policy,
                signer=dispatch_signer,
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
            # A signed dispatch carries only the frozen body string; shipping
            # the structured payload too would double the boundary bytes for
            # a value the data plane must not re-serialize anyway.
            "upstream_payload": None if upstream_body is not None else upstream_payload,
            "upstream_body": upstream_body,
            "idempotency_key": deployment_operation_key(route),
            "output_guardrail": bool(policy is not None and policy.output_checks),
        }
        if request.surface == GatewayApiSurface.RESPONSES:
            response["surface"] = "responses"
            response["envelope"] = responses_envelope(request)
        return json.dumps(response, separators=(",", ":"))

    def sign_dispatch(self, argument: str) -> str:
        """Sign one frozen dispatch body immediately before the provider POST.

        The data plane calls this after it acquires its bounded dispatch
        permit, so queue time can never age a signature toward AWS's short
        clock window; the immediate bounded open retry reuses the result
        within milliseconds.

        Args:
            argument: JSON object with ``request_id``, the exact ``url``, and
                the exact frozen ``body`` string the data plane will send.

        Returns:
            JSON object with the ``headers`` to send verbatim.

        Raises:
            NativeBridgeError: The attempt is unknown, carries no signer, or
                credential resolution failed.
        """
        data = json.loads(argument)
        with self._lock:
            entry = self._inflight.get(str(data.get("request_id") or ""))
        try:
            headers = dispatch_signature_headers(
                entry.signer if entry is not None else None,
                url=str(data["url"]),
                body=str(data["body"]),
            )
        except OpenAIProtocolError as exc:
            raise NativeBridgeError(exc) from exc
        return json.dumps({"headers": headers}, separators=(",", ":"))

    def enforce_output(self, argument: str) -> str:
        """Run one output-chain callback for a native buffered completion."""
        data = json.loads(argument)
        request_id = str(data.get("request_id") or "")
        with self._lock:
            entry = self._inflight.get(request_id)
            policy = None if entry is None else entry.policy
            deadline = time.monotonic() if entry is None else entry.deadline_monotonic
        return enforce_native_output(
            self._guardrails,
            policy,
            argument,
            deadline_monotonic=deadline,
        )

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
            idempotency_key=optional_text(data.get("idempotency_key")),
            client_request_id=optional_text(data.get("client_request_id")),
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
                app_referer=optional_text(data.get("app_referer")),
                app_title=optional_text(data.get("app_title")),
            )
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise _authority_error(exc) from exc
        if not isinstance(authorization.target, DirectTarget):
            return _escalation("project-backed aliases use learned selection on the python engine")
        try:
            route = self._components.routes.resolve_direct(authorization)
            resolve_wire_profile(self._components.runtime_catalogs, route)
        except NativeDialectUnavailableError as exc:
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
        terminal, failure = terminal_from_settlement(data)
        first_token_at = first_token_at_from_settlement(data)
        try:
            self._write_ledger.finish_attempt(
                attempt_id=entry.attempt_id,
                terminal_event=terminal,
                failure=failure,
                finalize_request=True,
                first_token_at=first_token_at,
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
        # Only the local composition (SQLite ledger) serves native usage;
        # hosted compositions disable it and own their own usage surface.
        return read_usage_report(
            cast("SQLiteAttemptLedger", self._components.ledger),
            organization_id=self._components.organization_id,
            identity_id=identity_id,
        )

    def metrics_snapshot(self) -> JsonObject:
        """Compose the one content-free observability snapshot.

        ``data_plane`` carries the native engine's registry when a provider
        is bound, otherwise ``None``; ``control_plane`` carries this bridge's
        own sweep recoveries, in-flight registry size, reconciliation counts,
        and accounting health. The native ``/metrics.json`` route serves
        exactly this body; ``/metrics`` serves its Prometheus text rendering.
        """
        data_plane: JsonObject | None = None
        if self._data_plane_metrics is not None:
            data_plane = json.loads(self._data_plane_metrics())
        with self._lock:
            control_plane: JsonObject = {
                "sweep_retained_settlements_replayed": self._sweep_retained_replayed,
                "sweep_abandoned_attempts_cancelled": self._sweep_abandoned_cancelled,
                "inflight_attempts": len(self._inflight),
                "reconciled_expired_requests": self._components.reconciled_expired_requests,
                "reconciled_unknown_attempts": self._components.reconciled_unknown_attempts,
                "accounting_healthy": self._accounting_healthy,
            }
        return {"data_plane": data_plane, "control_plane": control_plane}

    def metrics_json(self, argument: str) -> str:
        """Return the content-free metrics snapshot body for the data plane."""
        del argument
        return json.dumps(self.metrics_snapshot(), separators=(",", ":"))

    def metrics_text(self, argument: str) -> str:
        """Return ``{"text": ...}`` holding the snapshot's Prometheus exposition."""
        del argument
        text = render_metrics_text(self.metrics_snapshot())
        return json.dumps({"text": text}, separators=(",", ":"))

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
        """Decode one raw request body with the shared surface decoder."""
        try:
            return decode_native_body(
                body,
                surface=surface,
                idempotency_key=idempotency_key,
                client_request_id=client_request_id,
            )
        except NativeDecodeError as exc:
            raise NativeBridgeError(exc.error) from exc

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
            terminal, failure = terminal_from_settlement(settlement)
            if self._settle_swept(request_id, entry, terminal, failure, latch_on_failure=True):
                with self._lock:
                    self._sweep_retained_replayed += 1
        if not abandoned:
            return
        cancelled = GatewayFailure(
            failure_class=GatewayFailureClass.CANCELLED,
            safe_message="gateway request was abandoned before settlement",
        )
        terminal = GatewayEvent(kind=GatewayEventKind.FAILED, sequence_number=0, failure=cancelled)
        for request_id, entry in abandoned:
            if self._settle_swept(request_id, entry, terminal, cancelled, latch_on_failure=True):
                with self._lock:
                    self._sweep_abandoned_cancelled += 1

    def _settle_swept(
        self,
        request_id: str,
        entry: _InflightAttempt,
        terminal: GatewayEvent,
        failure: GatewayFailure | None,
        *,
        latch_on_failure: bool,
    ) -> bool:
        """Land one swept settlement; keep the entry for retry on failure.

        Returns:
            Whether the swept terminal write reached the ledger.
        """
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
            return False
        with self._lock:
            self._inflight.pop(request_id, None)
        return True

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
