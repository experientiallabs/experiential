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

Admission returns the full ordered certified route (one wire configuration
per deployment) plus the frozen retry-policy facts, accepting the request
without starting any attempt. The data plane then reserves each physical
dispatch through ``start_attempt`` immediately before network work and lands
each attempt's durable terminal through ``settle`` (finalizing the request
only on the terminal attempt); candidate selection stays here: the frozen
waterfall policy, health circuits, and budget skipping.

Boundary errors raise :class:`NativeBridgeError`, whose ``public_error_json``
attribute carries the sanitized OpenAI-shaped error the data plane returns to
the caller through the shared boundary mapping. Requests the native path
cannot serve (resolved clients exposing no native wire profile) are answered
with an ``{"escalate": reason}`` admission disposition after the accepted
request is finalized content-free; the data plane classifies the reason for
metrics and fails the request closed with the shared internal error.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable

from exp.common.core.artifacts import JsonObject, sha256_bytes
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayApiSurface,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
)
from exp.runtime.gateway.group_commit import SyncGroupCommitLedger
from exp.runtime.gateway.guardrails.client import assert_not_internal_classification
from exp.runtime.gateway.guardrails.contracts import GuardrailRejected
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.native import enforce_native_input, enforce_native_output
from exp.runtime.gateway.native_accounting import (
    NativeAttemptAccounting,
    NativeBridgeError,
)
from exp.runtime.gateway.native_accounting import (
    authority_error as _authority_error,
)
from exp.runtime.gateway.native_components import NativeGatewayComponents, SyncWriteLedger
from exp.runtime.gateway.native_decode import NativeDecodeError, decode_native_body
from exp.runtime.gateway.native_dispatch import dispatch_signature_headers, frozen_dispatch
from exp.runtime.gateway.native_execution import (
    MAXIMUM_SAME_DEPLOYMENT_ATTEMPTS,
    MAXIMUM_TOTAL_ATTEMPTS,
    DeadRung,
    FrozenDispatchBinding,
    InflightRequest,
    NativeDialectUnavailableError,
    deployment_health_key,
    deployment_wire_entry,
    dispatchable_route_profiles,
    resolve_route_profiles,
    select_route_deployments,
)
from exp.runtime.gateway.native_observability import NativeObservabilityMixin
from exp.runtime.gateway.native_responses import (
    ContinuationContext,
    continued_request,
    remember_turn,
    responses_envelope,
)
from exp.runtime.gateway.native_settlement import (
    optional_text,
)
from exp.runtime.gateway.routing import GatewayRoute, GatewayRoutingError
from exp.runtime.models.providers import (
    preflight_gateway_request,
    require_gateway_provider,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import (
    ProviderCapabilityError,
    ProviderParameterError,
    normalized_provider_failure,
)
from exp.runtime.models.providers.generation_route_compat import (
    compatible_generation_parameter_profile_indexes,
)
from exp.runtime.models.providers.protocol import GatewayDispatchSigner, NativeWireClient
from exp.runtime.models.providers.streaming_requests import (
    dialect_stream_payload,
    route_generation_parameter_requests,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, public_failure_error
from exp.runtime.openai_protocol.requests import DecodedGatewayRequest
from exp.runtime.openai_protocol.state import (
    BoundedContinuationStore,
    ProtocolNamespace,
    episode_namespace,
    replay_key,
)

_REQUEST_TIMEOUT_SECONDS = 120.0
_PUBLIC_REQUEST_CAPABILITY_PARAMS = {
    GatewayApiSurface.CHAT_COMPLETIONS: {
        "developer_messages": "messages",
        "function_tools": "tools",
        "parallel_tool_calls": "parallel_tool_calls",
        "stop_sequences": "stop",
        "streaming": "stream",
        "streaming_tool_arguments": "stream",
        "strict_tools": "tools",
        "structured_output": "response_format",
        "structured_text": "response_format",
    },
    GatewayApiSurface.RESPONSES: {
        "developer_messages": "instructions",
        "function_tools": "tools",
        "parallel_tool_calls": "parallel_tool_calls",
        "streaming": "stream",
        "streaming_tool_arguments": "stream",
        "strict_tools": "tools",
        "structured_output": "text.format",
        "structured_text": "text.format",
    },
    GatewayApiSurface.MESSAGES: {
        "developer_messages": "system",
        "function_tools": "tools",
        "parallel_tool_calls": "tool_choice.disable_parallel_tool_use",
        "stop_sequences": "stop_sequences",
        "streaming": "stream",
        "streaming_tool_arguments": "stream",
        "strict_tools": "tools",
    },
}


def _public_capability_param(capability: str, surface: GatewayApiSurface) -> str | None:
    """Translate an internal capability label to the caller's request field."""
    return _PUBLIC_REQUEST_CAPABILITY_PARAMS[surface].get(capability)


_logger = logging.getLogger(__name__)


def _escalation(reason: str) -> str:
    """Return the admission disposition for a request the plane cannot serve.

    The data plane classifies the reason for content-free metrics and fails
    the request closed.

    Args:
        reason: Display-safe reason the native path cannot serve the request.

    Returns:
        The JSON admission body carrying the escalation disposition.
    """
    return json.dumps({"escalate": reason}, separators=(",", ":"))


class NativeControlPlane(NativeObservabilityMixin):
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
            continuation_store: Optional injected Responses continuation
                state; a host supplies its own bounded namespaced history,
                and the default is one in-process bounded store.
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
        # The accounting registry owns in-flight requests, per-dispatch
        # reservations, deployment-health circuits, and the deadline sweep.
        self._accounting = NativeAttemptAccounting(
            self._write_ledger,
            budget_error_factory=budget_error_factory,
        )

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
        """Decode, authorize, inspect, route, and durably accept one request.

        The raw body is decoded with the same ``decode_chat`` the python
        engine uses, and every deployment's upstream payload is built with
        the same shared payload builders, so the two engines cannot drift at
        the protocol or provider boundary. No attempt row is written here:
        each physical dispatch is reserved by :meth:`start_attempt`.

        Args:
            argument: JSON object with ``raw_key``, ``body`` (raw request
                body text), optional ``surface`` (``"chat"`` or
                ``"responses"``, defaulting to chat), and optional
                ``app_referer``/``app_title`` caller app identity.

        Returns:
            JSON wire configuration carrying the full ordered certified
            ``route`` (one dialect, endpoint, headers, payload, and
            per-deployment idempotency key entry per deployment) plus the
            frozen retry-policy facts, or an ``{"escalate": reason}``
            disposition (its accepted request already finalized, with no
            attempt row) naming why the native plane cannot serve the
            request.

        Raises:
            NativeBridgeError: Decoding, authorization, routing, or
                capability admission failed.
        """
        assert_not_internal_classification()
        self._accounting.sweep_expired()
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
        # ledger write; unavailable, expired, evicted, or cross-namespace
        # state fails closed here.
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

        # The ledger accepts the logical request before route selection, so a
        # keyed operation whose durable terminal already exists (or whose key
        # was reused with different content) fails closed here, before
        # learned selection can run request-time embedding or any other
        # provider-touching work.
        try:
            self._write_ledger.accept_request(authorization=authorization)
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise _authority_error(exc) from exc

        # Escalation runs after acceptance; the accepted request is finished
        # quietly before the disposition returns, so an unservable request is
        # accounted content-free and never billed. Routing failures found by
        # the probe are raised against the accepted request below.
        probe_failure: Exception | None = None
        route: GatewayRoute | None = None
        resolved_wires: tuple[tuple[GatewayWireProfile, NativeWireClient], ...] | None = None
        try:
            route = self._resolve_route(
                authorization,
                request,
                continuation=continuation_context,
            )
            # A rung that is dead at admission (a lost credential, a drifted
            # connection) is skipped so a live fallback still serves the
            # request instead of the whole request failing on a dead lead.
            dispatchable = dispatchable_route_profiles(self._components.runtime_catalogs, route)
            self._record_dead_admission_rungs(
                authorization,
                dispatchable.dead,
                fallback_available=bool(dispatchable.indexes),
            )
            if not dispatchable.indexes:
                # Every certified rung was operationally dead at admission;
                # there is nothing live to serve, so the accepted request is
                # finished closed.
                return self._escalate_accepted(
                    authorization,
                    "every certified deployment was unavailable at admission",
                )
            route = select_route_deployments(route, dispatchable.indexes)
            resolved_wires = dispatchable.resolved_wires
        except NativeDialectUnavailableError as exc:
            return self._escalate_accepted(authorization, str(exc))
        except Exception as exc:  # noqa: BLE001 - raised after route packaging below.
            probe_failure = exc
        if route is not None and self._native_route_eligible is not None:
            try:
                native_route_eligible = self._native_route_eligible(route, request)
            except Exception:  # noqa: BLE001 - hosted policy fails closed.
                native_route_eligible = False
            if not native_route_eligible:
                return self._escalate_accepted(
                    authorization,
                    "host policy does not permit native execution of this route",
                )

        # Admission returns the full ordered route; no attempt row exists
        # until the data plane's first `start_attempt`.
        public_request = request
        provider_request = request.model_copy(update={"stream": True, "include_usage": True})
        try:
            if probe_failure is not None or route is None or resolved_wires is None:
                raise probe_failure or GatewayRoutingError("authorized route did not resolve")
            compatible_indexes = compatible_generation_parameter_profile_indexes(
                tuple(profile for profile, _client in resolved_wires),
                request,
            )
            route = select_route_deployments(route, compatible_indexes)
            resolved_wires = tuple(resolved_wires[index] for index in compatible_indexes)
            public_request, provider_request = route_generation_parameter_requests(
                tuple(profile for profile, _client in resolved_wires),
                request,
            )
            provider_request = provider_request.model_copy(
                update={"stream": True, "include_usage": True}
            )
            protocol_indexes: list[int] = []
            first_protocol_error: ProviderParameterError | ProviderCapabilityError | None = None
            for index, (deployment, (profile, _client)) in enumerate(
                zip(route.deployments, resolved_wires, strict=True)
            ):
                try:
                    preflight_gateway_request(
                        provider_request,
                        deployment.gateway.capabilities,
                        model_capabilities=deployment.capabilities,
                    )
                    dialect_stream_payload(profile, provider_request)
                except (ProviderParameterError, ProviderCapabilityError) as exc:
                    if first_protocol_error is None:
                        first_protocol_error = exc
                    continue
                protocol_indexes.append(index)
            if not protocol_indexes:
                if first_protocol_error is None:
                    raise GatewayRoutingError("authorized route has no compatible deployment")
                raise first_protocol_error
            if len(protocol_indexes) != len(route.deployments):
                selected_indexes = tuple(protocol_indexes)
                route = select_route_deployments(route, selected_indexes)
                resolved_wires = tuple(resolved_wires[index] for index in selected_indexes)
                public_request, provider_request = route_generation_parameter_requests(
                    tuple(profile for profile, _client in resolved_wires),
                    request,
                )
                provider_request = provider_request.model_copy(
                    update={"stream": True, "include_usage": True}
                )
            wire_route: list[JsonObject] = []
            signers: list[GatewayDispatchSigner | None] = []
            dispatch_bindings: list[FrozenDispatchBinding | None] = []
            for deployment, (profile, client) in zip(
                route.deployments, resolved_wires, strict=True
            ):
                require_gateway_provider(deployment.provider)
                preflight_gateway_request(
                    provider_request,
                    deployment.gateway.capabilities,
                    model_capabilities=deployment.capabilities,
                )
                upstream_payload = dialect_stream_payload(profile, provider_request)
                upstream_body, dispatch_signer = frozen_dispatch(profile, client, upstream_payload)
                wire_route.append(
                    deployment_wire_entry(
                        route,
                        deployment,
                        profile,
                        upstream_payload,
                        upstream_body,
                    )
                )
                signers.append(dispatch_signer)
                dispatch_bindings.append(
                    None
                    if dispatch_signer is None or upstream_body is None
                    else FrozenDispatchBinding(
                        url=profile.url,
                        body_sha256=sha256_bytes(upstream_body.encode("utf-8")),
                    )
                )
        except (ProviderParameterError, ProviderCapabilityError) as exc:
            # One shared normalizer keeps both pre-dispatch rejections
            # field-specific: the parameter path names the parameter and the
            # capability path names the capability, so a triager sees which
            # request feature the route cannot preserve.
            failure = normalized_provider_failure(exc)
            self._accounting.finish_request_quietly(authorization, failure)
            failure_param = (
                _public_capability_param(exc.capability, provider_request.surface)
                if isinstance(exc, ProviderCapabilityError)
                else exc.param
            )
            raise NativeBridgeError(public_failure_error(failure, param=failure_param)) from exc
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            error = _authority_error(exc)
            failure = GatewayFailure(
                failure_class=GatewayFailureClass.INTERNAL,
                safe_message="gateway admission failed before provider dispatch",
            )
            self._accounting.finish_request_quietly(authorization, failure)
            raise error from exc

        self._accounting.register(
            InflightRequest(
                authorization=authorization,
                route=route,
                request=provider_request,
                deadline_monotonic=deadline,
                continuation=continuation_context,
                policy=policy,
                signers=tuple(signers),
                dispatch_bindings=tuple(dispatch_bindings),
            )
        )
        response: JsonObject = {
            "request_id": authorization.request_id,
            "alias": authorization.alias,
            "alias_revision_id": authorization.alias_revision_id,
            "stream": request.stream,
            "include_usage": request.include_usage,
            "exact_model_id": route.snapshot.exact_model_id,
            "route_reason": route.route_reason,
            "route": wire_route,
            "ignored_parameters": list(public_request.ignored_parameters),
            "maximum_total_attempts": MAXIMUM_TOTAL_ATTEMPTS,
            "maximum_same_deployment_attempts": MAXIMUM_SAME_DEPLOYMENT_ATTEMPTS,
            "refusal_failover": authorization.refusal_failover,
            "output_guardrail": bool(policy is not None and policy.output_checks),
        }
        if request.surface == GatewayApiSurface.RESPONSES:
            response["surface"] = "responses"
            response["envelope"] = responses_envelope(public_request)
        return json.dumps(response, separators=(",", ":"))

    def sign_dispatch(self, argument: str) -> str:
        """Sign one frozen dispatch body immediately before the provider POST.

        The data plane calls this after it acquires its bounded dispatch
        permit and immediately before the open attempt reserved by
        ``start_attempt``, so queue time can never age a signature toward
        AWS's short clock window; a same-deployment redial or a failover
        advance is a fresh physical attempt through ``start_attempt``, so it
        always signs afresh too.

        Args:
            argument: JSON object with ``request_id``, the exact ``url``, and
                the exact frozen ``body`` string the data plane will send.

        Returns:
            JSON object with the ``headers`` to send verbatim.

        Raises:
            NativeBridgeError: The attempt is unknown, its route depth
                carries no signer, or credential resolution failed.
        """
        data = json.loads(argument)
        entry = self._accounting.entry(str(data.get("request_id") or ""))
        signer = None
        binding = None
        if entry is not None and entry.active_attempt_id is not None:
            depth = entry.attempt_depths.get(entry.active_attempt_id)
            if depth is not None and depth < len(entry.signers):
                signer = entry.signers[depth]
            if depth is not None and depth < len(entry.dispatch_bindings):
                binding = entry.dispatch_bindings[depth]
        try:
            url = str(data["url"])
            body = str(data["body"])
            if (
                binding is None
                or url != binding.url
                or sha256_bytes(body.encode("utf-8")) != binding.body_sha256
            ):
                raise public_failure_error(
                    GatewayFailure(
                        failure_class=GatewayFailureClass.INTERNAL,
                        safe_message=(
                            "gateway dispatch differs from the admitted destination or frozen body"
                        ),
                    )
                )
            headers = dispatch_signature_headers(
                signer,
                url=url,
                body=body,
            )
        except OpenAIProtocolError as exc:
            raise NativeBridgeError(exc) from exc
        return json.dumps({"headers": headers}, separators=(",", ":"))

    def start_attempt(self, argument: str) -> str:
        """Reserve one physical dispatch through the accounting registry.

        Args:
            argument: JSON object with ``request_id``, ``attempt_ordinal``,
                optional ``current_depth``, and the optional classified
                ``failure``; see
                :meth:`NativeAttemptAccounting.start_attempt`.

        Returns:
            The registry's reservation or exhaustion disposition.

        Raises:
            NativeBridgeError: The reservation failed; the request is
                finalized before the error is raised.
        """
        return self._accounting.start_attempt(argument)

    def settle(self, argument: str) -> str:
        """Durably settle one reserved attempt through the accounting registry.

        Args:
            argument: JSON settlement payload; see
                :meth:`NativeAttemptAccounting.settle`.

        Returns:
            An empty JSON object; repeated settlement is a no-op.

        Raises:
            NativeBridgeError: The durable terminal write failed; the entry
                is retained so a retried settlement can still land.
        """
        return self._accounting.settle(argument)

    def abandon(self, argument: str) -> str:
        """Terminalize one accepted request through the accounting registry.

        Args:
            argument: JSON object with ``request_id`` and optional
                ``failure``; see :meth:`NativeAttemptAccounting.abandon`.

        Returns:
            An empty JSON object; an unknown request is a no-op.

        Raises:
            NativeBridgeError: The durable terminal write failed; the entry
                is kept so the deadline sweep can still close it.
        """
        return self._accounting.abandon(argument)

    def enforce_output(self, argument: str) -> str:
        """Run one output-chain callback for a native buffered completion."""
        data = json.loads(argument)
        request_id = str(data.get("request_id") or "")
        entry = self._accounting.entry(request_id)
        policy = None if entry is None else entry.policy
        deadline = time.monotonic() if entry is None else entry.deadline_monotonic
        return enforce_native_output(
            self._guardrails,
            policy,
            argument,
            deadline_monotonic=deadline,
        )

    def claim_scope(self, argument: str) -> str:
        """Resolve the replay-store scope for one keyed request.

        The data plane owns the bounded in-process replay store; this call
        performs the decode and authorization once so the store key (tenant
        namespace, hashed caller operation, canonical request digest) is
        computed by exactly one implementation. The surface is part of the
        key, so keyed Chat Completions and keyed Responses operations never
        collide. A direct route whose provider has no native dialect is
        escalated before any replay claim, so an unservable caller operation
        never occupies the replay store; project targets resolve their
        deployment at admission through the same frozen selection, so their
        scope claims natively.

        Args:
            argument: JSON object with ``raw_key``, ``body``, optional
                ``surface`` (``"chat"`` or ``"responses"``, defaulting to
                chat), and optional ``idempotency_key`` and
                ``client_request_id`` header values.

        Returns:
            JSON replay scope with ``organization_id``, ``identity_id``,
            ``alias_revision_id``, ``surface``, ``caller_operation_sha256``,
            and ``canonical_request_sha256``, or an ``{"escalate": reason}``
            disposition naming why the native plane cannot serve the request.

        Raises:
            NativeBridgeError: Decoding or authorization failed.
        """
        data = json.loads(argument)
        decoded = self._decode_body(
            data["body"],
            surface=str(data.get("surface", "chat")),
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
        if isinstance(authorization.target, DirectTarget):
            try:
                route = self._components.routes.resolve_direct(authorization)
                resolve_route_profiles(self._components.runtime_catalogs, route)
            except NativeDialectUnavailableError as exc:
                return _escalation(str(exc))
            except Exception:  # noqa: BLE001 - the owner's admission records this failure.
                pass
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
        entry = self._accounting.entry(request_id)
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

    def _record_dead_admission_rungs(
        self,
        authorization: AuthorizationSnapshot,
        dead: tuple[DeadRung, ...],
        *,
        fallback_available: bool,
    ) -> None:
        """Feed each admission-dead rung into its health circuit and surface it.

        A rung skipped because it was operationally dead at admission is
        recorded in the same deployment health circuit as a runtime failure,
        so the existing cooldown and half-open probe bring it back
        automatically when it heals; it is never permanently blacklisted. A
        skipped lead rung is logged and counted so a persistently dead lead
        reaches a human instead of being silently masked behind a healthy
        fallback forever. That masking signal only exists when a live
        fallback actually serves: a total route outage escalates loudly on
        its own path and must not read as one more fallback-served request.

        Args:
            authorization: Frozen authority for the accepted request.
            dead: Every rung skipped as operationally dead, in route order.
            fallback_available: Whether any live rung remains to serve.
        """
        if not dead:
            return
        health = self._accounting.health
        for rung in dead:
            health.failed(deployment_health_key(authorization, rung.deployment), rung.failure)
        lead = next((rung for rung in dead if rung.index == 0), None)
        lead_masked = lead is not None and fallback_available
        self._accounting.record_admission_rung_skips(len(dead), lead_skipped=lead_masked)
        if lead is not None and fallback_available:
            _logger.warning(
                "gateway admission skipped the lead rung for alias %r: served off a "
                "fallback because deployment %r (provider %r) was dead at admission",
                authorization.alias,
                lead.deployment.deployment_id,
                lead.deployment.provider,
            )

    def _escalate_accepted(self, authorization: AuthorizationSnapshot, reason: str) -> str:
        """Finish one accepted-but-unservable request and return its disposition.

        The request was durably accepted before route probing, so the plane
        finalizes it content-free (no attempt row ever exists) before the
        escalation disposition tells the data plane to fail the request
        closed.

        Args:
            authorization: Frozen authority for the accepted request.
            reason: Display-safe reason the native path cannot serve it.

        Returns:
            The JSON admission body carrying the escalation disposition.
        """
        self._accounting.finish_request_quietly(
            authorization,
            GatewayFailure(
                failure_class=GatewayFailureClass.INTERNAL,
                safe_message="the native engine cannot serve the authorized route",
            ),
        )
        return _escalation(reason)

    def _resolve_route(
        self,
        authorization: AuthorizationSnapshot,
        request: GatewayRequest,
        *,
        continuation: ContinuationContext | None = None,
    ) -> GatewayRoute:
        """Resolve one direct or project route without an event loop.

        Direct pools resolve entirely inside frozen in-memory catalogs.
        Project targets run frozen learned selection synchronously on this
        worker thread through the shared selection seam and episode identity
        derivation, so there is exactly one policy execution path. A
        Responses continuation carries its original turn's episode key, so a
        continued request joins the same selection episode instead of
        re-running request-time embedding for a fresh one. Request-time
        embedding failure falls back to the frozen conservative baseline
        inside the shared runtime, and neither path mutates policy or
        evidence.
        """
        if isinstance(authorization.target, DirectTarget):
            return self._components.routes.resolve_direct(authorization)
        if continuation is not None:
            episode = (
                authorization.organization_id,
                authorization.identity_id,
                authorization.alias_revision_id,
                continuation.episode_key,
            )
        else:
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
