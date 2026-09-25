"""Python control plane for the native (Rust) gateway data plane.

Rust owns sockets, streaming and normalization. Python owns authorization,
payloads, continuations and ledger transactions. Boundaries use JSON.
Admission returns the certified route without starting an attempt; Rust reserves
each dispatch through ``start_attempt`` and records its durable terminal through
``settle``. Candidate selection, health circuits and budget skipping stay here.
Boundary errors raise :class:`NativeBridgeError`, whose ``public_error_json``
attribute carries the sanitized OpenAI-shaped error returned to the caller.
Requests the native path cannot serve (clients without a wire profile) return
with an ``{"escalate": reason}`` admission disposition after the accepted
request is finalized content-free; the data plane classifies the reason for
metrics and fails the request closed with the shared internal error.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.attempt_tokens import counted_input_tokens
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayApiSurface,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
)
from exp.runtime.gateway.group_commit import SyncGroupCommitLedger
from exp.runtime.gateway.guardrails import deterministic
from exp.runtime.gateway.guardrails.client import assert_not_internal_classification
from exp.runtime.gateway.guardrails.contracts import GuardrailRejected
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.native import (
    enforce_native_input,
    enforce_native_output,
    enforce_native_output_segment,
    native_output_mode,
)
from exp.runtime.gateway.model_chain_authority import authorize_serving_model_chains
from exp.runtime.gateway.native_accounting import (
    NativeAttemptAccounting,
    NativeBridgeError,
)
from exp.runtime.gateway.native_accounting import (
    authority_error as _authority_error,
)
from exp.runtime.gateway.native_admission import (
    admitted_route_requests,
    fold_parallel_tool_call_disclosures,
    log_reasoning_continuation_rejection,
    record_dead_admission_rungs,
    resolve_admission_route,
    select_single_route_before_search,
)
from exp.runtime.gateway.native_authentication import NativeAuthenticationMixin
from exp.runtime.gateway.native_batches import NativeBatchRelayMixin
from exp.runtime.gateway.native_bridge_errors import (
    escalation as _escalation,
)
from exp.runtime.gateway.native_bridge_errors import (
    ledger_capability_message,
)
from exp.runtime.gateway.native_bridge_errors import (
    public_capability_error as _public_capability_error,
)
from exp.runtime.gateway.native_capture import (
    CaptureController,
    begin_capture,
    select_capture_model,
)
from exp.runtime.gateway.native_components import NativeGatewayComponents, SyncWriteLedger
from exp.runtime.gateway.native_continuation import (
    continuation_binding_error as _continuation_binding_error,
)
from exp.runtime.gateway.native_continuation import (
    remember_continuation,
)
from exp.runtime.gateway.native_continuation import (
    require_bound_wire_authority as _require_bound_wire_authority,
)
from exp.runtime.gateway.native_continuation import (
    select_bound_continuation_route as _select_bound_continuation_route,
)
from exp.runtime.gateway.native_count_tokens import NativeCountTokensMixin
from exp.runtime.gateway.native_decisions import NativeDecisionsMixin
from exp.runtime.gateway.native_decode import NativeDecodeError, decode_native_body
from exp.runtime.gateway.native_dispatch_signing import NativeDispatchSigningMixin
from exp.runtime.gateway.native_embeddings import NativeEmbeddingsMixin
from exp.runtime.gateway.native_execution import (
    FrozenDispatchBinding,
    InflightRequest,
    NativeDialectUnavailableError,
    dispatchable_route_profiles,
    resolve_route_profiles,
    select_route_deployments,
)
from exp.runtime.gateway.native_images import NativeImagesMixin
from exp.runtime.gateway.native_observability import NativeObservabilityMixin
from exp.runtime.gateway.native_reasoning import (
    authenticate_reasoning_history,
    has_active_reasoning_content,
    rung_provider_request,
    seal_reasoning_carrier_content,
    strip_stale_reasoning_history,
    unseal_reasoning_history,
)
from exp.runtime.gateway.native_replay import replay_scope_payload
from exp.runtime.gateway.native_request_policy import require_route_authority
from exp.runtime.gateway.native_responses import (
    ContinuationContext,
    continuation_route_binding,
    continued_request,
    responses_envelope,
)
from exp.runtime.gateway.native_rung_policy import throttle_redial_budgets
from exp.runtime.gateway.native_rungs import build_rung_dispatch
from exp.runtime.gateway.native_settlement import (
    gateway_updating_failure,
    optional_text,
)
from exp.runtime.gateway.native_tool_search import NativeToolSearchMixin
from exp.runtime.gateway.reasoning_carrier import (
    ReasoningCarrierAuthority,
)
from exp.runtime.gateway.recovery import RecoveryHost
from exp.runtime.gateway.recovery_binding import validated_recovery_binding
from exp.runtime.gateway.request_policy import attempt_policy
from exp.runtime.gateway.reservation_tokenizer import reservation_encoder
from exp.runtime.gateway.routing import GatewayRoute, GatewayRoutingError
from exp.runtime.gateway.tool_search.plan import plan_tool_search
from exp.runtime.gateway.web_search.backend import WebSearchBackend, default_web_search_backend
from exp.runtime.gateway.web_search.plan import plan_web_search
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import (
    ProviderCapabilityError,
    ProviderParameterError,
    normalized_provider_failure,
)
from exp.runtime.models.providers.logprobs import require_unmodified_probability_output
from exp.runtime.models.providers.protocol import GatewayDispatchSigner, NativeWireClient
from exp.runtime.openai_protocol.errors import (
    OpenAIProtocolError,
    invalid_field,
    public_failure_error,
)
from exp.runtime.openai_protocol.requests import DecodedGatewayRequest
from exp.runtime.openai_protocol.state import BoundedContinuationStore

_logger = logging.getLogger(__name__)


_REQUEST_TIMEOUT_SECONDS = 120.0


class NativeControlPlane(
    NativeAuthenticationMixin,
    NativeBatchRelayMixin,
    NativeDispatchSigningMixin,
    NativeToolSearchMixin,
    NativeCountTokensMixin,
    NativeDecisionsMixin,
    NativeEmbeddingsMixin,
    NativeImagesMixin,
    NativeObservabilityMixin,
):
    """Authority and accounting callbacks for the native data plane.

    Rust worker threads share the group-commit writer and the locked in-flight
    registry. Opportunistic sweeps bound abandoned reservations to the request
    deadline plus the sweep grace.
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
        cache_sample_gate: Callable[[str], bool] | None = None,
        recovery_host: RecoveryHost | None = None,
        native_route_eligible: Callable[[GatewayRoute, GatewayRequest], bool] | None = None,
        guardrails: GuardrailEngine | None = None,
        capture: CaptureController | None = None,
        web_search: WebSearchBackend | None = None,
        default_lane_bound: int | None = None,
    ) -> None:
        """Bind loaded gateway components for serving.

        Args:
            components: Authority, ledger, routes, and runtime catalogs.
            request_timeout_seconds: Total per-request budget from admission.
            data_plane_metrics: Optional native metrics JSON supplier, typically
                ``exp_gateway_native.metrics_snapshot_json``; otherwise reports ``None``.
            continuation_store: Optional injected Responses continuation
                state; a host supplies its own bounded namespaced history,
                and the default is one in-process bounded store.
            readiness_probe: Optional hosted lifecycle readiness callback.
            usage_reporter: Optional hosted usage report callback.
            budget_error_factory: Optional hosted mapping for a rejected reservation.
            cache_sample_gate: Optional hosted predicate deciding whether one
                settled attempt (by its ledger attempt id) may feed the
                cache-priority EWMA; the host excludes promo-funded attempts
                so subsidized replay cannot buy fair-share weight. ``None``
                admits every sample; a raising gate skips the sample.
            native_route_eligible: Optional hosted policy for complete native semantics.
            guardrails: Optional identity-scoped engine. ``None`` leaves traffic unguarded.
            capture: Optional identity-scoped native capture controller.
            web_search: Gateway web-search backend; ``None`` binds Exa from ``EXA_API_KEY``.
            default_lane_bound: Per-worker in-flight cap for rungs that author
                no ``concurrency_bound`` (``lane_saturation.default_lane_bound``
                of the data plane's ``max_active_requests``); ``None`` leaves
                unauthored rungs unbounded, the historical behavior.
        """
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self._components = components
        self._capture = capture
        # The optional batch lane: hosts without it leave every batch route
        # answering the uniform not-enabled error below.
        self._batches = getattr(components, "batches", None)
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
        self._web_search = web_search if web_search is not None else default_web_search_backend()
        # Deterministic rules compile once here, never per request.
        self._guardrail_detectors = deterministic.compile_native_detectors(
            {} if guardrails is None else guardrails.deterministic_specifications
        )
        # Shared accounting owns reservations, health, recovery and deadline cleanup.
        self._accounting = NativeAttemptAccounting(
            self._write_ledger,
            budget_error_factory=budget_error_factory,
            cache_sample_gate=cache_sample_gate,
            recovery_host=recovery_host,
            default_lane_bound=default_lane_bound,
        )
        # Every reservation tokenizes its prompt; build the packaged BPE now so
        # a fresh process pays that once at bind time, never on its first
        # request, and a corrupt table fails startup with its own message.
        reservation_encoder()

    @property
    def guardrail_detectors(self) -> dict[str, deterministic.NativeDetector]:
        """Return the compiled deterministic rules the data plane enforces."""
        return dict(self._guardrail_detectors)

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

    def admit(self, argument: str) -> str:
        """Decode, authorize, inspect, route, and durably accept one request.

        Args:
            argument: JSON object with ``raw_key``, ``body`` (raw request
                body text), optional ``surface`` (``"chat"`` or
                ``"responses"``, defaulting to chat), and optional
                ``app_referer``/``app_title`` caller app identity.

        Returns:
            The ordered certified ``route`` with dispatch and retry configuration,
            or ``{"escalate": reason}`` after finalizing an accepted request that
            the native plane cannot serve without writing an attempt row.

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
            anthropic_beta=optional_text(data.get("anthropic_beta")),
        )
        request = decoded.request
        deadline = time.monotonic() + self._request_timeout_seconds
        try:
            # Freeze native app attribution and the trusted client IP onto caller authority.
            authorization = self._components.store.authorize_request(
                raw_key=data["raw_key"],
                alias=decoded.alias,
                request=request,
                deadline_monotonic=deadline,
                app_referer=optional_text(data.get("app_referer")),
                app_title=optional_text(data.get("app_title")),
                client_ip=optional_text(data.get("client_ip")),
            )
            authorization = authorize_serving_model_chains(self._components, authorization)
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            mapped = _authority_error(exc)
            pointer = self._batch_pointer_error(alias=decoded.alias, mapped=mapped)
            if pointer is not None:
                raise pointer from exc
            raise mapped from exc

        # Resolve continuation after authorization and before any durable acceptance.
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

        pinned_reasoning_route: GatewayRoute | None = None
        try:
            request, pinned_reasoning_route = authenticate_reasoning_history(
                self._components,
                authorization,
                request,
            )
        except ProviderParameterError as exc:
            raise NativeBridgeError(invalid_field(exc.param, str(exc))) from exc
        except Exception as exc:  # noqa: BLE001 - one public shape prevents an oracle.
            log_reasoning_continuation_rejection(authorization, "authenticate", exc)
            error = invalid_field(
                "messages.reasoning_content",
                "'messages.reasoning_content' must be an authentic continuation for this route.",
            )
            raise NativeBridgeError(error) from exc

        policy = None
        try:
            request, policy = enforce_native_input(
                self._guardrails,
                authorization=authorization,
                request=request,
                deadline_monotonic=deadline,
                detectors=self._guardrail_detectors,
            )
        except GuardrailRejected as exc:
            raise NativeBridgeError(public_failure_error(exc.failure)) from exc
        captured_request = request
        retention_request = strip_stale_reasoning_history(request)
        try:
            request, verified_reasoning_route = unseal_reasoning_history(
                self._components,
                authorization,
                request,
            )
        except ProviderParameterError as exc:
            raise NativeBridgeError(invalid_field(exc.param, str(exc))) from exc
        except Exception as exc:  # noqa: BLE001 - one public shape prevents an oracle.
            log_reasoning_continuation_rejection(authorization, "unseal", exc)
            error = invalid_field(
                "messages.reasoning_content",
                "'messages.reasoning_content' must be an authentic continuation for this route.",
            )
            raise NativeBridgeError(error) from exc
        if (
            pinned_reasoning_route is not None
            and verified_reasoning_route is not None
            and pinned_reasoning_route.deployment != verified_reasoning_route.deployment
        ):
            log_reasoning_continuation_rejection(
                authorization, "route_pin", "authenticate and unseal resolved different deployments"
            )
            raise NativeBridgeError(
                invalid_field(
                    "messages.reasoning_content",
                    "'messages.reasoning_content' must be an authentic continuation "
                    "for this route.",
                )
            )
        pinned_reasoning_route = verified_reasoning_route
        request = strip_stale_reasoning_history(request)
        if pinned_reasoning_route is not None and not has_active_reasoning_content(request):
            pinned_reasoning_route = None
        if continuation_context is not None:
            # Execution receives authenticated plaintext, but the bounded
            # continuation store keeps the post-guardrail history sealed.
            continuation_context.messages = retention_request.messages

        # The ledger accepts the logical request before route selection, so a
        # keyed operation whose durable terminal already exists (or whose key
        # was reused with different content) fails closed here, before
        # learned selection can run request-time embedding or any other
        # provider-touching work.
        try:
            self._write_ledger.accept_request(authorization=authorization)
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise _authority_error(exc) from exc

        if not begin_capture(
            self._capture,
            authorization,
            captured_request,
            session_id=optional_text(data.get("capture_session_id")),
        ):
            message = "Traffic capture is unavailable or at capacity. Restore capacity and retry."
            self._accounting.finish_request_quietly(
                authorization,
                GatewayFailure(failure_class=GatewayFailureClass.UNAVAILABLE, safe_message=message),
            )
            raise NativeBridgeError(
                OpenAIProtocolError(status_code=503, code="capture_unavailable", message=message)
            )
        # Escalation finishes the accepted request quietly before returning, so it is
        # accounted content-free and never billed. Routing failures found by
        # the probe are raised against the accepted request below.
        probe_failure: Exception | None = None
        web_search_admission: JsonObject | None = None
        tool_search_admission: JsonObject | None = None
        tool_search_state = None
        route: GatewayRoute | None = None
        resolved_wires: tuple[tuple[GatewayWireProfile, NativeWireClient], ...] | None = None
        try:
            route = pinned_reasoning_route or self._resolve_route(
                authorization,
                request,
                continuation=continuation_context,
            )
            require_route_authority(authorization, request, route)
            route = _select_bound_continuation_route(
                route,
                None
                if continuation_context is None
                else continuation_context.required_route_binding,
            )
            # A rung that is dead at admission (a lost credential, a drifted
            # connection) is skipped so a live fallback still serves the
            # request instead of the whole request failing on a dead lead.
            dispatchable = dispatchable_route_profiles(self._components.runtime_catalogs, route)
            record_dead_admission_rungs(
                self._accounting,
                authorization,
                dispatchable.dead,
                fallback_available=bool(dispatchable.indexes),
            )
            if not dispatchable.indexes:
                if (
                    continuation_context is not None
                    and continuation_context.required_route_binding is not None
                ):
                    raise _continuation_binding_error()
                # No dispatchable rung remains; finalize the accepted request closed.
                return self._escalate_accepted(
                    authorization,
                    "every certified deployment was unavailable at admission",
                )
            route = select_route_deployments(route, dispatchable.indexes)
            resolved_wires = dispatchable.resolved_wires
            route, resolved_wires, selected_placement = select_single_route_before_search(
                route,
                resolved_wires,
                request,
                accounting=self._accounting,
                authorization=authorization,
                continuation=continuation_context,
            )
            searched = plan_web_search(
                request,
                [profile.dialect for profile, _client in resolved_wires],
                self._web_search,
                deadline_monotonic=deadline,
            )
            request, web_search_admission = searched.request, searched.admission
            # Caller-declared tool search on a route with no native one (tool_search.plan).
            planned = plan_tool_search(request, [p.dialect for p, _c in resolved_wires])
            request, tool_search_state = planned.request, planned.state
            tool_search_admission = planned.admission
            _require_bound_wire_authority(
                None
                if continuation_context is None
                else continuation_context.required_route_binding,
                route,
                resolved_wires,
            )
        except NativeDialectUnavailableError as exc:
            return self._escalate_accepted(authorization, str(exc))
        except OpenAIProtocolError as exc:
            # A continuation whose bound provider authority is no longer
            # available is a CLIENT error (400 previous_response_not_found: resend
            # the full conversation), not a gateway-internal fault. Class the
            # durable failure by the public status so usage and health read it
            # as a client failure and it never pages as internal; the caller
            # still receives the exact public error unchanged.
            self._accounting.finish_request_quietly(
                authorization,
                GatewayFailure(
                    failure_class=(
                        GatewayFailureClass.INTERNAL
                        if exc.status_code >= 500
                        else GatewayFailureClass.INVALID_REQUEST
                    ),
                    safe_message=exc.detail.message,
                ),
            )
            raise NativeBridgeError(exc) from exc
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
            route, resolved_wires, public_request, provider_request, placement = (
                admitted_route_requests(
                    route,
                    resolved_wires,
                    request,
                    accounting=self._accounting,
                    authorization=authorization,
                    continuation=continuation_context,
                )
            )
            placement = selected_placement or placement
            require_route_authority(authorization, request, route)
            require_unmodified_probability_output(request, bool(policy and policy.output_checks))
            wire_route: list[JsonObject] = []
            parallel_disclosures: set[str] = set()
            output_bounds: list[int] = []
            signers: list[GatewayDispatchSigner | None] = []
            dispatch_bindings: list[FrozenDispatchBinding | None] = []
            carrier_authorities: list[ReasoningCarrierAuthority | None] = []
            # How long a throttle is worth waiting on per rung for THIS
            # request (the pool's schedule scaled by the cache at stake),
            # decided here so the data plane never waits on a rung whose
            # throttle should fail over cold at once. `route` is the admitted
            # route (dead and incompatible rungs already removed), so its last
            # rung is the one with no cold alternative; the sticky binding is
            # the one placement already read.
            redial_budgets = throttle_redial_budgets(
                self._accounting.loads,
                route,
                authorization.organization_id,
                sticky_deployment_id=placement.sticky_deployment_id,
            )
            for deployment, (profile, client), budget in zip(
                route.deployments, resolved_wires, redial_budgets, strict=True
            ):
                # A reasoning-pinned route's fallback rung is frozen WITHOUT
                # the pinned provider's sealed reasoning (it cannot unseal
                # it), so a failover past the issuing rung dispatches the
                # conversation minus that turn's thinking, never a foreign
                # sealed block.
                dispatch = build_rung_dispatch(
                    route,
                    deployment,
                    profile,
                    client,
                    provider_request=rung_provider_request(route, deployment, provider_request),
                    public_request=public_request,
                    authorization=authorization,
                    throttle_redial_budget=budget,
                )
                if dispatch.parallel_disclosure is not None:
                    parallel_disclosures.add(dispatch.parallel_disclosure)
                if dispatch.output_disclosure is not None:
                    parallel_disclosures.add(dispatch.output_disclosure)
                output_bounds.append(dispatch.reserved_output_tokens)
                wire_route.append(dispatch.wire_entry)
                signers.append(dispatch.signer)
                dispatch_bindings.append(dispatch.binding)
                carrier_authorities.append(dispatch.carrier_authority)
            public_request = fold_parallel_tool_call_disclosures(
                public_request,
                parallel_disclosures,
                accounting=self._accounting,
                authorization=authorization,
            )
            if continuation_context is not None:
                continuation_context.route_bindings = tuple(
                    continuation_route_binding(deployment, profile)
                    for deployment, (profile, _client) in zip(
                        route.deployments,
                        resolved_wires,
                        strict=True,
                    )
                )
        except NativeBridgeError:
            # The enriched fail-closed capability rejection above already
            # finished the accepted request; let it cross the boundary as-is.
            raise
        except (ProviderParameterError, ProviderCapabilityError) as exc:
            # One shared normalizer keeps both pre-dispatch rejections
            # field-specific: the parameter path names the parameter and the
            # capability path names the capability, so a triager sees which
            # request feature the route cannot preserve.
            failure = normalized_provider_failure(exc)
            if isinstance(exc, ProviderCapabilityError):
                public_error = _public_capability_error(
                    exc,
                    provider_request.surface,
                    public_stream=public_request.stream,
                    public_tools=bool(public_request.tools),
                    developer_messages_param=decoded.developer_messages_param,
                )
                # The ledger keeps the capability-free generic sentence, but a
                # bare "cannot preserve a requested capability" is untriageable
                # from an alert. Append the PUBLIC field the caller was told
                # about (never the internal literal), so operators read
                # "(field: stop)" without opening the request.
                failure = failure.model_copy(
                    update={
                        "safe_message": ledger_capability_message(
                            failure.safe_message, public_error.detail.param
                        )
                    }
                )
            else:
                public_error = public_failure_error(failure, param=exc.param)
            self._accounting.finish_request_quietly(authorization, failure)
            raise NativeBridgeError(public_error) from exc
        except GatewayRoutingError as exc:
            # A route/catalog that cannot be built during a rolling deploy is a
            # transient control-plane condition, not a bug: record it retryable
            # so it never pages as INTERNAL. The public error is already a 503.
            failure = gateway_updating_failure()
            self._accounting.finish_request_quietly(authorization, failure)
            raise _authority_error(exc) from exc
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            error = _authority_error(exc)
            failure = GatewayFailure(
                failure_class=GatewayFailureClass.INTERNAL,
                safe_message="gateway admission failed before provider dispatch",
            )
            # The public error and the ledger row carry only the sanitized
            # text, so this record is the ONLY place the real exception
            # survives: an unlogged INTERNAL here left a granted alias failing
            # 500 for hours with nothing to diagnose (platform staging,
            # 2026-09-03). The message names the request and alias; the
            # traceback rides exc_info. Nothing here carries a credential.
            _logger.exception(
                "gateway admission failed before provider dispatch",
                extra={
                    "operation": "native_admit",
                    "request_id": authorization.request_id,
                    "alias": authorization.alias,
                    "alias_revision_id": authorization.alias_revision_id,
                    "exception_type": type(exc).__name__,
                },
            )
            self._accounting.finish_request_quietly(authorization, failure)
            raise error from exc

        plan = deterministic.native_output_plan(policy, self._guardrail_detectors)
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
                reasoning_carrier_authorities=tuple(carrier_authorities),
                tier_forwarded_by_depth=tuple(
                    profile.forwards_tier(provider_request.service_tier)
                    for profile, _client in resolved_wires
                ),
                reserved_output_tokens_by_depth=tuple(output_bounds),
                affinity_fingerprint=placement.fingerprint,
                verified_warm_deployment_id=placement.verified_warm_deployment_id,
                verified_warm_until_monotonic=placement.verified_warm_until_monotonic,
                recovery_scoped=placement.recovery_scoped,
                sticky_preferred=placement.sticky_preferred,
                throttle_redial_budgets=redial_budgets,
                recovery_reason=placement.recovery_reason,
                recovery_bindings={
                    deployment.deployment_id: binding
                    for deployment, (profile, _) in zip(
                        route.deployments, resolved_wires, strict=True
                    )
                    if (
                        binding := validated_recovery_binding(
                            deployment, profile, authorization.organization_id
                        )
                    )
                    is not None
                },
                resolved_wires=None if tool_search_state is None else tuple(resolved_wires),
                public_request=None if tool_search_state is None else public_request,
                tool_search=tool_search_state,
            )
        )
        select_capture_model(self._capture, authorization.request_id, route.snapshot.exact_model_id)
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
            **attempt_policy(request.gateway).model_dump(mode="json"),
            "refusal_failover": authorization.refusal_failover,
            "output_guardrail": native_output_mode(
                self._guardrails,
                policy,
                public_request,
                image_output=any(wire.get("image_output") is True for wire in wire_route),
            ).value,
            "caller_scope": f"{authorization.organization_id}:{authorization.identity_id}",
        }
        if route.snapshot.throttle_redial is not None:
            # The pool's frozen backoff-and-redial schedule; absent (not null) on
            # pools that keep throttles failover-only: their admission is byte-identical.
            response["throttle_redial"] = route.snapshot.throttle_redial.model_dump(mode="json")
        if plan is not None:
            response["guardrail_output_plan"] = plan
        if web_search_admission is not None:
            # The gateway searched: the data plane cites the sources and counts it.
            response["web_search"] = web_search_admission
        if tool_search_admission is not None:
            response["tool_search"] = tool_search_admission
        if request.surface == GatewayApiSurface.MESSAGES:
            # Display-only: what `message_start` shows as input when the
            # upstream reports nothing before its final chunk. The ledger
            # never reads it; settlement keeps the provider's meters.
            response["input_token_estimate"] = counted_input_tokens(public_request)
        if public_request.maximum_output_tokens is not None:
            # The caller's cap: classifies an output-less, usage-less `stop` (capped -> length).
            response["maximum_output_tokens"] = public_request.maximum_output_tokens
        if request.surface == GatewayApiSurface.RESPONSES:
            response["surface"] = "responses"
            response["envelope"] = responses_envelope(public_request)
        return json.dumps(response, separators=(",", ":"))

    def start_attempt(self, argument: str) -> str:
        """Reserve one physical dispatch through the accounting registry.

        Args:
            argument: JSON payload for ``NativeAttemptAccounting.start_attempt``.

        Returns:
            The registry's reservation or exhaustion disposition.

        Raises:
            NativeBridgeError: Reservation failed after finalizing the request.
        """
        return self._accounting.start_attempt(argument)

    def settle(self, argument: str) -> str:
        """Durably settle one reserved attempt through the accounting registry.

        Args:
            argument: JSON payload for ``NativeAttemptAccounting.settle``.

        Returns:
            An empty JSON object; repeated settlement is a no-op.

        Raises:
            NativeBridgeError: Write failed; the entry remains available for retry.
        """
        return self._accounting.settle(argument)

    def abandon(self, argument: str) -> str:
        """Terminalize one accepted request through the accounting registry.

        Args:
            argument: JSON payload for ``NativeAttemptAccounting.abandon``.

        Returns:
            An empty JSON object; an unknown request is a no-op.

        Raises:
            NativeBridgeError: Write failed; the entry remains for the deadline sweep.
        """
        return self._accounting.abandon(argument)

    def enforce_output_segment(self, argument: str) -> str:
        """Release the settled part of one streamed ``stream`` mode tail."""
        entry = self._accounting.entry(str(json.loads(argument).get("request_id") or ""))
        policy = None if entry is None else entry.policy
        deadline = time.monotonic() if entry is None else entry.deadline_monotonic
        return enforce_native_output_segment(
            self._guardrails, policy, argument, deadline_monotonic=deadline
        )

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

    def seal_reasoning_content(self, argument: str) -> str:
        """Seal one winning Fireworks turn before terminal settlement."""
        return seal_reasoning_carrier_content(self._accounting, argument)

    def claim_scope(self, argument: str) -> str:
        """Resolve the replay-store scope for one keyed request.

        Decode and authorize once to scope replay by tenant, surface, operation,
        and canonical request digest. Unsupported direct routes escalate before
        any claim; project targets use the frozen admission deployment selection.

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
        # Only the standard Idempotency-Key names a retriable operation;
        # client_request_id is a session correlation id real callers reuse
        # across distinct requests, so it never keys replay.
        caller_operation = request.idempotency_key
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
            authorization = authorize_serving_model_chains(self._components, authorization)
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
        return replay_scope_payload(authorization, request)

    def remember(self, argument: str) -> str:
        """Retain one finished Responses continuation within strict bounds.

        Args:
            argument: JSON object with ``request_id``, aggregated ``text``,
                ``refusal`` presence, and completed ``tool_calls``; an
                output-less turn carries all of them empty and is retained as
                the conversation so far.

        Returns:
            An empty JSON object; retention that does not apply (a
            ``store: false`` caller, a refusal) is a no-op.

        Raises:
            NativeBridgeError: The continuation exceeds the bounded store or
                a completed tool call carried malformed fields.
        """
        try:
            return remember_continuation(self._accounting, self._continuations, argument)
        except OpenAIProtocolError as exc:
            raise NativeBridgeError(exc) from exc
        except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
            raise _authority_error(exc) from exc

    def _decode_body(
        self,
        body: str,
        *,
        surface: str = "chat",
        idempotency_key: str | None = None,
        client_request_id: str | None = None,
        anthropic_beta: str | None = None,
    ) -> DecodedGatewayRequest:
        """Decode one raw request body with the shared surface decoder."""
        try:
            return decode_native_body(
                body,
                surface=surface,
                idempotency_key=idempotency_key,
                client_request_id=client_request_id,
                anthropic_beta=anthropic_beta,
            )
        except NativeDecodeError as exc:
            raise NativeBridgeError(exc.error) from exc

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
        """Resolve one direct or project route; see ``resolve_admission_route``."""
        return resolve_admission_route(
            self._components, authorization, request, continuation=continuation
        )
