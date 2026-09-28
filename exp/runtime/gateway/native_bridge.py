"""Python control plane for the native (Rust) gateway data plane.

Rust owns sockets, streaming and normalization. Python owns authorization,
payloads, continuations and ledger transactions. Boundaries use JSON.
Admission returns the certified route without starting an attempt; Rust reserves
each dispatch through ``start_attempt`` and records its durable terminal through
``settle``. Eligible unkeyed direct Chat requests queue durable acceptance while
route assembly runs, then wait for the commit before returning; keyed and complex
requests retain early acceptance. Candidate selection, health circuits and budget
skipping stay here.
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
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
)
from exp.runtime.gateway.explicit_cache import ExplicitCacheHost
from exp.runtime.gateway.group_commit import SyncGroupCommitLedger
from exp.runtime.gateway.guardrails import deterministic
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.native import (
    enforce_native_output,
    enforce_native_output_segment,
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
    resolve_admission_route,
)
from exp.runtime.gateway.native_authentication import NativeAuthenticationMixin
from exp.runtime.gateway.native_batches import NativeBatchRelayMixin
from exp.runtime.gateway.native_bridge_admission import NativeAdmissionMixin
from exp.runtime.gateway.native_bridge_errors import (
    escalation as _escalation,
)
from exp.runtime.gateway.native_capture import (
    CaptureController,
)
from exp.runtime.gateway.native_components import NativeGatewayComponents, SyncWriteLedger
from exp.runtime.gateway.native_continuation import (
    remember_continuation,
)
from exp.runtime.gateway.native_count_tokens import NativeCountTokensMixin
from exp.runtime.gateway.native_decisions import NativeDecisionsMixin
from exp.runtime.gateway.native_decode_boundary import NativeDecodeMixin
from exp.runtime.gateway.native_dispatch_signing import NativeDispatchSigningMixin
from exp.runtime.gateway.native_embeddings import NativeEmbeddingsMixin
from exp.runtime.gateway.native_execution import (
    NativeDialectUnavailableError,
    resolve_route_profiles,
)
from exp.runtime.gateway.native_explicit_cache import (
    NativeExplicitCacheMixin,
    validate_explicit_cache_host,
)
from exp.runtime.gateway.native_images import NativeImagesMixin
from exp.runtime.gateway.native_observability import (
    ControlPlaneTimingDiagnostics,
    NativeObservabilityMixin,
)
from exp.runtime.gateway.native_reasoning import (
    seal_reasoning_carrier_content,
)
from exp.runtime.gateway.native_replay import replay_scope_payload
from exp.runtime.gateway.native_responses import (
    ContinuationContext,
)
from exp.runtime.gateway.native_settlement import (
    optional_text,
)
from exp.runtime.gateway.native_tool_search import NativeToolSearchMixin
from exp.runtime.gateway.recovery import RecoveryHost
from exp.runtime.gateway.reservation_tokenizer import reservation_encoder
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.gateway.web_search.backend import WebSearchBackend, default_web_search_backend
from exp.runtime.openai_protocol.errors import (
    OpenAIProtocolError,
)
from exp.runtime.openai_protocol.state import BoundedContinuationStore

_logger = logging.getLogger(__name__)


_REQUEST_TIMEOUT_SECONDS = 120.0


class NativeControlPlane(
    NativeAdmissionMixin,
    NativeAuthenticationMixin,
    NativeDecodeMixin,
    NativeExplicitCacheMixin,
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
        explicit_cache: ExplicitCacheHost | None = None,
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
            explicit_cache: Durable host policy and resource accounting for marked Google
                prefixes. None disables explicit cache operations; generation is unchanged.
        """
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self._components = components
        self._explicit_cache = validate_explicit_cache_host(explicit_cache)
        self._control_plane_timing = ControlPlaneTimingDiagnostics()
        self._accounting_timing = ControlPlaneTimingDiagnostics(
            (
                "attempt_preparation_ms",
                "attempt_writer_ms",
                "attempt_postprocess_ms",
                "settlement_preparation_ms",
                "settlement_writer_ms",
                "settlement_postprocess_ms",
            )
        )
        set_authority_timing = getattr(
            components.store, "set_request_authority_timing_recorder", None
        )
        if callable(set_authority_timing):
            set_authority_timing(self._control_plane_timing.record)
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
            timing_recorder=self._accounting_timing.record,
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
