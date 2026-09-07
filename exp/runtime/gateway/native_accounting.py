"""Durable attempt accounting for the native data plane's waterfall.

One registry owns every admitted request from acceptance to its terminal
settlement: it reserves each physical dispatch (``start_attempt``), lands each
attempt's durable terminal (``settle``), terminalizes requests with no
settleable outcome (``abandon``), and sweeps retained settlements and
abandoned reservations on a timer. Candidate selection enforces the frozen
waterfall policy, including deployment-health circuits and per-deployment
budget skipping. Every method takes and returns one JSON
string, matching the bridge boundary.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable

from exp.common.core.artifacts import JsonObject
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.attempt_tokens import worst_case_attempt_tokens
from exp.runtime.gateway.boundary import boundary_protocol_error
from exp.runtime.gateway.budgets import (
    BudgetReservationRejected,
    BudgetScopeKind,
    maximum_attempt_cost_micro_usd,
)
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayUsage,
)
from exp.runtime.gateway.health import DeploymentHealthRegistry
from exp.runtime.gateway.native_components import SyncWriteLedger
from exp.runtime.gateway.native_execution import (
    DeadRung,
    InflightRequest,
    claim_route_from,
    deployment_health_key,
    deployment_priced_for_service_tier,
    dispatch_disclosure,
    next_route_candidate,
    rung_load_key,
)
from exp.runtime.gateway.native_settlement import (
    all_routes_throttled_failure,
    all_routes_unavailable_failure,
    budget_quota_failure,
    budget_quota_protocol_error,
    failure_from_boundary_payload,
    first_token_at_from_settlement,
    ledger_failure,
    settlement_rate_limit,
    terminal_from_settlement,
)
from exp.runtime.gateway.rate_limit_headers import RateLimitObservation
from exp.runtime.gateway.rung_admission import RungLoadRegistry, RungShed
from exp.runtime.gateway.sticky_affinity import StickySpillRegistry
from exp.runtime.openai_protocol.errors import (
    OpenAIProtocolError,
    public_failure_error,
)

_SWEEP_GRACE_SECONDS = 5.0
_SWEEP_INTERVAL_SECONDS = 5.0
_SWEEP_BATCH = 16
_logger = logging.getLogger(__name__)


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


def authority_error(exception: Exception) -> NativeBridgeError:
    """Map boundary failures through the shared service-layer mapper.

    Args:
        exception: Store, grant, routing, or execution failure.

    Returns:
        A boundary error carrying the matching public OpenAI error.
    """
    return NativeBridgeError(boundary_protocol_error(exception))


def internal_protocol_error() -> OpenAIProtocolError:
    """Return the public internal error for a broken data-plane wire contract."""
    return OpenAIProtocolError(
        status_code=500,
        code="internal_error",
        message="The gateway request failed.",
        error_type="api_error",
    )


class NativeAttemptAccounting:
    """Registry of admitted requests and their durable attempt settlements.

    Methods are called from multiple Rust worker threads; the registry is
    guarded by one lock and swept both opportunistically and on a timer so an
    abandoned reservation cannot outlive its request deadline by more than
    the sweep grace. A lost durable terminal write latches the registry
    unhealthy so readiness fails until the next startup reconciliation.
    """

    def __init__(
        self,
        write_ledger: SyncWriteLedger,
        *,
        budget_error_factory: Callable[[str], NativeBridgeError] | None = None,
    ) -> None:
        """Bind the durable ledger and start the settlement sweep.

        Args:
            write_ledger: Blocking durable request and attempt ledger.
            budget_error_factory: Optional hosted mapping for a rejected
                reservation.
        """
        self._write_ledger = write_ledger
        self._budget_error_factory = budget_error_factory
        # The native waterfall's deployment-health circuits, revision-scoped
        # to the traffic this plane serves.
        self._health = DeploymentHealthRegistry()
        # Per-worker in-flight counters and rate windows for rungs that author
        # a dispatch policy (concurrency bound, rate caps, weighted fair
        # share); pure in-memory arithmetic, physical-lane scoped so counters
        # survive catalog rolls.
        self._loads = RungLoadRegistry()
        # Worker-local conversation-to-rung bindings under
        # maximize_cache_affinity, so a spilled conversation keeps serving off
        # the rung holding its warm cache instead of bouncing back.
        self._sticky = StickySpillRegistry()
        self._inflight: dict[str, InflightRequest] = {}
        self._lock = threading.Lock()
        self._accounting_healthy = True
        self._sweep_retained_replayed = 0
        self._sweep_abandoned_cancelled = 0
        # Admission-time dead-rung skips: a request served off a fallback
        # because a certified rung could not be resolved for dispatch, and the
        # subset of those where the skipped rung was the lead.
        self._admission_dead_rungs_skipped = 0
        self._admission_parameter_coercions = 0
        self._admission_lead_rungs_skipped = 0
        # Dispatch-policy outcomes: rungs bypassed at their bound, rate
        # window, fresh-session threshold, or their organization's fair
        # share, and dispatches forced past a bound because no other rung
        # could serve. The per-reason counters split the aggregate.
        self._rung_admission_sheds = 0
        self._rung_saturated_overflows = 0
        self._rung_rate_limit_sheds = 0
        self._rung_fresh_session_spills = 0
        # The sweep also runs on a timer so retained settlements and abandoned
        # attempts are recovered even when no further requests arrive.
        self._sweeper = threading.Thread(
            target=self._sweep_loop,
            name="exp-native-settlement-sweep",
            daemon=True,
        )
        self._sweeper.start()

    @property
    def accounting_healthy(self) -> bool:
        """Return whether every durable terminal write has landed."""
        return self._accounting_healthy

    @property
    def health(self) -> DeploymentHealthRegistry:
        """Return the native waterfall's deployment-health circuits."""
        return self._health

    @property
    def loads(self) -> RungLoadRegistry:
        """Return the per-worker rung in-flight registry for bounded admission."""
        return self._loads

    @property
    def sticky(self) -> StickySpillRegistry:
        """Return the worker-local sticky conversation-to-rung bindings."""
        return self._sticky

    def _reserve_rung_slot(
        self,
        entry: InflightRequest,
        deployment: ExactModelDeployment,
        *,
        reserved_tokens: int,
        force: bool,
    ) -> str | RungShed | None:
        """Reserve one policy-bounded slot on a rung, or report the shed.

        Args:
            entry: The owning in-flight request (organization and weight).
            deployment: The claimed rung about to dispatch.
            reserved_tokens: Worst-case tokens this dispatch reserves, counted
                against the rung's token window when one is authored.
            force: Admit past every policy limit because no other rung can
                serve.

        Returns:
            An opaque reservation ticket, the shed disclosure, or ``None``
            when the rung authors no admission policy (the untouched default).
        """
        policy = deployment.gateway.dispatch
        if policy is None or (
            policy.concurrency_bound is None
            and policy.requests_per_minute is None
            and policy.tokens_per_minute is None
        ):
            return None
        # Warm standing: the request's affinity fingerprint holds a live
        # sticky binding on THIS rung, so its provider cache lives here and
        # the fresh-session early threshold does not apply to it. The early
        # threshold only exists on affinity pools AND for requests that carry
        # a fingerprint (chat/Responses admission): a surface with no session
        # concept (embeddings, images) must never be classed fresh wholesale.
        fresh_fraction = (
            policy.fresh_session_spill_fraction
            if entry.route.snapshot.failover_mode == "maximize_cache_affinity"
            and entry.affinity_fingerprint is not None
            else None
        )
        warm_session = True
        if fresh_fraction is not None and entry.affinity_fingerprint is not None:
            warm_session = (
                self._sticky.bound_deployment(entry.affinity_fingerprint)
                == deployment.deployment_id
            )
        result = self._loads.reserve(
            (deployment.deployment_id, deployment.connection_sha256),
            organization_id=entry.authorization.organization_id,
            weight=entry.authorization.fair_share_weight,
            bound=policy.concurrency_bound,
            fair_share=policy.fair_share,
            requests_per_minute=policy.requests_per_minute,
            tokens_per_minute=policy.tokens_per_minute,
            cache_priority_alpha=policy.cache_priority_alpha,
            reserved_tokens=reserved_tokens if policy.tokens_per_minute is not None else 0,
            warm_session=warm_session,
            fresh_spill_fraction=fresh_fraction,
            force=force,
        )
        if isinstance(result, RungShed):
            with self._lock:
                self._rung_admission_sheds += 1
                if result.reason == "rate_limit":
                    self._rung_rate_limit_sheds += 1
                elif result.reason == "fresh_session_spill":
                    self._rung_fresh_session_spills += 1
            if result.reason == "rate_limit":
                _logger.debug(
                    "gateway rate-limit shed on deployment %r (learned ceiling %s/min)",
                    deployment.deployment_id,
                    result.learned_requests_per_minute,
                )
        return result

    def rung_admission_counters(self) -> tuple[int, int]:
        """Return ``(sheds, saturated_overflows)`` for the metrics snapshot."""
        with self._lock:
            return (self._rung_admission_sheds, self._rung_saturated_overflows)

    def rung_rate_counters(self) -> tuple[int, int]:
        """Return ``(rate_limit_sheds, fresh_session_spills)`` for metrics."""
        with self._lock:
            return (self._rung_rate_limit_sheds, self._rung_fresh_session_spills)

    def mark_unhealthy(self) -> None:
        """Latch an unhealthy state after a lost terminal accounting write."""
        self._accounting_healthy = False

    def counters(self) -> tuple[int, int, int]:
        """Return sweep recoveries and registry size for the metrics snapshot."""
        with self._lock:
            return (
                self._sweep_retained_replayed,
                self._sweep_abandoned_cancelled,
                len(self._inflight),
            )

    def record_admission_coercions(self, count: int) -> None:
        """Count disclosed request coercions applied at admission.

        Args:
            count: Number of disclosed substitutions on one admission.
        """
        with self._lock:
            self._admission_parameter_coercions += count

    def admission_parameter_coercions(self) -> int:
        """Return the total disclosed request coercions for metrics."""
        with self._lock:
            return self._admission_parameter_coercions

    def record_admission_rung_skips(self, dead_count: int, *, lead_skipped: bool) -> None:
        """Count admission-time dead-rung skips for the metrics snapshot.

        Args:
            dead_count: Number of certified rungs skipped as dead at admission.
            lead_skipped: Whether the skipped set included the lead rung.
        """
        with self._lock:
            self._admission_dead_rungs_skipped += dead_count
            if lead_skipped:
                self._admission_lead_rungs_skipped += 1

    def admission_rung_skips(self) -> tuple[int, int]:
        """Return ``(lead_rungs_skipped, dead_rungs_skipped)`` for metrics."""
        with self._lock:
            return (self._admission_lead_rungs_skipped, self._admission_dead_rungs_skipped)

    def register(self, entry: InflightRequest) -> None:
        """Track one accepted request until its terminal settlement."""
        with self._lock:
            self._inflight[entry.authorization.request_id] = entry

    def entry(self, request_id: str) -> InflightRequest | None:
        """Return one tracked request, or ``None`` after settlement."""
        with self._lock:
            return self._inflight.get(request_id)

    def start_attempt(self, argument: str) -> str:
        """Reserve one physical dispatch immediately before network work.

        Candidate selection mirrors the executor: the first dispatch claims
        the first healthy route in authored order (with bounded last-resort
        and forced claims through open circuits), a classified failure either
        redials the same deployment or advances to the next claimable one,
        and a deployment whose hard monthly allocation cannot admit this call
        is skipped. Exhaustion finalizes the accepted request here, so the
        data plane only has to answer with the last classified failure.

        Args:
            argument: JSON object with ``request_id``, ``attempt_ordinal``
                (the count of physical dispatches already reserved), optional
                ``current_depth`` (the route position of the failed dispatch,
                absent for the first), and the optional classified
                ``failure`` with its ``retryable_same_deployment`` and
                ``failover_eligible`` flags.

        Returns:
            ``{"attempt_id", "route_depth"}`` for one durably reserved
            dispatch, or ``{"exhausted": true, "failure": {...}}`` after the
            request was finalized with that failure.

        Raises:
            NativeBridgeError: The request is unknown, its deadline passed, a
                non-deployment budget scope rejected the reservation, or the
                reservation write failed; the request is finalized before the
                error is raised.
        """
        data = json.loads(argument)
        request_id = str(data["request_id"])
        with self._lock:
            entry = self._inflight.get(request_id)
        if entry is None or int(data["attempt_ordinal"]) != entry.total_attempts:
            raise NativeBridgeError(internal_protocol_error())
        route = entry.route
        keys = tuple(
            deployment_health_key(entry.authorization, deployment)
            for deployment in route.deployments
        )
        if time.monotonic() >= entry.deadline_monotonic:
            failure = GatewayFailure(
                failure_class=GatewayFailureClass.TIMEOUT,
                safe_message="gateway execution deadline exceeded",
            )
            self.finish_request_quietly(entry.authorization, failure)
            with self._lock:
                self._inflight.pop(request_id, None)
            raise NativeBridgeError(public_failure_error(failure))
        failure = failure_from_boundary_payload(data.get("failure"))
        current_depth = data.get("current_depth")
        if failure is not None and isinstance(current_depth, int):
            candidate = next_route_candidate(
                health=self._health,
                keys=keys,
                failure=failure,
                current_depth=current_depth,
                attempt_counts=entry.attempt_counts,
                total_attempts=entry.total_attempts,
                refusal_failover=entry.authorization.refusal_failover,
                failover_mode=route.snapshot.failover_mode,
            )
            last_failure: GatewayFailure | None = failure
        else:
            candidate = claim_route_from(self._health, keys, 0)
            last_failure = None
        # Rung dispatch policies shed a claimed rung SIDEWAYS to the next
        # claimable one instead of queueing on it (spill in seconds, never a
        # deadline death). Each shed is remembered so the dispatched attempt
        # can disclose the bypassed preferred rung, and so a ladder exhausted
        # ONLY by policy sheds can force-admit past the bound rather than
        # manufacture a failure unbounded admission would not have had.
        policy_sheds: list[tuple[int, str]] = []
        forced_overflow = False
        while True:
            if candidate is None:
                if policy_sheds and last_failure is None and not forced_overflow:
                    forced_overflow = True
                    candidate = policy_sheds[0][0]
                else:
                    break
            deployment = deployment_priced_for_service_tier(
                route.deployments[candidate],
                getattr(entry.request, "service_tier", None),
                forwards_tier=(
                    candidate < len(entry.tier_forwarded_by_depth)
                    and entry.tier_forwarded_by_depth[candidate]
                ),
            )
            # Reserve the worst-case in-flight tokens alongside the worst-case
            # cost. The platform's token windows (promo free-tier, strict; org
            # rate limits, soft) count these dispatched reservations, and the
            # rung's own authored token window counts the same conservative
            # bound, so a concurrent burst binds instead of leaking past caps.
            reserved_input_tokens, reserved_output_tokens = worst_case_attempt_tokens(
                entry.request, deployment
            )
            ticket = self._reserve_rung_slot(
                entry,
                deployment,
                reserved_tokens=reserved_input_tokens + reserved_output_tokens,
                force=forced_overflow,
            )
            if isinstance(ticket, RungShed):
                policy_sheds.append((candidate, ticket.reason))
                self._health.release_probe(keys[candidate])
                candidate = claim_route_from(self._health, keys, candidate + 1)
                continue
            dispatch_reason, preferred_deployment = dispatch_disclosure(
                route,
                candidate,
                policy_sheds=policy_sheds,
                forced_overflow=forced_overflow,
                sticky_preferred=entry.sticky_preferred,
            )
            try:
                attempt_id = self._write_ledger.start_attempt(
                    snapshot=route.snapshot,
                    deployment=deployment,
                    attempt_ordinal=entry.total_attempts,
                    route_depth=candidate,
                    maximum_cost_micro_usd=maximum_attempt_cost_micro_usd(
                        entry.request, deployment
                    ),
                    reserved_input_tokens=reserved_input_tokens,
                    reserved_output_tokens=reserved_output_tokens,
                    route_reason=route.route_reason,
                    fallback_reason=route.fallback_reason,
                    dispatch_reason=dispatch_reason,
                    preferred_deployment=preferred_deployment,
                )
            except BudgetReservationRejected as exc:
                if ticket is not None:
                    self._loads.release_ticket(ticket)
                self._health.release_probe(keys[candidate])
                if exc.scope_kind is not BudgetScopeKind.DEPLOYMENT:
                    error = (
                        NativeBridgeError(budget_quota_protocol_error())
                        if self._budget_error_factory is None
                        else self._budget_error_factory(str(data.get("raw_key", "")))
                    )
                    self.finish_request_quietly(entry.authorization, budget_quota_failure())
                    with self._lock:
                        self._inflight.pop(request_id, None)
                    raise error from exc
                # A route whose hard monthly allocation cannot admit this
                # call is skipped; a later certified route may still serve.
                last_failure = (
                    budget_quota_failure()
                    if candidate == len(route.deployments) - 1
                    else all_routes_unavailable_failure()
                )
                candidate = claim_route_from(self._health, keys, candidate + 1)
                continue
            except Exception as exc:  # noqa: BLE001 - boundary sanitizes every failure.
                # A reservation that raised before returning an attempt id
                # wrote nothing durable; the accepted request is terminalized
                # and the sanitized failure answers the caller.
                if ticket is not None:
                    self._loads.release_ticket(ticket)
                self._health.release_probe(keys[candidate])
                error = authority_error(exc)
                self.finish_request_quietly(
                    entry.authorization,
                    GatewayFailure(
                        failure_class=GatewayFailureClass.INTERNAL,
                        safe_message=(
                            "gateway could not reserve attempt accounting before dispatch"
                        ),
                    ),
                )
                with self._lock:
                    self._inflight.pop(request_id, None)
                raise error from exc
            if ticket is not None:
                self._loads.bind(ticket, attempt_id)
            if forced_overflow:
                with self._lock:
                    self._rung_saturated_overflows += 1
            self._bind_sticky_dispatch(entry, deployment)
            with self._lock:
                entry.attempt_counts[candidate] += 1
                entry.total_attempts += 1
                entry.active_attempt_id = attempt_id
                entry.attempt_depths[attempt_id] = candidate
            return json.dumps(
                {"attempt_id": attempt_id, "route_depth": candidate},
                separators=(",", ":"),
            )
        exhaustion = last_failure
        if exhaustion is None:
            # Nothing dispatched and nothing classified: forced claims admit
            # any non-throttled circuit, so an empty first claim means every
            # deployment sits inside a provider throttle window.
            throttled_remaining = self._health.throttled_remaining_seconds(keys)
            exhaustion = (
                all_routes_throttled_failure(throttled_remaining)
                if throttled_remaining is not None
                else all_routes_unavailable_failure()
            )
        self.finish_request_quietly(entry.authorization, ledger_failure(exhaustion))
        with self._lock:
            self._inflight.pop(request_id, None)
        failure_payload: JsonObject = {
            "failure_class": exhaustion.failure_class.value,
            "safe_message": exhaustion.safe_message,
        }
        if exhaustion.customer_owned:
            # Echoed back so the data plane renders the caller's 400, not the
            # house 502, from the failure that ended the ladder.
            failure_payload["customer_owned"] = True
        if exhaustion.rejected_parameter is not None:
            failure_payload["rejected_parameter"] = exhaustion.rejected_parameter
        if exhaustion.provider_detail is not None:
            failure_payload["provider_detail"] = exhaustion.provider_detail
        if exhaustion.refusal_reason is not None:
            # Echoed back so an exhausted refusal ladder re-renders with its
            # bounded category on the caller-facing 400.
            failure_payload["refusal_reason"] = exhaustion.refusal_reason.value
        if exhaustion.retry_after_seconds is not None:
            failure_payload["retry_after_seconds"] = exhaustion.retry_after_seconds
        return json.dumps(
            {"exhausted": True, "failure": failure_payload},
            separators=(",", ":"),
        )

    def _bind_sticky_dispatch(
        self,
        entry: InflightRequest,
        deployment: ExactModelDeployment,
    ) -> None:
        """Record or refresh the conversation's binding to its serving rung.

        Every dispatch on a ``maximize_cache_affinity`` pool records where the
        conversation's provider cache is being built, with the serving rung's
        authored ``sticky_spill_seconds`` as the lifetime: a spilled
        conversation thereby stays on its spill target instead of bouncing
        back to the higher-ranked rung the moment it stops shedding, and a
        conversation on its preferred rung simply refreshes a no-op binding.
        A rung authoring no lifetime records nothing.

        Args:
            entry: The owning in-flight request.
            deployment: The rung about to receive the dispatch.
        """
        if entry.affinity_fingerprint is None:
            return
        if entry.route.snapshot.failover_mode != "maximize_cache_affinity":
            return
        policy = deployment.gateway.dispatch
        if policy is None or policy.sticky_spill_seconds is None:
            return
        self._sticky.bind(
            entry.affinity_fingerprint,
            deployment.deployment_id,
            ttl_seconds=float(policy.sticky_spill_seconds),
        )

    def settle(self, argument: str) -> str:
        """Durably settle one previously reserved attempt exactly once.

        A finalizing settlement also terminalizes the request and removes the
        in-flight entry; a non-finalizing one (a failed precommit dispatch
        with a successor still possible) closes only the attempt so the
        waterfall can reserve its next dispatch. Deployment-health circuits
        record every settled outcome, restoring admission first when the
        dispatch had opened.

        Args:
            argument: JSON object with ``request_id``, ``attempt_id``,
                ``outcome``, optional ``usage``, ``tool_names``, ``failure``,
                ``finalize`` (default true), and ``opened`` (default false).

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
        attempt_id = str(data["attempt_id"])
        finalize = bool(data.get("finalize", True))
        opened = bool(data.get("opened", False))
        terminal, failure = terminal_from_settlement(data)
        first_token_at = first_token_at_from_settlement(data)
        rate_limit = settlement_rate_limit(data)
        try:
            self._write_ledger.finish_attempt(
                attempt_id=attempt_id,
                terminal_event=terminal,
                failure=failure,
                finalize_request=finalize,
                first_token_at=first_token_at,
                retry_after_seconds=rate_limit.retry_after_seconds,
                ratelimit_limit_requests=rate_limit.limit_requests,
                ratelimit_remaining_requests=rate_limit.remaining_requests,
                ratelimit_limit_tokens=rate_limit.limit_tokens,
                ratelimit_remaining_tokens=rate_limit.remaining_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - the data plane retries.
            # The exact settlement is retained so a retry (from the data
            # plane or the timer sweep) lands the ORIGINAL outcome and usage,
            # never a downgraded cancellation.
            with self._lock:
                entry.pending_settlement = data
            raise authority_error(exc) from exc
        self._record_health(entry, attempt_id, opened=opened, failure=failure)
        self._record_cache_fraction(entry, attempt_id, terminal.usage)
        with self._lock:
            if finalize:
                self._inflight.pop(request_id, None)
            elif entry.active_attempt_id == attempt_id:
                entry.active_attempt_id = None
        return "{}"

    def abandon(self, argument: str) -> str:
        """Terminalize one accepted request with no settleable outcome.

        The data plane calls this when a request ends before any attempt is
        active (a queue-deadline expiry, a drained permit, admission wire
        drift, or a dropped handler between attempts). An active attempt is
        settled with the given failure; otherwise only the accepted request
        is finalized.

        Args:
            argument: JSON object with ``request_id`` and an optional
                ``failure`` (defaulting to a cancellation).

        Returns:
            An empty JSON object; an unknown request is a no-op.

        Raises:
            NativeBridgeError: The durable terminal write failed; the entry
                is kept so the deadline sweep can still close it.
        """
        data = json.loads(argument)
        request_id = str(data["request_id"])
        with self._lock:
            entry = self._inflight.get(request_id)
        if entry is None:
            return "{}"
        failure = failure_from_boundary_payload(data.get("failure")) or GatewayFailure(
            failure_class=GatewayFailureClass.CANCELLED,
            safe_message="gateway request was cancelled",
        )
        try:
            if entry.active_attempt_id is not None:
                self._record_health(entry, entry.active_attempt_id, opened=False, failure=failure)
                self._write_ledger.finish_attempt(
                    attempt_id=entry.active_attempt_id,
                    terminal_event=GatewayEvent(
                        kind=GatewayEventKind.FAILED,
                        sequence_number=0,
                        failure=failure,
                    ),
                    failure=failure,
                    finalize_request=True,
                )
            else:
                self._write_ledger.finish_request(
                    authorization=entry.authorization,
                    failure=failure,
                )
        except Exception as exc:  # noqa: BLE001 - the data plane retries.
            raise authority_error(exc) from exc
        with self._lock:
            self._inflight.pop(request_id, None)
        return "{}"

    def finish_request_quietly(
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

    def _record_health(
        self,
        entry: InflightRequest,
        attempt_id: str,
        *,
        opened: bool,
        failure: GatewayFailure | None,
    ) -> None:
        """Apply one settled attempt's outcome to the deployment circuits.

        Mirrors the executor's recording order: a successful dispatch opening
        restores admission first, then the terminal outcome either closes the
        circuit or counts against it.

        Args:
            entry: The owning in-flight request.
            attempt_id: The settled attempt.
            opened: Whether the provider dispatch opened successfully.
            failure: The terminal failure, or ``None`` for a success.
        """
        # The rung's bounded-admission slot frees with the health recording:
        # both releases are idempotent, so settle, abandon, and the sweep can
        # each fire without double-counting.
        self._loads.release_attempt(attempt_id)
        depth = entry.attempt_depths.get(attempt_id)
        if depth is None:
            return
        deployment = entry.route.deployments[depth]
        key = deployment_health_key(entry.authorization, deployment)
        if opened:
            self._health.dispatch_opened(key)
        if failure is not None:
            policy = deployment.gateway.dispatch
            if failure.failure_class == GatewayFailureClass.THROTTLED and (
                policy is not None
                and (
                    policy.concurrency_bound is not None
                    or policy.requests_per_minute is not None
                    or policy.tokens_per_minute is not None
                )
            ):
                # Passive-adaptive calibration: the provider just proved the
                # observed dispatch rate is too high for this physical lane,
                # so the rung's learned ceiling clamps to it. Recovery creep
                # rediscovers headroom without synthetic probes. Gated on an
                # admission-participating policy: only such rungs reserve
                # through the registry, so only they carry a real observed
                # window (and only they would ever enforce the ceiling).
                self._loads.record_throttle(rung_load_key(deployment))
            self._health.failed(key, failure)
        else:
            self._health.succeeded(key)

    def _record_cache_fraction(
        self,
        entry: InflightRequest,
        attempt_id: str,
        usage: GatewayUsage | None,
    ) -> None:
        """Fold one settled attempt's cached-token fraction into the registry.

        Feeds the congestion-dependent cache-priority term: the registry keeps
        a per-(organization, rung) EWMA of the settled cached fraction, so
        fairness can favor the organization whose traffic actually reuses warm
        provider cache. Attempts without observed token usage record nothing,
        and one attempt folds at most once: a settlement can reach the ledger
        through both the direct path and the retained-settlement sweep (each
        idempotent there), and a duplicate fold would skew the estimate.

        Args:
            entry: The owning in-flight request.
            attempt_id: The settled attempt.
            usage: The terminal event's usage, if any.
        """
        if usage is None or usage.input_tokens is None:
            return
        depth = entry.attempt_depths.get(attempt_id)
        if depth is None:
            return
        with self._lock:
            if attempt_id in entry.cache_recorded_attempts:
                return
            entry.cache_recorded_attempts.add(attempt_id)
        cached = usage.cached_input_tokens
        self._loads.record_settle(
            rung_load_key(entry.route.deployments[depth]),
            entry.authorization.organization_id,
            cached_tokens=0 if cached is None else cached,
            input_tokens=usage.input_tokens,
        )

    def _sweep_loop(self) -> None:
        """Run the settlement sweep on a timer for the process lifetime."""
        while True:
            time.sleep(_SWEEP_INTERVAL_SECONDS)
            self.sweep_expired()

    def sweep_expired(self) -> None:
        """Recover retained settlements and close abandoned requests.

        A retained settlement (the data plane's terminal write failed) is
        replayed verbatim so the original outcome, usage, and finalize flag
        land. A request with no settlement at all past its deadline plus
        grace is closed as cancelled through its active attempt when one
        exists, otherwise through its request row; that is the backstop for
        wire-contract failures and data-plane crashes short of process death.
        A retained settlement that fails again here latches
        accounting-unhealthy as a durable loss.
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
            if self._settle_swept(
                request_id,
                entry,
                attempt_id=str(settlement["attempt_id"]),
                terminal=terminal,
                failure=failure,
                finalize=bool(settlement.get("finalize", True)),
                rate_limit=settlement_rate_limit(settlement),
            ):
                with self._lock:
                    entry.pending_settlement = None
                    self._sweep_retained_replayed += 1
        if not abandoned:
            return
        cancelled = GatewayFailure(
            failure_class=GatewayFailureClass.CANCELLED,
            safe_message="gateway request was abandoned before settlement",
        )
        terminal = GatewayEvent(kind=GatewayEventKind.FAILED, sequence_number=0, failure=cancelled)
        for request_id, entry in abandoned:
            if entry.active_attempt_id is None:
                # An accepted request with every attempt already settled (or
                # none reserved) closes through its request row alone.
                try:
                    self._write_ledger.finish_request(
                        authorization=entry.authorization,
                        failure=cancelled,
                    )
                except Exception:  # noqa: BLE001 - keep the entry; the sweep retries.
                    self._accounting_healthy = False
                    continue
                with self._lock:
                    self._inflight.pop(request_id, None)
                    self._sweep_abandoned_cancelled += 1
                continue
            if self._settle_swept(
                request_id,
                entry,
                attempt_id=entry.active_attempt_id,
                terminal=terminal,
                failure=cancelled,
                finalize=True,
            ):
                with self._lock:
                    self._sweep_abandoned_cancelled += 1

    def _settle_swept(
        self,
        request_id: str,
        entry: InflightRequest,
        *,
        attempt_id: str,
        terminal: GatewayEvent,
        failure: GatewayFailure | None,
        finalize: bool,
        rate_limit: RateLimitObservation | None = None,
    ) -> bool:
        """Land one swept settlement; keep the entry for retry on failure.

        Returns:
            Whether the swept terminal write reached the ledger.
        """
        observation = RateLimitObservation() if rate_limit is None else rate_limit
        try:
            self._write_ledger.finish_attempt(
                attempt_id=attempt_id,
                terminal_event=terminal,
                failure=failure,
                finalize_request=finalize,
                retry_after_seconds=observation.retry_after_seconds,
                ratelimit_limit_requests=observation.limit_requests,
                ratelimit_remaining_requests=observation.remaining_requests,
                ratelimit_limit_tokens=observation.limit_tokens,
                ratelimit_remaining_tokens=observation.remaining_tokens,
            )
        except Exception:  # noqa: BLE001 - keep the entry; the sweep retries.
            self._accounting_healthy = False
            return False
        self._record_health(entry, attempt_id, opened=False, failure=failure)
        # A retained settlement that finally lands through the sweep carries
        # the same observed usage as the direct path, so the cache-priority
        # EWMA must not depend on WHICH recovery path succeeded.
        self._record_cache_fraction(entry, attempt_id, terminal.usage)
        with self._lock:
            if finalize:
                self._inflight.pop(request_id, None)
            elif entry.active_attempt_id == attempt_id:
                entry.active_attempt_id = None
        return True


def record_dead_admission_rungs(
    accounting: NativeAttemptAccounting,
    authorization: AuthorizationSnapshot,
    dead: tuple[DeadRung, ...],
    *,
    fallback_available: bool,
) -> None:
    """Record admission-dead rungs and surface a lead masked by fallback."""
    if not dead:
        return
    for rung in dead:
        accounting.health.failed(
            deployment_health_key(authorization, rung.deployment),
            rung.failure,
        )
    lead = next((rung for rung in dead if rung.index == 0), None)
    lead_masked = lead is not None and fallback_available
    accounting.record_admission_rung_skips(len(dead), lead_skipped=lead_masked)
    if lead is not None and fallback_available:
        _logger.warning(
            "gateway admission skipped the lead rung for alias %r: served off a "
            "fallback because deployment %r (provider %r) was dead at admission",
            authorization.alias,
            lead.deployment.deployment_id,
            lead.deployment.provider,
        )
