"""Durable native attempt reservations, scoped recovery, settlement and deadline cleanup."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final

from exp.common.core.artifacts import JsonObject
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.attempt_tokens import worst_case_output_tokens
from exp.runtime.gateway.budget_continuation import denied_destination_pool
from exp.runtime.gateway.budgets import BudgetReservationRejected, BudgetScopeKind
from exp.runtime.gateway.class_throttles import (
    attempt_output_pace,
    started_attempt_payload,
    tier_settlement,
)
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayEvent,
    GatewayFailure,
    GatewayFailureClass,
)
from exp.runtime.gateway.disconnect_estimate import settled_terminal
from exp.runtime.gateway.failover_pass import (
    FailoverPass,
    failover_refusal,
    forced_failover_target,
    sideways_claim,
)
from exp.runtime.gateway.guardrails.session import discard_guardrails
from exp.runtime.gateway.health import DeploymentHealthRegistry
from exp.runtime.gateway.lane_saturation import lane_saturated_failure, overflow_target
from exp.runtime.gateway.ledger import AttemptRejectedError
from exp.runtime.gateway.model_chain_authority import require_bound_model_chain_authority
from exp.runtime.gateway.native_accounting_errors import (
    NativeBridgeError,
    accounting_recovering_error,
    authority_error,
    internal_protocol_error,
)
from exp.runtime.gateway.native_accounting_recovery import (
    RecoveryRotation,
    recover_expired,
    recover_until_stopped,
    recovery_drained,
)
from exp.runtime.gateway.native_attempt_pricing import deployment_priced_at, priced_attempt
from exp.runtime.gateway.native_components import SyncWriteLedger
from exp.runtime.gateway.native_execution import (
    THROTTLE_BACKOFF,
    THROTTLE_FAILOVER_COLD,
    THROTTLE_SURFACED_CACHE_PRESERVING,
    InflightRequest,
    ThrottleDisposition,
    claim_route_from,
    deployment_health_key,
    dispatch_disclosure,
)
from exp.runtime.gateway.native_fallback_rules import eligible_ladder, rule_fallback_reason
from exp.runtime.gateway.native_recovery import (
    observe_reserved_attempt,
    record_departure,
    record_session_outcome,
    recovery_failure,
    retain_recovery_observation_time,
)
from exp.runtime.gateway.native_request_settlement import RequestSettlements
from exp.runtime.gateway.native_rung_policy import (
    bind_sticky_dispatch,
    ejection_disclosure,
    failed_dispatch_candidate,
    record_attempt_health,
    record_cache_fraction,
    reserve_rung_slot,
    shed_keeps_rung,
)
from exp.runtime.gateway.native_service_tiers import admission_kwarg, tier_ceiling
from exp.runtime.gateway.native_settlement import (
    all_routes_throttled_failure,
    all_routes_unavailable_failure,
    budget_quota_failure,
    budget_quota_protocol_error,
    exhausted_attempt_payload,
    failure_from_boundary_payload,
    ledger_failure,
    settlement_metadata,
    terminal_from_settlement,
    tool_search_requests_from_terminal,
    tool_search_requests_kwarg,
    web_search_requests_from_terminal,
    web_search_requests_kwarg,
)
from exp.runtime.gateway.recovery import RecoveryHost, SessionRecoveryRegistry
from exp.runtime.gateway.rung_admission import RungLoadRegistry, RungShed
from exp.runtime.gateway.sticky_affinity import StickySpillRegistry
from exp.runtime.openai_protocol.errors import public_failure_error

_SWEEP_INTERVAL_SECONDS = 5.0

# The deadline expired before a provider answered; the relay names the streaming case.
DEADLINE_BEFORE_ANSWER_MESSAGE: Final = (
    "the request reached the gateway's time limit before any provider finished "
    "answering; retry the request"
)

_logger = logging.getLogger(__name__)


TOOL_SEARCH_ROUND: Final = "tool_search_round"
"""Dispatch reason of a same-rung re-dial after a gateway tool-search round."""


class NativeAttemptAccounting:
    """Registry of admitted requests and their durable attempt settlements.

    A lock protects Rust callbacks; a background sweep owns abandoned reservations.
    Failed terminal writes block this control plane's admission until recovery.
    """

    def __init__(
        self,
        write_ledger: SyncWriteLedger,
        *,
        budget_error_factory: Callable[[str], NativeBridgeError] | None = None,
        cache_sample_gate: Callable[[str], bool] | None = None,
        recovery_host: RecoveryHost | None = None,
        default_lane_bound: int | None = None,
        background_recovery: bool = True,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """Bind the durable ledger and start the settlement sweep.

        Args:
            write_ledger: Blocking durable request and attempt ledger.
            budget_error_factory: Optional mapping for a rejected reservation.
            default_lane_bound: Per-worker cap for rungs authoring no bound (lane_saturation).
            background_recovery: Start the owned recovery thread. False requires
                the host to serialize periodic sweeps and keep doing so through close.
            clock: Aware wall clock that dates each attempt's start for off-peak pricing.
            cache_sample_gate: Predicate admitting a settled attempt to the cache-priority
                EWMA. Hosts exclude promo-funded samples so subsidized replay cannot buy
                fair-share weight. ``None`` admits every sample; exceptions skip it.
        """
        self._write_ledger = write_ledger
        self._clock = clock
        self.request_settlements = RequestSettlements(write_ledger)
        self._finish_attempt: Callable[..., None] = write_ledger.finish_attempt
        self._budget_error_factory = budget_error_factory
        self._cache_sample_gate = cache_sample_gate
        self.recovery_host = recovery_host
        self.recovery = SessionRecoveryRegistry()
        self._health = DeploymentHealthRegistry()
        self._loads = RungLoadRegistry(default_bound=default_lane_bound)
        # Cache-affinity spills stay on the rung holding the warmed conversation.
        self._sticky = StickySpillRegistry()
        self._inflight: dict[str, InflightRequest] = {}
        self._lock = threading.Lock()
        self._active_settlements = 0
        self._retained_recovery = RecoveryRotation[tuple[str, InflightRequest]](
            lambda item: item[0]
        )
        self._abandoned_recovery = RecoveryRotation[tuple[str, InflightRequest]](
            lambda item: item[0]
        )
        self._sweep_retained_replayed = 0
        self._sweep_abandoned_cancelled = 0
        # Unresolvable certified rungs skipped at admission, including skipped leads.
        self._admission_dead_rungs_skipped = 0
        self._admission_parameter_coercions = 0
        self._admission_lead_rungs_skipped = 0
        # Count bypasses by bound, rate, fresh-session and tenant-share limits,
        # plus forced dispatches when no other rung can serve.
        self._rung_admission_sheds = 0
        self._rung_saturated_overflows = 0
        self._rung_saturation_refusals = 0
        self._rung_rate_limit_sheds = 0
        self._rung_fresh_session_spills = 0
        # Cache-stakes throttle dispositions on pools authoring a
        # throttle_cache_threshold (surfaced: no further attempt row, so the
        # only worker-side trace; failed over cold: fallback reserved), plus
        # post-backoff redials on pools authoring a throttle_redial schedule
        # and the redials force-admitted past the warm rung's own rate shed.
        self._throttles_surfaced = 0
        self._throttles_failed_over = 0
        self._throttle_backoff_redials = 0
        self._throttle_backoff_forced = 0
        # Reconcile retained writes and abandoned requests even without new traffic.
        self._stop_sweeper = threading.Event()
        self._sweeper: threading.Thread | None = None
        if background_recovery:
            self._sweeper = threading.Thread(
                target=self._sweep_loop,
                name="exp-native-settlement-sweep",
                daemon=True,
            )
            self._sweeper.start()

    @property
    def accounting_healthy(self) -> bool:
        """Return whether every durable terminal write has landed."""
        with self._lock:
            pending = any(
                entry.settlement_write_failed or entry.pending_abandon is not None
                for entry in self._inflight.values()
            )
        return not pending and not self.request_settlements.pending

    def require_clear(self) -> None:
        """Refuse new work while this control plane owns a failed accounting write."""
        self.request_settlements.require_clear()
        with self._lock:
            pending = any(
                entry.settlement_write_failed or entry.abandon_write_failed
                for entry in self._inflight.values()
            )
        if pending:
            raise NativeBridgeError(accounting_recovering_error())

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
        rate_retry: bool = False,
        failover: bool = False,
    ) -> str | RungShed | None:
        """Reserve one slot through ``reserve_rung_slot``, counting every shed.

        Args:
            entry: The owning in-flight request (organization and weight).
            deployment: The claimed rung about to dispatch.
            reserved_tokens: Worst-case tokens this dispatch reserves.
            force: Admit past soft policy limits when overflow is explicitly allowed.
            rate_retry: Skip only rate-window checks after scheduled backoff.
            failover: The reservation is a failover's forced pass.

        Returns:
            A ticket, the shed disclosure, or ``None`` without an admission policy.
        """
        result = reserve_rung_slot(
            self._loads,
            self._sticky,
            entry,
            deployment,
            reserved_tokens=reserved_tokens,
            force=force,
            rate_retry=rate_retry,
            failover_pass=failover,
        )
        if isinstance(result, RungShed):
            with self._lock:
                self._rung_admission_sheds += 1
                if result.reason == "rate_limit":
                    self._rung_rate_limit_sheds += 1
                elif result.reason == "fresh_session_spill":
                    self._rung_fresh_session_spills += 1
        return result

    def rung_admission_counters(self) -> tuple[int, int, int]:
        """Return ``(sheds, saturated_overflows, saturation_refusals)`` for metrics."""
        with self._lock:
            sheds, overflows = self._rung_admission_sheds, self._rung_saturated_overflows
            return (sheds, overflows, self._rung_saturation_refusals)

    def rung_rate_counters(self) -> tuple[int, int]:
        """Return ``(rate_limit_sheds, fresh_session_spills)`` for metrics."""
        with self._lock:
            return (self._rung_rate_limit_sheds, self._rung_fresh_session_spills)

    def throttle_cache_counters(self) -> tuple[int, int, int, int]:
        """Return ``(surfaced, failed_over, backoff_redials, backoff_forced)`` for metrics."""
        with self._lock:
            return (
                self._throttles_surfaced,
                self._throttles_failed_over,
                self._throttle_backoff_redials,
                self._throttle_backoff_forced,
            )

    def _count_throttle_disposition(self, disposition: ThrottleDisposition) -> None:
        """Count one throttle decision for the metrics snapshot."""
        with self._lock:
            if disposition == THROTTLE_SURFACED_CACHE_PRESERVING:
                self._throttles_surfaced += 1
            elif disposition == THROTTLE_BACKOFF:
                self._throttle_backoff_redials += 1
            else:
                self._throttles_failed_over += 1

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
        """Return one live request, or None after settlement."""
        with self._lock:
            return self._inflight.get(request_id)

    def start_attempt(self, argument: str) -> str:
        """Reserve a dispatch while keeping concurrent abandonment discoverable.

        The request-local gate serializes physical reservations and cleanup,
        never unrelated requests. Abandonment records intent without waiting
        for ledger I/O; a late reservation is closed before any dispatch reply.
        The native ordinal is revalidated under the gate by the selection path.
        """
        request_id = str(json.loads(argument)["request_id"])
        entry = self.entry(request_id)
        if entry is None:
            raise NativeBridgeError(internal_protocol_error())
        try:
            with entry.execution_lock:
                result = self._start_attempt(argument)
        finally:
            with self._lock:
                cancelled = entry.pending_abandon
            if cancelled is not None:
                self.abandon(json.dumps({"request_id": request_id}))
        if cancelled is not None:
            return exhausted_attempt_payload(cancelled)
        return result

    def _start_attempt(self, argument: str) -> str:
        """Select and reserve one ordinal under its request-local execution gate."""
        data = json.loads(argument)
        request_id = str(data["request_id"])
        with self._lock:
            entry = self._inflight.get(request_id)
        if entry is None or int(data["attempt_ordinal"]) != entry.total_attempts:
            raise NativeBridgeError(internal_protocol_error())
        if entry.pending_abandon is not None:
            return exhausted_attempt_payload(entry.pending_abandon)
        route = entry.route
        if entry.authorization.model_chain_authority is not None:
            # The host refreshes an expired receipt inside reservation before
            # binding it to the durable attempt; identity and mode cannot drift.
            require_bound_model_chain_authority(entry.authorization, require_current=False)
        keys = tuple(
            deployment_health_key(entry.authorization, deployment)
            for deployment in route.deployments
        )
        if time.monotonic() >= entry.deadline_monotonic:
            failure = GatewayFailure(
                failure_class=GatewayFailureClass.TIMEOUT,
                safe_message=DEADLINE_BEFORE_ANSWER_MESSAGE,
            )
            self.finish_request_quietly(entry.authorization, failure)
            with self._lock:
                discard_guardrails(self._inflight.pop(request_id, None))
            raise NativeBridgeError(public_failure_error(failure))
        failure = failure_from_boundary_payload(data.get("failure"))
        current_depth = data.get("current_depth")
        selected_first = route.resolved_route_id is not None and entry.total_attempts == 0
        ladder = (0,) if selected_first else eligible_ladder(route, failure)
        # A caller-selected first dial cannot spill or override health and load gates.
        # Other ladders may spill sideways or use their operator-authored overflow.
        policy_sheds: list[tuple[int, str]] = []
        disposition: ThrottleDisposition | None = None
        redial_depth: int | None = None  # The rung a post-backoff redial re-dials.
        tool_search_round = False
        reasoning_repair = data.get("reasoning_repair") is True and isinstance(current_depth, int)
        failover = FailoverPass.begin(entry, failure, current_depth)
        if failure is not None and isinstance(current_depth, int):
            candidate, disposition = failed_dispatch_candidate(
                health=self._health,
                loads=self._loads,
                keys=keys,
                entry=entry,
                failure=failure,
                current_depth=current_depth,
                throttle_backoff=data.get("throttle_backoff") is True,
            )
            if disposition == THROTTLE_BACKOFF:
                redial_depth = current_depth
            elif disposition == THROTTLE_SURFACED_CACHE_PRESERVING:
                # Terminal by construction, so counted at the decision; the
                # cold branch counts only once a fallback is reserved below.
                self._count_throttle_disposition(disposition)
            elif disposition == THROTTLE_FAILOVER_COLD:
                # A cold failover bypasses the warm rung by policy and is
                # disclosed like a shed, with the throttled rung as the
                # preferred counterfactual. It never force-admits.
                policy_sheds.append((current_depth, THROTTLE_FAILOVER_COLD))
            last_failure: GatewayFailure | None = failure
        else:
            # Semantic tool turns prefer the preceding rung; repairs stay on that rung.
            tool_search_round = data.get("tool_search_round") is True and isinstance(
                current_depth, int
            )
            if selected_first:
                candidate = 0 if self._health.claim(keys[0]) else None
            else:
                candidate = claim_route_from(
                    self._health,
                    keys,
                    current_depth if tool_search_round or reasoning_repair else 0,
                    (current_depth,) if reasoning_repair else ladder,
                )
            if reasoning_repair:
                ladder = (current_depth,)
            last_failure = None
        forced_overflow = False
        capacity_refused = False
        shed_records: dict[int, RungShed] = {}
        # Tokenize the input reservation once per request for every attempt.
        reserved_input_tokens = entry.reserved_input_tokens()
        while True:
            with self._lock:
                active = self._inflight.get(request_id) is entry and entry.pending_abandon is None
            if not active or time.monotonic() >= entry.deadline_monotonic:
                if candidate is not None:
                    self._health.release_probe(keys[candidate])
                last_failure = GatewayFailure(
                    failure_class=GatewayFailureClass.TIMEOUT
                    if active
                    else GatewayFailureClass.CANCELLED,
                    safe_message=DEADLINE_BEFORE_ANSWER_MESSAGE
                    if active
                    else "gateway request was cancelled",
                )
                break
            if candidate is None:
                if policy_sheds and last_failure is None and not forced_overflow:
                    candidate = (
                        None
                        if selected_first and not entry.authorization.priority_admission
                        else overflow_target(route, policy_sheds, shed_records)
                    )
                    forced_overflow = candidate is not None
                    if candidate is None:
                        last_failure = lane_saturated_failure(entry.authorization)
                        capacity_refused = True
                        with self._lock:
                            self._rung_saturation_refusals += 1
                        break
                else:
                    # A failover is never ended by a sibling's bound (failover_pass).
                    candidate = forced_failover_target(
                        failover, policy_sheds, shed_records, self._health, keys
                    )
                    if candidate is None:
                        refusal = failover_refusal(failover, entry.authorization, shed_records)
                        if refusal is not None:
                            last_failure = refusal
                            with self._lock:
                                self._rung_saturation_refusals += 1
                        break
                    forced_overflow = True
            if route.snapshot.stage_for_depth(candidate).pool_id in entry.denied_destination_pools:
                self._health.release_probe(keys[candidate])
                candidate = sideways_claim(failover, self._health, keys, ladder, candidate)
                continue
            if not entry.attempt_policy.permits(
                entry.total_attempts, entry.attempt_counts[candidate]
            ):
                last_failure = last_failure or GatewayFailure(
                    failure_class=GatewayFailureClass.INVALID_REQUEST,
                    safe_message=(
                        "The request exhausted gateway.retry attempt limits before "
                        "producing an answer. Increase the requested limits and resend."
                    ),
                    rejected_parameter="gateway.retry",
                )
                self._health.release_probe(keys[candidate])
                break
            priced_at = self._clock()
            deployment, tier = priced_attempt(
                route.deployments[candidate],
                getattr(entry.request, "service_tier", None),
                forwards_tier=candidate < len(entry.tier_forwarded_by_depth)
                and entry.tier_forwarded_by_depth[candidate],
                instant=priced_at,
            )
            reservation_request = entry.request
            if entry.reserved_output_tokens_by_depth:
                frozen_bound = entry.reserved_output_tokens_by_depth[candidate]
                reservation_request = entry.request.model_copy(
                    update={"maximum_output_tokens": frozen_bound}
                )
            reserved_output_tokens = worst_case_output_tokens(reservation_request, deployment)
            ticket = self._reserve_rung_slot(
                entry,
                deployment,
                reserved_tokens=reserved_input_tokens + reserved_output_tokens,
                force=forced_overflow,
                rate_retry=forced_overflow and candidate == redial_depth,
                failover=failover is not None and failover.forcing,
            )
            if isinstance(ticket, RungShed):
                record_departure(
                    self.recovery, self.recovery_host, entry, deployment, "local_capacity"
                )
                policy_sheds.append((candidate, ticket.reason))
                shed_records[candidate] = ticket
                self._health.release_probe(keys[candidate])
                forced_overflow = not selected_first and shed_keeps_rung(
                    route, candidate, redial_depth, last_failure, ticket.reason
                )
                if (
                    forced_overflow
                    and not (candidate == redial_depth and ticket.reason == "rate_limit")
                    and overflow_target(route, [(candidate, ticket.reason)], {candidate: ticket})
                    is None
                ):
                    last_failure = lane_saturated_failure(entry.authorization)
                    capacity_refused = True
                    with self._lock:
                        self._rung_saturation_refusals += 1
                    break
                if not forced_overflow:
                    candidate = sideways_claim(failover, self._health, keys, ladder, candidate)
                continue
            throttle_backoff = candidate == redial_depth
            dispatch_reason, preferred_deployment = dispatch_disclosure(
                route,
                candidate,
                policy_sheds=policy_sheds,
                forced_overflow=forced_overflow,
                sticky_preferred=entry.sticky_preferred,
                throttle_backoff=throttle_backoff,
            )
            if tool_search_round:
                dispatch_reason = TOOL_SEARCH_ROUND
            elif candidate != current_depth:
                dispatch_reason, preferred_deployment = ejection_disclosure(
                    self._health,
                    keys,
                    route,
                    failover.untried if failover is not None else ladder,
                    candidate,
                    (dispatch_reason, preferred_deployment),
                )
            try:
                attempt_id = self._write_ledger.start_attempt(
                    snapshot=route.snapshot,
                    deployment=deployment,
                    attempt_ordinal=entry.total_attempts,
                    route_depth=candidate,
                    maximum_cost_nano_usd=tier_ceiling(
                        reservation_request, deployment, tier, input_tokens=reserved_input_tokens
                    ),
                    **admission_kwarg(tier),
                    reserved_input_tokens=reserved_input_tokens,
                    reserved_output_tokens=reserved_output_tokens,
                    route_reason=route.attempt_route_reason(route.deployments[candidate]),
                    fallback_reason=rule_fallback_reason(route, candidate, current_depth, failure),
                    dispatch_reason=entry.recovery_reason
                    if candidate == 0
                    and entry.total_attempts == 0
                    and entry.recovery_reason is not None
                    and dispatch_reason in (None, "affinity", "affinity_sticky")
                    else dispatch_reason,
                    preferred_deployment=deployment_priced_at(preferred_deployment, priced_at),
                )
            except BudgetReservationRejected as exc:
                if ticket is not None:
                    self._loads.release_ticket(ticket)
                self._health.release_probe(keys[candidate])
                denied_pool = denied_destination_pool(exc, route.snapshot, candidate)
                if denied_pool is not None:
                    entry.denied_destination_pools.add(denied_pool)
                    last_failure = budget_quota_failure()
                    forced_overflow = False
                    candidate = sideways_claim(failover, self._health, keys, ladder, candidate)
                    continue
                if exc.scope_kind is not BudgetScopeKind.DEPLOYMENT:
                    error = (
                        NativeBridgeError(budget_quota_protocol_error())
                        if self._budget_error_factory is None
                        else self._budget_error_factory(str(data.get("raw_key", "")))
                    )
                    self.finish_request_quietly(entry.authorization, budget_quota_failure())
                    with self._lock:
                        discard_guardrails(self._inflight.pop(request_id, None))
                    raise error from exc
                # A route whose hard monthly allocation cannot admit this
                # call is skipped; a later certified route may still serve.
                last_failure = (
                    budget_quota_failure()
                    if candidate == len(route.deployments) - 1
                    else all_routes_unavailable_failure()
                )
                # A forced reservation belongs only to the selected rung, never
                # a later destination reached after this one's budget rejection.
                forced_overflow = False
                candidate = sideways_claim(failover, self._health, keys, ladder, candidate)
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
                    exc.failure
                    if isinstance(exc, AttemptRejectedError)
                    else GatewayFailure(
                        failure_class=GatewayFailureClass.INTERNAL,
                        safe_message="gateway could not reserve attempt accounting before dispatch",
                    ),
                )
                with self._lock:
                    discard_guardrails(self._inflight.pop(request_id, None))
                raise error from exc
            if ticket is not None:
                self._loads.bind(ticket, attempt_id)
            if disposition == THROTTLE_FAILOVER_COLD:
                # Real only now: an exhausted ladder is a plain exhausted throttle.
                self._count_throttle_disposition(disposition)
            elif throttle_backoff:
                self._count_throttle_disposition(THROTTLE_BACKOFF)
            if self.recovery_host is None:
                bind_sticky_dispatch(self._sticky, entry, deployment)
            observe_reserved_attempt(self.recovery_host, entry, attempt_id, deployment)
            with self._lock:
                if forced_overflow and not throttle_backoff:
                    self._rung_saturated_overflows += 1
                elif forced_overflow:
                    self._throttle_backoff_forced += 1
                entry.attempt_counts[candidate] += 1
                if not tool_search_round and not reasoning_repair:
                    entry.ordinary_attempt_counts[candidate] += 1
                if throttle_backoff:
                    entry.throttle_redials[candidate] += 1
                entry.total_attempts += 1
                entry.active_attempt_id = attempt_id
                entry.attempt_depths[attempt_id] = candidate
                if tier is not None:
                    entry.attempt_service_tiers[attempt_id] = tier
            return started_attempt_payload(
                attempt_id, candidate, attempt_output_pace(entry, deployment, self._loads)
            )
        exhaustion = last_failure
        if exhaustion is None:
            # An empty forced claim means every deployment is provider-throttled.
            # Otherwise report unavailable without a free-capacity certificate.
            throttled_remaining = self._health.throttled_remaining_seconds(keys)
            exhaustion = (
                all_routes_throttled_failure(throttled_remaining)
                if throttled_remaining is not None
                else all_routes_unavailable_failure()
            )
        certified = active and self.finish_request_quietly(
            entry.authorization,
            ledger_failure(exhaustion),
            certify_no_effects=capacity_refused and entry.no_paid_prework,
        )
        with self._lock:
            if active or entry.pending_abandon is None:  # Settlement owns active exhaustion.
                discard_guardrails(self._inflight.pop(request_id, None))
        return exhausted_attempt_payload(exhaustion, known_unbilled=capacity_refused and certified)

    def settle(self, argument: str) -> str:
        """Durably settle one previously reserved attempt exactly once.

        Finalization closes the request; otherwise the waterfall continues. Health records both.

        Args:
            argument: JSON request/attempt identity, outcome, optional usage/failure,
                finalization flag, and witnessed provider metadata.

        Returns:
            An empty JSON object; repeated settlement is a no-op.

        Raises:
            NativeBridgeError: The terminal write failed; the retained entry lets either
                the data plane or the deadline sweep retry the original settlement.
        """
        with self._lock:
            self._active_settlements += 1
        try:
            return self._settle(argument)
        finally:
            with self._lock:
                self._active_settlements -= 1

    def _settle(self, argument: str) -> str:
        """Commit one observed outcome while the caller retains shutdown ownership."""
        data = json.loads(argument)
        request_id = str(data["request_id"])
        with self._lock:
            entry = self._inflight.get(request_id)
            if entry is None:
                return "{}"
            if entry.pending_settlement is not None:
                if entry.pending_settlement["attempt_id"] != data["attempt_id"]:
                    raise NativeBridgeError(accounting_recovering_error())
                data = entry.pending_settlement
            else:
                entry.pending_settlement = data
        with entry.execution_lock:
            if self.entry(request_id) is not entry or entry.pending_settlement is not data:
                return "{}"
            return self._commit_settlement(request_id, entry, data)

    def _commit_settlement(self, request_id: str, entry: InflightRequest, data: JsonObject) -> str:
        """Write the exact observed meter while holding the request execution gate."""
        attempt_id = str(data["attempt_id"])
        finalize = bool(data.get("finalize", True))
        opened = bool(data.get("opened", False))
        parsed = terminal_from_settlement(data, surface=entry.authorization.surface)
        retain_recovery_observation_time(self.recovery, entry, attempt_id)
        terminal, failure = settled_terminal(data, entry, parsed=parsed, loads=self._loads)
        try:
            self._finish_attempt(
                attempt_id=attempt_id,
                terminal_event=terminal,
                failure=failure,
                finalize_request=finalize,
                **settlement_metadata(
                    data,
                    self._finish_attempt,
                    **tier_settlement(entry, attempt_id),
                ),
                **web_search_requests_kwarg(
                    self._finish_attempt, web_search_requests_from_terminal(terminal)
                ),
                **tool_search_requests_kwarg(
                    self._finish_attempt, tool_search_requests_from_terminal(terminal)
                ),
            )
        except Exception as exc:  # noqa: BLE001 - the data plane retries.
            # Both retry paths retain the original outcome, usage and tier evidence.
            with self._lock:
                entry.pending_settlement = data
                entry.settlement_write_failed = True
            raise authority_error(exc) from exc
        failure = terminal.failure
        record_attempt_health(
            self._health,
            self._loads,
            entry,
            attempt_id,
            opened=opened,
            failure=failure,
            settlement=data,
        )
        self._record_cache_fraction(entry, attempt_id, terminal)
        record_session_outcome(
            self.recovery,
            self.recovery_host,
            entry,
            attempt_id,
            None if terminal.usage_estimated else terminal.usage,
            recovery_failure(data, failure),
        )
        with self._lock:
            if entry.pending_settlement == data:
                entry.pending_settlement = None
                entry.settlement_write_failed = False
            if finalize:
                discard_guardrails(self._inflight.pop(request_id, None))
            elif entry.active_attempt_id == attempt_id:
                entry.active_attempt_id = None
        return "{}"

    def abandon(self, argument: str) -> str:
        """Terminalize one accepted request with no settleable outcome.

        Preserve paid admission work on the active attempt, or on the request
        when none was ever reserved. Failed writes remain owned for retry.

        Args:
            argument: JSON object with ``request_id`` and an optional
                ``failure`` (defaulting to a cancellation).

        Returns:
            Whether the request is terminalized; an unknown request is already closed.

        Raises:
            NativeBridgeError: The terminal write failed; the entry remains for retry.
        """
        data = json.loads(argument)
        request_id = str(data["request_id"])
        with self._lock:
            entry = self._inflight.get(request_id)
        if entry is None:
            return '{"terminalized":true}'
        failure = failure_from_boundary_payload(data.get("failure")) or GatewayFailure(
            failure_class=GatewayFailureClass.CANCELLED,
            safe_message="gateway request was cancelled",
        )
        with self._lock:
            if self._inflight.get(request_id) is not entry:
                return '{"terminalized":true}'
            if entry.pending_abandon is None:
                entry.pending_abandon = failure
            failure = entry.pending_abandon
        if entry.guardrails is not None:
            failure = entry.guardrails.settlement_failure() or failure
        if not entry.execution_lock.acquire(blocking=False):
            return '{"terminalized":false}'
        try:
            with self._lock:
                if self._inflight.get(request_id) is not entry:
                    return '{"terminalized":true}'
                if entry.pending_settlement is not None:
                    return '{"terminalized":false}'
            try:
                if entry.active_attempt_id is not None:
                    record_attempt_health(
                        self._health, self._loads, entry, entry.active_attempt_id, failure=failure
                    )
                    self._finish_attempt(
                        attempt_id=entry.active_attempt_id,
                        terminal_event=GatewayEvent.for_accounting_recovery(failure=failure),
                        failure=failure,
                        finalize_request=True,
                        **web_search_requests_kwarg(
                            self._finish_attempt, entry.web_search_requests
                        ),
                    )
                else:
                    self.request_settlements.finish(
                        entry.authorization,
                        failure,
                        web_search_requests=entry.web_search_requests
                        if not entry.total_attempts
                        else 0,
                    )
            except Exception as exc:  # noqa: BLE001 - the data plane retries.
                with self._lock:
                    entry.abandon_write_failed = True
                raise authority_error(exc) from exc
            with self._lock:
                # A meter may arrive while the unknown cancellation is writing.
                # MemoryDB can upgrade pending usage; a conflicting terminal
                # owner must retain the evidence instead of reporting a drain.
                if entry.pending_settlement is not None:
                    return '{"terminalized":false}'
                discard_guardrails(self._inflight.pop(request_id, None))
            return '{"terminalized":true}'
        finally:
            entry.execution_lock.release()

    def finish_request_quietly(
        self,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
        *,
        certify_no_effects: bool = False,
        web_search_requests: int = 0,
    ) -> bool:
        """Retain failed terminal writes and their exact meters for the existing sweep."""
        entry = self.entry(authorization.request_id)
        if entry is not None and not entry.total_attempts:
            web_search_requests = max(web_search_requests, entry.web_search_requests)
        try:
            return self.request_settlements.finish(
                authorization,
                failure,
                certify_no_effects=certify_no_effects,
                web_search_requests=web_search_requests,
            )
        except Exception:  # noqa: BLE001 - the original sanitized admission error wins.
            return False

    def _record_cache_fraction(
        self, entry: InflightRequest, attempt_id: str, terminal: GatewayEvent
    ) -> None:
        """Apply the observed-only cache sample once through the existing rung-policy owner."""
        record_cache_fraction(
            self._loads,
            entry,
            attempt_id,
            terminal,
            lock=self._lock,
            sample_gate=self._cache_sample_gate,
        )

    def _sweep_loop(self) -> None:
        """Periodically reconcile retained or expired attempts for the process lifetime."""
        recover_until_stopped(self, _SWEEP_INTERVAL_SECONDS)

    def close(self, timeout_seconds: float = 10.0) -> bool:
        """Stop owned recovery before a host closes its durable transport.

        Hosts drain data-plane callbacks first. A timeout leaves the transport
        owned by recovery; the caller must not close it underneath that thread.
        Durable reservations remain pending when no trustworthy meter exists.
        """
        if not 0 <= timeout_seconds <= 300:
            raise ValueError("recovery shutdown timeout must be between zero and 300 seconds")
        self._stop_sweeper.set()
        if self._sweeper is not None:
            self._sweeper.join(timeout_seconds)
            if self._sweeper.is_alive():
                return False
        return recovery_drained(self)

    def sweep_expired(self) -> None:
        """Recover a fair bounded selection without running in admission callbacks."""
        recover_expired(self)

    def _settle_swept(
        self,
        request_id: str,
        entry: InflightRequest,
        settlement: JsonObject,
    ) -> bool:
        """Replay owned evidence, skipping busy requests and stale sweep selections."""
        if not entry.execution_lock.acquire(blocking=False):
            return False
        try:
            if self.entry(request_id) is not entry or entry.pending_settlement != settlement:
                return False
            self._commit_settlement(request_id, entry, settlement)
            return True
        except Exception:  # noqa: BLE001 - keep the entry; the sweep retries.
            return False
        finally:
            entry.execution_lock.release()
