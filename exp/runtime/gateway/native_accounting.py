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
from typing import Final

from exp.common.core.artifacts import JsonObject
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.attempt_tokens import worst_case_input_tokens, worst_case_output_tokens
from exp.runtime.gateway.budgets import (
    BudgetReservationRejected,
    BudgetScopeKind,
    maximum_attempt_cost_nano_usd,
)
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
)
from exp.runtime.gateway.disconnect_estimate import settled_terminal
from exp.runtime.gateway.health import DeploymentHealthRegistry
from exp.runtime.gateway.lane_saturation import lane_saturated_failure, overflow_target
from exp.runtime.gateway.native_accounting_errors import (
    NativeBridgeError,
    authority_error,
    internal_protocol_error,
)
from exp.runtime.gateway.native_components import SyncWriteLedger
from exp.runtime.gateway.native_execution import (
    THROTTLE_BACKOFF,
    THROTTLE_FAILOVER_COLD,
    THROTTLE_SURFACED_CACHE_PRESERVING,
    InflightRequest,
    ThrottleDisposition,
    claim_route_from,
    deployment_health_key,
    deployment_priced_for_service_tier,
    dispatch_disclosure,
    rung_load_key,
)
from exp.runtime.gateway.native_fallback_rules import eligible_ladder, rule_fallback_reason
from exp.runtime.gateway.native_rung_policy import (
    failed_dispatch_candidate,
    reserve_rung_slot,
    shed_keeps_rung,
)
from exp.runtime.gateway.native_settlement import (
    all_routes_throttled_failure,
    all_routes_unavailable_failure,
    budget_quota_failure,
    budget_quota_protocol_error,
    failure_from_boundary_payload,
    first_token_at_from_settlement,
    ledger_failure,
    pins_native_responses_route,
    settlement_rate_limit,
    tool_search_requests_from_terminal,
    tool_search_requests_kwarg,
    upstream_provider_from_settlement,
    upstream_provider_kwarg,
    web_search_requests_from_terminal,
    web_search_requests_kwarg,
)
from exp.runtime.gateway.rung_admission import RungLoadRegistry, RungShed
from exp.runtime.gateway.sticky_affinity import StickySpillRegistry
from exp.runtime.openai_protocol.errors import public_failure_error

_SWEEP_GRACE_SECONDS = 5.0
_SWEEP_INTERVAL_SECONDS = 5.0
_SWEEP_BATCH = 16
_logger = logging.getLogger(__name__)


TOOL_SEARCH_ROUND: Final = "tool_search_round"
"""Dispatch reason of a same-rung re-dial after a gateway tool-search round."""


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
        cache_sample_gate: Callable[[str], bool] | None = None,
        default_lane_bound: int | None = None,
    ) -> None:
        """Bind the durable ledger and start the settlement sweep.

        Args:
            write_ledger: Blocking durable request and attempt ledger.
            budget_error_factory: Optional hosted mapping for a rejected
                reservation.
            default_lane_bound: Per-worker cap for rungs authoring no bound (lane_saturation).
            cache_sample_gate: Optional hosted predicate deciding whether one
                settled attempt (by attempt id) may feed the cache-priority
                EWMA. The hosted store knows which attempts are promo-funded;
                admitting those samples would let free-tier replay traffic
                (whose cached prefixes are costless under a promotion) buy
                fair-share weight with the very replay the promotion already
                subsidizes. ``None`` admits every sample; a raising gate
                skips the sample (fails closed).
        """
        self._write_ledger = write_ledger
        self._finish_attempt: Callable[..., None] = write_ledger.finish_attempt
        self._budget_error_factory = budget_error_factory
        self._cache_sample_gate = cache_sample_gate
        # Revision-scoped deployment health; physical-lane load survives catalog rolls.
        self._health = DeploymentHealthRegistry()
        self._loads = RungLoadRegistry(default_bound=default_lane_bound)
        # Cache-affinity spills stay on the rung holding the warmed conversation.
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
        result = reserve_rung_slot(
            self._loads,
            self._sticky,
            entry,
            deployment,
            reserved_tokens=reserved_tokens,
            force=force,
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
        """Return one tracked request, or ``None`` after settlement."""
        with self._lock:
            return self._inflight.get(request_id)

    def start_attempt(self, argument: str) -> str:
        """Reserve one physical dispatch immediately before network work.

        Each callback carries the ordinal, prior depth, failure and dispatch reason.
        Returns a durable attempt or finalized exhaustion; authority, deadline and
        accounting refusals raise NativeBridgeError.
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
        selected_first = route.resolved_route_id is not None and entry.total_attempts == 0
        ladder = (0,) if selected_first else eligible_ladder(route, failure)
        # A caller-selected first dial cannot spill or override health and load gates.
        # Other ladders may spill sideways or use their operator-authored overflow.
        policy_sheds: list[tuple[int, str]] = []
        disposition: ThrottleDisposition | None = None
        redial_depth: int | None = None  # The rung a post-backoff redial re-dials.
        tool_search_round = False
        reasoning_repair = data.get("reasoning_repair") is True and isinstance(current_depth, int)
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
        shed_records: dict[int, RungShed] = {}
        # The input half of the reservation tokenizes the whole prompt, so it
        # is computed once per ladder walk and shared with every candidate.
        reserved_input_tokens = worst_case_input_tokens(entry.request)
        while True:
            if candidate is None:
                if policy_sheds and last_failure is None and not forced_overflow:
                    candidate = (
                        None
                        if selected_first
                        else overflow_target(route, policy_sheds, shed_records)
                    )
                    forced_overflow = candidate is not None
                    if candidate is None:
                        last_failure = lane_saturated_failure()
                        with self._lock:
                            self._rung_saturation_refusals += 1
                        break
                else:
                    break
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
            deployment = deployment_priced_for_service_tier(
                route.deployments[candidate],
                getattr(entry.request, "service_tier", None),
                forwards_tier=(
                    candidate < len(entry.tier_forwarded_by_depth)
                    and entry.tier_forwarded_by_depth[candidate]
                ),
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
            )
            if isinstance(ticket, RungShed):
                policy_sheds.append((candidate, ticket.reason))
                shed_records.setdefault(candidate, ticket)
                self._health.release_probe(keys[candidate])
                forced_overflow = not selected_first and shed_keeps_rung(
                    route, candidate, redial_depth, last_failure, ticket.reason
                )
                if not forced_overflow:
                    candidate = claim_route_from(self._health, keys, candidate + 1, ladder)
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
            try:
                attempt_id = self._write_ledger.start_attempt(
                    snapshot=route.snapshot,
                    deployment=deployment,
                    attempt_ordinal=entry.total_attempts,
                    route_depth=candidate,
                    maximum_cost_nano_usd=maximum_attempt_cost_nano_usd(
                        reservation_request, deployment, input_tokens=reserved_input_tokens
                    ),
                    reserved_input_tokens=reserved_input_tokens,
                    reserved_output_tokens=reserved_output_tokens,
                    route_reason=route.attempt_route_reason(route.deployments[candidate]),
                    fallback_reason=rule_fallback_reason(route, candidate, current_depth, failure),
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
                if candidate == redial_depth:
                    # The forced admission belonged to the redialed rung alone.
                    forced_overflow = False
                candidate = claim_route_from(self._health, keys, candidate + 1, ladder)
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
            if disposition == THROTTLE_FAILOVER_COLD:
                # Real only now: an exhausted ladder is a plain exhausted throttle.
                self._count_throttle_disposition(disposition)
            elif throttle_backoff:
                self._count_throttle_disposition(THROTTLE_BACKOFF)
            self._bind_sticky_dispatch(entry, deployment)
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
            pinned = pins_native_responses_route(entry.request)
            exhaustion = (
                all_routes_throttled_failure(throttled_remaining, native_responses_pinned=pinned)
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
        terminal, failure = settled_terminal(data, entry, loads=self._loads)
        first_token_at = first_token_at_from_settlement(data)
        rate_limit = settlement_rate_limit(data)
        upstream = upstream_provider_from_settlement(data)
        try:
            self._finish_attempt(
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
                **upstream_provider_kwarg(self._finish_attempt, upstream),
                **web_search_requests_kwarg(
                    self._finish_attempt, web_search_requests_from_terminal(terminal)
                ),
                **tool_search_requests_kwarg(
                    self._finish_attempt, tool_search_requests_from_terminal(terminal)
                ),
            )
        except Exception as exc:  # noqa: BLE001 - the data plane retries.
            # The exact settlement is retained so a retry (from the data
            # plane or the timer sweep) lands the ORIGINAL outcome and usage,
            # never a downgraded cancellation.
            with self._lock:
                entry.pending_settlement = data
            raise authority_error(exc) from exc
        self._record_health(entry, attempt_id, opened=opened, failure=failure)
        self._record_cache_fraction(entry, attempt_id, terminal)
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
        terminal: GatewayEvent,
    ) -> None:
        """Fold one settled attempt's cached-token fraction into the registry.

        Feeds the congestion-dependent cache-priority term (a per-(organization,
        rung) EWMA of the settled cached fraction, so fairness favors traffic
        that reuses warm provider cache). Attempts without an observed input
        count record nothing: an estimated disconnect meter has no cache legs.
        One attempt folds at most once, whichever of the direct path and the
        retained-settlement sweep lands it, so a duplicate cannot skew the EWMA.

        Args:
            entry: The owning in-flight request.
            attempt_id: The settled attempt.
            terminal: The settled terminal event.
        """
        usage = terminal.usage
        if usage is None or usage.input_tokens is None or terminal.usage_estimated:
            return
        depth = entry.attempt_depths.get(attempt_id)
        if depth is None:
            return
        if self._cache_sample_gate is not None:
            # Promo-funded (or otherwise excluded) attempts must not buy
            # fair-share weight; an erroring gate skips the sample rather
            # than admit one the host meant to exclude.
            try:
                if not self._cache_sample_gate(attempt_id):
                    return
            except Exception:  # noqa: BLE001 - the sample is telemetry, never worth failing settle.
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
            terminal, failure = settled_terminal(settlement, entry, loads=self._loads)
            if self._settle_swept(
                request_id,
                entry,
                attempt_id=str(settlement["attempt_id"]),
                terminal=terminal,
                failure=failure,
                finalize=bool(settlement.get("finalize", True)),
                settlement=settlement,
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
        settlement: JsonObject | None = None,
    ) -> bool:
        """Land one swept settlement from its retained payload; keep the entry to retry on failure.

        Returns:
            Whether the swept terminal write reached the ledger.
        """
        observation = settlement_rate_limit(settlement)
        upstream = upstream_provider_from_settlement(settlement)
        try:
            self._finish_attempt(
                attempt_id=attempt_id,
                terminal_event=terminal,
                failure=failure,
                finalize_request=finalize,
                retry_after_seconds=observation.retry_after_seconds,
                ratelimit_limit_requests=observation.limit_requests,
                ratelimit_remaining_requests=observation.remaining_requests,
                ratelimit_limit_tokens=observation.limit_tokens,
                ratelimit_remaining_tokens=observation.remaining_tokens,
                **upstream_provider_kwarg(self._finish_attempt, upstream),
                **web_search_requests_kwarg(
                    self._finish_attempt, web_search_requests_from_terminal(terminal)
                ),
                **tool_search_requests_kwarg(
                    self._finish_attempt, tool_search_requests_from_terminal(terminal)
                ),
            )
        except Exception:  # noqa: BLE001 - keep the entry; the sweep retries.
            self._accounting_healthy = False
            return False
        self._record_health(entry, attempt_id, opened=False, failure=failure)
        # A retained settlement that finally lands through the sweep carries
        # the same observed usage as the direct path, so the cache-priority
        # EWMA must not depend on WHICH recovery path succeeded.
        self._record_cache_fraction(entry, attempt_id, terminal)
        with self._lock:
            if finalize:
                self._inflight.pop(request_id, None)
            elif entry.active_attempt_id == attempt_id:
                entry.active_attempt_id = None
        return True
