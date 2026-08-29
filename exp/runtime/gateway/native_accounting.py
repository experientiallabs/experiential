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
from typing import cast

from exp.common.core.artifacts import JsonObject
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
)
from exp.runtime.gateway.health import DeploymentHealthRegistry
from exp.runtime.gateway.native_components import SyncWriteLedger
from exp.runtime.gateway.native_execution import (
    DeadRung,
    InflightRequest,
    claim_route_from,
    deployment_health_key,
    next_route_candidate,
)
from exp.runtime.gateway.native_settlement import (
    budget_quota_protocol_error,
    first_token_at_from_settlement,
    terminal_from_settlement,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, public_failure_error

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


def budget_quota_failure() -> GatewayFailure:
    """Return the sanitized quota failure after no route can reserve its cost."""
    return GatewayFailure(
        failure_class=GatewayFailureClass.QUOTA_EXCEEDED,
        safe_message="monthly gateway allocation is exhausted",
    )


def all_routes_unavailable_failure() -> GatewayFailure:
    """Return the sanitized terminal failure for an exhausted certified pool."""
    return GatewayFailure(
        failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
        safe_message="all exact-model deployments are unavailable",
    )


def _failure_from_payload(payload: object) -> GatewayFailure | None:
    """Parse one optional classified failure from a boundary payload."""
    if not isinstance(payload, dict):
        return None
    data = cast("JsonObject", payload)
    rejected_parameter = data.get("rejected_parameter")
    provider_detail = data.get("provider_detail")
    return GatewayFailure(
        failure_class=GatewayFailureClass(str(data["failure_class"])),
        safe_message=str(data["safe_message"]),
        retryable_same_deployment=bool(data.get("retryable_same_deployment", False)),
        failover_eligible=bool(data.get("failover_eligible", False)),
        rejected_parameter=(
            rejected_parameter
            if isinstance(rejected_parameter, str) and rejected_parameter
            else None
        ),
        provider_detail=(
            provider_detail if isinstance(provider_detail, str) and provider_detail else None
        ),
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
        failure = _failure_from_payload(data.get("failure"))
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
            )
            last_failure: GatewayFailure | None = failure
        else:
            candidate = claim_route_from(self._health, keys, 0)
            last_failure = None
        while candidate is not None:
            deployment = route.deployments[candidate]
            try:
                attempt_id = self._write_ledger.start_attempt(
                    snapshot=route.snapshot,
                    deployment=deployment,
                    attempt_ordinal=entry.total_attempts,
                    route_depth=candidate,
                    maximum_cost_micro_usd=maximum_attempt_cost_micro_usd(
                        entry.request, deployment
                    ),
                    route_reason=route.route_reason,
                    fallback_reason=route.fallback_reason,
                )
            except BudgetReservationRejected as exc:
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
            with self._lock:
                entry.attempt_counts[candidate] += 1
                entry.total_attempts += 1
                entry.active_attempt_id = attempt_id
                entry.attempt_depths[attempt_id] = candidate
            return json.dumps(
                {"attempt_id": attempt_id, "route_depth": candidate},
                separators=(",", ":"),
            )
        exhaustion = last_failure or all_routes_unavailable_failure()
        self.finish_request_quietly(entry.authorization, exhaustion)
        with self._lock:
            self._inflight.pop(request_id, None)
        failure_payload: JsonObject = {
            "failure_class": exhaustion.failure_class.value,
            "safe_message": exhaustion.safe_message,
        }
        if exhaustion.rejected_parameter is not None:
            failure_payload["rejected_parameter"] = exhaustion.rejected_parameter
        if exhaustion.provider_detail is not None:
            failure_payload["provider_detail"] = exhaustion.provider_detail
        return json.dumps(
            {"exhausted": True, "failure": failure_payload},
            separators=(",", ":"),
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
        try:
            self._write_ledger.finish_attempt(
                attempt_id=attempt_id,
                terminal_event=terminal,
                failure=failure,
                finalize_request=finalize,
                first_token_at=first_token_at,
            )
        except Exception as exc:  # noqa: BLE001 - the data plane retries.
            # The exact settlement is retained so a retry (from the data
            # plane or the timer sweep) lands the ORIGINAL outcome and usage,
            # never a downgraded cancellation.
            with self._lock:
                entry.pending_settlement = data
            raise authority_error(exc) from exc
        self._record_health(entry, attempt_id, opened=opened, failure=failure)
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
        failure = _failure_from_payload(data.get("failure")) or GatewayFailure(
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
        depth = entry.attempt_depths.get(attempt_id)
        if depth is None:
            return
        key = deployment_health_key(entry.authorization, entry.route.deployments[depth])
        if opened:
            self._health.dispatch_opened(key)
        if failure is not None:
            self._health.failed(key, failure)
        else:
            self._health.succeeded(key)

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
    ) -> bool:
        """Land one swept settlement; keep the entry for retry on failure.

        Returns:
            Whether the swept terminal write reached the ledger.
        """
        try:
            self._write_ledger.finish_attempt(
                attempt_id=attempt_id,
                terminal_event=terminal,
                failure=failure,
                finalize_request=finalize,
            )
        except Exception:  # noqa: BLE001 - keep the entry; the sweep retries.
            self._accounting_healthy = False
            return False
        self._record_health(entry, attempt_id, opened=False, failure=failure)
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
