"""Initial admission and lifecycle access to the sole Rust accounting owner."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable

from exp_gateway_native import NativeAccounting, NativeCallbacks

from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.guardrails.session import GuardrailSession, discard_guardrails
from exp.runtime.gateway.native_accounting_errors import (
    NativeBridgeError,
    accounting_recovering_error,
)
from exp.runtime.gateway.native_capture import CaptureController, discard_capture
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.native_handoff import NativeRequestFactory, NativeRequestHandoff
from exp.runtime.gateway.native_loads import NativeLoads
from exp.runtime.gateway.native_recovery_state import NativeRecovery
from exp.runtime.gateway.native_state import NativeHealth, NativeState, NativeSticky
from exp.runtime.gateway.recovery import RecoveryHost
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


class NativeAttemptAccounting:
    """Own one native registry; Python retains no post-admission request or settlement."""

    def __init__(
        self,
        *,
        native_request_factory: NativeRequestFactory,
        recovery_host: RecoveryHost | None = None,
        default_lane_bound: int | None = None,
        background_recovery: bool = True,
        capture: CaptureController | None = None,
    ) -> None:
        """Compose one interpreter-independent owner and its initial admission views."""
        self.recovery = NativeRecovery()
        self.native = NativeAccounting(
            default_bound=default_lane_bound,
            background_recovery=background_recovery,
            recovery=self.recovery.native,
        )
        self._native_factory = native_request_factory
        self._native_capture = capture
        self.recovery_host = recovery_host
        state = NativeState(self.native)
        self.health = NativeHealth(state)
        self.loads = NativeLoads(state, default_lane_bound)
        self.sticky = NativeSticky(state)
        self._lock = threading.Lock()
        self._admission_coercions = 0
        self._admission_dead = 0
        self._admission_lead = 0

    def register(
        self, entry: InflightRequest, *, prepare: Callable[[], None] | None = None
    ) -> None:
        """Transfer the resolved route, inspection and financial resources exactly once."""
        # Registration imports admission helpers that use this facade for annotations.
        from exp.runtime.gateway.native_registration import registration_json

        transferred = False
        try:
            handoff = self._native_factory(entry)
            target = handoff.callbacks if isinstance(handoff, NativeRequestHandoff) else handoff
            if not isinstance(target, NativeCallbacks):
                raise TypeError("Admission requires an interpreter-independent native ledger")
            if entry.guardrails is not None:
                entry.guardrails.native_execution = True
            if prepare is not None:
                prepare()
            if (
                entry.guardrails is not None
                and entry.guardrails.input_pending
                and entry.guardrails.native_input is None
            ):
                raise ValueError("Native ownership cannot retain a Python input inspection task")
            pending = entry.pending_capture
            capture = self._native_capture
            record = (
                capture.prepare(
                    entry.authorization,
                    pending.request,
                    entry.route.snapshot.exact_model_id,
                    session_id=pending.session_id,
                )
                if capture is not None and pending is not None
                else None
            )
            self.native.register(
                registration_json(entry, bind_sticky=self.recovery_host is None),
                target,
                input=None if entry.guardrails is None else entry.guardrails.native_input,
                observation=None
                if entry.guardrails is None
                else entry.guardrails.native_observation,
                capture=None if capture is None else capture.native,
                pending_capture=record,
            )
            transferred = True
            entry.pending_capture = None
            if entry.guardrails is not None:
                for inspection in entry.guardrails.native_completed:
                    self.native.record_input(
                        inspection, target, str(entry.authorization.request_id)
                    )
                entry.guardrails.native_completed.clear()
            if isinstance(handoff, NativeRequestHandoff):
                handoff.registered()
        except Exception as exc:
            discard_capture(self._native_capture, entry.authorization.request_id)
            discard_guardrails(entry)
            if transferred:
                try:
                    self.native.abort_admission(
                        str(entry.authorization.request_id), entry.web_search_requests
                    )
                except Exception:  # noqa: BLE001 - the native owner retains the exact failed write.
                    pass
            else:
                self.finish_request_quietly(
                    entry.authorization,
                    exc.failure
                    if isinstance(exc, NativeBridgeError) and exc.failure is not None
                    else GatewayFailure(
                        failure_class=GatewayFailureClass.INTERNAL,
                        safe_message="gateway could not hand off admitted request accounting",
                    ),
                    web_search_requests=entry.web_search_requests,
                    input_session=entry.guardrails,
                )
            raise

    def call(self, method: str, argument: str) -> str:
        """Expose the same Rust owner to administrative callers and local checks."""
        try:
            return self.native.call(method, argument)
        except ValueError as error:
            try:
                public = OpenAIProtocolError(**json.loads(str(error)))
            except (TypeError, ValueError):
                public = OpenAIProtocolError(
                    status_code=500,
                    code="internal_error",
                    message="The gateway request failed.",
                    error_type="api_error",
                )
            raise NativeBridgeError(public) from error

    def start_attempt(self, argument: str) -> str:
        """Reserve with the native owner, including its retry and capacity policy."""
        return self.call("start_attempt", argument)

    def settle(self, argument: str) -> str:
        """Settle through the native financial writer and retained recovery owner."""
        return self.call("settle", argument)

    def price(self, argument: str) -> str:
        """Price a terminal with the same native owner that will settle it."""
        return self.call("price", argument)

    def abandon(self, argument: str) -> str:
        """Retain cancellation in Rust until financial acknowledgement."""
        return self.call("abandon", argument)

    def finish_request_quietly(
        self,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
        *,
        certify_no_effects: bool = False,
        web_search_requests: int = 0,
        input_session: GuardrailSession | None = None,
    ) -> bool:
        """Hand failed admission to native retention without replacing its original error."""
        try:
            target = self._native_factory.admission(authorization)
            if input_session is not None:
                try:
                    if input_session.native_input is not None:
                        self.native.record_input(
                            input_session.native_input, target, str(authorization.request_id)
                        )
                        input_session.native_input = None
                    if input_session.native_observation is not None:
                        self.native.record_input(
                            input_session.native_observation, target, str(authorization.request_id)
                        )
                        input_session.native_observation = None
                    for inspection in input_session.native_completed:
                        self.native.record_input(inspection, target, str(authorization.request_id))
                    input_session.native_completed.clear()
                except Exception:  # noqa: BLE001 - reporting cannot prevent financial cleanup.
                    pass
            return self.native.finish_admission(
                authorization.organization_id,
                json.dumps(
                    {
                        "request_id": authorization.request_id,
                        "failure": failure.model_dump(mode="json"),
                        "certify_no_effects": certify_no_effects,
                        "web_search_requests": web_search_requests,
                    },
                    separators=(",", ":"),
                ),
                target,
            )
        except Exception:  # noqa: BLE001 - the original sanitized admission failure wins.
            return False

    def require_clear(self, organization_id: str) -> None:
        """Fence only organizations with unacknowledged native accounting writes."""
        if not self.native.is_clear(organization_id):
            raise NativeBridgeError(accounting_recovering_error())

    @property
    def accounting_healthy(self) -> bool:
        """Report whether all required native writes are acknowledged."""
        return not self.native.has_failed_writes()

    def fenced_organizations(self) -> frozenset[str]:
        """Read native accounting fences without scanning Python request objects."""
        return frozenset(self.native.fenced_organizations())

    def _counts(self, *keys: str) -> tuple[int, ...]:
        """Read one coherent snapshot of native execution counters."""
        counts = json.loads(self.native.counters_json())
        return tuple(counts[key] for key in keys)

    def counters(self) -> tuple[int, int, int]:
        """Return recovered terminals, expired requests and retained request count."""
        replayed, abandoned = self._counts("replayed", "abandoned")
        return replayed, abandoned, self.native.inflight()

    def rung_admission_counters(self) -> tuple[int, int, int]:
        """Return capacity sheds, forced dispatches and capacity refusals."""
        sheds, overflows, refusals = self._counts("sheds", "overflows", "refusals")
        return sheds, overflows, refusals

    def rung_rate_counters(self) -> tuple[int, int]:
        """Return request-rate sheds and fresh-session spills."""
        rates, fresh = self._counts("rates", "fresh")
        return rates, fresh

    def throttle_cache_counters(self) -> tuple[int, int, int, int]:
        """Return throttle outcomes counted at native dispatch."""
        surfaced, cold, backoff, forced = self._counts(
            "surfaced", "cold", "backoff", "forced_backoff"
        )
        return surfaced, cold, backoff, forced

    def record_admission_coercions(self, count: int) -> None:
        """Count request substitutions disclosed during initial admission."""
        with self._lock:
            self._admission_coercions += count

    def admission_parameter_coercions(self) -> int:
        """Read the initial admission substitution counter."""
        with self._lock:
            return self._admission_coercions

    def record_admission_rung_skips(self, dead_count: int, *, lead_skipped: bool) -> None:
        """Count unavailable rungs excluded before native ownership begins."""
        with self._lock:
            self._admission_dead += dead_count
            self._admission_lead += int(lead_skipped)

    def admission_rung_skips(self) -> tuple[int, int]:
        """Read initial admission's lead and total unavailable-rung counts."""
        with self._lock:
            return self._admission_lead, self._admission_dead

    def sweep_expired(self) -> None:
        """Run bounded native recovery when the host schedules its own maintenance."""
        self.native.recover()

    def close(self, timeout_seconds: float = 10.0) -> bool:
        """Retain native financial ownership until all writes drain or timeout expires."""
        return self.native.close(min(60.0, max(0.0, timeout_seconds)))
