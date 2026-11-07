"""Explicit host handoff from Python admission to native transaction ownership."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from exp_gateway_native import NativeAccounting, NativeCallbacks

from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_accounting_errors import (
    NativeBridgeError,
    accounting_recovering_error,
)
from exp.runtime.gateway.native_accounting_recovery import recovery_drained
from exp.runtime.gateway.native_capture import CaptureController
from exp.runtime.gateway.native_components import SyncWriteLedger
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.native_loads import NativeLoads
from exp.runtime.gateway.native_recovery_state import NativeRecovery
from exp.runtime.gateway.native_registration import registration_json
from exp.runtime.gateway.native_state import NativeHealth, NativeState, NativeSticky
from exp.runtime.gateway.recovery import RecoveryHost


@dataclass(frozen=True, slots=True)
class NativeRequestHandoff:
    """Release host admission memos only after native registration succeeds.

    The registered callback runs synchronously during admission and is never
    retained by the native request. Failed registration leaves the original
    memos available to the existing admission-failure cleanup path.
    """

    callbacks: NativeCallbacks
    registered: Callable[[], None]


class NativeRequestFactory(Protocol):
    """Select a ledger owner once, at initial admission.

    A native backend returns its retained C ABI target. ``None`` explicitly
    selects the composition's synchronous ledger for this request. A failure
    raises and fails admission; it never selects another backend.
    """

    def __call__(self, entry: InflightRequest, /) -> NativeCallbacks | NativeRequestHandoff | None:
        """Select a retained native owner or the explicit synchronous ledger."""
        ...


def create_accounting(
    write_ledger: SyncWriteLedger,
    *,
    native_request_factory: NativeRequestFactory | None,
    budget_error_factory: Callable[[str], NativeBridgeError] | None,
    cache_sample_gate: Callable[[str], bool] | None,
    recovery_host: RecoveryHost | None,
    default_lane_bound: int | None,
    background_recovery: bool,
    capture: CaptureController | None = None,
) -> NativeAttemptAccounting:
    """Choose the explicitly configured request-owner contract at composition."""
    if native_request_factory is None:
        return NativeAttemptAccounting(
            write_ledger,
            budget_error_factory=budget_error_factory,
            cache_sample_gate=cache_sample_gate,
            recovery_host=recovery_host,
            default_lane_bound=default_lane_bound,
            background_recovery=background_recovery,
        )
    return NativeRequestAccounting(
        write_ledger,
        capture=capture,
        native_request_factory=native_request_factory,
        budget_error_factory=budget_error_factory,
        cache_sample_gate=cache_sample_gate,
        recovery_host=recovery_host,
        default_lane_bound=default_lane_bound,
        background_recovery=background_recovery,
    )


class NativeRequestAccounting(NativeAttemptAccounting):
    """Initial admission and lifecycle facade over native request owners."""

    def __init__(
        self,
        write_ledger: SyncWriteLedger,
        *,
        native_request_factory: NativeRequestFactory,
        budget_error_factory: Callable[[str], NativeBridgeError] | None = None,
        cache_sample_gate: Callable[[str], bool] | None = None,
        recovery_host: RecoveryHost | None = None,
        default_lane_bound: int | None = None,
        background_recovery: bool = True,
        capture: CaptureController | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        recovery = NativeRecovery()
        self.native = NativeAccounting(
            default_bound=default_lane_bound,
            background_recovery=background_recovery,
            recovery=recovery.native,
        )
        self._native_factory = native_request_factory
        self._native_capture = capture
        super().__init__(
            write_ledger,
            budget_error_factory=budget_error_factory,
            cache_sample_gate=cache_sample_gate,
            recovery_host=recovery_host,
            default_lane_bound=default_lane_bound,
            background_recovery=background_recovery,
            clock=clock,
        )
        self.recovery = recovery
        state = NativeState(self.native)
        self._health = NativeHealth(state)
        self._loads = NativeLoads(state, default_lane_bound)
        self._sticky = NativeSticky(state)

    def register(self, entry: InflightRequest) -> None:
        """Transfer ownership before returning the admitted wire contract."""
        try:
            handoff = self._native_factory(entry)
            target = handoff.callbacks if isinstance(handoff, NativeRequestHandoff) else handoff
            if target is None:
                super().register(entry)
            else:
                if (
                    entry.guardrails is not None
                    and entry.guardrails.input_pending
                    and entry.guardrails.native_input is None
                ):
                    message = "Native ownership cannot retain a Python input inspection task"
                    raise ValueError(message)
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
                    capture=None if capture is None else capture.native,
                    pending_capture=record,
                )
                entry.pending_capture = None
                if isinstance(handoff, NativeRequestHandoff):
                    handoff.registered()
        except Exception:
            self.finish_request_quietly(
                entry.authorization,
                GatewayFailure(
                    failure_class=GatewayFailureClass.INTERNAL,
                    safe_message="gateway could not hand off admitted request accounting",
                ),
            )
            raise

    def require_clear(self, organization_id: str) -> None:
        """Refuse admission while either owner retains failed writes for this organization."""
        super().require_clear(organization_id)
        if not self.native.is_clear(organization_id):
            raise NativeBridgeError(accounting_recovering_error())

    @property
    def accounting_healthy(self) -> bool:
        """Report whether both owners have acknowledged their pending writes."""
        return super().accounting_healthy and self.native.is_clear()

    def fenced_organizations(self) -> frozenset[str]:
        """Combine admission fences from both explicitly configured ledger contracts."""
        return super().fenced_organizations() | frozenset(self.native.fenced_organizations())

    def counters(self) -> tuple[int, int, int]:
        """Include native requests in the control plane activity snapshot."""
        replayed, abandoned, inflight = super().counters()
        return replayed, abandoned, inflight + self.native.inflight()

    def sweep_expired(self) -> None:
        """Recover retained writes through each owner without performing admission I/O."""
        super().sweep_expired()
        self.native.recover()

    def close(self, timeout_seconds: float = 10.0) -> bool:
        """Keep both recovery owners alive until their required writes drain."""
        deadline = time.monotonic() + timeout_seconds
        synchronous_closed = super().close(timeout_seconds)
        native_closed = self.native.close(min(60.0, max(0.0, deadline - time.monotonic())))
        return synchronous_closed and native_closed


def uses_native(accounting: NativeAttemptAccounting, request_id: str) -> bool:
    """Emit an explicit serving contract for the owner selected at admission."""
    return isinstance(accounting, NativeRequestAccounting) and accounting.native.owns_request(
        request_id
    )


def accounting_callbacks(accounting: NativeAttemptAccounting) -> NativeCallbacks | None:
    """Return the native owner registry bound to this control plane."""
    return (
        accounting.native.callbacks() if isinstance(accounting, NativeRequestAccounting) else None
    )


def accounting_drained(accounting: NativeAttemptAccounting) -> bool:
    """Require both ledger contracts to drain before the host closes resources."""
    return recovery_drained(accounting) and (
        not isinstance(accounting, NativeRequestAccounting) or accounting.native.drained()
    )
