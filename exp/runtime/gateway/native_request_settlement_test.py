"""Paid admission work survives a failed terminal write without a synthetic model attempt."""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import patch

import pytest

from exp.runtime.gateway.budgets import BudgetScopeKind
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayFailure,
    GatewayFailureClass,
)
from exp.runtime.gateway.native_accounting_errors import NativeBridgeError
from exp.runtime.gateway.native_accounting_test import (
    _RecordingLedger,
    _registry,
    _retryable_failure,
    _settle,
    _start,
)
from exp.runtime.gateway.native_request_settlement import RequestSettlements


def test_pre_dispatch_search_meter_is_retried_by_the_accounting_sweep() -> None:
    """Block further paid work until the original verdict and exact meter commit."""
    accounting, ledger, entry = _registry()
    ledger.fail_request_finishes = 2
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.GUARDRAIL,
        safe_message="blocked",
        safe_details={"input_guardrail_denied": True},
    )
    assert not accounting.finish_request_quietly(
        entry.authorization,
        failure,
        web_search_requests=1,
    )
    assert not accounting.accounting_healthy
    with pytest.raises(NativeBridgeError, match="accounting is recovering"):
        accounting.request_settlements.require_clear()
    accounting.sweep_expired()
    assert not accounting.accounting_healthy
    accounting.sweep_expired()
    assert accounting.accounting_healthy
    accounting.request_settlements.require_clear()
    accounting.sweep_expired()
    assert ledger.finished_requests == [failure]
    assert ledger.request_search_meters == [1]
    assert ledger.started == []


def test_ledger_without_unattempted_search_contract_is_rejected_at_startup() -> None:
    """A host must support the hard cutover before the gateway can spend on a search."""
    ledger = _RecordingLedger()

    def incomplete_contract(
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
        certify_no_effects: bool = False,
    ) -> bool:
        """Expose an incomplete ledger signature without allowing a write."""
        raise AssertionError("startup validation must not execute the ledger")

    with patch.object(ledger, "finish_request", incomplete_contract):
        with pytest.raises(ValueError, match="persist unattempted search expense"):
            RequestSettlements(ledger)


def test_normal_in_progress_terminal_write_does_not_block_unrelated_admission() -> None:
    """Only failed writes close admission; a normal commit may still be in flight."""
    accounting, ledger, entry = _registry()
    entered, release = threading.Event(), threading.Event()
    calls: list[int] = []

    def commit(
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
        certify_no_effects: bool = False,
        web_search_requests: int = 0,
    ) -> bool:
        """Hold a successful terminal write open while another request checks admission."""
        calls.append(1)
        entered.set()
        assert release.wait(3)
        return True

    failure = GatewayFailure(failure_class=GatewayFailureClass.GUARDRAIL, safe_message="blocked")
    results: list[bool] = []
    worker = threading.Thread(
        target=lambda: results.append(
            accounting.finish_request_quietly(entry.authorization, failure)
        )
    )
    with patch.object(ledger, "finish_request", commit):
        worker.start()
        try:
            assert entered.wait(3)
            assert accounting.accounting_healthy
            accounting.request_settlements.require_clear()
            accounting.request_settlements.retry(16)
            assert calls == [1]
        finally:
            release.set()
            worker.join(3)
    assert not worker.is_alive()
    assert results == [True]


@pytest.mark.parametrize("termination", ["deadline", "budget", "reservation", "abandon"])
def test_registered_search_meter_survives_unattempted_termination(termination: str) -> None:
    """Every registered terminal path preserves paid search with no invented attempt."""
    accounting, ledger, entry = _registry()
    entry.web_search_requests = 1
    if termination == "deadline":
        entry.deadline_monotonic = time.monotonic() - 1
    elif termination == "budget":
        ledger.budget_rejections["deployment-a"] = BudgetScopeKind.TEAM
    elif termination == "reservation":
        ledger.typed_rejection = GatewayFailure(
            failure_class=GatewayFailureClass.UNAVAILABLE, safe_message="unavailable"
        )
    if termination == "abandon":
        accounting.abandon(json.dumps({"request_id": entry.authorization.request_id}))
    else:
        with pytest.raises(NativeBridgeError):
            _start(accounting, ordinal=0)
    assert ledger.request_search_meters == [1]
    assert not ledger.started
    assert accounting.entry(entry.authorization.request_id) is None


def test_registered_search_survives_retained_terminal_write() -> None:
    """The meter outlives removal of the in-flight entry while a terminal write retries."""
    accounting, ledger, entry = _registry()
    entry.web_search_requests = 1
    entry.deadline_monotonic = time.monotonic() - 1
    ledger.fail_request_finishes = 2
    with pytest.raises(NativeBridgeError):
        _start(accounting, ordinal=0)
    assert accounting.entry(entry.authorization.request_id) is None
    assert not accounting.accounting_healthy
    accounting.sweep_expired()
    assert not accounting.accounting_healthy
    accounting.sweep_expired()
    assert accounting.accounting_healthy
    assert ledger.request_search_meters == [1]
    assert ledger.finished_requests[0].failure_class is GatewayFailureClass.TIMEOUT
    assert not ledger.started


def test_failed_abandonment_retains_search_meter_and_original_failure() -> None:
    """Recover the original expense and refusal even if abandonment is delivered again."""
    accounting, ledger, entry = _registry()
    entry.web_search_requests = 1
    ledger.fail_request_finishes = 1
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.TIMEOUT, safe_message="permit deadline exceeded"
    )
    with pytest.raises(NativeBridgeError):
        accounting.abandon(
            json.dumps(
                {
                    "request_id": entry.authorization.request_id,
                    "failure": failure.model_dump(mode="json"),
                }
            )
        )
    assert not accounting.accounting_healthy
    with pytest.raises(NativeBridgeError, match="accounting is recovering"):
        accounting.request_settlements.require_clear()
    accounting.abandon(json.dumps({"request_id": entry.authorization.request_id}))
    accounting.sweep_expired()
    assert accounting.accounting_healthy
    assert ledger.request_search_meters == [1]
    assert ledger.finished_requests == [failure]
    assert not ledger.started
    assert accounting.entry(entry.authorization.request_id) is None


@pytest.mark.parametrize("previous_attempt", [False, True])
def test_active_abandonment_owns_search_only_on_final_settlement(previous_attempt: bool) -> None:
    """Search belongs to the final attempt even after a failed fallback rung."""
    accounting, ledger, entry = _registry()
    entry.web_search_requests = 1
    started = _start(accounting, ordinal=0)
    if previous_attempt:
        _settle(
            accounting,
            attempt_id=str(started["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure=_retryable_failure(),
        )
        assert ledger.web_search_requests == [None]
        _start(
            accounting,
            ordinal=1,
            current_depth=0,
            failure=_retryable_failure(),
        )
    accounting.abandon(json.dumps({"request_id": entry.authorization.request_id}))
    accounting.sweep_expired()
    assert ledger.web_search_requests == ([None, 1] if previous_attempt else [1])
    assert ledger.request_search_meters == []
    assert len(ledger.finished) == (2 if previous_attempt else 1)
    assert accounting.entry(entry.authorization.request_id) is None
