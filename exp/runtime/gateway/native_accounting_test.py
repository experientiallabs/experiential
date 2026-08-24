"""Tests for the native attempt-accounting registry's waterfall reservations."""

from __future__ import annotations

import json
import time
from datetime import datetime

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.budgets import BudgetReservationRejected, BudgetScopeKind
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayEvent,
    GatewayFailure,
)
from exp.runtime.gateway.native_accounting import (
    NativeAttemptAccounting,
    NativeBridgeError,
)
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.tests.waterfall_test import _deployment, _request, _route


class _RecordingLedger:
    """Blocking write-ledger fake recording every waterfall write."""

    def __init__(self) -> None:
        """Start with empty write logs and no scripted rejections."""
        self.started: list[JsonObject] = []
        self.finished: list[JsonObject] = []
        self.finished_requests: list[GatewayFailure] = []
        self.budget_rejections: dict[str, BudgetScopeKind] = {}
        self._counter = 0

    def accept_request(self, *, authorization: AuthorizationSnapshot) -> None:
        """Record one accepted request (unused by the registry itself)."""
        del authorization

    def start_attempt(
        self,
        *,
        snapshot: object,
        deployment: ExactModelDeployment,
        attempt_ordinal: int,
        route_depth: int,
        maximum_cost_micro_usd: int | None = None,
        route_reason: str | None = None,
        fallback_reason: str | None = None,
    ) -> str:
        """Reserve one recorded attempt row, honoring scripted rejections."""
        del snapshot, maximum_cost_micro_usd, route_reason, fallback_reason
        scope = self.budget_rejections.get(deployment.deployment_id)
        if scope is not None:
            raise BudgetReservationRejected(scope_kind=scope, reason="scripted")
        self._counter += 1
        attempt_id = f"attempt-{self._counter}"
        self.started.append(
            {
                "attempt_id": attempt_id,
                "deployment_id": deployment.deployment_id,
                "attempt_ordinal": attempt_ordinal,
                "route_depth": route_depth,
            }
        )
        return attempt_id

    def finish_attempt(
        self,
        *,
        attempt_id: str,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
        first_token_at: datetime | None = None,
    ) -> None:
        """Record one settled attempt."""
        del terminal_event, first_token_at
        self.finished.append(
            {
                "attempt_id": attempt_id,
                "failure_class": None if failure is None else failure.failure_class.value,
                "finalize": finalize_request,
            }
        )

    def finish_request(
        self,
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
    ) -> None:
        """Record one request-only terminalization."""
        del authorization
        self.finished_requests.append(failure)


def _registry() -> tuple[NativeAttemptAccounting, _RecordingLedger, InflightRequest]:
    """Compose one registry over a two-deployment certified route."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)  # type: ignore[arg-type]
    deployments = (
        _deployment("deployment-a", connection_sha256="b" * 64),
        _deployment("deployment-b", connection_sha256="c" * 64),
    )
    route = _route(deployments)
    entry = InflightRequest(
        authorization=route.snapshot.authorization,
        route=route,
        request=_request(),
        deadline_monotonic=time.monotonic() + 30,
    )
    registry.register(entry)
    return registry, ledger, entry


def _start(
    registry: NativeAttemptAccounting,
    *,
    ordinal: int,
    current_depth: int | None = None,
    failure: JsonObject | None = None,
) -> JsonObject:
    """Call one start_attempt with the data plane's wire shape."""
    return json.loads(
        registry.start_attempt(
            json.dumps(
                {
                    "request_id": "request-one",
                    "attempt_ordinal": ordinal,
                    "current_depth": current_depth,
                    "failure": failure,
                }
            )
        )
    )


def _settle(
    registry: NativeAttemptAccounting,
    *,
    attempt_id: str,
    outcome: str,
    finalize: bool,
    failure: JsonObject | None = None,
) -> str:
    """Call one settle with the data plane's wire shape."""
    return registry.settle(
        json.dumps(
            {
                "request_id": "request-one",
                "attempt_id": attempt_id,
                "outcome": outcome,
                "usage": None,
                "tool_names": [],
                "failure": failure,
                "finalize": finalize,
                "opened": True,
            }
        )
    )


def _retryable_failure() -> JsonObject:
    """One wire failure the executor may redial on the same deployment."""
    return {
        "failure_class": "provider_internal",
        "safe_message": "provider service failed; retry after a short delay",
        "retryable_same_deployment": True,
        "failover_eligible": True,
    }


def test_waterfall_reservations_count_every_physical_dispatch() -> None:
    """Ordinals count all dispatches; depth tracks the deployment position."""
    registry, ledger, _entry = _registry()
    first = _start(registry, ordinal=0)
    assert first == {"attempt_id": "attempt-1", "route_depth": 0}
    assert (
        _settle(
            registry,
            attempt_id="attempt-1",
            outcome="failed",
            finalize=False,
            failure=_retryable_failure(),
        )
        == "{}"
    )
    redial = _start(registry, ordinal=1, current_depth=0, failure=_retryable_failure())
    assert redial == {"attempt_id": "attempt-2", "route_depth": 0}
    assert (
        _settle(
            registry,
            attempt_id="attempt-2",
            outcome="failed",
            finalize=False,
            failure=_retryable_failure(),
        )
        == "{}"
    )
    failover = _start(registry, ordinal=2, current_depth=0, failure=_retryable_failure())
    assert failover == {"attempt_id": "attempt-3", "route_depth": 1}
    assert _settle(registry, attempt_id="attempt-3", outcome="completed", finalize=True) == "{}"
    assert [(row["attempt_ordinal"], row["route_depth"]) for row in ledger.started] == [
        (0, 0),
        (1, 0),
        (2, 1),
    ]
    assert [row["finalize"] for row in ledger.finished] == [False, False, True]
    assert registry.entry("request-one") is None
    assert ledger.finished_requests == []


def test_deployment_budget_rejection_skips_to_the_next_route() -> None:
    """A deployment-scope budget rejection advances without a caller error."""
    registry, ledger, _entry = _registry()
    ledger.budget_rejections["deployment-a"] = BudgetScopeKind.DEPLOYMENT
    started = _start(registry, ordinal=0)
    assert started["route_depth"] == 1
    assert ledger.started[0]["deployment_id"] == "deployment-b"


def test_non_deployment_budget_rejection_finalizes_with_quota() -> None:
    """A team-scope rejection raises the public quota error and finalizes."""
    registry, ledger, _entry = _registry()
    ledger.budget_rejections["deployment-a"] = BudgetScopeKind.TEAM
    with pytest.raises(NativeBridgeError) as excinfo:
        _start(registry, ordinal=0)
    payload = json.loads(excinfo.value.public_error_json)
    assert payload["status_code"] == 429
    assert payload["code"] == "insufficient_quota"
    assert [failure.failure_class.value for failure in ledger.finished_requests] == [
        "quota_exceeded"
    ]
    assert registry.entry("request-one") is None


def test_exhaustion_finalizes_the_request_with_the_last_failure() -> None:
    """An ineligible failure class exhausts the ladder and finalizes."""
    registry, ledger, _entry = _registry()
    started = _start(registry, ordinal=0)
    assert (
        _settle(
            registry,
            attempt_id=str(started["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure={"failure_class": "invalid_request", "safe_message": "bad request"},
        )
        == "{}"
    )
    exhausted = _start(
        registry,
        ordinal=1,
        current_depth=0,
        failure={
            "failure_class": "invalid_request",
            "safe_message": "bad request",
            "retryable_same_deployment": False,
            "failover_eligible": False,
        },
    )
    assert exhausted["exhausted"] is True
    failure_payload = exhausted["failure"]
    assert isinstance(failure_payload, dict)
    assert failure_payload["failure_class"] == "invalid_request"
    assert [failure.failure_class.value for failure in ledger.finished_requests] == [
        "invalid_request"
    ]
    assert registry.entry("request-one") is None


def test_ordinal_mismatch_is_a_wire_contract_failure() -> None:
    """A desynchronized dispatch count fails closed as an internal error."""
    registry, _ledger, _entry = _registry()
    with pytest.raises(NativeBridgeError):
        _start(registry, ordinal=3)


def test_abandon_without_an_active_attempt_finalizes_the_request_row() -> None:
    """Abandoning an accepted request with no reservation closes the request."""
    registry, ledger, _entry = _registry()
    assert registry.abandon(json.dumps({"request_id": "request-one"})) == "{}"
    assert [failure.failure_class.value for failure in ledger.finished_requests] == ["cancelled"]
    assert registry.entry("request-one") is None


def test_sweep_cancels_the_active_attempt_after_the_deadline() -> None:
    """The deadline sweep closes an unsettled reservation as cancelled."""
    registry, ledger, entry = _registry()
    started = _start(registry, ordinal=0)
    entry.deadline_monotonic = time.monotonic() - 60.0
    registry.sweep_expired()
    assert ledger.finished == [
        {
            "attempt_id": started["attempt_id"],
            "failure_class": "cancelled",
            "finalize": True,
        }
    ]
    assert registry.entry("request-one") is None
    assert registry.counters()[1] == 1
