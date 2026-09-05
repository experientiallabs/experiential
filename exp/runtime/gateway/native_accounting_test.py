"""Tests for the native attempt-accounting registry's waterfall reservations."""

from __future__ import annotations

import json
import time
from datetime import datetime

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.catalog import (
    GatewayDeploymentCapabilities,
    GatewayDeploymentMetadata,
)
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.budgets import BudgetReservationRejected, BudgetScopeKind
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayEvent,
    GatewayFailure,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.native_accounting import (
    NativeAttemptAccounting,
    NativeBridgeError,
    _failure_from_payload,
)
from exp.runtime.gateway.native_execution import InflightRequest, deployment_health_key
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.openai_protocol.errors import (
    THROTTLED_RETRY_AFTER_SECONDS,
    public_failure_error,
)

_DIGEST = "a" * 64


def _deployment(deployment_id: str, *, connection_sha256: str) -> ExactModelDeployment:
    """Build one deployment in the shared certified exact-model pool."""
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=deployment_id,
        exact_model_id="exact-one",
        connection=f"connection-{deployment_id}",
        provider="openai",
        provider_model="provider-model",
        connection_sha256=connection_sha256,
        capabilities_sha256="d" * 64,
        gateway=GatewayDeploymentMetadata(
            capabilities=GatewayDeploymentCapabilities(supports_streaming=True)
        ),
    )


def _authorization(catalog_sha256: str) -> AuthorizationSnapshot:
    """Build one direct authority snapshot pinned to the test catalog."""
    return AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256=catalog_sha256,
        canonical_request_sha256=_DIGEST,
        deadline_monotonic=1.0,
    )


def _request() -> GatewayRequest:
    """Build one canonical request for physical execution tests."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
    )


def _route(
    deployments: tuple[ExactModelDeployment, ...],
    *,
    refusal_failover: bool = False,
) -> GatewayRoute:
    """Build one frozen certified route with a live request deadline."""
    authorization = _authorization(_DIGEST).model_copy(
        update={
            "deadline_monotonic": time.monotonic() + 30,
            "refusal_failover": refusal_failover,
        }
    )
    return GatewayRoute(
        snapshot=ExecutionSnapshot(
            authorization=authorization,
            exact_model_id="exact-one",
            pool_id="pool-one",
            deployment_ids=tuple(item.deployment_id for item in deployments),
        ),
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )


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


def test_a_fully_throttled_route_exhausts_as_throttled_not_provider_internal() -> None:
    """Pre-dispatch exhaustion caused only by provider throttle windows is
    caller-facing rate limiting, never platform deadness.

    Production signal (2026-09-04): a single-rung alias whose rung sat inside
    the 30s throttle window after provider 429s reported every shadowed
    request as provider_internal "all exact-model deployments are
    unavailable", misfiling a 429 storm as an outage.
    """
    registry, ledger, entry = _registry()
    throttle = GatewayFailure(
        failure_class=GatewayFailureClass.THROTTLED,
        safe_message="provider throttled the request",
    )
    for deployment in entry.route.deployments:
        registry.health.failed(deployment_health_key(entry.authorization, deployment), throttle)

    exhausted = _start(registry, ordinal=0)

    assert exhausted["exhausted"] is True
    failure_payload = exhausted["failure"]
    assert isinstance(failure_payload, dict)
    assert failure_payload["failure_class"] == "throttled"
    message = str(failure_payload["safe_message"])
    assert "throttle window" in message
    retry_after = failure_payload["retry_after_seconds"]
    assert isinstance(retry_after, int)
    # The advertised Retry-After covers the whole remaining window (floored
    # at the default backoff) and the message names the same wait, so a
    # client honoring the header never retries into the window it was told
    # to sit out.
    assert THROTTLED_RETRY_AFTER_SECONDS <= retry_after <= 30
    assert f"retry in {retry_after}s" in message
    public = public_failure_error(GatewayFailure.model_validate(failure_payload))
    assert public.retry_after_seconds == retry_after
    assert [failure.failure_class.value for failure in ledger.finished_requests] == ["throttled"]
    assert registry.entry("request-one") is None


def test_an_open_circuit_route_still_dispatches_and_never_reports_throttled() -> None:
    """Circuit-open deployments stay dispatchable through forced claims, so
    the throttled exhaustion class is reserved for real throttle windows."""
    registry, ledger, entry = _registry()
    dead = GatewayFailure(
        failure_class=GatewayFailureClass.PROVIDER_AUTHENTICATION,
        safe_message="provider authentication failed",
    )
    for deployment in entry.route.deployments:
        registry.health.failed(deployment_health_key(entry.authorization, deployment), dead)

    started = _start(registry, ordinal=0)

    assert started["route_depth"] == 0
    assert ledger.finished_requests == []


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


def test_rejected_parameter_crosses_the_boundary_only_as_a_string() -> None:
    """The provider-named parameter path survives the failure payload decode."""
    registry, _ledger, _entry = _registry()
    started = _start(registry, ordinal=0)
    assert (
        _settle(
            registry,
            attempt_id=str(started["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure={
                "failure_class": "invalid_request",
                "safe_message": "provider rejected the request",
                "rejected_parameter": "input[1].status",
            },
        )
        == "{}"
    )
    exhausted = _start(
        registry,
        ordinal=1,
        current_depth=0,
        failure={
            "failure_class": "invalid_request",
            "safe_message": "provider rejected the request",
            "rejected_parameter": "input[1].status",
        },
    )
    assert exhausted["exhausted"] is True
    failure_payload = exhausted["failure"]
    assert isinstance(failure_payload, dict)
    assert failure_payload["rejected_parameter"] == "input[1].status"
    # Non-string or empty payload values decode to None, never a coerced str.
    numeric = _failure_from_payload(
        {"failure_class": "invalid_request", "safe_message": "x", "rejected_parameter": 7}
    )
    assert numeric is not None and numeric.rejected_parameter is None
    empty = _failure_from_payload(
        {"failure_class": "invalid_request", "safe_message": "x", "rejected_parameter": ""}
    )
    assert empty is not None and empty.rejected_parameter is None


def test_provider_detail_crosses_the_boundary_only_as_a_string() -> None:
    """The provider explanation survives the failure payload decode."""
    registry, _ledger, _entry = _registry()
    _start(registry, ordinal=0)
    exhausted = _start(
        registry,
        ordinal=1,
        current_depth=0,
        failure={
            "failure_class": "invalid_request",
            "safe_message": "provider rejected the request",
            "provider_detail": "`top_p` is deprecated for this model.",
        },
    )
    failure_payload = exhausted["failure"]
    assert isinstance(failure_payload, dict)
    assert failure_payload["provider_detail"] == "`top_p` is deprecated for this model."
    numeric = _failure_from_payload(
        {"failure_class": "invalid_request", "safe_message": "x", "provider_detail": 7}
    )
    assert numeric is not None and numeric.provider_detail is None
    empty_detail = _failure_from_payload(
        {"failure_class": "invalid_request", "safe_message": "x", "provider_detail": ""}
    )
    assert empty_detail is not None and empty_detail.provider_detail is None
